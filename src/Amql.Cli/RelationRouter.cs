using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Cli;

/// <summary>The router's inputs and tunables.</summary>
public sealed record RouteOptions(int Top = 5, int MaxTemplates = 8, int TraceLayerStart = 8, int TraceLayerEnd = 24, bool NoTrace = false, string CorruptToken = "the");

/// <summary>One discovered relationship: the template that names it, the
/// model's propensity for B under that template, and the (layer, head,
/// position) tensors that carry the A→B link.</summary>
public sealed record RouteLink(
    string Relation,
    string Template,
    float Score,
    int Apos,
    int CtxLength,
    IReadOnlyList<RouteCoordinate> Coordinates,
    CausalAttribution? Attribution);

/// <summary>One tensor address inside the model: layer L, head H, query
/// position, key position, and the attention weight at that cell (final
/// prediction row, softmax layers only).</summary>
public sealed record RouteCoordinate(int Layer, int Head, int QueryPos, int KeyPos, float Weight);

/// <summary>
/// The relationship router ("stage one" of the larql-style probe
/// machinery): given two tokens A and B, it (1) probes a set of relation
/// templates, scoring each by how strongly the model predicts B after
/// template(A) — naming the relation; (2) locates the (layer, head,
/// position) attention tensors that carry A's content into B's prediction;
/// and (3) computes causal-tracing attribution weights per layer — the
/// tensors you would adjust (patch / LoRA) to change the propensity.
/// The template set is plain text data: a future trained-probe pipeline
/// slots in by swapping the template list for probe weights.
/// </summary>
public static class RelationRouter
{
    public static readonly (string Name, string Template)[] Templates =
    {
        ("Capital", "The capital of {0} is"),
        ("Language", "The official language of {0} is"),
        ("Contains", "{0} contains"),
        ("Located in", "{0} is located in"),
        ("Part of", "{0} is part of"),
        ("Borders", "{0} borders"),
        ("Known for", "{0} is known for"),
        ("City", "The largest city in {0} is"),
        ("Continent", "The continent of {0} is"),
        ("Currency", "The currency of {0} is"),
    };

    public sealed record RouteResult(string A, string BList, IReadOnlyList<RouteLink> Links);

    /// <summary>
    /// Routes A→B: template-scored links (top-N), each with its coordinate
    /// map; the strongest link additionally gets causal attribution over
    /// <c>options.TraceLayerStart..TraceLayerEnd</c> (unless NoTrace).
    /// Returns (links, notes) — notes carries multi-token warnings.
    /// </summary>
    public static (List<RouteLink> Links, List<string> Notes) Route(
        Vindex3Container container,
        string component,
        HfTokenizer tokenizer,
        string a,
        string b,
        RouteOptions options,
        Action<string>? progress = null)
    {
        var aIds = tokenizer.EncodeToIds(a).ToArray();
        var bIds = tokenizer.EncodeToIds(b).ToArray();
        var notes = new List<string>();
        if (aIds.Length == 0 || bIds.Length == 0)
        {
            throw new CliException($"cannot tokenize '{a}' or '{b}' — no token output");
        }
        if (aIds.Length > 1)
        {
            notes.Add($"'{a}' tokenizes to {aIds.Length} tokens; probing uses its first token id {aIds[0]}.");
        }
        if (bIds.Length > 1)
        {
            notes.Add($"'{b}' tokenizes to {bIds.Length} tokens; the score targets its first token id {bIds[0]}.");
        }
        int aId = aIds[0];
        int bId = bIds[0];

        // The model continues with the space-merged form ("ĠParis"), which
        // tokenises to a different id than the standalone word — score both.
        var bTargets = bIds
            .Concat(tokenizer.EncodeToIds(" " + b).ToArray())
            .Distinct()
            .ToArray();
        int corruptId = tokenizer.EncodeToIds(options.CorruptToken).FirstOrDefault(3);

        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, component, store);
        var rt = new GenericRuntime(plan, store);

        // ── phase 1: template probing (name + propensity + coordinates) ──
        var links = new List<RouteLink>();
        foreach (var (name, template) in Templates.Take(options.MaxTemplates))
        {
            string filled = template.Replace("{0}", a);
            var ctx = tokenizer.EncodeToIds(filled).ToArray();
            int apos = FindA(tokenizer, filled, a);
            if (apos < 0)
            {
                notes.Add($"template '{name}': '{a}' not found in the tokenized template; skipped.");
                continue;
            }
            progress?.Invoke($"probing '{name}' ({filled}) …");

            var logits = CausalTracer.RunPositionMajor(rt, plan, ctx, captureTrace: !options.NoTrace);
            float score = CausalTracer.SoftmaxProb(logits, bTargets);

            var coords = CoordinatesFromTrace(rt.AttentionTrace, ctx.Length, apos);
            links.Add(new RouteLink(name, template, score, apos, ctx.Length, coords, null));
        }

        links.Sort((x, y) => y.Score.CompareTo(x.Score));
        if (links.Count == 0)
        {
            return (links, notes);
        }

        // ── phase 2: causal attribution for the strongest link ───────────
        if (!options.NoTrace)
        {
            var best = links[0];
            var ctx = tokenizer.EncodeToIds(best.Template.Replace("{0}", a)).ToArray();
            int apos = FindA(tokenizer, best.Template.Replace("{0}", a), a);
            progress?.Invoke($"attributing '{best.Relation}' across layers {options.TraceLayerStart}..{options.TraceLayerEnd} …");
            var attribution = CausalTracer.Trace(
                rt, plan, ctx, apos, bTargets, corruptId,
                options.TraceLayerStart, options.TraceLayerEnd,
                () => progress?.Invoke("·"));
            var updated = links
                .Select((l, i) => i == 0 ? l with { Attribution = attribution } : l)
                .ToList();
            return (updated, notes);
        }
        return (links, notes);
    }

    /// <summary>Position of the first piece whose decoded text contains the
    /// bare word (handles the space-prefixed "ĠFrance" token inside a
    /// sentence — the standalone token id differs from its in-sentence
    /// space-merge form).</summary>
    private static int FindA(HfTokenizer tokenizer, string text, string a)
    {
        var pieces = tokenizer.Encode(text).Pieces;
        for (int i = 0; i < pieces.Count; i++)
        {
            var decoded = pieces[i].DecodedText;
            if (decoded is not null && decoded.Contains(a, StringComparison.Ordinal))
            {
                return i;
            }
        }
        return -1;
    }

    /// <summary>Per (softmax layer, head), the final prediction row's
    /// attention weight on A's position — the cheapest tensor-level
    /// addresses of the link.</summary>
    private static List<RouteCoordinate> CoordinatesFromTrace(List<GenericRuntime.LayerHeadAttention>? trace, int ctxLength, int apos)
    {
        var coords = new List<RouteCoordinate>();
        if (trace is null)
        {
            return coords;
        }
        int queryPos = ctxLength - 1;
        foreach (var row in trace)
        {
            float weight = apos < row.Weights.Length ? row.Weights[apos] : 0f;
            coords.Add(new RouteCoordinate(row.Layer, row.Head, queryPos, apos, weight));
        }
        coords.Sort((x, y) => y.Weight.CompareTo(x.Weight));
        return coords;
    }
}