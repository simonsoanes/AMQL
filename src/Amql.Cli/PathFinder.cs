using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Cli;

public sealed record PathSearchOptions(int TopK = 6, int MaxNodes = 48, int MaxDepth = 6, bool Debug = false);

/// <summary>One hop of a found path: the token reached, its text, and the
/// edge cost (−log P) of the transition.</summary>
public sealed record PathHop(int TokenId, string TokenText, double EdgeCost);

public sealed record PathResult(
    bool Found,
    IReadOnlyList<PathHop> Hops,
    double TotalCost,
    double MeetingForwardCost,
    double MeetingBackwardCost,
    int Forwards,
    int NodesVisited);

/// <summary>
/// Bidirectional best-first search between two tokens over the model's
/// on-demand continuation graph. Forward: from A, node = the token chain;
/// edges = the model's top-K next tokens with cost −log P (each expansion
/// is one model forward). Backward: from B, walking reverse edges
/// accumulated from every forward expansion (the same model defines both
/// directions of each discovered edge). The search stops at the first
/// contact point — a token reachable from A and leading to B — and
/// returns the token chain WITHOUT any relation names.
///
/// The graph is genuinely context-dependent (next-token probabilities
/// condition on the whole chain), so this is Dijkstra-style best-first
/// search over an implicit graph, not a static Dĳkstra: cheapest-first
/// expansion, both ends, meet in the middle.
/// </summary>
public static class PathFinder
{
    public static PathResult Search(
        Vindex3Container container,
        string component,
        HfTokenizer tokenizer,
        int a,
        int b,
        PathSearchOptions options,
        Action<string>? progress = null,
        WeightPatch? patch = null)
    {
        if (a == b)
        {
            return new PathResult(true,
                new[] { new PathHop(a, tokenizer.TokenInfo(a).DecodedText ?? a.ToString(), 0) },
                0, 0, 0, 0, 0);
        }

        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, component, store);
        var rt = new GenericRuntime(plan, store, patch);

        var fwdQueue = new PriorityQueue<ChainEntry, double>();
        var bwdQueue = new PriorityQueue<ChainEntry, double>();
        var fwdCost = new Dictionary<string, double>();
        var bwdCost = new Dictionary<string, double>();
        var revCache = new Dictionary<int, List<(int From, double Cost)>>();

        // Endpoints of the other side, keyed by the meeting token.
        var fwdEnds = new Dictionary<int, ChainEntry>(); // chain id → chain whose LAST token is id
        var bwdEnds = new Dictionary<int, ChainEntry>(); // chain id → chain whose FIRST token is id

        string Key(int[] chain) => string.Join('\u0001', chain);

        var start = NewEntry(new[] { a }, new[] { 0.0 });
        fwdQueue.Enqueue(start, 0);
        fwdCost[Key(start.Chain)] = 0;
        fwdEnds[a] = start;

        var bStart = NewEntry(new[] { b }, new[] { 0.0 });
        bwdQueue.Enqueue(bStart, 0);
        bwdCost[Key(bStart.Chain)] = 0;
        bwdEnds[b] = bStart;

        // A fixed pool of grammatical bridges, scored with their true model
        // probabilities — without them the cheapest-first frontier drowns
        // in punctuation highways (top continuations of a bare token are
        // often ",", ".", ":").
        var bridges = ResolveBridges(tokenizer);

        int forwards = 0;
        int visited = 0;

        while (forwards < options.MaxNodes)
        {
            bool expandFwd = bwdQueue.Count == 0 ||
                             (fwdQueue.TryPeek(out _, out var fPrio) && bwdQueue.TryPeek(out _, out var bPrio) && fPrio <= bPrio);
            if (expandFwd)
            {
                if (!fwdQueue.TryDequeue(out var entry, out _))
                {
                    break;
                }
                if (fwdCost.TryGetValue(Key(entry.Chain), out var known) && known < entry.Cost)
                {
                    continue;
                }
                var contact = TryContact(entry, fwdEnds, bwdEnds, forwardSide: true);
                if (contact is not null)
                {
                    return Build(entry, contact.Bwd, contact.MeetingFwdCost, contact.MeetingBwdCost, forwards, visited, tokenizer);
                }
                if (entry.Chain.Length >= options.MaxDepth)
                {
                    continue;
                }
                progress?.Invoke($"·f{forwards}");
                forwards++;
                visited++;
                var candidates = ExpandCandidates(rt, plan, entry.Chain, options.TopK, bridges, tokenizer);
                if (options.Debug && progress is not null)
                {
                    string chainText = string.Join(" ", entry.Chain.Select(t => tokenizer.TokenInfo(t).DecodedText ?? t.ToString()));
                    string candidatesText = string.Join(" | ", candidates.Select(c => $"{tokenizer.TokenInfo(c.Id).DecodedText ?? c.Id.ToString()} {c.Prob:0.0000}"));
                    progress($"\n  fwd [{chainText}]  →  [{candidatesText}]");
                }
                foreach (var (id, prob) in candidates)
                {
                    double edge = -Math.Log(Math.Max(prob, 1e-9));
                    var chain = entry.Chain.Append(id).ToArray();
                    var costs = entry.CumCosts.Concat(new[] { entry.Cost + edge }).ToArray();
                    var next = NewEntry(chain, costs);
                    string key = Key(chain);
                    if (fwdCost.TryGetValue(key, out var k2) && k2 <= next.Cost)
                    {
                        continue;
                    }
                    fwdCost[key] = next.Cost;
                    revCache[id] = revCache.TryGetValue(id, out var rev) ? rev : new List<(int, double)>();
                    revCache[id].Add((entry.Chain[^1], edge));
                    var contact2 = TryContact(next, fwdEnds, bwdEnds, forwardSide: true);
                    if (contact2 is not null)
                    {
                        return Build(next, contact2.Bwd, contact2.MeetingFwdCost, contact2.MeetingBwdCost, forwards, visited, tokenizer);
                    }
                    fwdQueue.Enqueue(next, next.Cost);
                    if (!fwdEnds.ContainsKey(id))
                    {
                        fwdEnds[id] = next;
                    }
                }
            }
            else
            {
                if (!bwdQueue.TryDequeue(out var entry, out _))
                {
                    continue;
                }
                if (bwdCost.TryGetValue(Key(entry.Chain), out var known) && known < entry.Cost)
                {
                    continue;
                }
                var contact = TryContact(entry, bwdEnds, fwdEnds, forwardSide: false);
                if (contact is not null)
                {
                    return Build(contact.Fwd, entry, contact.MeetingFwdCost, contact.MeetingBwdCost, forwards, visited, tokenizer);
                }
                if (revCache.TryGetValue(entry.Chain[0], out var revs))
                {
                    foreach (var (from, edge) in revs)
                    {
                        var chain = new[] { from }.Concat(entry.Chain).ToArray();
                        var costs = new[] { entry.Cost + edge }.Concat(entry.CumCosts).ToArray();
                        var next = NewEntry(chain, costs);
                        string key = Key(chain);
                        if (bwdCost.TryGetValue(key, out var k2) && k2 <= next.Cost)
                        {
                            continue;
                        }
                        bwdCost[key] = next.Cost;
                        var contact2 = TryContact(next, bwdEnds, fwdEnds, forwardSide: false);
                        if (contact2 is not null)
                        {
                            return Build(contact2.Fwd, next, contact2.MeetingFwdCost, contact2.MeetingBwdCost, forwards, visited, tokenizer);
                        }
                        bwdQueue.Enqueue(next, next.Cost);
                        bwdEnds[chain[0]] = next;
                    }
                }
            }
            visited++;
        }

        return new PathResult(false, Array.Empty<PathHop>(), 0, 0, 0, forwards, visited);
    }

    private sealed record ChainEntry(int[] Chain, double[] CumCosts)
    {
        public double Cost => CumCosts[^1];
    }

    private sealed record Contact(ChainEntry Fwd, ChainEntry Bwd, double MeetingFwdCost, double MeetingBwdCost);

    private static ChainEntry NewEntry(int[] chain, double[] costs) => new(chain, costs);

    /// <summary>Checks whether <c>entry</c> meets the other side: a token
    /// that is the meeting point of both directions. <c>entry</c> is on the
    /// Fwd side (last token) or Bwd side (first token); <c>mine</c> is the
    /// endpoint map of my side and <c>theirs</c> of the opposite side.</summary>
    private static Contact? TryContact(
        ChainEntry entry,
        Dictionary<int, ChainEntry> mine,
        Dictionary<int, ChainEntry> theirs,
        bool forwardSide)
    {
        int token = forwardSide ? entry.Chain[^1] : entry.Chain[0];
        if (theirs.TryGetValue(token, out var other))
        {
            return forwardSide
                ? new Contact(entry, other, entry.Cost, other.Cost)
                : new Contact(other, entry, other.Cost, entry.Cost);
        }
        return null;
    }

    private static PathResult Build(
        ChainEntry fwd,
        ChainEntry bwd,
        double meetingFwd,
        double meetingBwd,
        int forwards,
        int visited,
        HfTokenizer tokenizer)
    {
        // fwd.Chain = A → … → X ; bwd.Chain = X → … → B.
        var hops = new List<PathHop>();
        for (int i = 0; i < fwd.Chain.Length; i++)
        {
            double edge = i == 0 ? 0 : fwd.CumCosts[i] - fwd.CumCosts[i - 1];
            hops.Add(new PathHop(fwd.Chain[i], Txt(tokenizer, fwd.Chain[i]), edge));
        }
        for (int i = 1; i < bwd.Chain.Length; i++)
        {
            double edge = bwd.CumCosts[i] - bwd.CumCosts[i - 1];
            hops.Add(new PathHop(bwd.Chain[i], Txt(tokenizer, bwd.Chain[i]), edge));
        }
        return new PathResult(true, hops, meetingFwd + meetingBwd, meetingFwd, meetingBwd, forwards, visited);
    }

    private static string Txt(HfTokenizer tokenizer, int id) =>
        tokenizer.TokenInfo(id).DecodedText ?? id.ToString();

    private static readonly string[] BridgeWords =
    {
        " is", " of", " the", " and", " in", " a", " 's", " to", " for",
        " on", " with", " it", " as", " by", " from", " at", " was", " has",
    };

    /// <summary>Resolves the bridge tokens once (their ids depend only on
    /// the tokenizer), dropping any the vocab lacks (or cannot encode —
    /// e.g. the demo alphabet cannot spell " is").</summary>
    private static List<int> ResolveBridges(HfTokenizer tokenizer)
    {
        var ids = new List<int>();
        foreach (var word in BridgeWords)
        {
            try
            {
                var encoded = tokenizer.EncodeToIds(word);
                if (encoded.Count == 1)
                {
                    ids.Add(encoded[0]);
                }
            }
            catch (TokenizerException)
            {
                // not in this vocabulary — skip the bridge
            }
        }
        return ids;
    }

    private static bool IsWordLike(HfTokenizer tokenizer, int id)
    {
        var text = tokenizer.TokenInfo(id).DecodedText;
        if (string.IsNullOrEmpty(text))
        {
            return false;
        }
        foreach (var ch in text)
        {
            if (char.IsLetterOrDigit(ch))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>A forward expansion's candidate set: the top-k word-like
    /// continuations plus the bridge tokens, each scored with its true
    /// model probability, punctuation excluded (it is a dead highway for
    /// connection finding).</summary>
    private static List<(int Id, double Prob)> ExpandCandidates(
        GenericRuntime rt, ComponentOpPlan plan, int[] chain, int k,
        List<int> bridges, HfTokenizer tokenizer)
    {
        var logits = CausalTracer.RunPositionMajor(rt, plan, chain);
        var row = logits.Row(logits.Rows - 1);
        double max = double.NegativeInfinity;
        for (int i = 0; i < row.Length; i++)
        {
            if (row[i] > max)
            {
                max = row[i];
            }
        }
        var probs = new double[row.Length];
        double sum = 0;
        for (int i = 0; i < row.Length; i++)
        {
            probs[i] = Math.Exp(row[i] - max);
            sum += probs[i];
        }
        double inv = 1.0 / sum;

        var order = Enumerable.Range(0, row.Length).ToArray();
        Array.Sort(order, (x, y) => probs[y].CompareTo(probs[x]));

        var chosen = new List<(int Id, double Prob)>();
        var seen = new HashSet<int>();
        foreach (var id in order)
        {
            if (!IsWordLike(tokenizer, id))
            {
                continue;
            }
            if (chosen.Count >= k)
            {
                break;
            }
            chosen.Add((id, probs[id] * inv));
            seen.Add(id);
        }
        foreach (var id in bridges)
        {
            if (!seen.Contains(id) && IsWordLike(tokenizer, id))
            {
                chosen.Add((id, probs[id] * inv));
            }
        }
        return chosen;
    }
}