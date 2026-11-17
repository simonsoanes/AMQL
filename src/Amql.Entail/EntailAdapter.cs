using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Amql.Entail;

/// <summary>What an output lane of an adapter means to its partner.</summary>
public enum EntailOutputKind
{
    /// <summary>This token must come next.</summary>
    Say,
    /// <summary>The answer is over: none of these tokens may come next.</summary>
    End,
    /// <summary>Only these tokens may come next.</summary>
    Only,
    /// <summary>None of these tokens may come next.</summary>
    Forbid,
    /// <summary>This token comes next if the partner gives it at least the output's threshold,
    /// or, when the output is relative, that many times the partner's prior for it.</summary>
    Prefer,
}

/// <summary>One output row of an adapter, in the partner's token ids.</summary>
public sealed record EntailAdapterOutput(int Row, EntailOutputKind Kind, IReadOnlyList<int> Tokens,
    double Threshold = 0, bool TimesPrior = false, bool Strict = false);

/// <summary>
/// Joins an adapter to its partner: which of the partner's tokens the program
/// reads, what each output row means, and which tokenizer the ids belong to.
/// </summary>
public sealed record EntailAdapterManifest(string Program, string Tokenizer,
    IReadOnlyDictionary<int, int> Rows, IReadOnlyList<EntailAdapterOutput> Outputs)
{
    public const string FileName = "adapter.json";
    private const string Format = "amql-entail-adapter-v1";

    /// <summary>True when an output weighs a token against the partner's prior, so the host
    /// must measure that prior (see <c>SideBySideSession.PriorPrompt</c>).</summary>
    public bool NeedsPrior => Outputs.Any(o => o.Kind == EntailOutputKind.Prefer && o.TimesPrior);

    /// <summary>The adapter's row for a partner token. Row 0 is a token the
    /// program doesn't know, which still takes up its place in the text.</summary>
    public int Row(int partnerToken) => Rows.GetValueOrDefault(partnerToken);

    public static string TokenizerHash(string tokenizerDir) =>
        Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(Path.Combine(tokenizerDir, "tokenizer.json"))));

    /// <summary>Refuses a partner that cuts text differently: its ids would
    /// mean other tokens, and nothing else would show it.</summary>
    public void RequirePartner(string tokenizerDir)
    {
        if (!string.Equals(TokenizerHash(tokenizerDir), Tokenizer, StringComparison.OrdinalIgnoreCase))
        {
            throw new EntailException($"adapter '{Program}' was built for a different tokenizer than the one in {tokenizerDir}");
        }
    }

    public void Save(string dir)
    {
        var rows = new JsonArray();
        foreach (var (token, row) in Rows.OrderBy(x => x.Value))
        {
            rows.Add(new JsonArray(token, row));
        }
        var outputs = new JsonArray();
        foreach (var o in Outputs)
        {
            var output = new JsonObject
            {
                ["row"] = o.Row,
                ["kind"] = o.Kind.ToString().ToLowerInvariant(),
                ["tokens"] = new JsonArray(o.Tokens.Select(t => (JsonNode?)t).ToArray()),
            };
            if (o.Kind == EntailOutputKind.Prefer)
            {
                output["threshold"] = o.Threshold;
                if (o.TimesPrior)
                {
                    output["times_prior"] = true;
                }
                if (o.Strict)
                {
                    output["strict"] = true;
                }
            }
            outputs.Add(output);
        }
        var root = new JsonObject
        {
            ["format"] = Format, ["program"] = Program, ["tokenizer"] = Tokenizer, ["rows"] = rows, ["outputs"] = outputs,
        };
        File.WriteAllText(Path.Combine(dir, FileName), root.ToJsonString());
    }

    public static EntailAdapterManifest Load(string dir)
    {
        var path = Path.Combine(dir, FileName);
        if (!File.Exists(path))
        {
            throw new EntailException($"{dir} has no {FileName}: build it with 'entail adapter'");
        }
        var root = JsonNode.Parse(File.ReadAllText(path))!.AsObject();
        if (root["format"]?.GetValue<string>() != Format)
        {
            throw new EntailException($"{path}: not a {Format} manifest");
        }
        var rows = root["rows"]!.AsArray().ToDictionary(r => r![0]!.GetValue<int>(), r => r![1]!.GetValue<int>());
        var outputs = root["outputs"]!.AsArray().Select(o => new EntailAdapterOutput(
            o!["row"]!.GetValue<int>(),
            Enum.Parse<EntailOutputKind>(o["kind"]!.GetValue<string>(), ignoreCase: true),
            o["tokens"]!.AsArray().Select(t => t!.GetValue<int>()).ToArray(),
            o["threshold"]?.GetValue<double>() ?? 0,
            o["times_prior"]?.GetValue<bool>() ?? false,
            o["strict"]?.GetValue<bool>() ?? false)).ToList();
        return new EntailAdapterManifest(root["program"]!.GetValue<string>(), root["tokenizer"]!.GetValue<string>(), rows, outputs);
    }
}

/// <summary>A manifest, with the program's side of each row.</summary>
public sealed record EntailAdapterBinding(EntailAdapterManifest Manifest, IReadOnlyList<(int Row, string Text)> Inputs,
    IReadOnlyList<(int Row, int Lane)> Outputs, List<string> Missing);

/// <summary>Who decided a token.</summary>
public enum EntailTokenSource
{
    /// <summary>The partner, unaided.</summary>
    Partner,
    /// <summary>The partner, after a rule ruled out what it preferred.</summary>
    Narrowed,
    /// <summary>A program.</summary>
    Entail,
}

/// <summary>
/// What the adapters require of the next token. Applied at the sampler, so it
/// doesn't depend on the partner's logit scale or precision.
/// </summary>
public sealed class EntailVerdict
{
    public static readonly EntailVerdict Silence = new();

    /// <summary>The token that must come next, if any. It outranks every mask.</summary>
    public int? Say { get; private set; }

    /// <summary>The tokens that may come next, or null for all of them.</summary>
    public IReadOnlySet<int>? Allowed { get; private set; }

    /// <summary>The tokens that may not come next.</summary>
    public IReadOnlySet<int> Banned { get; private set; } = new HashSet<int>();

    /// <summary>Tokens to say if the partner gives them at least their threshold, in
    /// program order; the first that qualifies is said.</summary>
    public IReadOnlyList<(int Token, double Threshold, bool TimesPrior, bool Strict)> Prefers => _prefers;
    private readonly List<(int Token, double Threshold, bool TimesPrior, bool Strict)> _prefers = new();

    /// <summary>Two adapters demanded different tokens. The first one listed won.</summary>
    public bool Conflict { get; private set; }

    public bool Silent => Say is null && Allowed is null && Banned.Count == 0 && _prefers.Count == 0;

    /// <summary>Reads an adapter's logits. Emits are in program order and the
    /// first set one counts, as in the interpreter; every set mask applies.</summary>
    public static EntailVerdict Read(EntailAdapterManifest manifest, ReadOnlySpan<float> logits)
    {
        EntailVerdict? verdict = null;
        bool emitted = false;
        foreach (var output in manifest.Outputs)
        {
            if (logits[output.Row] <= 0.5f)
            {
                continue;
            }
            verdict ??= new EntailVerdict();
            bool emit = output.Kind is EntailOutputKind.Say or EntailOutputKind.End;
            if (emit && emitted)
            {
                continue;
            }
            emitted |= emit;
            switch (output.Kind)
            {
                case EntailOutputKind.Say:
                    verdict.Say = output.Tokens[0];
                    break;
                case EntailOutputKind.Only:
                    verdict.Allow(output.Tokens);
                    break;
                case EntailOutputKind.Prefer:
                    verdict._prefers.Add((output.Tokens[0], output.Threshold, output.TimesPrior, output.Strict));
                    break;
                default:
                    verdict.Ban(output.Tokens);
                    break;
            }
        }
        return verdict ?? Silence;
    }

    /// <summary>Joins verdicts in adapter order.</summary>
    public static EntailVerdict Merge(IReadOnlyList<EntailVerdict> verdicts)
    {
        var loud = verdicts.Where(v => !v.Silent).ToList();
        if (loud.Count < 2)
        {
            return loud.Count == 1 ? loud[0] : Silence;
        }
        var merged = new EntailVerdict();
        foreach (var v in loud)
        {
            if (v.Say is { } say)
            {
                merged.Conflict |= merged.Say is { } first && first != say;
                merged.Say ??= say;
            }
            if (v.Allowed is { } allowed)
            {
                merged.Allow(allowed);
            }
            merged._prefers.AddRange(v._prefers);
            merged.Ban(v.Banned);
        }
        return merged;
    }

    /// <summary>Chooses greedily from the partner's logits. <paramref name="prior"/> is the
    /// partner's next-token distribution with the latest message blanked, for a prefer that
    /// is relative to it; without one, such a prefer stays silent.</summary>
    public (int Token, EntailTokenSource Source) Choose(ReadOnlySpan<float> logits, IReadOnlyList<double>? prior = null)
    {
        if (Say is { } say)
        {
            return (say, EntailTokenSource.Entail);
        }
        HashSet<int>? below = null;
        if (_prefers.Count > 0)
        {
            float max = float.NegativeInfinity;
            foreach (float v in logits)
            {
                max = Math.Max(max, v);
            }
            double sum = 0;
            foreach (float v in logits)
            {
                sum += Math.Exp(v - max);
            }
            foreach (var (token, threshold, timesPrior, strict) in _prefers)
            {
                if (timesPrior && prior is null)
                {
                    continue;
                }
                double needed = timesPrior ? threshold * prior![token] : threshold;
                if (Math.Exp(logits[token] - max) / sum >= needed)
                {
                    if (Permits(token))
                    {
                        return (token, EntailTokenSource.Entail);
                    }
                }
                else if (strict)
                {
                    (below ??= new HashSet<int>()).Add(token);
                }
            }
        }
        int free = 0, held = -1;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > logits[free])
            {
                free = i;
            }
            if (Permits(i) && below?.Contains(i) != true && (held < 0 || logits[i] > logits[held]))
            {
                held = i;
            }
        }
        // Rules that leave nothing can't be kept, so the partner goes free.
        if (held < 0)
        {
            Conflict = true;
            return (free, EntailTokenSource.Partner);
        }
        return (held, held == free ? EntailTokenSource.Partner : EntailTokenSource.Narrowed);
    }

    public bool Permits(int token) => (Allowed is null || Allowed.Contains(token)) && !Banned.Contains(token);

    private void Allow(IEnumerable<int> tokens)
    {
        var set = tokens.ToHashSet();
        if (Allowed is not null)
        {
            set.IntersectWith(Allowed);
        }
        Allowed = set;
    }

    private void Ban(IEnumerable<int> tokens) => Banned = Banned.Concat(tokens).ToHashSet();
}
