using System.Text;
using System.Text.Json;

namespace Amql.Inference.NGram;

/// <summary>
/// Next-token statistics of a corpus, counted in the model's own tokens: for every context of 1 to
/// <see cref="Order"/> − 1 tokens seen at least <c>minCount</c> times, how often each of its most
/// frequent next tokens followed it. A domain pack (.NET docs and code, say) blended into the
/// model's logits (<see cref="NGramPrior"/>) supplies what a small model cannot recall: the exact
/// spelling of a rare member name after <c>Directory.</c>, the usual next token of an idiom.
/// </summary>
public sealed class NGramPack
{
    public const string Magic = "AMQLNGR1";

    /// <summary>One context's continuations: its total count and its most frequent next tokens.</summary>
    public readonly record struct Entry(uint Total, int[] Tokens, uint[] Counts);

    private readonly Dictionary<ulong, Entry>[] _contexts;   // [length] for contexts of 1 .. Order-1 tokens

    public string Name { get; }

    /// <summary>The longest n-gram counted: contexts are up to Order − 1 tokens.</summary>
    public int Order { get; }

    /// <summary>Words that select this pack for a request (see the server's pack choice).</summary>
    public IReadOnlyList<string> Keywords { get; }

    /// <summary>What the pack was built from, and how many tokens.</summary>
    public string Source { get; }

    public long CorpusTokens { get; }

    /// <summary>The tokenizer the counts are in (vocabulary size and a hash of its pieces).</summary>
    public string Tokenizer { get; }

    public int ContextCount => _contexts.Skip(1).Sum(c => c.Count);

    private NGramPack(string name, int order, IReadOnlyList<string> keywords, string source, long corpusTokens, string tokenizer,
        Dictionary<ulong, Entry>[] contexts)
    {
        Name = name;
        Order = order;
        Keywords = keywords;
        Source = source;
        CorpusTokens = corpusTokens;
        Tokenizer = tokenizer;
        _contexts = contexts;
    }

    /// <summary>The continuations of the longest context ending the text that the pack knows, of
    /// at least <paramref name="minLength"/> tokens and seen at least <paramref name="minTotal"/>
    /// times; null when there is none.</summary>
    public (int Length, Entry Entry)? Lookup(IReadOnlyList<int> text, int minLength = 1, uint minTotal = 2)
    {
        for (int length = Math.Min(Order - 1, text.Count); length >= Math.Max(1, minLength); length--)
        {
            if (_contexts[length].TryGetValue(Hash(text, text.Count - length, length), out var entry) && entry.Total >= minTotal)
            {
                return (length, entry);
            }
        }
        return null;
    }

    /// <summary>FNV-1a over the context's tokens and its length.</summary>
    public static ulong Hash(IReadOnlyList<int> tokens, int start, int length)
    {
        ulong h = 14695981039346656037UL ^ (ulong)length;
        for (int i = start; i < start + length; i++)
        {
            uint t = (uint)tokens[i];
            for (int b = 0; b < 4; b++)
            {
                h ^= (t >> (8 * b)) & 0xff;
                h *= 1099511628211UL;
            }
        }
        return h;
    }

    /// <summary>
    /// Counts the n-grams of <paramref name="documents"/> (each a token sequence; n-grams never
    /// cross from one to the next). Contexts seen fewer than <paramref name="minCount"/> times are
    /// dropped, and each keeps its <paramref name="topK"/> most frequent next tokens.
    /// </summary>
    public static NGramPack Build(string name, IEnumerable<int[]> documents, int order, string tokenizer,
        IReadOnlyList<string>? keywords = null, string source = "", int minCount = 2, int topK = 8)
    {
        if (order < 2 || order > 8)
        {
            throw new ArgumentException("an n-gram pack's order is 2 to 8");
        }
        var pairs = new List<(ulong Context, int Next)>[order];
        for (int l = 1; l < order; l++)
        {
            pairs[l] = new List<(ulong, int)>();
        }
        long total = 0;
        foreach (var doc in documents)
        {
            total += doc.Length;
            for (int i = 1; i < doc.Length; i++)
            {
                for (int l = 1; l < order && l <= i; l++)
                {
                    pairs[l].Add((Hash(doc, i - l, l), doc[i]));
                }
            }
        }

        var contexts = new Dictionary<ulong, Entry>[order];
        contexts[0] = new();
        for (int l = 1; l < order; l++)
        {
            var list = pairs[l];
            list.Sort((a, b) => a.Context != b.Context ? a.Context.CompareTo(b.Context) : a.Next.CompareTo(b.Next));
            var dict = new Dictionary<ulong, Entry>();
            int at = 0;
            while (at < list.Count)
            {
                ulong ctx = list[at].Context;
                var nexts = new List<(int Token, uint Count)>();
                uint sum = 0;
                while (at < list.Count && list[at].Context == ctx)
                {
                    int token = list[at].Next;
                    uint n = 0;
                    while (at < list.Count && list[at].Context == ctx && list[at].Next == token)
                    {
                        n++;
                        at++;
                    }
                    nexts.Add((token, n));
                    sum += n;
                }
                if (sum >= minCount)
                {
                    var top = nexts.OrderByDescending(x => x.Count).ThenBy(x => x.Token).Take(topK).ToArray();
                    dict[ctx] = new Entry(sum, top.Select(x => x.Token).ToArray(), top.Select(x => x.Count).ToArray());
                }
            }
            contexts[l] = dict;
            pairs[l] = null!;   // let the pairs go before the next length is sorted
        }
        return new NGramPack(name, order, keywords ?? Array.Empty<string>(), source, total, tokenizer, contexts);
    }

    public void Save(string path)
    {
        using var w = new BinaryWriter(File.Create(path), Encoding.UTF8);
        w.Write(Encoding.ASCII.GetBytes(Magic));
        string header = JsonSerializer.Serialize(new Dictionary<string, object>
        {
            ["name"] = Name, ["order"] = Order, ["keywords"] = Keywords, ["source"] = Source,
            ["corpus_tokens"] = CorpusTokens, ["tokenizer"] = Tokenizer,
        });
        w.Write(header);
        for (int l = 1; l < Order; l++)
        {
            w.Write(_contexts[l].Count);
            foreach (var (ctx, e) in _contexts[l])
            {
                w.Write(ctx);
                w.Write(e.Total);
                w.Write((byte)e.Tokens.Length);
                for (int i = 0; i < e.Tokens.Length; i++)
                {
                    w.Write(e.Tokens[i]);
                    w.Write(e.Counts[i]);
                }
            }
        }
    }

    public static NGramPack Load(string path)
    {
        using var r = new BinaryReader(File.OpenRead(path), Encoding.UTF8);
        if (Encoding.ASCII.GetString(r.ReadBytes(Magic.Length)) != Magic)
        {
            throw new InvalidDataException($"{path} is not an AMQL n-gram pack");
        }
        using var header = JsonDocument.Parse(r.ReadString());
        var h = header.RootElement;
        int order = h.GetProperty("order").GetInt32();
        var contexts = new Dictionary<ulong, Entry>[order];
        contexts[0] = new();
        for (int l = 1; l < order; l++)
        {
            int count = r.ReadInt32();
            var dict = new Dictionary<ulong, Entry>(count);
            for (int c = 0; c < count; c++)
            {
                ulong ctx = r.ReadUInt64();
                uint total = r.ReadUInt32();
                int k = r.ReadByte();
                var tokens = new int[k];
                var counts = new uint[k];
                for (int i = 0; i < k; i++)
                {
                    tokens[i] = r.ReadInt32();
                    counts[i] = r.ReadUInt32();
                }
                dict[ctx] = new Entry(total, tokens, counts);
            }
            contexts[l] = dict;
        }
        return new NGramPack(h.GetProperty("name").GetString()!, order,
            h.GetProperty("keywords").EnumerateArray().Select(x => x.GetString()!).ToList(),
            h.GetProperty("source").GetString() ?? "", h.GetProperty("corpus_tokens").GetInt64(),
            h.GetProperty("tokenizer").GetString()!, contexts);
    }

    /// <summary>A tokenizer's identity for a pack: its vocabulary size and a hash of how it
    /// spells its first 4096 tokens (enough to tell tokenizers apart).</summary>
    public static string Fingerprint(int vocab, Func<int, string> piece)
    {
        ulong h = 14695981039346656037UL;
        for (int id = 0; id < Math.Min(vocab, 4096); id++)
        {
            foreach (char c in piece(id))
            {
                h ^= c;
                h *= 1099511628211UL;
            }
        }
        return $"{vocab}:{h:x16}";
    }
}

/// <summary>
/// An n-gram pack blended into a model's next-token distribution, kNN-LM style:
/// p = (1 − λ)·p_model + λ·p_pack over the tokens the pack has for the current context, with λ
/// scaled by how much evidence it has (λ_max · total / (total + <see cref="HalfCount"/>)) and only
/// for contexts of at least <see cref="MinContext"/> tokens, so a pack speaks where it knows the
/// text and is silent elsewhere. Applied in logit space: the pack's tokens are raised to
/// log((1 − λ)·p_model + λ·p_pack), which leaves every other token's relative odds alone.
/// </summary>
public sealed class NGramPrior
{
    public IReadOnlyList<NGramPack> Packs { get; }

    /// <summary>The largest share of the distribution a pack may take.</summary>
    public double Lambda { get; init; } = 0.3;

    /// <summary>A context seen this many times gets half of <see cref="Lambda"/>.</summary>
    public double HalfCount { get; init; } = 4;

    /// <summary>The shortest context (in tokens) a pack may speak from.</summary>
    public int MinContext { get; init; } = 2;

    /// <summary>The pack is silent where the model's own top token already has this much
    /// probability: it supplies what the model does not know, not a second opinion on what it does.</summary>
    public double ModelConfident { get; init; } = 1.0;

    /// <summary>The pack is silent where its own top continuation has less than this share of
    /// the context's count: a context followed by anything says nothing.</summary>
    public double PackConfident { get; init; } = 0.0;

    /// <summary>
    /// When set, the pack speaks only where the next token names a member: the text so far ends
    /// with a '.' token, or the pack's own likeliest continuation is a ".Name" token. That is where
    /// a small model's spelling fails (it invents <c>Directory.EnumerateSubdirectories</c>), while
    /// punctuation and structure, which it gets right, are left to it. The function gives a
    /// token's text.
    /// </summary>
    public Func<int, string>? MembersOnly { get; init; }

    /// <summary>Where the pack may speak: <c>code</c> (a fenced code block once its opening line is
    /// complete, or a tool call's value), or <c>all</c>. A code pack in prose does harm: blended
    /// into a fence line it turned "```csharp" into "```csharp.tm".</summary>
    public string Scope { get; init; } = "code";

    /// <summary>Whether the reply's text so far ends inside a fenced code block, past its opening line.</summary>
    public static bool InFencedCode(string text)
    {
        int fences = 0, at = 0, last = -1;
        while ((at = text.IndexOf("```", at, StringComparison.Ordinal)) >= 0)
        {
            fences++;
            last = at;
            at += 3;
        }
        return fences % 2 == 1 && text.IndexOf('\n', last) >= 0;
    }

    /// <summary>Whether the next token names a member (see <see cref="MembersOnly"/>).</summary>
    public static bool AtMember(Func<int, string> piece, IReadOnlyList<int> text, NGramPack.Entry entry)
    {
        if (text.Count > 0 && piece(text[^1]).EndsWith('.'))
        {
            return true;
        }
        string top = piece(entry.Tokens[0]);
        return top.Length > 1 && top[0] == '.' && char.IsLetter(top[1]);
    }

    public NGramPrior(IReadOnlyList<NGramPack> packs) => Packs = packs;

    /// <summary>
    /// Writes the blended logits of <paramref name="logits"/> into <paramref name="output"/> and
    /// returns true, or returns false (output untouched) when no pack knows the context.
    /// </summary>
    public bool Apply(ReadOnlySpan<float> logits, IReadOnlyList<int> text, Span<float> output)
    {
        // The pack that knows the longest context speaks; ties go to the one with more evidence.
        (int Length, NGramPack.Entry Entry)? best = null;
        foreach (var pack in Packs)
        {
            if (pack.Lookup(text, MinContext) is { } hit &&
                (best is null || hit.Length > best.Value.Length || hit.Length == best.Value.Length && hit.Entry.Total > best.Value.Entry.Total))
            {
                best = hit;
            }
        }
        if (best is not { } found)
        {
            return false;
        }
        var e = found.Entry;
        double lambda = Lambda * e.Total / (e.Total + HalfCount);
        if (lambda <= 0 || (double)e.Counts[0] / e.Total < PackConfident || MembersOnly is { } piece && !AtMember(piece, text, e))
        {
            return false;
        }

        // log Z of the model's distribution, so a pack probability can be put in logit units.
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
        double logZ = max + Math.Log(sum);
        if (1 / sum >= ModelConfident)
        {
            return false;   // the model's top token already has 1/sum of the probability
        }

        logits.CopyTo(output);
        // Every token is scaled by (1 − λ), which changes no relative odds, so only the pack's
        // tokens move: logit' = log(exp(logit) + λ/(1 − λ) · p_pack · Z).
        double boost = Math.Log(lambda / (1 - lambda)) + logZ;
        for (int i = 0; i < e.Tokens.Length; i++)
        {
            int t = e.Tokens[i];
            if ((uint)t >= (uint)output.Length)
            {
                continue;
            }
            double add = boost + Math.Log((double)e.Counts[i] / e.Total);
            double a = logits[t], hi = Math.Max(a, add);
            output[t] = (float)(hi + Math.Log(Math.Exp(a - hi) + Math.Exp(add - hi)));
        }
        return true;
    }
}
