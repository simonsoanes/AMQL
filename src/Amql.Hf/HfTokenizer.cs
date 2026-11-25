using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Amql.Hf.Tokenizers;

namespace Amql.Hf;

/// <summary>Raised on malformed tokenizer files or unencodable text — the
/// tokenizer's fail-closed counterpart of <see cref="ModelConfigException"/>.</summary>
public sealed class TokenizerException : Exception
{
    public TokenizerException(string message) : base(message) { }

    public TokenizerException(string message, Exception inner) : base(message, inner) { }
}

/// <summary>One added (special) token of the tokenizer file.
/// <c>Lstrip</c>/<c>Rstrip</c> mean the token absorbs the whitespace on that
/// side of it (ModernBERT's <c>[MASK]</c> is lstrip: "a [MASK]" is "a" + [MASK]).</summary>
public sealed record AddedToken(int Id, string Content, bool IsSpecial, bool Lstrip = false, bool Rstrip = false);

/// <summary>One encoded piece: the produced id plus its representations —
/// the byte-level form (the "Ġ"-style vocabulary string) and the decoded
/// human-readable text it contributes.</summary>
public sealed record TokenPiece(int Id, string? Representation, string? DecodedText, bool IsSpecial);

public sealed record TokenizationResult(IReadOnlyList<TokenPiece> Pieces, IReadOnlyList<int> Ids)
{
    /// <summary>The concatenated decoded text of the pieces.</summary>
    public string ToDecodedText() => string.Concat(Pieces.Select(p => p.DecodedText ?? string.Empty));
}

/// <summary>
/// An HF <c>tokenizer.json</c> reader for the byte-level BPE family: NFC
/// normalisation, the configured regex pre-tokenizer (the GPT-2-style
/// split), added/special-token carving, greedy BPE over the byte-level
/// characters, and ByteLevel decoding. Also the SentencePiece-style BPE family
/// (Gemma): spaces replaced by "▁", no split, BPE over characters with a byte
/// fallback (<c>&lt;0x0A&gt;</c>) for those the vocabulary lacks, and the matching
/// decoder. Unsupported pipeline stages refuse the file by name — the
/// tokenizer is only trusted when every stage is one it has judged.
/// </summary>
public sealed class HfTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<(string A, string B), int> _mergeRanks;
    private readonly Dictionary<string, AddedToken> _addedByContent;
    private readonly IReadOnlyList<AddedToken> _addedTokens;
    private readonly Regex _splitRegex;
    private readonly HashSet<int> _specialIds;
    private readonly string[] _tokensById;
    private readonly (int[] Prefix, int[] Suffix)? _singleTemplate;
    private readonly Dictionary<int, AddedToken> _addedById;

    /// <summary>SentencePiece-style: what a space is written as ("▁"); null for byte-level BPE.</summary>
    private readonly string? _spaceMark;

    private HfTokenizer(
        Dictionary<string, int> vocab,
        Dictionary<(string, string), int> mergeRanks,
        Dictionary<string, AddedToken> addedByContent,
        IReadOnlyList<AddedToken> addedTokens,
        Regex splitRegex,
        string[] tokensById,
        HashSet<int> specialIds,
        (int[] Prefix, int[] Suffix)? singleTemplate,
        string? spaceMark)
    {
        _spaceMark = spaceMark;
        _singleTemplate = singleTemplate;
        _addedById = new Dictionary<int, AddedToken>();
        foreach (var t in addedTokens)
        {
            _addedById[t.Id] = t;
        }
        _vocab = vocab;
        _mergeRanks = mergeRanks;
        _addedByContent = addedByContent;
        _addedTokens = addedTokens;
        _splitRegex = splitRegex;
        _tokensById = tokensById;
        _specialIds = specialIds;
    }

    public int VocabSize => _vocab.Count;

    public IReadOnlyList<AddedToken> AddedTokens => _addedTokens;

    // ── loading ───────────────────────────────────────────────────────────

    /// <summary>Loads <c>tokenizer.json</c> from a model directory.</summary>
    public static HfTokenizer FromModelDir(string modelDir)
    {
        var path = Path.Combine(modelDir, "tokenizer.json");
        if (!File.Exists(path))
        {
            throw new TokenizerException(
                $"'{modelDir}' has no tokenizer.json — only the HF tokenizers format is supported by this build");
        }
        return FromTokenizerFile(path);
    }

    public static HfTokenizer FromTokenizerFile(string path)
    {
        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(File.ReadAllBytes(path));
        }
        catch (JsonException e)
        {
            throw new TokenizerException($"'{path}' is not valid JSON: {e.Message}", e);
        }

        using (doc)
        {
            var root = doc.RootElement;
            var model = root.GetProperty("model");
            if (model.GetProperty("type").GetString() != "BPE")
            {
                throw new TokenizerException($"'{path}': model type '{model.GetProperty("type").GetString()}' is not the BPE this build serves");
            }

            // Vocab + merges.
            var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var entry in model.GetProperty("vocab").EnumerateObject())
            {
                vocab[entry.Name] = entry.Value.GetInt32();
            }
            var mergeRanks = new Dictionary<(string, string), int>();
            // One pass with an enumerator: JsonElement's array indexer walks
            // from the start for arrays of arrays (the ≥ 0.21 pair form), so
            // merges[i] in a loop is quadratic — minutes for a 247k-merge file.
            int rank = -1;
            foreach (var entry in model.GetProperty("merges").EnumerateArray())
            {
                rank++;
                // tokenizers ≤ 0.20 writes a merge as one "left right" string;
                // ≥ 0.21 (Granite's file) writes the pair as a two-element
                // array. Joining the parts with the separator reproduces the
                // string form exactly, so both rank identically.
                string pair;
                if (entry.ValueKind == JsonValueKind.Array)
                {
                    if (entry.GetArrayLength() != 2)
                    {
                        throw new TokenizerException(
                            $"'{path}': merge entry {rank} is an array of {entry.GetArrayLength()} parts, expected 2");
                    }
                    pair = $"{entry[0].GetString()} {entry[1].GetString()}";
                }
                else
                {
                    pair = entry.GetString() ?? throw new TokenizerException($"'{path}': null merge entry at {rank}");
                }
                int splitAt = pair.LastIndexOf(' ');
                if (splitAt <= 0)
                {
                    throw new TokenizerException($"'{path}': malformed merge '{pair}'");
                }
                mergeRanks[(pair[..splitAt], pair[(splitAt + 1)..])] = rank;
            }

            // Added / special tokens.
            var addedTokens = new List<AddedToken>();
            var addedByContent = new Dictionary<string, AddedToken>(StringComparer.Ordinal);
            if (root.TryGetProperty("added_tokens", out var added) && added.ValueKind == JsonValueKind.Array)
            {
                foreach (var entry in added.EnumerateArray())
                {
                    int id = entry.GetProperty("id").GetInt32();
                    string content = entry.GetProperty("content").GetString() ?? string.Empty;
                    bool special = entry.TryGetProperty("special", out var sp) && sp.GetBoolean();
                    bool lstrip = entry.TryGetProperty("lstrip", out var ls) && ls.GetBoolean();
                    bool rstrip = entry.TryGetProperty("rstrip", out var rs) && rs.GetBoolean();
                    var token = new AddedToken(id, content, special, lstrip, rstrip);
                    addedTokens.Add(token);
                    addedByContent[content] = token;
                }
            }

            // SentencePiece-style BPE: every stage must be the one that goes with
            // the others, or the file is refused below by the stage that is not.
            string? spaceMark = ReadSpaceMark(root, model, path);

            // Normalizer: only NFC is judged.
            if (spaceMark is null && root.TryGetProperty("normalizer", out var normalizer) &&
                normalizer.ValueKind == JsonValueKind.Object &&
                normalizer.GetProperty("type").GetString() is { } normType &&
                normType != "NFC")
            {
                throw new TokenizerException($"'{path}': normalizer '{normType}' is not the NFC this build serves");
            }

            // Pre-tokenizer: Sequence(Split(regex), ByteLevel) or bare ByteLevel.
            var splitRegex = spaceMark is null ? BuildSplitRegex(root) : NoSplit;

            // Decoder: only ByteLevel is judged.
            if (spaceMark is null && root.TryGetProperty("decoder", out var decoder) &&
                decoder.GetProperty("type").GetString() is { } decoderType &&
                decoderType != "ByteLevel")
            {
                throw new TokenizerException($"'{path}': decoder '{decoderType}' is not the ByteLevel this build serves");
            }

            // Reverse index for token text lookup. Added tokens live
            // OUTSIDE the BPE vocab (ids ≥ vocab size), so the table spans
            // the union of both id ranges.
            int maxId = Math.Max(
                vocab.Count > 0 ? vocab.Values.Max() : -1,
                addedTokens.Count > 0 ? addedTokens.Max(a => a.Id) : -1);
            var tokensById = new string[maxId + 1];
            foreach (var (token, id) in vocab)
            {
                tokensById[id] = token;
            }
            var specialIds = new HashSet<int>(addedTokens.Where(a => a.IsSpecial).Select(a => a.Id));

            return new HfTokenizer(vocab, mergeRanks, addedByContent, addedTokens, splitRegex, tokensById, specialIds,
                ReadSingleTemplate(root, path), spaceMark);
        }
    }

    private static readonly Regex NoSplit = new(".+", RegexOptions.Singleline | RegexOptions.CultureInvariant);

    /// <summary>The space mark of a SentencePiece-style file: normalizer
    /// <c>Replace " " → mark</c>, a pre-tokenizer that splits only on spaces (there
    /// are none left) or none, byte fallback, and the decoder that reverses it.
    /// Null when the normalizer is not that one.</summary>
    private static string? ReadSpaceMark(JsonElement root, JsonElement model, string path)
    {
        static string? Type(JsonElement e) => e.ValueKind == JsonValueKind.Object ? e.GetProperty("type").GetString() : null;
        static string? Literal(JsonElement e) =>
            e.TryGetProperty("pattern", out var p) && p.TryGetProperty("String", out var s) ? s.GetString() : null;

        if (!root.TryGetProperty("normalizer", out var normalizer) || Type(normalizer) != "Replace" || Literal(normalizer) != " ")
        {
            return null;
        }
        string mark = normalizer.GetProperty("content").GetString() ?? "";
        if (mark.Length == 0)
        {
            throw new TokenizerException($"'{path}': the normalizer deletes spaces, which is not served by this build");
        }
        if (root.TryGetProperty("pre_tokenizer", out var pre) && pre.ValueKind != JsonValueKind.Null &&
            (Type(pre) != "Split" || Literal(pre) != " "))
        {
            throw new TokenizerException($"'{path}': pre-tokenizer '{Type(pre)}' over a '{mark}' normalizer is not served by this build");
        }
        if (!model.TryGetProperty("byte_fallback", out var fallback) || fallback.ValueKind != JsonValueKind.True)
        {
            throw new TokenizerException($"'{path}': a '{mark}' normalizer without byte fallback is not served by this build");
        }
        foreach (var key in new[] { "continuing_subword_prefix", "end_of_word_suffix", "dropout" })
        {
            if (model.TryGetProperty(key, out var v) && v.ValueKind != JsonValueKind.Null &&
                !(v.ValueKind == JsonValueKind.String && v.GetString() == ""))
            {
                throw new TokenizerException($"'{path}': BPE with {key} is not served by this build");
            }
        }
        bool reversed = root.TryGetProperty("decoder", out var decoder) && Type(decoder) == "Sequence";
        if (reversed)
        {
            var stages = decoder.GetProperty("decoders").EnumerateArray().ToList();
            reversed = stages.Count == 3 && Type(stages[0]) == "Replace" && Literal(stages[0]) == mark &&
                       stages[0].GetProperty("content").GetString() == " " && Type(stages[1]) == "ByteFallback" && Type(stages[2]) == "Fuse";
        }
        return reversed
            ? mark
            : throw new TokenizerException($"'{path}': the decoder is not Replace, ByteFallback, Fuse, which is the one that goes with a '{mark}' normalizer");
    }

    /// <summary>Reads a <c>TemplateProcessing</c> post-processor's single-sequence
    /// template as the special-token ids around <c>$A</c> (BERT-style
    /// <c>[CLS] $A [SEP]</c>). Other post-processors add nothing to the ids and
    /// yield no template.</summary>
    private static (int[] Prefix, int[] Suffix)? ReadSingleTemplate(JsonElement root, string path)
    {
        if (!root.TryGetProperty("post_processor", out var post) || post.ValueKind != JsonValueKind.Object ||
            post.GetProperty("type").GetString() != "TemplateProcessing")
        {
            return null;
        }
        var prefix = new List<int>();
        var suffix = new List<int>();
        bool seenSequence = false;
        var specials = post.GetProperty("special_tokens");
        foreach (var item in post.GetProperty("single").EnumerateArray())
        {
            if (item.TryGetProperty("Sequence", out _))
            {
                seenSequence = true;
                continue;
            }
            string name = item.GetProperty("SpecialToken").GetProperty("id").GetString()!;
            var ids = specials.GetProperty(name).GetProperty("ids").EnumerateArray().Select(e => e.GetInt32());
            (seenSequence ? suffix : prefix).AddRange(ids);
        }
        if (!seenSequence)
        {
            throw new TokenizerException($"'{path}': TemplateProcessing single template has no $A sequence");
        }
        return (prefix.ToArray(), suffix.ToArray());
    }

    /// <summary>Whether the file declares special tokens to wrap a single
    /// sequence in (see <see cref="EncodeWithSpecialTokens"/>).</summary>
    public bool HasSingleTemplate => _singleTemplate is not null;

    /// <summary>Encodes and applies the post-processor's single-sequence template —
    /// what HF's <c>tokenizer(text)</c> returns with <c>add_special_tokens=True</c>.</summary>
    public int[] EncodeWithSpecialTokens(string text)
    {
        var (prefix, suffix) = _singleTemplate
            ?? throw new TokenizerException("this tokenizer declares no TemplateProcessing post-processor");
        var body = EncodeToIds(text);
        var ids = new int[prefix.Length + body.Count + suffix.Length];
        prefix.CopyTo(ids, 0);
        for (int i = 0; i < body.Count; i++)
        {
            ids[prefix.Length + i] = body[i];
        }
        suffix.CopyTo(ids, prefix.Length + body.Count);
        return ids;
    }

    /// <summary>
    /// What HF's <c>tokenizer(text, truncation=True, max_length=n)</c> returns:
    /// the body, truncated from the right so that it and the post-processor's
    /// special tokens fit in <paramref name="maxLength"/>, then wrapped by the
    /// single-sequence template when the tokenizer declares one (a ByteLevel
    /// post-processor, as Qwen's, adds nothing). Reports whether it truncated.
    /// </summary>
    public int[] EncodeForModel(string text, int? maxLength, out bool truncated)
    {
        var (prefix, suffix) = _singleTemplate ?? (Array.Empty<int>(), Array.Empty<int>());
        var body = EncodeToIds(text);
        int keep = body.Count;
        truncated = false;
        if (maxLength is { } max)
        {
            int room = max - prefix.Length - suffix.Length;
            if (room < 1)
            {
                throw new TokenizerException($"max length {max} leaves no room beside {prefix.Length + suffix.Length} special tokens");
            }
            if (keep > room)
            {
                keep = room;
                truncated = true;
            }
        }
        var ids = new int[prefix.Length + keep + suffix.Length];
        prefix.CopyTo(ids, 0);
        for (int i = 0; i < keep; i++)
        {
            ids[prefix.Length + i] = body[i];
        }
        suffix.CopyTo(ids, prefix.Length + keep);
        return ids;
    }

    /// <summary>The canonical GPT-2 split regex tokenizers applies under
    /// ByteLevel(use_regex=true) — the pre-tokenizer Granite's file declares.</summary>
    private const string Gpt2SplitPattern =
        @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

    private static Regex BuildSplitRegex(JsonElement root)
    {
        string? pattern = null;
        if (root.TryGetProperty("pre_tokenizer", out var pre) && pre.ValueKind == JsonValueKind.Object)
        {
            switch (pre.GetProperty("type").GetString())
            {
                case "ByteLevel":
                    // A bare ByteLevel with use_regex=true IS the GPT-2
                    // pre-tokenizer: tokenizers applies the canonical GPT-2
                    // split regex under that flag (Granite's tokenizer.json
                    // is exactly this shape). Without the flag there is no
                    // split of our own to judge.
                    if (pre.TryGetProperty("use_regex", out var useRegex) && useRegex.GetBoolean())
                    {
                        pattern = Gpt2SplitPattern;
                    }
                    break;
                case "Sequence":
                {
                    foreach (var child in pre.GetProperty("pretokenizers").EnumerateArray())
                    {
                        if (child.GetProperty("type").GetString() == "Split")
                        {
                            var regexElement = child.GetProperty("pattern").GetProperty("Regex");
                            pattern = regexElement.GetString();
                            // Only Isolated non-inverted splits are judged.
                            if (child.TryGetProperty("behavior", out var behavior) && behavior.GetString() != "Isolated")
                            {
                                throw new TokenizerException(
                                    $"pre-tokenizer split behavior '{behavior.GetString()}' is not the 'Isolated' this build serves");
                            }
                            if (child.TryGetProperty("invert", out var invert) && invert.GetBoolean())
                            {
                                throw new TokenizerException("pre-tokenizer split with invert=true is not served by this build");
                            }
                        }
                    }
                    break;
                }
                case var other when other is not null:
                    throw new TokenizerException($"pre-tokenizer '{other}' is not served by this build");
            }
        }
        return pattern is null ? new Regex("\\S+", RegexOptions.Compiled) : new Regex(pattern, RegexOptions.Compiled | RegexOptions.CultureInvariant);
    }

    // ── encoding ──────────────────────────────────────────────────────────

    public TokenizationResult Encode(string text)
    {
        if (_spaceMark is not null)
        {
            return EncodeMarked(text);
        }
        // Stage 1: NF C normalisation (the "NFC" normalizer).
        string normalized = text.Normalize(NormalizationForm.FormC);

        var pieces = new List<TokenPiece>();
        // Stage 2: added tokens are carved at their start positions
        // (longest content first); the runs between them are regex-split
        // (the configured split, Isolated behavior: matches are the pieces).
        foreach (var (start, end, token) in Scatter(normalized))
        {
            if (token is not null)
            {
                pieces.Add(PieceFor(token, token.Content));
            }
            else
            {
                foreach (Match match in _splitRegex.Matches(normalized.Substring(start, end - start)))
                {
                    string byteLevel = ByteLevel.EncodeWord(match.Value);
                    foreach (var id in BpeToIds(byteLevel, match.Value))
                    {
                        pieces.Add(PieceFor(id));
                    }
                }
            }
        }
        return new TokenizationResult(pieces, pieces.Select(p => p.Id).ToArray());
    }

    public IReadOnlyList<int> EncodeToIds(string text) => Encode(text).Ids;

    /// <summary>SentencePiece-style: added tokens are carved from the text as it is,
    /// and each run between them is one word of the BPE, its spaces marked.</summary>
    private TokenizationResult EncodeMarked(string text)
    {
        var pieces = new List<TokenPiece>();
        foreach (var (start, end, token) in Scatter(text))
        {
            if (token is not null)
            {
                pieces.Add(PieceFor(token, token.Content));
                continue;
            }
            foreach (int id in BpeRun(text.Substring(start, end - start).Replace(" ", _spaceMark)))
            {
                pieces.Add(PieceFor(id));
            }
        }
        return new TokenizationResult(pieces, pieces.Select(p => p.Id).ToArray());
    }

    /// <summary>BPE over a run of any length: the lowest-ranked merge first, the
    /// leftmost of equals, from a queue rather than a scan (a document is one run).</summary>
    private List<int> BpeRun(string run)
    {
        var tokens = new List<string>(run.Length);
        foreach (var rune in run.EnumerateRunes())
        {
            string ch = rune.ToString();
            if (_vocab.ContainsKey(ch))
            {
                tokens.Add(ch);
                continue;
            }
            foreach (byte b in Encoding.UTF8.GetBytes(ch))
            {
                tokens.Add($"<0x{b:X2}>");
            }
        }
        int n = tokens.Count;
        var next = new int[n];
        var previous = new int[n];
        var alive = new bool[n];
        for (int i = 0; i < n; i++)
        {
            next[i] = i + 1;
            previous[i] = i - 1;
            alive[i] = true;
        }
        // Each entry keeps the very strings it was queued for, so one whose tokens
        // have since merged is told apart.
        var queue = new PriorityQueue<(int At, string Left, string Right), (int Rank, int At)>();
        void Offer(int at)
        {
            if (at >= 0 && next[at] < n && _mergeRanks.TryGetValue((tokens[at], tokens[next[at]]), out int rank))
            {
                queue.Enqueue((at, tokens[at], tokens[next[at]]), (rank, at));
            }
        }
        for (int i = 0; i < n; i++)
        {
            Offer(i);
        }
        while (queue.TryDequeue(out var entry, out _))
        {
            int at = entry.At;
            if (!alive[at] || next[at] >= n || !ReferenceEquals(tokens[at], entry.Left) || !ReferenceEquals(tokens[next[at]], entry.Right))
            {
                continue;
            }
            int gone = next[at];
            tokens[at] = entry.Left + entry.Right;
            alive[gone] = false;
            next[at] = next[gone];
            if (next[at] < n)
            {
                previous[next[at]] = at;
            }
            Offer(previous[at]);
            Offer(at);
        }
        var ids = new List<int>();
        for (int i = 0; i < n; i = next[i])
        {
            ids.Add(_vocab.TryGetValue(tokens[i], out int id)
                ? id
                : throw new TokenizerException($"piece '{tokens[i]}' is not in the vocabulary"));
        }
        return ids;
    }

    /// <summary>The bytes a SentencePiece-style token stands for.</summary>
    private byte[] MarkedBytes(string token) =>
        token.Length == 6 && token.StartsWith("<0x", StringComparison.Ordinal) && token[5] == '>' &&
        byte.TryParse(token.AsSpan(3, 2), System.Globalization.NumberStyles.AllowHexSpecifier, System.Globalization.CultureInfo.InvariantCulture, out byte b)
            ? [b]
            : Encoding.UTF8.GetBytes(token.Replace(_spaceMark!, " "));

    /// <summary>
    /// Walks the string once, carving added tokens leftmost-longest the way
    /// tokenizers does; the plain runs between them go to the regex split. An
    /// <c>lstrip</c> token's match may begin on the whitespace before its
    /// content and an <c>rstrip</c> token's may run over the whitespace after
    /// it, so that whitespace belongs to the token — and because the longest
    /// match wins, "  [MASK]" beats a two-space added token that starts at the
    /// same place (ModernBERT registers such whitespace tokens).
    /// </summary>
    private List<(int Start, int End, AddedToken? Token)> Scatter(string text)
    {
        var runs = new List<(int, int, AddedToken?)>();
        var ordered = _addedTokens.OrderByDescending(t => t.Content.Length).ToArray();
        int plainStart = 0;
        int i = 0;
        while (i < text.Length)
        {
            if (MatchAt(text, i, ordered, out var token, out int matchEnd))
            {
                if (i > plainStart)
                {
                    runs.Add((plainStart, i, null));
                }
                runs.Add((i, matchEnd, token));
                i = matchEnd;
                plainStart = i;
            }
            else
            {
                i++;
            }
        }
        if (text.Length > plainStart)
        {
            runs.Add((plainStart, text.Length, null));
        }
        return runs;
    }

    private static bool MatchAt(string text, int index, AddedToken[] ordered, out AddedToken token, out int end)
    {
        token = null!;
        end = -1;
        foreach (var candidate in ordered)
        {
            if (candidate.Content.Length == 0)
            {
                continue;
            }
            int at = index;
            if (candidate.Lstrip)
            {
                while (at < text.Length && char.IsWhiteSpace(text[at]))
                {
                    at++;
                }
            }
            if (at + candidate.Content.Length > text.Length ||
                string.CompareOrdinal(text, at, candidate.Content, 0, candidate.Content.Length) != 0)
            {
                continue;
            }
            int e = at + candidate.Content.Length;
            if (candidate.Rstrip)
            {
                while (e < text.Length && char.IsWhiteSpace(text[e]))
                {
                    e++;
                }
            }
            if (e > end)
            {
                token = candidate;
                end = e;
            }
        }
        return end > index;
    }

    // ── BPE ────────────────────────────────────────────────────────────────

    private int[] BpeToIds(string byteLevel, string originalWord)
    {
        // Initial segmentation: one byte-level character per byte.
        var tokens = new List<string>(byteLevel.Length);
        foreach (var ch in byteLevel)
        {
            tokens.Add(ch.ToString());
        }

        // Greedy merges: always merge the pair with the lowest merge index.
        while (tokens.Count > 1)
        {
            int bestRank = int.MaxValue;
            int bestIndex = -1;
            for (int i = 0; i < tokens.Count - 1; i++)
            {
                if (_mergeRanks.TryGetValue((tokens[i], tokens[i + 1]), out var rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }
            if (bestIndex < 0)
            {
                break;
            }
            string merged = tokens[bestIndex] + tokens[bestIndex + 1];
            tokens[bestIndex] = merged;
            tokens.RemoveAt(bestIndex + 1);
        }

        var ids = new int[tokens.Count];
        for (int i = 0; i < tokens.Count; i++)
        {
            if (!_vocab.TryGetValue(tokens[i], out var id))
            {
                throw new TokenizerException(
                    $"cannot tokenize '{originalWord}': byte-level piece '{tokens[i]}' is not in the vocabulary " +
                    "(the string needs characters unknown to this tokenizer)");
            }
            ids[i] = id;
        }
        return ids;
    }

    // ── pieces / decoding ──────────────────────────────────────────────────

    private TokenPiece PieceFor(int id)
    {
        // Added tokens are judged FIRST: they live outside the BPE vocab and
        // always decode to their content — special or not (ModernBERT's
        // whitespace runs are non-special added tokens).
        if (_addedById.TryGetValue(id, out var added))
        {
            return new TokenPiece(id, added.Content, added.Content, added.IsSpecial);
        }
        if (id < 0 || id >= _tokensById.Length || _tokensById[id] is null)
        {
            throw new TokenizerException($"cannot decode unknown id {id}");
        }
        string token = _tokensById[id];
        if (_spaceMark is not null)
        {
            return new TokenPiece(id, token, Encoding.UTF8.GetString(MarkedBytes(token)), false);
        }
        ByteLevel.TryDecodeWord(token, out var word);
        return new TokenPiece(id, token, word, false);
    }

    private TokenPiece PieceFor(AddedToken added, string content) =>
        new(added.Id, content, content, added.IsSpecial);

    /// <summary>
    /// The UTF-8 bytes a token sequence stands for, joined before any text
    /// decoding — what HF's ByteLevel decoder does. Added tokens contribute
    /// their content (special ones only when <paramref name="skipSpecialTokens"/>
    /// is false).
    /// </summary>
    public byte[] DecodeBytes(IReadOnlyList<int> ids, bool skipSpecialTokens)
    {
        var bytes = new List<byte>();
        foreach (int id in ids)
        {
            if (_addedById.TryGetValue(id, out var added))
            {
                if (!(added.IsSpecial && skipSpecialTokens))
                {
                    bytes.AddRange(Encoding.UTF8.GetBytes(added.Content));
                }
                continue;
            }
            if (_spaceMark is not null && id >= 0 && id < _tokensById.Length && _tokensById[id] is { } marked)
            {
                bytes.AddRange(MarkedBytes(marked));
                continue;
            }
            if (id < 0 || id >= _tokensById.Length || _tokensById[id] is null ||
                !ByteLevel.TryAppendBytes(_tokensById[id], bytes))
            {
                throw new TokenizerException($"cannot decode id {id}");
            }
        }
        return bytes.ToArray();
    }

    /// <summary>Decodes to text through <see cref="DecodeBytes"/>; invalid
    /// UTF-8 (a sequence cut mid-character) becomes U+FFFD.</summary>
    public string DecodeText(IReadOnlyList<int> ids, bool skipSpecialTokens = true) =>
        Encoding.UTF8.GetString(DecodeBytes(ids, skipSpecialTokens));

    /// <summary>Whether an id is a special added token.</summary>
    public bool IsSpecial(int id) => _specialIds.Contains(id);

    /// <summary>The id of an added token by its content, if there is one.</summary>
    public int? AddedTokenId(string content) =>
        _addedByContent.TryGetValue(content, out var t) ? t.Id : null;

    /// <summary>Decodes ids to text: special tokens contribute their
    /// content, byte-level tokens their byte-decoded word.</summary>
    public string Decode(IReadOnlyList<int> ids)
    {
        var builder = new StringBuilder();
        foreach (var id in ids)
        {
            builder.Append(PieceFor(id).DecodedText);
        }
        return builder.ToString();
    }

    /// <summary>The display facts for one id (repr + decoded text + special
    /// flag) — what <c>inspect-token</c> renders for a token.</summary>
    public TokenPiece TokenInfo(int id)
    {
        if (id < 0 || id >= _tokensById.Length)
        {
            throw new TokenizerException($"cannot inspect unknown id {id}");
        }
        return PieceFor(id);
    }
}