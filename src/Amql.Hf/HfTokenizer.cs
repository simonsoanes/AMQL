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

/// <summary>One added (special) token of the tokenizer file.</summary>
public sealed record AddedToken(int Id, string Content, bool IsSpecial);

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
/// characters, and ByteLevel decoding. Unsupported pipeline stages refuse
/// the file by name — the tokenizer is only trusted when every stage is
/// one it has judged.
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

    private HfTokenizer(
        Dictionary<string, int> vocab,
        Dictionary<(string, string), int> mergeRanks,
        Dictionary<string, AddedToken> addedByContent,
        IReadOnlyList<AddedToken> addedTokens,
        Regex splitRegex,
        string[] tokensById,
        HashSet<int> specialIds)
    {
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
            var merges = model.GetProperty("merges");
            for (int i = 0; i < merges.GetArrayLength(); i++)
            {
                var pair = merges[i].GetString() ?? throw new TokenizerException($"'{path}': null merge entry at {i}");
                int splitAt = (pair.Length - 1) - ' ';
                splitAt = pair.LastIndexOf(' ');
                if (splitAt <= 0)
                {
                    throw new TokenizerException($"'{path}': malformed merge '{pair}'");
                }
                mergeRanks[(pair[..splitAt], pair[(splitAt + 1)..])] = i;
            }

            // Added / special tokens.
            var addedTokens = new List<AddedToken>();
            var addedByContent = new Dictionary<string, AddedToken>(StringComparer.Ordinal);
            if (root.TryGetProperty("added_tokens", out var added) && added.ValueKind == JsonValueKind.Array)
            {
                for (int i = 0; i < added.GetArrayLength(); i++)
                {
                    var entry = added[i];
                    int id = entry.GetProperty("id").GetInt32();
                    string content = entry.GetProperty("content").GetString() ?? string.Empty;
                    bool special = entry.TryGetProperty("special", out var sp) && sp.GetBoolean();
                    var token = new AddedToken(id, content, special);
                    addedTokens.Add(token);
                    addedByContent[content] = token;
                }
            }

            // Normalizer: only NFC is judged.
            if (root.TryGetProperty("normalizer", out var normalizer) &&
                normalizer.ValueKind == JsonValueKind.Object &&
                normalizer.GetProperty("type").GetString() is { } normType &&
                normType != "NFC")
            {
                throw new TokenizerException($"'{path}': normalizer '{normType}' is not the NFC this build serves");
            }

            // Pre-tokenizer: Sequence(Split(regex), ByteLevel) or bare ByteLevel.
            var splitRegex = BuildSplitRegex(root);

            // Decoder: only ByteLevel is judged.
            if (root.TryGetProperty("decoder", out var decoder) &&
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

            return new HfTokenizer(vocab, mergeRanks, addedByContent, addedTokens, splitRegex, tokensById, specialIds);
        }
    }

    private static Regex BuildSplitRegex(JsonElement root)
    {
        string? pattern = null;
        if (root.TryGetProperty("pre_tokenizer", out var pre) && pre.ValueKind == JsonValueKind.Object)
        {
            switch (pre.GetProperty("type").GetString())
            {
                case "ByteLevel":
                    break; // no splitting; regex = null → whole-string tokens
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
        // Stage 1: NF C normalisation (the "NFC" normalizer).
        string normalized = text.Normalize(NormalizationForm.FormC);

        var pieces = new List<TokenPiece>();
        // Stage 2: added tokens are carved at their start positions
        // (longest content first); the runs between them are regex-split
        // (the configured split, Isolated behavior: matches are the pieces).
        foreach (var (start, end, isAdded) in Scatter(normalized))
        {
            if (isAdded)
            {
                string content = normalized[start..end];
                pieces.Add(PieceFor(_addedByContent[content], content));
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

    /// <summary>Walks the string once: added-token runs (exact content
    /// match, longest-first) alternate with plain runs that end where the
    /// next added token begins.</summary>
    private static List<(int Start, int End, bool IsAdded)> Scatter(string normalized, IReadOnlyCollection<string> addedContents)
    {
        var runs = new List<(int, int, bool)>();
        var ordered = addedContents.OrderByDescending(c => c.Length).ToArray();
        int i = 0;
        while (i < normalized.Length)
        {
            if (TryAddedAt(normalized, i, ordered, out var content))
            {
                runs.Add((i, i + content.Length, true));
                i += content.Length;
            }
            else
            {
                int j = i;
                while (j < normalized.Length && !TryAddedAt(normalized, j, ordered, out _))
                {
                    j++;
                }
                if (j > i)
                {
                    runs.Add((i, j, false));
                }
                i = j;
            }
        }
        return runs;

        static bool TryAddedAt(string text, int index, string[] ordered, out string content)
        {
            foreach (var candidate in ordered)
            {
                if (index + candidate.Length <= text.Length &&
                    string.CompareOrdinal(text, index, candidate, 0, candidate.Length) == 0)
                {
                    content = candidate;
                    return true;
                }
            }
            content = string.Empty;
            return false;
        }
    }

    private IEnumerable<(int Start, int End, bool IsAdded)> Scatter(string normalized) =>
        Scatter(normalized, _addedByContent.Keys);

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
        // Added/special tokens are judged FIRST: they live outside the BPE
        // vocab and always decode to their content.
        if (_specialIds.Contains(id))
        {
            var added = _addedTokens.FirstOrDefault(a => a.Id == id);
            string content = added?.Content ?? string.Empty;
            return new TokenPiece(id, content, content, true);
        }
        if (id < 0 || id >= _tokensById.Length || _tokensById[id] is null)
        {
            throw new TokenizerException($"cannot decode unknown id {id}");
        }
        string token = _tokensById[id];
        ByteLevel.TryDecodeWord(token, out var word);
        return new TokenPiece(id, token, word, false);
    }

    private TokenPiece PieceFor(AddedToken added, string content) =>
        new(added.Id, content, content, added.IsSpecial);

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