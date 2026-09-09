using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>
/// A model's token vocabulary as recorded in its tokenizer.json: the
/// ordered token-string list (id = index, matching the HF contiguous-id
/// convention) and the reverse lookup. This is the authority for token
/// relationships during a merge — tokenizer ids are opaque, the token
/// strings are the relationship key.
/// </summary>
public sealed class TokenVocabulary
{
    private TokenVocabulary(IReadOnlyList<string> ordered, IReadOnlyDictionary<string, int> byToken)
    {
        Ordered = ordered;
        ByToken = byToken;
    }

    public IReadOnlyList<string> Ordered { get; }
    public IReadOnlyDictionary<string, int> ByToken { get; }
    public int Count => Ordered.Count;

    public static TokenVocabulary ReadFromFile(string tokenizerPath)
    {
        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(File.ReadAllBytes(tokenizerPath));
        }
        catch (Exception e) when (e is IOException or JsonException)
        {
            throw new MergeException($"tokenizer '{tokenizerPath}' is not readable JSON: {e.Message}", e);
        }

        using (doc)
        {
            if (!doc.RootElement.TryGetProperty("model", out var model) ||
                !model.TryGetProperty("vocab", out var vocab) ||
                vocab.ValueKind != JsonValueKind.Object)
            {
                throw new MergeException(
                    $"tokenizer '{tokenizerPath}' has no model.vocab object — a token relationship map cannot be built");
            }

            var ids = new List<(int Id, string Token)>();
            var byToken = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var prop in vocab.EnumerateObject())
            {
                if (!prop.Value.TryGetInt32(out int id) || id < 0)
                {
                    throw new MergeException(
                        $"tokenizer '{tokenizerPath}': token '{prop.Name}' has a non-integer id — refusing to guess");
                }
                ids.Add((id, prop.Name));
                if (!byToken.TryAdd(prop.Name, id))
                {
                    throw new MergeException($"tokenizer '{tokenizerPath}': duplicate token '{prop.Name}'");
                }
            }

            ids.Sort((a, b) => a.Id.CompareTo(b.Id));
            for (int i = 0; i < ids.Count; i++)
            {
                if (ids[i].Id != i)
                {
                    throw new MergeException(
                        $"tokenizer '{tokenizerPath}': ids are not contiguous (expected 0..{ids.Count - 1}, got {ids[i].Id} at rank {i}) — refusing to renumber");
                }
            }
            return new TokenVocabulary(ids.Select(x => x.Token).ToArray(), byToken);
        }
    }

    public static TokenVocabulary ReadFromContainer(Vindex3Container container)
    {
        var path = Path.Combine(container.Root, "tokenizer.json");
        if (!File.Exists(path))
        {
            throw new MergeException(
                $"container '{container.Root}' carries no tokenizer.json — a token relationship map cannot be built");
        }
        return ReadFromFile(path);
    }

    /// <summary>
    /// Appends extra token strings to a base tokenizer.json (the exported
    /// vocabulary) so the merged container's tokenizer covers the union
    /// vocabulary. Base ids are untouched; new tokens take the next ids in
    /// the given order.
    /// </summary>
    public static string WithAppendedTokens(string tokenizerPath, IReadOnlyList<string> append)
    {
        JsonNode root;
        try
        {
            root = JsonNode.Parse(File.ReadAllBytes(tokenizerPath))
                   ?? throw new MergeException($"tokenizer '{tokenizerPath}' is empty");
        }
        catch (JsonException e)
        {
            throw new MergeException($"tokenizer '{tokenizerPath}' is not readable JSON: {e.Message}", e);
        }

        var vocab = root["model"]?["vocab"]?.AsObject()
            ?? throw new MergeException($"tokenizer '{tokenizerPath}' has no model.vocab object");
        int nextId = vocab.Count;
        foreach (var token in append)
        {
            if (vocab.ContainsKey(token))
            {
                continue;
            }
            vocab[token] = nextId++;
        }
        return root.ToJsonString(ViJson.Options);
    }
}