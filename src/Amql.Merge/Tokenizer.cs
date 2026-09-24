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

            // Special tokens live in the top-level `added_tokens` array, not in
            // model.vocab, and occupy the ids directly above it. Dropping them
            // silently truncates the vocabulary: the merged embedding would be
            // sized to model.vocab while the tokenizer still hands out the
            // special ids, so every bos/eos lookup reads past the end of the
            // table. They are part of the vocabulary and must be carried.
            if (doc.RootElement.TryGetProperty("added_tokens", out var addedTokens) &&
                addedTokens.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in addedTokens.EnumerateArray())
                {
                    if (item.ValueKind != JsonValueKind.Object ||
                        !item.TryGetProperty("id", out var idElement) ||
                        !idElement.TryGetInt32(out int addedId) || addedId < 0)
                    {
                        continue;
                    }
                    string? content = item.TryGetProperty("content", out var contentElement)
                        ? contentElement.GetString()
                        : null;
                    if (content is null)
                    {
                        continue;
                    }
                    if (byToken.TryGetValue(content, out int existingId))
                    {
                        // The same string can legitimately appear in both places
                        // with the same id; a conflicting id is a defect.
                        if (existingId != addedId)
                        {
                            throw new MergeException(
                                $"tokenizer '{tokenizerPath}': token '{content}' has id {existingId} in model.vocab " +
                                $"but id {addedId} in added_tokens — refusing to guess");
                        }
                        continue;
                    }
                    ids.Add((addedId, content));
                    byToken[content] = addedId;
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

        // Next free id must clear BOTH model.vocab and added_tokens — the
        // special tokens sit above model.vocab, so sizing nextId from
        // vocab.Count would hand new tokens ids that already belong to
        // bos/eos/etc and silently overwrite them.
        int nextId = vocab.Count;
        foreach (var entry in vocab)
        {
            if (entry.Value is JsonValue v && v.TryGetValue<int>(out int vid) && vid >= nextId)
            {
                nextId = vid + 1;
            }
        }
        if (root["added_tokens"] is JsonArray addedArray)
        {
            foreach (var item in addedArray)
            {
                if (item?["id"] is JsonValue av && av.TryGetValue<int>(out int aid) && aid >= nextId)
                {
                    nextId = aid + 1;
                }
            }
        }

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