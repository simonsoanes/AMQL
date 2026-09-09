namespace Amql.Merge;

/// <summary>Raised on anything that prevents a merge — fail closed, never
/// guess a token relationship or a weight value.</summary>
public sealed class MergeException : Exception
{
    public MergeException(string message) : base(message) { }

    public MergeException(string message, Exception inner) : base(message, inner) { }
}

/// <summary>How one merged vocabulary row relates to the two source
/// models, per the tokenization mapping layer.</summary>
public enum TokenMergeKind
{
    /// <summary>Token exists only in the base model — row stays the base
    /// model's (zero-extended to the merged width).</summary>
    BaseOnly,

    /// <summary>Token exists in both models — the two rows are normalised
    /// together (base plus aligned-imported, averaged).</summary>
    Blended,

    /// <summary>Token exists only in the imported model — its row is added
    /// through the alignment map that fits the imported space into the
    /// base space.</summary>
    AlignedNew,
}

/// <summary>One row of the merged vocabulary: the token string, how it
/// relates to the two sources, and the source ids (base ids are stable in
/// the merged model; imported ids are recorded for the tracker).</summary>
public sealed record TokenMapEntry(int Id, string Token, TokenMergeKind Kind, int? BaseId, int? ImportedId)
{
    public string KindLabel => Kind switch
    {
        TokenMergeKind.BaseOnly => "base_only",
        TokenMergeKind.Blended => "blended",
        TokenMergeKind.AlignedNew => "aligned_new",
        _ => "unknown",
    };
}

/// <summary>
/// The tokenization mapping layer: the relationship between two models'
/// vocabularies. Both tokenizers are read in id order; two tokens relate
/// when their strings are identical (the only relationship this build
/// judges — prefix/byte relationships would be fabrications). The merged
/// vocabulary is the base vocabulary verbatim (ids stable) followed by the
/// imported-only tokens in imported order (new ids), so every merged id
/// tracks back to its sources.
/// </summary>
public sealed class TokenMapping
{
    public required string BaseModel { get; init; }
    public required string ImportedModel { get; init; }
    public required int BaseVocab { get; init; }
    public required int ImportedVocab { get; init; }

    /// <summary>Merged vocabulary in id order (id == index).</summary>
    public required List<TokenMapEntry> Entries { get; init; }

    /// <summary>Tokens present in both models — the anchors of the
    /// space-alignment fit.</summary>
    public IReadOnlyList<TokenMapEntry> Anchors => Entries.Where(e => e.Kind == TokenMergeKind.Blended).ToList();

    public int Count => Entries.Count;
    public int BlendedCount => Anchors.Count;

    public static TokenMapping Build(
        TokenVocabulary baseVocab,
        TokenVocabulary importedVocab,
        string baseModel,
        string importedModel)
    {
        var entries = new List<TokenMapEntry>(baseVocab.Count + importedVocab.Count);
        for (int id = 0; id < baseVocab.Count; id++)
        {
            string token = baseVocab.Ordered[id];
            entries.Add(importedVocab.ByToken.TryGetValue(token, out int importedId)
                ? new TokenMapEntry(id, token, TokenMergeKind.Blended, id, importedId)
                : new TokenMapEntry(id, token, TokenMergeKind.BaseOnly, id, null));
        }

        int nextId = baseVocab.Count;
        for (int importedId = 0; importedId < importedVocab.Count; importedId++)
        {
            string token = importedVocab.Ordered[importedId];
            if (baseVocab.ByToken.ContainsKey(token))
            {
                continue; // already placed by its base id
            }
            entries.Add(new TokenMapEntry(nextId++, token, TokenMergeKind.AlignedNew, null, importedId));
        }

        return new TokenMapping
        {
            BaseModel = baseModel,
            ImportedModel = importedModel,
            BaseVocab = baseVocab.Count,
            ImportedVocab = importedVocab.Count,
            Entries = entries,
        };
    }
}