using Amql.Vindex3;

namespace Amql.Hf;

/// <summary>Summary of one encoder run, printed by the CLI.</summary>
public sealed record EncodeReport(
    string ModelId,
    string ContainerRoot,
    string Encoding,
    long PayloadBytes,
    int Tensors,
    bool TokenizerCopied,
    IReadOnlyDictionary<string, SegmentWriteResult> Segments);

/// <summary>
/// The G0→G3 loader pipeline: inventory → architecture facts → system
/// graph → canonical container. This is the .NET analogue of the
/// reference's <c>inspect / invent → represent → encode</c> chain, bounded
/// to canonical (unquantised) materialisation of the text decoder.
/// </summary>
public static class ModelToContainer
{
    public static EncodeReport Encode(
        string modelDir,
        string containerOut,
        string? modelId = null,
        ArchMapper.EncodeOptions? options = null)
    {
        string modelName = modelId ?? Path.GetFileName(modelDir.TrimEnd('\\', '/'));
        var facts = ModelConfig.ReadTextFacts(Path.Combine(modelDir, "config.json"));
        using var inventory = HfInventory.Open(modelDir);
        var spec = ArchMapper.MapToContainerSpec(modelName, facts, inventory, options ?? new ArchMapper.EncodeOptions());
        var result = ContainerEncoder.Encode(containerOut, spec);

        // The tokenizer travels with the container: if the checkpoint ships
        // tokenizer.json it is copied into the container root, so text
        // commands can run without an explicit --tokenizer.
        bool tokenizerCopied = false;
        var tokenizerPath = Path.Combine(modelDir, "tokenizer.json");
        if (File.Exists(tokenizerPath))
        {
            File.Copy(tokenizerPath, Path.Combine(containerOut, "tokenizer.json"), overwrite: false);
            tokenizerCopied = true;
        }

        long payload = spec.Representations.Sum(r => r.Tensors.Sum(t => (long)t.Data.Length));
        int tensors = spec.Representations.Sum(r => r.Tensors.Count);
        var encoding = result.Index.Representations.Values.FirstOrDefault()?.Encoding ?? "?";
        return new EncodeReport(modelName, containerOut, encoding, payload, tensors, tokenizerCopied, result.Segments);
    }
}