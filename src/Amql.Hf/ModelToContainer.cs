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
    IReadOnlyList<string> AncillaryCopied,
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
        var classification = ModelConfig.ReadClassificationFacts(Path.Combine(modelDir, "config.json"), inventory);
        var spec = ArchMapper.MapToContainerSpec(modelName, facts, inventory, options ?? new ArchMapper.EncodeOptions(),
            classification);
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

        // The chat template and the processor configs travel with it too.
        // Without them a container re-exports as a checkpoint that can
        // tokenize but cannot format a conversation, and a multi-modal model
        // loses its image/video processor settings — neither is recoverable
        // from the tensors, so the container is the only place to keep them.
        var ancillaryCopied = HfAncillaryFiles.CopyInto(modelDir, containerOut);

        // A classifier's top-level facts, mirrored beside the graph (which is
        // the authority) so they can be read without parsing it.
        if (classification is not null)
        {
            var mirror = new System.Text.Json.Nodes.JsonObject
            {
                ["num_labels"] = classification.NumLabels,
                ["problem_type"] = classification.ProblemType,
                ["id2label"] = classification.Labels is { } labels
                    ? new System.Text.Json.Nodes.JsonObject(labels.Select((l, i) =>
                        KeyValuePair.Create(i.ToString(System.Globalization.CultureInfo.InvariantCulture), (System.Text.Json.Nodes.JsonNode?)l)))
                    : null,
                ["nli_template"] = classification.Template,
                ["pad_token_id"] = classification.PadTokenId,
                ["score_bias"] = classification.BiasTensor is not null,
            };
            File.WriteAllText(Path.Combine(containerOut, "classifier.json"),
                mirror.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
            ancillaryCopied = ancillaryCopied.Append("classifier.json").ToList();
        }

        long payload = spec.Representations.Sum(r => r.Tensors.Sum(t => (long)t.Data.Length));
        int tensors = spec.Representations.Sum(r => r.Tensors.Count);
        var encoding = result.Index.Representations.Values.FirstOrDefault()?.Encoding ?? "?";
        return new EncodeReport(modelName, containerOut, encoding, payload, tensors, tokenizerCopied,
            ancillaryCopied, result.Segments);
    }
}