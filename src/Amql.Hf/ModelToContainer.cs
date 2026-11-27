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
    /// <summary>The source checkpoint's config.json, kept in the container root.</summary>
    public const string SourceConfigFile = "source_config.json";

    public static EncodeReport Encode(
        string modelDir,
        string containerOut,
        string? modelId = null,
        ArchMapper.EncodeOptions? options = null)
    {
        string modelName = modelId ?? Path.GetFileName(modelDir.TrimEnd('\\', '/'));
        string configPath = Path.Combine(modelDir, "config.json");
        // Inkling 2 looped shares no roles with the stacks the generic reader
        // judges (its router alone would be refused there): it has its own.
        bool inkling = File.Exists(configPath) && Inkling2.Describes(configPath);
        var facts = inkling ? null : ModelConfig.ReadTextFacts(configPath);
        using var inventory = HfInventory.Open(modelDir);
        var classification = inkling ? null : ModelConfig.ReadClassificationFacts(configPath, inventory);
        var spec = inkling
            ? ArchMapper.MapInkling2(modelName, configPath, inventory, options ?? new ArchMapper.EncodeOptions())
            : ArchMapper.MapToContainerSpec(modelName, facts!, inventory, options ?? new ArchMapper.EncodeOptions(),
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

        // The config itself rides along too, as the base an export overwrites
        // with the container's judged facts: keys this build has no notion of
        // survive the round trip instead of being dropped.
        if (inkling)
        {
            File.Copy(configPath, Path.Combine(containerOut, SourceConfigFile), overwrite: true);
            ancillaryCopied = ancillaryCopied.Append(SourceConfigFile).ToList();
            // Its experts are kept as stored (NVFP4, MXFP8), so the file that
            // says how they are stored is still true of an export.
            string quant = Path.Combine(modelDir, Inkling2.QuantConfigFile);
            if (File.Exists(quant))
            {
                File.Copy(quant, Path.Combine(containerOut, Inkling2.QuantConfigFile), overwrite: true);
                ancillaryCopied = ancillaryCopied.Append(Inkling2.QuantConfigFile).ToList();
            }
        }
        else if (facts!.SourceConfig is { } sourceConfig)
        {
            File.WriteAllText(Path.Combine(containerOut, SourceConfigFile), sourceConfig.GetRawText());
            ancillaryCopied = ancillaryCopied.Append(SourceConfigFile).ToList();
        }

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

        long payload = spec.Representations.Sum(r => r.Tensors.Sum(t => t.PayloadLength));
        int tensors = spec.Representations.Sum(r => r.Tensors.Count);
        var encoding = result.Index.Representations.Values.FirstOrDefault()?.Encoding ?? "?";
        return new EncodeReport(modelName, containerOut, encoding, payload, tensors, tokenizerCopied,
            ancillaryCopied, result.Segments);
    }
}