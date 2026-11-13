using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Cli;

/// <summary>
/// The family-independent half of an HF export: the checkpoint's own
/// <c>config.json</c>, carried from import (<see cref="SourceConfigFile"/>),
/// with every fact the container judges written over it — so the export
/// keeps keys this build has never heard of, and never states a dimension the
/// tensors contradict — and the fused tensors the importer split, joined back.
/// </summary>
internal static class GenericExport
{
    /// <summary>The source checkpoint's config.json, as imported.</summary>
    public const string SourceConfigFile = Amql.Hf.ModelToContainer.SourceConfigFile;

    /// <summary>The regenerated config, or null when the container carries no
    /// source config (encoded before it was kept, or built from scratch).</summary>
    public static string? TryBuildConfig(Vindex3Container container, bool quantizeMxfp4)
    {
        string path = Path.Combine(container.Root, SourceConfigFile);
        if (!File.Exists(path))
        {
            return null;
        }
        var root = JsonNode.Parse(File.ReadAllText(path))?.AsObject()
                   ?? throw new CliException($"{SourceConfigFile} is not a JSON object");
        var graph = container.Graph
            ?? throw new CliException("container records no system graph — cannot regenerate config.json");
        var component = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.PrimaryText)
            ?? throw new CliException("container has no primary text component — cannot regenerate config.json");
        var surface = component.Execution
            ?? throw new CliException($"component '{component.Id}' declares no execution surface — cannot regenerate config.json");
        var policies = component.Attention
            ?? throw new CliException($"component '{component.Id}' declares no per-layer attention table — cannot regenerate config.json");

        // A wrapper config keeps its text stack in text_config.
        var text = root["text_config"] as JsonObject ?? root;
        int sourceLayers = text["num_hidden_layers"]?.GetValue<int>() ?? component.NumLayers;

        void Set(string key, JsonNode? value) => text[key] = value;
        Set("num_hidden_layers", component.NumLayers);
        Set("hidden_size", component.HiddenSize);
        if (surface.Attention is { } attention)
        {
            Set("num_attention_heads", attention.NumQHeads);
            Set("num_key_value_heads", attention.NumKvHeads);
            if (text.ContainsKey("head_dim") || attention.HeadDim * attention.NumQHeads != component.HiddenSize)
            {
                Set("head_dim", attention.HeadDim);
            }
        }
        if (surface.Ffn is { } ffn)
        {
            Set("intermediate_size", ffn.IntermediateSize);
            if (ffn.Moe is { } moe)
            {
                Set("moe", new JsonObject
                {
                    ["experts"] = moe.Experts,
                    ["top_k"] = moe.TopK,
                    ["expert_intermediate_size"] = moe.ExpertIntermediateSize,
                    ["routing_policy"] = moe.RoutingPolicy == ExpertRoutingPolicy.NormalisedOverSelected
                        ? "normalised_over_selected"
                        : "softmax_then_select",
                });
            }
        }
        if (surface.Head is { } head)
        {
            Set("vocab_size", head.VocabSize);
            // tie_word_embeddings may sit on the wrapper rather than the text stack.
            (root.ContainsKey("tie_word_embeddings") && !text.ContainsKey("tie_word_embeddings") ? root : text)
                ["tie_word_embeddings"] = head.HeadReusesEmbedding;
        }
        if (surface.ContextLength is { } context)
        {
            Set("max_position_embeddings", context);
        }

        // Per-layer tables are regenerated from the policy table; any other
        // per-layer array the build cannot regenerate refuses if the layer
        // count changed, rather than ship a table for the wrong stack.
        if (text.ContainsKey("layer_types"))
        {
            Set("layer_types", new JsonArray(policies.Select(p => (JsonNode?)LayerType(p)).ToArray()));
        }
        if (text.ContainsKey("no_rope_layers"))
        {
            Set("no_rope_layers", new JsonArray(policies.Select(p => (JsonNode?)(p.Position is PositionNone ? 0 : 1)).ToArray()));
        }
        if (component.NumLayers != sourceLayers)
        {
            foreach (var (key, value) in text.ToList())
            {
                if (value is JsonArray array && array.Count == sourceLayers && key is not "layer_types" and not "no_rope_layers")
                {
                    throw new CliException(
                        $"config key '{key}' holds one entry per layer, the stack went from {sourceLayers} to {component.NumLayers} " +
                        "layers, and this build cannot tell which entries survived — refusing to export a stale table");
                }
            }
        }

        // A materialised vision tower travels with the model. The checkpoint's
        // own vision_config wins; a source without one (a synthetic or
        // re-assembled model) gets the tower's judged facts, as the
        // fact-only config builder writes them.
        var perception = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.Perception);
        if (perception is { } vision && !root.ContainsKey("vision_config") &&
            graph.Objects.Any(o => o.Component == vision.Id && o.Representations.Count > 0))
        {
            var visionConfig = new JsonObject
            {
                ["modality"] = "image",
                ["hidden_size"] = vision.HiddenSize,
                ["num_hidden_layers"] = vision.NumLayers,
            };
            if (vision.Perception is { ValueKind: JsonValueKind.Object } facts && facts.TryGetProperty("transform", out var transform))
            {
                visionConfig["transform"] = JsonNode.Parse(transform.GetRawText());
            }
            root["vision_config"] = visionConfig;
        }

        if (quantizeMxfp4)
        {
            root["quantization_config"] = new JsonObject
            {
                ["quant_method"] = "mxfp4",
                ["element_dtype"] = "FP4",
                ["element_grid"] = new JsonArray(BitPattern.Fp4PositiveGrid.Select(v => (JsonNode?)JsonValue.Create(v)).ToArray()),
                ["block_elements"] = Mxfp4.BlockElements,
                ["block_scale_dtype"] = "F8_E8M0",
            };
        }
        return root.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
    }

    private static string LayerType(AttentionLayerPolicy policy) => policy.Operator switch
    {
        LayerOperators.Softmax => policy.Span == AttentionSpan.Sliding ? "sliding_attention" : "full_attention",
        LayerOperators.LinearAttention => "linear_attention",
        LayerOperators.Conv => "conv",
        var other => throw new CliException($"operator '{other}' has no layer_types spelling — refusing to approximate"),
    };

    /// <summary>
    /// Joins the parts of every fused tensor the importer split back into the
    /// checkpoint's own fused tensor (row blocks concatenated in order), in
    /// place. A part that was quantised refuses: its packed rows no longer
    /// concatenate into the fused tensor's layout.
    /// </summary>
    public static int Refuse(List<TensorPayload> payloads, LayerRoleMap? roles, string stackPrefix)
    {
        if (roles?.Fused is not { Count: > 0 } fusions)
        {
            return 0;
        }
        var byName = payloads.ToDictionary(p => p.Name, StringComparer.Ordinal);
        int joined = 0;
        foreach (var fusion in fusions)
        {
            string firstSuffix = "." + fusion.Parts[0];
            var layers = byName.Keys
                .Where(n => n.StartsWith(stackPrefix + ".", StringComparison.Ordinal) && n.EndsWith(firstSuffix, StringComparison.Ordinal))
                .Select(n => n[(stackPrefix.Length + 1)..^firstSuffix.Length])
                .ToList();
            foreach (var layer in layers)
            {
                var parts = fusion.Parts.Select(p => byName.GetValueOrDefault($"{stackPrefix}.{layer}.{p}")).ToList();
                if (parts.Any(p => p is null))
                {
                    throw new CliException(
                        $"layer {layer}: '{fusion.Name}' was imported fused, but not all of its parts ({string.Join(", ", fusion.Parts)}) are in the export");
                }
                if (parts.Select(p => p!.Dtype).Distinct().Count() != 1 ||
                    parts.Any(p => p!.Shape.Skip(1).SequenceEqual(parts[0]!.Shape.Skip(1)) is false))
                {
                    throw new CliException(
                        $"layer {layer}: the parts of '{fusion.Name}' no longer share a dtype and row width (quantised?) — cannot fuse them back");
                }
                var shape = (long[])parts[0]!.Shape.Clone();
                shape[0] = parts.Sum(p => p!.Shape[0]);
                var chunks = parts.SelectMany(p => p!.Chunks ?? new[] { p.Data }).ToList();
                var fused = new TensorPayload
                {
                    Name = $"{stackPrefix}.{layer}.{fusion.Name}",
                    Dtype = parts[0]!.Dtype,
                    Shape = shape,
                    Data = chunks.Count == 1 ? chunks[0] : Array.Empty<byte>(),
                    Chunks = chunks.Count == 1 ? null : chunks,
                };
                foreach (var part in parts)
                {
                    payloads.Remove(part!);
                    byName.Remove(part!.Name);
                }
                payloads.Add(fused);
                byName[fused.Name] = fused;
                joined++;
            }
        }
        return joined;
    }
}
