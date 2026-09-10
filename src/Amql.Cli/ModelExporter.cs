using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Cli;

/// <summary>Everything written by <c>amql-cli export</c>: the checkpoint
/// directory, its shard, and the objects that could not be materialised
/// (carried-only objects like the tied output head).</summary>
public sealed record ExportReport(
    string OutDir,
    string Model,
    int Tensors,
    long PayloadBytes,
    IReadOnlyList<string> Notes);

/// <summary>
/// The inverse of <c>encode</c>: materialises an HF-style checkpoint
/// directory (config.json + model.safetensors + tokenizer.json) from a
/// VINDEX3 container. Segment tensor names are object-relative, so the
/// HF tensor name is rebuilt from the graph's source binding
/// (<c>TensorPrefix + "." + name</c>); an optional weight patch is baked
/// in by widening to f32, adding the delta, and re-encoding to the
/// tensor's stored dtype. Tensors a patch never touches are copied
/// verbatim, so an unpatched export is byte-identical to the source.
/// config.json is regenerated from judged graph facts only — an operator
/// without a judged <c>layer_types</c> spelling refuses the export rather
/// than approximating it.
/// </summary>
public static class ModelExporter
{
    public const string ExportFormat = "amql-export-v1";
    private const string ShardName = "model.safetensors";

    public static ExportReport Export(
        Vindex3Container container,
        string outDir,
        WeightPatch? patch,
        bool quantizeMxfp4 = false)
    {
        if (Directory.Exists(outDir))
        {
            throw new CliException($"export output '{outDir}' already exists");
        }

        // Build config.json first: an unjudged operator refuses here,
        // before anything is written.
        string configJson = ExportConfig.BuildJson(container, quantizeMxfp4);
        var graph = container.Graph
            ?? throw new CliException("container records no system graph — cannot rebuild HF tensor names");

        using var store = container.CreateOperandStore();
        var payloads = new List<TensorPayload>();
        var names = new Dictionary<string, string>(StringComparer.Ordinal);
        var notes = new List<string>();
        int quantized = 0;

        foreach (var obj in graph.Objects)
        {
            if (obj.Representations.Count == 0)
            {
                notes.Add($"object '{obj.Id}': carried only (no materialised tensors) — skipped");
                continue;
            }
            string? prefix = obj.SourceBindings.FirstOrDefault()?.TensorPrefix;
            if (string.IsNullOrEmpty(prefix))
            {
                throw new CliException(
                    $"object '{obj.Id}' materialises tensors but declares no source-binding tensor prefix — cannot rebuild HF tensor names");
            }

            string? segmentPath = store.SegmentPathFor(obj.Id);
            if (segmentPath is null)
            {
                notes.Add($"object '{obj.Id}': representation recorded but no segment on disk — skipped");
                continue;
            }

            using var segment = SegmentFile.Open(Path.Combine(container.Root, segmentPath));
            foreach (var tensor in segment.Header.Tensors)
            {
                string hfName = prefix + "." + tensor.Name;
                if (names.TryGetValue(hfName, out var other))
                {
                    throw new CliException(
                        $"export collision: tensor name '{hfName}' rebuilt for both '{other}' and '{obj.Id}'");
                }
                names[hfName] = obj.Id;

                if (quantizeMxfp4 && ShouldQuantize(obj.Id, tensor.Name, tensor.Shape))
                {
                    AddQuantizedPayloads(payloads, names, obj.Id, hfName, tensor, store, patch);
                    quantized++;
                }
                else
                {
                    payloads.Add(new TensorPayload
                    {
                        Name = hfName,
                        Dtype = DtypeExtensions.FromLabel(tensor.Dtype),
                        Shape = tensor.Shape,
                        Data = ExportTensor(store, obj.Id, tensor, patch),
                    });
                }
            }
        }

        if (quantized > 0)
        {
            notes.Add($"{quantized} stack projection tensors exported as MXFP4 (FP4 E2M1 grid elements, " +
                      $"per-{Mxfp4.BlockElements}-element {Dtype.F8_E8M0.Label()} scales) — " +
                      "embeddings, norms, biases and the output head keep their full precision");
        }

        if (payloads.Count == 0)
        {
            throw new CliException("the container holds no materialised tensors — nothing to export");
        }

        Directory.CreateDirectory(outDir);
        SafetensorsWriter.Write(Path.Combine(outDir, ShardName), payloads, new Dictionary<string, string>
        {
            ["format"] = ExportFormat,
            ["model"] = container.Index.Model,
        });
        File.WriteAllText(Path.Combine(outDir, "config.json"), configJson);

        var tokenizerPath = Path.Combine(container.Root, "tokenizer.json");
        if (File.Exists(tokenizerPath))
        {
            File.Copy(tokenizerPath, Path.Combine(outDir, "tokenizer.json"));
        }

        return new ExportReport(
            outDir,
            container.Index.Model,
            payloads.Count,
            payloads.Sum(p => (long)p.Data.Length),
            notes);
    }

    /// <summary>
    /// The stored bytes of one tensor in the exported checkpoint. An
    /// unpatched tensor's payload is copied verbatim (byte-identical
    /// regeneration); a patched tensor is widened to f32, the patch delta
    /// is added, and the result is re-encoded to the stored dtype. Dtypes
    /// without an encode path refuse if a patch touches them.
    /// </summary>
    private static byte[] ExportTensor(OperandStore store, string objectId, SegmentTensor tensor, WeightPatch? patch)
    {
        var dtype = DtypeExtensions.FromLabel(tensor.Dtype);
        return EncodeToDtype(dtype, WidenedValues(store, objectId, tensor, patch));
    }

    /// <summary>The tensor's f32 values (patch deltas applied when given)
    /// — the shared input for the encode and MXFP4 paths.</summary>
    private static float[] WidenedValues(OperandStore store, string objectId, SegmentTensor tensor, WeightPatch? patch)
    {
        var resolution = store.Resolve(objectId, tensor.Name);
        var dtype = DtypeExtensions.FromLabel(tensor.Dtype);
        var widened = BitPattern.WidenToF32(dtype, resolution.Payload);
        if (patch is not null && patch.TryGet(objectId, tensor.Name, out var entry))
        {
            if (entry.Delta.Length != WeightPatch.ElementCount(tensor.Shape))
            {
                throw new CliException(
                    $"patch entry '{entry.Key}' holds {entry.Delta.Length} deltas but the tensor " +
                    $"'{objectId}/{tensor.Name}' has {WeightPatch.ElementCount(tensor.Shape)} elements");
            }
            for (int i = 0; i < widened.Length; i++)
            {
                widened[i] += entry.Delta[i];
            }
        }
        return widened;
    }

    /// <summary>MXFP4 quantises the stack's projection matrices: 2-D
    /// per-layer weights in the decoder stack, leaving the synthetic
    /// log-space A_log tensor and any biases at full precision.</summary>
    private static bool ShouldQuantize(string objectId, string tensorName, long[] shape) =>
        objectId == "target.decoder_stack" &&
        shape.Length == 2 &&
        shape[0] > 0 &&
        shape[1] > 0 &&
        !tensorName.Contains("A_log") &&
        !tensorName.Contains("bias");

    /// <summary>Replaces one weight tensor with its MXFP4 pair: the FP4
    /// packed weight (logical shape, two elements per byte) and the
    /// per-32-element E8M0 block scales.</summary>
    private static void AddQuantizedPayloads(
        List<TensorPayload> payloads,
        Dictionary<string, string> names,
        string objectId,
        string hfName,
        SegmentTensor tensor,
        OperandStore store,
        WeightPatch? patch)
    {
        var values = WidenedValues(store, objectId, tensor, patch);
        long rows = tensor.Shape[0];
        long columns = tensor.Shape[1];
        var quantized = Mxfp4.Quantize(values, rows, columns);

        // The weight name itself was registered by the caller; only the
        // companion tensor needs its own collision guard.
        payloads.Add(new TensorPayload
        {
            Name = hfName,
            Dtype = Dtype.FP4,
            Shape = tensor.Shape,
            Data = quantized.Packed,
        });
        string scaleName = hfName + Mxfp4.ScaleSuffix;
        if (names.TryGetValue(scaleName, out var owner))
        {
            throw new CliException(
                $"export collision: MXFP4 companion tensor '{scaleName}' rebuilt for both '{owner}' and '{objectId}'");
        }
        names[scaleName] = objectId;
        payloads.Add(new TensorPayload
        {
            Name = scaleName,
            Dtype = Dtype.F8_E8M0,
            Shape = new[] { rows, Mxfp4.BlocksPerRow(columns) },
            Data = quantized.BlockScales,
        });
    }

    /// <summary>f32 → the tensor's stored dtype bytes. Only dtypes with a
    /// round-trip encode path are re-encodable; anything else fails closed
    /// naming the dtype.</summary>
    private static byte[] EncodeToDtype(Dtype dtype, float[] values)
    {
        switch (dtype)
        {
            case Dtype.F32:
            {
                var bytes = new byte[values.Length * 4];
                Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
                return bytes;
            }
            case Dtype.BF16:
            {
                var bytes = new byte[values.Length * 2];
                for (int i = 0; i < values.Length; i++)
                {
                    ushort bits = BitPattern.EncodeBf16(values[i]);
                    bytes[2 * i] = (byte)(bits & 0xFF);
                    bytes[2 * i + 1] = (byte)(bits >> 8);
                }
                return bytes;
            }
            case Dtype.F16:
            {
                var bytes = new byte[values.Length * 2];
                for (int i = 0; i < values.Length; i++)
                {
                    ushort bits = BitPattern.EncodeF16(values[i]);
                    bytes[2 * i] = (byte)(bits & 0xFF);
                    bytes[2 * i + 1] = (byte)(bits >> 8);
                }
                return bytes;
            }
            default:
                throw new CliException(
                    $"a patched '{dtype.Label()}' tensor cannot be re-encoded — export merges patches into F32/BF16/F16 weights only");
        }
    }
}

/// <summary>
/// Regenerates <c>config.json</c> for the exported checkpoint from judged
/// graph facts: the primary text component's execution surface and
/// per-layer attention policy table. Mirrors <see cref="ModelConfig"/>'s
/// field set so the exported checkpoint re-encodes; a fact this build has
/// not judged (an unjudged layer operator, activation or position policy)
/// refuses the export by name instead of fabricating a value.
/// </summary>
internal static class ExportConfig
{
    public static string BuildJson(Vindex3Container container, bool quantizeMxfp4 = false)
    {
        var graph = container.Graph
            ?? throw new CliException("container records no system graph — cannot regenerate config.json");
        var component = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.PrimaryText)
            ?? throw new CliException("container has no primary text component — cannot regenerate config.json");
        var surface = component.Execution
            ?? throw new CliException($"component '{component.Id}' declares no execution surface — cannot regenerate config.json");
        var attention = surface.Attention
            ?? throw new CliException($"component '{component.Id}' declares no attention surface — cannot regenerate config.json");
        var ffn = surface.Ffn
            ?? throw new CliException($"component '{component.Id}' declares no FFN surface — cannot regenerate config.json");
        var head = surface.Head
            ?? throw new CliException($"component '{component.Id}' declares no head surface — cannot regenerate config.json");
        if (surface.ContextLength is not { } contextLength)
        {
            throw new CliException($"component '{component.Id}' declares no context length — cannot regenerate config.json");
        }
        if (component.Attention is not { Count: > 0 } policies)
        {
            throw new CliException($"component '{component.Id}' declares no per-layer attention table — cannot regenerate config.json");
        }

        var layerTypes = new List<string>();
        bool hasLinearAttention = false;
        for (int l = 0; l < component.NumLayers; l++)
        {
            string op = policies[l].Operator;
            switch (op)
            {
                case LayerOperators.Softmax:
                    layerTypes.Add("full_attention");
                    break;
                case LayerOperators.LinearAttention:
                    layerTypes.Add("linear_attention");
                    hasLinearAttention = true;
                    break;
                default:
                    throw new CliException(
                        $"layer {l}: operator '{op}' has no judged layer_types spelling — this build regenerates " +
                        "'full_attention' and 'linear_attention' only; refusing to approximate");
            }
        }

        string hiddenAct = ffn.Activation switch
        {
            Activation.Silu => "silu",
            var other => throw new CliException(
                $"FFN activation '{other}' has no judged hidden_act spelling — refusing to approximate"),
        };

        var config = new JsonObject
        {
            ["model_type"] = container.Index.Family,
            ["hidden_size"] = component.HiddenSize,
            ["num_hidden_layers"] = component.NumLayers,
            ["num_attention_heads"] = attention.NumQHeads,
            ["num_key_value_heads"] = attention.NumKvHeads,
            ["head_dim"] = attention.HeadDim,
            ["intermediate_size"] = ffn.IntermediateSize,
            ["hidden_act"] = hiddenAct,
            ["rms_norm_eps"] = surface.Norm.Pre.Eps,
            ["vocab_size"] = head.VocabSize,
            ["max_position_embeddings"] = contextLength,
            ["layer_types"] = new JsonArray(layerTypes.Select(t => JsonValue.Create(t)).ToArray()),
        };

        if (attention.AttentionBias == true)
        {
            config["attention_bias"] = true;
        }
        if (attention.OutputGate is not null)
        {
            config["attn_output_gate"] = true;
        }
        if (head.HeadReusesEmbedding)
        {
            config["tie_word_embeddings"] = true;
        }

        if (hasLinearAttention)
        {
            if (surface.LinearAttention is not { } linearAttn)
            {
                throw new CliException(
                    "component declares linear_attention layers but no linear-attention surface facts — cannot regenerate config.json");
            }
            config["linear_conv_kernel_dim"] = LinearField(linearAttn, "conv_kernel");
            config["linear_num_key_heads"] = LinearField(linearAttn, "key_heads");
            config["linear_key_head_dim"] = LinearField(linearAttn, "key_head_dim");
            config["linear_num_value_heads"] = LinearField(linearAttn, "value_heads");
            config["linear_value_head_dim"] = LinearField(linearAttn, "value_head_dim");
        }

        if (RopeParameters(policies[0].Position) is { } rope)
        {
            config["rope_parameters"] = rope;
        }

        if (quantizeMxfp4)
        {
            config["quantization_config"] = new JsonObject
            {
                ["quant_method"] = "mxfp4",
                ["element_dtype"] = "FP4",
                ["element_grid"] = new JsonArray(BitPattern.Fp4PositiveGrid
                    .Select(v => JsonValue.Create(v)).ToArray()),
                ["block_elements"] = Mxfp4.BlockElements,
                ["block_scale_dtype"] = "F8_E8M0",
            };
        }

        return config.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
    }

    /// <summary>The encoder reads one text-level rope block; this build's
    /// per-layer policies are homogeneous, so layer 0's policy stands for
    /// the stack.</summary>
    private static JsonNode? RopeParameters(PositionPolicy position) => position switch
    {
        PositionRope rope => new JsonObject
        {
            ["rope_type"] = "default",
            ["rope_theta"] = rope.Theta,
        },
        PositionPartialRope partial => new JsonObject
        {
            ["rope_type"] = "default",
            ["rope_theta"] = partial.Theta,
            ["partial_rotary_factor"] = partial.RotaryFactor,
        },
        PositionNone => null,
        PositionUnresolved unresolved => throw new CliException(
            $"position policy '{unresolved.Kind}' is carried unjudged — cannot regenerate rope_parameters"),
        _ => throw new CliException("unsupported position policy — cannot regenerate rope_parameters"),
    };

    private static long LinearField(JsonElement surface, string name)
    {
        if (surface.ValueKind == JsonValueKind.Object &&
            surface.TryGetProperty(name, out var value) &&
            value.TryGetInt64(out long result))
        {
            return result;
        }
        throw new CliException(
            $"linear-attention surface facts declare no '{name}' — cannot regenerate config.json");
    }
}