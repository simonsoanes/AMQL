using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Cli;

/// <summary>
/// The Qwen3.8-Flash-Next (qwen4_exp) tensor-name layout and config.json
/// writer. Maps AMQL logical objects onto the Flash-Next checkpoint tensor
/// contract and builds the <c>qwen4_exp</c> config.json from judged graph
/// facts. An object or surface with no Flash-Next judgement refuses the
/// export by name — exactly like the existing <c>layer_types</c> refusal.
/// </summary>
/// <remarks>
/// The name table is derived from vLLM source
/// (<c>vllm/models/qwen4_exp/</c> on main, 2026-09-22) and pinned against
/// a reference <c>config.json</c> +
/// <c>model.safetensors.index.json</c> from the published
/// <c>Qwen/Qwen3.8-Flash-Next</c> Hub repository.
/// </remarks>
public static class Qwen4NextLayout
{
    public const string Arch = "qwen4-next";
    public const string ModelType = "qwen4_exp";
    public const string TextModelType = "qwen4_exp_text";
    public const string Architectures = "Qwen4ExpForCausalLM";

    // ── HC defaults (required by the config; emitted as zero tensors) ──
    public const int HcCount = 4;
    public const int HcLowrank = 320;
    public const string OutputGateType = "sigmoid";

    // ── PLE defaults (disabled until the value gate passes) ────────────
    public const int NgramSize = 3;
    public const int HeadsPerNgram = 8;
    public const int NgramVocabSizeBase = 20_000_000;
    public const int MakeNgramVocabSizeDivisibleBy = 128;
    public const int SplitNgramParts = 512;

    // ── Tensor-name remapping ──────────────────────────────────────────

    /// <summary>
    /// Translates an AMQL container tensor name (source-binding prefix +
    /// segment-relative name) to the Flash-Next checkpoint name.
    /// Returns the remapped name, or the original if no mapping applies.
    /// </summary>
    public static string RemapTensorName(string objectId, string hfName)
    {
        // Router rename: moe-ify writes {layer}.mlp.router.weight;
        // Flash-Next expects {layer}.mlp.gate.weight.
        if (objectId == "target.decoder_stack" && hfName.Contains(".mlp.router.weight"))
        {
            return hfName.Replace(".mlp.router.weight", ".mlp.gate.weight");
        }

        // The shared expert placeholder follows the Flash-Next convention.
        // (Phase A emits a zero tensor; later phases will copy expert 0.)

        return hfName;
    }

    /// <summary>
    /// Returns every (name, shape, dtype) placeholder tensor the Flash-Next
    /// format requires but the AMQL container does not carry. These are
    /// emitted as zero-initialised tensors so the checkpoint is structurally
    /// loadable. Callers should skip any name that already exists in the
    /// real tensor set.
    /// </summary>
    public static IEnumerable<(string Name, long[] Shape, Dtype Dtype)> PlaceholderTensors(
        Vindex3Container container)
    {
        var graph = container.Graph
            ?? throw new CliException("container records no system graph");
        var component = graph.Components.First(c => c.Role == ComponentRole.PrimaryText);
        int hidden = component.HiddenSize;
        int layers = component.NumLayers;
        int hcHidden = HcCount * hidden;
        var moe = component.Execution?.Ffn?.Moe;
        bool hasMoe = moe is not null;

        // ── HyperConnection tensors (per-layer attn_hc + mlp_hc, final mixer) ──
        for (int l = 0; l < layers; l++)
        {
            // Per-layer attention HC
            yield return ($"model.layers.{l}.attn_hc.hc_norm.weight",
                new long[] { hcHidden }, Dtype.F32);
            yield return ($"model.layers.{l}.attn_hc.input_mix_weight_down.weight",
                new long[] { HcLowrank, hcHidden }, Dtype.F32);
            yield return ($"model.layers.{l}.attn_hc.input_mix_weight_up.weight",
                new long[] { hcHidden, HcLowrank }, Dtype.F32);
            yield return ($"model.layers.{l}.attn_hc.block_inject_weight.weight",
                new long[] { HcCount, hcHidden }, Dtype.F32);

            // Per-layer MLP HC
            yield return ($"model.layers.{l}.mlp_hc.hc_norm.weight",
                new long[] { hcHidden }, Dtype.F32);
            yield return ($"model.layers.{l}.mlp_hc.input_mix_weight_down.weight",
                new long[] { HcLowrank, hcHidden }, Dtype.F32);
            yield return ($"model.layers.{l}.mlp_hc.input_mix_weight_up.weight",
                new long[] { hcHidden, HcLowrank }, Dtype.F32);
            yield return ($"model.layers.{l}.mlp_hc.block_inject_weight.weight",
                new long[] { HcCount, hcHidden }, Dtype.F32);
        }

        // Final HyperConnection mixer
        yield return ("model.hyper_connection_mixer.hc_norm.weight",
            new long[] { hcHidden }, Dtype.F32);
        yield return ("model.hyper_connection_mixer.input_mix_weight_down.weight",
            new long[] { HcLowrank, hcHidden }, Dtype.F32);
        yield return ("model.hyper_connection_mixer.input_mix_weight_up.weight",
            new long[] { hcHidden, HcLowrank }, Dtype.F32);

        // ── Shared expert (one always-on expert) ──
        if (hasMoe)
        {
            int expertIntermediate = moe!.ExpertIntermediateSize;
            for (int l = 0; l < layers; l++)
            {
                yield return ($"model.layers.{l}.mlp.shared_expert.gate_proj.weight",
                    new long[] { expertIntermediate, hidden }, Dtype.F32);
                yield return ($"model.layers.{l}.mlp.shared_expert.up_proj.weight",
                    new long[] { expertIntermediate, hidden }, Dtype.F32);
                yield return ($"model.layers.{l}.mlp.shared_expert.down_proj.weight",
                    new long[] { hidden, expertIntermediate }, Dtype.F32);
            }
        }
    }

    /// <summary>Zeros payload for one placeholder tensor.</summary>
    public static byte[] ZeroPayload(long[] shape, Dtype dtype)
    {
        long elements = 1;
        foreach (long dim in shape) elements *= dim;
        return new byte[elements * dtype.ElementSize()];
    }

    // ── Config.json builder ────────────────────────────────────────────

    /// <summary>
    /// Builds the <c>qwen4_exp</c> config.json from judged graph facts.
    /// An operator or surface fact this build has not judged refuses the
    /// export by name.
    /// </summary>
    public static string BuildConfigJson(
        Vindex3Container container,
        bool quantizeMxfp4,
        IReadOnlyList<string> layerTypes)
    {
        var graph = container.Graph
            ?? throw new CliException("container records no system graph");
        var component = graph.Components.First(c => c.Role == ComponentRole.PrimaryText);
        var surface = component.Execution
            ?? throw new CliException($"component '{component.Id}' declares no execution surface");
        var attention = surface.Attention
            ?? throw new CliException($"component '{component.Id}' declares no attention surface");
        var ffn = surface.Ffn
            ?? throw new CliException($"component '{component.Id}' declares no FFN surface");
        var head = surface.Head
            ?? throw new CliException($"component '{component.Id}' declares no head surface");
        if (surface.ContextLength is not { } contextLength)
        {
            throw new CliException($"component '{component.Id}' declares no context length");
        }

        int hidden = component.HiddenSize;
        int layers = component.NumLayers;
        int intermediate = ffn.IntermediateSize;

        string hiddenAct = ffn.Activation switch
        {
            Activation.Silu => "silu",
            var other => throw new CliException(
                $"FFN activation '{other}' has no judged hidden_act spelling — refusing to approximate"),
        };

        bool hasLinearAttention = layerTypes.Any(t => t == "linear_attention");

        // ── text_config (the inner config) ──
        var textConfig = new JsonObject
        {
            ["model_type"] = TextModelType,
            ["hidden_size"] = hidden,
            ["num_hidden_layers"] = layers,
            ["num_attention_heads"] = attention.NumQHeads,
            ["num_key_value_heads"] = attention.NumKvHeads,
            ["head_dim"] = attention.HeadDim,
            ["intermediate_size"] = intermediate,
            ["hidden_act"] = hiddenAct,
            ["rms_norm_eps"] = surface.Norm.Pre.Eps,
            ["vocab_size"] = head.VocabSize,
            ["max_position_embeddings"] = contextLength,
            ["layer_types"] = new JsonArray(layerTypes.Select(t => JsonValue.Create(t)!).ToArray()),
            ["hc_count"] = HcCount,
            ["hc_lowrank"] = HcLowrank,
            ["ple_layer_ids"] = new JsonArray(),  // disabled by default
            ["ple_embed_dim"] = hidden,
            ["ple_conv_kernel_size"] = 4,
            ["ngram_size"] = NgramSize,
            ["heads_per_ngram"] = HeadsPerNgram,
            ["ngram_vocab_size_base"] = NgramVocabSizeBase,
            ["make_ngram_vocab_size_divisible_by"] = MakeNgramVocabSizeDivisibleBy,
            ["output_gate_type"] = OutputGateType,
        };

        if (ffn.Moe is { } moe)
        {
            textConfig["num_experts"] = moe.Experts;
            textConfig["num_experts_per_tok"] = moe.TopK;
            textConfig["moe_intermediate_size"] = moe.ExpertIntermediateSize;
            textConfig["shared_expert_intermediate_size"] = moe.ExpertIntermediateSize;
            textConfig["routing_policy"] = moe.RoutingPolicy == ExpertRoutingPolicy.NormalisedOverSelected
                ? "normalised_over_selected"
                : "softmax_then_select";
        }
        else
        {
            textConfig["num_experts"] = 0;
            textConfig["num_experts_per_tok"] = 0;
            textConfig["moe_intermediate_size"] = 0;
            textConfig["shared_expert_intermediate_size"] = 0;
        }

        // Rope parameters
        var policies = component.Attention
            ?? throw new CliException($"component '{component.Id}' declares no per-layer attention table");
        if (RopeParameters(policies[0].Position) is { } rope)
        {
            textConfig["rope_parameters"] = rope;
        }

        if (attention.AttentionBias == true)
        {
            textConfig["attention_bias"] = true;
        }
        if (attention.OutputGate is not null)
        {
            textConfig["attn_output_gate"] = true;
        }
        if (head.HeadReusesEmbedding)
        {
            textConfig["tie_word_embeddings"] = true;
        }

        if (hasLinearAttention)
        {
            if (surface.LinearAttention is not { } linearAttn)
            {
                throw new CliException(
                    "component declares linear_attention layers but no linear-attention surface facts");
            }
            textConfig["linear_conv_kernel_dim"] = LinearField(linearAttn, "conv_kernel");
            textConfig["linear_num_key_heads"] = LinearField(linearAttn, "key_heads");
            textConfig["linear_key_head_dim"] = LinearField(linearAttn, "key_head_dim");
            textConfig["linear_num_value_heads"] = LinearField(linearAttn, "value_heads");
            textConfig["linear_value_head_dim"] = LinearField(linearAttn, "value_head_dim");
        }

        if (quantizeMxfp4)
        {
            textConfig["quantization_config"] = new JsonObject
            {
                ["quant_method"] = "mxfp4",
                ["element_dtype"] = "FP4",
                ["element_grid"] = new JsonArray(BitPattern.Fp4PositiveGrid
                    .Select(v => JsonValue.Create(v)).ToArray()),
                ["block_elements"] = Mxfp4.BlockElements,
                ["block_scale_dtype"] = "F8_E8M0",
            };
        }

        // ── top-level config ──
        var config = new JsonObject
        {
            ["model_type"] = ModelType,
            ["architectures"] = new JsonArray(JsonValue.Create(Architectures)!),
            ["text_config"] = textConfig,
        };

        if (head.HeadReusesEmbedding)
        {
            config["tie_word_embeddings"] = true;
        }

        // Vision tower (when materialised)
        var perception = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.Perception);
        if (perception is { } vision &&
            graph.Objects.Any(o => o.Component == vision.Id && o.Representations.Count > 0))
        {
            var visionConfig = new JsonObject
            {
                ["model_type"] = ModelType,
                ["modality"] = "image",
                ["hidden_size"] = vision.HiddenSize,
                ["num_hidden_layers"] = vision.NumLayers,
            };
            if (vision.Perception is { } perceptionFacts &&
                perceptionFacts.ValueKind == JsonValueKind.Object &&
                perceptionFacts.TryGetProperty("transform", out var transform))
            {
                visionConfig["transform"] = JsonNode.Parse(transform.GetRawText());
            }
            config["vision_config"] = visionConfig;
        }
        else
        {
            // vLLM requires vision_config even for text-only models
            config["vision_config"] = new JsonObject
            {
                ["model_type"] = ModelType,
            };
        }

        return config.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
    }

    // ── Layer type judgement ───────────────────────────────────────────

    /// <summary>
    /// Builds the <c>layer_types</c> table for the Flash-Next format.
    /// Every layer operator must be judged; an unjudged operator refuses
    /// the export by name.
    /// </summary>
    public static List<string> BuildLayerTypes(Vindex3Container container)
    {
        var graph = container.Graph
            ?? throw new CliException("container records no system graph");
        var component = graph.Components.First(c => c.Role == ComponentRole.PrimaryText);
        var policies = component.Attention
            ?? throw new CliException($"component '{component.Id}' declares no per-layer attention table");

        var layerTypes = new List<string>();
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
                    break;
                default:
                    throw new CliException(
                        $"layer {l}: operator '{op}' has no judged layer_types spelling — " +
                        "this build regenerates 'full_attention' and 'linear_attention' only; " +
                        "refusing to approximate");
            }
        }
        return layerTypes;
    }

    // ── Helpers (mirror ExportConfig) ──────────────────────────────────

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