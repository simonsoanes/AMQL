using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Cli;

/// <summary>
/// The Qwen4 (<c>qwen4_exp</c>: Qwen3.8-Flash-Next, the Qwen4-Max lineage)
/// tensor layout and config.json writer. Rewrites a Qwen3.5-family container
/// into the qwen4_exp checkpoint contract <em>exactly</em> — the exported
/// model computes the same function as the source — and refuses by name
/// whatever it cannot express exactly.
/// </summary>
/// <remarks>
/// <para>The contract is pinned against three Qwen-team-aligned sources
/// (2026-11-17): the transformers <c>models/qwen4_exp</c> modelling and
/// conversion mapping, llama.cpp's <c>conversion/qwen4exp.py</c>, and the
/// published <c>Qwen/Qwen3.8-Flash-Next</c> config.json and
/// model.safetensors.index.json, with vLLM's <c>models/qwen4_exp</c>
/// (<c>.research/</c>) as a cross-check.</para>
/// <para>What changes against Qwen3.5, and how each maps exactly:</para>
/// <list type="bullet">
/// <item><b>Gated Residual (hyper-connections)</b> replaces every layer norm:
/// <c>hc_count</c> residual streams, each sub-block reading a gated mix of the
/// normalised streams and writing back through per-stream inject gates. With
/// the mix/inject projections zero the read gate is sigmoid(0) = ½ and the
/// write gate 2·sigmoid(0) = 1, so the streams stay identical and the read is
/// ½·RMSNorm(x)·(1+w_hc). Folding the source pre-norm gain g into
/// w_hc = 2g − 1 reproduces pre-norm residual exactly; the final mixer
/// absorbs the final norm the same way.</item>
/// <item><b>QSA</b>: every full-attention layer carries a block indexer that
/// keeps <c>indexer_budget</c> tokens per query. A budget at least the
/// context length keeps every token, so attention stays dense and exact.
/// vLLM requires budget / compress_ratio ∈ {512, 2048}; the compress ratio
/// (a power of two, as vLLM's fused path wants) grows with the context.</item>
/// <item><b>MoE everywhere</b>: every layer is a routed MoE with an always-on
/// gated shared expert. A dense MLP becomes one routed expert under a zero
/// router (softmax over one logit = 1); a missing shared expert is zero.</item>
/// <item><b>PLE n-gram embedding</b> is optional (<c>ple_layer_ids: []</c>)
/// and left off: there is nothing to carry from a source without one.</item>
/// <item>The linear-attention output gate activation is configurable;
/// Qwen3.5's gated delta net is SiLU.</item>
/// </list>
/// </remarks>
public static class Qwen4NextLayout
{
    public const string Arch = "qwen4-next";
    public const string ModelType = "qwen4_exp";
    public const string TextModelType = "qwen4_exp_text";
    public const string CausalLmArchitecture = "Qwen4ExpForCausalLM";
    public const string ConditionalGenerationArchitecture = "Qwen4ExpForConditionalGeneration";

    // ── Gated Residual (hyper-connections) ─────────────────────────────
    public const int HcCount = 4;
    public const int HcLowrank = 320;

    /// <summary>The gated-delta-net output gate. qwen4_exp defaults to
    /// <c>hidden_act</c>; Flash-Next itself ships "sigmoid", but every gated
    /// delta net AMQL carries is Qwen3.5's, whose z-gate is SiLU.</summary>
    public const string LinearOutputGateType = "silu";

    // ── QSA indexer ────────────────────────────────────────────────────
    public const int IndexerHeads = 4;
    public const int IndexerKvHeads = 1;
    public const int IndexerHeadDim = 128;
    public const int IndexerBlockTopK = 2048;
    public const int MinCompressRatio = 4;

    // ── PLE (disabled: ple_layer_ids is empty) ─────────────────────────
    public const int NgramSize = 3;
    public const int HeadsPerNgram = 8;
    public const int NgramVocabSizeBase = 20_000_000;
    public const int MakeNgramVocabSizeDivisibleBy = 128;
    public const int SplitNgramParts = 512;

    /// <summary>Width of the zero shared expert a dense source gets: any
    /// positive width is exact (its output is zero), so keep it small.</summary>
    public const int ZeroSharedExpertWidth = 128;

    /// <summary>What <see cref="Rewrite"/> established, for the config.</summary>
    public sealed record Facts(
        string TextRoot,
        string Architecture,
        bool DenseAsSingleExpert,
        int SharedExpertWidth,
        int FoldedNorms,
        int Placeholders);

    // ── Tensor-name remapping ──────────────────────────────────────────

    /// <summary>
    /// Translates an AMQL container tensor name to the qwen4_exp checkpoint
    /// name. Returns the original if no mapping applies.
    /// </summary>
    public static string RemapTensorName(string objectId, string hfName)
    {
        // moe-ify writes {layer}.mlp.router.weight; qwen4_exp names it mlp.gate.
        if (objectId == "target.decoder_stack" && hfName.Contains(".mlp.router.weight"))
        {
            return hfName.Replace(".mlp.router.weight", ".mlp.gate.weight");
        }
        return hfName;
    }

    /// <summary>
    /// The QSA compress ratio for a context: the smallest power of two
    /// (at least <see cref="MinCompressRatio"/>) with
    /// <see cref="IndexerBlockTopK"/> × ratio ≥ context, so the token budget
    /// covers the whole context and selection keeps every token.
    /// </summary>
    public static int CompressRatioFor(long contextLength)
    {
        int ratio = MinCompressRatio;
        while ((long)IndexerBlockTopK * ratio < contextLength)
        {
            ratio *= 2;
        }
        return ratio;
    }

    // ── Payload rewrite ────────────────────────────────────────────────

    /// <summary>
    /// Rewrites the exported payloads in place into the qwen4_exp layout:
    /// folds the layer and final norms into Gated Residual tensors, re-centres
    /// QK norms, turns a dense MLP into one routed expert, and adds the
    /// indexer and shared-expert tensors. Refuses by name anything that would
    /// not compute the source function exactly. Runs before anything is
    /// written, so a refusal leaves no output behind.
    /// </summary>
    public static Facts Rewrite(
        Vindex3Container container,
        IReadOnlyList<string> layerTypes,
        string stackPrefix,
        List<TensorPayload> payloads,
        Dictionary<string, string> names,
        bool quantized)
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

        if (attention.OutputGate is null)
        {
            throw new CliException(
                "qwen4_exp attention always applies Qwen3.5's sigmoid output gate (q_proj is 2×q wide); " +
                "the source declares no attention output gate — refusing to approximate");
        }
        // PreOnly's "Post" spec is the FFN pre-norm (post_attention_layernorm);
        // only true post norms (PrePost, Sandwich) have no exact Gated Residual form.
        if (surface.Norm.Placement is NormPlacement.Sandwich or NormPlacement.PrePost)
        {
            throw new CliException(
                "qwen4_exp's Gated Residual folds exactly one pre-norm per sub-block; " +
                "the source declares post norms — refusing to approximate");
        }
        if (!stackPrefix.EndsWith(".layers", StringComparison.Ordinal))
        {
            throw new CliException(
                $"decoder stack prefix '{stackPrefix}' does not end in '.layers' — cannot place qwen4_exp tensors");
        }
        string root = stackPrefix[..^".layers".Length];
        var moe = ffn.Moe;
        if (moe is null && quantized)
        {
            throw new CliException(
                "a dense source exports to qwen4_exp as one routed expert, which fuses gate/up into " +
                "experts.gate_up_proj — that is not done over quantised payloads; export without --quantize");
        }
        if (moe is { DensePrefixLayers: > 0 } || moe is { Hybrid: true })
        {
            throw new CliException(
                "qwen4_exp routes every layer through the MoE block; the source mixes dense and MoE layers — " +
                "refusing to approximate");
        }

        var byName = payloads.ToDictionary(p => p.Name, StringComparer.Ordinal);
        TensorPayload Take(string name, string why)
        {
            if (!byName.Remove(name, out var payload))
            {
                throw new CliException($"qwen4_exp export needs '{name}' ({why}) but the container carries none");
            }
            payloads.Remove(payload);
            names.Remove(name);
            return payload;
        }
        int placeholders = 0;
        void Add(TensorPayload payload, bool placeholder)
        {
            if (byName.ContainsKey(payload.Name))
            {
                throw new CliException($"qwen4_exp export: tensor '{payload.Name}' would be written twice");
            }
            byName[payload.Name] = payload;
            payloads.Add(payload);
            names[payload.Name] = "(qwen4_exp)";
            if (placeholder) placeholders++;
        }

        int hidden = component.HiddenSize;
        int hcHidden = HcCount * hidden;
        Dtype zeroDtype = byName.TryGetValue(root + ".embed_tokens.weight", out var embed) &&
                          embed.Dtype is Dtype.BF16 or Dtype.F16 or Dtype.F32
            ? embed.Dtype
            : Dtype.BF16;
        TensorPayload Zeros(string name, params long[] shape) => new()
        {
            Name = name,
            Dtype = zeroDtype,
            Shape = shape,
            Data = ZeroPayload(shape, zeroDtype),
        };

        void AddGatedResidual(string prefix, TensorPayload norm, float offset, bool combine)
        {
            Add(FoldedHcNorm(prefix + ".hc_norm.weight", norm, offset, hidden), placeholder: false);
            Add(Zeros(prefix + ".input_mix_weight_down.weight", HcLowrank, hcHidden), placeholder: true);
            Add(Zeros(prefix + ".input_mix_weight_up.weight", hcHidden, HcLowrank), placeholder: true);
            if (combine)
            {
                Add(Zeros(prefix + ".block_inject_weight.weight", HcCount, hcHidden), placeholder: true);
            }
        }

        float preOffset = surface.Norm.Pre.WeightOffset;
        float ffnOffset = (surface.Norm.Post ?? surface.Norm.Pre).WeightOffset;
        int folded = 0;
        int? sharedWidth = null;
        for (int l = 0; l < component.NumLayers; l++)
        {
            string p = $"{root}.layers.{l}.";

            // Gated Residual in place of both pre-norms.
            var inputNorm = Take(p + "input_layernorm.weight", "the attention pre-norm, folded into attn_hyper_connection.hc_norm");
            var postNorm = Take(p + "post_attention_layernorm.weight", "the FFN pre-norm, folded into mlp_hyper_connection.hc_norm");
            AddGatedResidual(p + "attn_hyper_connection", inputNorm, preOffset, combine: true);
            AddGatedResidual(p + "mlp_hyper_connection", postNorm, ffnOffset, combine: true);
            folded += 2;

            if (layerTypes[l] == "full_attention")
            {
                // qwen4_exp's q_norm/k_norm are (1 + w); re-centre other conventions.
                foreach (string qk in new[] { "q_norm", "k_norm" })
                {
                    string name = p + "self_attn." + qk + ".weight";
                    if (!byName.ContainsKey(name))
                    {
                        throw new CliException(
                            $"layer {l}: qwen4_exp attention always RMS-normalises q and k per head; " +
                            $"the source carries no '{qk}' — refusing to approximate");
                    }
                    if (attention.QkNormWeightOffset != 1f)
                    {
                        var source = Take(name, "the per-head QK norm");
                        Add(ShiftedNorm(name, source, attention.QkNormWeightOffset - 1f), placeholder: false);
                    }
                }

                // QSA indexer: weights are irrelevant while the budget covers the context.
                string ix = p + "self_attn.indexer.";
                Add(Zeros(ix + "index_qk_proj.weight", (IndexerHeads + IndexerKvHeads) * IndexerHeadDimFor(component), hidden), placeholder: true);
                Add(Zeros(ix + "q_layernorm.weight", IndexerHeadDimFor(component)), placeholder: true);
                Add(Zeros(ix + "k_layernorm.weight", IndexerHeadDimFor(component)), placeholder: true);
            }

            // FFN: qwen4_exp is MoE on every layer.
            if (moe is null)
            {
                var gate = Take(p + "mlp.gate_proj.weight", "the dense MLP gate, routed expert 0");
                var up = Take(p + "mlp.up_proj.weight", "the dense MLP up, routed expert 0");
                var down = Take(p + "mlp.down_proj.weight", "the dense MLP down, routed expert 0");
                if (gate.Dtype != up.Dtype || gate.Shape.Length != 2 || up.Shape.Length != 2 ||
                    gate.Shape[0] != up.Shape[0] || gate.Shape[1] != up.Shape[1] || down.Shape.Length != 2)
                {
                    throw new CliException($"layer {l}: dense MLP gate/up/down shapes or dtypes disagree — cannot fuse expert 0");
                }
                // gate_up_proj [1, 2I, H]: gate rows then up rows (chunk(2) on the output).
                Add(new TensorPayload
                {
                    Name = p + "mlp.experts.gate_up_proj",
                    Dtype = gate.Dtype,
                    Shape = new[] { 1, 2 * gate.Shape[0], gate.Shape[1] },
                    Chunks = Parts(gate).Concat(Parts(up)).ToList(),
                }, placeholder: false);
                Add(new TensorPayload
                {
                    Name = p + "mlp.experts.down_proj",
                    Dtype = down.Dtype,
                    Shape = new[] { 1, down.Shape[0], down.Shape[1] },
                    Chunks = Parts(down).ToList(),
                }, placeholder: false);
                Add(Zeros(p + "mlp.gate.weight", 1, hidden), placeholder: true);
            }

            int width;
            if (byName.TryGetValue(p + "mlp.shared_expert.down_proj.weight", out var sharedDown))
            {
                width = checked((int)sharedDown.Shape[1]);
            }
            else
            {
                width = moe?.ExpertIntermediateSize ?? ZeroSharedExpertWidth;
                Add(Zeros(p + "mlp.shared_expert.gate_proj.weight", width, hidden), placeholder: true);
                Add(Zeros(p + "mlp.shared_expert.up_proj.weight", width, hidden), placeholder: true);
                Add(Zeros(p + "mlp.shared_expert.down_proj.weight", hidden, width), placeholder: true);
            }
            if (sharedWidth is { } seen && seen != width)
            {
                throw new CliException(
                    $"layer {l}: shared expert width {width} differs from {seen} — qwen4_exp has one shared_expert_intermediate_size");
            }
            sharedWidth = width;
            if (!byName.ContainsKey(p + "mlp.shared_expert_gate.weight"))
            {
                Add(Zeros(p + "mlp.shared_expert_gate.weight", 1, hidden), placeholder: true);
            }
        }

        // Final mixer in place of the final norm.
        var finalNorm = Take(root + ".norm.weight", "the final norm, folded into hyper_connection_mixer.hc_norm");
        AddGatedResidual(root + ".hyper_connection_mixer", finalNorm, surface.Norm.FinalNorm.WeightOffset, combine: false);
        folded++;

        bool vision = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.Perception) is { } tower &&
                      graph.Objects.Any(o => o.Component == tower.Id && o.Representations.Count > 0);
        return new Facts(
            TextRoot: root,
            Architecture: vision ? ConditionalGenerationArchitecture : CausalLmArchitecture,
            DenseAsSingleExpert: moe is null,
            SharedExpertWidth: sharedWidth ?? ZeroSharedExpertWidth,
            FoldedNorms: folded,
            Placeholders: placeholders);
    }

    /// <summary>The indexer head must cover the attention's rotary dims.</summary>
    private static int IndexerHeadDimFor(Component component)
    {
        int rotary = component.Attention?[0].Position is PositionPartialRope partial
            ? (int)(component.Execution!.Attention!.HeadDim * partial.RotaryFactor)
            : component.Execution!.Attention!.HeadDim;
        return Math.Max(IndexerHeadDim, rotary);
    }

    /// <summary>hc_norm = 2·(offset + w) − 1 per stream: the read gate's ½
    /// times (1 + hc_norm) then equals the source gain offset + w.</summary>
    private static TensorPayload FoldedHcNorm(string name, TensorPayload norm, float offset, int hidden)
    {
        float[] w = Widen(norm);
        if (w.Length != hidden)
        {
            throw new CliException($"'{norm.Name}' holds {w.Length} values, expected hidden size {hidden}");
        }
        var folded = new float[HcCount * hidden];
        for (int s = 0; s < HcCount; s++)
        {
            for (int i = 0; i < hidden; i++)
            {
                folded[s * hidden + i] = 2f * (offset + w[i]) - 1f;
            }
        }
        return F32(name, folded, new long[] { folded.Length });
    }

    private static TensorPayload ShiftedNorm(string name, TensorPayload norm, float shift)
    {
        float[] w = Widen(norm);
        for (int i = 0; i < w.Length; i++) w[i] += shift;
        return F32(name, w, norm.Shape);
    }

    private static float[] Widen(TensorPayload payload)
    {
        if (!payload.Dtype.IsWidenableToF32())
        {
            throw new CliException($"'{payload.Name}' is stored as {payload.Dtype.Label()}, which does not widen to f32");
        }
        return BitPattern.WidenToF32(payload.Dtype, Parts(payload).SelectMany(b => b).ToArray());
    }

    private static TensorPayload F32(string name, float[] values, long[] shape) => new()
    {
        Name = name,
        Dtype = Dtype.F32,
        Shape = shape,
        Data = MemoryMarshal.AsBytes(values.AsSpan()).ToArray(),
    };

    private static IEnumerable<byte[]> Parts(TensorPayload payload) =>
        payload.Chunks ?? (IEnumerable<byte[]>)new[] { payload.Data };

    /// <summary>Zeros payload for one placeholder tensor.</summary>
    public static byte[] ZeroPayload(long[] shape, Dtype dtype)
    {
        long elements = 1;
        foreach (long dim in shape) elements *= dim;
        return new byte[elements * dtype.ElementSize()];
    }

    // ── Config.json builder ────────────────────────────────────────────

    /// <summary>
    /// Builds the <c>qwen4_exp</c> config.json from judged graph facts and
    /// what <see cref="Rewrite"/> established. An operator or surface fact
    /// this build has not judged refuses the export by name.
    /// </summary>
    public static string BuildConfigJson(
        Vindex3Container container,
        bool quantizeMxfp4,
        IReadOnlyList<string> layerTypes,
        Facts facts)
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

        string hiddenAct = ffn.Activation switch
        {
            Activation.Silu => "silu",
            var other => throw new CliException(
                $"FFN activation '{other}' has no judged hidden_act spelling — refusing to approximate"),
        };

        bool hasLinearAttention = layerTypes.Any(t => t == "linear_attention");
        int compressRatio = CompressRatioFor(contextLength);

        // ── text_config (the inner config) ──
        var textConfig = new JsonObject
        {
            ["model_type"] = TextModelType,
            ["hidden_size"] = hidden,
            ["num_hidden_layers"] = layers,
            ["num_attention_heads"] = attention.NumQHeads,
            ["num_key_value_heads"] = attention.NumKvHeads,
            ["head_dim"] = attention.HeadDim,
            ["hidden_act"] = hiddenAct,
            ["rms_norm_eps"] = surface.Norm.Pre.Eps,
            ["vocab_size"] = head.VocabSize,
            ["max_position_embeddings"] = contextLength,
            ["layer_types"] = new JsonArray(layerTypes.Select(t => JsonValue.Create(t)!).ToArray()),
            ["hc_count"] = HcCount,
            ["hc_lowrank"] = HcLowrank,
            ["indexer_n_heads"] = IndexerHeads,
            ["indexer_kv_heads"] = IndexerKvHeads,
            ["indexer_head_dim"] = IndexerHeadDimFor(component),
            ["indexer_compress_ratio"] = compressRatio,
            // budget ≥ context: every token is selected, attention stays dense.
            ["indexer_budget"] = IndexerBlockTopK * compressRatio,
            ["ple_layer_ids"] = new JsonArray(),
            ["ple_embed_dim"] = hidden,
            ["ple_conv_kernel_size"] = 4,
            ["ngram_size"] = NgramSize,
            ["heads_per_ngram"] = HeadsPerNgram,
            ["ngram_vocab_size_base"] = NgramVocabSizeBase,
            ["make_ngram_vocab_size_divisible_by"] = MakeNgramVocabSizeDivisibleBy,
            ["split_ngram_parts"] = SplitNgramParts,
            ["output_gate_type"] = LinearOutputGateType,
            ["shared_expert_intermediate_size"] = facts.SharedExpertWidth,
        };

        if (ffn.Moe is { } moe)
        {
            textConfig["num_experts"] = moe.Experts;
            textConfig["num_experts_per_tok"] = moe.TopK;
            textConfig["moe_intermediate_size"] = moe.ExpertIntermediateSize;
            textConfig["norm_topk_prob"] = moe.RoutingPolicy == ExpertRoutingPolicy.NormalisedOverSelected;
        }
        else
        {
            // The dense MLP is routed expert 0 under a zero router.
            textConfig["num_experts"] = 1;
            textConfig["num_experts_per_tok"] = 1;
            textConfig["moe_intermediate_size"] = ffn.IntermediateSize;
            textConfig["norm_topk_prob"] = true;
        }

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
        textConfig["attn_output_gate"] = true;
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
            ["architectures"] = new JsonArray(JsonValue.Create(facts.Architecture)!),
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
                ["model_type"] = "qwen4_exp_vision",
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
                ["model_type"] = "qwen4_exp_vision",
            };
        }

        return config.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
    }

    // ── Layer type judgement ───────────────────────────────────────────

    /// <summary>
    /// Builds the <c>layer_types</c> table. qwen4_exp checkpoints spell their
    /// QSA layers "full_attention" (transformers reads them as indexed
    /// attention). Every layer operator must be judged; an unjudged operator
    /// refuses the export by name.
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
