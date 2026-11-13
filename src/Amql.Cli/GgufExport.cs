using Amql.Gguf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Cli;

/// <summary>
/// Writes a llama.cpp GGUF straight from a container's plan — no family code.
/// llama.cpp names every layer tensor the same way in every architecture
/// (<c>blk.N.attn_q</c>, <c>blk.N.ffn_gate</c>, <c>blk.N.post_attention_norm</c>…),
/// so the plan's operands map onto those names by the role they play. What
/// does differ between llama.cpp architectures is the arithmetic each one
/// assumes; the architecture is therefore chosen from the model's judged
/// properties, and a model whose properties no supported architecture
/// reproduces exactly is refused, naming the property, never approximated.
/// </summary>
public static class GgufExport
{
    public sealed record Report(string OutFile, string Architecture, int Tensors, long Bytes, IReadOnlyList<string> Notes);

    /// <summary>The model's shape as llama.cpp would need to know it.</summary>
    private sealed record Shape(
        NormType NormKind, float NormOffset, bool PreAttn, bool PreFfn, bool PostAttn, bool PostFfn,
        string QkNorm, bool QkvBias, bool OBias, bool FfnBias, bool NormBias, bool Gated, Activation Activation,
        bool Interleaved, double PartialFactor, bool AnyNope, bool Window, bool Parallel,
        bool Multipliers, bool Softcap, bool EmbedScale, bool ScoreScaleStandard, bool HeadGates);

    public static Report Export(Vindex3Container container, string outFile, string componentId = "target")
    {
        if (File.Exists(outFile))
        {
            throw new CliException($"export output '{outFile}' already exists");
        }
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, componentId, store);
        var component = container.Graph!.Component(componentId);
        var surface = component.Execution!;
        var attention = surface.Attention!;
        var head = surface.Head!;

        if (plan.Layers.Any(l => l.Attention is null))
        {
            throw new CliException("GGUF export covers softmax-attention stacks; this one has linear-attention or conv layers " +
                                   "(Qwen3.5 / LFM checkpoints have their own converters: amql-cli gguf <checkpoint>)");
        }
        if (plan.Layers.Any(l => l.Ffn?.Routed is not null))
        {
            throw new CliException("GGUF export of routed (MoE) FFNs is not generic yet — refusing");
        }
        if (plan.Embedding is null || plan.Output is null)
        {
            throw new CliException("GGUF export needs an embedding and an output head");
        }

        var shape = Judge(plan, surface);
        string arch = ChooseArchitecture(shape, out bool permuteQk);
        var notes = new List<string> { $"llama.cpp architecture '{arch}' chosen from the model's properties" };

        var first = plan.Layers[0].Attention!;
        int hidden = plan.HiddenSize;
        var ropes = plan.Layers.Select(l => l.Attention!.Position).OfType<PositionRope>().ToList();
        var ropeTheta = ropes.Select(p => p.Theta).Distinct().ToList();
        if (ropeTheta.Count > 1)
        {
            throw new CliException($"layers use {ropeTheta.Count} different rope bases — llama.cpp's '{arch}' has one");
        }
        // One rescaling for the whole stack, spelled the way llama.cpp reads it:
        // linear as rope.scaling.*, llama3 as the rope_freqs divisor tensor
        // (what convert_hf_to_gguf.py writes for Llama 3.1).
        var scalings = ropes.Select(p => p.Scaling is null ? "none" : System.Text.Json.JsonSerializer.Serialize(p.Scaling)).Distinct().ToList();
        if (scalings.Count > 1)
        {
            throw new CliException($"layers use {scalings.Count} different rope scalings — llama.cpp's '{arch}' has one");
        }
        var scaling = ropes.FirstOrDefault()?.Scaling;
        float[]? ropeFreqs = null;
        switch (scaling?.Type)
        {
            case null:
                break;
            case "linear":
                break;
            case "llama3" when arch == "llama":
            {
                var (baseFreq, _) = RopeScaling.Frequencies(ropeTheta[0], first.HeadDim, null);
                var (scaled, _) = RopeScaling.Frequencies(ropeTheta[0], first.HeadDim, scaling);
                ropeFreqs = baseFreq.Select((f, i) => (float)(f / scaled[i])).ToArray();
                break;
            }
            default:
                throw new CliException(
                    $"rope scaling '{scaling.Type}' has no verified spelling for llama.cpp's '{arch}' in the generic GGUF export — " +
                    "refusing rather than risk a mismatched rescale (linear: any arch; llama3: llama)");
        }

        using var stream = File.Create(outFile);
        using var writer = new GgufWriter(stream);
        writer.Kv("general.architecture", GgufValue.String(arch));
        writer.Kv("general.name", GgufValue.String(container.Index.Model));
        writer.Kv($"{arch}.block_count", GgufValue.Uint32((uint)plan.Layers.Count));
        writer.Kv($"{arch}.context_length", GgufValue.Uint32((uint)(surface.ContextLength ?? 4096)));
        writer.Kv($"{arch}.embedding_length", GgufValue.Uint32((uint)hidden));
        writer.Kv($"{arch}.feed_forward_length", GgufValue.Uint32((uint)surface.Ffn!.IntermediateSize));
        writer.Kv($"{arch}.attention.head_count", GgufValue.Uint32((uint)first.NumQHeads));
        writer.Kv($"{arch}.attention.head_count_kv", GgufValue.Uint32((uint)first.NumKvHeads));
        writer.Kv($"{arch}.attention.key_length", GgufValue.Uint32((uint)first.HeadDim));
        writer.Kv($"{arch}.attention.value_length", GgufValue.Uint32((uint)first.HeadDim));
        writer.Kv(shape.NormKind == NormType.RmsNorm
                ? $"{arch}.attention.layer_norm_rms_epsilon"
                : $"{arch}.attention.layer_norm_epsilon",
            GgufValue.Float32((float)plan.FinalNorm.Eps));
        writer.Kv($"{arch}.rope.freq_base", GgufValue.Float32((float)(ropeTheta.FirstOrDefault() is var t and > 0 ? t : 10_000.0)));
        writer.Kv($"{arch}.rope.dimension_count", GgufValue.Uint32((uint)first.HeadDim));
        writer.Kv($"{arch}.vocab_size", GgufValue.Uint32((uint)head.VocabSize));
        if (scaling is { Type: "linear" })
        {
            writer.Kv($"{arch}.rope.scaling.type", GgufValue.String("linear"));
            writer.Kv($"{arch}.rope.scaling.factor", GgufValue.Float32((float)scaling.Factor));
        }
        if (arch == "granite")
        {
            writer.Kv("granite.attention.scale", GgufValue.Float32((float)attention.ScoreScale));
            writer.Kv("granite.embedding_scale", GgufValue.Float32((float)(head.EmbedScale ?? 1.0)));
            writer.Kv("granite.residual_scale", GgufValue.Float32((float)(surface.ResidualScale ?? 1.0)));
            // llama.cpp divides the logits by logit_scale; the head multiplies.
            writer.Kv("granite.logit_scale", GgufValue.Float32((float)(1.0 / (head.OutputMultiplier ?? 1.0))));
        }
        WriteTokenizer(writer, container.Root, head.VocabSize, notes);

        // ── tensors, by role ────────────────────────────────────────────
        var tensors = new List<(string Name, OperandRef? Operand, long[] Shape, Func<float[], float[]>? Transform)>();
        void Add(string name, OperandRef? operand, Func<float[], long[], float[]>? transform = null)
        {
            if (operand is null)
            {
                return;
            }
            var dims = store.ShapeOf(operand.ObjectId, operand.TensorName);
            tensors.Add((name, operand, dims, transform is null ? null : v => transform(v, dims)));
        }
        Add("token_embd.weight", plan.Embedding.Table);
        for (int l = 0; l < plan.Layers.Count; l++)
        {
            var layer = plan.Layers[l];
            var attn = layer.Attention!;
            string b = $"blk.{l}.";
            Add(b + "attn_norm.weight", layer.PreAttentionNorm?.Weight);
            Add(b + "attn_norm.bias", layer.PreAttentionNorm?.Bias);
            Add(b + "post_attention_norm.weight", layer.PostAttentionNorm?.Weight);
            Add(b + "ffn_norm.weight", layer.PreFfnNorm?.Weight);
            Add(b + "ffn_norm.bias", layer.PreFfnNorm?.Bias);
            Add(b + "post_ffw_norm.weight", layer.PostFfnNorm?.Weight);
            Func<float[], long[], float[]>? permuteQ = permuteQk ? (v, d) => PermuteRows(v, d, attn.NumQHeads) : null;
            Func<float[], long[], float[]>? permuteK = permuteQk ? (v, d) => PermuteRows(v, d, attn.NumKvHeads) : null;
            Add(b + "attn_q.weight", attn.QProj, permuteQ);
            Add(b + "attn_q.bias", attn.QBias, permuteQ);
            Add(b + "attn_k.weight", attn.KProj, permuteK);
            Add(b + "attn_k.bias", attn.KBias, permuteK);
            Add(b + "attn_v.weight", attn.VProj);
            Add(b + "attn_v.bias", attn.VBias);
            Add(b + "attn_output.weight", attn.OProj);
            Add(b + "attn_output.bias", attn.OBias);
            Add(b + "attn_q_norm.weight", attn.QNorm?.Weight);
            Add(b + "attn_k_norm.weight", attn.KNorm?.Weight);
            var ffn = layer.Ffn?.Dense;
            Add(b + "ffn_gate.weight", ffn?.Gate);
            Add(b + "ffn_up.weight", ffn?.Up);
            Add(b + "ffn_down.weight", ffn?.Down);
        }
        Add("output_norm.weight", plan.FinalNorm.Weight);
        Add("output_norm.bias", plan.FinalNorm.Bias);
        if (!plan.Output.ReusesEmbedding)
        {
            Add("output.weight", plan.Output.Projection);
        }
        if (ropeFreqs is not null)
        {
            var freqs = ropeFreqs;
            tensors.Add(("rope_freqs.weight", null, new long[] { freqs.Length }, _ => freqs));
        }

        // Stored floats keep their precision; anything else widens to F32.
        var types = tensors.Select(t =>
        {
            if (t.Operand is null)
            {
                return GgufType.F32;
            }
            var dtype = store.Resolve(t.Operand).Dtype;
            return t.Transform is null && dtype is Dtype.F16 ? GgufType.F16
                : t.Transform is null && dtype is Dtype.BF16 ? GgufType.BF16
                : GgufType.F32;
        }).ToList();
        for (int i = 0; i < tensors.Count; i++)
        {
            writer.Tensor(tensors[i].Name, types[i], tensors[i].Shape.Reverse().ToArray());
        }
        writer.WriteHeader();
        for (int i = 0; i < tensors.Count; i++)
        {
            writer.SeekTensor(i);
            var (_, operand, _, transform) = tensors[i];
            if (types[i] != GgufType.F32)
            {
                foreach (var chunk in store.ReadPayloadChunks(operand!.ObjectId, operand.TensorName))
                {
                    writer.Data.Write(chunk);
                }
            }
            else
            {
                var values = operand is null ? Array.Empty<float>() : store.ResolveWidened(operand).Values;
                if (transform is not null)
                {
                    values = transform(values);
                }
                var bytes = new byte[values.Length * 4];
                Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
                writer.Data.Write(bytes);
            }
            writer.FinishTensor(i);
        }
        writer.Data.Flush();
        if (permuteQk)
        {
            notes.Add("q/k rows permuted to llama.cpp's adjacent-pair rotary layout (as convert_hf_to_gguf.py does for Llama)");
        }
        return new Report(outFile, arch, tensors.Count, new FileInfo(outFile).Length, notes);
    }

    private static Shape Judge(ComponentOpPlan plan, ExecutionSurface surface)
    {
        var layers = plan.Layers;
        bool All(Func<LayerPlan, bool> f) => layers.All(f);
        bool Any(Func<LayerPlan, bool> f) => layers.Any(f);
        var attention = surface.Attention!;
        var a0 = layers[0].Attention!;
        string qk = a0.QNorm is null ? "none" : a0.QNorm.Width == a0.HeadDim ? "head" : "full";
        double partial = layers.Select(l => l.Attention!.Position).OfType<PositionPartialRope>().Select(p => p.RotaryFactor).FirstOrDefault(1.0);
        if (layers.Select(l => l.Attention!.Position).Any(p => p is not (PositionRope or PositionNone or PositionPartialRope)))
        {
            throw new CliException("a layer's position encoding has no llama.cpp spelling in the generic GGUF export");
        }
        foreach (var site in new (string Name, Func<LayerPlan, bool> Has)[]
                 {
                     ("pre-attention", l => l.PreAttentionNorm is not null), ("pre-FFN", l => l.PreFfnNorm is not null),
                     ("attention output", l => l.PostAttentionNorm is not null), ("FFN output", l => l.PostFfnNorm is not null),
                 })
        {
            if (Any(site.Has) && !All(site.Has))
            {
                throw new CliException($"only some layers carry a {site.Name} norm — llama.cpp layouts are uniform");
            }
        }
        long context = surface.ContextLength ?? long.MaxValue;
        return new Shape(
            NormKind: plan.FinalNorm.Kind,
            NormOffset: plan.FinalNorm.WeightOffset,
            PreAttn: layers[0].PreAttentionNorm is not null,
            PreFfn: layers[0].PreFfnNorm is not null,
            PostAttn: layers[0].PostAttentionNorm is not null,
            PostFfn: layers[0].PostFfnNorm is not null,
            QkNorm: qk,
            QkvBias: Any(l => l.Attention!.QBias is not null || l.Attention.KBias is not null || l.Attention.VBias is not null),
            OBias: Any(l => l.Attention!.OBias is not null),
            FfnBias: Any(l => l.Ffn?.Dense is { } d && (d.GateBias ?? d.UpBias ?? d.DownBias) is not null),
            NormBias: plan.FinalNorm.Bias is not null || Any(l => l.PreAttentionNorm?.Bias is not null),
            Gated: All(l => l.Ffn?.Dense?.Gate is not null),
            Activation: surface.Ffn!.Activation,
            Interleaved: attention.RopeInterleaved,
            PartialFactor: partial,
            AnyNope: Any(l => l.Attention!.Position is PositionNone),
            Window: Any(l => l.Attention!.Window is { } w && w < context),
            Parallel: surface.ParallelResidual,
            Multipliers: plan.ResidualScale != 1.0 || surface.Head?.EmbedScale is not null || surface.Head?.OutputMultiplier is not null,
            Softcap: attention.LogitSoftcapping is not null || surface.Head?.FinalLogitSoftcapping is not null,
            EmbedScale: surface.Head?.EmbedScale is not null,
            ScoreScaleStandard: Math.Abs(attention.ScoreScale - 1.0 / Math.Sqrt(attention.HeadDim)) < 1e-9,
            HeadGates: Any(l => l.Attention!.OutputGate || l.Attention.HeadGate is not null));
    }

    /// <summary>
    /// The llama.cpp architecture whose arithmetic is exactly this model's.
    /// Each candidate states what it computes; the first that matches every
    /// property wins. <paramref name="permuteQk"/> is set where llama.cpp
    /// rotates adjacent pairs and the model rotates half-split ones.
    /// </summary>
    private static string ChooseArchitecture(Shape s, out bool permuteQk)
    {
        var common = new List<string>();
        void Need(bool ok, string what)
        {
            if (!ok)
            {
                common.Add(what);
            }
        }
        Need(s.NormKind == NormType.RmsNorm && !s.NormBias, "LayerNorm");
        Need(s.NormOffset == 0f, "(1 + w) norm weights");
        Need(s.Gated && s.Activation == Activation.Silu, $"a {(s.Gated ? "gated" : "plain")} {s.Activation} FFN");
        Need(!s.FfnBias, "FFN biases");
        Need(!s.Window, "sliding-window attention");
        Need(!s.Parallel, "parallel attention/FFN blocks");
        Need(!s.Softcap, "logit softcapping");
        Need(!s.AnyNope, "NoPE layers");
        Need(s.PartialFactor == 1.0, "partial rotary");
        Need(!s.HeadGates, "attention output gates");
        if (common.Count > 0)
        {
            throw new CliException(
                $"no llama.cpp architecture in the generic GGUF export computes this model exactly — it has {string.Join(", ", common)}");
        }

        bool preOnly = s.PreAttn && s.PreFfn && !s.PostAttn && !s.PostFfn;
        bool postOnly = !s.PreAttn && !s.PreFfn && s.PostAttn && s.PostFfn;
        permuteQk = false;
        if (postOnly && s.QkNorm == "full" && !s.QkvBias && !s.OBias && !s.Multipliers && s.ScoreScaleStandard && !s.Interleaved)
        {
            return "olmo2";
        }
        if (preOnly && s.QkNorm == "head" && !s.QkvBias && !s.OBias && !s.Multipliers && s.ScoreScaleStandard && !s.Interleaved)
        {
            return "qwen3";
        }
        if (preOnly && s.QkNorm == "none" && s.QkvBias && !s.OBias && !s.Multipliers && s.ScoreScaleStandard && !s.Interleaved)
        {
            return "qwen2";
        }
        if (preOnly && s.QkNorm == "none" && s.Multipliers && !s.QkvBias && !s.OBias)
        {
            permuteQk = !s.Interleaved;
            return "granite";
        }
        if (preOnly && s.QkNorm == "none" && !s.Multipliers && s.ScoreScaleStandard)
        {
            permuteQk = !s.Interleaved;
            return "llama";
        }
        throw new CliException(
            "no llama.cpp architecture in the generic GGUF export matches this combination: " +
            $"norms {(preOnly ? "pre-only" : postOnly ? "post-only" : "sandwich")}, QK norm {s.QkNorm}, " +
            $"q/k/v bias {s.QkvBias}, o bias {s.OBias}, multipliers {s.Multipliers}, standard score scale {s.ScoreScaleStandard}");
    }

    /// <summary>convert_hf_to_gguf.py's Llama permute: within each head, rows
    /// [first half | second half] become interleaved pairs, so llama.cpp's
    /// adjacent-pair rotary sees what HF's half-split rotary saw.</summary>
    private static float[] PermuteRows(float[] values, long[] shape, int heads)
    {
        long rows = shape[0];
        long cols = values.Length / rows;
        long headRows = rows / heads;
        long half = headRows / 2;
        var result = new float[values.Length];
        for (long h = 0; h < heads; h++)
        {
            for (long i = 0; i < half; i++)
            {
                for (int side = 0; side < 2; side++)
                {
                    long from = h * headRows + side * half + i;
                    long to = h * headRows + 2 * i + side;
                    Array.Copy(values, from * cols, result, to * cols, cols);
                }
            }
        }
        return result;
    }

    private static void WriteTokenizer(GgufWriter writer, string containerRoot, int vocab, List<string> notes)
    {
        string path = Path.Combine(containerRoot, "tokenizer.json");
        if (!File.Exists(path))
        {
            notes.Add("the container carries no tokenizer.json — the GGUF has no tokenizer block and llama.cpp will not load it for text");
            return;
        }
        var tok = GgufConverter.LoadTokenizer(path, vocab);
        writer.Kv("tokenizer.ggml.model", GgufValue.String("gpt2"));
        writer.Kv("tokenizer.ggml.pre", GgufValue.String("default"));
        writer.Kv("tokenizer.ggml.tokens", GgufValue.StringArray(tok.Tokens));
        writer.Kv("tokenizer.ggml.token_type", GgufValue.Int32Array(tok.Types.Select(t => (int)t).ToArray()));
        if (tok.Merges.Length > 0)
        {
            writer.Kv("tokenizer.ggml.merges", GgufValue.StringArray(tok.Merges));
        }
        if (tok.EosTokenId is { } eos)
        {
            writer.Kv("tokenizer.ggml.eos_token_id", GgufValue.Uint32((uint)eos));
        }
        if (tok.PaddingTokenId is { } pad)
        {
            writer.Kv("tokenizer.ggml.padding_token_id", GgufValue.Uint32((uint)pad));
        }
        if (GgufConverter.LoadChatTemplate(containerRoot) is { } template)
        {
            writer.Kv("tokenizer.chat_template", GgufValue.String(template));
        }
        notes.Add("tokenizer written as BPE ('gpt2') with the generic 'default' pre-tokenizer — llama.cpp identifies " +
                  "family-specific pre-tokenizers by hash, so split points may differ slightly from the HF tokenizer");
    }
}
