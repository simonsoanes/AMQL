using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// A ModernBERT bidirectional encoder, written directly against the
/// container's judged facts — no ONNX, no external runtime. One forward pass
/// over a whole sequence:
/// <code>
///   x = LayerNorm(tok_embeddings[ids])
///   per layer l:
///     h   = l == 0 ? x : LayerNorm(x)                      (layer 0 has no attn norm)
///     q,k,v = split(h · Wqkvᵀ); RoPE(q, k; θ_l, position_ids)
///     x  += softmax(q·kᵀ/√d + mask_l) · v · Woᵀ
///     a, g = split(LayerNorm(x) · Wiᵀ)                    (GeGLU: input first, gate second)
///     x  += (gelu(a) ⊙ g) · Woᵀ
///   return LayerNorm(x)
/// </code>
/// Global layers attend over every allowed key; sliding layers additionally
/// require <c>|pos_q − pos_k| ≤ window/2</c>, measured on the position ids,
/// not the raw index — which is what makes Von's order-invariant mode work
/// (see <see cref="Forward"/>). Every norm is a weight-only LayerNorm; the
/// arithmetic is float32 with double-precision norm statistics.
/// </summary>
public sealed class ModernBertEncoder
{
    private sealed record Layer(
        float[]? AttnNorm, Tensor2D Wqkv, Tensor2D Wo, float[] MlpNorm, Tensor2D Wi, Tensor2D MlpWo,
        bool Sliding, int HalfWindow, float[] InvFreq);

    private readonly float[] _embeddings;
    private readonly float[] _embeddingNorm;
    private readonly Layer[] _layers;
    private readonly float[] _finalNorm;
    private readonly double _eps;

    public int HiddenSize { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public int IntermediateSize { get; }
    public int VocabSize { get; }
    public int NumLayers => _layers.Length;
    public long ContextLength { get; }

    private ModernBertEncoder(float[] embeddings, float[] embeddingNorm, Layer[] layers, float[] finalNorm,
        double eps, int hidden, int heads, int headDim, int intermediate, int vocab, long context)
    {
        _embeddings = embeddings;
        _embeddingNorm = embeddingNorm;
        _layers = layers;
        _finalNorm = finalNorm;
        _eps = eps;
        HiddenSize = hidden;
        NumHeads = heads;
        HeadDim = headDim;
        IntermediateSize = intermediate;
        VocabSize = vocab;
        ContextLength = context;
    }

    /// <summary>Whether the component is an encoder this class serves.</summary>
    public static bool Serves(Vindex3Container container, string componentId = "target") =>
        container.Graph?.Components.FirstOrDefault(c => c.Id == componentId)?.Execution?.Encoder?.Architecture == "ModernBertModel";

    /// <summary>Loads the encoder weights (widened to f32) and checks every
    /// fact the kernels depend on. A fact they do not implement refuses by name.</summary>
    public static ModernBertEncoder Load(Vindex3Container container, OperandStore store, string componentId = "target")
    {
        var graph = container.Graph ?? throw new ContainerException("container records no system graph");
        var component = graph.Component(componentId);
        var surface = component.Execution
            ?? throw new UnsupportedOperatorException($"component '{componentId}' has no execution surface");
        var encoder = surface.Encoder
            ?? throw new UnsupportedOperatorException($"component '{componentId}' is not a bidirectional encoder");
        if (encoder.Architecture != "ModernBertModel")
        {
            throw new UnsupportedOperatorException($"encoder architecture '{encoder.Architecture}' has no kernel in this build");
        }
        if (encoder.NormBias || encoder.MlpBias || surface.Attention?.AttentionBias == true)
        {
            throw new UnsupportedOperatorException("modernbert with biased norms or projections has no kernel in this build");
        }
        if (encoder.FusedFfnLayout != "input_then_gate" || surface.Ffn is not { FfnType: FfnType.Gated, Activation: Activation.Gelu } ffn)
        {
            throw new UnsupportedOperatorException("the encoder FFN is not the GeGLU (input_then_gate, exact gelu) this build serves");
        }
        if (!encoder.EmbeddingNorm || surface.Norm.Pre.Kind != NormType.LayerNorm || surface.Norm.FinalNorm.Kind != NormType.LayerNorm)
        {
            throw new UnsupportedOperatorException("the encoder norms are not the weight-only LayerNorms this build serves");
        }
        var attention = surface.Attention!;
        if (attention.NumKvHeads != attention.NumQHeads)
        {
            throw new UnsupportedOperatorException("grouped-query attention in an encoder has no kernel in this build");
        }
        if (Math.Abs(attention.ScoreScale - 1.0 / Math.Sqrt(attention.HeadDim)) > 1e-12)
        {
            throw new UnsupportedOperatorException($"attention score scale {attention.ScoreScale} is not 1/sqrt(head_dim)");
        }

        int hidden = component.HiddenSize;
        int heads = attention.NumQHeads;
        int headDim = attention.HeadDim;
        int inter = ffn.IntermediateSize;
        if (heads * headDim != hidden)
        {
            throw new UnsupportedOperatorException($"heads x head_dim ({heads}x{headDim}) != hidden {hidden}");
        }

        float[] Vec(string obj, string name, int width)
        {
            var r = store.Resolve(obj, name);
            if (r.Shape.Length != 1 || r.Shape[0] != width)
            {
                throw new ContainerException($"{obj}/{name}: expected [{width}], found [{string.Join("x", r.Shape)}]");
            }
            return BitPattern.WidenToF32(r.Dtype, r.Payload);
        }
        Tensor2D Mat(string obj, string name, int rows, int cols)
        {
            var r = store.Resolve(obj, name);
            if (r.Shape.Length != 2 || r.Shape[0] != rows || r.Shape[1] != cols)
            {
                throw new ContainerException($"{obj}/{name}: expected [{rows}x{cols}], found [{string.Join("x", r.Shape)}]");
            }
            var t = new Tensor2D(BitPattern.WidenToF32(r.Dtype, r.Payload), rows, cols);
            if (CudaShim.Enabled)
            {
                t.DeviceWeightF16 = CudaShim.UploadRawF16(
                    $"{obj}/{name}", Bf16ToFp16Raw(r.Payload), rows, cols);
            }
            return t;
        }

        int vocab = surface.Head?.VocabSize ?? throw new ContainerException("encoder declares no vocabulary size");
        var embeddings = Mat("target.embedding", "tok_embeddings.weight", vocab, hidden).Data;
        var embNorm = Vec("target.embedding", "norm.weight", hidden);

        var policies = component.Attention ?? throw new ContainerException("encoder declares no per-layer attention table");
        var layers = new Layer[component.NumLayers];
        const string stack = "target.encoder_stack";
        for (int l = 0; l < layers.Length; l++)
        {
            var policy = policies[l];
            if (policy.Operator != LayerOperators.Softmax || policy.Position is not PositionRope rope)
            {
                throw new UnsupportedOperatorException($"layer {l}: only RoPE softmax attention is served in an encoder");
            }
            bool sliding = policy.Span switch
            {
                AttentionSpan.Full => false,
                AttentionSpan.Sliding when policy.Window is > 0 => true,
                _ => throw new UnsupportedOperatorException($"layer {l}: span '{policy.Span}' has no encoder kernel"),
            };
            bool hasAttnNorm = store.ContainsTensor(stack, $"{l}.attn_norm.weight");
            if (!hasAttnNorm && !(l == 0 && encoder.FirstLayerSkipsAttentionNorm))
            {
                throw new ContainerException($"layer {l} has no attn_norm and the surface does not say it may skip one");
            }
            layers[l] = new Layer(
                hasAttnNorm ? Vec(stack, $"{l}.attn_norm.weight", hidden) : null,
                Mat(stack, $"{l}.attn.Wqkv.weight", 3 * hidden, hidden),
                Mat(stack, $"{l}.attn.Wo.weight", hidden, hidden),
                Vec(stack, $"{l}.mlp_norm.weight", hidden),
                Mat(stack, $"{l}.mlp.Wi.weight", 2 * inter, hidden),
                Mat(stack, $"{l}.mlp.Wo.weight", hidden, inter),
                sliding,
                sliding ? (int)(policy.Window!.Value / 2) : 0,
                InverseFrequencies(rope.Theta, headDim));
        }
        var finalNorm = Vec("target.final_norm", "weight", hidden);

        return new ModernBertEncoder(embeddings, embNorm, layers, finalNorm, surface.Norm.Pre.Eps,
            hidden, heads, headDim, inter, vocab, surface.ContextLength ?? long.MaxValue);
    }

    /// <summary>Converts raw BF16 bytes → FP16 bytes (both 2-byte per element).
    /// BF16: 1 sign + 8 exponent + 7 mantissa. FP16: 1+5+10. The BF16
    /// exponent range fits in FP16 without clipping; only the mantissa is
    /// truncated.</summary>
    private static byte[] Bf16ToFp16Raw(byte[] bf16)
    {
        int n = bf16.Length / 2;
        var fp16 = new byte[n * 2];
        for (int i = 0; i < n; i++)
        {
            ushort bits = (ushort)(bf16[i * 2] | (bf16[i * 2 + 1] << 8));
            // BF16 → F32: shift left 16 bits, reinterpret as IEEE 754 single
            float f = BitConverter.Int32BitsToSingle(bits << 16);
            // F32 → FP16: round to nearest even (Half ctor then ToUInt16Bits)
            ushort h = BitConverter.HalfToUInt16Bits((Half)f);
            fp16[i * 2] = (byte)h;
            fp16[i * 2 + 1] = (byte)(h >> 8);
        }
        return fp16;
    }

    private static float[] InverseFrequencies(double theta, int headDim)
    {
        var inv = new float[headDim / 2];
        for (int i = 0; i < inv.Length; i++)
        {
            inv[i] = 1f / MathF.Pow((float)theta, (2 * i) / (float)headDim);
        }
        return inv;
    }

    /// <summary>
    /// Runs the encoder over one sequence and returns its final hidden states
    /// (T × hidden).
    /// </summary>
    /// <param name="ids">Token ids, special tokens included.</param>
    /// <param name="positions">Position id per token (RoPE and the sliding
    /// window both use it). <c>null</c> means 0..T-1.</param>
    /// <param name="allowed">Row-major T×T: may query i attend to key j.
    /// <c>null</c> means everything. The diagonal is always allowed, so no row
    /// is ever fully masked.</param>
    public Tensor2D Forward(IReadOnlyList<int> ids, IReadOnlyList<int>? positions = null, bool[]? allowed = null)
    {
        int t = ids.Count;
        if (t == 0)
        {
            throw new ArgumentException("cannot encode an empty sequence", nameof(ids));
        }
        if (t > ContextLength)
        {
            throw new ArgumentException($"sequence of {t} tokens exceeds the {ContextLength}-token context");
        }
        if (allowed is not null && allowed.Length != t * t)
        {
            throw new ArgumentException("allowed must be T x T", nameof(allowed));
        }
        var pos = new int[t];
        for (int i = 0; i < t; i++)
        {
            pos[i] = positions?[i] ?? i;
        }

        int h = HiddenSize;
        var x = new Tensor2D(new float[t * h], t, h);
        for (int i = 0; i < t; i++)
        {
            int id = ids[i];
            if ((uint)id >= (uint)VocabSize)
            {
                throw new ArgumentException($"token id {id} is outside the {VocabSize}-entry vocabulary");
            }
            _embeddings.AsSpan(id * h, h).CopyTo(x.Row(i));
        }
        Norms.ApplyInPlace(x, NormType.LayerNorm, _eps, _embeddingNorm, 0f);

        bool[] fullMask = BuildMask(t, pos, allowed, halfWindow: -1);
        var slidingMasks = new Dictionary<int, bool[]>();

        foreach (var layer in _layers)
        {
            bool[] mask = layer.Sliding
                ? slidingMasks.TryGetValue(layer.HalfWindow, out var m) ? m
                    : slidingMasks[layer.HalfWindow] = BuildMask(t, pos, allowed, layer.HalfWindow)
                : fullMask;

            // ── attention ─────────────────────────────────────────────
            Tensor2D attnIn = x;
            if (layer.AttnNorm is { } an)
            {
                attnIn = new Tensor2D((float[])x.Data.Clone(), t, h);
                Norms.ApplyInPlace(attnIn, NormType.LayerNorm, _eps, an, 0f);
            }
            var qkv = EncoderGemm.MatMulTransposedB(attnIn, layer.Wqkv);
            ApplyRope(qkv, pos, layer.InvFreq);
            var context = Attend(qkv, t, mask);
            var attnOut = EncoderGemm.MatMulTransposedB(context, layer.Wo);
            AddInPlace(x.Data, attnOut.Data);

            // ── GeGLU MLP ─────────────────────────────────────────────
            var mlpIn = new Tensor2D((float[])x.Data.Clone(), t, h);
            Norms.ApplyInPlace(mlpIn, NormType.LayerNorm, _eps, layer.MlpNorm, 0f);
            var wi = EncoderGemm.MatMulTransposedB(mlpIn, layer.Wi);
            var gated = GeGlu(wi, IntermediateSize);
            var mlpOut = EncoderGemm.MatMulTransposedB(gated, layer.MlpWo);
            AddInPlace(x.Data, mlpOut.Data);
        }

        Norms.ApplyInPlace(x, NormType.LayerNorm, _eps, _finalNorm, 0f);
        return x;
    }

    /// <summary>Global layers: the caller's mask. Sliding layers: the caller's
    /// mask AND <c>|pos_i − pos_j| ≤ halfWindow</c>. The diagonal is always on.</summary>
    private static bool[] BuildMask(int t, int[] pos, bool[]? allowed, int halfWindow)
    {
        var mask = new bool[t * t];
        for (int i = 0; i < t; i++)
        {
            for (int j = 0; j < t; j++)
            {
                bool ok = allowed?[i * t + j] ?? true;
                if (halfWindow >= 0 && Math.Abs(pos[i] - pos[j]) > halfWindow)
                {
                    ok = false;
                }
                mask[i * t + j] = ok || i == j;
            }
        }
        return mask;
    }

    /// <summary>Rotate-half RoPE on the q and k thirds of the fused projection,
    /// cos/sin computed in float32 from <c>pos · inv_freq</c> as the reference does.</summary>
    private void ApplyRope(Tensor2D qkv, int[] pos, float[] invFreq)
    {
        int h = HiddenSize, d = HeadDim, half = d / 2;
        Parallel.For(0, qkv.Rows, i =>
        {
            Span<float> cos = stackalloc float[half];
            Span<float> sin = stackalloc float[half];
            for (int f = 0; f < half; f++)
            {
                float angle = pos[i] * invFreq[f];
                cos[f] = MathF.Cos(angle);
                sin[f] = MathF.Sin(angle);
            }
            var row = qkv.Row(i);
            for (int part = 0; part < 2; part++)            // q, then k
            {
                for (int head = 0; head < NumHeads; head++)
                {
                    var v = row.Slice(part * h + head * d, d);
                    for (int f = 0; f < half; f++)
                    {
                        float a = v[f], b = v[f + half];
                        v[f] = a * cos[f] - b * sin[f];
                        v[f + half] = b * cos[f] + a * sin[f];
                    }
                }
            }
        });
    }

    /// <summary>Softmax attention per head over the masked score matrix; the
    /// output is heads concatenated back to T × hidden.</summary>
    private Tensor2D Attend(Tensor2D qkv, int t, bool[] mask)
    {
        int h = HiddenSize, d = HeadDim;
        float scale = 1f / MathF.Sqrt(d);
        var output = new float[t * h];
        Parallel.For(0, NumHeads * t, work =>
        {
            int head = work / t, i = work % t;
            var q = qkv.Row(i).Slice(head * d, d);
            var scores = new float[t];
            float max = float.NegativeInfinity;
            for (int j = 0; j < t; j++)
            {
                if (!mask[i * t + j])
                {
                    scores[j] = float.NegativeInfinity;
                    continue;
                }
                float s = TensorOps.Dot(q, qkv.Row(j).Slice(h + head * d, d)) * scale;
                scores[j] = s;
                if (s > max)
                {
                    max = s;
                }
            }
            double sum = 0;
            for (int j = 0; j < t; j++)
            {
                float e = float.IsNegativeInfinity(scores[j]) ? 0f : MathF.Exp(scores[j] - max);
                scores[j] = e;
                sum += e;
            }
            var o = output.AsSpan(i * h + head * d, d);
            float inv = (float)(1.0 / sum);
            for (int j = 0; j < t; j++)
            {
                float p = scores[j] * inv;
                if (p == 0f)
                {
                    continue;
                }
                var v = qkv.Row(j).Slice(2 * h + head * d, d);
                for (int c = 0; c < d; c++)
                {
                    o[c] += p * v[c];
                }
            }
        });
        return new Tensor2D(output, t, h);
    }

    private static Tensor2D GeGlu(Tensor2D wi, int inter)
    {
        var result = new float[wi.Rows * inter];
        Parallel.For(0, wi.Rows, i =>
        {
            var row = wi.Row(i);
            var dst = result.AsSpan(i * inter, inter);
            for (int c = 0; c < inter; c++)
            {
                dst[c] = Gelu.Exact(row[c]) * row[inter + c];
            }
        });
        return new Tensor2D(result, wi.Rows, inter);
    }

    private static void AddInPlace(float[] dst, float[] src)
    {
        for (int i = 0; i < dst.Length; i++)
        {
            dst[i] += src[i];
        }
    }
}

/// <summary>
/// The encoder's GEMM, <c>C = X·Wᵀ</c>, for many-row activations: a 4×2
/// register-blocked micro-kernel with fused multiply-add, so each vector of X
/// and W loaded from cache feeds several accumulators instead of one. It sums
/// in a different order from <see cref="TensorOps.Dot"/>, which is why it is
/// the encoder's alone — the decoder keeps its kernel and its exact numerics.
/// </summary>
public static class EncoderGemm
{
    /// <summary>The GPU dispatch floor — same threshold as
    /// <see cref="TensorOps"/>. Below this, launch and copy-back overhead
    /// dominate the work.</summary>
    private const long CudaMinimumMacs = 8L << 20; // 8M MACs

    public static Tensor2D MatMulTransposedB(Tensor2D x, Tensor2D w)
    {
        int m = x.Rows, k = x.Cols, n = w.Rows;
        if (w.Cols != k)
        {
            throw new ArgumentException($"shape mismatch: x {m}x{k}, weight {n}x{w.Cols}");
        }

        // GPU fast path: when the weight carries a device-resident FP16
        // copy (uploaded by the encoder loader or the MXFP4/OnDemand paths),
        // dispatch to CUDA before falling into the CPU kernels.
        long macs = (long)m * k * n;
        if (w.DeviceWeightF16 != IntPtr.Zero && macs >= CudaMinimumMacs &&
            CudaShim.TryGemmTransposedB(x.Data, m, k, w.DeviceWeightF16, n, out var gpuResult))
        {
            return new Tensor2D(gpuResult!, m, n);
        }

        int vw = System.Numerics.Vector<float>.Count;
        if (k % vw != 0 || m < 4)
        {
            return TensorOps.MatMulTransposedB(x, w);
        }

        var c = new float[m * n];
        int pairs = (n + 1) / 2;
        int workers = Math.Max(1, Math.Min(ComputeBudget.Cores, pairs / 8));
        int pairsPerWorker = (pairs + workers - 1) / workers;
        // Keep a block of weight rows hot in L2 while every activation row passes over it.
        int blockPairs = Math.Max(1, (128 * 1024) / (2 * k * sizeof(float)));
        Parallel.For(0, workers, worker =>
        {
            int p0 = worker * pairsPerWorker, p1 = Math.Min(pairs, p0 + pairsPerWorker);
            for (int pb = p0; pb < p1; pb += blockPairs)
            {
                int pe = Math.Min(p1, pb + blockPairs);
                int i = 0;
                for (; i + 4 <= m; i += 4)
                {
                    for (int p = pb; p < pe; p++)
                    {
                        int j = 2 * p;
                        if (j + 1 < n)
                        {
                            Kernel4x2(x.Data, w.Data, c, i, j, k, n);
                        }
                        else
                        {
                            for (int r = i; r < i + 4; r++)
                            {
                                c[r * n + j] = TensorOps.Dot(x.Row(r), w.Row(j));
                            }
                        }
                    }
                }
                for (; i < m; i++)
                {
                    for (int j = 2 * pb; j < Math.Min(n, 2 * pe); j++)
                    {
                        c[i * n + j] = TensorOps.Dot(x.Row(i), w.Row(j));
                    }
                }
            }
        });
        return new Tensor2D(c, m, n);
    }

    private static void Kernel4x2(float[] x, float[] w, float[] c, int i, int j, int k, int n)
    {
        var x0 = System.Runtime.InteropServices.MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(x.AsSpan(i * k, k));
        var x1 = System.Runtime.InteropServices.MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(x.AsSpan((i + 1) * k, k));
        var x2 = System.Runtime.InteropServices.MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(x.AsSpan((i + 2) * k, k));
        var x3 = System.Runtime.InteropServices.MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(x.AsSpan((i + 3) * k, k));
        var w0 = System.Runtime.InteropServices.MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(w.AsSpan(j * k, k));
        var w1 = System.Runtime.InteropServices.MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(w.AsSpan((j + 1) * k, k));
        System.Numerics.Vector<float> a00 = default, a01 = default, a10 = default, a11 = default,
            a20 = default, a21 = default, a30 = default, a31 = default;
        for (int t = 0; t < w0.Length; t++)
        {
            var b0 = w0[t];
            var b1 = w1[t];
            var v = x0[t];
            a00 = System.Numerics.Vector.FusedMultiplyAdd(v, b0, a00);
            a01 = System.Numerics.Vector.FusedMultiplyAdd(v, b1, a01);
            v = x1[t];
            a10 = System.Numerics.Vector.FusedMultiplyAdd(v, b0, a10);
            a11 = System.Numerics.Vector.FusedMultiplyAdd(v, b1, a11);
            v = x2[t];
            a20 = System.Numerics.Vector.FusedMultiplyAdd(v, b0, a20);
            a21 = System.Numerics.Vector.FusedMultiplyAdd(v, b1, a21);
            v = x3[t];
            a30 = System.Numerics.Vector.FusedMultiplyAdd(v, b0, a30);
            a31 = System.Numerics.Vector.FusedMultiplyAdd(v, b1, a31);
        }
        c[i * n + j] = System.Numerics.Vector.Sum(a00);
        c[i * n + j + 1] = System.Numerics.Vector.Sum(a01);
        c[(i + 1) * n + j] = System.Numerics.Vector.Sum(a10);
        c[(i + 1) * n + j + 1] = System.Numerics.Vector.Sum(a11);
        c[(i + 2) * n + j] = System.Numerics.Vector.Sum(a20);
        c[(i + 2) * n + j + 1] = System.Numerics.Vector.Sum(a21);
        c[(i + 3) * n + j] = System.Numerics.Vector.Sum(a30);
        c[(i + 3) * n + j + 1] = System.Numerics.Vector.Sum(a31);
    }
}

/// <summary>The exact (erf) GELU, <c>0.5·x·(1 + erf(x/√2))</c> — what
/// <c>nn.GELU()</c> and HF's <c>"gelu"</c> compute. Distinct from the tanh
/// approximation the decoder FFNs use.</summary>
public static class Gelu
{
    public static float Exact(float x) => (float)(0.5 * x * (1.0 + Erf(x * 0.70710678118654752440)));

    /// <summary>erf via the Numerical Recipes erfc Chebyshev fit (fractional
    /// error below 1.2e-7 everywhere — under float32 resolution).</summary>
    public static double Erf(double x)
    {
        double z = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.5 * z);
        double erfc = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
            t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
            t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? 1.0 - erfc : erfc - 1.0;
    }
}

/// <summary>
/// Von's option-marker scoring head: at each marker position,
/// <c>LayerNorm → Linear(H, S) → GELU → LayerNorm → Linear(S, 1)</c>, all with
/// bias, giving one logit per option.
/// </summary>
public sealed class OptionMarkerScorer
{
    private readonly float[] _inNormW, _inNormB, _denseB, _normW, _normB, _outW;
    private readonly Tensor2D _dense;
    private readonly float _outB;
    private readonly double _eps;

    private OptionMarkerScorer(float[] inNormW, float[] inNormB, Tensor2D dense, float[] denseB,
        float[] normW, float[] normB, float[] outW, float outB, double eps)
    {
        (_inNormW, _inNormB, _dense, _denseB, _normW, _normB, _outW, _outB, _eps) =
            (inNormW, inNormB, dense, denseB, normW, normB, outW, outB, eps);
    }

    public OptionMarkerSurface Facts { get; private init; } = null!;

    public static OptionMarkerScorer Load(Vindex3Container container, OperandStore store, string componentId = "target")
    {
        var component = container.Graph!.Component(componentId);
        var facts = component.Execution?.OptionMarker
            ?? throw new UnsupportedOperatorException("the container has no option-marker head — it is not a Von decision model");
        if (facts.Activation != Activation.Gelu)
        {
            throw new UnsupportedOperatorException($"option-marker activation '{facts.Activation}' has no kernel");
        }
        const string obj = "target.option_marker_head";
        float[] V(string name) { var r = store.Resolve(obj, name); return BitPattern.WidenToF32(r.Dtype, r.Payload); }
        var dense = store.Resolve(obj, "dense.weight");
        int s = (int)dense.Shape[0], h = (int)dense.Shape[1];
        if (s != facts.ScorerHiddenSize || h != component.HiddenSize)
        {
            throw new ContainerException($"scorer dense is [{s}x{h}], the surface says [{facts.ScorerHiddenSize}x{component.HiddenSize}]");
        }
        var denseTensor = new Tensor2D(BitPattern.WidenToF32(dense.Dtype, dense.Payload), s, h);
        if (CudaShim.Enabled)
        {
            var fp16 = BitPattern.F32ToFp16Bytes(denseTensor.Data);
            denseTensor.DeviceWeightF16 = CudaShim.UploadRawF16($"{obj}/dense.weight", fp16, s, h);
        }
        return new OptionMarkerScorer(
            V("input_norm.weight"), V("input_norm.bias"),
            denseTensor, V("dense.bias"),
            V("norm.weight"), V("norm.bias"), V("out_proj.weight"), V("out_proj.bias")[0], facts.NormEps)
        {
            Facts = facts,
        };
    }

    /// <summary>One logit per marker row of the encoder output.
    /// Marker rows are gathered into one matrix so the dense projection
    /// beams a single GEMM — fast on CPU (cache-blocked) and dispatchable
    /// on GPU when the scorer's weights carry a device pointer.</summary>
    public float[] Score(Tensor2D hidden, IReadOnlyList<int> markerRows)
    {
        int k = markerRows.Count;
        if (k == 0)
        {
            return Array.Empty<float>();
        }
        int h = hidden.Cols;
        int s = _dense.Rows;

        // Gather marker hidden states → [K, H], apply input LayerNorm.
        var hMat = new Tensor2D(new float[k * h], k, h);
        for (int i = 0; i < k; i++)
        {
            var row = hidden.Row(markerRows[i]);
            row.CopyTo(hMat.Row(i));
            LayerNorm(hMat.Row(i), _inNormW, _inNormB, _eps);
        }

        // Dense projection: [K, H] × [S, H]^T → [K, S] — the single GEMM
        // that replaces the per-marker loop.  When _dense carries a device
        // pointer (scorer weights uploaded at load time) this runs on GPU.
        var projected = TensorOps.MatMulTransposedB(hMat, _dense);

        // GeLU + bias → second LayerNorm → out projection, per marker.
        var logits = new float[k];
        for (int i = 0; i < k; i++)
        {
            var y = projected.Row(i);
            for (int c = 0; c < s; c++)
            {
                y[c] = Gelu.Exact(y[c] + _denseB[c]);
            }
            LayerNorm(y, _normW, _normB, _eps);
            logits[i] = TensorOps.Dot(y, _outW) + _outB;
        }
        return logits;
    }

    private static void LayerNorm(Span<float> row, float[] w, float[] b, double eps)
    {
        double mean = 0;
        foreach (float v in row)
        {
            mean += v;
        }
        mean /= row.Length;
        double variance = 0;
        foreach (float v in row)
        {
            variance += (v - mean) * (v - mean);
        }
        double inv = 1.0 / Math.Sqrt(variance / row.Length + eps);
        for (int i = 0; i < row.Length; i++)
        {
            row[i] = (float)((row[i] - mean) * inv * w[i] + b[i]);
        }
    }
}
