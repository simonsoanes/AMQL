using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// The multi-stream gated residual (hyper-connections) on the managed path:
/// the residual carried as several parallel streams, each sub-block reading a
/// gated mean of them and writing back into every stream with its own gate;
/// and the hashed n-gram embedding with its sharded table, added to those
/// streams before the mixer reads them. Qwen3.8-Flash-Next (qwen4_exp) is one
/// model built this way.
/// </summary>
public sealed partial class GenericRuntime
{
    /// <summary>The token ids of the pass in flight (set by <see cref="Embed"/>):
    /// the n-gram embedding reads ids, not hidden states.</summary>
    private int[]? _passTokens;

    // Per n-gram-embedding layer: the last (n-1) tokens, and the dilated conv's history.
    private readonly Dictionary<int, int[]> _ngramHistory = new();
    private readonly Dictionary<int, float[]> _ngramConvState = new();
    private readonly Dictionary<int, NgramHashing.Layout> _ngramLayouts = new();

    private void ResetHyperConnectionState()
    {
        _passTokens = null;
        _ngramHistory.Clear();
        _ngramConvState.Clear();
    }

    private static Tensor2D RepeatStreams(Tensor2D hidden, int streams)
    {
        int width = hidden.Cols;
        var data = GC.AllocateUninitializedArray<float>(hidden.Rows * width * streams);
        for (int r = 0; r < hidden.Rows; r++)
        {
            var row = hidden.Data.AsSpan(r * width, width);
            for (int s = 0; s < streams; s++)
            {
                row.CopyTo(data.AsSpan((r * streams + s) * width, width));
            }
        }
        return new Tensor2D(data, hidden.Rows, width * streams);
    }

    // ── the layer: n-gram embedding, then a gated residual round each sub-block

    private Tensor2D RunHyperLayer(Tensor2D x, int layer, LayerPlan layerPlan, int[] queryPositions, int[] kvPositions, bool appendKv,
        int[][]? positionStreams)
    {
        if (x.Cols != _plan.ResidualWidth)
        {
            throw new ArgumentException($"layer {layer}: a {x.Cols}-wide residual, the gated residual carries {_plan.ResidualWidth}");
        }
        if (layerPlan.Ngram is { } ngram)
        {
            long tn = TraceStart;
            var ple = RunNgramEmbedding(x, layer, ngram);
            AddInPlace(x, ple);
            Report(layer, "ngram_embedding", ngram.KeyProj, ple, tn);
        }

        var mixerHc = layerPlan.MixerHyperConnection!;
        long t = TraceStart;
        var (h, inject) = HyperRead(x, mixerHc);
        Report(layer, "attn_hyper_read", mixerHc.Norm, h, t);
        var mixerOut = RunMixer(h, layer, layerPlan, queryPositions, kvPositions, appendKv, positionStreams);
        HyperWrite(x, mixerOut, inject!, mixerHc);

        var ffnHc = layerPlan.FfnHyperConnection!;
        t = TraceStart;
        var (hf, ffnInject) = HyperRead(x, ffnHc);
        Report(layer, "mlp_hyper_read", ffnHc.Norm, hf, t);
        var ffnOut = RunFfn(hf, layer, layerPlan.Ffn!);
        HyperWrite(x, ffnOut, ffnInject!, ffnHc);

        ApplyPatches(x, layer, queryPositions);
        Report(layer, "residual_out", null, x, 0);
        return x;
    }

    // ── gated residual: read and write ───────────────────────────────────────

    /// <summary>The read: normalise each stream, gate it with a low-rank mix of
    /// all of them, and average the streams into one hidden state. With an
    /// inject weight, also the per-stream write gates
    /// <c>2·sigmoid(inject·n / S)</c>.</summary>
    private (Tensor2D Read, Tensor2D? Inject) HyperRead(Tensor2D x, HyperConnectionOp op)
    {
        int streams = op.Streams, hidden = op.HiddenSize, width = op.Width;
        if (x.Cols != width)
        {
            throw new ArgumentException($"gated residual reads {width} features, the residual has {x.Cols}");
        }
        var normed = x.Clone();
        ApplyGroupedNorm(normed, _weights.Vector(op.Norm, width), hidden, op.Eps, op.NormWeightOffset);

        var mix = TensorOps.MatMulTransposedB(normed, _weights.Matrix(op.Down, op.Lowrank, width));
        for (int i = 0; i < mix.Data.Length; i++)
        {
            mix.Data[i] = Activations.Silu(mix.Data[i] / streams);
        }
        var gates = TensorOps.MatMulTransposedB(mix, _weights.Matrix(op.Up, width, op.Lowrank));

        var read = new float[x.Rows * hidden];
        for (int r = 0; r < x.Rows; r++)
        {
            var g = gates.Data.AsSpan(r * width, width);
            var n = normed.Data.AsSpan(r * width, width);
            var outRow = read.AsSpan(r * hidden, hidden);
            for (int s = 0; s < streams; s++)
            {
                for (int i = 0; i < hidden; i++)
                {
                    outRow[i] += Sigmoid(g[s * hidden + i]) * n[s * hidden + i];
                }
            }
            for (int i = 0; i < hidden; i++)
            {
                outRow[i] /= streams;
            }
        }

        Tensor2D? inject = null;
        if (op.Inject is { } injectRef)
        {
            inject = TensorOps.MatMulTransposedB(normed, _weights.Matrix(injectRef, streams, width));
            for (int i = 0; i < inject.Data.Length; i++)
            {
                inject.Data[i] = 2f * Sigmoid(inject.Data[i] / streams);
            }
        }
        return (new Tensor2D(read, x.Rows, hidden), inject);
    }

    /// <summary>The write: <c>x_s += out · inject_s</c> on every stream.</summary>
    private static void HyperWrite(Tensor2D x, Tensor2D output, Tensor2D inject, HyperConnectionOp op)
    {
        int streams = op.Streams, hidden = op.HiddenSize;
        for (int r = 0; r < x.Rows; r++)
        {
            var o = output.Data.AsSpan(r * hidden, hidden);
            for (int s = 0; s < streams; s++)
            {
                float g = inject.Data[r * streams + s];
                var stream = x.Data.AsSpan(r * x.Cols + s * hidden, hidden);
                for (int i = 0; i < hidden; i++)
                {
                    stream[i] += o[i] * g;
                }
            }
        }
    }

    /// <summary>Grouped RMSNorm: each <paramref name="group"/>-wide slice
    /// of a row normalised on its own, then scaled by its slice of the weight.</summary>
    private static void ApplyGroupedNorm(Tensor2D x, float[] weight, int group, double eps, float offset)
    {
        int cols = x.Cols;
        var data = x.Data;
        void Row(int r)
        {
            for (int start = 0; start < cols; start += group)
            {
                Norms.ApplyRow(data.AsSpan(r * cols + start, group), NormType.RmsNorm, eps, weight.AsSpan(start, group), offset);
            }
        }
        if (x.Rows >= 8)
        {
            KernelPool.For(x.Rows, Row);
            return;
        }
        for (int r = 0; r < x.Rows; r++)
        {
            Row(r);
        }
    }

    // ── n-gram embedding ─────────────────────────────────────────────────────

    private Tensor2D RunNgramEmbedding(Tensor2D x, int layer, NgramEmbeddingOp op)
    {
        var tokens = _passTokens ?? throw new InvalidOperationException(
            $"layer {layer}: the n-gram embedding reads token ids, and this pass did not start at the embedding");
        int rows = x.Rows;
        if (tokens.Length != rows)
        {
            throw new InvalidOperationException($"layer {layer}: {rows} residual rows but {tokens.Length} tokens in the pass");
        }
        var surface = op.Surface;
        int n = surface.NgramSize, heads = surface.NgramHeads, rowWidth = op.RowWidth, embedDim = surface.EmbedDim;
        int hidden = op.HiddenSize, streams = op.Streams, width = op.Width;
        if (!_ngramLayouts.TryGetValue(layer, out var layout))
        {
            _ngramLayouts[layer] = layout = NgramHashing.For(surface, op.PleIndex);
        }
        if (!_ngramHistory.TryGetValue(layer, out var previous))
        {
            previous = Enumerable.Repeat(surface.EosTokenId, n - 1).ToArray();
        }

        // Each token's hashed n-gram rows, gathered from whichever shard holds them.
        var history = new int[previous.Length + rows];
        previous.CopyTo(history, 0);
        tokens.CopyTo(history, previous.Length);
        var embedding = new float[rows * embedDim];
        Span<long> ids = stackalloc long[heads];
        Span<long> local = stackalloc long[1];
        for (int r = 0; r < rows; r++)
        {
            int at = previous.Length + r;
            NgramHashing.Rows(layout, surface, history.AsSpan(0, at), history[at], ids);
            for (int head = 0; head < heads; head++)
            {
                int shard = ShardOf(op.ShardStarts, ids[head]);
                local[0] = ids[head] - op.ShardStarts[shard];
                _weights.ReadRows(op.Shards[shard], local, embedding.AsSpan(r * embedDim + head * rowWidth, rowWidth));
            }
        }
        _ngramHistory[layer] = history[^(n - 1)..];
        var embeddings = new Tensor2D(embedding, rows, embedDim);

        // key per stream, one shared value; the normalised streams gate it.
        var key = TensorOps.MatMulTransposedB(embeddings, _weights.Matrix(op.KeyProj, width, embedDim));
        ApplyGroupedNorm(key, _weights.Vector(op.NormKey, width), hidden, op.Eps, op.NormWeightOffset);
        var value = TensorOps.MatMulTransposedB(embeddings, _weights.Matrix(op.ValueProj, hidden, embedDim));
        var query = x.Clone();
        ApplyGroupedNorm(query, _weights.Vector(op.NormQuery, width), hidden, op.Eps, op.NormWeightOffset);

        var gated = new float[rows * width];
        double scale = 1.0 / Math.Sqrt(hidden);
        for (int r = 0; r < rows; r++)
        {
            for (int s = 0; s < streams; s++)
            {
                var k = key.Data.AsSpan(r * width + s * hidden, hidden);
                var q = query.Data.AsSpan(r * width + s * hidden, hidden);
                double gate = TensorOps.Dot(k, q) * scale;
                // gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
                gate = Math.Sign(gate) * Math.Sqrt(Math.Max(Math.Abs(gate), 1e-6));
                float g = (float)(1.0 / (1.0 + Math.Exp(-gate)));
                var v = value.Data.AsSpan(r * hidden, hidden);
                var target = gated.AsSpan(r * width + s * hidden, hidden);
                for (int i = 0; i < hidden; i++)
                {
                    target[i] = g * v[i];
                }
            }
        }
        var gatedValue = new Tensor2D(gated, rows, width);
        var convInput = gatedValue.Clone();
        ApplyGroupedNorm(convInput, _weights.Vector(op.NormConv, width), hidden, op.Eps, op.NormWeightOffset);

        // Dilated depthwise causal conv (dilation = n-gram size), SiLU, added to
        // the gated value. The history is the last (K-1)·D rows of conv input.
        int kernel = surface.ConvKernel, dilation = n;
        int span = (kernel - 1) * dilation;
        var weight = _weights.Vector(op.Conv, width * kernel);
        if (!_ngramConvState.TryGetValue(layer, out var carried))
        {
            carried = new float[span * width];
        }
        var input = convInput.Data;
        var result = gatedValue.Data;
        KernelPool.For(rows, r =>
        {
            for (int c = 0; c < width; c++)
            {
                float sum = 0f;
                for (int j = 0; j < kernel; j++)
                {
                    int pos = r - (kernel - 1 - j) * dilation;
                    float v = pos >= 0 ? input[pos * width + c] : carried[(span + pos) * width + c];
                    sum += weight[c * kernel + j] * v;
                }
                result[r * width + c] += Activations.Silu(sum);
            }
        });
        var next = new float[span * width];
        for (int i = 0; i < span; i++)
        {
            int pos = rows - span + i;
            if (pos >= 0)
            {
                Array.Copy(input, pos * width, next, i * width, width);
            }
            else
            {
                Array.Copy(carried, (span + pos) * width, next, i * width, width);
            }
        }
        _ngramConvState[layer] = next;
        return gatedValue;
    }

    private static int ShardOf(IReadOnlyList<long> starts, long row)
    {
        int lo = 0, hi = starts.Count - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) / 2;
            if (starts[mid] <= row)
            {
                lo = mid;
            }
            else
            {
                hi = mid - 1;
            }
        }
        return lo;
    }
}
