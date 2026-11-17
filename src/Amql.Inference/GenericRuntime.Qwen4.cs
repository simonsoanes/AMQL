using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// Qwen4 (qwen4_exp) on the managed path: the Gated Residual that carries the
/// residual as several streams, the hashed n-gram embedding (PLE) with its
/// sharded table, and QSA's indexer-selected sparse attention. Transcribed
/// from transformers' <c>modeling_qwen4_exp.py</c>; each method names the
/// upstream module it mirrors.
/// </summary>
public sealed partial class GenericRuntime
{
    /// <summary>The token ids of the pass in flight (set by <see cref="Embed"/>):
    /// the n-gram embedding reads ids, not hidden states.</summary>
    private int[]? _passTokens;

    // Per PLE layer: the last (n-1) tokens, and the dilated conv's history.
    private readonly Dictionary<int, int[]> _ngramHistory = new();
    private readonly Dictionary<int, float[]> _pleConvState = new();
    private readonly Dictionary<int, NgramHashing.Layout> _ngramLayouts = new();

    // Per QSA layer: every position's raw indexer key, and the finished block keys.
    private readonly Dictionary<int, List<float>> _indexKeys = new();
    private readonly Dictionary<int, List<float[]>> _indexBlocks = new();

    private void ResetQwen4State()
    {
        _passTokens = null;
        _ngramHistory.Clear();
        _pleConvState.Clear();
        _indexKeys.Clear();
        _indexBlocks.Clear();
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

    // ── the layer (Qwen4ExpTextDecoderLayer) ────────────────────────────────

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

    // ── Gated Residual (Qwen4ExpTextGatedResidual) ─────────────────────────

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

    /// <summary>Qwen4's grouped RMSNorm: each <paramref name="group"/>-wide slice
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

    // ── n-gram embedding (Qwen4ExpTextPLELayer) ────────────────────────────

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
        if (!_pleConvState.TryGetValue(layer, out var carried))
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
        _pleConvState[layer] = next;
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

    // ── QSA (Qwen4ExpTextQSAIndexer) ───────────────────────────────────────

    /// <summary>
    /// Runs the indexer over this pass's rows and returns, per query row, the
    /// key positions it keeps (ascending) — or null when no row exceeds its
    /// budget, so the dense attention is exactly what QSA computes. A row
    /// within budget gets a null entry (every visible key).
    /// </summary>
    private int[]?[]? SelectSparseKeys(Tensor2D h, int layer, SparseIndexerOp ix, int[] queryPositions, int[] kvPositions,
        bool appendKv, int kvSeq, int[][]? positionStreams)
    {
        int hd = ix.HeadDim, heads = ix.Heads, ratio = ix.CompressRatio;
        int hidden = _plan.HiddenSize;
        var qk = TensorOps.MatMulTransposedB(h, _weights.Matrix(ix.QkProj, (heads + 1) * hd, hidden));
        if (!_indexKeys.TryGetValue(layer, out var keys))
        {
            _indexKeys[layer] = keys = new List<float>();
        }
        if (appendKv)
        {
            for (int r = 0; r < qk.Rows; r++)
            {
                keys.AddRange(qk.Data.AsSpan(r * qk.Cols + heads * hd, hd).ToArray());
            }
        }
        if (keys.Count != kvSeq * hd)
        {
            throw new InvalidOperationException(
                $"layer {layer}: the QSA indexer holds {keys.Count / hd} keys for {kvSeq} cached positions");
        }
        int first = kvPositions.Length - kvSeq;
        for (int j = 0; j < kvSeq; j++)
        {
            if (first < 0 || kvPositions[first + j] != j)
            {
                throw new UnsupportedOperatorException($"layer {layer}: QSA blocks are judged over contiguous positions from 0");
            }
        }

        bool any = false;
        foreach (int p in queryPositions)
        {
            any |= (p + 1) / ratio > ix.BlockTopK;
        }
        if (!any)
        {
            return null;
        }
        if (positionStreams is not null)
        {
            throw new UnsupportedOperatorException(
                $"layer {layer}: QSA block selection over multimodal positions has no managed implementation");
        }

        // Queries: per-head norm, then the attention's rotary.
        var q = ColRange(qk, 0, heads * hd);
        ApplyPerHead(q, heads, hd, _weights.Vector(ix.QNorm, hd), ix.Eps, ix.NormWeightOffset);
        Rope.Apply(q, heads, hd, ix.Position, queryPositions, ix.RopeInterleaved);

        // Block keys: the mean of each complete block's raw keys, normalised and
        // rotated at the block's first position. A block never changes once
        // complete, so each is made once.
        if (!_indexBlocks.TryGetValue(layer, out var blocks))
        {
            _indexBlocks[layer] = blocks = new List<float[]>();
        }
        int complete = kvSeq / ratio;
        if (blocks.Count < complete)
        {
            int fresh = complete - blocks.Count;
            var pooled = new float[fresh * hd];
            var starts = new int[fresh];
            for (int b = 0; b < fresh; b++)
            {
                int block = blocks.Count + b;
                starts[b] = block * ratio;
                for (int d = 0; d < hd; d++)
                {
                    float sum = 0f;
                    for (int i = 0; i < ratio; i++)
                    {
                        sum += keys[(block * ratio + i) * hd + d];
                    }
                    pooled[b * hd + d] = sum / ratio;
                }
            }
            var pooledKeys = new Tensor2D(pooled, fresh, hd);
            ApplyPerHead(pooledKeys, 1, hd, _weights.Vector(ix.KNorm, hd), ix.Eps, ix.NormWeightOffset);
            Rope.Apply(pooledKeys, 1, hd, ix.Position, starts, ix.RopeInterleaved);
            for (int b = 0; b < fresh; b++)
            {
                blocks.Add(pooledKeys.Data.AsSpan(b * hd, hd).ToArray());
            }
        }

        var selection = new int[]?[queryPositions.Length];
        double scale = 1.0 / Math.Sqrt(hd);
        KernelPool.For(queryPositions.Length, r =>
        {
            int p = queryPositions[r];
            int visible = p + 1;
            int nb = visible / ratio;
            if (nb <= ix.BlockTopK)
            {
                return; // every visible key
            }
            var scores = new double[nb];
            for (int b = 0; b < nb; b++)
            {
                double total = 0;
                for (int head = 0; head < heads; head++)
                {
                    double dot = TensorOps.Dot(q.Data.AsSpan(r * q.Cols + head * hd, hd), blocks[b]);
                    total += Math.Max(dot, 0.0);
                }
                scores[b] = total * scale;
            }
            var order = Enumerable.Range(0, nb).OrderByDescending(b => scores[b]).ThenBy(b => b).Take(ix.BlockTopK).ToList();
            var keep = new List<int>(ix.BlockTopK * ratio + ratio);
            foreach (int b in order)
            {
                for (int i = 0; i < ratio; i++)
                {
                    keep.Add(b * ratio + i);
                }
            }
            for (int pos = nb * ratio; pos <= p; pos++)
            {
                keep.Add(pos); // the incomplete tail block is always kept
            }
            keep.Sort();
            selection[r] = keep.ToArray();
        });
        return selection;
    }

    /// <summary>Softmax attention over each query's selected keys (null: every
    /// key up to its position), GQA as the dense kernel does it.</summary>
    private static Tensor2D SparseAttention(Tensor2D q, float[] keys, int keyWidth, float[] values, int valueWidth,
        AttentionOp attn, int[]?[] selection, int[] queryPositions)
    {
        int heads = attn.NumQHeads, hd = attn.HeadDim, rep = attn.NumQHeads / attn.NumKvHeads;
        var output = Tensor2D.Zeros(q.Rows, attn.QDim);
        KernelPool.For(q.Rows * heads, index =>
        {
            int r = index / heads, head = index % heads, kvHead = head / rep;
            var chosen = selection[r];
            int count = chosen?.Length ?? queryPositions[r] + 1;
            var qSlice = q.Data.AsSpan(r * q.Cols + head * hd, hd);
            var weights = new float[count];
            for (int i = 0; i < count; i++)
            {
                int j = chosen?[i] ?? i;
                double score = TensorOps.Dot(qSlice, keys.AsSpan(j * keyWidth + kvHead * hd, hd)) * attn.ScoreScale;
                if (attn.LogitSoftcapping is { } cap)
                {
                    score = Math.Tanh(score / cap) * cap;
                }
                weights[i] = (float)score;
            }
            AttentionKernel.SoftmaxInPlace(weights);
            var outSlice = output.Data.AsSpan(r * output.Cols + head * hd, hd);
            for (int i = 0; i < count; i++)
            {
                int j = chosen?[i] ?? i;
                var v = values.AsSpan(j * valueWidth + kvHead * hd, hd);
                float w = weights[i];
                for (int d = 0; d < hd; d++)
                {
                    outSlice[d] += w * v[d];
                }
            }
        });
        return output;
    }
}
