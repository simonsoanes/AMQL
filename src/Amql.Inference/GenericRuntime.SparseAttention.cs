using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// Block-sparse attention on the managed path: an indexer pools each layer's
/// keys into fixed-size blocks, scores them per query, and each query attends
/// only to its best blocks plus the incomplete tail (QSA in
/// Qwen3.8-Flash-Next is one such operator).
/// </summary>
public sealed partial class GenericRuntime
{
    // Per sparse-attention layer: every position's raw indexer key, and the finished block keys.
    private readonly Dictionary<int, List<float>> _indexKeys = new();
    private readonly Dictionary<int, List<float[]>> _indexBlocks = new();

    private void ResetSparseAttentionState()
    {
        _indexKeys.Clear();
        _indexBlocks.Clear();
    }

    // ── block selection and the attention over it ──────────────────────────

    /// <summary>
    /// Runs the indexer over this pass's rows and returns, per query row, the
    /// key positions it keeps (ascending) — or null when no row exceeds its
    /// budget, so the dense attention is exactly what the sparse one computes. A row
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
                $"layer {layer}: the sparse-attention indexer holds {keys.Count / hd} keys for {kvSeq} cached positions");
        }
        int first = kvPositions.Length - kvSeq;
        for (int j = 0; j < kvSeq; j++)
        {
            if (first < 0 || kvPositions[first + j] != j)
            {
                throw new UnsupportedOperatorException($"layer {layer}: sparse-attention blocks are judged over contiguous positions from 0");
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
                $"layer {layer}: sparse-attention block selection over multimodal positions has no managed implementation");
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
