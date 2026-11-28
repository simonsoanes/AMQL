namespace Amql.Inference;

/// <summary>
/// Softmax attention on the row-exact path, with its conditioning: separate
/// q, k and v inputs, a per-channel query scale and an L2 clip of query and key
/// heads, position log-scaling, residual convolutions on the key and value
/// projections, a gainless value RMS, a learned relative score bias, a sink
/// logit, and an output gate with its own projection. Position-free: the
/// relative bias stands where a rotary embedding would.
/// </summary>
public sealed partial class GenericRuntime
{
    private float[] RowAttention(AttentionOp attn, VisitSlot slot, float[] qIn, float[] kIn, float[] vIn)
    {
        int d = qIn.Length, heads = attn.NumQHeads, kvHeads = attn.NumKvHeads, hd = attn.HeadDim;
        int t = SessionPosition;
        double tau = attn.LogScaling is { } log
            ? 1.0 + log.Alpha * Math.Log(Math.Max(1.0, (t + 1) / (double)log.Floor))
            : 1.0;

        // Queries: the learned per-channel scale, the clip per head, the log-scale.
        var q = Rows.Matrix(attn.QProj, attn.QDim, d).MatVec(qIn);
        float[]? qScale = attn.QueryChannelScale is { } qs ? Rows.Vector(qs, attn.QDim) : null;
        for (int h = 0; h < heads; h++)
        {
            var head = q.AsSpan(h * hd, hd);
            if (qScale is not null)
            {
                for (int i = 0; i < hd; i++)
                {
                    head[i] *= qScale[h * hd + i];
                }
            }
            if (attn.QkClip is { } clip)
            {
                RowMath.Clip(head, clip);
            }
            RowMath.Scale(head, tau);
        }

        // Keys and values: projected, through their residual convolutions, then
        // the keys clipped per head and the values RMS-normed without a gain.
        var k = Rows.Matrix(attn.KProj, attn.KvDim, d).MatVec(kIn);
        var v = Rows.Matrix(attn.VProj, attn.KvDim, d).MatVec(vIn);
        if (attn.KeyConv is { } keyConv)
        {
            k = ResidualConv(keyConv, slot.KeyHistory, k, kIn);
        }
        if (attn.ValueConv is { } valueConv)
        {
            v = ResidualConv(valueConv, slot.ValueHistory, v, vIn);
        }
        for (int h = 0; h < kvHeads; h++)
        {
            if (attn.QkClip is { } clip)
            {
                RowMath.Clip(k.AsSpan(h * hd, hd), clip);
            }
            if (attn.ParameterFreeQkNorm.V)
            {
                RowMath.Unweighted(v.AsSpan(h * hd, hd), attn.ParameterFreeQkNormEps).CopyTo(v, h * hd);
            }
        }
        slot.Keys.Add(k);
        slot.Values.Add(v);

        // The relative bias: per head, a clipped (and log-scaled) mix of the
        // table's profiles over backward distance.
        float[]? r = null;
        float[]? table = null;
        var bias = attn.RelativeBias;
        if (bias is not null)
        {
            r = Rows.Matrix(bias.Projection, heads * bias.Rank, d).MatVec(qIn);
            var scale = Rows.Vector(bias.ChannelScale, heads * bias.Rank);
            for (int h = 0; h < heads; h++)
            {
                var head = r.AsSpan(h * bias.Rank, bias.Rank);
                for (int i = 0; i < bias.Rank; i++)
                {
                    head[i] *= scale[h * bias.Rank + i];
                }
                RowMath.Clip(head, bias.Clip);
                RowMath.Scale(head, tau);
            }
            table = Rows.Vector(bias.Table, bias.Rank * bias.Extent);
        }
        float[]? sinks = attn.Sink switch
        {
            null => null,
            { Projected: true } s => Rows.Matrix(s.Weight, heads, d).MatVec(qIn),
            { } s => Rows.Vector(s.Weight, heads),
        };

        int first = attn.Window is { } window ? (int)Math.Max(0, t - window + 1) : 0;
        var output = new float[attn.QDim];
        double scoreScale = attn.ScoreScale;
        int group = heads / kvHeads;
        var scores = new double[t - first + 1];
        var acc = new double[hd];
        for (int h = 0; h < heads; h++)
        {
            int kvh = h / group;
            var qh = q.AsSpan(h * hd, hd);
            double max = sinks is not null ? sinks[h] : double.NegativeInfinity;
            for (int j = first; j <= t; j++)
            {
                int dist = t - j;
                double s = RowMath.Dot(qh, slot.Keys[j].AsSpan(kvh * hd, hd)) * scoreScale;
                if (bias is not null && dist < bias.Extent)
                {
                    double b = 0;
                    for (int i = 0; i < bias.Rank; i++)
                    {
                        b += r![h * bias.Rank + i] * (double)table![i * bias.Extent + dist];
                    }
                    s += b;
                }
                scores[j - first] = s;
                max = Math.Max(max, s);
            }
            double sum = sinks is not null ? Math.Exp(sinks[h] - max) : 0;
            for (int j = 0; j < scores.Length; j++)
            {
                scores[j] = Math.Exp(scores[j] - max);
                sum += scores[j];
            }
            Array.Clear(acc);
            for (int j = first; j <= t; j++)
            {
                double p = scores[j - first] / sum;
                var vj = slot.Values[j].AsSpan(kvh * hd, hd);
                for (int i = 0; i < hd; i++)
                {
                    acc[i] += p * vj[i];
                }
            }
            for (int i = 0; i < hd; i++)
            {
                output[h * hd + i] = (float)acc[i];
            }
        }

        // The output gate reads the query input through its own projection.
        if (attn.OutputGateProjection is { } gateProj)
        {
            var gate = Rows.Matrix(gateProj, attn.QDim, d).MatVec(qIn);
            for (int i = 0; i < output.Length; i++)
            {
                output[i] *= (float)(1.0 / (1.0 + Math.Exp(-gate[i])));
            }
        }
        return Rows.Matrix(attn.OProj, d, attn.QDim).MatVec(output);
    }
}
