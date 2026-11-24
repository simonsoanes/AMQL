using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>How a model reads a text against a reference: its own perplexity,
/// and how far its next-token distributions are from the reference's.</summary>
public sealed record QuantComparison(string Model, int Positions, double Nll, double Perplexity,
    double KlMean, double KlP95, double KlMax, double Top1Agreement, double Seconds)
{
    public override string ToString() =>
        $"{Model}: NLL {Nll:F4} (perplexity {Perplexity:F3}), KL to reference mean {KlMean:F4} p95 {KlP95:F4} max {KlMax:F3}, " +
        $"top-1 agrees {100 * Top1Agreement:F1}%";
}

/// <summary>
/// Compares quantised containers with the one they came from: each reads the
/// same text, teacher-forced, on the GPU, and at every position its next-token
/// distribution is measured against the reference's (KL(reference ‖ model),
/// nats), along with its own perplexity and whether its top token is the
/// reference's. KL is the measure that matters for a quantisation: perplexity
/// can move little while the model's choices move a lot. The reference's
/// log-probabilities are held in memory (positions × vocabulary floats), so
/// only one model is on the device at a time.
/// </summary>
public static class QuantCompare
{
    public static (QuantComparison Reference, List<QuantComparison> Candidates) Run(
        string referenceDir, IReadOnlyList<string> candidateDirs, string text, int maxTokens, Action<string>? log = null, int scoreFrom = 0)
    {
        int kvAt = referenceDir.LastIndexOf("@kv=", StringComparison.Ordinal);
        var tokenizer = HfTokenizer.FromModelDir(kvAt > 0 ? referenceDir[..kvAt] : referenceDir);
        var ids = tokenizer.EncodeToIds(text).Take(maxTokens + 1).ToArray();
        if (ids.Length < 16)
        {
            throw new MergeException($"the text gives {ids.Length} tokens; compare needs at least 16");
        }
        int positions = ids.Length - 1;
        // Only positions from scoreFrom on are scored (the text before still runs as context).
        scoreFrom = Math.Clamp(scoreFrom, 0, positions - 1);
        int scored = positions - scoreFrom;
        var reference = new float[positions][];
        var refTop = new int[positions];
        var started = System.Diagnostics.Stopwatch.StartNew();
        double refNll = 0;
        Read(referenceDir, ids, (p, logits) =>
        {
            var lp = LogSoftmax(logits);
            reference[p] = lp;
            refTop[p] = ArgMax(lp);
            if (p >= scoreFrom)
            {
                refNll -= lp[ids[p + 1]];
            }
        });
        var refResult = new QuantComparison(Path.GetFileName(Path.TrimEndingDirectorySeparator(referenceDir)), scored,
            refNll / scored, Math.Exp(refNll / scored), 0, 0, 0, 1, started.Elapsed.TotalSeconds);
        log?.Invoke(refResult.ToString());

        var results = new List<QuantComparison>();
        foreach (var dir in candidateDirs)
        {
            started.Restart();
            double nll = 0;
            int agree = 0;
            var kl = new double[scored];
            Read(dir, ids, (p, logits) =>
            {
                if (p < scoreFrom)
                {
                    return;
                }
                var lp = LogSoftmax(logits);
                nll -= lp[ids[p + 1]];
                if (ArgMax(lp) == refTop[p])
                {
                    agree++;
                }
                var r = reference[p];
                double sum = 0;
                for (int v = 0; v < r.Length; v++)
                {
                    double pr = Math.Exp(r[v]);
                    if (pr > 0)
                    {
                        sum += pr * (r[v] - lp[v]);
                    }
                }
                kl[p - scoreFrom] = Math.Max(0, sum);
            });
            var sorted = kl.OrderBy(x => x).ToArray();
            var result = new QuantComparison(Path.GetFileName(Path.TrimEndingDirectorySeparator(dir)), scored,
                nll / scored, Math.Exp(nll / scored), kl.Average(), sorted[(int)(0.95 * (scored - 1))], sorted[^1],
                (double)agree / scored, started.Elapsed.TotalSeconds);
            log?.Invoke(result.ToString());
            results.Add(result);
        }
        return (refResult, results);
    }

    /// <summary>Each position's next-token logits for the text, in order,
    /// read 8 rows at a time through the engine's verify pass. A directory
    /// written <c>&lt;dir&gt;@kv=nvfp4</c> (or <c>@kv=f32</c>) reads with
    /// that KV cache format, so a container can be compared with itself.</summary>
    private static void Read(string spec, int[] ids, Action<int, float[]> each)
    {
        string saved = GpuEngine.RequestedKvFormat;
        // <dir>@exact=N keeps the first N positions at f32 in an NVFP4 cache; @exact16=N at FP16.
        int exact = 0;
        bool f16 = false;
        foreach (var (tag, half) in new[] { ("@exact16=", true), ("@exact=", false) })
        {
            int ex = spec.LastIndexOf(tag, StringComparison.Ordinal);
            if (ex > 0)
            {
                exact = int.TryParse(spec[(ex + tag.Length)..], out int n) && n >= 0 ? n
                    : throw new MergeException($"'{spec}': {tag} takes a token count");
                f16 = half;
                spec = spec[..ex];
                break;
            }
        }
        string dir = spec;
        int at0 = spec.LastIndexOf("@kv=", StringComparison.Ordinal);
        if (at0 > 0)
        {
            dir = spec[..at0];
            GpuEngine.RequestedKvFormat = spec[(at0 + 4)..].ToLowerInvariant() switch
            {
                "nvfp4" => "nvfp4",
                "fp16" => "fp16",
                "f32" => "f32",
                var other => throw new MergeException($"unknown KV format '{other}' in '{spec}': use f32, fp16 or nvfp4"),
            };
        }
        try
        {
            ReadWith(dir, ids, each, exact, f16);
        }
        finally
        {
            GpuEngine.RequestedKvFormat = saved;
        }
    }

    private static void ReadWith(string dir, int[] ids, Action<int, float[]> each, int exact, bool f16)
    {
        using var container = Vindex3Container.Open(dir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        using var session = new DecodeSession(plan, store);
        session.Prefill([ids[0]]);
        var engine = session.DeviceEngine()
            ?? throw new MergeException($"compare runs on the GPU, which is not available ({session.GpuRefusal})");
        if (engine.KvFormatRefusal is { } refused)
        {
            throw new MergeException($"{dir}: {refused}");
        }
        var seq = engine.NewSequence();
        engine.SetExactRows(seq, exact, f16);
        try
        {
            each(0, engine.Prefill(seq, ids.AsSpan(0, 1)));
            for (int at = 1; at + 1 < ids.Length; at += 8)
            {
                int rows = Math.Min(8, ids.Length - 1 - at);
                engine.Verify(seq, ids.AsSpan(at, rows));
                for (int r = 0; r < rows; r++)
                {
                    each(at + r, engine.HostLogits(r).ToArray());
                }
                engine.Accept(seq, rows);
            }
        }
        finally
        {
            engine.Release(seq);
        }
    }

    private static float[] LogSoftmax(float[] logits)
    {
        float max = float.NegativeInfinity;
        foreach (var v in logits)
        {
            max = MathF.Max(max, v);
        }
        double sum = 0;
        foreach (var v in logits)
        {
            sum += Math.Exp(v - max);
        }
        float log = (float)(max + Math.Log(sum));
        var lp = new float[logits.Length];
        for (int i = 0; i < lp.Length; i++)
        {
            lp[i] = logits[i] - log;
        }
        return lp;
    }

    private static int ArgMax(float[] x)
    {
        int best = 0;
        for (int i = 1; i < x.Length; i++)
        {
            if (x[i] > x[best])
            {
                best = i;
            }
        }
        return best;
    }
}
