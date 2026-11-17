using System.Diagnostics;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>One supervised example: the prompt the model reads and the
/// completion it should write. Loss is taken on the completion only.</summary>
public sealed record LoraExample(string Id, int[] Prompt, int[] Completion);

public sealed record LoraTrainOptions
{
    /// <summary>Decoder layers whose FFN down-projection gets a LoRA. Null
    /// means the last layer only, whose gradient is exact.</summary>
    public IReadOnlyList<int>? Layers { get; init; }
    public int Rank { get; init; } = 8;
    public double Alpha { get; init; } = 16;
    public double LearningRate { get; init; } = 2e-3;
    public int Steps { get; init; } = 60;
    public int EvalEvery { get; init; } = 10;
    public double ClipNorm { get; init; } = 1.0;
    public double WeightDecay { get; init; }
    public int Seed { get; init; } = 1234;

    /// <summary>Stop once the mean training loss is below this, every training
    /// example decodes exactly, and an evaluation has confirmed it.</summary>
    public double TargetLoss { get; init; } = 0.02;

    /// <summary>Undo a step that raised the loss by more than this factor and
    /// halve the learning rate: a cheap trust region for tiny datasets.</summary>
    public double SpikeFactor { get; init; } = 1.5;

    /// <summary>Directional finite-difference checks of the gradient, per
    /// layer, at the first step. 0 turns them off.</summary>
    public int GradChecks { get; init; } = 2;

    public IReadOnlySet<int>? StopTokens { get; init; }
    public int ExtraDecodeTokens { get; init; } = 8;

    /// <summary>Tokens of its own reply the base model writes for each anchor
    /// prompt. Anchors hold the model to what it already did there.</summary>
    public int AnchorTokens { get; init; } = 40;

    /// <summary>Weight of an anchor row against a training row in the loss.</summary>
    public double AnchorWeight { get; init; } = 1.0;
}

/// <summary>A rank-r update ΔW = scale · B · A on one down-projection.</summary>
public sealed class LoraLayer
{
    public required int Layer { get; init; }
    public required OperandRef Down { get; init; }
    public required int Hidden { get; init; }
    public required int Inter { get; init; }
    public required int Rank { get; init; }
    public required double Scale { get; init; }
    /// <summary>r × inter, row-major.</summary>
    public required float[] A { get; init; }
    /// <summary>hidden × r, row-major.</summary>
    public required float[] B { get; init; }

    public float[] Delta()
    {
        var delta = new float[Hidden * Inter];
        for (int h = 0; h < Hidden; h++)
        {
            var row = delta.AsSpan(h * Inter, Inter);
            for (int k = 0; k < Rank; k++)
            {
                float b = (float)(Scale * B[h * Rank + k]);
                if (b == 0f)
                    continue;
                var a = A.AsSpan(k * Inter, Inter);
                for (int i = 0; i < Inter; i++)
                    row[i] += b * a[i];
            }
        }
        return delta;
    }
}

public sealed record LoraEvalCase(string Id, bool Exact, double Loss, int FirstWrong, int[] Output);

public sealed record LoraTrainReport(
    IReadOnlyList<LoraLayer> Layers,
    int StepsRun,
    double InitialLoss,
    double FinalLoss,
    IReadOnlyList<LoraEvalCase> TrainBefore,
    IReadOnlyList<LoraEvalCase> TrainAfter,
    IReadOnlyList<LoraEvalCase> EvalBefore,
    IReadOnlyList<LoraEvalCase> EvalAfter,
    IReadOnlyList<string> Notes);

/// <summary>
/// LoRA training on FFN down-projections, without an autograd engine.
/// <para>
/// The forward pass is the runtime's own and exact. The backward pass runs
/// through the output head and the final norm exactly, giving the loss
/// gradient on the last residual. For the LAST layer's down-projection that is
/// the whole chain, so its gradient is exact. For an earlier layer it is the
/// direct residual path: the update's effect on the layers above is ignored.
/// The trainer measures how good that is (a directional finite-difference
/// check per layer, logged) rather than assuming it, and the loss it reports
/// and acts on is always the exact forward loss.
/// </para>
/// <para>
/// Layers below the first LoRA layer never change, so their output is computed
/// once per example and cached; each step runs only the layers from there up,
/// and the head only on the rows that predict the completion.
/// </para>
/// </summary>
public static class LoraTrainer
{
    public static LoraTrainReport Train(
        Vindex3Container container,
        string componentId,
        IReadOnlyList<LoraExample> train,
        IReadOnlyList<LoraExample> eval,
        LoraTrainOptions options,
        Action<object> log,
        IReadOnlyList<LoraExample>? anchors = null)
    {
        if (train.Count == 0)
            throw new MergeException("LoRA training needs at least one example");

        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, componentId, store);
        if (plan.Output is not { } output)
            throw new MergeException($"component '{componentId}' has no output head");
        if (output.LogitSoftcapping is not null)
            throw new MergeException("logit softcapping is not supported by the LoRA trainer's backward pass");
        if (plan.FinalNorm.Kind != NormType.RmsNorm)
            throw new MergeException("the LoRA trainer's backward pass needs an RMSNorm final norm");

        int n = plan.Layers.Count;
        var layerIds = (options.Layers ?? new[] { n - 1 }).Distinct().OrderBy(l => l).ToList();
        foreach (var l in layerIds)
        {
            if (l < 0 || l >= n)
                throw new MergeException($"layer {l} is outside [0, {n})");
            if (!PredicateBaker.IsEligible(plan.Layers[l]))
                throw new MergeException($"layer {l} has no dense FFN reaching the residual directly");
        }
        int hidden = plan.HiddenSize;
        var rng = new Random(options.Seed);
        var loras = layerIds.Select(l =>
        {
            var dense = plan.Layers[l].Ffn!.Dense!;
            int inter = dense.IntermediateSize;
            var a = new float[options.Rank * inter];
            // Kaiming-uniform A, zero B: the adapter starts as the identity.
            double bound = Math.Sqrt(3.0 / inter);
            for (int i = 0; i < a.Length; i++)
                a[i] = (float)((rng.NextDouble() * 2 - 1) * bound);
            return new LoraLayer
            {
                Layer = l, Down = dense.Down, Hidden = hidden, Inter = inter, Rank = options.Rank,
                Scale = options.Alpha / options.Rank, A = a, B = new float[hidden * options.Rank],
            };
        }).ToList();

        // The device engine has no seams to add a LoRA through, so the session runs the
        // managed path. That path's GEMMs go to the GPU whenever CUDA is on (no --cpu):
        // the base weights stay frozen and the LoRA runs beside them as its own f32
        // low-rank path, so their device copies, f16 or not, never go stale.
        var session = new DecodeSession(plan, store, workingSet: WeightWorkingSet.ResidentF32) { AllowGpuEngine = false };
        var rt = session.Runtime;
        var notes = new List<string>();
        int first = layerIds[0];
        var byLayer = loras.ToDictionary(l => l.Layer);
        rt.FfnAdapter = (layer, hf) =>
            byLayer.TryGetValue(layer, out var l) ? LoraPath(rt, plan.Layers[layer].Ffn!.Dense!, l, hf) : null;
        log(new { kind = "device", gpu = CudaShim.Enabled, note = CudaShim.Enabled
            ? "managed path, GEMMs on the GPU; the LoRA path in f32"
            : "managed path on the CPU" });

        // The LoRA path reads A and B as they are, so there is nothing to apply.
        static void Apply()
        {
        }

        try
        {
            var sw = Stopwatch.StartNew();
            // Anchors: the base model's own replies become targets, so the
            // adapter learns the new behaviour without unlearning the old.
            var weights = Enumerable.Repeat(1.0, train.Count).ToList();
            if (anchors is { Count: > 0 })
            {
                var own = anchors.Select(a => new LoraExample("anchor:" + a.Id, a.Prompt,
                    Greedy(session, a.Prompt, options.AnchorTokens, options.StopTokens))).ToList();
                foreach (var a in own)
                    log(new { kind = "anchor", id = a.Id, tokens = a.Completion.Length });
                train = train.Concat(own).ToList();
                weights.AddRange(own.Select(_ => options.AnchorWeight));
                log(new { kind = "anchors", count = own.Count, seconds = sw.Elapsed.TotalSeconds });
            }
            var trainCache = train.Select(e => Prefix(rt, e, first)).ToList();
            var evalCache = eval.Select(e => Prefix(rt, e, first)).ToList();
            log(new { kind = "prefix", layers = first, examples = train.Count + eval.Count, seconds = sw.Elapsed.TotalSeconds });

            var trainBefore = Evaluate(session, train, trainCache, first, options);
            var evalBefore = Evaluate(session, eval, evalCache, first, options);
            LogEval(log, 0, "train", trainBefore);
            LogEval(log, 0, "eval", evalBefore);

            var adamM = loras.Select(l => (A: new float[l.A.Length], B: new float[l.B.Length])).ToList();
            var adamV = loras.Select(l => (A: new float[l.A.Length], B: new float[l.B.Length])).ToList();
            double lr = options.LearningRate;
            double initialLoss = double.NaN, lastLoss = double.NaN;
            int step = 0, adamT = 0;
            List<(float[] A, float[] B)>? snapshot = null;
            List<LoraEvalCase> trainAfter = trainBefore, evalAfter = evalBefore;
            for (step = 1; step <= options.Steps; step++)
            {
                var t0 = Stopwatch.GetTimestamp();
                var grads = loras.Select(l => (A: new float[l.A.Length], B: new float[l.B.Length])).ToList();
                double loss = 0;
                double rowsTotal = 0;
                var perExample = new List<double>();
                for (int e = 0; e < train.Count; e++)
                {
                    var (exLoss, rows) = LossAndGrad(rt, plan, train[e], trainCache[e], first, loras, grads, weights[e]);
                    loss += exLoss * weights[e];
                    rowsTotal += rows * weights[e];
                    perExample.Add(exLoss / rows);
                }
                loss /= rowsTotal;
                foreach (var g in grads)
                {
                    Scale(g.A, 1.0 / rowsTotal);
                    Scale(g.B, 1.0 / rowsTotal);
                }
                if (step == 1)
                    initialLoss = loss;

                // A step that made things much worse is undone, and the rate halved.
                if (!double.IsNaN(lastLoss) && loss > lastLoss * options.SpikeFactor && snapshot is not null)
                {
                    lr /= 2;
                    for (int i = 0; i < loras.Count; i++)
                    {
                        Array.Copy(snapshot[i].A, loras[i].A, loras[i].A.Length);
                        Array.Copy(snapshot[i].B, loras[i].B, loras[i].B.Length);
                    }
                    Apply();
                    snapshot = null;
                    log(new { kind = "spike", step, loss, previous = lastLoss, lr, action = "undone, rate halved" });
                    continue;
                }
                snapshot = loras.Select(l => (A: (float[])l.A.Clone(), B: (float[])l.B.Clone())).ToList();

                if (step == 1 && options.GradChecks > 0)
                    GradCheck(rt, plan, train, trainCache, first, loras, grads, options.GradChecks, rng, Apply, log);

                double gradNorm = Math.Sqrt(grads.Sum(g => SumSq(g.A) + SumSq(g.B)));
                double clip = gradNorm > options.ClipNorm ? options.ClipNorm / gradNorm : 1.0;
                adamT++;
                double b1 = 0.9, b2 = 0.999, eps = 1e-8;
                double c1 = 1 - Math.Pow(b1, adamT), c2 = 1 - Math.Pow(b2, adamT);
                for (int i = 0; i < loras.Count; i++)
                {
                    AdamStep(loras[i].A, grads[i].A, adamM[i].A, adamV[i].A, lr, clip, b1, b2, eps, c1, c2, options.WeightDecay);
                    AdamStep(loras[i].B, grads[i].B, adamM[i].B, adamV[i].B, lr, clip, b1, b2, eps, c1, c2, options.WeightDecay);
                }
                Apply();
                lastLoss = loss;
                double seconds = Stopwatch.GetElapsedTime(t0).TotalSeconds;
                log(new
                {
                    kind = "step", step, loss, lr, gradNorm, clipped = clip < 1,
                    deltaNorm = loras.Select(l => Math.Sqrt(SumSq(l.Delta()))).ToArray(),
                    perExample, seconds,
                });

                bool evalNow = step % options.EvalEvery == 0 || step == options.Steps || loss < options.TargetLoss;
                if (evalNow)
                {
                    trainAfter = Evaluate(session, train, trainCache, first, options);
                    evalAfter = Evaluate(session, eval, evalCache, first, options);
                    LogEval(log, step, "train", trainAfter);
                    LogEval(log, step, "eval", evalAfter);
                    if (loss < options.TargetLoss && trainAfter.All(c => c.Exact))
                    {
                        log(new { kind = "stop", step, reason = "target loss reached and every training example decodes exactly" });
                        break;
                    }
                }
            }
            int stepsRun = Math.Min(step, options.Steps);
            double finalLoss = trainAfter.Average(c => c.Loss);
            return new LoraTrainReport(loras, stepsRun, initialLoss, finalLoss, trainBefore, trainAfter, evalBefore, evalAfter, notes);
        }
        finally
        {
            rt.FfnAdapter = null;
            session.Dispose();
        }
    }

    private static void LogEval(Action<object> log, int step, string set, List<LoraEvalCase> cases) =>
        log(new
        {
            kind = "eval", step, set,
            exact = cases.Count(c => c.Exact), of = cases.Count,
            loss = cases.Count == 0 ? 0 : cases.Average(c => c.Loss),
            cases = cases.Select(c => new { c.Id, c.Exact, c.Loss, c.FirstWrong }).ToArray(),
        });

    // ── forward pieces ──────────────────────────────────────────────────

    /// <summary>The residual entering <paramref name="first"/> for every
    /// position of prompt + completion[..^1].</summary>
    private static Tensor2D Prefix(GenericRuntime rt, LoraExample e, int first)
    {
        var tokens = Tokens(e);
        rt.ResetKvCache();
        var positions = Enumerable.Range(0, tokens.Length).ToArray();
        var x = rt.Embed(tokens);
        for (int l = 0; l < first; l++)
            x = rt.RunLayerInternal(x, l, positions, positions, appendKv: true);
        rt.ResetKvCache();
        return x;
    }

    private static int[] Tokens(LoraExample e) => e.Prompt.Concat(e.Completion.Take(e.Completion.Length - 1)).ToArray();

    private sealed record Upper(Tensor2D Final, Dictionary<int, Tensor2D> FfnInputs);

    /// <summary>Runs the layers from <paramref name="first"/> up and keeps, for
    /// the completion rows, the final residual and each LoRA layer's FFN input.</summary>
    private static Upper RunUpper(GenericRuntime rt, LoraExample e, Tensor2D prefix, int first, ISet<int> capture)
    {
        int rowStart = e.Prompt.Length - 1, rows = e.Completion.Length;
        var inputs = new Dictionary<int, Tensor2D>();
        rt.ResetKvCache();
        rt.FfnInputCapture = (layer, hf) =>
        {
            if (capture.Contains(layer))
                inputs[layer] = Slice(hf, rowStart, rows);
        };
        try
        {
            var x = prefix.Clone();
            int t = x.Rows;
            var positions = Enumerable.Range(0, t).ToArray();
            for (int l = first; l < rt.Plan.Layers.Count; l++)
                x = rt.RunLayerInternal(x, l, positions, positions, appendKv: true);
            return new Upper(Slice(x, rowStart, rows), inputs);
        }
        finally
        {
            rt.FfnInputCapture = null;
            rt.ResetKvCache();
        }
    }

    /// <summary>Summed cross-entropy over the completion, and (when
    /// <paramref name="grads"/> is given) the LoRA gradients added into it.</summary>
    private static (double Loss, int Rows) LossAndGrad(GenericRuntime rt, ComponentOpPlan plan, LoraExample e, Tensor2D prefix,
        int first, List<LoraLayer> loras, List<(float[] A, float[] B)>? grads, double weight = 1.0)
    {
        var upper = RunUpper(rt, e, prefix, first, loras.Select(l => l.Layer).ToHashSet());
        var x = upper.Final; // pre-norm residual, rows × hidden
        int rows = x.Rows, hidden = x.Cols;
        var normed = x.Clone();
        var logits = rt.FinalNormAndHead(normed);
        int vocab = logits.Cols;
        double mult = plan.Output!.Multiplier ?? 1.0;
        double loss = 0;
        var dNormed = new float[rows * hidden];
        var head = rt.Weights.Matrix(plan.Output.Projection, plan.Output.VocabSize, plan.Output.HiddenSize);
        for (int r = 0; r < rows; r++)
        {
            var row = logits.Row(r);
            float max = float.NegativeInfinity;
            foreach (var v in row)
                max = Math.Max(max, v);
            double sum = 0;
            foreach (var v in row)
                sum += Math.Exp(v - max);
            double logZ = max + Math.Log(sum);
            int target = e.Completion[r];
            loss += logZ - row[target];
            if (grads is null)
                continue;
            // dL/dlogit = p - onehot; probabilities below 1e-9 are dropped,
            // which keeps the head backward sparse at no measurable cost.
            var d = dNormed.AsSpan(r * hidden, hidden);
            for (int v = 0; v < vocab; v++)
            {
                double p = Math.Exp(row[v] - logZ) - (v == target ? 1 : 0);
                if (Math.Abs(p) < 1e-9)
                    continue;
                float g = (float)(p * mult * weight);
                var w = head.Data.AsSpan(v * hidden, hidden);
                for (int h = 0; h < hidden; h++)
                    d[h] += g * w[h];
            }
        }
        if (grads is null)
            return (loss, rows);

        // Back through the final RMSNorm: y = x·inv·w', inv = (mean x² + eps)^-½.
        var norm = plan.FinalNorm;
        var nw = rt.Weights.Vector(norm.Weight, hidden);
        var dx = new float[rows * hidden];
        for (int r = 0; r < rows; r++)
        {
            var xr = x.Row(r);
            double ss = 0;
            for (int h = 0; h < hidden; h++)
                ss += (double)xr[h] * xr[h];
            double inv = 1.0 / Math.Sqrt(ss / hidden + norm.Eps);
            double dot = 0;
            for (int h = 0; h < hidden; h++)
                dot += dNormed[r * hidden + h] * (nw[h] + norm.WeightOffset) * xr[h];
            for (int h = 0; h < hidden; h++)
            {
                double gy = dNormed[r * hidden + h] * (nw[h] + norm.WeightOffset);
                dx[r * hidden + h] = (float)(gy * inv - xr[h] * dot * inv * inv * inv / hidden);
            }
        }

        // Each LoRA layer reads the gradient on the residual it writes to: exact
        // for the last layer, the direct path for the rest.
        for (int i = 0; i < loras.Count; i++)
        {
            var l = loras[i];
            var dense = plan.Layers[l.Layer].Ffn!.Dense!;
            var act = Intermediate(rt, dense, upper.FfnInputs[l.Layer], hidden); // rows × inter
            var (gA, gB) = grads[i];
            int rank = l.Rank, inter = l.Inter;
            for (int r = 0; r < rows; r++)
            {
                var a = act.Row(r);
                var g = dx.AsSpan(r * hidden, hidden);
                Span<double> u = stackalloc double[rank];  // A·act
                Span<double> gu = stackalloc double[rank]; // Bᵀ·g
                for (int k = 0; k < rank; k++)
                {
                    var ak = l.A.AsSpan(k * inter, inter);
                    double s = 0;
                    for (int j = 0; j < inter; j++)
                        s += ak[j] * a[j];
                    u[k] = s;
                    double t = 0;
                    for (int h = 0; h < hidden; h++)
                        t += l.B[h * rank + k] * g[h];
                    gu[k] = t;
                }
                for (int h = 0; h < hidden; h++)
                    for (int k = 0; k < rank; k++)
                        gB[h * rank + k] += (float)(l.Scale * g[h] * u[k]);
                for (int k = 0; k < rank; k++)
                {
                    float c = (float)(l.Scale * gu[k]);
                    if (c == 0f)
                        continue;
                    var gak = gA.AsSpan(k * inter, inter);
                    for (int j = 0; j < inter; j++)
                        gak[j] += c * a[j];
                }
            }
        }
        return (loss, rows);
    }

    /// <summary>For each LoRA layer, compares the gradient's prediction along a
    /// random direction with the exact central difference of the loss.</summary>
    private static void GradCheck(GenericRuntime rt, ComponentOpPlan plan, IReadOnlyList<LoraExample> train,
        List<Tensor2D> cache, int first, List<LoraLayer> loras, List<(float[] A, float[] B)> grads, int checks,
        Random rng, Action apply, Action<object> log)
    {
        int rowsTotal = train.Sum(e => e.Completion.Length);
        double TotalLoss()
        {
            apply();
            double s = 0;
            for (int e = 0; e < train.Count; e++)
                s += LossAndGrad(rt, plan, train[e], cache[e], first, loras, null).Loss;
            return s / rowsTotal;
        }
        for (int i = 0; i < loras.Count; i++)
        {
            var l = loras[i];
            for (int c = 0; c < checks; c++)
            {
                // B starts at zero, so only B's direction moves the loss at step 1.
                var dirB = new float[l.B.Length];
                for (int j = 0; j < dirB.Length; j++)
                    dirB[j] = (float)(rng.NextDouble() * 2 - 1);
                double dn = Math.Sqrt(SumSq(dirB));
                for (int j = 0; j < dirB.Length; j++)
                    dirB[j] /= (float)dn;
                double predicted = 0;
                for (int j = 0; j < dirB.Length; j++)
                    predicted += grads[i].B[j] * dirB[j];
                // Align the direction with the gradient half the time, so the
                // check also sees the component that matters.
                if (c % 2 == 1)
                {
                    double gn = Math.Sqrt(SumSq(grads[i].B));
                    for (int j = 0; j < dirB.Length; j++)
                        dirB[j] = (float)(grads[i].B[j] / gn);
                    predicted = gn;
                }
                double h = 1e-2;
                var saved = (float[])l.B.Clone();
                for (int j = 0; j < l.B.Length; j++)
                    l.B[j] = saved[j] + (float)(h * dirB[j]);
                double plus = TotalLoss();
                for (int j = 0; j < l.B.Length; j++)
                    l.B[j] = saved[j] - (float)(h * dirB[j]);
                double minus = TotalLoss();
                Array.Copy(saved, l.B, saved.Length);
                apply();
                double measured = (plus - minus) / (2 * h);
                log(new
                {
                    kind = "gradcheck", layer = l.Layer, direction = c % 2 == 1 ? "gradient" : "random",
                    predicted, measured, ratio = measured == 0 ? double.NaN : predicted / measured,
                });
            }
        }
    }

    // ── evaluation ──────────────────────────────────────────────────────

    private static List<LoraEvalCase> Evaluate(DecodeSession session, IReadOnlyList<LoraExample> examples,
        List<Tensor2D> cache, int first, LoraTrainOptions options)
    {
        var rt = session.Runtime;
        var result = new List<LoraEvalCase>();
        for (int i = 0; i < examples.Count; i++)
        {
            var e = examples[i];
            var (loss, rows) = LossAndGrad(rt, rt.Plan, e, cache[i], first, new List<LoraLayer>(), null);
            var output = Greedy(session, e.Prompt, e.Completion.Length + options.ExtraDecodeTokens, options.StopTokens);
            int firstWrong = -1;
            for (int t = 0; t < e.Completion.Length; t++)
            {
                if (t >= output.Length || output[t] != e.Completion[t])
                {
                    firstWrong = t;
                    break;
                }
            }
            result.Add(new LoraEvalCase(e.Id, firstWrong < 0, loss / rows, firstWrong, output));
        }
        return result;
    }

    public static int[] Greedy(DecodeSession session, int[] prompt, int max, IReadOnlySet<int>? stop)
    {
        session.Reset();
        var output = new List<int>(max);
        var logits = session.Prefill(prompt);
        while (output.Count < max)
        {
            int next = ArgMax(logits.Row(0));
            output.Add(next);
            if (stop is not null && stop.Contains(next))
                break;
            if (output.Count < max)
                logits = session.Step(next);
        }
        session.Reset();
        return output.ToArray();
    }

    // ── small helpers ───────────────────────────────────────────────────

    /// <summary>A LoRA's own contribution to its layer's FFN output: scale · B · (A · act),
    /// with act the down-projection's input, rows × hidden.</summary>
    private static Tensor2D? LoraPath(GenericRuntime rt, DenseFfnOp dense, LoraLayer l, Tensor2D hf)
    {
        // B starts at zero, and then the path adds nothing.
        if (Array.TrueForAll(l.B, b => b == 0f))
            return null;
        int rows = hf.Rows, rank = l.Rank, inter = l.Inter, hidden = l.Hidden;
        var act = Intermediate(rt, dense, hf, hidden);
        var u = new double[rank];
        var output = new float[rows * hidden];
        for (int r = 0; r < rows; r++)
        {
            var a = act.Row(r);
            for (int k = 0; k < rank; k++)
            {
                var ak = l.A.AsSpan(k * inter, inter);
                double s = 0;
                for (int j = 0; j < inter; j++)
                    s += ak[j] * a[j];
                u[k] = s * l.Scale;
            }
            var o = output.AsSpan(r * hidden, hidden);
            for (int h = 0; h < hidden; h++)
            {
                double s = 0;
                for (int k = 0; k < rank; k++)
                    s += l.B[h * rank + k] * u[k];
                o[h] = (float)s;
            }
        }
        return new Tensor2D(output, rows, hidden);
    }

    public static Tensor2D Intermediate(GenericRuntime rt, DenseFfnOp dense, Tensor2D h, int hidden)
    {
        var act = Activations.For(dense.Activation, dense.IsGated);
        var up = TensorOps.MatMulTransposedB(h, rt.Weights.Matrix(dense.Up, dense.IntermediateSize, hidden));
        var gate = dense.Gate is null
            ? null
            : TensorOps.MatMulTransposedB(h, rt.Weights.Matrix(dense.Gate, dense.IntermediateSize, hidden));
        for (int i = 0; i < up.Data.Length; i++)
            up.Data[i] = gate is null ? act(up.Data[i]) : act(gate.Data[i]) * up.Data[i];
        return up;
    }

    private static void AdamStep(float[] p, float[] g, float[] m, float[] v, double lr, double clip,
        double b1, double b2, double eps, double c1, double c2, double wd)
    {
        for (int i = 0; i < p.Length; i++)
        {
            double gi = g[i] * clip;
            m[i] = (float)(b1 * m[i] + (1 - b1) * gi);
            v[i] = (float)(b2 * v[i] + (1 - b2) * gi * gi);
            double update = (m[i] / c1) / (Math.Sqrt(v[i] / c2) + eps) + wd * p[i];
            p[i] -= (float)(lr * update);
        }
    }

    private static double SumSq(float[] a)
    {
        double s = 0;
        foreach (var v in a)
            s += (double)v * v;
        return s;
    }

    private static void Scale(float[] a, double s)
    {
        for (int i = 0; i < a.Length; i++)
            a[i] = (float)(a[i] * s);
    }

    private static Tensor2D Slice(Tensor2D source, int firstRow, int rows)
    {
        var data = new float[rows * source.Cols];
        Array.Copy(source.Data, firstRow * source.Cols, data, 0, data.Length);
        return new Tensor2D(data, rows, source.Cols);
    }

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best])
                best = i;
        return best;
    }
}
