using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>One probe: the same user turn rendered twice — once with the
/// predicate in context (the teacher sees it, e.g. as the system message)
/// and once without (what the baked model will be given).</summary>
public sealed record PredicateProbe(int[] TeacherPrompt, int[] StudentPrompt);

public sealed record PredicateBakeOptions
{
    /// <summary>Tokens the teacher generates per probe — its own reading
    /// of the predicate, and the positions the fit aligns.</summary>
    public int MaxNewTokens { get; init; } = 48;

    /// <summary>Decoder layers whose FFN down-projection is solved, in any
    /// order (they are fitted shallow to deep). Null picks evenly spaced
    /// eligible layers in the middle half of the stack.</summary>
    public IReadOnlyList<int>? Layers { get; init; }

    /// <summary>How many layers to pick when <see cref="Layers"/> is null.</summary>
    public int LayerCount { get; init; } = 4;

    /// <summary>Relative ridge on the sample Gram — larger keeps the update
    /// smaller and closer to the base model, smaller follows the teacher
    /// more tightly.</summary>
    public double Ridge { get; init; } = 1e-2;

    /// <summary>Tokens that end a teacher completion (kept as its last
    /// token so the baked model also learns where to stop).</summary>
    public IReadOnlySet<int>? StopTokens { get; init; }

    /// <summary>How many probes to re-generate with and without the patch
    /// for the report.</summary>
    public int Samples { get; init; } = 3;
}

/// <summary>Per edited layer: the tensor and how much of the
/// teacher/student hidden-state gap the closed-form update closed.</summary>
public sealed record PredicateLayerFit(int Layer, string ObjectId, string TensorName, double Explained, double GapNorm);

public sealed record PredicateSample(int ProbeIndex, int[] Teacher, int[] StudentBefore, int[] StudentAfter);

public sealed record PredicateBakeReport(
    string PatchPath,
    string ComponentId,
    int Probes,
    int AlignedRows,
    IReadOnlyList<PredicateLayerFit> Layers,
    double KlBefore,
    double KlAfter,
    double Top1Before,
    double Top1After,
    IReadOnlyList<PredicateSample> Samples,
    IReadOnlyList<string> Notes);

/// <summary>
/// Bakes a predicate into the weights: the model is shown the predicate
/// in context and interprets it itself (the teacher), then the FFN
/// down-projections of a few layers are solved so the same model WITHOUT
/// the predicate reproduces the teacher's hidden states — a closed-form
/// context distillation.
/// <para>
/// For each probe the teacher generates its completion y with the
/// predicate present. Both sequences end in the same tokens (the user turn
/// tail and y), so the rows that predict y line up one to one. Layers are
/// fitted shallow to deep: at layer L the student (already carrying the
/// shallower updates) runs, the gap between the teacher's residual and the
/// student's at the aligned rows is the target, the student's FFN
/// activation at those rows is the key, and the down-projection update is
/// the minimum-norm ridge solution ΔWᵀ = Kᵀ(KKᵀ + λI)⁻¹T mapping one to the other
/// (<see cref="LeastSquares.FitMinimumNormUpdate"/>). Each layer takes
/// its share of the remaining gap, so the change is spread over depth
/// rather than forced into one layer. No gradients, deterministic; the
/// result is an ordinary weight patch.
/// </para>
/// </summary>
public static class PredicateBaker
{
    public static PredicateBakeReport Bake(
        Vindex3Container container,
        string componentId,
        IReadOnlyList<PredicateProbe> probes,
        string outPatchPath,
        PredicateBakeOptions? options = null)
    {
        options ??= new PredicateBakeOptions();
        if (probes.Count == 0)
            throw new MergeException("baking a predicate needs at least one probe prompt");
        if (options.MaxNewTokens <= 0)
            throw new MergeException("max new tokens must be positive");
        if (options.Ridge <= 0)
            throw new MergeException("ridge must be positive");

        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, componentId, store);
        if (plan.Output is null)
            throw new MergeException($"component '{componentId}' carries no output head — a predicate is read through generation");
        if (plan.Diffusion is not null)
            throw new MergeException("predicate baking aligns causal continuations — a masked diffusion component is not supported");

        var layers = ResolveLayers(plan, options);
        int hidden = plan.HiddenSize;
        var notes = new List<string>();

        // ── teacher: the model reads the predicate and answers ──────────
        var teacher = new DecodeSession(plan, store);
        var completions = new List<int[]>(probes.Count);
        var teacherResiduals = new List<Dictionary<int, float[]>>(probes.Count);
        var teacherLogProbs = new List<float[][]>(probes.Count);
        foreach (var probe in probes)
        {
            var y = Greedy(teacher, probe.TeacherPrompt, options.MaxNewTokens, options.StopTokens);
            completions.Add(y);
            var pass = Forward(teacher.Runtime, probe.TeacherPrompt, y, layers, ffnLayer: -1);
            teacherResiduals.Add(pass.Residuals);
            teacherLogProbs.Add(pass.LogProbs);
        }
        int alignedRows = completions.Sum(y => y.Length);

        // ── before: how far the bare model is from the teacher ───────────
        var (klBefore, top1Before) = Divergence(teacher.Runtime, probes, completions, teacherLogProbs);

        // ── fit, shallow to deep ────────────────────────────────────────
        var entries = new List<WeightPatchEntry>();
        var fits = new List<PredicateLayerFit>();
        for (int li = 0; li < layers.Count; li++)
        {
            int layer = layers[li];
            var dense = plan.Layers[layer].Ffn!.Dense!;
            int inter = dense.IntermediateSize;
            var rt = new GenericRuntime(plan, store, entries.Count == 0 ? null : WeightPatch.FromEntries(entries));

            var keys = new float[alignedRows * inter];
            var targets = new float[alignedRows * hidden];
            float share = 1f / (layers.Count - li);
            double gap = 0;
            int row = 0;
            for (int p = 0; p < probes.Count; p++)
            {
                var pass = Forward(rt, probes[p].StudentPrompt, completions[p], new[] { layer }, ffnLayer: layer);
                var act = Intermediate(rt, dense, pass.FfnInput!, hidden);
                Array.Copy(act.Data, 0, keys, row * inter, act.Data.Length);
                var student = pass.Residuals[layer];
                var target = teacherResiduals[p][layer];
                for (int i = 0; i < student.Length; i++)
                {
                    float d = target[i] - student[i];
                    gap += (double)d * d;
                    targets[row * hidden + i] = share * d;
                }
                row += act.Rows;
            }

            var (delta, explained) = LeastSquares.FitMinimumNormUpdate(
                targets, keys, alignedRows, hidden, inter, options.Ridge);
            var shape = store.Resolve(dense.Down).Shape;
            if (WeightPatch.ElementCount(shape) != delta.Length)
                throw new MergeException(
                    $"layer {layer} down-projection is [{string.Join("x", shape)}], expected {hidden}x{inter}");
            entries.Add(new WeightPatchEntry(dense.Down.ObjectId, dense.Down.TensorName, shape, delta));
            fits.Add(new PredicateLayerFit(layer, dense.Down.ObjectId, dense.Down.TensorName, explained,
                Math.Sqrt(gap / alignedRows)));
        }

        var patch = WeightPatch.FromEntries(entries, container.Index.Model);
        var (klAfter, top1After) = Divergence(new GenericRuntime(plan, store, patch), probes, completions, teacherLogProbs);
        WeightPatch.Save(outPatchPath, entries, container.Index.Model);

        // ── samples: the bare prompt, before and after the patch ────────
        var samples = new List<PredicateSample>();
        var before = new DecodeSession(plan, store);
        var after = new DecodeSession(plan, store, patch);
        for (int p = 0; p < Math.Min(options.Samples, probes.Count); p++)
        {
            samples.Add(new PredicateSample(p, completions[p],
                Greedy(before, probes[p].StudentPrompt, options.MaxNewTokens, options.StopTokens),
                Greedy(after, probes[p].StudentPrompt, options.MaxNewTokens, options.StopTokens)));
        }

        if (klAfter >= klBefore)
            notes.Add("the patch did not bring the bare model closer to the teacher — try more probes, other layers or a larger ridge");
        if (alignedRows < 16)
            notes.Add($"only {alignedRows} aligned positions — the update is fitted to very little evidence");

        return new PredicateBakeReport(outPatchPath, componentId, probes.Count, alignedRows, fits,
            klBefore, klAfter, top1Before, top1After, samples, notes);
    }

    /// <summary>The layers whose FFN can carry a linear residual update: a
    /// dense FFN whose output reaches the residual without a post-norm
    /// (a post-FFN norm would rescale the update nonlinearly).</summary>
    public static bool IsEligible(LayerPlan layer) =>
        layer.Ffn?.Dense is not null && layer.PostFfnNorm is null;

    private static List<int> ResolveLayers(ComponentOpPlan plan, PredicateBakeOptions options)
    {
        int n = plan.Layers.Count;
        if (options.Layers is { } requested)
        {
            var list = requested.Distinct().OrderBy(l => l).ToList();
            if (list.Count == 0)
                throw new MergeException("no layers given to bake the predicate into");
            foreach (var l in list)
            {
                if (l < 0 || l >= n)
                    throw new MergeException($"layer {l} is outside [0, {n})");
                if (!IsEligible(plan.Layers[l]))
                    throw new MergeException(
                        $"layer {l} has no dense FFN reaching the residual directly (routed or post-normed) — pick another layer");
            }
            return list;
        }

        var eligible = Enumerable.Range(0, n).Where(l => IsEligible(plan.Layers[l])).ToList();
        if (eligible.Count == 0)
            throw new MergeException("no layer has a dense FFN without a post-norm — nothing can carry the update");
        var band = eligible.Where(l => l >= n / 4 && l < Math.Max(n / 4 + 1, 3 * n / 4)).ToList();
        if (band.Count == 0)
            band = eligible;
        int k = Math.Clamp(options.LayerCount, 1, band.Count);
        return Enumerable.Range(0, k)
            .Select(i => band[(int)Math.Round(i * (band.Count - 1) / (double)Math.Max(1, k - 1))])
            .Distinct().ToList();
    }

    private sealed record Pass(Dictionary<int, float[]> Residuals, Tensor2D? FfnInput, float[][] LogProbs);

    /// <summary>Runs prompt + y through the stack and keeps, at the rows
    /// that predict y, the residual after each requested layer, the
    /// pre-FFN input at <paramref name="ffnLayer"/>, and the output
    /// log-probabilities.</summary>
    private static Pass Forward(GenericRuntime rt, int[] prompt, int[] y, IReadOnlyList<int> layers, int ffnLayer)
    {
        // The row of the last prompt token predicts y[0]; the row of y[i]
        // predicts y[i+1]. The final y token predicts nothing we know.
        var tokens = prompt.Concat(y.Take(y.Length - 1)).ToArray();
        int first = prompt.Length - 1;
        int rows = y.Length;

        rt.ResetKvCache();
        Tensor2D? ffnInput = null;
        rt.FfnInputCapture = (layer, hf) =>
        {
            if (layer == ffnLayer)
                ffnInput = Slice(hf, first, rows);
        };
        var residuals = new Dictionary<int, float[]>();
        try
        {
            var hidden = rt.Embed(tokens);
            var positions = Enumerable.Range(0, tokens.Length).ToArray();
            for (int layer = 0; layer < rt.Plan.Layers.Count; layer++)
            {
                hidden = rt.RunLayerInternal(hidden, layer, positions, positions, appendKv: true);
                if (layers.Contains(layer))
                    residuals[layer] = Slice(hidden, first, rows).Data;
            }
            var logits = rt.FinalNormAndHead(Slice(hidden, first, rows));
            var logProbs = new float[rows][];
            for (int r = 0; r < rows; r++)
                logProbs[r] = LogSoftmax(logits.Row(r));
            return new Pass(residuals, ffnInput, logProbs);
        }
        finally
        {
            rt.FfnInputCapture = null;
            rt.ResetKvCache();
        }
    }

    /// <summary>The down-projection's input: act(gate·h) ⊙ (up·h), or
    /// act(up·h) ungated — <see cref="FfnKernel.Dense"/> without the last
    /// GEMM.</summary>
    private static Tensor2D Intermediate(GenericRuntime rt, DenseFfnOp dense, Tensor2D h, int hidden)
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

    /// <summary>Mean KL(teacher ‖ student) per aligned row, and how often
    /// the student's top token is the teacher's.</summary>
    private static (double Kl, double Top1) Divergence(
        GenericRuntime rt, IReadOnlyList<PredicateProbe> probes, List<int[]> completions, List<float[][]> teacher)
    {
        double kl = 0;
        int agree = 0, rows = 0;
        for (int p = 0; p < probes.Count; p++)
        {
            var student = Forward(rt, probes[p].StudentPrompt, completions[p], Array.Empty<int>(), ffnLayer: -1).LogProbs;
            for (int r = 0; r < student.Length; r++)
            {
                var t = teacher[p][r];
                var s = student[r];
                for (int v = 0; v < t.Length; v++)
                    kl += Math.Exp(t[v]) * (t[v] - s[v]);
                if (ArgMax(t) == ArgMax(s))
                    agree++;
                rows++;
            }
        }
        return (kl / rows, agree / (double)rows);
    }

    /// <summary>Greedy continuation of <paramref name="prompt"/>; a stop
    /// token ends it and is kept.</summary>
    private static int[] Greedy(DecodeSession session, int[] prompt, int max, IReadOnlySet<int>? stop)
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

    private static Tensor2D Slice(Tensor2D source, int firstRow, int rows)
    {
        var data = new float[rows * source.Cols];
        Array.Copy(source.Data, firstRow * source.Cols, data, 0, data.Length);
        return new Tensor2D(data, rows, source.Cols);
    }

    private static float[] LogSoftmax(ReadOnlySpan<float> logits)
    {
        float max = float.NegativeInfinity;
        foreach (var v in logits)
            max = Math.Max(max, v);
        double sum = 0;
        foreach (var v in logits)
            sum += Math.Exp(v - max);
        float log = (float)(max + Math.Log(sum));
        var result = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
            result[i] = logits[i] - log;
        return result;
    }

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
                best = i;
        }
        return best;
    }
}
