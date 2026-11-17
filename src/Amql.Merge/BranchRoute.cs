using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>One branch of a branch-adapt-route model: a LoRA trained on one topic,
/// by <c>lora-train</c>, over some of the base's FFN down-projections.</summary>
public sealed class LoraExpert
{
    public required string Name { get; init; }
    public required IReadOnlyDictionary<int, LoraLayer> Layers { get; init; }

    /// <summary>Reads an amql-lora-v1 adapter directory (adapter_config.json +
    /// adapter_model.safetensors) as written by <c>lora-train</c>.</summary>
    public static LoraExpert Load(string dir, string name, ComponentOpPlan plan)
    {
        var configPath = Path.Combine(dir, "adapter_config.json");
        var tensorsPath = Path.Combine(dir, "adapter_model.safetensors");
        if (!File.Exists(configPath) || !File.Exists(tensorsPath))
            throw new MergeException($"expert '{name}': '{dir}' holds no adapter_config.json + adapter_model.safetensors");
        var config = JsonNode.Parse(File.ReadAllText(configPath))!.AsObject();
        using var file = SafetensorsFile.Open(tensorsPath);
        var layers = new Dictionary<int, LoraLayer>();
        foreach (var target in config["targets"]!.AsArray())
        {
            int layer = (int?)target!["layer"] ?? throw new MergeException($"expert '{name}': a target records no layer");
            if (layer < 0 || layer >= plan.Layers.Count || plan.Layers[layer].Ffn?.Dense is not { } dense)
                throw new MergeException($"expert '{name}': layer {layer} has no dense FFN in this model");
            if ((string?)target["tensor_name"] != dense.Down.TensorName)
                throw new MergeException($"expert '{name}': layer {layer}'s adapter targets '{target["tensor_name"]}', " +
                                         $"not the down-projection '{dense.Down.TensorName}'");
            int rank = (int)target["rank"]!;
            var shape = target["shape"]!.AsArray();
            int hidden = (int)shape[0]!, inter = (int)shape[1]!;
            if (hidden != plan.HiddenSize || inter != dense.IntermediateSize)
                throw new MergeException($"expert '{name}': layer {layer}'s adapter is {hidden}×{inter}, the model's " +
                                         $"down-projection {plan.HiddenSize}×{dense.IntermediateSize}");
            layers[layer] = new LoraLayer
            {
                Layer = layer, Down = dense.Down, Hidden = hidden, Inter = inter, Rank = rank,
                Scale = (double)target["scale"]!,
                A = file.DecodeF32((string)target["lora_a"]!),
                B = file.DecodeF32((string)target["lora_b"]!),
            };
        }
        if (layers.Count == 0)
            throw new MergeException($"expert '{name}': the adapter has no targets");
        return new LoraExpert { Name = name, Layers = layers };
    }
}

/// <summary>A per-layer linear router over the experts, BAR's "small linear layer at
/// each transformer block": logits = W · x for the FFN's (normed) input x. Expert 0
/// is the anchor, the base FFN unchanged; experts 1..N are the branches.</summary>
public sealed class MoeRouter
{
    public required IReadOnlyList<string> Experts { get; init; }
    public required IReadOnlyList<int> Layers { get; init; }
    public required int Hidden { get; init; }
    /// <summary>Per layer, experts × hidden, row-major.</summary>
    public required Dictionary<int, float[]> Weights { get; init; }
    /// <summary>Route every row by the mean of the prompt rows' logits, held for the
    /// whole reply, instead of by each row's own.</summary>
    public bool HoldPrompt { get; init; }
    public string Method { get; init; } = "";

    public static MoeRouter Zero(IReadOnlyList<string> experts, IReadOnlyList<int> layers, int hidden, bool holdPrompt) => new()
    {
        Experts = experts, Layers = layers, Hidden = hidden, HoldPrompt = holdPrompt,
        Weights = layers.ToDictionary(l => l, _ => new float[experts.Count * hidden]),
    };

    /// <summary>This router over a new expert list: rows are carried over by expert
    /// name, and an expert it has not seen starts at zero. So adding a topic keeps
    /// what the router learnt about the others, and training starts from there.</summary>
    public MoeRouter Extend(IReadOnlyList<string> experts, IReadOnlyList<int> layers, bool holdPrompt)
    {
        var weights = new Dictionary<int, float[]>();
        foreach (var l in layers)
        {
            var w = new float[experts.Count * Hidden];
            if (Weights.TryGetValue(l, out var old))
            {
                for (int e = 0; e < experts.Count; e++)
                {
                    int from = Experts.ToList().IndexOf(experts[e]);
                    if (from >= 0)
                        Array.Copy(old, from * Hidden, w, e * Hidden, Hidden);
                }
            }
            weights[l] = w;
        }
        return new MoeRouter { Experts = experts, Layers = layers, Hidden = Hidden, Weights = weights, HoldPrompt = holdPrompt, Method = Method };
    }

    public void Save(string path)
    {
        var tensors = Layers.Select(l => new TensorPayload
        {
            Name = $"{l}.router.weight", Dtype = Dtype.F32,
            Shape = new long[] { Experts.Count, Hidden }, Data = MoeIfy.ToF32Bytes(Weights[l]),
        }).ToList();
        SafetensorsWriter.Write(path, tensors, new Dictionary<string, string>
        {
            ["format"] = "amql-moe-router-v1",
            ["experts"] = string.Join(",", Experts),
            ["layers"] = string.Join(",", Layers),
            ["hold"] = HoldPrompt ? "prompt" : "token",
            ["method"] = Method,
        });
    }

    public static MoeRouter Load(string path)
    {
        using var file = SafetensorsFile.Open(path);
        var meta = file.Metadata ?? throw new MergeException($"'{path}' records no router metadata");
        if (!meta.TryGetValue("format", out var format) || format != "amql-moe-router-v1")
            throw new MergeException($"'{path}' is not an amql-moe-router-v1 file");
        var experts = meta["experts"].Split(',');
        var layers = meta["layers"].Split(',').Select(int.Parse).ToArray();
        var weights = layers.ToDictionary(l => l, l => file.DecodeF32($"{l}.router.weight"));
        return new MoeRouter
        {
            Experts = experts, Layers = layers, Hidden = weights[layers[0]].Length / experts.Length, Weights = weights,
            HoldPrompt = meta.GetValueOrDefault("hold") == "prompt", Method = meta.GetValueOrDefault("method") ?? "",
        };
    }
}

/// <summary>How one forward pass mixes the experts: fixed weights (the base, one
/// expert, an average, a sum), or a router.</summary>
public sealed record Mixing(double[]? Fixed, MoeRouter? Router, int TopK)
{
    public static Mixing Of(params double[] weights) => new(weights, null, 0);
    public static Mixing Routed(MoeRouter router, int topK) => new(null, router, topK);
}

public sealed record RouterTrainOptions
{
    public int Steps { get; init; } = 30;
    public double LearningRate { get; init; } = 1e-2;
    public double WeightDecay { get; init; }
    public double ClipNorm { get; init; } = 1.0;
    /// <summary>For the label method: keep every n-th row of each example.</summary>
    public int RowStride { get; init; } = 2;
    public bool HoldPrompt { get; init; }
    /// <summary>A router to start from, extended to the bench's experts by name.</summary>
    public MoeRouter? Init { get; init; }
}

/// <summary>
/// Branch-Adapt-Route (Morrison et al., 2026) on the AMQL runtime. Topic experts are
/// LoRAs on the base's FFN down-projections, the anchor is the base FFN, and a
/// per-layer linear router mixes them; all else is shared and frozen.
/// <para>
/// Because every expert shares the base's gate and up projections, expert e's FFN
/// output is the base's plus ΔW_e · act, and a mixture with weights p summing to 1 is
/// the base's plus Σ p_e ΔW_e · act. So the bench runs any mixture on the dense base
/// through the runtime's FFN adapter seam, and <see cref="WriteMoe"/> materialises the
/// same function as a routed container, each expert a full FFN.
/// </para>
/// <para>
/// BAR trains its router on the language-model loss with everything else frozen.
/// Here the gradient on a router's logits is exact at the last layer and the direct
/// residual path below it, the same approximation <see cref="LoraTrainer"/> makes.
/// A second method fits each router to the topic labels instead: a per-layer
/// logistic regression on the FFN inputs, no backward pass at all.
/// </para>
/// </summary>
public sealed class MoeBench : IDisposable
{
    private readonly OperandStore _store;
    private readonly DecodeSession _session;
    private readonly Dictionary<int, LoraLayer?[]> _byLayer = new();
    private Mixing _mixing = Mixing.Of();
    private int _promptRows;
    private Dictionary<int, Capture>? _capture;

    public ComponentOpPlan Plan { get; }
    public GenericRuntime Runtime { get; }
    public IReadOnlyList<LoraExpert> Experts { get; }
    /// <summary>"anchor", then each expert's name.</summary>
    public IReadOnlyList<string> Names { get; }
    public IReadOnlyList<int> Layers { get; }
    /// <summary>The first layer a mixture touches; with no experts, every layer runs per pass.</summary>
    public int First => Layers.Count == 0 ? 0 : Layers[0];

    /// <summary>Per routed layer, what a forward pass saw: the FFN input, each
    /// expert's delta output, and the mixing weights (rows × experts).</summary>
    private sealed record Capture(Tensor2D Input, Tensor2D?[] Deltas, float[] P, float[] Logits);

    public MoeBench(Vindex3Container container, string componentId, Func<ComponentOpPlan, IReadOnlyList<LoraExpert>> experts)
    {
        _store = container.CreateOperandStore();
        Plan = Planner.Plan(container, componentId, _store);
        if (Plan.Output is null)
            throw new MergeException($"component '{componentId}' has no output head");
        _session = new DecodeSession(Plan, _store, workingSet: WeightWorkingSet.ResidentF32) { AllowGpuEngine = false };
        Runtime = _session.Runtime;
        Experts = experts(Plan);
        Names = new[] { "anchor" }.Concat(Experts.Select(e => e.Name)).ToArray();
        if (Names.Distinct().Count() != Names.Count)
            throw new MergeException("expert names must be distinct, and none may be 'anchor'");
        Layers = Experts.SelectMany(e => e.Layers.Keys).Distinct().OrderBy(l => l).ToArray();
        foreach (var l in Layers)
            _byLayer[l] = Names.Select((_, i) => i == 0 ? null : Experts[i - 1].Layers.GetValueOrDefault(l)).ToArray();
        Runtime.FfnAdapter = Adapter;
    }

    public void Dispose()
    {
        Runtime.FfnAdapter = null;
        _session.Dispose();
        _store.Dispose();
    }

    public Tensor2D Prefix(LoraExample e) => LoraTrainer.Prefix(Runtime, e, First);

    /// <summary>Mean completion loss (nats per token) of one example under a mixture.</summary>
    public double Loss(LoraExample e, Tensor2D prefix, Mixing mixing)
    {
        var (loss, _) = Forward(e, prefix, mixing, wantGrad: false);
        return loss / e.Completion.Length;
    }

    private (double Loss, float[]? Dx) Forward(LoraExample e, Tensor2D prefix, Mixing mixing, bool wantGrad)
    {
        _mixing = mixing;
        _promptRows = e.Prompt.Length;
        try
        {
            var upper = LoraTrainer.RunUpper(Runtime, e, prefix, First, new HashSet<int>());
            return LoraTrainer.HeadLossGrad(Runtime, Plan, upper.Final, e.Completion, 1.0, wantGrad);
        }
        finally
        {
            _mixing = Mixing.Of();
        }
    }

    private Tensor2D? Adapter(int layer, Tensor2D hf)
    {
        if (!_byLayer.TryGetValue(layer, out var loras))
            return null;
        int rows = hf.Rows, hidden = hf.Cols, n = Names.Count;
        var p = new float[rows * n];
        var logits = new float[rows * n];
        if (_mixing.Fixed is { } w)
        {
            for (int r = 0; r < rows; r++)
                for (int e = 0; e < n; e++)
                    p[r * n + e] = e < w.Length ? (float)w[e] : 0f;
        }
        else
        {
            RouterWeights(_mixing.Router!, layer, hf, _mixing.TopK, p, logits);
        }
        bool any = false;
        for (int e = 1; e < n && !any; e++)
            for (int r = 0; r < rows && !any; r++)
                any = loras[e] is not null && p[r * n + e] != 0f;
        if (!any && _capture is null)
            return null;

        var dense = Plan.Layers[layer].Ffn!.Dense!;
        var act = LoraTrainer.Intermediate(Runtime, dense, hf, hidden);
        var deltas = new Tensor2D?[n];
        var output = new Tensor2D(new float[rows * hidden], rows, hidden);
        for (int e = 1; e < n; e++)
        {
            if (loras[e] is not { } l)
                continue;
            var delta = Delta(l, act);
            deltas[e] = delta;
            for (int r = 0; r < rows; r++)
            {
                float pe = p[r * n + e];
                if (pe == 0f)
                    continue;
                var o = output.Row(r);
                var d = delta.Row(r);
                for (int h = 0; h < hidden; h++)
                    o[h] += pe * d[h];
            }
        }
        if (_capture is not null)
            _capture[layer] = new Capture(hf.Clone(), deltas, p, logits);
        return output;
    }

    /// <summary>ΔW · act = scale · B · (A · act), rows × hidden.</summary>
    private static Tensor2D Delta(LoraLayer l, Tensor2D act)
    {
        var u = TensorOps.MatMulTransposedB(act, new Tensor2D(l.A, l.Rank, l.Inter)); // rows × rank
        var d = TensorOps.MatMulTransposedB(u, new Tensor2D(l.B, l.Hidden, l.Rank));  // rows × hidden
        if (l.Scale != 1.0)
        {
            float s = (float)l.Scale;
            var data = d.Data;
            for (int i = 0; i < data.Length; i++)
                data[i] *= s;
        }
        return d;
    }

    /// <summary>Softmax over the router's logits, kept to the top k and renormalised
    /// over them (k = 0 or k ≥ experts keeps all). With the prompt held, every row
    /// takes the mean of the prompt rows' logits.</summary>
    private void RouterWeights(MoeRouter router, int layer, Tensor2D hf, int topK, float[] p, float[] logits)
    {
        int rows = hf.Rows, hidden = hf.Cols, n = Names.Count;
        if (!router.Weights.TryGetValue(layer, out var w))
        {
            for (int r = 0; r < rows; r++)
                p[r * n] = 1f; // a layer the router does not cover keeps the anchor
            return;
        }
        for (int r = 0; r < rows; r++)
        {
            var x = hf.Row(r);
            for (int e = 0; e < n; e++)
                logits[r * n + e] = TensorOps.Dot(x, w.AsSpan(e * hidden, hidden));
        }
        if (router.HoldPrompt)
        {
            int held = Math.Min(_promptRows, rows);
            var mean = new float[n];
            for (int r = 0; r < held; r++)
                for (int e = 0; e < n; e++)
                    mean[e] += logits[r * n + e] / held;
            for (int r = 0; r < rows; r++)
                mean.CopyTo(logits.AsSpan(r * n, n));
        }
        for (int r = 0; r < rows; r++)
            Softmax(logits.AsSpan(r * n, n), p.AsSpan(r * n, n), topK);
    }

    private static void Softmax(ReadOnlySpan<float> logits, Span<float> p, int topK)
    {
        int n = logits.Length;
        float max = float.NegativeInfinity;
        for (int e = 0; e < n; e++)
            max = Math.Max(max, logits[e]);
        Span<bool> keep = stackalloc bool[n];
        if (topK <= 0 || topK >= n)
        {
            keep.Fill(true);
        }
        else
        {
            for (int k = 0; k < topK; k++)
            {
                int best = -1;
                for (int e = 0; e < n; e++)
                    if (!keep[e] && (best < 0 || logits[e] > logits[best]))
                        best = e;
                keep[best] = true;
            }
        }
        double sum = 0;
        for (int e = 0; e < n; e++)
        {
            p[e] = keep[e] ? MathF.Exp(logits[e] - max) : 0f;
            sum += p[e];
        }
        for (int e = 0; e < n; e++)
            p[e] = (float)(p[e] / sum);
    }

    // ── router training ─────────────────────────────────────────────────

    /// <summary>BAR's router training: the language-model loss on the routing set,
    /// with every expert and every shared weight frozen. Soft routing over all
    /// experts, Adam, full batch. The router starts at zero: a uniform mixture.</summary>
    public MoeRouter TrainByLoss(IReadOnlyList<LoraExample> data, IReadOnlyList<Tensor2D> prefixes,
        RouterTrainOptions options, Action<object> log)
    {
        int n = Names.Count, hidden = Plan.HiddenSize;
        var router = Start(options);
        var m = Layers.ToDictionary(l => l, _ => new float[n * hidden]);
        var v = Layers.ToDictionary(l => l, _ => new float[n * hidden]);
        int t = 0;
        for (int step = 1; step <= options.Steps; step++)
        {
            var t0 = Stopwatch.GetTimestamp();
            var (loss, grads, usage) = RouterLossGradient(data, prefixes, router);
            double norm = Math.Sqrt(grads.Values.Sum(LoraTrainer.SumSq));
            double clip = norm > options.ClipNorm ? options.ClipNorm / norm : 1.0;
            t++;
            double c1 = 1 - Math.Pow(0.9, t), c2 = 1 - Math.Pow(0.999, t);
            foreach (var l in Layers)
                LoraTrainer.AdamStep(router.Weights[l], grads[l], m[l], v[l], options.LearningRate, clip,
                    0.9, 0.999, 1e-8, c1, c2, options.WeightDecay);
            log(new
            {
                kind = "router-step", step, loss, gradNorm = norm,
                usage = Layers.ToDictionary(l => l.ToString(), l => usage[l]),
                seconds = Stopwatch.GetElapsedTime(t0).TotalSeconds,
            });
        }
        return new MoeRouter
        {
            Experts = router.Experts, Layers = router.Layers, Hidden = hidden, Weights = router.Weights,
            HoldPrompt = options.HoldPrompt, Method = "loss",
        };
    }

    private MoeRouter Start(RouterTrainOptions options)
    {
        if (options.Init is not { } init)
            return MoeRouter.Zero(Names, Layers, Plan.HiddenSize, options.HoldPrompt);
        if (init.Hidden != Plan.HiddenSize)
            throw new MergeException($"the initial router is for hidden size {init.Hidden}, the model's is {Plan.HiddenSize}");
        return init.Extend(Names, Layers, options.HoldPrompt);
    }

    /// <summary>The mean completion loss over <paramref name="data"/> under soft routing
    /// by <paramref name="router"/>, its gradient on each layer's router weights, and
    /// each layer's mean expert weights over the completion rows.</summary>
    public (double Loss, Dictionary<int, float[]> Grads, Dictionary<int, double[]> Usage) RouterLossGradient(
        IReadOnlyList<LoraExample> data, IReadOnlyList<Tensor2D> prefixes, MoeRouter router)
    {
        int n = Names.Count, hidden = Plan.HiddenSize;
        var grads = Layers.ToDictionary(l => l, _ => new float[n * hidden]);
        var usage = Layers.ToDictionary(l => l, _ => new double[n]);
        double loss = 0;
        int tokens = 0;
        for (int i = 0; i < data.Count; i++)
        {
            var e = data[i];
            var captures = _capture = new Dictionary<int, Capture>();
            try
            {
                var (exLoss, dx) = Forward(e, prefixes[i], Mixing.Routed(router, 0), wantGrad: true);
                loss += exLoss;
                tokens += e.Completion.Length;
                AccumulateLossGrad(captures, e, dx!, grads, usage, router);
            }
            finally
            {
                _capture = null;
            }
        }
        foreach (var g in grads.Values)
            for (int j = 0; j < g.Length; j++)
                g[j] /= tokens;
        foreach (var u in usage.Values)
            for (int k = 0; k < u.Length; k++)
                u[k] /= tokens;
        return (loss / tokens, grads, usage);
    }

    /// <summary>dL/dlogit_e = p_e · g·(f_e − y) for each completion row, with g the
    /// loss gradient on the residual the layer writes to, f_e expert e's delta, and
    /// y the mixture's; then dL/dW = dlogit ⊗ x (or the prompt mean of x, held).</summary>
    private void AccumulateLossGrad(Dictionary<int, Capture> captures, LoraExample e, float[] dx,
        Dictionary<int, float[]> grads, Dictionary<int, double[]> usage, MoeRouter router)
    {
        int n = Names.Count, hidden = Plan.HiddenSize;
        int rowStart = e.Prompt.Length - 1, rows = e.Completion.Length;
        foreach (var (layer, c) in captures)
        {
            for (int r = rowStart; r < rowStart + rows; r++)
                for (int k = 0; k < n; k++)
                    usage[layer][k] += c.P[r * n + k];
            var gw = grads[layer];
            var dlogitHeld = new double[n];
            for (int r = 0; r < rows; r++)
            {
                int row = rowStart + r;
                var g = dx.AsSpan(r * hidden, hidden);
                Span<double> gf = stackalloc double[n];
                double gy = 0;
                for (int k = 1; k < n; k++)
                {
                    if (c.Deltas[k] is not { } d)
                        continue;
                    gf[k] = TensorOps.Dot(g, d.Row(row));
                    gy += c.P[row * n + k] * gf[k];
                }
                for (int k = 0; k < n; k++)
                {
                    double dl = c.P[row * n + k] * (gf[k] - gy);
                    if (router.HoldPrompt)
                    {
                        dlogitHeld[k] += dl;
                        continue;
                    }
                    if (dl == 0)
                        continue;
                    var x = c.Input.Row(row);
                    var gk = gw.AsSpan(k * hidden, hidden);
                    float f = (float)dl;
                    for (int h = 0; h < hidden; h++)
                        gk[h] += f * x[h];
                }
            }
            if (router.HoldPrompt)
            {
                int held = Math.Min(e.Prompt.Length, c.Input.Rows);
                var mean = new double[hidden];
                for (int r = 0; r < held; r++)
                {
                    var x = c.Input.Row(r);
                    for (int h = 0; h < hidden; h++)
                        mean[h] += x[h] / (double)held;
                }
                for (int k = 0; k < n; k++)
                {
                    var gk = gw.AsSpan(k * hidden, hidden);
                    for (int h = 0; h < hidden; h++)
                        gk[h] += (float)(dlogitHeld[k] * mean[h]);
                }
            }
        }
    }

    /// <summary>Fits each layer's router to the topic labels: a multinomial logistic
    /// regression on the FFN inputs, the inputs captured with each example routed to
    /// its own topic's expert (what a good router will reproduce). No backward pass
    /// through the model. Rows of a topic with no expert go to the anchor.</summary>
    public MoeRouter TrainByLabel(IReadOnlyList<LoraExample> data, IReadOnlyList<Tensor2D> prefixes,
        IReadOnlyList<string> topics, RouterTrainOptions options, Action<object> log)
    {
        int n = Names.Count, hidden = Plan.HiddenSize;
        var rowsByLayer = Layers.ToDictionary(l => l, _ => new List<float[]>());
        var labels = new List<int>();
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < data.Count; i++)
        {
            var e = data[i];
            int label = Math.Max(0, Names.ToList().IndexOf(topics[i]));
            var oracle = new double[n];
            oracle[label] = 1;
            _capture = new Dictionary<int, Capture>();
            try
            {
                Forward(e, prefixes[i], Mixing.Of(oracle), wantGrad: false);
                var picked = new List<int>();
                int total = _capture[First].Input.Rows;
                if (options.HoldPrompt)
                {
                    picked.Add(-1); // one row: the prompt mean
                }
                else
                {
                    for (int r = 0; r < total; r += Math.Max(1, options.RowStride))
                        picked.Add(r);
                }
                foreach (var l in Layers)
                {
                    var input = _capture[l].Input;
                    foreach (var r in picked)
                    {
                        if (r >= 0)
                        {
                            rowsByLayer[l].Add(input.Row(r).ToArray());
                            continue;
                        }
                        int held = Math.Min(e.Prompt.Length, input.Rows);
                        var mean = new float[hidden];
                        for (int q = 0; q < held; q++)
                        {
                            var x = input.Row(q);
                            for (int h = 0; h < hidden; h++)
                                mean[h] += x[h] / held;
                        }
                        rowsByLayer[l].Add(mean);
                    }
                }
                labels.AddRange(picked.Select(_ => label));
            }
            finally
            {
                _capture = null;
            }
        }
        log(new { kind = "label-rows", rows = labels.Count, seconds = sw.Elapsed.TotalSeconds });

        var router = Start(options);
        foreach (var l in Layers)
        {
            var xs = rowsByLayer[l];
            var w = router.Weights[l];
            var m = new float[w.Length];
            var v = new float[w.Length];
            double loss = 0, acc = 0;
            for (int step = 1; step <= options.Steps; step++)
            {
                var g = new float[w.Length];
                loss = 0;
                acc = 0;
                Span<float> logits = stackalloc float[n];
                Span<float> p = stackalloc float[n];
                for (int r = 0; r < xs.Count; r++)
                {
                    var x = xs[r];
                    for (int k = 0; k < n; k++)
                        logits[k] = TensorOps.Dot(x, w.AsSpan(k * hidden, hidden));
                    Softmax(logits, p, 0);
                    int y = labels[r];
                    loss -= Math.Log(Math.Max(p[y], 1e-12));
                    int arg = 0;
                    for (int k = 1; k < n; k++)
                        if (p[k] > p[arg])
                            arg = k;
                    acc += arg == y ? 1 : 0;
                    for (int k = 0; k < n; k++)
                    {
                        float d = p[k] - (k == y ? 1f : 0f);
                        var gk = g.AsSpan(k * hidden, hidden);
                        for (int h = 0; h < hidden; h++)
                            gk[h] += d * x[h];
                    }
                }
                for (int j = 0; j < g.Length; j++)
                    g[j] /= xs.Count;
                double norm = Math.Sqrt(LoraTrainer.SumSq(g));
                double clip = norm > options.ClipNorm ? options.ClipNorm / norm : 1.0;
                double c1 = 1 - Math.Pow(0.9, step), c2 = 1 - Math.Pow(0.999, step);
                LoraTrainer.AdamStep(w, g, m, v, options.LearningRate, clip, 0.9, 0.999, 1e-8, c1, c2, options.WeightDecay);
            }
            log(new { kind = "label-fit", layer = l, loss = loss / xs.Count, accuracy = acc / xs.Count, rows = xs.Count });
        }
        return new MoeRouter
        {
            Experts = router.Experts, Layers = router.Layers, Hidden = hidden, Weights = router.Weights,
            HoldPrompt = options.HoldPrompt, Method = "label",
        };
    }

    // ── materialising the MoE ───────────────────────────────────────────

    /// <summary>
    /// Writes the mixture as a routed container: each routed layer gets the router
    /// and one full FFN per expert (the anchor's is the base's; a branch's shares the
    /// base's gate and up and carries down + ΔW in f32), in the AMQL expert layout the
    /// planner and <see cref="FfnKernel.Routed"/> run. Layers no expert changed keep
    /// their dense FFN. Top-k selection renormalises over the selected experts, the
    /// same rule the bench routes by.
    /// </summary>
    public static string WriteMoe(string containerDir, string outDir, IReadOnlyList<LoraExpert> experts,
        MoeRouter router, int topK)
    {
        if (Directory.Exists(outDir))
            throw new MergeException($"moe-merge output '{outDir}' already exists");
        if (router.HoldPrompt)
            throw new MergeException("a prompt-held router decides from the whole prompt, which a per-token routed " +
                                     "FFN cannot express; train a per-token router to materialise");
        var names = new[] { "anchor" }.Concat(experts.Select(e => e.Name)).ToArray();
        if (!names.SequenceEqual(router.Experts))
            throw new MergeException($"the router's experts ({string.Join(",", router.Experts)}) are not " +
                                     $"anchor + the experts given ({string.Join(",", names)}), in order");
        int n = names.Length;
        if (topK <= 0 || topK > n)
            topK = n;

        using var container = Vindex3Container.Open(containerDir);
        var graph = container.Graph ?? throw new MergeException("container records no system graph");
        var component = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.PrimaryText)
            ?? throw new MergeException("container records no primary text component");
        var surface = component.Execution ?? throw new MergeException($"component '{component.Id}' declares no execution surface");
        var ffn = surface.Ffn ?? throw new MergeException($"component '{component.Id}' declares no FFN surface");
        if (ffn.Moe is not null)
            throw new MergeException("the base is already routed; moe-merge starts from a dense model");
        if (ffn.FfnType != FfnType.Gated)
            throw new MergeException($"moe-merge needs a gated FFN, the surface declares {ffn.FfnType}");

        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, component.Id, store);
        var weights = new GenericRuntime(plan, store).Weights;
        int hidden = component.HiddenSize, inter = ffn.IntermediateSize;
        var stackId = graph.Objects.First(o => o.Component == component.Id && o.Kind == ObjectKind.DecoderStack).Id;
        var stackRep = container.CanonicalRepresentationId(stackId);
        var stackEncoding = container.Index.Representations[stackRep].Encoding;
        var stackSegmentPath = container.Index.Representations[stackRep].Segment;

        var routed = router.Layers.ToHashSet();
        var drop = new HashSet<string>(StringComparer.Ordinal);
        foreach (var l in routed)
        {
            var dense = plan.Layers[l].Ffn?.Dense ?? throw new MergeException($"layer {l} has no dense FFN");
            if (dense.Up.TensorName != $"{l}.mlp.up_proj.weight" || dense.Down.TensorName != $"{l}.mlp.down_proj.weight" ||
                dense.Gate?.TensorName != $"{l}.mlp.gate_proj.weight")
                throw new MergeException($"layer {l}: moe-merge writes the AMQL expert layout and needs the dense FFN " +
                                         $"named {l}.mlp.{{gate,up,down}}_proj.weight, found '{dense.Down.TensorName}'");
            drop.Add(dense.Gate!.TensorName);
            drop.Add(dense.Up.TensorName);
            drop.Add(dense.Down.TensorName);
        }

        var rebuilt = new List<NamedTensorData>();
        using (var source = SegmentFile.Open(Path.Combine(container.Root, stackSegmentPath)))
        {
            foreach (var tensor in source.Header.Tensors)
            {
                if (drop.Contains(tensor.Name))
                    continue;
                rebuilt.Add(new NamedTensorData
                {
                    Name = tensor.Name, Dtype = DtypeExtensions.FromLabel(tensor.Dtype),
                    Shape = tensor.Shape, Data = source.ReadBytes(tensor.Name),
                });
            }
        }
        foreach (var l in router.Layers)
        {
            var dense = plan.Layers[l].Ffn!.Dense!;
            var gate = MoeIfy.Resolve(store, stackId, dense.Gate!.TensorName);
            var up = MoeIfy.Resolve(store, stackId, dense.Up.TensorName);
            var down = MoeIfy.Resolve(store, stackId, dense.Down.TensorName);
            var baseDown = weights.Matrix(dense.Down, hidden, inter).Data;
            rebuilt.Add(new NamedTensorData
            {
                Name = $"{l}.mlp.router.weight", Dtype = Dtype.F32,
                Shape = new long[] { n, hidden }, Data = MoeIfy.ToF32Bytes(router.Weights[l]),
            });
            for (int e = 0; e < n; e++)
            {
                string stem = $"{l}.mlp.experts.{e}";
                rebuilt.Add(new NamedTensorData { Name = $"{stem}.gate_proj.weight", Dtype = gate.Dtype, Shape = new long[] { inter, hidden }, Data = gate.Payload });
                rebuilt.Add(new NamedTensorData { Name = $"{stem}.up_proj.weight", Dtype = up.Dtype, Shape = new long[] { inter, hidden }, Data = up.Payload });
                var lora = e == 0 ? null : experts[e - 1].Layers.GetValueOrDefault(l);
                if (lora is null)
                {
                    rebuilt.Add(new NamedTensorData { Name = $"{stem}.down_proj.weight", Dtype = down.Dtype, Shape = new long[] { hidden, inter }, Data = down.Payload });
                    continue;
                }
                var delta = lora.Delta();
                var merged = new float[delta.Length];
                for (int j = 0; j < merged.Length; j++)
                    merged[j] = baseDown[j] + delta[j];
                rebuilt.Add(new NamedTensorData { Name = $"{stem}.down_proj.weight", Dtype = Dtype.F32, Shape = new long[] { hidden, inter }, Data = MoeIfy.ToF32Bytes(merged) });
            }
        }

        Directory.CreateDirectory(outDir);
        var representations = new Dictionary<string, RepresentationEntry>(StringComparer.Ordinal);
        var segments = new Dictionary<string, int>(StringComparer.Ordinal);
        var entry = SegmentWriter.Write(Path.Combine(outDir, stackSegmentPath), $"{stackId}@{stackEncoding}", rebuilt);
        representations[stackRep] = new RepresentationEntry
        {
            Object = stackId, Encoding = stackEncoding, Segment = stackSegmentPath, TensorCount = rebuilt.Count,
            PayloadBytes = entry.PayloadBytes, PayloadSha256 = entry.PayloadSha256Hex, SegmentSha256 = entry.SegmentSha256Hex,
        };
        segments[stackSegmentPath[..^4]] = 1;
        MoeIfy.CopyTokenizer(container, outDir);
        foreach (var (repId, other) in container.Index.Representations)
        {
            if (representations.ContainsKey(repId))
                continue;
            Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(outDir, other.Segment))!);
            File.Copy(Path.Combine(container.Root, other.Segment), Path.Combine(outDir, other.Segment), overwrite: true);
            representations[repId] = other;
            segments[other.Segment[..^4]] = 1;
        }
        var moeFfn = new FfnSurface
        {
            IntermediateSize = ffn.IntermediateSize, Activation = ffn.Activation, FfnType = ffn.FfnType, GatePolicy = ffn.GatePolicy,
            Moe = new MoeSurface
            {
                Experts = n, TopK = topK, ExpertIntermediateSize = inter,
                RoutingPolicy = ExpertRoutingPolicy.NormalisedOverSelected,
            },
        };
        MoeIfy.WriteGraph(outDir, graph, component, surface, moeFfn);
        string model = $"{container.Index.Model}-bar{n}x{topK}";
        MoeIfy.WriteIndex(outDir, container, component, model, representations, segments);
        File.WriteAllText(Path.Combine(outDir, "moe_experts.json"), JsonSerializer.Serialize(new
        {
            format = "amql-bar-v1", experts = names, layers = router.Layers, top_k = topK, router_method = router.Method,
        }, new JsonSerializerOptions { WriteIndented = true }));
        return model;
    }
}
