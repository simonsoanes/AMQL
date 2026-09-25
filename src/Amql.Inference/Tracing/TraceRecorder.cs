using System.Text.Json;
using System.Text.Json.Serialization;
using Amql.Vindex3;

namespace Amql.Inference.Tracing;

/// <summary>
/// A traced point in the graph: one operator in one layer, plus the weight
/// tensor it reads when there is exactly one. Nodes are interned once per run
/// and referred to by id from every step, which is the difference between a
/// usable trace file and one that repeats a few dozen strings tens of
/// thousands of times.
/// <para>
/// The weight is an <see cref="OperandRef"/> rather than a display string so a
/// consumer can feed it straight back into <c>TensorPatchTools.ApplyEdit</c> —
/// identifying a tensor is only useful if you can then edit it.
/// </para>
/// </summary>
public sealed record OpNode(int Id, int Layer, string Op, OperandRef? Weight);

/// <summary>
/// What one operator contributed during one forward pass. Only these floats
/// are kept — the statistics are computed at the observation site so a trace
/// never holds an activation alive. A 64-layer model at roughly ten operators
/// per layer is ~640 samples per generated token; retaining the tensors
/// instead would be gigabytes for a run that is trivial to re-perform.
/// </summary>
public readonly record struct OpSample(int NodeId, float L2, float MaxAbs, float MeanAbs, double Ms);

/// <summary>One entry of the output distribution at a step. The text is
/// optional because the runtime has no tokenizer — whoever captures the trace
/// supplies a resolver, so the saved file is self-describing and a run opened
/// days later still reads as words rather than ids.</summary>
public sealed record TokenCandidate(int Id, float Logit, float Probability, string? Text = null);

/// <summary>Everything observed during one generated token.</summary>
public sealed record StepTrace(
    int Index,
    int Position,
    int TokenId,
    string? TokenText,
    float Entropy,
    float Top1Margin,
    IReadOnlyList<TokenCandidate> TopK,
    IReadOnlyList<int> RoutedExperts,
    IReadOnlyList<OpSample> Ops);

/// <summary>A complete traced run: the graph it walked, then every step.
/// Serialisable so a run can be saved, reopened, and diffed against another —
/// comparing two runs on the same map is how an edit to a weight tensor gets
/// judged.</summary>
public sealed record RunTrace(
    string Model,
    string ComponentId,
    int HiddenSize,
    int Layers,
    IReadOnlyList<int> PromptTokens,
    string Sampling,
    string WeightWorkingSet,
    string CapturedUtc,
    IReadOnlyList<OpNode> Nodes,
    IReadOnlyList<StepTrace> Steps)
{
    /// <summary>Per-node reduction over the whole run, which is what a map
    /// colours itself by: a single token says little, the aggregate says which
    /// tensors carry the computation.</summary>
    public IReadOnlyDictionary<int, NodeAggregate> Aggregate()
    {
        var acc = new Dictionary<int, (double SumL2, double MaxL2, int Count, double Ms)>();
        foreach (var step in Steps)
        {
            foreach (var op in step.Ops)
            {
                acc.TryGetValue(op.NodeId, out var a);
                acc[op.NodeId] = (a.SumL2 + op.L2, Math.Max(a.MaxL2, (double)op.L2),
                    a.Count + 1, a.Ms + op.Ms);
            }
        }
        var result = new Dictionary<int, NodeAggregate>(acc.Count);
        foreach (var (id, a) in acc)
        {
            result[id] = new NodeAggregate(
                MeanL2: (float)(a.SumL2 / Math.Max(1, a.Count)),
                MaxL2: (float)a.MaxL2,
                MeanMs: a.Ms / Math.Max(1, a.Count),
                Steps: a.Count);
        }
        return result;
    }
}

/// <summary>Per-node totals over a run.</summary>
public sealed record NodeAggregate(float MeanL2, float MaxL2, double MeanMs, int Steps);

/// <summary>
/// One operator observation as the runtime reports it. A struct so the hot
/// loop does not allocate per operator, and deliberately carrying only floats:
/// the runtime computes the statistics at the observation site and never hands
/// out the activation, so a trace cannot keep tensors alive.
/// </summary>
public readonly record struct OpObservation(
    int Layer, string Op, OperandRef? Weight, float L2, float MaxAbs, float MeanAbs, double Ms);

/// <summary>
/// Collects <see cref="OpSample"/>s into a <see cref="RunTrace"/>. Attach its
/// <see cref="Observe"/> to <c>GenericRuntime.OpTrace</c>; call
/// <see cref="BeginStep"/> before a forward and <see cref="EndStep"/> once the
/// sampled token is known.
/// </summary>
public sealed class TraceRecorder
{
    private readonly Dictionary<string, int> _nodeIds = new(StringComparer.Ordinal);
    private readonly List<OpNode> _nodes = new();
    private readonly List<StepTrace> _steps = new();
    private List<OpSample> _samples = new();
    private readonly List<int> _experts = new();
    private int _stepIndex;
    private int _position;

    /// <summary>Number of operators recorded so far — a cheap liveness check
    /// for tests and for the GUI's status line.</summary>
    public int SampleCount { get; private set; }

    public IReadOnlyList<OpNode> Nodes => _nodes;
    public IReadOnlyList<StepTrace> Steps => _steps;

    /// <summary>Interns an operator and returns its stable id.</summary>
    public int NodeId(int layer, string op, OperandRef? weight)
    {
        // The weight is part of the key because the same operator name in the
        // same layer can read different tensors across a mixed plan.
        string key = weight is null
            ? $"{layer}:{op}"
            : $"{layer}:{op}:{weight.ObjectId}\0{weight.TensorName}";
        if (_nodeIds.TryGetValue(key, out int id))
        {
            return id;
        }
        id = _nodes.Count;
        _nodes.Add(new OpNode(id, layer, op, weight));
        _nodeIds[key] = id;
        return id;
    }

    public void BeginStep(int position)
    {
        _position = position;
        _samples = new List<OpSample>();
        _experts.Clear();
    }

    /// <summary>Records one operator observation from the runtime hook. Taken
    /// by value so it binds directly to <c>Action&lt;OpObservation&gt;</c> —
    /// an <c>in</c> parameter would not, and the struct is small enough that
    /// the copy is cheaper than the indirection.</summary>
    public void Observe(OpObservation o)
    {
        _samples.Add(new OpSample(NodeId(o.Layer, o.Op, o.Weight), o.L2, o.MaxAbs, o.MeanAbs, o.Ms));
        SampleCount++;
    }

    /// <summary>Records which experts a routed FFN selected this step. Genuinely
    /// sparse, and the most interpretable signal a MoE layer produces.</summary>
    public void ObserveExperts(IReadOnlyList<int> expertIds)
    {
        for (int i = 0; i < expertIds.Count; i++)
        {
            _experts.Add(expertIds[i]);
        }
    }

    public void EndStep(int tokenId, string? tokenText, IReadOnlyList<TokenCandidate> topK,
        float entropy, float top1Margin)
    {
        _steps.Add(new StepTrace(_stepIndex++, _position, tokenId, tokenText, entropy, top1Margin,
            topK, _experts.ToArray(), _samples.ToArray()));
    }

    public RunTrace ToRunTrace(string model, string componentId, int hiddenSize, int layers,
        IReadOnlyList<int> promptTokens, string sampling, string weightWorkingSet)
        => new(model, componentId, hiddenSize, layers, promptTokens, sampling, weightWorkingSet,
            DateTime.UtcNow.ToString("O"), _nodes.ToArray(), _steps.ToArray());

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };

    public static void WriteJson(RunTrace trace, string path)
    {
        string? dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }
        File.WriteAllText(path, JsonSerializer.Serialize(trace, JsonOptions));
    }

    public static RunTrace ReadJson(string path)
        => JsonSerializer.Deserialize<RunTrace>(File.ReadAllText(path), JsonOptions)
           ?? throw new InvalidOperationException($"'{path}' did not deserialize to a run trace");
}
