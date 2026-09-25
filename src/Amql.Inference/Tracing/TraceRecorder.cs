using System.Text;
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

/// <summary>The logit lens at one layer: what the model would have predicted had
/// it stopped there and emitted immediately. Reading this down the layers shows
/// the depth at which the eventual token forms — something activation magnitude
/// cannot tell you, since a layer can be quiet and still be where the decision
/// lands.</summary>
public sealed record LensRow(int Layer, int TokenId, float Probability, IReadOnlyList<TokenCandidate> TopK);

/// <summary>One head's post-softmax attention row at one step: the weight this
/// head's last query position put on each key position. Only softmax-attention
/// layers produce one — in a Qwen3.5 model three layers in four are recurrent and
/// have no attention matrix at all, so a viewer must not imply otherwise.</summary>
public sealed record AttentionRow(int Layer, int Head, IReadOnlyList<float> Weights);

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
    IReadOnlyList<OpSample> Ops,
    IReadOnlyList<LensRow>? Lens = null,
    IReadOnlyList<AttentionRow>? Attention = null);

/// <summary>
/// Layer-level causal attribution captured alongside a run, when asked for.
/// ROME-style: corrupt one prompt token, then restore each layer's clean
/// residual in turn and re-measure the target token's probability. The share a
/// layer recovers is how much of that propensity actually lives there — a much
/// stronger statement than activation magnitude, which only says a tensor was
/// loud. It costs one forward per traced layer plus two, so it is opt-in.
/// <para>
/// Attribution is per layer, not per operator: causal tracing restores a whole
/// layer's residual, so it cannot resolve finer than that. A map coloured by it
/// should say so rather than imply operator-level resolution.
/// </para>
/// </summary>
public sealed record CausalInfo(
    int SourceRow,
    int SourceTokenId,
    int CorruptTokenId,
    int TargetTokenId,
    float CleanProbability,
    float CorruptProbability,
    IReadOnlyList<float> LayerDelta,
    IReadOnlyList<float> LayerShare)
{
    /// <summary>Propensity lost by corrupting the source token.</summary>
    public float TotalEffect => CleanProbability - CorruptProbability;

    /// <summary>The layer holding the largest share, or -1 if none moved.</summary>
    public int PeakLayer
    {
        get
        {
            int best = -1;
            float bestShare = 0f;
            for (int i = 0; i < LayerShare.Count; i++)
            {
                if (LayerShare[i] > bestShare)
                {
                    bestShare = LayerShare[i];
                    best = i;
                }
            }
            return best;
        }
    }
}

/// <summary>A complete traced run: the graph it walked, then every step.
/// Serialisable so a run can be saved, reopened, and diffed against another —
/// comparing two runs on the same map is how an edit to a weight tensor gets
/// judged.</summary>
public sealed record RunTrace(
    string Model,
    string ComponentId,
    string ContainerPath,
    int HiddenSize,
    int Layers,
    IReadOnlyList<int> PromptTokens,
    string Sampling,
    string WeightWorkingSet,
    string CapturedUtc,
    IReadOnlyList<OpNode> Nodes,
    IReadOnlyList<StepTrace> Steps,
    CausalInfo? Causal = null)
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

/// <summary>The identity of a run, written as the first record of a trace
/// stream so a consumer knows what it is watching before any step lands.</summary>
public sealed record RunHeader(
    string Model,
    string ComponentId,
    string ContainerPath,
    int HiddenSize,
    int Layers,
    IReadOnlyList<int> PromptTokens,
    string Sampling,
    string WeightWorkingSet);

/// <summary>
/// Line-delimited trace output, flushed per record so a consumer can tail the
/// file while generation is still running. The single JSON document
/// <see cref="TraceRecorder.WriteJson"/> produces is only complete once the run
/// ends, which is exactly what makes it useless for watching a run happen.
/// <para>
/// Each line is an object with a <c>kind</c> of <c>header</c>, <c>node</c>,
/// <c>step</c>, <c>causal</c> or <c>end</c>. Node definitions are interleaved
/// with the steps rather than collected up front, because operators are
/// discovered as the run proceeds.
/// </para>
/// </summary>
public sealed class TraceStream : IDisposable
{
    private readonly StreamWriter _writer;

    public TraceStream(string path)
    {
        string? dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }
        // Truncate: a stale stream from a previous run must not be mistaken for
        // the current one by whoever is tailing it.
        _writer = new StreamWriter(path, append: false, new UTF8Encoding(false))
        {
            AutoFlush = true,
        };
    }

    public void WriteHeader(RunHeader header) => Write("header", header);

    public void WriteNode(OpNode node) => Write("node", node);

    public void WriteStep(StepTrace step) => Write("step", step);

    public void WriteCausal(CausalInfo causal) => Write("causal", causal);

    /// <summary>Marks the stream complete. A consumer that never sees this knows
    /// the run died partway, which is worth distinguishing from a short one.</summary>
    public void WriteEnd() => _writer.WriteLine("{\"kind\":\"end\"}");

    private void Write(string kind, object payload)
    {
        string json = JsonSerializer.Serialize(payload, TraceRecorder.JsonOptions);
        // Splice the discriminator in rather than wrapping every record in an
        // envelope type: the payload stays exactly the shape the document
        // format uses, so one set of records serves both.
        _writer.WriteLine(json.Insert(1, $"\"kind\":\"{kind}\","));
    }

    public void Dispose() => _writer.Dispose();
}

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
    private readonly List<LensRow> _lens = new();
    private readonly List<AttentionRow> _attention = new();
    private int _stepIndex;
    private int _position;
    private TraceStream? _stream;

    /// <summary>Starts mirroring every record to a line-delimited file as it
    /// happens, so a consumer can watch the run rather than wait for it.</summary>
    public void AttachStream(string path, RunHeader header)
    {
        _stream = new TraceStream(path);
        _stream.WriteHeader(header);
        // Anything interned before the stream was attached still has to appear,
        // or the consumer sees step records referencing node ids it never got.
        foreach (var node in _nodes)
        {
            _stream.WriteNode(node);
        }
    }

    public void StreamCausal(CausalInfo causal) => _stream?.WriteCausal(causal);

    /// <summary>Finishes the stream. A consumer that never sees the end record
    /// knows the run was cut short rather than merely short.</summary>
    public void CloseStream()
    {
        _stream?.WriteEnd();
        _stream?.Dispose();
        _stream = null;
    }

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
        var node = new OpNode(id, layer, op, weight);
        _nodes.Add(node);
        _nodeIds[key] = id;
        _stream?.WriteNode(node);
        return id;
    }

    public void BeginStep(int position)
    {
        _position = position;
        _samples = new List<OpSample>();
        _experts.Clear();
        _lens.Clear();
        _attention.Clear();
    }

    /// <summary>Records one head's attention row for this step.</summary>
    public void ObserveAttention(int layer, int head, IReadOnlyList<float> weights)
        => _attention.Add(new AttentionRow(layer, head, weights));

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

    /// <summary>Records the logit lens at one layer, already reduced to its top
    /// candidates. The full distribution is vocab-sized per layer per step,
    /// which on a 24-layer model would dwarf everything else in the trace.</summary>
    public void ObserveLens(int layer, int tokenId, float probability, IReadOnlyList<TokenCandidate> topK)
    {
        _lens.Add(new LensRow(layer, tokenId, probability, topK));
    }

    public void EndStep(int tokenId, string? tokenText, IReadOnlyList<TokenCandidate> topK,
        float entropy, float top1Margin)
    {
        var step = new StepTrace(_stepIndex++, _position, tokenId, tokenText, entropy, top1Margin,
            topK, _experts.ToArray(), _samples.ToArray(),
            _lens.Count == 0 ? null : _lens.ToArray(),
            _attention.Count == 0 ? null : _attention.ToArray());
        _steps.Add(step);
        _stream?.WriteStep(step);
    }

    public RunTrace ToRunTrace(string model, string componentId, int hiddenSize, int layers,
        IReadOnlyList<int> promptTokens, string sampling, string weightWorkingSet,
        string containerPath = "", CausalInfo? causal = null)
        => new(model, componentId, containerPath, hiddenSize, layers, promptTokens, sampling,
            weightWorkingSet, DateTime.UtcNow.ToString("O"), _nodes.ToArray(), _steps.ToArray(), causal);

    internal static readonly JsonSerializerOptions JsonOptions = new()
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

    /// <summary>
    /// Rebuilds a run trace from a line-delimited stream file — the format the
    /// GUI tails during a live run. Tolerates a truncated final line and missing
    /// <c>end</c>, because a run that is still going, or one that was cancelled,
    /// leaves the file in exactly that state and the steps captured so far are
    /// still worth showing.
    /// </summary>
    public static RunTrace ReadStream(string path)
    {
        RunHeader? header = null;
        var nodes = new List<OpNode>();
        var steps = new List<StepTrace>();
        CausalInfo? causal = null;

        foreach (string line in File.ReadLines(path))
        {
            if (line.Length == 0)
            {
                continue;
            }
            try
            {
                using var doc = JsonDocument.Parse(line);
                string kind = doc.RootElement.TryGetProperty("kind", out var k)
                    ? k.GetString() ?? string.Empty
                    : string.Empty;
                switch (kind)
                {
                    case "header":
                        header = doc.RootElement.Deserialize<RunHeader>(JsonOptions);
                        break;
                    case "node":
                        if (doc.RootElement.Deserialize<OpNode>(JsonOptions) is { } node)
                        {
                            nodes.Add(node);
                        }
                        break;
                    case "step":
                        if (doc.RootElement.Deserialize<StepTrace>(JsonOptions) is { } step)
                        {
                            steps.Add(step);
                        }
                        break;
                    case "causal":
                        causal = doc.RootElement.Deserialize<CausalInfo>(JsonOptions);
                        break;
                }
            }
            catch (JsonException)
            {
                // A partially flushed trailing line. Everything before it is
                // still valid, and the next read will pick up the rest.
            }
        }

        header ??= new RunHeader(string.Empty, string.Empty, string.Empty, 0,
            nodes.Count == 0 ? 0 : nodes.Max(n => n.Layer) + 1,
            Array.Empty<int>(), string.Empty, string.Empty);
        return new RunTrace(header.Model, header.ComponentId, header.ContainerPath,
            header.HiddenSize, header.Layers, header.PromptTokens, header.Sampling,
            header.WeightWorkingSet, DateTime.UtcNow.ToString("O"), nodes, steps, causal);
    }
}
