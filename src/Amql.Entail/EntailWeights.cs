using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Entail;

/// <summary>What a compile reports about the model it wrote.</summary>
public sealed record EntailBuildReport(
    string Directory, int Hidden, int Layers, int BlockLayers, int LanesUsed, int LanesPeak,
    int Neurons, int Heads, double MinAttentionGap, int TokensBound, IReadOnlyList<string> TokensMissing);

/// <summary>
/// Turns an <see cref="EntailProgram"/> into tensors in a target's layer
/// shape (<see cref="EntailTarget"/>): the Qwen3.5 shape for the empty model
/// and the adapter, the host's own shape for a graft. The compiled block reads
/// and writes only its own lanes, so it is the same set of layers whether it
/// stands alone in an empty model or is grafted in front of a host's.
/// <para>
/// Three facts make a stack of RMSNorms behave like exact integer logic. A
/// constant "ballast", far larger than everything else, fixes the norm's
/// divisor. It is spread over just enough dimensions that the mean of squares
/// is a power of four, so the divisor is a power of two; everything else is
/// too small to survive the rounding back to float, so a lane reads back as
/// precisely the number written to it, and integers stay integers from layer
/// to layer with no drift at all. And a lane's norm weight, also a power of
/// two, undoes the divisor, while a host dimension's is zero, so nothing else
/// can reach a lane. A host whose layers normalise their sub-blocks' outputs
/// too (Gemma's sandwich, OLMo 2's post-norms) gets a ballast in each
/// sub-block's output as well, from a head and a neuron kept for it, so those
/// norms divide by a power of two as well.
/// </para>
/// </summary>
public static class EntailWeights
{
    /// <summary>Raw value of each ballast dimension; a power of two, so
    /// bf16-exact.</summary>
    public const double Ballast = 1048576;
    /// <summary>Rotary pairs a lookup uses, at most.</summary>
    private const int MaxPairs = 16;
    /// <summary>The score a head gives the token at its own offset: in the
    /// Qwen3.5 shape, a q/k norm gain of 28 over 16 of head_dim 256's pairs.</summary>
    private const double ReferencePeak = 12544;
    private const double OutputGate = 30; // sigmoid(30) = 1

    /// <summary>Lane → dimension assignment by liveness: a lane nothing reads
    /// any more is cleared by one neuron and its dimension is reused.</summary>
    private sealed class Allocation
    {
        public int[] Dim = Array.Empty<int>();             // lane → dim index, -1 if unused
        public int Peak;
        public int Used;
        public List<(int Layer, int Lane)> Clears = new(); // neuron that zeroes a dead lane
        public List<int> LiveAtEnd = new();
    }

    private static Allocation Allocate(EntailProgram p, int firstDim, bool keepEmitsLive)
    {
        int n = p.Lanes.Count;
        var firstWrite = new int[n];
        var lastRead = new int[n];
        var written = new bool[n];
        Array.Fill(firstWrite, int.MaxValue);
        Array.Fill(lastRead, -1);
        foreach (var writes in p.Tokens.Values)
        {
            foreach (var lane in writes.Keys)
            {
                firstWrite[lane] = -1;
                written[lane] = true;
            }
        }
        void Read(Lin l, int time)
        {
            foreach (var lane in l.Terms.Keys)
            {
                lastRead[lane] = Math.Max(lastRead[lane], time);
            }
        }
        foreach (var op in p.Ops)
        {
            if (op is Neuron neuron)
            {
                int t = 2 * neuron.Layer + 1;
                Read(neuron.Gate, t);
                Read(neuron.Up, t);
                firstWrite[neuron.Dest] = Math.Min(firstWrite[neuron.Dest], t);
                written[neuron.Dest] = true;
            }
            else if (op is Head head)
            {
                int t = 2 * head.Layer;
                foreach (var (dest, source) in head.Copies)
                {
                    Read(source, t);
                    firstWrite[dest] = Math.Min(firstWrite[dest], t);
                    written[dest] = true;
                }
                if (head.Pointer is { } pointer)
                {
                    foreach (var (lane, _) in pointer)
                    {
                        lastRead[lane] = Math.Max(lastRead[lane], t);
                    }
                }
            }
        }

        int end = 2 * p.Layers + 1;
        foreach (int lane in p.Emits.Select(e => e.Lane).Concat(p.Masks.Select(m => m.Lane)))
        {
            lastRead[lane] = keepEmitsLive ? int.MaxValue : end;
            if (keepEmitsLive)
            {
                // With a tied head an output lane is an embedding column, so
                // it may never share a dimension with an input lane.
                firstWrite[lane] = -1;
            }
        }

        var result = new Allocation { Dim = new int[n] };
        Array.Fill(result.Dim, -1);
        var free = new PriorityQueue<int, int>();          // dim, usable from time
        var order = Enumerable.Range(0, n).Where(l => written[l] && lastRead[l] >= 0)
            .OrderBy(l => firstWrite[l]).ThenBy(l => l).ToList();
        int nextDim = firstDim, live = 0;
        var active = new PriorityQueue<int, int>();        // lane by last read
        foreach (int lane in order)
        {
            while (active.TryPeek(out int old, out int until) && until < firstWrite[lane] - 1)
            {
                active.Dequeue();
                live--;
                // Cleared by the FFN that runs at or just after the last read.
                int clearLayer = lastRead[old] / 2;
                int clearTime = 2 * clearLayer + 1;
                if (clearTime < lastRead[old])
                {
                    clearLayer++;
                    clearTime += 2;
                }
                free.Enqueue(result.Dim[old], clearTime);
                result.Clears.Add((clearLayer, old));
            }
            if (free.TryPeek(out int dim, out int from) && from < firstWrite[lane])
            {
                free.Dequeue();
                result.Dim[lane] = dim;
            }
            else
            {
                result.Dim[lane] = nextDim++;
            }
            active.Enqueue(lane, lastRead[lane]);
            live++;
            result.Peak = Math.Max(result.Peak, live);
        }
        while (active.TryDequeue(out int lane, out _))
        {
            if (!p.Lanes[lane].Emits)
            {
                result.LiveAtEnd.Add(lane);
            }
        }
        result.Used = nextDim - firstDim;
        return result;
    }

    // ── attention geometry ──────────────────────────────────────────────────

    /// <summary>The rotary pairs a target's heads look up with, and the
    /// score they give their own offset.</summary>
    private sealed class Lookup
    {
        public required RopePair[] Pairs { get; init; }
        public required double Peak { get; init; }
        public float? Softcap { get; init; }
        public double Gap { get; set; }
        /// <summary>|q|·|k| per pair, in the units attention sees, before the
        /// score scale and the rope attention factor.</summary>
        public double Amplitude(EntailTarget t) =>
            Math.Sqrt(Peak / (t.ScoreScale * t.AttentionFactor * t.AttentionFactor * Pairs.Length));
    }

    /// <summary>A query, before rotation, whose score against the constant
    /// key peaks at exactly <paramref name="offset"/> tokens back.</summary>
    private static double[] OffsetQuery(Lookup look, int headDim, int offset)
    {
        var q = new double[headDim];
        foreach (var pair in look.Pairs)
        {
            q[pair.A] = Math.Cos(pair.Omega * offset);
            q[pair.B] = -Math.Sin(pair.Omega * offset);
        }
        return q;
    }

    /// <summary>The score a head looking <paramref name="target"/> back gives
    /// the token <paramref name="delta"/> back, softcapped as the host caps.</summary>
    private static double Score(Lookup look, int target, int delta)
    {
        double sum = 0;
        foreach (var pair in look.Pairs)
        {
            sum += Math.Cos(pair.Omega * (delta - target));
        }
        double score = look.Peak * sum / look.Pairs.Length;
        return look.Softcap is { } cap ? cap * Math.Tanh(score / cap) : score;
    }

    private static double Gap(Lookup look, IEnumerable<int> offsets, int context)
    {
        double gap = double.MaxValue;
        foreach (int target in offsets)
        {
            double peak = Score(look, target, target);
            for (int delta = 0; delta < context; delta++)
            {
                if (delta != target)
                {
                    gap = Math.Min(gap, peak - Score(look, target, delta));
                }
            }
        }
        return gap;
    }

    private static IEnumerable<int> Offsets(EntailProgram p, bool ballastHead) =>
        p.Ops.OfType<Head>()
            .SelectMany(h => h.Pointer?.Select(x => x.Offset) ?? new[] { h.Offset })
            .Concat(ballastHead ? new[] { 0 } : Array.Empty<int>())
            .Distinct();

    /// <summary>The smallest margin by which any head's intended token beats
    /// every other token, over the whole context, in the empty model's shape.
    /// Compile-time proof that attention is a lookup, not a blend.</summary>
    public static double AttentionGap(EntailProgram p, int context) =>
        ChooseLookup(EntailTarget.Fresh(0), p, context, strict: false).Gap;

    /// <summary>Picks the rotary pairs and score scale the heads use. The
    /// highest-frequency pairs are tried first (the empty model's choice);
    /// pairs spread across the host's whole range next, for a host whose
    /// first pairs repeat within the context. A pair is usable only if it
    /// turns at the same rate in every regime the context reaches. With a
    /// softcap, the peak is chosen to keep the most margin under it. A graft
    /// refuses a host where no choice makes every head a lookup: the softmax
    /// mass off the intended token must stay below float resolution.</summary>
    private static Lookup ChooseLookup(EntailTarget t, EntailProgram p, int context, bool strict)
    {
        if (t.FrequenciesChangeAfter is { } after && context > after)
        {
            throw new EntailException(
                $"the host's dynamic rope changes every frequency past {after} tokens, which would move every lookup: graft with --context {after} or less");
        }
        var regimes = t.SecondRegimeAfter is { } second && context <= second ? t.RopeRegimes.Take(1).ToList() : t.RopeRegimes;
        var usable = regimes[0]
            .Where((pair, i) => regimes.All(r => r.Count > i && r[i].Omega == pair.Omega))
            .ToList();
        if (usable.Count == 0)
        {
            throw new EntailException("the host has no rotary pair whose frequency stays fixed over the context");
        }
        int n = Math.Min(MaxPairs, usable.Count);
        var selections = new List<RopePair[]> { usable.Take(n).ToArray() };
        if (usable.Count > n)
        {
            selections.Add(Enumerable.Range(0, n).Select(k => usable[(int)Math.Round(k * (usable.Count - 1.0) / (n - 1))]).ToArray());
        }
        var peaks = t.Softcap is { } cap
            ? new[] { 1.0, 1.5, 2, 3, 0.75, 0.5 }.Select(m => m * cap).ToArray()
            : new[] { ReferencePeak };
        var offsets = Offsets(p, t.PostNorms).ToList();
        double required = Math.Log(Math.Max(2, context)) + 25 * Math.Log(2);
        Lookup? best = null;
        foreach (var pairs in selections)
        {
            foreach (var peak in peaks)
            {
                var look = new Lookup { Pairs = pairs, Peak = peak, Softcap = t.Softcap };
                look.Gap = offsets.Count == 0 ? double.MaxValue : Gap(look, offsets, context);
                // The empty model keeps its own pairs, whatever the margin: it is
                // reported, not enforced, there.
                if (look.Gap >= required || !strict)
                {
                    return look;
                }
                if (best is null || look.Gap > best.Gap)
                {
                    best = look;
                }
            }
        }
        if (strict)
        {
            throw new EntailException(
                $"no choice of the host's rotary pairs makes every head a lookup over {context} tokens: the best margin is " +
                $"{best!.Gap:F1} logits and float resolution needs {required:F1} (lower --context)");
        }
        return best!;
    }

    // ── the block ───────────────────────────────────────────────────────────

    private sealed class Layer
    {
        public required float[] Q, K, V, O, Gate, Up, Down;
        public float[]? InNorm, FfnInNorm, AttnOutNorm, FfnOutNorm, QNorm, KNorm, SharedExpertGate;
    }

    private sealed class Geometry
    {
        public required int Hidden, Intermediate, Ballast, BallastDims, Shift;
        public required Allocation Lanes;
        public required double[] Scale;   // lane → raw per unit, this target
        public required bool[] HostDim;   // dims the block must not read
        public required EntailTarget Target;
        public required Lookup Look;
        /// <summary>What a ballast dim reads as at a matrix input: the norm's
        /// divisor undone (2^shift), or the raw ballast with no input norm.</summary>
        public double BallastRead => Target.PreNorms ? Math.Pow(2, Shift) : EntailWeights.Ballast;
        public double LaneGamma => EntailWeights.Ballast / Math.Pow(2, Shift);
    }

    /// <summary>Splits hidden = n · 4^shift with shift as large as possible:
    /// n ballast dimensions make the mean of squares Ballast² / 4^shift, so
    /// the norm's divisor is the power of two Ballast / 2^shift.</summary>
    private static (int Dims, int Shift) BallastShape(int hidden)
    {
        int n = hidden, shift = 0;
        while (n % 4 == 0)
        {
            n /= 4;
            shift++;
        }
        return (n, shift);
    }

    /// <summary>The smallest width holding <paramref name="fixedDims"/> plus
    /// its own ballast. With <paramref name="powerOfFour"/> the width is a
    /// power of four, so sqrt(hidden), which a Gemma scales its embedding by,
    /// is a power of two.</summary>
    private static int WidthFor(int fixedDims, bool powerOfFour = false)
    {
        if (powerOfFour)
        {
            int width = 4;
            while (width < fixedDims + 1)
            {
                width *= 4;
            }
            return width;
        }
        for (int hidden = RoundUp(fixedDims + 1, 64); ; hidden += 64)
        {
            if (fixedDims + BallastShape(hidden).Dims <= hidden)
            {
                return hidden;
            }
        }
    }

    private static void ReadInto(float[] matrix, int row, int hidden, Lin form, Geometry g, double factor = 1)
    {
        int at = row * hidden;
        matrix[at + g.Ballast] += (float)(form.Const * factor / g.BallastRead);
        foreach (var (lane, c) in form.Terms)
        {
            int dim = g.Lanes.Dim[lane];
            if (dim >= 0)
            {
                matrix[at + dim] += (float)(c * factor / g.Scale[lane]);
            }
        }
    }

    /// <summary>A norm weight over the widened stream: <paramref name="laneGamma"/>
    /// on lanes, <paramref name="ballastGamma"/> on the first ballast dim,
    /// <paramref name="spareGamma"/> on the rest, zero on host dims.</summary>
    private static float[] NormRow(Geometry g, double laneGamma, double ballastGamma = 1, double spareGamma = 0)
    {
        var w = new float[g.Hidden];
        for (int d = 0; d < g.Hidden; d++)
        {
            bool spare = d > g.Ballast && d < g.Ballast + g.BallastDims;
            double gamma = g.HostDim[d] ? 0 : spare ? spareGamma : d == g.Ballast ? ballastGamma : laneGamma;
            w[d] = (float)(gamma - g.Target.NormOffset);
        }
        return w;
    }

    private static Layer BuildLayer(EntailProgram p, Geometry g, int index,
        IEnumerable<(double GateConst, int Lane, bool ClearBallast)> clears)
    {
        var t = g.Target;
        var look = g.Look;
        int h = g.Hidden, inter = g.Intermediate, headDim = t.HeadDim;
        var clearList = clears.ToList();
        bool clearsBallast = clearList.Any(c => c.ClearBallast);
        double lane = g.LaneGamma;
        var layer = new Layer
        {
            Q = new float[(t.OutputGate ? 2 : 1) * t.QDim * h],
            K = new float[t.KvDim * h],
            V = new float[t.KvDim * h],
            O = new float[h * t.QDim],
            Gate = new float[inter * h],
            Up = new float[inter * h],
            Down = new float[h * inter],
            InNorm = t.PreNorms ? NormRow(g, lane) : null,
            FfnInNorm = t.PreNorms ? NormRow(g, lane) : null,
            // A sub-block's output ballast fixes its post-norm's divisor and
            // stays out of the stream; the layer that clears the ballast lets
            // its FFN's −ballast through instead.
            AttnOutNorm = t.PostNorms ? NormRow(g, lane, 0, 0) : null,
            FfnOutNorm = t.PostNorms ? clearsBallast ? NormRow(g, lane, lane, lane) : NormRow(g, lane, 0, 0) : null,
        };
        double amplitude = look.Amplitude(t);
        double rotaryRms = Math.Sqrt(look.Pairs.Length / (double)headDim);
        int qStride = t.OutputGate ? 2 * headDim : headDim;
        bool normed = t.QkNorm != QkNormKind.None;

        // Keys: each group's constant key, one per rotary pair. A q/k norm
        // rescales it, so its height matters only without one.
        float key = (float)(normed ? 1 : amplitude / g.BallastRead);
        for (int group = 0; group < t.KvGroups; group++)
        {
            foreach (var pair in look.Pairs)
            {
                layer.K[(group * headDim + pair.A) * h + g.Ballast] = key;
            }
        }

        // Heads: each group shares one constant key and one value table.
        var heads = p.Ops.OfType<Head>().Where(x => x.Layer == index).ToList();
        var slots = Enumerable.Range(0, t.KvGroups).Select(_ => new Dictionary<string, int>()).ToArray();
        var counts = new int[t.KvGroups];
        int Capacity(int group) => t.PostNorms && group == t.KvGroups - 1 ? t.HeadsPerGroup - 1 : t.HeadsPerGroup;
        int activeHeads = 0;
        int Slot(int group, Lin source)
        {
            var slotKey = source.Key();
            if (!slots[group].TryGetValue(slotKey, out int slot))
            {
                slot = slots[group].Count;
                if (slot >= headDim)
                {
                    throw new EntailException($"layer {index}: more than {headDim} values in one attention group");
                }
                slots[group][slotKey] = slot;
                ReadInto(layer.V, group * headDim + slot, h, source, g);
            }
            return slot;
        }
        void Query(int qHead, IEnumerable<(int Dim, double Unit, int Offset)> parts)
        {
            int block = qHead * qStride;
            if (t.OutputGate)
            {
                for (int d = 0; d < headDim; d++)
                {
                    layer.Q[(block + headDim + d) * h + g.Ballast] = (float)(OutputGate / g.BallastRead);
                }
            }
            foreach (var (dim, unit, offset) in parts)
            {
                var q = OffsetQuery(look, headDim, offset);
                for (int d = 0; d < headDim; d++)
                {
                    // A q/k norm sets the height; without one it is written in.
                    layer.Q[(block + d) * h + dim] += (float)(normed ? q[d] : q[d] * amplitude / unit);
                }
            }
            activeHeads++;
        }
        foreach (var head in heads)
        {
            int group = Enumerable.Range(0, t.KvGroups).FirstOrDefault(gr => counts[gr] < Capacity(gr), -1);
            if (group < 0)
            {
                throw new EntailException($"layer {index}: more than {t.ProgramHeads} heads");
            }
            int qHead = group * t.HeadsPerGroup + counts[group]++;
            if (head.Pointer is { } pointer)
            {
                Query(qHead, pointer.Where(x => g.Lanes.Dim[x.Lane] >= 0)
                    .Select(x => (g.Lanes.Dim[x.Lane], g.Scale[x.Lane], x.Offset)));
            }
            else
            {
                Query(qHead, new[] { (g.Ballast, g.BallastRead, head.Offset) });
            }
            foreach (var (dest, source) in head.Copies)
            {
                if (g.Lanes.Dim[dest] < 0)
                {
                    continue;
                }
                int slot = Slot(group, source);
                layer.O[g.Lanes.Dim[dest] * t.QDim + qHead * headDim + slot] = (float)(g.Scale[dest] / t.ResidualScale);
            }
        }
        if (t.PostNorms)
        {
            // The kept head reads a constant 1, so the attention output carries
            // the ballast its post-norm divides by.
            int group = t.KvGroups - 1;
            int qHead = group * t.HeadsPerGroup + t.HeadsPerGroup - 1;
            Query(qHead, new[] { (g.Ballast, g.BallastRead, 0) });
            int slot = Slot(group, Lin.Of(1));
            for (int d = 0; d < g.BallastDims; d++)
            {
                layer.O[(g.Ballast + d) * t.QDim + qHead * headDim + slot] = (float)Ballast;
            }
        }

        // q/k norm weights: the height a lookup needs, whatever the width
        // the norm divides over.
        switch (t.QkNorm)
        {
            case QkNormKind.PerHead:
                layer.QNorm = Enumerable.Repeat((float)(amplitude * rotaryRms - t.QkNormOffset), headDim).ToArray();
                layer.KNorm = Enumerable.Repeat((float)(amplitude * rotaryRms - t.QkNormOffset), headDim).ToArray();
                break;
            case QkNormKind.Whole:
            {
                // Over all heads at once: the rms counts every head with a query.
                double qRms = Math.Sqrt(Math.Max(1, activeHeads) * look.Pairs.Length / (double)t.QDim);
                layer.QNorm = Enumerable.Repeat((float)(amplitude * qRms - t.QkNormOffset), t.QDim).ToArray();
                layer.KNorm = Enumerable.Repeat((float)(amplitude * rotaryRms - t.QkNormOffset), t.KvDim).ToArray();
                break;
            }
        }

        // Neurons.
        int at = 0;
        double ffnScale = t.ResidualScale * t.ExpertWeight;
        void Place(Lin gate, Lin up, IEnumerable<(int Dim, double Raw)> dests)
        {
            if (at >= inter)
            {
                throw new EntailException($"layer {index}: more than {inter} neurons, once dead lanes are cleared; lower --neurons-per-layer to leave room");
            }
            ReadInto(layer.Gate, at, h, gate, g);
            ReadInto(layer.Up, at, h, up, g);
            foreach (var (dim, raw) in dests)
            {
                layer.Down[dim * inter + at] = (float)(raw / ffnScale);
            }
            at++;
        }
        foreach (var n in p.Ops.OfType<Neuron>().Where(x => x.Layer == index))
        {
            int dim = g.Lanes.Dim[n.Dest];
            if (dim >= 0)
            {
                Place(n.Gate, n.Up, new[] { (dim, n.Coef * g.Scale[n.Dest]) });
            }
        }
        foreach (var (gateConst, laneIndex, clearBallast) in clearList)
        {
            var gate = Lin.Of(gateConst);
            if (clearBallast)
            {
                for (int d = 0; d < g.BallastDims; d++)
                {
                    Place(gate, Lin.Of(1 / gateConst), new[] { (g.Ballast + d, -Ballast) });
                }
            }
            else
            {
                // Up reads the lane's meaning; the down weight restores raw.
                Place(gate, Lin.OfLane(laneIndex).Scaled(1 / gateConst), new[] { (g.Lanes.Dim[laneIndex], -g.Scale[laneIndex]) });
            }
        }
        if (t.PostNorms && !clearsBallast)
        {
            // The FFN output's ballast, for its post-norm.
            Place(Lin.Of(EntailCompiler.G), Lin.Of(1 / EntailCompiler.G),
                Enumerable.Range(0, g.BallastDims).Select(d => (g.Ballast + d, Ballast * ffnScale)));
        }
        if (t.Ffn == BlockFfn.SharedExpert && t.Names.ContainsKey($"{TensorRoles.SharedExpertGate}.weight"))
        {
            layer.SharedExpertGate = new float[h];
            layer.SharedExpertGate[g.Ballast] = (float)(OutputGate / g.BallastRead);
        }
        return layer;
    }

    /// <summary>A block layer's tensors under the target's own names, fused
    /// where the target stores them fused, with zero biases where it has them.</summary>
    private static IEnumerable<TensorPayload> LayerTensors(Layer l, EntailTarget t, string prefix, int h, int inter)
    {
        var tensors = new List<(string Name, float[] Data, long[] Shape)>();
        string? Name(string role, string part = "weight") => t.Names.TryGetValue($"{role}.{part}", out var n) ? n : null;
        void Add(string role, float[] data, params long[] shape)
        {
            tensors.Add((Name(role) ?? throw new EntailException($"the target has no name for '{role}'"), data, shape));
            if (Name(role, "bias") is { } bias)
            {
                tensors.Add((bias, new float[shape[0]], new[] { shape[0] }));
            }
        }
        if (l.InNorm is not null)
        {
            Add(TensorRoles.AttnIn, l.InNorm, h);
        }
        if (l.AttnOutNorm is not null)
        {
            Add(TensorRoles.AttnOut, l.AttnOutNorm, h);
        }
        if (l.FfnInNorm is not null)
        {
            Add(TensorRoles.FfnIn, l.FfnInNorm, h);
        }
        if (l.FfnOutNorm is not null)
        {
            Add(TensorRoles.FfnOut, l.FfnOutNorm, h);
        }
        int qRows = l.Q.Length / h;
        Add(TensorRoles.Q, l.Q, qRows, h);
        Add(TensorRoles.K, l.K, t.KvDim, h);
        Add(TensorRoles.V, l.V, t.KvDim, h);
        Add(TensorRoles.O, l.O, h, t.QDim);
        if (l.QNorm is not null)
        {
            Add(TensorRoles.QNorm, l.QNorm, l.QNorm.Length);
            Add(TensorRoles.KNorm, l.KNorm!, l.KNorm!.Length);
        }
        switch (t.Ffn)
        {
            case BlockFfn.Dense:
                Add(TensorRoles.Gate, l.Gate, inter, h);
                Add(TensorRoles.Up, l.Up, inter, h);
                Add(TensorRoles.Down, l.Down, h, inter);
                break;
            case BlockFfn.SharedExpert:
            case BlockFfn.Experts:
            {
                var moe = t.ExpertNames!;
                int experts = t.ExpertCount;
                Add(TensorRoles.Router, new float[experts * h], experts, h);
                bool shared = t.Ffn == BlockFfn.SharedExpert;
                int expertInter = shared ? t.ExpertIntermediate : inter;
                for (int e = 0; e < experts; e++)
                {
                    tensors.Add((moe.Name(e, moe.Gate!), shared ? new float[expertInter * h] : l.Gate, new long[] { expertInter, h }));
                    tensors.Add((moe.Name(e, moe.Up), shared ? new float[expertInter * h] : l.Up, new long[] { expertInter, h }));
                    tensors.Add((moe.Name(e, moe.Down), shared ? new float[h * expertInter] : l.Down, new long[] { h, expertInter }));
                }
                if (shared)
                {
                    Add(TensorRoles.SharedGate, l.Gate, inter, h);
                    Add(TensorRoles.SharedUp, l.Up, inter, h);
                    Add(TensorRoles.SharedDown, l.Down, h, inter);
                    if (l.SharedExpertGate is not null)
                    {
                        Add(TensorRoles.SharedExpertGate, l.SharedExpertGate, 1, h);
                    }
                }
                break;
            }
        }

        // Fused spellings: concatenate the parts along the first axis.
        foreach (var fused in t.Fused ?? Array.Empty<FusedTensor>())
        {
            var parts = fused.Parts.Select(part => tensors.FirstOrDefault(x => x.Name == part)).ToList();
            if (parts.Any(x => x.Name is null))
            {
                continue;
            }
            var data = parts.SelectMany(x => x.Data).ToArray();
            long rows = parts.Sum(x => x.Shape[0]);
            tensors.RemoveAll(x => fused.Parts.Contains(x.Name));
            tensors.Add((fused.Name, data, parts[0].Shape.Length == 1 ? new[] { rows } : new[] { rows, parts[0].Shape[1] }));
        }
        foreach (var (name, data, shape) in tensors)
        {
            yield return F32($"{prefix}.{name}", data, shape);
        }
    }

    private static TensorPayload F32(string name, float[] data, params long[] shape)
    {
        var bytes = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        return new TensorPayload { Name = name, Dtype = Dtype.F32, Shape = shape, Data = bytes };
    }

    private static IEnumerable<(double, int, bool)> ClearsFor(Allocation a, int layer) =>
        a.Clears.Where(c => c.Layer == layer).Select(c => (EntailCompiler.G, c.Lane, false));

    // ── target 1: a new, empty model ────────────────────────────────────────

    /// <summary>
    /// Writes a complete checkpoint holding nothing but the program: a
    /// byte-level tokenizer over printable ASCII, the compiled layers, and an
    /// output head with one row per <c>emit</c>. No weight is initialised
    /// randomly and none is trained.
    /// </summary>
    public static EntailBuildReport WriteFresh(EntailProgram p, string outDir, int context = 8192)
    {
        RefuseVotes(p);
        RefuseCarries(p);
        if (p.Masks.Count > 0)
        {
            throw new EntailException("`only` and `forbid` narrow a host's predictions: graft this program instead");
        }
        // Vocabulary: two control tokens, then printable ASCII.
        var vocab = new List<string> { "<none>", "<eos>" };
        for (char c = ' '; c <= '~'; c++)
        {
            vocab.Add(c.ToString());
        }
        var ids = vocab.Select((t, i) => (t, i)).ToDictionary(x => x.t, x => x.i, StringComparer.Ordinal);
        var missing = p.Tokens.Keys.Where(t => !ids.ContainsKey(t)).ToList();
        var inputs = p.Tokens.Keys.Where(ids.ContainsKey).Select(t => (ids[t], t)).ToList();
        var outputs = p.Emits.Select(e => (e.Text is null ? 1 : ids[e.Text], e.Lane, 64f)).ToList();
        return WriteBlank(p, outDir, context, vocab, inputs, outputs, missing);
    }

    /// <summary>
    /// Writes a program as an adapter to run beside a partner model: the same
    /// blank model as <see cref="WriteFresh"/>, reading the partner's token ids
    /// through a table, and with one output row per output lane. Its logits are
    /// therefore its verdict, and all zero where it has nothing to say.
    /// </summary>
    public static (EntailBuildReport Report, EntailAdapterManifest Manifest) WriteAdapter(
        EntailProgram p, string partnerTokenizerDir, string outDir, int context = 8192)
    {
        var binding = BindAdapter(p, partnerTokenizerDir);
        int size = Math.Max(binding.Manifest.Rows.Count + 2, binding.Manifest.Outputs.Count);
        var vocab = new List<string> { "<none>", "<eos>" };
        vocab.AddRange(Enumerable.Range(2, size - 2).Select(i => $"r{i}"));
        var outputs = binding.Outputs.Select(o => (o.Row, o.Lane, 1f)).ToList();
        var report = WriteBlank(p, outDir, context, vocab, binding.Inputs, outputs, binding.Missing);
        binding.Manifest.Save(outDir);
        return (report, binding.Manifest);
    }

    /// <summary>Joins a program to a partner's tokens: which token each input
    /// row is, and what each output row means.</summary>
    public static EntailAdapterBinding BindAdapter(EntailProgram p, string partnerTokenizerDir)
    {
        RefuseVotes(p);
        var (ids, missing) = BindTokens(p, partnerTokenizerDir);

        // Row 0 stands for every token the program does not know, row 1 is unused.
        var rows = new Dictionary<int, int>();
        var inputs = new List<(int Row, string Text)>();
        foreach (var text in p.Tokens.Keys.Where(ids.ContainsKey))
        {
            int row = rows.Count + 2;
            rows[ids[text]] = row;
            inputs.Add((row, text));
        }

        var bound = p.Emits.Where(e => e.Text is not null && ids.ContainsKey(e.Text)).ToList();
        var said = bound.Where(e => e.Text != " ").Select(e => ids[e.Text!]).Distinct().ToArray();
        var entries = new List<EntailAdapterOutput>();
        var outputs = new List<(int Row, int Lane)>();
        void Add(EntailOutputKind kind, int lane, int[] tokens)
        {
            outputs.Add((entries.Count, lane));
            entries.Add(new EntailAdapterOutput(entries.Count, kind, tokens));
        }
        foreach (var (text, lane) in p.Emits)
        {
            if (text is null)
            {
                Add(EntailOutputKind.End, lane, said);
            }
            else if (ids.TryGetValue(text, out int id))
            {
                Add(EntailOutputKind.Say, lane, new[] { id });
            }
        }
        foreach (var (allow, texts, lane) in p.Masks)
        {
            Add(allow ? EntailOutputKind.Only : EntailOutputKind.Forbid, lane,
                texts.Where(ids.ContainsKey).Select(t => ids[t]).Distinct().ToArray());
        }

        var manifest = new EntailAdapterManifest(p.Name,
            EntailAdapterManifest.TokenizerHash(partnerTokenizerDir), rows, entries);
        return new EntailAdapterBinding(manifest, inputs, outputs, missing);
    }

    private static EntailBuildReport WriteBlank(EntailProgram p, string outDir, int context, IReadOnlyList<string> vocab,
        IReadOnlyList<(int Row, string Text)> inputs, IReadOnlyList<(int Row, int Lane, float Weight)> outputs, List<string> missing)
    {
        Directory.CreateDirectory(outDir);
        int hidden = WidthFor(Allocate(p, firstDim: 0, keepEmitsLive: true).Used);
        var (ballastDims, shift) = BallastShape(hidden);
        var alloc = Allocate(p, firstDim: ballastDims, keepEmitsLive: true);
        int clearPeak = Enumerable.Range(0, p.Layers).Max(l =>
            p.Ops.OfType<Neuron>().Count(n => n.Layer == l) + alloc.Clears.Count(c => c.Layer == l));
        int intermediate = RoundUp(clearPeak, 64);
        var scale = p.Lanes.Select(l => l.Scale).ToArray();
        var target = EntailTarget.Fresh(intermediate);
        var look = ChooseLookup(target, p, context, strict: false);
        var geometry = new Geometry
        {
            Hidden = hidden, Intermediate = intermediate, Ballast = 0, BallastDims = ballastDims, Shift = shift, Lanes = alloc,
            Scale = scale, HostDim = new bool[hidden], Target = target, Look = look,
        };

        var embed = new float[(long)vocab.Count * hidden];
        for (int t = 0; t < vocab.Count; t++)
        {
            for (int d = 0; d < ballastDims; d++)
            {
                embed[(long)t * hidden + d] = (float)Ballast;
            }
        }
        foreach (var (row, text) in inputs)
        {
            foreach (var (lane, value) in p.Tokens[text])
            {
                if (alloc.Dim[lane] >= 0)
                {
                    embed[(long)row * hidden + alloc.Dim[lane]] = (float)(value * scale[lane]);
                }
            }
        }

        var head = new float[(long)vocab.Count * hidden];
        foreach (var (row, lane, weight) in outputs)
        {
            if (alloc.Dim[lane] >= 0)
            {
                head[(long)row * hidden + alloc.Dim[lane]] = weight;
            }
        }

        var tensors = new List<TensorPayload> { F32("model.embed_tokens.weight", embed, vocab.Count, hidden) };
        for (int l = 0; l < p.Layers; l++)
        {
            var layer = BuildLayer(p, geometry, l, ClearsFor(alloc, l));
            tensors.AddRange(LayerTensors(layer, target, $"model.layers.{l}", hidden, intermediate));
        }
        tensors.Add(F32("model.norm.weight", NormRow(geometry, geometry.LaneGamma), hidden));
        tensors.Add(F32("lm_head.weight", head, vocab.Count, hidden));
        SafetensorsWriter.Write(Path.Combine(outDir, "model.safetensors"), tensors);

        var config = new JsonObject
        {
            ["architectures"] = new JsonArray("Qwen3_5ForConditionalGeneration"),
            ["model_type"] = "qwen3_5_text",
            ["hidden_size"] = hidden,
            ["num_hidden_layers"] = p.Layers,
            ["num_attention_heads"] = EntailProgram.HeadsPerLayer,
            ["num_key_value_heads"] = EntailProgram.KvGroups,
            ["head_dim"] = EntailProgram.HeadDim,
            ["intermediate_size"] = intermediate,
            ["hidden_act"] = "silu",
            ["rms_norm_eps"] = 1e-6,
            ["vocab_size"] = vocab.Count,
            ["max_position_embeddings"] = context,
            ["layer_types"] = new JsonArray(Enumerable.Repeat("full_attention", p.Layers).Select(x => (JsonNode?)x).ToArray()),
            ["attn_output_gate"] = true,
            ["tie_word_embeddings"] = false,
            ["eos_token_id"] = 1,
            ["rope_parameters"] = new JsonObject
            {
                ["rope_type"] = "default",
                ["rope_theta"] = EntailTarget.FreshTheta,
                ["partial_rotary_factor"] = 0.25,
            },
        };
        File.WriteAllText(Path.Combine(outDir, "config.json"), config.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        File.WriteAllText(Path.Combine(outDir, "tokenizer.json"), TokenizerJson(vocab));
        File.WriteAllText(Path.Combine(outDir, "generation_config.json"), "{ \"eos_token_id\": 1 }");

        return new EntailBuildReport(outDir, hidden, p.Layers, p.Layers, alloc.Used, alloc.Peak,
            p.Ops.Count(o => o is Neuron), p.Ops.Count(o => o is Head), look.Gap, inputs.Count, missing);
    }

    private static string TokenizerJson(IReadOnlyList<string> vocab)
    {
        var map = new JsonObject();
        for (int i = 0; i < vocab.Count; i++)
        {
            map[vocab[i] == " " ? "Ġ" : vocab[i]] = i;
        }
        var root = new JsonObject
        {
            ["version"] = "1.0.0",
            ["truncation"] = null,
            ["padding"] = null,
            ["added_tokens"] = new JsonArray(
                new JsonObject { ["id"] = 0, ["content"] = "<none>", ["single_word"] = false, ["lstrip"] = false, ["rstrip"] = false, ["normalized"] = false, ["special"] = true },
                new JsonObject { ["id"] = 1, ["content"] = "<eos>", ["single_word"] = false, ["lstrip"] = false, ["rstrip"] = false, ["normalized"] = false, ["special"] = true }),
            ["normalizer"] = null,
            ["pre_tokenizer"] = new JsonObject
            {
                ["type"] = "Sequence",
                ["pretokenizers"] = new JsonArray(
                    new JsonObject
                    {
                        ["type"] = "Split",
                        ["pattern"] = new JsonObject { ["Regex"] = "[\\s\\S]" },
                        ["behavior"] = "Isolated",
                        ["invert"] = false,
                    },
                    new JsonObject { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["trim_offsets"] = false, ["use_regex"] = false }),
            },
            ["post_processor"] = null,
            ["decoder"] = new JsonObject { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["trim_offsets"] = false, ["use_regex"] = false },
            ["model"] = new JsonObject
            {
                ["type"] = "BPE",
                ["dropout"] = null,
                ["unk_token"] = null,
                ["continuing_subword_prefix"] = null,
                ["end_of_word_suffix"] = null,
                ["fuse_unk"] = false,
                ["byte_fallback"] = false,
                ["ignore_merges"] = false,
                ["vocab"] = map,
                ["merges"] = new JsonArray(),
            },
        };
        return root.ToJsonString();
    }

    private static void RefuseCarries(EntailProgram p)
    {
        if (p.Carries.Count > 0)
        {
            throw new EntailException(
                "this program carries lanes from token to token (`carry`), which a fixed-depth transformer cannot do: " +
                "run it with `entail run`, or beside a model as a sparse adapter (`entail adapter`, without --dense)");
        }
    }

    private static void RefuseVotes(EntailProgram p)
    {
        if (p.Votes.Count > 0)
        {
            throw new EntailException("this program still has soft votes: refine them into `emit` before compiling it");
        }
    }

    private static int RoundUp(int value, int unit) => (value + unit - 1) / unit * unit;

    // ── target 2: grafted into a host checkpoint, in situ ───────────────────

    /// <summary>Raw height of a set output lane: small enough that the host's
    /// final norm cannot feel it.</summary>
    private const double EmitLaneHeight = 1.0 / 1024;
    /// <summary>The token's own entry in its output lane. With a tied head this
    /// column is both an input and an output, so the block clears it on the
    /// way in.</summary>
    private const double EmitRow = 1.0 / 1048576;
    /// <summary>Residual norm a host reaches at its final norm (measured on
    /// Qwen 3.5 0.8B), and how many logits a set output lane adds at that norm.</summary>
    public const double DefaultHostOutputNorm = 115;
    private const double OutputForce = 256;

    /// <summary>
    /// Restructures a host checkpoint around the program, leaving every host
    /// weight's value in place. The host is judged by the importer's rules
    /// (<see cref="EntailHost.Judge"/>) and the block is written in its own
    /// layer shape and spelling:
    /// <list type="bullet">
    /// <item>The hidden size widens; the new dimensions are the program's
    /// lanes. Every host matrix gains zeros there, by the role of each axis,
    /// so no host weight can read or write a lane, and its norm weights are
    /// rescaled so the wider mean leaves them exactly where they were.</item>
    /// <item>The compiled layers go in front of the host's first layer. They
    /// read only lanes, so they compute a pure function of the tokens. They
    /// finish by clearing every lane and the ballast: the host then starts
    /// from its own unchanged embedding.</item>
    /// <item>Each output lane is a column of the output head, non-zero in its
    /// own token's row alone, and the only lanes the final norm passes. A set
    /// lane therefore raises exactly one logit; the end-of-answer lane lowers
    /// the logits of the tokens an answer is made of.</item>
    /// </list>
    /// Wherever the program says nothing, the host's computation is unchanged.
    /// </summary>
    public static EntailBuildReport WriteGraft(EntailProgram p, string hostDir, string outDir, int context = 8192,
        double hostOutputNorm = DefaultHostOutputNorm) =>
        WriteGraft(p, EntailHost.Judge(hostDir), outDir, context, hostOutputNorm);

    public static EntailBuildReport WriteGraft(EntailProgram p, EntailHost hostFacts, string outDir, int context = 8192,
        double hostOutputNorm = DefaultHostOutputNorm)
    {
        RefuseVotes(p);
        RefuseCarries(p);
        var t = hostFacts.Target;
        if (p.Ops.OfType<Head>().GroupBy(h => h.Layer).Any(g => g.Count() > t.ProgramHeads))
        {
            throw new EntailException($"the host's layers hold {t.ProgramHeads} program heads: compile the program for this host");
        }
        if (t.Window is { } window && window < context)
        {
            // Every layer is windowed (Mistral): a lookup reaches the window, no further.
            context = (int)window;
        }
        Directory.CreateDirectory(outDir);
        string hostDir = hostFacts.Directory;
        int hostHidden = hostFacts.Hidden;
        string root = hostFacts.Prefix;

        int hidden = WidthFor(hostHidden + Allocate(p, firstDim: 0, keepEmitsLive: true).Used, t.EmbedScaleSqrtHidden);
        var (ballastDims, shift) = BallastShape(hidden);
        var alloc = Allocate(p, firstDim: hostHidden + ballastDims, keepEmitsLive: true);
        int ballast = hostHidden;
        var scale = p.Lanes.Select(l => l.Emits ? EmitLaneHeight : l.Scale).ToArray();
        var hostDim = new bool[hidden];
        for (int d = 0; d < hostHidden; d++)
        {
            hostDim[d] = true;
        }
        var look = ChooseLookup(t, p, context, strict: true);
        int inter = t.Intermediate;
        var geometry = new Geometry
        {
            Hidden = hidden, Intermediate = inter, Ballast = ballast, BallastDims = ballastDims, Shift = shift, Lanes = alloc,
            Scale = scale, HostDim = hostDim, Target = t, Look = look,
        };
        // What the embedding is multiplied by, before and after widening. The
        // block's entries are divided by the new one (a power of two); the
        // host's are moved so the product is unchanged.
        double embedOld = t.EmbedScaleSqrtHidden ? Math.Sqrt(hostHidden) : t.EmbedMultiplier;
        double embedNew = t.EmbedScaleSqrtHidden ? Math.Sqrt(hidden) : t.EmbedMultiplier;
        double hostEmbedFactor = embedOld / embedNew;

        // Everything below streams: the host is read through its memory map
        // and each output tensor goes to disk as it is made, so building a
        // graft needs one compiled layer and a few row blocks in memory, not
        // the whole widened model. A host larger than RAM grafts the same way.
        using var host = ModelDirectory.OpenCheckpoint(hostDir);
        string embedName = hostFacts.EmbeddingName;
        var hostNames = host.TensorNames
            .Where(n => (n.StartsWith(root, StringComparison.Ordinal) || n == hostFacts.HeadName) && n != embedName)
            .Distinct(StringComparer.Ordinal)
            .ToList();
        int block = p.Layers + 2;

        var (vocabIds, missing) = BindTokens(p, hostDir);
        var (embedShard, embedInfo) = host.Get(embedName);
        long vocab = embedInfo.Shape[0];

        // Input writes go to the embedding; output writes to the output head,
        // which is the embedding when tied. A later write wins.
        var inputWrites = new Dictionary<long, List<(int Dim, double Value)>>();
        var outputWrites = new Dictionary<long, List<(int Dim, double Value)>>();
        void Put(Dictionary<long, List<(int, double)>> writes, long id, int dim, double value)
        {
            if (!writes.TryGetValue(id, out var list))
            {
                writes[id] = list = new List<(int, double)>();
            }
            list.Add((dim, value));
        }
        foreach (var (textToken, id) in vocabIds)
        {
            // A token that is only ever said has no input facts.
            if (!p.Tokens.TryGetValue(textToken, out var writes))
            {
                continue;
            }
            foreach (var (lane, value) in writes)
            {
                if (alloc.Dim[lane] >= 0)
                {
                    Put(inputWrites, id, alloc.Dim[lane], value * scale[lane] / embedNew);
                }
            }
        }
        var outputs = p.Emits.Where(e => e.Text is not null && vocabIds.ContainsKey(e.Text)).ToList();
        foreach (var (token, lane) in outputs)
        {
            Put(outputWrites, vocabIds[token!], alloc.Dim[lane], EmitRow);
        }
        // "End of answer" has no token in a host, whose text simply goes on.
        // Its lane is a column too, negative in the rows of the tokens an
        // answer is made of: the number stops, and the host says what follows.
        var ends = p.Emits.Where(e => e.Text is null).Select(e => e.Lane).ToList();
        foreach (int lane in ends)
        {
            foreach (var (token, _) in outputs.Where(o => o.Text != " "))
            {
                Put(outputWrites, vocabIds[token!], alloc.Dim[lane], -EmitRow);
            }
        }
        // A mask's lane is a column too, negative in the rows it rules out.
        // It can cover most of the vocabulary, so it is a rule per row.
        var masks = p.Masks
            .Select(m => (m.Allow, Listed: m.Texts.Where(vocabIds.ContainsKey).Select(x => (long)vocabIds[x]).ToHashSet(),
                Dim: alloc.Dim[m.Lane]))
            .ToList();
        var outputLanes = outputs.Select(o => o.Lane).Concat(ends).Concat(p.Masks.Select(m => m.Lane)).ToList();

        long reserve = 4096 + 256L * (hostNames.Count + 1 + block * (16 + 3 * t.ExpertCount));
        using var writer = new SafetensorsStreamWriter(Path.Combine(outDir, "model.safetensors"), reserve);

        // Embedding: host dims (moved, if the embedding scale changes), then
        // ballast and token lanes; the output columns too, when tied.
        WriteRows(writer, host, embedName, embedName, hostHidden, hidden, hostEmbedFactor, (row, put) =>
        {
            for (int d = 0; d < ballastDims; d++)
            {
                put(ballast + d, Ballast / embedNew);
            }
            if (inputWrites.TryGetValue(row, out var list))
            {
                foreach (var (dim, value) in list)
                {
                    put(dim, value);
                }
            }
            if (hostFacts.Tied)
            {
                WriteOutputs(row, put);
            }
        });
        void WriteOutputs(long row, Action<int, double> put)
        {
            if (outputWrites.TryGetValue(row, out var list))
            {
                foreach (var (dim, value) in list)
                {
                    put(dim, value);
                }
            }
            foreach (var (allow, listed, dim) in masks)
            {
                if (listed.Contains(row) != allow)
                {
                    put(dim, -EmitRow);
                }
            }
        }

        // The compiled block, then the two layers that hand over to the host,
        // each written and dropped before the next is built. With a tied head
        // the output columns are also input entries, cleared on the way in.
        for (int l = 0; l < p.Layers; l++)
        {
            var clears = ClearsFor(alloc, l);
            if (l == 0 && hostFacts.Tied)
            {
                clears = clears.Concat(outputLanes.Select(lane => (EntailCompiler.G, lane, false)));
            }
            WriteLayer(writer, BuildLayer(p, geometry, l, clears), t, $"{root}layers.{l}", hidden, inter);
        }
        var finalClears = alloc.LiveAtEnd.Select(lane => (EntailCompiler.G, lane, false));
        WriteLayer(writer, BuildLayer(p, geometry, -1, finalClears), t, $"{root}layers.{p.Layers}", hidden, inter);
        WriteLayer(writer, BuildLayer(p, geometry, -1, new[] { (EntailCompiler.G, 0, true) }), t,
            $"{root}layers.{p.Layers + 1}", hidden, inter);

        // The host's own tensors, widened by the role of each axis.
        double shrink = Math.Sqrt(hostHidden / (double)hidden);
        double headFactor = hostFacts.Tied ? hostEmbedFactor : 1;
        var axes = HostAxes(hostFacts);
        foreach (var name in hostNames)
        {
            string target = name;
            string? relative = null;
            if (name.StartsWith($"{root}layers.", StringComparison.Ordinal))
            {
                string rest = name[$"{root}layers.".Length..];
                int dot = rest.IndexOf('.');
                int index = int.Parse(rest[..dot]);
                relative = rest[(dot + 1)..];
                target = $"{root}layers.{index + block}.{relative}";
            }
            if (name == hostFacts.FinalNormName)
            {
                // A tied head reads the moved embedding: the host's logits move back here.
                var widened = WidenNorm(host, name, hostHidden, hidden, shrink / headFactor, t.NormOffset, isFinalNorm: true);
                // logit += EmitRow · gamma · lane/rms: OutputForce at the host's usual norm.
                double gamma = OutputForce * hostOutputNorm / (EmitRow * EmitLaneHeight * Math.Sqrt(hidden) * t.OutputMultiplier);
                foreach (int lane in outputLanes)
                {
                    widened[alloc.Dim[lane]] = (float)(gamma - t.NormOffset);
                }
                writer.WriteF32(target, widened, hidden);
                continue;
            }
            if (name == hostFacts.HeadName)
            {
                WriteRows(writer, host, name, name, hostHidden, hidden, 1, (row, put) => WriteOutputs(row, put));
                continue;
            }
            var axis = relative is not null && axes.TryGetValue(relative, out var known) ? known : AxisByShape(host, name, hostHidden);
            Widen(writer, host, name, target, hostHidden, hidden, shrink, axis, t.NormOffset);
        }

        writer.Complete();

        WriteGraftConfig(hostFacts, outDir, hidden, block);
        foreach (var file in new[] { "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
                     "special_tokens_map.json", "tokenizer.model", "vocab.json", "merges.txt" })
        {
            var source = Path.Combine(hostDir, file);
            if (File.Exists(source))
            {
                File.Copy(source, Path.Combine(outDir, file), overwrite: true);
            }
        }

        return new EntailBuildReport(outDir, hidden, hostFacts.Layers + block, block, alloc.Used, alloc.Peak,
            p.Ops.Count(o => o is Neuron), p.Ops.Count(o => o is Head), look.Gap,
            vocabIds.Count, missing);
    }

    private static void WriteLayer(SafetensorsStreamWriter writer, Layer layer, EntailTarget t, string prefix, int hidden, int inter)
    {
        foreach (var tensor in LayerTensors(layer, t, prefix, hidden, inter))
        {
            writer.Write(tensor);
        }
    }

    /// <summary>How a host tensor meets the residual stream.</summary>
    private enum Axis
    {
        /// <summary>Touches no residual axis: copied.</summary>
        None,
        /// <summary>A norm over the stream: rescaled, zero on new dims.</summary>
        Norm,
        /// <summary>Reads the stream (columns): new columns are zero.</summary>
        Columns,
        /// <summary>Writes the stream (rows, or a 1-D bias): new rows are zero.</summary>
        Rows,
    }

    /// <summary>Each host layer tensor's residual axis, by its role — a
    /// square projection (Llama's o_proj when heads × head_dim = hidden) has
    /// two axes of the stream's width, and only its role says which is which.</summary>
    private static Dictionary<string, Axis> HostAxes(EntailHost host)
    {
        var axes = new Dictionary<string, Axis>(StringComparer.Ordinal);
        var roles = host.Roles;
        void Set(string role, Axis weight, Axis bias)
        {
            if (roles.Name(role) is { } w)
            {
                axes[w] = weight;
            }
            if (roles.Name(role, "bias") is { } b)
            {
                axes[b] = bias;
            }
        }
        foreach (var norm in new[] { TensorRoles.AttnIn, TensorRoles.AttnOut, TensorRoles.FfnIn, TensorRoles.FfnOut })
        {
            Set(norm, Axis.Norm, Axis.Rows);
        }
        foreach (var reads in new[] { TensorRoles.Q, TensorRoles.K, TensorRoles.V, TensorRoles.Gate, TensorRoles.Up, TensorRoles.Router,
                     TensorRoles.SharedGate, TensorRoles.SharedUp, TensorRoles.SharedExpertGate })
        {
            Set(reads, Axis.Columns, Axis.None);
        }
        foreach (var writes in new[] { TensorRoles.O, TensorRoles.Down, TensorRoles.SharedDown })
        {
            Set(writes, Axis.Rows, Axis.Rows);
        }
        Set(TensorRoles.QNorm, Axis.None, Axis.None);
        Set(TensorRoles.KNorm, Axis.None, Axis.None);
        foreach (var fused in roles.Fused ?? new List<FusedTensor>())
        {
            // Every known fusion is of input projections (qkv, gate_up).
            axes[fused.Name] = fused.Name.EndsWith(".bias", StringComparison.Ordinal) ? Axis.None : Axis.Columns;
        }
        if (roles.Experts is { } experts)
        {
            for (int e = 0; e < host.Target.ExpertCount; e++)
            {
                if (experts.Gate is { } g)
                {
                    axes[experts.Name(e, g)] = Axis.Columns;
                }
                axes[experts.Name(e, experts.Up)] = Axis.Columns;
                axes[experts.Name(e, experts.Down)] = Axis.Rows;
            }
        }
        return axes;
    }

    /// <summary>A tensor no role names: its axis of the stream's width, if
    /// exactly one of its last two is.</summary>
    private static Axis AxisByShape(ModelDirectory host, string name, int hostHidden)
    {
        var shape = host.Get(name).Info.Shape;
        if (shape.Length == 1)
        {
            return shape[0] != hostHidden ? Axis.None
                : name.Contains("norm", StringComparison.Ordinal) ? Axis.Norm : Axis.Rows;
        }
        bool rows = shape[^2] == hostHidden, columns = shape[^1] == hostHidden;
        if (rows && columns)
        {
            throw new EntailException($"cannot graft into this host: '{name}' is {hostHidden}×{hostHidden} and no role says which side reads the residual stream");
        }
        return columns ? Axis.Columns : rows ? Axis.Rows : Axis.None;
    }

    /// <summary>A norm's weight over the wider stream: (offset+w)·shrink keeps
    /// the host's normalised values where they were under a mean over more
    /// dimensions. The final norm additionally zeroes the new dims, so lanes
    /// never reach the output head.</summary>
    private static float[] WidenNorm(ModelDirectory host, string name, int hostHidden, int hidden, double shrink,
        float offset, bool isFinalNorm)
    {
        var (file, info) = host.Get(name);
        var widened = BitPattern.WidenToF32(info.Dtype, file.ReadBytes(info));
        var w = new float[hidden];
        for (int d = 0; d < hidden; d++)
        {
            w[d] = d < hostHidden ? (float)((offset + (double)widened[d]) * shrink - offset) : isFinalNorm ? -offset : 0f;
        }
        return w;
    }

    /// <summary>Writes a host tensor with zeros where the stream widened,
    /// streaming it from the host a block of rows at a time.</summary>
    private static void Widen(SafetensorsStreamWriter writer, ModelDirectory host, string name, string target,
        int hostHidden, int hidden, double shrink, Axis axis, float offset)
    {
        var (file, info) = host.Get(name);
        const long BlockBytes = 32L << 20;
        int element = (int)(info.DataLength / Math.Max(1, info.ElementCount));
        switch (axis)
        {
            case Axis.Norm:
                writer.WriteF32(target, WidenNorm(host, name, hostHidden, hidden, shrink, offset, isFinalNorm: false), hidden);
                return;
            case Axis.Columns:
            {
                // [.., rows, hostHidden] → [.., rows, hidden]
                long rows = info.ElementCount / hostHidden;
                var shape = (long[])info.Shape.Clone();
                shape[^1] = hidden;
                writer.Write(target, info.Dtype, shape, stream =>
                {
                    long rowsPerBlock = Math.Max(1, BlockBytes / ((long)hidden * element));
                    for (long first = 0; first < rows; first += rowsPerBlock)
                    {
                        int count = (int)Math.Min(rowsPerBlock, rows - first);
                        var source = file.ReadBytes(info, first * hostHidden * element, count * hostHidden * element);
                        var data = new byte[(long)count * hidden * element];
                        for (int r = 0; r < count; r++)
                        {
                            Buffer.BlockCopy(source, r * hostHidden * element, data, r * hidden * element, hostHidden * element);
                        }
                        stream.Write(data);
                    }
                });
                return;
            }
            case Axis.Rows:
            {
                // [.., hostHidden, cols] → [.., hidden, cols]; a 1-D bias is cols = 1.
                var shape = (long[])info.Shape.Clone();
                int axisIndex = shape.Length == 1 ? 0 : shape.Length - 2;
                long cols = shape.Length == 1 ? 1 : shape[^1];
                long batches = info.ElementCount / (hostHidden * cols);
                shape[axisIndex] = hidden;
                writer.Write(target, info.Dtype, shape, stream =>
                {
                    long batchBytes = hostHidden * cols * element;
                    long zeros = (hidden - hostHidden) * cols * element;
                    var blank = new byte[(int)Math.Min(zeros, BlockBytes)];
                    for (long b = 0; b < batches; b++)
                    {
                        for (long done = 0; done < batchBytes; done += BlockBytes)
                        {
                            stream.Write(file.ReadBytes(info, b * batchBytes + done, (int)Math.Min(BlockBytes, batchBytes - done)));
                        }
                        for (long done = 0; done < zeros; done += blank.Length)
                        {
                            stream.Write(blank, 0, (int)Math.Min(blank.Length, zeros - done));
                        }
                    }
                });
                return;
            }
            default:
                writer.Write(target, info.Dtype, info.Shape, stream => host.CopyTo(name, stream));
                return;
        }
    }

    /// <summary>Writes a [vocab, hostHidden] host matrix at [vocab, hidden]:
    /// the host's entries times <paramref name="hostFactor"/>, then whatever
    /// <paramref name="extra"/> puts in each row. It stays in the host's dtype
    /// when every value is exact there (bf16 and f32 hold the block's powers of
    /// two and small integers), else it is written in f32.</summary>
    private static void WriteRows(SafetensorsStreamWriter writer, ModelDirectory host, string name, string target,
        int hostHidden, int hidden, double hostFactor, Action<long, Action<int, double>> extra)
    {
        var (file, info) = host.Get(name);
        long vocab = info.Shape[0];
        bool keep = hostFactor == 1 && info.Dtype is Dtype.BF16 or Dtype.F32;
        var dtype = keep ? info.Dtype : Dtype.F32;
        int inElement = (int)(info.DataLength / Math.Max(1, info.ElementCount));
        int outElement = dtype == Dtype.BF16 ? 2 : 4;
        writer.Write(target, dtype, new[] { vocab, (long)hidden }, stream =>
        {
            long rowsPerBlock = Math.Max(1, (32L << 20) / ((long)hidden * outElement));
            for (long first = 0; first < vocab; first += rowsPerBlock)
            {
                int rows = (int)Math.Min(rowsPerBlock, vocab - first);
                var source = file.ReadBytes(info, first * hostHidden * inElement, rows * hostHidden * inElement);
                var rowsOut = new byte[(long)rows * hidden * outElement];
                float[]? wide = keep ? null : BitPattern.WidenToF32(info.Dtype, source);
                for (int r = 0; r < rows; r++)
                {
                    long at = (long)r * hidden;
                    if (keep)
                    {
                        Buffer.BlockCopy(source, r * hostHidden * inElement, rowsOut, (int)(at * outElement), hostHidden * inElement);
                    }
                    else
                    {
                        for (int d = 0; d < hostHidden; d++)
                        {
                            WriteValue(rowsOut, at + d, dtype, wide![(long)r * hostHidden + d] * hostFactor);
                        }
                    }
                    extra(first + r, (dim, value) => WriteValue(rowsOut, at + dim, dtype, value));
                }
                stream.Write(rowsOut);
            }
        });
    }

    private static void WriteValue(byte[] data, long element, Dtype dtype, double value)
    {
        if (dtype == Dtype.BF16)
        {
            ushort bits = Bf16(value);
            data[element * 2] = (byte)bits;
            data[element * 2 + 1] = (byte)(bits >> 8);
        }
        else
        {
            BitConverter.TryWriteBytes(data.AsSpan((int)(element * 4), 4), (float)value);
        }
    }

    private static ushort Bf16(double value)
    {
        uint bits = BitConverter.SingleToUInt32Bits((float)value);
        return (ushort)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
    }

    /// <summary>
    /// The host's config with the block's layers in front: the widened
    /// hidden size and layer count, and every per-layer fact shifted —
    /// <c>layer_types</c> (written out when the host gives only a period),
    /// <c>no_rope_layers</c>, Qwen2's <c>max_window_layers</c> and Qwen-MoE's
    /// <c>mlp_only_layers</c>. Any other per-layer table refuses: the build
    /// cannot tell what a block layer's entry should be.
    /// </summary>
    private static void WriteGraftConfig(EntailHost host, string outDir, int hidden, int block)
    {
        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(host.Directory, "config.json")))!.AsObject();
        var text = config["text_config"] as JsonObject ?? config;
        var t = host.Target;
        int layers = host.Layers;
        text["hidden_size"] = hidden;
        text["num_hidden_layers"] = layers + block;
        // A config that leaves head_dim to hidden_size / heads would derive a
        // new one from the wider stream: state the host's.
        text["head_dim"] ??= t.HeadDim;

        foreach (var key in new[] { "no_rope_layer_interval", "interleave_moe_layer_step", "first_k_dense_replace", "global_attn_every_n_layers" })
        {
            // An explicit no_rope_layers list wins over its interval (SmolLM3).
            if (text.ContainsKey(key) && !(key == "no_rope_layer_interval" && text["no_rope_layers"] is JsonArray))
            {
                throw new EntailException($"cannot graft into this host: config key '{key}' sets layers by position, and the block's layers would shift it");
            }
        }
        if (text["decoder_sparse_step"] is JsonValue step && step.GetValue<int>() != 1)
        {
            throw new EntailException("cannot graft into this host: decoder_sparse_step routes layers by position, and the block's layers would shift it");
        }

        var blockTypes = Enumerable.Range(0, block).Select(_ => (JsonNode?)JsonValue.Create(t.LayerType)).ToList();
        if (text["layer_types"] is JsonArray declared)
        {
            text["layer_types"] = new JsonArray(blockTypes.Concat(declared.Select(x => x?.DeepClone())).ToArray());
        }
        else if (host.Policies.Select(PolicyType).Distinct().Count() > 1)
        {
            // A period (sliding_window_pattern) would shift: spell each layer out.
            text["layer_types"] = new JsonArray(blockTypes.Concat(host.Policies.Select(p => (JsonNode?)PolicyType(p))).ToArray());
        }
        if (text["no_rope_layers"] is JsonArray noRope)
        {
            text["no_rope_layers"] = new JsonArray(Enumerable.Range(0, block).Select(_ => (JsonNode?)JsonValue.Create(1)).Concat(noRope.Select(x => x?.DeepClone())).ToArray());
        }
        if (text["use_sliding_window"] is JsonValue sliding && sliding.GetValue<bool>() && text["max_window_layers"] is JsonValue mwl)
        {
            text["max_window_layers"] = mwl.GetValue<int>() + block;
        }
        if (text["mlp_only_layers"] is JsonArray dense)
        {
            var shifted = dense.Select(x => x!.GetValue<int>() + block);
            // Only an MoE stack reads it: there the block's dense layers are listed.
            var blockDense = t.Ffn == BlockFfn.Dense && t.ExpertCount > 0 ? Enumerable.Range(0, block) : Enumerable.Empty<int>();
            text["mlp_only_layers"] = new JsonArray(blockDense.Concat(shifted).Select(x => (JsonNode?)x).ToArray());
        }
        foreach (var (key, value) in text.ToList())
        {
            if (value is JsonArray array && array.Count == layers && key is not "layer_types" and not "no_rope_layers" and not "mlp_only_layers")
            {
                throw new EntailException($"cannot graft into this host: config key '{key}' holds one entry per layer and the build cannot tell a block layer's");
            }
        }
        // rms_norm_eps stays as it is: the same value serves norms that do not
        // widen (per-head norms, linear attention's), which are far more
        // sensitive to it than the layer norms are.
        config.Remove("vision_config");
        File.WriteAllText(Path.Combine(outDir, "config.json"), config.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    }

    private static string PolicyType(AttentionLayerPolicy p) => p.Operator switch
    {
        LayerOperators.LinearAttention => "linear_attention",
        LayerOperators.Conv => "conv",
        _ => p.Span == AttentionSpan.Sliding ? "sliding_attention" : "full_attention",
    };

    /// <summary>Finds each program token in the host's vocabulary: byte-level
    /// BPE (GPT-2, Llama 3, Qwen), metaspace (SentencePiece, ▁ for a space),
    /// or a byte-fallback piece (<c>&lt;0x41&gt;</c>), in a dictionary or
    /// Unigram-list vocabulary. A token the vocabulary does not hold as one
    /// piece is simply not part of this model's alphabet.</summary>
    private static (Dictionary<string, int> Ids, List<string> Missing) BindTokens(EntailProgram p, string hostDir)
    {
        var root = JsonNode.Parse(File.ReadAllText(Path.Combine(hostDir, "tokenizer.json")))!.AsObject();
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        switch (root["model"]!["vocab"])
        {
            case JsonObject map:
                foreach (var (piece, id) in map)
                {
                    vocab[piece] = id!.GetValue<int>();
                }
                break;
            case JsonArray list:
                for (int i = 0; i < list.Count; i++)
                {
                    vocab.TryAdd(list[i]![0]!.GetValue<string>(), i);
                }
                break;
        }
        var byteMap = ByteLevelMap();
        var added = (root["added_tokens"]?.AsArray() ?? new JsonArray()).ToDictionary(
            t => t!["content"]!.GetValue<string>(), t => t!["id"]!.GetValue<int>(), StringComparer.Ordinal);
        var ids = new Dictionary<string, int>(StringComparer.Ordinal);
        var missing = new List<string>();
        foreach (var token in p.Tokens.Keys.Concat(p.Emits.Where(e => e.Text is not null).Select(e => e.Text!))
            .Concat(p.Masks.SelectMany(m => m.Texts)).Distinct())
        {
            var bytes = Encoding.UTF8.GetBytes(token);
            var spellings = new List<string>
            {
                string.Concat(bytes.Select(b => byteMap[b])),
                token.Replace(' ', '▁'),
            };
            if (bytes.Length == 1)
            {
                spellings.Add($"<0x{bytes[0]:X2}>");
            }
            int? found = spellings.Select(s => vocab.TryGetValue(s, out int id) ? id : (int?)null).FirstOrDefault(id => id is not null);
            if (found is { } id)
            {
                ids[token] = id;
            }
            else if (added.TryGetValue(token, out int special))
            {
                ids[token] = special;
            }
            else
            {
                missing.Add(token);
            }
        }
        return (ids, missing);
    }

    private static char[] ByteLevelMap()
    {
        var map = new char[256];
        int extra = 0;
        for (int b = 0; b < 256; b++)
        {
            bool printable = b is >= 33 and <= 126 or >= 161 and <= 172 or >= 174 and <= 255;
            map[b] = printable ? (char)b : (char)(256 + extra++);
        }
        return map;
    }
}
