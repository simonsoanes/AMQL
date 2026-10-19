using System.Text.Json;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>Star-Elastic prune inputs: the whole-output byte budget, the
/// calibration corpus, and the FFN-channel floor.</summary>
public sealed record StarElasticOptions(
    long TargetBytes,
    string? CorpusText = null,
    int SampleCap = 8192,
    int MinIntermediate = 1);

/// <summary>Outcome of one star-elastic prune: whole-output sizes, the
/// FFN-channel and attention-head counts, and the ranking/edge notes.</summary>
public sealed record StarElasticReport(
    string OutDir,
    string Model,
    long OriginalBytes,
    long TargetBytes,
    long FinalBytes,
    int OriginalIntermediate,
    int KeptIntermediate,
    int Layers,
    int DenseLayers,
    /// <summary>The kept FFN channel indices, in original order — the same
    /// set on every dense FFN layer (homogeneous width).</summary>
    IReadOnlyList<int> KeptChannels,
    int OriginalNumQHeads,
    int KeptNumQHeads,
    int OriginalNumKvHeads,
    int KeptNumKvHeads,
    /// <summary>The kept Q-head indices and KV-head indices, in original
    /// order — the same set on every softmax-attention layer. Empty when the
    /// model has no softmax attention.</summary>
    IReadOnlyList<int> KeptQHeads,
    IReadOnlyList<int> KeptKvHeads,
    IReadOnlyList<string> Notes);

/// <summary>
/// Star-Elastic structured <em>width</em> pruning — the zero-shot core of
/// the Star Elastic paper (importance estimation + nested slicing). One
/// forward pass over a calibration corpus ranks two axes by activation
/// magnitude:
/// <list type="bullet">
/// <item>FFN channels — <c>F_j = Σ_layers Σ_t |(hf · W_upᵀ)[t,j]|</c>;</item>
/// <item>attention heads — <c>F_h = Σ_layers Σ_t ‖Attn(XW_q^h, XW_k^h, XW_v^h)‖₂</c>,
/// aggregated to a KV head and its <c>NumQHeads/NumKvHeads</c> group so the
/// uniform GQA grouping stays intact.</item>
/// </list>
/// The axes are pruned <em>jointly</em> under <c>--target-bytes</c>: each
/// unit's raw score is normalised by its axis maximum (so a channel and a
/// KV group are both mapped to [0,1] "importance relative to its siblings"),
/// the units are merged least-important-first, and the pruner drops the
/// smallest least-important prefix that fits the budget — channels down to
/// <c>--min-intermediate</c>, heads down to one KV group.
///
/// Every axis is sliced to the same indices on every layer it applies to
/// (AMQL's surface is homogeneous), and kept units are restored to their
/// original index order — a plain smaller model, no permutation. The byte
/// budget is honoured with an exact estimate and a fail-closed floor, like
/// <see cref="LayerPruner"/>. This is stages 1+2 of the paper; the elastic
/// training (stage 3) that recovers accuracy is out of scope — see
/// docs/star-elastic-pruning.md.
/// </summary>
public static class StarElasticPruner
{
    private const string IndexFile = "index.json";

    private sealed class SlicedFfn
    {
        public SegmentTensor? Gate;
        public required SegmentTensor Up;
        public required SegmentTensor Down;
    }

    private sealed class SlicedAttn
    {
        public required SegmentTensor Q;
        public required SegmentTensor K;
        public required SegmentTensor V;
        public required SegmentTensor O;
        public required bool OutputGate;
    }

    private sealed class HeadSliceInfo
    {
        public required int NumQHeads;
        public required int NumKvHeads;
        public required int HeadDim;
        public required Dictionary<int, SlicedAttn> Layers;
    }

    private sealed class SliceContext
    {
        public required List<SegmentTensor> Carried;
        public required Dictionary<int, SlicedFfn> Ffn;
        public HeadSliceInfo? Head;
        public required int TensorCount;
    }

    private sealed record SegmentEstimate(long SegmentBytes, long PayloadBytes);

    public static StarElasticReport Prune(string containerDir, string outDir, StarElasticOptions options)
    {
        if (Directory.Exists(outDir))
        {
            throw new MergeException($"star-elastic output '{outDir}' already exists");
        }
        if (options.TargetBytes <= 0)
        {
            throw new MergeException("--target-bytes must be a positive byte count");
        }
        if (options.MinIntermediate < 1)
        {
            throw new MergeException("--min-intermediate must be at least 1");
        }

        using var container = Vindex3Container.Open(containerDir);
        var graph = container.Graph
            ?? throw new MergeException($"'{containerDir}' records no system graph — nothing to prune");
        var component = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.PrimaryText)
            ?? throw new MergeException("the container records no primary text component — nothing to prune");
        var surface = component.Execution
            ?? throw new MergeException($"component '{component.Id}' declares no execution surface");
        var ffn = surface.Ffn
            ?? throw new MergeException($"component '{component.Id}' declares no FFN surface");
        if (ffn.FfnType != FfnType.Gated && ffn.FfnType != FfnType.Dense)
        {
            throw new MergeException($"component '{component.Id}' declares FFN type '{ffn.FfnType}' — unsupported");
        }
        int intermediate = ffn.IntermediateSize;
        int hidden = component.HiddenSize;
        if (intermediate < options.MinIntermediate)
        {
            throw new MergeException(
                $"cannot keep {options.MinIntermediate} FFN channels of a {intermediate}-channel FFN");
        }
        bool drafterPresent = graph.Components.Any(c => c.Role == ComponentRole.Drafter);

        int[] tokens = Tokenize(container.Root, options.CorpusText, options.SampleCap);
        if (tokens.Length == 0)
        {
            throw new MergeException("the corpus encodes to zero tokens — nothing to score");
        }

        // ── plan ──────────────────────────────────────────────────────────
        using var store = container.CreateOperandStore();
        ComponentOpPlan plan;
        try
        {
            plan = Planner.Plan(container, component.Id, store);
        }
        catch (UnsupportedOperatorException e)
        {
            throw new MergeException(
                "--approach star-elastic needs a runnable stack, and the planner refuses: " + e.Message, e);
        }
        if (plan.Layers.Count != component.NumLayers)
        {
            throw new MergeException(
                $"planned {plan.Layers.Count} layers for a {component.NumLayers}-layer component — refusing to score a mismatch");
        }

        int denseLayers = plan.Layers.Count(l => l.Ffn?.Dense is not null);
        if (denseLayers == 0)
        {
            throw new MergeException(
                "star-elastic FFN pruning needs at least one dense FFN layer — routed-only networks are not " +
                "supported yet (MoE expert slicing is a later phase)");
        }

        // Homogeneous attention geometry from the first softmax layer.
        bool hasSoftmax = false;
        int numQHeads = 0, numKvHeads = 0, headDim = 0;
        foreach (var lp in plan.Layers)
        {
            if (lp.Attention is { } a)
            {
                numQHeads = a.NumQHeads;
                numKvHeads = a.NumKvHeads;
                headDim = a.HeadDim;
                hasSoftmax = true;
                break;
            }
        }
        if (hasSoftmax && (numQHeads <= 0 || numKvHeads <= 0 || numQHeads % numKvHeads != 0))
        {
            throw new MergeException(
                $"degenerate attention geometry: num_q_heads={numQHeads}, num_kv_heads={numKvHeads}");
        }
        int groupSize = hasSoftmax ? numQHeads / numKvHeads : 0;

        // ── one forward pass scores both axes ─────────────────────────────
        double[] channelScore = new double[intermediate];
        double[] headScore = hasSoftmax ? new double[numQHeads] : Array.Empty<double>();
        var runtime = new GenericRuntime(plan, store);
        runtime.FfnInputCapture = (layer, hf) =>
        {
            if (plan.Layers[layer].Ffn?.Dense is { } dense)
            {
                var up = runtime.Weights.Matrix(dense.Up, dense.IntermediateSize, hidden);
                var hu = TensorOps.MatMulTransposedB(hf, up); // [rows, intermediate]
                for (int r = 0; r < hu.Rows; r++)
                {
                    int off = r * intermediate;
                    for (int j = 0; j < intermediate; j++)
                    {
                        channelScore[j] += Math.Abs(hu.Data[off + j]);
                    }
                }
            }
        };
        if (hasSoftmax)
        {
            runtime.AttentionContextCapture = (layer, output) =>
            {
                if (plan.Layers[layer].Attention is null)
                {
                    return;
                }
                for (int r = 0; r < output.Rows; r++)
                {
                    int off = r * output.Cols;
                    for (int h = 0; h < numQHeads; h++)
                    {
                        double n = 0;
                        for (int d = 0; d < headDim; d++)
                        {
                            float x = output.Data[off + h * headDim + d];
                            n += (double)x * x;
                        }
                        headScore[h] += Math.Sqrt(n);
                    }
                }
            };
        }
        foreach (var token in tokens)
        {
            runtime.StepForward(token);
        }

        double[] groupScore = new double[hasSoftmax ? numKvHeads : 0];
        if (hasSoftmax)
        {
            for (int h = 0; h < numQHeads; h++)
            {
                groupScore[h / groupSize] += headScore[h];
            }
        }

        int[] channelDesc = Enumerable.Range(0, intermediate)
            .OrderByDescending(j => channelScore[j]).ThenBy(j => j).ToArray();
        int[] groupDesc = hasSoftmax
            ? Enumerable.Range(0, numKvHeads)
                .OrderByDescending(g => groupScore[g]).ThenBy(g => g).ToArray()
            : Array.Empty<int>();

        // ── classify stack tensors: slice targets vs carried ─────────────
        var stackObj = graph.Objects.First(o => o.Component == component.Id && o.Kind == ObjectKind.DecoderStack);
        string stackId = stackObj.Id;
        string stackRepId = container.CanonicalRepresentationId(stackId);
        var stackEntry = container.Index.Representations[stackRepId];
        string stackSegmentPath = Path.Combine(container.Root, stackEntry.Segment);

        List<SegmentTensor> sourceTensors;
        using (var segment = SegmentFile.Open(stackSegmentPath))
        {
            sourceTensors = segment.Header.Tensors.ToList();
        }
        var byName = sourceTensors.ToDictionary(t => t.Name, StringComparer.Ordinal);

        var slicedNames = new HashSet<string>(StringComparer.Ordinal);
        var slicedFfn = new Dictionary<int, SlicedFfn>();
        bool gated = false;
        for (int l = 0; l < plan.Layers.Count; l++)
        {
            var dense = plan.Layers[l].Ffn?.Dense;
            if (dense is null)
            {
                continue;
            }
            var gate = dense.Gate is { } g ? Lookup(byName, g.TensorName, stackSegmentPath) : null;
            var up = Lookup(byName, dense.Up.TensorName, stackSegmentPath);
            var down = Lookup(byName, dense.Down.TensorName, stackSegmentPath);
            foreach (var t in new[] { gate, up, down })
            {
                RequireFlat(t, stackSegmentPath);
            }
            if (gate is not null)
            {
                gated = true;
                slicedNames.Add(gate.Name);
            }
            slicedNames.Add(up.Name);
            slicedNames.Add(down.Name);
            slicedFfn[l] = new SlicedFfn { Gate = gate, Up = up, Down = down };
        }

        HeadSliceInfo? headInfo = null;
        if (hasSoftmax)
        {
            var attnSurface = surface.Attention
                ?? throw new MergeException("the plan serves softmax attention but the surface declares no attention facts");
            var attnLayers = new Dictionary<int, SlicedAttn>();
            for (int l = 0; l < plan.Layers.Count; l++)
            {
                if (plan.Layers[l].Attention is not { } attn)
                {
                    continue;
                }
                var q = Lookup(byName, attn.QProj.TensorName, stackSegmentPath);
                var k = Lookup(byName, attn.KProj.TensorName, stackSegmentPath);
                var v = Lookup(byName, attn.VProj.TensorName, stackSegmentPath);
                var o = Lookup(byName, attn.OProj.TensorName, stackSegmentPath);
                foreach (var t in new[] { q, k, v, o })
                {
                    RequireFlat(t, stackSegmentPath);
                }
                slicedNames.Add(q.Name);
                slicedNames.Add(k.Name);
                slicedNames.Add(v.Name);
                slicedNames.Add(o.Name);
                attnLayers[l] = new SlicedAttn { Q = q, K = k, V = v, O = o, OutputGate = attn.OutputGate };
            }
            headInfo = new HeadSliceInfo
            {
                NumQHeads = attnSurface.NumQHeads,
                NumKvHeads = attnSurface.NumKvHeads,
                HeadDim = attnSurface.HeadDim,
                Layers = attnLayers,
            };
        }

        var carried = sourceTensors.Where(t => !slicedNames.Contains(t.Name)).ToList();
        int ffnTensorCount = slicedFfn.Count * (gated ? 3 : 2);
        int headTensorCount = headInfo is null ? 0 : headInfo.Layers.Count * 4;
        var ctx = new SliceContext
        {
            Carried = carried,
            Ffn = slicedFfn,
            Head = headInfo,
            TensorCount = carried.Count + ffnTensorCount + headTensorCount,
        };

        // ── baseline byte census ──────────────────────────────────────────
        long inputTotal = DirectorySize(containerDir);
        string graphRel = container.Index.SystemGraph
            ?? throw new MergeException("container graph is present but index records no system graph path");
        string indexPath = Path.Combine(container.Root, IndexFile);
        string graphPath = Path.Combine(container.Root, graphRel);
        long copiedBytes = inputTotal
            - new FileInfo(stackSegmentPath).Length
            - new FileInfo(indexPath).Length
            - new FileInfo(graphPath).Length;

        // ── joint allocation: drop the least-important prefix until it fits ─
        double cMax = channelScore.Max();
        double gMax = hasSoftmax ? groupScore.Max() : 0.0;
        int maxDropChannels = intermediate - options.MinIntermediate;
        int maxDropGroups = hasSoftmax ? numKvHeads - 1 : 0;

        var units = new List<(bool IsGroup, int Index, double Norm)>(
            maxDropChannels + maxDropGroups);
        foreach (var j in Enumerable.Range(0, intermediate)
                     .OrderBy(j => channelScore[j]).ThenBy(j => j).Take(maxDropChannels))
        {
            units.Add((false, j, cMax > 0 ? channelScore[j] / cMax : 0.0));
        }
        foreach (var g in Enumerable.Range(0, numKvHeads)
                     .OrderBy(g => groupScore[g]).ThenBy(g => g).Take(maxDropGroups))
        {
            units.Add((true, g, gMax > 0 ? groupScore[g] / gMax : 0.0));
        }
        units.Sort((a, b) =>
        {
            int c = a.Norm.CompareTo(b.Norm);
            if (c != 0)
            {
                return c;
            }
            c = (a.IsGroup ? 1 : 0).CompareTo(b.IsGroup ? 1 : 0);
            if (c != 0)
            {
                return c;
            }
            return a.Index.CompareTo(b.Index);
        });

        long floorBytes = -1;
        int droppedChannels = 0, droppedGroups = 0;
        for (int m = 0; ; m++)
        {
            int keptInter = intermediate - droppedChannels;
            int keptKv = hasSoftmax ? numKvHeads - droppedGroups : 0;

            var estimate = MeasureStack(stackRepId, ctx, keptInter, keptKv);
            var newFfn = RebuildFfn(ffn, keptInter);
            var newAttn = hasSoftmax
                ? RebuildAttention(surface.Attention!, keptKv * groupSize, keptKv)
                : surface.Attention;
            var newSurface = RebuildSurface(surface, newFfn, newAttn);
            var newGraph = BuildGraph(graph, component, newSurface);
            long graphLen = JsonSerializer.SerializeToUtf8Bytes(newGraph, ViJson.Options).Length;
            var indexEstimate = BuildIndex(
                container.Index, stackEntry, stackRepId, keptInter, intermediate,
                hasSoftmax ? numQHeads : 0, keptKv * groupSize,
                hasSoftmax ? numKvHeads : 0, keptKv,
                options, ctx.TensorCount, estimate.PayloadBytes,
                payloadSha: new string('0', 64), segmentSha: new string('0', 64));
            long indexLen = JsonSerializer.SerializeToUtf8Bytes(indexEstimate, ViJson.Options).Length;
            long total = copiedBytes + estimate.SegmentBytes + indexLen + graphLen;

            if (total <= options.TargetBytes)
            {
                int[] keptChannels = channelDesc.Take(keptInter).OrderBy(j => j).ToArray();
                int[] keptKvGroups = hasSoftmax ? groupDesc.Take(keptKv).OrderBy(g => g).ToArray() : Array.Empty<int>();
                int[] keptQHeads = hasSoftmax
                    ? keptKvGroups.SelectMany(g => Enumerable.Range(g * groupSize, groupSize)).ToArray()
                    : Array.Empty<int>();

                var report = Commit(
                    container, outDir, graph, component, surface, ffn, stackRepId, stackEntry, ctx,
                    keptChannels, keptKvGroups, keptQHeads, keptInter, keptKv, intermediate,
                    hasSoftmax ? numQHeads : 0, hasSoftmax ? numKvHeads : 0,
                    options, copiedBytes, inputTotal, gated, headInfo is not null, drafterPresent, tokens.Length);
                if (report is not null)
                {
                    return report;
                }
            }
            floorBytes = total;

            if (m >= units.Count)
            {
                break;
            }
            if (units[m].IsGroup)
            {
                droppedGroups++;
            }
            else
            {
                droppedChannels++;
            }
        }

        throw new MergeException(
            $"cannot reach target-bytes '{FormatBytes(options.TargetBytes)}': the floor is {FormatBytes(floorBytes)} " +
            $"(FFN {intermediate}→{options.MinIntermediate} channels" +
            (hasSoftmax && numKvHeads > 1 ? $", attention {numKvHeads}→1 KV groups" : "") + ")");
    }

    // ── ranking / tokenization ────────────────────────────────────────────

    private static int[] Tokenize(string containerRoot, string? corpusText, int sampleCap)
    {
        if (corpusText is null)
        {
            throw new MergeException("--approach star-elastic requires '--text <corpus.txt>'");
        }
        int[] ids;
        try
        {
            ids = HfTokenizer.FromModelDir(containerRoot).EncodeToIds(corpusText).ToArray();
        }
        catch (TokenizerException e)
        {
            throw new MergeException(
                "--approach star-elastic needs a BPE tokenizer.json beside the container: " + e.Message, e);
        }
        int cap = sampleCap > 0 ? Math.Min(sampleCap, ids.Length) : ids.Length;
        return ids.Take(cap).ToArray();
    }

    // ── slice helpers ─────────────────────────────────────────────────────

    private static SegmentTensor Lookup(Dictionary<string, SegmentTensor> byName, string name, string segmentPath)
    {
        if (byName.TryGetValue(name, out var tensor))
        {
            return tensor;
        }
        throw new MergeException($"segment '{segmentPath}' records no tensor '{name}' — the plan/segment disagree");
    }

    private static int Elem(SegmentTensor tensor) => DtypeExtensions.FromLabel(tensor.Dtype).ElementSize();

    private static void RequireFlat(SegmentTensor? tensor, string segmentPath)
    {
        if (tensor is not null && tensor.Dtype is not ("F32" or "F16" or "BF16"))
        {
            throw new MergeException(
                $"tensor '{tensor.Name}' is stored as '{tensor.Dtype}' — star-elastic width pruning slices at byte " +
                $"granularity and needs a flat F32/F16/BF16 stack (segment '{segmentPath}'); re-encode with " +
                "'--strip --keep F32' first");
        }
    }

    /// <summary>FFN channel gather: rows keep a channel for gate/up
    /// (<c>[I, H]</c>), columns for down (<c>[H, I]</c>).</summary>
    private static byte[] GatherRows(byte[] source, int width, int elem, int[] units)
    {
        var result = new byte[units.Length * width * elem];
        for (int r = 0; r < units.Length; r++)
        {
            Buffer.BlockCopy(source, units[r] * width * elem, result, r * width * elem, width * elem);
        }
        return result;
    }

    private static byte[] GatherCols(byte[] source, int srcCols, int rows, int elem, int[] cols)
    {
        var result = new byte[rows * cols.Length * elem];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols.Length; c++)
            {
                Buffer.BlockCopy(
                    source, (r * srcCols + cols[c]) * elem,
                    result, (r * cols.Length + c) * elem, elem);
            }
        }
        return result;
    }

    /// <summary>Head-block row gather: each kept head owns <paramref name="blockRows"/>
    /// contiguous rows of <paramref name="width"/> elements (q/k/v: <c>[heads·blockRows, hidden]</c>).</summary>
    private static byte[] GatherHeadRows(byte[] source, int hidden, int elem, int[] keptHeads, int blockRows)
    {
        int blockBytes = blockRows * hidden * elem;
        var result = new byte[keptHeads.Length * blockBytes];
        for (int i = 0; i < keptHeads.Length; i++)
        {
            Buffer.BlockCopy(source, keptHeads[i] * blockBytes, result, i * blockBytes, blockBytes);
        }
        return result;
    }

    /// <summary>Head-block column gather: each kept Q head owns <paramref name="headDim"/>
    /// contiguous columns (o: <c>[hidden, heads·headDim]</c>).</summary>
    private static byte[] GatherHeadCols(byte[] source, int hidden, int srcCols, int elem, int[] keptHeads, int headDim)
    {
        var result = new byte[hidden * keptHeads.Length * headDim * elem];
        for (int r = 0; r < hidden; r++)
        {
            for (int i = 0; i < keptHeads.Length; i++)
            {
                Buffer.BlockCopy(
                    source, (r * srcCols + keptHeads[i] * headDim) * elem,
                    result, (r * keptHeads.Length * headDim + i * headDim) * elem,
                    headDim * elem);
            }
        }
        return result;
    }

    // ── shape/length geometry (shared by estimate and commit) ─────────────

    private static (long[] Shape, long Len) FfnTensor(SegmentTensor t, int keptInter, int hidden, bool isDown) =>
        isDown
            ? (new long[] { hidden, keptInter }, (long)hidden * keptInter * Elem(t))
            : (new long[] { keptInter, hidden }, (long)keptInter * hidden * Elem(t));

    private static (long[] Shape, long Len) QTensor(SegmentTensor q, int hidden, int headDim, int keptQCount, bool outputGate)
    {
        int blockRows = outputGate ? 2 * headDim : headDim;
        return (new long[] { (long)keptQCount * blockRows, hidden }, (long)keptQCount * blockRows * hidden * Elem(q));
    }

    private static (long[] Shape, long Len) KvTensor(SegmentTensor k, int hidden, int headDim, int keptKvCount) =>
        (new long[] { (long)keptKvCount * headDim, hidden }, (long)keptKvCount * headDim * hidden * Elem(k));

    private static (long[] Shape, long Len) OTensor(SegmentTensor o, int hidden, int headDim, int keptQCount) =>
        (new long[] { hidden, (long)keptQCount * headDim }, (long)hidden * keptQCount * headDim * Elem(o));

    // ── exact stack-size estimate ─────────────────────────────────────────

    private static SegmentEstimate MeasureStack(string repId, SliceContext ctx, int keptInter, int keptKv)
    {
        int keptQCount = ctx.Head is { } h ? keptKv * (h.NumQHeads / h.NumKvHeads) : 0;
        var desc = new List<(string Name, string Dtype, long[] Shape, long Len)>(
            ctx.Carried.Count + ctx.Ffn.Count * 3 + (ctx.Head?.Layers.Count * 4 ?? 0));

        foreach (var t in ctx.Carried)
        {
            desc.Add((t.Name, t.Dtype, t.Shape, t.Len));
        }
        foreach (var sf in ctx.Ffn.Values)
        {
            int hidden = (int)sf.Up.Shape[1];
            if (sf.Gate is { } gate)
            {
                var g = FfnTensor(gate, keptInter, hidden, isDown: false);
                desc.Add((gate.Name, gate.Dtype, g.Shape, g.Len));
            }
            var u = FfnTensor(sf.Up, keptInter, hidden, isDown: false);
            desc.Add((sf.Up.Name, sf.Up.Dtype, u.Shape, u.Len));
            var d = FfnTensor(sf.Down, keptInter, hidden, isDown: true);
            desc.Add((sf.Down.Name, sf.Down.Dtype, d.Shape, d.Len));
        }
        if (ctx.Head is { } head)
        {
            foreach (var sa in head.Layers.Values)
            {
                int hidden = (int)sa.O.Shape[0];
                var q = QTensor(sa.Q, hidden, head.HeadDim, keptQCount, sa.OutputGate);
                var k = KvTensor(sa.K, hidden, head.HeadDim, keptKv);
                var v = KvTensor(sa.V, hidden, head.HeadDim, keptKv);
                var o = OTensor(sa.O, hidden, head.HeadDim, keptQCount);
                desc.Add((sa.Q.Name, sa.Q.Dtype, q.Shape, q.Len));
                desc.Add((sa.K.Name, sa.K.Dtype, k.Shape, k.Len));
                desc.Add((sa.V.Name, sa.V.Dtype, v.Shape, v.Len));
                desc.Add((sa.O.Name, sa.O.Dtype, o.Shape, o.Len));
            }
        }
        desc.Sort((a, b) => StringComparer.Ordinal.Compare(a.Name, b.Name));

        long payload = 0;
        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WriteNumber("schema", SegmentFormat.CurrentSchema);
            writer.WriteString("representation", repId);
            writer.WritePropertyName("tensors");
            writer.WriteStartArray();
            long cursor = 0;
            foreach (var (name, dtype, shape, len) in desc)
            {
                writer.WriteStartObject();
                writer.WriteString("name", name);
                writer.WriteString("dtype", dtype);
                writer.WritePropertyName("shape");
                writer.WriteStartArray();
                foreach (var dim in shape)
                {
                    writer.WriteNumberValue(dim);
                }
                writer.WriteEndArray();
                writer.WriteNumber("offset", cursor);
                writer.WriteNumber("len", len);
                writer.WriteEndObject();
                cursor += len;
            }
            writer.WriteEndArray();
            writer.WriteEndObject();
            writer.Flush();
            payload = cursor;
        }
        var headerJson = buffer.ToArray();
        int pad = (SegmentFormat.PayloadAlignment -
                   ((SegmentFormat.HeaderLengthBytes + headerJson.Length) % SegmentFormat.PayloadAlignment)) %
                  SegmentFormat.PayloadAlignment;
        int storedLength = headerJson.Length + pad;
        return new SegmentEstimate(
            SegmentFormat.HeaderLengthBytes + storedLength + payload,
            payload);
    }

    // ── surface / graph / index rebuilders ────────────────────────────────

    private static FfnSurface RebuildFfn(FfnSurface source, int keptIntermediate) => new()
    {
        IntermediateSize = keptIntermediate,
        Activation = source.Activation,
        FfnType = source.FfnType,
        GatePolicy = source.GatePolicy,
        Moe = source.Moe,
    };

    private static AttentionSurface RebuildAttention(AttentionSurface source, int keptQHeads, int keptKvHeads) => new()
    {
        NumQHeads = keptQHeads,
        NumKvHeads = keptKvHeads,
        HeadDim = source.HeadDim,
        QueryScale = source.QueryScale,
        ScoreScale = source.ScoreScale,
        LogitSoftcapping = source.LogitSoftcapping,
        QkNormScope = source.QkNormScope,
        QkNormWeightOffset = source.QkNormWeightOffset,
        ParameterFreeQkNorm = source.ParameterFreeQkNorm,
        OutputGate = source.OutputGate,
        Sinks = source.Sinks,
        AttentionBias = source.AttentionBias,
    };

    private static ExecutionSurface RebuildSurface(ExecutionSurface source, FfnSurface newFfn, AttentionSurface? newAttn) => new()
    {
        ContextLength = source.ContextLength,
        Attention = newAttn,
        Ffn = newFfn,
        Norm = source.Norm,
        Head = source.Head,
        ResidualScale = source.ResidualScale,
        LinearAttention = source.LinearAttention,
        Kda = source.Kda,
        KdaGateLowerBound = source.KdaGateLowerBound,
        Mla = source.Mla,
        Mamba2 = source.Mamba2,
        ConvQkv = source.ConvQkv,
        ResidualInFp32 = source.ResidualInFp32,
        Classifier = source.Classifier,
        Encoder = source.Encoder,
        OptionMarker = source.OptionMarker,
        Embedding = source.Embedding,
    };

    private static SystemGraph BuildGraph(SystemGraph source, Component primary, ExecutionSurface newSurface)
    {
        var components = new List<Component>(source.Components.Count);
        foreach (var component in source.Components)
        {
            components.Add(component.Id == primary.Id
                ? new Component
                {
                    Id = component.Id,
                    Role = component.Role,
                    SourceArtifact = component.SourceArtifact,
                    NumLayers = component.NumLayers,
                    HiddenSize = component.HiddenSize,
                    Attention = component.Attention,
                    Execution = newSurface,
                    Perception = component.Perception,
                }
                : component);
        }
        return new SystemGraph
        {
            Schema = source.Schema,
            Components = components,
            Objects = new List<LogicalObject>(source.Objects),
            Edges = new List<HiddenStateEdge>(source.Edges),
        };
    }

    private static Vindex3Index BuildIndex(
        Vindex3Index source,
        RepresentationEntry stackEntry,
        string stackRepId,
        int keptIntermediate,
        int originalIntermediate,
        int keptQHeads,
        int originalQHeads,
        int keptKvHeads,
        int originalKvHeads,
        StarElasticOptions options,
        int tensorCount,
        long payloadBytes,
        string payloadSha,
        string segmentSha)
    {
        var reps = new Dictionary<string, RepresentationEntry>(StringComparer.Ordinal);
        foreach (var (repId, entry) in source.Representations)
        {
            if (repId != stackRepId)
            {
                reps[repId] = entry;
                continue;
            }
            reps[repId] = new RepresentationEntry
            {
                Object = entry.Object,
                Encoding = entry.Encoding,
                Segment = entry.Segment,
                TensorCount = tensorCount,
                PayloadBytes = payloadBytes,
                PayloadSha256 = payloadSha,
                SegmentSha256 = segmentSha,
                CompiledFrom = entry.CompiledFrom,
            };
        }

        var extra = source.Extra is null
            ? new Dictionary<string, JsonElement>()
            : new Dictionary<string, JsonElement>(source.Extra, StringComparer.Ordinal);
        extra["prune"] = JsonSerializer.SerializeToElement(new
        {
            Approach = "star-elastic",
            Axis = new[] { "ffn_channels", "attention_heads" },
            FromIntermediate = originalIntermediate,
            ToIntermediate = keptIntermediate,
            FromNumQHeads = originalQHeads,
            ToNumQHeads = keptQHeads,
            FromNumKvHeads = originalKvHeads,
            ToNumKvHeads = keptKvHeads,
            TargetBytes = options.TargetBytes,
        }, ViJson.Options);

        return new Vindex3Index
        {
            Version = source.Version,
            Model = source.Model,
            Family = source.Family,
            HiddenSize = source.HiddenSize,
            NumLayers = source.NumLayers,
            MoeManifest = source.MoeManifest,
            SystemGraph = source.SystemGraph,
            Representations = reps,
            Profiles = new List<Profile>(source.Profiles),
            Segments = new Dictionary<string, int>(source.Segments, StringComparer.Ordinal),
            Authority = source.Authority,
            PrecisionMap = source.PrecisionMap,
            DerivedFromModel = source.DerivedFromModel,
            TokenMap = source.TokenMap,
            Extra = extra,
        };
    }

    // ── commit ────────────────────────────────────────────────────────────

    private static StarElasticReport? Commit(
        Vindex3Container container,
        string outDir,
        SystemGraph graph,
        Component component,
        ExecutionSurface surface,
        FfnSurface ffn,
        string stackRepId,
        RepresentationEntry stackEntry,
        SliceContext ctx,
        int[] keptChannels,
        int[] keptKvGroups,
        int[] keptQHeads,
        int keptIntermediate,
        int keptKv,
        int originalIntermediate,
        int originalQHeads,
        int originalKvHeads,
        StarElasticOptions options,
        long copiedBytes,
        long inputTotal,
        bool gated,
        bool headPruned,
        bool drafterPresent,
        int sampledTokens)
    {
        try
        {
            Directory.CreateDirectory(outDir);

            var named = new List<NamedTensorData>(ctx.TensorCount);
            using (var sourceSegment = SegmentFile.Open(Path.Combine(container.Root, stackEntry.Segment)))
            {
                foreach (var tensor in ctx.Carried)
                {
                    named.Add(new NamedTensorData
                    {
                        Name = tensor.Name,
                        Dtype = DtypeExtensions.FromLabel(tensor.Dtype),
                        Shape = tensor.Shape,
                        Data = sourceSegment.ReadBytes(tensor.Name),
                    });
                }
                foreach (var sf in ctx.Ffn.Values)
                {
                    int hidden = (int)sf.Up.Shape[1];
                    if (sf.Gate is { } gate)
                    {
                        named.Add(new NamedTensorData
                        {
                            Name = gate.Name,
                            Dtype = DtypeExtensions.FromLabel(gate.Dtype),
                            Shape = new long[] { keptIntermediate, hidden },
                            Data = GatherRows(sourceSegment.ReadBytes(gate.Name), hidden, Elem(gate), keptChannels),
                        });
                    }
                    named.Add(new NamedTensorData
                    {
                        Name = sf.Up.Name,
                        Dtype = DtypeExtensions.FromLabel(sf.Up.Dtype),
                        Shape = new long[] { keptIntermediate, hidden },
                        Data = GatherRows(sourceSegment.ReadBytes(sf.Up.Name), hidden, Elem(sf.Up), keptChannels),
                    });
                    named.Add(new NamedTensorData
                    {
                        Name = sf.Down.Name,
                        Dtype = DtypeExtensions.FromLabel(sf.Down.Dtype),
                        Shape = new long[] { hidden, keptIntermediate },
                        Data = GatherCols(sourceSegment.ReadBytes(sf.Down.Name),
                            (int)sf.Down.Shape[1], hidden, Elem(sf.Down), keptChannels),
                    });
                }
                if (ctx.Head is { } head)
                {
                    foreach (var sa in head.Layers.Values)
                    {
                        int hidden = (int)sa.O.Shape[0];
                        int srcCols = (int)sa.O.Shape[1];
                        int blockRows = sa.OutputGate ? 2 * head.HeadDim : head.HeadDim;
                        named.Add(new NamedTensorData
                        {
                            Name = sa.Q.Name,
                            Dtype = DtypeExtensions.FromLabel(sa.Q.Dtype),
                            Shape = new long[] { (long)keptQHeads.Length * blockRows, hidden },
                            Data = GatherHeadRows(sourceSegment.ReadBytes(sa.Q.Name), hidden, Elem(sa.Q), keptQHeads, blockRows),
                        });
                        named.Add(new NamedTensorData
                        {
                            Name = sa.K.Name,
                            Dtype = DtypeExtensions.FromLabel(sa.K.Dtype),
                            Shape = new long[] { (long)keptKvGroups.Length * head.HeadDim, hidden },
                            Data = GatherHeadRows(sourceSegment.ReadBytes(sa.K.Name), hidden, Elem(sa.K), keptKvGroups, head.HeadDim),
                        });
                        named.Add(new NamedTensorData
                        {
                            Name = sa.V.Name,
                            Dtype = DtypeExtensions.FromLabel(sa.V.Dtype),
                            Shape = new long[] { (long)keptKvGroups.Length * head.HeadDim, hidden },
                            Data = GatherHeadRows(sourceSegment.ReadBytes(sa.V.Name), hidden, Elem(sa.V), keptKvGroups, head.HeadDim),
                        });
                        named.Add(new NamedTensorData
                        {
                            Name = sa.O.Name,
                            Dtype = DtypeExtensions.FromLabel(sa.O.Dtype),
                            Shape = new long[] { hidden, (long)keptQHeads.Length * head.HeadDim },
                            Data = GatherHeadCols(sourceSegment.ReadBytes(sa.O.Name), hidden, srcCols, Elem(sa.O), keptQHeads, head.HeadDim),
                        });
                    }
                }
            }

            string stackDir = Path.Combine(outDir, Path.GetDirectoryName(stackEntry.Segment)!);
            Directory.CreateDirectory(stackDir);
            var stackWrite = SegmentWriter.Write(Path.Combine(outDir, stackEntry.Segment), stackRepId, named);

            var newFfn = RebuildFfn(ffn, keptIntermediate);
            var newAttn = headPruned
                ? RebuildAttention(surface.Attention!, keptQHeads.Length, keptKvGroups.Length)
                : surface.Attention;
            var newSurface = RebuildSurface(surface, newFfn, newAttn);
            var newGraph = BuildGraph(graph, component, newSurface);
            File.WriteAllBytes(Path.Combine(outDir, container.Index.SystemGraph!),
                JsonSerializer.SerializeToUtf8Bytes(newGraph, ViJson.Options));

            var newIndex = BuildIndex(
                container.Index, stackEntry, stackRepId, keptIntermediate, originalIntermediate,
                keptQHeads.Length, originalQHeads, keptKvGroups.Length, originalKvHeads,
                options, named.Count, stackWrite.PayloadBytes,
                stackWrite.PayloadSha256Hex, stackWrite.SegmentSha256Hex);
            File.WriteAllBytes(Path.Combine(outDir, IndexFile),
                JsonSerializer.SerializeToUtf8Bytes(newIndex, ViJson.Options));

            long exactTotal = copiedBytes
                + new FileInfo(Path.Combine(outDir, stackEntry.Segment)).Length
                + new FileInfo(Path.Combine(outDir, IndexFile)).Length
                + new FileInfo(Path.Combine(outDir, container.Index.SystemGraph!)).Length;
            if (exactTotal > options.TargetBytes)
            {
                Directory.Delete(outDir, recursive: true);
                return null;
            }

            foreach (var file in Directory.EnumerateFiles(container.Root, "*", SearchOption.AllDirectories))
            {
                string rel = Path.GetRelativePath(container.Root, file).Replace('\\', '/');
                if (rel.Equals(IndexFile, StringComparison.Ordinal) ||
                    rel.Equals(container.Index.SystemGraph, StringComparison.Ordinal) ||
                    rel.Equals(stackEntry.Segment, StringComparison.Ordinal))
                {
                    continue;
                }
                string dest = Path.Combine(outDir, rel);
                Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
                File.Copy(file, dest);
            }

            var notes = new List<string>
            {
                $"ranked {originalIntermediate} FFN channels by Σ|x·W_upᵀ| and" +
                (headPruned ? $" {originalKvHeads} KV-head groups by Σ‖head context‖₂" : " no softmax heads") +
                $" over {sampledTokens} sampled positions",
                gated
                    ? "gated FFN — gate/up rows and down columns sliced to the same kept channels"
                    : "dense FFN — up rows and down columns sliced to the same kept channels",
                "homogeneous width — the same channels/heads are kept in every layer (AMQL's surface is per-model, not per-layer)",
            };
            if (headPruned)
            {
                notes.Add($"attention heads — {originalKvHeads} → {keptKvGroups.Length} KV groups " +
                          $"({originalQHeads} → {keptQHeads.Length} Q heads)");
            }
            if (keptIntermediate == originalIntermediate && (!headPruned || keptKvGroups.Length == originalKvHeads))
            {
                notes.Add("already within the budget — nothing dropped, the stack was re-encoded unchanged");
            }
            if (drafterPresent)
            {
                notes.Add("an MTP drafter is present and its trunk (a copy of a full-attention layer) predates the " +
                          "slice — regenerate it with generate-mtp, or export without it");
            }

            return new StarElasticReport(
                OutDir: outDir,
                Model: container.Index.Model,
                OriginalBytes: inputTotal,
                TargetBytes: options.TargetBytes,
                FinalBytes: exactTotal,
                OriginalIntermediate: originalIntermediate,
                KeptIntermediate: keptIntermediate,
                Layers: component.NumLayers,
                DenseLayers: ctx.Ffn.Count,
                KeptChannels: keptChannels,
                OriginalNumQHeads: originalQHeads,
                KeptNumQHeads: keptQHeads.Length,
                OriginalNumKvHeads: originalKvHeads,
                KeptNumKvHeads: keptKvGroups.Length,
                KeptQHeads: keptQHeads,
                KeptKvHeads: keptKvGroups,
                Notes: notes);
        }
        catch
        {
            if (Directory.Exists(outDir))
            {
                Directory.Delete(outDir, recursive: true);
            }
            throw;
        }
    }

    // ── small helpers ─────────────────────────────────────────────────────

    private static long DirectorySize(string dir) =>
        Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories)
            .Sum(f => new FileInfo(f).Length);

    private static string FormatBytes(long bytes) => bytes switch
    {
        >= 1 << 30 => $"{bytes / 1073741824.0:0.00} GiB",
        >= 1 << 20 => $"{bytes / 1048576.0:0.00} MiB",
        >= 1 << 10 => $"{bytes / 1024.0:0.00} KiB",
        _ => $"{bytes} B",
    };
}