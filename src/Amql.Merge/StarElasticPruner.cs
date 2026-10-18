using System.Text.Json;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>Star-Elastic prune inputs: the whole-output byte budget, the
/// calibration corpus, and the FFN-channel controls.</summary>
public sealed record StarElasticOptions(
    long TargetBytes,
    string? CorpusText = null,
    int SampleCap = 8192,
    int MinIntermediate = 1);

/// <summary>Outcome of one star-elastic prune: the sizes, the FFN-channel
/// counts, and the ranking/edge notes.</summary>
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
    /// set on every dense layer (homogeneous width).</summary>
    IReadOnlyList<int> KeptChannels,
    IReadOnlyList<string> Notes);

/// <summary>
/// Star-Elastic structured <em>width</em> pruning — the zero-shot core of
/// the Star Elastic paper (importance estimation + nested slicing), the
/// FFN-channel axis. One forward pass over a calibration corpus ranks the
/// FFN intermediate channels by their activation magnitude
/// (<c>F_j = Σ_layers Σ_t |(hf · W_upᵀ)[t,j]|</c>); the pruner then keeps the
/// top-K channels — the same K and the same indices on <em>every</em> dense
/// FFN layer, because AMQL's <see cref="FfnSurface.IntermediateSize"/> is a
/// single homogeneous value — and slices gate/up rows and down columns to
/// produce a smaller, complete, planable container.
///
/// The output is a plain smaller model (kept channels restored to their
/// original index order — no permutation). The byte budget is honored with
/// an exact estimate and a fail-closed floor, exactly like
/// <see cref="LayerPruner"/>. This is stages 1+2 of the paper; the elastic
/// training (stage 3) that recovers accuracy is out of scope here — see
/// docs/star-elastic-pruning.md.
/// </summary>
public static class StarElasticPruner
{
    private const string IndexFile = "index.json";

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
        if (intermediate <= options.MinIntermediate)
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

        // ── plan + activation-based ranking (one forward pass) ────────────
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

        double[] score = new double[intermediate];
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
                        score[j] += Math.Abs(hu.Data[off + j]);
                    }
                }
            }
        };
        foreach (var token in tokens)
        {
            runtime.StepForward(token);
        }

        int[] ranked = Enumerable.Range(0, intermediate)
            .OrderByDescending(j => score[j])
            .ThenBy(j => j)
            .ToArray();

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

        var sliced = new Dictionary<int, SlicedFfn>();
        var slicedNames = new HashSet<string>(StringComparer.Ordinal);
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
                if (t is not null && t.Dtype is not ("F32" or "F16" or "BF16"))
                {
                    throw new MergeException(
                        $"tensor '{t.Name}' is stored as '{t.Dtype}' — star-elastic width pruning slices at byte " +
                        "granularity and needs a flat F32/F16/BF16 stack; re-encode with '--strip --keep F32' first");
                }
            }
            if (gate is not null)
            {
                gated = true;
                slicedNames.Add(gate.Name);
            }
            slicedNames.Add(up.Name);
            slicedNames.Add(down.Name);
            sliced[l] = new SlicedFfn { Gate = gate, Up = up, Down = down };
        }
        var carried = sourceTensors.Where(t => !slicedNames.Contains(t.Name)).ToList();
        int tensorCount = carried.Count + sliced.Count * (gated ? 3 : 2);

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

        // ── scan K = intermediate … MinIntermediate for the largest fit ──
        long floor = -1;
        for (int keptInter = intermediate; keptInter >= options.MinIntermediate; keptInter--)
        {
            int[] kept = ranked.Take(keptInter).OrderBy(j => j).ToArray();
            var estimate = MeasureStack(stackRepId, carried, sliced, keptInter);
            var newFfn = RebuildFfn(ffn, keptInter);
            var newSurface = RebuildSurface(surface, newFfn);
            var newGraph = BuildGraph(graph, component, newSurface);
            long graphLen = JsonSerializer.SerializeToUtf8Bytes(newGraph, ViJson.Options).Length;
            var indexEstimate = BuildIndex(
                container.Index, stackEntry, stackRepId, keptInter, intermediate, options,
                tensorCount, estimate.PayloadBytes,
                payloadSha: new string('0', 64), segmentSha: new string('0', 64));
            long indexLen = JsonSerializer.SerializeToUtf8Bytes(indexEstimate, ViJson.Options).Length;
            long estimatedTotal = copiedBytes + estimate.SegmentBytes + indexLen + graphLen;
            if (estimatedTotal > options.TargetBytes)
            {
                floor = Math.Max(floor, estimatedTotal);
                continue;
            }

            var report = Commit(
                container, outDir, graph, component, surface, ffn, stackRepId, stackEntry,
                carried, sliced, kept, keptInter, intermediate, options, copiedBytes, inputTotal,
                gated, drafterPresent, tokens.Length);
            if (report is not null)
            {
                return report;
            }
            floor = Math.Max(floor, estimatedTotal);
        }

        throw new MergeException(
            $"cannot reach target-bytes '{FormatBytes(options.TargetBytes)}': " +
            $"at --min-intermediate {options.MinIntermediate} the floor is {FormatBytes(floor)} " +
            $"(from {intermediate} FFN channels down to {options.MinIntermediate})");
    }

    // ── ranking / scoring ─────────────────────────────────────────────────

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

    // ── container / tensor helpers ────────────────────────────────────────

    private sealed class SlicedFfn
    {
        public SegmentTensor? Gate;
        public required SegmentTensor Up;
        public required SegmentTensor Down;
    }

    private static SegmentTensor Lookup(Dictionary<string, SegmentTensor> byName, string name, string segmentPath)
    {
        if (byName.TryGetValue(name, out var tensor))
        {
            return tensor;
        }
        throw new MergeException($"segment '{segmentPath}' records no tensor '{name}' — the plan/segment disagree");
    }

    private static int Elem(SegmentTensor tensor) => DtypeExtensions.FromLabel(tensor.Dtype).ElementSize();

    /// <summary>sliced-byte helpers (same byte-granular row/column gather as
    /// moe-ify): rows keep a channel for gate/up (<c>[I, H]</c>), columns for
    /// down (<c>[H, I]</c>).</summary>
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

    private sealed record SegmentEstimate(long SegmentBytes, long PayloadBytes);

    /// <summary>Exact serialised size of the rebuilt stack segment at a
    /// given kept width, without reading any payload: the header JSON is
    /// serialised exactly as <see cref="SegmentWriter.Write"/> would (sorted
    /// names, 16-byte padding) and payload lengths are summed from the slice
    /// geometry.</summary>
    private static SegmentEstimate MeasureStack(
        string repId, List<SegmentTensor> carried, Dictionary<int, SlicedFfn> sliced, int keptInter)
    {
        var desc = new List<(string Name, string Dtype, long[] Shape, long Len)>(carried.Count + sliced.Count * 3);
        foreach (var t in carried)
        {
            desc.Add((t.Name, t.Dtype, t.Shape, t.Len));
        }
        foreach (var sf in sliced.Values)
        {
            int hidden = (int)sf.Up.Shape[1];
            if (sf.Gate is { } gate)
            {
                desc.Add((gate.Name, gate.Dtype, new long[] { keptInter, hidden }, (long)keptInter * hidden * Elem(gate)));
            }
            desc.Add((sf.Up.Name, sf.Up.Dtype, new long[] { keptInter, hidden }, (long)keptInter * hidden * Elem(sf.Up)));
            desc.Add((sf.Down.Name, sf.Down.Dtype, new long[] { hidden, keptInter }, (long)hidden * keptInter * Elem(sf.Down)));
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

    private static FfnSurface RebuildFfn(FfnSurface source, int keptIntermediate) => new()
    {
        IntermediateSize = keptIntermediate,
        Activation = source.Activation,
        FfnType = source.FfnType,
        GatePolicy = source.GatePolicy,
        Moe = source.Moe,
    };

    private static ExecutionSurface RebuildSurface(ExecutionSurface source, FfnSurface newFfn) => new()
    {
        ContextLength = source.ContextLength,
        Attention = source.Attention,
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
        StarElasticOptions options,
        int tensorCount,
        long payloadBytes,
        string payloadSha,
        string segmentSha)
    {
        var reps = new Dictionary<string, RepresentationEntry>(StringComparer.Ordinal);
        foreach (var (repId, entry) in source.Representations)
        {
            reps[repId] = repId == stackRepId
                ? new RepresentationEntry
                {
                    Object = entry.Object,
                    Encoding = entry.Encoding,
                    Segment = entry.Segment,
                    TensorCount = tensorCount,
                    PayloadBytes = payloadBytes,
                    PayloadSha256 = payloadSha,
                    SegmentSha256 = segmentSha,
                    CompiledFrom = entry.CompiledFrom,
                }
                : entry;
        }

        var extra = source.Extra is null
            ? new Dictionary<string, JsonElement>()
            : new Dictionary<string, JsonElement>(source.Extra, StringComparer.Ordinal);
        extra["prune"] = JsonSerializer.SerializeToElement(new
        {
            Approach = "star-elastic",
            Axis = "ffn_channels",
            FromIntermediate = originalIntermediate,
            ToIntermediate = keptIntermediate,
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
        List<SegmentTensor> carried,
        Dictionary<int, SlicedFfn> sliced,
        int[] kept,
        int keptIntermediate,
        int originalIntermediate,
        StarElasticOptions options,
        long copiedBytes,
        long inputTotal,
        bool gated,
        bool drafterPresent,
        int sampledTokens)
    {
        try
        {
            Directory.CreateDirectory(outDir);

            // ── rebuilt stack segment (sliced FFN + verbatim carried) ───
            var named = new List<NamedTensorData>(carried.Count + sliced.Count * 3);
            using (var sourceSegment = SegmentFile.Open(Path.Combine(container.Root, stackEntry.Segment)))
            {
                foreach (var tensor in carried)
                {
                    named.Add(new NamedTensorData
                    {
                        Name = tensor.Name,
                        Dtype = DtypeExtensions.FromLabel(tensor.Dtype),
                        Shape = tensor.Shape,
                        Data = sourceSegment.ReadBytes(tensor.Name),
                    });
                }
                foreach (var sf in sliced.Values)
                {
                    int hidden = (int)sf.Up.Shape[1];
                    if (sf.Gate is { } gate)
                    {
                        named.Add(new NamedTensorData
                        {
                            Name = gate.Name,
                            Dtype = DtypeExtensions.FromLabel(gate.Dtype),
                            Shape = new long[] { keptIntermediate, hidden },
                            Data = GatherRows(sourceSegment.ReadBytes(gate.Name), hidden, Elem(gate), kept),
                        });
                    }
                    named.Add(new NamedTensorData
                    {
                        Name = sf.Up.Name,
                        Dtype = DtypeExtensions.FromLabel(sf.Up.Dtype),
                        Shape = new long[] { keptIntermediate, hidden },
                        Data = GatherRows(sourceSegment.ReadBytes(sf.Up.Name), hidden, Elem(sf.Up), kept),
                    });
                    named.Add(new NamedTensorData
                    {
                        Name = sf.Down.Name,
                        Dtype = DtypeExtensions.FromLabel(sf.Down.Dtype),
                        Shape = new long[] { hidden, keptIntermediate },
                        Data = GatherCols(sourceSegment.ReadBytes(sf.Down.Name),
                            (int)sf.Down.Shape[1], hidden, Elem(sf.Down), kept),
                    });
                }
            }

            string stackDir = Path.Combine(outDir, Path.GetDirectoryName(stackEntry.Segment)!);
            Directory.CreateDirectory(stackDir);
            var stackWrite = SegmentWriter.Write(Path.Combine(outDir, stackEntry.Segment), stackRepId, named);

            // ── rebuilt metadata ────────────────────────────────────────
            var newFfn = RebuildFfn(ffn, keptIntermediate);
            var newSurface = RebuildSurface(surface, newFfn);
            var newGraph = BuildGraph(graph, component, newSurface);
            File.WriteAllBytes(Path.Combine(outDir, container.Index.SystemGraph!),
                JsonSerializer.SerializeToUtf8Bytes(newGraph, ViJson.Options));

            var newIndex = BuildIndex(
                container.Index, stackEntry, stackRepId, keptIntermediate, originalIntermediate, options,
                named.Count, stackWrite.PayloadBytes,
                stackWrite.PayloadSha256Hex, stackWrite.SegmentSha256Hex);
            File.WriteAllBytes(Path.Combine(outDir, IndexFile),
                JsonSerializer.SerializeToUtf8Bytes(newIndex, ViJson.Options));

            // ── exact total before copying anything else ────────────────
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
                $"ranked {originalIntermediate} FFN channels by Σ|x·W_upᵀ| over {sampledTokens} sampled positions",
                gated
                    ? "gated FFN — gate/up rows and down columns sliced to the same kept channels"
                    : "dense FFN — up rows and down columns sliced to the same kept channels",
                "homogeneous width — the same channels are kept in every dense layer (AMQL's surface is per-model, not per-layer)",
            };
            if (keptIntermediate == originalIntermediate)
            {
                notes.Add("already within the budget — no channels dropped, the stack was re-encoded unchanged");
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
                DenseLayers: sliced.Count,
                KeptChannels: kept,
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