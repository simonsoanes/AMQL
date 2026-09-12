using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>How a prune ranks the decoder layers to drop.</summary>
public enum PruneApproach
{
    /// <summary>token-map.json layer provenance: layers whose tensors were
    /// grown (zero-padded from the other model) during the merge drop
    /// first, then position descending — deterministic, corpus-free.</summary>
    Provenance,

    /// <summary>One forward pass over a corpus: per-layer residual delta;
    /// layers that change the hidden state least drop first (a
    /// contribution-magnitude proxy, single pass, no ablation loop).</summary>
    Corpus,

    /// <summary>Seeded uniform random layer order — the ablation baseline.</summary>
    Random,
}

/// <summary>Prune inputs: the byte budget for the whole output container,
/// the layer-ranking approach, and the approach controls.</summary>
public sealed record PruneOptions(
    long TargetBytes,
    PruneApproach Approach = PruneApproach.Provenance,
    int Seed = 42,
    string? CorpusText = null,
    int SampleCap = 8192,
    int MinLayers = 1);

/// <summary>Outcome of one prune: the sizes, the layer counts, and which
/// original layers were dropped (old indices, in drop order).</summary>
public sealed record PruneReport(
    string OutDir,
    string Model,
    long OriginalBytes,
    long TargetBytes,
    long FinalBytes,
    int OriginalLayers,
    int KeptLayers,
    IReadOnlyList<int> DroppedLayers,
    string Approach,
    IReadOnlyList<string> Notes);

/// <summary>
/// Prunes a container down to a byte budget by dropping whole decoder
/// layers. The stack segment is rebuilt without the dropped layers (the
/// kept layers are renumbered 0..K-1), the vocabulary/embedding/head and
/// every other file are untouched (byte-copied), and the graph, index and
/// token-map are rewritten to match — so the result is a complete,
/// mergeable container: import, export and the planner all work on it
/// unchanged.
///
/// The drop order is the approach's ranking; the prune drops the smallest
/// prefix of that order that gets the whole output container under the
/// budget (never below <c>MinLayers</c>). Selection predicts each
/// candidate's exact on-disk size (the rebuilt stack segment and JSON
/// files are sized before anything is written), and the commit verifies
/// the exact total again before copying, so an over-budget output is
/// impossible rather than merely unlikely.
/// </summary>
public static class LayerPruner
{
    private const string IndexFile = "index.json";

    public static PruneReport Prune(string containerDir, string outDir, PruneOptions options)
    {
        if (Directory.Exists(outDir))
        {
            throw new MergeException($"prune output '{outDir}' already exists");
        }
        if (options.TargetBytes <= 0)
        {
            throw new MergeException("--target-bytes must be a positive byte count");
        }
        if (options.MinLayers < 1)
        {
            throw new MergeException("--min-layers must be at least 1");
        }

        using var container = Vindex3Container.Open(containerDir);
        var graph = container.Graph
            ?? throw new MergeException($"'{containerDir}' records no system graph — nothing to prune");
        // An MTP drafter's trunk is a self-contained COPY in its own
        // segment (generate-mtp materialises the last full-attention
        // layer); pruning renumbers only the primary stack, so the
        // container stays structurally valid — the drafter's trunk just
        // no longer matches any live layer. Recorded, not refused.
        bool drafterPresent = graph.Components.Any(c => c.Role == ComponentRole.Drafter);

        var component = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.PrimaryText)
            ?? throw new MergeException("the container records no primary text component — nothing to prune");
        var attention = component.Attention
            ?? throw new MergeException($"component '{component.Id}' carries no per-layer attention table");
        if (attention.Count != component.NumLayers)
        {
            throw new MergeException(
                $"component '{component.Id}' has {attention.Count} attention rows for {component.NumLayers} layers — inconsistent, refusing");
        }
        if (graph.Edges.Any(e => e.ProducerComponent == component.Id && e.ProducerLayers.Count > 0))
        {
            throw new MergeException(
                "an edge leaves the text component by explicit layer numbers; pruning renumbers layers " +
                "and cannot keep such an edge honest");
        }

        var stack = graph.Objects.FirstOrDefault(o => o.Component == component.Id && o.Kind == ObjectKind.DecoderStack)
            ?? throw new MergeException($"component '{component.Id}' owns no decoder_stack object");
        if (stack.Representations.Count != 1)
        {
            throw new MergeException(
                $"decoder_stack has {stack.Representations.Count} representations — refusing to rewrite a multi-representation stack");
        }
        string stackRepId = $"{stack.Id}@{stack.Representations[0].Encoding}";
        if (!container.Index.Representations.TryGetValue(stackRepId, out var stackEntry))
        {
            throw new MergeException($"index records no representation '{stackRepId}'");
        }

        int layers = component.NumLayers;
        if (options.MinLayers > layers)
        {
            throw new MergeException($"cannot keep {options.MinLayers} layers of a {layers}-layer stack");
        }

        // ── the stack segment's layer census ─────────────────────────────
        string stackFilePath = Path.Combine(container.Root, stackEntry.Segment);
        List<SegmentTensor> stackTensors;
        using (var segment = SegmentFile.Open(stackFilePath))
        {
            stackTensors = segment.Header.Tensors.ToList();
        }

        var byLayer = new Dictionary<int, List<SegmentTensor>>();
        var carried = new List<SegmentTensor>(); // tensors with no numeric layer prefix — never dropped
        foreach (var tensor in stackTensors)
        {
            int layer = LayerIndex(tensor.Name);
            if (layer < 0)
            {
                carried.Add(tensor);
            }
            else
            {
                if (!byLayer.TryGetValue(layer, out var list))
                {
                    byLayer[layer] = list = new List<SegmentTensor>();
                }
                list.Add(tensor);
            }
        }
        for (int l = 0; l < layers; l++)
        {
            if (!byLayer.TryGetValue(l, out var list) || list.Count == 0)
            {
                throw new MergeException($"decoder stack carries no tensors for layer {l} — incomplete, refusing");
            }
        }
        if (byLayer.Keys.Any(l => l >= layers))
        {
            throw new MergeException("decoder stack carries tensors beyond the declared layer count — refusing");
        }

        // ── files the prune rebuilds vs byte-copies ──────────────────────
        string graphRel = container.Index.SystemGraph
            ?? throw new MergeException("container graph is present but index records no system graph path");
        string? manifestPath = container.Index.TokenMap is { } tm ? Path.Combine(container.Root, tm) : null;
        var rebuilt = new HashSet<string>(StringComparer.Ordinal)
        {
            Path.Combine(container.Root, IndexFile),
            Path.Combine(container.Root, graphRel),
            stackFilePath,
        };
        if (manifestPath is not null)
        {
            rebuilt.Add(manifestPath);
        }

        long inputTotal = DirectorySize(containerDir);
        long copiedBytes = inputTotal;
        long manifestInLen = 0;
        foreach (var path in rebuilt)
        {
            long len = new FileInfo(path).Length;
            copiedBytes -= len;
            if (manifestPath is not null && path == manifestPath)
            {
                manifestInLen = len;
            }
        }

        JsonArray? manifestLayerTable = null;
        if (manifestPath is not null)
        {
            var manifestNode = JsonNode.Parse(File.ReadAllBytes(manifestPath))
                ?? throw new MergeException($"'{container.Index.TokenMap}' is empty");
            manifestLayerTable = manifestNode["layer_table"]?.AsArray()
                ?? throw new MergeException($"token-map '{container.Index.TokenMap}' has no layer_table");
            if (manifestLayerTable.Count != layers)
            {
                throw new MergeException(
                    $"token-map layer_table has {manifestLayerTable.Count} rows for {layers} layers — inconsistent, refusing");
            }
        }

        // ── drop ranking (ascending importance → dropped first) ──────────
        int[] dropOrder = options.Approach switch
        {
            PruneApproach.Provenance => ProvenanceRank(manifestLayerTable, layers),
            PruneApproach.Random => RandomRank(layers, options.Seed),
            PruneApproach.Corpus => CorpusRank(container, component.Id, layers, options),
            _ => throw new MergeException($"unknown prune approach '{options.Approach}'"),
        };

        // ── selection + exact commit ─────────────────────────────────────
        // Each candidate is the smallest drop prefix whose estimated total
        // is under the budget; the commit re-verifies the exact total
        // before copying, and on overflow advances to the next drop count
        // (at most one corrective step in practice).
        int maxDrop = layers - options.MinLayers;
        long exactFloor = -1;
        for (int drop = 0; drop <= maxDrop; drop++)
        {
            var (dropped, keptOld) = DropMask(dropOrder, drop, layers);
            long keptPayload = carried.Sum(t => t.Len) + keptOld.Sum(l => byLayer[l].Sum(t => t.Len));
            int keptTensorCount = carried.Count + keptOld.Sum(l => byLayer[l].Count);
            long stackEstimate = StackSegmentFileBytes(stackRepId, keptOld, byLayer, carried);
            long indexEstimate = IndexJsonBytes(
                stackEntry, stackRepId, keptOld.Length, dropped, options, container,
                drafterPresent,
                keptTensorCount: keptTensorCount,
                keptPayloadBytes: keptPayload,
                payloadSha: null, segmentSha: null);
            long graphEstimate = GraphJsonBytes(graph, component, attention, keptOld);
            long manifestEstimate = manifestLayerTable is null
                ? 0
                : manifestInLen - DroppedLayerTableBytes(manifestLayerTable, dropped);
            long estimatedTotal = copiedBytes + stackEstimate + indexEstimate + graphEstimate + manifestEstimate;
            if (estimatedTotal > options.TargetBytes)
            {
                exactFloor = estimatedTotal;
                continue;
            }

            var (report, exactTotal) = Commit(
                containerDir, outDir, container, graph, component, attention, stackRepId, stackEntry,
                byLayer, carried, dropOrder, drop, dropped, keptOld, manifestPath, manifestLayerTable,
                options, copiedBytes, inputTotal, drafterPresent);
            if (report is not null)
            {
                return report;
            }
            exactFloor = exactTotal; // exact total overflowed — drop one more
        }

        throw new MergeException(
            $"cannot reach target-bytes '{FormatBytes(options.TargetBytes)}': " +
            $"at --min-layers {options.MinLayers} the floor is {FormatBytes(exactFloor)} " +
            $"({layers - options.MinLayers} layer(s) droppable of {layers})");
    }

    // ── ranking approaches ────────────────────────────────────────────────

    /// <summary>token-map layer provenance: layers that contain grown
    /// tensors (zero-padded for the other model during the merge) are the
    /// least authentic and drop first; everything else ties and drops by
    /// position descending (keep the input-proximal side of the stack).
    /// Without a token-map every layer ties — pure position descending.
    /// </summary>
    private static int[] ProvenanceRank(JsonArray? layerTable, int layers)
    {
        var grown = new HashSet<int>();
        if (layerTable is not null)
        {
            foreach (var entry in layerTable)
            {
                if (entry is not JsonObject obj)
                {
                    continue;
                }
                int layer = obj["layer"]?.GetValue<int>() ?? -1;
                if (layer < 0 || layer >= layers)
                {
                    throw new MergeException($"token-map layer_table names layer {layer} outside 0..{layers - 1}");
                }
                if (obj["tensors"] is JsonObject tensors &&
                    tensors.Any(p => p.Value?["operation"]?.GetValue<string>() == "grown"))
                {
                    grown.Add(layer);
                }
            }
        }
        return Enumerable.Range(0, layers)
            .OrderBy(l => grown.Contains(l) ? 1 : 0)
            .ThenByDescending(l => l)
            .ToArray();
    }

    /// <summary>Seeded uniform random layer order — deterministic under a
    /// fixed <c>--seed</c>, the ablation baseline.</summary>
    private static int[] RandomRank(int layers, int seed)
    {
        var order = Enumerable.Range(0, layers).ToArray();
        var rng = new Random(seed);
        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
        return order;
    }

    /// <summary>Corpus approach: plan the full stack and run one forward
    /// pass over the corpus, accumulating each layer's residual delta
    /// energy (<c>‖h(l+1) − h(l)‖²</c> per position, mean over positions).
    /// Layers whose output moves the hidden state least drop first —
    /// removing them changes the model least, the one-shot
    /// contribution-magnitude criterion.</summary>
    private static int[] CorpusRank(Vindex3Container container, string componentId, int layers, PruneOptions options)
    {
        if (options.CorpusText is null)
        {
            throw new MergeException("--approach corpus requires '--text <corpus.txt>'");
        }

        int[] tokens;
        try
        {
            var ids = HfTokenizer.FromModelDir(container.Root).EncodeToIds(options.CorpusText);
            int cap = options.SampleCap > 0 ? Math.Min(options.SampleCap, ids.Count) : ids.Count;
            tokens = ids.Take(cap).ToArray();
        }
        catch (TokenizerException e)
        {
            throw new MergeException(
                "--approach corpus needs a BPE tokenizer.json beside the container: " + e.Message, e);
        }
        if (tokens.Length == 0)
        {
            throw new MergeException("the corpus encodes to zero tokens — nothing to score");
        }

        ComponentOpPlan plan;
        try
        {
            using var store = container.CreateOperandStore();
            plan = Planner.Plan(container, componentId, store);
        }
        catch (UnsupportedOperatorException e)
        {
            throw new MergeException(
                "--approach corpus needs a runnable stack, and the planner refuses: " + e.Message, e);
        }
        if (plan.Layers.Count != layers)
        {
            throw new MergeException(
                $"planned {plan.Layers.Count} layers for a {layers}-layer component — refusing to score a mismatch");
        }

        var scores = new double[layers];
        var counts = new long[layers];
        using var runtimeStore = container.CreateOperandStore();
        var runtime = new GenericRuntime(plan, runtimeStore);
        for (int t = 0; t < tokens.Length; t++)
        {
            var hidden = runtime.Embed(new[] { tokens[t] });
            var queryPositions = new[] { t };
            var kvPositions = Enumerable.Range(0, t + 1).ToArray();
            for (int l = 0; l < layers; l++)
            {
                var before = hidden.Clone();
                hidden = runtime.RunLayerInternal(hidden, l, queryPositions, kvPositions, appendKv: true);
                double sq = 0;
                for (int i = 0; i < hidden.Data.Length; i++)
                {
                    double d = hidden.Data[i] - before.Data[i];
                    sq += d * d;
                }
                scores[l] += sq / hidden.Data.Length;
                counts[l]++;
            }
        }
        for (int l = 0; l < layers; l++)
        {
            scores[l] = counts[l] == 0 ? double.PositiveInfinity : scores[l] / counts[l];
        }
        return Enumerable.Range(0, layers)
            .OrderBy(l => scores[l])
            .ThenByDescending(l => l)
            .ToArray();
    }

    // ── candidate selection + commit ──────────────────────────────────────

    private static (bool[] Dropped, int[] KeptOld) DropMask(int[] dropOrder, int dropCount, int layers)
    {
        var dropped = new bool[layers];
        for (int i = 0; i < dropCount; i++)
        {
            dropped[dropOrder[i]] = true;
        }
        var keptOld = new int[layers - dropCount];
        int k = 0;
        for (int l = 0; l < layers; l++)
        {
            if (!dropped[l])
            {
                keptOld[k++] = l;
            }
        }
        return (dropped, keptOld);
    }

    /// <summary>Exact bytes of the rebuilt stack segment for a kept-layer
    /// set: the header JSON is serialised exactly as <see cref="SegmentWriter"/>
    /// would (kept tensors renumbered to their new layer index, sorted by
    /// name, 16-byte payload alignment), payload lengths from the
    /// segment's own table. No payload bytes are read.</summary>
    private static long StackSegmentFileBytes(
        string stackRepId, int[] keptOld, Dictionary<int, List<SegmentTensor>> byLayer, List<SegmentTensor> carried)
    {
        var ordered = new List<(string Name, SegmentTensor Tensor)>(carried.Count + keptOld.Length * 4);
        foreach (var tensor in carried)
        {
            ordered.Add((tensor.Name, tensor)); // carried tensors keep their names
        }
        for (int j = 0; j < keptOld.Length; j++)
        {
            int old = keptOld[j];
            foreach (var tensor in byLayer[old])
            {
                ordered.Add((RenameLayerTensor(tensor.Name, old, j), tensor));
            }
        }
        ordered.Sort((a, b) => StringComparer.Ordinal.Compare(a.Name, b.Name));

        long payload = 0;
        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WriteNumber("schema", SegmentFormat.CurrentSchema);
            writer.WriteString("representation", stackRepId);
            writer.WritePropertyName("tensors");
            writer.WriteStartArray();
            long cursor = 0;
            foreach (var (name, tensor) in ordered)
            {
                writer.WriteStartObject();
                writer.WriteString("name", name);
                writer.WriteString("dtype", tensor.Dtype);
                writer.WritePropertyName("shape");
                writer.WriteStartArray();
                foreach (var dim in tensor.Shape)
                {
                    writer.WriteNumberValue(dim);
                }
                writer.WriteEndArray();
                writer.WriteNumber("offset", cursor);
                writer.WriteNumber("len", tensor.Len);
                writer.WriteEndObject();
                cursor += tensor.Len;
            }
            writer.WriteEndArray();
            writer.WriteEndObject();
            writer.Flush();
            payload = cursor;
        }
        var headerJson = buffer.ToArray();
        int storedLength = headerJson.Length + SegmentPad(headerJson.Length);
        return SegmentFormat.HeaderLengthBytes + storedLength + payload;
    }

    private static int SegmentPad(int headerJsonLength) =>
        (SegmentFormat.PayloadAlignment - ((SegmentFormat.HeaderLengthBytes + headerJsonLength) % SegmentFormat.PayloadAlignment)) %
        SegmentFormat.PayloadAlignment;

    /// <summary>Exact serialised size of the candidate index (the prune
    /// provenance record included; the two hashes are fixed-width 64-char
    /// hex, so placeholder values do not change the length).</summary>
    private static long IndexJsonBytes(
        RepresentationEntry stackEntry,
        string stackRepId,
        int keptLayers,
        bool[] dropped,
        PruneOptions options,
        Vindex3Container container,
        bool drafterPresent,
        int keptTensorCount,
        long keptPayloadBytes,
        string? payloadSha,
        string? segmentSha)
    {
        var index = BuildIndex(stackEntry, stackRepId, keptLayers, dropped, options, container,
            drafterPresent, keptTensorCount, keptPayloadBytes,
            payloadSha ?? new string('0', 64), segmentSha ?? new string('0', 64));
        return JsonSerializer.SerializeToUtf8Bytes(index, ViJson.Options).Length;
    }

    private static Vindex3Index BuildIndex(
        RepresentationEntry stackEntry,
        string stackRepId,
        int keptLayers,
        bool[] dropped,
        PruneOptions options,
        Vindex3Container container,
        bool drafterPresent,
        int keptTensorCount,
        long keptPayloadBytes,
        string payloadSha,
        string segmentSha)
    {
        var source = container.Index;
        var reps = new Dictionary<string, RepresentationEntry>(StringComparer.Ordinal);
        foreach (var (repId, entry) in source.Representations)
        {
            reps[repId] = repId == stackRepId
                ? new RepresentationEntry
                {
                    Object = entry.Object,
                    Encoding = entry.Encoding,
                    Segment = entry.Segment,
                    TensorCount = keptTensorCount,
                    PayloadBytes = keptPayloadBytes,
                    PayloadSha256 = payloadSha,
                    SegmentSha256 = segmentSha,
                    CompiledFrom = entry.CompiledFrom,
                }
                : entry;
        }

        var droppedLayers = Enumerable.Range(0, dropped.Length).Where(l => dropped[l]).ToArray();

        var extra = source.Extra is null
            ? new Dictionary<string, JsonElement>()
            : new Dictionary<string, JsonElement>(source.Extra, StringComparer.Ordinal);
        extra["prune"] = JsonSerializer.SerializeToElement(new
        {
            Approach = options.Approach.ToString().ToLowerInvariant(),
            FromLayers = source.NumLayers,
            ToLayers = keptLayers,
            DroppedLayers = droppedLayers,
            TargetBytes = options.TargetBytes,
            DrafterStale = drafterPresent,
        }, ViJson.Options);

        return new Vindex3Index
        {
            Version = source.Version,
            Model = source.Model,
            Family = source.Family,
            HiddenSize = source.HiddenSize,
            NumLayers = keptLayers,
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

    private static long GraphJsonBytes(SystemGraph graph, Component component, List<AttentionLayerPolicy> attention, int[] keptOld)
    {
        var rebuilt = BuildGraph(graph, component, attention, keptOld);
        return JsonSerializer.SerializeToUtf8Bytes(rebuilt, ViJson.Options).Length;
    }

    private static SystemGraph BuildGraph(SystemGraph source, Component primary, List<AttentionLayerPolicy> attention, int[] keptOld)
    {
        var keptPolicies = new List<AttentionLayerPolicy>(keptOld.Length);
        foreach (var old in keptOld)
        {
            keptPolicies.Add(attention[old]);
        }
        var components = new List<Component>(source.Components.Count);
        foreach (var component in source.Components)
        {
            if (component.Id == primary.Id)
            {
                components.Add(new Component
                {
                    Id = component.Id,
                    Role = component.Role,
                    SourceArtifact = component.SourceArtifact,
                    NumLayers = keptOld.Length,
                    HiddenSize = component.HiddenSize,
                    Attention = keptPolicies,
                    Execution = component.Execution,
                    Perception = component.Perception,
                });
            }
            else
            {
                components.Add(component);
            }
        }
        return new SystemGraph
        {
            Schema = source.Schema,
            Components = components,
            Objects = new List<LogicalObject>(source.Objects),
            Edges = new List<HiddenStateEdge>(source.Edges),
        };
    }

    /// <summary>Estimate of the rebuilt token-map's size: the input file
    /// minus the exact serialised bytes of the dropped layer-table entries
    /// (the tables' JSON formatting otherwise round-trips verbatim).</summary>
    private static long DroppedLayerTableBytes(JsonArray layerTable, bool[] dropped)
    {
        long total = 0;
        for (int l = 0; l < dropped.Length; l++)
        {
            if (!dropped[l])
            {
                continue;
            }
            var node = layerTable.FirstOrDefault(e => e is JsonObject o && o["layer"]?.GetValue<int>() == l);
            if (node is not null)
            {
                total += JsonSerializer.SerializeToUtf8Bytes(node, ViJson.Options).Length + 1; // entry + comma
            }
        }
        return total;
    }

    /// <summary>Writes the candidate: the stack segment first (its hashes
    /// feed the index), then the index/graph/token-map, then byte-copies
    /// of everything else. The exact total is verified before the copies
    /// are committed — on overflow the output tree is removed and the
    /// caller retries with one more drop; the returned exact total is
    /// whatever was measured (used as the floor in the failure message).
    /// </summary>
    private static (PruneReport? Report, long ExactTotal) Commit(
        string containerDir,
        string outDir,
        Vindex3Container container,
        SystemGraph graph,
        Component component,
        List<AttentionLayerPolicy> attention,
        string stackRepId,
        RepresentationEntry stackEntry,
        Dictionary<int, List<SegmentTensor>> byLayer,
        List<SegmentTensor> carried,
        int[] dropOrder,
        int dropCount,
        bool[] dropped,
        int[] keptOld,
        string? manifestPath,
        JsonArray? manifestLayerTable,
        PruneOptions options,
        long copiedBytes,
        long inputTotal,
        bool drafterPresent)
    {
        try
        {
            Directory.CreateDirectory(outDir);

            // ── rebuilt stack segment (final tensor names known up front) ─
            var named = new List<NamedTensorData>(carried.Count + byLayer.Count * 4);
            using (var segment = SegmentFile.Open(Path.Combine(container.Root, stackEntry.Segment)))
            {
                foreach (var tensor in carried)
                {
                    named.Add(new NamedTensorData
                    {
                        Name = tensor.Name,
                        Dtype = DtypeExtensions.FromLabel(tensor.Dtype),
                        Shape = tensor.Shape,
                        Data = segment.ReadBytes(tensor.Name),
                    });
                }
                for (int j = 0; j < keptOld.Length; j++)
                {
                    int old = keptOld[j];
                    foreach (var tensor in byLayer[old])
                    {
                        named.Add(new NamedTensorData
                        {
                            Name = RenameLayerTensor(tensor.Name, old, j),
                            Dtype = DtypeExtensions.FromLabel(tensor.Dtype),
                            Shape = tensor.Shape,
                            Data = segment.ReadBytes(tensor.Name),
                        });
                    }
                }
            }
            var stackDir = Path.Combine(outDir, Path.GetDirectoryName(stackEntry.Segment)!);
            Directory.CreateDirectory(stackDir);
            var stackWrite = SegmentWriter.Write(Path.Combine(outDir, stackEntry.Segment), stackRepId, named);

            // ── rebuilt metadata ─────────────────────────────────────────
            var index = BuildIndex(stackEntry, stackRepId, keptOld.Length, dropped, options, container,
                drafterPresent, named.Count, stackWrite.PayloadBytes,
                stackWrite.PayloadSha256Hex, stackWrite.SegmentSha256Hex);
            File.WriteAllBytes(Path.Combine(outDir, IndexFile),
                JsonSerializer.SerializeToUtf8Bytes(index, ViJson.Options));

            var newGraph = BuildGraph(graph, component, attention, keptOld);
            File.WriteAllBytes(Path.Combine(outDir, container.Index.SystemGraph!),
                JsonSerializer.SerializeToUtf8Bytes(newGraph, ViJson.Options));

            long manifestLen = 0;
            if (manifestPath is not null)
            {
                var manifest = JsonNode.Parse(File.ReadAllBytes(manifestPath))
                    ?? throw new MergeException($"'{Path.GetFileName(manifestPath)}' is empty");
                RebuildLayerTable(manifest, keptOld);
                var manifestBytes = JsonSerializer.SerializeToUtf8Bytes(manifest, ViJson.Options);
                manifestLen = manifestBytes.Length;
                File.WriteAllBytes(Path.Combine(outDir, container.Index.TokenMap!), manifestBytes);
            }

            // ── exact total check before copying anything else ────────────
            long exactTotal = copiedBytes
                + new FileInfo(Path.Combine(outDir, stackEntry.Segment)).Length
                + new FileInfo(Path.Combine(outDir, IndexFile)).Length
                + new FileInfo(Path.Combine(outDir, container.Index.SystemGraph!)).Length
                + manifestLen;
            if (exactTotal > options.TargetBytes)
            {
                Directory.Delete(outDir, recursive: true);
                return (null, exactTotal);
            }

            // ── byte-copy everything else ────────────────────────────────
            foreach (var file in Directory.EnumerateFiles(containerDir, "*", SearchOption.AllDirectories))
            {
                // Index paths use forward slashes; normalise the relative
                // path so the rebuilt files (stack segment included) are
                // never copied over themselves.
                string rel = Path.GetRelativePath(containerDir, file).Replace('\\', '/');
                bool rebuiltFile = rel.Equals(IndexFile, StringComparison.Ordinal)
                    || rel.Equals(container.Index.SystemGraph, StringComparison.Ordinal)
                    || rel.Equals(container.Index.TokenMap, StringComparison.Ordinal)
                    || rel.Equals(stackEntry.Segment, StringComparison.Ordinal);
                if (rebuiltFile)
                {
                    continue;
                }
                string dest = Path.Combine(outDir, rel);
                Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
                File.Copy(file, dest);
            }

            var notes = new List<string>();
            if (dropCount == 0)
            {
                notes.Add("already within the budget — copied unchanged");
            }
            if (carried.Count > 0)
            {
                notes.Add($"{carried.Count} carried (non-layer) stack tensors retained verbatim");
            }
            if (options.Approach == PruneApproach.Corpus)
            {
                notes.Add("corpus ranking is one-shot — layers are scored on the full stack, not re-scored after each drop");
            }
            if (options.Approach == PruneApproach.Provenance && manifestLayerTable is null)
            {
                notes.Add("no token-map recorded — provenance ties across all layers, so the drop order is positional " +
                          "(top of the stack first)");
            }
            if (drafterPresent)
            {
                notes.Add("an MTP drafter is present and its trunk is now stale (it was copied from the original last " +
                          "full-attention layer) — regenerate it with generate-mtp, or export without it");
            }

            var droppedLayers = new List<int>(dropCount);
            for (int i = 0; i < dropCount; i++)
            {
                droppedLayers.Add(dropOrder[i]);
            }

            return (new PruneReport(
                OutDir: outDir,
                Model: container.Index.Model,
                OriginalBytes: inputTotal,
                TargetBytes: options.TargetBytes,
                FinalBytes: exactTotal,
                OriginalLayers: container.Index.NumLayers,
                KeptLayers: keptOld.Length,
                DroppedLayers: droppedLayers,
                Approach: options.Approach.ToString().ToLowerInvariant(),
                Notes: notes), exactTotal);
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

    private static void RebuildLayerTable(JsonNode manifest, int[] keptOld)
    {
        var table = manifest["layer_table"]?.AsArray();
        if (table is null)
        {
            return;
        }
        var rebuilt = new JsonArray();
        foreach (var entry in table)
        {
            if (entry is not JsonObject obj)
            {
                continue;
            }
            int old = obj["layer"]?.GetValue<int>() ?? -1;
            int keptAt = Array.IndexOf(keptOld, old);
            if (keptAt < 0)
            {
                continue;
            }
            var tensors = new JsonObject();
            if (obj["tensors"] is JsonObject source)
            {
                foreach (var (name, info) in source)
                {
                    string newName = old != keptAt ? RenameLayerTensor(name, old, keptAt) : name;
                    tensors[newName] = info?.DeepClone();
                }
            }
            rebuilt.Add(new JsonObject
            {
                ["layer"] = keptAt,
                ["tensors"] = tensors,
            });
        }
        manifest["layer_table"] = rebuilt;
        if (manifest["scaffold"] is JsonObject scaffold)
        {
            scaffold["layers"] = keptOld.Length;
        }
    }

    // ── small helpers ──────────────────────────────────────────────────────

    private static int LayerIndex(string name)
    {
        int dot = name.IndexOf('.');
        if (dot <= 0)
        {
            return -1;
        }
        return int.TryParse(name.AsSpan(0, dot), out int layer) ? layer : -1;
    }

    private static string RenameLayerTensor(string name, int oldLayer, int newLayer)
    {
        int dot = name.IndexOf('.');
        return $"{newLayer}{name[dot..]}";
    }

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