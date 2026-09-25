using System.Text.Json;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>The outcome of one MoE-ification: the restructured container,
/// the routing geometry, and the dense-vs-MoE perplexity gate.</summary>
public sealed record MoeIfyReport(
    string OutDir,
    string Model,
    int Layers,
    int Experts,
    int TopK,
    int ExpertIntermediateSize,
    int SampledTokens,
    double DensePpl,
    double MoePpl,
    IReadOnlyList<string> Notes);

/// <summary>
/// MoE-ifies a dense transformer in the MoEfication lineage: sample FFN
/// input activations through the model, cluster each layer's intermediate
/// units by co-activation into K balanced experts, slice the gate/up rows
/// and down columns into per-expert tensors, materialise a per-layer
/// linear router (the cluster mean gate directions), and restructure the
/// container so the planner judges every FFN as routed
/// (<c>mlp.router.weight</c> + <c>mlp.experts.{e}.{gate,up,down}_proj.weight</c>)
/// — the exact layout the existing <see cref="FfnKernel.Routed"/> executes.
/// The output stays a normal container: exportable, and each expert carries
/// its own tensors for separate placement. Quality is gated by a
/// next-token perplexity comparison against the dense original.
/// </summary>
public static class MoeIfy
{
    public static MoeIfyReport Transform(
        string containerDir,
        string outDir,
        IReadOnlyList<int> clusterTokens,
        IReadOnlyList<int> evalTokens,
        int experts,
        int topK,
        ExpertRoutingPolicy policy = ExpertRoutingPolicy.SoftmaxThenSelect)
    {
        if (Directory.Exists(outDir))
        {
            throw new MergeException($"moе-ify output '{outDir}' already exists");
        }
        if (experts < 2)
        {
            throw new MergeException($"moе-ify needs at least 2 experts, got {experts}");
        }
        if (topK < 1 || topK > experts)
        {
            throw new MergeException($"top-k {topK} must be within 1..{experts}");
        }

        using var container = Vindex3Container.Open(containerDir);
        var graph = container.Graph ?? throw new MergeException("container records no system graph");
        var component = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.PrimaryText)
            ?? throw new MergeException("container records no primary text component");
        var surface = component.Execution
            ?? throw new MergeException($"component '{component.Id}' declares no execution surface");
        var ffn = surface.Ffn
            ?? throw new MergeException($"component '{component.Id}' declares no FFN surface");
        if (ffn.FfnType != FfnType.Gated)
        {
            throw new MergeException(
                $"moе-ify requires a gated FFN (gate_proj + up_proj), the surface declares {ffn.FfnType}");
        }
        int intermediate = ffn.IntermediateSize;
        if (intermediate % experts != 0)
        {
            throw new MergeException(
                $"{experts} experts do not divide the FFN intermediate {intermediate} — pick a divisor");
        }
        int expertIntermediate = intermediate / experts;
        int hidden = component.HiddenSize;

        var notes = new List<string>();

        // ── 1. sample the pre-FFN layer inputs ─────────────────────────
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, component.Id, store);
        var runtime = new GenericRuntime(plan, store);
        int layers = plan.Layers.Count;
        var stackId = graph.Objects
            .First(o => o.Component == component.Id && o.Kind == ObjectKind.DecoderStack).Id;

        var hfRows = new float[layers][];
        int[] captured = new int[layers];
        runtime.FfnInputCapture = (layer, hf) =>
        {
            var buffer = hfRows[layer] ??= new float[clusterTokens.Count * hidden];
            Array.Copy(hf.Data, 0, buffer, captured[layer] * hidden, hf.Data.Length);
            captured[layer] += hf.Rows;
        };
        foreach (var token in clusterTokens)
        {
            runtime.StepForward(token);
        }

        // ── 2. cluster each layer's units into balanced experts ────────
        var partitions = new int[layers][];
        for (int l = 0; l < layers; l++)
        {
            var parsed = plan.Layers[l].Ffn?.Dense?.Gate ??
                throw new MergeException($"layer {l} carries no dense gated FFN after planning");
            var gate = runtime.Weights.Matrix(parsed, intermediate, hidden);
            var hfStack = new Tensor2D(hfRows[l], clusterTokens.Count, hidden);
            var usage = TensorOps.MatMulTransposedB(hfStack, gate); // [T, I] up-states
            partitions[l] = ClusterUnits(usage, clusterTokens.Count, intermediate, experts);
        }
        notes.Add($"sampled {clusterTokens.Count} tokens; {layers} layers clustered into {experts} " +
                  $"balanced experts of {expertIntermediate} units");

        // ── 3. rebuild the decoder stack: router + expert tensors ──────
        var stackEncoding = container.Index.Representations[
            container.CanonicalRepresentationId(stackId)].Encoding;
        var stackSegmentPath = container.Index.Representations[
            container.CanonicalRepresentationId(stackId)].Segment;
        var rebuilt = new List<NamedTensorData>();
        using (var sourceSegment = SegmentFile.Open(Path.Combine(container.Root, stackSegmentPath)))
        {
            foreach (var tensor in sourceSegment.Header.Tensors)
            {
                // The dense FFN projections leave the stack; everything
                // else (attention, norms, linear-attention tensors) stays.
                bool isDenseProjection = tensor.Name.EndsWith(".mlp.gate_proj.weight") ||
                                         tensor.Name.EndsWith(".mlp.up_proj.weight") ||
                                         tensor.Name.EndsWith(".mlp.down_proj.weight");
                if (!isDenseProjection)
                {
                    rebuilt.Add(new NamedTensorData
                    {
                        Name = tensor.Name,
                        Dtype = DtypeExtensions.FromLabel(tensor.Dtype),
                        Shape = tensor.Shape,
                        Data = sourceSegment.ReadBytes(tensor.Name),
                    });
                }
            }
        }

        for (int l = 0; l < layers; l++)
        {
            string layerStem = $"{l}.mlp";
            var gateInfo = Resolve(store, stackId, $"{layerStem}.gate_proj.weight");
            var upInfo = Resolve(store, stackId, $"{layerStem}.up_proj.weight");
            var downInfo = Resolve(store, stackId, $"{layerStem}.down_proj.weight");
            int elem = gateInfo.Dtype.ElementSize();

            var partition = partitions[l];
            var routerValues = RouterRows(gateInfo, partition, experts, hidden, elem);

            rebuilt.Add(new NamedTensorData
            {
                Name = $"{layerStem}.router.weight",
                Dtype = Dtype.F32,
                Shape = new[] { (long)experts, hidden },
                Data = ToF32Bytes(routerValues),
            });

            for (int e = 0; e < experts; e++)
            {
                var units = Enumerable.Range(0, intermediate).Where(j => partition[j] == e).ToArray();
                var gateRows = GatherRows(gateInfo.Payload, hidden, elem, units);
                var upRows = GatherRows(upInfo.Payload, hidden, elem, units);
                var downCols = GatherCols(downInfo.Payload, intermediate, hidden, elem, units);
                string stem = $"{layerStem}.experts.{e}";
                rebuilt.Add(new NamedTensorData
                {
                    Name = $"{stem}.gate_proj.weight",
                    Dtype = gateInfo.Dtype,
                    Shape = new[] { (long)expertIntermediate, hidden },
                    Data = gateRows,
                });
                rebuilt.Add(new NamedTensorData
                {
                    Name = $"{stem}.up_proj.weight",
                    Dtype = upInfo.Dtype,
                    Shape = new[] { (long)expertIntermediate, hidden },
                    Data = upRows,
                });
                rebuilt.Add(new NamedTensorData
                {
                    Name = $"{stem}.down_proj.weight",
                    Dtype = downInfo.Dtype,
                    Shape = new[] { (long)hidden, expertIntermediate },
                    Data = downCols,
                });
            }
        }

        // ── 4. write the moе container ─────────────────────────────────
        Directory.CreateDirectory(outDir);
        var representations = new Dictionary<string, RepresentationEntry>(StringComparer.Ordinal);
        var segments = new Dictionary<string, int>(StringComparer.Ordinal);

        var rebuiltEntry = SegmentWriter.Write(
            Path.Combine(outDir, stackSegmentPath),
            $"target.decoder_stack@{stackEncoding}",
            rebuilt);
        representations[$"target.decoder_stack@{stackEncoding}"] = new RepresentationEntry
        {
            Object = "target.decoder_stack",
            Encoding = stackEncoding,
            Segment = stackSegmentPath,
            TensorCount = rebuilt.Count,
            PayloadBytes = rebuiltEntry.PayloadBytes,
            PayloadSha256 = rebuiltEntry.PayloadSha256Hex,
            SegmentSha256 = rebuiltEntry.SegmentSha256Hex,
        };
        segments[stackSegmentPath[..^4]] = 1;

        foreach (var objectId in new[] { "target.embedding", "target.final_norm" })
        {
            CopySegment(container, outDir, objectId, representations, segments);
        }
        if (representations.Keys.Any(k => k.StartsWith("target.output_head", StringComparison.Ordinal)) == false &&
            container.Index.Representations.Values.Any(e => e.Object == "target.output_head"))
        {
            CopySegment(container, outDir, "target.output_head", representations, segments);
        }
        CopyTokenizer(container, outDir);

        // The moе-ified container keeps every segment the rebuild does not
        // touch — the vision tower, a carried MTP drafter, anything else
        // the encoder materialised. The graph already references those
        // objects unchanged; index entries and segment files must follow
        // or the container is structurally incomplete.
        foreach (var (repId, entry) in container.Index.Representations)
        {
            if (representations.ContainsKey(repId))
            {
                continue;
            }
            Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(outDir, entry.Segment))!);
            File.Copy(Path.Combine(container.Root, entry.Segment), Path.Combine(outDir, entry.Segment), overwrite: true);
            representations[repId] = entry;
            segments[entry.Segment[..^4]] = 1;
        }

        var moeSurface = new MoeSurface
        {
            Experts = experts,
            TopK = topK,
            ExpertIntermediateSize = expertIntermediate,
            RoutingPolicy = policy,
        };
        var moeFfn = new FfnSurface
        {
            IntermediateSize = ffn.IntermediateSize,
            Activation = ffn.Activation,
            FfnType = ffn.FfnType,
            GatePolicy = ffn.GatePolicy,
            Moe = moeSurface,
        };
        WriteGraph(outDir, graph, component, surface, moeFfn);
        string moeModel = $"{container.Index.Model}-moe{experts}x{topK}";
        WriteIndex(outDir, container, component, moeModel, representations, segments);

        // ── 5. the perplexity gate ─────────────────────────────────────
        double densePpl = EvaluatePpl(containerDir, evalTokens);
        double moePpl = EvaluatePpl(outDir, evalTokens);
        notes.Add($"perplexity over {evalTokens.Count} held-out tokens: dense {densePpl:0.00} → moе {moePpl:0.00} " +
                  $"({(moePpl / densePpl - 1) * 100:+0.0;-0.0}%)");

        return new MoeIfyReport(
            OutDir: outDir,
            Model: moeModel,
            Layers: layers,
            Experts: experts,
            TopK: topK,
            ExpertIntermediateSize: expertIntermediate,
            SampledTokens: clusterTokens.Count,
            DensePpl: densePpl,
            MoePpl: moePpl,
            Notes: notes);
    }

    // ── activation clustering (balanced k-means over unit usage) ──────────

    /// <summary>Clusters the intermediate units by their usage pattern
    /// across the sampled tokens (cosine in the normalised space), then
    /// force-balances every cluster to exactly <c>intermediate / experts</c>
    /// units — the runtime resolves each expert at the nominal width.</summary>
    public static int[] ClusterUnits(Tensor2D usage, int tokens, int intermediate, int experts)
    {
        int target = intermediate / experts;
        var rows = new float[intermediate * tokens]; // unit j's usage across tokens
        for (int j = 0; j < intermediate; j++)
        {
            for (int t = 0; t < tokens; t++)
            {
                rows[j * tokens + t] = usage.Data[t * intermediate + j];
            }
        }
        // L2-normalise each unit's usage vector; zero-usage units keep zeros.
        for (int j = 0; j < intermediate; j++)
        {
            double norm = 0;
            for (int t = 0; t < tokens; t++)
            {
                double v = rows[j * tokens + t];
                norm += v * v;
            }
            if (norm > 0)
            {
                double inv = 1.0 / Math.Sqrt(norm);
                for (int t = 0; t < tokens; t++)
                {
                    rows[j * tokens + t] = (float)(rows[j * tokens + t] * inv);
                }
            }
        }

        var centroids = new float[experts * tokens];
        // Farthest-first seeding.
        {
            Array.Copy(rows, 0, centroids, 0, tokens);
            double[] best = new double[intermediate];
            for (int k = 1; k < experts; k++)
            {
                double farthest = -1;
                int farthestUnit = 0;
                for (int j = 0; j < intermediate; j++)
                {
                    double d = DistanceSquared(rows, j, centroids, 0, tokens);
                    if (d > best[j])
                    {
                        best[j] = d;
                    }
                    if (best[j] > farthest)
                    {
                        farthest = best[j];
                        farthestUnit = j;
                    }
                }
                Array.Copy(rows, farthestUnit * tokens, centroids, k * tokens, tokens);
            }
        }

        var assign = new int[intermediate];
        int[] counts = new int[experts];
        for (int iter = 0; iter < 40; iter++)
        {
            bool changed = false;
            Array.Clear(counts);
            for (int j = 0; j < intermediate; j++)
            {
                int bestCluster = -1;
                double bestD = double.PositiveInfinity;
                double bestMin = double.PositiveInfinity;
                for (int k = 0; k < experts; k++)
                {
                    if (counts[k] >= target && iter < 39)
                    {
                        continue; // keep balance during refinement
                    }
                    double d = DistanceSquared(rows, j, centroids, k * tokens, tokens);
                    if (d < bestD)
                    {
                        bestD = d;
                        bestCluster = k;
                    }
                }
                if (bestCluster < 0)
                {
                    double d = DistanceSquared(rows, j, centroids, 0, tokens);
                    bestCluster = 0;
                    bestMin = d;
                    for (int k = 1; k < experts; k++)
                    {
                        d = DistanceSquared(rows, j, centroids, k * tokens, tokens);
                        if (d < bestMin)
                        {
                            bestMin = d;
                            bestCluster = k;
                        }
                    }
                }
                if (assign[j] != bestCluster)
                {
                    changed = true;
                    assign[j] = bestCluster;
                }
                counts[bestCluster]++;
            }
            RecomputedCentroids(rows, assign, counts, centroids, intermediate, experts, tokens);
            if (!changed)
            {
                break;
            }
        }

        // Enforce the exact balance: move the worst-fitting unit from every
        // overfull cluster into the underfull cluster it fits best.
        for (int pass = 0; pass < intermediate; pass++)
        {
            var underfull = Enumerable.Range(0, experts)
                .Where(k => counts[k] < target)
                .ToArray();
            if (underfull.Length == 0)
            {
                break;
            }
            var targetCluster = underfull[0];
            var candidates = Enumerable.Range(0, intermediate)
                .Where(j => counts[assign[j]] > target)
                .ToArray();
            if (candidates.Length == 0)
            {
                break;
            }
            double best = double.PositiveInfinity;
            int bestUnit = candidates[0];
            foreach (var j in candidates)
            {
                double d = DistanceSquared(rows, j, centroids, targetCluster * tokens, tokens);
                if (d < best)
                {
                    best = d;
                    bestUnit = j;
                }
            }
            counts[assign[bestUnit]]--;
            counts[targetCluster]++;
            assign[bestUnit] = targetCluster;
        }
        return assign;
    }

    private static double DistanceSquared(float[] rows, int unit, float[] centroids, int centroidStart, int tokens)
    {
        double sum = 0;
        for (int t = 0; t < tokens; t++)
        {
            double d = rows[unit * tokens + t] - centroids[centroidStart + t];
            sum += d * d;
        }
        return sum;
    }

    private static void RecomputedCentroids(
        float[] rows, int[] assign, int[] counts, float[] centroids,
        int intermediate, int experts, int tokens)
    {
        Array.Clear(centroids);
        var sums = new float[experts * tokens];
        for (int j = 0; j < intermediate; j++)
        {
            int cluster = assign[j];
            for (int t = 0; t < tokens; t++)
            {
                sums[cluster * tokens + t] += rows[j * tokens + t];
            }
        }
        for (int k = 0; k < experts; k++)
        {
            if (counts[k] > 0)
            {
                float inv = 1f / counts[k];
                for (int t = 0; t < tokens; t++)
                {
                    centroids[k * tokens + t] = sums[k * tokens + t] * inv;
                }
            }
        }
    }

    // ── tensor slicing ─────────────────────────────────────────────────────

    private sealed record ResolvedTable(Dtype Dtype, byte[] Payload);

    private static ResolvedTable Resolve(OperandStore store, string stackId, string tensorName)
    {
        var resolution = store.Resolve(stackId, tensorName);
        return new ResolvedTable(resolution.Dtype, resolution.Payload);
    }

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

    /// <summary>The linear router: each expert's row is the mean of its
    /// cluster's gate directions — token affinity <c>x · routerᵀ</c>,
    /// exactly what <see cref="FfnKernel.Routed"/> scores.</summary>
    private static float[] RouterRows(ResolvedTable gate, int[] partition, int experts, int hidden, int elem)
    {
        var router = new float[experts * hidden];
        var counts = new int[experts];
        var gateValues = BitPattern.WidenToF32(gate.Dtype, gate.Payload);
        for (int j = 0; j < partition.Length; j++)
        {
            int cluster = partition[j];
            counts[cluster]++;
            for (int c = 0; c < hidden; c++)
            {
                router[cluster * hidden + c] += gateValues[j * hidden + c];
            }
        }
        for (int k = 0; k < experts; k++)
        {
            if (counts[k] > 0)
            {
                float inv = 1f / counts[k];
                for (int c = 0; c < hidden; c++)
                {
                    router[k * hidden + c] *= inv;
                }
            }
        }
        return router;
    }

    private static byte[] ToF32Bytes(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    // ── container writing ──────────────────────────────────────────────────

    private static void CopySegment(
        Vindex3Container source,
        string outDir,
        string objectId,
        Dictionary<string, RepresentationEntry> representations,
        Dictionary<string, int> segments)
    {
        string repId = source.CanonicalRepresentationId(objectId);
        if (!source.Index.Representations.TryGetValue(repId, out var entry))
        {
            return;
        }
        Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(outDir, entry.Segment))!);
        File.Copy(Path.Combine(source.Root, entry.Segment), Path.Combine(outDir, entry.Segment), overwrite: true);
        representations[repId] = entry;
        segments[entry.Segment[..^4]] = 1;
    }

    private static void CopyTokenizer(Vindex3Container source, string outDir)
    {
        var tokenizer = Path.Combine(source.Root, "tokenizer.json");
        if (File.Exists(tokenizer))
        {
            File.Copy(tokenizer, Path.Combine(outDir, "tokenizer.json"));
        }
        HfAncillaryFiles.CopyInto(source.Root, outDir);
    }

    private static void WriteGraph(
        string outDir,
        SystemGraph graph,
        Component component,
        ExecutionSurface surface,
        FfnSurface moeFfn)
    {
        var rebuilt = new ExecutionSurface
        {
            ContextLength = surface.ContextLength,
            Attention = surface.Attention,
            Ffn = moeFfn,
            Norm = surface.Norm,
            Head = surface.Head,
            ResidualScale = surface.ResidualScale,
            LinearAttention = surface.LinearAttention,
            Kda = surface.Kda,
            KdaGateLowerBound = surface.KdaGateLowerBound,
            Mla = surface.Mla,
            Mamba2 = surface.Mamba2,
            ConvQkv = surface.ConvQkv,
            ResidualInFp32 = surface.ResidualInFp32,
        };
        var components = graph.Components.Select(c => c.Role == ComponentRole.PrimaryText
            ? new Component
            {
                Id = c.Id,
                Role = c.Role,
                SourceArtifact = c.SourceArtifact,
                NumLayers = c.NumLayers,
                HiddenSize = c.HiddenSize,
                Attention = c.Attention,
                Execution = rebuilt,
                Perception = null,
            }
            : c).ToList();
        File.WriteAllText(Path.Combine(outDir, "system_graph.json"),
            JsonSerializer.Serialize(new SystemGraph
            {
                Schema = graph.Schema,
                Components = components,
                Objects = graph.Objects,
                Edges = graph.Edges,
            }, ViJson.Options));
    }

    private static void WriteIndex(
        string outDir,
        Vindex3Container source,
        Component component,
        string model,
        Dictionary<string, RepresentationEntry> representations,
        Dictionary<string, int> segments)
    {
        var index = new Vindex3Index
        {
            Version = Vindex3Index.CurrentSchema,
            Model = model,
            Family = source.Index.Family,
            HiddenSize = component.HiddenSize,
            NumLayers = component.NumLayers,
            SystemGraph = "system_graph.json",
            Representations = representations,
            Profiles = new List<Profile> { Profile.Exact() },
            Segments = segments,
            Authority = ContainerAuthority.Derived,
            DerivedFromModel = source.Index.Model,
            PrecisionMap = source.Index.PrecisionMap,
        };
        File.WriteAllText(Path.Combine(outDir, "index.json"),
            JsonSerializer.Serialize(index, ViJson.Options));
    }

    // ── the perplexity gate ────────────────────────────────────────────────

    private static double EvaluatePpl(string containerPath, IReadOnlyList<int> tokens)
    {
        if (tokens.Count < 2)
        {
            return double.NaN;
        }
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var runtime = new GenericRuntime(plan, store);
        double totalCe = 0;
        int counted = 0;
        for (int t = 0; t < tokens.Count - 1; t++)
        {
            var hidden = runtime.StepForward(tokens[t]);
            if (plan.Output is not { } output)
            {
                continue;
            }
            var logits = runtime.FinalNormAndHead(hidden);
            var row = logits.Row(0);
            float max = float.NegativeInfinity;
            for (int i = 0; i < row.Length; i++)
            {
                if (row[i] > max)
                {
                    max = row[i];
                }
            }
            double sum = 0;
            for (int i = 0; i < row.Length; i++)
            {
                sum += Math.Exp(row[i] - max);
            }
            double prob = Math.Exp(row[tokens[t + 1]] - max) / sum;
            totalCe += -Math.Log(Math.Max(prob, 1e-12));
            counted++;
        }
        return counted == 0 ? double.NaN : Math.Exp(totalCe / counted);
    }
}