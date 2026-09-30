using System.Text.Json;

namespace Amql.Vindex3;

/// <summary>
/// Produces a slim VIndex3 container from an existing full container by
/// keeping only the representation that matches a target encoding for
/// every logical object.  The output is structurally a valid VIndex3
/// container with <see cref="ContainerAuthority.Slim"/>.
/// </summary>
public static class ContainerStripper
{
    /// <summary>
    /// Strip every logical object down to the representation whose
    /// <paramref name="keepEncoding"/> matches, copying the corresponding
    /// segment and producing a new <c>index.json</c> and lightweight
    /// <c>system_graph.json</c> at <paramref name="outputRoot"/>.
    /// </summary>
    /// <param name="sourceRoot">Path to the existing full container.</param>
    /// <param name="outputRoot">Destination directory — must not exist.</param>
    /// <param name="keepEncoding">Case-insensitive encoding label to keep
    /// (e.g. <c>"Q4_0"</c>, <c>"BF16"</c>, <c>"FP4"</c>).</param>
    public static void Strip(string sourceRoot, string outputRoot, string keepEncoding)
    {
        if (Directory.Exists(outputRoot))
            throw new ArgumentException($"output directory '{outputRoot}' already exists");

        using var source = Vindex3Container.Open(sourceRoot);
        var sourceIndex = source.Index;
        var sourceGraph = source.Graph;

        // ── 1. Build the slim object→representation map ──
        var slimReps = new Dictionary<string, RepresentationEntry>(StringComparer.Ordinal);
        int discardedCount = 0;

        foreach (var (repId, entry) in sourceIndex.Representations)
        {
            if (string.Equals(entry.Encoding, keepEncoding, StringComparison.OrdinalIgnoreCase))
            {
                slimReps[repId] = entry;
            }
            else
            {
                discardedCount++;
            }
        }

        if (slimReps.Count == 0)
        {
            throw new InvalidOperationException(
                $"container '{sourceRoot}' has no representations matching '{keepEncoding}'. " +
                $"Available encodings: {string.Join(", ", sourceIndex.Representations.Values.Select(e => e.Encoding).Distinct())}");
        }

        // Every logical object must have exactly one kept representation,
        // otherwise the slim container is incomplete.
        if (sourceGraph is not null)
        {
            foreach (var obj in sourceGraph.Objects)
            {
                var keptForObj = obj.Representations
                    .Where(r => string.Equals(r.Encoding, keepEncoding, StringComparison.OrdinalIgnoreCase))
                    .ToList();
                if (keptForObj.Count == 0)
                {
                    throw new InvalidOperationException(
                        $"object '{obj.Id}' has no '{keepEncoding}' representation");
                }
            }
        }

        // ── 2. Create output directory and copy segments ──
        Directory.CreateDirectory(outputRoot);
        var segmentsDir = Path.Combine(outputRoot, ".segments");
        Directory.CreateDirectory(segmentsDir);

        var segmentMap = new Dictionary<string, string>(StringComparer.Ordinal); // old → new (relative)
        foreach (var (repId, entry) in slimReps)
        {
            var srcPath = Path.Combine(sourceRoot, entry.Segment);
            var segmentName = Path.GetFileName(entry.Segment);
            var dstPath = Path.Combine(segmentsDir, segmentName);
            File.Copy(srcPath, dstPath, overwrite: false);

            var relPath = Path.GetRelativePath(outputRoot, dstPath).Replace('\\', '/');
            segmentMap[entry.Segment] = relPath;
        }

        // ── 3. Remap segments in the kept representation entries ──
        var remappedReps = new Dictionary<string, RepresentationEntry>(StringComparer.Ordinal);
        foreach (var (repId, entry) in slimReps)
        {
            var newSegment = segmentMap[entry.Segment];
            remappedReps[repId] = new RepresentationEntry
            {
                Object = entry.Object,
                Encoding = entry.Encoding,
                Segment = newSegment,
                TensorCount = entry.TensorCount,
                PayloadBytes = entry.PayloadBytes,
                PayloadSha256 = entry.PayloadSha256,
                SegmentSha256 = entry.SegmentSha256,
            };
        }

        // ── 4. Build the segment census ──
        var segmentStems = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var entry in remappedReps.Values)
        {
            var stem = Path.GetFileNameWithoutExtension(entry.Segment);
            segmentStems.TryGetValue(stem, out var count);
            segmentStems[stem] = count + 1;
        }

        // ── 5. Write slim index.json ──
        var slimIndex = new Vindex3Index
        {
            Version = sourceIndex.Version,
            Model = sourceIndex.Model,
            Family = sourceIndex.Family,
            HiddenSize = sourceIndex.HiddenSize,
            NumLayers = sourceIndex.NumLayers,
            SystemGraph = sourceIndex.SystemGraph,
            Representations = remappedReps,
            Profiles = sourceIndex.Profiles,
            Segments = segmentStems,
            Authority = ContainerAuthority.Slim,
            PrecisionMap = sourceIndex.PrecisionMap,
            DerivedFromModel = sourceIndex.DerivedFromModel,
            TokenMap = sourceIndex.TokenMap,
            Extra = sourceIndex.Extra,
        };

        var indexPath = Path.Combine(outputRoot, "index.json");
        File.WriteAllBytes(indexPath, JsonSerializer.SerializeToUtf8Bytes(slimIndex, ViJson.Options));

        // ── 6. Write slim system_graph.json ──
        if (sourceGraph is not null)
        {
            var slimObjects = new List<LogicalObject>();
            foreach (var obj in sourceGraph.Objects)
            {
                var keptReps = obj.Representations
                    .Where(r => string.Equals(r.Encoding, keepEncoding, StringComparison.OrdinalIgnoreCase))
                    .ToList();
                slimObjects.Add(new LogicalObject
                {
                    Id = obj.Id,
                    Component = obj.Component,
                    Kind = obj.Kind,
                    SourceBindings = obj.SourceBindings,
                    Representations = keptReps,
                });
            }

            var slimGraph = new SystemGraph
            {
                Schema = sourceGraph.Schema,
                Components = sourceGraph.Components,
                Objects = slimObjects,
                Edges = sourceGraph.Edges,
            };

            var graphPath = sourceIndex.SystemGraph ?? "system_graph.json";
            var graphDst = Path.Combine(outputRoot, graphPath);
            var graphDir = Path.GetDirectoryName(graphDst);
            if (graphDir is not null && !Directory.Exists(graphDir))
                Directory.CreateDirectory(graphDir);
            File.WriteAllBytes(graphDst, JsonSerializer.SerializeToUtf8Bytes(slimGraph, ViJson.Options));
        }

        // ── 7. Copy auxiliary files ──
        var auxFiles = new[] { "tokenizer.json", "classifier.json", "chat_template.jinja" };
        foreach (var f in auxFiles)
        {
            var src = Path.Combine(sourceRoot, f);
            if (File.Exists(src))
            {
                File.Copy(src, Path.Combine(outputRoot, f), overwrite: false);
            }
        }

        if (sourceGraph is not null)
        {
            Console.WriteLine(
                $"Stripped {discardedCount} representation(s) — kept {slimReps.Count} × {keepEncoding} " +
                $"(Authority: Slim)");
        }
        else
        {
            Console.WriteLine(
                $"Stripped {discardedCount} representation(s) — kept {slimReps.Count} × {keepEncoding}");
        }
    }
}