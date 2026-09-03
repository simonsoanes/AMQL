using System.Text.Json;

namespace Amql.Vindex3;

/// <summary>One representation to materialise: a logical object realised by
/// a set of named tensors in one encoding. This build stays canonical —
/// the reference's represent/quantisation machinery is out of scope.</summary>
public sealed class RepresentationSpec
{
    public required string ObjectId { get; init; }

    /// <summary>Encoding label recorded in the graph's representations
    /// ("BF16", "F32", ...).</summary>
    public required string Encoding { get; init; }

    public required List<NamedTensorData> Tensors { get; init; }
}

/// <summary>The full description of a container to build — the .NET stand-in
/// for the reference's plan pipeline output (the reference derives this from
/// an HF checkpoint via inventories and the representability plan; this
/// build takes the graph and bindings explicitly).</summary>
public sealed class ContainerSpec
{
    public required string Model { get; init; }
    public required string Family { get; init; }
    public required int HiddenSize { get; init; }
    public required int NumLayers { get; init; }
    public required SystemGraph SystemGraph { get; init; }
    public required List<RepresentationSpec> Representations { get; init; }

    /// <summary>Stored-precision policy documenting per-tensor dtype
    /// deviations from the canonical encoding.</summary>
    public PrecisionMap? PrecisionMap { get; init; }
}

public sealed record EncodeResult(
    string Root,
    Vindex3Index Index,
    IReadOnlyDictionary<string, SegmentWriteResult> Segments);

/// <summary>
/// G3-style materialisation: one logical object → one canonical
/// representation → one simple contiguous segment, plus the index and
/// system graph that make the container self-describing. Encodes into a
/// fresh directory and never touches the source after writing — the G3
/// gate ("validation gets no access to the source") is preserved by
/// construction.
/// </summary>
public static class ContainerEncoder
{
    public static EncodeResult Encode(string containerRoot, ContainerSpec spec)
    {
        if (Directory.Exists(containerRoot))
        {
            throw new ContainerException($"container output '{containerRoot}' already exists");
        }

        var byId = new HashSet<string>(spec.SystemGraph.Objects.Select(o => o.Id), StringComparer.Ordinal);
        var representations = new Dictionary<string, RepresentationEntry>(StringComparer.Ordinal);
        var segments = new Dictionary<string, int>(StringComparer.Ordinal);
        var results = new Dictionary<string, SegmentWriteResult>(StringComparer.Ordinal);

        foreach (var rep in spec.Representations)
        {
            if (!byId.Contains(rep.ObjectId))
            {
                throw new ContainerException(
                    $"representation '{rep.ObjectId}' is not a logical object of the system graph");
            }
            if (representations.ContainsKey($"{rep.ObjectId}@{rep.Encoding}"))
            {
                throw new ContainerException($"duplicate representation '{rep.ObjectId}@{rep.Encoding}'");
            }

            var representationId = $"{rep.ObjectId}@{rep.Encoding}";
            var segmentPath = $"segments/{rep.ObjectId}.bin";
            var fullPath = Path.Combine(containerRoot, segmentPath);

            var result = SegmentWriter.Write(fullPath, representationId, rep.Tensors);
            representations[representationId] = new RepresentationEntry
            {
                Object = rep.ObjectId,
                Encoding = rep.Encoding,
                Segment = segmentPath,
                TensorCount = rep.Tensors.Count,
                PayloadBytes = result.PayloadBytes,
                PayloadSha256 = result.PayloadSha256Hex,
                SegmentSha256 = result.SegmentSha256Hex,
            };
            segments[$"segments/{rep.ObjectId}"] = 1;
            results[representationId] = result;
        }

        var index = new Vindex3Index
        {
            Version = Vindex3Index.CurrentSchema,
            Model = spec.Model,
            Family = spec.Family,
            HiddenSize = spec.HiddenSize,
            NumLayers = spec.NumLayers,
            SystemGraph = "system_graph.json",
            Representations = representations,
            Profiles = new List<Profile> { Profile.Exact() },
            Segments = segments,
            Authority = ContainerAuthority.Canonical,
            PrecisionMap = spec.PrecisionMap,
        };

        // Write the container: sole root authority, then the graph, then
        // the physical segments (already on disk).
        Directory.CreateDirectory(containerRoot);
        File.WriteAllText(Path.Combine(containerRoot, "index.json"),
            JsonSerializer.Serialize(index, ViJson.Options));
        File.WriteAllText(Path.Combine(containerRoot, "system_graph.json"),
            JsonSerializer.Serialize(spec.SystemGraph, ViJson.Options));

        return new EncodeResult(containerRoot, index, results);
    }
}