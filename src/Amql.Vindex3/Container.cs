using System.Security.Cryptography;
using System.Text.Json;

namespace Amql.Vindex3;

/// <summary>One integrity check result for one representation.</summary>
public sealed record IntegrityCheck(
    string Representation,
    bool Ok,
    string? Detail);

/// <summary>The full result of re-verifying a container's byte
/// equivalence (G4-style): every representation's payload hash and whole
/// segment hash are recomputed from disk and compared against the index.</summary>
public sealed record IntegrityReport(bool Ok, List<IntegrityCheck> Checks);

/// <summary>
/// An opened VINDEX3 container: <c>index.json</c> (root authority) plus,
/// when recorded, <c>system_graph.json</c> (semantic IR). Opening is
/// schema-gated and fails closed — an unreadable schema, a dangling
/// reference or a missing file is a typed error, never a guess.
/// </summary>
public sealed class Vindex3Container : IDisposable
{
    private readonly List<SegmentFile> _segments = new();

    public Vindex3Container(string root, Vindex3Index index, SystemGraph? graph)
    {
        Root = root;
        Index = index;
        Graph = graph;
    }

    public string Root { get; }
    public Vindex3Index Index { get; }
    public SystemGraph? Graph { get; }

    public static Vindex3Container Open(string root)
    {
        var indexPath = Path.Combine(root, "index.json");
        if (!File.Exists(indexPath))
        {
            throw new ContainerException($"container '{root}' has no index.json");
        }

        Vindex3Index index;
        try
        {
            index = JsonSerializer.Deserialize<Vindex3Index>(File.ReadAllBytes(indexPath), ViJson.Options)
                    ?? throw new ContainerException($"container '{root}': index.json is empty");
        }
        catch (JsonException e)
        {
            throw new ContainerException($"container '{root}': malformed index.json: {e.Message}", e);
        }

        if (index.Version is < Vindex3Index.MinReadableSchema or > Vindex3Index.CurrentSchema)
        {
            throw new ContainerException(
                $"container '{root}': index schema {index.Version} is not readable " +
                $"(expected {Vindex3Index.MinReadableSchema}..{Vindex3Index.CurrentSchema})");
        }

        SystemGraph? graph = null;
        if (index.SystemGraph is { } graphPath)
        {
            var graphFile = Path.Combine(root, graphPath);
            if (!File.Exists(graphFile))
            {
                throw new ContainerException(
                    $"container '{root}': index references system graph '{graphPath}' but the file is missing");
            }
            try
            {
                graph = JsonSerializer.Deserialize<SystemGraph>(File.ReadAllBytes(graphFile), ViJson.Options)
                        ?? throw new ContainerException($"container '{root}': {graphPath} is empty");
            }
            catch (JsonException e)
            {
                throw new ContainerException($"container '{root}': malformed {graphPath}: {e.Message}", e);
            }

            if (graph.Schema != SystemGraph.CurrentSchema)
            {
                throw new ContainerException(
                    $"container '{root}': system graph schema {graph.Schema} is not supported " +
                    $"(expected {SystemGraph.CurrentSchema})");
            }
        }

        var container = new Vindex3Container(root, index, graph);
        container.ValidateStructure();
        return container;
    }

    /// <summary>Coherence validation: every recorded reference resolves —
    /// index representations name known objects (when a graph exists),
    /// graph components/objects/edges resolve to each other, and every
    /// representation entry's segment file exists on disk.</summary>
    private void ValidateStructure()
    {
        if (Graph is null)
        {
            return; // no graph recorded: representation refs cannot be checked
        }

        var objectIds = new HashSet<string>(Graph.Objects.Select(o => o.Id), StringComparer.Ordinal);
        var componentIds = new HashSet<string>(Graph.Components.Select(c => c.Id), StringComparer.Ordinal);

        foreach (var (representationId, entry) in Index.Representations)
        {
            if (!objectIds.Contains(entry.Object))
            {
                throw new ContainerException(
                    $"representation '{representationId}' names object '{entry.Object}' which the system graph does not contain");
            }
            var segmentFile = Path.Combine(Root, entry.Segment);
            if (!File.Exists(segmentFile))
            {
                throw new ContainerException(
                    $"representation '{representationId}' points at '{entry.Segment}' but the file is missing");
            }
        }

        foreach (var obj in Graph.Objects)
        {
            if (!componentIds.Contains(obj.Component))
            {
                throw new ContainerException(
                    $"object '{obj.Id}' names component '{obj.Component}' which does not exist");
            }
        }

        foreach (var edge in Graph.Edges)
        {
            if (!componentIds.Contains(edge.ProducerComponent))
            {
                throw new ContainerException(
                    $"edge producer '{edge.ProducerComponent}' does not exist");
            }
            if (!componentIds.Contains(edge.ConsumerComponent))
            {
                throw new ContainerException(
                    $"edge consumer '{edge.ConsumerComponent}' does not exist");
            }
            if (!objectIds.Contains(edge.ConsumerObject))
            {
                throw new ContainerException(
                    $"edge consumer object '{edge.ConsumerObject}' does not exist");
            }
        }
    }

    /// <summary>The canonical representation id of an object: its first
    /// declared representation's encoding (<c>{object}@{encoding}</c>).</summary>
    public string CanonicalRepresentationId(string objectId)
    {
        var graph = Graph ?? throw new ContainerException("container records no system graph");
        var obj = graph.Object(objectId);
        if (obj.Representations.Count == 0)
        {
            throw new ContainerException($"object '{objectId}' declares no representations");
        }
        return $"{objectId}@{obj.Representations[0].Encoding}";
    }

    /// <summary>Re-verifies byte equivalence from disk alone: payload and
    /// whole-segment SHA-256s recomputed and compared against
    /// <c>index.representations</c>. A drifted checkpoint (source ≠
    /// recorded) and a corrupted container (encoded ≠ recorded) both fail
    /// here, though with different details.</summary>
    public IntegrityReport VerifyIntegrity()
    {
        var checks = new List<IntegrityCheck>();
        bool ok = true;

        foreach (var (representationId, entry) in Index.Representations)
        {
            var segmentFile = Path.Combine(Root, entry.Segment);
            if (!File.Exists(segmentFile))
            {
                checks.Add(new IntegrityCheck(representationId, false, "segment file missing"));
                ok = false;
                continue;
            }

            byte[] fileBytes;
            try
            {
                fileBytes = File.ReadAllBytes(segmentFile);
            }
            catch (IOException e)
            {
                checks.Add(new IntegrityCheck(representationId, false, $"unreadable: {e.Message}"));
                ok = false;
                continue;
            }

            // Re-derive payload bounds from the segment header, exactly what
            // the writer produced.
            long headerLength = BitConverter.ToInt64(fileBytes, 0);
            long payloadStart = checked(8 + headerLength);

            string whole = Convert.ToHexStringLower(SHA256.HashData(fileBytes));
            string payload = payloadStart >= fileBytes.Length
                ? string.Empty
                : Convert.ToHexStringLower(SHA256.HashData(
                    fileBytes.AsSpan((int)payloadStart).ToArray()));

            bool wholeOk = whole == entry.SegmentSha256;
            bool payloadOk = payload == entry.PayloadSha256;
            if (!wholeOk || !payloadOk)
            {
                ok = false;
                string wholeDetail = wholeOk ? "✓" : $"✗ (actual {whole})";
                string payloadDetail = payloadOk ? "✓" : $"✗ (actual {payload})";
                checks.Add(new IntegrityCheck(
                    representationId,
                    false,
                    $"segment_sha256 {wholeDetail} | payload_sha256 {payloadDetail}"));
            }
            else
            {
                checks.Add(new IntegrityCheck(representationId, true, null));
            }
        }

        return new IntegrityReport(ok, checks);
    }

    /// <summary>Opens the operand store for this container: object id →
    /// representation → segment → tensor resolution, with segments opened
    /// lazily and cached. The returned store is independent of this
    /// container and must be disposed by the caller.</summary>
    public OperandStore CreateOperandStore() => new(this);

    /// <summary>Opens a segment file, caching it on the container so
    /// repeated representation resolution reuses the mapping.</summary>
    internal SegmentFile OpenSegment(string relativePath)
    {
        var full = System.IO.Path.Combine(Root, relativePath);
        var existing = _segments.Find(s => s.Path == full);
        if (existing is not null)
        {
            return existing;
        }
        var opened = SegmentFile.Open(full);
        _segments.Add(opened);
        return opened;
    }

    public void Dispose()
    {
        foreach (var segment in _segments)
        {
            segment.Dispose();
        }
        _segments.Clear();
    }
}