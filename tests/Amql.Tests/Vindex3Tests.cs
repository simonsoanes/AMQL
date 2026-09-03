using System.Text.Json;
using Amql.Safetensors;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

public class Vindex3Tests
{
    // ── segment byte layout ────────────────────────────────────────────────

    [Fact]
    public void Segment_Layout_Golden()
    {
        using var dir = new TempDir();
        var path = Path.Combine(dir.Path, "segments", "target.decoder_stack.bin");

        var payloadA = SyntheticModel.ToBytes(new[] { 1f, 2f, 3f, 4f });
        var payloadB = SyntheticModel.ToBytes(new[] { 5f, 6f });

        var result = SegmentWriter.Write(path, "target.decoder_stack@F32", new[]
        {
            new NamedTensorData { Name = "zb.ffn", Dtype = Dtype.F32, Shape = new long[] { 1, 2 }, Data = payloadB },
            new NamedTensorData { Name = "aa.attn", Dtype = Dtype.F32, Shape = new long[] { 2, 2 }, Data = payloadA },
        });

        var bytes = File.ReadAllBytes(path);
        long headerLength = BitConverter.ToInt64(bytes, 0);
        Assert.Equal(24, result.PayloadBytes);

        // Framing: [u64 LE header length][padded JSON header][payload].
        // The payload must start 16-byte aligned.
        long payloadStart = 8 + headerLength;
        Assert.Equal(0L, payloadStart % SegmentFormat.PayloadAlignment);
        Assert.True(payloadStart < bytes.Length);
        Assert.Equal(payloadA, bytes.AsSpan((int)payloadStart, payloadA.Length).ToArray());

        // Header JSON parses; table is sorted by name; offsets are relative
        // to payload start.
        using var segment = SegmentFile.Open(path);
        Assert.Equal((long)SegmentFormat.CurrentSchema, (long)segment.Header.Schema);
        Assert.Equal("target.decoder_stack@F32", segment.Header.Representation);
        Assert.Equal(2, segment.Header.Tensors.Count);
        Assert.Equal("aa.attn", segment.Header.Tensors[0].Name);
        Assert.Equal("zb.ffn", segment.Header.Tensors[1].Name);
        Assert.Equal(0L, segment.Header.Tensors[0].Offset);
        Assert.Equal((long)payloadA.Length, segment.Header.Tensors[1].Offset);
        Assert.Equal(payloadA, segment.ReadBytes("aa.attn"));
        Assert.Equal(payloadB, segment.ReadBytes("zb.ffn"));
    }

    [Fact]
    public void Segment_Refuses_WrongSchema()
    {
        // Build a header with schema 99 and confirm the reader refuses
        // rather than guessing.
        var header = JsonSerializer.Serialize(new
        {
            schema = 99,
            representation = "x@F32",
            tensors = Array.Empty<object>(),
        });
        // pad to 16
        var payloadAlign = SegmentFormat.PayloadAlignment;
        int headerLen = System.Text.Encoding.UTF8.GetByteCount(header);
        int pad = (payloadAlign - ((8 + headerLen) % payloadAlign)) % payloadAlign;
        using var dir = new TempDir();
        var path = Path.Combine(dir.Path, "bad.bin");
        using (var stream = new FileStream(path, FileMode.Create))
        {
            var len = BitConverter.GetBytes((ulong)headerLen);
            stream.Write(len, 0, 8);
            var headerBytes = System.Text.Encoding.UTF8.GetBytes(header);
            stream.Write(headerBytes, 0, headerBytes.Length);
            stream.Write(new byte[pad]);
        }
        Assert.Throws<ContainerException>(() => SegmentFile.Open(path));
    }

    // ── encode → open round trip ───────────────────────────────────────────

    [Fact]
    public void Encode_Then_Open_RoundTrip()
    {
        using var dir = new TempDir();
        var containerPath = Path.Combine(dir.Path, "container");
        var spec = SyntheticModel.BuildSpec(new Dims());
        var encoded = ContainerEncoder.Encode(containerPath, spec);

        Assert.True(File.Exists(Path.Combine(containerPath, "index.json")));
        Assert.True(File.Exists(Path.Combine(containerPath, "system_graph.json")));
        Assert.Equal(4, encoded.Index.Representations.Count);

        using var container = Vindex3Container.Open(containerPath);
        Assert.Equal("synth-lm", container.Index.Model);
        Assert.Equal(Vindex3Index.CurrentSchema, container.Index.Version);
        Assert.NotNull(container.Graph);
        Assert.Single(container.Graph!.Components);
        Assert.Equal(4, container.Graph.Objects.Count);
        Assert.Equal("F32", container.CanonicalRepresentationId("target.decoder_stack").Split('@')[1]);

        // Representability question: does the store resolve every object?
        using var store = container.CreateOperandStore();
        var stack = "target.decoder_stack";
        Assert.Equal("segments/target.decoder_stack.bin", store.SegmentPathFor(stack));
        var res = store.Resolve(stack, "0.self_attn.q_proj.weight");
        Assert.Equal(Dtype.F32, res.Dtype);
        Assert.Equal(new long[] { 4, 4 }, res.Shape);
        Assert.Equal(64, res.Payload.Length);
        Assert.True(store.ContainsTensor(stack, "1.mlp.up_proj.weight"));

        // Integrity: byte equivalence verified from the container alone.
        var report = container.VerifyIntegrity();
        Assert.True(report.Ok, string.Join("\n", report.Checks.Select(c => $"{c.Representation}: {c.Detail ?? "ok"}")));
        Assert.Equal(4, report.Checks.Count);
    }

    [Fact]
    public void Integrity_Detects_Payload_Mutation()
    {
        using var dir = new TempDir();
        var containerPath = Path.Combine(dir.Path, "container");
        ContainerEncoder.Encode(containerPath, SyntheticModel.BuildSpec(new Dims()));

        // Corrupt one payload byte of the decoder stack segment.
        var segmentPath = Path.Combine(containerPath, "segments", "target.decoder_stack.bin");
        var bytes = File.ReadAllBytes(segmentPath);
        long headerLength = BitConverter.ToInt64(bytes, 0);
        bytes[8 + headerLength] ^= 0xFF; // first payload byte
        File.WriteAllBytes(segmentPath, bytes);

        using var container = Vindex3Container.Open(containerPath);
        var report = container.VerifyIntegrity();
        Assert.False(report.Ok);
        var failing = report.Checks.First(c => !c.Ok);
        Assert.Contains("target.decoder_stack@F32", failing.Representation);
        Assert.Contains("payload_sha256", failing.Detail);
    }

    // ── JSON round trips (serde-parity spellings) ──────────────────────────

    [Fact]
    public void PositionPolicy_Json_RoundTrip()
    {
        var ropeJson = JsonSerializer.Serialize(PositionPolicy.CreateRope(10_000.0));
        Assert.Equal("{\"kind\":\"rope\",\"theta\":10000}", ropeJson);

        var noneJson = JsonSerializer.Serialize(PositionPolicy.None);
        Assert.Equal("{\"kind\":\"none\"}", noneJson);

        var rope = JsonSerializer.Deserialize<PositionPolicy>(ropeJson);
        var none = JsonSerializer.Deserialize<PositionPolicy>(noneJson);
        Assert.IsType<PositionRope>(rope);
        Assert.Equal(10_000.0, ((PositionRope)rope).Theta);
        Assert.IsType<PositionNone>(none);
    }

    [Fact]
    public void PositionPolicy_UnknownKind_Carried_Verbatim()
    {
        const string yarnJson = "{\"kind\":\"yarn\",\"theta\":500000,\"scaling\":{\"original_max_position_embeddings\":8192}}";
        var policy = JsonSerializer.Deserialize<PositionPolicy>(yarnJson)!;
        var unresolved = Assert.IsType<PositionUnresolved>(policy);
        Assert.Equal("yarn", unresolved.Kind);
        Assert.Contains("500000", unresolved.Payload.GetRawText());
        Assert.Contains("8192", unresolved.Payload.GetRawText());

        // Serialising the unresolved policy re-emits the original object.
        Assert.Equal(yarnJson, JsonSerializer.Serialize(policy));
    }

    [Fact]
    public void Graph_Json_Uses_SnakeCase_Spellings()
    {
        var graph = new SystemGraph
        {
            Schema = SystemGraph.CurrentSchema,
            Components = new List<Component>
            {
                new()
                {
                    Id = "target",
                    Role = ComponentRole.PrimaryText,
                    SourceArtifact = "x",
                    NumLayers = 2,
                    HiddenSize = 4,
                    Attention = new List<AttentionLayerPolicy>
                    {
                        new()
                        {
                            Operator = LayerOperators.Softmax,
                            Span = AttentionSpan.Sliding,
                            Window = 128,
                            Position = PositionPolicy.None,
                            VFromK = false,
                        },
                    },
                },
            },
            Objects = new List<LogicalObject>(),
            Edges = new List<HiddenStateEdge>(),
        };

        var json = JsonSerializer.Serialize(graph, ViJson.Options);
        Assert.Contains("\"num_layers\":2", json);
        Assert.Contains("\"hidden_size\":4", json);
        Assert.Contains("\"operator\":\"softmax\"", json);
        Assert.Contains("\"span\":\"sliding\"", json);
        Assert.Contains("\"position\":{\"kind\":\"none\"}", json);
        Assert.Contains("\"window\":128", json);
        Assert.Contains("\"source_artifact\":\"x\"", json);

        // Round trip.
        var back = JsonSerializer.Deserialize<SystemGraph>(json, ViJson.Options)!;
        Assert.Equal(ComponentRole.PrimaryText, back.Components[0].Role);
        Assert.Equal(AttentionSpan.Sliding, back.Components[0].Attention![0].Span);
        Assert.Equal(128, back.Components[0].Attention![0].Window);
    }

    [Fact]
    public void Index_UnknownFields_PassThrough()
    {
        const string json = """
            {"version":4,"model":"m","family":"f","hidden_size":8,"num_layers":2,
             "system_graph":"system_graph.json","representations":{},"profiles":["exact"],
             "segments":{},"future_flag":true,"future_note":"kept"}
            """;
        var index = JsonSerializer.Deserialize<Vindex3Index>(json, ViJson.Options)!;
        Assert.NotNull(index.Extra);
        Assert.Equal("kept", index.Extra!["future_note"].GetString());
        Assert.Single(index.Profiles);
        Assert.Equal("exact", index.Profiles[0].Name);

        // Bare-string profile form deserialises (untagged).
        var roundTrip = JsonSerializer.Serialize(index);
        Assert.Contains("\"future_flag\":true", roundTrip);
        Assert.Contains("\"exact\"", roundTrip);
    }
}