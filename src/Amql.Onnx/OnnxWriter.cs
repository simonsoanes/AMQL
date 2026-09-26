namespace Amql.Onnx;

/// <summary>Minimal protobuf binary writer for the ONNX subset we emit.
/// Encodes the field numbers and wire types directly — no external
/// protobuf dependency.</summary>
internal sealed class ProtoWriter : IDisposable
{
    private readonly Stream _stream;
    public ProtoWriter(Stream stream) => _stream = stream;

    public void Dispose() => _stream.Dispose();

    // ── wire types ─────────────────────────────────────────────────────
    private const int Varint = 0;     // int32, int64, uint32, bool, enum
    private const int Fixed64 = 1;    // double, fixed64
    private const int LengthDelim = 2;// string, bytes, embedded, packed
    private const int Fixed32 = 5;    // float, fixed32

    private void Tag(int field, int wire) =>
        WriteVarint((ulong)field << 3 | (uint)wire);

    // ── primitives ─────────────────────────────────────────────────────

    public void Int32(int field, int v) { Tag(field, Varint); WriteVarint((ulong)v); }
    public void Int64(int field, long v) { Tag(field, Varint); WriteVarint((ulong)v); }
    public void Float(int field, float v) { Tag(field, Fixed32); WriteFixed32(v); }
    public void String(int field, string v) { Tag(field, LengthDelim); WriteBytes(System.Text.Encoding.UTF8.GetBytes(v)); }
    public void Bytes(int field, byte[] v) { Tag(field, LengthDelim); WriteBytes(v); }
    public void Message(int field, Action<ProtoWriter> write) { Tag(field, LengthDelim); var ms = new MemoryStream(); var inner = new ProtoWriter(ms); write(inner); inner.Dispose(); var b = ms.ToArray(); WriteBytes(b); }

    // ── wire encoders ──────────────────────────────────────────────────

    private void WriteVarint(ulong v)
    {
        while (v >= 0x80)
        {
            _stream.WriteByte((byte)(v & 0x7F | 0x80));
            v >>= 7;
        }
        _stream.WriteByte((byte)v);
    }

    private void WriteFixed32(float v)
    {
        var bytes = BitConverter.GetBytes(v);
        if (!BitConverter.IsLittleEndian) Array.Reverse(bytes);
        _stream.Write(bytes);
    }

    private void WriteFixed64(double v)
    {
        var bytes = BitConverter.GetBytes(v);
        if (!BitConverter.IsLittleEndian) Array.Reverse(bytes);
        _stream.Write(bytes);
    }

    private void WriteBytes(byte[] v)
    {
        WriteVarint((ulong)v.Length);
        _stream.Write(v);
    }
}

/// <summary>Known ONNX data types — the integer codes the spec
/// assigns.</summary>
internal static class OnnxTypes
{
    public const int Float = 1;
    public const int Int64 = 7;
    public const int String = 8;
    public const int Float16 = 10;
    public const int Int4 = 22; // opset 21+
}

/// <summary>ONNX operator set we target.</summary>
internal static class OnnxOpsets
{
    public const string Domain = "";
    public const long Version = 21; // latest stable
}

/// <summary>Describes one tensor shape dimension — a known value or
/// a dynamic parameter name.</summary>
public sealed class OnnxDim
{
    public long? Value { get; init; }
    public string? Param { get; init; }
    public static OnnxDim Fixed(long v) => new() { Value = v };
    public static OnnxDim Parametric(string p) => new() { Param = p };
}

/// <summary>One tensor input or output of the graph.</summary>
public sealed class OnnxVar
{
    public required string Name { get; init; }
    public required int ElementType { get; init; }
    public IReadOnlyList<OnnxDim>? Shape { get; init; }
}

/// <summary>One operation node.</summary>
public sealed class OnnxNode
{
    public required string OpType { get; init; }
    public required IReadOnlyList<string> Inputs { get; init; }
    public required IReadOnlyList<string> Outputs { get; init; }
    public Dictionary<string, object> Attributes { get; init; } = new();
}

/// <summary>One weight initializer (stored in the graph).</summary>
public sealed class OnnxInitializer
{
    public required string Name { get; init; }
    public required int DataType { get; init; }
    public required long[] Dims { get; init; }
    public required byte[] RawData { get; init; }
}

/// <summary>The complete graph description the exporter produces.</summary>
public sealed class OnnxGraph
{
    public string Name { get; init; } = "amql";
    public required IReadOnlyList<OnnxVar> Inputs { get; init; }
    public required IReadOnlyList<OnnxVar> Outputs { get; init; }
    public required IReadOnlyList<OnnxNode> Nodes { get; init; }
    public required IReadOnlyList<OnnxInitializer> Initializers { get; init; }
}

/// <summary>Serialises an OnnxGraph to the binary .onnx format
/// (ModelProto wrapped in a protobuf stream).
///
/// When <paramref name="externalDataPath"/> is non-null, initializer
/// payloads are written to that file and referenced by offset/length
/// in the protobuf — required for models whose on-disk weights exceed
/// the 2 GiB protobuf message limit.</summary>
public static class OnnxWriter
{
    private const string ExtLocation = "location";
    private const string ExtOffset = "offset";
    private const string ExtLength = "length";

    public static void Write(string path, OnnxGraph graph, string? externalDataPath = null)
    {
        string? dataFileName = externalDataPath is { } fname
            ? Path.GetFileName(fname)
            : null;
        using var dataFile = externalDataPath is { } extPath
            ? File.Create(extPath)
            : null;

        // Pre-compute external-data offsets so we don't need mutable
        // state smuggled into lambdas.
        long offset = 0;
        var offsets = new long[graph.Initializers.Count];
        for (int i = 0; i < graph.Initializers.Count; i++)
        {
            offsets[i] = offset;
            offset += graph.Initializers[i].RawData.Length;
        }

        using var fs = File.Create(path);
        using var w = new ProtoWriter(fs);

        // ModelProto — fields written directly; no wrapper.
        w.Int64(1, 10);                          // ir_version = 10
        w.Message(8, inner =>                    // opset_import
        {
            inner.String(1, OnnxOpsets.Domain);
            inner.Int64(2, OnnxOpsets.Version);
        });
        w.String(2, "amql-cli");                  // producer_name

        // GraphProto (field 7 of ModelProto)
        w.Message(7, g =>
        {
            g.String(2, graph.Name);

            foreach (var node in graph.Nodes)
            {
                g.Message(1, n => WriteNode(n, node));
            }

            int initIdx = 0;
            foreach (var init in graph.Initializers)
            {
                long initOffset = offsets[initIdx++];
                g.Message(5, i => WriteInitializer(i, init, dataFile, dataFileName, initOffset));
            }

            foreach (var input in graph.Inputs)
            {
                g.Message(11, vi => WriteValueInfo(vi, input));
            }

            foreach (var output in graph.Outputs)
            {
                g.Message(12, vo => WriteValueInfo(vo, output));
            }
        });
    }

    private static void WriteNode(ProtoWriter w, OnnxNode node)
    {
        foreach (var i in node.Inputs) w.String(1, i);
        foreach (var o in node.Outputs) w.String(2, o);
        w.String(4, node.OpType);
        foreach (var (name, value) in node.Attributes)
        {
            w.Message(5, a =>
            {
                a.String(1, name);
                switch (value)
                {
                    case float f:
                        a.Int32(20, 1); a.Float(2, f); break;
                    case long l:
                        a.Int32(20, 2); a.Int64(3, l); break;
                    case string s:
                        a.Int32(20, 3); a.String(4, s); break;
                    case int[] ints:
                        a.Int32(20, 7);
                        foreach (var ii in ints) a.Int64(8, ii);
                        break;
                    case long[] longs:
                        a.Int32(20, 7);
                        foreach (var ll in longs) a.Int64(8, ll);
                        break;
                }
            });
        }
    }

    private static void WriteInitializer(ProtoWriter w, OnnxInitializer init,
        Stream? dataFile, string? dataFileName, long offset)
    {
        foreach (var d in init.Dims) w.Int64(1, d);
        w.Int32(2, init.DataType);
        w.String(8, init.Name);

        if (dataFile is not null && dataFileName is not null)
        {
            long length = init.RawData.Length;
            dataFile.Write(init.RawData);
            w.Int32(14, 1);
            w.Message(13, e => { e.String(1, ExtLocation); e.String(2, dataFileName); });
            w.Message(13, e => { e.String(1, ExtOffset); e.String(2, offset.ToString()); });
            w.Message(13, e => { e.String(1, ExtLength); e.String(2, length.ToString()); });
        }
        else
        {
            w.Bytes(9, init.RawData);
        }
    }

    private static void WriteValueInfo(ProtoWriter w, OnnxVar var)
    {
        w.String(1, var.Name);
        w.Message(2, t =>
        {
            t.Message(1, tt =>
            {
                tt.Int32(1, var.ElementType);
                if (var.Shape is { } shape)
                {
                    tt.Message(2, s =>
                    {
                        foreach (var dim in shape)
                        {
                            s.Message(1, d =>
                            {
                                if (dim.Value is { } v && dim.Param is null)
                                    d.Int64(1, v);
                                else if (dim.Param is { } p)
                                    d.String(2, p);
                            });
                        }
                    });
                }
            });
        });
    }
}