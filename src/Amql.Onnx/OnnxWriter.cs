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
    public void Message(int field, Action write) { Tag(field, LengthDelim); var ms = new MemoryStream(); var w = new ProtoWriter(ms); write(); w.Dispose(); var b = ms.ToArray(); WriteBytes(b); }

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
/// (ModelProto wrapped in a protobuf stream).</summary>
public static class OnnxWriter
{
    public static void Write(string path, OnnxGraph graph)
    {
        using var fs = File.Create(path);
        using var w = new ProtoWriter(fs);

        // ModelProto
        w.Message(1, () =>
        {
            w.Int64(1, 10);                          // ir_version = 10
            w.Message(8, () =>                        // opset_import
            {
                w.String(1, OnnxOpsets.Domain);
                w.Int64(2, OnnxOpsets.Version);
            });
            w.String(2, "amql-cli");                  // producer_name

            // GraphProto (field 7 of ModelProto)
            w.Message(7, () =>
            {
                // name
                w.String(2, graph.Name);

                // node (repeated, field 1)
                foreach (var node in graph.Nodes)
                {
                    w.Message(1, () => WriteNode(w, node));
                }

                // initializer (repeated, field 5)
                foreach (var init in graph.Initializers)
                {
                    w.Message(5, () => WriteInitializer(w, init));
                }

                // input (repeated, field 11)
                foreach (var input in graph.Inputs)
                {
                    w.Message(11, () => WriteValueInfo(w, input));
                }

                // output (repeated, field 12)
                foreach (var output in graph.Outputs)
                {
                    w.Message(12, () => WriteValueInfo(w, output));
                }
            });
        });
    }

    private static void WriteNode(ProtoWriter w, OnnxNode node)
    {
        // input (repeated string, field 1)
        foreach (var i in node.Inputs) w.String(1, i);
        // output (repeated string, field 2)
        foreach (var o in node.Outputs) w.String(2, o);
        // op_type (string, field 4)
        w.String(4, node.OpType);
        // attribute (repeated, field 5)
        foreach (var (name, value) in node.Attributes)
        {
            w.Message(5, () =>
            {
                w.String(1, name); // name
                switch (value)
                {
                    case float f: w.Float(2, f); break;            // f
                    case long l: w.Int64(3, l); break;             // i
                    case string s: w.String(4, s); break;          // s
                    case int[] ints:                               // ints
                        foreach (var ii in ints) w.Int64(8, ii);   // packed repeated
                        break;
                }
            });
        }
    }

    private static void WriteInitializer(ProtoWriter w, OnnxInitializer init)
    {
        // dims (repeated int64, field 1)
        foreach (var d in init.Dims) w.Int64(1, d);
        // data_type (int32, field 2)
        w.Int32(2, init.DataType);
        // name (string, field 8)
        w.String(8, init.Name);
        // raw_data (bytes, field 9)
        w.Bytes(9, init.RawData);
    }

    private static void WriteValueInfo(ProtoWriter w, OnnxVar var)
    {
        // name (string, field 1)
        w.String(1, var.Name);
        // type (TypeProto, field 2)
        w.Message(2, () =>
        {
            // tensor_type (TensorTypeProto, field 1)
            w.Message(1, () =>
            {
                // elem_type (int32, field 1)
                w.Int32(1, var.ElementType);
                // shape (TensorShapeProto, field 2)
                if (var.Shape is { } shape)
                {
                    w.Message(2, () =>
                    {
                        foreach (var dim in shape)
                        {
                            w.Message(1, () =>
                            {
                                if (dim.Value is { } v && dim.Param is null)
                                {
                                    w.Int64(1, v);  // dim_value
                                }
                                else if (dim.Param is { } p)
                                {
                                    w.String(2, p); // dim_param
                                }
                                // else: empty dimension (unknown)
                            });
                        }
                    });
                }
            });
        });
    }
}