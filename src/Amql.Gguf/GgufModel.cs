namespace Amql.Gguf;

/// <summary>
/// GGUF tensor data types. The numeric values are the GGUF/GGML_type
/// enum as written by llama.cpp's gguf writer — only the float/integer
/// dtypes this converter emits are used, the quantised values are listed
/// for completeness so a reader can parse foreign files.
/// </summary>
public enum GgufType : uint
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q4_2 = 4,
    Q4_3 = 5,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    /// <summary>OCP microscaling FP4: 32 E2M1 elements sharing one E8M0
    /// exponent, 17 bytes per block. What gpt-oss ships natively and what
    /// Unsloth's MXFP4_MOE recipe uses for the expert stacks.</summary>
    Mxfp4 = 39,
    Nvfp4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
    /// <summary>PrismML Bonsai ternary PQ2_0: 128 {-1,0,+1} weights at 2 bits
    /// per trit (32 packed bytes) plus one FP16 scale, 34 bytes per block.
    /// The id is the Bonsai fork's own numbering — stock ggml's ternary types
    /// are TQ1_0/TQ2_0 at 34/35 with a different trit packing, so a file using
    /// this type loads only in that fork.</summary>
    Pq2_0 = 142,
    /// <summary>PrismML Bonsai ternary PTQ1_0: 128 weights as 5 trits per byte
    /// (26 packed bytes) plus one FP16 scale, 28 bytes per block. See
    /// <see cref="Pq2_0"/> for the fork-vs-stock caveat.</summary>
    Ptq1_0 = 143,
}

/// <summary>
/// Byte sizing for a GGUF tensor payload, shared by the writer and the reader
/// so the two cannot drift. Quantised types are block-packed rather than
/// per-element, so a reader that sized them as plain dtypes could not parse a
/// file the writer had just produced.
/// </summary>
public static class GgufTypeSizing
{
    /// <summary>Bytes occupied by the payload of a tensor with these dims.</summary>
    public static long DataBytes(GgufType type, long[] dims)
    {
        long elements = 1;
        foreach (long d in dims)
        {
            elements = checked(elements * d);
        }
        return DataBytes(type, elements);
    }

    /// <summary>(elements, bytes) per block of every block-packed ggml type,
    /// as ggml-common.h lays them out.</summary>
    private static readonly Dictionary<GgufType, (int Elements, int Bytes)> Blocks = new()
    {
        [GgufType.Q4_0] = (32, 18),
        [GgufType.Q4_1] = (32, 20),
        [GgufType.Q5_0] = (32, 22),
        [GgufType.Q5_1] = (32, 24),
        [GgufType.Q8_0] = (32, 34),
        [GgufType.Q2_K] = (256, 84),
        [GgufType.Q3_K] = (256, 110),
        [GgufType.Q4_K] = (256, 144),
        [GgufType.Q5_K] = (256, 176),
        [GgufType.Q6_K] = (256, 210),
        [GgufType.IQ2_XXS] = (256, 66),
        [GgufType.IQ2_XS] = (256, 74),
        [GgufType.IQ2_S] = (256, 82),
        [GgufType.IQ3_XXS] = (256, 98),
        [GgufType.IQ3_S] = (256, 110),
        [GgufType.IQ1_S] = (256, 50),
        [GgufType.IQ1_M] = (256, 56),
        [GgufType.IQ4_NL] = (32, 18),
        [GgufType.IQ4_XS] = (256, 136),
        [GgufType.TQ1_0] = (256, 54),
        [GgufType.TQ2_0] = (256, 66),
        // MXFP4: 1-byte E8M0 scale + 16 nibbles per 32.
        [GgufType.Mxfp4] = (32, 17),
        // NVFP4: 4 UE4M3 sub-block scales + 32 nibbles per 64.
        [GgufType.Nvfp4] = (64, 36),
        [GgufType.Q1_0] = (128, 18),
        [GgufType.Q2_0] = (64, 18),
        // Bonsai ternary: 128-element blocks, FP16 scale then packed trits.
        [GgufType.Pq2_0] = (128, 34),
        [GgufType.Ptq1_0] = (128, 28),
    };

    /// <summary>Bytes occupied by the payload of that many values.</summary>
    public static long DataBytes(GgufType type, long elements) =>
        Blocks.TryGetValue(type, out var block)
            ? (elements + block.Elements - 1) / block.Elements * block.Bytes
            : checked(elements * ElementSize(type));

    /// <summary>True for the block-packed types, whose payload size is a
    /// function of the block count rather than the element count.</summary>
    public static bool IsBlockQuantized(GgufType type) => Blocks.ContainsKey(type);

    /// <summary>Elements per scale block for a block-quantized type, which is
    /// what an eligibility check has to divide by — Q4_0 and friends block at
    /// 32, the K-quants and most IQ types at 256, the Bonsai ternary packings
    /// at 128.</summary>
    public static int BlockElements(GgufType type) =>
        Blocks.TryGetValue(type, out var block) ? block.Elements : 32;

    /// <summary>Bytes per value, for the plain dtypes only.</summary>
    public static int ElementSize(GgufType type) => type switch
    {
        GgufType.F32 or GgufType.I32 => 4,
        GgufType.F16 or GgufType.BF16 or GgufType.I16 => 2,
        GgufType.I8 => 1,
        _ => throw new GgufException($"gguf type {type} is not a plain dtype"),
    };
}

/// <summary>The GGUF metadata value-type tags.</summary>
public enum GgufValueType : uint
{
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

/// <summary>A GGUF metadata value — one of the scalar kinds or an array
/// of uniformly-typed scalars.</summary>
public sealed class GgufValue
{
    public GgufValueType Kind { get; }
    private readonly object _value;

    private GgufValue(GgufValueType kind, object value)
    {
        Kind = kind;
        _value = value;
    }

    /// <summary>Reader-side factory for arrays read from a file.</summary>
    internal static GgufValue ArrayOf(GgufValueType elementKind, IReadOnlyList<object> items)
        => new(GgufValueType.Array, (elementKind, items));

    /// <summary>The boxed payload: string, uint, ulong, int, long, float,
    /// bool, or (GgufValueType, IReadOnlyList&lt;object&gt;) for arrays.
    /// Internal so the writer and reader share one canonical form.</summary>
    internal object Boxed => _value;

    public static GgufValue String(string s) => new(GgufValueType.String, s);
    public static GgufValue Uint32(uint v) => new(GgufValueType.Uint32, v);
    public static GgufValue Uint64(ulong v) => new(GgufValueType.Uint64, v);
    public static GgufValue Int32(int v) => new(GgufValueType.Int32, v);
    public static GgufValue Int64(long v) => new(GgufValueType.Int64, v);
    public static GgufValue Float32(float v) => new(GgufValueType.Float32, v);
    public static GgufValue Float64(double v) => new(GgufValueType.Float64, v);
    public static GgufValue Bool(bool v) => new(GgufValueType.Bool, v);

    public static GgufValue StringArray(IReadOnlyList<string> items)
        => new(GgufValueType.Array, (GgufValueType.String, Box(items)));

    public static GgufValue Uint32Array(IReadOnlyList<uint> items)
        => new(GgufValueType.Array, (GgufValueType.Uint32, Box(items)));

    public static GgufValue Int32Array(IReadOnlyList<int> items)
        => new(GgufValueType.Array, (GgufValueType.Int32, Box(items)));

    public static GgufValue FloatArray(IReadOnlyList<float> items)
        => new(GgufValueType.Array, (GgufValueType.Float32, Box(items)));

    public static GgufValue BoolArray(IReadOnlyList<bool> items)
        => new(GgufValueType.Array, (GgufValueType.Bool, Box(items)));

    private static IReadOnlyList<object> Box<T>(IReadOnlyList<T> items)
    {
        var result = new object[items.Count];
        for (int i = 0; i < items.Count; i++)
        {
            result[i] = items[i]!;
        }
        return result;
    }

    public string AsString() => (string)_value;

    public uint AsUInt32() => Kind == GgufValueType.Uint32 ? (uint)_value
        : throw new InvalidOperationException($"value is {Kind}, not Uint32");

    public ulong AsUInt64() => Kind == GgufValueType.Uint64 ? (ulong)_value
        : throw new InvalidOperationException($"value is {Kind}, not Uint64");

    public int AsInt32() => Kind is GgufValueType.Int32 or GgufValueType.Uint32 ? Convert.ToInt32(_value)
        : throw new InvalidOperationException($"value is {Kind}, not Int32");

    public long AsInt64() => Kind is GgufValueType.Int64 or GgufValueType.Uint64 ? Convert.ToInt64(_value)
        : throw new InvalidOperationException($"value is {Kind}, not Int64");

    public float AsFloat() => Kind is GgufValueType.Float32 or GgufValueType.Float64 ? Convert.ToSingle(_value)
        : throw new InvalidOperationException($"value is {Kind}, not Float32");

    public bool AsBool() => Kind == GgufValueType.Bool ? (bool)_value
        : throw new InvalidOperationException($"value is {Kind}, not Bool");

    public (GgufValueType ElemType, IReadOnlyList<object> Items) AsArray()
        => ((GgufValueType, IReadOnlyList<object>))_value;
}

/// <summary>One tensor as described by a GGUF file's tensor-info table.</summary>
public sealed class GgufTensor
{
    public string Name { get; init; } = string.Empty;
    public GgufType Type { get; init; }
    /// <summary>Dimensions as stored in the file — GGUF stores the logical
    /// order reversed (fastest-varying first), matching ggml's ne[].</summary>
    public long[] Dims { get; init; } = Array.Empty<long>();
    public ulong Offset { get; init; }
}