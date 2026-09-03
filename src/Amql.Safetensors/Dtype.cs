namespace Amql.Safetensors;

/// <summary>
/// The safetensors dtype vocabulary. Labels match the uppercase spellings
/// used in the file header ("F32", "BF16", "F8_E4M3", ...) which the
/// VINDEX3 segment format carries through verbatim.
/// </summary>
public enum Dtype
{
    F64 = 0,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    BOOL,
    F8_E4M3,
    F8_E5M2,
    F8_E8M0,
}

public static class DtypeExtensions
{
    private static readonly Dictionary<string, Dtype> ByLabel = new(StringComparer.Ordinal)
    {
        ["F64"] = Dtype.F64,
        ["F32"] = Dtype.F32,
        ["F16"] = Dtype.F16,
        ["BF16"] = Dtype.BF16,
        ["I64"] = Dtype.I64,
        ["I32"] = Dtype.I32,
        ["I16"] = Dtype.I16,
        ["I8"] = Dtype.I8,
        ["U8"] = Dtype.U8,
        ["BOOL"] = Dtype.BOOL,
        ["F8_E4M3"] = Dtype.F8_E4M3,
        ["F8_E5M2"] = Dtype.F8_E5M2,
        ["F8_E8M0"] = Dtype.F8_E8M0,
    };

    public static string Label(this Dtype dtype) => dtype switch
    {
        Dtype.F64 => "F64",
        Dtype.F32 => "F32",
        Dtype.F16 => "F16",
        Dtype.BF16 => "BF16",
        Dtype.I64 => "I64",
        Dtype.I32 => "I32",
        Dtype.I16 => "I16",
        Dtype.I8 => "I8",
        Dtype.U8 => "U8",
        Dtype.BOOL => "BOOL",
        Dtype.F8_E4M3 => "F8_E4M3",
        Dtype.F8_E5M2 => "F8_E5M2",
        Dtype.F8_E8M0 => "F8_E8M0",
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, null),
    };

    public static Dtype FromLabel(string label)
    {
        if (ByLabel.TryGetValue(label, out var dtype))
        {
            return dtype;
        }
        throw new SafetensorsException($"unknown safetensors dtype label '{label}'");
    }

    /// <summary>Number of bytes one element occupies.</summary>
    public static int ElementSize(this Dtype dtype) => dtype switch
    {
        Dtype.F64 or Dtype.I64 => 8,
        Dtype.F32 or Dtype.I32 => 4,
        Dtype.F16 or Dtype.BF16 or Dtype.I16 => 2,
        _ => 1,
    };

    /// <summary>
    /// Whether the byte pattern can be widened to f32 by
    /// <see cref="BitPattern"/>. Mirrors the reference's
    /// <c>tensor_to_f32</c> dispatch: F16/BF16/FP8/I8 are decoded, F32 is
    /// copied; the rest are refused rather than guessed.
    /// </summary>
    public static bool IsWidenableToF32(this Dtype dtype) => dtype switch
    {
        Dtype.F32 or Dtype.F16 or Dtype.BF16 or Dtype.I8 or Dtype.F8_E4M3 or Dtype.F8_E5M2 or Dtype.F8_E8M0 => true,
        _ => false,
    };
}