using System.Runtime.InteropServices;

namespace Amql.Gguf;

/// <summary>
/// ggml's own reference quantizers, called in-process from a llama.cpp
/// build's <c>ggml-base</c> library. AMQL encodes the simple block formats
/// itself (Q8_0, Q4_0, MXFP4); the K-quants and the importance-guided IQ
/// family (IQ1_S … IQ4_XS) are lattice/codebook searches whose exact bytes
/// only ggml defines, so they go through <c>ggml_quantize_chunk</c> — the
/// same function <c>llama-quantize</c> calls — with an optional importance
/// matrix. The library is located from <c>AMQL_GGML</c> (a file or a
/// directory holding <c>ggml-base.dll</c> / <c>libggml-base.so</c>).
/// </summary>
public static unsafe class GgmlNative
{
    private delegate nuint QuantizeChunk(uint type, float* src, void* dst, long start, long nrows, long nPerRow, float* imatrix);
    private delegate byte RequiresImatrix(uint type);
    private delegate nuint RowSize(uint type, long ne);

    private static QuantizeChunk? _quantize;
    private static RequiresImatrix? _requires;
    private static RowSize? _rowSize;
    private static readonly object Gate = new();

    public static string? LoadedFrom { get; private set; }

    /// <summary>Loads the library once; throws naming what was searched when it cannot.</summary>
    public static void EnsureLoaded(string? hint = null)
    {
        lock (Gate)
        {
            if (_quantize is not null)
            {
                return;
            }
            var candidates = new List<string>();
            foreach (var root in new[] { hint, Environment.GetEnvironmentVariable("AMQL_GGML") })
            {
                if (string.IsNullOrEmpty(root))
                {
                    continue;
                }
                if (File.Exists(root))
                {
                    candidates.Add(root);
                }
                else
                {
                    candidates.Add(Path.Combine(root, "ggml-base.dll"));
                    candidates.Add(Path.Combine(root, "libggml-base.so"));
                    candidates.Add(Path.Combine(root, "libggml-base.dylib"));
                }
            }
            foreach (var path in candidates.Where(File.Exists))
            {
                // ggml-base resolves its siblings from its own directory.
                var handle = NativeLibrary.Load(Path.GetFullPath(path));
                _quantize = Marshal.GetDelegateForFunctionPointer<QuantizeChunk>(NativeLibrary.GetExport(handle, "ggml_quantize_chunk"));
                _requires = Marshal.GetDelegateForFunctionPointer<RequiresImatrix>(NativeLibrary.GetExport(handle, "ggml_quantize_requires_imatrix"));
                _rowSize = Marshal.GetDelegateForFunctionPointer<RowSize>(NativeLibrary.GetExport(handle, "ggml_row_size"));
                LoadedFrom = path;
                return;
            }
            throw new GgufException(
                "this quantization type is encoded by ggml itself — point AMQL_GGML (or --ggml) at a llama.cpp build's " +
                "ggml-base library; searched: " + (candidates.Count == 0 ? "(nothing set)" : string.Join(", ", candidates)));
        }
    }

    public static bool RequiresImportance(GgufType type)
    {
        EnsureLoaded();
        return _requires!((uint)type) != 0;
    }

    /// <summary>Quantizes a row-major [rows × cols] matrix. <paramref name="importance"/>
    /// (length <paramref name="cols"/>) weights each input column's error, as
    /// llama-quantize's <c>--imatrix</c> does.</summary>
    public static unsafe byte[] Quantize(GgufType type, float[] values, int cols, float[]? importance)
    {
        EnsureLoaded();
        long rows = values.Length / cols;
        long rowBytes = (long)_rowSize!((uint)type, cols);
        long expected = GgufTypeSizing.DataBytes(type, values.Length);
        if (rowBytes * rows != expected)
        {
            throw new GgufException($"{type}: ggml sizes a {cols}-wide row at {rowBytes} B, AMQL's table at {expected / rows} B");
        }
        if (importance is not null && importance.Length != cols)
        {
            throw new GgufException($"{type}: importance vector has {importance.Length} entries for {cols} columns");
        }
        if (importance is null && RequiresImportance(type))
        {
            throw new GgufException($"ggml refuses {type} without an importance matrix — pass --imatrix");
        }
        var output = new byte[expected];
        // Rows are independent: quantize in row bands across the cores.
        int bands = Math.Max(1, Math.Min(Environment.ProcessorCount, (int)(rows / 16)));
        long perBand = (rows + bands - 1) / bands;
        fixed (float* src = values)
        fixed (byte* dst = output)
        fixed (float* imp = importance)
        {
            nint srcPtr = (nint)src, dstPtr = (nint)dst, impPtr = (nint)imp;
            Parallel.For(0, bands, band =>
            {
                long start = band * perBand;
                long count = Math.Min(perBand, rows - start);
                if (count <= 0)
                {
                    return;
                }
                _quantize!((uint)type, (float*)srcPtr, (byte*)dstPtr, start * cols, count, cols, (float*)impPtr);
            });
        }
        return output;
    }
}

/// <summary>
/// An importance matrix in llama.cpp's GGUF imatrix format: per weight
/// tensor, <c>&lt;name&gt;.in_sum2</c> (Σ x² per input column) and
/// <c>&lt;name&gt;.counts</c>. The importance of column j is in_sum2[j] / count.
/// </summary>
public sealed class ImportanceMatrix
{
    private readonly Dictionary<string, float[]> _columns = new(StringComparer.Ordinal);

    public int Count => _columns.Count;

    public static ImportanceMatrix Load(string path)
    {
        var result = new ImportanceMatrix();
        using var reader = GgufReader.Open(path);
        foreach (var tensor in reader.Tensors.Where(t => t.Name.EndsWith(".in_sum2", StringComparison.Ordinal)))
        {
            string name = tensor.Name[..^".in_sum2".Length];
            var sums = MemoryMarshal.Cast<byte, float>(reader.ReadBytes(tensor.Name)).ToArray();
            float count = 1f;
            if (reader.Contains(name + ".counts"))
            {
                var counts = MemoryMarshal.Cast<byte, float>(reader.ReadBytes(name + ".counts")).ToArray();
                count = Math.Max(1f, counts[0]);
            }
            result._columns[name] = sums.Select(v => v / count).ToArray();
        }
        return result;
    }

    public float[]? For(string ggufName) => _columns.TryGetValue(ggufName, out var v) ? v : null;
}

/// <summary>
/// A variable-quantization recipe: ordered <c>regex type</c> lines matched
/// against GGUF tensor names (first match wins), <c>#</c> comments, and a
/// final <c>* type</c> default. Norms, the conv kernel and other 1-D tensors
/// are never quantized whatever the recipe says.
/// </summary>
public sealed class QuantRecipe
{
    private readonly List<(System.Text.RegularExpressions.Regex Pattern, GgufType Type)> _rules = new();

    public string Source { get; }

    private QuantRecipe(string source) => Source = source;

    public static QuantRecipe Parse(string text, string source = "recipe")
    {
        var recipe = new QuantRecipe(source);
        foreach (var raw in text.Split('\n'))
        {
            string line = raw.Split('#')[0].Trim();
            if (line.Length == 0)
            {
                continue;
            }
            var parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != 2)
            {
                throw new GgufException($"{source}: '{raw.Trim()}' is not '<regex> <type>'");
            }
            string pattern = parts[0] == "*" ? ".*" : parts[0];
            recipe._rules.Add((new System.Text.RegularExpressions.Regex($"^(?:{pattern})$"), ParseType(parts[1])));
        }
        return recipe;
    }

    public GgufType? TypeFor(string ggufName) =>
        _rules.Where(r => r.Pattern.IsMatch(ggufName)).Select(r => (GgufType?)r.Type).FirstOrDefault();

    public static GgufType ParseType(string name) => name.ToLowerInvariant() switch
    {
        "f32" => GgufType.F32,
        "f16" => GgufType.F16,
        "bf16" => GgufType.BF16,
        "q8_0" => GgufType.Q8_0,
        "q5_0" => GgufType.Q5_0,
        "q5_1" => GgufType.Q5_1,
        "q4_0" => GgufType.Q4_0,
        "q4_1" => GgufType.Q4_1,
        "q2_k" => GgufType.Q2_K,
        "q3_k" => GgufType.Q3_K,
        "q4_k" => GgufType.Q4_K,
        "q5_k" => GgufType.Q5_K,
        "q6_k" => GgufType.Q6_K,
        "iq1_s" => GgufType.IQ1_S,
        "iq1_m" => GgufType.IQ1_M,
        "iq2_xxs" => GgufType.IQ2_XXS,
        "iq2_xs" => GgufType.IQ2_XS,
        "iq2_s" => GgufType.IQ2_S,
        "iq3_xxs" => GgufType.IQ3_XXS,
        "iq3_s" => GgufType.IQ3_S,
        "iq4_nl" => GgufType.IQ4_NL,
        "iq4_xs" => GgufType.IQ4_XS,
        "tq1_0" => GgufType.TQ1_0,
        "tq2_0" => GgufType.TQ2_0,
        "mxfp4" => GgufType.Mxfp4,
        "nvfp4" => GgufType.Nvfp4,
        "q1_0" => GgufType.Q1_0,
        "q2_0" => GgufType.Q2_0,
        _ => throw new GgufException($"unknown ggml type '{name}'"),
    };
}
