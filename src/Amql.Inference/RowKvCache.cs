namespace Amql.Inference;

/// <summary>
/// Row-based, caller-owned KV state: per absolute layer, position-ordered
/// key/value rows (post-norm, post-RoPE). Every row is retained — sliding
/// windows mask at attention time, never evict. Mirrors the reference's
/// <c>RowKvState</c>.
/// <para>
/// Each layer's rows live in one contiguous, geometrically grown buffer, so a
/// decode step appends its row in amortised constant time and attention reads
/// the whole span in place — no per-step rebuild of a [rows × width] matrix,
/// which made every step cost O(context) copying per layer.
/// </para>
/// </summary>
public sealed class RowKvCache
{
    private sealed class LayerRows
    {
        public float[] Keys = Array.Empty<float>();
        public float[] Values = Array.Empty<float>();
        public int Count;
        public int KeyWidth;
        public int ValueWidth;
    }

    private readonly Dictionary<int, LayerRows> _layers = new();

    /// <summary>Rows cached for layer 0 — the session position.</summary>
    public int Position => _layers.TryGetValue(0, out var rows) ? rows.Count : 0;

    public void Append(int layer, ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        if (!_layers.TryGetValue(layer, out var rows))
        {
            rows = new LayerRows { KeyWidth = key.Length, ValueWidth = value.Length };
            _layers[layer] = rows;
        }
        if (key.Length != rows.KeyWidth || value.Length != rows.ValueWidth)
        {
            throw new ArgumentException(
                $"layer {layer} caches {rows.KeyWidth}/{rows.ValueWidth}-wide key/value rows, not {key.Length}/{value.Length}");
        }
        if ((rows.Count + 1) * rows.KeyWidth > rows.Keys.Length || (rows.Count + 1) * rows.ValueWidth > rows.Values.Length)
        {
            int capacity = Math.Max(16, rows.Count * 2);
            Array.Resize(ref rows.Keys, capacity * rows.KeyWidth);
            Array.Resize(ref rows.Values, capacity * rows.ValueWidth);
        }
        key.CopyTo(rows.Keys.AsSpan(rows.Count * rows.KeyWidth));
        value.CopyTo(rows.Values.AsSpan(rows.Count * rows.ValueWidth));
        rows.Count++;
    }

    /// <summary>Number of rows cached for a layer.</summary>
    public int Count(int layer) => _layers.TryGetValue(layer, out var rows) ? rows.Count : 0;

    /// <summary>The layer's rows in place: row j of the keys is
    /// <c>Keys[j·KeyWidth ..]</c>. The buffers may be longer than
    /// <c>Count</c> rows; anything past them is unused capacity. They are
    /// replaced (not mutated) when they grow, so a caller holding them keeps a
    /// consistent view of the rows it read.</summary>
    public (float[] Keys, int KeyWidth, float[] Values, int ValueWidth, int Count) Buffers(int layer) =>
        _layers.TryGetValue(layer, out var rows)
            ? (rows.Keys, rows.KeyWidth, rows.Values, rows.ValueWidth, rows.Count)
            : (Array.Empty<float>(), 0, Array.Empty<float>(), 0, 0);

    public IReadOnlyList<float[]> Keys(int layer) => CopyRows(layer, keys: true);

    public IReadOnlyList<float[]> Values(int layer) => CopyRows(layer, keys: false);

    private IReadOnlyList<float[]> CopyRows(int layer, bool keys)
    {
        if (!_layers.TryGetValue(layer, out var rows))
        {
            return Array.Empty<float[]>();
        }
        int width = keys ? rows.KeyWidth : rows.ValueWidth;
        var source = keys ? rows.Keys : rows.Values;
        var result = new float[rows.Count][];
        for (int j = 0; j < rows.Count; j++)
        {
            result[j] = source.AsSpan(j * width, width).ToArray();
        }
        return result;
    }

    /// <summary>Clears all cached rows (a fresh session).</summary>
    public void Reset() => _layers.Clear();
}

/// <summary>Per-layer KV geometry the runtime derives from the plan.</summary>
public sealed record LayerKvGeometry(int KvDim, long? Window);
