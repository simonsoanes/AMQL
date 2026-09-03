namespace Amql.Inference;

/// <summary>
/// Row-based, caller-owned KV state: per absolute layer, position-ordered
/// key/value rows (post-norm, post-RoPE). Every row is retained — sliding
/// windows mask at attention time, never evict. Mirrors the reference's
/// <c>RowKvState</c>.
/// </summary>
public sealed class RowKvCache
{
    private readonly Dictionary<int, List<float[]>> _keys;
    private readonly Dictionary<int, List<float[]>> _values;

    public RowKvCache()
    {
        _keys = new Dictionary<int, List<float[]>>();
        _values = new Dictionary<int, List<float[]>>();
    }

    /// <summary>Rows cached for layer 0 — the session position.</summary>
    public int Position => _keys.TryGetValue(0, out var rows) ? rows.Count : 0;

    public void Append(int layer, ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        if (!_keys.TryGetValue(layer, out var keys))
        {
            keys = new List<float[]>();
            _keys[layer] = keys;
            _values[layer] = new List<float[]>();
        }
        keys.Add(key.ToArray());
        _values[layer].Add(value.ToArray());
    }

    public IReadOnlyList<float[]> Keys(int layer) =>
        _keys.TryGetValue(layer, out var keys) ? keys : Array.Empty<float[]>();

    public IReadOnlyList<float[]> Values(int layer) =>
        _values.TryGetValue(layer, out var values) ? values : Array.Empty<float[]>();

    /// <summary>Clears all cached rows (a fresh session).</summary>
    public void Reset()
    {
        _keys.Clear();
        _values.Clear();
    }
}

/// <summary>Per-layer KV geometry the runtime derives from the plan.</summary>
public sealed record LayerKvGeometry(int KvDim, long? Window);