using Amql.Safetensors;

namespace Amql.Hf;

/// <summary>
/// G0 inventory: a model directory's safetensors shards, answering "which
/// tensors exist, with what facts, and where are the bytes". Header-only
/// lookups stay cheap; payloads are read on demand (the encoder copies
/// them verbatim — no widening at this layer).
/// </summary>
public sealed class HfInventory : IDisposable
{
    private readonly ModelDirectory _model;

    private HfInventory(string modelDir, ModelDirectory model)
    {
        Root = modelDir;
        _model = model;
    }

    public string Root { get; }

    public static HfInventory Open(string modelDir)
    {
        var shards = ModelDirectory.DiscoverShards(modelDir);
        if (shards.Count == 0)
        {
            throw new ModelConfigException(
                $"model directory '{modelDir}' contains no *.safetensors shards");
        }
        return new HfInventory(modelDir, ModelDirectory.Open(modelDir));
    }

    public IEnumerable<string> TensorNames => _model.TensorNames;

    public bool TryGet(string fullName, out TensorInfo info)
    {
        if (_model.TryGet(fullName, out _, out info))
        {
            return true;
        }
        info = null!;
        return false;
    }

    public TensorInfo Get(string fullName) => _model.Get(fullName).Info;

    /// <summary>Raw payload bytes — read verbatim from the mapping.</summary>
    public byte[] ReadBytes(string fullName) => _model.ReadRawBytes(fullName);

    /// <summary>Number of tensors whose name starts with the prefix.</summary>
    public int CountUnder(string prefix) => _model.TensorNames.Count(n => n.StartsWith(prefix, StringComparison.Ordinal));

    /// <summary>Total payload bytes of tensors under the prefix.</summary>
    public long BytesUnder(string prefix)
    {
        long total = 0;
        foreach (var name in _model.TensorNames)
        {
            if (name.StartsWith(prefix, StringComparison.Ordinal) && _model.TryGet(name, out _, out var info))
            {
                total += info.DataLength;
            }
        }
        return total;
    }

    public void Dispose() => _model.Dispose();
}