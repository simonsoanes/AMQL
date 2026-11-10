namespace Amql.Safetensors;

/// <summary>
/// A model directory of safetensors shards, mirroring the reference loader
/// (<c>larql-models/src/loading/safetensors/mod.rs</c>) at the level of its
/// building blocks: shard discovery + sorting, the MLX <c>weights/</c>
/// fallback, cross-shard name lookup and on-demand dtype widening.
/// Architecture detection / key-prefix stripping are the caller's concern
/// here — the directory answers "where is this tensor, and what are its
/// bytes".</summary>
public sealed class ModelDirectory : IDisposable
{
    private readonly List<SafetensorsFile> _shards = new();

    private ModelDirectory(string root, IReadOnlyList<string> shardPaths)
    {
        Root = root;
        foreach (var shardPath in shardPaths)
        {
            _shards.Add(SafetensorsFile.Open(shardPath));
        }
    }

    public string Root { get; }

    public IReadOnlyList<SafetensorsFile> Shards => _shards;

    /// <summary>All tensor names across the shards, in shard-then-header order.</summary>
    public IEnumerable<string> TensorNames => _shards.SelectMany(s => s.TensorNames);

    /// <summary>Discovers the safetensors shards of a model directory:
    /// <c>*.safetensors</c> at the root, or under <c>weights/</c> when the
    /// root holds none (MLX convention), sorted by path.</summary>
    public static IReadOnlyList<string> DiscoverShards(string directory)
    {
        if (!Directory.Exists(directory))
        {
            throw new SafetensorsException($"model directory '{directory}' does not exist");
        }

        var files = Directory.EnumerateFiles(directory, "*.safetensors").ToList();
        if (files.Count == 0)
        {
            var weightsDir = Path.Combine(directory, "weights");
            if (Directory.Exists(weightsDir))
            {
                files = Directory.EnumerateFiles(weightsDir, "*.safetensors").ToList();
            }
        }
        files.Sort(StringComparer.Ordinal);
        return files;
    }

    public static ModelDirectory Open(string directory) =>
        new(directory, DiscoverShards(directory));

    /// <summary>Opens a Hugging Face checkpoint's own weights: the shards
    /// <c>model.safetensors.index.json</c> names, else <c>model.safetensors</c>,
    /// else every shard <see cref="DiscoverShards"/> finds. Side files such as
    /// <c>mtp.safetensors</c> are left out whenever the checkpoint says which
    /// files are its own.</summary>
    public static ModelDirectory OpenCheckpoint(string directory)
    {
        var index = System.IO.Path.Combine(directory, "model.safetensors.index.json");
        if (File.Exists(index))
        {
            using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(index));
            var files = doc.RootElement.GetProperty("weight_map").EnumerateObject()
                .Select(p => p.Value.GetString()!)
                .Distinct(StringComparer.Ordinal)
                .OrderBy(f => f, StringComparer.Ordinal)
                .Select(f => System.IO.Path.Combine(directory, f))
                .ToList();
            return new ModelDirectory(directory, files);
        }
        var single = System.IO.Path.Combine(directory, "model.safetensors");
        return File.Exists(single) ? new ModelDirectory(directory, new[] { single }) : Open(directory);
    }

    public bool TryGet(string name, out SafetensorsFile file, out TensorInfo info)
    {
        foreach (var shard in _shards)
        {
            if (shard.Contains(name))
            {
                file = shard;
                info = shard.GetTensor(name);
                return true;
            }
        }
        file = null!;
        info = null!;
        return false;
    }

    public (SafetensorsFile File, TensorInfo Info) Get(string name)
    {
        if (TryGet(name, out var file, out var info))
        {
            return (file, info);
        }
        throw new SafetensorsException($"no tensor '{name}' in any shard of '{Root}'");
    }

    public byte[] ReadRawBytes(string name) => Get(name).File.ReadBytes(name);

    /// <summary>Copies a whole tensor payload into <paramref name="stream"/>
    /// in bounded chunks, never holding it as one buffer.</summary>
    public void CopyTo(string name, Stream stream, int chunkBytes = 64 * 1024 * 1024)
    {
        var (file, info) = Get(name);
        for (long done = 0; done < info.DataLength; done += chunkBytes)
        {
            stream.Write(file.ReadBytes(info, done, (int)Math.Min(chunkBytes, info.DataLength - done)));
        }
    }

    /// <summary>Reads a sub-range of a shard tensor's payload — chunked
    /// reads for tensors beyond the 2 GiB single-buffer ceiling.</summary>
    public byte[] ReadBytes(string name, long offset, int count)
    {
        var (file, info) = Get(name);
        return file.ReadBytes(info, offset, count);
    }

    public float[] DecodeF32(string name) => Get(name).File.DecodeF32(name);

    /// <summary>
    /// Key normalisation mirroring the reference <c>normalize_key</c>: the
    /// first matching prefix (longest-first) is stripped, otherwise the key
    /// passes through unchanged.
    /// </summary>
    public static string NormalizeKey(string key, IReadOnlyList<string> prefixes)
    {
        foreach (var prefix in prefixes)
        {
            if (key.StartsWith(prefix, StringComparison.Ordinal))
            {
                return key[prefix.Length..];
            }
        }
        return key;
    }

    public void Dispose()
    {
        foreach (var shard in _shards)
        {
            shard.Dispose();
        }
        _shards.Clear();
    }
}