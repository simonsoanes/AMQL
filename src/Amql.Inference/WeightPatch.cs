using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// One edited weight: the f32 delta to ADD to the base tensor — patched
/// value minus original, dense, element-aligned with the base tensor's
/// row-major layout. The entry's <see cref="Shape"/> is the BASE tensor's
/// shape, so application can verify the delta against the container's
/// actual tensor before merging.
/// </summary>
public sealed record WeightPatchEntry(
    string ObjectId,
    string TensorName,
    long[] Shape,
    float[] Delta)
{
    /// <summary>The patch-file tensor name: <c>objectId/tensorName</c>.
    /// Object ids and segment tensor names never contain '/', so the first
    /// slash unambiguously splits the two.</summary>
    public string Key => ObjectId + "/" + TensorName;

    public long ElementCount => WeightPatch.ElementCount(Shape);
}

/// <summary>
/// A persistent weight patch: a set of f32 deltas over base tensors,
/// stored as a safetensors file (one dense F32 tensor per edited weight,
/// named <c>objectId/tensorName</c>, shaped like the base tensor) plus a
/// <c>__metadata__</c> block naming the format. Applying a patch ADDs every
/// delta to the base weight as the runtime loads it — the container itself
/// is never rewritten, so integrity verification and the original model
/// stay untouched.
/// </summary>
public sealed class WeightPatch
{
    public const string PatchFormat = "amql-patch-v1";
    private const string MetadataFormat = "format";
    private const string MetadataModel = "model";

    private readonly Dictionary<string, WeightPatchEntry> _byKey = new(StringComparer.Ordinal);

    private WeightPatch(IEnumerable<WeightPatchEntry> entries, string? model)
    {
        Model = model;
        foreach (var entry in entries)
        {
            _byKey[entry.Key] = entry;
        }
    }

    public IReadOnlyList<WeightPatchEntry> Entries => _byKey.Values.ToList();

    /// <summary>The base model id recorded at authoring time (null when the
    /// patch is hand-built without one).</summary>
    public string? Model { get; }

    public bool TryGet(string objectId, string tensorName, out WeightPatchEntry entry) =>
        _byKey.TryGetValue(objectId + "/" + tensorName, out entry!);

    /// <summary>Builds an in-memory patch (no file round trip) — the
    /// composer path for code that already holds entry lists.</summary>
    public static WeightPatch FromEntries(IReadOnlyList<WeightPatchEntry> entries, string? model = null) =>
        new(entries, model);

    // ── file I/O ───────────────────────────────────────────────────────────

    /// <summary>Reads a patch file. Refuses anything that is not an
    /// <c>amql-patch-v1</c> safetensors file.</summary>
    public static WeightPatch Load(string path)
    {
        using var file = SafetensorsFile.Open(path);
        if (file.Metadata is null || !file.Metadata.TryGetValue(MetadataFormat, out var format) ||
            format != PatchFormat)
        {
            throw new SafetensorsException(
                $"'{path}' is not an AMQL weight patch (missing metadata format '{PatchFormat}')");
        }

        var entries = new List<WeightPatchEntry>();
        foreach (var tensorName in file.TensorNames)
        {
            (string objectId, string tensor) = SplitKey(path, tensorName);
            var info = file.GetTensor(tensorName);
            if (info.Dtype != Dtype.F32)
            {
                throw new SafetensorsException(
                    $"patch tensor '{tensorName}' is {info.Dtype.Label()} — a patch stores f32 deltas");
            }
            var delta = BitPattern.WidenToF32(info.Dtype, file.ReadBytes(info));
            entries.Add(new WeightPatchEntry(objectId, tensor, info.Shape, delta));
        }

        file.Metadata.TryGetValue(MetadataModel, out var model);
        return new WeightPatch(entries, model);
    }

    /// <summary>Writes a patch file: one F32 tensor per entry, shaped like
    /// the base tensor, plus format/model metadata.</summary>
    public static void Save(string path, IEnumerable<WeightPatchEntry> entries, string? model = null)
    {
        var list = entries.Where(e => HasNonZero(e.Delta)).ToList();
        if (list.Count == 0)
        {
            throw new SafetensorsException("refusing to write an empty patch — nothing changed");
        }

        var metadata = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            [MetadataFormat] = PatchFormat,
        };
        if (model is not null)
        {
            metadata[MetadataModel] = model;
        }

        SafetensorsWriter.Write(path, list.Select(e => new TensorPayload
        {
            Name = e.Key,
            Dtype = Dtype.F32,
            Shape = e.Shape,
            Data = F32Bytes(e.Delta),
        }), metadata);
    }

    /// <summary>Whether an entry still describes a real change (exact-zero
    /// deltas carry no information and are dropped at write time).</summary>
    public static bool HasNonZero(float[] delta)
    {
        foreach (var v in delta)
        {
            if (v != 0f)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Shape binds the patch to the container: every entry must
    /// resolve to a real object/tensor whose element count matches the
    /// entry's base shape. Throws a typed error naming the first
    /// mismatch.</summary>
    public void ValidateAgainst(Vindex3Container container)
    {
        using var store = container.CreateOperandStore();
        foreach (var entry in Entries)
        {
            OperandResolution resolution;
            try
            {
                resolution = store.Resolve(entry.ObjectId, entry.TensorName);
            }
            catch (ContainerException e)
            {
                throw new ContainerException(
                    $"patch entry '{entry.Key}' does not resolve in the container: {e.Message}", e);
            }
            if (ElementCount(resolution.Shape) != entry.ElementCount)
            {
                throw new ContainerException(
                    $"patch entry '{entry.Key}' holds {entry.ElementCount} deltas shaped " +
                    $"[{string.Join("x", entry.Shape)}] but the container's tensor is " +
                    $"[{string.Join("x", resolution.Shape)}]");
            }
        }
    }

    // ── helpers ────────────────────────────────────────────────────────────

    /// <summary>Product of the shape's dimensions.</summary>
    public static long ElementCount(long[] shape)
    {
        long n = 1;
        foreach (var dim in shape)
        {
            n = checked(n * dim);
        }
        return n;
    }

    private static (string ObjectId, string Tensor) SplitKey(string path, string key)
    {
        int slash = key.IndexOf('/');
        if (slash <= 0 || slash == key.Length - 1)
        {
            throw new SafetensorsException(
                $"'{path}': patch tensor name '{key}' is not '<objectId>/<tensorName>'");
        }
        return (key[..slash], key[(slash + 1)..]);
    }

    private static byte[] F32Bytes(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}