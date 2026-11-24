using System.Text.Json.Nodes;

namespace Amql.Merge;

/// <summary>
/// What running a container at 4 bits means by default: its GPTQ-rounded
/// copy, built once beside it (<c>{container}-nvfp4-gptq</c>) and reused.
/// Quantising each weight to nearest as it loads costs the 4B +4.3%
/// perplexity; the GPTQ copy costs +1.1%, for about a minute's work the first
/// time. A container already stored at 4 bits is used as it is.
/// </summary>
public static class Fp4Cache
{
    /// <summary>The container to load for <paramref name="format"/>:
    /// <paramref name="containerDir"/> itself when it is stored at 4 bits,
    /// otherwise its GPTQ copy, built now if missing or older than it.</summary>
    public static string Resolve(string containerDir, Fp4Format format, Action<string>? log = null)
    {
        containerDir = Path.GetFullPath(containerDir).TrimEnd('/', '\\');
        if (Quantisation(containerDir) is not null || !Quantisable(containerDir))
        {
            return containerDir;
        }
        string name = format == Fp4Format.Nvfp4 ? "nvfp4" : "mxfp4";
        string cached = $"{containerDir}-{name}-gptq";
        if (Quantisation(cached) is { } q && (string?)q["format"] == name &&
            File.GetLastWriteTimeUtc(Path.Combine(cached, "index.json")) >= Newest(containerDir))
        {
            return cached;
        }
        log?.Invoke($"building {Path.GetFileName(cached)} (GPTQ, once; about a minute for a 4B)");
        string building = $"{cached}.building";
        if (Directory.Exists(building))
        {
            Directory.Delete(building, recursive: true);
        }
        try
        {
            QuantiseContainer.Transform(containerDir, building, new QuantiseOptions(format) { Log = log });
        }
        catch (Exception e) when (e is MergeException or InvalidOperationException or NotSupportedException or IOException)
        {
            // Not a stack GPTQ can calibrate (no GPU, not BF16, another architecture):
            // the weights are then quantised to nearest as they load, as before.
            log?.Invoke($"no GPTQ copy ({e.Message}); quantising to nearest as it loads");
            if (Directory.Exists(building))
            {
                Directory.Delete(building, recursive: true);
            }
            return containerDir;
        }
        if (Directory.Exists(cached))
        {
            Directory.Delete(cached, recursive: true);
        }
        Directory.Move(building, cached);
        return cached;
    }

    /// <summary>The <c>quantisation</c> record of a container's index, or null
    /// when it has none (or is not a container).</summary>
    public static JsonObject? Quantisation(string containerDir)
    {
        string index = Path.Combine(containerDir, "index.json");
        if (!File.Exists(index))
        {
            return null;
        }
        return JsonNode.Parse(File.ReadAllText(index))?["quantisation"] as JsonObject;
    }

    /// <summary>Whether the container has a decoder stack whose projections
    /// are not already at 4 bits (an imported NVFP4 checkpoint is used as it is).</summary>
    private static bool Quantisable(string containerDir)
    {
        using var container = Amql.Vindex3.Vindex3Container.Open(containerDir);
        const string stack = "target.decoder_stack";
        if (!container.Index.Representations.Values.Any(r => r.Object == stack))
        {
            return false;
        }
        using var store = container.CreateOperandStore();
        return !store.TensorNames(stack).Any(n => store.DtypeOf(stack, n) == Amql.Safetensors.Dtype.FP4);
    }

    /// <summary>When the container's own files last changed (adapters and
    /// other folders beside them do not count).</summary>
    private static DateTime Newest(string dir) =>
        Directory.EnumerateFiles(dir).Select(File.GetLastWriteTimeUtc).DefaultIfEmpty(DateTime.MinValue).Max();
}
