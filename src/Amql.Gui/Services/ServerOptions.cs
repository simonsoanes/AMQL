using System.IO;

namespace Amql.Gui.Services;

/// <summary>
/// The server set-up both front ends offer: the optional behaviours, the
/// format and context choices, and the summaries shown beside them. Kept here
/// so the desktop and text UIs cannot drift apart in what they let you set.
/// </summary>
public static class ServerOptions
{
    /// <summary>The server's optional behaviours, each a flag that turns a default off (or on).</summary>
    public static readonly (string Label, string Help, Func<ServerProfile, bool> Get, Action<ServerProfile, bool> Set)[] Behaviours =
    {
        ("Don't warm up models", "--no-warm: skip the warm-up pass at start (faster start, slower first request)", p => p.NoWarm, (p, v) => p.NoWarm = v),
        ("One request at a time", "--no-batch: no continuous batching, even on the GPU", p => p.NoBatch, (p, v) => p.NoBatch = v),
        ("No prefix reuse", "--no-prefix-reuse: prefill every prompt whole", p => p.NoPrefixReuse, (p, v) => p.NoPrefixReuse = v),
        ("Page prompt state to disk", "--kv-disk: idle prompt state goes to kv-cache in the model's container (1 GiB, deleted after 30 min) and loads back instead of prefilling", p => p.KvDisk, (p, v) => p.KvDisk = v),
        ("BF16 batches (faster)", "--batch-bf16: batches of 4+ on tensor cores, ~1.6x throughput; replies depend slightly on their batch", p => p.BatchBf16, (p, v) => p.BatchBf16 = v),
        ("No loop guard", "--no-loop-guard: let a reply repeat itself to its token limit", p => p.NoLoopGuard, (p, v) => p.NoLoopGuard = v),
        ("Free tool calls", "--free-tool-calls: do not hold calls to the request's declared tools", p => p.FreeToolCalls, (p, v) => p.FreeToolCalls = v),
        ("No mention rule", "--no-mention-rule: do not open optional parameters the conversation names", p => p.NoMentionRule, (p, v) => p.NoMentionRule = v),
        ("No announce rule", "--no-announce-rule: allow a reply to end on a colon without a call", p => p.NoAnnounceRule, (p, v) => p.NoAnnounceRule = v),
        ("Allow repeated calls", "--allow-repeated-calls: allow a call that already returned the same result twice", p => p.AllowRepeatedCalls, (p, v) => p.AllowRepeatedCalls = v),
        ("Keep line marks", "--keep-line-marks: show tool results with their line decoration", p => p.KeepLineMarks, (p, v) => p.KeepLineMarks = v),
    };

    /// <summary>--gpu-format for the whole server.</summary>
    public static readonly (string Label, string Value)[] GpuFormats =
    {
        ("As stored", ""), ("FP16", "fp16"), ("NVFP4 (4-bit, GPTQ copy)", "nvfp4"), ("MXFP4 (4-bit, GPTQ copy)", "mxfp4"),
    };

    /// <summary>--kv-format.</summary>
    public static readonly (string Label, string Value)[] KvFormats =
    {
        ("Server default (FP16)", ""), ("NVFP4 (4-bit, long contexts)", "nvfp4"), ("F32", "f32"),
    };

    /// <summary>A model's own GPU format.</summary>
    public static readonly (string Label, string Value)[] ModelFormats =
    {
        ("Server default", ""), ("As stored / FP16", "fp16"), ("NVFP4 (4-bit)", "nvfp4"), ("MXFP4 (4-bit)", "mxfp4"),
    };

    /// <summary>A chat model's reasoning default.</summary>
    public static readonly (string Label, bool? Value)[] ThinkChoices =
    {
        ("Server default", null), ("Think", true), ("Don't think", false),
    };

    /// <summary>Context lengths offered before "Other…" (a typed value).</summary>
    public static readonly (string Label, string Value)[] ContextChoices =
    {
        ("The model's own (default)", ""), ("512K tokens", "512K"), ("1M tokens", "1M"),
    };

    public static bool SamePath(string a, string b)
    {
        try
        {
            return string.Equals(Path.GetFullPath(a).TrimEnd('\\', '/'), Path.GetFullPath(b).TrimEnd('\\', '/'),
                OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException or PathTooLongException)
        {
            return false;
        }
    }

    public static string ShortPath(string p) => p.Length > 48 ? "…" + p[^46..] : p;

    /// <summary>The entry the profile holds for a container, if any.</summary>
    public static ServedModel? Served(ServerProfile profile, string path) =>
        profile.Models.FirstOrDefault(s => SamePath(s.Path, path));

    /// <summary>The profile's entry for a container, made (unticked) when it has none.</summary>
    public static ServedModel Ensure(ServerProfile profile, string path)
    {
        var served = Served(profile, path);
        if (served is null)
        {
            served = new ServedModel { Path = path, Enabled = false };
            profile.Models.Add(served);
        }
        return served;
    }

    /// <summary>Ticks or unticks a model. Unticked models with nothing set are dropped,
    /// so the list does not fill with leftovers.</summary>
    public static void SetEnabled(ServerProfile profile, string path, bool enabled)
    {
        Ensure(profile, path).Enabled = enabled;
        profile.Models.RemoveAll(s => !s.Enabled && s.Id.Length == 0 && s.Format.Length == 0 && s.MaxContext.Length == 0 && s.Think is null
                                      && s.MaxTokens is null && s.Temperature is null && s.TopP is null && s.TopK is null && s.Adapters.Count == 0);
    }

    /// <summary>One line on what is chosen and whether it fits.</summary>
    public static string SelectionSummary(IReadOnlyList<ModelEntry> found, ServerProfile profile)
    {
        var chosen = found.Where(f => Served(profile, f.Path)?.Enabled == true).ToList();
        if (chosen.Count == 0)
        {
            return "No models chosen — tick the ones to load.";
        }
        long bytes = chosen.Sum(c => c.SizeBytes);
        var kinds = chosen.GroupBy(c => c.KindLabel).Select(g => $"{g.Count()} {g.Key.ToLowerInvariant()}");
        string fit = profile.MaxRamGb is null && bytes > SystemInfo.TotalRamBytes * 0.7
            ? " — more than fits comfortably in memory; consider a RAM limit"
            : "";
        return $"{chosen.Count} model{(chosen.Count == 1 ? "" : "s")} chosen ({string.Join(", ", kinds)}) · {Format.Bytes(bytes)} on disk{fit}";
    }

    /// <summary>What the server would be started with, and how it was last started.</summary>
    public static string CommandLinePreview(ServerProfile profile)
    {
        var models = profile.Models.Where(m => m.Enabled).Select(m => $"{m.EffectiveId}={m.Path}");
        string text = "amql-server --config <" + string.Join(", ", models) + "> " +
                      string.Join(' ', ServerController.Arguments(profile).Select(a => a.Contains(' ') ? $"\"{a}\"" : a));
        if (ServerController.Instance.LastCommandLine is { Length: > 0 } last)
        {
            text += Environment.NewLine + Environment.NewLine + "Last started as:" + Environment.NewLine + last;
        }
        return text;
    }

    /// <summary>The help line under the RAM limit.</summary>
    public static string RamHelp(double? limitGb)
    {
        double totalRam = SystemInfo.TotalRamBytes / (double)(1L << 30);
        return limitGb is { } v
            ? $"The server aims to stay under {v:0} of {totalRam:0} GB: weights stream through a cache sized to fit. Slower, but runs models larger than memory."
            : $"Models are held in memory whole ({totalRam:0} GB installed). Limit it to run models bigger than that, or to leave room for other work.";
    }

    /// <summary>The help line under the GPU memory limit.</summary>
    public static string GpuHelp(double? limitGb, GpuSample? gpu)
    {
        if (gpu is null)
        {
            return "No NVIDIA GPU was found (nvidia-smi did not answer), so models run on the CPU.";
        }
        double total = gpu.TotalBytes / (double)(1L << 30);
        string unified = gpu.Unified ? " The GPU shares the system's memory, so this also comes out of RAM." : "";
        return (limitGb switch
        {
            null => "The server takes the GPU's free memory, less a margin.",
            0 => "0 GB keeps every weight on the host; the GPU still computes.",
            { } v => $"Weights may take up to {v:0.#} of {total:0} GB on the GPU; the rest stays on the host.",
        }) + unified;
    }
}
