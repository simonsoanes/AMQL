using System.Globalization;

namespace Amql.Inference;

/// <summary>
/// Process-wide compute budget: the core count the parallel kernels may
/// saturate and the working-memory ceiling a session may hold. The
/// defaults are deliberately conservative — a core margin is left free and
/// the memory budget holds back a share of physical RAM instead of filling
/// it to the last GiB — because a large job that pegs every core or drives
/// the working set into the pagefile makes the machine unresponsive and
/// its own runtime slow (the widened-f32 model plus pagefile thrash).
///
/// Both knobs are overridable for machines with a strict job profile:
/// <c>AMQL_CORES</c> (int) caps the parallel degree and
/// <c>AMQL_MEMORY_GB</c> (decimal) sets the session memory ceiling in
/// GiB. The MXFP4 working set adds two more: <c>AMQL_WEIGHTS</c>
/// (<c>bf16</c> | <c>mxfp4</c>) selects the resident weight format, and
/// <c>AMQL_F32_CACHE_GB</c> bounds the dequantised f32 LRU that MXFP4
/// mode caches on top of the packs (a small default: packs are the
/// resident truth, dequant is cheap). Any parallel site that matters
/// (matmul, export, merge alignment) and the weight loader — which
/// widens whole models to f32 — read this budget, so a single change
/// applies everywhere.
/// </summary>
public static class ComputeBudget
{
    // Reserve up to one eighth of the logical cores, at least two, so the
    // machine stays responsive while a big job runs.
    private const int CoreMarginMax = 8;
    private const int CoreMarginMin = 2;

    // Hold back 30% of physical RAM: enough for the OS page cache and the
    // user's other work, little enough that a normal model still fits.
    private const double MemoryShare = 0.70;

    /// <summary>The core count the parallel kernels may saturate.</summary>
    public static int Cores { get; } = DiscoverCores();

    /// <summary>The working-memory ceiling (bytes) a session may hold.</summary>
    public static long MemoryBudgetBytes { get; } = DiscoverMemory();

    /// <summary>The MXFP4 dequant LRU cap: how many f32 bytes of
    /// dequantised stack projections stay resident on top of the packs.
    /// Defaults to a quarter of the memory budget (and is bounded by it).</summary>
    public static long ResidentCacheBytes { get; } = DiscoverResidentCache();

    public static string Describe() => $"{Cores} cores, {FormatBytes(MemoryBudgetBytes)} memory budget";

    private static int DiscoverCores()
    {
        string? raw = Environment.GetEnvironmentVariable("AMQL_CORES");
        if (int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int requested))
        {
            return Math.Clamp(requested, 1, Environment.ProcessorCount);
        }
        int margin = Math.Clamp(Environment.ProcessorCount / 8, CoreMarginMin, CoreMarginMax);
        return Math.Max(1, Environment.ProcessorCount - margin);
    }

    private static long DiscoverMemory()
    {
        string? raw = Environment.GetEnvironmentVariable("AMQL_MEMORY_GB");
        if (double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out double gb) && gb > 0)
        {
            return (long)(gb * 1024.0 * 1024.0 * 1024.0);
        }
        long total = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        if (total <= 0)
        {
            total = 8L << 30; // unknowable — assume a modest default
        }
        return (long)(total * MemoryShare);
    }

    private static long DiscoverResidentCache()
    {
        string? raw = Environment.GetEnvironmentVariable("AMQL_F32_CACHE_GB");
        if (double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out double gb) && gb > 0)
        {
            return (long)(gb * 1024.0 * 1024.0 * 1024.0);
        }
        long quarter = MemoryBudgetBytes / 4;
        return Math.Max(1L << 30, quarter); // at least 1 GiB, at most a quarter of the budget
    }

    private static string FormatBytes(long bytes) =>
        bytes >= 1L << 30
            ? $"{bytes / (double)(1L << 30):0.0} GiB"
            : $"{bytes / (double)(1L << 20):0.0} MiB";
}

/// <summary>Resident weight format a session's <see cref="WeightLoader"/>
/// uses. Chosen by <c>AMQL_WEIGHTS</c> at construction; every runtime
/// consumer inherits it.</summary>
public enum WeightWorkingSet
{
    /// <summary>Widen every tensor to f32 and keep it resident (the
    /// byte-exact reference path). Simple and fast for small models; the
    /// working set doubles the container payload for large ones.</summary>
    ResidentF32,

    /// <summary>Stack projections resident as their stored bytes (BF16),
    /// widened into a bounded f32 LRU on access. Same numerics as the
    /// resident path (widening is lossless) at ~2× memory saving — the
    /// CPU mode when the f32 working set would thrash but byte-exact
    /// output matters.</summary>
    OnDemandBf16,

    /// <summary>Stack projections resident as MXFP4 packs, dequantised
    /// into a bounded f32 LRU on access. Deterministic, never
    /// bit-exact; ~4× the memory saving of BF16 and the exact input
    /// Blackwell FP4 tensor cores consume (CUDA phase).</summary>
    Mxfp4,
}

public static class WeightWorkingSetExtensions
{
    public static WeightWorkingSet FromEnv()
    {
        string? raw = Environment.GetEnvironmentVariable("AMQL_WEIGHTS");
        return raw?.Trim().ToLowerInvariant() switch
        {
            "mxfp4" or "fp4" or "quantized" => WeightWorkingSet.Mxfp4,
            "bf16" or "on-demand" or "ondemand" => WeightWorkingSet.OnDemandBf16,
            _ => WeightWorkingSet.ResidentF32,
        };
    }
}