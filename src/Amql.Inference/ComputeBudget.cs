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
/// (<c>bf16</c> | <c>mxfp4</c> | <c>disk</c> | <c>lean</c>) selects the resident weight format, and
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

    /// <summary>How many f32 bytes of weights the disk working set keeps in
    /// memory: <c>AMQL_F32_CACHE_GB</c> when set, else half the memory
    /// budget. Weights past it are read from disk on every pass.</summary>
    public static long DiskCacheBytes { get; } = DiscoverDiskCache();

    /// <summary>The lean working set's process target: <c>AMQL_MAX_RAM_GB</c>
    /// when set, else 10 GiB (capped by the memory budget). Weights, caches,
    /// activations and the runtime's own state are meant to fit under it.</summary>
    public static long LeanRamBytes { get; set; } = DiscoverLeanRam();

    /// <summary>The lean working set's GPU target: <c>AMQL_GPU_MEMORY_GB</c>
    /// when set; otherwise null, and the loader takes the device's free memory
    /// less a margin. Zero keeps every weight on the host.</summary>
    public static long? GpuMemoryBytes { get; set; } = ParseGb("AMQL_GPU_MEMORY_GB", allowZero: true);

    /// <summary>What the lean working set holds back from its RAM target for
    /// everything that is not a cached weight: activations, KV and recurrent
    /// state, streamed matrices in flight, the runtime and the GC's slack.</summary>
    public static long LeanReserveBytes(long ramBytes) => Math.Max(Math.Min(1536L << 20, ramBytes / 2), ramBytes / 4);

    /// <summary>The form a weight takes on the GPU when the checkpoint did not
    /// choose one: FP16 (the default, exact for BF16 up to FP16's range) or
    /// NVFP4 (four bits, lossy, 3.5 times as many weights in the same memory).
    /// <c>AMQL_GPU_FORMAT</c> or <c>--gpu-format</c>. A weight stored as NVFP4
    /// always goes as NVFP4.</summary>
    public static GpuWeightFormat GpuWeightFormat { get; set; } =
        Environment.GetEnvironmentVariable("AMQL_GPU_FORMAT")?.Trim().ToLowerInvariant() switch
        {
            "nvfp4" or "fp4" => GpuWeightFormat.Nvfp4,
            "mxfp4" => GpuWeightFormat.Mxfp4,
            _ => GpuWeightFormat.Fp16,
        };

    /// <summary>Host memory the CUDA runtime, cuBLAS and the shim's staging
    /// buffers take once the GPU is in use; measured at 1.5 to 3 GiB.</summary>
    public const long CudaHostBytes = 2L << 30;

    /// <summary>The lean working set's weight cache: the RAM target less the
    /// reserve, and less <see cref="CudaHostBytes"/> when the GPU is in use,
    /// since the CUDA runtime's own host memory counts against the target too.</summary>
    public static long LeanCacheBytes(bool gpu)
    {
        long ram = LeanRamBytes;
        return Math.Max(0, ram - LeanReserveBytes(ram) - (gpu ? CudaHostBytes : 0));
    }

    private static long DiscoverLeanRam() =>
        Math.Min(ParseGb("AMQL_MAX_RAM_GB", allowZero: false) ?? (10L << 30), MemoryBudgetBytes);

    private static long? ParseGb(string variable, bool allowZero)
    {
        string? raw = Environment.GetEnvironmentVariable(variable);
        if (double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out double gb) &&
            (gb > 0 || (allowZero && gb == 0)))
        {
            return (long)(gb * 1024.0 * 1024.0 * 1024.0);
        }
        return null;
    }

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

    private static long DiscoverDiskCache()
    {
        string? raw = Environment.GetEnvironmentVariable("AMQL_F32_CACHE_GB");
        if (double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out double gb) && gb > 0)
        {
            return (long)(gb * 1024.0 * 1024.0 * 1024.0);
        }
        return Math.Max(1L << 30, MemoryBudgetBytes / 2);
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

    /// <summary>Nothing held for the session: every matrix is widened to
    /// f32 from the container's memory map when it is used, and kept only
    /// while it fits in <see cref="ComputeBudget.DiskCacheBytes"/>; the rest
    /// is read again on every pass. The resident path's exact f32 numerics
    /// on the CPU (nothing goes to the GPU), for a model whose f32 weights
    /// exceed memory: slower, bounded by the cache rather than the model.</summary>
    Disk,

    /// <summary>The bare minimum held: weights stay as their stored bits
    /// (BF16 is never widened), the embedding is read a row at a time, routed
    /// experts are read only when the router picks them, and what is cached
    /// is bounded by a RAM target (<see cref="ComputeBudget.LeanRamBytes"/>):
    /// dense weights first, the hottest experts in what is left. With CUDA,
    /// dense matrices also go to the GPU as FP16 up to a VRAM budget
    /// (<see cref="ComputeBudget.GpuMemoryBytes"/>). The resident path's
    /// numerics on the CPU, for a model far larger than memory.</summary>
    Lean,
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
            "disk" or "stream" or "streamed" => WeightWorkingSet.Disk,
            "lean" or "minimal" or "min" => WeightWorkingSet.Lean,
            _ => WeightWorkingSet.ResidentF32,
        };
    }
}

/// <summary>The form a weight takes on the GPU (see <see cref="ComputeBudget.GpuWeightFormat"/>).</summary>
public enum GpuWeightFormat
{
    Fp16,
    Nvfp4,

    /// <summary>OCP MXFP4: the GPU engine's projections at 4 bits with E8M0
    /// scales per 32 (the lean path's device copies stay FP16).</summary>
    Mxfp4,
}
