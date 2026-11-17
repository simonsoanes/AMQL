using System.Diagnostics;
using Amql.Inference;

namespace Amql.Cli;

// ── lean working set: targets and memory reporting ──────────────────────────

internal static partial class Program
{
    /// <summary>The lean working set's RAM and GPU targets, for the header line.</summary>
    private static string LeanTargets()
    {
        long ram = ComputeBudget.LeanRamBytes;
        long cache = ComputeBudget.LeanCacheBytes(CudaShim.Enabled);
        string gpu = !CudaShim.Enabled
            ? "no GPU"
            : ComputeBudget.GpuMemoryBytes is { } vram
                ? $"GPU up to {Gib(vram)}"
                : "GPU up to its free memory";
        if (CudaShim.Enabled && ComputeBudget.GpuWeightFormat == GpuWeightFormat.Nvfp4)
        {
            gpu += " as NVFP4";
        }
        return $"RAM target {Gib(ram)}, weight cache {Gib(cache)}, {gpu}";
    }

    /// <summary>The process's private memory now and at its peak — what the
    /// lean working set's RAM target bounds. Mapped container pages are not
    /// in it: the OS can drop them whenever it needs the memory.</summary>
    private static string ProcessMemoryLine()
    {
        using var self = Process.GetCurrentProcess();
        self.Refresh();
        return $"private {Gib(self.PrivateMemorySize64)} (peak {Gib(self.PeakPagedMemorySize64)}), " +
               $"managed heap {Gib(GC.GetTotalMemory(false))}, {GC.CollectionCount(2)} full GCs ({GC.GetTotalPauseDuration().TotalSeconds:0.0}s paused)";
    }

    /// <summary>What a lean loader holds, has read, and keeps on the GPU.</summary>
    private static string LeanLine(LeanStats s) =>
        $"cached {Gib(s.DenseBytes)} dense + {Gib(s.ExpertBytes)} experts ({s.ExpertsCached} matrices) " +
        $"of {Gib(s.CacheBudget)}; GPU {Gib(s.DeviceBytes)} of {Gib(s.DeviceBudget)}" +
        (s.DeviceExperts > 0 ? $" ({s.DeviceExperts} expert matrices)" : string.Empty) + "; " +
        $"{s.Reads:N0} reads, {Gib(s.BytesRead)} read" +
        (s.HostReloads > 0 ? $"; {s.HostReloads} GPU weights read back to the host" : string.Empty) +
        (s.ScaledWidenings > 0 ? $"; {s.ScaledWidenings} quantised weights widened whole" : string.Empty);

    private static string Gib(long bytes) =>
        bytes >= 1L << 30
            ? $"{bytes / (double)(1L << 30):0.00} GiB"
            : $"{bytes / (double)(1L << 20):0.0} MiB";
}
