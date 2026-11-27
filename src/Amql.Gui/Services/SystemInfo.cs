using System.Diagnostics;
using System.Globalization;
using System.IO;

namespace Amql.Gui.Services;

/// <summary>One GPU as nvidia-smi reports it. <see cref="Unified"/> is set for a GPU
/// that shares the system's memory (the DGX Spark's GB10), whose memory figures
/// are the machine's: nvidia-smi reports them as not available.</summary>
public sealed record GpuSample(string Name, long TotalBytes, long UsedBytes, int UtilisationPercent, int? TemperatureC, bool Unified = false);

/// <summary>Machine facts for the memory sliders and the monitor: physical RAM,
/// and the GPU through nvidia-smi (so the GUI never loads CUDA itself).</summary>
public static class SystemInfo
{
    /// <summary>Physical memory the OS reports to .NET.</summary>
    public static long TotalRamBytes { get; } = Math.Max(GC.GetGCMemoryInfo().TotalAvailableMemoryBytes, 1L << 30);

    private static bool? _haveSmi;

    /// <summary>The first GPU's state, or null without an NVIDIA GPU / nvidia-smi.</summary>
    public static async Task<GpuSample?> SampleGpuAsync()
    {
        if (_haveSmi == false)
        {
            return null;
        }
        var lines = await RunSmiAsync("--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits");
        if (lines is null || lines.Count == 0)
        {
            _haveSmi = false;
            return null;
        }
        _haveSmi = true;
        var f = lines[0].Split(',').Select(s => s.Trim()).ToArray();
        if (f.Length < 5)
        {
            return null;
        }
        static long Mib(string s) => long.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) ? v << 20 : 0;
        static int? Int(string s) => int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) ? v : null;
        long total = Mib(f[1]), used = Mib(f[2]);
        if (total == 0 && ReadMemInfo() is { } mem)
        {
            // Unified memory (GB10, Jetson-class parts): "[N/A]" for the GPU's own memory.
            // The GPU can address the system's memory, so that is what it has.
            return new GpuSample(f[0], mem.Total, mem.Total - mem.Available, Int(f[3]) ?? 0, Int(f[4]), Unified: true);
        }
        return new GpuSample(f[0], total, used, Int(f[3]) ?? 0, Int(f[4]));
    }

    /// <summary>MemTotal and MemAvailable from /proc/meminfo (Linux), or null elsewhere.</summary>
    public static (long Total, long Available)? ReadMemInfo()
    {
        if (!OperatingSystem.IsLinux())
        {
            return null;
        }
        try
        {
            long total = 0, available = 0;
            foreach (var line in File.ReadLines("/proc/meminfo"))
            {
                static long Kib(string l) =>
                    long.TryParse(l.Split(' ', StringSplitOptions.RemoveEmptyEntries)[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) ? v << 10 : 0;
                if (line.StartsWith("MemTotal:", StringComparison.Ordinal)) total = Kib(line);
                else if (line.StartsWith("MemAvailable:", StringComparison.Ordinal)) available = Kib(line);
            }
            return total > 0 ? (total, available) : null;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or IndexOutOfRangeException)
        {
            return null;
        }
    }

    /// <summary>GPU memory one process holds, when the driver reports it per process (not under WDDM).</summary>
    public static async Task<long?> GpuMemoryOfProcessAsync(int pid)
    {
        if (_haveSmi == false)
        {
            return null;
        }
        var lines = await RunSmiAsync("--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits");
        foreach (var line in lines ?? new List<string>())
        {
            var f = line.Split(',').Select(s => s.Trim()).ToArray();
            if (f.Length == 2 && int.TryParse(f[0], out var p) && p == pid && long.TryParse(f[1], out var mib))
            {
                return mib << 20;
            }
        }
        return null;
    }

    private static async Task<List<string>?> RunSmiAsync(params string[] args)
    {
        try
        {
            var psi = new ProcessStartInfo("nvidia-smi")
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            foreach (var a in args)
            {
                psi.ArgumentList.Add(a);
            }
            using var p = Process.Start(psi);
            if (p is null)
            {
                return null;
            }
            string output = await p.StandardOutput.ReadToEndAsync();
            await p.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(5));
            return p.ExitCode == 0
                ? output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList()
                : null;
        }
        catch (Exception e) when (e is System.ComponentModel.Win32Exception or InvalidOperationException or TimeoutException or IOException)
        {
            return null;
        }
    }
}
