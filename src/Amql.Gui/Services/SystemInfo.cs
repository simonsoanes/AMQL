using System.Diagnostics;
using System.Globalization;
using System.IO;

namespace Amql.Gui.Services;

/// <summary>One GPU as nvidia-smi reports it.</summary>
public sealed record GpuSample(string Name, long TotalBytes, long UsedBytes, int UtilisationPercent, int? TemperatureC);

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
        return new GpuSample(f[0], Mib(f[1]), Mib(f[2]), Int(f[3]) ?? 0, Int(f[4]));
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
