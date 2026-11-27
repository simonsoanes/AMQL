using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Amql.Manager;

// The Ariadne ModelHost desired-state contract (apiVersion 2), so Ariadne.Router manages an
// AMQL install as a deployment location of kind AriadneModelHost. Wire names are camelCase,
// the key comes as x-api-key, and /healthz needs none, as the router's ModelHostClient expects.

public sealed class InstanceSpec
{
    public string Name { get; set; } = "";
    public string Engine { get; set; } = "";
    public string? Runtime { get; set; }
    public string Source { get; set; } = "huggingface";
    public string Model { get; set; } = "";
    public string? Revision { get; set; }
    public List<string> Files { get; set; } = new();
    public string? ServedModelName { get; set; }
    public int? Port { get; set; }
    public int? ContextLength { get; set; }
    public int? MemoryMb { get; set; }
    public double? GpuMemoryUtilization { get; set; }
    public List<string> ExtraArgs { get; set; } = new();
    public Dictionary<string, string> Environment { get; set; } = new();
    public string? Image { get; set; }
    public string? ApiKey { get; set; }
}

public sealed class DesiredRequest
{
    public List<InstanceSpec> Instances { get; set; } = new();
}

public static class ModelHostApi
{
    public const string Engine = "amql";

    internal static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    public static void Map(WebApplication app, ManagerConfig config, ModelPool pool, CliJobs jobs, Updater updater,
        Func<ReleaseInfo, Task<IResult>> applyAndRestart)
    {
        app.MapGet("/healthz", () => Results.Json(new { status = "ok", apiVersion = 2, version = Updater.RunningVersion() }, Json));

        var v2 = app.MapGroup("/api/v2");
        v2.MapGet("/capacity", () => Results.Json(Capacity(config, pool), Json));
        v2.MapGet("/instances", (HttpContext ctx) => Results.Json(
            pool.All.Where(m => m.Desired).Select(m => Status(m, PublicUrl(config, ctx))).ToList(), Json));
        v2.MapPut("/desired", async (HttpContext ctx) =>
        {
            // Read with the ModelHost's camelCase names, not the management API's snake_case.
            DesiredRequest? request;
            try
            {
                request = await JsonSerializer.DeserializeAsync<DesiredRequest>(ctx.Request.Body, Json, ctx.RequestAborted);
            }
            catch (JsonException e)
            {
                return Results.Json(new { message = $"the desired set is not valid JSON: {e.Message}" }, Json, statusCode: 400);
            }
            var rejected = await ReconcileAsync(request ?? new DesiredRequest(), config, pool, jobs);
            return Results.Json(new
            {
                instances = pool.All.Where(m => m.Desired).Select(m => Status(m, PublicUrl(config, ctx))).ToList(),
                rejected,
            }, Json);
        });
        v2.MapGet("/instances/{name}/logs", (string name, int? lines) =>
        {
            var m = pool.Get(name);
            var process = m?.Process ?? m?.Preparing?.Process;
            if (process is null)
            {
                return Results.Text(m is null ? $"no instance '{name}'" : "", "text/plain", statusCode: m is null ? 404 : 200);
            }
            return Results.Text(process.Tail(Math.Clamp(lines ?? 200, 1, ChildProcess.Keep)), "text/plain");
        });
        // The router's "install engine" is AMQL's update: the newest release, if there is one.
        v2.MapPost("/engines/{engine}/install", async (string engine, bool? force, CancellationToken cancel) =>
        {
            if (!string.Equals(engine, Engine, StringComparison.OrdinalIgnoreCase))
            {
                return Results.Json(new { message = $"this host runs only the '{Engine}' engine" }, Json, statusCode: 404);
            }
            var release = await updater.CheckAsync(cancel);
            if (release is null)
            {
                return Results.Json(new
                {
                    message = updater.Status.LastError is { } e ? $"update check failed: {e}" : $"AMQL {updater.Status.Current} is the newest release",
                }, Json, statusCode: updater.Status.LastError is null ? 200 : 502);
            }
            var result = await applyAndRestart(release);
            return Results.Json(new { message = $"installing AMQL {release.Version}; the manager restarts into it" }, Json,
                statusCode: result is IStatusCodeHttpResult { StatusCode: >= 400 } failed ? failed.StatusCode ?? 502 : 202);
        });
    }

    internal static string PublicUrl(ManagerConfig config, HttpContext ctx) =>
        (config.PublicUrl ?? $"{ctx.Request.Scheme}://{ctx.Request.Host}").TrimEnd('/');

    /// <summary>Brings the desired instances to what was asked: new ones started, changed ones
    /// restarted, missing ones stopped. Models of manager.json are left alone. Returns the
    /// instances that cannot be run, with why.</summary>
    internal static async Task<Dictionary<string, string>> ReconcileAsync(DesiredRequest request, ManagerConfig config, ModelPool pool, CliJobs jobs)
    {
        var rejected = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var wanted = new HashSet<string>(StringComparer.Ordinal);
        foreach (var spec in request.Instances)
        {
            string? reason = Validate(spec, pool);
            if (reason is not null)
            {
                rejected[spec.Name] = reason;
                continue;
            }
            wanted.Add(spec.Name);
            string key = JsonSerializer.Serialize(spec, Json);
            if (pool.Get(spec.Name) is { Desired: true } existing && existing.SpecKey == key)
            {
                if (existing.State is ModelState.Stopped or ModelState.Failed && existing.Preparing is null)
                {
                    _ = pool.StartAsync(spec.Name, CancellationToken.None);
                }
                continue;
            }
            if (pool.Get(spec.Name) is { Desired: true })
            {
                await pool.RemoveDesiredAsync(spec.Name);
            }
            var (path, encode) = Locate(spec, config, pool);
            var args = new List<string>();
            if (spec.ContextLength is > 0 and var context)
            {
                args.AddRange(new[] { "--max-context", context.ToString(System.Globalization.CultureInfo.InvariantCulture) });
            }
            args.AddRange(spec.ExtraArgs);
            var instance = new ModelInstance
            {
                Entry = new ModelEntry
                {
                    Id = spec.Name,
                    Path = path,
                    ServedName = string.IsNullOrWhiteSpace(spec.ServedModelName) ? spec.Name : spec.ServedModelName,
                    Args = args,
                    StartOnDemand = true,
                },
                Desired = true,
                ApiKey = string.IsNullOrEmpty(spec.ApiKey) ? null : spec.ApiKey,
                SpecKey = key,
            };
            if (!pool.TryAddDesired(instance))
            {
                rejected[spec.Name] = $"'{spec.Name}' is a model of manager.json";
                continue;
            }
            if (encode is not null)
            {
                instance.Preparing = jobs.Submit(encode, null);
                _ = StartWhenPreparedAsync(instance, pool);
            }
            else
            {
                _ = pool.StartAsync(spec.Name, CancellationToken.None);
            }
        }
        foreach (var m in pool.All.Where(m => m.Desired && !wanted.Contains(m.Entry.Id)).ToList())
        {
            await pool.RemoveDesiredAsync(m.Entry.Id);
        }
        return rejected;
    }

    private static string? Validate(InstanceSpec spec, ModelPool pool)
    {
        if (string.IsNullOrWhiteSpace(spec.Name) || spec.Name.Any(c => !(char.IsLetterOrDigit(c) || c is '-' or '_' or '.')))
        {
            return "an instance needs a name of letters, digits, '-', '_' and '.'";
        }
        if (!string.Equals(spec.Engine, Engine, StringComparison.OrdinalIgnoreCase))
        {
            return $"this host runs the '{Engine}' engine, not '{spec.Engine}'";
        }
        if (string.IsNullOrWhiteSpace(spec.Model))
        {
            return "no model given";
        }
        if (spec.Source is not ("huggingface" or "amql" or "local"))
        {
            return $"source '{spec.Source}' is not served: give a Hugging Face repo, an AMQL container folder or a model of manager.json";
        }
        if (pool.Get(spec.Name) is { Desired: false })
        {
            return $"'{spec.Name}' is a model of manager.json";
        }
        return null;
    }

    /// <summary>Where the instance's container is: a model of manager.json, a container folder,
    /// or a Hugging Face repo encoded into the containers folder (the command doing it when it is
    /// not there yet).</summary>
    private static (string Path, List<string>? Encode) Locate(InstanceSpec spec, ManagerConfig config, ModelPool pool)
    {
        if (pool.Get(spec.Model) is { Desired: false } configured)
        {
            return (config.Resolve(configured.Entry.Path), null);
        }
        string containers = config.Resolve(config.ContainersDir);
        foreach (string candidate in new[] { spec.Model, Path.Combine(containers, spec.Model) })
        {
            if (Path.IsPathFullyQualified(candidate) || candidate.StartsWith(containers, StringComparison.Ordinal))
            {
                if (File.Exists(Path.Combine(candidate, "index.json")))
                {
                    return (candidate, null);
                }
            }
        }
        string folder = ModelPool.Safe(spec.Model.Replace('/', '-').Replace("--", "-")) +
                        (string.IsNullOrWhiteSpace(spec.Revision) || spec.Revision == "main" ? "" : "@" + ModelPool.Safe(spec.Revision));
        string target = Path.Combine(containers, folder);
        if (File.Exists(Path.Combine(target, "index.json")))
        {
            return (target, null);
        }
        var encode = new List<string> { "encode", spec.Model, "--out", target };
        if (!string.IsNullOrWhiteSpace(spec.Revision))
        {
            encode.AddRange(new[] { "--revision", spec.Revision });
        }
        return (target, encode);
    }

    private static async Task StartWhenPreparedAsync(ModelInstance instance, ModelPool pool)
    {
        var job = instance.Preparing!;
        while (job.State is JobState.Queued or JobState.Running)
        {
            await Task.Delay(1000).ConfigureAwait(false);
        }
        if (job.State != JobState.Succeeded)
        {
            instance.State = ModelState.Failed;
            instance.Error = $"encoding {instance.Entry.Path} failed: {job.Error ?? job.Process?.Tail(6)}";
            return;
        }
        instance.Preparing = null;
        if (pool.Get(instance.Entry.Id) == instance)
        {
            await pool.StartAsync(instance.Entry.Id, CancellationToken.None).ConfigureAwait(false);
        }
    }

    internal static object Status(ModelInstance m, string publicUrl) => new
    {
        name = m.Entry.Id,
        engine = Engine,
        runtime = "native",
        model = m.Entry.Path,
        state = m.Preparing is { State: JobState.Queued or JobState.Running } ? "downloading"
            : m.State switch
            {
                ModelState.Ready => "ready",
                ModelState.Failed => "failed",
                ModelState.Stopped => m.Error is null ? "pending" : "failed",
                _ => "pending",
            },
        detail = m.Preparing is { State: JobState.Queued or JobState.Running } p ? $"encoding: {p.Process?.Tail(1)}"
            : m.State == ModelState.Starting ? "loading" : null,
        error = m.Error,
        port = m.Port,
        // Each model's server listens on loopback: clients reach it through this manager,
        // which routes by the model name and takes this instance's own key for it.
        url = publicUrl,
        servedModelName = m.Entry.ServedName ?? m.Entry.Id,
        apiKeyEnforced = m.ApiKey is not null,
        downloadedBytes = 0L,
        totalBytes = (long?)null,
        startedAtUtc = m.StartedAt?.UtcDateTime,
        readyAtUtc = m.ReadyAt?.UtcDateTime,
    };

    internal static object Capacity(ManagerConfig config, ModelPool pool)
    {
        var gpus = Gpus();
        var (total, available) = SystemMemoryMb();
        long acceleratorTotal = gpus.Sum(g => (long)(g.MemoryTotalMb ?? 0));
        long acceleratorFree = gpus.Sum(g => (long)(g.MemoryFreeMb ?? 0));
        long diskFree = 0;
        try
        {
            string containers = config.Resolve(config.ContainersDir);
            diskFree = new DriveInfo(Path.GetPathRoot(Path.GetFullPath(containers))!).AvailableFreeSpace / (1024 * 1024);
        }
        catch (Exception e) when (e is IOException or ArgumentException or UnauthorizedAccessException)
        {
        }
        string? server = null;
        try
        {
            server = Tools.Find(config.BinDirectory, "amql-server");
        }
        catch (ManagerException)
        {
        }
        return new
        {
            hostName = Environment.MachineName,
            os = RuntimeInformation.OSDescription,
            architecture = RuntimeInformation.OSArchitecture.ToString().ToLowerInvariant(),
            cpuCount = Environment.ProcessorCount,
            gpus,
            unifiedMemory = gpus.Count == 0 ? false : gpus.All(g => g.Name.Contains("GB10", StringComparison.OrdinalIgnoreCase)),
            systemMemoryTotalMb = total,
            systemMemoryAvailableMb = available,
            acceleratorMemoryTotalMb = acceleratorTotal,
            freeMemoryMb = gpus.Count > 0 ? acceleratorFree : available,
            plannableMemoryMb = gpus.Count > 0 ? acceleratorFree : available,
            diskFreeMb = diskFree,
            dockerAvailable = false,
            dockerError = (string?)"AMQL runs natively",
            engines = new[]
            {
                new
                {
                    name = Engine,
                    displayName = "AMQL",
                    runtime = "native",
                    image = (string?)null,
                    executable = server,
                    available = server is not null,
                    reason = server is null ? $"amql-server is not in {config.BinDirectory}" : null,
                    supportsApiKey = true,
                    installable = true,
                    installsOnDemand = false,
                    installedVersion = Updater.RunningVersion(),
                    installState = server is null ? "missing" : "installed",
                    installDetail = (string?)null,
                    installError = (string?)null,
                },
            },
            managedInstanceCount = pool.All.Count(m => m.Desired),
            measuredAtUtc = DateTime.UtcNow,
        };
    }

    public sealed record Gpu(int Index, string Name, int? MemoryTotalMb, int? MemoryUsedMb, int? MemoryFreeMb, int? UtilizationPercent);

    /// <summary>The GPUs nvidia-smi lists; none when it is absent.</summary>
    internal static List<Gpu> Gpus()
    {
        try
        {
            var start = new ProcessStartInfo("nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using var p = Process.Start(start)!;
            string output = p.StandardOutput.ReadToEnd();
            if (!p.WaitForExit(5000) || p.ExitCode != 0)
            {
                return new List<Gpu>();
            }
            return ParseGpus(output);
        }
        catch (Exception e) when (e is System.ComponentModel.Win32Exception or InvalidOperationException)
        {
            return new List<Gpu>();
        }
    }

    internal static List<Gpu> ParseGpus(string csv)
    {
        static int? Int(string s) => int.TryParse(s.Trim(), out int v) ? v : null;
        var gpus = new List<Gpu>();
        foreach (string line in csv.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var f = line.Split(',');
            if (f.Length >= 6 && Int(f[0]) is { } index)
            {
                gpus.Add(new Gpu(index, f[1].Trim(), Int(f[2]), Int(f[3]), Int(f[4]), Int(f[5])));
            }
        }
        return gpus;
    }

    /// <summary>Total and available system memory, in MB.</summary>
    internal static (long Total, long Available) SystemMemoryMb()
    {
        if (OperatingSystem.IsLinux() && File.Exists("/proc/meminfo"))
        {
            long total = 0, available = 0;
            foreach (string line in File.ReadLines("/proc/meminfo"))
            {
                var parts = line.Split(':', 2);
                if (parts.Length == 2 && long.TryParse(parts[1].Trim().Split(' ')[0], out long kb))
                {
                    if (parts[0] == "MemTotal")
                    {
                        total = kb / 1024;
                    }
                    else if (parts[0] == "MemAvailable")
                    {
                        available = kb / 1024;
                    }
                }
            }
            return (total, available);
        }
        if (OperatingSystem.IsWindows())
        {
            var status = new MemoryStatus { Length = (uint)Marshal.SizeOf<MemoryStatus>() };
            if (GlobalMemoryStatusEx(ref status))
            {
                return ((long)(status.TotalPhys / (1024 * 1024)), (long)(status.AvailPhys / (1024 * 1024)));
            }
        }
        long gc = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / (1024 * 1024);
        return (gc, gc);
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryStatus
    {
        public uint Length;
        public uint MemoryLoad;
        public ulong TotalPhys;
        public ulong AvailPhys;
        public ulong TotalPageFile;
        public ulong AvailPageFile;
        public ulong TotalVirtual;
        public ulong AvailVirtual;
        public ulong AvailExtendedVirtual;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GlobalMemoryStatusEx(ref MemoryStatus status);
}
