using System.Net;
using System.Net.Sockets;
using System.Text.Json;

namespace Amql.Manager;

public enum ModelState
{
    Stopped,
    Starting,
    Ready,
    Stopping,
    Failed,
}

/// <summary>One configured model and the server process running it, if any.</summary>
public sealed class ModelInstance
{
    public required ModelEntry Entry { get; set; }

    /// <summary>Asked for by a desired-state client (Ariadne.Router through <c>PUT /api/v2/desired</c>)
    /// rather than listed in manager.json.</summary>
    public bool Desired { get; internal set; }

    /// <summary>The key a client may use for this instance's model alone.</summary>
    public string? ApiKey { get; internal set; }

    /// <summary>What a desired-state client asked for, so an unchanged request changes nothing.</summary>
    public string? SpecKey { get; internal set; }

    /// <summary>Set while the instance's model is still being fetched and encoded.</summary>
    public CliJob? Preparing { get; internal set; }

    public DateTimeOffset? StartedAt { get; internal set; }
    public ModelState State { get; internal set; } = ModelState.Stopped;
    public int Port { get; internal set; }
    public ChildProcess? Process { get; internal set; }
    public IReadOnlyList<string> Served { get; internal set; } = Array.Empty<string>();
    public string? Error { get; internal set; }
    public DateTimeOffset? ReadyAt { get; internal set; }
    public DateTimeOffset LastUsed { get; internal set; } = DateTimeOffset.UtcNow;
    internal int InFlight;
    internal Task<bool>? Starting;
    internal readonly SemaphoreSlim Gate = new(1, 1);

    public string? BaseUrl => State is ModelState.Ready or ModelState.Starting && Port > 0 ? $"http://127.0.0.1:{Port}" : null;
}

/// <summary>
/// The models the manager can serve, each in its own <c>amql-server</c>
/// process on a loopback port. Models start on request (or with the manager),
/// stop on request or after an idle time, and the ones running are written
/// down so a restart, an update's included, brings them back.
/// </summary>
public sealed class ModelPool : IAsyncDisposable
{
    private readonly ManagerConfig _config;
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, ModelInstance> _models;
    private readonly HttpClient _probe = new() { Timeout = TimeSpan.FromSeconds(5) };
    private readonly string _statePath;
    private readonly object _ports = new();
    private Timer? _idle;

    public ModelPool(ManagerConfig config)
    {
        _config = config;
        _models = new(config.Models.Select(m => KeyValuePair.Create(m.Id, new ModelInstance { Entry = m })), StringComparer.Ordinal);
        _statePath = Path.Combine(config.DataDirectory, "running-models.json");
    }

    public IReadOnlyCollection<ModelInstance> All => _models.Values.OrderBy(m => m.Entry.Id, StringComparer.Ordinal).ToList();

    public ModelInstance? Get(string id) => _models.GetValueOrDefault(id);

    /// <summary>Adds an instance a desired-state client asked for; false when the id is
    /// already a model of manager.json, which such a client cannot replace.</summary>
    internal bool TryAddDesired(ModelInstance instance) =>
        _models.TryAdd(instance.Entry.Id, instance) || _models.TryGetValue(instance.Entry.Id, out var existing) && existing.Desired;

    /// <summary>Stops and forgets an instance a desired-state client no longer asks for.</summary>
    internal async Task RemoveDesiredAsync(string id)
    {
        if (_models.TryGetValue(id, out var m) && m.Desired)
        {
            m.Preparing?.Cancel.Cancel();
            await StopAsync(id).ConfigureAwait(false);
            _models.TryRemove(id, out _);
            m.Process?.Dispose();
            Remember();
        }
    }

    /// <summary>The instance serving <paramref name="model"/>: one whose server lists it, or
    /// the configured model of that id.</summary>
    public ModelInstance? Resolve(string model) =>
        _models.Values.FirstOrDefault(m => m.Served.Contains(model, StringComparer.Ordinal)) ?? Get(model);

    /// <summary>Starts the autostart models and those running when the manager last stopped.</summary>
    public void StartRemembered()
    {
        var remembered = new HashSet<string>(StringComparer.Ordinal);
        try
        {
            if (File.Exists(_statePath))
            {
                remembered.UnionWith(JsonSerializer.Deserialize<string[]>(File.ReadAllText(_statePath)) ?? Array.Empty<string>());
            }
        }
        catch (JsonException)
        {
        }
        foreach (var m in _models.Values.Where(m => m.Entry.Autostart || remembered.Contains(m.Entry.Id)))
        {
            _ = StartAsync(m.Entry.Id, CancellationToken.None);
        }
        _idle = new Timer(_ => _ = StopIdleAsync(), null, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
    }

    /// <summary>Starts a model and waits until its server answers; true when it is ready.</summary>
    public async Task<bool> StartAsync(string id, CancellationToken cancel)
    {
        var m = Get(id) ?? throw new ManagerException($"no model '{id}' is configured");
        Task<bool> starting;
        await m.Gate.WaitAsync(cancel).ConfigureAwait(false);
        try
        {
            if (m.State == ModelState.Ready && m.Process is { Running: true })
            {
                return true;
            }
            if (m.State != ModelState.Starting || m.Starting is null)
            {
                m.Starting = LaunchAsync(m);
            }
            starting = m.Starting;
        }
        finally
        {
            m.Gate.Release();
        }
        return await starting.WaitAsync(cancel).ConfigureAwait(false);
    }

    private async Task<bool> LaunchAsync(ModelInstance m)
    {
        m.State = ModelState.Starting;
        m.StartedAt = DateTimeOffset.UtcNow;
        m.Error = null;
        m.Served = Array.Empty<string>();
        m.Port = m.Entry.Port > 0 ? m.Entry.Port : FreePort();
        string path = _config.Resolve(m.Entry.Path);
        var args = new List<string>();
        args.AddRange(path.EndsWith(".json", StringComparison.OrdinalIgnoreCase) || path.EndsWith(".jsonc", StringComparison.OrdinalIgnoreCase)
            ? new[] { "--config", path }
            : new[] { "--model", $"{m.Entry.ServedName ?? m.Entry.Id}={path}" });
        args.AddRange(new[] { "--urls", $"http://127.0.0.1:{m.Port}" });
        args.AddRange(_config.ServerArgs);
        args.AddRange(m.Entry.Args);
        string log = Path.Combine(_config.DataDirectory, "models", $"{Safe(m.Entry.Id)}.log");
        try
        {
            m.Process?.Dispose();
            m.Process = ChildProcess.Start(Tools.Find(_config.BinDirectory, "amql-server"), args,
                lowPriority: _config.Jobs.LowPriority, logPath: log);
        }
        catch (Exception e)
        {
            m.State = ModelState.Failed;
            m.Error = e.Message;
            return false;
        }

        var deadline = DateTimeOffset.UtcNow + TimeSpan.FromSeconds(m.Entry.StartTimeoutSeconds);
        while (DateTimeOffset.UtcNow < deadline)
        {
            if (!m.Process.Running)
            {
                m.State = ModelState.Failed;
                m.Error = $"amql-server exited with {m.Process.ExitCode}: {m.Process.Tail(8)}";
                Remember();
                return false;
            }
            try
            {
                using var response = await _probe.GetAsync($"http://127.0.0.1:{m.Port}/v1/models").ConfigureAwait(false);
                if (response.IsSuccessStatusCode)
                {
                    using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync().ConfigureAwait(false));
                    m.Served = doc.RootElement.TryGetProperty("data", out var data)
                        ? data.EnumerateArray().Select(d => d.GetProperty("id").GetString()!).ToList()
                        : new List<string> { m.Entry.Id };
                    m.State = ModelState.Ready;
                    m.ReadyAt = DateTimeOffset.UtcNow;
                    m.LastUsed = DateTimeOffset.UtcNow;
                    Remember();
                    return true;
                }
            }
            catch (Exception e) when (e is HttpRequestException or TaskCanceledException)
            {
                // Not listening yet.
            }
            await Task.Delay(500).ConfigureAwait(false);
        }
        m.Error = $"not ready after {m.Entry.StartTimeoutSeconds:0} s";
        await m.Process.StopAsync(TimeSpan.FromSeconds(10)).ConfigureAwait(false);
        m.State = ModelState.Failed;
        Remember();
        return false;
    }

    public async Task StopAsync(string id, bool remember = true)
    {
        var m = Get(id) ?? throw new ManagerException($"no model '{id}' is configured");
        await m.Gate.WaitAsync().ConfigureAwait(false);
        try
        {
            if (m.Process is { Running: true } p)
            {
                m.State = ModelState.Stopping;
                await p.StopAsync(TimeSpan.FromSeconds(15)).ConfigureAwait(false);
            }
            m.State = ModelState.Stopped;
            m.Served = Array.Empty<string>();
            m.Starting = null;
        }
        finally
        {
            m.Gate.Release();
        }
        if (remember)
        {
            Remember();
        }
    }

    /// <summary>Marks a request in flight on the model, so the idle stop leaves it alone.</summary>
    public IDisposable Use(ModelInstance m)
    {
        Interlocked.Increment(ref m.InFlight);
        m.LastUsed = DateTimeOffset.UtcNow;
        return new Release(() =>
        {
            m.LastUsed = DateTimeOffset.UtcNow;
            Interlocked.Decrement(ref m.InFlight);
        });
    }

    private async Task StopIdleAsync()
    {
        foreach (var m in _models.Values)
        {
            if (m.Entry.IdleMinutes > 0 && m.State == ModelState.Ready && Volatile.Read(ref m.InFlight) == 0 &&
                DateTimeOffset.UtcNow - m.LastUsed > TimeSpan.FromMinutes(m.Entry.IdleMinutes))
            {
                await StopAsync(m.Entry.Id).ConfigureAwait(false);
            }
        }
    }

    /// <summary>Stops every model, remembering which were running so the next start brings them back.</summary>
    public async Task StopAllForRestartAsync()
    {
        Remember();
        foreach (var m in _models.Values)
        {
            await StopAsync(m.Entry.Id, remember: false).ConfigureAwait(false);
        }
    }

    private void Remember()
    {
        var running = _models.Values.Where(m => !m.Desired && (m.State is ModelState.Ready or ModelState.Starting)).Select(m => m.Entry.Id).ToArray();
        File.WriteAllText(_statePath, JsonSerializer.Serialize(running));
    }

    private int FreePort()
    {
        lock (_ports)
        {
            var taken = _models.Values.Where(m => m.State != ModelState.Stopped).Select(m => m.Port).ToHashSet();
            for (int port = _config.FirstModelPort; port < _config.FirstModelPort + 1000; port++)
            {
                if (taken.Contains(port))
                {
                    continue;
                }
                try
                {
                    using var listener = new TcpListener(IPAddress.Loopback, port);
                    listener.Start();
                    return port;
                }
                catch (SocketException)
                {
                }
            }
            throw new ManagerException($"no free port from {_config.FirstModelPort}");
        }
    }

    internal static string Safe(string id) => string.Concat(id.Select(c => char.IsLetterOrDigit(c) || c is '-' or '_' or '.' ? c : '_'));

    public async ValueTask DisposeAsync()
    {
        _idle?.Dispose();
        foreach (var m in _models.Values)
        {
            if (m.Process is { } p)
            {
                await p.StopAsync(TimeSpan.FromSeconds(10)).ConfigureAwait(false);
                p.Dispose();
            }
        }
        _probe.Dispose();
    }

    private sealed class Release(Action action) : IDisposable
    {
        private int _done;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _done, 1) == 0)
            {
                action();
            }
        }
    }
}

internal static class Tools
{
    /// <summary>An AMQL executable in the bin folder: the apphost, or its .dll run by dotnet.</summary>
    public static string Find(string binDir, string name)
    {
        string exe = Path.Combine(binDir, OperatingSystem.IsWindows() ? name + ".exe" : name);
        if (File.Exists(exe))
        {
            return exe;
        }
        throw new ManagerException($"'{name}' is not in {binDir} (set 'bin_dir' in the manager's config)");
    }
}
