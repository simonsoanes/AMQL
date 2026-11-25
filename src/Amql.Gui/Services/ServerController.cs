using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Windows;

namespace Amql.Gui.Services;

public enum ServerState
{
    Stopped,
    Starting,
    Running,
    Stopping,
    Failed,

    /// <summary>A server this app did not start answers on the port: it can be watched, not stopped.</summary>
    External,
}

/// <summary>
/// Starts and stops amql-server for the app. The models and settings of a
/// <see cref="ServerProfile"/> become a --config file and flags; the server's
/// own output is read only to follow its start-up (which model is loading,
/// when it listens) and kept in a short buffer for diagnosing a failure. What
/// the server is doing once it runs comes from its /amql/status endpoint.
/// Properties change on the UI thread.
/// </summary>
public sealed class ServerController : INotifyPropertyChanged
{
    private const int MaxConsoleLines = 600;

    public static ServerController Instance { get; } = new();

    private readonly LinkedList<string> _console = new();
    private Process? _process;
    private bool _stopping;
    private ServerState _state = ServerState.Stopped;
    private string _detail = "Not running";
    private string? _loadingModel;
    private int _modelsLoaded, _modelsTotal;

    private ServerController()
    {
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    /// <summary>A new line of server output arrived (UI thread).</summary>
    public event Action<string>? ConsoleLine;

    public ServerState State
    {
        get => _state;
        private set
        {
            if (Set(ref _state, value))
            {
                OnPropertyChanged(nameof(IsRunning));
                OnPropertyChanged(nameof(IsBusy));
                OnPropertyChanged(nameof(CanStart));
                OnPropertyChanged(nameof(CanStop));
            }
        }
    }

    /// <summary>One line saying what the server is doing, for a status pill.</summary>
    public string Detail { get => _detail; private set => Set(ref _detail, value); }

    public string? LoadingModel { get => _loadingModel; private set => Set(ref _loadingModel, value); }

    public int ModelsLoaded { get => _modelsLoaded; private set => Set(ref _modelsLoaded, value); }

    public int ModelsTotal { get => _modelsTotal; private set => Set(ref _modelsTotal, value); }

    public bool IsRunning => State is ServerState.Running or ServerState.External;

    public bool IsBusy => State is ServerState.Starting or ServerState.Stopping;

    public bool CanStart => State is ServerState.Stopped or ServerState.Failed;

    public bool CanStop => State is ServerState.Running or ServerState.Starting;

    /// <summary>The URL clients reach the running server at.</summary>
    public string BaseUrl { get; private set; } = AppSettings.Current.Server.BaseUrl;

    public string ApiKey { get; private set; } = AppSettings.Current.Server.ApiKey;

    public int? ProcessId => _process is { HasExited: false } p ? p.Id : null;

    public DateTime? StartedAt { get; private set; }

    /// <summary>The command line of the last start, for the Advanced view.</summary>
    public string LastCommandLine { get; private set; } = string.Empty;

    public IReadOnlyList<string> ConsoleSnapshot()
    {
        lock (_console)
        {
            return _console.ToList();
        }
    }

    /// <summary>Looks for a server already answering on the profile's port (one started
    /// outside the app, or left running) and adopts it for watching.</summary>
    public async Task DetectExternalAsync(ServerProfile profile)
    {
        if (State != ServerState.Stopped)
        {
            return;
        }
        if (await ServerClient.IsHealthyAsync(profile.BaseUrl, profile.ApiKey))
        {
            BaseUrl = profile.BaseUrl;
            ApiKey = profile.ApiKey;
            State = ServerState.External;
            Detail = $"Running on port {profile.Port} (started outside AMQL Studio)";
        }
    }

    /// <summary>Forgets an external server (it keeps running).</summary>
    public void Detach()
    {
        if (State == ServerState.External)
        {
            State = ServerState.Stopped;
            Detail = "Not running";
        }
    }

    public async Task StartAsync(ServerProfile profile)
    {
        if (!CanStart)
        {
            return;
        }
        var models = profile.Models.Where(m => m.Enabled && Directory.Exists(m.Path)).ToList();
        if (models.Count == 0)
        {
            throw new InvalidOperationException("Choose at least one model to load.");
        }
        if (models.GroupBy(m => m.EffectiveId, StringComparer.Ordinal).FirstOrDefault(g => g.Count() > 1) is { } dup)
        {
            throw new InvalidOperationException($"Two models are both called '{dup.Key}'. Give one of them a different name.");
        }
        if (await ServerClient.IsHealthyAsync(profile.BaseUrl, profile.ApiKey))
        {
            throw new InvalidOperationException(
                $"Something is already answering on port {profile.Port}. Stop it, or choose another port under Advanced.");
        }

        var tool = ToolLocator.Require(ToolLocator.Server);
        string configPath = WriteConfig(models);
        var argv = new List<string>(tool.PrefixArguments) { "--config", configPath };
        argv.AddRange(Arguments(profile));

        var psi = new ProcessStartInfo
        {
            FileName = tool.FileName,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = tool.WorkingDirectory,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        foreach (var a in argv)
        {
            psi.ArgumentList.Add(a);
        }
        if (!string.IsNullOrWhiteSpace(profile.ApiKey))
        {
            psi.Environment["AMQL_API_KEY"] = profile.ApiKey.Trim();
        }
        LastCommandLine = Quote(tool.FileName) + " " + string.Join(' ', argv.Select(Quote));

        lock (_console)
        {
            _console.Clear();
        }
        _stopping = false;
        BaseUrl = profile.BaseUrl;
        ApiKey = profile.ApiKey;
        ModelsTotal = models.Count;
        ModelsLoaded = 0;
        LoadingModel = null;
        State = ServerState.Starting;
        Detail = "Starting…";
        Append("> " + LastCommandLine);

        var process = new Process { StartInfo = psi, EnableRaisingEvents = true };
        process.OutputDataReceived += (_, e) => OnOutput(e.Data);
        process.ErrorDataReceived += (_, e) => OnOutput(e.Data);
        process.Exited += (_, _) => Dispatch(() => OnExited(process));
        try
        {
            process.Start();
        }
        catch (Exception e)
        {
            State = ServerState.Failed;
            Detail = $"Could not start: {e.Message}";
            throw;
        }
        _process = process;
        StartedAt = DateTime.Now;
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        OnPropertyChanged(nameof(ProcessId));
    }

    public async Task StopAsync()
    {
        if (_process is not { } process || !CanStop)
        {
            return;
        }
        _stopping = true;
        State = ServerState.Stopping;
        Detail = "Stopping…";
        try
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
            }
            await process.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(10));
        }
        catch (Exception e) when (e is InvalidOperationException or TimeoutException or Win32Exception)
        {
            // already gone, or will be shortly; Exited finishes the state change
        }
        if (State == ServerState.Stopping)
        {
            OnExited(process);
        }
    }

    /// <summary>Kills the server synchronously, for app shutdown.</summary>
    public void StopNow()
    {
        try
        {
            if (_process is { HasExited: false } p)
            {
                _stopping = true;
                p.Kill(entireProcessTree: true);
            }
        }
        catch (Exception e) when (e is InvalidOperationException or Win32Exception)
        {
            // exiting anyway
        }
    }

    private void OnExited(Process process)
    {
        if (!ReferenceEquals(process, _process))
        {
            return;
        }
        int? code = null;
        try
        {
            code = process.ExitCode;
        }
        catch (InvalidOperationException)
        {
            // no exit code available
        }
        _process = null;
        StartedAt = null;
        LoadingModel = null;
        OnPropertyChanged(nameof(ProcessId));
        if (_stopping)
        {
            State = ServerState.Stopped;
            Detail = "Stopped";
            return;
        }
        State = ServerState.Failed;
        Detail = LastError() is { } error
            ? $"Stopped unexpectedly: {error}"
            : $"Stopped unexpectedly (exit code {code})";
    }

    /// <summary>The most useful line of output for a failure: the last error-looking one.</summary>
    private string? LastError()
    {
        var lines = ConsoleSnapshot();
        return lines.LastOrDefault(l =>
                   l.Contains("error", StringComparison.OrdinalIgnoreCase) ||
                   l.Contains("exception", StringComparison.OrdinalIgnoreCase) ||
                   l.Contains("needs", StringComparison.OrdinalIgnoreCase) ||
                   l.Contains("cannot", StringComparison.OrdinalIgnoreCase) ||
                   l.Contains("unknown", StringComparison.OrdinalIgnoreCase))
               ?? lines.LastOrDefault(l => !l.StartsWith('>'));
    }

    private void OnOutput(string? line)
    {
        if (line is null)
        {
            return;
        }
        Dispatch(() =>
        {
            Append(line);
            // Start-up milestones (see src/Amql.Server/Program.cs).
            if (line.StartsWith("loading ", StringComparison.Ordinal) && line.Contains(" from ", StringComparison.Ordinal))
            {
                LoadingModel = line["loading ".Length..line.IndexOf(" from ", StringComparison.Ordinal)];
                Detail = $"Loading {LoadingModel} ({ModelsLoaded + 1} of {ModelsTotal})…";
            }
            else if (LoadingModel is { } id && line.TrimStart().StartsWith(id + ":", StringComparison.Ordinal) && State == ServerState.Starting
                     && !line.Contains("note:", StringComparison.Ordinal))
            {
                ModelsLoaded++;
            }
            else if (State == ServerState.Starting && line.StartsWith("  ", StringComparison.Ordinal) && line.TrimStart() is var t
                     && t.IndexOf(": ", StringComparison.Ordinal) is > 0 and var colon
                     && !t.StartsWith("note", StringComparison.Ordinal) && !t[..colon].Contains(' '))
            {
                // Before a model loads, its 4-bit copy may be built ("  id: quantising …").
                Detail = $"Preparing {t[..colon]}: {t[(colon + 2)..]}";
            }
            else if (line.TrimStart().StartsWith("cuda:", StringComparison.Ordinal) && State == ServerState.Starting)
            {
                Detail = "Preparing the GPU…";
            }
            else if (line.StartsWith("listening on ", StringComparison.Ordinal))
            {
                LoadingModel = null;
                ModelsLoaded = ModelsTotal;
                State = ServerState.Running;
                Detail = $"Running on port {new Uri(BaseUrl).Port}";
            }
        });
    }

    private void Append(string line)
    {
        lock (_console)
        {
            _console.AddLast(line);
            while (_console.Count > MaxConsoleLines)
            {
                _console.RemoveFirst();
            }
        }
        ConsoleLine?.Invoke(line);
    }

    /// <summary>Server flags from the profile (everything but the models, which go in the config file).</summary>
    public static List<string> Arguments(ServerProfile p)
    {
        string Num(double v) => v.ToString("0.###", CultureInfo.InvariantCulture);
        var a = new List<string>
        {
            "--urls", $"http://{(p.ListenOnNetwork ? "0.0.0.0" : "127.0.0.1")}:{p.Port}",
        };
        if (p.Device == "cpu") a.Add("--cpu");
        if (p.MaxRamGb is { } ram) { a.Add("--max-ram"); a.Add(Num(ram)); }
        if (p.GpuMemoryGb is { } vram && p.Device != "cpu") { a.Add("--gpu-memory"); a.Add(Num(vram)); }
        if (p.GpuFormat is { Length: > 0 } f) { a.Add("--gpu-format"); a.Add(f); }
        if (p.KvFormat is { Length: > 0 } kv) { a.Add("--kv-format"); a.Add(kv); }
        if (p.Think) a.Add("--think");
        if (p.MaxTokens is { } mt) { a.Add("--max-tokens"); a.Add(mt.ToString(CultureInfo.InvariantCulture)); }
        if (p.Temperature is { } t) { a.Add("--temperature"); a.Add(Num(t)); }
        if (p.TopP is { } tp) { a.Add("--top-p"); a.Add(Num(tp)); }
        if (p.TopK is { } tk) { a.Add("--top-k"); a.Add(tk.ToString(CultureInfo.InvariantCulture)); }
        if (p.NoWarm) a.Add("--no-warm");
        if (p.NoBatch) a.Add("--no-batch");
        if (p.NoPrefixReuse) a.Add("--no-prefix-reuse");
        if (p.KvDisk && !p.NoPrefixReuse) a.Add("--kv-disk");
        if (p.BatchBf16) a.Add("--batch-bf16");
        if (p.NoLoopGuard) a.Add("--no-loop-guard");
        if (p.FreeToolCalls) a.Add("--free-tool-calls");
        if (p.NoMentionRule) a.Add("--no-mention-rule");
        if (p.NoAnnounceRule) a.Add("--no-announce-rule");
        if (p.AllowRepeatedCalls) a.Add("--allow-repeated-calls");
        if (p.KeepLineMarks) a.Add("--keep-line-marks");
        foreach (var pack in p.NGramPacks.Where(x => !string.IsNullOrWhiteSpace(x)))
        {
            a.Add("--ngram-pack");
            a.Add(pack);
        }
        a.AddRange(SplitArguments(p.ExtraArguments));
        return a;
    }

    /// <summary>Writes the models as an amql-server --config file and returns its path.</summary>
    private static string WriteConfig(IEnumerable<ServedModel> models)
    {
        var array = new JsonArray();
        foreach (var m in models)
        {
            var o = new JsonObject
            {
                ["id"] = m.EffectiveId,
                ["path"] = Path.GetFullPath(m.Path),
            };
            if (m.Adapters.Count > 0) o["adapters"] = new JsonArray(m.Adapters.Select(x => (JsonNode?)Path.GetFullPath(x)).ToArray());
            if (m.Format is { Length: > 0 } f) o["format"] = f;
            if (m.MaxContext.Trim() is { Length: > 0 } context) o["max_context"] = context;
            if (m.Think is { } think) o["think"] = think;
            if (m.MaxTokens is { } mt) o["max_tokens"] = mt;
            if (m.Temperature is { } t) o["temperature"] = t;
            if (m.TopP is { } tp) o["top_p"] = tp;
            if (m.TopK is { } tk) o["top_k"] = tk;
            array.Add(o);
        }
        Directory.CreateDirectory(AppSettings.Folder);
        string path = Path.Combine(AppSettings.Folder, "server-models.json");
        File.WriteAllText(path, new JsonObject { ["models"] = array }.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        return path;
    }

    public static IEnumerable<string> SplitArguments(string text)
    {
        var current = new StringBuilder();
        bool quoted = false, any = false;
        foreach (char c in text ?? string.Empty)
        {
            if (c == '"')
            {
                quoted = !quoted;
                any = true;
            }
            else if (char.IsWhiteSpace(c) && !quoted)
            {
                if (any)
                {
                    yield return current.ToString();
                    current.Clear();
                    any = false;
                }
            }
            else
            {
                current.Append(c);
                any = true;
            }
        }
        if (any)
        {
            yield return current.ToString();
        }
    }

    private static string Quote(string a) => a.Length == 0 || a.Any(char.IsWhiteSpace) ? $"\"{a}\"" : a;

    private static void Dispatch(Action action)
    {
        var dispatcher = Application.Current?.Dispatcher;
        if (dispatcher is null || dispatcher.CheckAccess())
        {
            action();
        }
        else
        {
            dispatcher.BeginInvoke(action);
        }
    }

    private bool Set<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return false;
        }
        field = value;
        OnPropertyChanged(name);
        return true;
    }

    private void OnPropertyChanged([CallerMemberName] string? name = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}
