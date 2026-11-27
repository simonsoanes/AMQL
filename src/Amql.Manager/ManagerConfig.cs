using System.Text.Json;
using System.Text.Json.Serialization;

namespace Amql.Manager;

/// <summary>
/// <c>manager.json</c>: what the manager serves and how. Comments and trailing
/// commas are allowed; an unknown setting is refused, so a typo is not silently
/// a default.
/// </summary>
public sealed class ManagerConfig
{
    /// <summary>Where the management API and the model endpoint listen.</summary>
    public string Urls { get; set; } = "http://127.0.0.1:8100";

    /// <summary>The bearer key every request must carry (or <c>AMQL_MANAGER_API_KEY</c>).
    /// Required when <see cref="Urls"/> reaches beyond this machine.</summary>
    public string? ApiKey { get; set; }

    /// <summary>The folder holding amql-cli and amql-server; by default the manager's own.</summary>
    public string? BinDir { get; set; }

    /// <summary>Where job logs and the record of running models are kept.</summary>
    public string? DataDir { get; set; }

    /// <summary>The first port given to a model's server; each takes the next free one.</summary>
    public int FirstModelPort { get; set; } = 8201;

    /// <summary>Arguments every model's server gets (e.g. <c>["--kv-format", "nvfp4"]</c>).</summary>
    public List<string> ServerArgs { get; set; } = new();

    public List<ModelEntry> Models { get; set; } = new();

    /// <summary>The URL clients reach this manager at, reported as each model's address
    /// (e.g. <c>http://spark.local:8100</c>); by default the address a request came in on.</summary>
    public string? PublicUrl { get; set; }

    /// <summary>Where a Hugging Face model asked for by a desired-state client is encoded to.</summary>
    public string ContainersDir { get; set; } = "containers";

    public JobSettings Jobs { get; set; } = new();

    public UpdateSettings Update { get; set; } = new();

    [JsonIgnore]
    public string? SourcePath { get; private set; }

    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        WriteIndented = true,
    };

    public static ManagerConfig Load(string? path)
    {
        ManagerConfig config;
        if (path is null || !File.Exists(path))
        {
            config = new ManagerConfig();
        }
        else
        {
            try
            {
                config = JsonSerializer.Deserialize<ManagerConfig>(File.ReadAllText(path), Options) ?? new ManagerConfig();
            }
            catch (JsonException e)
            {
                throw new ManagerException($"{path}: {e.Message}");
            }
        }
        config.SourcePath = path is null ? null : Path.GetFullPath(path);
        config.ApiKey ??= Environment.GetEnvironmentVariable("AMQL_MANAGER_API_KEY");
        config.Validate();
        return config;
    }

    public void Save(string path) => File.WriteAllText(path, JsonSerializer.Serialize(this, Options));

    private void Validate()
    {
        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var m in Models)
        {
            if (string.IsNullOrWhiteSpace(m.Id) || string.IsNullOrWhiteSpace(m.Path))
            {
                throw new ManagerException("every model needs an 'id' and a 'path'");
            }
            if (!ids.Add(m.Id))
            {
                throw new ManagerException($"model '{m.Id}' is listed twice");
            }
        }
        if (string.IsNullOrEmpty(ApiKey) && !LoopbackOnly(Urls))
        {
            throw new ManagerException(
                $"'{Urls}' listens beyond this machine: set 'api_key' (or AMQL_MANAGER_API_KEY), since the API runs commands and installs updates");
        }
        if (Update.IntervalHours <= 0)
        {
            throw new ManagerException("update.interval_hours must be positive");
        }
    }

    internal static bool LoopbackOnly(string urls) =>
        urls.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).All(u =>
            Uri.TryCreate(u.Replace("://+", "://0.0.0.0").Replace("://*", "://0.0.0.0"), UriKind.Absolute, out var uri) &&
            (uri.IsLoopback || uri.Host == "localhost"));

    /// <summary>A path in the config is relative to the config file.</summary>
    public string Resolve(string path) =>
        Path.IsPathRooted(path) || SourcePath is null ? path : Path.GetFullPath(Path.Combine(Path.GetDirectoryName(SourcePath)!, path));

    public string BinDirectory => BinDir is { } dir ? Resolve(dir) : AppContext.BaseDirectory;

    public string DataDirectory
    {
        get
        {
            string dir = DataDir is { } d ? Resolve(d)
                : Environment.GetEnvironmentVariable("AMQL_ROOT") is { } root ? Path.Combine(root, "data")
                : Path.Combine(AppContext.BaseDirectory, "manager-data");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }
}

/// <summary>A model the manager can start: its container, and how.</summary>
public sealed class ModelEntry
{
    public string Id { get; set; } = "";

    /// <summary>The container folder (relative to the config file), or an amql-server
    /// <c>--config</c> file when it ends in <c>.json</c>/<c>.jsonc</c>.</summary>
    public string Path { get; set; } = "";

    /// <summary>The name the model answers to in requests; by default its id.</summary>
    public string? ServedName { get; set; }

    /// <summary>Arguments added after the shared server arguments.</summary>
    public List<string> Args { get; set; } = new();

    /// <summary>Started with the manager.</summary>
    public bool Autostart { get; set; }

    /// <summary>A request for this model starts it if it is stopped.</summary>
    public bool StartOnDemand { get; set; } = true;

    /// <summary>Stopped after this many minutes without a request; 0 keeps it running.</summary>
    public double IdleMinutes { get; set; }

    /// <summary>How long a start may take before it counts as failed.</summary>
    public double StartTimeoutSeconds { get; set; } = 600;

    /// <summary>A fixed port; 0 takes the next free one from <c>first_model_port</c>.</summary>
    public int Port { get; set; }
}

public sealed class JobSettings
{
    /// <summary>CLI jobs running at once; the rest wait their turn.</summary>
    public int MaxConcurrent { get; set; } = 1;

    /// <summary>Jobs (and models' servers) run below normal priority, so the machine stays usable.</summary>
    public bool LowPriority { get; set; } = true;

    /// <summary>How many finished jobs are kept for reading back.</summary>
    public int KeepFinished { get; set; } = 200;
}

public sealed class UpdateSettings
{
    public bool Enabled { get; set; } = true;

    /// <summary>The GitHub repository whose releases carry the builds.</summary>
    public string Repository { get; set; } = "simonsoanes/AMQL";

    /// <summary>Prereleases are taken too.</summary>
    public bool Prerelease { get; set; }

    public double IntervalHours { get; set; } = 24;

    /// <summary>A newer release found by the daily check is installed then, when no job is running;
    /// otherwise it waits for <c>POST /v1/manage/update/apply</c>.</summary>
    public bool AutoApply { get; set; } = true;

    /// <summary>The environment variable holding a GitHub token, for a private repository
    /// or a higher rate limit.</summary>
    public string TokenVariable { get; set; } = "AMQL_UPDATE_TOKEN";

    /// <summary>The API base, for GitHub Enterprise or a test.</summary>
    public string ApiBase { get; set; } = "https://api.github.com";
}

public sealed class ManagerException(string message) : Exception(message);
