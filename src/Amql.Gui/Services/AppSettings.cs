using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Amql.Gui.Services;

/// <summary>
/// Settings that belong to this machine rather than to a project: where the
/// models live, how the inference server is set up, and the playground's
/// choices. Stored as JSON under %LOCALAPPDATA%\AMQL Studio\settings.json and
/// saved whenever something changes, so the next start picks up where the
/// last one left off.
/// </summary>
public sealed class AppSettings
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    public static string Folder { get; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "AMQL Studio");

    private static string FilePath => Path.Combine(Folder, "settings.json");

    /// <summary>The settings for this run of the app.</summary>
    public static AppSettings Current { get; } = Load();

    /// <summary>Folders scanned for containers (each scanned two levels deep).</summary>
    public List<string> ModelFolders { get; set; } = new();

    /// <summary>Containers added one at a time, outside the scanned folders.</summary>
    public List<string> ExtraModels { get; set; } = new();

    public ServerProfile Server { get; set; } = new();

    public PlaygroundSettings Playground { get; set; } = new();

    /// <summary>Page shown when the app last closed.</summary>
    public string? LastPage { get; set; }

    /// <summary>Main window bounds (device-independent units) and whether it was maximised.</summary>
    public double[]? WindowBounds { get; set; }
    public bool WindowMaximized { get; set; }

    /// <summary>Project opened when the app last closed (reopened on start).</summary>
    public string? LastProject { get; set; }

    private static AppSettings Load()
    {
        try
        {
            if (File.Exists(FilePath))
            {
                var loaded = JsonSerializer.Deserialize<AppSettings>(File.ReadAllText(FilePath), JsonOptions);
                if (loaded is not null)
                {
                    return loaded;
                }
            }
        }
        catch (Exception e) when (e is IOException or JsonException or UnauthorizedAccessException)
        {
            // a damaged file starts over from the defaults
        }
        var fresh = new AppSettings();
        if (ToolLocator.RepoRoot is { } repo)
        {
            fresh.ModelFolders.Add(repo);
        }
        return fresh;
    }

    public void Save()
    {
        try
        {
            Directory.CreateDirectory(Folder);
            var tmp = FilePath + ".tmp";
            File.WriteAllText(tmp, JsonSerializer.Serialize(this, JsonOptions));
            File.Move(tmp, FilePath, overwrite: true);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            // settings are a convenience; failing to save them is not worth an error dialog
        }
    }
}

/// <summary>How amql-server is started: what it loads and the limits it runs under.</summary>
public sealed class ServerProfile
{
    /// <summary>The models to load, in order.</summary>
    public List<ServedModel> Models { get; set; } = new();

    public int Port { get; set; } = 8000;

    /// <summary>Listen on every interface instead of only this machine.</summary>
    public bool ListenOnNetwork { get; set; }

    public string ApiKey { get; set; } = string.Empty;

    /// <summary>--max-ram GiB; null lets the server use what it needs.</summary>
    public double? MaxRamGb { get; set; }

    /// <summary>--gpu-memory GiB; null lets the server take the GPU's free memory.</summary>
    public double? GpuMemoryGb { get; set; }

    /// <summary>"gpu" (auto) or "cpu".</summary>
    public string Device { get; set; } = "gpu";

    /// <summary>--gpu-format for every model that does not choose: "", "nvfp4", "mxfp4", "fp16".</summary>
    public string GpuFormat { get; set; } = string.Empty;

    // Defaults for chat models.
    public bool Think { get; set; }
    public int? MaxTokens { get; set; }
    public double? Temperature { get; set; }
    public double? TopP { get; set; }
    public int? TopK { get; set; }

    // Serving behaviour (advanced): each is the server's default when false.
    public bool NoWarm { get; set; }
    public bool NoBatch { get; set; }
    public bool NoPrefixReuse { get; set; }
    public bool KvDisk { get; set; }
    public bool BatchBf16 { get; set; }
    public bool NoLoopGuard { get; set; }
    public bool FreeToolCalls { get; set; }
    public bool NoMentionRule { get; set; }
    public bool NoAnnounceRule { get; set; }
    public bool AllowRepeatedCalls { get; set; }
    public bool KeepLineMarks { get; set; }

    /// <summary>N-gram pack files loaded with --ngram-pack.</summary>
    public List<string> NGramPacks { get; set; } = new();

    /// <summary>Anything else to pass, as typed (split on spaces, quotes respected).</summary>
    public string ExtraArguments { get; set; } = string.Empty;

    /// <summary>Stop the server when AMQL Studio closes.</summary>
    public bool StopOnExit { get; set; } = true;

    [JsonIgnore]
    public string BaseUrl => $"http://127.0.0.1:{Port}";
}

/// <summary>One model the server loads, with what it may override.</summary>
public sealed class ServedModel
{
    public string Path { get; set; } = string.Empty;

    /// <summary>The id requests name it by; empty uses the folder name.</summary>
    public string Id { get; set; } = string.Empty;

    public bool Enabled { get; set; } = true;

    /// <summary>"", "nvfp4", "mxfp4" or "fp16": empty follows the server's default.</summary>
    public string Format { get; set; } = string.Empty;

    /// <summary>Per-model think default (null: the server's).</summary>
    public bool? Think { get; set; }

    public int? MaxTokens { get; set; }
    public double? Temperature { get; set; }
    public double? TopP { get; set; }
    public int? TopK { get; set; }

    /// <summary>Entail adapter folders run beside a chat model.</summary>
    public List<string> Adapters { get; set; } = new();

    [JsonIgnore]
    public string EffectiveId => string.IsNullOrWhiteSpace(Id)
        ? System.IO.Path.GetFileName(Path.TrimEnd('/', '\\'))
        : Id.Trim();
}

public sealed class PlaygroundSettings
{
    public string? Model { get; set; }
    public string SystemPrompt { get; set; } = string.Empty;
    public bool Think { get; set; }
    public double? Temperature { get; set; }
    public int? MaxTokens { get; set; }
}
