using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Amql.Gui.Model;

/// <summary>
/// The .amqlproj file format (JSON). One file stores everything:
///   • the CLI's global options (device, memory limits, weights env),
///   • project-wide defaults (container / tokenizer / patch / component),
///   • the last-entered parameter values for every command,
///   • the run history — status, progress, timing, exit code and an output
///     tail for each run, so reopening a project shows where work stands.
///
/// The format is additive: unknown JSON properties are ignored on load, and
/// <see cref="FormatVersion"/> guards future structural changes.
/// </summary>
public sealed class ProjectModel
{
    public const string FileExtension = ".amqlproj";
    public const string FileFilter = "AMQL Project (*.amqlproj)|*.amqlproj|All files (*.*)|*.*";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    public int FormatVersion { get; set; } = 1;
    public string Name { get; set; } = "Untitled AMQL Project";
    public DateTime? SavedUtc { get; set; }

    public CliSettings Cli { get; set; } = new();
    public ProjectDefaults Defaults { get; set; } = new();

    /// <summary>Per-command parameter values, keyed by command name then parameter key.</summary>
    public Dictionary<string, CommandState> Commands { get; set; } = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Run history, oldest first. Status and progress live here.</summary>
    public List<RunRecord> Runs { get; set; } = new();

    /// <summary>Command selected when the project was last saved.</summary>
    public string? LastCommand { get; set; }

    public CommandState Command(string name)
    {
        if (!Commands.TryGetValue(name, out var state))
        {
            state = new CommandState();
            Commands[name] = state;
        }
        return state;
    }

    public static ProjectModel Load(string path)
    {
        var json = File.ReadAllText(path);
        var model = JsonSerializer.Deserialize<ProjectModel>(json, JsonOptions)
            ?? throw new InvalidDataException($"'{path}' does not contain a project object");
        if (model.FormatVersion > 1)
        {
            throw new InvalidDataException(
                $"project file format version {model.FormatVersion} is newer than this build supports (1)");
        }
        return model;
    }

    public void Save(string path)
    {
        SavedUtc = DateTime.UtcNow;
        var json = JsonSerializer.Serialize(this, JsonOptions);
        // Write via a temp file + replace so a crash mid-write cannot corrupt the project.
        var tmp = path + ".tmp";
        File.WriteAllText(tmp, json);
        if (File.Exists(path))
        {
            File.Copy(tmp, path, overwrite: true);
            File.Delete(tmp);
        }
        else
        {
            File.Move(tmp, path);
        }
    }
}

/// <summary>The CLI's global options for a project's runs. Where amql-cli
/// lives is not a setting: <see cref="Services.ToolLocator"/> finds it.</summary>
public sealed class CliSettings
{
    /// <summary>"auto" (no flag), "cpu" (--cpu) or "gpu" (--gpu).</summary>
    public string Device { get; set; } = "auto";

    /// <summary>Value for the AMQL_WEIGHTS environment variable; empty means "do not set".</summary>
    public string WeightsEnv { get; set; } = string.Empty;

    /// <summary>--max-ram GiB (the lean working set's process target); null leaves it to the CLI.</summary>
    public double? MaxRamGb { get; set; }

    /// <summary>--gpu-memory GiB the weights may take on the GPU; null leaves it to the CLI.</summary>
    public double? GpuMemoryGb { get; set; }

    /// <summary>--gpu-format: "" (as stored), "nvfp4", "mxfp4" or "fp16".</summary>
    public string GpuFormat { get; set; } = string.Empty;

    /// <summary>--fp4-prefill: "" (default), "on" or "off".</summary>
    public string Fp4Prefill { get; set; } = string.Empty;

    /// <summary>--verbose.</summary>
    public bool Verbose { get; set; }

    /// <summary>The global flags these settings add before the command name.</summary>
    public IEnumerable<string> GlobalArguments()
    {
        if (Device == "cpu") yield return "--cpu";
        else if (Device == "gpu") yield return "--gpu";
        if (MaxRamGb is { } ram) { yield return "--max-ram"; yield return Num(ram); }
        if (GpuMemoryGb is { } vram) { yield return "--gpu-memory"; yield return Num(vram); }
        if (GpuFormat is { Length: > 0 } f) { yield return "--gpu-format"; yield return f; }
        if (Fp4Prefill is { Length: > 0 } p) { yield return "--fp4-prefill"; yield return p; }
        if (Verbose) yield return "--verbose";
    }

    private static string Num(double v) => v.ToString("0.##", System.Globalization.CultureInfo.InvariantCulture);
}

/// <summary>Project-wide defaults the user can stamp into any command form.</summary>
public sealed class ProjectDefaults
{
    public string ContainerDir { get; set; } = string.Empty;
    public string TokenizerDir { get; set; } = string.Empty;
    public string PatchFile { get; set; } = string.Empty;
    public string Component { get; set; } = "target";
}

/// <summary>Persisted state of one command tab: its parameter values.</summary>
public sealed class CommandState
{
    public Dictionary<string, string> Values { get; set; } = new(StringComparer.OrdinalIgnoreCase);
}

/// <summary>One CLI invocation and its outcome — the status/progress record.</summary>
public sealed class RunRecord
{
    public int Id { get; set; }
    public string Command { get; set; } = string.Empty;
    /// <summary>The full CLI argument list (after the command name), as executed.</summary>
    public List<string> Arguments { get; set; } = new();
    public DateTime StartedUtc { get; set; }
    public DateTime? FinishedUtc { get; set; }
    /// <summary>Running | Succeeded | Failed | Cancelled.</summary>
    public string Status { get; set; } = "Running";
    public int? ExitCode { get; set; }
    /// <summary>Output lines seen so far; updated while running so saves capture progress.</summary>
    public int OutputLines { get; set; }
    /// <summary>Last parsed percentage (0–100) from the CLI's progress output, when any.</summary>
    public double? ProgressPercent { get; set; }
    /// <summary>Trailing output lines kept in the project file (bounded).</summary>
    public List<string> OutputTail { get; set; } = new();
    public string? Error { get; set; }

    public string StartedLocal => StartedUtc.ToLocalTime().ToString("d MMM HH:mm");

    public string Display =>
        $"#{Id} {Command} — {Status}" +
        (ExitCode is { } code ? $" (exit {code})" : string.Empty) +
        $"  {StartedUtc.ToLocalTime():yyyy-MM-dd HH:mm:ss}";
}
