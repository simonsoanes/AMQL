using System.IO;
using System.Text;
using Amql.Gui.Model;
using Amql.Gui.Run;
using Amql.Gui.Services;

namespace Amql.Gui.Commands;

/// <summary>
/// The rules a command form follows, whichever front end draws it: which
/// parameters are shown up front, how a parameter is labelled, the command
/// line it previews, what must be filled in, and how a finished run is
/// described. The desktop Tools page and the text UI's Tools screen both use
/// these, so a form means the same thing in both.
/// </summary>
public static class CommandForms
{
    public const int MaxTailLines = 400;
    public const int MaxHistoryRuns = 200;

    /// <summary>Shown above "Advanced options": required parameters, and positionals a project default fills.</summary>
    public static bool IsBasic(ParamDef p) => p.Required || (p.Kind == ParamKind.Positional && p.DefaultFrom is not null);

    /// <summary>The basic and advanced parameters of a command. A command with no basic
    /// parameters shows them all up front.</summary>
    public static (List<ParamDef> Basic, List<ParamDef> Advanced) Split(CommandDef cmd)
    {
        var basic = cmd.Params.Where(IsBasic).ToList();
        var advanced = cmd.Params.Where(p => !IsBasic(p)).ToList();
        return basic.Count == 0 ? (advanced, new List<ParamDef>()) : (basic, advanced);
    }

    /// <summary>A parameter's label: its flag or name without the dashes, capitalised.</summary>
    public static string Friendly(ParamDef p)
    {
        string text = p.Label;
        if (text.StartsWith("--", StringComparison.Ordinal))
        {
            text = text[2..];
        }
        text = text.Replace('-', ' ').Trim('<', '>');
        return text.Length == 0 ? p.Key : char.ToUpperInvariant(text[0]) + text[1..];
    }

    /// <summary>The value a form starts with: what was saved for the command, else the parameter's default.</summary>
    public static string InitialValue(ParamDef p, CommandState state) =>
        state.Values.TryGetValue(p.Key, out var saved) ? saved : p.DefaultValue;

    public static bool IsOn(string value) => value is "true" or "1" or "on";

    /// <summary>The project default a parameter can be filled from.</summary>
    public static string? DefaultFor(ProjectDefaults d, string? key) => key switch
    {
        "container" => d.ContainerDir,
        "tokenizer" => d.TokenizerDir,
        "patch" => d.PatchFile,
        "component" => d.Component,
        _ => null,
    };

    public static bool NeedsQuoting(string arg) => arg.Length == 0 || arg.Any(char.IsWhiteSpace) || arg.Contains('"');

    public static string Quote(string arg) => "\"" + arg.Replace("\"", "\\\"") + "\"";

    public static string QuoteIfNeeded(string arg) => NeedsQuoting(arg) ? Quote(arg) : arg;

    /// <summary>The full amql-cli command line a form would run.</summary>
    public static string CommandLine(CliSettings cli, CommandDef cmd, IReadOnlyDictionary<string, string> values)
    {
        var sb = new StringBuilder("amql-cli");
        foreach (var g in cli.GlobalArguments())
        {
            sb.Append(' ').Append(g);
        }
        sb.Append(' ').Append(cmd.Name);
        foreach (var a in cmd.BuildArguments(values))
        {
            sb.Append(' ').Append(QuoteIfNeeded(a));
        }
        return sb.ToString();
    }

    /// <summary>The labels of required parameters left empty.</summary>
    public static List<string> MissingRequired(CommandDef cmd, IReadOnlyDictionary<string, string> values) =>
        cmd.Params
            .Where(p => p.Required && p.Kind != ParamKind.Switch && (!values.TryGetValue(p.Key, out var v) || v.Length == 0))
            .Select(Friendly).ToList();

    /// <summary>Where a new container goes when the form does not say: the repository's
    /// containers folder, the first model folder, or Documents/AMQL/containers.</summary>
    public static string ContainersFolder() =>
        ToolLocator.RepoRoot is { } repo ? Path.Combine(repo, "containers")
        : AppSettings.Current.ModelFolders.FirstOrDefault()
          ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "AMQL", "containers");

    /// <summary>The status a CLI exit code records.</summary>
    public static string StatusFor(int exit) => exit switch
    {
        CliExitCodes.Ok => "Succeeded",
        CliExitCodes.NegativeResult => "Negative",
        _ => "Failed",
    };

    /// <summary>
    /// After encode makes a container, makes sure the Server screen will list it:
    /// added to the extra models unless a model folder already covers it. Returns
    /// the line to show in the output, or null when the folder is not a container.
    /// </summary>
    public static string? AnnounceContainer(string folder)
    {
        try
        {
            string full = Path.GetFullPath(folder);
            if (!File.Exists(Path.Combine(full, "index.json")))
            {
                return null;
            }
            char sep = Path.DirectorySeparatorChar;
            var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
            bool covered = AppSettings.Current.ModelFolders.Any(root =>
            {
                string r = Path.GetFullPath(root).TrimEnd(sep);
                return full.StartsWith(r + sep, comparison)
                       && full.Count(c => c == sep) - r.Count(c => c == sep) <= 2;
            });
            if (!covered && !AppSettings.Current.ExtraModels.Contains(full, StringComparer.FromComparison(comparison)))
            {
                AppSettings.Current.ExtraModels.Add(full);
                AppSettings.Current.Save();
            }
            return $"[studio] {Path.GetFileName(full)} is ready: tick it on the Server page to serve it (rescan if the page is open).";
        }
        catch (Exception e) when (e is ArgumentException or IOException or NotSupportedException)
        {
            return null;
        }
    }

    /// <summary>A finished run as text: its command line, times, status and output tail.</summary>
    public static string DescribeRun(RunRecord record)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"# Run {record.Id}: amql-cli {record.Command} {string.Join(' ', record.Arguments.Select(QuoteIfNeeded))}");
        sb.AppendLine($"# started {record.StartedUtc.ToLocalTime():yyyy-MM-dd HH:mm:ss}" +
                      (record.FinishedUtc is { } end ? $", finished {end.ToLocalTime():HH:mm:ss}" : string.Empty));
        sb.AppendLine($"# status: {record.Status}" +
                      (record.ExitCode is { } code ? $", exit {code}" : string.Empty) +
                      (record.ProgressPercent is { } pct ? $", progress {pct:0.#}%" : string.Empty));
        if (record.Error is { Length: > 0 })
        {
            sb.AppendLine($"# error: {record.Error}");
        }
        sb.AppendLine($"# output tail ({record.OutputTail.Count} of {record.OutputLines} lines):");
        foreach (var line in record.OutputTail)
        {
            sb.AppendLine(line);
        }
        return sb.ToString();
    }

    /// <summary>The one-line result after a run: "succeeded in 4.2 s", "failed (exit 2) in 1m 03s".</summary>
    public static string RunSummary(RunRecord record)
    {
        var took = (record.FinishedUtc ?? DateTime.UtcNow) - record.StartedUtc;
        return record.Status.ToLowerInvariant() +
               (record.ExitCode is { } code and not 0 ? $" (exit {code})" : string.Empty) +
               $" in {Format.Duration(took)}";
    }

    public static string SanitizeFileName(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var result = new string(name.Select(c => invalid.Contains(c) ? '_' : c).ToArray()).Trim();
        return result.Length == 0 ? "project" : result;
    }
}
