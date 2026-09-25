using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Amql.Gui.Model;

namespace Amql.Gui.Run;

/// <summary>How to invoke amql-cli, resolved from <see cref="CliSettings"/>.</summary>
public sealed record CliInvocation(string FileName, IReadOnlyList<string> PrefixArguments);

/// <summary>
/// Launches amql-cli as a child process and streams its output. Transforms and
/// anything long-running go through the CLI, which keeps the GUI at feature
/// parity with the command line by construction and insulated from concurrent
/// core changes. The GUI does reference Amql.Vindex3, Amql.Safetensors and
/// Amql.Inference directly — for container inspection in the Explorer and for
/// the inference visualiser's trace model — so this is the default path for
/// commands, not a rule that nothing is linked in.
/// </summary>
public sealed class CliRunner
{
    /// <summary>Protocol line prefixes emitted by the CLI under --progress
    /// (see src/Amql.Cli/CliProgress.cs). Machine-readable, so the GUI binds
    /// progress exactly instead of scraping 'NN%' out of prose.</summary>
    public const string ProgressPrefix = "##amql-progress ";
    public const string ResultPrefix = "##amql-result ";

    private static readonly Regex PercentRegex = new(@"(?<![\w.])(\d{1,3}(?:[.,]\d+)?)\s*%", RegexOptions.Compiled);
    private static readonly Regex FractionRegex = new(@"\b(\d+)\s*/\s*(\d+)\b", RegexOptions.Compiled);

    /// <summary>Resolves the CLI invocation, or throws with an actionable message.</summary>
    public static CliInvocation Resolve(CliSettings settings)
    {
        if (!string.IsNullOrWhiteSpace(settings.CliExePath))
        {
            var exe = settings.CliExePath.Trim();
            if (!File.Exists(exe))
            {
                throw new FileNotFoundException($"CLI executable not found: '{exe}'", exe);
            }
            return new CliInvocation(exe, Array.Empty<string>());
        }

        if (!string.IsNullOrWhiteSpace(settings.RepoPath))
        {
            var repo = settings.RepoPath.Trim();
            var cliProject = Path.Combine(repo, "src", "Amql.Cli", "Amql.Cli.csproj");
            if (!File.Exists(cliProject))
            {
                throw new FileNotFoundException(
                    $"no src/Amql.Cli/Amql.Cli.csproj under repo path '{repo}' — set the CLI exe path or fix the repo path", cliProject);
            }
            return new CliInvocation("dotnet", new[] { "run", "--no-launch-profile", "--project", cliProject, "--" });
        }

        // No explicit exe and no repo path: try autodetect before giving up on
        // PATH. This is what makes the shared bin/<Configuration>/ output
        // useful — a CLI built beside the GUI is found with no configuration,
        // which matters for windows that are not opened from a project (the
        // inference visualiser launched via `amql-gui --trace`, say) and so
        // have default settings.
        if (CliAutoDetect.FindExe() is { } detected)
        {
            return new CliInvocation(detected, Array.Empty<string>());
        }

        // Last resort: amql-cli on PATH.
        return new CliInvocation("amql-cli", Array.Empty<string>());
    }

    /// <summary>
    /// Runs one command, invoking <paramref name="onLine"/> for every output
    /// line (on a thread-pool thread — marshal to the UI thread yourself) and
    /// returning the exit code. Cancellation kills the process tree.
    /// </summary>
    public static async Task<int> RunAsync(
        CliSettings settings,
        string command,
        IReadOnlyList<string> args,
        Action<string> onLine,
        Action<double>? onProgress,
        CancellationToken cancellationToken,
        bool machineProgress = true)
    {
        var invocation = Resolve(settings);

        var argv = new List<string>(invocation.PrefixArguments);
        if (settings.Device == "cpu") argv.Add("--cpu");
        else if (settings.Device == "gpu") argv.Add("--gpu");
        // Ask for the machine-readable protocol: the CLI emits newline-delimited
        // ##amql-progress / ##amql-result JSON on stdout, which the parser below
        // binds to exactly. ParseProgress remains as a fallback for the
        // human-readable output the protocol does not cover.
        if (machineProgress) argv.Add("--progress");
        argv.Add(command);
        argv.AddRange(args);

        var psi = new ProcessStartInfo
        {
            FileName = invocation.FileName,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = !string.IsNullOrWhiteSpace(settings.RepoPath)
                ? settings.RepoPath.Trim()
                : Directory.GetCurrentDirectory(),
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        foreach (var a in argv)
        {
            psi.ArgumentList.Add(a);
        }
        if (!string.IsNullOrWhiteSpace(settings.WeightsEnv))
        {
            psi.Environment["AMQL_WEIGHTS"] = settings.WeightsEnv.Trim();
        }

        using var process = new Process { StartInfo = psi, EnableRaisingEvents = true };

        void Handle(string? line, bool isError)
        {
            if (line is null) return;

            // Protocol lines are consumed, not echoed verbatim: the progress
            // bar binds to the JSON and a one-line readable summary replaces
            // the raw payload in the output pane.
            if (!isError && CliRunnerProtocol.TryHandleProtocolLine(line, onLine, onProgress))
            {
                return;
            }

            onLine(isError ? "[stderr] " + line : line);
            // Heuristic fallback for output the protocol does not cover.
            var pct = ParseProgress(line);
            if (!double.IsNaN(pct))
            {
                onProgress?.Invoke(pct);
            }
        }

        process.OutputDataReceived += (_, e) => Handle(e.Data, isError: false);
        process.ErrorDataReceived += (_, e) => Handle(e.Data, isError: true);

        try
        {
            process.Start();
        }
        catch (Exception e)
        {
            throw new InvalidOperationException(
                $"cannot start '{invocation.FileName}' — check the CLI settings in the project (exe path or repo path). {e.Message}", e);
        }
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        using var registration = cancellationToken.Register(() =>
        {
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                }
            }
            catch (Exception)
            {
                // best-effort kill; the wait below observes the real outcome
            }
        });

        try
        {
            await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            try { await process.WaitForExitAsync(CancellationToken.None).ConfigureAwait(false); }
            catch (Exception) { /* process died mid-wait */ }
            throw;
        }

        // Give the async output readers a moment to drain after exit.
        await Task.Delay(100).ConfigureAwait(false);

        return process.ExitCode;
    }

    /// <summary>Best-effort progress extraction: 'NN%' anywhere, or an 'N/M' counter.</summary>
    public static double ParseProgress(string line)
    {
        var m = PercentRegex.Match(line);
        if (m.Success && double.TryParse(m.Groups[1].Value.Replace(',', '.'),
                System.Globalization.CultureInfo.InvariantCulture, out var pct) && pct is >= 0 and <= 100)
        {
            return pct;
        }
        var f = FractionRegex.Match(line);
        if (f.Success &&
            long.TryParse(f.Groups[1].Value, out var done) &&
            long.TryParse(f.Groups[2].Value, out var total) &&
            total > 0 && done <= total && done > 0)
        {
            return Math.Round(100.0 * done / total, 1);
        }
        return double.NaN;
    }
}

/// <summary>The CLI's documented exit codes (see 'amql-cli help'):</summary>
/// <remarks>
///   0 — success; 1 — a legitimate negative result (path found nothing within
///   budget, verify's integrity check failed), which is an answer rather than
///   a fault; 2 — usage or runtime error.
/// </remarks>
public static class CliExitCodes
{
    public const int Ok = 0;
    public const int NegativeResult = 1;
    public const int Usage = 2;

    public static bool IsFault(int code) => code != Ok && code != NegativeResult;
}

/// <summary>Protocol-line parsing used by <see cref="CliRunner"/>.</summary>
internal static class CliRunnerProtocol
{
    /// <summary>
    /// Handles one stdout line if it is a protocol line; returns false
    /// otherwise (the caller then falls back to heuristics). Emits a compact
    /// human-readable summary through <paramref name="onLine"/> and feeds the
    /// percentage to <paramref name="onProgress"/>.
    /// </summary>
    internal static bool TryHandleProtocolLine(
        string line,
        Action<string> onLine,
        Action<double>? onProgress)
    {
        bool isProgress = line.StartsWith(CliRunner.ProgressPrefix, StringComparison.Ordinal);
        bool isResult = line.StartsWith(CliRunner.ResultPrefix, StringComparison.Ordinal);
        if (!isProgress && !isResult)
        {
            return false;
        }

        string json = line[(isProgress ? CliRunner.ProgressPrefix : CliRunner.ResultPrefix).Length..];
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            if (isProgress)
            {
                // The CLI emits camelCase; accept any casing so a renamed or
                // older payload still drives the bar.
                if (TryGetNumber(root, "percent", out var pct))
                {
                    onProgress?.Invoke(pct);
                }
                else if (TryGetPropertyIgnoreCase(root, "phase", out _) ||
                         TryGetPropertyIgnoreCase(root, "command", out _))
                {
                    // A new phase without a percentage: restart the bar.
                    onProgress?.Invoke(0);
                }
                return true; // progress lines are noise in the output pane
            }

            onLine("[result] " + Summarise(root));
            return true;
        }
        catch (JsonException)
        {
            // Not valid JSON after all — treat it as ordinary output.
            return false;
        }
    }

    private static bool TryGetPropertyIgnoreCase(JsonElement root, string name, out JsonElement value)
    {
        if (root.ValueKind == JsonValueKind.Object)
        {
            foreach (var property in root.EnumerateObject())
            {
                if (string.Equals(property.Name, name, StringComparison.OrdinalIgnoreCase))
                {
                    value = property.Value;
                    return true;
                }
            }
        }
        value = default;
        return false;
    }

    private static bool TryGetNumber(JsonElement root, string name, out double value)
    {
        value = double.NaN;
        return TryGetPropertyIgnoreCase(root, name, out var element) &&
               element.ValueKind == JsonValueKind.Number &&
               element.TryGetDouble(out value);
    }

    private static string Summarise(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
        {
            return root.ToString();
        }
        var parts = new List<string>();
        foreach (var property in root.EnumerateObject())
        {
            parts.Add(property.Value.ValueKind switch
            {
                JsonValueKind.String => $"{property.Name}={property.Value.GetString()}",
                JsonValueKind.Number or JsonValueKind.True or JsonValueKind.False or JsonValueKind.Null =>
                    $"{property.Name}={property.Value}",
                _ => $"{property.Name}=…",
            });
        }
        return string.Join(", ", parts);
    }
}
