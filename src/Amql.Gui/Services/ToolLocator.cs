using System.IO;

namespace Amql.Gui.Services;

/// <summary>An AMQL executable the GUI can start, and how it was found.</summary>
public sealed record ToolLocation(string FileName, IReadOnlyList<string> PrefixArguments, string Source, DateTime? BuiltUtc)
{
    /// <summary>The working directory to start it in: the repository when running from source.</summary>
    public string WorkingDirectory { get; init; } = Directory.GetCurrentDirectory();

    public string Describe() =>
        PrefixArguments.Count > 0
            ? $"{Source} (dotnet run)"
            : $"{FileName}{(BuiltUtc is { } b ? $" — built {b.ToLocalTime():d MMM HH:mm}" : "")}";
}

/// <summary>
/// Finds amql-cli and amql-server with no configuration. Every candidate
/// location is looked at and the most recently built executable wins, so a
/// fresh Release build is picked over a stale Debug one (and the other way
/// round) without anyone pointing the GUI at a path:
///   1. beside the GUI (the shared bin/&lt;Configuration&gt;/ output and installs);
///   2. the folder above it;
///   3. the repository's bin/Debug, bin/Release, bin, bin-dev and publish outputs,
///      found by walking up to AMQL.slnx from the GUI and the working directory;
///   4. every folder on PATH.
/// When no executable exists but the repository does, the tool is run from
/// source with <c>dotnet run</c>.
/// </summary>
public static class ToolLocator
{
    public const string Cli = "amql-cli";
    public const string Server = "amql-server";

    private static readonly Dictionary<string, ToolLocation?> Cache = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>The repository root, if the GUI runs from (or inside) a checkout.</summary>
    public static string? RepoRoot { get; } =
        FindRepoRoot(AppContext.BaseDirectory) ?? FindRepoRoot(Directory.GetCurrentDirectory());

    public static ToolLocation? Find(string tool)
    {
        lock (Cache)
        {
            if (!Cache.TryGetValue(tool, out var found))
            {
                found = Discover(tool);
                Cache[tool] = found;
            }
            return found;
        }
    }

    /// <summary>Forgets what was found, so a build made while the GUI runs is picked up.</summary>
    public static void Refresh()
    {
        lock (Cache)
        {
            Cache.Clear();
        }
    }

    /// <summary>The tool, or an exception that says how to make it available.</summary>
    public static ToolLocation Require(string tool) =>
        Find(tool) ?? throw new FileNotFoundException(
            $"{tool}.exe was not found beside AMQL Studio, in the repository's bin folders or on PATH. " +
            $"Build it (dotnet build src/{ProjectFor(tool)}) or put it on PATH.");

    private static ToolLocation? Discover(string tool)
    {
        string exe = tool + ".exe";
        var candidates = new List<(string Path, string Source)>();
        string baseDir = AppContext.BaseDirectory;
        candidates.Add((Path.Combine(baseDir, exe), "beside AMQL Studio"));
        if (Directory.GetParent(baseDir.TrimEnd(Path.DirectorySeparatorChar)) is { } parent)
        {
            candidates.Add((Path.Combine(parent.FullName, exe), "beside AMQL Studio"));
        }
        if (RepoRoot is { } repo)
        {
            foreach (var sub in new[] { "bin/Release", "bin/Debug", "bin", "bin-dev", "bin/Release/win-x64/publish" })
            {
                candidates.Add((Path.Combine(repo, sub, exe), "repository build"));
            }
        }
        foreach (var dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            try
            {
                candidates.Add((Path.Combine(dir.Trim('"'), exe), "PATH"));
            }
            catch (ArgumentException)
            {
                // a malformed PATH entry
            }
        }

        ToolLocation? best = null;
        foreach (var (path, source) in candidates)
        {
            try
            {
                if (!File.Exists(path))
                {
                    continue;
                }
                // A framework-dependent apphost is useless without its dll, and a build folder
                // can hold an exe whose libraries were written somewhere else.
                if (!IsComplete(path) && source != "PATH")
                {
                    continue;
                }
                var built = File.GetLastWriteTimeUtc(Path.ChangeExtension(path, ".dll") is var dll && File.Exists(dll) ? dll : path);
                if (best is null || built > best.BuiltUtc)
                {
                    best = new ToolLocation(Path.GetFullPath(path), Array.Empty<string>(), source, built)
                    {
                        WorkingDirectory = RepoRoot ?? Path.GetDirectoryName(path)!,
                    };
                }
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException or ArgumentException)
            {
                // unreadable candidate — skip it
            }
        }
        if (best is not null)
        {
            return best;
        }

        if (RepoRoot is { } root)
        {
            var project = Path.Combine(root, "src", ProjectFor(tool), ProjectFor(tool) + ".csproj");
            if (File.Exists(project))
            {
                return new ToolLocation("dotnet", new[] { "run", "--no-launch-profile", "-c", "Release", "--project", project, "--" },
                    "source in " + root, null)
                {
                    WorkingDirectory = root,
                };
            }
        }
        return null;
    }

    /// <summary>Whether an apphost has its own dll and every project library its deps.json lists beside it.</summary>
    private static bool IsComplete(string exe)
    {
        string dir = Path.GetDirectoryName(exe)!;
        if (!File.Exists(Path.ChangeExtension(exe, ".dll")))
        {
            return false;
        }
        string deps = Path.ChangeExtension(exe, ".deps.json");
        if (!File.Exists(deps))
        {
            return true;
        }
        try
        {
            using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(deps));
            if (!doc.RootElement.TryGetProperty("libraries", out var libraries))
            {
                return true;
            }
            foreach (var library in libraries.EnumerateObject())
            {
                if (library.Value.TryGetProperty("type", out var type) && type.GetString() == "project")
                {
                    string name = library.Name.Split('/')[0];
                    string own = Path.GetFileNameWithoutExtension(exe);
                    if (!string.Equals(name, own, StringComparison.OrdinalIgnoreCase) &&
                        !File.Exists(Path.Combine(dir, name + ".dll")))
                    {
                        return false;
                    }
                }
            }
            return true;
        }
        catch (System.Text.Json.JsonException)
        {
            return true;    // unreadable deps file: let the runtime decide
        }
    }

    private static string ProjectFor(string tool) => tool == Server ? "Amql.Server" : "Amql.Cli";

    /// <summary>Walks up from a directory to the folder holding AMQL.slnx.</summary>
    public static string? FindRepoRoot(string startDir)
    {
        try
        {
            for (var dir = new DirectoryInfo(startDir); dir is not null; dir = dir.Parent)
            {
                if (File.Exists(Path.Combine(dir.FullName, "AMQL.slnx")))
                {
                    return dir.FullName;
                }
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or System.Security.SecurityException)
        {
            // unreadable directory — no repository
        }
        return null;
    }
}
