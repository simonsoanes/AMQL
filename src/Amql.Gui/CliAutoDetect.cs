using System.IO;

namespace Amql.Gui;

/// <summary>Scans nearby directories for the amql-cli executable, supporting
/// both the debug build location and a locally copied adjacent binary.</summary>
public static class CliAutoDetect
{
    /// <summary>Returns the first found amql-cli.exe path, or null.</summary>
    public static string? FindExe()
    {
        var baseDir = AppContext.BaseDirectory;

        // 1. Adjacent copy (same directory as the GUI)
        var adjacent = Path.Combine(baseDir, "amql-cli.exe");
        if (File.Exists(adjacent)) return adjacent;

        // 2. Parent directory (common for side-by-side deployment)
        var parent = Path.Combine(Path.GetDirectoryName(baseDir) ?? baseDir, "amql-cli.exe");
        if (File.Exists(parent)) return parent;

        // 3. Shared solution output: every project under src/ writes to
        // <repo>/bin/<Configuration>/ (see src/Directory.Build.props), so the
        // CLI built from this working tree is here whether the GUI itself was
        // launched from there or from somewhere else.
        var repoRoot = FindRepoRoot(baseDir);
        if (repoRoot is not null)
        {
            foreach (var configuration in new[] { "Debug", "Release" })
            {
                var built = Path.Combine(repoRoot, "bin", configuration, "amql-cli.exe");
                if (File.Exists(built)) return built;
            }
        }

        // 4. Release publish (self-contained)
        if (repoRoot is not null)
        {
            var releaseExe = Path.Combine(repoRoot, "bin", "Release", "win-x64", "publish", "amql-cli.exe");
            if (File.Exists(releaseExe)) return releaseExe;
        }

        // 5. Via dotnet run — check if repo root exists (fallback, returns null)
        return null;
    }

    /// <summary>Finds the AMQL repository root by walking up from the given
    /// directory looking for AMQL.slnx.</summary>
    public static string? FindRepoRoot(string startDir)
    {
        try
        {
            var dir = new DirectoryInfo(startDir);
            while (dir is not null)
            {
                if (File.Exists(Path.Combine(dir.FullName, "AMQL.slnx")))
                {
                    return dir.FullName;
                }
                dir = dir.Parent;
            }
        }
        catch
        {
            // permissions or deleted directories — ignore
        }
        return null;
    }
}