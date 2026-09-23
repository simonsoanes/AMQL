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

        // 3. Debug build: src/Amql.Cli/bin/Debug/net10.0/amql-cli.exe
        var repoRoot = FindRepoRoot(baseDir);
        if (repoRoot is not null)
        {
            var debugExe = Path.Combine(repoRoot, "src", "Amql.Cli", "bin", "Debug", "net10.0", "amql-cli.exe");
            if (File.Exists(debugExe)) return debugExe;
        }

        // 4. Release publish (self-contained): .../Release/net10.0/win-x64/publish/amql-cli.exe
        if (repoRoot is not null)
        {
            var releaseExe = Path.Combine(repoRoot, "src", "Amql.Cli", "bin", "Release",
                "net10.0", "win-x64", "publish", "amql-cli.exe");
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