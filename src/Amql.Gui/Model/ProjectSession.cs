using System.IO;
using Amql.Gui.Commands;
using Amql.Gui.Run;
using Amql.Gui.Services;

namespace Amql.Gui.Model;

/// <summary>
/// The open project and its run history, without any UI: opening, creating
/// and saving .amqlproj files, and running a command so that the run is
/// recorded the way the project format expects (status, exit code, progress,
/// output tail). The text UI's Tools screen is built on it.
/// </summary>
public sealed class ProjectSession
{
    private int _nextRunId = 1;

    public ProjectModel Project { get; private set; } = new() { Name = "Default" };

    public string Path { get; private set; } = DefaultProjectPath;

    /// <summary>Runs, newest first.</summary>
    public List<RunRecord> History { get; } = new();

    public static string DefaultProjectPath => System.IO.Path.Combine(AppSettings.Folder, "Default" + ProjectModel.FileExtension);

    public bool IsDefault => Path == DefaultProjectPath;

    public string DisplayName => IsDefault ? "Default project" : Project.Name;

    public CliSettings Cli => Project.Cli;

    /// <summary>Opens the project used last, or the default one.</summary>
    public void OpenLastOrDefault()
    {
        var last = AppSettings.Current.LastProject;
        if (!TryOpen(last is not null && File.Exists(last) ? last : DefaultProjectPath, out _))
        {
            Project = new ProjectModel { Name = "Default" };
            Path = DefaultProjectPath;
            History.Clear();
            _nextRunId = 1;
        }
    }

    /// <summary>Opens a project file. Runs it recorded as still running are marked Interrupted.</summary>
    public bool TryOpen(string path, out string? error)
    {
        error = null;
        try
        {
            if (!File.Exists(path))
            {
                error = "the file does not exist";
                return false;
            }
            Project = ProjectModel.Load(path);
            Path = path;
            foreach (var run in Project.Runs.Where(r => r.Status == "Running"))
            {
                run.Status = "Interrupted";
                run.Error = "the app exited while this run was active";
                run.FinishedUtc ??= run.StartedUtc;
            }
            _nextRunId = Project.Runs.Count == 0 ? 1 : Project.Runs.Max(r => r.Id) + 1;
            History.Clear();
            History.AddRange(Enumerable.Reverse(Project.Runs));
            return true;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return false;
        }
    }

    /// <summary>Starts a new project at <paramref name="path"/> and saves it.</summary>
    public string? CreateNew(string name, string path, string container, string tokenizer)
    {
        try
        {
            var dir = System.IO.Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir))
            {
                Directory.CreateDirectory(dir);
            }
        }
        catch (Exception ex)
        {
            return $"Cannot create the project folder: {ex.Message}";
        }
        Project = new ProjectModel { Name = name };
        Project.Defaults.ContainerDir = container;
        Project.Defaults.TokenizerDir = tokenizer;
        Path = path;
        History.Clear();
        _nextRunId = 1;
        return Save();
    }

    /// <summary>Saves under a new path (and name).</summary>
    public string? SaveAs(string path)
    {
        Path = path;
        Project.Name = System.IO.Path.GetFileNameWithoutExtension(path);
        return Save();
    }

    /// <summary>Saves the project; returns an error message, or null.</summary>
    public string? Save()
    {
        try
        {
            Project.Runs = Enumerable.Reverse(History).ToList();
            Directory.CreateDirectory(System.IO.Path.GetDirectoryName(Path)!);
            Project.Save(Path);
            AppSettings.Current.LastProject = IsDefault ? null : Path;
            AppSettings.Current.Save();
            return null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return $"Could not save: {ex.Message}";
        }
    }

    public void ClearHistory()
    {
        History.Clear();
        Save();
    }

    /// <summary>
    /// Runs one command and records it. <paramref name="onLine"/> and
    /// <paramref name="onProgress"/> are called on the UI thread. The record is
    /// in <see cref="History"/> from the start, so a save mid-run captures it.
    /// </summary>
    public async Task<RunRecord> RunAsync(
        CommandDef cmd,
        IReadOnlyDictionary<string, string> values,
        IReadOnlyDictionary<string, string>? environment,
        Action<RunRecord> onStarted,
        Action<string> onLine,
        Action<double> onProgress,
        CancellationToken cancel)
    {
        var args = cmd.BuildArguments(values);
        var record = new RunRecord
        {
            Id = _nextRunId++,
            Command = cmd.Name,
            Arguments = args,
            StartedUtc = DateTime.UtcNow,
            Status = "Running",
        };
        History.Insert(0, record);
        while (History.Count > CommandForms.MaxHistoryRuns)
        {
            History.RemoveAt(History.Count - 1);
        }
        onStarted(record);
        var tail = new Queue<string>();
        int lines = 0;
        try
        {
            int exit = await CliRunner.RunAsync(
                Cli, cmd.Name, args,
                line => UiThread.Run(() =>
                {
                    lines++;
                    record.OutputLines = lines;
                    tail.Enqueue(line);
                    while (tail.Count > CommandForms.MaxTailLines)
                    {
                        tail.Dequeue();
                    }
                    onLine(line);
                }),
                pct => UiThread.Run(() =>
                {
                    record.ProgressPercent = pct;
                    onProgress(pct);
                }),
                cancel,
                environment: environment);
            // The output readers post their last lines just before the exit is seen.
            await Task.Delay(50);
            record.ExitCode = exit;
            record.Status = CommandForms.StatusFor(exit);
            if (exit == CliExitCodes.Ok && cmd.Name == "encode" && values.TryGetValue("out", out var made)
                && CommandForms.AnnounceContainer(made) is { } note)
            {
                onLine(note);
                tail.Enqueue(note);
            }
        }
        catch (OperationCanceledException)
        {
            record.Status = "Cancelled";
            record.Error = "cancelled by user";
        }
        catch (Exception ex)
        {
            record.Status = "Failed";
            record.Error = ex.Message;
            onLine("[studio] " + ex.Message);
            tail.Enqueue("[studio] " + ex.Message);
        }
        finally
        {
            record.FinishedUtc = DateTime.UtcNow;
            record.OutputTail = tail.ToList();
            if (record.Status == "Running")
            {
                record.Status = "Failed";
            }
            Save();
        }
        return record;
    }
}
