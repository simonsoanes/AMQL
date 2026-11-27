using System.Collections.Concurrent;
using System.Text.RegularExpressions;

namespace Amql.Manager;

public enum JobState
{
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

public sealed class CliJob
{
    public required string Id { get; init; }
    public required IReadOnlyList<string> Args { get; init; }
    public string? WorkingDirectory { get; init; }
    public DateTimeOffset Queued { get; } = DateTimeOffset.UtcNow;
    public JobState State { get; internal set; } = JobState.Queued;
    public ChildProcess? Process { get; internal set; }
    public string? Error { get; internal set; }
    public required string LogPath { get; init; }
    internal readonly CancellationTokenSource Cancel = new();
}

/// <summary>
/// Every amql-cli command, as a job: its arguments are passed as a list, never
/// through a shell. Jobs run in turn up to <see cref="JobSettings.MaxConcurrent"/>
/// at once, below normal priority by default, each logged to its own file.
/// </summary>
public sealed class CliJobs : IDisposable
{
    private readonly ManagerConfig _config;
    private readonly ConcurrentDictionary<string, CliJob> _jobs = new();
    private readonly SemaphoreSlim _slots;
    private readonly string _logs;
    private IReadOnlyList<CliCommand>? _commands;

    public CliJobs(ManagerConfig config)
    {
        _config = config;
        _slots = new SemaphoreSlim(Math.Max(1, config.Jobs.MaxConcurrent));
        _logs = Path.Combine(config.DataDirectory, "jobs");
        Directory.CreateDirectory(_logs);
    }

    public IEnumerable<CliJob> All => _jobs.Values.OrderByDescending(j => j.Queued);

    public CliJob? Get(string id) => _jobs.GetValueOrDefault(id);

    public bool Busy => _jobs.Values.Any(j => j.State is JobState.Queued or JobState.Running);

    public CliJob Submit(IReadOnlyList<string> args, string? workingDirectory)
    {
        if (args.Count == 0)
        {
            throw new ManagerException("a job needs the command's arguments, e.g. [\"verify\", \"containers/x\"]");
        }
        string id = $"{DateTimeOffset.UtcNow:yyyyMMdd-HHmmss}-{Guid.NewGuid().ToString("N")[..6]}";
        var job = new CliJob
        {
            Id = id,
            Args = args.ToArray(),
            WorkingDirectory = workingDirectory,
            LogPath = Path.Combine(_logs, id + ".log"),
        };
        _jobs[id] = job;
        _ = RunAsync(job);
        Trim();
        return job;
    }

    private async Task RunAsync(CliJob job)
    {
        try
        {
            await _slots.WaitAsync(job.Cancel.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            job.State = JobState.Cancelled;
            return;
        }
        try
        {
            job.State = JobState.Running;
            job.Process = ChildProcess.Start(Tools.Find(_config.BinDirectory, "amql-cli"), job.Args, job.WorkingDirectory,
                lowPriority: _config.Jobs.LowPriority, logPath: job.LogPath);
            using (job.Cancel.Token.Register(() => _ = job.Process.StopAsync(TimeSpan.FromSeconds(10))))
            {
                await job.Process.Completion.ConfigureAwait(false);
            }
            job.State = job.Cancel.IsCancellationRequested ? JobState.Cancelled
                : job.Process.ExitCode == 0 ? JobState.Succeeded : JobState.Failed;
        }
        catch (Exception e)
        {
            job.Error = e.Message;
            job.State = JobState.Failed;
        }
        finally
        {
            _slots.Release();
        }
    }

    public bool CancelJob(string id)
    {
        if (Get(id) is not { State: JobState.Queued or JobState.Running } job)
        {
            return false;
        }
        job.Cancel.Cancel();
        return true;
    }

    private void Trim()
    {
        var finished = _jobs.Values.Where(j => j.State is JobState.Succeeded or JobState.Failed or JobState.Cancelled)
            .OrderByDescending(j => j.Queued).Skip(_config.Jobs.KeepFinished).ToList();
        foreach (var j in finished)
        {
            if (_jobs.TryRemove(j.Id, out _))
            {
                j.Process?.Dispose();
            }
        }
    }

    /// <summary>The commands amql-cli lists in its own help: the name and the usage lines under it.</summary>
    public async Task<IReadOnlyList<CliCommand>> CommandsAsync()
    {
        if (_commands is not null)
        {
            return _commands;
        }
        using var help = ChildProcess.Start(Tools.Find(_config.BinDirectory, "amql-cli"), Array.Empty<string>(), lowPriority: false);
        await help.Completion.WaitAsync(TimeSpan.FromSeconds(30)).ConfigureAwait(false);
        var (lines, _, _) = help.Read(0, ChildProcess.Keep);
        _commands = ParseHelp(lines);
        return _commands;
    }

    private static readonly Regex Usage = new(@"^  amql-cli (?<name>[a-z][a-z0-9-]*(?: [a-z][a-z0-9-]*)?)(?=\s|$)", RegexOptions.Compiled);

    internal static IReadOnlyList<CliCommand> ParseHelp(IReadOnlyList<string> lines)
    {
        var commands = new List<CliCommand>();
        List<string>? usage = null;
        foreach (string line in lines)
        {
            var m = Usage.Match(line);
            if (m.Success)
            {
                string name = m.Groups["name"].Value;
                // "amql-cli context bench" is a command of two words; "amql-cli encode <dir>" is one.
                string[] words = name.Split(' ');
                if (words.Length == 2 && line.Length > m.Length && line[m.Length..].TrimStart().StartsWith('-'))
                {
                    name = words[0];
                }
                usage = new List<string> { line.Trim() };
                commands.Add(new CliCommand(name, usage));
            }
            else if (usage is not null && line.StartsWith("    ", StringComparison.Ordinal))
            {
                usage.Add(line.Trim());
            }
            else if (line.Length > 0 && !line.StartsWith(' '))
            {
                usage = null;
            }
        }
        return commands.GroupBy(c => c.Name).Select(g => new CliCommand(g.Key, g.SelectMany(c => c.Usage).ToList())).ToList();
    }

    public void Dispose()
    {
        foreach (var j in _jobs.Values)
        {
            j.Cancel.Cancel();
            j.Process?.Dispose();
        }
    }
}

public sealed record CliCommand(string Name, IReadOnlyList<string> Usage);
