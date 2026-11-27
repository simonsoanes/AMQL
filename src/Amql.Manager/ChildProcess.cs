using System.Diagnostics;

namespace Amql.Manager;

/// <summary>
/// A process the manager starts: an executable and its arguments, never a
/// shell, so no argument is ever interpreted. Its output is kept as numbered
/// lines (the latest <see cref="Keep"/> in memory, all of them in a log file
/// when one is given), so a client can read it incrementally.
/// </summary>
public sealed class ChildProcess : IDisposable
{
    public const int Keep = 2000;

    private readonly Process _process;
    private readonly LinkedList<(long Number, string Line)> _lines = new();
    private readonly object _gate = new();
    private readonly StreamWriter? _log;
    private long _count;

    public DateTimeOffset Started { get; } = DateTimeOffset.UtcNow;
    public DateTimeOffset? Exited { get; private set; }
    public int? ExitCode { get; private set; }
    public int Id => _process.Id;
    public Task Completion { get; }

    private ChildProcess(Process process, StreamWriter? log)
    {
        _process = process;
        _log = log;
        var done = new TaskCompletionSource();
        _process.EnableRaisingEvents = true;
        _process.OutputDataReceived += (_, e) => Add(e.Data);
        _process.ErrorDataReceived += (_, e) => Add(e.Data);
        _process.Exited += (_, _) =>
        {
            // The output streams can still be draining: WaitForExit() with no
            // timeout waits for them too.
            try
            {
                _process.WaitForExit();
            }
            catch (InvalidOperationException)
            {
            }
            ExitCode = _process.ExitCode;
            Exited = DateTimeOffset.UtcNow;
            lock (_gate)
            {
                _log?.Flush();
            }
            done.TrySetResult();
        };
        Completion = done.Task;
    }

    public static ChildProcess Start(string executable, IEnumerable<string> arguments, string? workingDirectory = null,
        IReadOnlyDictionary<string, string>? environment = null, bool lowPriority = true, string? logPath = null)
    {
        var start = new ProcessStartInfo(executable)
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = true,
            CreateNoWindow = true,
            WorkingDirectory = workingDirectory ?? Environment.CurrentDirectory,
        };
        foreach (string a in arguments)
        {
            start.ArgumentList.Add(a);
        }
        if (environment is not null)
        {
            foreach (var (k, v) in environment)
            {
                start.Environment[k] = v;
            }
        }
        StreamWriter? log = null;
        if (logPath is not null)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(logPath)!);
            log = new StreamWriter(logPath, append: true) { AutoFlush = false };
        }
        var process = new Process { StartInfo = start };
        var child = new ChildProcess(process, log);
        try
        {
            if (!process.Start())
            {
                throw new ManagerException($"could not start {executable}");
            }
        }
        catch
        {
            // A start that fails leaves nothing open: the log file especially, or every
            // later start that logs to it fails too.
            log?.Dispose();
            process.Dispose();
            throw;
        }
        process.StandardInput.Close();
        if (lowPriority)
        {
            try
            {
                process.PriorityClass = ProcessPriorityClass.BelowNormal;
            }
            catch (Exception e) when (e is InvalidOperationException or System.ComponentModel.Win32Exception)
            {
                // It exited already, or the platform refused; it runs as it is.
            }
        }
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        return child;
    }

    public bool Running => Exited is null;

    private void Add(string? line)
    {
        if (line is null)
        {
            return;
        }
        lock (_gate)
        {
            _lines.AddLast((_count++, line));
            if (_lines.Count > Keep)
            {
                _lines.RemoveFirst();
            }
            _log?.WriteLine(line);
            if (_count % 64 == 0)
            {
                _log?.Flush();
            }
        }
    }

    /// <summary>The lines numbered from <paramref name="from"/> on (as many as are still
    /// held), and the number the next read starts at.</summary>
    public (IReadOnlyList<string> Lines, long First, long Next) Read(long from = 0, int max = 1000)
    {
        lock (_gate)
        {
            var lines = _lines.Where(l => l.Number >= from).Take(max).ToList();
            long first = lines.Count > 0 ? lines[0].Number : _count;
            return (lines.Select(l => l.Line).ToList(), first, lines.Count > 0 ? lines[^1].Number + 1 : _count);
        }
    }

    public string Tail(int count = 20)
    {
        lock (_gate)
        {
            return string.Join("\n", _lines.Skip(Math.Max(0, _lines.Count - count)).Select(l => l.Line));
        }
    }

    /// <summary>Stops the process and everything it started.</summary>
    public async Task StopAsync(TimeSpan wait)
    {
        if (!Running)
        {
            return;
        }
        try
        {
            _process.Kill(entireProcessTree: true);
        }
        catch (InvalidOperationException)
        {
            return;
        }
        await Task.WhenAny(Completion, Task.Delay(wait)).ConfigureAwait(false);
    }

    public void Dispose()
    {
        if (Running)
        {
            try
            {
                _process.Kill(entireProcessTree: true);
            }
            catch (InvalidOperationException)
            {
            }
        }
        _process.Dispose();
        lock (_gate)
        {
            _log?.Dispose();
        }
    }
}
