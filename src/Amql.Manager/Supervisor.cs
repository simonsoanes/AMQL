using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Amql.Manager;

/// <summary>
/// The service itself (a Windows service or a systemd unit runs
/// <c>amql-manager supervise --root &lt;dir&gt;</c>): it runs the current
/// version's manager and keeps it running.
/// <list type="bullet">
/// <item>An exit with <see cref="Updater.RestartExitCode"/> is a requested restart (an update or a
/// rollback was made current): the next manager starts at once, from whichever version is current now.</item>
/// <item>Any other exit restarts after a pause that grows to a minute.</item>
/// <item>A version made current that fails within a minute of starting, three times running, is
/// rolled back to the version before it.</item>
/// </list>
/// The supervisor is the service's own copy (in <c>&lt;root&gt;/supervisor</c>), so an update never
/// replaces the running executable.
/// </summary>
public sealed class Supervisor(string root, ILogger<Supervisor> log) : BackgroundService
{
    private ChildProcess? _child;

    protected override async Task ExecuteAsync(CancellationToken stop)
    {
        int quickFailures = 0;
        TimeSpan pause = TimeSpan.FromSeconds(2);
        while (!stop.IsCancellationRequested)
        {
            string? version = Layout.ReadCurrent(root);
            if (version is null)
            {
                log.LogError("{Root}/current names no version: install one with scripts/service/install-*", root);
                await Task.Delay(TimeSpan.FromSeconds(30), stop).ConfigureAwait(false);
                continue;
            }
            string dir = Layout.VersionDir(root, version);
            string exe = Layout.Executable(dir, "amql-manager");
            if (!File.Exists(exe))
            {
                log.LogError("version {Version} is current but {Exe} is missing", version, exe);
                if (!RollBack(version))
                {
                    await Task.Delay(TimeSpan.FromSeconds(30), stop).ConfigureAwait(false);
                }
                continue;
            }

            log.LogInformation("starting AMQL manager {Version}", version);
            var started = DateTimeOffset.UtcNow;
            try
            {
                _child = ChildProcess.Start(exe, new[] { "serve", "--config", Path.Combine(root, "manager.json") }, dir,
                    new Dictionary<string, string> { ["AMQL_ROOT"] = root, ["AMQL_SUPERVISED"] = "1" },
                    lowPriority: false, logPath: Path.Combine(root, "data", "manager.log"));
            }
            catch (Exception e) when (e is ManagerException or System.ComponentModel.Win32Exception or IOException)
            {
                // A version whose manager cannot even start fails as one that exits at once.
                quickFailures++;
                log.LogError("manager {Version} did not start: {Error} ({Failures} running)", version, e.Message, quickFailures);
                if (quickFailures >= 3 && RollBack(version))
                {
                    quickFailures = 0;
                    continue;
                }
                await Task.Delay(pause, stop).ConfigureAwait(false);
                pause = TimeSpan.FromSeconds(Math.Min(60, pause.TotalSeconds * 2));
                continue;
            }
            try
            {
                await _child.Completion.WaitAsync(stop).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            int code = _child.ExitCode ?? -1;
            _child.Dispose();
            _child = null;

            if (code == Updater.RestartExitCode)
            {
                log.LogInformation("manager {Version} asked for a restart (now current: {Current})", version, Layout.ReadCurrent(root));
                quickFailures = 0;
                pause = TimeSpan.FromSeconds(2);
                continue;
            }
            bool quick = DateTimeOffset.UtcNow - started < TimeSpan.FromMinutes(1);
            quickFailures = quick ? quickFailures + 1 : 0;
            log.LogWarning("manager {Version} exited with {Code}{Quick}", version, code, quick ? $" within a minute ({quickFailures} running)" : "");
            if (quickFailures >= 3 && RollBack(version))
            {
                quickFailures = 0;
                continue;
            }
            await Task.Delay(pause, stop).ConfigureAwait(false);
            pause = TimeSpan.FromSeconds(Math.Min(60, pause.TotalSeconds * 2));
        }
    }

    // Versions that kept failing in this run: never rolled back to, so two broken
    // versions cannot swap back and forth.
    private readonly HashSet<string> _failed = new(StringComparer.Ordinal);

    private bool RollBack(string failing)
    {
        _failed.Add(failing);
        string? previous = Layout.ReadPrevious(root);
        if (previous is null || _failed.Contains(previous) || !Directory.Exists(Layout.VersionDir(root, previous)))
        {
            log.LogError("version {Failing} keeps failing and there is no working version to roll back to", failing);
            return false;
        }
        log.LogError("version {Failing} keeps failing: rolling back to {Previous}", failing, previous);
        Layout.MakeCurrent(root, previous);
        return true;
    }

    public override async Task StopAsync(CancellationToken cancel)
    {
        if (_child is { Running: true } child)
        {
            await child.StopAsync(TimeSpan.FromSeconds(20)).ConfigureAwait(false);
        }
        await base.StopAsync(cancel).ConfigureAwait(false);
    }
}
