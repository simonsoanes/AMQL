using System.IO;
using System.Reflection;
using Amql.Gui.Services;
using Terminal.Gui.App;
using Terminal.Gui.Input;

namespace Amql.Gui.Tui;

/// <summary>
/// Runs AMQL Studio's text UI: the same pages as the desktop window (Home,
/// Server, Playground, Tools, Explorer) plus the Monitor and the inference
/// Visualiser, which are windows of their own on the desktop and pages here.
/// </summary>
public static class TuiApp
{
    public static string Version =>
        Assembly.GetExecutingAssembly().GetName().Version is { } v ? v.ToString(3) : "";

    public static int Run(string[] args)
    {
        string? page = Option(args, "--page");
        string? trace = Option(args, "--trace") ?? Option(args, "--trace-json");
        if (Option(args, "--tui-snapshot") is { } snapshotDir)
        {
            return Snapshot(snapshotDir, args);
        }

        if (!OperatingSystem.IsWindows() && (Console.IsInputRedirected || Console.IsOutputRedirected))
        {
            // Terminal.Gui waits for the terminal to answer its start-up queries, so without one it would hang.
            Console.Error.WriteLine("AMQL Studio's text UI needs a terminal: run it in an interactive shell (ssh -t, tmux, a console).");
            return 2;
        }
        Configure();
        using IApplication app = Application.Create();
        app.Init();
        TuiKit.Attach(app);
        UiThread.Install(() => app.MainThreadId is not { } id || id == Environment.CurrentManagedThreadId, a => app.Invoke(a));

        var shell = new Shell(page, trace);
        if (Environment.GetEnvironmentVariable("AMQL_TUI_TRACE") is { Length: > 0 })
        {
            // Where focus is, for diagnosing keyboard navigation.
            app.Navigation!.FocusedChanged += (_, _) =>
            {
                var chain = new List<string>();
                for (var v = app.Navigation.GetFocused(); v is not null; v = v.SuperView)
                {
                    chain.Add(v.GetType().Name + (string.IsNullOrEmpty(v.Title) ? "" : $"({v.Title})"));
                }
                Console.Error.WriteLine("[focus] " + string.Join(" < ", chain));
            };
            app.Keyboard.KeyDown += (_, key) => Console.Error.WriteLine($"[key] app {key} handled={key.Handled}");
        }
        try
        {
            app.Run(shell);
        }
        finally
        {
            shell.Closing();
            shell.Dispose();
        }
        return 0;
    }

    /// <summary>Process-wide settings for the text UI (none needed yet; buttons are made without drop shadows where they are created).</summary>
    private static void Configure()
    {
    }

    private static void Step(string what)
    {
        if (Environment.GetEnvironmentVariable("AMQL_TUI_TRACE") is { Length: > 0 })
        {
            Console.Error.WriteLine($"[tui] {DateTime.Now:HH:mm:ss.fff} {what}");
        }
    }

    private static string? Option(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
            {
                return args[i + 1];
            }
        }
        return null;
    }

    /// <summary>
    /// <c>--tui-snapshot &lt;dir&gt;</c>: draws every page off-screen at a fixed
    /// size and writes each as a text file, then exits. A check that the text UI
    /// builds and lays out on a machine (a fresh Spark, CI) without anyone at a
    /// terminal; <c>--size 120x36</c> chooses the screen size.
    /// </summary>
    private static int Snapshot(string dir, string[] args)
    {
        int cols = 120, rows = 36;
        if (Option(args, "--size") is { } size && size.Split('x') is [var c, var r]
            && int.TryParse(c, out var cc) && int.TryParse(r, out var rr))
        {
            cols = cc;
            rows = rr;
        }
        Directory.CreateDirectory(dir);
        Configure();
        Step("create");
        using IApplication app = Application.Create();
        Step("init");
        app.Init("ansi");
        app.Driver!.SetScreenSize(cols, rows);
        TuiKit.Attach(app);
        UiThread.Install(() => true, a => a());
        Step("shell");
        var shell = new Shell(null, Option(args, "--trace"), snapshot: true);
        Step("begin");
        var token = app.Begin(shell);
        int failures = 0;
        foreach (var key in Shell.PageKeys)
        {
            try
            {
                Step(key);
                shell.Navigate(key);
                // Let page set-up that awaits (model scans, nvidia-smi) land.
                for (int i = 0; i < 20; i++)
                {
                    Thread.Sleep(50);
                    app.LayoutAndDraw(true);
                }
                File.WriteAllText(Path.Combine(dir, $"{key}.txt"), app.Driver.ToString());
                Console.WriteLine($"{key}: ok");
            }
            catch (Exception e)
            {
                failures++;
                Console.WriteLine($"{key}: {e}");
            }
        }
        app.End(token!);
        shell.Dispose();
        return failures == 0 ? 0 : 1;
    }
}
