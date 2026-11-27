using System.Runtime.InteropServices;
using Amql.Gui.Tui;

namespace Amql.Gui;

/// <summary>
/// Chooses the front end. The Windows build opens the WPF window unless asked
/// for the text UI (<c>--tui</c>, or AMQL_STUDIO_UI=tui); the Linux build has no
/// window to open, so it always runs the text UI, which has the same screens
/// and options drawn in a terminal (over SSH on a DGX Spark, say).
/// </summary>
public static class Program
{
    [STAThread]
    public static int Main(string[] args)
    {
        if (args.Any(a => a is "--help" or "-h" or "-?" or "/?"))
        {
            EnsureConsole(outputOnly: true);
            Console.WriteLine(Usage);
            return 0;
        }

        bool snapshot = args.Contains("--tui-snapshot", StringComparer.OrdinalIgnoreCase);
        bool wantTui = snapshot || args.Contains("--tui", StringComparer.OrdinalIgnoreCase)
                       || string.Equals(Environment.GetEnvironmentVariable("AMQL_STUDIO_UI"), "tui", StringComparison.OrdinalIgnoreCase);
        bool wantGui = args.Contains("--gui", StringComparer.OrdinalIgnoreCase);

#if AMQL_DESKTOP
        if (!wantTui || wantGui)
        {
            var app = new App();
            Services.UiThread.Install(() => app.Dispatcher.CheckAccess(), a => app.Dispatcher.BeginInvoke(a));
            app.InitializeComponent();
            return app.Run();
        }
        EnsureConsole(outputOnly: snapshot);
#else
        if (wantGui)
        {
            Console.Error.WriteLine("This build of AMQL Studio has no desktop window (the WPF window is Windows-only); running the text UI.");
        }
#endif
        return TuiApp.Run(args);
    }

    private const string Usage =
        """
        AMQL Studio

          amql-gui                     the desktop window on Windows; the text UI on Linux
          amql-gui --tui               the text UI (on Windows it opens in a console window of its own)
          amql-gui --trace <file>      open the inference visualiser on a captured trace
          amql-gui --page <name>       text UI: start on a page (home, server, chat, tools, explorer, monitor, visualiser)
          amql-gui --tui-snapshot <dir> [--size 120x36]
                                       draw every text-UI page off-screen into <dir>/<page>.txt and exit
                                       (a check that it runs on a machine; on Linux it still needs a terminal)

        AMQL_STUDIO_UI=tui chooses the text UI without the switch.
        In the text UI: F1 help, Alt+1…7 or the list on the left change page, F2 starts or stops
        the server, F5 runs the Tools command, F7 shows the Monitor,
        Ctrl+PgUp/PgDn change tab, Ctrl+Q quits.
        """;

    /// <summary>
    /// The Windows build is a GUI-subsystem program, so it starts without a
    /// console. The text UI needs one: a new console window. (Attaching to the
    /// parent's would share it with a shell that has already gone back to
    /// reading the keyboard, since it does not wait for a GUI program.)
    /// </summary>
    private static void EnsureConsole(bool outputOnly = false)
    {
        if (!OperatingSystem.IsWindows() || GetConsoleWindow() != IntPtr.Zero)
        {
            return;
        }
        // Printing the usage is fine in the parent's console; an interactive UI is not.
        if (outputOnly && AttachConsole(-1))
        {
            return;
        }
        AllocConsole();
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool AttachConsole(int processId);

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetConsoleWindow();

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool AllocConsole();
}
