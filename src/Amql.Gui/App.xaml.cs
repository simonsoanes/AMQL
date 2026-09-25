using System.IO;
using System.Windows;
using System.Windows.Threading;
using Amql.Gui.Visualiser;

namespace Amql.Gui;

public partial class App : Application
{
    /// <summary>
    /// <c>amql-gui --trace &lt;file.json&gt;</c> opens the inference visualiser
    /// straight onto a captured run, so a generate command can hand its trace
    /// over without the user having to browse for the file a second time.
    /// </summary>
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        string? tracePath = null;
        for (int i = 0; i < e.Args.Length - 1; i++)
        {
            if (string.Equals(e.Args[i], "--trace", StringComparison.OrdinalIgnoreCase)
                || string.Equals(e.Args[i], "--trace-json", StringComparison.OrdinalIgnoreCase))
            {
                tracePath = e.Args[i + 1];
                break;
            }
        }
        if (tracePath is null)
        {
            return;
        }
        if (!File.Exists(tracePath))
        {
            MessageBox.Show($"Trace file not found:\n{tracePath}", "AMQL Studio",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        // Deferred until after StartupUri has built the main window, so the
        // visualiser can be owned by it and sit beside the container explorer.
        Dispatcher.BeginInvoke(new Action(() =>
        {
            var window = new InferenceVisualiserWindow(tracePath);
            if (MainWindow is not null)
            {
                window.Owner = MainWindow;
            }
            window.Show();
        }), DispatcherPriority.Loaded);
    }
}
