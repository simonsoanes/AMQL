using System.Globalization;
using System.Windows;
using Microsoft.Win32;

namespace Amql.Gui.Visualiser;

/// <summary>
/// Collects the parameters for a live run. The visualiser drives the CLI as a
/// child process rather than hosting inference here, so the model never loads
/// into the GUI: a 27B run needs its memory in the child, cancellation kills a
/// process instead of unwinding a thread inside the UI, and a crash in the
/// runtime cannot take the window with it.
/// </summary>
public sealed partial class RunInferenceDialog : Window
{
    public RunInferenceDialog(string containerPath, string prompt)
    {
        InitializeComponent();
        ContainerBox.Text = containerPath;
        PromptBox.Text = prompt;
    }

    public string ContainerPath => ContainerBox.Text.Trim();
    public string Prompt => PromptBox.Text;
    public int Steps { get; private set; } = 8;
    public float Temperature { get; private set; }
    public bool LogitLens => LensCheck.IsChecked == true;
    public bool TraceAttention => AttentionCheck.IsChecked == true;
    public bool AutoFit => AutoFitCheck.IsChecked == true;

    private void OnBrowse(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog { Title = "Open VINDEX3 Container" };
        if (dialog.ShowDialog(this) == true)
        {
            ContainerBox.Text = dialog.FolderName;
        }
    }

    private void OnRun(object sender, RoutedEventArgs e)
    {
        if (ContainerPath.Length == 0)
        {
            Reject("A container directory is required.");
            return;
        }
        if (Prompt.Length == 0)
        {
            Reject("A prompt is required — there is nothing to trace without one.");
            return;
        }
        if (!int.TryParse(StepsBox.Text.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int steps)
            || steps < 1)
        {
            Reject("Steps must be a whole number of at least 1.");
            return;
        }
        if (!float.TryParse(TempBox.Text.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out float temp)
            || temp < 0f)
        {
            Reject("Temperature must be a number, 0 or greater. Use 0 for greedy decoding.");
            return;
        }

        Steps = steps;
        Temperature = temp;
        DialogResult = true;
    }

    private void Reject(string message)
        => MessageBox.Show(this, message, "Run Inference", MessageBoxButton.OK, MessageBoxImage.Warning);
}
