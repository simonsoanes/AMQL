using System.Windows;
using System.Windows.Controls;
using Amql.Gui.Explorer;
using Amql.Gui.Visualiser;
using Microsoft.Win32;

namespace Amql.Gui.Pages;

/// <summary>The container explorer, with the inference visualiser one click away.</summary>
public sealed class ExplorerPage : UserControl
{
    private readonly MainWindow _shell;
    private readonly ContainerExplorer _explorer = new();

    public ExplorerPage(MainWindow shell)
    {
        _shell = shell;
        var root = new DockPanel { Margin = new Thickness(16, 16, 16, 12) };

        var header = new DockPanel { Margin = new Thickness(0, 0, 0, 12) };
        var openTrace = new Button { Content = "Open a saved trace…", ToolTip = "Open a captured operator trace (--trace-json) in the inference visualiser" };
        openTrace.Click += (_, _) => OpenTrace();
        DockPanel.SetDock(openTrace, Dock.Right);
        header.Children.Add(openTrace);
        var titles = new StackPanel();
        titles.Children.Add(new TextBlock { Text = "Explorer", Style = (Style)FindResource("PageTitle") });
        titles.Children.Add(new TextBlock
        {
            Text = "Look inside a container: its objects and tensors, the values in any tensor, and which tensors carry a prompt when you run it.",
            Style = (Style)FindResource("PageSubtitle"),
            Margin = new Thickness(0),
        });
        header.Children.Add(titles);
        DockPanel.SetDock(header, Dock.Top);
        root.Children.Add(header);

        _explorer.TensorSelected += OnTensorSelected;
        _explorer.RunInferenceRequested += path =>
            InferenceVisualiserWindow.RunAndDisplay(Window.GetWindow(this), path, _shell.Tools.Cli);
        root.Children.Add(new Border { Style = (Style)FindResource("Card"), Padding = new Thickness(0), Child = _explorer });
        Content = root;
    }

    /// <summary>Opens a container (from the Server page's model list, say).</summary>
    public void Open(string containerPath) => _explorer.Open(containerPath);

    private void OpenTrace()
    {
        var dlg = new OpenFileDialog { Title = "Open an inference trace", Filter = "Inference trace (*.json)|*.json|All files (*.*)|*.*" };
        if (dlg.ShowDialog(Window.GetWindow(this)) != true) return;
        var window = new InferenceVisualiserWindow(dlg.FileName) { Owner = Window.GetWindow(this) };
        window.SetCliSettings(_shell.Tools.Cli);
        window.Show();
    }

    private void OnTensorSelected(string objectId, string tensorName)
    {
        var data = _explorer.Model.ReadTensor(objectId, tensorName);
        if (data is null)
        {
            MessageBox.Show(Window.GetWindow(this), $"Could not read tensor '{tensorName}' of '{objectId}'.", "Explorer",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        new TensorGridDialog(objectId, tensorName, data) { Owner = Window.GetWindow(this) }.ShowDialog();
    }
}
