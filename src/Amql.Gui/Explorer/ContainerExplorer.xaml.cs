using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using Microsoft.Win32;

namespace Amql.Gui.Explorer;

public partial class ContainerExplorer : UserControl
{
    private readonly ContainerExplorerModel _model = new();

    public ContainerExplorer()
    {
        InitializeComponent();
        DataContext = _model;
    }

    public ContainerExplorerModel Model => _model;

    public event Action<string, string>? TensorSelected;

    private void OnOpenContainer(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog
        {
            Title = "Open VINDEX3 Container Directory",
        };
        if (dlg.ShowDialog() != true) return;

        _model.Load(dlg.FolderName);
        PathText.Text = dlg.FolderName;
    }

    private void OnCloseContainer(object sender, RoutedEventArgs e)
    {
        _model.Unload();
        PathText.Text = "No container loaded";
    }

    private void OnNodeSelected(object sender, RoutedPropertyChangedEventArgs<object> e)
    {
        if (e.NewValue is not ExplorerNode node) return;

        if (node.Kind == NodeKind.Object)
        {
            // Drill into tensor names for this object
            if (_model.IsLoaded)
            {
                var tensors = _model.ListTensors(node.Id);
                foreach (var name in tensors)
                {
                    if (!node.Children.Any(c => c.Label == name && c.Kind == NodeKind.Property))
                    {
                        node.Children.Add(new ExplorerNode($"Tensor: {name}", NodeKind.Property, node.Id + "/" + name));
                    }
                }
            }
        }
        else if (node.Kind == NodeKind.Property && node.Id.Contains("/"))
        {
            // A tensor node — notify parent to show grid
            var parts = node.Id.Split('/');
            if (parts.Length == 2)
            {
                TensorSelected?.Invoke(parts[0], parts[1]);
            }
        }
    }
}