using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Amql.Vindex3;
using Microsoft.Win32;

namespace Amql.Gui.Explorer;

public partial class ContainerExplorer : UserControl
{
    private readonly ContainerExplorerModel _model = new();
    private ExplorerNode? _rightClickedNode;

    public ContainerExplorer()
    {
        InitializeComponent();
        DataContext = _model;
        BuildContextMenu();
    }

    public ContainerExplorerModel Model => _model;

    public event Action<string, string>? TensorSelected;

    // ── context menu ────────────────────────────────────────────────────

    private ContextMenu _contextMenu = null!;
    private MenuItem _viewTensorsItem = null!;
    private MenuItem _viewTokenizerItem = null!;
    private MenuItem _viewConnectionsItem = null!;
    private MenuItem _viewLayerTensorsItem = null!;
    private Separator _sepItem = null!;
    private MenuItem _toLinearItem = null!;
    private MenuItem _toFullItem = null!;

    private void BuildContextMenu()
    {
        _contextMenu = new ContextMenu();
        _viewTensorsItem = new MenuItem { Header = "View Tensor Values" };
        _viewTensorsItem.Click += OnViewTensors;
        _contextMenu.Items.Add(_viewTensorsItem);

        _viewLayerTensorsItem = new MenuItem { Header = "View Layer Tensors" };
        _viewLayerTensorsItem.Click += OnViewLayerTensors;
        _contextMenu.Items.Add(_viewLayerTensorsItem);

        _viewTokenizerItem = new MenuItem { Header = "View Tokenizer" };
        _viewTokenizerItem.Click += OnViewTokenizer;
        _contextMenu.Items.Add(_viewTokenizerItem);

        _viewConnectionsItem = new MenuItem { Header = "View Token Connections" };
        _viewConnectionsItem.Click += OnViewConnections;
        _contextMenu.Items.Add(_viewConnectionsItem);

        _sepItem = new Separator();
        _contextMenu.Items.Add(_sepItem);

        _toLinearItem = new MenuItem { Header = "Change to linear_attention" };
        _toLinearItem.Click += OnChangeToLinear;
        _contextMenu.Items.Add(_toLinearItem);

        _toFullItem = new MenuItem { Header = "Change to full_attention" };
        _toFullItem.Click += OnChangeToFull;
        _contextMenu.Items.Add(_toFullItem);
    }

    private void OnNodeRightClick(object sender, MouseButtonEventArgs e)
    {
        if (e.OriginalSource is not FrameworkElement fe) return;
        if (fe.DataContext is not ExplorerNode node) return;

        _rightClickedNode = node;

        // Update visibility based on node kind
        bool isObject = node.Kind == NodeKind.Object;
        bool isTensor = node.Kind == NodeKind.Property && node.Id.Contains("/");
        bool isLayer = node.Kind == NodeKind.Layer;
        bool isLayerObject = isLayer || (
            node.Kind == NodeKind.Property &&
            node.Id.StartsWith("layer.", StringComparison.Ordinal));

        _viewTensorsItem.Visibility = (isObject || isTensor) ? Visibility.Visible : Visibility.Collapsed;
        _viewLayerTensorsItem.Visibility = isLayer ? Visibility.Visible : Visibility.Collapsed;
        _viewTokenizerItem.Visibility = _model.IsLoaded ? Visibility.Visible : Visibility.Collapsed;
        _viewConnectionsItem.Visibility = _model.IsLoaded ? Visibility.Visible : Visibility.Collapsed;
        _sepItem.Visibility = isLayer ? Visibility.Visible : Visibility.Collapsed;
        _toLinearItem.Visibility = isLayer ? Visibility.Visible : Visibility.Collapsed;
        _toFullItem.Visibility = isLayer ? Visibility.Visible : Visibility.Collapsed;

        _contextMenu.IsOpen = true;
        e.Handled = true;
    }

    // ── context menu handlers ───────────────────────────────────────────

    private void OnViewTensors(object sender, RoutedEventArgs e)
    {
        if (_rightClickedNode is null) return;

        if (_rightClickedNode.Kind == NodeKind.Property && _rightClickedNode.Id.Contains("/"))
        {
            var parts = _rightClickedNode.Id.Split('/');
            if (parts.Length == 2)
            {
                TensorSelected?.Invoke(parts[0], parts[1]);
            }
        }
        else if (_rightClickedNode.Kind == NodeKind.Object)
        {
            // Show the first tensor of this object
            var tensors = _model.ListTensors(_rightClickedNode.Id);
            if (tensors.Count > 0)
            {
                TensorSelected?.Invoke(_rightClickedNode.Id, tensors[0]);
            }
        }
    }

    private void OnViewLayerTensors(object sender, RoutedEventArgs e)
    {
        if (_rightClickedNode?.Kind != NodeKind.Layer) return;
        // The layer's parent is the component; we need to find the decoder/encoder
        // stack object. For now: show the decoder stack's first tensor.
        if (_model.Container?.Graph is { } graph)
        {
            var decoderObj = graph.Objects.FirstOrDefault(o =>
                o.Kind == ObjectKind.DecoderStack || o.Kind == ObjectKind.EncoderStack);
            if (decoderObj is not null)
            {
                TensorSelected?.Invoke(decoderObj.Id, "weight");
            }
        }
    }

    private void OnViewTokenizer(object sender, RoutedEventArgs e)
    {
        if (!_model.IsLoaded) return;
        var tokenizerPath = Path.Combine(_model.ContainerPath, "tokenizer.json");
        if (!File.Exists(tokenizerPath))
        {
            MessageBox.Show("This container has no tokenizer.json.", "Tokenizer", MessageBoxButton.OK,
                MessageBoxImage.Information);
            return;
        }
        try
        {
            var json = File.ReadAllText(tokenizerPath);
            var doc = JsonDocument.Parse(json);
            var vocab = doc.RootElement.TryGetProperty("model", out var model) &&
                        model.TryGetProperty("vocab", out var v)
                ? v : doc.RootElement;

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("Tokenizer (vocabulary)");
            sb.AppendLine("=====================");
            if (vocab.ValueKind == JsonValueKind.Object)
            {
                foreach (var prop in vocab.EnumerateObject().Take(200))
                {
                    sb.AppendLine($"{prop.Name}: {prop.Value}");
                }
                if (vocab.EnumerateObject().Count() > 200)
                {
                    sb.AppendLine($"... ({vocab.EnumerateObject().Count() - 200} more entries)");
                }
            }
            var dialog = new TensorViewDialog("tokenizer.json", sb.ToString());
            dialog.Owner = Window.GetWindow(this);
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Failed to read tokenizer: {ex.Message}", "Tokenizer",
                MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void OnViewConnections(object sender, RoutedEventArgs e)
    {
        if (!_model.IsLoaded || _model.Container?.Graph is not { } graph) return;
        var component = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.PrimaryText);
        if (component is null) return;

        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Hidden-State Edges (component: {component.Id})");
        sb.AppendLine("============================================");
        sb.AppendLine();

        foreach (var edge in graph.Edges)
        {
            sb.AppendLine($"Producer: {edge.ProducerComponent}");
            sb.AppendLine($"  Layers: [{string.Join(", ", edge.ProducerLayers)}]");
            sb.AppendLine($"  Block size: {edge.BlockSize?.ToString() ?? "none"}");
            sb.AppendLine($"  → {edge.ConsumerComponent}.{edge.ConsumerObject}");
            sb.AppendLine();
        }

        if (graph.Edges.Count == 0)
        {
            sb.AppendLine("(no hidden-state edges in this container)");
        }

        var dialog = new TensorViewDialog("Token Connections", sb.ToString());
        dialog.Owner = Window.GetWindow(this);
    }

    private void OnChangeToLinear(object sender, RoutedEventArgs e)
    {
        if (_rightClickedNode?.Kind != NodeKind.Layer) return;
        MessageBox.Show(
            "Converting a layer from full_attention to linear_attention requires " +
            "re-encoding the container with a linear-attention checkpoint (e.g., Qwen3-Next). " +
            "This operation is available via 'amql-cli encode' with a linear-attention model.\n\n" +
            "For containers already encoded with linear_attention, the layer type is detected " +
            "automatically from the source checkpoint's layer_types table.",
            "Layer Type Conversion",
            MessageBoxButton.OK, MessageBoxImage.Information);
    }

    private void OnChangeToFull(object sender, RoutedEventArgs e)
    {
        if (_rightClickedNode?.Kind != NodeKind.Layer) return;
        MessageBox.Show(
            "Converting a layer from linear_attention to full_attention requires " +
            "re-encoding the container with a full-attention checkpoint. " +
            "This operation is available via 'amql-cli encode' with a standard Qwen3.5 model.\n\n" +
            "For containers already encoded with full_attention, the layer type is detected " +
            "automatically from the source checkpoint's layer_types table.",
            "Layer Type Conversion",
            MessageBoxButton.OK, MessageBoxImage.Information);
    }

    // ── open / close ────────────────────────────────────────────────────

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

    // ── node selection (lazy-loaded tensor viewer) ──────────────────────

    private void OnNodeSelected(object sender, RoutedPropertyChangedEventArgs<object> e)
    {
        if (e.NewValue is not ExplorerNode node) return;

        if (node.Kind == NodeKind.Object)
        {
            // Lazy-load: drill into tensor names for this object only when clicked
            if (_model.IsLoaded && node.Children.Count == 0)
            {
                var tensors = _model.ListTensors(node.Id);
                foreach (var name in tensors)
                {
                    node.Children.Add(new ExplorerNode($"Tensor: {name}",
                        NodeKind.Property, node.Id + "/" + name));
                }
            }
        }
        else if (node.Kind == NodeKind.Layer)
        {
            // Lazy-load: populate layer tensor nodes on first click
            if (_model.IsLoaded && node.Children.Count == 0)
            {
                // The layer node already has its children populated during build
                // (operator, span, position, geometry). No additional tensor nodes
                // here — tensors are accessed via the parent Object node.
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

/// <summary>Minimal text viewer dialog for tokenizer and connections.</summary>
public class TensorViewDialog : Window
{
    public TensorViewDialog(string title, string content)
    {
        Title = title;
        Width = 700;
        Height = 500;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        WindowStyle = WindowStyle.ToolWindow;

        var textBox = new TextBox
        {
            Text = content,
            IsReadOnly = true,
            FontFamily = new System.Windows.Media.FontFamily("Cascadia Mono, Consolas"),
            FontSize = 12,
            VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
            HorizontalScrollBarVisibility = ScrollBarVisibility.Auto,
            TextWrapping = TextWrapping.NoWrap,
        };

        var dock = new DockPanel();
        var header = new Border
        {
            Padding = new Thickness(8),
            Child = new TextBlock { Text = title, FontWeight = FontWeights.Bold },
        };
        DockPanel.SetDock(header, Dock.Top);
        dock.Children.Add(header);
        dock.Children.Add(textBox);
        Content = dock;

        ShowDialog();
    }
}