using System.IO;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Amql.Gui.Explorer;
using Amql.Safetensors;
using Amql.Vindex3;
using Terminal.Gui.Input;
using Terminal.Gui.ViewBase;
using Terminal.Gui.Views;

namespace Amql.Gui.Tui;

/// <summary>
/// Look inside a container: its index, components and layers, objects and
/// edges, the values in any tensor, and — one action away — a run whose trace
/// opens in the Visualiser. The desktop explorer's right-click menu is the
/// action list beside the tree here (and Enter on a node).
/// </summary>
internal sealed class ExplorerPageTui : ITuiPage
{
    private sealed record NodeAction(string Label, Action Run);

    private readonly Shell _shell;
    private readonly ContainerExplorerModel _model = new();
    private readonly View _root = new() { CanFocus = true };
    private readonly Label _path = new() { X = 0, Y = 0, Width = Dim.Fill(56), Height = 1, Text = "No container loaded" };
    private readonly TreeView<ExplorerNode> _tree = new() { X = 0, Y = 2, Width = Dim.Percent(55), Height = Dim.Fill() };
    private readonly Label _detail = new() { X = 0, Y = 0, Width = Dim.Fill(), Height = 5 };
    private readonly ListView _actions = new() { X = 0, Y = 6, Width = Dim.Fill(), Height = Dim.Fill() };
    private readonly ObservableCollection<string> _actionRows = new();
    private List<NodeAction> _nodeActions = new();

    public ExplorerPageTui(Shell shell)
    {
        _shell = shell;
        _root.Add(new Label { Text = "Look inside a container: its objects and tensors, the values in any tensor, and which tensors carry a prompt when you run it.", X = 0, Y = 1, Width = Dim.Fill(), Height = 1 });
        _root.Add(_path);
        _root.Add(TuiKit.Button("Open container…", OpenContainer, Pos.AnchorEnd(56), 0));
        _root.Add(TuiKit.Button("Close", Close, Pos.AnchorEnd(36), 0));
        _root.Add(TuiKit.Button("Open a saved trace…", OpenTrace, Pos.AnchorEnd(26), 0));

        _tree.TreeBuilder = new DelegateTreeBuilder<ExplorerNode>(n => n.Children, n => n.Children.Count > 0 || n.Kind is NodeKind.Object);
        _tree.AspectGetter = n => n.Label;
        _tree.SelectionChanged += (_, e) => OnNodeSelected(e.NewValue);
        _tree.Accepting += (_, e) =>
        {
            e.Handled = true;
            ChooseAction();
        };
        _root.Add(_tree);

        var side = new FrameView { Title = "Selected", X = Pos.Right(_tree) + 1, Y = 2, Width = Dim.Fill(), Height = Dim.Fill() };
        _actions.SetSource(_actionRows);
        _actions.Accepting += (_, e) =>
        {
            e.Handled = true;
            if (_actions.SelectedItem is { } i && i >= 0 && i < _nodeActions.Count)
            {
                TuiKit.Guard(_nodeActions[i].Run);
            }
        };
        side.Add(_detail, new Label { Text = "Actions (Enter)", X = 0, Y = 5 }, _actions);
        _root.Add(side);
        ShowActions(null);
    }

    public View View => _root;

    public void Shown()
    {
    }

    public void Focus() => _tree.SetFocus();

    public void Hidden()
    {
    }

    // ── open / close ────────────────────────────────────────────────────────

    private void OpenContainer()
    {
        if (TuiKit.PickPath("Open VINDEX3 Container Directory", directory: true, _model.ContainerPath) is { } path)
        {
            Open(path);
        }
    }

    /// <summary>Opens a container without asking (from the Server page's model list, say).</summary>
    public void Open(string containerPath)
    {
        try
        {
            _model.Load(containerPath);
        }
        catch (Exception ex)
        {
            TuiKit.Error("Explorer", $"Cannot open the container:\n{ex.Message}");
            return;
        }
        _path.Text = containerPath;
        _tree.ClearObjects();
        _tree.AddObjects(_model.Roots);
        ShowActions(_model.Roots.FirstOrDefault());
    }

    private void Close()
    {
        _model.Unload();
        _tree.ClearObjects();
        _path.Text = "No container loaded";
        ShowActions(null);
    }

    private void OpenTrace()
    {
        if (TuiKit.PickPath("Open an inference trace", directory: false) is { } path)
        {
            _shell.Navigate("visualiser");
            _shell.Visualiser.OpenTrace(path);
        }
    }

    // ── nodes ───────────────────────────────────────────────────────────────

    private void OnNodeSelected(ExplorerNode? node)
    {
        if (node?.Kind == NodeKind.Object && _model.IsLoaded && node.Children.All(c => !c.Id.Contains('/')))
        {
            // Lazy-load: the object's tensor names, when it is first selected.
            foreach (var name in _model.ListTensors(node.Id))
            {
                node.Children.Add(new ExplorerNode($"Tensor: {name}", NodeKind.Property, node.Id + "/" + name));
            }
            _tree.RefreshObject(node);
        }
        ShowActions(node);
    }

    private void ShowActions(ExplorerNode? node)
    {
        var actions = new List<NodeAction>();
        if (_model.IsLoaded)
        {
            actions.Add(new("▶ Run and display inference…", () => _shell.Visualiser.RunAndDisplay(_model.ContainerPath, _shell.Tools.Session.Cli)));
        }
        if (node is not null)
        {
            bool isTensor = node.Kind == NodeKind.Property && node.Id.Contains('/');
            if (node.Kind == NodeKind.Object || isTensor)
            {
                actions.Add(new("View tensor values", () => ViewTensors(node)));
            }
            if (node.Kind == NodeKind.Layer)
            {
                actions.Add(new("View layer tensors", () => ViewLayerTensors(node)));
            }
        }
        if (_model.IsLoaded)
        {
            actions.Add(new("View tokenizer", ViewTokenizer));
            actions.Add(new("View token connections", ViewConnections));
        }
        if (node?.Kind == NodeKind.Layer)
        {
            actions.Add(new("Change to linear_attention", () => TuiKit.Info("Layer Type Conversion",
                "Converting a layer from full_attention to linear_attention requires re-encoding the container with a linear-attention checkpoint (e.g., Qwen3-Next). " +
                "This operation is available via 'amql-cli encode' with a linear-attention model.\n\n" +
                "For containers already encoded with linear_attention, the layer type is detected automatically from the source checkpoint's layer_types table.")));
            actions.Add(new("Change to full_attention", () => TuiKit.Info("Layer Type Conversion",
                "Converting a layer from linear_attention to full_attention requires re-encoding the container with a full-attention checkpoint. " +
                "This operation is available via 'amql-cli encode' with a standard Qwen3.5 model.\n\n" +
                "For containers already encoded with full_attention, the layer type is detected automatically from the source checkpoint's layer_types table.")));
        }
        _nodeActions = actions;
        _actionRows.Clear();
        foreach (var a in actions)
        {
            _actionRows.Add(a.Label);
        }
        _detail.Text = node is null
            ? (_model.IsLoaded ? "Choose a node in the tree." : "Open a container to explore it (or pick one on the Server page and press e).")
            : $"{node.Label}\n{node.Kind}{(node.Id.Length > 0 ? " · " + node.Id : "")}\n{node.Tooltip}";
    }

    private void ChooseAction()
    {
        if (_nodeActions.Count == 0)
        {
            return;
        }
        if (_tree.SelectedObject is { } node && node.Kind == NodeKind.Property && node.Id.Contains('/'))
        {
            // A tensor: Enter shows its values, as clicking it does on the desktop.
            ViewTensors(node);
            return;
        }
        if (TuiKit.Choose(_tree.SelectedObject?.Label ?? "Container", _nodeActions.Select(a => a.Label).ToList()) is { } i)
        {
            TuiKit.Guard(_nodeActions[i].Run);
        }
    }

    private void ViewTensors(ExplorerNode node)
    {
        if (node.Kind == NodeKind.Property && node.Id.Split('/') is [var obj, var tensor])
        {
            ShowTensor(obj, tensor);
        }
        else if (node.Kind == NodeKind.Object && _model.ListTensors(node.Id) is { Count: > 0 } tensors)
        {
            ShowTensor(node.Id, tensors[0]);
        }
    }

    private void ViewLayerTensors(ExplorerNode node)
    {
        if (!node.Id.StartsWith("layer.", StringComparison.Ordinal) || !int.TryParse(node.Id["layer.".Length..], out int layer)
            || _model.Container?.Graph is not { } graph)
        {
            return;
        }
        var stack = graph.Objects.FirstOrDefault(o => o.Kind == ObjectKind.DecoderStack || o.Kind == ObjectKind.EncoderStack);
        if (stack is null)
        {
            return;
        }
        try
        {
            using var store = _model.Container!.CreateOperandStore();
            if (store.SegmentPathFor(stack.Id) is not { } segmentPath)
            {
                return;
            }
            using var segment = SegmentFile.Open(Path.Combine(_model.ContainerPath, segmentPath));
            string prefix = $"{layer}.";
            var names = segment.Header.Tensors.Select(t => t.Name).Where(n => n.StartsWith(prefix, StringComparison.Ordinal)).ToList();
            if (names.Count == 0)
            {
                TuiKit.Info("Layer Tensors", $"No tensors found for layer {layer} in the segment.");
                return;
            }
            foreach (var tn in names)
            {
                string display = tn[prefix.Length..];
                if (!node.Children.Any(c => c.Kind == NodeKind.Property && c.Label == display))
                {
                    node.Children.Add(new ExplorerNode(display, NodeKind.Property, $"{stack.Id}/{tn}"));
                }
            }
            _tree.RefreshObject(node);
            _tree.Expand(node);
            if (TuiKit.Choose($"Layer {layer} tensors", names.Select(n => n[prefix.Length..]).ToList()) is { } i)
            {
                ShowTensor(stack.Id, names[i]);
            }
        }
        catch (Exception ex)
        {
            TuiKit.Error("Layer Tensors", $"Could not list layer tensors: {ex.Message}");
        }
    }

    private void ViewTokenizer()
    {
        var path = Path.Combine(_model.ContainerPath, "tokenizer.json");
        if (!File.Exists(path))
        {
            TuiKit.Info("Tokenizer", "This container has no tokenizer.json.");
            return;
        }
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var vocab = doc.RootElement.TryGetProperty("model", out var model) && model.TryGetProperty("vocab", out var v) ? v : doc.RootElement;
            var sb = new StringBuilder("Tokenizer (vocabulary)\n=====================\n");
            if (vocab.ValueKind == JsonValueKind.Object)
            {
                int count = 0;
                foreach (var prop in vocab.EnumerateObject())
                {
                    if (count++ < 200)
                    {
                        sb.AppendLine($"{prop.Name}: {prop.Value}");
                    }
                }
                if (count > 200)
                {
                    sb.AppendLine($"... ({count - 200} more entries)");
                }
            }
            TuiKit.ShowText("tokenizer.json", sb.ToString());
        }
        catch (Exception ex)
        {
            TuiKit.Error("Tokenizer", $"Failed to read tokenizer: {ex.Message}");
        }
    }

    private void ViewConnections()
    {
        if (_model.Container?.Graph is not { } graph)
        {
            return;
        }
        var component = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.PrimaryText);
        var sb = new StringBuilder($"Hidden-State Edges (component: {component?.Id})\n============================================\n\n");
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
        TuiKit.ShowText("Token Connections", sb.ToString());
    }

    // ── tensor values ───────────────────────────────────────────────────────

    private void ShowTensor(string objectId, string tensorName)
    {
        var data = _model.ReadTensor(objectId, tensorName);
        if (data is null)
        {
            TuiKit.Error("Explorer", $"Could not read tensor '{tensorName}' of '{objectId}'.");
            return;
        }
        var source = new TensorTableSource(data);
        var dialog = new Dialog { Title = $"{objectId} / {tensorName}", Width = Dim.Percent(95), Height = Dim.Percent(90) };
        var info = new Label { X = 0, Y = 0, Width = Dim.Fill(), Height = 1, Text = $"[{string.Join(" × ", data.Shape)}]  {data.Dtype.Label()}" };
        var filterLabel = new Label { Text = "Show rows with |value| >", X = 0, Y = 1 };
        var filter = new TextField { X = 26, Y = 1, Width = 12 };
        var stats = new Label { X = 40, Y = 1, Width = Dim.Fill(), Height = 1 };
        var table = new TableView { X = 0, Y = 2, Width = Dim.Fill(), Height = Dim.Fill(1), Table = source, FullRowSelect = false };
        var cell = new Label { X = 0, Y = Pos.AnchorEnd(1), Width = Dim.Fill(), Height = 1 };
        void Update()
        {
            stats.Text = source.Summary();
            if (table.Value is { } sel)
            {
                cell.Text = source.Describe(sel.SelectedCell.Y, sel.SelectedCell.X);
            }
        }
        filter.TextChanged += (_, _) =>
        {
            source.Filter(float.TryParse(filter.Text.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var t) ? t : null);
            table.Update();
            Update();
        };
        table.ValueChanged += (_, _) => Update();
        var copy = new Button { ShadowStyle = ShadowStyles.None, Text = "Cop_y cell" };
        copy.Accepting += (_, e) =>
        {
            e.Handled = true;
            if (table.Value is { } sel)
            {
                TuiKit.Copy(source.ValueText(sel.SelectedCell.Y, sel.SelectedCell.X));
            }
        };
        dialog.Add(info, filterLabel, filter, stats, table, cell);
        dialog.AddButton(copy);
        dialog.AddButton(new Button { ShadowStyle = ShadowStyles.None, Text = "_Close" });
        Update();
        table.SetFocus();
        TuiKit.App.Run(dialog);
        dialog.Dispose();
    }

    /// <summary>
    /// A tensor as a table: one row of values per row of a 2-D tensor (at most
    /// 256 columns, as on the desktop), or index and value for any other shape.
    /// The filter keeps rows holding a value beyond ±threshold.
    /// </summary>
    private sealed class TensorTableSource : ITableSource
    {
        private const int MaxColumns = 256;
        private readonly TensorData _data;
        private readonly int _rows, _cols;
        private readonly bool _matrix;
        private List<int> _visible;

        public TensorTableSource(TensorData data)
        {
            _data = data;
            _matrix = data.Shape.Length == 2;
            _rows = _matrix ? (int)data.Shape[0] : data.Values.Length;
            _cols = _matrix ? (int)data.Shape[1] : 1;
            _visible = Enumerable.Range(0, _rows).ToList();
            ColumnNames = _matrix
                ? new[] { "Row" }.Concat(Enumerable.Range(0, Math.Min(_cols, MaxColumns)).Select(c => $"C{c}")).ToArray()
                : new[] { "#", "Value" };
        }

        public string[] ColumnNames { get; }

        public int Columns => ColumnNames.Length;

        public int Rows => _visible.Count;

        public object this[int row, int col] => col == 0 ? _visible[row] : ValueText(row, col);

        private float? Value(int row, int col)
        {
            if (row < 0 || row >= _visible.Count || col < 1)
            {
                return null;
            }
            int r = _visible[row];
            return _matrix ? _data.Values[(long)r * _cols + col - 1] : _data.Values[r];
        }

        public string ValueText(int row, int col) =>
            Value(row, col) is { } v ? v.ToString(_matrix ? "F4" : "F6", CultureInfo.InvariantCulture) : "";

        public void Filter(float? threshold)
        {
            _visible = threshold is not { } t
                ? Enumerable.Range(0, _rows).ToList()
                : Enumerable.Range(0, _rows).Where(r => Enumerable.Range(0, _cols).Any(c =>
                    Math.Abs(_matrix ? _data.Values[(long)r * _cols + c] : _data.Values[r]) > t)).ToList();
        }

        public string Summary() =>
            $"{_visible.Count} of {_rows} rows" + (_matrix && _cols > MaxColumns ? $" · showing {MaxColumns} of {_cols} columns" : "");

        public string Describe(int row, int col) =>
            row < 0 || row >= _visible.Count ? "" : col == 0 ? $"row {_visible[row]}" : $"R{_visible[row]} C{col - 1}: {ValueText(row, col)}";
    }
}
