using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Gui.Explorer;

/// <summary>Wraps a Vindex3Container for the explorer tree, building the
/// drill-down hierarchy on demand.</summary>
public sealed class ContainerExplorerModel : INotifyPropertyChanged
{
    private Vindex3Container? _container;
    private string _containerPath = string.Empty;

    public string ContainerPath
    {
        get => _containerPath;
        private set { _containerPath = value; OnPropertyChanged(); }
    }

    public ObservableCollection<ExplorerNode> Roots { get; } = new();

    public bool IsLoaded => _container is not null;

    public void Load(string containerPath)
    {
        _container?.Dispose();
        _container = Vindex3Container.Open(containerPath);
        ContainerPath = containerPath;
        Roots.Clear();
        BuildTree();
        OnPropertyChanged(nameof(IsLoaded));
    }

    public void Unload()
    {
        _container?.Dispose();
        _container = null;
        ContainerPath = string.Empty;
        Roots.Clear();
        OnPropertyChanged(nameof(IsLoaded));
    }

    public Vindex3Container? Container => _container;

    private void BuildTree()
    {
        if (_container?.Graph is not { } graph) return;

        var indexNode = new ExplorerNode("Index", NodeKind.Index, "index.json")
        {
            Tooltip = $"Model: {_container.Index.Model}, Family: {_container.Index.Family}, " +
                      $"Hidden: {_container.Index.HiddenSize}, Layers: {_container.Index.NumLayers}",
        };
        indexNode.Children.Add(new ExplorerNode($"Model: {_container.Index.Model}", NodeKind.Property, ""));
        indexNode.Children.Add(new ExplorerNode($"Family: {_container.Index.Family}", NodeKind.Property, ""));
        indexNode.Children.Add(new ExplorerNode($"Hidden Size: {_container.Index.HiddenSize}", NodeKind.Property, ""));
        indexNode.Children.Add(new ExplorerNode($"Num Layers: {_container.Index.NumLayers}", NodeKind.Property, ""));
        indexNode.Children.Add(new ExplorerNode($"Authority: {_container.Index.Authority}", NodeKind.Property, ""));
        if (_container.Index.DerivedFromModel is { } derived)
        {
            indexNode.Children.Add(new ExplorerNode($"Derived From: {derived}", NodeKind.Property, ""));
        }
        Roots.Add(indexNode);

        foreach (var component in graph.Components)
        {
            var compNode = new ExplorerNode($"{component.Role}: {component.Id}",
                NodeKind.Component, component.Id)
            {
                Tooltip = $"Hidden: {component.HiddenSize}, Layers: {component.NumLayers}, " +
                          $"Source: {component.SourceArtifact}",
            };
            compNode.Children.Add(new ExplorerNode($"Hidden Size: {component.HiddenSize}", NodeKind.Property, ""));
            compNode.Children.Add(new ExplorerNode($"Num Layers: {component.NumLayers}", NodeKind.Property, ""));
            compNode.Children.Add(new ExplorerNode($"Source Artifact: {component.SourceArtifact}", NodeKind.Property, ""));

            if (component.Attention is { } policies)
            {
                for (int l = 0; l < component.NumLayers; l++)
                {
                    var policy = policies[l];
                    var layerNode = new ExplorerNode($"Layer {l}: {policy.Operator}",
                        NodeKind.Layer, $"layer.{l}")
                    {
                        Tooltip = $"Operator: {policy.Operator}, Span: {policy.Span}",
                    };
                    layerNode.Children.Add(new ExplorerNode($"Operator: {policy.Operator}", NodeKind.Property, ""));
                    layerNode.Children.Add(new ExplorerNode($"Span: {policy.Span}", NodeKind.Property, ""));
                    if (policy.Position is { } pos)
                    {
                        layerNode.Children.Add(new ExplorerNode($"Position: {pos}", NodeKind.Property, ""));
                    }
                    if (policy.Geometry is { } geom)
                    {
                        layerNode.Children.Add(new ExplorerNode(
                            $"Heads: KV={geom.NumKvHeads} Dim={geom.HeadDim}",
                            NodeKind.Property, ""));
                    }
                    compNode.Children.Add(layerNode);
                }
            }

            Roots.Add(compNode);
        }

        // Objects
        foreach (var obj in graph.Objects)
        {
            var objNode = new ExplorerNode($"{obj.Kind}: {obj.Id}", NodeKind.Object, obj.Id)
            {
                Tooltip = $"Component: {obj.Component}, Representations: {obj.Representations.Count}",
            };
            objNode.Children.Add(new ExplorerNode($"Component: {obj.Component}", NodeKind.Property, ""));
            objNode.Children.Add(new ExplorerNode($"Representations: {obj.Representations.Count}", NodeKind.Property, ""));
            foreach (var rep in obj.Representations)
            {
                objNode.Children.Add(new ExplorerNode($"{rep.Encoding} ({rep.Fidelity})", NodeKind.Property, ""));
            }
            if (obj.SourceBindings is { } bindings)
            {
                foreach (var binding in bindings)
                {
                    objNode.Children.Add(new ExplorerNode(
                        $"Source: {binding.Artifact} ({binding.Tensors} tensors, {binding.TensorPrefix})",
                        NodeKind.Property, ""));
                }
            }
            Roots.Add(objNode);
        }

        // Edges (connections)
        foreach (var edge in graph.Edges)
        {
            var edgeNode = new ExplorerNode(
                $"{edge.ProducerComponent} → {edge.ConsumerComponent}.{edge.ConsumerObject}",
                NodeKind.Edge, $"{edge.ProducerComponent}>{edge.ConsumerComponent}")
            {
                Tooltip = $"Producer layers: [{string.Join(",", edge.ProducerLayers)}]",
            };
            edgeNode.Children.Add(new ExplorerNode(
                $"Producer layers: [{string.Join(",", edge.ProducerLayers)}]", NodeKind.Property, ""));
            edgeNode.Children.Add(new ExplorerNode($"Block size: {edge.BlockSize?.ToString() ?? "none"}", NodeKind.Property, ""));
            Roots.Add(edgeNode);
        }
    }

    /// <summary>Reads a tensor's values for display in the grid dialog.
    /// Returns null if the tensor can't be resolved.</summary>
    public TensorData? ReadTensor(string objectId, string tensorName)
    {
        if (_container is null) return null;
        using var store = _container.CreateOperandStore();
        try
        {
            var resolution = store.ResolveWidened(objectId, tensorName);
            return new TensorData(
                objectId, tensorName,
                resolution.Dtype,
                resolution.Shape,
                resolution.Values);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Lists tensor names for an object.</summary>
    public IReadOnlyList<string> ListTensors(string objectId)
    {
        if (_container is null) return Array.Empty<string>();
        return _container.Index.Representations
            .Where(kvp => kvp.Key.StartsWith(objectId + "@"))
            .Select(_ => "weight") // decode later — for now return common names
            .Distinct()
            .ToList();
    }

    // ── INotifyPropertyChanged ──────────────────────────────────────────

    public event PropertyChangedEventHandler? PropertyChanged;
    private void OnPropertyChanged([CallerMemberName] string? name = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}

public sealed record TensorData(
    string ObjectId, string TensorName, Dtype Dtype,
    long[] Shape, float[] Values);

public enum NodeKind
{
    Index, Component, Layer, Object, Edge, Property,
}

public sealed class ExplorerNode
{
    public string Label { get; set; }
    public NodeKind Kind { get; set; }
    public string Id { get; set; }
    public string? Tooltip { get; set; }
    public ObservableCollection<ExplorerNode> Children { get; } = new();
    public bool IsExpanded { get; set; }

    public ExplorerNode(string label, NodeKind kind, string id)
    {
        Label = label;
        Kind = kind;
        Id = id;
    }
}