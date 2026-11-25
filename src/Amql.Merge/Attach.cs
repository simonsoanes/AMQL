using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>What <see cref="Attach"/> added to a container.</summary>
public sealed record AttachReport(string Container, IReadOnlyList<string> Objects, int Tensors, long PayloadBytes, IReadOnlyList<string> Notes);

/// <summary>
/// Adds a part to a container that is there already, beside what it holds: a drafter that reads
/// the model's caches. Only new segments are written; the model's own are not touched, and the
/// index and graph are replaced once the segments are whole.
/// </summary>
public static class Attach
{
    private const string DrafterComponent = "drafter";

    /// <summary>The drafter's matrices that a 4-bit copy rounds.</summary>
    internal static bool DrafterMatrix(string name) =>
        name is "pre_projection.weight" or "post_projection.weight" or "embed_tokens.weight" ||
        (name.StartsWith("layers.", StringComparison.Ordinal) && name.EndsWith("_proj.weight", StringComparison.Ordinal));

    /// <summary>
    /// Adds an assistant checkpoint (Gemma 4's) as the container's drafter, replacing one it
    /// has. <paramref name="format"/> rounds its matrices to 4 bits; null keeps them as stored.
    /// </summary>
    public static AttachReport Drafter(string containerDir, string checkpointDir, Fp4Format? format)
    {
        using var config = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(checkpointDir, "config.json")));
        var root = config.RootElement;
        string type = Str(root, "model_type") ?? "";
        if (!type.StartsWith("gemma4", StringComparison.Ordinal) || !type.EndsWith("assistant", StringComparison.Ordinal))
        {
            throw new MergeException($"'{checkpointDir}' is a '{type}' checkpoint, not a drafter this build attaches (gemma4_assistant)");
        }
        var text = root.GetProperty("text_config");
        if (root.TryGetProperty("use_ordered_embeddings", out var ordered) && ordered.ValueKind == JsonValueKind.True)
        {
            throw new MergeException("the drafter's head is clustered (use_ordered_embeddings), which this build does not run");
        }
        int layers = text.GetProperty("num_hidden_layers").GetInt32();
        if (Int(text, "num_kv_shared_layers") != layers)
        {
            throw new MergeException("the drafter has layers with caches of their own, which this build does not run");
        }
        if (Int(text, "hidden_size_per_layer_input") is > 0 || (text.TryGetProperty("enable_moe_block", out var moe) && moe.ValueKind == JsonValueKind.True) ||
            (text.TryGetProperty("final_logit_softcapping", out var cap) && cap.ValueKind == JsonValueKind.Number))
        {
            throw new MergeException("the drafter has per-layer inputs, experts or a logit softcap, which this build does not run");
        }
        var spans = text.GetProperty("layer_types").EnumerateArray().Select(t => t.GetString()!).ToList();
        long window = Int(text, "sliding_window") ?? 0;

        using var container = Vindex3Container.Open(containerDir);
        var graph = container.Graph ?? throw new MergeException("the container records no system graph");
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        if (Int(root, "backbone_hidden_size") != plan.HiddenSize || Int(text, "vocab_size") != plan.Output?.VocabSize)
        {
            throw new MergeException($"the drafter is for a model {Int(root, "backbone_hidden_size")} wide with {Int(text, "vocab_size")} tokens; " +
                                     $"this one is {plan.HiddenSize} wide with {plan.Output?.VocabSize}");
        }
        SameAsTarget(container, text);

        var policies = new List<AttentionLayerPolicy>();
        for (int i = 0; i < layers; i++)
        {
            bool sliding = spans[i] == "sliding_attention";
            int source = plan.Layers.FindLastIndex(l => l.Attention is { } a && (a.Window is not null) == sliding);
            if (source < 0)
            {
                throw new MergeException($"the drafter's layer {i} ({spans[i]}) reads a cache the model has no layer for");
            }
            var attn = plan.Layers[source].Attention!;
            if (sliding && attn.Window != window)
            {
                throw new MergeException($"the drafter's window is {window}, the model's {attn.Window}");
            }
            policies.Add(new AttentionLayerPolicy
            {
                Operator = LayerOperators.Softmax,
                Span = sliding ? AttentionSpan.Sliding : AttentionSpan.Full,
                Window = sliding ? window : null,
                Position = attn.Position,
                Geometry = new HeadGeometry { HeadDim = attn.HeadDim, NumKvHeads = attn.NumKvHeads },
            });
        }

        using var inventory = HfInventory.Open(checkpointDir);
        int hidden = text.GetProperty("hidden_size").GetInt32();
        var tensors = new List<NamedTensorData>();
        int rounded = 0;
        foreach (string name in inventory.TensorNames.OrderBy(n => n, StringComparer.Ordinal))
        {
            var info = inventory.Get(name);
            string relative = name.StartsWith("model.", StringComparison.Ordinal) ? name["model.".Length..] : name;
            if (format is { } fp4 && DrafterMatrix(relative) && info.Shape.Length == 2 && info.Shape[1] % 32 == 0)
            {
                var values = BitPattern.WidenToF32(info.Dtype, inventory.ReadBytes(name));
                tensors.AddRange(QuantiseContainer.Quantise(relative, values, (int)info.Shape[0], (int)info.Shape[1], fp4));
                rounded++;
                continue;
            }
            tensors.Add(new NamedTensorData
            {
                Name = relative, Dtype = info.Dtype, Shape = info.Shape,
                Source = (offset, count) => inventory.ReadBytes(name, offset, count), SourceLength = info.DataLength,
            });
        }
        var component = new Component
        {
            Id = DrafterComponent, Role = ComponentRole.Drafter, SourceArtifact = "model", NumLayers = layers, HiddenSize = hidden, Attention = policies,
        };
        var obj = new LogicalObject
        {
            Id = DrafterPlan.Stack, Component = DrafterComponent, Kind = ObjectKind.DecoderStack,
            SourceBindings = [new SourceBinding { Artifact = "model", TensorPrefix = "model", Tensors = tensors.Count, Bytes = tensors.Sum(t => t.PayloadLength) }],
            Representations = [new Representation { Encoding = Encoding(container), Fidelity = format is null ? Fidelity.Canonical : Fidelity.Approximate }],
        };
        store.Dispose();
        return Write(containerDir, container, [component], [(obj, tensors)],
            [$"{layers}-layer drafter, {hidden} wide: {tensors.Count} tensors" + (format is null ? "" : $", {rounded} matrices to {format}")]);
    }

    /// <summary>Adds a checkpoint's vision tower and projector, as encoding it with them would have.</summary>
    public static AttachReport Vision(string containerDir, string checkpointDir)
    {
        string configPath = Path.Combine(checkpointDir, "config.json");
        var facts = ModelConfig.ReadTextFacts(configPath);
        using var inventory = HfInventory.Open(checkpointDir);
        var spec = ArchMapper.MapToContainerSpec(Path.GetFileName(Path.TrimEndingDirectorySeparator(checkpointDir)), facts, inventory,
            new ArchMapper.EncodeOptions { IncludeVision = true, IncludeMtp = false }, null);
        var objects = spec.SystemGraph.Objects.Where(o => o.Component == "vision" && o.Representations.Count > 0)
            .Select(o => (Object: o, spec.Representations.First(r => r.ObjectId == o.Id).Tensors)).ToList();
        if (objects.Count == 0)
        {
            throw new MergeException($"'{checkpointDir}' has no vision tower");
        }
        using var container = Vindex3Container.Open(containerDir);
        if (container.Index.HiddenSize != facts.HiddenSize)
        {
            throw new MergeException($"the tower is for a model {facts.HiddenSize} wide; this one is {container.Index.HiddenSize} wide");
        }
        var report = Write(containerDir, container, spec.SystemGraph.Components.Where(c => c.Id == "vision").ToList(), objects,
            [$"vision tower and projector: {objects.Sum(o => o.Tensors.Count)} tensors"]);
        foreach (string file in new[] { "processor_config.json", "preprocessor_config.json" })
        {
            string from = Path.Combine(checkpointDir, file), to = Path.Combine(containerDir, file);
            if (File.Exists(from) && !File.Exists(to))
            {
                File.Copy(from, to);
            }
        }
        return report;
    }

    /// <summary>What the drafter shares with the model rather than states: its rotary
    /// embeddings, norm epsilon and activation.</summary>
    private static void SameAsTarget(Vindex3Container container, JsonElement drafter)
    {
        string path = Path.Combine(container.Root, ModelToContainer.SourceConfigFile);
        if (!File.Exists(path))
        {
            return;
        }
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var target = doc.RootElement.TryGetProperty("text_config", out var t) ? t : doc.RootElement;
        foreach (string key in new[] { "rope_parameters", "rms_norm_eps", "hidden_activation" })
        {
            string? a = target.TryGetProperty(key, out var x) ? Canonical(x) : null, b = drafter.TryGetProperty(key, out var y) ? Canonical(y) : null;
            if (a is not null && b is not null && a != b)
            {
                throw new MergeException($"the drafter's {key} is {b}, the model's {a}");
            }
        }
    }

    private static string Canonical(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.Object => "{" + string.Join(",", e.EnumerateObject().OrderBy(p => p.Name, StringComparer.Ordinal).Select(p => p.Name + ":" + Canonical(p.Value))) + "}",
        JsonValueKind.Number => e.GetDouble().ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        _ => e.GetRawText(),
    };

    private static string Encoding(Vindex3Container container) =>
        container.Index.Representations[container.CanonicalRepresentationId("target.decoder_stack")].Encoding;

    /// <summary>Writes the objects' segments, then the index and the graph with them in it.</summary>
    internal static AttachReport Write(string containerDir, Vindex3Container container, IReadOnlyList<Component> components,
        IReadOnlyList<(LogicalObject Object, List<NamedTensorData> Tensors)> objects, IReadOnlyList<string> notes)
    {
        string encoding = Encoding(container);
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(containerDir, "index.json")))!.AsObject();
        string graphFile = Path.Combine(containerDir, container.Index.SystemGraph ?? "system_graph.json");
        var graph = JsonNode.Parse(File.ReadAllText(graphFile))!.AsObject();
        var segments = new List<(string Temporary, string Final)>();
        long bytes = 0;
        int tensorCount = 0;
        void Put(JsonArray list, string id, object value)
        {
            var node = JsonSerializer.SerializeToNode(value, ViJson.Options);
            for (int i = 0; i < list.Count; i++)
            {
                if (list[i]!["id"]!.GetValue<string>() == id)
                {
                    list[i] = node;
                    return;
                }
            }
            list.Add(node);
        }
        foreach (var component in components)
        {
            Put(graph["components"]!.AsArray(), component.Id, component);
        }
        foreach (var (obj, tensors) in objects)
        {
            Put(graph["objects"]!.AsArray(), obj.Id, obj);
            string relative = $"segments/{obj.Id}.bin", final = Path.Combine(containerDir, relative);
            var result = SegmentWriter.Write(final + ".new", $"{obj.Id}@{encoding}", tensors);
            segments.Add((final + ".new", final));
            index["representations"]!.AsObject()[$"{obj.Id}@{encoding}"] = JsonSerializer.SerializeToNode(new RepresentationEntry
            {
                Object = obj.Id, Encoding = encoding, Segment = relative, TensorCount = tensors.Count,
                PayloadBytes = result.PayloadBytes, PayloadSha256 = result.PayloadSha256Hex, SegmentSha256 = result.SegmentSha256Hex,
            }, ViJson.Options);
            index["segments"]!.AsObject()[$"segments/{obj.Id}"] = 1;
            bytes += result.PayloadBytes;
            tensorCount += tensors.Count;
        }
        container.Dispose();
        foreach (var (temporary, final) in segments)
        {
            File.Move(temporary, final, overwrite: true);
        }
        Replace(graphFile, graph.ToJsonString());
        Replace(Path.Combine(containerDir, "index.json"), index.ToJsonString());
        return new AttachReport(containerDir, objects.Select(o => o.Object.Id).ToList(), tensorCount, bytes, notes);
    }

    private static void Replace(string path, string text)
    {
        File.WriteAllText(path + ".new", text);
        File.Move(path + ".new", path, overwrite: true);
    }

    private static string? Str(JsonElement e, string name) => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static int? Int(JsonElement e, string name) => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : null;
}
