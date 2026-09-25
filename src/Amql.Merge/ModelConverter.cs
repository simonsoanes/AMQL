using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Hf;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Merge;

public sealed record ConvertReport(string OutDir, string Model, IReadOnlyList<string> Notes);

/// <summary>
/// Converts a generative (decoder) container into a classifier or embedding
/// container by adding the head/surface facts the target type needs.
/// The base model's weights stay byte-identical; only the system graph and
/// (for classifiers) a new score-head segment are added.
/// </summary>
public static class ModelConverter
{
    public static ConvertReport ConvertToClassifier(
        string containerDir, string outDir, int numLabels, string problemType = "single_label_classification")
    {
        if (Directory.Exists(outDir))
        {
            throw new MergeException($"convert output '{outDir}' already exists");
        }
        if (numLabels < 2)
        {
            throw new MergeException($"num-labels must be ≥ 2, got {numLabels}");
        }

        using var container = Vindex3Container.Open(containerDir);
        var graph = container.Graph ?? throw new MergeException("container records no system graph");
        var component = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.PrimaryText)
            ?? throw new MergeException("container records no primary text component");
        var surface = component.Execution
            ?? throw new MergeException($"component '{component.Id}' declares no execution surface");
        if (surface.Classifier is not null)
        {
            throw new MergeException(
                $"component '{component.Id}' already carries a ClassifierSurface — the container is already a classifier");
        }
        var head = surface.Head
            ?? throw new MergeException($"component '{component.Id}' declares no head surface");
        int hidden = component.HiddenSize;

        var notes = new List<string>();

        // ── 1. Create the random score head ──────────────────────────────
        var rng = new Random(42); // deterministic seed for reproducibility
        var scoreBytes = new byte[numLabels * hidden * sizeof(float)];
        float scale = MathF.Sqrt(2.0f / (hidden + numLabels)); // Xavier uniform
        Span<float> score = stackalloc float[numLabels * hidden];
        // Use a Box-Muller-inspired approximation: uniform [-scale, scale]
        // amortised over the long vector for speed.
        for (int i = 0; i < score.Length; i++)
        {
            score[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        }
        Buffer.BlockCopy(score.ToArray(), 0, scoreBytes, 0, scoreBytes.Length);

        // ── 2. Build the container ───────────────────────────────────────
        Directory.CreateDirectory(outDir);

        // Copy existing segments verbatim
        var representations = new Dictionary<string, RepresentationEntry>(container.Index.Representations);
        var segments = new Dictionary<string, int>(container.Index.Segments);
        foreach (var (_, entry) in container.Index.Representations)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(outDir, entry.Segment))!);
            File.Copy(Path.Combine(container.Root, entry.Segment), Path.Combine(outDir, entry.Segment), overwrite: true);
        }

        // Copy the tokenizer
        var tokenizer = Path.Combine(container.Root, "tokenizer.json");
        if (File.Exists(tokenizer))
        {
            File.Copy(tokenizer, Path.Combine(outDir, "tokenizer.json"));
        }
        HfAncillaryFiles.CopyInto(container.Root, outDir);

        // Write the score-head segment
        string scoreSegmentPath = "segments/target.classifier_head.bin";
        var scoreTensors = new List<NamedTensorData>
        {
            new() { Name = "weight", Dtype = Dtype.F32, Shape = new long[] { numLabels, hidden }, Data = scoreBytes },
        };
        var scoreResult = SegmentWriter.Write(
            Path.Combine(outDir, scoreSegmentPath),
            "target.classifier_head@F32", scoreTensors);
        representations["target.classifier_head@F32"] = new RepresentationEntry
        {
            Object = "target.classifier_head",
            Encoding = "F32",
            Segment = scoreSegmentPath,
            TensorCount = 1,
            PayloadBytes = scoreResult.PayloadBytes,
            PayloadSha256 = scoreResult.PayloadSha256Hex,
            SegmentSha256 = scoreResult.SegmentSha256Hex,
        };
        segments["segments/target.classifier_head"] = 1;

        // ── 3. Build the updated system graph ────────────────────────────
        var classifierSurface = new ClassifierSurface
        {
            NumLabels = numLabels,
            ProblemType = problemType,
            Pooling = new PoolingSurface { Kind = PoolingKind.Last, LastNonPad = true },
        };
        var newSurface = RebuildSurface(surface, classifier: classifierSurface);
        var newComponents = new List<Component>(graph.Components.Count);
        foreach (var c in graph.Components)
        {
            if (c.Role == ComponentRole.PrimaryText)
            {
                newComponents.Add(new Component
                {
                    Id = c.Id,
                    Role = c.Role,
                    SourceArtifact = c.SourceArtifact,
                    NumLayers = c.NumLayers,
                    HiddenSize = c.HiddenSize,
                    Attention = c.Attention,
                    Execution = newSurface,
                    Perception = c.Perception,
                });
            }
            else
            {
                newComponents.Add(c);
            }
        }

        var newObjects = new List<LogicalObject>(graph.Objects)
        {
            new()
            {
                Id = "target.classifier_head",
                Component = "target",
                Kind = ObjectKind.ClassifierHead,
                SourceBindings = new List<SourceBinding>
                {
                    new() { Artifact = "model.score", TensorPrefix = "model.score", Tensors = 1, Bytes = 0 },
                },
                Representations = new List<Representation>
                {
                    new() { Encoding = "F32", Fidelity = Fidelity.Canonical },
                },
            },
        };

        var newGraph = new SystemGraph
        {
            Schema = graph.Schema,
            Components = newComponents,
            Objects = newObjects,
            Edges = graph.Edges,
        };
        File.WriteAllText(Path.Combine(outDir, "system_graph.json"),
            JsonSerializer.Serialize(newGraph, ViJson.Options));

        // ── 4. Write the index ───────────────────────────────────────────
        var index = new Vindex3Index
        {
            Version = Vindex3Index.CurrentSchema,
            Model = container.Index.Model + "-classifier",
            Family = container.Index.Family,
            HiddenSize = hidden,
            NumLayers = component.NumLayers,
            SystemGraph = "system_graph.json",
            Representations = representations,
            Profiles = new List<Profile> { Profile.Exact() },
            Segments = segments,
            Authority = ContainerAuthority.Derived,
            DerivedFromModel = container.Index.Model,
            PrecisionMap = container.Index.PrecisionMap,
        };
        File.WriteAllText(Path.Combine(outDir, "index.json"),
            JsonSerializer.Serialize(index, ViJson.Options));

        notes.Add($"score head: [{numLabels} x {hidden}] F32, Xavier-uniform init (seed 42)");
        notes.Add("the head weights are a warm start — apply classification training data afterwards");

        return new ConvertReport(outDir, index.Model, notes);
    }

    public static ConvertReport ConvertToEmbedding(string containerDir, string outDir)
    {
        if (Directory.Exists(outDir))
        {
            throw new MergeException($"convert output '{outDir}' already exists");
        }

        using var container = Vindex3Container.Open(containerDir);
        var graph = container.Graph ?? throw new MergeException("container records no system graph");
        var component = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.PrimaryText)
            ?? throw new MergeException("container records no primary text component");
        var surface = component.Execution
            ?? throw new MergeException($"component '{component.Id}' declares no execution surface");

        var notes = new List<string>();
        int hidden = component.HiddenSize;

        // ── 1. Copy existing segments verbatim (no new tensors) ──────────
        Directory.CreateDirectory(outDir);
        var representations = new Dictionary<string, RepresentationEntry>(container.Index.Representations);
        var segments = new Dictionary<string, int>(container.Index.Segments);
        foreach (var (_, entry) in container.Index.Representations)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(outDir, entry.Segment))!);
            File.Copy(Path.Combine(container.Root, entry.Segment), Path.Combine(outDir, entry.Segment), overwrite: true);
        }

        var tokenizer = Path.Combine(container.Root, "tokenizer.json");
        if (File.Exists(tokenizer))
        {
            File.Copy(tokenizer, Path.Combine(outDir, "tokenizer.json"));
        }
        HfAncillaryFiles.CopyInto(container.Root, outDir);

        // ── 2. Build the updated system graph ────────────────────────────
        // An embedding surface records the pooling recipe; the stack stays
        // as-is (causal mask is disabled at runtime by the consumer).
        var embeddingFacts = JsonSerializer.SerializeToElement(new
        {
            pooling = new
            {
                kind = "mean",
                l2_normalise = true,
            },
        });
        var newSurface = RebuildSurface(surface, embedding: embeddingFacts);
        var newComponents = new List<Component>(graph.Components.Count);
        foreach (var c in graph.Components)
        {
            if (c.Role == ComponentRole.PrimaryText)
            {
                newComponents.Add(new Component
                {
                    Id = c.Id,
                    Role = c.Role,
                    SourceArtifact = c.SourceArtifact,
                    NumLayers = c.NumLayers,
                    HiddenSize = c.HiddenSize,
                    Attention = c.Attention,
                    Execution = newSurface,
                    Perception = c.Perception,
                });
            }
            else
            {
                newComponents.Add(c);
            }
        }

        var newGraph = new SystemGraph
        {
            Schema = graph.Schema,
            Components = newComponents,
            Objects = graph.Objects,
            Edges = graph.Edges,
        };
        File.WriteAllText(Path.Combine(outDir, "system_graph.json"),
            JsonSerializer.Serialize(newGraph, ViJson.Options));

        // ── 3. Write the index ───────────────────────────────────────────
        var index = new Vindex3Index
        {
            Version = Vindex3Index.CurrentSchema,
            Model = container.Index.Model + "-embedding",
            Family = container.Index.Family,
            HiddenSize = hidden,
            NumLayers = component.NumLayers,
            SystemGraph = "system_graph.json",
            Representations = representations,
            Profiles = new List<Profile> { Profile.Exact() },
            Segments = segments,
            Authority = ContainerAuthority.Derived,
            DerivedFromModel = container.Index.Model,
            PrecisionMap = container.Index.PrecisionMap,
        };
        File.WriteAllText(Path.Combine(outDir, "index.json"),
            JsonSerializer.Serialize(index, ViJson.Options));

        notes.Add("pooling: mean over last non-padding token, L2-normalised");
        notes.Add("no new tensors — the decoder stack produces hidden states; pooling is a runtime operation");

        return new ConvertReport(outDir, index.Model, notes);
    }

    // ── helpers ──────────────────────────────────────────────────────────

    private static ExecutionSurface RebuildSurface(
        ExecutionSurface source,
        ClassifierSurface? classifier = null,
        JsonElement? embedding = null) => new()
        {
            ContextLength = source.ContextLength,
            Attention = source.Attention,
            Ffn = source.Ffn,
            Norm = source.Norm,
            Head = source.Head,
            ResidualScale = source.ResidualScale,
            LinearAttention = source.LinearAttention,
            Kda = source.Kda,
            KdaGateLowerBound = source.KdaGateLowerBound,
            Mla = source.Mla,
            Mamba2 = source.Mamba2,
            ConvQkv = source.ConvQkv,
            ResidualInFp32 = source.ResidualInFp32,
            Classifier = classifier,
            Embedding = embedding,
        };
}