using System.Text.Json;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>The outcome of generating an MTP drafter for a model that has
/// none: the container now carries the bootstrapped module, and
/// <c>DraftAcceptance</c> measures the honest zero-shot draft quality —
/// the share of positions where the drafter's top-1 second-next-token
/// draft matches the model's actual continuation.</summary>
public sealed record GenerateMtpReport(
    string OutDir,
    string Model,
    int TrunkLayer,
    int Tensors,
    double DraftAcceptance,
    int SampledTokens,
    IReadOnlyList<string> Notes);

/// <summary>
/// Generates an MTP drafter for a dense container that has none, entirely
/// from its own weights (the bootstrapping init the MTP line uses):
///
/// <list type="bullet">
/// <item>the trunk layer is a verbatim copy of the model's LAST
/// full-attention decoder layer (self-attn + MLP + the two norms) — the
/// MTP trunk is always a softmax layer, so the last softmax layer is the
/// copy source;</item>
/// <item>the <c>pre_fc_norm_hidden</c>/<c>pre_fc_norm_embedding</c> norms
/// and the drafter's <c>norm.weight</c> copy the model's final norm;</item>
/// <item>the <c>fc</c> projector [2h → h] boots deterministically as the
/// mean-combination <c>0.5·(norm(h_t) + norm(e_{t+1}))</c>;</item>
/// <item>the head and conditioning embedding are the model's shared
/// tables (<c>mtp_use_dedicated_embeddings: false</c>).</item>
/// </list>
///
/// The module is materialised as <c>mtp.stack</c> (index entry, graph
/// object, its own segment) so <c>export</c> emits the drafter companion
/// automatically. <c>DraftAcceptance</c> is measured by running the
/// drafter's real pipeline — pre-fc norms, projector, one trunk
/// attention+FFN pass, shared head — against the model's hidden states.
/// </summary>
public static class GenerateMtp
{
    public static GenerateMtpReport Transform(
        string containerDir,
        string outDir,
        IReadOnlyList<int> acceptanceTokens,
        int sampleLimit = 1024)
    {
        if (Directory.Exists(outDir))
        {
            throw new MergeException($"generate-mtp output '{outDir}' already exists");
        }

        using var container = Vindex3Container.Open(containerDir);
        var graph = container.Graph ?? throw new MergeException("container records no system graph");
        var store = container.CreateOperandStore();
        using (store)
        {
            var plan = Planner.Plan(container, "target", store);
            int hidden = plan.HiddenSize;

            // The trunk is a copy of the last full-attention layer.
            int trunk = -1;
            for (int l = plan.Layers.Count - 1; l >= 0; l--)
            {
                if (plan.Layers[l].Attention is not null)
                {
                    trunk = l;
                    break;
                }
            }
            if (trunk < 0)
            {
                throw new MergeException(
                    "no full-attention layer to copy as the MTP trunk — the MTP trunk is always a softmax layer");
            }
            if (plan.FinalNorm.Weight is not { } finalNormRef ||
                plan.Embedding is not { } embedPlan ||
                plan.Output is null)
            {
                throw new MergeException(
                    "the model lacks the final-norm / embedding / head the drafter must share");
            }
            var srcLayer = plan.Layers[trunk];
            var srcAttention = srcLayer.Attention!;
            var srcFfn = srcLayer.Ffn?.Dense
                ?? throw new MergeException($"layer {trunk} has no dense FFN to copy as the trunk");

            // ── 1. the module tensors, all from the model's own weights ─
            const string stem = "mtp.";
            var tensors = new List<NamedTensorData>();

            // Build the trunk's tensor set from the plan's operand refs so
            // the rebound plan and the written segment agree exactly. The
            // segment stores RELATIVE names ("fc.weight", "layers.0.*"…)
            // like the encode's carried modules; the export rebuilds the
            // outer "mtp." names for the companion shard.
            void CopyRef(OperandRef source, string fullName)
            {
                var resolution = store.Resolve(source.ObjectId, source.TensorName);
                tensors.Add(new NamedTensorData
                {
                    Name = fullName[stem.Length..],
                    Dtype = resolution.Dtype,
                    Shape = resolution.Shape,
                    Data = resolution.Payload,
                });
            }

            static string Mod(string relative) => $"{stem}{relative}";

            CopyRef(srcAttention.QProj, Mod("layers.0.self_attn.q_proj.weight"));
            CopyRef(srcAttention.KProj, Mod("layers.0.self_attn.k_proj.weight"));
            if (!srcAttention.VFromK)
            {
                CopyRef(srcAttention.VProj, Mod("layers.0.self_attn.v_proj.weight"));
            }
            CopyRef(srcAttention.OProj, Mod("layers.0.self_attn.o_proj.weight"));
            if (srcAttention.QNorm is { } qNorm)
            {
                CopyRef(qNorm.Weight, Mod("layers.0.self_attn.q_norm.weight"));
            }
            if (srcAttention.KNorm is { } kNorm)
            {
                CopyRef(kNorm.Weight, Mod("layers.0.self_attn.k_norm.weight"));
            }
            CopyRef(srcFfn.Gate!, Mod("layers.0.mlp.gate_proj.weight"));
            CopyRef(srcFfn.Up, Mod("layers.0.mlp.up_proj.weight"));
            CopyRef(srcFfn.Down, Mod("layers.0.mlp.down_proj.weight"));
            if (srcLayer.PreAttentionNorm is { } preAttn)
            {
                CopyRef(preAttn.Weight, Mod("layers.0.input_layernorm.weight"));
            }
            if (srcLayer.PreFfnNorm is { } preFfn)
            {
                CopyRef(preFfn.Weight, Mod("layers.0.post_attention_layernorm.weight"));
            }

            // The drafter's final norm and its two pre-fc norms all copy
            // the model's final norm.
            var finalNorm = store.Resolve(finalNormRef.ObjectId, finalNormRef.TensorName);
            foreach (var fullName in new[] { Mod("norm.weight"), Mod("pre_fc_norm_hidden.weight"), Mod("pre_fc_norm_embedding.weight") })
            {
                tensors.Add(new NamedTensorData
                {
                    Name = fullName[stem.Length..],
                    Dtype = finalNorm.Dtype,
                    Shape = finalNorm.Shape,
                    Data = finalNorm.Payload,
                });
            }

            // The fc projector [h, 2h]: mean-combination boot.
            var fc = new float[hidden * 2 * hidden];
            for (int i = 0; i < hidden; i++)
            {
                fc[i * (2 * hidden) + i] = 0.5f;
                fc[i * (2 * hidden) + hidden + i] = 0.5f;
            }
            tensors.Add(new NamedTensorData
            {
                Name = "fc.weight",
                Dtype = Dtype.F32,
                Shape = new[] { (long)hidden, 2L * hidden },
                Data = ToF32Bytes(fc),
            });

            // ── 2. materialise into a new container ─────────────────────
            Directory.CreateDirectory(outDir);
            var representations = new Dictionary<string, RepresentationEntry>(StringComparer.Ordinal);
            var segments = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var (repId, entry) in container.Index.Representations)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(outDir, entry.Segment))!);
                File.Copy(Path.Combine(container.Root, entry.Segment), Path.Combine(outDir, entry.Segment), overwrite: true);
                representations[repId] = entry;
                segments[entry.Segment[..^4]] = 1;
            }

            string encoding = container.Index.Representations[
                container.CanonicalRepresentationId("target.decoder_stack")].Encoding;
            var result = SegmentWriter.Write(
                Path.Combine(outDir, "segments", "mtp.stack.bin"),
                $"mtp.stack@{encoding}",
                tensors);
            representations[$"mtp.stack@{encoding}"] = new RepresentationEntry
            {
                Object = "mtp.stack",
                Encoding = encoding,
                Segment = "segments/mtp.stack.bin",
                TensorCount = tensors.Count,
                PayloadBytes = result.PayloadBytes,
                PayloadSha256 = result.PayloadSha256Hex,
                SegmentSha256 = result.SegmentSha256Hex,
            };
            segments["segments/mtp.stack"] = 1;

            WriteGraph(outDir, container, graph, encoding, tensors.Count);
            WriteIndex(outDir, container, plan, encoding, representations, segments);
            CopyTokenizer(container, outDir);

            // ── 3. the zero-shot acceptance gate ────────────────────────
            double acceptance = double.NaN;
            var notes = new List<string>
            {
                $"trunk = a copy of layer {trunk} (the last full-attention layer); pre-fc norms and the drafter norm copy the final norm; " +
                "fc boots as the mean-combination 0.5·(norm(h)+norm(e))",
            };
            int sampled = Math.Min(sampleLimit, acceptanceTokens.Count);
            if (sampled >= 4)
            {
                using var acceptanceCtx = new DraftContext(outDir, acceptanceTokens, sampled);
                acceptance = acceptanceCtx.Score();
                notes.Add($"zero-shot draft acceptance over {sampled - 2} positions: " +
                          $"{(double.IsNaN(acceptance) ? "n/a" : acceptance.ToString("0.0%"))} — " +
                          "the bootstrapped drafter is a warm start; training (frozen model) is the next step");
            }
            else
            {
                notes.Add("no corpus tokens supplied — the acceptance gate was skipped (pass --text)");
            }

            return new GenerateMtpReport(
                OutDir: outDir,
                Model: container.Index.Model,
                TrunkLayer: trunk,
                Tensors: tensors.Count,
                DraftAcceptance: acceptance,
                SampledTokens: sampled,
                Notes: notes);
        }
    }

    // ── the acceptance gate ────────────────────────────────────────────────

    /// <summary>Zero-shot draft acceptance over the first <paramref name="sample"/>
    /// corpus tokens, measured through the runtime path — the same gate the
    /// boot and the fit use.</summary>
    public static double MeasureDraftAcceptance(string containerDir, IReadOnlyList<int> tokens, int sample)
    {
        using var ctx = new DraftContext(containerDir, tokens, sample);
        return ctx.Score();
    }

    internal static int LastFullAttentionLayer(ComponentOpPlan plan)
    {
        for (int l = plan.Layers.Count - 1; l >= 0; l--)
        {
            if (plan.Layers[l].Attention is not null)
            {
                return l;
            }
        }
        throw new MergeException("no full-attention layer to serve as the MTP trunk");
    }

    
    internal static ComponentOpPlan BuildTrunkPlan(
        Vindex3Container source, string generatedDir, ComponentOpPlan densePlan, int trunk, int hidden)
    {
        var src = densePlan.Layers[trunk];
        var srcAttn = src.Attention!;
        var srcFfn = src.Ffn!.Dense!;

        OperandRef R(string objectId, string sourceSuffix) =>
            new(objectId, sourceSuffix.Replace($"{trunk}.", "layers.0."));

        // Rebuild the ops with the operands rebound onto mtp.stack.
        return new ComponentOpPlan
        {
            ComponentId = "mtp",
            Embedding = densePlan.Embedding,
            Layers = new List<LayerPlan>
            {
                new()
                {
                    PreAttentionNorm = src.PreAttentionNorm is { } pa
                        ? NormOp.From(R("mtp.stack", pa.Weight.TensorName), NormSpecOf(pa), hidden)
                        : null,
                    PreFfnNorm = src.PreFfnNorm is { } pf
                        ? NormOp.From(R("mtp.stack", pf.Weight.TensorName), NormSpecOf(pf), hidden)
                        : null,
                    Attention = new AttentionOp
                    {
                        NumQHeads = srcAttn.NumQHeads,
                        NumKvHeads = srcAttn.NumKvHeads,
                        HeadDim = srcAttn.HeadDim,
                        ScoreScale = srcAttn.ScoreScale,
                        LogitSoftcapping = srcAttn.LogitSoftcapping,
                        Window = srcAttn.Window,
                        Position = srcAttn.Position,
                        VFromK = srcAttn.VFromK,
                        OutputGate = srcAttn.OutputGate,
                        QkNormScope = srcAttn.QkNormScope,
                        QkNormWeightOffset = srcAttn.QkNormWeightOffset,
                        ParameterFreeQkNorm = srcAttn.ParameterFreeQkNorm,
                        ParameterFreeQkNormEps = srcAttn.ParameterFreeQkNormEps,
                        QNorm = srcAttn.QNorm is { } qn ? NormOp.From(R("mtp.stack", qn.Weight.TensorName), NormSpecOf(qn), srcAttn.HeadDim) : null,
                        KNorm = srcAttn.KNorm is { } kn ? NormOp.From(R("mtp.stack", kn.Weight.TensorName), NormSpecOf(kn), srcAttn.HeadDim) : null,
                        QProj = R("mtp.stack", srcAttn.QProj.TensorName),
                        KProj = R("mtp.stack", srcAttn.KProj.TensorName),
                        VProj = R("mtp.stack", srcAttn.VProj.TensorName),
                        OProj = R("mtp.stack", srcAttn.OProj.TensorName),
                    },
                    Ffn = new LayerFfn
                    {
                        Dense = new DenseFfnOp
                        {
                            Gate = R("mtp.stack", srcFfn.Gate!.TensorName),
                            Up = R("mtp.stack", srcFfn.Up.TensorName),
                            Down = R("mtp.stack", srcFfn.Down.TensorName),
                            Activation = srcFfn.Activation,
                            IntermediateSize = srcFfn.IntermediateSize,
                            HiddenSize = hidden,
                        },
                    },
                },
            },
            FinalNorm = NormOp.From(new OperandRef("mtp.stack", "norm.weight"),
                NormSpecOf(densePlan.FinalNorm), hidden),
            Output = densePlan.Output,
        };
    }

    private static NormSpec NormSpecOf(NormOp op) => new()
    {
        Kind = op.Kind,
        Eps = op.Eps,
        WeightOffset = op.WeightOffset,
    };

    // ── container writing ──────────────────────────────────────────────────

    private static void WriteGraph(string outDir, Vindex3Container source, SystemGraph graph, string encoding, int mtpTensors)
    {
        var mtpObject = graph.Objects.FirstOrDefault(o => o.Component == "mtp" && o.Kind == ObjectKind.DecoderStack);
        if (mtpObject is null)
        {
            throw new MergeException("the graph has no mtp.stack object to materialise into");
        }
        var objects = graph.Objects.Select(o =>
            o.Id == mtpObject.Id
                ? new LogicalObject
                {
                    Id = o.Id,
                    Component = o.Component,
                    Kind = o.Kind,
                    SourceBindings = new List<SourceBinding>
                    {
                        new()
                        {
                            Artifact = "mtp",
                            TensorPrefix = "mtp",
                            Tensors = mtpTensors,
                            Bytes = mtpTensors, // census only; the segment holds the payloads
                        },
                    },
                    Representations = new List<Representation>
                    {
                        new() { Encoding = encoding, Fidelity = Fidelity.Canonical },
                    },
                }
                : o).ToList();

        // The mtp component may already exist (carried placeholder) or not.
        var components = graph.Components.ToList();
        if (!components.Any(c => c.Id == "mtp"))
        {
            var target = components.First(c => c.Role == ComponentRole.PrimaryText);
            components.Add(new Component
            {
                Id = "mtp",
                Role = ComponentRole.Drafter,
                SourceArtifact = "mtp",
                NumLayers = 1,
                HiddenSize = target.HiddenSize,
            });
        }

        File.WriteAllText(Path.Combine(outDir, "system_graph.json"),
            JsonSerializer.Serialize(new SystemGraph
            {
                Schema = graph.Schema,
                Components = components,
                Objects = objects,
                Edges = graph.Edges,
            }, ViJson.Options));
    }

    private static void WriteIndex(
        string outDir,
        Vindex3Container source,
        ComponentOpPlan plan,
        string encoding,
        Dictionary<string, RepresentationEntry> representations,
        Dictionary<string, int> segments)
    {
        var index = new Vindex3Index
        {
            Version = Vindex3Index.CurrentSchema,
            Model = source.Index.Model,
            Family = source.Index.Family,
            HiddenSize = plan.HiddenSize,
            NumLayers = plan.Layers.Count,
            SystemGraph = "system_graph.json",
            Representations = representations,
            Profiles = new List<Profile> { Profile.Exact() },
            Segments = segments,
            Authority = ContainerAuthority.Derived,
            DerivedFromModel = source.Index.Model,
            PrecisionMap = source.Index.PrecisionMap,
        };
        File.WriteAllText(Path.Combine(outDir, "index.json"),
            JsonSerializer.Serialize(index, ViJson.Options));
    }

    private static void CopyTokenizer(Vindex3Container source, string outDir)
    {
        var tokenizer = Path.Combine(source.Root, "tokenizer.json");
        if (File.Exists(tokenizer))
        {
            File.Copy(tokenizer, Path.Combine(outDir, "tokenizer.json"));
        }
    }

    private static byte[] ToF32Bytes(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}