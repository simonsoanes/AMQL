using System.Text.Json;
using Amql.Hf;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Entail;

/// <summary>How a block layer normalises q and k: not at all, per head
/// (Qwen3, Gemma 3), or across the whole projection (OLMo 2).</summary>
public enum QkNormKind
{
    None,
    PerHead,
    Whole,
}

/// <summary>Where a block layer's neurons go: the dense FFN, the always-on
/// shared expert of an MoE layer (its routed experts set to zero), or every
/// routed expert alike (so whichever the router picks computes the same).</summary>
public enum BlockFfn
{
    Dense,
    SharedExpert,
    Experts,
}

/// <summary>One rotating pair of a head: dims <see cref="A"/> and
/// <see cref="B"/> turn together by <see cref="Omega"/> radians per token.</summary>
public readonly record struct RopePair(int A, int B, double Omega);

/// <summary>
/// The layer shape a compiled block is written in. A graft judges it from
/// the host (<see cref="EntailHost.Judge"/>) with the same rules the importer
/// uses, so the block is spelled and computed the way the host's own layers
/// are: head geometry, the host's rotary pairs and frequencies, its score
/// scale, its q/k norm, norm offset and placement, output gate, residual and
/// embedding multipliers, and FFN. The empty model and the adapter use
/// <see cref="Fresh"/>, the Qwen3.5 shape.
/// </summary>
public sealed class EntailTarget
{
    public required int Heads { get; init; }
    public required int KvGroups { get; init; }
    public required int HeadDim { get; init; }
    /// <summary>q_proj carries a sigmoid gate per head after its query (Qwen3.5).</summary>
    public required bool OutputGate { get; init; }
    public required QkNormKind QkNorm { get; init; }
    public float QkNormOffset { get; init; }
    /// <summary>The rotating pairs, highest frequency first, for each
    /// frequency regime the context can reach (two for longrope). A pair the
    /// block may use turns at the same rate in every regime.</summary>
    public required IReadOnlyList<IReadOnlyList<RopePair>> RopeRegimes { get; init; }
    /// <summary>longrope: the second regime starts past this length, so a
    /// context that never gets there needs only the first.</summary>
    public long? SecondRegimeAfter { get; init; }
    /// <summary>Dynamic NTK: every frequency changes past this length.</summary>
    public long? FrequenciesChangeAfter { get; init; }
    public required double ScoreScale { get; init; }
    /// <summary>cos and sin are multiplied by this (yarn, longrope).</summary>
    public double AttentionFactor { get; init; } = 1;
    public float? Softcap { get; init; }
    public required float NormOffset { get; init; }
    /// <summary>attn_in and ffn_in norms exist.</summary>
    public required bool PreNorms { get; init; }
    /// <summary>attn_out and ffn_out norms exist (sandwich or post-only).</summary>
    public required bool PostNorms { get; init; }
    public double ResidualScale { get; init; } = 1;
    /// <summary>Embedding rows are multiplied by sqrt(hidden_size) (Gemma).</summary>
    public bool EmbedScaleSqrtHidden { get; init; }
    /// <summary>A fixed embedding multiplier (Granite), else 1.</summary>
    public double EmbedMultiplier { get; init; } = 1;
    /// <summary>Logits are multiplied by this (Granite, Cohere), else 1.</summary>
    public double OutputMultiplier { get; init; } = 1;
    public required BlockFfn Ffn { get; init; }
    /// <summary>Neurons one block layer can hold.</summary>
    public required int Intermediate { get; init; }
    /// <summary>Routed experts in an MoE layer, and their width.</summary>
    public int ExpertCount { get; init; }
    public int ExpertIntermediate { get; init; }
    /// <summary>Replicated experts: the total routing weight of the
    /// selected experts when every router score is equal.</summary>
    public double ExpertWeight { get; init; } = 1;
    /// <summary>Block tensor names, layer-relative without the index, keyed
    /// <c>role.weight</c> / <c>role.bias</c> as <see cref="LayerRoleMap"/> is.</summary>
    public required IReadOnlyDictionary<string, string> Names { get; init; }
    public IReadOnlyList<FusedTensor>? Fused { get; init; }
    public ExpertLayout? ExpertNames { get; init; }
    /// <summary>The host's spelling of the block's layer type.</summary>
    public string LayerType { get; init; } = "full_attention";
    /// <summary>A sliding window the block layers inherit (Mistral).</summary>
    public long? Window { get; init; }

    /// <summary>Heads the program may use: one is kept back to fix the scale
    /// of a post-norm's input.</summary>
    public int ProgramHeads => PostNorms ? Heads - 1 : Heads;
    public int QDim => Heads * HeadDim;
    public int KvDim => KvGroups * HeadDim;
    public int HeadsPerGroup => Heads / KvGroups;

    /// <summary>The Qwen3.5 layer shape of the empty model and the adapter:
    /// 8 query heads over 2 KV groups of head_dim 256, an output gate, per-head
    /// q/k norm, (1+w) norms, and 16 of the 32 pairs a 0.25 partial rope turns
    /// at base 1e7.</summary>
    public static EntailTarget Fresh(int intermediate) => new()
    {
        Heads = EntailProgram.HeadsPerLayer,
        KvGroups = EntailProgram.KvGroups,
        HeadDim = EntailProgram.HeadDim,
        OutputGate = true,
        QkNorm = QkNormKind.PerHead,
        QkNormOffset = 1,
        RopeRegimes = new[] { HalfSplitPairs(FreshRotary, FreshTheta, 0) },
        ScoreScale = 1 / Math.Sqrt(EntailProgram.HeadDim),
        NormOffset = 1,
        PreNorms = true,
        PostNorms = false,
        Ffn = BlockFfn.Dense,
        Intermediate = intermediate,
        Names = new Dictionary<string, string>
        {
            [$"{TensorRoles.AttnIn}.weight"] = "input_layernorm.weight",
            [$"{TensorRoles.FfnIn}.weight"] = "post_attention_layernorm.weight",
            [$"{TensorRoles.Q}.weight"] = "self_attn.q_proj.weight",
            [$"{TensorRoles.K}.weight"] = "self_attn.k_proj.weight",
            [$"{TensorRoles.V}.weight"] = "self_attn.v_proj.weight",
            [$"{TensorRoles.O}.weight"] = "self_attn.o_proj.weight",
            [$"{TensorRoles.QNorm}.weight"] = "self_attn.q_norm.weight",
            [$"{TensorRoles.KNorm}.weight"] = "self_attn.k_norm.weight",
            [$"{TensorRoles.Gate}.weight"] = "mlp.gate_proj.weight",
            [$"{TensorRoles.Up}.weight"] = "mlp.up_proj.weight",
            [$"{TensorRoles.Down}.weight"] = "mlp.down_proj.weight",
        },
    };

    internal const int FreshRotary = 64;
    internal const double FreshTheta = 1e7;

    /// <summary>Unscaled half-split pairs over a rotary window starting at
    /// dim <paramref name="first"/>: (i, i + width/2) at theta^(-2i/width).</summary>
    internal static RopePair[] HalfSplitPairs(int width, double theta, int first) =>
        Enumerable.Range(0, width / 2)
            .Select(i => new RopePair(first + i, first + i + width / 2, Math.Pow(theta, -2.0 * i / width)))
            .ToArray();
}

/// <summary>
/// A host checkpoint as a graft sees it: the judged block shape, and where its
/// tensors and config facts are. Everything is judged by the importer's own
/// <see cref="ArchMapper"/>; what a block cannot be written exactly into is
/// refused, naming the property.
/// </summary>
public sealed class EntailHost
{
    public required EntailTarget Target { get; init; }
    public required string Directory { get; init; }
    /// <summary>The text stack's tensor prefix, with a trailing dot.</summary>
    public required string Prefix { get; init; }
    public required int Hidden { get; init; }
    public required int Layers { get; init; }
    public required bool Tied { get; init; }
    public required LayerRoleMap Roles { get; init; }
    public required IReadOnlyList<AttentionLayerPolicy> Policies { get; init; }
    public required string EmbeddingName { get; init; }
    public required string FinalNormName { get; init; }
    public string? HeadName { get; init; }

    public static EntailHost Judge(string hostDir)
    {
        string configPath = Path.Combine(hostDir, "config.json");
        var facts = ModelConfig.ReadTextFacts(configPath);
        using var inventory = HfInventory.Open(hostDir);
        ContainerSpec spec;
        try
        {
            spec = ArchMapper.MapToContainerSpec("graft-host", facts, inventory,
                new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });
        }
        catch (ModelConfigException e)
        {
            throw new EntailException($"the host is not a model this build can judge: {e.Message}");
        }
        var component = spec.SystemGraph.Components.First(c => c.Role == ComponentRole.PrimaryText);
        var surface = component.Execution ?? throw new EntailException("the host has no execution surface");
        var policies = component.Attention ?? throw new EntailException("the host has no per-layer attention table");
        var roles = surface.Roles ?? throw new EntailException("the host's layer tensors were not judged into roles");
        var attention = surface.Attention ?? throw new EntailException("the host has no attention: a block needs heads");
        string prefix = ArchMapper.DetectTextPrefix(inventory) + ".";

        void Refuse(string property) => throw new EntailException($"cannot graft into this host: {property}");
        if (surface.Norm.Pre.Kind != NormType.RmsNorm)
        {
            Refuse("its norms are LayerNorms, whose mean subtraction moves every lane — the block needs RMSNorm");
        }
        if (surface.ParallelResidual)
        {
            Refuse("its attention and FFN run in parallel off one norm, so a neuron cannot read what a head wrote in the same layer");
        }
        if (attention.ParameterFreeQkNorm is { Q: true } or { K: true } or { V: true })
        {
            Refuse("parameter-free q/k/v norms");
        }
        if (attention.Sinks is not null)
        {
            Refuse("attention sinks");
        }
        bool outputGate = false;
        if (attention.OutputGate is { } gate)
        {
            if (gate.TryGetProperty("attn_output_gate", out var g) && g.ValueKind == JsonValueKind.True)
            {
                outputGate = true;
            }
            else
            {
                Refuse($"an attention output gate of kind {gate.GetRawText()}");
            }
        }

        // ── the template: a softmax layer that rotates ───────────────────
        int template = Enumerable.Range(0, policies.Count).FirstOrDefault(l => Roped(policies[l]) && policies[l].Span != AttentionSpan.Sliding, -1);
        if (template < 0)
        {
            template = Enumerable.Range(0, policies.Count).FirstOrDefault(l => Roped(policies[l]), -1);
        }
        if (template < 0)
        {
            Refuse("none of its layers is a softmax attention layer with rotary positions, which the block's lookups rely on");
        }
        var policy = policies[template];
        int headDim = policy.Geometry?.HeadDim ?? attention.HeadDim;
        var (regimes, attentionFactor, changesAfter, secondAfter) = Pairs(policy.Position, headDim, attention.RopeInterleaved);

        // ── which norms a layer carries ─────────────────────────────────
        bool attnIn = roles.Name(TensorRoles.AttnIn) is not null, ffnIn = roles.Name(TensorRoles.FfnIn) is not null;
        bool attnOut = roles.Name(TensorRoles.AttnOut) is not null, ffnOut = roles.Name(TensorRoles.FfnOut) is not null;
        if (attnIn != ffnIn || attnOut != ffnOut)
        {
            Refuse("a layer normalises one sub-block's input or output but not the other's");
        }
        double residualScale = surface.ResidualScale ?? 1;
        if (!PowerOfTwo(residualScale))
        {
            Refuse($"residual multiplier {residualScale} is not a power of two, so the block's lanes would not stay exact");
        }
        if (attnOut && residualScale != 1)
        {
            Refuse("post-norms together with a residual multiplier");
        }
        var conventions = FamilyConventions.For(facts.ModelType);
        var head = surface.Head!;
        double embedMultiplier = 1;
        if (!conventions.EmbedScaleSqrtHidden && head.EmbedScale is { } scale && scale != 1)
        {
            if (!PowerOfTwo(scale))
            {
                Refuse($"embedding multiplier {scale} is not a power of two, so the block's lanes would not stay exact");
            }
            embedMultiplier = scale;
        }
        var ffnSurface = surface.Ffn!;
        // Every activation served is relu at the block's gate levels (±32):
        // silu, gelu and gelu-tanh round to exactly x or 0 there in f32.
        if (ffnSurface.Activation is not (Activation.Silu or Activation.Gelu or Activation.GeluTanh or Activation.Relu))
        {
            Refuse($"activation {ffnSurface.Activation} is not relu-like at the block's gate levels");
        }

        // ── q/k norm, from the template layer's own tensors ─────────────
        string LayerTensor(string relative) => $"{prefix}layers.{template}.{relative}";
        var qkNorm = QkNormKind.None;
        if (roles.Name(TensorRoles.QNorm) is { } qNormName && inventory.TryGet(LayerTensor(qNormName), out var qNormInfo))
        {
            qkNorm = qNormInfo.Shape[0] == headDim ? QkNormKind.PerHead : QkNormKind.Whole;
        }

        // ── the FFN the block's neurons live in ─────────────────────────
        var names = new Dictionary<string, string>(roles.Roles, StringComparer.Ordinal);
        string? routerName = roles.Name(TensorRoles.Router);
        bool routed = routerName is not null && inventory.TryGet(LayerTensor(routerName), out _);
        bool denseNamed = roles.Name(TensorRoles.Gate) is not null && roles.Name(TensorRoles.Up) is not null;
        bool canDeclareDense = routed && denseNamed && TextConfig(configPath).TryGetProperty("mlp_only_layers", out _);
        BlockFfn ffn;
        int intermediate;
        double expertWeight = 1;
        var routedMoe = ffnSurface.Moe;
        if (!routed || canDeclareDense)
        {
            if (!denseNamed)
            {
                Refuse("its FFN is not gated (act(gate)·up), and the block's neurons multiply two lanes");
            }
            ffn = BlockFfn.Dense;
            intermediate = ffnSurface.IntermediateSize;
        }
        else if (roles.Name(TensorRoles.SharedUp) is { } sharedUp && inventory.TryGet(LayerTensor(sharedUp), out var sharedInfo))
        {
            ffn = BlockFfn.SharedExpert;
            intermediate = (int)sharedInfo.Shape[0];
            if (roles.Name(TensorRoles.SharedGate) is null)
            {
                Refuse("its shared expert is not gated (act(gate)·up)");
            }
        }
        else
        {
            var moe = ffnSurface.Moe!;
            ffn = BlockFfn.Experts;
            intermediate = moe.ExpertIntermediateSize;
            // Every router score equal: softmax gives 1/E each; renormalising
            // over the k selected gives 1/k each. Either is exact only as a
            // power of two, and so is the sum the down weights divide by.
            double each = moe.RoutingPolicy == ExpertRoutingPolicy.NormalisedOverSelected ? 1.0 / moe.TopK : 1.0 / moe.Experts;
            expertWeight = each * moe.TopK;
            if (!PowerOfTwo(each) || !PowerOfTwo(expertWeight))
            {
                Refuse($"with {moe.Experts} experts and top-{moe.TopK} routing, an even split of the routing weight is not a power of two");
            }
            if (roles.Experts?.Gate is null)
            {
                Refuse("its experts are not gated (act(gate)·up)");
            }
        }

        // ── every tensor of a template layer must be one the block can set ──
        var known = new HashSet<string>(names.Values, StringComparer.Ordinal);
        foreach (var fused in roles.Fused ?? new List<FusedTensor>())
        {
            known.Add(fused.Name);
        }
        foreach (var name in inventory.TensorNames.Where(n => n.StartsWith($"{prefix}layers.{template}.", StringComparison.Ordinal)))
        {
            string relative = name[$"{prefix}layers.{template}.".Length..];
            if (!known.Contains(relative) && !(roles.Experts is { } ex && relative.StartsWith(ex.Prefix, StringComparison.Ordinal)))
            {
                Refuse($"its attention layers carry '{relative}', which the block has no value for");
            }
        }

        var target = new EntailTarget
        {
            Heads = attention.NumQHeads,
            KvGroups = policy.Geometry?.NumKvHeads ?? attention.NumKvHeads,
            HeadDim = headDim,
            OutputGate = outputGate,
            QkNorm = qkNorm,
            QkNormOffset = attention.QkNormWeightOffset,
            RopeRegimes = regimes,
            FrequenciesChangeAfter = changesAfter,
            SecondRegimeAfter = secondAfter,
            ScoreScale = attention.ScoreScale,
            AttentionFactor = attentionFactor,
            Softcap = attention.LogitSoftcapping,
            NormOffset = surface.Norm.Pre.WeightOffset,
            PreNorms = attnIn,
            PostNorms = attnOut,
            ResidualScale = residualScale,
            EmbedScaleSqrtHidden = conventions.EmbedScaleSqrtHidden,
            EmbedMultiplier = embedMultiplier,
            OutputMultiplier = head.OutputMultiplier ?? 1,
            Ffn = ffn,
            Intermediate = intermediate,
            ExpertCount = routedMoe?.Experts ?? 0,
            ExpertIntermediate = routedMoe?.ExpertIntermediateSize ?? 0,
            ExpertWeight = expertWeight,
            Names = names,
            Fused = roles.Fused,
            ExpertNames = roles.Experts,
            LayerType = policy.Span == AttentionSpan.Sliding ? "sliding_attention" : "full_attention",
            Window = policy.Span == AttentionSpan.Sliding ? policy.Window : null,
        };
        if (target.Heads % target.KvGroups != 0 || target.ProgramHeads < 1)
        {
            Refuse($"{target.Heads} query heads over {target.KvGroups} KV groups leaves the block no head to use");
        }

        string embedding = inventory.TensorNames.FirstOrDefault(n => n == $"{prefix}embed_tokens.weight")
            ?? inventory.TensorNames.FirstOrDefault(n => n.StartsWith(prefix, StringComparison.Ordinal) && !n.StartsWith($"{prefix}layers.", StringComparison.Ordinal) &&
                                                         n.Contains("embed", StringComparison.Ordinal))
            ?? throw new EntailException("the host has no token embedding under its text prefix");
        var finalNorms = inventory.TensorNames
            .Where(n => n.StartsWith(prefix, StringComparison.Ordinal) && !n.StartsWith($"{prefix}layers.", StringComparison.Ordinal) &&
                        n.EndsWith("norm.weight", StringComparison.Ordinal) && ShapeOf(inventory, n) is { Length: 1 } s && s[0] == facts.HiddenSize)
            .ToList();
        if (finalNorms.Count != 1)
        {
            Refuse($"{finalNorms.Count} candidates for its final norm");
        }
        string parent = prefix.TrimEnd('.').Contains('.') ? prefix[..prefix.TrimEnd('.').LastIndexOf('.')] + "." : string.Empty;
        string? headName = head.HeadReusesEmbedding ? null
            : new[] { "lm_head.weight", $"{prefix}lm_head.weight", $"{parent}lm_head.weight" }.FirstOrDefault(n => inventory.TryGet(n, out _))
              ?? throw new EntailException("the host's output head is untied but no lm_head tensor was found");

        return new EntailHost
        {
            Target = target,
            Directory = hostDir,
            Prefix = prefix,
            Hidden = facts.HiddenSize,
            Layers = facts.NumLayers,
            Tied = head.HeadReusesEmbedding,
            Roles = roles,
            Policies = policies,
            EmbeddingName = embedding,
            FinalNormName = finalNorms[0],
            HeadName = headName,
        };
    }

    private static bool Roped(AttentionLayerPolicy p) =>
        p.Operator == LayerOperators.Softmax && p.Position is PositionRope or PositionPartialRope or PositionZeroTailRope;

    internal static bool PowerOfTwo(double x) => x > 0 && Math.Abs(Math.Log2(x) - Math.Round(Math.Log2(x))) < 1e-12;

    private static long[] ShapeOf(HfInventory inventory, string name) => inventory.Get(name).Shape;

    private static JsonElement TextConfig(string configPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(configPath));
        var root = doc.RootElement;
        return (root.TryGetProperty("text_config", out var t) && t.ValueKind == JsonValueKind.Object ? t : root).Clone();
    }

    /// <summary>The rotating pairs of a position policy, for each regime the
    /// frequencies can be in, and the factor cos and sin are multiplied by.</summary>
    private static (IReadOnlyList<IReadOnlyList<RopePair>> Regimes, double AttentionFactor, long? ChangesAfter, long? SecondAfter) Pairs(
        PositionPolicy position, int headDim, bool interleaved)
    {
        RopePair[] Layout(double[] omegas, int width, bool zeroTailHalf)
        {
            int half = zeroTailHalf ? headDim / 2 : width / 2;
            return omegas.Select((w, i) => interleaved
                ? new RopePair(2 * i, 2 * i + 1, w)
                : new RopePair(i, i + half, w)).ToArray();
        }
        switch (position)
        {
            case PositionZeroTailRope zeroTail:
            {
                int rotated = zeroTail.RotatedPairs(headDim);
                var omegas = Enumerable.Range(0, rotated).Select(i => Math.Pow(zeroTail.Theta, -2.0 * i / headDim)).ToArray();
                return (new[] { Layout(omegas, headDim, zeroTailHalf: true) }, 1, null, null);
            }
            case PositionRope or PositionPartialRope:
            {
                var (theta, scaling, width) = position switch
                {
                    PositionRope r => (r.Theta, r.Scaling, headDim),
                    PositionPartialRope p => (p.Theta, p.Scaling, p.RotaryWidth(headDim)),
                    _ => throw new InvalidOperationException(),
                };
                if (scaling is null)
                {
                    // Computed exactly as the empty model's pairs are.
                    var plain = EntailTarget.HalfSplitPairs(width, theta, 0).Select(p => p.Omega).ToArray();
                    return (new[] { Layout(plain, width, false) }, 1, null, null);
                }
                var (shortInv, factor) = RopeScaling.Frequencies(theta, width, scaling, 0);
                var regimes = new List<IReadOnlyList<RopePair>> { Layout(shortInv, width, false) };
                if (scaling.Type == "longrope")
                {
                    regimes.Add(Layout(RopeScaling.Frequencies(theta, width, scaling, scaling.Threshold + 1).InvFreq, width, false));
                }
                return (regimes, factor, scaling.Type == "dynamic" ? scaling.Threshold : null,
                    scaling.Type == "longrope" ? scaling.Threshold : null);
            }
            default:
                throw new EntailException("cannot graft into this host: its attention layers have no rotary positions");
        }
    }
}
