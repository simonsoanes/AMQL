using System.Text.Json.Serialization;

namespace Amql.Vindex3;

/// <summary>
/// The role each layer tensor plays, judged once from the stack's own tensor
/// names and persisted on the execution surface. The planner reads roles, never
/// spellings, so a family whose tensors use a known spelling runs without code
/// that names the family.
/// <para>Keys are <c>&lt;role&gt;.weight</c> / <c>&lt;role&gt;.bias</c>; values
/// are the layer-relative tensor name without the layer index
/// (<c>q.weight → self_attn.q_proj.weight</c>, read at layer 3 as
/// <c>3.self_attn.q_proj.weight</c>). A role a given layer lacks (a linear
/// layer has no <c>q</c>) is simply absent from that layer's segment.</para>
/// </summary>
public sealed class LayerRoleMap
{
    public required Dictionary<string, string> Roles { get; init; }

    /// <summary>Checkpoint tensors stored fused (Phi-3's <c>qkv_proj</c>, GLM's
    /// <c>gate_up_proj</c>) and split on import into their role tensors. Export
    /// concatenates the parts back, in order, along the first axis.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<FusedTensor>? Fused { get; init; }

    public string? Name(string role, string part = "weight") =>
        Roles.TryGetValue($"{role}.{part}", out var name) ? name : null;
}

/// <summary>One fused checkpoint tensor: <see cref="Name"/> is the concatenation,
/// along axis 0, of <see cref="Parts"/> (all layer-relative, no layer index).</summary>
public sealed class FusedTensor
{
    public required string Name { get; init; }
    public required List<string> Parts { get; init; }
}

/// <summary>
/// The catalogue of layer tensor roles: every spelling a checkpoint is known to
/// use for each role, and the structural rules that tell the norm sites apart
/// (the same name, <c>post_attention_layernorm</c>, is the pre-FFN norm in Llama
/// and the attention output norm in Gemma 2 or OLMo 2 — which one it is follows
/// from which other norms the layer carries).
/// <para>Extending support to a family with a new spelling is one alias here;
/// nothing else names a tensor.</para>
/// </summary>
public static class TensorRoles
{
    // Norm sites around the two sub-blocks.
    public const string AttnIn = "attn_in_norm";
    public const string AttnOut = "attn_out_norm";
    public const string FfnIn = "ffn_in_norm";
    public const string FfnOut = "ffn_out_norm";

    // Softmax attention.
    public const string Q = "q";
    public const string K = "k";
    public const string V = "v";
    public const string O = "o";
    public const string QNorm = "q_norm";
    public const string KNorm = "k_norm";

    // Dense FFN.
    public const string Gate = "ffn_gate";
    public const string Up = "ffn_up";
    public const string Down = "ffn_down";

    /// <summary>Role → known module stems (layer-relative, no <c>.weight</c>).
    /// First match wins, so the most common spelling leads.</summary>
    public static readonly IReadOnlyDictionary<string, string[]> Aliases = new Dictionary<string, string[]>
    {
        [Q] = new[] { "self_attn.q_proj", "attention.wq", "attn.q_proj", "self_attention.q_proj" },
        [K] = new[] { "self_attn.k_proj", "attention.wk", "attn.k_proj", "self_attention.k_proj" },
        [V] = new[] { "self_attn.v_proj", "attention.wv", "attn.v_proj", "self_attention.v_proj" },
        [O] = new[] { "self_attn.o_proj", "self_attn.out_proj", "attention.wo", "attn.o_proj", "self_attn.dense", "self_attention.o_proj" },
        [QNorm] = new[] { "self_attn.q_norm", "self_attn.q_layernorm", "attention.q_norm", "attn.q_norm" },
        [KNorm] = new[] { "self_attn.k_norm", "self_attn.k_layernorm", "attention.k_norm", "attn.k_norm" },
        [Gate] = new[] { "mlp.gate_proj", "feed_forward.w1", "mlp.w1", "feed_forward.gate_proj" },
        [Up] = new[] { "mlp.up_proj", "feed_forward.w3", "mlp.w3", "feed_forward.up_proj", "mlp.c_fc", "mlp.fc1", "mlp.dense_h_to_4h" },
        [Down] = new[] { "mlp.down_proj", "feed_forward.w2", "mlp.w2", "feed_forward.down_proj", "mlp.c_proj", "mlp.fc2", "mlp.dense_4h_to_h" },
    };

    /// <summary>Fused spellings whose parts are contiguous row blocks — safe to
    /// split by byte range. Interleaved fusions (GPT-NeoX's per-head
    /// <c>query_key_value</c>) are deliberately absent: splitting them by rows
    /// would scramble heads, so such a checkpoint refuses instead.</summary>
    public static readonly IReadOnlyDictionary<string, string[]> ContiguousFusions = new Dictionary<string, string[]>
    {
        ["self_attn.qkv_proj"] = new[] { "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" },
        ["attn.qkv_proj"] = new[] { "attn.q_proj", "attn.k_proj", "attn.v_proj" },
        ["mlp.gate_up_proj"] = new[] { "mlp.gate_proj", "mlp.up_proj" },
        ["feed_forward.gate_up_proj"] = new[] { "feed_forward.gate_proj", "feed_forward.up_proj" },
    };

    /// <summary>Fused spellings known to interleave per head; refused by name.</summary>
    public static readonly IReadOnlyList<string> InterleavedFusions = new[]
    {
        "attention.query_key_value", "self_attention.query_key_value", "attn.c_attn", "attn.Wqkv",
    };

    private static readonly string[] AttnInNorms = { "input_layernorm", "operator_norm", "attention_norm", "ln_1", "ln_attn" };

    /// <summary>
    /// Judges the role map from the stack's layer-relative tensor names
    /// (<c>"3.self_attn.q_proj.weight"</c>). The layer index is stripped, so the
    /// map is the union over all layers: hybrid stacks carry softmax roles from
    /// their attention layers and nothing from their recurrent ones.
    /// </summary>
    public static LayerRoleMap Judge(IEnumerable<string> layerRelativeNames, List<FusedTensor>? fused = null)
    {
        var suffixes = new HashSet<string>(StringComparer.Ordinal);
        foreach (var name in layerRelativeNames)
        {
            int dot = name.IndexOf('.');
            if (dot > 0 && int.TryParse(name.AsSpan(0, dot), out _))
            {
                suffixes.Add(name[(dot + 1)..]);
            }
        }
        bool Has(string stem) => suffixes.Contains(stem + ".weight");

        foreach (var interleaved in InterleavedFusions)
        {
            if (Has(interleaved))
            {
                throw new ContainerException(
                    $"layer tensor '{interleaved}.weight' fuses Q/K/V interleaved per head — splitting it by rows " +
                    "would scramble the heads, so this build refuses it rather than guess the layout");
            }
        }

        var roles = new Dictionary<string, string>(StringComparer.Ordinal);
        void Bind(string role, string? stem)
        {
            if (stem is null)
            {
                return;
            }
            foreach (var part in new[] { "weight", "bias" })
            {
                if (suffixes.Contains($"{stem}.{part}"))
                {
                    roles[$"{role}.{part}"] = $"{stem}.{part}";
                }
            }
        }

        foreach (var (role, stems) in Aliases)
        {
            Bind(role, stems.FirstOrDefault(Has));
        }

        // ── norm sites, told apart by which norms coexist ──────────────────
        string? attnIn = AttnInNorms.FirstOrDefault(Has);
        Bind(AttnIn, attnIn);
        if (Has("pre_feedforward_layernorm") || Has("mlp.pre_layernorm"))
        {
            // Gemma 2/3 (and the four-norm PrePost layout): a dedicated pre-FFN
            // norm exists, so post_attention_layernorm normalises the attention
            // output before its residual add.
            Bind(AttnOut, "post_attention_layernorm");
            Bind(FfnIn, Has("pre_feedforward_layernorm") ? "pre_feedforward_layernorm" : "mlp.pre_layernorm");
            Bind(FfnOut, Has("post_feedforward_layernorm") ? "post_feedforward_layernorm" : "mlp.post_layernorm");
        }
        else if (Has("post_self_attn_layernorm"))
        {
            // GLM-4: post_attention_layernorm stays the pre-FFN norm; the two
            // sub-block outputs carry their own.
            Bind(AttnOut, "post_self_attn_layernorm");
            Bind(FfnIn, "post_attention_layernorm");
            Bind(FfnOut, "post_mlp_layernorm");
        }
        else if (Has("post_operator_norm"))
        {
            // LFM3's sandwich.
            Bind(AttnOut, "post_operator_norm");
            Bind(FfnIn, "ffn_norm");
            Bind(FfnOut, "post_ffn_norm");
        }
        else if (attnIn is null && Has("post_feedforward_layernorm"))
        {
            // OLMo 2 / EXAONE 4: no input norms at all — each sub-block's
            // output is normalised before its residual add.
            Bind(AttnOut, "post_attention_layernorm");
            Bind(FfnOut, "post_feedforward_layernorm");
        }
        else
        {
            Bind(FfnIn, new[] { "post_attention_layernorm", "ffn_norm", "ln_2", "ln_mlp" }.FirstOrDefault(Has));
        }

        return new LayerRoleMap
        {
            Roles = roles,
            Fused = fused is { Count: > 0 } ? fused : null,
        };
    }

    /// <summary>A block with one norm feeding both the attention and the FFN
    /// (Cohere, GPT-J) runs them in parallel off the same normalised input:
    /// <c>x + attn(n(x)) + ffn(n(x))</c>. Judged from structure: an input norm
    /// and an FFN, but no norm of any kind between the two sub-blocks.</summary>
    public static bool IsParallelBlock(LayerRoleMap map) =>
        map.Name(AttnIn) is not null &&
        map.Name(Down) is not null &&
        map.Name(FfnIn) is null && map.Name(AttnOut) is null && map.Name(FfnOut) is null;
}
