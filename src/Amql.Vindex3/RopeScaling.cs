using System.Text.Json;

namespace Amql.Vindex3;

/// <summary>
/// A rotary frequency rescaling — the <c>rope_type</c> a checkpoint's rope
/// block declares beyond <c>default</c>. One computation, shared by the
/// runtime, the ONNX export and the GGUF export, mirroring transformers'
/// <c>modeling_rope_utils</c> exactly:
/// <list type="bullet">
/// <item><c>linear</c> — every inverse frequency divided by <c>factor</c>.</item>
/// <item><c>llama3</c> — wavelengths beyond <c>original/low_freq_factor</c>
/// divided by <c>factor</c>, those under <c>original/high_freq_factor</c> kept,
/// a linear blend between.</item>
/// <item><c>yarn</c> — interpolated (÷factor) and extrapolated frequencies
/// blended over a ramp between the <c>beta_fast</c>/<c>beta_slow</c>
/// correction dims, with cos/sin multiplied by an attention factor.</item>
/// <item><c>longrope</c> — each pair's frequency divided by its own
/// <c>short_factor</c> entry, or its <c>long_factor</c> entry once the
/// sequence is longer than <c>original_max_position_embeddings</c>; cos/sin
/// multiplied by <c>sqrt(1 + ln(factor)/ln(original))</c>.</item>
/// <item><c>dynamic</c> — NTK: once the sequence is longer than
/// <c>max_position_embeddings</c>, the base grows to
/// <c>theta·(factor·len/max − (factor − 1))^(d/(d−2))</c>.</item>
/// <item><c>ntk</c> — NTK-aware at a fixed factor: the base is
/// <c>theta·factor^(d/(d−2))</c>.</item>
/// </list>
/// Any of the fixed kinds may name a length it is for (<see cref="EngageBeyond"/>):
/// a sequence no longer than that rotates unscaled.
/// The last two depend on the sequence length, which transformers takes as
/// the largest position in the forward pass plus one; <see cref="Frequencies"/>
/// takes it the same way. Keys already in a cache keep the rotation they were
/// given, as transformers' do.
/// Serialised as the position policy's <c>scaling</c> object.
/// </summary>
public sealed class RopeScaling
{
    public static readonly IReadOnlyList<string> Types = new[] { "linear", "llama3", "yarn", "longrope", "dynamic", "ntk" };

    public required string Type { get; init; }
    public required double Factor { get; init; }
    public double? LowFreqFactor { get; init; }
    public double? HighFreqFactor { get; init; }
    public long? OriginalMaxPositions { get; init; }
    public double? BetaFast { get; init; }
    public double? BetaSlow { get; init; }
    public double? AttentionFactor { get; init; }
    public double? Mscale { get; init; }
    public double? MscaleAllDim { get; init; }
    public bool Truncate { get; init; } = true;
    /// <summary>longrope: per-pair divisors at or under the original length.</summary>
    public double[]? ShortFactor { get; init; }
    /// <summary>longrope: per-pair divisors beyond the original length.</summary>
    public double[]? LongFactor { get; init; }

    /// <summary>A context extension for long sequences only: one no longer than
    /// this rotates at the trained frequencies. The engine settles which a
    /// sequence is when it starts, from the length it is to reach; the managed
    /// path reads it off each call's positions, as it does for longrope.</summary>
    public long? EngageBeyond { get; init; }

    /// <summary>Whether the frequencies change with the sequence length.</summary>
    [System.Text.Json.Serialization.JsonIgnore]
    public bool DependsOnLength => Type is "longrope" or "dynamic" || EngageBeyond is not null;

    /// <summary>The sequence length past which a length-dependent scaling
    /// changes regime (longrope: the original length; dynamic: the model's
    /// maximum).</summary>
    [System.Text.Json.Serialization.JsonIgnore]
    public long Threshold => EngageBeyond ?? OriginalMaxPositions ?? throw Missing("original_max_position_embeddings");

    /// <summary>Inverse frequencies for a rotary window of <paramref name="width"/>
    /// dims at base <paramref name="theta"/>, and the factor cos/sin are
    /// multiplied by (1 except for yarn and longrope). <paramref name="sequenceLength"/>
    /// (largest position + 1) matters only to longrope and dynamic; 0 means
    /// "within the original length".</summary>
    public static (double[] InvFreq, double AttentionScale) Frequencies(double theta, int width, RopeScaling? scaling,
        long sequenceLength = 0)
    {
        if (scaling?.EngageBeyond is { } beyond && sequenceLength <= beyond)
        {
            scaling = null;
        }
        if (scaling is { Type: "ntk" })
        {
            theta *= Math.Pow(scaling.Factor, width / (width - 2.0));
            scaling = null;
        }
        if (scaling is { Type: "dynamic" })
        {
            double max = scaling.Threshold;
            double length = Math.Max(sequenceLength, max);
            theta *= Math.Pow(scaling.Factor * length / max - (scaling.Factor - 1), width / (width - 2.0));
            scaling = null;
        }
        int pairs = width / 2;
        var inv = new double[pairs];
        for (int i = 0; i < pairs; i++)
        {
            inv[i] = 1.0 / Math.Pow(theta, 2.0 * i / width);
        }
        return scaling is null ? (inv, 1.0) : scaling.Apply(inv, theta, width, sequenceLength);
    }

    private (double[], double) Apply(double[] inv, double theta, int width, long sequenceLength)
    {
        switch (Type)
        {
            case "longrope":
            {
                double original = Threshold;
                var factors = sequenceLength > original
                    ? LongFactor ?? throw Missing("long_factor")
                    : ShortFactor ?? throw Missing("short_factor");
                if (factors.Length != inv.Length)
                {
                    throw new InvalidOperationException(
                        $"rope scaling 'longrope' has {factors.Length} factors for {inv.Length} rotary pairs");
                }
                double attention = AttentionFactor
                    ?? (Factor <= 1 ? 1.0 : Math.Sqrt(1 + Math.Log(Factor) / Math.Log(original)));
                return (inv.Select((f, i) => f / factors[i]).ToArray(), attention);
            }
            case "linear":
                return (inv.Select(f => f / Factor).ToArray(), 1.0);
            case "llama3":
            {
                double original = OriginalMaxPositions ?? throw Missing("original_max_position_embeddings");
                double low = LowFreqFactor ?? throw Missing("low_freq_factor");
                double high = HighFreqFactor ?? throw Missing("high_freq_factor");
                double lowWavelen = original / low, highWavelen = original / high;
                var result = new double[inv.Length];
                for (int i = 0; i < inv.Length; i++)
                {
                    double wavelen = 2 * Math.PI / inv[i];
                    double f = wavelen > lowWavelen ? inv[i] / Factor : inv[i];
                    if (!(wavelen < highWavelen) && !(wavelen > lowWavelen))
                    {
                        double smooth = (original / wavelen - low) / (high - low);
                        f = (1 - smooth) * f / Factor + smooth * f;
                    }
                    result[i] = f;
                }
                return (result, 1.0);
            }
            case "yarn":
            {
                double original = OriginalMaxPositions ?? throw Missing("original_max_position_embeddings");
                static double GetMscale(double scale, double m = 1) => scale <= 1 ? 1.0 : 0.1 * m * Math.Log(scale) + 1.0;
                double attention = AttentionFactor
                    ?? (Mscale is { } ms && ms != 0 && MscaleAllDim is { } msa && msa != 0
                        ? GetMscale(Factor, ms) / GetMscale(Factor, msa)
                        : GetMscale(Factor));
                double betaFast = BetaFast is { } bf && bf != 0 ? bf : 32;
                double betaSlow = BetaSlow is { } bs && bs != 0 ? bs : 1;
                double CorrectionDim(double rotations) =>
                    width * Math.Log(original / (rotations * 2 * Math.PI)) / (2 * Math.Log(theta));
                double lowDim = CorrectionDim(betaFast), highDim = CorrectionDim(betaSlow);
                if (Truncate)
                {
                    lowDim = Math.Floor(lowDim);
                    highDim = Math.Ceiling(highDim);
                }
                lowDim = Math.Max(lowDim, 0);
                highDim = Math.Min(highDim, width - 1);
                if (lowDim == highDim)
                {
                    highDim += 0.001;
                }
                var result = new double[inv.Length];
                for (int i = 0; i < inv.Length; i++)
                {
                    double ramp = Math.Clamp((i - lowDim) / (highDim - lowDim), 0, 1);
                    double extrapolation = 1 - ramp;
                    result[i] = inv[i] / Factor * (1 - extrapolation) + inv[i] * extrapolation;
                }
                return (result, attention);
            }
            default:
                throw new InvalidOperationException($"rope scaling '{Type}' has no implementation");
        }
    }

    private Exception Missing(string key) =>
        new InvalidOperationException($"rope scaling '{Type}' requires '{key}'");

    /// <summary>Judges a checkpoint rope block (<c>rope_type</c>, <c>factor</c>, …);
    /// null for a type this build does not implement.</summary>
    public static RopeScaling? FromRopeBlock(JsonElement block, long maxPositionEmbeddings)
    {
        string type = block.TryGetProperty("rope_type", out var rt) ? rt.GetString() ?? "default"
            : block.TryGetProperty("type", out var t) ? t.GetString() ?? "default" : "default";
        // Early Phi-3 configs call longrope "su".
        type = type == "su" ? "longrope" : type;
        if (!Types.Contains(type))
        {
            return null;
        }
        double? Num(string key) => block.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;
        long? original = type == "dynamic"
            // transformers' dynamic NTK measures against max_position_embeddings itself.
            ? maxPositionEmbeddings
            : block.TryGetProperty("original_max_position_embeddings", out var o) && o.ValueKind == JsonValueKind.Number
                ? o.GetInt64()
                : type == "linear" ? null : maxPositionEmbeddings;
        double[]? List(string key) => block.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Array
            ? v.EnumerateArray().Select(e => e.GetDouble()).ToArray()
            : null;
        double factor = Num("factor") ?? (type is "yarn" or "longrope" && original is { } orig && orig > 0
            ? (double)maxPositionEmbeddings / orig
            : throw new InvalidOperationException($"rope scaling '{type}' declares no factor"));
        return new RopeScaling
        {
            Type = type,
            Factor = factor,
            LowFreqFactor = Num("low_freq_factor"),
            HighFreqFactor = Num("high_freq_factor"),
            OriginalMaxPositions = original,
            BetaFast = Num("beta_fast"),
            BetaSlow = Num("beta_slow"),
            AttentionFactor = Num("attention_factor"),
            Mscale = Num("mscale"),
            MscaleAllDim = Num("mscale_all_dim"),
            Truncate = !block.TryGetProperty("truncate", out var tr) || tr.ValueKind != JsonValueKind.False,
            ShortFactor = List("short_factor"),
            LongFactor = List("long_factor"),
        };
    }

    /// <summary>The block's keys, as a checkpoint config spells them.</summary>
    public void WriteConfigKeys(System.Text.Json.Nodes.JsonObject block)
    {
        if (EngageBeyond is not null || Type == "ntk")
        {
            throw new InvalidOperationException(
                "a context extension that only long sequences use has no checkpoint spelling; name the method (--rope-extend none, yarn, ntk or linear) to export one");
        }
        block["rope_type"] = Type;
        block["factor"] = Factor;
        if (LowFreqFactor is { } lf) block["low_freq_factor"] = lf;
        if (HighFreqFactor is { } hf) block["high_freq_factor"] = hf;
        if (OriginalMaxPositions is { } om) block["original_max_position_embeddings"] = om;
        if (BetaFast is { } bfa) block["beta_fast"] = bfa;
        if (BetaSlow is { } bsl) block["beta_slow"] = bsl;
        if (AttentionFactor is { } af) block["attention_factor"] = af;
        if (Mscale is { } m) block["mscale"] = m;
        if (MscaleAllDim is { } ma) block["mscale_all_dim"] = ma;
        if (!Truncate) block["truncate"] = false;
        if (ShortFactor is { } sf) block["short_factor"] = new System.Text.Json.Nodes.JsonArray(sf.Select(x => (System.Text.Json.Nodes.JsonNode?)x).ToArray());
        if (LongFactor is { } lf2) block["long_factor"] = new System.Text.Json.Nodes.JsonArray(lf2.Select(x => (System.Text.Json.Nodes.JsonNode?)x).ToArray());
    }

    internal void Write(Utf8JsonWriter writer)
    {
        writer.WriteStartObject("scaling");
        writer.WriteString("type", Type);
        writer.WriteNumber("factor", Factor);
        if (LowFreqFactor is { } lf) writer.WriteNumber("low_freq_factor", lf);
        if (HighFreqFactor is { } hf) writer.WriteNumber("high_freq_factor", hf);
        if (OriginalMaxPositions is { } om) writer.WriteNumber("original_max_position_embeddings", om);
        if (BetaFast is { } bfa) writer.WriteNumber("beta_fast", bfa);
        if (BetaSlow is { } bsl) writer.WriteNumber("beta_slow", bsl);
        if (AttentionFactor is { } af) writer.WriteNumber("attention_factor", af);
        if (Mscale is { } m) writer.WriteNumber("mscale", m);
        if (MscaleAllDim is { } ma) writer.WriteNumber("mscale_all_dim", ma);
        if (!Truncate) writer.WriteBoolean("truncate", false);
        if (EngageBeyond is { } eb) writer.WriteNumber("engage_beyond", eb);
        WriteList(writer, "short_factor", ShortFactor);
        WriteList(writer, "long_factor", LongFactor);
        writer.WriteEndObject();
    }

    private static void WriteList(Utf8JsonWriter writer, string name, double[]? values)
    {
        if (values is null)
        {
            return;
        }
        writer.WriteStartArray(name);
        foreach (var v in values)
        {
            writer.WriteNumberValue(v);
        }
        writer.WriteEndArray();
    }

    internal static RopeScaling? Read(JsonElement policy)
    {
        if (!policy.TryGetProperty("scaling", out var s) || s.ValueKind != JsonValueKind.Object)
        {
            return null;
        }
        double? Num(string key) => s.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;
        return new RopeScaling
        {
            Type = s.GetProperty("type").GetString()!,
            Factor = s.GetProperty("factor").GetDouble(),
            LowFreqFactor = Num("low_freq_factor"),
            HighFreqFactor = Num("high_freq_factor"),
            OriginalMaxPositions = s.TryGetProperty("original_max_position_embeddings", out var o) ? o.GetInt64() : null,
            BetaFast = Num("beta_fast"),
            BetaSlow = Num("beta_slow"),
            AttentionFactor = Num("attention_factor"),
            Mscale = Num("mscale"),
            MscaleAllDim = Num("mscale_all_dim"),
            Truncate = !s.TryGetProperty("truncate", out var tr) || tr.ValueKind != JsonValueKind.False,
            EngageBeyond = s.TryGetProperty("engage_beyond", out var eb) ? eb.GetInt64() : null,
            ShortFactor = s.TryGetProperty("short_factor", out var sf) ? sf.EnumerateArray().Select(e => e.GetDouble()).ToArray() : null,
            LongFactor = s.TryGetProperty("long_factor", out var lf2) ? lf2.EnumerateArray().Select(e => e.GetDouble()).ToArray() : null,
        };
    }
}

/// <summary>
/// Multimodal rotary (MRoPE, Qwen2-VL onward): each rotary pair takes its
/// position from one of three streams: temporal, height, width. Contiguous
/// sections (Qwen2-VL, Qwen2.5-VL: the first <c>section[0]</c> pairs read t,
/// the next <c>section[1]</c> read h, …) or interleaved (Qwen3-VL, Qwen3.5:
/// pairs cycle t, h, w until a stream's share is spent, the rest read t).
/// With identical streams, as for every text token, it is plain rotary.
/// Serialised as the position policy's <c>mrope</c> object.
/// </summary>
public sealed class RopeMultimodal
{
    public required int[] Section { get; init; }
    public bool Interleaved { get; init; }

    /// <summary>The stream (0 = t, 1 = h, 2 = w) rotary pair
    /// <paramref name="pair"/> reads its position from.</summary>
    public int StreamOf(int pair)
    {
        if (Interleaved)
        {
            // transformers' apply_interleaved_mrope: h takes pairs 1, 4, 7, …
            // below 3·section[1]; w takes 2, 5, 8, … below 3·section[2].
            for (int stream = 1; stream < Section.Length; stream++)
            {
                if (pair % 3 == stream && pair < 3 * Section[stream])
                {
                    return stream;
                }
            }
            return 0;
        }
        int end = 0;
        for (int stream = 0; stream < Section.Length; stream++)
        {
            end += Section[stream];
            if (pair < end)
            {
                return stream;
            }
        }
        return Section.Length - 1;
    }

    /// <summary>Judges a rope block's <c>mrope_section</c> /
    /// <c>mrope_interleaved</c>; null when it has none.</summary>
    public static RopeMultimodal? FromRopeBlock(JsonElement block)
    {
        if (block.ValueKind != JsonValueKind.Object ||
            !block.TryGetProperty("mrope_section", out var section) || section.ValueKind != JsonValueKind.Array)
        {
            return null;
        }
        return new RopeMultimodal
        {
            Section = section.EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            Interleaved = block.TryGetProperty("mrope_interleaved", out var i) && i.ValueKind == JsonValueKind.True,
        };
    }

    internal void Write(Utf8JsonWriter writer)
    {
        writer.WriteStartObject("mrope");
        writer.WriteStartArray("section");
        foreach (var n in Section)
        {
            writer.WriteNumberValue(n);
        }
        writer.WriteEndArray();
        if (Interleaved) writer.WriteBoolean("interleaved", true);
        writer.WriteEndObject();
    }

    internal static RopeMultimodal? Read(JsonElement policy) =>
        policy.TryGetProperty("mrope", out var m) && m.ValueKind == JsonValueKind.Object
            ? new RopeMultimodal
            {
                Section = m.GetProperty("section").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
                Interleaved = m.TryGetProperty("interleaved", out var i) && i.ValueKind == JsonValueKind.True,
            }
            : null;
}
