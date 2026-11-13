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
/// </list>
/// Serialised as the position policy's <c>scaling</c> object.
/// </summary>
public sealed class RopeScaling
{
    public static readonly IReadOnlyList<string> Types = new[] { "linear", "llama3", "yarn" };

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

    /// <summary>Inverse frequencies for a rotary window of <paramref name="width"/>
    /// dims at base <paramref name="theta"/>, and the factor cos/sin are
    /// multiplied by (1 except for yarn).</summary>
    public static (double[] InvFreq, double AttentionScale) Frequencies(double theta, int width, RopeScaling? scaling)
    {
        int pairs = width / 2;
        var inv = new double[pairs];
        for (int i = 0; i < pairs; i++)
        {
            inv[i] = 1.0 / Math.Pow(theta, 2.0 * i / width);
        }
        return scaling is null ? (inv, 1.0) : scaling.Apply(inv, theta, width);
    }

    private (double[], double) Apply(double[] inv, double theta, int width)
    {
        switch (Type)
        {
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
        if (!Types.Contains(type))
        {
            return null;
        }
        double? Num(string key) => block.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;
        long? original = block.TryGetProperty("original_max_position_embeddings", out var o) && o.ValueKind == JsonValueKind.Number
            ? o.GetInt64()
            : type == "linear" ? null : maxPositionEmbeddings;
        double factor = Num("factor") ?? (type == "yarn" && original is { } orig && orig > 0
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
        };
    }

    /// <summary>The block's keys, as a checkpoint config spells them.</summary>
    public void WriteConfigKeys(System.Text.Json.Nodes.JsonObject block)
    {
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
        writer.WriteEndObject();
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
        };
    }
}
