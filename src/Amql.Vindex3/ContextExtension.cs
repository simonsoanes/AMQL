using System.Globalization;
using System.Text.Json.Nodes;

namespace Amql.Vindex3;

/// <summary>
/// Running a model at a context length other than the one it was trained to.
/// The change is made to the system graph as a container opens (or written into
/// it), so everything that reads the graph sees one model: the planner, the
/// engine's rotary tables, the server's context limit and the exports.
/// <list type="bullet">
/// <item><c>none</c> — positions past the trained length rotate at the trained
/// frequencies; only the limit moves.</item>
/// <item><c>yarn</c> — YaRN: the frequencies whose wavelength the trained length
/// never covered are interpolated, the fast ones are kept.</item>
/// <item><c>ntk</c> — NTK-aware: the rotary base grows, so the slowest frequency
/// is interpolated fully and the fastest not at all.</item>
/// <item><c>linear</c> — position interpolation: every frequency divided.</item>
/// </list>
/// <c>auto</c> keeps the trained frequencies for a sequence that fits the
/// trained length, so nothing a model does today changes, and extends only the
/// sequences that go past it.
/// Only softmax layers hold positions: recurrent layers and layers without a
/// position policy are untouched, and so are sliding-window layers whose window
/// is inside the trained length (no distance they see is new to them).
/// </summary>
public sealed record ContextExtension(long Length, string Method = ContextExtension.Auto)
{
    public const string Auto = "auto";
    public static readonly IReadOnlyList<string> Methods = new[] { Auto, "none", "yarn", "ntk", "linear" };

    /// <summary>What <c>auto</c> does to a sequence past the trained length
    /// (docs/context-extension.md has the measurements).</summary>
    public const string AutoMethod = "ntk";

    /// <summary>YaRN's cos/sin multiplier; null is its own formula.</summary>
    public double? AttentionFactor { get; init; }

    /// <summary>The extension containers opened from now on run with, unless
    /// the call that opens one gives its own (<c>--max-context</c>).</summary>
    public static ContextExtension? Requested { get; set; }

    /// <summary>A length as people write it: 1048576, 1M, 512K.</summary>
    public static bool TryParseLength(string? text, out long length)
    {
        length = 0;
        text = text?.Trim();
        if (string.IsNullOrEmpty(text))
        {
            return false;
        }
        long unit = char.ToUpperInvariant(text[^1]) switch { 'K' => 1L << 10, 'M' => 1L << 20, _ => 1 };
        string number = unit == 1 ? text : text[..^1];
        if (!double.TryParse(number, NumberStyles.Float, CultureInfo.InvariantCulture, out double n) || n <= 0 || n * unit > int.MaxValue / 2)
        {
            return false;
        }
        length = (long)(n * unit);
        return length > 0;
    }

    /// <summary>The extension for a <c>--max-context</c> and a <c>--rope-extend</c> value.</summary>
    public static ContextExtension Parse(string length, string? method)
    {
        if (!TryParseLength(length, out long n))
        {
            throw new FormatException($"'{length}' is not a context length (use tokens, or K and M: 1048576, 512K, 1M)");
        }
        method = method?.Trim().ToLowerInvariant() ?? Auto;
        if (!Methods.Contains(method))
        {
            throw new FormatException($"unknown rope extension '{method}': use {string.Join(", ", Methods)}");
        }
        return new ContextExtension(n, method);
    }

    /// <summary>Applies the extension to a <c>system_graph.json</c> tree, in
    /// place. Returns what changed, one line per component.</summary>
    public IReadOnlyList<string> Apply(JsonObject graph)
    {
        string method = Method == Auto ? AutoMethod : Method;
        var changed = new List<string>();
        foreach (var node in graph["components"] as JsonArray ?? new JsonArray())
        {
            if (node is not JsonObject component || component["execution"] is not JsonObject surface ||
                component["attention"] is not JsonArray policies || surface["context_length"] is not JsonValue trainedNode)
            {
                continue;
            }
            long trained = trainedNode.GetValue<long>();
            if (trained == Length)
            {
                continue;
            }
            surface["context_length"] = Length;
            string id = (string?)component["id"] ?? "?";
            if (Length < trained)
            {
                changed.Add($"{id}: context limited to {Length:N0} tokens (the model has {trained:N0})");
                continue;
            }
            int layers = 0;
            string? how = null;
            if (method != "none")
            {
                foreach (var row in policies.OfType<JsonObject>())
                {
                    if ((string?)row["operator"] != "softmax" || row["position"] is not JsonObject position ||
                        (row["window"] is JsonValue w && w.GetValue<long>() <= trained))
                    {
                        continue;
                    }
                    how = Extend(position, trained, HeadDim(row, surface), method) ?? how;
                    layers++;
                }
            }
            changed.Add(method == "none" || how is null
                ? $"{id}: context raised from {trained:N0} to {Length:N0} tokens, rotary frequencies as trained"
                : $"{id}: context raised from {trained:N0} to {Length:N0} tokens, {how} on {layers} layers");
        }
        return changed;
    }

    private static int HeadDim(JsonObject row, JsonObject surface) =>
        row["geometry"]?["head_dim"]?.GetValue<int>() ?? surface["attention"]?["head_dim"]?.GetValue<int>() ?? 0;

    // Rewrites one position policy; null when the kind has nothing to rescale.
    private string? Extend(JsonObject position, long trained, int headDim, string method)
    {
        string kind = (string?)position["kind"] ?? "";
        if (kind is "none")
        {
            return null;
        }
        if (Method == Auto)
        {
            if (kind is not ("rope" or "partial_rope"))
            {
                return null;   // nothing here rescales it: its frequencies as trained
            }
            if ((string?)position["scaling"]?["type"] is { } own)
            {
                // Already scaled: its own scaling, taken further for every sequence.
                method = own is "llama3" or "yarn" or "linear" ? own : method;
            }
            else
            {
                double times = (double)Length / trained;
                position["scaling"] = new JsonObject { ["type"] = method, ["factor"] = times, ["engage_beyond"] = trained };
                return $"{method} factor {times:G6} for sequences past {trained:N0} tokens";
            }
        }
        if (kind is not ("rope" or "partial_rope"))
        {
            throw new ContainerException($"position policy '{kind}' has no context extension; use --rope-extend none");
        }
        double ratio = (double)Length / trained;
        var existing = position["scaling"] as JsonObject;
        string? type = (string?)existing?["type"];
        if (type is "dynamic" or "longrope")
        {
            return $"{type} scaling as it stands";   // already follows the sequence length
        }
        if (method == "ntk")
        {
            if (existing is not null)
            {
                throw new ContainerException($"the model's rope is already scaled ({type}); use --rope-extend yarn, linear or none");
            }
            double factor = kind == "partial_rope" ? position["rotary_factor"]!.GetValue<double>() : 1.0;
            int width = Math.Max(2, (int)(headDim * factor));
            if (headDim <= 0 || width <= 2)
            {
                throw new ContainerException("the rotary width is not known, so ntk cannot grow its base; use --rope-extend yarn");
            }
            double theta = position["theta"]!.GetValue<double>();
            position["theta"] = theta * Math.Pow(ratio, width / (width - 2.0));
            return $"ntk (base {theta:G6} to {position["theta"]!.GetValue<double>():G6})";
        }
        if (existing is not null && (type == "llama3" || type == method))
        {
            // The model's own scaling, taken further.
            double factor = existing["factor"]!.GetValue<double>() * ratio;
            existing["factor"] = factor;
            if (method == "yarn" && AttentionFactor is { } a)
            {
                existing["attention_factor"] = a;
            }
            return $"{type} factor {factor:G6}";
        }
        if (existing is not null)
        {
            throw new ContainerException($"the model's rope is already scaled ({type}), which {method} does not extend; use --rope-extend {(type == "yarn" ? "yarn" : "linear")} or none");
        }
        var scaling = new JsonObject { ["type"] = method, ["factor"] = ratio };
        if (method == "yarn")
        {
            scaling["original_max_position_embeddings"] = trained;
            if (AttentionFactor is { } a)
            {
                scaling["attention_factor"] = a;
            }
        }
        position["scaling"] = scaling;
        return $"{method} factor {ratio:G6}";
    }
}
