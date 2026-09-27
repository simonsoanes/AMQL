using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Cli;

/// <summary>A request the decisions contract rejects — the API's 422.</summary>
public sealed class DecisionRequestException : Exception
{
    public DecisionRequestException(string message) : base(message) { }
}

/// <summary>
/// Answers TypeSafe-style decision requests (<c>POST /v1/decisions</c> on
/// jevai.org, <c>/v1/systemone</c> on openjev.sh) with a Von container:
/// <code>
///   { "model"?: string, "state": string | object | array,
///     "questions": { id: { "type": "choice" | "score" | "noul",
///                          "instructions": string | object | array,
///                          "criteria"?: … } } }
/// </code>
/// and returns <c>{ model, answers: { id: … }, usage }</c>.
/// <para>
/// Each question is one encoder pass over <c>[CLS] instructions state [SEP]
/// [MASK] option₁ [MASK] option₂ … [SEP]</c>; the scorer turns each marker's
/// hidden state into a logit. How a request becomes that sequence — state
/// formatting, the noul yes/no framing and its zero-shot debias pass, score
/// levels, the input-conditioned temperature and the confidence margin — is a
/// line-for-line port of the Von 1.2 SDK's <c>OptionMarkerBackend</c>, because
/// any drift there changes the tokens and so the answer.
/// </para>
/// </summary>
public sealed class VonDecisionEngine
{
    public const int MaxChoiceOptions = 255;
    public const int MaxScoreLevels = 10;

    private readonly ModernBertEncoder _encoder;
    private readonly OptionMarkerScorer _scorer;
    private readonly HfTokenizer _tokenizer;
    private readonly int _maskId;
    private readonly string _maskToken;
    private readonly string _sepToken;
    private readonly Calibration _calibration;

    /// <summary>Force the attention mode instead of taking it from the
    /// checkpoint: <c>true</c> = order-invariant (1.2), <c>false</c> = full
    /// cross-option attention (1.1). Weights trained in one mode should run in it.</summary>
    public bool IndependentOptions { get; set; }

    public bool DigitSplit { get; }

    /// <summary>The model id stamped on responses (Von reports the version it
    /// actually served, never the id the caller asked for).</summary>
    public string ModelId { get; }

    private sealed record Calibration(
        double Temperature,
        IReadOnlyDictionary<string, double>? Map,
        (double A, double B)? NoulPrior);

    private VonDecisionEngine(ModernBertEncoder encoder, OptionMarkerScorer scorer, HfTokenizer tokenizer,
        Calibration calibration, string modelId)
    {
        _encoder = encoder;
        _scorer = scorer;
        _tokenizer = tokenizer;
        _calibration = calibration;
        _maskToken = scorer.Facts.MarkerToken;
        _maskId = tokenizer.AddedTokens.FirstOrDefault(t => t.Content == _maskToken)?.Id
            ?? throw new CliException($"the tokenizer has no marker token '{_maskToken}'");
        _sepToken = tokenizer.AddedTokens.FirstOrDefault(t => t.Content == "[SEP]")?.Content
            ?? throw new CliException("the tokenizer has no [SEP] token");
        IndependentOptions = scorer.Facts.IndependentOptions;
        DigitSplit = scorer.Facts.DigitSplit;
        ModelId = modelId;
    }

    public static VonDecisionEngine Load(Vindex3Container container)
    {
        if (!ModernBertEncoder.Serves(container))
        {
            throw new CliException("this container is not a ModernBERT encoder — 'decide' serves Von decision models");
        }
        using var store = container.CreateOperandStore();
        var encoder = ModernBertEncoder.Load(container, store);
        var scorer = OptionMarkerScorer.Load(container, store);

        string tokenizerPath = Path.Combine(container.Root, "tokenizer.json");
        if (!File.Exists(tokenizerPath))
        {
            throw new CliException("the container carries no tokenizer.json — re-encode from a checkpoint that ships one");
        }
        var tokenizer = HfTokenizer.FromTokenizerFile(tokenizerPath);
        if (!tokenizer.HasSingleTemplate)
        {
            throw new CliException("tokenizer.json declares no [CLS] … [SEP] template post-processor");
        }

        var (calibration, modelId) = ReadCalibration(Path.Combine(container.Root, ModernBert.CalibrationFile));
        return new VonDecisionEngine(encoder, scorer, tokenizer, calibration, modelId ?? container.Index.Model);
    }

    /// <summary>Reads <c>marker_calibration.json</c> with the SDK's validation:
    /// a malformed map or prior is dropped (falling back to the scalar
    /// temperature / the flat 0.7 correction), never allowed to fail a request.</summary>
    private static (Calibration, string?) ReadCalibration(string path)
    {
        if (!File.Exists(path))
        {
            return (new Calibration(1.0, null, null), null);
        }
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
            var root = doc.RootElement;
            double temperature = root.TryGetProperty("temperature", out var t) && t.TryGetDouble(out double tv) ? tv : 1.0;
            string? modelId = root.TryGetProperty("model_id", out var id) && id.ValueKind == JsonValueKind.String ? id.GetString() : null;

            Dictionary<string, double>? map = null;
            if (root.TryGetProperty("calibration_map", out var m) && m.ValueKind == JsonValueKind.Object)
            {
                map = new Dictionary<string, double>(StringComparer.Ordinal);
                foreach (var p in m.EnumerateObject())
                {
                    if (!TryPyFloat(p.Value, out double v))
                    {
                        map = null;
                        break;
                    }
                    map[p.Name] = v;
                }
                if (map is not null)
                {
                    if (!new[] { "bias", "entropy", "log_tokens", "n_options" }.Any(map.ContainsKey))
                    {
                        map = null;
                    }
                    else
                    {
                        map.TryAdd("lo", 0.5);
                        map.TryAdd("hi", 12.0);
                        if (map["lo"] > map["hi"])
                        {
                            map = null;
                        }
                    }
                }
            }

            (double, double)? prior = null;
            if (root.TryGetProperty("noul_zero_shot_prior", out var np) && np.ValueKind == JsonValueKind.Object &&
                np.TryGetProperty("a", out var a) && np.TryGetProperty("b", out var b) &&
                TryPyFloat(a, out double av) && TryPyFloat(b, out double bv))
            {
                prior = (av, bv);
            }
            return (new Calibration(temperature, map, prior), modelId);
        }
        catch (JsonException)
        {
            // The SDK falls back to T=1.0 and no map when the file is unreadable.
            return (new Calibration(1.0, null, null), null);
        }
    }

    private static bool TryPyFloat(JsonElement e, out double value)
    {
        value = 0;
        return e.ValueKind switch
        {
            JsonValueKind.Number => e.TryGetDouble(out value),
            JsonValueKind.True => (value = 1) == 1,
            JsonValueKind.False => (value = 0) == 0,
            JsonValueKind.String => double.TryParse(e.GetString(), System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out value),
            _ => false,
        };
    }

    // ── request → response ──────────────────────────────────────────────

    /// <summary>Validates and answers one request. Throws
    /// <see cref="DecisionRequestException"/> for anything the API would 422.</summary>
    public JsonObject Decide(JsonElement request)
    {
        var questions = ValidateRequest(request);
        string state = FormatState(request.GetProperty("state"));

        var answers = new JsonObject();
        int totalQuestionChars = 0;
        foreach (var (id, q) in questions)
        {
            string instructions = Instructions(q.GetProperty("instructions"));
            totalQuestionChars += PyLen(instructions);
            answers[id] = q.GetProperty("type").GetString() switch
            {
                "choice" => AnswerChoice(state, instructions, q.GetProperty("criteria")),
                "noul" => AnswerNoul(state, instructions, q.TryGetProperty("criteria", out var c) ? c : default),
                "score" => AnswerScore(state, instructions, q.GetProperty("criteria")),
                _ => throw new InvalidOperationException(),
            };
        }

        return new JsonObject
        {
            ["model"] = ModelId,
            ["answers"] = answers,
            // The SDK's estimate: a quarter of the characters, not a token count.
            ["usage"] = new JsonObject
            {
                ["input_tokens"] = Math.Max(1, PyLen(state) / 4) + Math.Max(1, totalQuestionChars / 4),
                ["output_tokens"] = answers.Count,
            },
        };
    }

    /// <summary>Python's <c>len(str)</c>: code points, not UTF-16 units.</summary>
    private static int PyLen(string s) => s.EnumerateRunes().Count();

    private static List<(string Id, JsonElement Question)> ValidateRequest(JsonElement request)
    {
        if (request.ValueKind != JsonValueKind.Object)
        {
            throw new DecisionRequestException("the request body must be a JSON object");
        }
        if (request.TryGetProperty("model", out var model) && model.ValueKind is not (JsonValueKind.String or JsonValueKind.Null))
        {
            throw new DecisionRequestException("'model' must be a string");
        }
        if (!request.TryGetProperty("state", out var state) ||
            state.ValueKind is not (JsonValueKind.String or JsonValueKind.Object or JsonValueKind.Array))
        {
            throw new DecisionRequestException("'state' is required and must be a string, object or array");
        }
        if (!request.TryGetProperty("questions", out var qs) || qs.ValueKind != JsonValueKind.Object)
        {
            throw new DecisionRequestException("'questions' is required and must be an object of question ids to questions");
        }
        var questions = PyFormat.LastWins(qs);
        if (questions.Count == 0)
        {
            throw new DecisionRequestException("'questions' must not be empty");
        }

        foreach (var (id, q) in questions)
        {
            string where = $"questions.{id}";
            if (q.ValueKind != JsonValueKind.Object)
            {
                throw new DecisionRequestException($"{where} must be an object");
            }
            if (!q.TryGetProperty("type", out var type) || type.ValueKind != JsonValueKind.String ||
                type.GetString() is not ("choice" or "score" or "noul"))
            {
                throw new DecisionRequestException($"{where}.type must be \"choice\", \"score\" or \"noul\"");
            }
            if (!q.TryGetProperty("instructions", out var ins) ||
                ins.ValueKind is not (JsonValueKind.String or JsonValueKind.Object or JsonValueKind.Array))
            {
                throw new DecisionRequestException($"{where}.instructions is required and must be a string, object or array");
            }
            bool hasCriteria = q.TryGetProperty("criteria", out var criteria) && criteria.ValueKind != JsonValueKind.Null;
            switch (type.GetString())
            {
                case "choice":
                    if (!hasCriteria || criteria.ValueKind != JsonValueKind.Object)
                    {
                        throw new DecisionRequestException($"{where}.criteria must be an object of option names to descriptions");
                    }
                    int options = PyFormat.LastWins(criteria).Count;
                    if (options is 0 or > MaxChoiceOptions)
                    {
                        throw new DecisionRequestException($"{where}.criteria must have 1 to {MaxChoiceOptions} options, not {options}");
                    }
                    break;
                case "score":
                    if (!hasCriteria || criteria.ValueKind != JsonValueKind.Array)
                    {
                        throw new DecisionRequestException($"{where}.criteria must be an ordered array of levels");
                    }
                    int levels = criteria.GetArrayLength();
                    if (levels is 0 or > MaxScoreLevels)
                    {
                        throw new DecisionRequestException($"{where}.criteria must have 1 to {MaxScoreLevels} levels, not {levels}");
                    }
                    break;
                case "noul":
                    if (hasCriteria)
                    {
                        if (criteria.ValueKind != JsonValueKind.Object)
                        {
                            throw new DecisionRequestException($"{where}.criteria must be an object with 'true' and/or 'false'");
                        }
                        foreach (var p in criteria.EnumerateObject())
                        {
                            if (p.Name is not ("true" or "false"))
                            {
                                throw new DecisionRequestException($"{where}.criteria key '{p.Name}' — only 'true' and 'false' are allowed");
                            }
                        }
                    }
                    break;
            }
        }
        return questions;
    }

    /// <summary>The SDK's <c>_format_state</c>: a string as-is, an object as
    /// <c>"key: str(value)"</c> lines, anything else as Python's <c>str()</c>.</summary>
    public static string FormatState(JsonElement state) => state.ValueKind switch
    {
        JsonValueKind.String => state.GetString()!,
        JsonValueKind.Object => string.Join("\n", PyFormat.LastWins(state).Select(kv => $"{kv.Key}: {PyFormat.Str(kv.Value)}")),
        _ => PyFormat.Str(state),
    };

    /// <summary>Structured instructions become canonical JSON, keys sorted when
    /// the top level is an object — the SDK's <c>_stringify_instructions</c>.</summary>
    public static string Instructions(JsonElement instructions) => instructions.ValueKind == JsonValueKind.String
        ? instructions.GetString()!
        : PyFormat.JsonDumps(instructions, sortKeys: instructions.ValueKind == JsonValueKind.Object);

    /// <summary>A criteria description as text. Strings are used verbatim; the
    /// API also allows objects and arrays, which the model reads as the same
    /// canonical JSON structured instructions become.</summary>
    private static string? Description(JsonElement value) => value.ValueKind switch
    {
        JsonValueKind.Null or JsonValueKind.Undefined => null,
        JsonValueKind.String => value.GetString(),
        _ => PyFormat.JsonDumps(value, sortKeys: value.ValueKind == JsonValueKind.Object),
    };

    // ── the three primitives ────────────────────────────────────────────

    private JsonObject AnswerChoice(string state, string instructions, JsonElement criteria)
    {
        var options = PyFormat.LastWins(criteria);
        var descriptions = options.Select(o =>
        {
            string? d = Description(o.Value);
            return string.IsNullOrEmpty(d) ? PyFormat.Strip(o.Key) : PyFormat.Strip(d);
        }).ToList();

        var logits = OptionLogits(state, instructions, descriptions);
        var probs = Softmax(logits, EffectiveTemperature(logits, state, options.Count));
        int best = ArgMax(logits);

        var probabilities = new JsonObject();
        for (int i = 0; i < options.Count; i++)
        {
            probabilities[options[i].Key] = Round(probs[i], 4);
        }
        return new JsonObject
        {
            ["type"] = "choice",
            ["choice"] = options[best].Key,
            ["probabilities"] = probabilities,
            ["confidence"] = MarginConfidence(probs),
        };
    }

    private JsonObject AnswerNoul(string state, string instructions, JsonElement criteria)
    {
        string? pos = null, neg = null;
        if (criteria.ValueKind == JsonValueKind.Object)
        {
            foreach (var (key, value) in PyFormat.LastWins(criteria))
            {
                if (key == "true")
                {
                    pos = Description(value);
                }
                else
                {
                    neg = Description(value);
                }
            }
        }
        bool explicitCriteria = !string.IsNullOrEmpty(pos) || !string.IsNullOrEmpty(neg);
        if (string.IsNullOrEmpty(pos))
        {
            pos = "Yes, condition holds true.";
        }
        if (string.IsNullOrEmpty(neg))
        {
            neg = "No, condition is false.";
        }
        var descriptions = new List<string> { pos, neg };

        var logits = OptionLogits(state, instructions, descriptions);
        if (!explicitCriteria)
        {
            // Zero-shot debias: the model leans "yes" with no state at all, so
            // the same question against an empty state measures that lean and
            // it is subtracted from the yes logit.
            var nullLogits = OptionLogits("", instructions, descriptions);
            double bias = nullLogits[0] - nullLogits[1];
            double correction = _calibration.NoulPrior is var (a, b) ? a * bias + b : 0.7 * bias;
            logits = new[] { (float)(logits[0] - correction), logits[1] };
        }
        var probs = Softmax(logits, EffectiveTemperature(logits, state, 2));
        return new JsonObject
        {
            ["type"] = "noul",
            ["noul"] = Round(Math.Clamp(probs[0], 0.0, 1.0), 4),
        };
    }

    private JsonObject AnswerScore(string state, string instructions, JsonElement criteria)
    {
        var legend = new JsonObject();
        var descriptions = new List<string>();
        int i = 0;
        foreach (var level in criteria.EnumerateArray())
        {
            string desc = LevelDescription(level);
            legend[i.ToString(System.Globalization.CultureInfo.InvariantCulture)] = desc;
            descriptions.Add(desc);
            i++;
        }

        var logits = OptionLogits(state, instructions, descriptions);
        var probs = Softmax(logits, EffectiveTemperature(logits, state, descriptions.Count));
        var probabilities = new JsonObject();
        double expected = 0;
        for (int k = 0; k < probs.Length; k++)
        {
            probabilities[k.ToString(System.Globalization.CultureInfo.InvariantCulture)] = Round(probs[k], 4);
            expected += k * probs[k];
        }
        return new JsonObject
        {
            ["type"] = "score",
            ["score"] = Round(expected, 2),
            ["confidence"] = MarginConfidence(probs),
            ["legend"] = legend,
            ["probabilities"] = probabilities,
        };
    }

    /// <summary>A score level's text. A string is stripped; an object with a
    /// <c>what</c> field is rendered as the SDK does (<c>"what Examples: a, b"</c>).
    /// Any other structure is kept whole as canonical JSON — the SDK would read
    /// only <c>what</c>/<c>examples</c> and silently drop, say, a <c>meaning</c>
    /// field, leaving an empty level.</summary>
    private static string LevelDescription(JsonElement level)
    {
        if (level.ValueKind == JsonValueKind.String)
        {
            return PyFormat.Strip(level.GetString()!);
        }
        if (level.ValueKind == JsonValueKind.Object && level.TryGetProperty("what", out var what))
        {
            string whatText = what.ValueKind == JsonValueKind.String ? what.GetString()! : PyFormat.Str(what);
            string examples = level.TryGetProperty("examples", out var ex) && ex.ValueKind == JsonValueKind.Array &&
                              ex.GetArrayLength() > 0
                ? " Examples: " + string.Join(", ", ex.EnumerateArray().Select(PyFormat.Str))
                : string.Empty;
            return PyFormat.Strip(whatText + examples);
        }
        return PyFormat.JsonDumps(level, sortKeys: level.ValueKind == JsonValueKind.Object);
    }

    // ── one encoder pass ────────────────────────────────────────────────

    private static readonly Regex DigitRun = new(@"\d+", RegexOptions.Compiled);

    /// <summary>The SDK's <c>pack_sequence</c>: <c>"{instructions} {state}"</c>
    /// (stripped), then <c>[SEP]</c>, then <c>[MASK] option</c> for each option.</summary>
    public string Pack(string state, string instructions, IReadOnlyList<string> options)
    {
        string prefix = instructions.Length > 0 ? PyFormat.Strip($"{instructions} {state}") : PyFormat.Strip(state);
        string packed = $"{prefix} {_sepToken} " + string.Join(" ", options.Select(o => $"{_maskToken} {PyFormat.Strip(o)}"));
        return DigitSplit ? DigitRun.Replace(packed, m => string.Join(" ", m.Value.ToCharArray())) : packed;
    }

    /// <summary>One logit per option.</summary>
    public float[] OptionLogits(string state, string instructions, IReadOnlyList<string> options)
    {
        int[] ids = _tokenizer.EncodeWithSpecialTokens(Pack(state, instructions, options));
        var markers = new List<int>();
        for (int i = 0; i < ids.Length; i++)
        {
            if (ids[i] == _maskId)
            {
                markers.Add(i);
            }
        }
        if (markers.Count != options.Count)
        {
            // An option or the state that itself contains the marker text
            // would be scored as an extra option; the SDK would crash or
            // mis-score, so refuse it instead.
            throw new DecisionRequestException(
                $"the packed sequence has {markers.Count} '{_maskToken}' markers for {options.Count} options — " +
                $"the state, instructions or options must not contain '{_maskToken}'");
        }

        Tensor2D hidden;
        if (IndependentOptions)
        {
            var (positions, allowed) = IndependentOptionLayout(ids.Length, markers);
            hidden = _encoder.Forward(ids, positions, allowed);
        }
        else
        {
            hidden = _encoder.Forward(ids);
        }
        return _scorer.Score(hidden, markers);
    }

    /// <summary>
    /// Von 1.2's order-invariant layout (<c>build_independent_option_masks</c>
    /// and <c>build_option_invariant_position_ids</c>): option k spans from its
    /// marker to the next marker (the last one stops before the final [SEP]).
    /// Prefix tokens — everything else, the final [SEP] included — attend only
    /// to prefix tokens; option tokens attend to the prefix and their own span.
    /// Each option's positions restart at the prefix length, as if it were the
    /// only option present.
    /// </summary>
    public static (int[] Positions, bool[] Allowed) IndependentOptionLayout(int length, IReadOnlyList<int> markers)
    {
        int last = length - 1;
        var option = new int[length];
        Array.Fill(option, -1);
        var positions = new int[length];
        for (int i = 0; i < length; i++)
        {
            positions[i] = i;
        }
        int prefixLength = markers.Count > 0 ? markers[0] : 0;
        for (int k = 0; k < markers.Count; k++)
        {
            int start = markers[k];
            int end = k + 1 < markers.Count ? markers[k + 1] : last;
            for (int i = start; i < end; i++)
            {
                option[i] = k;
                positions[i] = prefixLength + (i - start);
            }
        }
        var allowed = new bool[length * length];
        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < length; j++)
            {
                bool queryPrefix = option[i] < 0, keyPrefix = option[j] < 0;
                allowed[i * length + j] = queryPrefix ? keyPrefix : keyPrefix || option[i] == option[j];
            }
        }
        return (positions, allowed);
    }

    // ── calibration ─────────────────────────────────────────────────────

    /// <summary>The SDK's input-conditioned temperature: a bounded linear
    /// function of the option distribution's normalised entropy, the state's
    /// token count and the option count. Monotonic, so it never moves the
    /// argmax — only how sure the answer claims to be.</summary>
    private double EffectiveTemperature(float[] logits, string state, int optionCount)
    {
        if (_calibration.Map is not { } map)
        {
            return _calibration.Temperature;
        }
        var probs = Softmax(logits, 1.0);
        double entropy = 0;
        if (probs.Length > 1)
        {
            foreach (double p in probs)
            {
                entropy -= p * Math.Log(Math.Max(p, 1e-12));
            }
            entropy /= Math.Log(probs.Length);
        }
        int tokens = Math.Max(_tokenizer.EncodeToIds(state).Count, 1);
        double raw = Get("bias") + Get("entropy") * entropy + Get("log_tokens") * (Math.Log10(tokens) / 4.0) +
                     Get("n_options") * (optionCount / 8.0);
        return Math.Min(map["hi"], Math.Max(map["lo"], raw));

        double Get(string key) => map.TryGetValue(key, out double v) ? v : 0.0;
    }

    private static double[] Softmax(float[] logits, double temperature)
    {
        double t = Math.Max(temperature, 1e-4);
        double max = logits.Max() / t;
        var e = logits.Select(l => Math.Exp(l / t - max)).ToArray();
        double sum = e.Sum();
        return e.Select(v => v / sum).ToArray();
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
        {
            if (v[i] > v[best])
            {
                best = i;
            }
        }
        return best;
    }

    /// <summary>TypeSafe's confidence: <c>(n·p_max − 1)/(n − 1)</c>, anchored to
    /// the n-way chance baseline; 1 when there is nothing to be unsure between.</summary>
    private static double MarginConfidence(double[] probs)
    {
        int n = probs.Length;
        if (n <= 1)
        {
            return 1.0;
        }
        double conf = (n * probs.Max() - 1) / (n - 1);
        return Round(Math.Clamp(conf, 0.0, 1.0), 3);
    }

    private static double Round(double v, int digits) => Math.Round(v, digits, MidpointRounding.ToEven);
}
