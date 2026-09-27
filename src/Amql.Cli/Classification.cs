using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Cli;

/// <summary>One thing to classify: a premise/hypothesis pair (formatted
/// through the recorded template) or a single text used verbatim.</summary>
public sealed record ClassificationInput(string? Premise, string? Hypothesis, string? Text)
{
    public static ClassificationInput Pair(string premise, string hypothesis) => new(premise, hypothesis, null);

    public static ClassificationInput Single(string text) => new(null, null, text);

    public bool IsPair => Premise is not null;
}

/// <summary>Classification input the container cannot honour — the API's 422.</summary>
public sealed class ClassificationRequestException : Exception
{
    public ClassificationRequestException(string message) : base(message) { }
}

/// <summary>
/// The whole classify path over a container: apply the recorded template,
/// tokenise with the container's own tokenizer, run <see cref="SequenceClassifier"/>,
/// and describe the result by the recorded label table. Shared by
/// <c>amql-cli classify</c> and the server.
/// </summary>
public sealed class ClassificationService
{
    /// <summary>The reference cross-encoder's default truncation length.</summary>
    public const int DefaultMaxTokens = 4096;

    private readonly SequenceClassifier _classifier;
    private readonly HfTokenizer _tokenizer;
    private readonly int _maxTokens;

    public ClassifierSurface Surface => _classifier.Surface;

    public string Model { get; }

    /// <summary>Label names by id; <c>LABEL_i</c> when the checkpoint named none, as HF does.</summary>
    public IReadOnlyList<string> Labels { get; }

    private ClassificationService(SequenceClassifier classifier, HfTokenizer tokenizer, int maxTokens, string model)
    {
        _classifier = classifier;
        _tokenizer = tokenizer;
        _maxTokens = maxTokens;
        Model = model;
        Labels = classifier.Surface.Labels
                 ?? Enumerable.Range(0, classifier.Surface.NumLabels).Select(i => $"LABEL_{i}").ToArray();
    }

    public static bool Serves(Vindex3Container container) =>
        container.Graph?.Components.FirstOrDefault(c => c.Role == ComponentRole.PrimaryText)?.Execution?.Classifier is not null;

    public static ClassificationService Load(Vindex3Container container, OperandStore store, WeightPatch? patch = null,
        int maxTokens = DefaultMaxTokens)
    {
        string tokenizerPath = Path.Combine(container.Root, "tokenizer.json");
        if (!File.Exists(tokenizerPath))
        {
            throw new CliException("the container carries no tokenizer.json — re-encode from a checkpoint that ships one");
        }
        var classifier = SequenceClassifier.Load(container, store, patch);
        return new ClassificationService(classifier, HfTokenizer.FromTokenizerFile(tokenizerPath), maxTokens, container.Index.Model);
    }

    /// <summary>The text the model reads for an input.</summary>
    public string FormatInput(ClassificationInput input)
    {
        if (!input.IsPair)
        {
            return input.Text ?? throw new ClassificationRequestException("an input needs a text or a premise and hypothesis");
        }
        string template = Surface.Template ?? throw new ClassificationRequestException(
            "the container records no input template, so a premise/hypothesis pair has no defined format — pass a single text");
        // The reference strips both sides before formatting.
        return FormatTemplate(template, PyFormat.Strip(input.Premise!), PyFormat.Strip(input.Hypothesis ?? string.Empty));
    }

    /// <summary>Python's <c>template.format(premise=…, hypothesis=…)</c>:
    /// <c>{premise}</c>/<c>{hypothesis}</c> substituted, <c>{{</c>/<c>}}</c>
    /// unescaped; any other field is refused rather than left in place.</summary>
    public static string FormatTemplate(string template, string premise, string hypothesis)
    {
        var sb = new StringBuilder(template.Length + premise.Length + hypothesis.Length);
        for (int i = 0; i < template.Length; i++)
        {
            char c = template[i];
            if (c == '{' && i + 1 < template.Length && template[i + 1] == '{')
            {
                sb.Append('{');
                i++;
            }
            else if (c == '}' && i + 1 < template.Length && template[i + 1] == '}')
            {
                sb.Append('}');
                i++;
            }
            else if (c == '{')
            {
                int close = template.IndexOf('}', i);
                string field = close < 0 ? template[(i + 1)..] : template[(i + 1)..close];
                sb.Append(field switch
                {
                    "premise" => premise,
                    "hypothesis" => hypothesis,
                    _ => throw new ClassificationRequestException(
                        $"the recorded template uses the field '{{{field}}}' — only {{premise}} and {{hypothesis}} can be filled"),
                });
                i = close;
            }
            else if (c == '}')
            {
                throw new ClassificationRequestException("the recorded template has an unmatched '}'");
            }
            else
            {
                sb.Append(c);
            }
        }
        return sb.ToString();
    }

    /// <summary>Classifies one input into a self-describing JSON object.</summary>
    public JsonObject Classify(ClassificationInput input)
    {
        string text = FormatInput(input);
        int[] ids = _tokenizer.EncodeForModel(text, _maxTokens, out bool truncated);
        if (ids.Length == 0)
        {
            throw new ClassificationRequestException("the input tokenises to nothing");
        }
        var result = _classifier.Classify(ids);

        var scores = new JsonObject();
        for (int i = 0; i < Labels.Count; i++)
        {
            scores[Labels[i]] = result.Scores[i];
        }

        var obj = new JsonObject();
        obj["input"] = input.IsPair
            ? new JsonObject { ["premise"] = input.Premise, ["hypothesis"] = input.Hypothesis }
            : new JsonObject { ["text"] = input.Text };
        switch (Surface.ProblemType)
        {
            case "single_label_classification":
                obj["label"] = Labels[ArgMax(result.Scores)];
                break;
            case "multi_label_classification":
                obj["labels"] = new JsonArray(Labels.Where((_, i) => result.Scores[i] > 0.5).Select(l => (JsonNode?)l).ToArray());
                break;
            default:
                obj["value"] = result.Scores.Length == 1 ? result.Scores[0] : null;
                break;
        }
        obj["scores"] = scores;
        obj["logits"] = new JsonArray(result.Logits.Select(l => (JsonNode?)(double)l).ToArray());
        obj["problem_type"] = Surface.ProblemType;
        obj["pooling"] = new JsonObject
        {
            ["kind"] = "last",
            ["last_non_pad"] = Surface.Pooling.LastNonPad,
            ["token_index"] = result.PooledIndex,
            ["tokens"] = result.Tokens,
            ["truncated"] = truncated,
        };
        if (input.IsPair)
        {
            obj["template"] = Surface.Template;
        }
        return obj;
    }

    /// <summary>One line for <c>--format labels</c>.</summary>
    public string LabelLine(JsonObject result) => Surface.ProblemType switch
    {
        "single_label_classification" => result["label"]!.GetValue<string>(),
        "multi_label_classification" => result["labels"]!.AsArray().Count == 0
            ? "(none)"
            : string.Join(",", result["labels"]!.AsArray().Select(n => n!.GetValue<string>())),
        _ => string.Join(",", result["scores"]!.AsObject().Select(kv => kv.Value!.GetValue<double>().ToString("R", CultureInfo.InvariantCulture))),
    };

    private static int ArgMax(double[] v)
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
}
