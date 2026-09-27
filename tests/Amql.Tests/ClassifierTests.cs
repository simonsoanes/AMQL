using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Cli;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// Sequence classification end to end (docs/classifier-models-jev.md, gaps
/// C1–C6): ingest carries the head, its bias, the label table, template and
/// pad token; <c>classify</c> reproduces transformers. <c>fixtures/qwen-cls-tiny</c>
/// is a small random Qwen3.5 classifier (GatedDeltaNet + gated attention) whose
/// goldens transformers computed over a right-padded batch pooled at
/// <c>mask.sum − 1</c> — so matching them, one unpadded input at a time, is the
/// mask-invariance gate as well as the numeric one. Regenerate with
/// <c>fixtures/make_qwen_cls_tiny.py</c>.
/// </summary>
public sealed class ClassifierTests : IClassFixture<ClassifierTests.TinyClassifier>
{
    public sealed class TinyClassifier : IDisposable
    {
        private readonly TempDir _dir = new();

        public TinyClassifier()
        {
            Report = ModelToContainer.Encode(Source, ContainerPath, "qwen-cls-tiny");
            Golden = JsonDocument.Parse(File.ReadAllText(Path.Combine(Source, "golden.json"))).RootElement.Clone();
        }

        public static string Source => Path.Combine(AppContext.BaseDirectory, "fixtures", "qwen-cls-tiny");
        public string Root => _dir.Path;
        public string ContainerPath => Path.Combine(_dir.Path, "c");
        public EncodeReport Report { get; }
        public JsonElement Golden { get; }

        public void Dispose() => _dir.Dispose();
    }

    private readonly TinyClassifier _fx;

    public ClassifierTests(TinyClassifier fx) => _fx = fx;

    [Fact]
    public void Ingest_Carries_Head_Bias_Labels_Template_And_Pad()
    {
        using var container = Vindex3Container.Open(_fx.ContainerPath);
        var surface = container.Graph!.Components.Single(c => c.Role == ComponentRole.PrimaryText).Execution!.Classifier!;

        Assert.Equal(3, surface.NumLabels);
        Assert.Equal(new[] { "contradiction", "entailment", "neutral" }, surface.Labels);
        Assert.Equal("single_label_classification", surface.ProblemType);
        Assert.Equal("Premise: {premise}\nHypothesis: {hypothesis}", surface.Template);
        Assert.True(surface.ScoreBias);
        Assert.NotNull(surface.PadTokenId);
        Assert.Equal(PoolingKind.Last, surface.Pooling.Kind);
        Assert.True(surface.Pooling.LastNonPad);

        var head = container.Graph.Objects.Single(o => o.Kind == ObjectKind.ClassifierHead);
        Assert.Equal("score", head.SourceBindings[0].TensorPrefix);
        using var store = container.CreateOperandStore();
        Assert.True(store.ContainsTensor(head.Id, "weight"));
        Assert.True(store.ContainsTensor(head.Id, "bias"));

        Assert.Contains("classifier.json", _fx.Report.AncillaryCopied);
        var mirror = JsonNode.Parse(File.ReadAllText(Path.Combine(_fx.ContainerPath, "classifier.json")))!;
        Assert.Equal("entailment", mirror["id2label"]!["1"]!.GetValue<string>());
    }

    [Fact]
    public void Classify_Matches_Transformers_On_A_Right_Padded_Batch()
    {
        using var container = Vindex3Container.Open(_fx.ContainerPath);
        using var store = container.CreateOperandStore();
        var service = ClassificationService.Load(container, store);
        var tokenizer = HfTokenizer.FromTokenizerFile(Path.Combine(_fx.ContainerPath, "tokenizer.json"));

        foreach (var g in _fx.Golden.EnumerateArray())
        {
            var input = ClassificationInput.Pair(g.GetProperty("premise").GetString()!, g.GetProperty("hypothesis").GetString()!);
            var expectedIds = g.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            Assert.Equal(expectedIds, tokenizer.EncodeForModel(service.FormatInput(input), 4096, out _));

            var result = service.Classify(input);
            var logits = result["logits"]!.AsArray().Select(n => n!.GetValue<double>()).ToArray();
            var expected = g.GetProperty("logits").EnumerateArray().Select(e => e.GetDouble()).ToArray();
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(Math.Abs(expected[i] - logits[i]) < 1e-4, $"logit {i}: {logits[i]} vs {expected[i]}");
            }
            var probs = g.GetProperty("probs").EnumerateArray().Select(e => e.GetDouble()).ToArray();
            string argmax = service.Labels[Array.IndexOf(probs, probs.Max())];
            Assert.Equal(argmax, result["label"]!.GetValue<string>());
            Assert.Equal(expectedIds.Length - 1, result["pooling"]!["token_index"]!.GetValue<int>());
        }
    }

    [Fact]
    public void Export_Regenerates_Labels_Pad_And_Bias_And_Reencodes_Identically()
    {
        string exported = Path.Combine(_fx.Root, "exported");
        using (var container = Vindex3Container.Open(_fx.ContainerPath))
        {
            ModelExporter.Export(container, exported, patch: null);
        }
        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(exported, "config.json")))!;
        Assert.Equal("neutral", config["id2label"]!["2"]!.GetValue<string>());
        Assert.Equal(1, config["label2id"]!["entailment"]!.GetValue<int>());
        Assert.NotNull(config["pad_token_id"]);
        Assert.Equal("Premise: {premise}\nHypothesis: {hypothesis}", config["nli_template"]!.GetValue<string>());

        var second = ModelToContainer.Encode(exported, Path.Combine(_fx.Root, "c2"), "qwen-cls-tiny");
        Assert.Equal(
            _fx.Report.Segments.ToDictionary(s => s.Key, s => s.Value.PayloadSha256Hex),
            second.Segments.ToDictionary(s => s.Key, s => s.Value.PayloadSha256Hex));
    }

    [Theory]
    [InlineData("id2label", """{"0":"a","1":"b"}""", "id2label names 2 labels")]
    [InlineData("id2label", """{"0":"a","1":"b","7":"c"}""", "not a unique string label")]
    [InlineData("label2id", """{"contradiction":0,"entailment":2,"neutral":1}""", "not the inverse of id2label")]
    [InlineData("problem_type", "\"ranking\"", "problem_type 'ranking'")]
    public void Ingest_Refuses_Inconsistent_Classifier_Facts(string key, string value, string message)
    {
        string dir = Path.Combine(_fx.Root, $"bad-{key}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        foreach (var f in Directory.GetFiles(TinyClassifier.Source))
        {
            File.Copy(f, Path.Combine(dir, Path.GetFileName(f)));
        }
        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(dir, "config.json")))!.AsObject();
        config[key] = JsonNode.Parse(value);
        File.WriteAllText(Path.Combine(dir, "config.json"), config.ToJsonString());

        var e = Assert.Throws<ModelConfigException>(() => ModelToContainer.Encode(dir, Path.Combine(dir, "out"), "bad"));
        Assert.Contains(message, e.Message);
    }

    [Fact]
    public void Pooling_Selects_The_Last_Real_Token_Or_The_Last_Row()
    {
        int[] padded = { 1, 1, 1, 0, 0 };
        Assert.Equal(2, ClassifierPooling.Index(new PoolingSurface { Kind = PoolingKind.Last, LastNonPad = true }, padded));
        Assert.Equal(4, ClassifierPooling.Index(new PoolingSurface { Kind = PoolingKind.Last, LastNonPad = false }, padded));
        Assert.Throws<UnsupportedOperatorException>(() =>
            ClassifierPooling.Index(new PoolingSurface { Kind = PoolingKind.Mean }, padded));
        Assert.Throws<UnsupportedOperatorException>(() => ClassifierPooling.Activate("ranking", new float[2]));

        var multi = ClassifierPooling.Activate("multi_label_classification", new[] { 0f, 2f });
        Assert.Equal(0.5, multi[0], 12);
        Assert.Equal(new[] { 1.5 }, ClassifierPooling.Activate("regression", new[] { 1.5f }));
    }

    [Fact]
    public void Template_Formats_Like_Python_And_Refuses_Unknown_Fields()
    {
        Assert.Equal("P: a | H: b {x}", ClassificationService.FormatTemplate("P: {premise} | H: {hypothesis} {{x}}", "a", "b"));
        Assert.Throws<ClassificationRequestException>(() => ClassificationService.FormatTemplate("{premise} {label}", "a", "b"));
    }
}
