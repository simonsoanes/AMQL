using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Cli;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// The in-process ModernBERT encoder and <c>decide</c> against goldens produced
/// by the real stack: <c>fixtures/von-tiny</c> is a small random checkpoint in
/// Von's exact layout, and <c>golden.json</c> holds what transformers'
/// ModernBertModel computed for it (hidden states, marker logits, both
/// attention modes) and what the Von 1.2 SDK answered for a set of requests.
/// Regenerate with <c>fixtures/make_von_tiny.py</c>.
/// </summary>
public sealed class VonDecisionTests : IClassFixture<VonDecisionTests.TinyVon>
{
    public sealed class TinyVon : IDisposable
    {
        private readonly TempDir _dir = new();

        public TinyVon()
        {
            ModelToContainer.Encode(Source, ContainerPath, "von-tiny");
            Golden = JsonDocument.Parse(File.ReadAllText(Path.Combine(Source, "golden.json"))).RootElement.Clone();
        }

        public static string Source => Path.Combine(AppContext.BaseDirectory, "fixtures", "von-tiny");
        public string ContainerPath => Path.Combine(_dir.Path, "c");
        public JsonElement Golden { get; }

        public void Dispose() => _dir.Dispose();
    }

    private readonly TinyVon _von;

    public VonDecisionTests(TinyVon von) => _von = von;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Encoder_Matches_Transformers_Hidden_States_And_Marker_Logits(bool independent)
    {
        var golden = _von.Golden.GetProperty("encoder").EnumerateArray()
            .Single(e => e.GetProperty("independent").GetBoolean() == independent);
        int[] ids = golden.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        int[] markers = golden.GetProperty("markers").EnumerateArray().Select(e => e.GetInt32()).ToArray();

        using var container = Vindex3Container.Open(_von.ContainerPath);
        using var store = container.CreateOperandStore();
        var encoder = ModernBertEncoder.Load(container, store);
        var scorer = OptionMarkerScorer.Load(container, store);

        Tensor2D hidden;
        if (independent)
        {
            var (positions, allowed) = VonDecisionEngine.IndependentOptionLayout(ids.Length, markers);
            hidden = encoder.Forward(ids, positions, allowed);
        }
        else
        {
            hidden = encoder.Forward(ids);
        }

        double worst = 0;
        int row = 0;
        foreach (var expectedRow in golden.GetProperty("hidden").EnumerateArray())
        {
            int col = 0;
            foreach (var v in expectedRow.EnumerateArray())
            {
                worst = Math.Max(worst, Math.Abs(v.GetDouble() - hidden.Row(row)[col++]));
            }
            row++;
        }
        Assert.Equal(ids.Length, row);
        Assert.True(worst < 1e-4, $"max |Δhidden| {worst}");

        var logits = scorer.Score(hidden, markers);
        var expected = golden.GetProperty("logits").EnumerateArray().Select(e => e.GetDouble()).ToArray();
        Assert.Equal(expected.Length, logits.Length);
        for (int i = 0; i < logits.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - logits[i]) < 1e-4, $"logit {i}: {logits[i]} vs {expected[i]}");
        }
    }

    [Fact]
    public void Decide_Matches_The_Von_Sdk()
    {
        using var container = Vindex3Container.Open(_von.ContainerPath);
        var engine = VonDecisionEngine.Load(container);
        Assert.True(engine.IndependentOptions);     // from marker_calibration.json

        foreach (var golden in _von.Golden.GetProperty("decisions").EnumerateArray())
        {
            var response = engine.Decide(golden.GetProperty("request"));
            var expected = JsonNode.Parse(golden.GetProperty("response").GetRawText())!;

            Assert.Equal(expected["usage"]!.ToJsonString(), response["usage"]!.ToJsonString());
            foreach (var (id, answer) in expected["answers"]!.AsObject())
            {
                AssertAnswer(id, answer!, response["answers"]![id]!);
            }
        }
    }

    /// <summary>Exact on every string (choice, legend, type); floats within one
    /// unit of the reported rounding, since the SDK runs float32 torch.</summary>
    private static void AssertAnswer(string path, JsonNode expected, JsonNode actual)
    {
        switch (expected)
        {
            case JsonObject obj:
                Assert.Equal(obj.Select(p => p.Key).OrderBy(k => k), actual.AsObject().Select(p => p.Key).OrderBy(k => k));
                foreach (var (key, value) in obj)
                {
                    AssertAnswer($"{path}.{key}", value!, actual[key]!);
                }
                break;
            case JsonValue v when v.TryGetValue(out double d):
                Assert.True(Math.Abs(d - actual.GetValue<double>()) <= 1e-4, $"{path}: {actual} vs {d}");
                break;
            default:
                Assert.Equal(expected.ToJsonString(), actual.ToJsonString());
                break;
        }
    }

    [Theory]
    [InlineData("[]", "must be a JSON object")]
    [InlineData("""{"questions":{"q":{"type":"noul","instructions":"x"}}}""", "'state' is required")]
    [InlineData("""{"state":3,"questions":{"q":{"type":"noul","instructions":"x"}}}""", "'state' is required")]
    [InlineData("""{"state":"s","questions":{}}""", "must not be empty")]
    [InlineData("""{"state":"s","questions":{"q":{"type":"rank","instructions":"x"}}}""", "type must be")]
    [InlineData("""{"state":"s","questions":{"q":{"type":"noul"}}}""", "instructions is required")]
    [InlineData("""{"state":"s","questions":{"q":{"type":"choice","instructions":"x"}}}""", "criteria must be an object")]
    [InlineData("""{"state":"s","questions":{"q":{"type":"score","instructions":"x","criteria":["0","1","2","3","4","5","6","7","8","9","10"]}}}""", "1 to 10 levels")]
    [InlineData("""{"state":"s","questions":{"q":{"type":"noul","instructions":"x","criteria":{"maybe":"?"}}}}""", "only 'true' and 'false'")]
    [InlineData("""{"state":"s [MASK] t","questions":{"q":{"type":"noul","instructions":"x"}}}""", "must not contain '[MASK]'")]
    public void Decide_Rejects_What_The_Api_Would_422(string body, string message)
    {
        using var container = Vindex3Container.Open(_von.ContainerPath);
        var engine = VonDecisionEngine.Load(container);
        using var doc = JsonDocument.Parse(body);
        var e = Assert.Throws<DecisionRequestException>(() => engine.Decide(doc.RootElement));
        Assert.Contains(message, e.Message);
    }

    [Fact]
    public void Tokenizer_Gives_Lstrip_Markers_Their_Leading_Whitespace_And_Wraps_Cls_Sep()
    {
        var tok = HfTokenizer.FromTokenizerFile(Path.Combine(TinyVon.Source, "tokenizer.json"));
        int mask = tok.AddedTokens.Single(t => t.Content == "[MASK]").Id;
        int cls = tok.AddedTokens.Single(t => t.Content == "[CLS]").Id;
        int sep = tok.AddedTokens.Single(t => t.Content == "[SEP]").Id;

        // "x  [MASK] y": the two spaces belong to [MASK], so x is followed by it directly.
        var ids = tok.EncodeToIds("x  [MASK] y");
        Assert.Equal(tok.EncodeToIds("x")[0], ids[0]);
        Assert.Equal(mask, ids[1]);

        var wrapped = tok.EncodeWithSpecialTokens("x");
        Assert.Equal(cls, wrapped[0]);
        Assert.Equal(sep, wrapped[^1]);
    }

    [Theory]
    [InlineData("""{"a": 0.00005}""", "a: 5e-05")]
    [InlineData("""{"a": 0.0001}""", "a: 0.0001")]
    [InlineData("""{"a": 1e16}""", "a: 1e+16")]
    [InlineData("""{"a": 123.0, "b": 12345678901234567890}""", "a: 123.0\nb: 12345678901234567890")]
    [InlineData("""{"a": {"x": true, "y": null, "z": "it's"}}""", "a: {'x': True, 'y': None, 'z': \"it's\"}")]
    [InlineData("""{"a": ["q\n", "é😀", "\u0007"]}""", "a: ['q\\n', 'é😀', '\\x07']")]
    [InlineData("""[1, "two"]""", "[1, 'two']")]
    public void State_Formats_As_Python_Str(string state, string expected)
    {
        using var doc = JsonDocument.Parse(state);
        Assert.Equal(expected, VonDecisionEngine.FormatState(doc.RootElement));
    }

    [Theory]
    [InlineData("""{"b": 1, "a": ["é", 2.5]}""", """{"a": ["\u00e9", 2.5], "b": 1}""")]
    [InlineData("""["x", {"b": 1, "a": 2}]""", """["x", {"b": 1, "a": 2}]""")]
    [InlineData("""{"k": "😀"}""", """{"k": "\ud83d\ude00"}""")]
    public void Structured_Instructions_Serialise_As_Python_Json_Dumps(string instructions, string expected)
    {
        using var doc = JsonDocument.Parse(instructions);
        Assert.Equal(expected, VonDecisionEngine.Instructions(doc.RootElement));
    }

    [Fact]
    public void Encoder_Gemm_Matches_The_Reference_Kernel_Including_Remainders()
    {
        var rng = new Random(5);
        foreach (var (m, k, n) in new[] { (7, 16, 13), (4, 32, 2), (9, 64, 31), (3, 16, 5) })
        {
            var x = new Tensor2D(Enumerable.Range(0, m * k).Select(_ => (float)rng.NextDouble() - 0.5f).ToArray(), m, k);
            var w = new Tensor2D(Enumerable.Range(0, n * k).Select(_ => (float)rng.NextDouble() - 0.5f).ToArray(), n, k);
            var expected = TensorOps.MatMulTransposedB(x, w);
            var actual = EncoderGemm.MatMulTransposedB(x, w);
            for (int i = 0; i < expected.Data.Length; i++)
            {
                Assert.True(Math.Abs(expected.Data[i] - actual.Data[i]) < 1e-5, $"{m}x{k}x{n} at {i}");
            }
        }
    }

    [Fact]
    public void Gelu_Is_The_Exact_Erf_Form()
    {
        // Values from torch.nn.functional.gelu (approximate='none').
        Assert.Equal(0.0, Gelu.Exact(0f), 7);
        Assert.Equal(0.8413447, Gelu.Exact(1f), 6);
        Assert.Equal(-0.1586553, Gelu.Exact(-1f), 6);
        Assert.Equal(2.9959502, Gelu.Exact(3f), 6);
        Assert.Equal(-0.0040496, Gelu.Exact(-3f), 6);
    }
}
