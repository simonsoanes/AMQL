using System.Text.Json.Nodes;
using Amql.Hf;
using Amql.Inference;
using Amql.Server;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>Running a model past its trained context length: what changes in
/// the graph by method and by the model's own position policy, and that
/// opening a container with an extension runs it.</summary>
public class ContextExtensionTests
{
    private static string Zoo => Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo");

    private static JsonObject Graph(string position, string extra = "") => JsonNode.Parse(
        "{\"components\":[{\"id\":\"target\",\"execution\":{\"context_length\":1000,\"attention\":{\"head_dim\":64}},\"attention\":[" +
        "{\"operator\":\"softmax\",\"position\":" + position + extra + "}," +
        "{\"operator\":\"linear_attention\",\"position\":" + position + "}]},{\"id\":\"vision\"}]}")!.AsObject();

    private static JsonObject Position(JsonObject graph, int layer) =>
        graph["components"]![0]!["attention"]![layer]!["position"]!.AsObject();

    private static long Length(JsonObject graph) =>
        graph["components"]![0]!["execution"]!["context_length"]!.GetValue<long>();

    private const string Plain = "{\"kind\":\"rope\",\"theta\":10000}";

    [Theory]
    [InlineData("1M", 1048576)]
    [InlineData("512K", 524288)]
    [InlineData("1.5k", 1536)]
    [InlineData("300000", 300000)]
    public void Lengths_Parse(string text, long expected)
    {
        Assert.True(ContextExtension.TryParseLength(text, out long n));
        Assert.Equal(expected, n);
    }

    [Theory]
    [InlineData("")]
    [InlineData("K")]
    [InlineData("-1")]
    [InlineData("lots")]
    [InlineData("99999M")]
    public void Bad_Lengths_Are_Refused(string text)
    {
        Assert.False(ContextExtension.TryParseLength(text, out _));
        Assert.Throws<FormatException>(() => ContextExtension.Parse(text, null));
    }

    [Fact]
    public void An_Unknown_Method_Is_Refused() =>
        Assert.Throws<FormatException>(() => ContextExtension.Parse("1M", "magic"));

    [Fact]
    public void Yarn_Is_Recorded_On_Softmax_Layers_Only()
    {
        var g = Graph(Plain);
        new ContextExtension(4000, "yarn").Apply(g);
        Assert.Equal(4000, Length(g));
        var s = Position(g, 0)["scaling"]!.AsObject();
        Assert.Equal("yarn", (string?)s["type"]);
        Assert.Equal(4.0, s["factor"]!.GetValue<double>());
        Assert.Equal(1000, s["original_max_position_embeddings"]!.GetValue<long>());
        Assert.Null(Position(g, 1)["scaling"]);
    }

    [Fact]
    public void Ntk_Grows_The_Base_Over_The_Rotary_Width()
    {
        var g = Graph("{\"kind\":\"partial_rope\",\"theta\":10000,\"rotary_factor\":0.25}");
        new ContextExtension(4000, "ntk").Apply(g);
        Assert.Equal(10000 * Math.Pow(4, 16 / 14.0), Position(g, 0)["theta"]!.GetValue<double>(), 6);
    }

    [Fact]
    public void None_Moves_The_Limit_And_Nothing_Else()
    {
        var g = Graph(Plain);
        new ContextExtension(4000, "none").Apply(g);
        Assert.Equal(4000, Length(g));
        Assert.Equal(Plain, Position(g, 0).ToJsonString());
    }

    [Fact]
    public void A_Shorter_Length_Only_Limits_The_Model()
    {
        var g = Graph(Plain);
        var notes = new ContextExtension(500, "yarn").Apply(g);
        Assert.Equal(500, Length(g));
        Assert.Null(Position(g, 0)["scaling"]);
        Assert.Contains("limited", Assert.Single(notes));
    }

    [Fact]
    public void A_Window_Inside_The_Trained_Length_Keeps_Its_Frequencies()
    {
        var g = Graph(Plain, ",\"window\":512");
        new ContextExtension(4000, "yarn").Apply(g);
        Assert.Null(Position(g, 0)["scaling"]);
    }

    [Theory]
    [InlineData("llama3", "yarn", 32.0)]
    [InlineData("llama3", "linear", 32.0)]
    [InlineData("yarn", "yarn", 32.0)]
    [InlineData("linear", "linear", 32.0)]
    [InlineData("dynamic", "yarn", 8.0)]
    public void A_Models_Own_Scaling_Is_Taken_Further(string type, string method, double factor)
    {
        var g = Graph("{\"kind\":\"rope\",\"theta\":10000,\"scaling\":{\"type\":\"" + type + "\",\"factor\":8,\"original_max_position_embeddings\":125}}");
        new ContextExtension(4000, method).Apply(g);
        var s = Position(g, 0)["scaling"]!;
        Assert.Equal(type, (string?)s["type"]);
        Assert.Equal(factor, s["factor"]!.GetValue<double>());
        Assert.Equal(125, s["original_max_position_embeddings"]!.GetValue<long>());
    }

    [Theory]
    [InlineData("{\"kind\":\"rope\",\"theta\":10000,\"scaling\":{\"type\":\"linear\",\"factor\":2}}", "yarn")]
    [InlineData("{\"kind\":\"rope\",\"theta\":10000,\"scaling\":{\"type\":\"yarn\",\"factor\":2}}", "ntk")]
    [InlineData("{\"kind\":\"zero_tail_rope\",\"theta\":10000,\"rotary_factor\":0.5}", "yarn")]
    public void What_Cannot_Be_Extended_Is_Refused(string position, string method) =>
        Assert.Throws<ContainerException>(() => new ContextExtension(4000, method).Apply(Graph(position)));

    [Fact]
    public void A_Container_Opens_With_The_Extension_And_Its_Files_Stay_As_They_Were()
    {
        using var dir = new TempDir();
        string c = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(Path.Combine(Zoo, "qwen3"), c, "qwen3", new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });
        string before = File.ReadAllText(Path.Combine(c, "system_graph.json"));
        long trained;
        using (var plain = Vindex3Container.Open(c))
        {
            trained = plain.Graph!.Component("target").Execution!.ContextLength!.Value;
            Assert.Empty(plain.ContextNotes);
        }
        using var longer = Vindex3Container.Open(c, new ContextExtension(trained * 4, "yarn"));
        Assert.Single(longer.ContextNotes);
        Assert.Equal(trained * 4, longer.Graph!.Component("target").Execution!.ContextLength);
        using var store = longer.CreateOperandStore();
        var plan = Planner.Plan(longer, "target", store);
        foreach (var l in plan.Layers.Where(l => l.Attention is not null))
        {
            var rope = Assert.IsType<PositionRope>(l.Attention!.Position);
            Assert.Equal("yarn", rope.Scaling!.Type);
            Assert.Equal(4.0, rope.Scaling.Factor);
        }
        Assert.Equal(before, File.ReadAllText(Path.Combine(c, "system_graph.json")));
    }

    [Fact]
    public void Auto_Extends_Only_Sequences_Past_The_Trained_Length()
    {
        var g = Graph(Plain);
        new ContextExtension(4000).Apply(g);
        var s = Position(g, 0)["scaling"]!;
        Assert.Equal(ContextExtension.AutoMethod, (string?)s["type"]);
        Assert.Equal(1000, s["engage_beyond"]!.GetValue<long>());
        var scaling = System.Text.Json.JsonSerializer.Deserialize<PositionPolicy>(Position(g, 0).ToJsonString()) is PositionRope r ? r.Scaling! : null!;
        Assert.True(scaling.DependsOnLength);
        var trained = RopeScaling.Frequencies(10000, 64, null);
        Assert.Equal(trained.InvFreq, RopeScaling.Frequencies(10000, 64, scaling, 1000).InvFreq);
        Assert.Equal(RopeScaling.Frequencies(10000 * Math.Pow(4, 64 / 62.0), 64, null).InvFreq, RopeScaling.Frequencies(10000, 64, scaling, 1001).InvFreq);
        // Nothing but AMQL has a spelling for it: exports refuse rather than drop it.
        Assert.Throws<InvalidOperationException>(() => scaling.WriteConfigKeys(new JsonObject()));
    }

    [Fact]
    public void Server_Config_Reads_Max_Context()
    {
        using var dir = new TempDir();
        string file = Path.Combine(dir.Path, "models.json");
        File.WriteAllText(file, "{\"models\":[{\"path\":\"a\",\"max_context\":\"1M\",\"rope_extend\":\"yarn\"},{\"path\":\"b\",\"max_context\":4096},{\"path\":\"c\"}]}");
        var models = ServerConfig.Read(file);
        Assert.Equal(new ContextExtension(1 << 20, "yarn"), models[0].Context);
        Assert.Equal(new ContextExtension(4096), models[1].Context);
        Assert.Null(models[2].Context);
        File.WriteAllText(file, "{\"models\":[{\"path\":\"a\",\"rope_extend\":\"yarn\"}]}");
        Assert.Throws<InvalidDataException>(() => ServerConfig.Read(file));
        File.WriteAllText(file, "{\"models\":[{\"path\":\"a\",\"max_context\":\"huge\"}]}");
        Assert.Throws<InvalidDataException>(() => ServerConfig.Read(file));
    }
}

/// <summary>Caches too long for one pass of attention are taken in segments;
/// the results must match the unsegmented kernels. Skips without a CUDA device.</summary>
[Collection("gpu")]
public class SegmentedAttentionTests
{
    private static string Zoo => Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo");

    [Theory]
    [InlineData("qwen3", "f32")]
    [InlineData("qwen3", "fp16")]
    [InlineData("qwen3", "nvfp4")]
    [InlineData("qwen3_5-gdn", "nvfp4")]
    [InlineData("gemma2", "f32")]
    [InlineData("olmo3", "fp16")]
    [InlineData("gemma3_text", "f32")]
    [InlineData("gemma4-wide", "fp16")]
    [InlineData("gemma4-wide", "nvfp4")]
    [InlineData("gemma4-moe", "f32")]
    public void Segments_Agree_With_One_Pass(string family, string kv)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        using var dir = new TempDir();
        string c = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(Path.Combine(Zoo, family), c, family, new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });
        using var container = Vindex3Container.Open(c);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var (ids, _) = ArchZooTests.Golden(family);
        var prompt = Enumerable.Range(0, 1300).Select(i => ids[(i * 7 + i / 50) % ids.Length]).ToArray();

        List<float[]> Run()
        {
            string saved = GpuEngine.RequestedKvFormat;
            GpuEngine.RequestedKvFormat = kv;
            try
            {
                using var host = new DecodeSession(plan, store);
                host.Prefill(new[] { ids[0] });
                var engine = host.DeviceEngine()!;
                engine.ExactPrefill = true;
                var outputs = new List<float[]>();
                var a = engine.NewSequence();
                var b = engine.NewSequence();
                outputs.Add(engine.Prefill(a, prompt.AsSpan(0, 900)));
                outputs.Add(engine.Prefill(a, prompt.AsSpan(900)));
                outputs.Add(engine.Prefill(b, prompt.AsSpan(0, 300)));
                for (int i = 0; i < 4; i++)
                {
                    // Decode alone, then both sequences in a batch.
                    outputs.Add(engine.Step(new[] { a }, new[] { prompt[i] })[0]);
                    outputs.AddRange(engine.Step(new[] { a, b }, new[] { prompt[i + 8], prompt[i + 16] }));
                }
                engine.Verify(a, prompt.AsSpan(40, 4));
                for (int r = 0; r < 4; r++)
                {
                    outputs.Add(engine.HostLogits(r).ToArray());
                }
                return outputs;
            }
            finally
            {
                GpuEngine.RequestedKvFormat = saved;
            }
        }

        var one = Run();
        List<float[]> many;
        GpuEngine.SetAttentionSegments(3, 100, 1024);
        try
        {
            many = Run();
        }
        finally
        {
            GpuEngine.SetAttentionSegments(0, 0, 0);
        }
        Assert.Equal(one.Count, many.Count);
        for (int i = 0; i < one.Count; i++)
        {
            double scale = Math.Max(1.0, one[i].Max(v => Math.Abs((double)v)));
            double worst = one[i].Zip(many[i], (x, y) => Math.Abs((double)x - y)).Max();
            Assert.True(worst <= 2e-4 * scale, $"{family} {kv} output {i}: worst diff {worst:G4} (largest logit {scale:G4})");
        }
    }
}

/// <summary>auto: one model, two sets of rotary frequencies. A sequence that
/// stays inside the trained length runs exactly as the model does without an
/// extension, one past it exactly as the static method does, together or
/// apart. Skips without a CUDA device.</summary>
[Collection("gpu")]
public class AdaptiveContextTests
{
    private static string Zoo => Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo");

    private sealed class Model : IDisposable
    {
        private readonly Vindex3Container _container;
        private readonly OperandStore _store;
        private readonly DecodeSession _host;
        public readonly GpuEngine Engine;

        public Model(string dir, ContextExtension? context)
        {
            _container = Vindex3Container.Open(dir, context);
            _store = _container.CreateOperandStore();
            _host = new DecodeSession(Planner.Plan(_container, "target", _store), _store);
            _host.Prefill(new[] { 1 });
            Engine = _host.DeviceEngine()!;
            Engine.ExactPrefill = true;
        }

        public void Dispose()
        {
            _host.Dispose();
            _store.Dispose();
            _container.Dispose();
        }
    }

    private static string Container(string root, string family, long trained)
    {
        string c = Path.Combine(root, "c");
        ModelToContainer.Encode(Path.Combine(Zoo, family), c, family, new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });
        string graph = Path.Combine(c, "system_graph.json");
        var tree = JsonNode.Parse(File.ReadAllText(graph))!.AsObject();
        tree["components"]![0]!["execution"]!["context_length"] = trained;
        File.WriteAllText(graph, tree.ToJsonString());
        return c;
    }

    [Theory]
    [InlineData("qwen3")]
    [InlineData("qwen3_5-gdn")]
    public void Short_Sequences_Are_Untouched_And_Long_Ones_Extended(string family)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        using var dir = new TempDir();
        string c = Container(dir.Path, family, 700);
        var (ids, _) = ArchZooTests.Golden(family);
        var shortPrompt = Enumerable.Range(0, 600).Select(i => ids[(i * 3 + 1) % ids.Length]).ToArray();
        var longPrompt = Enumerable.Range(0, 900).Select(i => ids[(i * 5 + 2) % ids.Length]).ToArray();

        float[] trainedShort, ntkLong, trainedStep, ntkStep;
        using (var plain = new Model(c, null))
        {
            var s = plain.Engine.NewSequence();
            trainedShort = plain.Engine.Prefill(s, shortPrompt);
            trainedStep = plain.Engine.Step(new[] { s }, new[] { 2 })[0];
            Assert.Null(plain.Engine.RopeExtendsBeyond);
        }
        using (var ntk = new Model(c, new ContextExtension(2800, ContextExtension.AutoMethod)))
        {
            var s = ntk.Engine.NewSequence();
            ntkLong = ntk.Engine.Prefill(s, longPrompt);
            ntkStep = ntk.Engine.Step(new[] { s }, new[] { 2 })[0];
            Assert.Null(ntk.Engine.RopeExtendsBeyond);
        }

        using var auto = new Model(c, new ContextExtension(2800));
        var engine = auto.Engine;
        Assert.Equal(700, engine.RopeExtendsBeyond);
        var a = engine.NewSequence();
        var b = engine.NewSequence();
        Assert.Equal(trainedShort, engine.Prefill(a, shortPrompt));
        Assert.Equal(ntkLong, engine.Prefill(b, longPrompt));
        Assert.False(a.Extended);
        Assert.True(b.Extended);

        // Alone through the captured graph, which holds each one's frequencies.
        engine.Mark(a);
        engine.Mark(b);
        Assert.Equal(trainedStep, engine.Step(new[] { a }, new[] { 2 })[0]);
        Assert.Equal(ntkStep, engine.Step(new[] { b }, new[] { 2 })[0]);

        // Together in one batch.
        engine.Rewind(a);
        engine.Rewind(b);
        var both = engine.Step(new[] { a, b }, new[] { 2, 2 });
        Close(trainedStep, both[0]);
        Close(ntkStep, both[1]);

        // A long prompt read in slices is planned for its whole length, and saved state keeps its kind.
        var d = engine.NewSequence();
        engine.Reserve(d, longPrompt.Length);
        engine.Prefill(d, longPrompt.AsSpan(0, 512));
        engine.Mark(d);
        Assert.Equal(ntkLong, engine.Prefill(d, longPrompt.AsSpan(512)));
        var state = new ChunkedBuffer();
        engine.SaveMark(d, state);
        state.Position = 0;
        var restored = engine.RestoreMark(state);
        Assert.True(restored.Extended);
        Assert.Equal(ntkLong, engine.Prefill(restored, longPrompt.AsSpan(512)));

        // Reset, the same sequence may be the other kind.
        engine.Reset(b);
        Assert.Equal(trainedShort, engine.Prefill(b, shortPrompt));
        Assert.Equal(trainedStep, engine.Step(new[] { b }, new[] { 2 })[0]);
    }

    private static void Close(float[] expected, float[] actual)
    {
        double scale = Math.Max(1.0, expected.Max(v => Math.Abs((double)v)));
        double worst = expected.Zip(actual, (x, y) => Math.Abs((double)x - y)).Max();
        Assert.True(worst <= 1e-4 * scale, $"worst diff {worst:G4} (largest logit {scale:G4})");
    }
}
