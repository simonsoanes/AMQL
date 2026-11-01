using System.Runtime.InteropServices;
using System.Text.Json;
using Amql.Cli;
using Amql.Gguf;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// LFM3 import, inference and export against <see cref="SyntheticLfm3Checkpoint"/>.
/// The golden (<c>golden_lfm3_synth.json</c>) holds every position's logits from
/// an independent float32 PyTorch forward written from Liquid's
/// <c>modeling_lfm3.py</c> — conv width 7, a window-4 sliding layer and twelve
/// tokens, so both the conv history and the window are exercised.
/// </summary>
public class Lfm3Tests
{
    private static readonly string Golden = Path.Combine(AppContext.BaseDirectory, "golden_lfm3_synth.json");

    private static (int[] Ids, float[][] Logits) LoadGolden()
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(Golden));
        var root = doc.RootElement;
        var ids = root.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var logits = root.GetProperty("logits").EnumerateArray()
            .Select(r => r.EnumerateArray().Select(e => e.GetSingle()).ToArray())
            .ToArray();
        return (ids, logits);
    }

    private static string EncodeSynthetic(TempDir dir)
    {
        var ckpt = Path.Combine(dir.Path, "ckpt");
        SyntheticLfm3Checkpoint.Write(ckpt);
        var facts = ModelConfig.ReadTextFacts(Path.Combine(ckpt, "config.json"));
        using var inventory = HfInventory.Open(ckpt);
        var spec = ArchMapper.MapToContainerSpec("lfm3-test", facts, inventory, new ArchMapper.EncodeOptions());
        var containerDir = Path.Combine(dir.Path, "container");
        ContainerEncoder.Encode(containerDir, spec);
        return containerDir;
    }

    private static double WorstDiff(ReadOnlySpan<float> mine, float[] reference)
    {
        double worst = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(mine[i] - reference[i]));
        }
        return worst;
    }

    [Fact]
    public void Config_Facts_Carry_Sliding_Window_And_Sandwich()
    {
        using var dir = new TempDir();
        SyntheticLfm3Checkpoint.Write(dir.Path);
        var facts = ModelConfig.ReadTextFacts(Path.Combine(dir.Path, "config.json"));
        Assert.Equal("lfm3", facts.ModelType);
        Assert.Equal(SyntheticLfm3Checkpoint.SlidingWindow, facts.SlidingWindow);
        Assert.True(facts.UseSandwichNorm);
        Assert.Null(facts.WrapperConfig);
        Assert.Equal(SyntheticLfm3Checkpoint.LayerTypes, facts.LayerTypes);
    }

    [Fact]
    public void Plan_Judges_Every_Lfm3_Operator()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(EncodeSynthetic(dir));
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);

        Assert.Equal(SyntheticLfm3Checkpoint.ConvKernel, plan.Layers[0].Conv!.KernelSize);
        Assert.Null(plan.Layers[0].Attention);

        var full = plan.Layers[1].Attention!;
        Assert.Null(full.Window);
        var zeroTail = Assert.IsType<PositionZeroTailRope>(full.Position);
        Assert.Equal(0.5, zeroTail.RotaryFactor);
        Assert.NotNull(full.HeadGate);
        Assert.False(full.OutputGate);

        var sliding = plan.Layers[3].Attention!;
        Assert.Equal(SyntheticLfm3Checkpoint.SlidingWindow, sliding.Window);
        Assert.IsType<PositionRope>(sliding.Position);

        foreach (var layer in plan.Layers)
        {
            Assert.NotNull(layer.PostAttentionNorm);
            Assert.NotNull(layer.PostFfnNorm);
        }
    }

    [Fact]
    public void Prefill_Matches_Reference_Logits()
    {
        var (ids, golden) = LoadGolden();
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(EncodeSynthetic(dir));
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var runtime = new GenericRuntime(plan, store);

        var hidden = runtime.Embed(ids);
        var positions = Enumerable.Range(0, ids.Length).ToArray();
        for (int l = 0; l < plan.Layers.Count; l++)
        {
            hidden = runtime.RunLayerInternal(hidden, l, positions, positions, appendKv: true);
        }
        var logits = runtime.FinalNormAndHead(hidden);
        for (int t = 0; t < ids.Length; t++)
        {
            double worst = WorstDiff(logits.Row(t), golden[t]);
            Assert.True(worst < 1e-3, $"position {t}: worst logit diff {worst:G4}");
        }
    }

    [Fact]
    public void Decode_Step_By_Step_Matches_Reference_Logits()
    {
        // One token at a time: the conv history and the sliding window are
        // carried across calls rather than seen in one batch.
        var (ids, golden) = LoadGolden();
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(EncodeSynthetic(dir));
        using var store = container.CreateOperandStore();
        var session = new DecodeSession(Planner.Plan(container, "target", store), store);

        var logits = session.Prefill(ids[..3]);
        Assert.True(WorstDiff(logits.Row(0), golden[2]) < 1e-3);
        for (int t = 3; t < ids.Length; t++)
        {
            logits = session.Step(ids[t]);
            double worst = WorstDiff(logits.Row(0), golden[t]);
            Assert.True(worst < 1e-3, $"step {t}: worst logit diff {worst:G4}");
        }
    }

    [Fact]
    public void Export_Round_Trips_Tensors_And_Config()
    {
        using var dir = new TempDir();
        var ckpt = Path.Combine(dir.Path, "ckpt");
        using (var container = Vindex3Container.Open(EncodeSynthetic(dir)))
        {
            ModelExporter.Export(container, Path.Combine(dir.Path, "exported"), patch: null);
        }

        using var original = SafetensorsFile.Open(Path.Combine(ckpt, "model.safetensors"));
        using var exported = SafetensorsFile.Open(Path.Combine(dir.Path, "exported", "model.safetensors"));
        Assert.Equal(original.Tensors.Keys.Order(), exported.Tensors.Keys.Order());
        foreach (var name in original.Tensors.Keys)
        {
            Assert.Equal(original.ReadBytes(name), exported.ReadBytes(name));
        }

        var facts = ModelConfig.ReadTextFacts(Path.Combine(dir.Path, "exported", "config.json"));
        Assert.Equal("lfm3", facts.ModelType);
        Assert.Equal(SyntheticLfm3Checkpoint.LayerTypes, facts.LayerTypes);
        Assert.Equal(SyntheticLfm3Checkpoint.SlidingWindow, facts.SlidingWindow);
        Assert.True(facts.UseSandwichNorm);
        using var config = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir.Path, "exported", "config.json")));
        var root = config.RootElement;
        Assert.Equal(SyntheticLfm3Checkpoint.ConvKernel, root.GetProperty("conv_L_cache").GetInt32());
        Assert.Equal(0.5, root.GetProperty("rope_parameters").GetProperty("full_attention")
            .GetProperty("partial_rotary_factor").GetDouble());
        Assert.Equal("Lfm3ForCausalLM", root.GetProperty("architectures")[0].GetString());
    }

    [Fact]
    public void MoeIfy_Reads_The_Lfm_Feed_Forward_Names()
    {
        // moe-ify used to hard-code Qwen's mlp.{gate,up,down}_proj; LFM names
        // its FFN feed_forward.w1/w3/w2. The dense names now come from the plan.
        using var dir = new TempDir();
        var containerDir = EncodeSynthetic(dir);
        var outDir = Path.Combine(dir.Path, "moe");
        var report = Amql.Merge.MoeIfy.Transform(containerDir, outDir,
            clusterTokens: new[] { 3, 17, 250, 4, 99, 128 }, evalTokens: new[] { 7, 7, 42, 200 },
            experts: 2, topK: 1);
        Assert.Equal(SyntheticLfm3Checkpoint.Intermediate / 2, report.ExpertIntermediateSize);

        using var moe = Vindex3Container.Open(outDir);
        using var store = moe.CreateOperandStore();
        var plan = Planner.Plan(moe, "target", store);
        Assert.All(plan.Layers, l => Assert.NotNull(l.Ffn!.Routed));
        Assert.False(store.ContainsTensor("target.decoder_stack", "0.feed_forward.w1.weight"));
        Assert.True(store.ContainsTensor("target.decoder_stack", "0.mlp.experts.1.down_proj.weight"));
    }

    [Fact]
    public void Gguf_Writes_The_Lfm3_Layer_Table_And_Rope_Mask()
    {
        using var dir = new TempDir();
        var ckpt = Path.Combine(dir.Path, "ckpt");
        SyntheticLfm3Checkpoint.Write(ckpt);
        // A byte-level BPE vocabulary of one token per id, no merges.
        var vocab = Enumerable.Range(0, SyntheticLfm3Checkpoint.Vocab).ToDictionary(i => $"t{i}", i => i);
        File.WriteAllText(Path.Combine(ckpt, "tokenizer.json"), JsonSerializer.Serialize(new
        {
            model = new { type = "BPE", vocab, merges = Array.Empty<string>() },
            added_tokens = Array.Empty<object>(),
        }));

        var outFile = Path.Combine(dir.Path, "lfm3.gguf");
        var report = Lfm3Gguf.Convert(ckpt, outFile, "q8_0");
        Assert.Equal("lfm3", report.Architecture);

        using var gguf = GgufReader.Open(outFile);
        Assert.Equal("lfm3", gguf.Arch);
        Assert.Equal(GgufType.F32, gguf.GetTensor("blk.0.shortconv.conv.weight").Type);
        Assert.Equal(new long[] { SyntheticLfm3Checkpoint.ConvKernel, SyntheticLfm3Checkpoint.Hidden },
            gguf.GetTensor("blk.0.shortconv.conv.weight").Dims);
        Assert.Equal(GgufType.Q8_0, gguf.GetTensor("blk.1.attn_q.weight").Type);
        Assert.Equal(new long[] { SyntheticLfm3Checkpoint.Hidden, SyntheticLfm3Checkpoint.NumQHeads },
            gguf.GetTensor("blk.1.attn_gate.weight").Dims);
        Assert.True(gguf.Contains("blk.3.post_attention_norm.weight"));
        Assert.True(gguf.Contains("blk.3.post_ffw_norm.weight"));
        Assert.False(gguf.Contains("blk.0.attn_q.weight"));

        // Half the 8 rotary pairs rotate; the rest are masked with 1e30.
        var mask = MemoryMarshal.Cast<byte, float>(gguf.ReadBytes("rope_freqs.weight")).ToArray();
        Assert.Equal(new[] { 1f, 1f, 1f, 1f, 1e30f, 1e30f, 1e30f, 1e30f }, mask);
    }

    [Fact]
    public void ZeroTail_Rope_Rotates_Only_The_Leading_Pairs_With_Full_Dim_Frequencies()
    {
        const int headDim = 8;
        var x = new Tensor2D(Enumerable.Range(1, headDim).Select(i => (float)i).ToArray(), 1, headDim);
        Rope.Apply(x, 1, headDim, new PositionZeroTailRope { Theta = 100.0, RotaryFactor = 0.5 }, new[] { 3 });

        // Pairs (i, i+4); two rotated with inv_freq = 100^(-2i/8), two untouched.
        for (int i = 0; i < 2; i++)
        {
            double angle = 3 * Math.Pow(100.0, -2.0 * i / headDim);
            double a = i + 1, b = i + 5;
            Assert.Equal(a * Math.Cos(angle) - b * Math.Sin(angle), x.Data[i], 4);
            Assert.Equal(a * Math.Sin(angle) + b * Math.Cos(angle), x.Data[i + 4], 4);
        }
        Assert.Equal(3f, x.Data[2]);
        Assert.Equal(4f, x.Data[3]);
        Assert.Equal(7f, x.Data[6]);
        Assert.Equal(8f, x.Data[7]);
    }
}
