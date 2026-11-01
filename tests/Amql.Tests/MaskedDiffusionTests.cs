using System.Runtime.InteropServices;
using System.Text.Json;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Masked diffusion (dLLM A2D-Qwen3) import, bidirectional inference and the
/// MDLM sampler, against <see cref="SyntheticA2dQwen3Checkpoint"/>. The goldens
/// come from <c>scripts/diffusion/mdlm_ref.py</c>, a float32 PyTorch forward
/// written from the checkpoint's published modeling code (Qwen3 with the causal
/// mask removed) and checked against transformers' Qwen3 in causal mode:
/// <c>golden_a2d_qwen3_synth.json</c> holds every position's logits over a
/// sequence with masks between prompt tokens, <c>golden_a2d_qwen3_sample.json</c>
/// the full low-confidence commit trace of a two-block generation.
/// </summary>
public class MaskedDiffusionTests
{
    private static JsonElement LoadJson(string name) =>
        JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, name))).RootElement;

    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(x => x.GetInt32()).ToArray();

    private static string EncodeSynthetic(TempDir dir, string modelType = SyntheticA2dQwen3Checkpoint.Family)
    {
        var ckpt = Path.Combine(dir.Path, "ckpt");
        SyntheticA2dQwen3Checkpoint.Write(ckpt, modelType);
        var facts = ModelConfig.ReadTextFacts(Path.Combine(ckpt, "config.json"));
        using var inventory = HfInventory.Open(ckpt);
        var spec = ArchMapper.MapToContainerSpec("a2d-test", facts, inventory, new ArchMapper.EncodeOptions());
        var containerDir = Path.Combine(dir.Path, "container");
        ContainerEncoder.Encode(containerDir, spec);
        return containerDir;
    }

    [Fact]
    public void Config_Facts_Resolve_The_Mask_Token()
    {
        using var dir = new TempDir();
        SyntheticA2dQwen3Checkpoint.Write(dir.Path);
        var facts = ModelConfig.ReadTextFacts(Path.Combine(dir.Path, "config.json"));
        Assert.Equal("a2d-qwen3", facts.ModelType);
        Assert.Equal(SyntheticA2dQwen3Checkpoint.MaskTokenId, facts.MaskTokenId);
        Assert.Equal("A2DQwen3LMHeadModel", facts.Architecture);
        Assert.NotNull(facts.AutoMap);
    }

    [Fact]
    public void Graph_Records_Bidirectional_Attention_And_The_Objective()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(EncodeSynthetic(dir));
        var surface = container.Graph!.Component("target").Execution!;
        Assert.True(surface.Attention!.Bidirectional);
        Assert.Equal("masked_diffusion", surface.Diffusion!.Objective);
        Assert.Equal(SyntheticA2dQwen3Checkpoint.MaskTokenId, surface.Diffusion.MaskTokenId);

        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        Assert.Equal(SyntheticA2dQwen3Checkpoint.MaskTokenId, plan.Diffusion!.MaskTokenId);
        Assert.All(plan.Layers, l => Assert.False(l.Attention!.Causal));
    }

    [Fact]
    public void Unjudged_A2d_Family_Refuses()
    {
        using var dir = new TempDir();
        var ex = Assert.Throws<ModelConfigException>(() => EncodeSynthetic(dir, "a2d-llama"));
        Assert.Contains("a2d-llama", ex.Message);
    }

    [Fact]
    public void Step_Decoding_A_Diffusion_Plan_Refuses()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(EncodeSynthetic(dir));
        using var store = container.CreateOperandStore();
        var session = new DecodeSession(Planner.Plan(container, "target", store), store);
        Assert.Throws<UnsupportedOperatorException>(() => session.Step(5));
    }

    [Fact]
    public void Bidirectional_Forward_Matches_Reference_Logits()
    {
        var golden = LoadJson("golden_a2d_qwen3_synth.json");
        var ids = Ints(golden.GetProperty("ids"));
        var rows = golden.GetProperty("logits").EnumerateArray()
            .Select(r => r.EnumerateArray().Select(x => x.GetSingle()).ToArray()).ToArray();

        using var dir = new TempDir();
        using var container = Vindex3Container.Open(EncodeSynthetic(dir));
        using var store = container.CreateOperandStore();
        var sampler = new MaskedDiffusionSampler(new GenericRuntime(Planner.Plan(container, "target", store), store));

        var logits = sampler.Logits(ids, 0, ids.Length);
        for (int t = 0; t < ids.Length; t++)
        {
            double worst = 0;
            for (int v = 0; v < rows[t].Length; v++)
            {
                worst = Math.Max(worst, Math.Abs(logits.Row(t)[v] - rows[t][v]));
            }
            Assert.True(worst < 1e-3, $"position {t}: worst logit diff {worst:G4}");
        }

        // A row slice is the same rows, not a recomputation over fewer tokens.
        var slice = sampler.Logits(ids, 4, 3);
        for (int r = 0; r < 3; r++)
        {
            Assert.Equal(logits.Row(4 + r).ToArray(), slice.Row(r).ToArray());
        }
    }

    [Fact]
    public void Sampler_Reproduces_The_Reference_Commit_Trace()
    {
        var golden = LoadJson("golden_a2d_qwen3_sample.json");
        var prompt = Ints(golden.GetProperty("prompt_ids"));
        var options = new DiffusionOptions
        {
            GenLength = golden.GetProperty("gen").GetInt32(),
            Steps = golden.GetProperty("steps").GetInt32(),
            BlockLength = golden.GetProperty("block").GetInt32(),
        };

        using var dir = new TempDir();
        using var container = Vindex3Container.Open(EncodeSynthetic(dir));
        using var store = container.CreateOperandStore();
        var sampler = new MaskedDiffusionSampler(new GenericRuntime(Planner.Plan(container, "target", store), store));
        var result = sampler.Generate(prompt, options);

        Assert.Equal(Ints(golden.GetProperty("output_ids")), result.Generated);
        var trace = golden.GetProperty("commit_trace").EnumerateArray().Select(Ints).ToList();
        Assert.Equal(trace.Count, result.CommitTrace.Count);
        for (int s = 0; s < trace.Count; s++)
        {
            Assert.Equal(trace[s], result.CommitTrace[s]);
        }
        Assert.Equal(options.Steps, result.Forwards);
    }

    [Fact]
    public void Sampler_Stops_After_The_Block_Holding_A_Stop_Token()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(EncodeSynthetic(dir));
        using var store = container.CreateOperandStore();
        var sampler = new MaskedDiffusionSampler(new GenericRuntime(Planner.Plan(container, "target", store), store));
        var prompt = new[] { 5, 17, 99, 3 };
        var full = sampler.Generate(prompt, new DiffusionOptions { GenLength = 12, Steps = 6, BlockLength = 6 });

        // Whatever the first block committed, declaring one of its tokens a
        // stop token must end generation there with the same block content.
        int stop = full.Generated[0];
        var early = sampler.Generate(prompt, new DiffusionOptions
        {
            GenLength = 12, Steps = 6, BlockLength = 6, StopTokens = new HashSet<int> { stop },
        });
        Assert.True(early.StoppedEarly);
        Assert.Equal(3, early.Forwards);
        Assert.Equal(full.Generated[..6], early.Generated[..6]);
    }

    [Fact]
    public void Export_Restores_The_Diffusion_Identity_And_Remote_Code()
    {
        using var dir = new TempDir();
        var ckpt = Path.Combine(dir.Path, "ckpt");
        SyntheticA2dQwen3Checkpoint.Write(ckpt);
        File.WriteAllText(Path.Combine(ckpt, "modeling_qwen3.py"), "# carried, never run\n");
        var containerDir = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(ckpt, containerDir);

        var exported = Path.Combine(dir.Path, "exported");
        using (var container = Vindex3Container.Open(containerDir))
        {
            Amql.Cli.ModelExporter.Export(container, exported, patch: null);
        }
        using var config = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(exported, "config.json")));
        var root = config.RootElement;
        Assert.Equal("a2d-qwen3", root.GetProperty("model_type").GetString());
        Assert.Equal("A2DQwen3LMHeadModel", root.GetProperty("architectures")[0].GetString());
        Assert.Equal("modeling_qwen3.A2DQwen3LMHeadModel",
            root.GetProperty("auto_map").GetProperty("AutoModelForMaskedLM").GetString());
        Assert.Equal(1_000_000.0, root.GetProperty("rope_theta").GetDouble());
        Assert.Equal("# carried, never run\n", File.ReadAllText(Path.Combine(exported, "modeling_qwen3.py")));

        // Re-encoding the export judges it the same way.
        var facts = ModelConfig.ReadTextFacts(Path.Combine(exported, "config.json"));
        Assert.Equal(SyntheticA2dQwen3Checkpoint.MaskTokenId, facts.MaskTokenId);
        var reencoded = Path.Combine(dir.Path, "reencoded");
        ModelToContainer.Encode(exported, reencoded);
        using var again = Vindex3Container.Open(reencoded);
        Assert.Equal("masked_diffusion", again.Graph!.Component("target").Execution!.Diffusion!.Objective);
        Assert.IsType<PositionRope>(again.Graph.Component("target").Attention![0].Position);
        Assert.Equal(1_000_000.0, ((PositionRope)again.Graph.Component("target").Attention![0].Position).Theta);
    }

    [Fact]
    public void Legacy_Top_Level_Rope_Theta_Is_Read()
    {
        // Configs from before rope_parameters keep rope_theta at the top level;
        // it used to be ignored, so every such checkpoint ran at theta 10 000.
        using var dir = new TempDir();
        SyntheticA2dQwen3Checkpoint.Write(dir.Path);
        var facts = ModelConfig.ReadTextFacts(Path.Combine(dir.Path, "config.json"));
        Assert.Equal(1_000_000.0, facts.RopeParameters.GetProperty("rope_theta").GetDouble());
        Assert.Equal("default", facts.RopeParameters.GetProperty("rope_type").GetString());
    }

    [Fact]
    public void Gguf_Writes_Rnd1_With_One_Expert_And_No_Logit_Shift()
    {
        using var dir = new TempDir();
        var ckpt = Path.Combine(dir.Path, "ckpt");
        SyntheticA2dQwen3Checkpoint.Write(ckpt);
        var vocab = Enumerable.Range(0, SyntheticA2dQwen3Checkpoint.Vocab - 1).ToDictionary(i => $"t{i}", i => i);
        File.WriteAllText(Path.Combine(ckpt, "tokenizer.json"), JsonSerializer.Serialize(new
        {
            model = new { type = "BPE", vocab, merges = Array.Empty<string>() },
            added_tokens = new[] { new { id = SyntheticA2dQwen3Checkpoint.MaskTokenId, content = SyntheticA2dQwen3Checkpoint.MaskToken, special = true } },
        }));

        var outFile = Path.Combine(dir.Path, "a2d.gguf");
        Assert.True(Amql.Gguf.A2dQwen3Gguf.Applies(ckpt));
        var report = Amql.Gguf.A2dQwen3Gguf.Convert(ckpt, outFile, "q8_0");
        Assert.Equal("rnd1", report.Architecture);

        using var gguf = Amql.Gguf.GgufReader.Open(outFile);
        Assert.Equal("rnd1", gguf.Arch);
        Assert.False(gguf.Get("rnd1.attention.causal").AsBool());
        Assert.False(gguf.Get("diffusion.shift_logits").AsBool());
        Assert.Equal((uint)SyntheticA2dQwen3Checkpoint.MaskTokenId, gguf.Get("tokenizer.ggml.mask_token_id").AsUInt32());
        Assert.Equal(1u, gguf.Get("rnd1.expert_count").AsUInt32());
        Assert.Equal(1u, gguf.Get("rnd1.expert_used_count").AsUInt32());
        Assert.Equal(1_000_000f, gguf.Get("rnd1.rope.freq_base").AsFloat());

        // ggml order: [hidden, ff, experts] for gate/up, [ff, hidden, experts] for down.
        int h = SyntheticA2dQwen3Checkpoint.Hidden, ff = SyntheticA2dQwen3Checkpoint.Intermediate;
        Assert.Equal(new long[] { h, ff, 1 }, gguf.GetTensor("blk.0.ffn_gate_exps.weight").Dims);
        Assert.Equal(new long[] { ff, h, 1 }, gguf.GetTensor("blk.1.ffn_down_exps.weight").Dims);
        Assert.Equal(Amql.Gguf.GgufType.Q8_0, gguf.GetTensor("blk.0.ffn_up_exps.weight").Type);
        Assert.Equal(new long[] { h, 1 }, gguf.GetTensor("blk.0.ffn_gate_inp.weight").Dims);
        Assert.All(MemoryMarshal.Cast<byte, float>(gguf.ReadBytes("blk.0.ffn_gate_inp.weight")).ToArray(), v => Assert.Equal(0f, v));
        Assert.True(gguf.Contains("blk.1.attn_q_norm.weight"));
        Assert.False(gguf.Contains("output.weight")); // tied
    }

    /// <summary>The published 0.6B checkpoint, when a container of it is to
    /// hand (<c>AMQL_MDLM_CONTAINER</c>): logits over the chat prompt plus 32
    /// masks against the reference, at a strided sample of the vocabulary.</summary>
    [Fact]
    public void Real_Checkpoint_Matches_Reference_When_Available()
    {
        if (Environment.GetEnvironmentVariable("AMQL_MDLM_CONTAINER") is not { Length: > 0 } containerDir)
        {
            return;
        }
        var golden = LoadJson(Path.Combine("fixtures", "golden_mdlm_real.json"));
        var ids = Ints(golden.GetProperty("ids"));
        int stride = golden.GetProperty("vocab_stride").GetInt32();
        var argmax = Ints(golden.GetProperty("argmax"));
        var rows = golden.GetProperty("logits").EnumerateArray()
            .Select(r => r.EnumerateArray().Select(x => x.GetSingle()).ToArray()).ToArray();

        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var sampler = new MaskedDiffusionSampler(new GenericRuntime(Planner.Plan(container, "target", store), store));
        var logits = sampler.Logits(ids, 0, ids.Length);

        int agree = 0;
        double worst = 0;
        for (int t = 0; t < ids.Length; t++)
        {
            var row = logits.Row(t);
            int best = 0;
            for (int v = 1; v < row.Length; v++)
            {
                if (row[v] > row[best])
                {
                    best = v;
                }
            }
            agree += best == argmax[t] ? 1 : 0;
            for (int i = 0; i < rows[t].Length; i++)
            {
                worst = Math.Max(worst, Math.Abs(row[i * stride] - rows[t][i]));
            }
        }
        Assert.True(agree == ids.Length, $"argmax agrees at {agree}/{ids.Length} positions");
        Assert.True(worst < 0.05, $"worst sampled logit diff {worst:G4}");
    }
}
