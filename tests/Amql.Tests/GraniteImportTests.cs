using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

public class GraniteImportTests
{
    // ── synthetic Granite checkpoint ────────────────────────────────────────
    // Llama-shaped tensor names with the four Granite scalings declared in
    // config.json, untied head, and no layer_types table (the family spells
    // none; every layer is full attention).

    private const int Hidden = 8;
    private const int Layers = 2;
    private const int NumQHeads = 2;
    private const int NumKvHeads = 1;
    private const int Intermediate = 16;
    private const int Vocab = 32;

    private static void WriteGraniteCheckpoint(string dir, double attentionMultiplier,
        double embeddingMultiplier, double residualMultiplier, double logitsScaling)
    {
        Directory.CreateDirectory(dir);
        var rng = new Random(7);

        float[] Matrix(int rows, int cols)
        {
            float scale = MathF.Sqrt(6.0f / (rows + cols));
            var data = new float[rows * cols];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
            }
            return data;
        }

        static byte[] F32(float[] values)
        {
            var b = new byte[values.Length * 4];
            Buffer.BlockCopy(values, 0, b, 0, b.Length);
            return b;
        }

        var config = new Dictionary<string, object>
        {
            ["architectures"] = new[] { "GraniteForCausalLM" },
            ["model_type"] = "granite",
            ["hidden_size"] = Hidden,
            ["num_hidden_layers"] = Layers,
            ["num_attention_heads"] = NumQHeads,
            ["num_key_value_heads"] = NumKvHeads,
            ["intermediate_size"] = Intermediate,
            ["hidden_act"] = "silu",
            ["rms_norm_eps"] = 1e-5,
            ["vocab_size"] = Vocab,
            ["tie_word_embeddings"] = false,
            ["max_position_embeddings"] = 128,
            ["attention_multiplier"] = attentionMultiplier,
            ["embedding_multiplier"] = embeddingMultiplier,
            ["residual_multiplier"] = residualMultiplier,
            ["logits_scaling"] = logitsScaling,
            ["rope_parameters"] = new Dictionary<string, object>
            {
                ["rope_type"] = "default",
                ["rope_theta"] = 10_000.0,
            },
        };
        File.WriteAllText(Path.Combine(dir, "config.json"),
            System.Text.Json.JsonSerializer.Serialize(config, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));

        int headDim = Hidden / NumQHeads;
        int qDim = NumQHeads * headDim;
        int kvDim = NumKvHeads * headDim;
        var tensors = new Dictionary<string, (long[], float[])>
        {
            ["model.embed_tokens.weight"] = (new long[] { Vocab, Hidden }, Matrix(Vocab, Hidden)),
            ["model.norm.weight"] = (new long[] { Hidden }, Enumerable.Repeat(1.0f, Hidden).ToArray()),
            ["lm_head.weight"] = (new long[] { Vocab, Hidden }, Matrix(Vocab, Hidden)),
        };
        for (int l = 0; l < Layers; l++)
        {
            string p = $"model.layers.{l}.";
            tensors[$"{p}input_layernorm.weight"] = (new long[] { Hidden }, Enumerable.Repeat(1.0f, Hidden).ToArray());
            tensors[$"{p}self_attn.q_proj.weight"] = (new long[] { qDim, Hidden }, Matrix(qDim, Hidden));
            tensors[$"{p}self_attn.k_proj.weight"] = (new long[] { kvDim, Hidden }, Matrix(kvDim, Hidden));
            tensors[$"{p}self_attn.v_proj.weight"] = (new long[] { kvDim, Hidden }, Matrix(kvDim, Hidden));
            tensors[$"{p}self_attn.o_proj.weight"] = (new long[] { qDim, Hidden }, Matrix(qDim, Hidden));
            tensors[$"{p}post_attention_layernorm.weight"] = (new long[] { Hidden }, Enumerable.Repeat(1.0f, Hidden).ToArray());
            tensors[$"{p}mlp.gate_proj.weight"] = (new long[] { Intermediate, Hidden }, Matrix(Intermediate, Hidden));
            tensors[$"{p}mlp.up_proj.weight"] = (new long[] { Intermediate, Hidden }, Matrix(Intermediate, Hidden));
            tensors[$"{p}mlp.down_proj.weight"] = (new long[] { Hidden, Intermediate }, Matrix(Hidden, Intermediate));
        }

        var payloads = tensors
            .Select(kv => new TensorPayload { Name = kv.Key, Dtype = Dtype.F32, Shape = kv.Value.Item1, Data = F32(kv.Value.Item2) })
            .ToList();
        SafetensorsWriter.Write(Path.Combine(dir, "model.safetensors"), payloads,
            new Dictionary<string, string> { ["format"] = "pt" });
    }

    [Fact]
    public void Granite_Config_Reads_Scalings_And_Suppresses_Identities()
    {
        using var dir = new TempDir();
        WriteGraniteCheckpoint(dir.Path, 0.02, 2.0, 0.5, 8.0);

        var facts = ModelConfig.ReadTextFacts(Path.Combine(dir.Path, "config.json"));
        Assert.Equal("granite", facts.ModelType);
        Assert.Equal(0.02, facts.AttentionMultiplier);
        Assert.Equal(2.0, facts.EmbeddingMultiplier);
        Assert.Equal(0.5, facts.ResidualMultiplier);
        Assert.Equal(8.0, facts.LogitsScaling);
        // No layer_types table: one synthesised full-attention entry per layer.
        Assert.Equal(Layers, facts.LayerTypes.Count);
        Assert.All(facts.LayerTypes, t => Assert.Equal("full_attention", t));

        // The shipped Granite-4.2-3B shape: only attention_multiplier departs
        // from the identity; the rest must read as absent, not as 1.0 noise.
        WriteGraniteCheckpoint(dir.Path, 0.015625, 1.0, 1.0, 1.0);
        facts = ModelConfig.ReadTextFacts(Path.Combine(dir.Path, "config.json"));
        Assert.Equal(0.015625, facts.AttentionMultiplier);
        Assert.Null(facts.EmbeddingMultiplier);
        Assert.Null(facts.ResidualMultiplier);
        Assert.Null(facts.LogitsScaling);
    }

    [Fact]
    public void Granite_Scalings_Reach_The_Execution_Surface()
    {
        using var dir = new TempDir();
        WriteGraniteCheckpoint(dir.Path, 0.02, 2.0, 0.5, 8.0);

        var facts = ModelConfig.ReadTextFacts(Path.Combine(dir.Path, "config.json"));
        using var inventory = HfInventory.Open(dir.Path);
        var spec = ArchMapper.MapToContainerSpec("granite-test", facts, inventory,
            new ArchMapper.EncodeOptions());

        var surface = spec.SystemGraph.Components[0].Execution!;
        Assert.Equal(0.02, surface.Attention!.ScoreScale);
        Assert.Equal(0.5, surface.ResidualScale);
        Assert.Equal(2.0, surface.Head!.EmbedScale);
        // logits_scaling divides; the head op multiplies by the reciprocal.
        Assert.Equal(0.125, surface.Head.OutputMultiplier);

        var containerDir = Path.Combine(dir.Path, "container");
        ContainerEncoder.Encode(containerDir, spec);

        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);

        Assert.Equal(0.5, plan.ResidualScale);
        Assert.Equal(2.0, plan.Embedding!.Scale);
        Assert.Equal(0.125, plan.Output!.Multiplier);
        Assert.Equal(0.02, plan.Layers[0].Attention!.ScoreScale);

        // The imported stack runs end to end.
        var session = new DecodeSession(plan, store);
        var logits = session.Prefill(new[] { 1, 2, 3 });
        Assert.Equal(Vocab, logits.Cols);
    }
}
