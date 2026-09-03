using System.Text.Json;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Tests for the G0→G3 loader (config facts, architecture mapping, encode,
/// integrity, planner boundary) on a small synthetic multimodal checkpoint
/// that mirrors the real Qwen3.5-0.8B layout — plus a guarded test reading
/// the actual checkpoint's config when it is present on this machine.
/// </summary>
public class HfTests
{
    private static readonly string RealModelDir = @"D:\Models\Qwen3.5-0.8B";
    private static bool RealModelConfigPresent => File.Exists(Path.Combine(RealModelDir, "config.json"));

    // ── guarded reality check: the real checkpoint's config ────────────────

    [Fact]
    public void Real_Qwen35_Config_Facts()
    {
        if (!RealModelConfigPresent)
        {
            return; // machine without the checkpoint: nothing to check
        }

        var facts = ModelConfig.ReadTextFacts(Path.Combine(RealModelDir, "config.json"));
        Assert.Equal("qwen3_5_text", facts.ModelType);
        Assert.Equal(1024, facts.HiddenSize);
        Assert.Equal(24, facts.NumLayers);
        Assert.Equal(8, facts.NumQueryHeads);
        Assert.Equal(2, facts.NumKvHeads);
        Assert.Equal(256, facts.HeadDim);
        Assert.Equal(3584, facts.IntermediateSize);
        Assert.Equal(1e-6, facts.RmsNormEps, 10);
        Assert.Equal(248320, facts.VocabSize);
        Assert.True(facts.TieWordEmbeddings);
        Assert.True(facts.AttentionOutputGate);
        Assert.False(facts.AttentionBias);
        Assert.Equal(262_144, facts.MaxPositionEmbeddings);
        Assert.Equal(24, facts.LayerTypes.Count);
        Assert.Equal(18, facts.LayerTypes.Count(t => t == "linear_attention"));
        Assert.Equal(6, facts.LayerTypes.Count(t => t == "full_attention"));
        Assert.Equal(0.25, facts.PartialRotaryFactor, 6);
        Assert.NotNull(facts.LinearAttention);
        Assert.Equal(4, facts.LinearAttention!.ConvKernelDim);
        Assert.Equal(16, facts.LinearAttention.KeyHeads);
        Assert.Equal(128, facts.LinearAttention.KeyHeadDim);
        Assert.Equal(16, facts.LinearAttention.ValueHeads);
        Assert.Equal(128, facts.LinearAttention.ValueHeadDim);
    }

    // ── synthetic multimodal checkpoint → container ────────────────────────

    [Fact]
    public void Multimodal_Checkpoint_Encodes_EndToEnd()
    {
        using var dir = new TempDir();
        WriteSyntheticCheckpoint(dir.Path);

        var containerPath = Path.Combine(dir.Path, "container");
        var report = ModelToContainer.Encode(dir.Path, containerPath, modelId: "synth-qwen3.5");

        Assert.Equal("synth-qwen3.5", report.ModelId);
        Assert.Equal("F16", report.Encoding); // canonical = shard majority

        using var container = Vindex3Container.Open(containerPath);
        var index = container.Index;
        Assert.Equal(Vindex3Index.CurrentSchema, index.Version);
        Assert.Equal(3, index.Representations.Count);
        Assert.NotNull(index.PrecisionMap);
        Assert.Contains("0.linear_attn.A_log", index.PrecisionMap!.Exceptions);
        Assert.Contains("0.linear_attn.norm.weight", index.PrecisionMap.Exceptions);

        var graph = container.Graph!;
        Assert.Equal(3, graph.Components.Count); // target + vision + mtp
        var target = graph.Component("target");
        Assert.Equal(3, target.Attention!.Count);
        Assert.Equal(LayerOperators.LinearAttention, target.Attention[0].Operator);
        Assert.Equal(LayerOperators.Softmax, target.Attention[1].Operator);
        Assert.Equal(LayerOperators.Softmax, target.Attention[2].Operator);
        Assert.IsType<PositionUnresolved>(target.Attention[0].Position);
        Assert.Equal("partial_mrope", ((PositionUnresolved)target.Attention[0].Position).Kind);
        Assert.NotNull(graph.Component("vision"));
        Assert.NotNull(graph.Component("mtp"));

        // Representation directory + physical segments round-trip.
        Assert.Equal("segments/target.decoder_stack.bin", index.Representations["target.decoder_stack@F16"].Segment);
        var report2 = container.VerifyIntegrity();
        Assert.True(report2.Ok);

        // Real payload resolution through the same store the executor uses.
        using var store = container.CreateOperandStore();
        var q = store.Resolve("target.decoder_stack", "1.self_attn.q_proj.weight");
        Assert.Equal(Dtype.F16, q.Dtype);
        Assert.Equal(2, q.Shape.Length);

        // The hybrid operator boundary fails closed, by name.
        var ex = Assert.Throws<UnsupportedOperatorException>(() => Planner.Plan(container, "target", store));
        Assert.Contains("linear_attention", ex.Message);
    }

    [Fact]
    public void Unknown_LayerType_Refuses_At_Mapping()
    {
        using var dir = new TempDir();
        var expected = new[] { "full_attention", "fancy_attention", "full_attention" };
        WriteSyntheticCheckpoint(dir.Path, layerTypes: expected);

        using var inventory = HfInventory.Open(dir.Path);
        var facts = ModelConfig.ReadTextFacts(Path.Combine(dir.Path, "config.json"));
        var ex = Assert.Throws<ModelConfigException>(() =>
            ArchMapper.MapToContainerSpec("synth", facts, inventory, new ArchMapper.EncodeOptions()));
        Assert.Contains("fancy_attention", ex.Message);
    }

    // ── synthetic checkpoint builder ───────────────────────────────────────

    private static void WriteSyntheticCheckpoint(string modelDir, string[]? layerTypes = null)
    {
        const string prefix = "model.language_model";
        int hidden = 4;
        var layerTypesList = layerTypes ?? new[] { "linear_attention", "full_attention", "full_attention" };
        int layers = layerTypesList.Length;

        var tensors = new List<TensorPayload>();
        void Matrix(string name, int rows, int cols)
        {
            int elements = rows * cols;
            tensors.Add(new TensorPayload
            {
                Name = name,
                Dtype = Dtype.F16,
                Shape = new long[] { rows, cols },
                Data = Enumerable.Range(0, elements).SelectMany(i => new[] { (byte)(i % 251), (byte)((i * 7) % 251) }).ToArray(),
            });
        }

        Matrix($"{prefix}.embed_tokens.weight", 10, hidden);
        for (int l = 0; l < layers; l++)
        {
            Matrix($"{prefix}.layers.{l}.input_layernorm.weight", hidden, 1);
            Matrix($"{prefix}.layers.{l}.post_attention_layernorm.weight", hidden, 1);
            Matrix($"{prefix}.layers.{l}.mlp.gate_proj.weight", 6, hidden);
            Matrix($"{prefix}.layers.{l}.mlp.up_proj.weight", 6, hidden);
            Matrix($"{prefix}.layers.{l}.mlp.down_proj.weight", hidden, 6);

            if (layerTypesList[l] == "linear_attention")
            {
                Matrix($"{prefix}.layers.{l}.linear_attn.in_proj_qkv.weight", 12, hidden);
                Matrix($"{prefix}.layers.{l}.linear_attn.in_proj_a.weight", 2, hidden);
                Matrix($"{prefix}.layers.{l}.linear_attn.in_proj_b.weight", 2, hidden);
                Matrix($"{prefix}.layers.{l}.linear_attn.in_proj_z.weight", 8, hidden);
                Matrix($"{prefix}.layers.{l}.linear_attn.out_proj.weight", hidden, 8);
                tensors.Add(new TensorPayload
                {
                    Name = $"{prefix}.layers.{l}.linear_attn.conv1d.weight",
                    Dtype = Dtype.F16,
                    Shape = new long[] { 12, 1, 2 },
                    Data = new byte[12 * 2 * 2], // 24 elements × 2 bytes
                });
                // Deliberate precision exception: F32 inside the F16 stack.
                tensors.Add(new TensorPayload
                {
                    Name = $"{prefix}.layers.{l}.linear_attn.A_log",
                    Dtype = Dtype.F32,
                    Shape = new long[] { 2 },
                    Data = new byte[2 * 4],
                });
                tensors.Add(new TensorPayload
                {
                    Name = $"{prefix}.layers.{l}.linear_attn.dt_bias",
                    Dtype = Dtype.F16,
                    Shape = new long[] { 2 },
                    Data = new byte[4],
                });
                tensors.Add(new TensorPayload
                {
                    Name = $"{prefix}.layers.{l}.linear_attn.norm.weight",
                    Dtype = Dtype.F32,
                    Shape = new long[] { 2 },
                    Data = new byte[8],
                });
            }
            else
            {
                Matrix($"{prefix}.layers.{l}.self_attn.q_proj.weight", 4, hidden);
                Matrix($"{prefix}.layers.{l}.self_attn.k_proj.weight", 2, hidden);
                Matrix($"{prefix}.layers.{l}.self_attn.v_proj.weight", 2, hidden);
                Matrix($"{prefix}.layers.{l}.self_attn.o_proj.weight", hidden, 4);
            }
        }
        Matrix($"{prefix}.norm.weight", hidden, 1);

        // Perception tower + MTP, carried without segments.
        Matrix("model.visual.patch_embed.weight", 6, hidden);
        Matrix("model.visual.blocks.0.mlp.fc1.weight", 4, hidden);
        Matrix("mtp.fc.weight", hidden, hidden);
        Matrix("mtp.pre_fc_norm_hidden.weight", hidden, 1);

        SafetensorsWriter.Write(Path.Combine(modelDir, "model.safetensors"), tensors);

        string ropeParametersJson = """
            {
              "rope_type": "default",
              "rope_theta": 10000000,
              "partial_rotary_factor": 0.25,
              "mrope_interleaved": true,
              "mrope_section": [11, 11, 10]
            }
            """;
        string configJson = $$"""
            {
              "architectures": ["Qwen3_5ForConditionalGeneration"],
              "model_type": "qwen3_5",
              "tie_word_embeddings": true,
              "text_config": {
                "attention_bias": false,
                "attn_output_gate": true,
                "dtype": "float16",
                "head_dim": 2,
                "hidden_act": "silu",
                "hidden_size": 4,
                "intermediate_size": 6,
                "layer_types": [{{string.Join(", ", layerTypesList.Select(t => $"\"{t}\""))}}],
                "linear_conv_kernel_dim": 2,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 2,
                "linear_num_value_heads": 2,
                "linear_value_head_dim": 2,
                "max_position_embeddings": 512,
                "model_type": "qwen3_5_text",
                "num_attention_heads": 2,
                "num_hidden_layers": {{layers}},
                "num_key_value_heads": 1,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": true,
                "vocab_size": 10,
                "rope_parameters": {{ropeParametersJson}}
              }
            }
            """;
        File.WriteAllText(Path.Combine(modelDir, "config.json"), configJson);
    }
}