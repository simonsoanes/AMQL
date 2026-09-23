using System.Text.Json;
using Amql.Safetensors;

namespace Amql.Hf;

/// <summary>
/// Minimal LFM 2.5 checkpoint for testing the import pipeline.
/// Produces a 2-layer hybrid model (1 conv + 1 attention) with
/// LFM-specific tensor names and config format.
/// </summary>
public static class SyntheticLfmCheckpoint
{
    public const int Hidden = 128;
    public const int Layers = 2;
    public const int NumQHeads = 4;
    public const int NumKvHeads = 1;
    public const int HeadDim = 32;
    public const int Intermediate = 256;
    public const int Vocab = 1000;
    public const int Context = 512;
    public const int ConvKernel = 4;
    public const string Family = "lfm2";

    private static float[] Matrix(Random rng, int rows, int cols)
    {
        float scale = MathF.Sqrt(6.0f / (rows + cols));
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return data;
    }

    private static float[] Ones(int n) => Enumerable.Repeat(1.0f, n).ToArray();
    private static byte[] F32(float[] values)
    {
        var b = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, b, 0, b.Length);
        return b;
    }

    public static void Write(string dir)
    {
        Directory.CreateDirectory(dir);
        var rng = new Random(42);

        var config = new Dictionary<string, object>
        {
            ["architectures"] = new[] { "Lfm2ForCausalLM" },
            ["model_type"] = Family,
            ["hidden_size"] = Hidden,
            ["num_hidden_layers"] = Layers,
            ["num_attention_heads"] = NumQHeads,
            ["num_key_value_heads"] = NumKvHeads,
            ["head_dim"] = HeadDim,
            ["intermediate_size"] = Intermediate,
            ["hidden_act"] = "silu",
            ["rms_norm_eps"] = 1e-6,
            ["vocab_size"] = Vocab,
            ["tie_word_embeddings"] = true,
            ["max_position_embeddings"] = Context,
            ["layer_types"] = new[] { "conv", "full_attention" },
            ["rope_parameters"] = new Dictionary<string, object>
            {
                ["rope_type"] = "default",
                ["rope_theta"] = 10_000_000.0,
            },
            ["conv_kernel"] = ConvKernel,
            ["conv_bias"] = false,
            ["block_use_swiglu"] = true,
        };
        File.WriteAllText(Path.Combine(dir, "config.json"),
            JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true }));

        var tensors = new Dictionary<string, (long[], float[])>
        {
            ["model.embed_tokens.weight"] = (new long[] { Vocab, Hidden }, Matrix(rng, Vocab, Hidden)),
        };

        // Layer 0: conv
        string l0 = "model.layers.0.";
        int qDim = NumQHeads * HeadDim;
        tensors[$"{l0}operator_norm.weight"] = (new long[] { Hidden }, Ones(Hidden));
        tensors[$"{l0}conv.in_proj.weight"] = (new long[] { qDim, Hidden }, Matrix(rng, qDim, Hidden));
        tensors[$"{l0}conv.conv.weight"] = (new long[] { Hidden, 1, ConvKernel }, Matrix(rng, Hidden * ConvKernel, 1));
        tensors[$"{l0}conv.out_proj.weight"] = (new long[] { qDim, Hidden }, Matrix(rng, qDim, Hidden));
        tensors[$"{l0}ffn_norm.weight"] = (new long[] { Hidden }, Ones(Hidden));
        tensors[$"{l0}feed_forward.w1.weight"] = (new long[] { Intermediate, Hidden }, Matrix(rng, Intermediate, Hidden));
        tensors[$"{l0}feed_forward.w3.weight"] = (new long[] { Intermediate, Hidden }, Matrix(rng, Intermediate, Hidden));
        tensors[$"{l0}feed_forward.w2.weight"] = (new long[] { Hidden, Intermediate }, Matrix(rng, Hidden, Intermediate));

        // Layer 1: full_attention
        string l1 = "model.layers.1.";
        int kvDim = NumKvHeads * HeadDim;
        tensors[$"{l1}operator_norm.weight"] = (new long[] { Hidden }, Ones(Hidden));
        tensors[$"{l1}self_attn.q_proj.weight"] = (new long[] { qDim, Hidden }, Matrix(rng, qDim, Hidden));
        tensors[$"{l1}self_attn.k_proj.weight"] = (new long[] { kvDim, Hidden }, Matrix(rng, kvDim, Hidden));
        tensors[$"{l1}self_attn.v_proj.weight"] = (new long[] { kvDim, Hidden }, Matrix(rng, kvDim, Hidden));
        tensors[$"{l1}self_attn.o_proj.weight"] = (new long[] { qDim, Hidden }, Matrix(rng, qDim, Hidden));
        tensors[$"{l1}self_attn.q_layernorm.weight"] = (new long[] { HeadDim }, Ones(HeadDim));
        tensors[$"{l1}self_attn.k_layernorm.weight"] = (new long[] { HeadDim }, Ones(HeadDim));
        tensors[$"{l1}ffn_norm.weight"] = (new long[] { Hidden }, Ones(Hidden));
        tensors[$"{l1}feed_forward.w1.weight"] = (new long[] { Intermediate, Hidden }, Matrix(rng, Intermediate, Hidden));
        tensors[$"{l1}feed_forward.w3.weight"] = (new long[] { Intermediate, Hidden }, Matrix(rng, Intermediate, Hidden));
        tensors[$"{l1}feed_forward.w2.weight"] = (new long[] { Hidden, Intermediate }, Matrix(rng, Hidden, Intermediate));

        tensors["model.norm.weight"] = (new long[] { Hidden }, Ones(Hidden));

        var payloads = new List<TensorPayload>();
        foreach (var (name, (shape, data)) in tensors)
            payloads.Add(new TensorPayload { Name = name, Dtype = Dtype.F32, Shape = shape, Data = F32(data) });

        SafetensorsWriter.Write(Path.Combine(dir, "model.safetensors"), payloads,
            new Dictionary<string, string> { ["format"] = "pt" });
    }

    public static void WriteMinimal(string dir)
    {
        Directory.CreateDirectory(dir);

        var config = new Dictionary<string, object>
        {
            ["architectures"] = new[] { "Lfm2ForCausalLM" },
            ["model_type"] = Family,
            ["hidden_size"] = Hidden,
            ["num_hidden_layers"] = 2,
            ["num_attention_heads"] = 4,
            ["num_key_value_heads"] = 1,
            ["head_dim"] = HeadDim,
            ["intermediate_size"] = Intermediate,
            ["hidden_act"] = "silu",
            ["rms_norm_eps"] = 1e-6,
            ["vocab_size"] = Vocab,
            ["tie_word_embeddings"] = true,
            ["max_position_embeddings"] = Context,
            ["layer_types"] = new[] { "conv", "full_attention" },
            ["rope_parameters"] = new Dictionary<string, object>
            {
                ["rope_type"] = "default",
                ["rope_theta"] = 10_000_000.0,
            },
        };
        File.WriteAllText(Path.Combine(dir, "config.json"),
            JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true }));

        // Minimal safetensors with just enough for prefix detection
        var tensors = new List<(string Name, long[] Shape, int Bytes)>
        {
            ("model.embed_tokens.weight", new long[] { Vocab, Hidden }, Vocab * Hidden * 4),
            ("model.layers.0.operator_norm.weight", new long[] { Hidden }, Hidden * 4),
            ("model.norm.weight", new long[] { Hidden }, Hidden * 4),
        };
        var payloads = tensors.Select(t =>
            new TensorPayload { Name = t.Name, Dtype = Dtype.F32, Shape = t.Shape, Data = new byte[t.Bytes] }).ToList();
        SafetensorsWriter.Write(Path.Combine(dir, "model.safetensors"), payloads,
            new Dictionary<string, string> { ["format"] = "pt" });
    }
}