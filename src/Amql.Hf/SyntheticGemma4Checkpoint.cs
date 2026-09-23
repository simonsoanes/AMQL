using System.Text.Json;
using Amql.Safetensors;

namespace Amql.Hf;

/// <summary>
/// Minimal Gemma 4 checkpoint for testing the import pipeline.
/// Produces a 2-layer, 128-hidden model with gelu_pytorch_tanh activation,
/// Gemma 4 tensor naming, and the nested text_config wrapper.
/// </summary>
public static class SyntheticGemma4Checkpoint
{
    public const int Hidden = 128;
    public const int Layers = 2;
    public const int NumQHeads = 4;
    public const int NumKvHeads = 1;
    public const int HeadDim = 32;
    public const int Intermediate = 256;
    public const int Vocab = 1000;
    public const int Context = 512;
    public const string Family = "gemma4";
    public const string TextModelType = "gemma4_text";

    // ── Weight helpers ──────────────────────────────────────────────────

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

    // ── Checkpoint builder ──────────────────────────────────────────────

    public static void Write(string dir)
    {
        Directory.CreateDirectory(dir);
        var rng = new Random(42);

        // config.json — Gemma 4 structure with text_config wrapper
        var config = new Dictionary<string, object>
        {
            ["architectures"] = new[] { "Gemma4ForConditionalGeneration" },
            ["model_type"] = "gemma4",
            ["text_config"] = new Dictionary<string, object>
            {
                ["model_type"] = TextModelType,
                ["hidden_size"] = Hidden,
                ["num_hidden_layers"] = Layers,
                ["num_attention_heads"] = NumQHeads,
                ["num_key_value_heads"] = NumKvHeads,
                ["head_dim"] = HeadDim,
                ["intermediate_size"] = Intermediate,
                ["hidden_act"] = "gelu_pytorch_tanh",
                ["rms_norm_eps"] = 1e-6,
                ["vocab_size"] = Vocab,
                ["tie_word_embeddings"] = true,
                ["max_position_embeddings"] = Context,
                ["layer_types"] = new[] { "full_attention", "full_attention" },
                ["rope_parameters"] = new Dictionary<string, object>
                {
                    ["rope_type"] = "default",
                    ["rope_theta"] = 10000.0,
                },
                ["query_pre_attn_scalar"] = 128.0, // head_dim
            },
        };
        File.WriteAllText(Path.Combine(dir, "config.json"),
            JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true }));

        // model.safetensors — Gemma 4 naming: model.language_model.*
        // (nested under the multimodal wrapper)
        var tensors = new Dictionary<string, (long[], float[])>
        {
            ["model.embed_tokens.weight"] = (new long[] { Vocab, Hidden }, Matrix(rng, Vocab, Hidden)),
        };

        string layerPrefix = "model.layers.";
        for (int l = 0; l < Layers; l++)
        {
            string lr = $"{layerPrefix}{l}.";
            int qDim = NumQHeads * HeadDim;
            int kvDim = NumKvHeads * HeadDim;
            tensors[$"{lr}self_attn.q_proj.weight"] = (new long[] { qDim, Hidden }, Matrix(rng, qDim, Hidden));
            tensors[$"{lr}self_attn.k_proj.weight"] = (new long[] { kvDim, Hidden }, Matrix(rng, kvDim, Hidden));
            tensors[$"{lr}self_attn.v_proj.weight"] = (new long[] { kvDim, Hidden }, Matrix(rng, kvDim, Hidden));
            tensors[$"{lr}self_attn.o_proj.weight"] = (new long[] { Hidden, qDim }, Matrix(rng, Hidden, qDim));
            tensors[$"{lr}input_layernorm.weight"] = (new long[] { Hidden }, Ones(Hidden));
            tensors[$"{lr}post_attention_layernorm.weight"] = (new long[] { Hidden }, Ones(Hidden));
            tensors[$"{lr}mlp.gate_proj.weight"] = (new long[] { Intermediate, Hidden }, Matrix(rng, Intermediate, Hidden));
            tensors[$"{lr}mlp.up_proj.weight"] = (new long[] { Intermediate, Hidden }, Matrix(rng, Intermediate, Hidden));
            tensors[$"{lr}mlp.down_proj.weight"] = (new long[] { Hidden, Intermediate }, Matrix(rng, Hidden, Intermediate));
        }

        tensors["model.norm.weight"] = (new long[] { Hidden }, Ones(Hidden));

        var payloads = new List<TensorPayload>();
        foreach (var (name, (shape, data)) in tensors)
        {
            payloads.Add(new TensorPayload { Name = name, Dtype = Dtype.F32, Shape = shape, Data = F32(data) });
        }

        SafetensorsWriter.Write(Path.Combine(dir, "model.safetensors"), payloads,
            new Dictionary<string, string> { ["format"] = "pt" });
    }

    /// <summary>Writes a minimal safetensors with enough decoder tensors
    /// to satisfy prefix detection during HfInventory.Open validation.</summary>
    public static void WriteMinimalSafetensors(string dir)
    {
        var payloads = new List<TensorPayload>
        {
            new() { Name = "model.embed_tokens.weight", Dtype = Dtype.F32,
                Shape = new long[] { Vocab, Hidden }, Data = new byte[Vocab * Hidden * 4] },
            new() { Name = "model.layers.0.self_attn.q_proj.weight", Dtype = Dtype.F32,
                Shape = new long[] { NumQHeads * HeadDim, Hidden }, Data = new byte[NumQHeads * HeadDim * Hidden * 4] },
            new() { Name = "model.norm.weight", Dtype = Dtype.F32,
                Shape = new long[] { Hidden }, Data = new byte[Hidden * 4] },
        };
        SafetensorsWriter.Write(Path.Combine(dir, "model.safetensors"), payloads,
            new Dictionary<string, string> { ["format"] = "pt" });
    }
}