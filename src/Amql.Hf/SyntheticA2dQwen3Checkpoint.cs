using System.Text.Json;
using Amql.Safetensors;

namespace Amql.Hf;

/// <summary>
/// Minimal dLLM A2D masked diffusion checkpoint (<c>A2DQwen3LMHeadModel</c>,
/// <c>model_type</c> <c>a2d-qwen3</c>): Qwen3 weights — per-head q/k norms,
/// GQA, SwiGLU — whose attention is bidirectional. The tokenizer files carry
/// only what the importer reads: a <c>mask_token</c> and its id. Norm weights
/// are random so a misplaced norm shows up in the logits.
/// </summary>
public static class SyntheticA2dQwen3Checkpoint
{
    public const int Hidden = 64;
    public const int Layers = 2;
    public const int NumQHeads = 4;
    public const int NumKvHeads = 2;
    public const int HeadDim = 16;
    public const int Intermediate = 96;
    public const int Vocab = 128;
    public const int Context = 512;
    public const int MaskTokenId = 127;
    public const string MaskToken = "<|mask|>";
    public const string Family = "a2d-qwen3";

    private static float[] Matrix(Random rng, int rows, int cols)
    {
        float scale = MathF.Sqrt(6.0f / (rows + cols));
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        }
        return data;
    }

    private static float[] NormWeight(Random rng, int n)
    {
        var data = new float[n];
        for (int i = 0; i < n; i++)
        {
            data[i] = (float)(0.5 + rng.NextDouble());
        }
        return data;
    }

    private static byte[] F32(float[] values)
    {
        var b = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, b, 0, b.Length);
        return b;
    }

    public static void Write(string dir, string modelType = Family)
    {
        Directory.CreateDirectory(dir);
        var rng = new Random(11);

        var config = new Dictionary<string, object>
        {
            ["architectures"] = new[] { "A2DQwen3LMHeadModel" },
            ["auto_map"] = new Dictionary<string, object>
            {
                ["AutoConfig"] = "modeling_qwen3.A2DQwen3Config",
                ["AutoModel"] = "modeling_qwen3.A2DQwen3Model",
                ["AutoModelForMaskedLM"] = "modeling_qwen3.A2DQwen3LMHeadModel",
            },
            ["model_type"] = modelType,
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
            ["attention_bias"] = false,
            ["max_position_embeddings"] = Context,
            ["layer_types"] = Enumerable.Repeat("full_attention", Layers).ToArray(),
            ["rope_theta"] = 1_000_000.0,
            ["rope_scaling"] = (object?)null!,
            ["sliding_window"] = (object?)null!,
            ["use_sliding_window"] = false,
            ["bos_token_id"] = 0,
            ["eos_token_id"] = 1,
            ["pad_token_id"] = 0,
        };
        File.WriteAllText(Path.Combine(dir, "config.json"),
            JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true }));
        File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"),
            JsonSerializer.Serialize(new Dictionary<string, object> { ["mask_token"] = MaskToken, ["eos_token"] = "<|im_end|>" }));
        File.WriteAllText(Path.Combine(dir, "added_tokens.json"),
            JsonSerializer.Serialize(new Dictionary<string, int> { [MaskToken] = MaskTokenId, ["<|im_end|>"] = 1 }));

        var tensors = new List<(string Name, long[] Shape, float[] Data)>
        {
            ("model.embed_tokens.weight", new long[] { Vocab, Hidden }, Matrix(rng, Vocab, Hidden)),
        };
        int qDim = NumQHeads * HeadDim;
        int kvDim = NumKvHeads * HeadDim;
        for (int l = 0; l < Layers; l++)
        {
            string p = $"model.layers.{l}.";
            tensors.Add(($"{p}input_layernorm.weight", new long[] { Hidden }, NormWeight(rng, Hidden)));
            tensors.Add(($"{p}self_attn.q_proj.weight", new long[] { qDim, Hidden }, Matrix(rng, qDim, Hidden)));
            tensors.Add(($"{p}self_attn.k_proj.weight", new long[] { kvDim, Hidden }, Matrix(rng, kvDim, Hidden)));
            tensors.Add(($"{p}self_attn.v_proj.weight", new long[] { kvDim, Hidden }, Matrix(rng, kvDim, Hidden)));
            tensors.Add(($"{p}self_attn.o_proj.weight", new long[] { Hidden, qDim }, Matrix(rng, Hidden, qDim)));
            tensors.Add(($"{p}self_attn.q_norm.weight", new long[] { HeadDim }, NormWeight(rng, HeadDim)));
            tensors.Add(($"{p}self_attn.k_norm.weight", new long[] { HeadDim }, NormWeight(rng, HeadDim)));
            tensors.Add(($"{p}post_attention_layernorm.weight", new long[] { Hidden }, NormWeight(rng, Hidden)));
            tensors.Add(($"{p}mlp.gate_proj.weight", new long[] { Intermediate, Hidden }, Matrix(rng, Intermediate, Hidden)));
            tensors.Add(($"{p}mlp.up_proj.weight", new long[] { Intermediate, Hidden }, Matrix(rng, Intermediate, Hidden)));
            tensors.Add(($"{p}mlp.down_proj.weight", new long[] { Hidden, Intermediate }, Matrix(rng, Hidden, Intermediate)));
        }
        tensors.Add(("model.norm.weight", new long[] { Hidden }, NormWeight(rng, Hidden)));

        var payloads = tensors
            .Select(t => new TensorPayload { Name = t.Name, Dtype = Dtype.F32, Shape = t.Shape, Data = F32(t.Data) })
            .ToList();
        SafetensorsWriter.Write(Path.Combine(dir, "model.safetensors"), payloads,
            new Dictionary<string, string> { ["format"] = "pt" });
    }
}
