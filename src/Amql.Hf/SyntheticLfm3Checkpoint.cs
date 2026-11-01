using System.Text.Json;
using Amql.Safetensors;

namespace Amql.Hf;

/// <summary>
/// Minimal LFM3 text checkpoint (<c>Lfm3ForCausalLM</c>) for testing import,
/// inference and export: four layers — conv, full attention, conv, sliding
/// attention — so one model exercises the gated short conv (width 7), the
/// per-head output gate, the zero-tail partial rotary on the full layer, plain
/// rotary over a small window on the sliding layer, and the sandwich norms.
/// Norm weights are random rather than ones so a misplaced norm shows up in
/// the logits.
/// </summary>
public static class SyntheticLfm3Checkpoint
{
    public const int Hidden = 64;
    public const int Layers = 4;
    public const int NumQHeads = 4;
    public const int NumKvHeads = 2;
    public const int HeadDim = 16;
    public const int Intermediate = 96;
    public const int Vocab = 256;
    public const int Context = 512;
    public const int ConvKernel = 7;
    public const int SlidingWindow = 4;
    public const string Family = "lfm3";
    public static readonly string[] LayerTypes = { "conv", "full_attention", "conv", "sliding_attention" };

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

    public static void Write(string dir)
    {
        Directory.CreateDirectory(dir);
        var rng = new Random(7);

        var config = new Dictionary<string, object>
        {
            ["architectures"] = new[] { "Lfm3ForCausalLM" },
            ["model_type"] = Family,
            ["hidden_size"] = Hidden,
            ["num_hidden_layers"] = Layers,
            ["num_attention_heads"] = NumQHeads,
            ["num_key_value_heads"] = NumKvHeads,
            ["head_dim"] = HeadDim,
            ["intermediate_size"] = Intermediate,
            ["norm_eps"] = 1e-5,
            ["vocab_size"] = Vocab,
            ["tie_word_embeddings"] = true,
            ["max_position_embeddings"] = Context,
            ["layer_types"] = LayerTypes,
            ["sliding_window"] = SlidingWindow,
            ["use_sandwich_norm"] = true,
            ["conv_L_cache"] = ConvKernel,
            ["conv_bias"] = false,
            ["rope_parameters"] = new Dictionary<string, object>
            {
                ["full_attention"] = new Dictionary<string, object>
                {
                    ["partial_rotary_factor"] = 0.5,
                    ["rope_theta"] = 1_000_000.0,
                    ["rope_type"] = "default",
                },
                ["sliding_attention"] = new Dictionary<string, object>
                {
                    ["rope_theta"] = 10_000.0,
                    ["rope_type"] = "default",
                },
            },
        };
        File.WriteAllText(Path.Combine(dir, "config.json"),
            JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true }));

        var tensors = new List<(string Name, long[] Shape, float[] Data)>
        {
            ("model.embed_tokens.weight", new long[] { Vocab, Hidden }, Matrix(rng, Vocab, Hidden)),
        };
        int qDim = NumQHeads * HeadDim;
        int kvDim = NumKvHeads * HeadDim;
        for (int l = 0; l < Layers; l++)
        {
            string p = $"model.layers.{l}.";
            tensors.Add(($"{p}operator_norm.weight", new long[] { Hidden }, NormWeight(rng, Hidden)));
            tensors.Add(($"{p}post_operator_norm.weight", new long[] { Hidden }, NormWeight(rng, Hidden)));
            if (LayerTypes[l] == "conv")
            {
                tensors.Add(($"{p}conv.in_proj.weight", new long[] { 3 * Hidden, Hidden }, Matrix(rng, 3 * Hidden, Hidden)));
                tensors.Add(($"{p}conv.conv.weight", new long[] { Hidden, 1, ConvKernel }, Matrix(rng, Hidden, ConvKernel)));
                tensors.Add(($"{p}conv.out_proj.weight", new long[] { Hidden, Hidden }, Matrix(rng, Hidden, Hidden)));
            }
            else
            {
                tensors.Add(($"{p}self_attn.q_proj.weight", new long[] { qDim, Hidden }, Matrix(rng, qDim, Hidden)));
                tensors.Add(($"{p}self_attn.k_proj.weight", new long[] { kvDim, Hidden }, Matrix(rng, kvDim, Hidden)));
                tensors.Add(($"{p}self_attn.v_proj.weight", new long[] { kvDim, Hidden }, Matrix(rng, kvDim, Hidden)));
                tensors.Add(($"{p}self_attn.o_proj.weight", new long[] { Hidden, qDim }, Matrix(rng, Hidden, qDim)));
                tensors.Add(($"{p}self_attn.gate_proj.weight", new long[] { NumQHeads, Hidden }, Matrix(rng, NumQHeads, Hidden)));
                tensors.Add(($"{p}self_attn.q_norm.weight", new long[] { HeadDim }, NormWeight(rng, HeadDim)));
                tensors.Add(($"{p}self_attn.k_norm.weight", new long[] { HeadDim }, NormWeight(rng, HeadDim)));
            }
            tensors.Add(($"{p}ffn_norm.weight", new long[] { Hidden }, NormWeight(rng, Hidden)));
            tensors.Add(($"{p}post_ffn_norm.weight", new long[] { Hidden }, NormWeight(rng, Hidden)));
            tensors.Add(($"{p}feed_forward.w1.weight", new long[] { Intermediate, Hidden }, Matrix(rng, Intermediate, Hidden)));
            tensors.Add(($"{p}feed_forward.w3.weight", new long[] { Intermediate, Hidden }, Matrix(rng, Intermediate, Hidden)));
            tensors.Add(($"{p}feed_forward.w2.weight", new long[] { Hidden, Intermediate }, Matrix(rng, Hidden, Intermediate)));
        }
        tensors.Add(("model.embedding_norm.weight", new long[] { Hidden }, NormWeight(rng, Hidden)));

        var payloads = tensors
            .Select(t => new TensorPayload { Name = t.Name, Dtype = Dtype.F32, Shape = t.Shape, Data = F32(t.Data) })
            .ToList();
        SafetensorsWriter.Write(Path.Combine(dir, "model.safetensors"), payloads,
            new Dictionary<string, string> { ["format"] = "pt" });
    }
}
