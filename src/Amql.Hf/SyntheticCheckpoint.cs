using System.Text;
using System.Text.Json;
using Amql.Safetensors;

namespace Amql.Hf;

/// <summary>
/// Writes a tiny, fully <em>executable</em> Qwen3.5-shaped text checkpoint
/// (config.json + one safetensors shard): a two-layer dense stack with
/// plain full-window softmax attention and standard rope. It exists so the
/// loader pipeline and the inference CLI can be exercised end-to-end when
/// no servable HF checkpoint is at hand — the real Qwen3.5-0.8B hybrid
/// kernels (linear attention, gates, partial MRoPE) are still refused by
/// name, and this fixture is the executable stand-in.
/// </summary>
public static class SyntheticCheckpoint
{
    public const int Vocab = 12;
    public const int Hidden = 4;
    public const int Layers = 2;
    public const int NumQueryHeads = 2;
    public const int NumKvHeads = 1;
    public const int HeadDim = 2;
    public const int Intermediate = 8;

    /// <summary>Deterministic smallish weight value — anything stable works
    /// for a demo fixture; tests recompute from the written checkpoint, not
    /// from this formula.</summary>
    public static float Value(int index) => 0.1f * ((index * 5 + 3) % 11 - 5);

    public static void Write(string modelDir)
    {
        Directory.CreateDirectory(modelDir);
        var tensors = new List<TensorPayload>();

        void Matrix(string name, int rows, int cols)
        {
            var data = new float[rows * cols];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = Value(i);
            }
            tensors.Add(new TensorPayload
            {
                Name = name,
                Dtype = Dtype.F32,
                Shape = new long[] { rows, cols },
                Data = ToBytes(data),
            });
        }

        void Vector(string name, int width)
        {
            var data = new float[width];
            for (int i = 0; i < width; i++)
            {
                data[i] = 1f + 0.1f * (i % 3); // slight variation so norms are meaningful
            }
            tensors.Add(new TensorPayload
            {
                Name = name,
                Dtype = Dtype.F32,
                Shape = new long[] { width },
                Data = ToBytes(data),
            });
        }

        int qDim = NumQueryHeads * HeadDim;
        int kvDim = NumKvHeads * HeadDim;

        Matrix("model.embed_tokens.weight", Vocab, Hidden);
        for (int l = 0; l < Layers; l++)
        {
            Matrix($"model.layers.{l}.self_attn.q_proj.weight", Hidden, qDim);
            Matrix($"model.layers.{l}.self_attn.k_proj.weight", Hidden, kvDim);
            Matrix($"model.layers.{l}.self_attn.v_proj.weight", Hidden, kvDim);
            Matrix($"model.layers.{l}.self_attn.o_proj.weight", Hidden, qDim);
            Vector($"model.layers.{l}.input_layernorm.weight", Hidden);
            Vector($"model.layers.{l}.post_attention_layernorm.weight", Hidden);
            Matrix($"model.layers.{l}.mlp.gate_proj.weight", Intermediate, Hidden);
            Matrix($"model.layers.{l}.mlp.up_proj.weight", Intermediate, Hidden);
            Matrix($"model.layers.{l}.mlp.down_proj.weight", Hidden, Intermediate);
        }
        Vector("model.norm.weight", Hidden);

        SafetensorsWriter.Write(Path.Combine(modelDir, "model.safetensors"), tensors);

        string config = $$"""
            {
              "architectures": ["Qwen3_5ForConditionalGeneration"],
              "model_type": "qwen3_5",
              "tie_word_embeddings": true,
              "text_config": {
                "attention_bias": false,
                "dtype": "float32",
                "head_dim": {{HeadDim}},
                "hidden_act": "silu",
                "hidden_size": {{Hidden}},
                "intermediate_size": {{Intermediate}},
                "layer_types": ["full_attention", "full_attention"],
                "max_position_embeddings": 2048,
                "model_type": "qwen3_5_text",
                "num_attention_heads": {{NumQueryHeads}},
                "num_hidden_layers": {{Layers}},
                "num_key_value_heads": {{NumKvHeads}},
                "rms_norm_eps": 1e-5,
                "tie_word_embeddings": true,
                "vocab_size": {{Vocab}},
                "rope_parameters": { "rope_type": "default", "rope_theta": 10000 }
              }
            }
            """;
        File.WriteAllText(Path.Combine(modelDir, "config.json"), config);
        WriteTokenizer(modelDir);
    }

    /// <summary>A matching tokenizer.json for the demo vocabulary: ids
    /// 0..11 = space, letters a..j, '?'. Plain text only — the demo
    /// alphabet is deliberately small so the demo model and its vocabulary
    /// stay in sync (any other letter refuses at encode time with a clear
    /// message, which is the honest boundary).</summary>
    private static void WriteTokenizer(string modelDir)
    {
        const string tokenizerJson = """
            {
              "version": "1.0.0",
              "truncation": null,
              "padding": null,
              "added_tokens": [],
              "normalizer": { "type": "NFC" },
              "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                  { "type": "Split", "pattern": { "Regex": "(?i:[a-j?])| +|[\\s\\S]" }, "behavior": "Isolated", "invert": false },
                  { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false }
                ]
              },
              "post_processor": null,
              "decoder": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false },
              "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": {
                  "\u0120": 0,
                  "a": 1,
                  "b": 2,
                  "c": 3,
                  "d": 4,
                  "e": 5,
                  "f": 6,
                  "g": 7,
                  "h": 8,
                  "i": 9,
                  "j": 10,
                  "?": 11
                },
                "merges": []
              }
            }
            """;
        File.WriteAllText(Path.Combine(modelDir, "tokenizer.json"), tokenizerJson);
    }

    private static byte[] ToBytes(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}