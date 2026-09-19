using System.Buffers.Binary;
using Amql.Gguf;
using Amql.Safetensors;

namespace Amql.Tests;

/// <summary>
/// GGUF conversion tests: a synthetic hybrid (linear + full attention)
/// MoE checkpoint is built on disk and converted with
/// <see cref="GgufConverter"/>, then read back with <see cref="GgufReader"/>
/// to verify the header/metadata, the tensor table (names, GGUF-dim order,
/// 32-byte-aligned offsets), the transposed layouts, the stacked experts
/// and the tokenizer arrays.
/// </summary>
public class GgufTests
{
    private const int Hidden = 32;
    private const int Vocab = 17;
    private const int Experts = 4;
    private const int ExpertMid = 16;

    [Fact]
    public void Convert_Writes_Valid_Readable_Gguf()
    {
        using var temp = new TempDir();
        BuildCheckpoint(temp.Path);

        string outFile = Path.Combine(temp.Path, "model.gguf");
        var report = GgufConverter.Convert(temp.Path, outFile);

        Assert.Equal("qwen35moe", report.Architecture);
        Assert.True(report.TensorsWritten > 0);

        using var reader = GgufReader.Open(outFile);
        Assert.Equal("qwen35moe", reader.Arch);
        Assert.Equal(2u, reader.Get("qwen35moe.block_count").AsUInt32());
        Assert.Equal((uint)Hidden, reader.Get("qwen35moe.embedding_length").AsUInt32());
        Assert.Equal(4u, reader.Get("qwen35moe.attention.head_count").AsUInt32());
        Assert.Equal(2u, reader.Get("qwen35moe.attention.head_count_kv").AsUInt32());
        Assert.Equal((uint)Experts, reader.Get("qwen35moe.expert_count").AsUInt32());
        Assert.Equal(2u, reader.Get("qwen35moe.expert_used_count").AsUInt32());
        Assert.Equal(10_000f, reader.Get("qwen35moe.rope.freq_base").AsFloat());
        Assert.Equal(2u, reader.Get("qwen35moe.rope.dimension_count").AsUInt32()); // head_dim 8 × 0.25

        // required Qwen3.5 MRoPE section + the ssm/recurrent metadata
        var sections = reader.Get("qwen35moe.rope.dimension_sections").AsArray().Items;
        Assert.Equal(4, sections.Count);
        Assert.Equal(11u, (uint)sections[0]);
        Assert.Equal(11u, (uint)sections[1]);
        Assert.Equal(10u, (uint)sections[2]);
        Assert.Equal(4u, reader.Get("qwen35moe.ssm.conv_kernel").AsUInt32());
        Assert.Equal(4u, reader.Get("qwen35moe.ssm.state_size").AsUInt32());
        var recurrent = reader.Get("qwen35moe.attention.recurrent_layers").AsArray().Items;
        Assert.Equal(2, recurrent.Count);
        Assert.Equal(true, recurrent[0]);
        Assert.Equal(false, recurrent[1]);

        // every registered tensor offset is 32-byte aligned
        foreach (var tensor in reader.Tensors)
        {
            Assert.Equal(0ul, tensor.Offset % 32);
        }

        // the tokenizer arrays round-trip
        var tokens = reader.Get("tokenizer.ggml.tokens").AsArray().Items;
        Assert.Equal(Vocab, tokens.Count);
        Assert.Equal("<unk>", (string)tokens[0]);
        Assert.Equal("z", (string)tokens[Vocab - 1]);
        var merges = reader.Get("tokenizer.ggml.merges").AsArray().Items;
        Assert.Equal(2, merges.Count);
    }

    [Fact]
    public void Transposed_Tensors_Are_Layout_Correct()
    {
        using var temp = new TempDir();
        BuildCheckpoint(temp.Path);

        string outFile = Path.Combine(temp.Path, "model.gguf");
        GgufConverter.Convert(temp.Path, outFile);

        using var reader = GgufReader.Open(outFile);

        // layer 1 is full_attention: k_proj is [kv*head_dim, hidden] = [16, 32].
        // The GGUF tensor is transposed to [hidden, kv*head_dim] = [32, 16]:
        // element (c, r) = source (r, c), in F16.
        var kBytes = reader.ReadBytes("blk.1.attn_k.weight");
        Assert.Equal(32 * 16 * 2, kBytes.Length);
        var tensor = reader.GetTensor("blk.1.attn_k.weight");
        Assert.Equal(new long[] { 16, 32 }, tensor.Dims);
        for (int r = 0; r < 16; r++)
        {
            for (int c = 0; c < 32; c++)
            {
                ushort actual = BinaryPrimitives.ReadUInt16LittleEndian(kBytes.AsSpan((c * 16 + r) * 2));
                ushort expected = BitConverter.HalfToUInt16Bits((Half)SourceValue(r * 32 + c));
                Assert.Equal(expected, actual);
            }
        }

        // the F32 router is transposed to [hidden, experts], kept F32;
        // the stored GGUF dims are the reversed logical order [experts, hidden]
        var router = reader.GetTensor("blk.0.ffn_gate_inp.weight");
        Assert.Equal(GgufType.F32, router.Type);
        Assert.Equal(new long[] { Experts, Hidden }, router.Dims);
    }

    [Fact]
    public void Experts_Are_Stacked_Into_3d_Tensors()
    {
        using var temp = new TempDir();
        BuildCheckpoint(temp.Path);

        string outFile = Path.Combine(temp.Path, "model.gguf");
        GgufConverter.Convert(temp.Path, outFile);

        using var reader = GgufReader.Open(outFile);
        var gate = reader.GetTensor("blk.0.ffn_gate_exps.weight");
        // logical [experts, mid, emb] stored reversed: [emb, mid, experts]
        Assert.Equal(new long[] { Hidden, ExpertMid, Experts }, gate.Dims);
        Assert.Equal(GgufType.F16, gate.Type);

        var bytes = reader.ReadBytes("blk.0.ffn_gate_exps.weight");
        Assert.Equal(Hidden * ExpertMid * Experts * 2, bytes.Length);
        // expert 2's first element: the synthetic source fills each tensor
        // from its own index 0, so it is SourceValue(0)
        ushort actual = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan((2 * Hidden * ExpertMid) * 2));
        ushort expected = BitConverter.HalfToUInt16Bits((Half)SourceValue(0));
        Assert.Equal(expected, actual);
    }

    // ── synthetic checkpoint ───────────────────────────────────────────────

    private sealed class TempDir : IDisposable
    {
        public string Path { get; } =
            System.IO.Path.Combine(System.IO.Path.GetTempPath(), "amql-gguf-" + Guid.NewGuid().ToString("N")[..8]);

        public TempDir() => Directory.CreateDirectory(Path);

        public void Dispose()
        {
            try
            {
                Directory.Delete(Path, true);
            }
            catch
            {
                // best-effort cleanup
            }
        }
    }

    private static float SourceValue(long index)
        => ((index % 33) - 16) / 16.0f; // dyadic — exactly representable in BF16 and F16

    private static byte[] Bf16(float[] values)
    {
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            ushort bits = BitPattern.EncodeBf16(values[i]);
            bytes[i * 2] = (byte)(bits & 0xFF);
            bytes[i * 2 + 1] = (byte)(bits >> 8);
        }
        return bytes;
    }

    private static long Mul(params long[] dims)
    {
        long total = 1;
        foreach (long d in dims)
        {
            total *= d;
        }
        return total;
    }

    private static void BuildCheckpoint(string dir)
    {
        string config =
            $$"""
            {
              "model_type": "qwen3_5_text",
              "hidden_size": {{Hidden}},
              "num_hidden_layers": 2,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 8,
              "intermediate_size": 64,
              "vocab_size": {{Vocab}},
              "max_position_embeddings": 512,
              "layer_types": ["linear_attention", "full_attention"],
              "moe": { "experts": {{Experts}}, "top_k": 2, "expert_intermediate_size": {{ExpertMid}}, "routing_policy": "softmax_then_select" },
              "attn_output_gate": true,
              "linear_conv_kernel_dim": 4,
              "linear_num_key_heads": 2,
              "linear_key_head_dim": 4,
              "linear_num_value_heads": 4,
              "linear_value_head_dim": 4,
              "rope_parameters": { "rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.25 }
            }
            """;
        File.WriteAllText(Path.Combine(dir, "config.json"), config);

        var tensors = new List<TensorPayload>();

        void Add(string name, long rows, long cols, Dtype dtype = Dtype.BF16)
        {
            long count = rows * cols;
            var values = new float[count];
            for (long i = 0; i < count; i++)
            {
                values[i] = SourceValue(i);
            }
            byte[] data = dtype == Dtype.F32
                ? values.SelectMany(v => BitConverter.GetBytes(v)).ToArray()
                : Bf16(values);
            tensors.Add(new TensorPayload
            {
                Name = name,
                Dtype = dtype,
                Shape = new long[] { rows, cols },
                Data = data,
            });
        }

        void Add1D(string name, long length)
        {
            var values = new float[length];
            for (long i = 0; i < length; i++)
            {
                values[i] = SourceValue(i);
            }
            tensors.Add(new TensorPayload
            {
                Name = name,
                Dtype = Dtype.BF16,
                Shape = new long[] { length },
                Data = Bf16(values),
            });
        }

        Add("model.language_model.embed_tokens.weight", Vocab, Hidden);
        Add1D("model.language_model.norm.weight", Hidden);
        Add("lm_head.weight", Vocab, Hidden);

        for (int layer = 0; layer < 2; layer++)
        {
            bool linear = layer == 0;
            string p = $"model.language_model.layers.{layer}.";
            Add1D(p + "input_layernorm.weight", Hidden);
            Add1D(p + "post_attention_layernorm.weight", Hidden);

            if (linear)
            {
                // consistent with the head config: q(2×4) + k(2×4) + v(4×4) = 32 rows
                Add(p + "linear_attn.in_proj_qkv.weight", 32, Hidden);
                Add(p + "linear_attn.in_proj_a.weight", 4, Hidden);
                Add(p + "linear_attn.in_proj_b.weight", 4, Hidden);
                Add(p + "linear_attn.in_proj_z.weight", 16, Hidden);
                Add(p + "linear_attn.out_proj.weight", Hidden, 16);
                tensors.Add(new TensorPayload
                {
                    Name = p + "linear_attn.conv1d.weight",
                    Dtype = Dtype.BF16,
                    Shape = new long[] { 32, 1, 4 },
                    Data = Bf16(Enumerable.Range(0, 128).Select(i => SourceValue(i)).ToArray()),
                });
                Add1D(p + "linear_attn.A_log", 4);
                Add1D(p + "linear_attn.dt_bias", 4);
                Add1D(p + "linear_attn.norm.weight", 8);
            }
            else
            {
                Add(p + "self_attn.q_proj.weight", Hidden, Hidden);
                Add(p + "self_attn.k_proj.weight", 16, Hidden);
                Add(p + "self_attn.v_proj.weight", 16, Hidden);
                Add(p + "self_attn.o_proj.weight", Hidden, 16);
                Add1D(p + "self_attn.q_norm.weight", Hidden);
                Add1D(p + "self_attn.k_norm.weight", 16);
            }

            Add(p + "mlp.router.weight", Experts, Hidden, Dtype.F32);
            for (int e = 0; e < Experts; e++)
            {
                Add(p + $"mlp.experts.{e}.gate_proj.weight", ExpertMid, Hidden);
                Add(p + $"mlp.experts.{e}.up_proj.weight", ExpertMid, Hidden);
                Add(p + $"mlp.experts.{e}.down_proj.weight", Hidden, ExpertMid);
            }
        }

        // a couple of vision tensors must be skipped, not error
        Add("model.visual.patch_embed.proj.weight", 8, 8);

        string tokenizer =
            $$"""
            {
              "model": {
                "vocab": { "z": {{Vocab - 1}}, "b": 2, "a": 1, "<unk>": 0 },
                "merges": ["a b", "a z"],
                "scores": [[1, -0.5], [2, -1.25]]
              }
            }
            """;
        File.WriteAllText(Path.Combine(dir, "tokenizer.json"), tokenizer);

        SafetensorsWriter.Write(Path.Combine(dir, "model.safetensors"), tensors,
            new Dictionary<string, string> { ["format"] = "pt" });
    }
}