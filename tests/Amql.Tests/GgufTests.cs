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
        // GGUF writes ne[] as the HF shape REVERSED ([32, 16]) but leaves the
        // row-major buffer untouched — gguf-py does shape[::-1] and writes the
        // data unchanged. An HF nn.Linear is already [out, in], which is
        // [in, out] in ggml's order, so physically transposing the bytes here
        // double-transposes every weight: the file still loads and then
        // generates garbage.
        var kBytes = reader.ReadBytes("blk.1.attn_k.weight");
        Assert.Equal(32 * 16 * 2, kBytes.Length);
        var tensor = reader.GetTensor("blk.1.attn_k.weight");
        // stored in llama.cpp's ne[] order: [in, out] for the reversed k
        Assert.Equal(new long[] { 32, 16 }, tensor.Dims);
        for (int r = 0; r < 16; r++)
        {
            for (int c = 0; c < 32; c++)
            {
                ushort actual = BinaryPrimitives.ReadUInt16LittleEndian(kBytes.AsSpan((r * 32 + c) * 2));
                ushort expected = BitConverter.HalfToUInt16Bits((Half)SourceValue(r * 32 + c));
                Assert.Equal(expected, actual);
            }
        }

        // the F32 router is transposed to [hidden, experts] (llama.cpp's ne[] order)
        var router = reader.GetTensor("blk.0.ffn_gate_inp.weight");
        Assert.Equal(GgufType.F32, router.Type);
        Assert.Equal(new long[] { Hidden, Experts }, router.Dims);
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
        // expert slices stacked; file dims are llama.cpp's ne[] order [s1, s0, n_expert]
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

    /// <summary>
    /// Pins the Q4_0 byte layout to ggml's. The golden bytes come from
    /// quantize_row_q4_0_ref (via gguf-py's numpy port) applied to the
    /// synthetic ramp SourceValue(0..31) = -1, -0.9375 … 0.9375, which is
    /// block 0 of blk.1.attn_k.weight since the buffer is a verbatim copy.
    /// Byte j holds element j in its low nibble and element j+16 in its high
    /// nibble. Pairing even with odd instead yields 0x10, 0x21, 0x32, … and
    /// every row decodes as a permutation of itself — the file still loads,
    /// and only inference reveals the damage, so this needs an explicit test.
    /// </summary>
    [Fact]
    public void Q4_0_Blocks_Use_Ggml_Nibble_Layout()
    {
        using var temp = new TempDir();
        BuildCheckpoint(temp.Path);

        string outFile = Path.Combine(temp.Path, "model-q4.gguf");
        GgufConverter.Convert(temp.Path, outFile, quantization: "q4_0");

        using var reader = GgufReader.Open(outFile);
        var tensor = reader.GetTensor("blk.1.attn_k.weight");
        Assert.Equal(GgufType.Q4_0, tensor.Type);
        Assert.Equal(new long[] { 32, 16 }, tensor.Dims);

        byte[] bytes = reader.ReadBytes("blk.1.attn_k.weight");
        Assert.Equal(32 * 16 / 32 * 18, bytes.Length); // 512 values → 16 blocks

        byte[] golden =
        {
            0x00, 0x30,                                     // scale 0.125 as F16 LE
            0x80, 0x91, 0x91, 0xA2, 0xA2, 0xB3, 0xB3, 0xC4, // elements 0-7 | 16-23
            0xC4, 0xD5, 0xD5, 0xE6, 0xE6, 0xF7, 0xF7, 0xF8, // elements 8-15 | 24-31
        };
        Assert.Equal(golden, bytes[..18]);

        // the skip list survives quantization: norms stay F32, embeddings F16
        Assert.Equal(GgufType.F32, reader.GetTensor("blk.1.attn_norm.weight").Type);
        Assert.Equal(GgufType.F32, reader.GetTensor("blk.0.ffn_gate_inp.weight").Type);
        Assert.Equal(GgufType.F16, reader.GetTensor("token_embd.weight").Type);
    }

    /// <summary>
    /// The V-head reorder must permute rows (and, for out_proj, columns) and
    /// then write the buffer verbatim, matching _reorder_v_heads in
    /// llama.cpp's converter: reshape the V window to
    /// [num_k_heads, num_v_per_k, head_dim], swap the first two axes, flatten
    /// back to the original shape. Nothing transposes, so the declared dims
    /// are the HF shape reversed and the payload stays row-major.
    /// The synthetic config has 2 key heads and 4 value heads, making this the
    /// only test that exercises the path — Qwen3.5-0.8B has equal head counts
    /// and skips the reorder entirely, which is how a double transpose here
    /// survived a fully verified 0.8B export.
    /// </summary>
    [Fact]
    public void Linear_Attention_V_Heads_Are_Reordered_Not_Transposed()
    {
        using var temp = new TempDir();
        BuildCheckpoint(temp.Path);

        string outFile = Path.Combine(temp.Path, "model.gguf");
        GgufConverter.Convert(temp.Path, outFile);

        using var reader = GgufReader.Open(outFile);

        // in_proj_qkv is [40, hidden]; rows 0..15 are the q|k window and keep
        // their place, rows 16..39 are the V window and get permuted.
        var qkv = reader.GetTensor("blk.0.attn_qkv.weight");
        Assert.Equal(new long[] { Hidden, 40 }, qkv.Dims);
        Assert.Equal(GgufType.F16, qkv.Type);
        var qkvBytes = reader.ReadBytes("blk.0.attn_qkv.weight");
        for (int dst = 0; dst < 40; dst++)
        {
            int src = VReorderSourceRow(dst);
            for (int c = 0; c < Hidden; c++)
            {
                ushort actual = BinaryPrimitives.ReadUInt16LittleEndian(qkvBytes.AsSpan((dst * Hidden + c) * 2));
                ushort expected = BitConverter.HalfToUInt16Bits((Half)SourceValue(src * Hidden + c));
                Assert.Equal(expected, actual);
            }
        }

        // out_proj is [hidden, 24] and permutes its input columns instead.
        // The non-square shape makes the declared dims discriminate a
        // transpose on their own, and num_v_per_k 3 against 2 key heads makes
        // the permutation non-self-inverse, so this also pins its direction.
        var outProj = reader.GetTensor("blk.0.ssm_out.weight");
        Assert.Equal(new long[] { 24, Hidden }, outProj.Dims);
        Assert.Equal(GgufType.F16, outProj.Type);
        var outBytes = reader.ReadBytes("blk.0.ssm_out.weight");
        for (int r = 0; r < Hidden; r++)
        {
            for (int c = 0; c < 24; c++)
            {
                ushort actual = BinaryPrimitives.ReadUInt16LittleEndian(outBytes.AsSpan((r * 24 + c) * 2));
                ushort expected = BitConverter.HalfToUInt16Bits((Half)SourceValue(r * 24 + VReorderSourceCol(c)));
                Assert.Equal(expected, actual);
            }
        }
    }

    /// <summary>The V-head projections are ordinary matrices, so they quantize
    /// like every other weight; only the scalar params and the depthwise conv
    /// must stay F32. Forcing all of them to F32 cost 4 bytes per element on
    /// three quarters of a hybrid model's layers.</summary>
    [Fact]
    public void Q4_0_Quantizes_The_Reordering_Projections()
    {
        using var temp = new TempDir();
        BuildCheckpoint(temp.Path);

        string outFile = Path.Combine(temp.Path, "model-q4.gguf");
        GgufConverter.Convert(temp.Path, outFile, quantization: "q4_0");

        using var reader = GgufReader.Open(outFile);
        Assert.Equal(GgufType.Q4_0, reader.GetTensor("blk.0.attn_qkv.weight").Type);
        Assert.Equal(GgufType.Q4_0, reader.GetTensor("blk.0.attn_gate.weight").Type);
        Assert.Equal(GgufType.Q4_0, reader.GetTensor("blk.0.ssm_out.weight").Type);
        Assert.Equal(GgufType.F32, reader.GetTensor("blk.0.ssm_alpha.weight").Type);
        Assert.Equal(GgufType.F32, reader.GetTensor("blk.0.ssm_beta.weight").Type);
        Assert.Equal(GgufType.F32, reader.GetTensor("blk.0.ssm_conv1d.weight").Type);
        Assert.Equal(GgufType.F32, reader.GetTensor("blk.0.ssm_a").Type);
    }

    /// <summary>llama.cpp reads the chat template out of the GGUF itself, not
    /// from a sidecar file, so chat_template.jinja has to be embedded as
    /// tokenizer.chat_template — otherwise the model loads but a runtime can
    /// only do raw completions.</summary>
    [Fact]
    public void Chat_Template_Is_Embedded_As_A_Gguf_Key()
    {
        using var temp = new TempDir();
        BuildCheckpoint(temp.Path);
        const string template = "{% for m in messages %}{{ m.content }}{% endfor %}";
        File.WriteAllText(Path.Combine(temp.Path, "chat_template.jinja"), template);

        string outFile = Path.Combine(temp.Path, "with-template.gguf");
        var report = GgufConverter.Convert(temp.Path, outFile);
        Assert.Contains(report.Notes, n => n.Contains("chat template embedded"));

        using var reader = GgufReader.Open(outFile);
        Assert.Equal(template, reader.Get("tokenizer.chat_template").AsString());
    }

    /// <summary>A missing template is not an error — a base model legitimately
    /// ships none — but it must be reported, since the result is a GGUF that
    /// cannot format a conversation and that is easy to mistake for a
    /// runtime bug.</summary>
    [Fact]
    public void Missing_Chat_Template_Leaves_The_Gguf_Completion_Only()
    {
        using var temp = new TempDir();
        BuildCheckpoint(temp.Path);

        string outFile = Path.Combine(temp.Path, "no-template.gguf");
        var report = GgufConverter.Convert(temp.Path, outFile);
        Assert.Contains(report.Notes, n => n.Contains("completion-only"));

        using var reader = GgufReader.Open(outFile);
        Assert.False(reader.TryGet("tokenizer.chat_template", out _));
    }

    /// <summary>
    /// Pins the MXFP4 and Q8_0 byte layouts to ggml's. The golden bytes come
    /// from gguf-py's numpy ports of quantize_row_mxfp4_ref and
    /// quantize_row_q8_0 applied to the synthetic ramp SourceValue(0..31)
    /// (amax 1.0), which is block 0 of blk.1.attn_k.weight because the payload
    /// is a verbatim copy. MXFP4 stores one E8M0 exponent byte then 16 nibble
    /// bytes; Q8_0 stores an F16 scale then 32 signed bytes.
    /// </summary>
    [Fact]
    public void Mxfp4_And_Q8_0_Blocks_Use_Ggml_Layout()
    {
        using var temp = new TempDir();
        BuildCheckpoint(temp.Path);

        // MXFP4: amax 1.0 → e = floor(log2(1)) - 2 + 127 = 125 = 0x7D, and
        // the scale that pairs with the doubled E2M1 grid is 2^(125-128) = 0.125.
        string mxfp4File = Path.Combine(temp.Path, "model-mxfp4.gguf");
        GgufConverter.Convert(temp.Path, mxfp4File, quantization: "mxfp4");
        using (var reader = GgufReader.Open(mxfp4File))
        {
            Assert.Equal(GgufType.Mxfp4, reader.GetTensor("blk.1.attn_k.weight").Type);
            Assert.Equal(38u, reader.Get("general.file_type").AsUInt32());
            byte[] bytes = reader.ReadBytes("blk.1.attn_k.weight");
            Assert.Equal(32 * 16 / 32 * 17, bytes.Length); // 512 values → 16 blocks
            byte[] golden =
            {
                0x7D,                                     // E8M0 exponent 125
                0x0E, 0x0E, 0x1D, 0x1D, 0x2D, 0x2D, 0x3C, 0x3C, // elements 0-7 | 16-23
                0x4C, 0x4B, 0x4B, 0x5A, 0x5A, 0x59, 0x59, 0x60, // elements 8-15 | 24-31
            };
            Assert.Equal(golden, bytes[..17]);
        }

        // Q8_0: d = amax/127 = 1/127, stored F16 little-endian as 0x2008.
        string moeFile = Path.Combine(temp.Path, "model-mxfp4-moe.gguf");
        GgufConverter.Convert(temp.Path, moeFile, quantization: "mxfp4_moe");
        using (var reader = GgufReader.Open(moeFile))
        {
            Assert.Equal(GgufType.Q8_0, reader.GetTensor("blk.1.attn_k.weight").Type);
            byte[] bytes = reader.ReadBytes("blk.1.attn_k.weight");
            Assert.Equal(32 * 16 / 32 * 34, bytes.Length);
            byte[] golden =
            {
                0x08, 0x20,                               // scale 1/127 as F16 LE
                0x81, 0x89, 0x91, 0x99, 0xA1, 0xA9, 0xB1, 0xB9,
                0xC0, 0xC8, 0xD0, 0xD8, 0xE0, 0xE8, 0xF0, 0xF8,
                0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38,
                0x40, 0x47, 0x4F, 0x57, 0x5F, 0x67, 0x6F, 0x77,
            };
            Assert.Equal(golden, bytes[..34]);
        }
    }

    /// <summary>
    /// llama.cpp's MXFP4_MOE recipe (ftype 38) keys on the tensor being a 3-D
    /// expert stack: those become MXFP4 and every other quantizable weight
    /// becomes Q8_0, which is what keeps the shared attention path out of
    /// 4-bit. Norms, routers, embeddings and 1-D tensors stay full precision.
    /// </summary>
    [Fact]
    public void Mxfp4_Moe_Follows_The_Llama_Cpp_Recipe()
    {
        using var temp = new TempDir();
        BuildCheckpoint(temp.Path);

        string outFile = Path.Combine(temp.Path, "model-moe.gguf");
        GgufConverter.Convert(temp.Path, outFile, quantization: "mxfp4_moe");

        using var reader = GgufReader.Open(outFile);
        Assert.Equal(38u, reader.Get("general.file_type").AsUInt32());
        Assert.Equal(2u, reader.Get("general.quantization_version").AsUInt32());

        // 3-D expert stacks → MXFP4
        Assert.Equal(GgufType.Mxfp4, reader.GetTensor("blk.0.ffn_gate_exps.weight").Type);
        Assert.Equal(GgufType.Mxfp4, reader.GetTensor("blk.0.ffn_up_exps.weight").Type);
        Assert.Equal(GgufType.Mxfp4, reader.GetTensor("blk.0.ffn_down_exps.weight").Type);

        // every other quantizable weight → Q8_0
        Assert.Equal(GgufType.Q8_0, reader.GetTensor("blk.1.attn_q.weight").Type);
        Assert.Equal(GgufType.Q8_0, reader.GetTensor("blk.0.attn_qkv.weight").Type);
        Assert.Equal(GgufType.Q8_0, reader.GetTensor("blk.0.ssm_out.weight").Type);

        // the skip list is unchanged: routers and norms stay F32, embeddings F16
        Assert.Equal(GgufType.F32, reader.GetTensor("blk.0.ffn_gate_inp.weight").Type);
        Assert.Equal(GgufType.F32, reader.GetTensor("blk.1.attn_norm.weight").Type);
        Assert.Equal(GgufType.F32, reader.GetTensor("blk.0.ssm_conv1d.weight").Type);
        Assert.Equal(GgufType.F16, reader.GetTensor("token_embd.weight").Type);

        // all-MXFP4 mode quantizes the non-expert weights too
        string allFile = Path.Combine(temp.Path, "model-all.gguf");
        GgufConverter.Convert(temp.Path, allFile, quantization: "mxfp4");
        using var all = GgufReader.Open(allFile);
        Assert.Equal(GgufType.Mxfp4, all.GetTensor("blk.1.attn_q.weight").Type);
        Assert.Equal(GgufType.Mxfp4, all.GetTensor("blk.0.ffn_gate_exps.weight").Type);
        Assert.Equal(GgufType.F32, all.GetTensor("blk.0.ffn_gate_inp.weight").Type);
    }

    // 2 key heads × 3 value heads per key × head_dim 4, so a grouped row
    // (k, vp, h) becomes the tiled row (vp, k, h). num_v_per_k differing from
    // num_k_heads is what makes the permutation non-self-inverse, matching the
    // 27B's 16 key / 48 value heads.
    private const int VWindowStart = 16, NumKHeads = 2, VPerK = 3, LinearHeadDim = 4;

    private static int VReorderSourceRow(int dst)
    {
        if (dst < VWindowStart) return dst;
        int tiled = dst - VWindowStart;
        int vp = tiled / (NumKHeads * LinearHeadDim);
        int rem = tiled % (NumKHeads * LinearHeadDim);
        int k = rem / LinearHeadDim;
        int h = rem % LinearHeadDim;
        return VWindowStart + (k * VPerK + vp) * LinearHeadDim + h;
    }

    private static int VReorderSourceCol(int dst)
    {
        int vp = dst / (NumKHeads * LinearHeadDim);
        int rem = dst % (NumKHeads * LinearHeadDim);
        int k = rem / LinearHeadDim;
        int h = rem % LinearHeadDim;
        return (k * VPerK + vp) * LinearHeadDim + h;
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
              "linear_num_value_heads": 6,
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
                // consistent with the head config: q(2×4) + k(2×4) + v(6×4) = 40 rows.
                // 6 value heads over 2 key heads gives num_v_per_k = 3, which
                // matters: the grouped→tiled permutation is its own inverse only
                // when num_v_per_k equals num_k_heads, so a fixture with 4 value
                // heads would pass a column reorder written in either direction.
                Add(p + "linear_attn.in_proj_qkv.weight", 40, Hidden);
                Add(p + "linear_attn.in_proj_a.weight", 6, Hidden);
                Add(p + "linear_attn.in_proj_b.weight", 6, Hidden);
                Add(p + "linear_attn.in_proj_z.weight", 24, Hidden);
                Add(p + "linear_attn.out_proj.weight", Hidden, 24);
                tensors.Add(new TensorPayload
                {
                    Name = p + "linear_attn.conv1d.weight",
                    Dtype = Dtype.BF16,
                    Shape = new long[] { 40, 1, 4 },
                    Data = Bf16(Enumerable.Range(0, 160).Select(i => SourceValue(i)).ToArray()),
                });
                Add1D(p + "linear_attn.A_log", 6);
                Add1D(p + "linear_attn.dt_bias", 6);
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