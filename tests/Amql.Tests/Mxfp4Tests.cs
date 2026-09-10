using System.Text.Json;
using Amql.Cli;
using Amql.Hf;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// MXFP4 quantization tests (the OCP microscaling standard): the E2M1
/// element grid, E8M0 block scales, the pack/dequant round trip, and the
/// quantized export shape (fp4 weight + per-32-element E8M0 scales) with
/// dequant fidelity against the source container.
/// </summary>
public class Mxfp4Tests
{
    // ── the FP4 E2M1 element grid ─────────────────────────────────────────

    [Fact]
    public void Fp4_Grid_Decodes()
    {
        // nibble: [sign][low three bits index the positive grid]
        Assert.Equal(0f, BitPattern.DecodeFp4(0x0));
        Assert.Equal(0.5f, BitPattern.DecodeFp4(0x1));
        Assert.Equal(1.5f, BitPattern.DecodeFp4(0x3));
        Assert.Equal(1f, BitPattern.DecodeFp4(0x2));
        Assert.Equal(3f, BitPattern.DecodeFp4(0x5));
        Assert.Equal(6f, BitPattern.DecodeFp4(0x7));
        Assert.Equal(-1.5f, BitPattern.DecodeFp4(0xB)); // sign + 0x3
        Assert.Equal(4f, BitPattern.DecodeFp4(0x6));
    }

    [Fact]
    public void Fp4_Encode_Is_Nearest()
    {
        for (int nibble = 0; nibble < 16; nibble++)
        {
            Assert.Equal((byte)nibble, BitPattern.EncodeFp4(BitPattern.DecodeFp4((byte)nibble)));
        }
        Assert.Equal(0x0, BitPattern.EncodeFp4(0.24f));  // → 0
        Assert.Equal(0x1, BitPattern.EncodeFp4(0.4f));   // → 0.5
        Assert.Equal(0x1, BitPattern.EncodeFp4(0.75f));  // tie {0.5, 1} → smaller magnitude
        Assert.Equal(0x3, BitPattern.EncodeFp4(1.51f));  // → 1.5
        Assert.Equal(0x7, BitPattern.EncodeFp4(10f));    // clamps to +6
        Assert.Equal(0xF, BitPattern.EncodeFp4(-10f));   // clamps to −6
    }

    [Fact]
    public void E8M0_Scale_Encode_Decode()
    {
        foreach (float v in new[] { 1f, 0.5f, 2f, 0.125f, 6f, 0.333f, 100f })
        {
            byte bits = BitPattern.EncodeF8E8M0(v);
            float decoded = BitPattern.DecodeF8E8M0(bits);
            Assert.True(MathF.Abs(decoded - v) <= MathF.Max(MathF.Abs(v), 1e-25f),
                $"E8M0 round trip of {v} gave {decoded} (bits {bits})");
        }
        Assert.Equal(0, BitPattern.EncodeF8E8M0(0f));
    }

    // ── pack / dequant round trip ─────────────────────────────────────────

    [Fact]
    public void Mxfp4_Pack_Pairs_Low_Then_High_Nibble_With_E8M0_Scale()
    {
        // Values 1.5 and −2 in one 32-element block (rows 1, cols 4): the
        // block's peak is 2, whose E8M0 scale ≈ 2/6 → byte 125 → 0.25.
        // Ratios 6 and −8 (clamped to −6) → nibbles 0x7 / 0xF.
        var q = Mxfp4.Quantize(new float[] { 1.5f, -2f, 0f, 0f }, rows: 1, columns: 4);
        Assert.Equal(2, q.Packed.Length);
        Assert.Single(q.BlockScales);
        Assert.Equal(0x7, q.Packed[0] & 0x0F);
        Assert.Equal(0xF, (q.Packed[0] >> 4) & 0x0F);

        var dequant = Mxfp4.Dequant(q.Packed, q.BlockScales, rows: 1, columns: 4);
        Assert.True(MathF.Abs(dequant[0] - 1.5f) <= 0.01f, $"got {dequant[0]}");
        Assert.True(MathF.Abs(dequant[1] + 2f) <= 1.0f, $"got {dequant[1]}"); // clamp loss on the outlier pair
        Assert.Equal(0f, dequant[2]);
        Assert.Equal(0f, dequant[3]);
    }

    [Fact]
    public void Mxfp4_Quantize_Dequant_Preserves_Magnitudes()
    {
        var rng = new Random(7);
        int rows = 16, columns = 256;
        var values = new float[rows * columns];
        double maxAbs = 0;
        for (int i = 0; i < values.Length; i++)
        {
            double roll = rng.NextDouble();
            values[i] = roll < 0.95
                ? (float)(rng.NextDouble() * 6 - 3)
                : (float)(rng.NextDouble() * 200 - 100);
            maxAbs = Math.Max(maxAbs, Math.Abs((double)values[i]));
        }

        var q = Mxfp4.Quantize(values, rows, columns);
        var dequant = Mxfp4.Dequant(q.Packed, q.BlockScales, rows, columns);
        Assert.Equal(Mxfp4.PackedLength(values.Length), q.Packed.Length);
        Assert.Equal(rows * columns / 32, q.BlockScales.Length);

        double sqErr = 0, sqSignal = 0, maxErr = 0;
        for (int i = 0; i < values.Length; i++)
        {
            double diff = dequant[i] - values[i];
            sqErr += diff * diff;
            sqSignal += (double)values[i] * values[i];
            maxErr = Math.Max(maxErr, Math.Abs(diff));
        }
        double nrmse = Math.Sqrt(sqErr / sqSignal);
        Assert.True(nrmse < 0.2, $"NRMSE {nrmse:0.000} too high");
        // MX blocks span 32 elements with one shared power-of-two scale and
        // a coarse top grid (4/6), so a block dominated by an outlier can
        // carry ~25% peak error — inherent to the standard, not a bug.
        Assert.True(maxErr < 0.25 * maxAbs, $"max abs error {maxErr:0.00} too high");
    }

    [Fact]
    public void Mxfp4_Zero_Values_Quantise_To_Zeros()
    {
        var q = Mxfp4.Quantize(new float[] { 0, 0, 0, 0, 0 }, rows: 1, columns: 5);
        Assert.All(q.Packed, b => Assert.Equal(0, b));
        Assert.All(q.BlockScales, b => Assert.Equal(0, b));
        Assert.All(Mxfp4.Dequant(q.Packed, q.BlockScales, rows: 1, columns: 5), v => Assert.Equal(0f, v));
    }

    // ── quantized export ──────────────────────────────────────────────────

    [Fact]
    public void Export_With_Mxfp4_Emits_Pairs_And_Keeps_Everything_Else()
    {
        using var dir = new TempDir();
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "synth-mxfp4");
        var outDir = Path.Combine(dir.Path, "exported");

        using (var container = Vindex3Container.Open(containerPath))
        {
            var report = ModelExporter.Export(container, outDir, patch: null, quantizeMxfp4: true);
            Assert.Contains(report.Notes, n => n.Contains("MXFP4"));
            Assert.True(report.PayloadBytes < (long)(128 * 1024), "quantized export should be small");
        }

        using var file = SafetensorsFile.Open(Path.Combine(outDir, "model.safetensors"));
        var names = file.TensorNames;

        // A stack projection becomes a pair: fp4 weight + E8M0 scales.
        Assert.Contains("model.layers.0.self_attn.q_proj.weight", names);
        Assert.Contains("model.layers.0.self_attn.q_proj.weight_scale", names);
        Assert.DoesNotContain(names, n => n.EndsWith("_global_scale"));
        var weight = file.GetTensor("model.layers.0.self_attn.q_proj.weight");
        Assert.Equal(Dtype.FP4, weight.Dtype);
        Assert.Equal(new long[] { 4, 4 }, weight.Shape);
        var scale = file.GetTensor("model.layers.0.self_attn.q_proj.weight_scale");
        Assert.Equal(Dtype.F8_E8M0, scale.Dtype);
        Assert.Equal(new long[] { 4, 1 }, scale.Shape); // one 32-element block per row

        // Everything else stays full precision: the embedding and final
        // norm are exported without fp4 companions, as are A_log tensors.
        Assert.Contains("model.embed_tokens.weight", names);
        Assert.DoesNotContain(names, n => n == "model.embed_tokens.weight_scale");
        Assert.Contains("model.norm.weight", names);
        Assert.DoesNotContain(names, n => n == "model.norm.weight_scale");
        Assert.DoesNotContain(names,
            n => n.Contains("A_log") && n.EndsWith(Mxfp4.ScaleSuffix));

        // config.json documents the scheme, including the element grid.
        using var config = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(outDir, "config.json")));
        var qc = config.RootElement.GetProperty("quantization_config");
        Assert.Equal("mxfp4", qc.GetProperty("quant_method").GetString());
        Assert.Equal(8, qc.GetProperty("element_grid").GetArrayLength());
        Assert.Equal(1.5, qc.GetProperty("element_grid")[3]!.GetDouble());
        Assert.Equal(32, qc.GetProperty("block_elements").GetInt32());
        Assert.Equal("F8_E8M0", qc.GetProperty("block_scale_dtype").GetString());
    }

    [Fact]
    public void Export_Mxfp4_Dequant_Matches_The_Source_Within_Bounds()
    {
        using var dir = new TempDir();
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "synth-mxfp4");
        var outDir = Path.Combine(dir.Path, "exported");

        using (var container = Vindex3Container.Open(containerPath))
        {
            ModelExporter.Export(container, outDir, patch: null, quantizeMxfp4: true);
        }

        using var file = SafetensorsFile.Open(Path.Combine(outDir, "model.safetensors"));
        var packed = file.ReadBytes("model.layers.0.self_attn.q_proj.weight");
        var scales = file.ReadBytes("model.layers.0.self_attn.q_proj.weight_scale");
        var dequant = Mxfp4.Dequant(packed, scales, rows: 4, columns: 4);

        using (var container = Vindex3Container.Open(containerPath))
        using (var store = container.CreateOperandStore())
        {
            var resolution = store.Resolve("target.decoder_stack", "0.self_attn.q_proj.weight");
            var original = BitPattern.WidenToF32(resolution.Dtype, resolution.Payload);
            double sqErr = 0, sqSignal = 0;
            for (int i = 0; i < 16; i++)
            {
                double diff = dequant[i] - original[i];
                sqErr += diff * diff;
                sqSignal += (double)original[i] * original[i];
            }
            Assert.True(Math.Sqrt(sqErr / sqSignal) < 0.35,
                $"dequant NRMSE over the source too high: {Math.Sqrt(sqErr / sqSignal):0.00}");
        }
    }
}