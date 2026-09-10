using System.Text.Json;
using Amql.Cli;
using Amql.Hf;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// NVFP4 quantization tests: the FP4 element grid, the
/// pack/dequant round trip, and the quantized export shape (fp4 weight +
/// per-2-element FP8 scales + FP32 tensor scale) with dequant fidelity
/// against the source container.
/// </summary>
public class Nvfp4Tests
{
    // ── the FP4 element grid ──────────────────────────────────────────────

    [Fact]
    public void Fp4_Grid_Decodes()
    {
        // nibble: [sign][low three bits index the positive grid]
        Assert.Equal(0f, BitPattern.DecodeFp4(0x0));
        Assert.Equal(0.25f, BitPattern.DecodeFp4(0x1));
        Assert.Equal(0.75f, BitPattern.DecodeFp4(0x3));
        Assert.Equal(0.5f, BitPattern.DecodeFp4(0x2));
        Assert.Equal(1.5f, BitPattern.DecodeFp4(0x5));
        Assert.Equal(3f, BitPattern.DecodeFp4(0x7));
        Assert.Equal(-0.75f, BitPattern.DecodeFp4(0xB)); // sign + 0x3
        Assert.Equal(2f, BitPattern.DecodeFp4(0x6));
    }

    [Fact]
    public void Fp4_Encode_Is_Nearest()
    {
        for (int nibble = 0; nibble < 16; nibble++)
        {
            Assert.Equal((byte)nibble, BitPattern.EncodeFp4(BitPattern.DecodeFp4((byte)nibble)));
        }
        Assert.Equal(0x0, BitPattern.EncodeFp4(0.1f));        // → 0
        Assert.Equal(0x1, BitPattern.EncodeFp4(0.24f));       // → 0.25
        Assert.Equal(0x7, BitPattern.EncodeFp4(10f));         // clamps to +3
        Assert.Equal(0xF, BitPattern.EncodeFp4(-10f));        // clamps to −3
        Assert.Equal(0x2, BitPattern.EncodeFp4(0.4f));        // → 0.5
        Assert.Equal(0x3, BitPattern.EncodeFp4(0.75f));       // exactly on the grid
    }

    [Fact]
    public void Fp8E4M3_Encode_Round_Trips_The_Decoder()
    {
        foreach (float v in new[] { 1f, 2.5f, 0.5f, 0.0625f, 1.875f, 240f, 0.001f, 123.4f, -0.75f, 300f, 447f })
        {
            byte bits = BitPattern.EncodeF8E4M3(v);
            float decoded = BitPattern.DecodeF8E4M3(bits);
            // E4M3 has a 3-bit mantissa: relative step ≈ 1/8, so the
            // round trip must land within ~8% (plus an absolute floor).
            Assert.True(MathF.Abs(decoded - v) <= MathF.Max(1e-3f, MathF.Abs(v) * 0.08f),
                $"E4M3 round trip of {v} gave {decoded} (bits {bits:X2})");
        }
        Assert.Equal(0x7E, BitPattern.EncodeF8E4M3(1e9f)); // clamps to 448 (max finite)
    }

    // ── pack / dequant round trip ─────────────────────────────────────────

    [Fact]
    public void Nvfp4_Pack_Pairs_Low_Then_High_Nibble()
    {
        // Values 1.5 and −2: global scale = 2/3 maps −2 → −3 (the grid
        // extreme); the pair's block scale is 3. Ratios 0.75 and −1 are
        // exactly on the grid → nibbles 0x3 (low) and 0xC (high).
        var q = Nvfp4.Quantize(new float[] { 1.5f, -2f, 0f, 0f }, 4);
        Assert.Equal(2, q.Packed.Length);
        Assert.Equal(2, q.BlockScales.Length);
        Assert.True(q.GlobalScale > 0);
        Assert.Equal(0x3, q.Packed[0] & 0x0F);
        Assert.Equal(0xC, (q.Packed[0] >> 4) & 0x0F);

        var dequant = Nvfp4.Dequant(q.Packed, q.BlockScales, q.GlobalScale, 4);
        Assert.True(MathF.Abs(dequant[0] - 1.5f) <= 0.01f, $"got {dequant[0]}");
        Assert.True(MathF.Abs(dequant[1] + 2f) <= 0.01f, $"got {dequant[1]}");
        Assert.Equal(0f, dequant[2]);
        Assert.Equal(0f, dequant[3]);
    }

    [Fact]
    public void Nvfp4_Quantize_Dequant_Preserves_Magnitudes()
    {
        var rng = new Random(7);
        var values = new float[4096];
        double maxAbs = 0;
        for (int i = 0; i < values.Length; i++)
        {
            // Mix of scales: big outliers and a dense body.
            double roll = rng.NextDouble();
            values[i] = roll < 0.95
                ? (float)(rng.NextDouble() * 6 - 3)
                : (float)(rng.NextDouble() * 200 - 100);
            maxAbs = Math.Max(maxAbs, Math.Abs((double)values[i]));
        }

        var q = Nvfp4.Quantize(values, values.Length);
        var dequant = Nvfp4.Dequant(q.Packed, q.BlockScales, q.GlobalScale, values.Length);
        Assert.Equal(Nvfp4.PackedLength(values.Length), q.Packed.Length);

        // Normalised RMS error (block-quant) is small; the largest absolute
        // error is bounded by the local scale step.
        double sqErr = 0, sqSignal = 0, maxErr = 0;
        for (int i = 0; i < values.Length; i++)
        {
            double diff = dequant[i] - values[i];
            sqErr += diff * diff;
            sqSignal += (double)values[i] * values[i];
            maxErr = Math.Max(maxErr, Math.Abs(diff));
        }
        double nrmse = Math.Sqrt(sqErr / sqSignal);
        Assert.True(nrmse < 0.15, $"NRMSE {nrmse:0.000} too high");
        // Absolute error is the fp4 grid step on the local scale, so it is
        // bounded by a fraction of the tensor's global magnitude.
        Assert.True(maxErr < 0.1 * maxAbs, $"max abs error {maxErr:0.00} too high");
    }

    [Fact]
    public void Nvfp4_Zero_Values_Quantise_To_Zeros()
    {
        var q = Nvfp4.Quantize(new float[] { 0, 0, 0, 0, 0 }, 5);
        Assert.All(q.Packed, b => Assert.Equal(0, b));
        Assert.All(q.BlockScales, b => Assert.Equal(0, b));
        Assert.Equal(0f, q.GlobalScale);
        Assert.All(Nvfp4.Dequant(q.Packed, q.BlockScales, q.GlobalScale, 5), v => Assert.Equal(0f, v));
    }

    // ── quantized export ──────────────────────────────────────────────────

    [Fact]
    public void Export_With_Nvfp4_Emits_Triples_And_Keeps_Everything_Else()
    {
        using var dir = new TempDir();
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "synth-nvfp4");
        var outDir = Path.Combine(dir.Path, "exported");

        using (var container = Vindex3Container.Open(containerPath))
        {
            var report = ModelExporter.Export(container, outDir, patch: null, quantizeNvfp4: true);
            Assert.Contains(report.Notes, n => n.Contains("NVFP4"));
            // Under a quarter of the stack payload survives the fp4 packing.
            Assert.True(report.PayloadBytes < (long)(128 * 1024), "quantized export should be small");
        }

        using var file = SafetensorsFile.Open(Path.Combine(outDir, "model.safetensors"));
        var names = file.TensorNames;

        // A stack projection becomes a triple: fp4 weight + fp8 scales +
        // fp32 global scale.
        Assert.Contains("model.layers.0.self_attn.q_proj.weight", names);
        Assert.Contains("model.layers.0.self_attn.q_proj.weight_scale", names);
        Assert.Contains("model.layers.0.self_attn.q_proj.weight_global_scale", names);
        var weight = file.GetTensor("model.layers.0.self_attn.q_proj.weight");
        Assert.Equal(Dtype.FP4, weight.Dtype);
        Assert.Equal(new long[] { 4, 4 }, weight.Shape);
        var scale = file.GetTensor("model.layers.0.self_attn.q_proj.weight_scale");
        Assert.Equal(Dtype.F8_E4M3, scale.Dtype);
        Assert.Equal(new long[] { 4, 2 }, scale.Shape);
        var globalScale = file.GetTensor("model.layers.0.self_attn.q_proj.weight_global_scale");
        Assert.Equal(Dtype.F32, globalScale.Dtype);
        Assert.Equal(new long[] { 1 }, globalScale.Shape);

        // Everything else stays full precision: the embedding and final
        // norm are exported (no fp4 companions for them), as are the
        // log-space A_log tensors the rule excludes.
        Assert.Contains("model.embed_tokens.weight", names);
        Assert.DoesNotContain(names, n => n == "model.embed_tokens.weight_scale");
        Assert.Contains("model.norm.weight", names);
        Assert.DoesNotContain(names, n => n == "model.norm.weight_scale");
        Assert.DoesNotContain(names,
            n => n.Contains("A_log") && (n.EndsWith(Nvfp4.ScaleSuffix) || n.EndsWith(Nvfp4.GlobalScaleSuffix)));

        // config.json documents the scheme, including the element grid.
        using var config = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(outDir, "config.json")));
        var qc = config.RootElement.GetProperty("quantization_config");
        Assert.Equal("nvfp4", qc.GetProperty("quant_method").GetString());
        Assert.Equal(8, qc.GetProperty("element_grid").GetArrayLength());
        Assert.Equal(0.75, qc.GetProperty("element_grid")[3]!.GetDouble());
        Assert.Equal(2, qc.GetProperty("block_elements").GetInt32());
        Assert.Equal("F8_E4M3", qc.GetProperty("block_scale_dtype").GetString());
    }

    [Fact]
    public void Export_Nvfp4_Dequant_Matches_The_Source_Within_Bounds()
    {
        using var dir = new TempDir();
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "synth-nvfp4");
        var outDir = Path.Combine(dir.Path, "exported");

        using (var container = Vindex3Container.Open(containerPath))
        {
            ModelExporter.Export(container, outDir, patch: null, quantizeNvfp4: true);
        }

        using var file = SafetensorsFile.Open(Path.Combine(outDir, "model.safetensors"));
        var packed = file.ReadBytes("model.layers.0.self_attn.q_proj.weight");
        var scales = file.ReadBytes("model.layers.0.self_attn.q_proj.weight_scale");
        var globalBytes = file.ReadBytes("model.layers.0.self_attn.q_proj.weight_global_scale");
        float global = BitConverter.ToSingle(globalBytes, 0);
        var dequant = Nvfp4.Dequant(packed, scales, global, 16);

        using (var container = Vindex3Container.Open(containerPath))
        using (var store = container.CreateOperandStore())
        {
            var original = BitPattern.WidenToF32(
                store.Resolve("target.decoder_stack", "0.self_attn.q_proj.weight").Dtype,
                store.Resolve("target.decoder_stack", "0.self_attn.q_proj.weight").Payload);
            Assert.Equal(16, original.Length);
            double sqErr = 0, sqSignal = 0;
            for (int i = 0; i < 16; i++)
            {
                double diff = dequant[i] - original[i];
                sqErr += diff * diff;
                sqSignal += (double)original[i] * original[i];
            }
            Assert.True(Math.Sqrt(sqErr / sqSignal) < 0.2,
                $"dequant NRMSE over the source too high: {Math.Sqrt(sqErr / sqSignal):0.00}");
        }
    }
}