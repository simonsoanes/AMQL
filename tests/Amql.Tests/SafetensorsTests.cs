using Amql.Safetensors;
using Xunit;

namespace Amql.Tests;

public class SafetensorsTests
{
    // ── dtype decoders: golden bit patterns ─────────────────────────────────

    [Theory]
    [InlineData(0x3C00, 1.0f)]                     // 1.0
    [InlineData(0xBC00, -1.0f)]                    // -1.0
    [InlineData(0x0000, 0.0f)]                     // +0
    [InlineData(0x8000, -0.0f)]                    // -0
    [InlineData(0x4000, 2.0f)]                     // 2.0
    [InlineData(0x3555, 0.33325195f)]              // (1 + 341/1024) * 2^-2
    [InlineData(0x7C00, float.PositiveInfinity)]   // +inf
    public void F16_Decode_Golden(ushort bits, float expected)
    {
        Assert.Equal(expected, BitPattern.DecodeF16(bits), 6);
    }

    [Theory]
    [InlineData(0x3F80u, 1.0f)]
    [InlineData(0xC000u, -2.0f)]
    [InlineData(0x0000u, 0.0f)]
    [InlineData(0x7F80u, float.PositiveInfinity)]
    public void Bf16_Decode_Golden(ushort bits, float expected)
    {
        Assert.Equal(expected, BitPattern.DecodeBf16(bits), 6);
    }

    [Fact]
    public void Fp8_Decoders_Golden()
    {
        // E4M3: 0 0111 000 → 1 * 2^(7-7) = 1
        Assert.Equal(1.0f, BitPattern.DecodeF8E4M3(0b0_0111_000), 6);
        // E4M3: 1 0111 001 → -(1 + 1/8) * 2^0 = -1.125
        Assert.Equal(-1.125f, BitPattern.DecodeF8E4M3(0b1_0111_001), 6);
        // E4M3 subnormal: 0 0000 101 → (5/8) * 2^(1-7)
        Assert.Equal((float)(5.0 / 8.0 * Math.Pow(2, 1 - 7)), BitPattern.DecodeF8E4M3(0b0_0000_101), 6);
        // E5M2: 0 11110 00 → 2^(30-15) = 32768
        Assert.Equal(32768.0f, BitPattern.DecodeF8E5M2(0b0_11110_00), 6);
        // E5M2: 0 11111 00 → +inf
        Assert.True(float.IsPositiveInfinity(BitPattern.DecodeF8E5M2(0b0_11111_00)));
        // E8M0: 0x80 → 2^(128-127) = 2
        Assert.Equal(2.0f, BitPattern.DecodeF8E8M0(0x80), 6);
        // E8M0: 0xFF → NaN
        Assert.True(float.IsNaN(BitPattern.DecodeF8E8M0(0xFF)));
        // I8 sign extension
        Assert.Equal(-3.0f, BitPattern.DecodeI8(0xFD), 6);
    }

    [Theory]
    [InlineData(1.0f)]
    [InlineData(0.5f)]
    [InlineData(-2.0f)]
    [InlineData(1.5f)]
    [InlineData(0.33325195f)]
    [InlineData(1024.0f)]
    public void F16_Encode_RoundTrip_Exact(float value)
    {
        float decoded = BitPattern.DecodeF16(BitPattern.EncodeF16(value));
        Assert.Equal(value, decoded, 2);
    }

    [Fact]
    public void F16_Encode_RoundTrip_Pi()
    {
        // π is not exactly representable in f16 (relative precision ~2^-10);
        // the round trip must land within its epsilon.
        float decoded = BitPattern.DecodeF16(BitPattern.EncodeF16(3.14159f));
        Assert.Equal(3.14159f, decoded, 2);
    }

    [Fact]
    public void Bf16_Encode_RoundTrip()
    {
        foreach (var value in new[] { 1.0f, -0.75f, 256.0f, 0.0001f })
        {
            Assert.Equal(value, BitPattern.DecodeBf16(BitPattern.EncodeBf16(value)), 3);
        }
    }

    [Fact]
    public void WidenToF32_Matches_ManualDecode()
    {
        var f16Bytes = new byte[] { 0x00, 0x3C, 0x00, 0xBC, 0x00, 0x40 }; // 1.0, -1.0, 2.0 (F16 LE)
        var widened = BitPattern.WidenToF32(Dtype.F16, f16Bytes);
        Assert.Equal(new[] { 1.0f, -1.0f, 2.0f }, widened);
    }

    // ── file round trip ────────────────────────────────────────────────────

    [Fact]
    public void Writer_Then_Reader_RoundTrip()
    {
        using var dir = new TempDir();
        var path = Path.Combine(dir.Path, "model.safetensors");

        var tensors = new[]
        {
            new TensorPayload { Name = "weights.0", Dtype = Dtype.F32, Shape = new long[] { 2, 3 }, Data = SyntheticModel.ToBytes(new float[] { 1, 2, 3, 4, 5, 6 }) },
            new TensorPayload { Name = "alpha", Dtype = Dtype.F16, Shape = new long[] { 2 }, Data = new byte[] { 0x00, 0x3C, 0x00, 0xBC } },
            new TensorPayload { Name = "beta", Dtype = Dtype.BF16, Shape = new long[] { 1 }, Data = new byte[] { 0x80, 0x3F } }, // BF16 1.0
        };
        SafetensorsWriter.Write(path, tensors);

        using var file = SafetensorsFile.Open(path);
        Assert.Equal(3, file.Tensors.Count);
        Assert.True(file.Contains("alpha"));
        Assert.Equal(Dtype.F16, file.GetTensor("alpha").Dtype);
        Assert.Equal(new long[] { 2 }, file.GetTensor("alpha").Shape);

        // Payload alignment invariant: 8 + headerLen ≡ 0 (mod 8).
        Assert.Equal(0, (8 + file.HeaderLength) % SafetensorsFile.HeaderByteAlignment);

        // Tensors restored byte-for-byte.
        Assert.Equal(tensors[0].Data, file.ReadBytes("weights.0"));
        Assert.Equal(tensors[1].Data, file.ReadBytes("alpha"));

        // Widening.
        Assert.Equal(new[] { 1.0f, -1.0f }, file.DecodeF32("alpha"));
        Assert.Equal(new[] { 1.0f }, file.DecodeF32("beta"));

        // Header-only inventory reads nothing of the payload: a tensor
        // lookup touches header data alone.
        Assert.Equal(6L, file.GetTensor("weights.0").ElementCount);
    }

    [Fact]
    public void Empty_Header_Has_No_Tensors_And_Reads_Fail_Loudly()
    {
        using var dir = new TempDir();
        var path = Path.Combine(dir.Path, "empty.safetensors");

        // A valid empty-header file: header JSON "{}" padded so the stored
        // length keeps 8 + len aligned to 8.
        var headerJson = "{}"u8.ToArray();
        int storedLength = 8; // "{}" (2 bytes) + 6 pad bytes
        var fileBytes = new byte[8 + storedLength];
        BitConverter.TryWriteBytes(fileBytes.AsSpan(0, 8), (ulong)storedLength);
        headerJson.CopyTo(fileBytes, 8);
        fileBytes.AsSpan(8 + headerJson.Length, storedLength - headerJson.Length).Fill(0x20);
        File.WriteAllBytes(path, fileBytes);

        using var file = SafetensorsFile.Open(path);
        Assert.Empty(file.Tensors);
        Assert.Throws<SafetensorsException>(() => file.ReadBytes("nope"));
    }

    // ── sharding / model directory ─────────────────────────────────────────

    [Fact]
    public void Sharded_ModelDirectory_Resolves_Across_Shards()
    {
        using var dir = new TempDir();
        SafetensorsWriter.Write(
            Path.Combine(dir.Path, "model-00001-of-00002.safetensors"),
            new[]
            {
                new TensorPayload { Name = "layers.0.q.weight", Dtype = Dtype.F32, Shape = new long[] { 2 }, Data = SyntheticModel.ToBytes(new[] { 1f, 2f }) },
            });
        SafetensorsWriter.Write(
            Path.Combine(dir.Path, "model-00002-of-00002.safetensors"),
            new[]
            {
                new TensorPayload { Name = "layers.1.q.weight", Dtype = Dtype.F32, Shape = new long[] { 2 }, Data = SyntheticModel.ToBytes(new[] { 3f, 4f }) },
            });

        var shards = ModelDirectory.DiscoverShards(dir.Path);
        Assert.Equal(2, shards.Count);
        Assert.EndsWith("model-00001-of-00002.safetensors", shards[0]);

        using var model = ModelDirectory.Open(dir.Path);
        Assert.Equal(new[] { 1f, 2f }, model.DecodeF32("layers.0.q.weight"));
        Assert.Equal(new[] { 3f, 4f }, model.DecodeF32("layers.1.q.weight"));

        // Key normalisation mirrors the reference normalize_key.
        Assert.Equal("layers.0.q.weight", ModelDirectory.NormalizeKey("model.layers.0.q.weight", new[] { "model.", "model.language_model." }));
        Assert.Equal("embed.weight", ModelDirectory.NormalizeKey("embed.weight", new[] { "model." }));
    }

    [Fact]
    public void Unsupported_Integer_Dtype_Refuses_Widening()
    {
        using var dir = new TempDir();
        var path = Path.Combine(dir.Path, "ints.safetensors");
        SafetensorsWriter.Write(path, new[]
        {
            new TensorPayload { Name = "t", Dtype = Dtype.I32, Shape = new long[] { 2 }, Data = new byte[] { 1, 0, 0, 0, 2, 0, 0, 0 } },
        });

        using var file = SafetensorsFile.Open(path);
        Assert.False(file.GetTensor("t").Dtype.IsWidenableToF32());
        Assert.Throws<SafetensorsException>(() => file.DecodeF32("t"));
    }
}