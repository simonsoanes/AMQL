using System.Text.Json;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Rope scaling frequencies against transformers' own
/// <c>ROPE_INIT_FUNCTIONS</c> (head dim 16, values printed by transformers
/// 5.16 in float32), and the policy's JSON round trip.
/// </summary>
public class RopeScalingTests
{
    private static void AssertFrequencies(double[] expected, double expectedScale, (double[] Inv, double Scale) actual)
    {
        Assert.Equal(expected.Length, actual.Inv.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(actual.Inv[i] - expected[i]) <= 1e-6 * Math.Max(1, Math.Abs(expected[i])) + 1e-12,
                $"pair {i}: {actual.Inv[i]:G9} vs {expected[i]:G9}");
        }
        Assert.Equal(expectedScale, actual.Scale, 6);
    }

    [Fact]
    public void Linear_Divides_Every_Frequency()
    {
        var scaling = new RopeScaling { Type = "linear", Factor = 4 };
        AssertFrequencies(
            new[] { 0.25, 0.079056941, 0.0250000004, 0.00790569466, 0.00249999994, 0.000790569466, 0.000250000012, 7.90569466e-05 },
            1.0, RopeScaling.Frequencies(10_000, 16, scaling));
    }

    [Fact]
    public void Llama3_Rescales_By_Wavelength_Band()
    {
        var scaling = new RopeScaling
        {
            Type = "llama3", Factor = 8, LowFreqFactor = 1, HighFreqFactor = 4, OriginalMaxPositions = 64,
        };
        AssertFrequencies(
            new[] { 1, 0.079403013, 0.00470075384, 0.000911583134, 0.000176776681, 3.42810235e-05, 6.64786967e-06, 1.28917316e-06 },
            1.0, RopeScaling.Frequencies(500_000, 16, scaling));
    }

    [Fact]
    public void Yarn_Ramps_Between_Correction_Dims_And_Scales_Attention()
    {
        var scaling = new RopeScaling
        {
            Type = "yarn", Factor = 8, OriginalMaxPositions = 32, BetaFast = 16, BetaSlow = 2, Truncate = false,
        };
        AssertFrequencies(
            new[] { 1, 0.0395284705, 0.0125000002, 0.00395284733, 0.00124999997, 0.000395284733, 0.000125000006, 3.95284733e-05 },
            1.20794415, RopeScaling.Frequencies(10_000, 16, scaling));
    }

    [Fact]
    public void Scaled_Policy_Round_Trips_Through_Json()
    {
        PositionPolicy policy = new PositionRope
        {
            Theta = 500_000,
            Scaling = new RopeScaling { Type = "llama3", Factor = 8, LowFreqFactor = 1, HighFreqFactor = 4, OriginalMaxPositions = 8192 },
        };
        string json = JsonSerializer.Serialize(policy, ViJson.Options);
        var back = Assert.IsType<PositionRope>(JsonSerializer.Deserialize<PositionPolicy>(json, ViJson.Options));
        Assert.Equal(500_000, back.Theta);
        Assert.Equal("llama3", back.Scaling!.Type);
        Assert.Equal(8192, back.Scaling.OriginalMaxPositions);
        Assert.Equal(JsonSerializer.Serialize(policy, ViJson.Options), JsonSerializer.Serialize<PositionPolicy>(back, ViJson.Options));
    }

    [Fact]
    public void Unsupported_Scaling_Stays_Unresolved()
    {
        using var block = JsonDocument.Parse("""{"rope_type":"longrope","factor":4,"rope_theta":10000}""");
        Assert.Null(RopeScaling.FromRopeBlock(block.RootElement, 4096));
    }
}
