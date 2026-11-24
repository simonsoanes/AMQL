using Amql.Inference;
using Xunit;

namespace Amql.Tests;

/// <summary>Top-k sampling over the survivors only draws what the
/// whole-vocabulary path draws, seed for seed.</summary>
public class SamplerTopKTests
{
    private static float[] Logits(int seed, int vocab, int masked)
    {
        var rng = new Random(seed);
        var logits = new float[vocab];
        for (int i = 0; i < vocab; i++)
        {
            logits[i] = (float)(rng.NextDouble() * 12 - 6);
        }
        // A few strong candidates, as a model's next token has.
        for (int j = 0; j < 30; j++)
        {
            logits[rng.Next(vocab)] = (float)(8 + rng.NextDouble() * 6);
        }
        // Constrained decoding masks all but a few tokens.
        for (int i = 0; i < masked; i++)
        {
            logits[i] = float.NegativeInfinity;
        }
        return logits;
    }

    [Theory]
    [InlineData(0.6f, 20, 0.95f, 0)]
    [InlineData(1.0f, 20, 0f, 0)]
    [InlineData(0.7f, 40, 0.8f, 0)]
    [InlineData(0.6f, 20, 0.95f, 248_310)]   // three tokens left: fewer than k
    [InlineData(0.6f, 1, 0f, 0)]
    public void Survivors_Only_Draw_As_The_Whole_Vocabulary_Does(float temperature, int topK, float topP, int masked)
    {
        var config = new SamplingConfig(Temperature: temperature, TopK: topK, TopP: topP);
        int agree = 0, runs = 0;
        for (int seed = 0; seed < 40; seed++)
        {
            var logits = Logits(seed, 248_313, masked);
            for (int draw = 0; draw < 25; draw++)
            {
                int fast = Sampler.Sample(logits, config, new Random(seed * 100 + draw));
                int whole = Sampler.SampleWholeVocabulary(logits, config, new Random(seed * 100 + draw));
                agree += fast == whole ? 1 : 0;
                runs++;
            }
        }
        // Equal but for a draw that lands within rounding of a boundary.
        Assert.True(agree >= runs - 1, $"{agree} of {runs} draws agree");
    }

    [Fact]
    public void Ties_At_The_Cutoff_Are_All_Kept()
    {
        var logits = new float[64];
        Array.Fill(logits, -5f);
        logits[3] = logits[10] = logits[40] = 2f;   // three tied for the top two places
        var config = new SamplingConfig(Temperature: 1f, TopK: 2);
        var seen = new HashSet<int>();
        for (int seed = 0; seed < 300; seed++)
        {
            int token = Sampler.Sample(logits, config, new Random(seed));
            Assert.Equal(Sampler.SampleWholeVocabulary(logits, config, new Random(seed)), token);
            seen.Add(token);
        }
        Assert.Equal(new HashSet<int> { 3, 10, 40 }, seen);
    }

    [Fact]
    public void Logits_With_No_Finite_Value_Fall_Back_To_The_Arg_Max()
    {
        var logits = new float[100];
        Array.Fill(logits, float.NegativeInfinity);
        var config = new SamplingConfig(Temperature: 0.6f, TopK: 20, TopP: 0.95f);
        Assert.Equal(Sampler.SampleWholeVocabulary(logits, config, new Random(1)), Sampler.Sample(logits, config, new Random(1)));
    }
}
