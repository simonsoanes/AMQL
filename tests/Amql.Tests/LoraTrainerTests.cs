using System.Text.Json;
using Amql.Hf;
using Amql.Merge;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Tests for <c>amql-cli lora-train</c>: LoRA on FFN down-projections with a
/// hand-written backward pass. The finite-difference check the trainer logs is
/// the proof that the backward pass is right for the last layer.
/// </summary>
[Collection("gpu")]
public class LoraTrainerTests
{
    private static string WriteContainer(TempDir dir)
    {
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.WriteWithDims(modelDir, vocab: 12, hidden: 16, layers: 4,
            numQHeads: 2, numKvHeads: 1, headDim: 8, intermediate: 64);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "synth-lora");
        return containerPath;
    }

    private static (LoraTrainReport Report, List<JsonElement> Log) Train(LoraTrainOptions options)
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(WriteContainer(dir));
        var log = new List<JsonElement>();
        var train = new[] { new LoraExample("one", new[] { 1, 2, 3, 4 }, new[] { 7, 9, 7 }) };
        var report = LoraTrainer.Train(container, "target", train, Array.Empty<LoraExample>(), options,
            entry => log.Add(JsonSerializer.SerializeToElement(entry, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
            })));
        return (report, log);
    }

    [Fact]
    public void The_Last_Layer_Gradient_Matches_Finite_Differences()
    {
        var (_, log) = Train(new LoraTrainOptions { Steps = 1, Rank = 4, GradChecks = 2, ExtraDecodeTokens = 0 });
        var checks = log.Where(e => e.GetProperty("kind").GetString() == "gradcheck").ToList();
        Assert.Equal(2, checks.Count);
        foreach (var c in checks)
        {
            double predicted = c.GetProperty("predicted").GetDouble(), measured = c.GetProperty("measured").GetDouble();
            Assert.True(Math.Abs(predicted - measured) <= 1e-3 * Math.Max(1e-3, Math.Abs(measured)) + 1e-6,
                $"predicted {predicted}, measured {measured}");
        }
    }

    [Fact]
    public void Training_Fits_One_Example()
    {
        var (report, _) = Train(new LoraTrainOptions { Steps = 200, Rank = 4, LearningRate = 1e-2, GradChecks = 0, EvalEvery = 50 });
        Assert.True(report.FinalLoss < report.InitialLoss / 4, $"loss {report.InitialLoss} -> {report.FinalLoss}");
        Assert.Single(report.Layers);
        Assert.Equal(3, report.Layers[0].Layer);
    }
}
