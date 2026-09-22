using Amql.Hf;
using Amql.Inference;
using Amql.Merge;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Tests for <c>amql-cli fine-tune</c>: teacher-forced output-head
/// adaptation over prompt/completion pairs.
/// </summary>
public class FineTuneTests
{
    private static string WriteContainer(TempDir dir)
    {
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "synth-ft");
        return containerPath;
    }

    [Fact]
    public void FineTune_Creates_Patch_That_Increases_Target_Probability()
    {
        using var dir = new TempDir();
        var containerPath = WriteContainer(dir);
        var patchPath = Path.Combine(dir.Path, "ft.safetensors");

        // The synthetic model has a 12-token character vocabulary.
        var tokenizer = HfTokenizer.FromModelDir(Path.Combine(dir.Path, "model"));

        // The synthetic model has a 12-token character vocabulary:
        //   Ġ(0) a(1) b(2) c(3) d(4) e(5) f(6) g(7) h(8) i(9) j(10) ?(11)
        // Use single-character prompts and target completions.
        string promptA = "a";
        string completionA = "b";
        string promptB = "c";
        string completionB = "d";

        var dataPath = Path.Combine(dir.Path, "pairs.tsv");
        File.WriteAllLines(dataPath, new[]
        {
            $"{promptA}\t{completionA}",
            $"{promptB}\t{completionB}",
        });

        // Baseline: measure P(completionA | promptA) and P(completionB | promptB)
        // without the patch.
        var targetA = tokenizer.EncodeToIds(completionA).First();
        var targetB = tokenizer.EncodeToIds(completionB).First();

        float beforeA, beforeB;
        using (var container = Vindex3Container.Open(containerPath))
        using (var store = container.CreateOperandStore())
        {
            var plan = Planner.Plan(container, "target", store);
            var rt = new GenericRuntime(plan, store);

            float ProbAfter(string prompt, int targetId)
            {
                rt.Kv.Reset();
                rt.ResetSession();
                var ids = tokenizer.EncodeToIds(prompt).ToArray();
                var hidden = rt.Embed(ids);
                var positions = Enumerable.Range(0, ids.Length).ToArray();
                for (int l = 0; l < plan.Layers.Count; l++)
                {
                    hidden = rt.RunLayerInternal(hidden, l, positions, positions, appendKv: true);
                }
                var logits = rt.FinalNormAndHead(hidden);
                return CausalTracer.SoftmaxProb(logits, targetId);
            }

            beforeA = ProbAfter(promptA, targetA);
            beforeB = ProbAfter(promptB, targetB);
            Assert.True(beforeA >= 0f && beforeA <= 1f,
                $"probability of '{completionA}' after '{promptA}' is {beforeA}");
            Assert.True(beforeB >= 0f && beforeB <= 1f,
                $"probability of '{completionB}' after '{promptB}' is {beforeB}");
        }

        // Fine-tune.
        using var container2 = Vindex3Container.Open(containerPath);
        var report = ModelFinetuner.FineTune(
            container2, "target", dataPath, patchPath, tokenizer, lr: 0.1, epochs: 3);

        Assert.Equal(2, report.Pairs);
        Assert.True(report.Steps > 0);
        Assert.True(File.Exists(patchPath));

        // Verify the patch can be loaded and validated.
        var patch = WeightPatch.Load(patchPath);
        Assert.Single(patch.Entries);
        var entry = patch.Entries[0];
        Assert.Equal(report.HeadObjectId + "/" + report.HeadTensorName, entry.Key);
        Assert.Equal(new long[] { report.VocabSize, report.HiddenSize }, entry.Shape);

        // After fine-tuning, the target probabilities should increase.
        using (var container3 = Vindex3Container.Open(containerPath))
        using (var store3 = container3.CreateOperandStore())
        {
            var plan3 = Planner.Plan(container3, "target", store3);
            var rt3 = new GenericRuntime(plan3, store3, patch);

            float ProbAfter(string prompt, int targetId)
            {
                rt3.Kv.Reset();
                rt3.ResetSession();
                var ids = tokenizer.EncodeToIds(prompt).ToArray();
                var hidden = rt3.Embed(ids);
                var positions = Enumerable.Range(0, ids.Length).ToArray();
                for (int l = 0; l < plan3.Layers.Count; l++)
                {
                    hidden = rt3.RunLayerInternal(hidden, l, positions, positions, appendKv: true);
                }
                var logits = rt3.FinalNormAndHead(hidden);
                return CausalTracer.SoftmaxProb(logits, targetId);
            }

            float afterA = ProbAfter(promptA, targetA);
            float afterB = ProbAfter(promptB, targetB);

            // The target probabilities should increase (or stay the same
            // if the model already predicted them with near-certainty).
            Assert.True(afterA >= beforeA - 0.01f,
                $"P('{completionA}'|'{promptA}') should not decrease: {beforeA:F4} → {afterA:F4}");
            Assert.True(afterB >= beforeB - 0.01f,
                $"P('{completionB}'|'{promptB}') should not decrease: {beforeB:F4} → {afterB:F4}");

            // At least one should have meaningfully increased with lr=0.1 × 3 epochs.
            Assert.True(afterA > beforeA + 0.001f || afterB > beforeB + 0.001f,
                $"fine-tuning should increase at least one target probability: " +
                $"A {beforeA:F4}→{afterA:F4}, B {beforeB:F4}→{afterB:F4}");
        }
    }

    [Fact]
    public void FineTune_Refuses_Empty_Data()
    {
        using var dir = new TempDir();
        var containerPath = WriteContainer(dir);
        var tokenizer = HfTokenizer.FromModelDir(Path.Combine(dir.Path, "model"));
        var dataPath = Path.Combine(dir.Path, "empty.tsv");
        File.WriteAllText(dataPath, "# just a comment\n\n"); // empty after parsing
        var patchPath = Path.Combine(dir.Path, "ft.safetensors");

        using var container = Vindex3Container.Open(containerPath);
        var ex = Assert.Throws<MergeException>(() =>
            ModelFinetuner.FineTune(container, "target", dataPath, patchPath, tokenizer));
        Assert.Contains("no valid training pairs", ex.Message);
    }

    [Fact]
    public void FineTune_Deterministic()
    {
        using var dir = new TempDir();
        var containerPath = WriteContainer(dir);
        var tokenizer = HfTokenizer.FromModelDir(Path.Combine(dir.Path, "model"));
        var dataPath = Path.Combine(dir.Path, "pairs.tsv");
        File.WriteAllLines(dataPath, new[] { "a\tb", "c\td" });

        var patchA = Path.Combine(dir.Path, "a.safetensors");
        var patchB = Path.Combine(dir.Path, "b.safetensors");

        using (var c = Vindex3Container.Open(containerPath))
        {
            ModelFinetuner.FineTune(c, "target", dataPath, patchA, tokenizer, lr: 0.1, epochs: 2);
        }
        using (var c = Vindex3Container.Open(containerPath))
        {
            ModelFinetuner.FineTune(c, "target", dataPath, patchB, tokenizer, lr: 0.1, epochs: 2);
        }

        var pa = WeightPatch.Load(patchA);
        var pb = WeightPatch.Load(patchB);
        var ea = Assert.Single(pa.Entries);
        var eb = Assert.Single(pb.Entries);

        Assert.Equal(ea.Delta.Length, eb.Delta.Length);
        for (int i = 0; i < ea.Delta.Length; i++)
        {
            Assert.Equal(ea.Delta[i], eb.Delta[i]);
        }
    }
}