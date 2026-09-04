using System.Text.Json;
using Amql.Cli;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Tests for the weight-patch machinery: authoring single-cell edits into
/// patch files (<c>change-tensor</c>), merging patches into the loaded
/// weights of every pathway, and factoring patches into LoRA adapters for
/// the original model (<c>save-lora</c>).
/// </summary>
public class PatchTests
{
    private static string WriteSynthContainer(TempDir dir)
    {
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "synth-patch");
        return containerPath;
    }

    private static float BaseCell(string containerPath, string objectId, string tensorName, long index)
    {
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var resolution = store.Resolve(objectId, tensorName);
        return BitPattern.WidenToF32(resolution.Dtype, resolution.Payload)[index];
    }

    private static IReadOnlyList<WeightPatchEntry> Edit(
        string containerPath, string objectId, string tensorName, long cell, TensorEditOp op, float value)
    {
        using var container = Vindex3Container.Open(containerPath);
        return TensorPatchTools.ApplyEdit(container, objectId, tensorName, op, value, cell,
            Array.Empty<WeightPatchEntry>()).Entries;
    }

    private static void Close(float expected, float actual, float tol = 1e-5f) =>
        Assert.True(MathF.Abs(expected - actual) <= tol,
            $"expected {expected}, got {actual}");

    // ── authoring + applying ───────────────────────────────────────────────

    [Fact]
    public void ChangeTensor_Creates_Patch_And_Merges_Into_Loaded_Weights()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var patchPath = Path.Combine(dir.Path, "patch.safetensors");

        long cell = 1 * 4 + 1; // embedding [12, 4], row 1 col 1
        float baseValue;
        IReadOnlyList<WeightPatchEntry> entries;
        using (var container = Vindex3Container.Open(containerPath))
        {
            baseValue = BaseCell(containerPath, "target.embedding", "weight", cell);
            var result = TensorPatchTools.ApplyEdit(container, "target.embedding", "weight",
                TensorEditOp.Set, -0.5f, cell, Array.Empty<WeightPatchEntry>());
            Close(baseValue, result.Before);
            Close(-0.5f, result.After);
            Assert.Single(result.Entries);
            entries = result.Entries;
            WeightPatch.Save(patchPath, entries, container.Index.Model);
        }

        // File round trip: model metadata, key, shape, delta.
        var patch = WeightPatch.Load(patchPath);
        Assert.Equal("synth-patch", patch.Model);
        var entry = Assert.Single(patch.Entries);
        Assert.Equal("target.embedding/weight", entry.Key);
        Assert.Equal(new long[] { 12, 4 }, entry.Shape);
        Close(-0.5f - baseValue, entry.Delta[cell]);

        // The loader merges the delta: patched cell is -0.5, the clean
        // loader still serves the base value.
        using var open = Vindex3Container.Open(containerPath);
        using var store = open.CreateOperandStore();
        var plan = Planner.Plan(open, "target", store);
        var clean = new WeightLoader(store).Matrix(plan.Embedding!.Table, 12, 4);
        var patched = new WeightLoader(store, patch).Matrix(plan.Embedding!.Table, 12, 4);
        Close(baseValue, clean.Data[cell]);
        Close(-0.5f, patched.Data[cell]);
    }

    [Fact]
    public void ApplyEdit_Add_Composes_Across_Runs()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        long cell = 1 * 4 + 1;
        float baseValue = BaseCell(containerPath, "target.embedding", "weight", cell);

        IReadOnlyList<WeightPatchEntry> entries;
        using (var container = Vindex3Container.Open(containerPath))
        {
            var first = TensorPatchTools.ApplyEdit(container, "target.embedding", "weight",
                TensorEditOp.Add, 0.25f, cell, Array.Empty<WeightPatchEntry>());
            Close(baseValue + 0.25f, first.After);
            entries = first.Entries;

            // second run sees base + existing delta and applies again
            var second = TensorPatchTools.ApplyEdit(container, "target.embedding", "weight",
                TensorEditOp.Add, 0.25f, cell, entries);
            Close(baseValue + 0.5f, second.After);
            Close(0.5f, second.Entries[0].Delta[cell]);
        }
    }

    [Fact]
    public void Edit_Back_To_Base_Clears_The_Entry()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        long cell = 1 * 4 + 1;
        float baseValue = BaseCell(containerPath, "target.embedding", "weight", cell);

        using var container = Vindex3Container.Open(containerPath);
        var edited = TensorPatchTools.ApplyEdit(container, "target.embedding", "weight",
            TensorEditOp.Set, baseValue + 0.3f, cell, Array.Empty<WeightPatchEntry>());
        Assert.Single(edited.Entries);

        var reverted = TensorPatchTools.ApplyEdit(container, "target.embedding", "weight",
            TensorEditOp.Set, baseValue, cell, edited.Entries);
        Assert.True(reverted.Removed, "an edit back to the base value must remove the entry");
        Assert.Empty(reverted.Entries);
    }

    // ── the patch reaches the pathways ─────────────────────────────────────

    [Fact]
    public void Patched_Runtime_Changes_Prefill_Logits()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var entries = Edit(containerPath, "target.embedding", "weight", 1 * 4 + 0,
            TensorEditOp.Set, -4f);
        var patch = WeightPatch.FromEntries(entries, "synth-patch");

        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var ids = new[] { 1, 4, 2 };

        var clean = CausalTracer.RunPositionMajor(new GenericRuntime(plan, store), plan, ids);
        var patched = CausalTracer.RunPositionMajor(
            new GenericRuntime(plan, store, patch), plan, ids);

        var cleanRow = clean.Row(clean.Rows - 1);
        var patchedRow = patched.Row(patched.Rows - 1);
        bool changed = false;
        for (int i = 0; i < cleanRow.Length; i++)
        {
            changed |= MathF.Abs(cleanRow[i] - patchedRow[i]) > 1e-3f;
        }
        Assert.True(changed, "a weight patch on a context token must alter the forward pass");
    }

    [Fact]
    public void InspectEmbedding_Reflects_Patch()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        long cell = 3 * 4 + 0; // token 3, dim 0
        float baseValue = BaseCell(containerPath, "target.embedding", "weight", cell);
        var patch = WeightPatch.FromEntries(
            Edit(containerPath, "target.embedding", "weight", cell, TensorEditOp.Set, 5f));

        using var container = Vindex3Container.Open(containerPath);
        var clean = TokenInspector.InspectEmbedding(container, "target", 3, 3);
        var patched = TokenInspector.InspectEmbedding(container, "target", 3, 3, patch);
        Close(baseValue, clean.Row[0]);
        Close(5f, patched.Row[0]);
    }

    [Fact]
    public void InspectLogits_Reflects_Patch()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var patch = WeightPatch.FromEntries(
            Edit(containerPath, "target.embedding", "weight", 1 * 4 + 0, TensorEditOp.Set, -4f));

        using var container = Vindex3Container.Open(containerPath);
        var clean = TokenInspector.InspectLogits(container, "target", 5, new[] { 1, 4, 2 }, 5);
        var patched = TokenInspector.InspectLogits(container, "target", 5, new[] { 1, 4, 2 }, 5, patch);
        Assert.True(Math.Abs(clean.Logit - patched.Logit) > 1e-4,
            "the patched embedding of a context token must move the logits");
    }

    [Fact]
    public void Generate_Replays_Deterministically_With_Patch()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var patch = WeightPatch.FromEntries(
            Edit(containerPath, "target.embedding", "weight", 1 * 4 + 0, TensorEditOp.Set, -4f));

        using var container = Vindex3Container.Open(containerPath);
        var config = new SamplingConfig(Seed: 42);
        var (_, steps1) = InferenceRunner.Generate(
            container, "target", new[] { 1, 4, 2 }, 3, config, showTopK: 4, patch);
        var (_, steps2) = InferenceRunner.Generate(
            container, "target", new[] { 1, 4, 2 }, 3, config, showTopK: 4, patch);

        Assert.Equal(steps1.Select(s => s.Token), steps2.Select(s => s.Token));
        Assert.All(steps1, s => Assert.NotNull(s.Candidates));
        Assert.All(steps1, s => Assert.InRange(s.Token, 0, 11));
    }

    // ── LoRA ───────────────────────────────────────────────────────────────

    [Fact]
    public void LoRA_Saves_Rank1_Delta_And_Reconstructs_With_Scale()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var patchPath = Path.Combine(dir.Path, "patch.safetensors");
        var loraDir = Path.Combine(dir.Path, "lora");

        long[] shape;
        using (var container = Vindex3Container.Open(containerPath))
        {
            shape = TensorPatchTools.ResolveShape(container, "target.decoder_stack",
                "0.self_attn.q_proj.weight");
        }
        int rows = (int)shape[0];
        int cols = (int)shape[1];
        long cell = 2 * cols + 1;

        var entries = Edit(containerPath, "target.decoder_stack", "0.self_attn.q_proj.weight",
            cell, TensorEditOp.Set, 0.75f);
        WeightPatch.Save(patchPath, entries);
        var report = LoraWriter.SaveAsLora(patchPath, loraDir, rank: 8, alpha: 16);

        var target = Assert.Single(report.Targets);
        Assert.Equal("target.decoder_stack/0.self_attn.q_proj.weight",
            target.ObjectId + "/" + target.TensorName);
        Assert.Equal(4, target.Rank); // rank clamped to min(8, rows, cols)
        Assert.True(target.ReconstructionError < 1e-4,
            $"rank-1 delta must reconstruct exactly, error {target.ReconstructionError}");

        // scale · lora_B · lora_A must reproduce the raw delta.
        using var file = SafetensorsFile.Open(Path.Combine(loraDir, "adapter_model.safetensors"));
        var aInfo = file.GetTensor(target.AName);
        var bInfo = file.GetTensor(target.BName);
        Assert.Equal(new long[] { 4, cols }, aInfo.Shape);
        Assert.Equal(new long[] { rows, 4 }, bInfo.Shape);
        var aMat = file.DecodeF32(aInfo);
        var bMat = file.DecodeF32(bInfo);

        var delta = entries[0].Delta;
        double scale = 16.0 / 4.0;
        double err = 0;
        double norm = 0;
        for (int m = 0; m < rows; m++)
        {
            for (int k = 0; k < cols; k++)
            {
                double rec = 0;
                for (int i = 0; i < 4; i++)
                {
                    rec += bMat[m * 4 + i] * aMat[i * cols + k];
                }
                double diff = delta[m * cols + k] - scale * rec;
                err += diff * diff;
                norm += (double)delta[m * cols + k] * delta[m * cols + k];
            }
        }
        Assert.True(Math.Sqrt(err / norm) < 1e-4,
            $"LoRA round trip must reproduce the delta, relative error {Math.Sqrt(err / norm)}");

        // adapter_config.json records the format and the mapping.
        using var config = JsonDocument.Parse(
            File.ReadAllText(Path.Combine(loraDir, "adapter_config.json")));
        Assert.Equal("amql-lora-v1", config.RootElement.GetProperty("format").GetString());
        Assert.Equal(16, config.RootElement.GetProperty("alpha").GetDouble());
        var targets = config.RootElement.GetProperty("targets");
        Assert.Equal("lora_A.0", targets[0].GetProperty("lora_a").GetString());
        Assert.Equal("lora_B.0", targets[0].GetProperty("lora_b").GetString());
    }

    [Fact]
    public void LoRA_Skips_OneDimensional_Targets()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var patchPath = Path.Combine(dir.Path, "patch.safetensors");
        var loraDir = Path.Combine(dir.Path, "lora");

        var entries = Edit(containerPath, "target.embedding", "weight", 5, TensorEditOp.Set, 0.5f);
        using (var container = Vindex3Container.Open(containerPath))
        {
            entries = TensorPatchTools.ApplyEdit(container, "target.final_norm", "weight",
                TensorEditOp.Set, 0.9f, 0, entries).Entries;
        }
        WeightPatch.Save(patchPath, entries);

        var report = LoraWriter.SaveAsLora(patchPath, loraDir, rank: 4, alpha: 8);
        Assert.Single(report.Targets);
        var skipped = Assert.Single(report.Skipped);
        Assert.Contains("final_norm", skipped);
    }

    // ── format guards ──────────────────────────────────────────────────────

    [Fact]
    public void WeightPatch_Load_Rejects_Plain_Safetensors()
    {
        using var dir = new TempDir();
        var path = Path.Combine(dir.Path, "plain.safetensors");
        SafetensorsWriter.Write(path, new[]
        {
            new TensorPayload { Name = "t", Dtype = Dtype.F32, Shape = new long[] { 4 }, Data = new byte[16] },
        });
        Assert.Throws<SafetensorsException>(() => WeightPatch.Load(path));
    }

    [Fact]
    public void ValidateAgainst_Rejects_Shape_Mismatch()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        var patchPath = Path.Combine(dir.Path, "bogus.safetensors");
        var wrongDelta = new float[36];
        wrongDelta[0] = 1f;
        WeightPatch.Save(patchPath, new[]
        {
            new WeightPatchEntry("target.embedding", "weight", new long[] { 6, 6 }, wrongDelta),
        });

        using var container = Vindex3Container.Open(containerPath);
        Assert.Throws<ContainerException>(() => WeightPatch.Load(patchPath).ValidateAgainst(container));
    }
}