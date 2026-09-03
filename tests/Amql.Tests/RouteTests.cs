using Amql.Cli;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>Tests for the relationship-probing machinery: the residual
/// patch seam, the causal tracer's invariants, and (guarded) the real
/// model's route output.</summary>
public class RouteTests
{
    private static string WriteSynthContainer(TempDir dir)
    {
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "synth-route");
        return containerPath;
    }

    // ── the patch seam ─────────────────────────────────────────────────────

    [Fact]
    public void PatchSeam_Overrides_The_Residual_Stream()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var rt = new GenericRuntime(plan, store);

        var ids = new[] { 1, 4, 2 };
        var cleanLogits = CausalTracer.RunPositionMajor(rt, plan, ids);
        var cleanRow = cleanLogits.Row(cleanLogits.Rows - 1);

        // Patching a constant vector into the residual stream at (layer 0,
        // row 0) must (a) change the output, and (b) be deterministic.
        var constant = new float[plan.HiddenSize];
        constant[0] = 3.5f;
        constant[1] = -1.25f;
        var patch = new GenericRuntime.ResidualPatch(0, 0, constant);

        var first = CausalTracer.RunPositionMajor(rt, plan, ids, patch);
        var second = CausalTracer.RunPositionMajor(rt, plan, ids, patch);
        Assert.Equal(first.Data, second.Data);

        bool changed = false;
        var row = first.Row(first.Rows - 1);
        for (int i = 0; i < row.Length; i++)
        {
            changed |= MathF.Abs(row[i] - cleanRow[i]) > 1e-3f;
        }
        Assert.True(changed, "a residual patch must alter the output stream");
    }

    // ── the causal tracer ──────────────────────────────────────────────────

    [Fact]
    public void CausalTracer_Invariants_And_Determinism()
    {
        using var dir = new TempDir();
        var containerPath = WriteSynthContainer(dir);
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var rt = new GenericRuntime(plan, store);

        var ids = new[] { 1, 4, 2 };
        var first = CausalTracer.Trace(rt, plan, ids, sourceRow: 0, targetIds: new[] { 2 }, corruptTokenId: 9, layerStart: 0, layerEnd: plan.Layers.Count);
        var second = CausalTracer.Trace(rt, plan, ids, sourceRow: 0, targetIds: new[] { 2 }, corruptTokenId: 9, layerStart: 0, layerEnd: plan.Layers.Count);

        // Shape + determinism.
        Assert.Equal(plan.Layers.Count, first.LayerDelta.Length);
        Assert.Equal(first.LayerDelta, second.LayerDelta);
        Assert.True(Math.Abs(first.CleanProbability - first.CorruptProbability) >= 0f);

        // Deltas are bounded by the total effect.
        foreach (var d in first.LayerDelta)
        {
            Assert.InRange(d, -2f, 2f);
        }
        // The shares of the traced layers roughly re-assemble the effect.
        double shareSum = first.LayerShare.Sum();
        Assert.True(shareSum > 0.5, $"share sum {shareSum} — attribution should reconstruct the effect");
    }

    // ── guarded real-model route ───────────────────────────────────────────

    private static readonly string RealContainer = @"D:\Dev\AMQL\containers\Qwen3.5-0.8B";
    private static readonly string RealModel = @"D:\Models\Qwen3.5-0.8B";
    private static bool RealAvailable => Directory.Exists(RealContainer) && Directory.Exists(RealModel);

    [Fact]
    public void Route_Real_Model_Emits_Links_And_Weights()
    {
        if (!RealAvailable)
        {
            return;
        }
        using var container = Vindex3Container.Open(RealContainer);
        var tokenizer = HfTokenizer.FromModelDir(RealModel);
        var (links, notes) = RelationRouter.Route(
            container, "target", tokenizer, "France", "Paris",
            new RouteOptions(Top: 3, MaxTemplates: 4, TraceLayerStart: 8, TraceLayerEnd: 10, NoTrace: false),
            null);

        Assert.NotEmpty(links);
        var top = links[0];
        Assert.InRange(top.Score, 0f, 1f);
        Assert.Contains(top.Relation, RelationRouter.Templates.Select(t => t.Name));

        // Coordinates: layer/head in range, weight in [0,1].
        foreach (var c in top.Coordinates.Take(3))
        {
            Assert.InRange(c.Layer, 0, 23);
            Assert.InRange(c.Head, 0, 7);
            Assert.InRange(c.Weight, 0f, 1f);
        }

        // Attribution: per-layer weights in range, at least one real Δ.
        var attr = top.Attribution;
        Assert.NotNull(attr);
        Assert.Equal(24, attr.LayerDelta.Length);
        Assert.True(attr.LayerDelta.Any(d => d > 1e-4f),
            "expected at least one layer to carry a measurable share of the link");
    }
}