using Amql.Inference.Tracing;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// Tests for the operator trace recorder: node interning, per-run aggregation,
/// and the JSON round trip a saved run has to survive to be reopened and
/// compared against another run.
/// </summary>
public class TraceTests
{
    private static readonly OperandRef QProj = new("target.decoder_stack", "layers.0.self_attn.q_proj.weight");
    private static readonly OperandRef KProj = new("target.decoder_stack", "layers.0.self_attn.k_proj.weight");

    private static OpObservation Op(int layer, string op, OperandRef? weight, float l2, double ms = 0)
        => new(layer, op, weight, l2, l2 / 2f, l2 / 16f, ms);

    [Fact]
    public void Nodes_Are_Interned_By_Layer_Op_And_Weight()
    {
        var recorder = new TraceRecorder();

        int first = recorder.NodeId(0, "attn_q", QProj);
        int again = recorder.NodeId(0, "attn_q", QProj);
        int otherWeight = recorder.NodeId(0, "attn_q", KProj);
        int otherLayer = recorder.NodeId(1, "attn_q", QProj);
        int noWeight = recorder.NodeId(0, "residual_out", null);

        Assert.Equal(first, again);
        Assert.NotEqual(first, otherWeight);
        Assert.NotEqual(first, otherLayer);
        Assert.NotEqual(first, noWeight);
        Assert.Equal(4, recorder.Nodes.Count);

        // The weight is carried as an OperandRef so a consumer can feed it
        // straight back into TensorPatchTools.ApplyEdit.
        Assert.Equal(QProj, recorder.Nodes[first].Weight);
        Assert.Null(recorder.Nodes[noWeight].Weight);
    }

    [Fact]
    public void Aggregate_Reduces_Every_Step_Per_Node()
    {
        var recorder = new TraceRecorder();
        for (int step = 0; step < 3; step++)
        {
            recorder.BeginStep(step);
            recorder.Observe(Op(0, "attn_q", QProj, l2: 2f + step, ms: 1.0));
            recorder.ObserveExperts(new[] { 3, 7 });
            recorder.EndStep(tokenId: 100 + step, tokenText: null, Array.Empty<TokenCandidate>(), 0.5f, 0.25f);
        }

        var trace = recorder.ToRunTrace("m", "c", 8, 1, new[] { 1, 2, 3 }, "greedy", "F32");
        Assert.Equal(3, trace.Steps.Count);
        Assert.Equal(3, trace.Steps.Sum(s => s.Ops.Count));
        Assert.Equal(6, trace.Steps.Sum(s => s.RoutedExperts.Count));

        var aggregate = trace.Aggregate();
        var node = Assert.Single(aggregate);
        Assert.Equal(3f, node.Value.MeanL2, precision: 5);   // (2 + 3 + 4) / 3
        Assert.Equal(4f, node.Value.MaxL2, precision: 5);
        Assert.Equal(3, node.Value.Steps);
        Assert.Equal(1.0, node.Value.MeanMs, precision: 5);
    }

    [Fact]
    public void Trace_Survives_The_Json_Round_Trip()
    {
        using var dir = new TempDir();
        string path = Path.Combine(dir.Path, "nested", "trace.json");

        var recorder = new TraceRecorder();
        recorder.BeginStep(5);
        recorder.Observe(Op(2, "ffn_dense", KProj, l2: 1.5f, ms: 0.25));
        recorder.ObserveExperts(new[] { 1, 4, 9 });
        recorder.EndStep(tokenId: 4242, tokenText: " Paris",
            new[] { new TokenCandidate(4242, 9.5f, 0.6f, " Paris"), new TokenCandidate(7, 8.25f, 0.18f, " Lyon") },
            entropy: 1.75f, top1Margin: 0.42f);

        var original = recorder.ToRunTrace("Qwen3.5-0.8B", "target.decoder_stack", 1024, 24,
            new[] { 760, 6511 }, "temperature=0", "ResidentF32",
            containerPath: Path.Combine("containers", "Qwen3.5-0.8B"),
            causal: new CausalInfo(3, 760, 9564, 279, 0.1245f, 0.0806f,
                new float[] { 0.02f, -0.01f, 0.05f }, new float[] { 0.34f, -0.17f, 0.83f }));
        TraceRecorder.WriteJson(original, path);

        Assert.True(File.Exists(path));
        var read = TraceRecorder.ReadJson(path);

        Assert.Equal(original.Model, read.Model);
        Assert.Equal(original.HiddenSize, read.HiddenSize);
        Assert.Equal(original.Layers, read.Layers);
        Assert.Equal(original.PromptTokens, read.PromptTokens);
        Assert.Equal(original.Sampling, read.Sampling);
        Assert.Equal(original.WeightWorkingSet, read.WeightWorkingSet);
        Assert.Equal(original.ContainerPath, read.ContainerPath);

        // the attribution block has to survive too, or the causal overlay is
        // only ever available in the session that produced it
        var causal = read.Causal;
        Assert.NotNull(causal);
        Assert.Equal(3, causal!.SourceRow);
        Assert.Equal(760, causal.SourceTokenId);
        Assert.Equal(9564, causal.CorruptTokenId);
        Assert.Equal(279, causal.TargetTokenId);
        Assert.Equal(0.1245f, causal.CleanProbability, precision: 5);
        Assert.Equal(0.0806f, causal.CorruptProbability, precision: 5);
        Assert.Equal(0.0439f, causal.TotalEffect, precision: 4);
        Assert.Equal(new float[] { 0.34f, -0.17f, 0.83f }, causal.LayerShare);
        Assert.Equal(2, causal.PeakLayer);

        var node = Assert.Single(read.Nodes);
        Assert.Equal(2, node.Layer);
        Assert.Equal("ffn_dense", node.Op);
        Assert.Equal(KProj.ObjectId, node.Weight!.ObjectId);
        Assert.Equal(KProj.TensorName, node.Weight.TensorName);

        var step = Assert.Single(read.Steps);
        Assert.Equal(5, step.Position);
        Assert.Equal(4242, step.TokenId);
        Assert.Equal(" Paris", step.TokenText);
        Assert.Equal(1.75f, step.Entropy);
        Assert.Equal(0.42f, step.Top1Margin);
        Assert.Equal(new[] { 1, 4, 9 }, step.RoutedExperts);
        Assert.Equal(2, step.TopK.Count);
        Assert.Equal(9.5f, step.TopK[0].Logit);
        Assert.Equal(0.6f, step.TopK[0].Probability);
        Assert.Equal(" Paris", step.TopK[0].Text);

        var sample = Assert.Single(step.Ops);
        Assert.Equal(node.Id, sample.NodeId);
        Assert.Equal(1.5f, sample.L2);
        Assert.Equal(0.25, sample.Ms, precision: 9);
    }

    // ── reductions and layout the visualiser draws from ───────────────────

    /// <summary>Builds a small two-layer run: layer 0 has a weighted projection
    /// and a norm, layer 1 only a norm, over three steps with rising L2.</summary>
    private static RunTrace SampleRun()
    {
        var recorder = new TraceRecorder();
        for (int step = 0; step < 3; step++)
        {
            recorder.BeginStep(step);
            recorder.Observe(Op(0, "attn_q", QProj, l2: 2f * (step + 1), ms: 1.0));
            recorder.Observe(Op(0, "pre_ffn_norm", null, l2: 1f * (step + 1), ms: 0.5));
            recorder.Observe(Op(1, "pre_ffn_norm", null, l2: 8f, ms: 0.25));
            recorder.EndStep(100 + step, null, Array.Empty<TokenCandidate>(), 0.5f, 0.25f);
        }
        return recorder.ToRunTrace("m", "c", 8, 2, new[] { 1 }, "greedy", "F32");
    }

    [Fact]
    public void Reduce_Computes_Mean_Max_And_Relative_Intensity()
    {
        var stats = TraceMetrics.Reduce(SampleRun()).ToDictionary(s => (s.Layer, s.Op));

        var q = stats[(0, "attn_q")];
        Assert.Equal(4f, q.MeanL2, precision: 5);   // (2 + 4 + 6) / 3
        Assert.Equal(6f, q.MaxL2, precision: 5);
        Assert.Equal(3, q.Steps);
        Assert.Equal(1.0, q.MeanMs, precision: 5);
        Assert.True(q.HasWeight);
        Assert.Equal(QProj.ObjectId, q.WeightObject);

        var norm0 = stats[(0, "pre_ffn_norm")];
        Assert.Equal(2f, norm0.MeanL2, precision: 5);
        Assert.False(norm0.HasWeight);

        // Intensity is relative to the busiest operator in the run, which is
        // layer 1's norm at a constant 8.
        var norm1 = stats[(1, "pre_ffn_norm")];
        Assert.Equal(1f, norm1.Intensity, precision: 5);
        Assert.Equal(0.5f, q.Intensity, precision: 5);
        Assert.Equal(0.25f, norm0.Intensity, precision: 5);
    }

    [Fact]
    public void Ranking_Puts_The_Busiest_And_The_Slowest_First()
    {
        var run = SampleRun();
        var byL2 = TraceMetrics.RankByMeanL2(run);
        Assert.Equal("pre_ffn_norm", byL2[0].Op);
        Assert.Equal(1, byL2[0].Layer);
        Assert.Equal("attn_q", TraceMetrics.RankByMeanMs(run)[0].Op);
    }

    [Fact]
    public void OpSlots_Keep_Canonical_Order_And_Append_Unknown_Ops()
    {
        var slots = TraceMetrics.OpSlots(SampleRun().Nodes);

        // the canonical order is what makes a layer column read as a computation
        Assert.True(slots["pre_attn_norm"] < slots["attn_q"]);
        Assert.True(slots["attn_q"] < slots["attn_k"]);
        Assert.True(slots["attn_output"] < slots["pre_ffn_norm"]);
        Assert.True(slots["pre_ffn_norm"] < slots["ffn_routed"]);
        Assert.True(slots["post_ffn_norm"] < slots["residual_out"]);

        // an operator that is not in the table still gets a stable slot rather
        // than disappearing from the map
        var extended = TraceMetrics.OpSlots(
            SampleRun().Nodes.Append(new OpNode(99, 0, "brand_new_op", null)).ToArray());
        Assert.True(extended["brand_new_op"] >= TraceMetrics.OpOrder.Count);
    }

    [Fact]
    public void StepValues_Normalise_Within_The_Selected_Step()
    {
        var run = SampleRun();
        var last = TraceMetrics.StepValues(run, run.Steps.Count - 1);
        int qId = run.Nodes.First(n => n.Op == "attn_q").Id;
        int norm1Id = run.Nodes.First(n => n.Layer == 1).Id;

        // layer 1's norm (8) outranks layer 0's projection (6) in this step
        Assert.Equal(1f, last[norm1Id], precision: 5);
        Assert.Equal(6f / 8f, last[qId], precision: 5);

        // out-of-range steps give an empty map rather than throwing
        Assert.Empty(TraceMetrics.StepValues(run, 99));
        // 2 layer columns; the row count is the canonical operator table, since
        // the sample run introduces no operators outside it
        Assert.Equal((2, TraceMetrics.OpOrder.Count), TraceMetrics.LayoutSize(run));
    }

    [Fact]
    public void Compare_Matches_Nodes_By_Identity_Not_By_Interned_Id()
    {
        // Two runs of the same model whose operators were first seen in a
        // different order, so the interned ids disagree. Matching on id would
        // silently compare the wrong operators against each other.
        var first = new TraceRecorder();
        first.BeginStep(0);
        first.Observe(Op(0, "attn_q", QProj, l2: 4f));
        first.Observe(Op(0, "ffn_dense", KProj, l2: 2f));
        first.EndStep(1, null, Array.Empty<TokenCandidate>(), 0f, 0f);
        var baseline = first.ToRunTrace("m", "c", 8, 1, new[] { 1 }, "greedy", "F32");

        var second = new TraceRecorder();
        second.BeginStep(0);
        second.Observe(Op(0, "ffn_dense", KProj, l2: 3f));   // seen first this run
        second.Observe(Op(0, "attn_q", QProj, l2: 4f));
        second.EndStep(1, null, Array.Empty<TokenCandidate>(), 0f, 0f);
        var modified = second.ToRunTrace("m", "c", 8, 1, new[] { 1 }, "greedy", "F32");

        // the ids genuinely do differ, so this test can fail
        Assert.NotEqual(
            baseline.Nodes.First(n => n.Op == "attn_q").Id,
            modified.Nodes.First(n => n.Op == "attn_q").Id);
        Assert.True(TraceMetrics.Comparable(baseline, modified));

        var deltas = TraceMetrics.Compare(baseline, modified).ToDictionary(d => d.Op);
        Assert.Equal(2, deltas.Count);

        // unchanged operator
        Assert.Equal(0f, deltas["attn_q"].AbsoluteChange, precision: 5);
        Assert.Equal(0f, deltas["attn_q"].Intensity, precision: 5);

        // the one that moved: 2 → 3
        var moved = deltas["ffn_dense"];
        Assert.Equal(2f, moved.BaselineMeanL2, precision: 5);
        Assert.Equal(3f, moved.ModifiedMeanL2, precision: 5);
        Assert.Equal(1f, moved.AbsoluteChange, precision: 5);
        Assert.True(moved.Increased);
        Assert.Equal(0.5f, moved.RelativeChange, precision: 5);   // 1 / 2
        Assert.Equal(1f, moved.Intensity, precision: 5);          // largest movement in the run
        Assert.True(moved.HasWeight);
        Assert.Equal(KProj.TensorName, moved.WeightTensor);
    }
}
