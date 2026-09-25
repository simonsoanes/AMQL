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
            recorder.EndStep(tokenId: 100 + step, Array.Empty<TokenCandidate>(), 0.5f, 0.25f);
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
        recorder.EndStep(
            tokenId: 4242,
            new[] { new TokenCandidate(4242, 9.5f, 0.6f), new TokenCandidate(7, 8.25f, 0.18f) },
            entropy: 1.75f,
            top1Margin: 0.42f);

        var original = recorder.ToRunTrace("Qwen3.5-0.8B", "target.decoder_stack", 1024, 24,
            new[] { 760, 6511 }, "temperature=0", "ResidentF32");
        TraceRecorder.WriteJson(original, path);

        Assert.True(File.Exists(path));
        var read = TraceRecorder.ReadJson(path);

        Assert.Equal(original.Model, read.Model);
        Assert.Equal(original.HiddenSize, read.HiddenSize);
        Assert.Equal(original.Layers, read.Layers);
        Assert.Equal(original.PromptTokens, read.PromptTokens);
        Assert.Equal(original.Sampling, read.Sampling);
        Assert.Equal(original.WeightWorkingSet, read.WeightWorkingSet);

        var node = Assert.Single(read.Nodes);
        Assert.Equal(2, node.Layer);
        Assert.Equal("ffn_dense", node.Op);
        Assert.Equal(KProj.ObjectId, node.Weight!.ObjectId);
        Assert.Equal(KProj.TensorName, node.Weight.TensorName);

        var step = Assert.Single(read.Steps);
        Assert.Equal(5, step.Position);
        Assert.Equal(4242, step.TokenId);
        Assert.Equal(1.75f, step.Entropy);
        Assert.Equal(0.42f, step.Top1Margin);
        Assert.Equal(new[] { 1, 4, 9 }, step.RoutedExperts);
        Assert.Equal(2, step.TopK.Count);
        Assert.Equal(9.5f, step.TopK[0].Logit);
        Assert.Equal(0.6f, step.TopK[0].Probability);

        var sample = Assert.Single(step.Ops);
        Assert.Equal(node.Id, sample.NodeId);
        Assert.Equal(1.5f, sample.L2);
        Assert.Equal(0.25, sample.Ms, precision: 9);
    }
}
