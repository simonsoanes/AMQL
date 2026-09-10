using Amql.Cli;
using Amql.Hf;
using Amql.Inference;
using Amql.Merge;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// MoE-ification tests: the balanced unit clustering, and the end-to-end
/// restructure of the demo container (routed judgment, per-expert tensors,
/// the dense→MoE perplexity gate, and export).
/// </summary>
public class MoeIfyTests
{
    // ── balanced clustering ────────────────────────────────────────────────

    [Fact]
    public void ClusterUnits_Splits_Two_Modes_And_Balances()
    {
        // 4 tokens × 8 units: units 0-3 share the pattern [1,0,1,0];
        // units 4-7 share [0,1,0,1]. The clustering must put the two modes
        // apart (labels arbitrary) and leave exactly 4 units per expert.
        const int tokens = 4, intermediate = 8;
        var usage = new float[tokens * intermediate];
        for (int t = 0; t < tokens; t++)
        {
            for (int j = 0; j < intermediate; j++)
            {
                usage[t * intermediate + j] = (j < 4) == (t % 2 == 0) ? 1f : 0f;
            }
        }

        var partition = MoeIfy.ClusterUnits(new Tensor2D(usage, tokens, intermediate), tokens, intermediate, experts: 2);

        Assert.Equal(4, partition.Count(j => j == partition[0]));
        Assert.Equal(4, partition.Count(j => j != partition[0]));
        for (int j = 0; j < 4; j++)
        {
            Assert.Equal(partition[0], partition[j]);
        }
        for (int j = 4; j < 8; j++)
        {
            Assert.Equal(partition[4], partition[j]);
        }
        Assert.NotEqual(partition[0], partition[4]);
    }

    // ── end-to-end on the demo container ───────────────────────────────────

    private static string DemoContainer(TempDir dir)
    {
        var modelDir = Path.Combine(dir.Path, "model");
        SyntheticCheckpoint.Write(modelDir);
        var containerPath = Path.Combine(dir.Path, "container");
        ModelToContainer.Encode(modelDir, containerPath, "demo-moe");
        return containerPath;
    }

    private static int[] DemoTokens(string containerPath)
    {
        // The synthetic tokenizer is char-level over a..j (no 'k', no space).
        return HfTokenizer.FromModelDir(containerPath)
            .EncodeToIds("bcdefghij")
            .ToArray();
    }

    [Fact]
    public void MoeIfy_Restructures_And_Passes_The_Ppl_Gate()
    {
        using var dir = new TempDir();
        var containerPath = DemoContainer(dir);
        var ids = DemoTokens(containerPath);
        Assert.True(ids.Length >= 8, $"demo corpus encoded to {ids.Length} tokens");

        var outDir = Path.Combine(dir.Path, "moe");
        var report = MoeIfy.Transform(
            containerPath, outDir,
            clusterTokens: ids.Take(6).ToArray(),
            evalTokens: ids.Skip(6).Take(4).ToArray(),
            experts: 2, topK: 1);

        Assert.Equal(2, report.Experts);
        Assert.Equal(1, report.TopK);
        Assert.Equal(2, report.Layers);
        Assert.Equal(4, report.ExpertIntermediateSize); // 8 units / 2 experts
        Assert.Contains(report.Notes, n => n.Contains("perplexity"));
        Assert.True(double.IsFinite(report.DensePpl), "dense PPL must be computed");
        Assert.True(double.IsFinite(report.MoePpl), "moe PPL must be computed");
        Assert.True(report.MoePpl < report.DensePpl * 20,
            $"moe PPL {report.MoePpl:0.00} moved drastically away from dense {report.DensePpl:0.00}");

        // ── the restructure is in place ────────────────────────────────
        using var moe = Vindex3Container.Open(outDir);
        Assert.StartsWith("demo-moe-moe", moe.Index.Model);

        var stackId = moe.Graph!.Objects.Single(o => o.Kind == ObjectKind.DecoderStack && o.Component == "target").Id;
        var stackRep = moe.Index.Representations[moe.CanonicalRepresentationId(stackId)];
        using var segment = SegmentFile.Open(Path.Combine(moe.Root, stackRep.Segment));
        var names = segment.Header.Tensors.Select(t => t.Name).ToList();
        foreach (var dense in new[] { "0.mlp.gate_proj.weight", "1.mlp.down_proj.weight", "0.mlp.up_proj.weight" })
        {
            Assert.DoesNotContain(names, n => n == dense);
        }
        foreach (var expected in new[]
                 {
                     "0.mlp.router.weight",
                     "0.mlp.experts.0.gate_proj.weight",
                     "0.mlp.experts.1.up_proj.weight",
                     "1.mlp.experts.0.down_proj.weight",
                     "1.mlp.router.weight",
                 })
        {
            Assert.Contains(names, n => n == expected);
        }
        var routerTensor = segment.GetTensor("0.mlp.router.weight");
        Assert.Equal(2, routerTensor.Shape[0]);
        Assert.Equal(4, routerTensor.Shape[1]);
        var expertGate = segment.GetTensor("0.mlp.experts.0.gate_proj.weight");
        Assert.Equal(new long[] { 4, 4 }, expertGate.Shape); // 8 units / 2, hidden 4

        // ── the planner judges every FFN routed ────────────────────────
        using (var store = moe.CreateOperandStore())
        {
            var plan = Planner.Plan(moe, "target", store);
            for (int l = 0; l < plan.Layers.Count; l++)
            {
                var routed = plan.Layers[l].Ffn?.Routed;
                Assert.NotNull(routed);
                Assert.Equal(2, routed!.NumExperts);
                Assert.Equal(1, routed.TopK);
            }
        }

        // ── the surface carries the moe facts, and export works ────────
        var ffn = moe.Graph.Components.Single(c => c.Role == ComponentRole.PrimaryText)
            .Execution!.Ffn!;
        if (ffn.Moe is not { } moeFacts)
        {
            Assert.Fail("the moe surface facts are missing");
            return;
        }
        Assert.Equal(2, moeFacts.Experts);

        var exported = Path.Combine(dir.Path, "exported");
        ModelExporter.Export(moe, exported, patch: null);
        using var shard = SafetensorsFile.Open(Path.Combine(exported, "model.safetensors"));
        Assert.Contains("model.layers.0.mlp.router.weight", shard.TensorNames);
        Assert.Contains("model.layers.0.mlp.experts.0.gate_proj.weight", shard.TensorNames);
    }
}