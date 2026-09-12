using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>The drafter's free block: the single map <c>fc.weight</c>
/// (phase 1) or the K-projector mixture's routed stack
/// <c>fc.router.weight</c> + <c>fc.experts.{{e}}.weight</c> (phase 2).
/// Rows route to the expert of highest router affinity
/// <c>p_t = W_{{argmax_j r_j·x_t}}·x_t</c>; the single map has no router.
/// Evaluating an in-memory block and measuring through the materialised
/// container run the identical math, so the K-sweep's in-memory numbers
/// are exactly the numbers the shipped container will reproduce.</summary>
public sealed class DraftProjector
{
    public int Hidden { get; }
    public float[]? W { get; }
    public float[]? Router { get; }
    public float[][]? Experts { get; }
    public int Clusters => Experts?.Length ?? 1;

    private DraftProjector(int hidden, float[]? w, float[]? router, float[][]? experts)
    {
        Hidden = hidden;
        W = w;
        Router = router;
        Experts = experts;
    }

    public static DraftProjector Single(float[] w, int hidden) => new(hidden, w, null, null);

    public static DraftProjector Routed(float[] router, float[][] experts, int hidden) =>
        new(hidden, null, router, experts);

    /// <summary>Loads the free block the container's mtp.stack currently
    /// carries — routed when <c>fc.router.weight</c> is materialised, else
    /// the single <c>fc.weight</c>; null when the drafter module is
    /// absent.</summary>
    public static DraftProjector? LoadFrom(Vindex3Container container, OperandStore store, int hidden)
    {
        if (container.Graph?.Objects.FirstOrDefault(o => o.Id == "mtp.stack") is not { Representations.Count: > 0 })
        {
            return null;
        }
        if (store.ContainsTensor("mtp.stack", "fc.router.weight"))
        {
            var routerResolved = store.Resolve("mtp.stack", "fc.router.weight");
            var router = BitPattern.WidenToF32(routerResolved.Dtype, routerResolved.Payload);
            int k = router.Length / (2 * hidden);
            var experts = new float[k][];
            for (int e = 0; e < k; e++)
            {
                var expert = store.Resolve("mtp.stack", $"fc.experts.{e}.weight");
                experts[e] = BitPattern.WidenToF32(expert.Dtype, expert.Payload);
            }
            return Routed(router, experts, hidden);
        }
        var fc = store.Resolve("mtp.stack", "fc.weight");
        return Single(BitPattern.WidenToF32(fc.Dtype, fc.Payload), hidden);
    }

    /// <summary>Projects the (rows × 2h) input to (rows × h): the single
    /// map via one matmul; the routed block per row via the argmax
    /// router.</summary>
    public float[] Project(float[] input, int rows)
    {
        var matrix = new Tensor2D(input, rows, 2 * Hidden);
        if (W is not null)
        {
            return TensorOps.MatMulTransposedB(matrix, new Tensor2D(W, Hidden, 2 * Hidden)).Data;
        }
        var perExpert = new float[Clusters][];
        for (int e = 0; e < Clusters; e++)
        {
            perExpert[e] = TensorOps.MatMulTransposedB(
                matrix, new Tensor2D(Experts![e], Hidden, 2 * Hidden)).Data;
        }
        var router = new Tensor2D(Router!, Clusters, 2 * Hidden);
        var output = new float[rows * Hidden];
        for (int t = 0; t < rows; t++)
        {
            int best = 0;
            double bestAffinity = double.NegativeInfinity;
            for (int e = 0; e < Clusters; e++)
            {
                double affinity = 0;
                for (int c = 0; c < 2 * Hidden; c++)
                {
                    affinity += router.Data[e * (2 * Hidden) + c] * input[t * (2 * Hidden) + c];
                }
                if (affinity > bestAffinity)
                {
                    bestAffinity = affinity;
                    best = e;
                }
            }
            Array.Copy(perExpert[best], t * Hidden, output, t * Hidden, Hidden);
        }
        return output;
    }
}

/// <summary>One capture of the drafter's inputs over the sample — the
/// model's final-normed hidden states and the next-token embedding rows —
/// plus the trunk pipeline bound to the container's mtp.stack. The dense
/// forward runs once; many free blocks are scored against the same
/// context, so the K-sweep never repeats the model forward. Scoring an
/// in-memory block and scoring through the materialised container share
/// the identical trunk, norms and head (the gate is the runtime path).</summary>
internal sealed class DraftContext : IDisposable
{
    private readonly Vindex3Container _container;
    private readonly OperandStore _store;
    private readonly ComponentOpPlan _densePlan;
    private readonly int _sampled;
    private readonly float[] _normed;
    private readonly float[] _embeddings;
    private readonly IReadOnlyList<int> _tokens;
    private readonly ComponentOpPlan _trunkPlan;
    private readonly float[] _preFcNorm;
    private readonly double _eps;
    private readonly float _weightOffset;

    public int Hidden => _densePlan.HiddenSize;
    public int Useable => _sampled - 2;

    internal DraftContext(string containerDir, IReadOnlyList<int> tokens, int sampled)
    {
        if (sampled < 4)
        {
            throw new MergeException("need at least 4 sample tokens to measure draft acceptance");
        }
        _container = Vindex3Container.Open(containerDir);
        _store = _container.CreateOperandStore();
        _densePlan = Planner.Plan(_container, "target", _store);
        int trunk = GenerateMtp.LastFullAttentionLayer(_densePlan);
        _tokens = tokens;
        _sampled = sampled;

        int hidden = Hidden;
        _normed = new float[Useable * hidden];
        _embeddings = new float[Useable * hidden];

        var dense = new GenericRuntime(_densePlan, _store);
        var finalNorm = BitPattern.WidenToF32(
            _store.Resolve(_densePlan.FinalNorm.Weight.ObjectId, _densePlan.FinalNorm.Weight.TensorName).Dtype,
            _store.Resolve(_densePlan.FinalNorm.Weight.ObjectId, _densePlan.FinalNorm.Weight.TensorName).Payload);
        var table = dense.Weights.Matrix(
            _densePlan.Embedding!.Table, _densePlan.Embedding.VocabSize, _densePlan.Embedding.HiddenSize);

        for (int t = 0; t < sampled; t++)
        {
            var h = dense.StepForward(tokens[t]);
            if (t < Useable)
            {
                var n = (float[])h.Data.Clone();
                var view = new Tensor2D(n, h.Rows, h.Cols);
                Norms.ApplyInPlace(view, _densePlan.FinalNorm.Kind, _densePlan.FinalNorm.Eps,
                    finalNorm, _densePlan.FinalNorm.WeightOffset);
                Array.Copy(n, 0, _normed, t * hidden, hidden);
                var embedRow = TensorOps.GatherRows(table, new[] { tokens[t + 1] });
                Array.Copy(embedRow.Data, 0, _embeddings, t * hidden, hidden);
            }
        }

        // The drafter's trunk plan: layer 0 = the copied full-attention
        // layer, rebound onto mtp.stack; the shared embedding/head remain.
        var trunkPlan = GenerateMtp.BuildTrunkPlan(_container, containerDir, _densePlan, trunk, hidden);
        _trunkPlan = trunkPlan;
        _eps = _densePlan.FinalNorm.Eps;
        _weightOffset = _densePlan.FinalNorm.WeightOffset;
        var preFc = _store.Resolve("mtp.stack", "pre_fc_norm_hidden.weight");
        _preFcNorm = BitPattern.WidenToF32(preFc.Dtype, preFc.Payload);
    }

    /// <summary>Scores the free block currently materialised in the
    /// container's mtp.stack (single or routed) through the runtime
    /// path.</summary>
    public double Score() =>
        Score(DraftProjector.LoadFrom(_container, _store, Hidden)
            ?? throw new MergeException(
                "the container carries no materialised MTP drafter — run 'amql-cli generate-mtp' first"));

    /// <summary>Scores an in-memory free block over the same capture —
    /// the K-sweep's per-candidate measurement through the same trunk,
    /// norms and head.</summary>
    public double Score(DraftProjector block)
    {
        int hidden = Hidden;
        int useable = Useable;
        var input = new float[useable * 2 * hidden];
        for (int t = 0; t < useable; t++)
        {
            // Each half is normalised with the copied final-norm weights
            // (pre_fc_norm_hidden and pre_fc_norm_embedding).
            var halfH = new float[hidden];
            var halfE = new float[hidden];
            Array.Copy(_normed, t * hidden, halfH, 0, hidden);
            Array.Copy(_embeddings, t * hidden, halfE, 0, hidden);
            Norms.ApplyInPlace(new Tensor2D(halfH, 1, hidden), NormType.RmsNorm, _eps, _preFcNorm, _weightOffset);
            Norms.ApplyInPlace(new Tensor2D(halfE, 1, hidden), NormType.RmsNorm, _eps, _preFcNorm, _weightOffset);
            Array.Copy(halfH, 0, input, t * 2 * hidden, hidden);
            Array.Copy(halfE, 0, input, t * 2 * hidden + hidden, hidden);
        }

        // A fresh trunk runtime per score: the trunk layer's KV cache must
        // start empty for each candidate — the sweep scores the same
        // sequence once per free block, exactly as the container path does.
        var trunkRuntime = new GenericRuntime(_trunkPlan, _store);
        var projected = new Tensor2D(block.Project(input, useable), useable, hidden);
        var positions = Enumerable.Range(0, useable).ToArray();
        var trunkOut = trunkRuntime.RunLayerInternal(projected, 0, positions, positions, appendKv: true);
        var logits = trunkRuntime.FinalNormAndHead(trunkOut);

        int matches = 0;
        for (int t = 0; t < useable; t++)
        {
            var row = logits.Row(t);
            int best = 0;
            for (int i = 1; i < row.Length; i++)
            {
                if (row[i] > row[best])
                {
                    best = i;
                }
            }
            if (best == _tokens[t + 2])
            {
                matches++;
            }
        }
        return (double)matches / useable;
    }

    public void Dispose()
    {
        _store.Dispose();
        _container.Dispose();
    }
}