using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// Routed experts stored one tensor per projection (see <see cref="PackedExpertsOp"/>)
/// on the row-exact path: sigmoid scores, the selection bias, weights normalised
/// over the chosen experts and the route scale; a softcapped GLU; only the
/// chosen experts' rows decoded from their stored format.
/// </summary>
public sealed partial class GenericRuntime
{
    private float[] PackedExpertsRow(int layer, RoutedFfnOp routed, float[] x)
    {
        var packed = routed.Packed!;
        var policy = (ExpertGateSoftcappedGlu)routed.GatePolicy;
        int d = routed.HiddenSize, experts = routed.NumExperts, width = routed.ExpertIntermediateSize;
        var router = Rows.Vector(routed.Router, experts * d);
        var selectionBias = routed.SelectionBias is { } b ? Rows.Vector(b, experts) : null;
        var scores = new double[experts];
        for (int e = 0; e < experts; e++)
        {
            scores[e] = 1.0 / (1.0 + Math.Exp(-RowMath.Dot(router.AsSpan(e * d, d), x)));
        }
        var chosen = Enumerable.Range(0, experts)
            .OrderByDescending(e => selectionBias is null ? scores[e] : scores[e] + selectionBias[e])
            .Take(routed.TopK)
            .ToArray();
        ExpertRoutingTrace?.Invoke(layer, chosen);
        double total = chosen.Sum(e => scores[e]) + 1e-20;

        var gateUp = Rows.Experts(packed.GateUp, experts, 2 * width, d);
        var down = Rows.Experts(packed.Down, experts, d, width);
        var acc = new double[d];
        var act = new float[width];
        double gateCap = policy.GateCap, upCap = policy.UpCap;
        // Each row's dot product is the same whichever worker takes it.
        bool spread = (long)width * d >= RowMatrix.ParallelElements;
        foreach (int e in chosen)
        {
            double weight = scores[e] / total * routed.RouteScale;
            void Up(int j, float[] buffer)
            {
                // Gate row 2j, up row 2j + 1.
                gateUp.Row(e, 2 * j, buffer);
                double g = RowMath.Dot(buffer, x);
                gateUp.Row(e, 2 * j + 1, buffer);
                double u = RowMath.Dot(buffer, x);
                act[j] = (float)(Math.Tanh(g / gateCap) * gateCap / (1.0 + Math.Exp(-g)) * (Math.Tanh(u / upCap) * upCap));
            }
            void Down(int i, float[] buffer)
            {
                down.Row(e, i, buffer);
                acc[i] += weight * RowMath.Dot(buffer, act);
            }
            if (spread)
            {
                Parallel.For(0, width, RowMatrix.Budget, () => new float[d], (j, _, buffer) => { Up(j, buffer); return buffer; }, _ => { });
                Parallel.For(0, d, RowMatrix.Budget, () => new float[width], (i, _, buffer) => { Down(i, buffer); return buffer; }, _ => { });
            }
            else
            {
                var upRow = new float[d];
                for (int j = 0; j < width; j++)
                {
                    Up(j, upRow);
                }
                var downRow = new float[width];
                for (int i = 0; i < d; i++)
                {
                    Down(i, downRow);
                }
            }
        }
        var y = new float[d];
        for (int i = 0; i < d; i++)
        {
            y[i] = (float)acc[i];
        }
        return y;
    }
}
