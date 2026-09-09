namespace Amql.Merge;

/// <summary>
/// Ridge-regularised least-squares fitter for the embedding alignment:
/// given anchor pairs (a_i, b_i) — base row vs imported row of the same
/// token — finds the linear map M (d×d) minimising
/// Σ‖a_i − M·b_i‖² + ridge·‖M‖²_F. Solved via the normal equations
/// (BᵀB + ridge·I)·Mᵀ = BᵀA with a Cholesky factorisation; ridge keeps the
/// system solvable when anchors are fewer than dimensions. This is the
/// better-than-averaging normalisation for shared tokens: the imported
/// space is mapped into the base space before anything is blended or
/// appended.
/// </summary>
public static class LeastSquares
{
    /// <param name="a">Anchor rows of the base model, row-major n×d.</param>
    /// <param name="b">Anchor rows of the imported model, row-major n×d.</param>
    /// <returns>The d×d map M (row-major).</returns>
    public static float[] Fit(float[] a, float[] b, int n, int d, double ridgeRel = 1e-4)
    {
        if (n <= 0)
        {
            throw new MergeException(
                "no anchor tokens to fit the alignment — the two vocabularies appear to share no tokens");
        }
        if (a.Length != n * d || b.Length != n * d)
        {
            throw new MergeException("anchor rows do not match (n, d) — alignment is corrupt");
        }

        // Normal equations in double precision, accumulated in parallel
        // over anchor chunks. Each chunk's partial matrices are reduced at
        // the end; the anchor range per worker walks the row-major input
        // sequentially, so the reduction stays cache-friendly for the
        // ~10⁵-anchor × 10³-dimension case.
        var totalGram = new double[d * d];   // BᵀB
        var totalCross = new double[d * d];  // BᵀA
        Parallel.For(
            0,
            n,
            () => (Gram: new double[d * d], Cross: new double[d * d]),
            (i, _, local) =>
            {
                var gram = local.Gram;
                var cross = local.Cross;
                int bRow = i * d;
                for (int r = 0; r < d; r++)
                {
                    double br = b[bRow + r];
                    int rowBase = r * d;
                    for (int c = 0; c < d; c++)
                    {
                        gram[rowBase + c] += br * b[bRow + c];
                        cross[rowBase + c] += br * a[bRow + c];
                    }
                }
                return local;
            },
            local =>
            {
                for (int k = 0; k < d * d; k++)
                {
                    totalGram[k] += local.Gram[k];
                    totalCross[k] += local.Cross[k];
                }
            });
        var gram = totalGram;
        var cross = totalCross;

        // Relative ridge keeps the fitted map from overfitting near-singular
        // anchor sets while staying scale-invariant; the absolute floor
        // keeps a degenerate (zero-variance) anchor set — e.g. an imported
        // model whose token tables are all-zero — solvable, yielding the
        // zero map so the consensus gate defers to the scaffold.
        double trace = 0;
        for (int i = 0; i < d; i++)
        {
            trace += gram[i * d + i];
        }
        double ridge = ridgeRel * (trace / d) + 1e-8;
        for (int i = 0; i < d; i++)
        {
            gram[i * d + i] += ridge;
        }

        // Solve gram · X = cross for X = Mᵀ (Cholesky, positive definite by
        // construction with the ridge).
        var chol = Cholesky(gram, d);
        var x = new double[d * d];
        SolveLower(chol, x, cross, d);
        SolveUpperTranspose(chol, x, d);

        var result = new float[d * d];
        for (int i = 0; i < d * d; i++)
        {
            result[i] = (float)x[i];
        }
        return result;
    }

    /// <summary>Σ‖a_i − M·b_i‖² over the anchor pairs — the alignment
    /// residual reported in the manifest (rows are independent, so rows
    /// accumulate in parallel).</summary>
    public static double ResidualL2(float[] a, float[] b, int n, int d, float[] m)
    {
        double total = 0;
        var sync = new object();
        Parallel.For(
            0,
            n,
            () => 0.0,
            (i, _, local) =>
            {
                int bRow = i * d;
                int aRow = i * d;
                for (int r = 0; r < d; r++)
                {
                    double predicted = 0;
                    for (int c = 0; c < d; c++)
                    {
                        predicted += m[r * d + c] * b[bRow + c];
                    }
                    double diff = a[aRow + r] - predicted;
                    local += diff * diff;
                }
                return local;
            },
            local =>
            {
                lock (sync)
                {
                    total += local;
                }
            });
        return total;
    }

    private static double[] Cholesky(double[] a, int d)
    {
        var l = new double[d * d];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = a[i * d + j];
                for (int k = 0; k < j; k++)
                {
                    sum -= l[i * d + k] * l[j * d + k];
                }
                if (i == j)
                {
                    if (sum <= 0)
                    {
                        throw new MergeException(
                            $"alignment system is not positive definite (diagonal {sum:g3} at {i}) — the anchor set cannot be aligned");
                    }
                    l[i * d + i] = Math.Sqrt(sum);
                }
                else
                {
                    l[i * d + j] = sum / l[j * d + j];
                }
            }
        }
        return l;
    }

    /// <summary>In-place forward substitution L·x = b (b row-major d×d,
    /// columns solved independently).</summary>
    private static void SolveLower(double[] l, double[] x, double[] b, int d)
    {
        for (int col = 0; col < d; col++)
        {
            for (int i = 0; i < d; i++)
            {
                double sum = b[i * d + col];
                for (int k = 0; k < i; k++)
                {
                    sum -= l[i * d + k] * x[k * d + col];
                }
                x[i * d + col] = sum / l[i * d + i];
            }
        }
    }

    /// <summary>In-place back substitution Lᵀ·x = b (x already holds the
    /// forward solution).</summary>
    private static void SolveUpperTranspose(double[] l, double[] x, int d)
    {
        for (int col = 0; col < d; col++)
        {
            for (int i = d - 1; i >= 0; i--)
            {
                double sum = x[i * d + col];
                for (int k = i + 1; k < d; k++)
                {
                    sum -= l[k * d + i] * x[k * d + col];
                }
                x[i * d + col] = sum / l[i * d + i];
            }
        }
    }
}