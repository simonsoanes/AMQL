using Amql.Inference;

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
            new ParallelOptions { MaxDegreeOfParallelism = ComputeBudget.Cores },
            () => (Gram: new double[d * d], Cross: new double[d * d], RowB: new double[d], RowA: new double[d]),
            (i, _, local) =>
            {
                var gram = local.Gram;
                var cross = local.Cross;
                int bRow = i * d;
                int aRow = i * d;
                // Hoist the anchor rows once: the original loop re-read the
                // same 40 KB rows from the float table d times (once per r)
                // and widened them per element; the double buffers below are
                // read d times from L1 instead. Rank-1 accumulation, so the
                // arithmetic per accumulator element is unchanged — the map
                // is bit-identical to the scalar form.
                var rowB = local.RowB;
                var rowA = local.RowA;
                for (int c = 0; c < d; c++)
                {
                    rowB[c] = b[bRow + c];
                    rowA[c] = a[aRow + c];
                }
                int vec = System.Numerics.Vector<double>.Count;
                for (int r = 0; r < d; r++)
                {
                    double br = rowB[r];
                    int rowBase = r * d;
                    var brv = new System.Numerics.Vector<double>(br);
                    int c = 0;
                    for (; c + vec <= d; c += vec)
                    {
                        (new System.Numerics.Vector<double>(gram, rowBase + c) + brv * new System.Numerics.Vector<double>(rowB, c))
                            .CopyTo(gram, rowBase + c);
                        (new System.Numerics.Vector<double>(cross, rowBase + c) + brv * new System.Numerics.Vector<double>(rowA, c))
                            .CopyTo(cross, rowBase + c);
                    }
                    for (; c < d; c++)
                    {
                        gram[rowBase + c] += br * rowB[c];
                        cross[rowBase + c] += br * rowA[c];
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
        SolveLower(chol, x, cross, d, d);
        SolveUpperTranspose(chol, x, d, d);

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
            new ParallelOptions { MaxDegreeOfParallelism = ComputeBudget.Cores },
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

    /// <summary>
    /// Rectangular ridge least squares: minimises Σ‖a_i − W·b_i‖² +
    /// ridge·‖W‖²_F over W ∈ R^{dOut × dIn} via its closed-form normal
    /// equations W = (Σ a_i b_iᵀ)·(Σ b_i b_iᵀ + λI)⁻¹. This is the
    /// calculated MTP projector fit (Phase 1 of the fit): the drafter's
    /// free block, fitted from the collected continuation pairs,
    /// closed-form and order-independent.
    /// </summary>
    public static float[] FitProjection(
        float[] a, float[] b, int n, int dOut, int dIn, double ridgeRel = 1e-4)
    {
        if (n <= 0)
        {
            throw new MergeException("no pairs to fit the MTP projector");
        }
        if (a.Length != n * dOut || b.Length != n * dIn)
        {
            throw new MergeException($"projection pairs do not match (n={n}, in={dIn}, out={dOut})");
        }

        var gram = new double[dIn * dIn];   // Σ x_i x_iᵀ
        var cross = new double[dIn * dOut]; // Σ x_i y_iᵀ
        for (int i = 0; i < n; i++)
        {
            for (int r = 0; r < dIn; r++)
            {
                double xr = b[i * dIn + r];
                for (int c = 0; c < dIn; c++)
                {
                    gram[r * dIn + c] += xr * b[i * dIn + c];
                }
                for (int c = 0; c < dOut; c++)
                {
                    cross[r * dOut + c] += xr * a[i * dOut + c];
                }
            }
        }

        double trace = 0;
        for (int i = 0; i < dIn; i++)
        {
            trace += gram[i * dIn + i];
        }
        double ridge = ridgeRel * (trace / dIn) + 1e-8;
        for (int i = 0; i < dIn; i++)
        {
            gram[i * dIn + i] += ridge;
        }

        var chol = Cholesky(gram, dIn);
        var x = new double[dIn * dOut]; // Wᵀ = (S + λI)⁻¹ · C
        SolveLower(chol, x, cross, dIn, dOut);
        SolveUpperTranspose(chol, x, dIn, dOut);

        var w = new float[dOut * dIn];
        for (int o = 0; o < dOut; o++)
        {
            for (int j = 0; j < dIn; j++)
            {
                w[o * dIn + j] = (float)x[j * dOut + o];
            }
        }
        return w;
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

    /// <summary>In-place forward substitution L·x = b (b row-major
    /// rows×cols, columns solved independently).</summary>
    private static void SolveLower(double[] l, double[] x, double[] b, int rows, int cols)
    {
        for (int col = 0; col < cols; col++)
        {
            for (int i = 0; i < rows; i++)
            {
                double sum = b[i * cols + col];
                for (int k = 0; k < i; k++)
                {
                    sum -= l[i * rows + k] * x[k * cols + col];
                }
                x[i * cols + col] = sum / l[i * rows + i];
            }
        }
    }

    /// <summary>In-place back substitution Lᵀ·x = b (x already holds the
    /// forward solution).</summary>
    private static void SolveUpperTranspose(double[] l, double[] x, int rows, int cols)
    {
        for (int col = 0; col < cols; col++)
        {
            for (int i = rows - 1; i >= 0; i--)
            {
                double sum = x[i * cols + col];
                for (int k = i + 1; k < rows; k++)
                {
                    sum -= l[k * rows + i] * x[k * cols + col];
                }
                x[i * cols + col] = sum / l[i * rows + i];
            }
        }
    }
}