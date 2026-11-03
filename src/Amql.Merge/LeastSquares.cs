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

        // CUDA path: the normal-equation Gram/Cross over the anchors is a
        // pair of blocked fp32 GEMMs (BᵀB, BᵀA) that the 5090 computes in
        // seconds where the fp64 CPU accumulation below takes ~40 minutes
        // at 27B scale. The fp64 per-block partials sum on the host, so
        // the Cholesky solve and the fitted map stay the fp64 authority.
        if (MergeGpu.TryGramCross(a, b, n, d, out var gpuGram, out var gpuCross))
        {
            var gram = gpuGram!;
            var cross = gpuCross!;
            return SolveNormalEquations(gram, cross, d, ridgeRel);
        }

        // CPU normal equations in double precision, accumulated in parallel
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
                // localFinally runs concurrently on the workers that finish
                // together; unlocked, their += race and lose partials.
                lock (totalGram)
                {
                    for (int k = 0; k < d * d; k++)
                    {
                        totalGram[k] += local.Gram[k];
                        totalCross[k] += local.Cross[k];
                    }
                }
            });
        var g = totalGram;
        var x = totalCross;
        return SolveNormalEquations(g, x, d, ridgeRel);
    }

    /// <summary>Shared tail of the alignment fit: relative ridge, Cholesky
    /// factorisation and the two triangular solves, in fp64. The GPU path
    /// and the CPU path both land here, so the fitted map is the same
    /// authority regardless of where the Gram/Cross products were formed.</summary>
    private static float[] SolveNormalEquations(double[] gram, double[] cross, int d, double ridgeRel)
    {
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

        // X is Mᵀ; every consumer (ApplyMap, ResidualL2, the GPU map GEMM)
        // computes M·b with M row-major, so transpose on the way out.
        var result = new float[d * d];
        for (int r = 0; r < d; r++)
        {
            for (int c = 0; c < d; c++)
            {
                result[r * d + c] = (float)x[c * d + r];
            }
        }
        return result;
    }

    /// <summary>Σ‖a_i − M·b_i‖² over the anchor pairs — the alignment
    /// residual reported in the manifest (rows are independent, so rows
    /// accumulate in parallel).</summary>
    public static double ResidualL2(float[] a, float[] b, int n, int d, float[] m)
    {
        // CUDA path: aligned = b @ Mᵀ in one blocked GEMM, then the
        // Σ‖a − aligned‖² reduction on the host.
        if (MergeGpu.TryResidualL2(a, b, n, d, m, out var gpuResidual))
        {
            return gpuResidual;
        }

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

    /// <summary>
    /// Minimum-norm ridge update: the ΔW ∈ R^{dOut × dIn} minimising
    /// Σ‖t_i − ΔW·k_i‖² + λ‖ΔW‖²_F, solved in the SAMPLE space —
    /// ΔWᵀ = Kᵀ(KKᵀ + λI)⁻¹T — so the system is n×n rather than dIn×dIn.
    /// That is the cheap side when the keys are FFN activations (dIn in the
    /// thousands) and the samples a few hundred aligned positions, and the
    /// solution is the smallest weight change that hits the targets: it
    /// lies in the span of the keys, so inputs orthogonal to every key are
    /// untouched. Returns the update and, per sample, how much of the
    /// target it closes.
    /// </summary>
    /// <param name="t">Targets, row-major n×dOut.</param>
    /// <param name="k">Keys, row-major n×dIn.</param>
    /// <returns>ΔW row-major dOut×dIn, and the fraction of ‖T‖² the fit
    /// removes (1 = exact, 0 = nothing).</returns>
    public static (float[] Delta, double Explained, bool Gpu) FitMinimumNormUpdate(
        float[] t, float[] k, int n, int dOut, int dIn, double ridgeRel = 1e-2)
    {
        if (n <= 0)
        {
            throw new MergeException("no samples to fit the weight update");
        }
        if (t.Length != n * dOut || k.Length != n * dIn)
        {
            throw new MergeException($"update samples do not match (n={n}, in={dIn}, out={dOut})");
        }

        // CUDA path: the three products (KKᵀ, XᵀK, G·X) are fp32 cuBLAS
        // GEMMs and the solve is cuSOLVER's fp64 Cholesky. Each step falls
        // back to the managed code on its own, so a device failure part way
        // through still yields the update.
        bool gpu = CudaShim.Enabled;
        bool usedGpu = false;
        var options = new ParallelOptions { MaxDegreeOfParallelism = ComputeBudget.Cores };

        // G = KKᵀ (n×n).
        var gram = new double[n * n];
        if (gpu && CudaShim.TryGemmTransposedBF32WithMap(k, k, n, dIn, n, out var gram32))
        {
            // Symmetrise: the fp32 GEMM need not round (i,j) and (j,i) alike,
            // and Cholesky reads only one triangle.
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double v = 0.5 * ((double)gram32![i * n + j] + gram32[j * n + i]);
                    gram[i * n + j] = v;
                    gram[j * n + i] = v;
                }
            }
            usedGpu = true;
        }
        else
        {
            Parallel.For(0, n, options, i =>
            {
                var ki = k.AsSpan(i * dIn, dIn);
                for (int j = 0; j <= i; j++)
                {
                    double dot = TensorOps.Dot(ki, k.AsSpan(j * dIn, dIn));
                    gram[i * n + j] = dot;
                    gram[j * n + i] = dot;
                }
            });
        }

        double trace = 0;
        for (int i = 0; i < n; i++)
        {
            trace += gram[i * n + i];
        }
        double ridge = ridgeRel * (trace / n) + 1e-8;
        var system = (double[])gram.Clone();
        for (int i = 0; i < n; i++)
        {
            system[i * n + i] += ridge;
        }

        // X = (G + λI)⁻¹ T, kept column-major (n rows, dOut columns): column
        // o is x[o·n .. o·n + n). The system is symmetric, so its row-major
        // array is also its column-major one.
        var x = new double[n * dOut];
        for (int i = 0; i < n; i++)
        {
            for (int o = 0; o < dOut; o++)
            {
                x[o * n + i] = t[i * dOut + o];
            }
        }
        bool solved = false;
        if (gpu)
        {
            var factor = (double[])system.Clone();
            var rhs = (double[])x.Clone();
            try
            {
                if (CudaShim.amql_cuda_cholesky_solve_f64(factor, n, rhs, dOut, 0) == 0)
                {
                    x = rhs;
                    solved = usedGpu = true;
                }
            }
            catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException)
            {
            }
        }
        if (!solved)
        {
            var rowMajor = new double[n * dOut];
            for (int i = 0; i < n; i++)
            {
                for (int o = 0; o < dOut; o++)
                {
                    rowMajor[i * dOut + o] = x[o * n + i];
                }
            }
            var chol = Cholesky(system, n);
            var solution = new double[n * dOut];
            SolveLower(chol, solution, rowMajor, n, dOut);
            SolveUpperTranspose(chol, solution, n, dOut);
            for (int i = 0; i < n; i++)
            {
                for (int o = 0; o < dOut; o++)
                {
                    x[o * n + i] = solution[i * dOut + o];
                }
            }
        }

        // Xᵀ as fp32 dOut×n row-major is exactly the column-major X.
        var xt = new float[n * dOut];
        for (int i = 0; i < xt.Length; i++)
        {
            xt[i] = (float)x[i];
        }

        // ΔW[o, j] = Σ_i X[i, o] · K[i, j]  =  Xᵀ · (Kᵀ)ᵀ.
        float[] delta;
        var kt = gpu ? Transpose(k, n, dIn) : null;
        if (gpu && CudaShim.TryGemmTransposedBF32WithMap(xt, kt!, dOut, n, dIn, out var gpuDelta))
        {
            delta = gpuDelta!;
            usedGpu = true;
        }
        else
        {
            delta = new float[dOut * dIn];
            Parallel.For(0, dOut, options, o =>
            {
                var acc = new double[dIn];
                for (int i = 0; i < n; i++)
                {
                    double xo = x[o * n + i];
                    if (xo == 0)
                    {
                        continue;
                    }
                    int kRow = i * dIn;
                    for (int j = 0; j < dIn; j++)
                    {
                        acc[j] += xo * k[kRow + j];
                    }
                }
                for (int j = 0; j < dIn; j++)
                {
                    delta[o * dIn + j] = (float)acc[j];
                }
            });
        }

        // The fitted targets are K·ΔWᵀ = G·X; report how much of ‖T‖² they
        // remove.
        double[] fitted;
        float[]? gram32b = null;
        if (gpu)
        {
            gram32b = new float[n * n];
            for (int i = 0; i < gram32b.Length; i++)
            {
                gram32b[i] = (float)gram[i];
            }
        }
        if (gpu && CudaShim.TryGemmTransposedBF32WithMap(gram32b!, xt, n, n, dOut, out var gpuFitted))
        {
            fitted = Array.ConvertAll(gpuFitted!, v => (double)v);
            usedGpu = true;
        }
        else
        {
            fitted = new double[n * dOut];
            Parallel.For(0, n, options, i =>
            {
                for (int j = 0; j < n; j++)
                {
                    double g = gram[i * n + j];
                    for (int o = 0; o < dOut; o++)
                    {
                        fitted[i * dOut + o] += g * x[o * n + j];
                    }
                }
            });
        }
        double total = 0, left = 0;
        for (int i = 0; i < t.Length; i++)
        {
            double diff = t[i] - fitted[i];
            total += (double)t[i] * t[i];
            left += diff * diff;
        }
        return (delta, total > 0 ? 1.0 - left / total : 1.0, usedGpu);
    }

    private static float[] Transpose(float[] a, int rows, int cols)
    {
        var result = new float[a.Length];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                result[c * rows + r] = a[r * cols + c];
            }
        }
        return result;
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