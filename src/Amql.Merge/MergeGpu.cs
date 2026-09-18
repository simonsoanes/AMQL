using System.Runtime.InteropServices;
using Amql.Inference;

namespace Amql.Merge;

/// <summary>
/// GPU acceleration for the merge's normal-equation and map-apply hot
/// loops. The merge is the import's long pole: the alignment fit
/// (BᵀB/BᵀA Gram over 250k anchors × 5120²) and the per-token map-apply
/// (M·other for every shared token) are both O(n·d²) — minutes of
/// memory-bandwidth-bound CPU work on a 27B-class merge, performed as
/// blocked cuBLAS fp32 GEMMs on the 5090 in seconds each.
///
/// The managed fp64 normal-equation solve stays the authority: the GPU
/// computes each anchor block's Gram/Cross as fp32 GEMMs and the host
/// fp64-accumulates the block partials, so the Cholesky and the fitted
/// map match the CPU path block-for-block (fp32 GEMM rounding only, no
/// fp64 reduction change). Opt-in via <c>AMQL_MERGE_GPU=1</c> — the CPU
/// path remains the byte-exact default and the test oracle.
/// </summary>
public static class MergeGpu
{
    /// <summary>Whether the merge should route its hot loops to CUDA:
    /// explicitly requested by <c>AMQL_MERGE_GPU</c> (1/on/true) AND the
    /// device/native shim is actually usable.</summary>
    public static bool Enabled
    {
        get
        {
            string raw = Environment.GetEnvironmentVariable("AMQL_MERGE_GPU") ?? string.Empty;
            bool requested = raw.Trim().ToLowerInvariant() is "1" or "on" or "true" or "yes";
            return requested && CudaShim.Enabled;
        }
    }

    /// <summary>Blocked Gram/Cross on the device: gram = BᵀB, cross = BᵀA
    /// over fp32 rows, fp64-accumulated per block on the host. Returns
    /// false to fall back to the managed normal equations.</summary>
    public static bool TryGramCross(
        float[] a, float[] b, int n, int d,
        out double[]? gram, out double[]? cross)
    {
        gram = null;
        cross = null;
        if (!Enabled)
        {
            return false;
        }
        return CudaShim.TryGramCross(a, b, n, d, out gram, out cross);
    }

    /// <summary>Blocked residual: aligned = b @ Mᵀ on the device, then the
    /// Σ‖a − aligned‖² reduction runs host-parallel over the downloaded
    /// rows (n×d element ops — seconds at 27B scale, not the O(n·d²)
    /// matvec it replaces). Returns false to fall back.</summary>
    public static bool TryResidualL2(
        float[] a, float[] b, int n, int d, float[] m,
        out double residual)
    {
        residual = 0;
        if (!Enabled)
        {
            return false;
        }
        if (!CudaShim.TryGemmTransposedBF32WithMap(b, m, n, d, d, out var aligned, blockRows: 8192))
        {
            return false;
        }
        // Host-parallel diff reduction over the downloaded aligned rows.
        var sync = new object();
        double total = 0;
        Parallel.For(
            0,
            n,
            new ParallelOptions { MaxDegreeOfParallelism = ComputeBudget.Cores },
            () => 0.0,
            (i, _, local) =>
            {
                int row = i * d;
                for (int c = 0; c < d; c++)
                {
                    double diff = a[row + c] - aligned![row + c];
                    local += diff * diff;
                }
                return local;
            },
            local => { lock (sync) { total += local; } });
        residual = total;
        return true;
    }

    /// <summary>Applies the alignment map to every other-model row in one
    /// blocked GEMM: aligned[o] = M·other[o] over all rows at once. The
    /// caller's per-token loop then copies the aligned row instead of
    /// running the O(d²) matvec per token. Returns false to fall back.</summary>
    public static bool TryMapApply(
        float[] otherRows, int otherCount, int d, float[] m,
        out float[]? aligned)
    {
        aligned = null;
        if (!Enabled)
        {
            return false;
        }
        return CudaShim.TryGemmTransposedBF32WithMap(
            otherRows, m, otherCount, d, d, out aligned, blockRows: 8192);
    }
}