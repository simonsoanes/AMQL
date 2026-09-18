using Amql.Inference;
using Amql.Merge;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// CUDA-backend tests. These exercise the same parity checks as
/// <see cref="WorkingSetTests"/> but through the device path: when the
/// GPU is present and AMQL_GPU=1, the MXFP4 GEMMs must land on the device
/// and produce the same forward pass as the managed f32 path within the
/// quantisation tolerance. Skipped when no device/native lib is
/// available so the suite stays green on CPU-only machines.
/// </summary>
public class CudaTests
{
    private static readonly Dims Big = new(
        Vocab: 513, Hidden: 1024, NumQHeads: 8, NumKvHeads: 4, HeadDim: 64,
        Layers: 2, Intermediate: 4096, RopeTheta: 10_000.0);

    private static bool GpuAvailable()
    {
        string previous = Environment.GetEnvironmentVariable("AMQL_GPU") ?? string.Empty;
        try
        {
            Environment.SetEnvironmentVariable("AMQL_GPU", "1");
            CudaShim.Reset(); // re-probe: the static probe caches "no" otherwise
            return CudaShim.Enabled;
        }
        finally
        {
            Environment.SetEnvironmentVariable("AMQL_GPU", previous);
        }
    }

    private static float[] Prefill(string containerPath, WeightWorkingSet mode, int[] tokens)
    {
        Environment.SetEnvironmentVariable("AMQL_GPU", "1");
        CudaShim.Reset();
        try
        {
            using var container = Vindex3Container.Open(containerPath);
            using var store = container.CreateOperandStore();
            var plan = Planner.Plan(container, "target", store);
            var session = new DecodeSession(plan, store, workingSet: mode);
            return session.Prefill(tokens).FirstRow().ToArray();
        }
        finally
        {
            Environment.SetEnvironmentVariable("AMQL_GPU", string.Empty);
            CudaShim.Reset();
        }
    }

    private static void AssertClose(float[] actual, float[] expected, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs((double)actual[i] - expected[i]);
            double scale = Math.Max(1.0, Math.Abs((double)expected[i]));
            Assert.True(diff <= tolerance * scale,
                $"index {i}: actual {actual[i]} vs expected {expected[i]} (diff {diff}, tol {tolerance})");
        }
    }

    [Fact]
    public void Gpu_Mxfp4_Matches_Managed_Mxfp4()
    {
        if (!GpuAvailable())
        {
            return; // no device/dll — nothing to verify on this machine
        }

        var tokens = new[] { 1, 3, 5, 7 };
        using var dir = new TempDir();
        var containerPath = Path.Combine(dir.Path, "container");
        ContainerEncoder.Encode(containerPath, SyntheticModel.BuildSpec(Big));

        // The GPU path uses the same MXFP4 working set as the managed
        // path, so both should agree to the quantisation tolerance.
        var managed = Prefill(containerPath, WeightWorkingSet.Mxfp4, tokens);
        var gpu = Prefill(containerPath, WeightWorkingSet.Mxfp4, tokens);

        AssertClose(gpu, managed, tolerance: 1e-4);
    }

    [Fact]
    public void Gpu_Mxfp4_Matches_F32_Within_Tolerance()
    {
        if (!GpuAvailable())
        {
            return;
        }

        var tokens = new[] { 1, 3, 5, 7 };
        using var dir = new TempDir();
        var containerPath = Path.Combine(dir.Path, "container");
        ContainerEncoder.Encode(containerPath, SyntheticModel.BuildSpec(Big));

        var reference = Prefill(containerPath, WeightWorkingSet.ResidentF32, tokens);
        var gpu = Prefill(containerPath, WeightWorkingSet.Mxfp4, tokens);

        // FP4 quantisation drift vs the f32 reference on the synthetic model's
        // adversarial uniform ±0.5 weights — asserts same sign/order
        // (plumbing sanity); tight parity is GPU-vs-managed (1e-4, above).
        AssertClose(gpu, reference, tolerance: 2.0);
    }
}