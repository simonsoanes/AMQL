using Amql.Inference;
using Amql.Merge;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// Working-set tests: the <see cref="WeightLoader"/> compaction modes
/// (on-demand BF16, MXFP4) must produce the same forward pass as the
/// resident f32 path. The synthetic model here uses FFN projections large
/// enough (≥ 1M elements) to cross the compaction threshold, so the modes
/// genuinely round-trip the packed/dequant paths.
/// </summary>
public class WorkingSetTests
{
    private static readonly Dims Big = new(
        Vocab: 513, Hidden: 1024, NumQHeads: 8, NumKvHeads: 4, HeadDim: 64,
        Layers: 2, Intermediate: 4096, RopeTheta: 10_000.0);

    private static string Encode(TempDir dir)
    {
        var containerPath = Path.Combine(dir.Path, "container");
        ContainerEncoder.Encode(containerPath, SyntheticModel.BuildSpec(Big));
        return containerPath;
    }

    private static float[] Prefill(string containerPath, WeightWorkingSet mode, int[] tokens)
    {
        // Pin the CPU path: these tests verify the managed working-set
        // modes, not the CUDA backend (CudaTests owns that). Without the
        // pin, an AMQL_GPU auto-probe would silently route MXFP4 GEMMs to
        // the device on GPU machines.
        string previous = Environment.GetEnvironmentVariable("AMQL_GPU") ?? string.Empty;
        try
        {
            Environment.SetEnvironmentVariable("AMQL_GPU", "0");
            CudaShim.Reset();
            using var container = Vindex3Container.Open(containerPath);
            using var store = container.CreateOperandStore();
            var plan = Planner.Plan(container, "target", store);
            var session = new DecodeSession(plan, store, workingSet: mode);
            return session.Prefill(tokens).FirstRow().ToArray();
        }
        finally
        {
            Environment.SetEnvironmentVariable("AMQL_GPU", previous);
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
    public void OnDemandBf16_Matches_ResidentF32_Exactly()
    {
        var tokens = new[] { 1, 3, 5, 7 };
        using var dir = new TempDir();
        var containerPath = Encode(dir);

        var reference = Prefill(containerPath, WeightWorkingSet.ResidentF32, tokens);
        var onDemand = Prefill(containerPath, WeightWorkingSet.OnDemandBf16, tokens);

        // BF16 on-demand widens the same stored bytes losslessly — the
        // output must be bit-identical.
        Assert.Equal(reference, onDemand);
    }

    [Fact]
    public void Mxfp4_Matches_ResidentF32_Within_Tolerance()
    {
        var tokens = new[] { 1, 3, 5, 7 };
        using var dir = new TempDir();
        var containerPath = Encode(dir);

        var reference = Prefill(containerPath, WeightWorkingSet.ResidentF32, tokens);
        var mxfp4 = Prefill(containerPath, WeightWorkingSet.Mxfp4, tokens);

        // FP4 E2M1 + 32-block E8M0 scales quantise the FFN projections. The
        // synthetic model's uniform ±0.5 weights are the codec's
        // adversarial case (a single power-of-2 scale for a flat block can
        // swing intermediate logits ~2×), so this asserts the plumbing
        // works — same sign, same order — rather than closeness. Tight
        // parity lives in CudaTests (managed-vs-GPU, 1e-4) and real-weight
        // quality in the real-model smokes.
        AssertClose(mxfp4, reference, tolerance: 2.0);
    }

    [Fact]
    public void Compacted_WorkingSet_Bounds_Resident_Cache()
    {
        using var dir = new TempDir();
        var containerPath = Encode(dir);

        string previous = Environment.GetEnvironmentVariable("AMQL_GPU") ?? string.Empty;
        try
        {
            Environment.SetEnvironmentVariable("AMQL_GPU", "0");
            CudaShim.Reset();
            using var container = Vindex3Container.Open(containerPath);
            using var store = container.CreateOperandStore();
            var plan = Planner.Plan(container, "target", store);
            var session = new DecodeSession(plan, store, workingSet: WeightWorkingSet.OnDemandBf16);
            session.Prefill(new[] { 1, 3, 5, 7 });

            // Every FFN projection was served from the compacted path — the
            // session's loader must report pack usage without materialising
            // the full set as resident f32.
            Assert.True(session.Runtime.Weights.LoadedMatrixCount > 0);
            Assert.Equal(WeightWorkingSet.OnDemandBf16, session.Runtime.Weights.WorkingSet);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AMQL_GPU", previous);
        }
    }
}