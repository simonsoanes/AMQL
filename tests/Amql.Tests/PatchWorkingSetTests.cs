using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// <c>--patch</c> under the compact working sets: the patch must land once,
/// at full precision, on the host view and on the device copy — and a
/// patched runtime must neither reuse nor leave behind an unpatched
/// runtime's device weights (the device cache is process-wide).
/// </summary>
[Collection("gpu")]
public class PatchWorkingSetTests
{
    private static readonly Dims Big = new(
        Vocab: 513, Hidden: 1024, NumQHeads: 8, NumKvHeads: 4, HeadDim: 64,
        Layers: 2, Intermediate: 4096, RopeTheta: 10_000.0);

    private static readonly int[] Tokens = { 1, 3, 5, 7 };

    private static (WeightPatch Patch, OperandRef Down) MakePatch(Vindex3Container container)
    {
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var down = plan.Layers[1].Ffn!.Dense!.Down;
        var rng = new Random(11);
        var delta = Enumerable.Range(0, Big.Hidden * Big.Intermediate)
            .Select(_ => (float)(rng.NextDouble() - 0.5) * 0.5f).ToArray();
        var patch = WeightPatch.FromEntries(new[]
        {
            new WeightPatchEntry(down.ObjectId, down.TensorName, new long[] { Big.Hidden, Big.Intermediate }, delta),
        });
        return (patch, down);
    }

    private static float[] Prefill(Vindex3Container container, WeightWorkingSet mode, WeightPatch? patch,
        bool expectDevice = false)
    {
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var session = new DecodeSession(plan, store, patch, mode);
        var logits = session.Prefill(Tokens).FirstRow().ToArray();
        if (expectDevice)
            Assert.True(session.Runtime.Weights.AnyDeviceWeight, "the forward should have used device weights");
        return logits;
    }

    /// <summary>‖(a − b) − (c − d)‖ / ‖c − d‖: how far the change from b to
    /// a strays from the change from d to c.</summary>
    private static double ChangeError(float[] a, float[] b, float[] c, float[] d)
    {
        double err = 0, norm = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double reference = (double)c[i] - d[i];
            err += Math.Pow((double)a[i] - b[i] - reference, 2);
            norm += reference * reference;
        }
        return Math.Sqrt(err / norm);
    }

    [Fact]
    public void Mxfp4_Applies_The_Patch_Once_On_Top_Of_The_Base_Pack()
    {
        using var dir = new TempDir();
        var containerPath = Path.Combine(dir.Path, "container");
        ContainerEncoder.Encode(containerPath, SyntheticModel.BuildSpec(Big));
        using var container = Vindex3Container.Open(containerPath);
        var (patch, down) = MakePatch(container);
        try
        {
            CudaShim.Reset();
            CudaShim.ForceDisable();
            using var store = container.CreateOperandStore();
            var bare = new WeightLoader(store, null, WeightWorkingSet.Mxfp4).Matrix(down, Big.Hidden, Big.Intermediate).Data;
            var patched = new WeightLoader(store, patch, WeightWorkingSet.Mxfp4).Matrix(down, Big.Hidden, Big.Intermediate).Data;

            Assert.True(patch.TryGet(down.ObjectId, down.TensorName, out var entry));
            for (int i = 0; i < bare.Length; i++)
                Assert.True(MathF.Abs(patched[i] - bare[i] - entry.Delta[i]) <= 1e-6f,
                    $"element {i}: patched {patched[i]} − base {bare[i]} ≠ delta {entry.Delta[i]}");
        }
        finally
        {
            CudaShim.Reset();
        }
    }

    [Theory]
    [InlineData(WeightWorkingSet.OnDemandBf16)]
    [InlineData(WeightWorkingSet.Mxfp4)]
    public void Gpu_Patched_Runtime_Matches_Managed_And_Does_Not_Share_Device_Weights(WeightWorkingSet mode)
    {
        using var dir = new TempDir();
        var containerPath = Path.Combine(dir.Path, "container");
        ContainerEncoder.Encode(containerPath, SyntheticModel.BuildSpec(Big));
        using var container = Vindex3Container.Open(containerPath);
        var (patch, _) = MakePatch(container);
        string previous = Environment.GetEnvironmentVariable("AMQL_GPU") ?? string.Empty;
        try
        {
            CudaShim.Reset();
            CudaShim.ForceDisable();
            var managedBare = Prefill(container, mode, null);
            var managedPatched = Prefill(container, mode, patch);

            Environment.SetEnvironmentVariable("AMQL_GPU", "1");
            CudaShim.Reset();
            if (!CudaShim.Enabled)
            {
                return; // no device here — the managed path above is all there is
            }
            // Unpatched first, so its device copies are already cached when
            // the patched runtime loads the same tensors.
            var bareBefore = Prefill(container, mode, null, expectDevice: true);
            var gpuPatched = Prefill(container, mode, patch, expectDevice: true);
            var bareAfter = Prefill(container, mode, null, expectDevice: true);

            // FP16 device weights round differently from the managed f32
            // view, so compare what the patch changes rather than the logits:
            // a patch that never reached the device, or landed twice, is far off.
            double error = ChangeError(gpuPatched, bareBefore, managedPatched, managedBare);
            Assert.True(error < 0.2, $"GPU patch effect strays {error:P1} from the managed one");
            Assert.Equal(bareBefore, bareAfter);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AMQL_GPU", previous);
            CudaShim.Reset();
        }
    }
}
