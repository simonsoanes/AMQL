using System.Runtime.InteropServices;

namespace Amql.Inference;

/// <summary>
/// P/Invoke surface of <c>amql_cuda.dll</c> — the optional CUDA execution
/// backend. Loaded lazily on first probe; every call is graceful: when the
/// native library is absent, the device is absent, or the process budget
/// says no, <see cref="Enabled"/> is false and the managed kernels run
/// unchanged. Enable via <c>AMQL_GPU</c> (<c>0</c>/<c>off</c> disables,
/// <c>1</c>/<c>on</c> requires and enables, anything else probes the
/// device). The GEMM path additionally requires the MXFP4 weight working
/// set (<c>AMQL_WEIGHTS=mxfp4</c>) so the resident packs are the device's
/// input — see <see cref="GpuDispatch"/>. Only GEMMs above the small-op
/// cutoff ever reach the device; everything else stays managed.
/// </summary>
public static class CudaShim
{
    private static readonly object Lock = new();
    private static bool _probed;
    private static bool _enabled;

    /// <summary>Whether the CUDA backend is loaded and primed.</summary>
    public static bool Enabled
    {
        get
        {
            lock (Lock)
            {
                if (!_probed)
                {
                    _probed = true;
                    _enabled = Probe();
                }
                return _enabled;
            }
        }
    }

    private static bool Probe()
    {
        string raw = Environment.GetEnvironmentVariable("AMQL_GPU")?.Trim().ToLowerInvariant() ?? "auto";
        if (raw is "0" or "off" or "no" or "false")
        {
            return false;
        }
        try
        {
            if (amql_cuda_available() != 1)
            {
                return false;
            }
            return amql_cuda_init() == 0;
        }
        catch (DllNotFoundException)
        {
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    /// <summary>Forces a fresh probe on the next <see cref="Enabled"/>
    /// read and drops any uploaded device weights — the test seam and the
    /// process-exit safety net.</summary>
    public static void Reset()
    {
        lock (Lock)
        {
            _probed = false;
            _enabled = false;
            foreach (var device in _deviceWeights.Values)
            {
                amql_cuda_free(device);
            }
            _deviceWeights.Clear();
            _deviceFailed = false;
        }
    }

    /// <summary>
    /// Device-resident FP16 copy of one weight matrix, cached by its
    /// container operand so a session's repeated projection calls reuse
    /// the upload. Entry <c>(objectId, tensor)</c> holds the device
    /// pointer (0 = not uploaded). Uploads are idempotent and never fail
    /// silently: on any native error the device is marked unusable and
    /// the managed path takes over for the rest of the process.
    /// </summary>
    private static Dictionary<(string, string), IntPtr> _deviceWeights = new();
    private static bool _deviceFailed;

    public static bool TryGetDeviceWeight(string objectId, string tensor, out IntPtr device, out long elements)
    {
        device = IntPtr.Zero;
        elements = 0;
        if (!Enabled || _deviceFailed)
        {
            return false;
        }
        lock (Lock)
        {
            if (_deviceWeights.TryGetValue((objectId, tensor), out device))
            {
                return device != IntPtr.Zero;
            }
        }
        return false;
    }

    public static IntPtr UploadWeightF16(string objectId, string tensor, byte[] packed, byte[] scales, int rows, int cols)
    {
        if (!Enabled || _deviceFailed)
        {
            return IntPtr.Zero;
        }
        lock (Lock)
        {
            if (_deviceWeights.TryGetValue((objectId, tensor), out var existing))
            {
                return existing;
            }
            if (amql_cuda_malloc(out var dev, checked((nuint)((long)rows * cols * 2))) != 0)
            {
                _deviceFailed = true;
                return IntPtr.Zero;
            }
            if (amql_cuda_dequant_to_f16(packed, scales, dev, rows, cols, 0) != 0)
            {
                amql_cuda_free(dev);
                _deviceFailed = true;
                return IntPtr.Zero;
            }
            _deviceWeights[(objectId, tensor)] = dev;
            return dev;
        }
    }

    /// <summary>Calls the transposed-B GEMM (C[m,n] = A[m,k]·W[n,k]ᵀ with
    /// A host f32, W device FP16) and copies the result back. Returns true
    /// when the kernel ran; false leaves <paramref name="result"/> set to
    /// null and the caller falls back to the managed path.</summary>
    public static bool TryGemmTransposedB(float[] a, int m, int k, IntPtr wF16, int n, out float[]? result)
    {
        result = null;
        if (!Enabled || _deviceFailed)
        {
            return false;
        }
        var c = new float[m * n];
        if (amql_cuda_gemm_transposed_b(a, wF16, c, m, k, n, 0) != 0)
        {
            _deviceFailed = true;
            return false;
        }
        result = c;
        return true;
    }

    // ── native surface ────────────────────────────────────────────────────

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_available();

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_init();

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_malloc(out IntPtr ptr, nuint bytes);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_free(IntPtr ptr);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_dequant_to_f16(
        byte[] packed, byte[] scales, IntPtr outF16, int rows, int cols, int streamOrdinal);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_gemm_transposed_b(
        float[] a, IntPtr wF16, float[] c, int m, int k, int n, int streamOrdinal);
}