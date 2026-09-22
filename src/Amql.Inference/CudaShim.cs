using System.Runtime.InteropServices;

namespace Amql.Inference;

/// <summary>
/// P/Invoke surface of <c>amql_cuda.dll</c> — the optional CUDA execution
/// backend. Loaded lazily on first probe; every call is graceful: when the
/// native library is absent, the device is absent, or the process budget
/// says no, <see cref="Enabled"/> is false and the managed kernels run
/// unchanged.
///
/// By default the device is auto-detected: a CUDA-capable GPU with the
/// native DLL present enables itself automatically.  The <c>AMQL_GPU</c>
/// env var (<c>0</c>/<c>off</c> disables, <c>1</c>/<c>on</c> requires)
/// overrides the auto-probe.  The <c>--cpu</c> and <c>--gpu</c> CLI flags
/// take priority over the env var: <c>--cpu</c> forces the managed path,
/// <c>--gpu</c> requires the device and fails the command when it is
/// unavailable.
///
/// The GEMM path additionally requires the MXFP4 weight working
/// set (<c>AMQL_WEIGHTS=mxfp4</c>) so the resident packs are the device's
/// input — see <see cref="GpuDispatch"/>. Only GEMMs above the small-op
/// cutoff ever reach the device; everything else stays managed.
/// </summary>
public static class CudaShim
{
    private static readonly object Lock = new();
    private static bool _probed;
    private static bool _enabled;

    /// <summary>Forced mode: null = auto (default), true = GPU required,
    /// false = CPU forced.  Set once before the first probe — the CLI
    /// reads <c>--cpu</c>/<c>--gpu</c> and calls
    /// <see cref="ForceDisable"/> or <see cref="ForceEnable"/> before any
    /// command runs.</summary>
    private static bool? _forceMode;

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

    /// <summary>
    /// Forces the CPU path for this process — must be called before the
    /// first <see cref="Enabled"/> read (typically from the CLI after
    /// parsing <c>--cpu</c>).  Takes priority over <c>AMQL_GPU</c>.
    /// </summary>
    public static void ForceDisable()
    {
        lock (Lock)
        {
            if (_probed)
            {
                throw new InvalidOperationException(
                    "CudaShim has already probed — ForceDisable must be called before any CUDA access");
            }
            _forceMode = false;
        }
    }

    /// <summary>
    /// Requires the GPU for this process — must be called before the
    /// first <see cref="Enabled"/> read (typically from the CLI after
    /// parsing <c>--gpu</c>).  Takes priority over <c>AMQL_GPU</c>.
    /// When the device or the native DLL is absent the probe throws
    /// rather than silently falling back to the CPU path.
    /// </summary>
    public static void ForceEnable()
    {
        lock (Lock)
        {
            if (_probed)
            {
                throw new InvalidOperationException(
                    "CudaShim has already probed — ForceEnable must be called before any CUDA access");
            }
            _forceMode = true;
        }
    }

    private static bool Probe()
    {
        // --cpu / --gpu take priority over the env var.
        if (_forceMode == false)
        {
            return false;
        }

        string raw = Environment.GetEnvironmentVariable("AMQL_GPU")?.Trim().ToLowerInvariant() ?? "auto";
        if (raw is "0" or "off" or "no" or "false")
        {
            return false;
        }

        bool required = _forceMode == true;
        try
        {
            if (amql_cuda_available() != 1)
            {
                if (required)
                {
                    throw new InvalidOperationException(
                        "CUDA device not found — --gpu was requested but no compatible GPU is present. " +
                        "Install the NVIDIA driver and ensure the device is visible to nvidia-smi.");
                }
                return false;
            }
            int initCode = amql_cuda_init();
            if (initCode != 0)
            {
                if (required)
                {
                    throw new InvalidOperationException(
                        $"cuBLAS initialisation failed (code {initCode}) — --gpu was requested but the " +
                        "CUDA runtime could not start. Check that the driver matches the CUDA toolkit version.");
                }
                return false;
            }
            return true;
        }
        catch (DllNotFoundException) when (!required)
        {
            return false;
        }
        catch (EntryPointNotFoundException) when (!required)
        {
            return false;
        }
        // When --gpu is set, DllNotFoundException / EntryPointNotFoundException
        // propagate as typed errors (not swallowed).
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
            _forceMode = null;
            foreach (var device in _deviceWeights.Values)
            {
                amql_cuda_free(device);
            }
            _deviceWeights.Clear();
            _deviceFailed = false;
        }
    }

    /// <summary>True once any native call has failed in this process and
    /// the backend has latched into managed fallback. Diagnostics and the
    /// test seam only — the latch is the session's safety net.</summary>
    public static bool DeviceFailed => _deviceFailed;

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

    /// <summary>How many weight matrices currently hold a device FP16
    /// copy (diagnostics and test observation only).</summary>
    public static int DeviceWeightCount
    {
        get
        {
            lock (Lock)
            {
                return _deviceWeights.Count;
            }
        }
    }

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

    /// <summary>Allocates a device buffer (bytes) and returns its pointer;
    /// 0 on failure or when the backend is disabled.</summary>
    public static int AllocDevice(long bytes, out IntPtr ptr)
    {
        ptr = IntPtr.Zero;
        if (!Enabled || _deviceFailed || bytes <= 0)
        {
            return -1;
        }
        return amql_cuda_malloc(out ptr, checked((nuint)bytes));
    }

    /// <summary>Frees a device buffer from a previous AllocDevice.</summary>
    public static void FreeDevice(IntPtr ptr)
    {
        if (ptr != IntPtr.Zero)
        {
            amql_cuda_free(ptr);
        }
    }

    /// <summary>Copies bytes host → device; 0 on success.</summary>
    public static int CopyHostToDevice(IntPtr dst, byte[] src, long bytes)
    {
        if (!Enabled || _deviceFailed)
        {
            return -1;
        }
        return amql_cuda_host_to_device(dst, src, checked((nuint)bytes));
    }

    /// <summary>Copies an fp32 array host → device; 0 on success.</summary>
    public static int CopyHostToDevice(IntPtr dst, float[] src, long bytes)
    {
        if (!Enabled || _deviceFailed)
        {
            return -1;
        }
        return amql_cuda_host_to_device_f32(dst, src, checked((nuint)bytes));
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
        int code = amql_cuda_gemm_transposed_b(a, wF16, c, m, k, n, 0);
        LastNativeError = code;
        if (code != 0)
        {
            _deviceFailed = true;
            return false;
        }
        result = c;
        return true;
    }

    /// <summary>FP32 transposed-B GEMM streamed over A in row blocks —
    /// the merge path's map-apply (aligned = other @ Mᵀ). Returns true on
    /// success; the caller falls back to the managed ApplyMap otherwise.</summary>
    public static bool TryGemmTransposedBF32(
        float[] a, int m, int k, IntPtr wF32, int n, out float[]? result, int blockRows = 4096)
    {
        result = null;
        if (!Enabled || _deviceFailed)
        {
            return false;
        }
        var c = new float[m * n];
        if (amql_cuda_gemm_transposed_b_f32(a, wF32, c, m, k, n, blockRows, 0) != 0)
        {
            _deviceFailed = true;
            return false;
        }
        result = c;
        return true;
    }

    /// <summary>Map-apply with the map uploaded inside the native call, on
    /// the cublas stream — the merge path never straddles a sync copy with
    /// cublas work (see the native function's contract). Returns true on
    /// success; the caller falls back to the managed path otherwise.</summary>
    public static bool TryGemmTransposedBF32WithMap(
        float[] a, float[] w, int m, int k, int n, out float[]? result, int blockRows = 8192)
    {
        result = null;
        if (!Enabled || _deviceFailed)
        {
            return false;
        }
        var c = new float[m * n];
        if (amql_cuda_gemm_transposed_b_f32_u(a, w, c, m, k, n, blockRows, 0) != 0)
        {
            _deviceFailed = true;
            return false;
        }
        result = c;
        return true;
    }

    /// <summary>The last native error code (or 0) — diagnostics for the merge
    /// path where a GEMM failure must not crash the import.</summary>
    public static int LastNativeError { get; private set; }

    /// <summary>Blocked Gram/Cross for the merge alignment:
    /// gram = BᵀB and cross = BᵀA over n fp32 rows of width d, each block
    /// computed as a pair of cuBLAS GEMMs and fp64-accumulated on the
    /// host (matching the managed normal-equation authority). Returns true
    /// on success.</summary>
    public static bool TryGramCross(
        float[] a, float[] b, int n, int d,
        out double[]? gram, out double[]? cross, int blockRows = 8192)
    {
        gram = null;
        cross = null;
        LastNativeError = 0;
        if (!Enabled || _deviceFailed)
        {
            return false;
        }
        var g = new double[d * d];
        var x = new double[d * d];
        int code = amql_cuda_gram_cross(a, b, g, x, n, d, blockRows, 0);
        LastNativeError = code;
        if (code != 0)
        {
            _deviceFailed = true;
            return false;
        }
        gram = g;
        cross = x;
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
    private static extern int amql_cuda_host_to_device(IntPtr dst, byte[] src, nuint bytes);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_host_to_device_f32(IntPtr dst, float[] src, nuint bytes);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_dequant_to_f16(
        byte[] packed, byte[] scales, IntPtr outF16, int rows, int cols, int streamOrdinal);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_gemm_transposed_b(
        float[] a, IntPtr wF16, float[] c, int m, int k, int n, int streamOrdinal);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_gemm_transposed_b_f32(
        float[] a, IntPtr wF32, float[] c, int m, int k, int n, int blockRows, int streamOrdinal);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_gemm_transposed_b_f32_u(
        float[] a, float[] w, float[] c, int m, int k, int n, int blockRows, int streamOrdinal);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_gram_cross(
        float[] a, float[] b, double[] gram, double[] cross, int n, int d, int blockRows, int streamOrdinal);
}