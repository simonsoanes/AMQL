using System.Runtime.InteropServices;
using Amql.Safetensors;

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
                    if (_enabled)
                    {
                        _unifiedMemory = amql_cuda_unified_memory() == 1;
                    }
                }
                return _enabled;
            }
        }
    }

    private static bool _unifiedMemory;

    /// <summary>
    /// True when the device has unified memory (Grace-Hopper, integrated GPU)
    /// — the GPU accesses host memory directly through the coherent fabric and
    /// the native backend skips scratch-buffer staging copies.
    /// </summary>
    public static bool IsUnifiedMemory => _unifiedMemory;

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

    /// <summary>Allocates device memory and copies <paramref name="fp16"/> raw
    /// FP16 bytes to it. Returns the device pointer (IntPtr.Zero on failure).
    /// Tracked in the device-weight LRU and freed on shutdown.</summary>
    public static IntPtr UploadRawF16(string key, byte[] fp16, int rows, int cols)
    {
        if (!Enabled || _deviceFailed) return IntPtr.Zero;
        lock (Lock)
        {
            if (_deviceWeights.TryGetValue((key, ""), out var existing)) return existing;
            nuint bytes = checked((nuint)((long)rows * cols * 2));
            if (amql_cuda_malloc(out var dev, bytes) != 0) { _deviceFailed = true; return IntPtr.Zero; }
            if (amql_cuda_host_to_device(dev, fp16, bytes) != 0) { amql_cuda_free(dev); _deviceFailed = true; return IntPtr.Zero; }
            _deviceWeights[(key, "")] = dev;
            return dev;
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

    // ── NVFP4 weights ──────────────────────────────────────────────────────

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_nvfp4_upload(byte[] packed, byte[] scales, int rows, int cols, out IntPtr device);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_nvfp4_gemm(float[] a, IntPtr w, float[] c, int m, int k, int n, float global);

    /// <summary>Device bytes held by NVFP4 and MXFP8 uploads.</summary>
    public static long Nvfp4Bytes => Interlocked.Read(ref _nvfp4Bytes);
    private static long _nvfp4Bytes;

    /// <summary>The device bytes an NVFP4 upload of a [rows, cols] weight takes.</summary>
    public static long Nvfp4DeviceBytes(int rows, int cols) =>
        (((long)rows * cols / 2 + 255) & ~255L) +
        // Block scales in the tensor cores' tiles: rows padded to 128, blocks to 4.
        (long)((rows + 127) / 128) * 128 * ((cols / Nvfp4.BlockElements + 3) / 4) * 4;

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_nvfp4_tensor_cores(int m, int n, int k);

    [DllImport("amql_cuda")]
    private static extern void amql_cuda_set_fp4_prefill(int enabled);

    /// <summary>
    /// Whether NVFP4 products of more than 8 rows (prefill) run on the FP4
    /// tensor cores, which quantise the activations to NVFP4 too (W4A4, what
    /// NVFP4 checkpoints are trained for), or decode FP16 tiles of the weight
    /// against FP16 activations (W4A16, slower, closer to the dequantised
    /// model). On by default; <c>AMQL_FP4_PREFILL=0</c> or
    /// <c>--fp4-prefill off</c> turns it off.
    /// </summary>
    public static void SetFp4Prefill(bool enabled)
    {
        if (!Enabled)
        {
            return;
        }
        try
        {
            amql_cuda_set_fp4_prefill(enabled ? 1 : 0);
        }
        catch (EntryPointNotFoundException)
        {
        }
    }

    /// <summary>Whether NVFP4 products of this shape ran on the FP4 tensor cores
    /// (W4A4 through cuBLASLt) rather than decoded FP16 tiles.</summary>
    public static bool Nvfp4UsedTensorCores(int m, int n, int k)
    {
        try
        {
            return Enabled && amql_cuda_nvfp4_tensor_cores(m, n, k) == 1;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    /// <summary>
    /// Puts an NVFP4 weight on the device as its packs and block scales, the
    /// tensor scale staying with the caller. Zero when there is no device or
    /// the allocation fails; a failed allocation (the device is full) does not
    /// turn the GPU off, the weight simply stays on the host.
    /// </summary>
    public static IntPtr UploadNvfp4(ScaledMatrix weight)
    {
        if (!Enabled || _deviceFailed || !weight.IsNvfp4)
        {
            return IntPtr.Zero;
        }
        try
        {
            if (amql_cuda_nvfp4_upload(weight.Elements, weight.Nvfp4ScaleBytes(), weight.Rows, weight.Cols, out var device) != 0)
            {
                return IntPtr.Zero;
            }
            Interlocked.Add(ref _nvfp4Bytes, Nvfp4DeviceBytes(weight.Rows, weight.Cols));
            return device;
        }
        catch (Exception ex) when (ex is EntryPointNotFoundException or DllNotFoundException)
        {
            return IntPtr.Zero; // a native library from before NVFP4
        }
    }

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_mxfp8_upload(byte[] elements, byte[] scales, int rows, int cols, out IntPtr device);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_mxfp8_gemm(float[] a, IntPtr w, float[] c, int m, int k, int n);

    /// <summary>The device bytes an MXFP8 upload of a [rows, cols] weight takes.</summary>
    public static long Mxfp8DeviceBytes(int rows, int cols) =>
        (((long)rows * cols + 255) & ~255L) + (long)rows * (cols / Mxfp8.BlockElements);

    /// <summary>
    /// Puts an MXFP8 weight on the device as stored: its E4M3 elements and E8M0
    /// exponents. Zero when there is no device or the allocation fails (the
    /// weight then stays on the host; the GPU stays on).
    /// </summary>
    public static IntPtr UploadMxfp8(ScaledMatrix weight)
    {
        if (!Enabled || _deviceFailed || !weight.IsMxfp8)
        {
            return IntPtr.Zero;
        }
        try
        {
            if (amql_cuda_mxfp8_upload(weight.Elements, weight.Mxfp8ScaleBytes(), weight.Rows, weight.Cols, out var device) != 0)
            {
                return IntPtr.Zero;
            }
            Interlocked.Add(ref _nvfp4Bytes, Mxfp8DeviceBytes(weight.Rows, weight.Cols));
            return device;
        }
        catch (Exception ex) when (ex is EntryPointNotFoundException or DllNotFoundException)
        {
            return IntPtr.Zero;
        }
    }

    /// <summary>Frees an <see cref="UploadMxfp8"/> weight of the given shape.</summary>
    public static void FreeMxfp8(IntPtr device, int rows, int cols)
    {
        if (device == IntPtr.Zero || !Enabled)
        {
            return;
        }
        amql_cuda_free(device);
        Interlocked.Add(ref _nvfp4Bytes, -Mxfp8DeviceBytes(rows, cols));
    }

    /// <summary>C[m,n] = A[m,k]·W[n,k]ᵀ for an MXFP8 device weight; false on any failure.</summary>
    public static bool TryGemmMxfp8(float[] a, int m, int k, IntPtr w, int n, out float[]? result)
    {
        result = null;
        if (!Enabled || _deviceFailed || w == IntPtr.Zero)
        {
            return false;
        }
        var c = new float[m * n];
        int code = amql_cuda_mxfp8_gemm(a, w, c, m, k, n);
        LastNativeError = code;
        if (code != 0)
        {
            return false;
        }
        result = c;
        return true;
    }

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_ffn(float[] x, int m, int k, int width,
        int jobs, int[] rowStart, int[] rows, float[] rowWeights,
        int[] inter, IntPtr[] weights, int[] formats, float[] globals, float[] output);

    /// <summary>One gated SiLU FFN over chosen rows: <c>rows[i]</c> contributes
    /// <c>weights[i]</c> times its output to the sum.</summary>
    public sealed record FfnJob(int[] Rows, float[] Weights, Tensor2D Gate, Tensor2D Up, Tensor2D Down);

    /// <summary>Calls made through <see cref="TryFfn"/>, and the jobs they ran.</summary>
    public static long FfnCalls => Interlocked.Read(ref _ffnCalls);
    public static long FfnJobs => Interlocked.Read(ref _ffnJobs);
    private static long _ffnCalls, _ffnJobs;

    private static (IntPtr Weight, int Format, float Global) DeviceForm(Tensor2D w) =>
        w.DeviceNvfp4 != IntPtr.Zero ? (w.DeviceNvfp4, 1, w.DeviceNvfp4Global)
        : w.DeviceMxfp8 != IntPtr.Zero ? (w.DeviceMxfp8, 2, 1f)
        : w.DeviceWeightF16 != IntPtr.Zero ? (w.DeviceWeightF16, 3, 1f)
        : (IntPtr.Zero, 0, 0f);

    /// <summary>Whether gate, up and down are all on the device, so the FFN
    /// can run there without its activations leaving it.</summary>
    public static bool CanFuse(Tensor2D gate, Tensor2D up, Tensor2D down) =>
        FuseFfn && gate.OnGpu && up.OnGpu && down.OnGpu && gate.Rows == up.Rows && down.Cols == gate.Rows &&
        gate.Cols == up.Cols;

    /// <summary>Whether FFNs whose weights are all on the device run as one
    /// fused call, and products sharing an input share its upload (the
    /// default), or each product runs on its own; <c>AMQL_FUSED_FFN=0</c>
    /// turns both off.</summary>
    public static bool FuseFfn { get; set; } =
        Environment.GetEnvironmentVariable("AMQL_FUSED_FFN")?.Trim() is not ("0" or "false" or "off");

    [DllImport("amql_cuda")]
    private static extern void amql_cuda_set_grouped_ffn(int on);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_get_grouped_ffn();

    [DllImport("amql_cuda")]
    private static extern long amql_cuda_grouped_ffn_calls();

    /// <summary>Whether a routed layer whose experts are all NVFP4, with up to
    /// 32 rows each (decode, small batches), runs every expert's gate and up
    /// as one launch and every down as another (grouped GEMV), instead of six
    /// launches an expert. Bit for bit the same output. On by default;
    /// <c>AMQL_GROUPED_FFN=0</c> turns it off.</summary>
    public static bool GroupedFfn
    {
        get => Enabled && amql_cuda_get_grouped_ffn() == 1;
        set
        {
            if (Enabled)
            {
                amql_cuda_set_grouped_ffn(value ? 1 : 0);
            }
        }
    }

    /// <summary>Routed layers run as grouped launches so far.</summary>
    public static long GroupedFfnCalls => Enabled ? amql_cuda_grouped_ffn_calls() : 0;

    /// <summary>
    /// Runs FFN jobs over rows of <paramref name="x"/> ([m, k]) on the device:
    /// x goes up once, each job gathers its rows, multiplies gate and up, takes
    /// SiLU(gate) times up, multiplies down and adds its weighted rows into the
    /// sum, which comes down once ([m, width]). False on any failure (the
    /// caller then runs the products one at a time).
    /// </summary>
    public static bool TryFfn(float[] x, int m, int k, int width, IReadOnlyList<FfnJob> jobs, out float[]? sum)
    {
        sum = null;
        if (!Enabled || _deviceFailed || jobs.Count == 0)
        {
            return false;
        }
        var rowStart = new int[jobs.Count + 1];
        for (int j = 0; j < jobs.Count; j++)
        {
            rowStart[j + 1] = rowStart[j] + jobs[j].Rows.Length;
        }
        var rows = new int[rowStart[^1]];
        var rowWeights = new float[rowStart[^1]];
        var inter = new int[jobs.Count];
        var weights = new IntPtr[jobs.Count * 3];
        var formats = new int[jobs.Count * 3];
        var globals = new float[jobs.Count * 3];
        for (int j = 0; j < jobs.Count; j++)
        {
            var job = jobs[j];
            job.Rows.CopyTo(rows, rowStart[j]);
            job.Weights.CopyTo(rowWeights, rowStart[j]);
            inter[j] = job.Gate.Rows;
            int i = 0;
            foreach (var w in new[] { job.Gate, job.Up, job.Down })
            {
                var (pointer, format, global) = DeviceForm(w);
                if (pointer == IntPtr.Zero)
                {
                    return false; // evicted since it was fetched
                }
                (weights[3 * j + i], formats[3 * j + i], globals[3 * j + i]) = (pointer, format, global);
                i++;
            }
        }
        var output = new float[(long)m * width];
        try
        {
            int code = amql_cuda_ffn(x, m, k, width, jobs.Count, rowStart, rows, rowWeights, inter, weights, formats, globals, output);
            LastNativeError = code;
            if (code != 0)
            {
                return false;
            }
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
        Interlocked.Increment(ref _ffnCalls);
        Interlocked.Add(ref _ffnJobs, jobs.Count);
        sum = output;
        return true;
    }

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_multi(float[] a, int m, int k, int count,
        IntPtr[] weights, int[] formats, float[] globals, int[] ns, float[] output);

    /// <summary>Calls made through <see cref="TryMulti"/>.</summary>
    public static long MultiCalls => Interlocked.Read(ref _multiCalls);
    private static long _multiCalls;

    /// <summary>
    /// Products of one input with several device weights (attention's q/k/v,
    /// the linear attention's input projections): the input goes up once and
    /// every product comes down in one copy. False on any failure, or if a
    /// weight has no device copy, and the caller multiplies them one at a time.
    /// </summary>
    public static bool TryMulti(float[] a, int m, int k, IReadOnlyList<Tensor2D> weights, out Tensor2D[]? products)
    {
        products = null;
        if (!Enabled || _deviceFailed || !FuseFfn)
        {
            return false;
        }
        var pointers = new IntPtr[weights.Count];
        var formats = new int[weights.Count];
        var globals = new float[weights.Count];
        var ns = new int[weights.Count];
        long total = 0;
        for (int i = 0; i < weights.Count; i++)
        {
            (pointers[i], formats[i], globals[i]) = DeviceForm(weights[i]);
            if (pointers[i] == IntPtr.Zero)
            {
                return false;
            }
            ns[i] = weights[i].Rows;
            total += (long)m * ns[i];
        }
        var output = new float[total];
        try
        {
            int code = amql_cuda_multi(a, m, k, weights.Count, pointers, formats, globals, ns, output);
            LastNativeError = code;
            if (code != 0)
            {
                return false;
            }
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
        products = new Tensor2D[weights.Count];
        long offset = 0;
        for (int i = 0; i < weights.Count; i++)
        {
            int length = m * ns[i];
            products[i] = new Tensor2D(output.AsSpan((int)offset, length).ToArray(), m, ns[i]);
            offset += length;
        }
        Interlocked.Increment(ref _multiCalls);
        return true;
    }

    /// <summary>Frees an <see cref="UploadNvfp4"/> weight of the given shape.</summary>
    public static void FreeNvfp4(IntPtr device, int rows, int cols)
    {
        if (device == IntPtr.Zero || !Enabled)
        {
            return;
        }
        amql_cuda_free(device);
        Interlocked.Add(ref _nvfp4Bytes, -Nvfp4DeviceBytes(rows, cols));
    }

    /// <summary>C[m,n] = A[m,k]·W[n,k]ᵀ × <paramref name="global"/> for an NVFP4
    /// device weight. False (the caller falls back to the host) on any failure.</summary>
    public static bool TryGemmNvfp4(float[] a, int m, int k, IntPtr w, int n, float global, out float[]? result)
    {
        result = null;
        if (!Enabled || _deviceFailed || w == IntPtr.Zero)
        {
            return false;
        }
        var c = new float[m * n];
        int code = amql_cuda_nvfp4_gemm(a, w, c, m, k, n, global);
        LastNativeError = code;
        if (code != 0)
        {
            return false;
        }
        result = c;
        return true;
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

    /// <summary>
    /// Launches the FP16-weight GEMM asynchronously on the CUDA stream
    /// and returns immediately — the host output buffer <paramref name="c"/>
    /// is NOT valid until <see cref="Sync"/> is called.  The caller batches
    /// several <c>LaunchGemmAsync</c> calls (all using the same activation
    /// shape) and calls <c>Sync</c> once to wait for all results.
    /// Returns true on launch success; false latches <see cref="DeviceFailed"/>.
    /// </summary>
    public static bool LaunchGemmAsync(float[] a, IntPtr wF16, float[] c, int m, int k, int n)
    {
        if (!Enabled || _deviceFailed)
        {
            return false;
        }
        int code = amql_cuda_gemm_transposed_b_async(a, wF16, c, m, k, n, 0);
        LastNativeError = code;
        if (code != 0)
        {
            _deviceFailed = true;
            return false;
        }
        return true;
    }

    /// <summary>
    /// Uploads a host FP32 activation matrix to the device as FP16 and
    /// returns the device pointer.  The caller then launches several
    /// <see cref="LaunchGemmDeviceA"/> calls against the same activation,
    /// calls <see cref="Sync"/>, and frees the activation with
    /// <see cref="FreeActivation"/>.  Returns <see cref="IntPtr.Zero"/>
    /// on failure (device latched).
    /// </summary>
    public static IntPtr UploadActivationF16(float[] a, int m, int k)
    {
        if (!Enabled || _deviceFailed)
        {
            return IntPtr.Zero;
        }
        int code = amql_cuda_upload_activation_f16(a, out var ptr, m, k, 0);
        LastNativeError = code;
        if (code != 0)
        {
            _deviceFailed = true;
            return IntPtr.Zero;
        }
        return ptr;
    }

    /// <summary>
    /// Launches a GEMM against a device-resident FP16 activation (no host→device
    /// copy, no FP32→FP16 cast).  Async — caller must <see cref="Sync"/> before
    /// reading <paramref name="c"/>.
    /// </summary>
    public static bool LaunchGemmDeviceA(IntPtr aF16, IntPtr wF16, float[] c, int m, int k, int n)
    {
        if (!Enabled || _deviceFailed)
        {
            return false;
        }
        int code = amql_cuda_gemm_device_a(aF16, wF16, c, m, k, n, 0);
        LastNativeError = code;
        if (code != 0)
        {
            _deviceFailed = true;
            return false;
        }
        return true;
    }

    /// <summary>
    /// Synchronises the CUDA stream — blocks until all previously launched
    /// async GEMMs complete.  After this returns, every host output buffer
    /// passed to <see cref="LaunchGemmAsync"/> or
    /// <see cref="LaunchGemmDeviceA"/> since the last sync contains its result.
    /// </summary>
    public static bool Sync()
    {
        if (!Enabled || _deviceFailed)
        {
            return false;
        }
        int code = amql_cuda_sync(0);
        LastNativeError = code;
        if (code != 0)
        {
            _deviceFailed = true;
            return false;
        }
        return true;
    }

    /// <summary>Frees a device activation buffer allocated by
    /// <see cref="UploadActivationF16"/>.</summary>
    public static void FreeActivation(IntPtr ptr)
    {
        if (ptr != IntPtr.Zero)
        {
            amql_cuda_free(ptr);
        }
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
    private static extern int amql_cuda_unified_memory();

    [DllImport("amql_cuda")]
    private static extern int amqe_mem_info(out nuint freeBytes, out nuint totalBytes);

    /// <summary>The device's free and total memory, or false when there is no
    /// device (or the native library predates the query).</summary>
    public static bool TryMemoryInfo(out long freeBytes, out long totalBytes)
    {
        freeBytes = totalBytes = 0;
        if (!Enabled)
        {
            return false;
        }
        try
        {
            if (amqe_mem_info(out var free, out var total) != 0)
            {
                return false;
            }
            freeBytes = (long)free;
            totalBytes = (long)total;
            return true;
        }
        catch (Exception ex) when (ex is EntryPointNotFoundException or DllNotFoundException)
        {
            return false;
        }
    }

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
    private static extern int amql_cuda_gemm_transposed_b_async(
        float[] a, IntPtr wF16, float[] c, int m, int k, int n, int streamOrdinal);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_upload_activation_f16(
        float[] a, out IntPtr outF16, int m, int k, int streamOrdinal);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_gemm_device_a(
        IntPtr aF16, IntPtr wF16, float[] c, int m, int k, int n, int streamOrdinal);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_sync(int streamOrdinal);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_gemm_transposed_b_f32(
        float[] a, IntPtr wF32, float[] c, int m, int k, int n, int blockRows, int streamOrdinal);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_gemm_transposed_b_f32_u(
        float[] a, float[] w, float[] c, int m, int k, int n, int blockRows, int streamOrdinal);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_gram_cross(
        float[] a, float[] b, double[] gram, double[] cross, int n, int d, int blockRows, int streamOrdinal);

    // ── Elementwise ops (P2) ────────────────────────────────────────────────

    /// <summary>RMSNorm on the GPU: copies x and weight to device, runs the
    /// kernel there, copies x back. Returns true on success (x modified
    /// in-place). Falls back to managed on failure.</summary>
    public static bool TryRmsNorm(float[] x, float[] weight, float weightOffset, double eps, int rows, int cols)
    {
        if (!Enabled || _deviceFailed) return false;
        int code = amql_cuda_rms_norm_f32_host(x, weight, weightOffset, (float)eps, rows, cols, 0);
        LastNativeError = code;
        if (code != 0) { _deviceFailed = true; return false; }
        return true;
    }

    /// <summary>SiLU activation on the GPU, in-place. Returns true on success.</summary>
    public static bool TrySilu(float[] x, int rows, int cols)
    {
        if (!Enabled || _deviceFailed) return false;
        int code = amql_cuda_silu_f32_host(x, rows, cols, 0);
        LastNativeError = code;
        if (code != 0) { _deviceFailed = true; return false; }
        return true;
    }

    /// <summary>RoPE on the GPU, in-place. Returns true on success.</summary>
    public static bool TryRope(float[] x, float[] invFreq, int[] positions, int rows, int heads, int headDim, int pairCount)
    {
        if (!Enabled || _deviceFailed) return false;
        int code = amql_cuda_rope_f32_host(x, invFreq, positions, rows, heads, headDim, pairCount, 0);
        LastNativeError = code;
        if (code != 0) { _deviceFailed = true; return false; }
        return true;
    }

    /// <summary>Embedding gather on the GPU. Returns true on success.</summary>
    public static bool TryGatherRows(float[] table, int vocab, int[] ids, float[] result, int rows, int hidden)
    {
        if (!Enabled || _deviceFailed) return false;
        int code = amql_cuda_gather_rows_f32_host(table, vocab, ids, result, rows, hidden, 0);
        LastNativeError = code;
        if (code != 0) { _deviceFailed = true; return false; }
        return true;
    }

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_rms_norm_f32_host(
        float[] x, float[] w, float wOff, float eps, int rows, int cols, int streamOrdinal);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_silu_f32_host(
        float[] x, int rows, int cols, int streamOrdinal);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_rope_f32_host(
        float[] x, float[] invFreq, int[] positions, int rows, int heads, int headDim, int pairCount, int streamOrdinal);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_gather_rows_f32_host(
        float[] table, int vocab, int[] ids, float[] result, int rows, int hidden, int streamOrdinal);

    // ── Attention + KV cache (P3) ───────────────────────────────────────────

    /// <summary>Fused GQA attention on the GPU.  Returns true on success
    /// (output written).  Falls back to managed on failure.</summary>
    public static bool TryAttentionGqa(
        float[] q, int seqQ,
        float[] kCache, int seqKV,
        float[] vCache,
        float[] output,
        int numQHeads, int numKvHeads, int headDim,
        float scoreScale, float softCap, int window,
        float[]? sinks,
        int[] qPositions, int[] kvPositions)
    {
        if (!Enabled || _deviceFailed) return false;
        int code = amql_cuda_attention_gqa_f32_host(
            q, seqQ, kCache, seqKV, vCache, output,
            numQHeads, numKvHeads, headDim,
            scoreScale, softCap, window,
            // null marshals as a null pointer, which the kernel reads as "no
            // sinks". An empty array marshals as a valid pointer to nothing:
            // the native side then copies numQHeads floats of garbage and adds
            // them to every head's position-0 score.
            sinks,
            qPositions, kvPositions, 0);
        LastNativeError = code;
        if (code != 0) { _deviceFailed = true; return false; }
        return true;
    }

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_attention_gqa_f32_host(
        float[] q, int seqQ,
        float[] kCache, int seqKV,
        float[] vCache,
        float[] output,
        int numQHeads, int numKvHeads, int headDim,
        float scoreScale, float softCap, int window,
        float[]? sinks,
        int[] qPositions, int[] kvPositions,
        int streamOrdinal);

    // ── Device KV cache (P3) ────────────────────────────────────────────────

    /// <summary>Allocate a device-side KV cache buffer for one layer.
    /// Returns the device pointer (IntPtr), or IntPtr.Zero on failure.</summary>
    public static IntPtr AllocateKvCache(int maxSeq, int numKvHeads, int headDim)
    {
        if (!Enabled || _deviceFailed) return IntPtr.Zero;
        int code = amql_cuda_alloc_kv_cache(out IntPtr ptr, maxSeq, numKvHeads, headDim);
        LastNativeError = code;
        if (code != 0) { _deviceFailed = true; return IntPtr.Zero; }
        return ptr;
    }

    /// <summary>Free a device-side KV cache buffer.</summary>
    public static void FreeKvCache(IntPtr ptr)
    {
        if (ptr == IntPtr.Zero) return;
        amql_cuda_free_kv_cache(ptr);
    }

    /// <summary>Append new KV rows to a device-side cache.  Returns true
    /// on success (data on device, no host readback).</summary>
    public static bool TryKvAppend(float[] newRows, int seqQ, IntPtr cache, int maxSeq,
        int numKvHeads, int headDim, int writePos)
    {
        if (!Enabled || _deviceFailed) return false;
        int code = amql_cuda_kv_append_f32(newRows, seqQ, cache, maxSeq, numKvHeads, headDim, writePos, 0);
        LastNativeError = code;
        if (code != 0) { _deviceFailed = true; return false; }
        return true;
    }

    /// <summary>Read KV cache rows from device to host.  Returns true on success.</summary>
    public static bool TryKvRead(IntPtr cache, int startRow, int numRows,
        float[] host, int numKvHeads, int headDim)
    {
        if (!Enabled || _deviceFailed) return false;
        int code = amql_cuda_kv_read_f32(cache, startRow, numRows, host, numKvHeads, headDim, 0);
        LastNativeError = code;
        if (code != 0) { _deviceFailed = true; return false; }
        return true;
    }

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_alloc_kv_cache(out IntPtr ptr, int maxSeq, int numKvHeads, int headDim);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_free_kv_cache(IntPtr ptr);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_kv_append_f32(
        float[] newRows, int seqQ, IntPtr cache, int maxSeq,
        int numKvHeads, int headDim, int writePos, int streamOrdinal);

    [DllImport("amql_cuda")]
    private static extern int amql_cuda_kv_read_f32(
        IntPtr cache, int startRow, int numRows, float[] host,
        int numKvHeads, int headDim, int streamOrdinal);

    // ── Merge: cuSOLVER Cholesky + solve (P6) ──────────────────────────────

    /// <summary>cuSOLVER fp64 Cholesky factorisation + triangular solve.
    /// A[n*n] column-major, B[n*nrhs] column-major. A is overwritten with
    /// the Cholesky factor; B is overwritten with the solution X. Returns
    /// 0 on success, positive value if the matrix is singular, negative
    /// on device error.</summary>
    [DllImport("amql_cuda")]
    public static extern int amql_cuda_cholesky_solve_f64(
        double[] A, int n, double[] B, int nrhs, int streamOrdinal);
}