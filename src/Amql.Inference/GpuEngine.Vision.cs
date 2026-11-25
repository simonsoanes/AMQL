using System.Runtime.InteropServices;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Inference;

public sealed partial class GpuEngine
{
    /// <summary>The container's vision tower, when images are to be read (<see cref="UseVision"/>).</summary>
    private Lazy<VisionHost>? _vision;
    private VisionPlan? _visionPlan;

    /// <summary>Whether the engine can turn an image into tokens.</summary>
    public bool HasVision => _vision is not null;

    /// <summary>Gives the engine the container's vision tower. Its weights are read when the first
    /// image comes, stay in host memory and are copied over a layer at a time while an image is
    /// read, so between images the GPU holds nothing of it.</summary>
    public void UseVision(VisionPlan plan, OperandStore store)
    {
        _visionPlan = plan;
        _vision = new Lazy<VisionHost>(() => new VisionHost(plan, store));
    }

    /// <summary>A tower's weights as BF16 in host memory.</summary>
    private sealed class VisionHost
    {
        public readonly VisionPlan Plan;
        public readonly ushort[] Patch, TableX, TableY, Projection;
        public readonly float[]? Bias, Scale;
        public readonly Layer[] Layers;

        public sealed record Layer(ushort[] Qkv, ushort[] O, ushort[] Up, ushort[] Down,
            float[] PreAttn, float[] PostAttn, float[] PreFfn, float[] PostFfn, float[] QNorm, float[] KNorm);

        public VisionHost(VisionPlan plan, OperandStore store)
        {
            Plan = plan;
            int h = plan.HiddenSize, q = plan.Heads * plan.HeadDim, inter = plan.Intermediate;
            // As stored, or rounded to nearest where it is stored wider.
            (long[] Shape, ushort[] Bits) Rounded(string obj, string name)
            {
                if (store.DtypeOf(obj, name) == Dtype.BF16)
                {
                    return store.ResolveBf16(obj, name);
                }
                var wide = store.ResolveWidened(obj, name);
                return (wide.Shape, Array.ConvertAll(wide.Values, v =>
                {
                    uint b = BitConverter.SingleToUInt32Bits(v), low = b & 0xFFFF, up = b >> 16;
                    return (ushort)(low > 0x8000 || (low == 0x8000 && (up & 1) == 1) ? up + 1 : up);
                }));
            }
            ushort[] Bf16(string name, params long[] shape)
            {
                var (actual, bits) = Rounded(VisionPlan.Tower, name);
                return actual.SequenceEqual(shape) ? bits
                    : throw new ContainerException($"the vision tower's '{name}' is [{string.Join(", ", actual)}], expected [{string.Join(", ", shape)}]");
            }
            float[] F32(string name) => store.ResolveWidened(VisionPlan.Tower, name).Values;
            Patch = Bf16("patch_embedder.input_proj.weight", h, 3L * plan.PatchSize * plan.PatchSize);
            var table = Bf16("patch_embedder.position_embedding_table", 2, plan.Positions, h);
            // A side is never longer than the largest budget's patches in one row.
            int longest = Math.Min(plan.Positions, VisionPlan.Budgets[^1] * plan.Side);
            TableX = table.AsSpan(0, longest * h).ToArray();
            TableY = table.AsSpan(plan.Positions * h, longest * h).ToArray();
            if (plan.Standardise)
            {
                Bias = F32("std_bias");
                Scale = F32("std_scale");
            }
            Projection = Rounded(VisionPlan.Projector, "embedding_projection.weight").Bits;
            Layers = new Layer[plan.Layers];
            for (int i = 0; i < Layers.Length; i++)
            {
                string p = $"encoder.layers.{i}.";
                Layers[i] = new Layer(
                    [.. Bf16(p + "self_attn.q_proj.linear.weight", q, h), .. Bf16(p + "self_attn.k_proj.linear.weight", q, h), .. Bf16(p + "self_attn.v_proj.linear.weight", q, h)],
                    Bf16(p + "self_attn.o_proj.linear.weight", h, q),
                    [.. Bf16(p + "mlp.gate_proj.linear.weight", inter, h), .. Bf16(p + "mlp.up_proj.linear.weight", inter, h)],
                    Bf16(p + "mlp.down_proj.linear.weight", h, inter),
                    F32(p + "input_layernorm.weight"), F32(p + "post_attention_layernorm.weight"),
                    F32(p + "pre_feedforward_layernorm.weight"), F32(p + "post_feedforward_layernorm.weight"),
                    F32(p + "self_attn.q_norm.weight"), F32(p + "self_attn.k_norm.weight"));
            }
        }
    }

    /// <summary>The image's tokens as the model reads them, [tokens × hidden].</summary>
    public float[] EncodeImage(PreparedImage image)
    {
        var host = _vision?.Value ?? throw new InvalidOperationException("the engine has no vision tower");
        var plan = host.Plan;
        int rows = image.Across * image.Down, h = plan.HiddenSize, q = plan.Heads * plan.HeadDim, inter = plan.Intermediate;
        int values = 3 * plan.PatchSize * plan.PatchSize;
        var held = new List<IntPtr>();
        IntPtr Take(long bytes)
        {
            Check(Native.amqe_alloc(out var ptr, (nuint)Math.Max(16, bytes)), "vision memory");
            held.Add(ptr);
            return ptr;
        }
        IntPtr Put<T>(T[] data) where T : unmanaged
        {
            var ptr = Take(data.LongLength * System.Runtime.CompilerServices.Unsafe.SizeOf<T>());
            Upload(ptr, data);
            return ptr;
        }
        // The caches' format does not apply to the tower's keys and values, which are f32.
        Check(Native.amqe_set_kv_format(_engine, 0), "vision");
        try
        {
            var position = new int[2 * rows];
            var reach = new int[rows];
            for (int r = 0; r < rows; r++)
            {
                position[2 * r] = r % image.Across;
                position[2 * r + 1] = r / image.Across;
                reach[r] = rows - 1 - r;   // every patch sees every patch
            }
            var invFreq = Enumerable.Range(0, plan.HeadDim / 4).Select(i => (float)(1.0 / Math.Pow(plan.Theta, 2.0 * i / (plan.HeadDim / 2)))).ToArray();
            IntPtr pos = Put(position), ahead = Put(reach), freq = Put(invFreq);
            IntPtr x = Take(4L * rows * h), hn = Take(4L * rows * h), branch = Take(4L * rows * h);
            IntPtr qkv = Take(12L * rows * q), qs = Take(4L * rows * q), ks = Take(4L * rows * q), vs = Take(4L * rows * q), ctx = Take(4L * rows * q);
            IntPtr ffn = Take(4L * rows * inter), raw = Take(4L * rows * Math.Max(2 * inter, 3 * q));

            void Mul(IntPtr w, int n, int k, IntPtr input, IntPtr output, int act = 0, bool glu = false, int r = 0) =>
                Check(Native.amqe_linear(_engine, w, BF16, n, k, input, k, r == 0 ? rows : r, output, n, IntPtr.Zero, IntPtr.Zero, n, 1f,
                    act, glu ? 1 : 0, raw, IntPtr.Zero, 0f, IntPtr.Zero, 0f, 0, IntPtr.Zero, IntPtr.Zero), "vision projection");
            void RmsNorm(IntPtr input, IntPtr output, IntPtr w) =>
                Check(Native.amqe_norm(_engine, input, h, output, h, rows, h, w, 0f, IntPtr.Zero, plan.Eps, 0), "vision norm");

            IntPtr pixels = Put(image.Patches), patch = Put(host.Patch);
            Mul(patch, h, values, pixels, x);
            Check(Native.amqe_vision_positions(_engine, x, Put(host.TableX), Put(host.TableY), pos, rows, h), "vision positions");

            // One layer's weights at a time, in buffers they take turns in.
            IntPtr wQkv = Take(2L * 3 * q * h), wO = Take(2L * h * q), wUp = Take(4L * inter * h), wDown = Take(2L * h * inter);
            IntPtr n1 = Take(4L * h), n2 = Take(4L * h), n3 = Take(4L * h), n4 = Take(4L * h), qn = Take(4L * plan.HeadDim), kn = Take(4L * plan.HeadDim);
            foreach (var l in host.Layers)
            {
                Check(Native.amqe_sync(_engine), "vision");   // the layer before is done with them
                Upload(wQkv, l.Qkv);
                Upload(wO, l.O);
                Upload(wUp, l.Up);
                Upload(wDown, l.Down);
                Upload(n1, l.PreAttn);
                Upload(n2, l.PostAttn);
                Upload(n3, l.PreFfn);
                Upload(n4, l.PostFfn);
                Upload(qn, l.QNorm);
                Upload(kn, l.KNorm);
                RmsNorm(x, hn, n1);
                Mul(wQkv, 3 * q, h, hn, qkv);
                Check(Native.amqe_vision_prep(_engine, qkv, 3 * q, rows, plan.Heads, plan.HeadDim, qs, ks, vs, pos, qn, kn, plan.Eps, freq), "vision heads");
                Check(Native.amqe_set_reach(_engine, ahead), "vision");
                Check(Native.amqe_attention_rows(_engine, qs, ks, vs, IntPtr.Zero, ctx, 0, rows, plan.Heads, plan.Heads, plan.HeadDim, 1f, 0f, 0, 0), "vision attention");
                Check(Native.amqe_set_reach(_engine, IntPtr.Zero), "vision");
                Mul(wO, h, q, ctx, branch);
                Check(Native.amqe_norm_join(_engine, x, branch, rows, h, n2, 0f, plan.Eps, 1f, 1f), "vision residual");
                RmsNorm(x, hn, n3);
                Mul(wUp, inter, h, hn, ffn, act: 2, glu: true);
                Mul(wDown, h, inter, ffn, branch);
                Check(Native.amqe_norm_join(_engine, x, branch, rows, h, n4, 0f, plan.Eps, 1f, 1f), "vision residual");
            }
            IntPtr pooled = Take(4L * image.Tokens * h), tokens = Take(4L * image.Tokens * _hidden);
            Check(Native.amqe_vision_pool(_engine, x, pooled, image.Tokens, h, image.Across, plan.Side, MathF.Sqrt(h),
                host.Bias is { } bias ? Put(bias) : IntPtr.Zero, host.Scale is { } scale ? Put(scale) : IntPtr.Zero, plan.Eps), "vision pooling");
            Mul(Put(host.Projection), _hidden, h, pooled, tokens, r: image.Tokens);
            var features = new float[(long)image.Tokens * _hidden];
            Download(features, tokens);
            return features;
        }
        finally
        {
            Native.amqe_sync(_engine);
            Native.amqe_set_reach(_engine, IntPtr.Zero);
            Native.amqe_set_kv_format(_engine, _kvFp4 ? 1 : 0);
            foreach (var ptr in held)
            {
                Native.amqe_free(ptr);
            }
        }
    }

    // ── images in a prompt ──────────────────────────────────────────────────

    /// <summary>The images of the prompt the single-sequence API reads next.</summary>
    public IReadOnlyList<ImageSpan>? Images
    {
        get => _main.Images;
        set => _main.Images = value;
    }

    private IntPtr _reach;       // [chunk rows]: how far past itself each row of the chunk sees, in windowed layers
    private bool _reaching;      // the chunk being run has an image in it

    /// <summary>Where the prompt chunk that starts at <paramref name="at"/> ends, so that no image is cut:
    /// <paramref name="rows"/> at most, up to the image that would cross its end.</summary>
    private static int WholeImages(GpuSequence seq, int at, int rows)
    {
        foreach (var image in seq.Images ?? [])
        {
            if (image.Start > at && image.Start < at + rows && image.Start + image.Tokens > at + rows)
            {
                return image.Start - at;
            }
        }
        return rows;
    }

    /// <summary>Puts the images of the chunk at <paramref name="at"/> in place of their embeddings
    /// in <c>_x</c>, and notes how far their rows see.</summary>
    private void PlaceImages(GpuSequence seq, int at, int rows)
    {
        _reaching = false;
        int[]? reach = null;
        foreach (var image in seq.Images ?? [])
        {
            int from = Math.Max(image.Start, at), to = Math.Min(image.Start + image.Tokens, at + rows);
            if (from >= to)
            {
                continue;
            }
            var handle = GCHandle.Alloc(image.Features, GCHandleType.Pinned);
            try
            {
                Check(Native.amqe_upload(_engine, _x + (nint)((long)(from - at) * _hidden * 4),
                    handle.AddrOfPinnedObject() + (nint)((long)(from - image.Start) * _hidden * 4), (nuint)((long)(to - from) * _hidden * 4)), "image tokens");
            }
            finally
            {
                handle.Free();
            }
            if (_visionPlan is { WholeImages: true })
            {
                reach ??= new int[rows];
                for (int p = from; p < to; p++)
                {
                    reach[p - at] = to - 1 - p;
                }
            }
        }
        if (reach is null)
        {
            return;
        }
        if (_reach == IntPtr.Zero)
        {
            _reach = Alloc(_chunkRows * 4L);
        }
        Upload(_reach, reach);
        _reaching = true;
    }
}
