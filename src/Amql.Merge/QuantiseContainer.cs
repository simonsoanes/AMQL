using System.Text.Json.Nodes;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>The 4-bit form a container's projections are written in.</summary>
public enum Fp4Format
{
    /// <summary>NVFP4: E2M1 elements, an E4M3 scale per 16, an F32 tensor scale.</summary>
    Nvfp4,
    /// <summary>MXFP4 (OCP MX): E2M1 elements, an E8M0 scale per 32.</summary>
    Mxfp4,
}

/// <summary>How each weight finds its 4-bit values.</summary>
public enum Fp4Rounding
{
    /// <summary>Each block's scale from its peak, each element to the nearest
    /// value. Needs no GPU.</summary>
    Nearest,
    /// <summary>Each block takes the scale, among those near its peak's, that
    /// leaves the least squared error; elements to nearest. On the GPU.</summary>
    ScaleSearch,
    /// <summary>GPTQ over calibration text: rounding error carried across each
    /// row so the layer's output over real inputs moves least, block scales
    /// searched against the same weighting. On the GPU.</summary>
    Gptq,
}

public sealed record QuantiseOptions(Fp4Format Format)
{
    public Fp4Rounding Rounding { get; init; } = Fp4Rounding.Nearest;

    /// <summary>Text the model reads to calibrate GPTQ (required for it).</summary>
    public string? CalibrationText { get; init; }

    /// <summary>Calibration as token ids, in place of text.</summary>
    public int[]? CalibrationIds { get; init; }

    /// <summary>At most this many calibration tokens.</summary>
    public int CalibrationTokens { get; init; } = 32768;

    /// <summary>Tokens per calibration sequence.</summary>
    public int SequenceTokens { get; init; } = 1024;

    /// <summary>GPTQ's damping, as a fraction of the Gram's mean diagonal.</summary>
    public double Damp { get; init; } = 0.01;

    /// <summary>Progress lines, if anyone is listening.</summary>
    public Action<string>? Log { get; init; }
}

public sealed record QuantiseReport(string OutDir, Fp4Format Format, int Quantised, int Kept,
    long BytesBefore, long BytesAfter, IReadOnlyList<string> Notes);

/// <summary>
/// Compiles a container's projections to 4 bits, once: every 2-D weight of the
/// decoder stack and the MTP drafter (attention and Gated DeltaNet projections,
/// the FFN, the drafter's projector) is written as FP4 elements with its block
/// scales beside it (<c>{name}_scale</c>, and NVFP4's <c>{name}_scale_2</c>),
/// the layout quantised checkpoints ship and the operand store decodes. Norms,
/// convolutions, gates' small vectors, the embedding and the head stay as they
/// were. Every runtime then multiplies by the same 4-bit weights: the managed
/// path decodes them exactly, and the GPU engine reads them as stored, so the
/// two agree as they do at full precision, and anything measured against the
/// 4-bit model (an Entail adapter's triggers, an MTP drafter's acceptance)
/// holds wherever it runs.
/// <para>How the 4-bit values are found is <see cref="Fp4Rounding"/>: to
/// nearest, with each block's scale searched, or GPTQ against calibration text
/// the source model reads on the GPU (each layer's input Gram, one layer at a
/// time). The stored layout is the same for all three.</para>
/// </summary>
public static class QuantiseContainer
{
    private static readonly string[] Objects = ["target.decoder_stack", "mtp.stack"];

    public static QuantiseReport Transform(string containerDir, string outDir, Fp4Format format) =>
        Transform(containerDir, outDir, new QuantiseOptions(format));

    public static QuantiseReport Transform(string containerDir, string outDir, QuantiseOptions options)
    {
        var format = options.Format;
        if (options.Rounding == Fp4Rounding.Gptq && string.IsNullOrWhiteSpace(options.CalibrationText) && options.CalibrationIds is null)
        {
            throw new MergeException("GPTQ needs calibration text");
        }
        if (Directory.Exists(outDir) && Directory.EnumerateFileSystemEntries(outDir).Any())
        {
            throw new MergeException($"quantise output '{outDir}' already exists and is not empty");
        }
        using var container = Vindex3Container.Open(containerDir);
        var projections = Projections(container);
        using var calibrator = options.Rounding == Fp4Rounding.Nearest ? null : new Calibrator(container, options);
        Directory.CreateDirectory(outDir);
        foreach (var file in Directory.EnumerateFiles(container.Root))
        {
            File.Copy(file, Path.Combine(outDir, Path.GetFileName(file)));
        }

        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(container.Root, "index.json")))!.AsObject();
        var reps = index["representations"]!.AsObject();
        int quantised = 0, kept = 0;
        long before = 0, after = 0;
        var notes = new List<string>();
        foreach (var (repId, entry) in container.Index.Representations)
        {
            string from = Path.Combine(container.Root, entry.Segment);
            string to = Path.Combine(outDir, entry.Segment);
            Directory.CreateDirectory(Path.GetDirectoryName(to)!);
            if (!Objects.Contains(entry.Object))
            {
                File.Copy(from, to);
                continue;
            }
            using var segment = SegmentFile.Open(from);
            var tensors = new List<NamedTensorData>();
            foreach (var t in segment.Header.Tensors)
            {
                var dtype = DtypeExtensions.FromLabel(t.Dtype);
                before += t.Len;
                if (Quantisable(entry.Object, t, dtype, segment, projections, out var reads))
                {
                    int rows = checked((int)t.Shape[0]), cols = checked((int)t.Shape[1]);
                    var values = BitPattern.WidenToF32(dtype, segment.ReadBytes(t.Name));
                    var written = calibrator is null
                        ? Quantise(t.Name, values, rows, cols, format)
                        : calibrator.Quantise(t.Name, values, rows, cols, reads);
                    tensors.AddRange(written);
                    after += written.Sum(w => (long)w.Data.Length);
                    quantised++;
                    continue;
                }
                var name = t.Name;
                long length = t.Len;
                tensors.Add(new NamedTensorData
                {
                    Name = t.Name,
                    Dtype = dtype,
                    Shape = t.Shape,
                    Source = (offset, count) => segment.ReadBytes(name, offset, count),
                    SourceLength = length,
                });
                after += length;
                kept++;
            }
            var result = SegmentWriter.Write(to, segment.Header.Representation, tensors);
            var node = reps[repId]!.AsObject();
            node["tensor_count"] = tensors.Count;
            node["payload_bytes"] = result.PayloadBytes;
            node["payload_sha256"] = result.PayloadSha256Hex;
            node["segment_sha256"] = result.SegmentSha256Hex;
        }
        foreach (var dir in Directory.EnumerateDirectories(container.Root))
        {
            string name = Path.GetFileName(dir);
            if (name != "segments")
            {
                CopyDirectory(dir, Path.Combine(outDir, name));
            }
        }
        index["authority"] = "derived";
        index["derived_from_model"] = container.Index.Model;
        var q = new JsonObject
        {
            ["format"] = format == Fp4Format.Nvfp4 ? "nvfp4" : "mxfp4",
            ["rounding"] = options.Rounding switch
            {
                Fp4Rounding.Gptq => "gptq",
                Fp4Rounding.ScaleSearch => "scale_search",
                _ => "nearest",
            },
            ["objects"] = new JsonArray(Objects.Select(o => (JsonNode?)o).ToArray()),
            ["quantised_tensors"] = quantised,
        };
        if (calibrator is not null && options.Rounding == Fp4Rounding.Gptq)
        {
            q["calibration_tokens"] = calibrator.Tokens;
            q["damp"] = options.Damp;
        }
        index["quantisation"] = q;
        File.WriteAllText(Path.Combine(outDir, "index.json"), index.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = false }));
        notes.Add($"{quantised} projections to {format} ({q["rounding"]}), {kept} tensors kept; payload {before / 1e9:F2} GB -> {after / 1e9:F2} GB");
        return new QuantiseReport(outDir, format, quantised, kept, before, after, notes);
    }

    /// <summary>What a projection reads: its layer and which of the layer's
    /// inputs (<see cref="GpuEngine.GramMixerIn"/> and the rest), or layer −1
    /// for weights no calibration reaches (the drafter's).</summary>
    private readonly record struct Reads(int Layer, int Input);

    /// <summary>The stack's projections as its plan reads them: attention's
    /// q/k/v/o, the Gated DeltaNet's in- and out-projections, the FFN. Not the
    /// head (the stack may carry an untied one) or anything else.</summary>
    private static Dictionary<string, Reads> Projections(Vindex3Container container)
    {
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var names = new Dictionary<string, Reads>(StringComparer.Ordinal);
        for (int i = 0; i < plan.Layers.Count; i++)
        {
            var l = plan.Layers[i];
            void Add(OperandRef? r, int input)
            {
                if (r is not null && r.ObjectId == "target.decoder_stack")
                {
                    names[r.TensorName] = new Reads(i, input);
                }
            }
            if (l.Attention is { } a)
            {
                Add(a.QProj, GpuEngine.GramMixerIn); Add(a.KProj, GpuEngine.GramMixerIn); Add(a.VProj, GpuEngine.GramMixerIn);
                Add(a.OProj, GpuEngine.GramMixerOut);
            }
            if (l.LinearAttention is { } g)
            {
                Add(g.InProjQkv, GpuEngine.GramMixerIn); Add(g.InProjZ, GpuEngine.GramMixerIn);
                Add(g.InProjA, GpuEngine.GramMixerIn); Add(g.InProjB, GpuEngine.GramMixerIn);
                Add(g.OutProj, GpuEngine.GramMixerOut);
            }
            if (l.Ffn?.Dense is { } d)
            {
                Add(d.Gate, GpuEngine.GramFfnIn); Add(d.Up, GpuEngine.GramFfnIn);
                Add(d.Down, GpuEngine.GramDownIn);
            }
        }
        return names;
    }

    /// <summary>A projection of the stack, or of the MTP drafter (its layer's
    /// projections and its fc), 2-D float with columns a multiple of 32 (both
    /// block sizes) and not already scaled. Every one, however small: weights
    /// the engine stacks into one matrix (q/k/v, a Gated DeltaNet's four
    /// in-projections) must all be 4-bit for the stack to be, and the GPU
    /// engine's quantising at load takes the same set.</summary>
    private static bool Quantisable(string objectId, SegmentTensor t, Dtype dtype, SegmentFile segment,
        Dictionary<string, Reads> projections, out Reads reads)
    {
        reads = new Reads(-1, -1);
        bool projection = objectId == "mtp.stack"
            ? t.Name == "fc.weight" || (t.Name.StartsWith("layers.", StringComparison.Ordinal) &&
                                        (t.Name.Contains("_proj.", StringComparison.Ordinal)))
            : projections.TryGetValue(t.Name, out reads);
        return projection && t.Shape.Length == 2 && dtype is Dtype.BF16 or Dtype.F16 or Dtype.F32 &&
               t.Shape[1] % 32 == 0 && !segment.Contains(t.Name + "_scale");
    }

    /// <summary>The tensor and its scale companions, as the operand store reads them.</summary>
    public static List<NamedTensorData> Quantise(string name, float[] values, int rows, int cols, Fp4Format format)
    {
        if (format == Fp4Format.Nvfp4)
        {
            var q = Nvfp4.Quantize(values, rows, cols);
            var scales = new byte[q.Scales.Length];
            for (int i = 0; i < scales.Length; i++)
            {
                scales[i] = BitPattern.EncodeF8E4M3(q.Scales[i]);
            }
            return Written(name, rows, cols, format, q.Elements, scales, q.Global);
        }
        var mx = Mxfp4.Quantize(values, rows, cols);
        return Written(name, rows, cols, format, mx.Packed, mx.BlockScales, 1f);
    }

    private static List<NamedTensorData> Written(string name, int rows, int cols, Fp4Format format, byte[] codes, byte[] scales, float global) =>
        format == Fp4Format.Nvfp4
            ?
            [
                new() { Name = name, Dtype = Dtype.FP4, Shape = [rows, cols], Data = codes },
                new() { Name = name + "_scale", Dtype = Dtype.F8_E4M3, Shape = [rows, cols / Nvfp4.BlockElements], Data = scales },
                new() { Name = name + "_scale_2", Dtype = Dtype.F32, Shape = [1], Data = BitConverter.GetBytes(global) },
            ]
            :
            [
                new() { Name = name, Dtype = Dtype.FP4, Shape = [rows, cols], Data = codes },
                new() { Name = name + "_scale", Dtype = Dtype.U8, Shape = [rows, cols / Mxfp4.BlockElements], Data = scales },
            ];

    /// <summary>
    /// The source model on the GPU, for searched scales and GPTQ. For GPTQ it
    /// reads the calibration text one layer at a time, collecting the Grams of
    /// that layer's inputs (XᵀX), and keeps them while the layer's weights
    /// are rounded; the inputs come from the full-precision model throughout.
    /// </summary>
    private sealed class Calibrator : IDisposable
    {
        private readonly QuantiseOptions _options;
        private readonly OperandStore _store;
        private readonly DecodeSession _session;
        private readonly GpuEngine _engine;
        private readonly int[][] _sequences = [];
        private int _gramLayer = -1;
        private float[]?[] _grams = [];

        public Calibrator(Vindex3Container container, QuantiseOptions options)
        {
            _options = options;
            _store = container.CreateOperandStore();
            var plan = Planner.Plan(container, "target", _store);
            _session = new DecodeSession(plan, _store);
            _session.Prefill([0]);
            _engine = _session.DeviceEngine()
                ?? throw new MergeException($"searched and GPTQ rounding run on the GPU, which is not available ({_session.GpuRefusal})");
            if (_engine.Fp4Matrices > 0)
            {
                throw new MergeException("the source must be at full precision (unset AMQL_GPU_FORMAT)");
            }
            if (options.Rounding == Fp4Rounding.Gptq)
            {
                var ids = (options.CalibrationIds ?? HfTokenizer.FromModelDir(container.Root).EncodeToIds(options.CalibrationText!).ToArray())
                    .Take(options.CalibrationTokens).ToArray();
                if (ids.Length < 64)
                {
                    throw new MergeException($"calibration gives {ids.Length} tokens, too few to calibrate on");
                }
                _sequences = ids.Chunk(options.SequenceTokens).Where(c => c.Length > 8).ToArray();
                Tokens = _sequences.Sum(s => s.Length);
                options.Log?.Invoke($"calibrating on {Tokens} tokens in {_sequences.Length} sequences");
            }
        }

        public int Tokens { get; }

        public List<NamedTensorData> Quantise(string name, float[] values, int rows, int cols, Reads reads)
        {
            float[]? h = null;
            if (_options.Rounding == Fp4Rounding.Gptq && reads.Layer >= 0)
            {
                if (reads.Layer != _gramLayer)
                {
                    _grams = Grams(reads.Layer);
                    _gramLayer = reads.Layer;
                }
                h = _grams[reads.Input]
                    ?? throw new MergeException($"no calibration input reached {name}");
                if (h.LongLength != (long)cols * cols)
                {
                    throw new MergeException($"{name} reads {cols} columns, its layer's input has {Math.Sqrt(h.LongLength)}");
                }
            }
            bool mx = _options.Format == Fp4Format.Mxfp4;
            float global = 1f;
            if (!mx)
            {
                float amax = 0f;
                foreach (var v in values)
                {
                    amax = MathF.Max(amax, MathF.Abs(v));
                }
                global = amax > 0 ? amax / (BitPattern.Fp4MaxValue * 448f) : 1f;
            }
            var (codes, scales) = _engine.QuantiseFp4(values, rows, cols, h, mx, global, _options.Damp);
            return Written(name, rows, cols, _options.Format, codes, scales, global);
        }

        private float[]?[] Grams(int layer)
        {
            var started = System.Diagnostics.Stopwatch.StartNew();
            _engine.BeginGrams(layer);
            foreach (var tokens in _sequences)
            {
                var seq = _engine.NewSequence();
                try
                {
                    _engine.Prefill(seq, tokens);
                }
                finally
                {
                    _engine.Release(seq);
                }
            }
            var grams = _engine.EndGrams(out long rows);
            _options.Log?.Invoke($"layer {layer}: Grams over {rows} rows in {started.Elapsed.TotalSeconds:F1} s");
            return grams;
        }

        public void Dispose()
        {
            _session.Dispose();
            _store.Dispose();
        }
    }

    private static void CopyDirectory(string from, string to)
    {
        Directory.CreateDirectory(to);
        foreach (var file in Directory.EnumerateFiles(from))
        {
            File.Copy(file, Path.Combine(to, Path.GetFileName(file)));
        }
        foreach (var dir in Directory.EnumerateDirectories(from))
        {
            CopyDirectory(dir, Path.Combine(to, Path.GetFileName(dir)));
        }
    }
}
