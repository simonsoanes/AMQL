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
    public Fp4Rounding Rounding { get; init; } = Fp4Rounding.Gptq;

    /// <summary>Text the model reads to calibrate GPTQ; without it (or
    /// <see cref="CalibrationIds"/>), <see cref="DefaultCalibration"/>.</summary>
    public string? CalibrationText { get; init; }

    /// <summary>Calibration as token ids, in place of text.</summary>
    public int[]? CalibrationIds { get; init; }

    /// <summary>A container of the same model, usually its own nearest-rounded NVFP4
    /// copy, that reads the calibration text in place of the source: for a model too
    /// large for the GPU at full precision. The weights rounded are the source's.</summary>
    public string? CalibrateWith { get; init; }

    /// <summary>Also the token embedding and the head (one table in a tied model), which
    /// stay as they are otherwise: with searched scales, whatever the rounding (nearest
    /// when the rounding is).</summary>
    public bool Embedding { get; init; }

    /// <summary>Projections whose names match stay at the source's precision, for
    /// weights whose rounding costs more than their bytes save (attention at long
    /// context, say). The engine stacks a layer's q, k and v, and stacks them at
    /// 4 bits only when all three are: keep them together.</summary>
    public System.Text.RegularExpressions.Regex? Keep { get; init; }

    /// <summary>A quantised container of the same source: a projection it already
    /// holds at 4 bits is copied from it rather than rounded again, so trying
    /// another <see cref="Keep"/> takes minutes, not a calibration.</summary>
    public string? Reuse { get; init; }

    /// <summary>Tensors kept as they are whose names match a pattern are multiplied by
    /// its factor: an edit with no training, such as sharpening a layer's attention
    /// through its query norm.</summary>
    public IReadOnlyList<(System.Text.RegularExpressions.Regex Pattern, float Factor)> Scale { get; init; } = [];

    /// <summary>At most this many calibration tokens.</summary>
    public int CalibrationTokens { get; init; } = 32768;

    /// <summary>Tokens per calibration sequence.</summary>
    public int SequenceTokens { get; init; } = 1024;

    /// <summary>GPTQ's damping, as a fraction of the Gram's mean diagonal.</summary>
    public double Damp { get; init; } = 0.01;

    /// <summary>Progress lines, if anyone is listening.</summary>
    public Action<string>? Log { get; init; }

    /// <summary>The calibration GPTQ reads when none is given: about 130 KB of
    /// this repository's docs and C#. On Qwen 3.5 4B it took NVFP4's added
    /// perplexity from +4.9% to +1.3% on Python it never saw, so it carries
    /// beyond its own domain; text from the model's real use does better.</summary>
    public static string DefaultCalibration => _default.Value;

    private static readonly Lazy<string> _default = new(() =>
    {
        using var stream = typeof(QuantiseOptions).Assembly.GetManifestResourceStream("Amql.Merge.Calibration.default.txt")
            ?? throw new MergeException("the default calibration text is missing from Amql.Merge");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    });
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
/// time, and each routed expert's over the rows sent to it). A model too large
/// for the GPU reads it as a 4-bit copy of itself (<see cref="QuantiseOptions.CalibrateWith"/>).
/// The stored layout is the same for all three.</para>
/// </summary>
public static class QuantiseContainer
{
    private static readonly string[] Objects = ["target.decoder_stack", "mtp.stack", DrafterPlan.Stack];
    private static readonly string[] Tables = ["target.embedding", "target.output_head"];

    public static QuantiseReport Transform(string containerDir, string outDir, Fp4Format format) =>
        Transform(containerDir, outDir, new QuantiseOptions(format));

    public static QuantiseReport Transform(string containerDir, string outDir, QuantiseOptions options)
    {
        var format = options.Format;
        if (options.Rounding == Fp4Rounding.Gptq && string.IsNullOrWhiteSpace(options.CalibrationText) && options.CalibrationIds is null)
        {
            options = options with { CalibrationText = QuantiseOptions.DefaultCalibration };
            options.Log?.Invoke("calibrating on the built-in text (--calibration for your own)");
        }
        if (Directory.Exists(outDir) && Directory.EnumerateFileSystemEntries(outDir).Any())
        {
            throw new MergeException($"quantise output '{outDir}' already exists and is not empty");
        }
        if (options.Rounding == Fp4Rounding.Gptq && options.CalibrateWith is null && options.Reuse is null && TooLargeForTheGpu(containerDir, out long stackBytes))
        {
            // The whole model has to be on the GPU to read the calibration text: a
            // nearest-rounded copy of it, made for that and then deleted.
            string reader = Path.TrimEndingDirectorySeparator(Path.GetFullPath(outDir)) + ".calibration";
            if (Directory.Exists(reader))
            {
                throw new MergeException($"'{reader}' is in the way: the {stackBytes / 1e9:F0} GB stack does not fit the GPU, and its calibration copy goes there");
            }
            options.Log?.Invoke($"the {stackBytes / 1e9:F0} GB stack does not fit the GPU: calibrating with a 4-bit copy of it");
            try
            {
                Transform(containerDir, reader, new QuantiseOptions(format) { Rounding = Fp4Rounding.Nearest });
                return Transform(containerDir, outDir, options with { CalibrateWith = reader });
            }
            finally
            {
                if (Directory.Exists(reader))
                {
                    Directory.Delete(reader, recursive: true);
                }
            }
        }
        using var container = Vindex3Container.Open(containerDir);
        var projections = Projections(container);
        var normOffsets = options.Scale.Count > 0 ? NormOffsets(container) : [];
        using var reuse = options.Reuse is null ? null : Vindex3Container.Open(options.Reuse);
        using var calibrator = options.Rounding == Fp4Rounding.Nearest || reuse is not null ? null : new Calibrator(container, options);
        Directory.CreateDirectory(outDir);
        foreach (var file in Directory.EnumerateFiles(container.Root))
        {
            File.Copy(file, Path.Combine(outDir, Path.GetFileName(file)));
        }

        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(container.Root, "index.json")))!.AsObject();
        var reps = index["representations"]!.AsObject();
        var objects = options.Embedding ? [.. Objects, .. Tables] : Objects;
        int quantised = 0, kept = 0;
        long before = 0, after = 0;
        var notes = new List<string>();
        foreach (var (repId, entry) in container.Index.Representations)
        {
            string from = Path.Combine(container.Root, entry.Segment);
            string to = Path.Combine(outDir, entry.Segment);
            Directory.CreateDirectory(Path.GetDirectoryName(to)!);
            if (!objects.Contains(entry.Object))
            {
                File.Copy(from, to);
                continue;
            }
            using var segment = SegmentFile.Open(from);
            using var reused = reuse is not null && reuse.Index.Representations.TryGetValue(repId, out var other)
                ? SegmentFile.Open(Path.Combine(reuse.Root, other.Segment)) : null;
            var tensors = new List<NamedTensorData>();
            foreach (var t in segment.Header.Tensors)
            {
                var dtype = DtypeExtensions.FromLabel(t.Dtype);
                before += t.Len;
                bool keep = options.Keep?.IsMatch(t.Name) == true;
                if (!keep && reuse is not null && Quantisable(entry.Object, t, dtype, segment, projections, out _))
                {
                    if (reused is null || !reused.Contains(t.Name + "_scale"))
                    {
                        throw new MergeException($"'{t.Name}' is not quantised in '{options.Reuse}'");
                    }
                    foreach (var part in new[] { t.Name, t.Name + "_scale", t.Name + "_scale_2" }.Where(reused.Contains))
                    {
                        var rt = reused.GetTensor(part);
                        var seg = reused;
                        tensors.Add(new NamedTensorData
                        {
                            Name = part, Dtype = DtypeExtensions.FromLabel(rt.Dtype), Shape = rt.Shape,
                            Source = (offset, count) => seg.ReadBytes(part, offset, count), SourceLength = rt.Len,
                        });
                        after += rt.Len;
                    }
                    quantised++;
                    continue;
                }
                if (!keep && Quantisable(entry.Object, t, dtype, segment, projections, out var reads))
                {
                    int rows = checked((int)t.Shape[0]), cols = checked((int)t.Shape[1]);
                    // Quantised as the segment is written, one weight at a time: the whole
                    // of a large model's 4-bit weights is never held.
                    var written = Deferred(t.Name, rows, cols, format, () =>
                    {
                        var values = BitPattern.WidenToF32(dtype, segment.ReadBytes(t.Name));
                        return calibrator is null
                            ? Quantise(t.Name, values, rows, cols, format)
                            : calibrator.Quantise(t.Name, values, rows, cols, reads);
                    });
                    tensors.AddRange(written);
                    after += written.Sum(w => w.SourceLength);
                    quantised++;
                    continue;
                }
                var name = t.Name;
                long length = t.Len;
                if (options.Scale.FirstOrDefault(s => s.Pattern.IsMatch(name)) is { Pattern: not null } rule)
                {
                    // A norm read as x·(offset + w) is scaled as a whole: the stored w becomes
                    // factor·(offset + w) − offset, so the factor multiplies what the norm applies.
                    float shift = entry.Object == "target.decoder_stack" && normOffsets.TryGetValue(name, out var o) ? o : 0f;
                    tensors.Add(Scaled(t, dtype, segment.ReadBytes(name), rule.Factor, shift));
                    after += length;
                    kept++;
                    continue;
                }
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
            ["objects"] = new JsonArray(objects.Select(o => (JsonNode?)o).ToArray()),
            ["quantised_tensors"] = quantised,
        };
        if (options.Keep is not null)
        {
            q["kept"] = options.Keep.ToString();
        }
        if (options.Scale.Count > 0)
        {
            q["scaled"] = new JsonArray(options.Scale.Select(s => (JsonNode)$"{s.Pattern}={s.Factor.ToString(System.Globalization.CultureInfo.InvariantCulture)}").ToArray());
        }
        if (reuse is not null && JsonNode.Parse(File.ReadAllText(Path.Combine(reuse.Root, "index.json")))?["quantisation"] is JsonObject was)
        {
            // The 4-bit tensors are the reused container's: so is how they were rounded.
            foreach (var key in new[] { "format", "rounding", "calibration_tokens", "damp", "calibrated_with", "searched_experts" })
            {
                q[key] = was[key]?.DeepClone();
            }
            q["reused"] = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(options.Reuse!)));
        }
        if (calibrator is not null && options.Rounding == Fp4Rounding.Gptq)
        {
            q["calibration_tokens"] = calibrator.Tokens;
            q["damp"] = options.Damp;
            if (options.CalibrateWith is { } with)
            {
                q["calibrated_with"] = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(with)));
            }
            if (calibrator.Thin > 0)
            {
                q["searched_experts"] = calibrator.Thin;
            }
        }
        index["quantisation"] = q;
        File.WriteAllText(Path.Combine(outDir, "index.json"), index.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = false }));
        notes.Add($"{quantised} projections to {format} ({q["rounding"]}), {kept} tensors kept; payload {before / 1e9:F2} GB -> {after / 1e9:F2} GB");
        return new QuantiseReport(outDir, format, quantised, kept, before, after, notes);
    }

    /// <summary>A float tensor whose effective value, <paramref name="offset"/> + w, is
    /// multiplied by <paramref name="factor"/>, written back in its own dtype.</summary>
    internal static NamedTensorData Scaled(SegmentTensor t, Dtype dtype, byte[] bytes, float factor, float offset = 0f)
    {
        var values = BitPattern.WidenToF32(dtype, bytes);
        var data = new byte[bytes.Length];
        for (int i = 0; i < values.Length; i++)
        {
            float v = factor * (offset + values[i]) - offset;
            switch (dtype)
            {
                case Dtype.BF16: BitConverter.TryWriteBytes(data.AsSpan(2 * i), BitPattern.EncodeBf16(v)); break;
                case Dtype.F16: BitConverter.TryWriteBytes(data.AsSpan(2 * i), BitPattern.EncodeF16(v)); break;
                case Dtype.F32: BitConverter.TryWriteBytes(data.AsSpan(4 * i), v); break;
                default: throw new MergeException($"'{t.Name}' is {dtype}: only float tensors can be scaled");
            }
        }
        return new NamedTensorData { Name = t.Name, Dtype = dtype, Shape = t.Shape, Data = data };
    }

    private static bool TooLargeForTheGpu(string containerDir, out long stackBytes)
    {
        using var container = Vindex3Container.Open(containerDir);
        stackBytes = container.Index.Representations.Values.Where(r => Objects.Contains(r.Object)).Sum(r => r.PayloadBytes);
        // Room is needed beside the weights for the Grams and the scratch.
        return CudaShim.TryMemoryInfo(out _, out long total) && stackBytes > total * 0.75;
    }

    /// <summary>Every norm weight of the stack the plan applies, by tensor name, with the
    /// offset it is read with: 0 for x·w, 1 for x·(1 + w).</summary>
    private static Dictionary<string, float> NormOffsets(Vindex3Container container)
    {
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var found = new Dictionary<string, float>(StringComparer.Ordinal);
        var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
        void Walk(object? node, int depth)
        {
            if (node is null || depth > 12 || node is string || node.GetType().IsPrimitive || node.GetType().IsEnum || !seen.Add(node))
            {
                return;
            }
            if (node is NormOp n)
            {
                if (n.Weight.ObjectId == "target.decoder_stack")
                {
                    found[n.Weight.TensorName] = n.WeightOffset;
                }
                return;
            }
            if (node is System.Collections.IEnumerable items)
            {
                foreach (var item in items)
                {
                    Walk(item, depth + 1);
                }
                return;
            }
            if (node.GetType().Namespace?.StartsWith("Amql", StringComparison.Ordinal) != true)
            {
                return;
            }
            foreach (var p in node.GetType().GetProperties(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance))
            {
                if (p.GetIndexParameters().Length == 0)
                {
                    Walk(p.GetValue(node), depth + 1);
                }
            }
        }
        foreach (var layer in plan.Layers)
        {
            Walk(layer, 0);
        }
        return found;
    }

    /// <summary>What a projection reads: its layer and which of the layer's
    /// inputs (<see cref="GpuEngine.GramMixerIn"/> and the rest), or layer −1
    /// for weights no calibration reaches (the drafter's).</summary>
    private readonly record struct Reads(int Layer, int Input, int Expert = -1);

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
            if ((l.Ffn?.Dense ?? l.Ffn?.Routed?.Hybrid?.Dense) is { } d)
            {
                Add(d.Gate, GpuEngine.GramFfnIn); Add(d.Up, GpuEngine.GramFfnIn);
                Add(d.Down, GpuEngine.GramDownIn);
            }
            if (l.Ffn?.Routed is { } moe)
            {
                // Each expert reads the rows the router sends it. The router itself stays.
                for (int e = 0; e < moe.NumExperts; e++)
                {
                    names[$"{moe.ExpertGatePrefix}{e}.{moe.ExpertGateStem}"] = new Reads(i, GpuEngine.GramExpertIn, e);
                    names[$"{moe.ExpertUpPrefix}{e}.{moe.ExpertUpStem}"] = new Reads(i, GpuEngine.GramExpertIn, e);
                    names[$"{moe.ExpertDownPrefix}{e}.{moe.ExpertDownStem}"] = new Reads(i, GpuEngine.GramExpertDown, e);
                }
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
        bool projection = Tables.Contains(objectId) ? t.Name == "weight" : objectId == DrafterPlan.Stack
            ? Attach.DrafterMatrix(t.Name) : objectId == "mtp.stack"
            ? t.Name == "fc.weight" || (t.Name.StartsWith("layers.", StringComparison.Ordinal) &&
                                        (t.Name.Contains("_proj.", StringComparison.Ordinal)))
            : projections.TryGetValue(t.Name, out reads);
        return projection && t.Shape.Length == 2 && dtype is Dtype.BF16 or Dtype.F16 or Dtype.F32 &&
               t.Shape[1] % 32 == 0 && !segment.Contains(t.Name + "_scale");
    }

    /// <summary>The tensors a weight's quantisation writes, quantised when the
    /// writer first asks for one of them and let go once it has read them all
    /// (it reads each from its start to its end, once).</summary>
    private static List<NamedTensorData> Deferred(string name, int rows, int cols, Fp4Format format,
        Func<List<NamedTensorData>> quantise)
    {
        Dictionary<string, byte[]>? held = null;
        bool ran = false;
        byte[] Read(string tensor, long offset, int count)
        {
            if (held is null)
            {
                if (ran)
                {
                    throw new MergeException($"'{name}' was quantised and written already");
                }
                ran = true;
                held = quantise().ToDictionary(t => t.Name, t => t.Data, StringComparer.Ordinal);
            }
            var data = held[tensor];
            var slice = data.AsSpan(checked((int)offset), count).ToArray();
            if (offset + count == data.LongLength)
            {
                held.Remove(tensor);
                if (held.Count == 0)
                {
                    held = null;
                }
            }
            return slice;
        }
        long elements = (long)rows * cols;
        NamedTensorData Tensor(string tensor, Dtype dtype, long[] shape, long bytes) => new()
        {
            Name = tensor, Dtype = dtype, Shape = shape, Source = (offset, count) => Read(tensor, offset, count), SourceLength = bytes,
        };
        return format == Fp4Format.Nvfp4
            ?
            [
                Tensor(name, Dtype.FP4, [rows, cols], elements / 2),
                Tensor(name + "_scale", Dtype.F8_E4M3, [rows, cols / Nvfp4.BlockElements], elements / Nvfp4.BlockElements),
                Tensor(name + "_scale_2", Dtype.F32, [1], 4),
            ]
            :
            [
                Tensor(name, Dtype.FP4, [rows, cols], elements / 2),
                Tensor(name + "_scale", Dtype.U8, [rows, cols / Mxfp4.BlockElements], elements / Mxfp4.BlockElements),
            ];
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
        private readonly Vindex3Container? _reader;
        private readonly OperandStore? _store;
        private readonly DecodeSession? _session;
        private readonly GpuEngine _engine;
        private readonly int[][] _sequences = [];
        private int _gramLayer = -1;
        private float[]?[] _grams = [];

        public Calibrator(Vindex3Container container, QuantiseOptions options)
        {
            _options = options;
            if (options.Rounding != Fp4Rounding.Gptq)
            {
                // Searched scales need the weights and a GPU, not the model on it.
                _engine = GpuEngine.ForRounding()
                    ?? throw new MergeException("searched rounding runs on the GPU, which is not available");
                return;
            }
            try
            {
                if (options.CalibrateWith is { } with)
                {
                    _reader = Vindex3Container.Open(with);
                    if (_reader.Index.Model != container.Index.Model)
                    {
                        throw new MergeException($"'{with}' is a copy of '{_reader.Index.Model}', not of '{container.Index.Model}'");
                    }
                }
                var reader = _reader ?? container;
                _store = reader.CreateOperandStore();
                var plan = Planner.Plan(reader, "target", _store);
                _session = new DecodeSession(plan, _store);
                _session.Prefill([0]);
                _engine = _session.DeviceEngine()
                    ?? throw new MergeException($"GPTQ rounding runs on the GPU, which is not available ({_session.GpuRefusal}); " +
                                                "a model too large for it calibrates with a 4-bit copy of itself (--calibrate-with)");
                if (_engine.Fp4Matrices > 0 && _reader is null)
                {
                    throw new MergeException("the source must be at full precision (unset AMQL_GPU_FORMAT), or name the copy to calibrate with");
                }
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
            catch
            {
                // Nothing calibrated: let go of the model before the error leaves.
                _session?.Dispose();
                _store?.Dispose();
                _reader?.Dispose();
                throw;
            }
        }

        public int Tokens { get; }

        /// <summary>Expert projections that saw too little calibration for a Gram.</summary>
        public int Thin => _thin;
        private int _thin;

        public List<NamedTensorData> Quantise(string name, float[] values, int rows, int cols, Reads reads)
        {
            float[]? h = null;
            string? key = null;
            if (_options.Rounding == Fp4Rounding.Gptq && reads.Layer >= 0)
            {
                if (reads.Layer != _gramLayer)
                {
                    _grams = Grams(reads.Layer);
                    _gramLayer = reads.Layer;
                }
                if (reads.Expert >= 0)
                {
                    h = _engine.ExpertGram(reads.Input, reads.Expert, out long seen);
                    if (seen < cols / 8)
                    {
                        h = null;   // too few rows to weigh columns by: its scales are searched
                        _thin++;
                    }
                    else
                    {
                        key = $"{reads.Layer}/{reads.Input}/{reads.Expert}";
                    }
                }
                else
                {
                    h = _grams[reads.Input]
                        ?? throw new MergeException($"no calibration input reached {name}");
                }
                if (h is not null && h.LongLength != (long)cols * cols)
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
            var (codes, scales) = _engine.QuantiseFp4(values, rows, cols, h, mx, global, _options.Damp, key);
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
            if (_session is null)
            {
                _engine.Dispose();
            }
            _session?.Dispose();
            _store?.Dispose();
            _reader?.Dispose();
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
