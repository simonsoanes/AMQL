using System.Text.Json.Nodes;
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
/// </summary>
public static class QuantiseContainer
{
    private static readonly string[] Objects = ["target.decoder_stack", "mtp.stack"];

    public static QuantiseReport Transform(string containerDir, string outDir, Fp4Format format)
    {
        if (Directory.Exists(outDir) && Directory.EnumerateFileSystemEntries(outDir).Any())
        {
            throw new MergeException($"quantise output '{outDir}' already exists and is not empty");
        }
        using var container = Vindex3Container.Open(containerDir);
        var projections = Projections(container);
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
                if (Quantisable(entry.Object, t, dtype, segment, projections))
                {
                    int rows = checked((int)t.Shape[0]), cols = checked((int)t.Shape[1]);
                    var values = BitPattern.WidenToF32(dtype, segment.ReadBytes(t.Name));
                    var written = Quantise(t.Name, values, rows, cols, format);
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
        index["quantisation"] = new JsonObject
        {
            ["format"] = format == Fp4Format.Nvfp4 ? "nvfp4" : "mxfp4",
            ["objects"] = new JsonArray(Objects.Select(o => (JsonNode?)o).ToArray()),
            ["quantised_tensors"] = quantised,
        };
        File.WriteAllText(Path.Combine(outDir, "index.json"), index.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = false }));
        notes.Add($"{quantised} projections to {format}, {kept} tensors kept; payload {before / 1e9:F2} GB -> {after / 1e9:F2} GB");
        return new QuantiseReport(outDir, format, quantised, kept, before, after, notes);
    }

    /// <summary>The stack's projections as its plan reads them: attention's
    /// q/k/v/o, the Gated DeltaNet's in- and out-projections, the FFN. Not the
    /// head (the stack may carry an untied one) or anything else.</summary>
    private static HashSet<string> Projections(Vindex3Container container)
    {
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var names = new HashSet<string>(StringComparer.Ordinal);
        void Add(OperandRef? r)
        {
            if (r is not null && r.ObjectId == "target.decoder_stack")
            {
                names.Add(r.TensorName);
            }
        }
        foreach (var l in plan.Layers)
        {
            if (l.Attention is { } a)
            {
                Add(a.QProj); Add(a.KProj); Add(a.VProj); Add(a.OProj);
            }
            if (l.LinearAttention is { } g)
            {
                Add(g.InProjQkv); Add(g.InProjZ); Add(g.InProjA); Add(g.InProjB); Add(g.OutProj);
            }
            if (l.Ffn?.Dense is { } d)
            {
                Add(d.Gate); Add(d.Up); Add(d.Down);
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
    private static bool Quantisable(string objectId, SegmentTensor t, Dtype dtype, SegmentFile segment, HashSet<string> projections)
    {
        bool projection = objectId == "mtp.stack"
            ? t.Name == "fc.weight" || (t.Name.StartsWith("layers.", StringComparison.Ordinal) &&
                                        (t.Name.Contains("_proj.", StringComparison.Ordinal)))
            : projections.Contains(t.Name);
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
            return
            [
                new() { Name = name, Dtype = Dtype.FP4, Shape = [rows, cols], Data = q.Elements },
                new() { Name = name + "_scale", Dtype = Dtype.F8_E4M3, Shape = [rows, cols / Nvfp4.BlockElements], Data = scales },
                new() { Name = name + "_scale_2", Dtype = Dtype.F32, Shape = [1], Data = BitConverter.GetBytes(q.Global) },
            ];
        }
        var mx = Mxfp4.Quantize(values, rows, cols);
        return
        [
            new() { Name = name, Dtype = Dtype.FP4, Shape = [rows, cols], Data = mx.Packed },
            new() { Name = name + "_scale", Dtype = Dtype.U8, Shape = [rows, cols / Mxfp4.BlockElements], Data = mx.BlockScales },
        ];
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
