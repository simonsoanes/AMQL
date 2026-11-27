using System.Text;
using System.Text.Json;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;
using Xunit;
using Xunit.Abstractions;

namespace Amql.Tests;

/// <summary>
/// Inkling 2 looped with its experts as the real checkpoint stores them:
/// NVFP4 (<c>U8 [E, rows, cols/2]</c>, an E4M3 scale per 16, a per-expert
/// <c>amax / outer_scale_denominator</c>) and MXFP8 (E4M3 with an E8M0
/// exponent per 32), mixed with plain layers. The tiny arch-zoo model is
/// quantised here; the same model with those values decoded to F32 by
/// transformers' formula is the reference, so the runtime must match it
/// exactly, and the export must give back the stored bytes.
/// </summary>
public sealed class Inkling2Tests
{
    private readonly ITestOutputHelper _output;

    public Inkling2Tests(ITestOutputHelper output) => _output = output;

    private static string Tiny => Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo", "inkling2_loop");

    private static readonly float[] E2M1 = { 0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f, -0f, -0.5f, -1f, -1.5f, -2f, -3f, -4f, -6f };

    /// <summary>Layer l's experts: plain (l % 3 == 0), NVFP4 (1) or MXFP8 (2),
    /// so the loop's core (4-7) has all three.</summary>
    private static string FormOf(int layer) => (layer % 3) switch { 0 => "plain", 1 => "NVFP4", _ => "MXFP8" };

    /// <summary>Writes the quantised checkpoint and its decoded twin.</summary>
    private static void WriteCheckpoints(string quantised, string decoded)
    {
        Directory.CreateDirectory(quantised);
        Directory.CreateDirectory(decoded);
        File.Copy(Path.Combine(Tiny, "config.json"), Path.Combine(quantised, "config.json"));
        File.Copy(Path.Combine(Tiny, "config.json"), Path.Combine(decoded, "config.json"));
        var q = new List<TensorPayload>();
        var dq = new List<TensorPayload>();
        using var source = ModelDirectory.Open(Tiny);
        foreach (string name in source.TensorNames)
        {
            var info = source.Get(name).Info;
            var bytes = source.ReadRawBytes(name);
            var plain = new TensorPayload { Name = name, Dtype = info.Dtype, Shape = info.Shape, Data = bytes };
            bool expert = name.EndsWith(".mlp.experts.w13_weight", StringComparison.Ordinal) ||
                          name.EndsWith(".mlp.experts.w2_weight", StringComparison.Ordinal);
            int layer = expert ? int.Parse(name.Split('.')[3]) : -1;
            if (!expert || FormOf(layer) == "plain")
            {
                q.Add(plain);
                dq.Add(plain);
                continue;
            }
            Assert.Equal(Dtype.F32, info.Dtype);
            int experts = (int)info.Shape[0], rows = (int)info.Shape[1], cols = (int)info.Shape[2];
            var values = source.DecodeF32(name);
            var decodedValues = new float[values.Length];
            int each = rows * cols;
            if (FormOf(layer) == "NVFP4")
            {
                const float denominator = 6f * 448f;
                var packed = new byte[(long)experts * each / 2];
                var scales = new byte[(long)experts * each / 16];
                var amax = new float[experts];
                for (int e = 0; e < experts; e++)
                {
                    var m = Nvfp4.Quantize(values.AsSpan(e * each, each), rows, cols);
                    m.Elements.CopyTo(packed, e * each / 2);
                    m.Nvfp4ScaleBytes().CopyTo(scales, e * each / 16);
                    amax[e] = m.Global * denominator;
                    float factor = amax[e] / denominator;
                    // transformers: (E2M1 value x block scale) x factor, in f32.
                    for (int i = 0; i < each; i++)
                    {
                        byte b = m.Elements[i / 2];
                        float v = E2M1[(i & 1) == 0 ? b & 0xF : b >> 4];
                        float s = BitPattern.DecodeF8E4M3(scales[(e * each + i) / 16]);
                        decodedValues[e * each + i] = v * s * factor;
                    }
                }
                q.Add(new TensorPayload { Name = name, Dtype = Dtype.U8, Shape = new long[] { experts, rows, cols / 2 }, Data = packed });
                q.Add(new TensorPayload { Name = name + "_scale", Dtype = Dtype.F8_E4M3, Shape = new long[] { experts, rows, cols / 16 }, Data = scales });
                q.Add(new TensorPayload { Name = name + "_amax", Dtype = Dtype.F32, Shape = new long[] { experts }, Data = F32Bytes(amax) });
                q.Add(new TensorPayload { Name = name + "_outer_scale_denominator", Dtype = Dtype.F32, Shape = new long[] { 1 }, Data = F32Bytes(new[] { denominator }) });
            }
            else
            {
                var elements = new byte[(long)experts * each];
                var scales = new byte[(long)experts * each / 32];
                for (int e = 0; e < experts; e++)
                {
                    var m = Mxfp8.Quantize(values.AsSpan(e * each, each), rows, cols);
                    m.Elements.CopyTo(elements, e * each);
                    m.Mxfp8ScaleBytes().CopyTo(scales, e * each / 32);
                    // transformers: E4M3 value x 2^(scale - 127).
                    for (int i = 0; i < each; i++)
                    {
                        decodedValues[e * each + i] = BitPattern.DecodeF8E4M3(elements[e * each + i]) *
                                                      MathF.Pow(2f, scales[(e * each + i) / 32] - 127);
                    }
                }
                q.Add(new TensorPayload { Name = name, Dtype = Dtype.F8_E4M3, Shape = info.Shape, Data = elements });
                q.Add(new TensorPayload { Name = name + "_scale", Dtype = Dtype.U8, Shape = new long[] { experts, rows, cols / 32 }, Data = scales });
            }
            dq.Add(new TensorPayload { Name = name, Dtype = Dtype.F32, Shape = info.Shape, Data = F32Bytes(decodedValues) });
        }
        SafetensorsWriter.Write(Path.Combine(quantised, "model.safetensors"), q);
        SafetensorsWriter.Write(Path.Combine(decoded, "model.safetensors"), dq);
    }

    private static byte[] F32Bytes(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static readonly ArchMapper.EncodeOptions NoVision = new() { IncludeVision = false, IncludeMtp = false };

    [Fact]
    public void Quantised_Experts_Are_Kept_As_Stored_And_Match_Their_Decoded_Values()
    {
        using var dir = new TempDir();
        string quantised = Path.Combine(dir.Path, "q"), decoded = Path.Combine(dir.Path, "dq");
        WriteCheckpoints(quantised, decoded);
        string qc = Path.Combine(dir.Path, "qc"), dqc = Path.Combine(dir.Path, "dqc");
        ModelToContainer.Encode(quantised, qc, "inkling2-q", NoVision);
        ModelToContainer.Encode(decoded, dqc, "inkling2-dq", NoVision);

        using (var container = Vindex3Container.Open(qc))
        using (var store = container.CreateOperandStore())
        {
            const string stack = "target.decoder_stack";
            // NVFP4: U8 [8, 64, 32] in the checkpoint, FP4 [8, 64, 64] in the container.
            Assert.Equal(Dtype.FP4, store.DtypeOf(stack, "1.mlp.experts.w13_weight"));
            Assert.Equal(new long[] { 8, 64, 64 }, store.ShapeOf(stack, "1.mlp.experts.w13_weight"));
            Assert.Equal(Dtype.F8_E4M3, store.DtypeOf(stack, "1.mlp.experts.w13_weight_scale"));
            Assert.Equal(Dtype.F32, store.DtypeOf(stack, "1.mlp.experts.w13_weight_amax"));
            Assert.Equal(Dtype.F8_E4M3, store.DtypeOf(stack, "2.mlp.experts.w13_weight"));
            Assert.Equal(Dtype.U8, store.DtypeOf(stack, "2.mlp.experts.w13_weight_scale"));
            var model = Inkling2Model.Load(container, store);
            Assert.Equal(Enumerable.Range(0, 12).Select(FormOf), model.ExpertForms);
        }

        var (ids, golden) = ArchZooTests.Golden("inkling2_loop");
        var mine = ArchZooTests.RunPrefill(qc, ids);
        var reference = ArchZooTests.RunPrefill(dqc, ids);
        Assert.Equal(0.0, ArchZooTests.WorstDiff(mine, reference));
        // Quantising the experts moves the logits, but only so far.
        _output.WriteLine($"quantised vs transformers' unquantised logits: worst {ArchZooTests.WorstDiff(mine, golden):G4}");
    }

    [Fact]
    public void Quantised_Experts_Export_As_Stored()
    {
        using var dir = new TempDir();
        string quantised = Path.Combine(dir.Path, "q"), decoded = Path.Combine(dir.Path, "dq");
        WriteCheckpoints(quantised, decoded);
        string qc = Path.Combine(dir.Path, "qc"), exported = Path.Combine(dir.Path, "hf"), again = Path.Combine(dir.Path, "qc2");
        ModelToContainer.Encode(quantised, qc, "inkling2-q", NoVision);
        using (var container = Vindex3Container.Open(qc))
        {
            Amql.Cli.ModelExporter.Export(container, exported, patch: null);
        }
        using (var a = ModelDirectory.Open(quantised))
        using (var b = ModelDirectory.Open(exported))
        {
            Assert.Equal(a.TensorNames.Order(StringComparer.Ordinal), b.TensorNames.Order(StringComparer.Ordinal));
            foreach (string name in a.TensorNames)
            {
                var (ia, ib) = (a.Get(name).Info, b.Get(name).Info);
                Assert.True(ia.Dtype == ib.Dtype && ia.Shape.SequenceEqual(ib.Shape),
                    $"{name}: {ia.Dtype.Label()} [{string.Join(",", ia.Shape)}] came back {ib.Dtype.Label()} [{string.Join(",", ib.Shape)}]");
                Assert.Equal(a.ReadRawBytes(name), b.ReadRawBytes(name));
            }
        }
        ModelToContainer.Encode(exported, again, "inkling2-q", NoVision);
        var (ids, _) = ArchZooTests.Golden("inkling2_loop");
        Assert.Equal(0.0, ArchZooTests.WorstDiff(ArchZooTests.RunPrefill(again, ids), ArchZooTests.RunPrefill(qc, ids)));
    }

    // ── the real checkpoint against golden.py ─────────────────────────────────

    /// <summary>
    /// Inkling-2-Mini against transformers (bf16, CPU), when both are to hand:
    /// <c>AMQL_INKLING_CONTAINER</c> (encoded with <c>--no-vision</c>) and
    /// <c>AMQL_INKLING_GOLDEN</c> (golden.py's output: per prompt
    /// <c>{name}.ids.npy</c>, <c>{name}.logits.npy</c>, optionally
    /// <c>{name}.hidden.npy</c>, and <c>summary.json</c> with the greedy
    /// continuation). The reference runs in bf16 and this runtime in f32, so
    /// it is judged on the decisions: the argmax at (nearly) every position,
    /// the top five at the last, and the start of the greedy continuation.
    /// </summary>
    [Fact]
    public void Real_Checkpoint_Matches_Transformers_When_Available()
    {
        if (Environment.GetEnvironmentVariable("AMQL_INKLING_CONTAINER") is not { Length: > 0 } containerDir ||
            Environment.GetEnvironmentVariable("AMQL_INKLING_GOLDEN") is not { Length: > 0 } goldenDir)
        {
            return;
        }
        int greedySteps = int.TryParse(Environment.GetEnvironmentVariable("AMQL_INKLING_GREEDY"), out var g) ? g : 8;
        using var summary = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(goldenDir, "summary.json")));
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var model = Inkling2Model.Load(container, store);
        _output.WriteLine($"experts: {string.Join(" ", model.ExpertForms)}");

        foreach (var prompt in summary.RootElement.EnumerateObject())
        {
            string name = prompt.Name;
            var ids = Npy.Read(Path.Combine(goldenDir, $"{name}.ids.npy")).AsInts();
            var logits = Npy.Read(Path.Combine(goldenDir, $"{name}.logits.npy"));
            int vocab = logits.Shape[^1];
            Assert.Equal(model.VocabSize, vocab);

            model.Reset();
            var visits = new List<float[]>();
            model.OnVisit = (_, h) => visits.Add(h);
            var mine = model.Prefill(ids);
            model.OnVisit = null;

            int agree = 0;
            double worst = 0;
            for (int t = 0; t < ids.Length; t++)
            {
                var reference = logits.Data.AsSpan(t * vocab, vocab);
                agree += ArgMax(mine[t]) == ArgMax(reference) ? 1 : 0;
                for (int v = 0; v < vocab; v++)
                {
                    worst = Math.Max(worst, Math.Abs(mine[t][v] - reference[v]));
                }
            }
            var last = logits.Data.AsSpan((ids.Length - 1) * vocab, vocab).ToArray();
            var top5Mine = TopK(mine[^1], 5);
            var top5Ref = TopK(last, 5);
            int overlap = top5Mine.Intersect(top5Ref).Count();
            _output.WriteLine($"{name}: {ids.Length} tokens, argmax agrees at {agree}/{ids.Length}, worst logit diff {worst:G4}, " +
                              $"last top-5 mine [{string.Join(",", top5Mine)}] ref [{string.Join(",", top5Ref)}]");

            string hiddenPath = Path.Combine(goldenDir, $"{name}.hidden.npy");
            if (File.Exists(hiddenPath))
            {
                // [states, T, d]; the visits' inputs in the order transformers records them.
                var hidden = Npy.Read(hiddenPath);
                int states = hidden.Shape[0], d = hidden.Shape[2];
                int perToken = visits.Count / ids.Length;
                var lines = new StringBuilder();
                for (int s = 0; s < Math.Min(states, perToken); s++)
                {
                    var theirs = hidden.Data.AsSpan((s * ids.Length + ids.Length - 1) * d, d);
                    var ours = visits[(ids.Length - 1) * perToken + s];
                    lines.Append($"{s}:{Cosine(ours, theirs):F4} ");
                }
                _output.WriteLine($"{name}: hidden-state cosine at the last token by visit ({states} recorded, {perToken} visits): {lines}");
            }

            var expected = prompt.Value.GetProperty("greedy").EnumerateArray().Select(e => e.GetInt32()).Take(greedySteps).ToArray();
            var generated = new List<int>();
            var next = mine[^1];
            for (int i = 0; i < expected.Length; i++)
            {
                int token = ArgMax(next);
                generated.Add(token);
                next = model.Step(token);
            }
            _output.WriteLine($"{name}: greedy mine [{string.Join(",", generated)}] ref [{string.Join(",", expected)}]");

            Assert.Equal(ArgMax(last), ArgMax(mine[^1]));
            Assert.True(overlap >= 4, $"{name}: last-position top-5 overlap {overlap}/5");
            Assert.True(agree >= ids.Length * 0.9, $"{name}: argmax agrees at {agree}/{ids.Length}");
            int prefix = Math.Min(4, expected.Length);
            Assert.Equal(expected.Take(prefix), generated.Take(prefix));
        }
    }

    /// <summary>The real checkpoint without a reference (<c>AMQL_INKLING_CONTAINER</c>
    /// alone): a plain prompt whose continuation is not in doubt.</summary>
    [Fact]
    public void Real_Checkpoint_Completes_A_Plain_Prompt_When_Available()
    {
        if (Environment.GetEnvironmentVariable("AMQL_INKLING_CONTAINER") is not { Length: > 0 } containerDir)
        {
            return;
        }
        var tokenizer = HfTokenizer.FromModelDir(containerDir);
        var ids = tokenizer.HasSingleTemplate ? tokenizer.EncodeWithSpecialTokens("The capital of France is") : tokenizer.Encode("The capital of France is").Ids.ToArray();
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var started = System.Diagnostics.Stopwatch.StartNew();
        var model = Inkling2Model.Load(container, store);
        _output.WriteLine($"loaded in {started.Elapsed.TotalSeconds:F0}s; experts: {string.Join(" ", model.ExpertForms)}");
        started.Restart();
        var next = model.Prefill(ids)[^1];
        _output.WriteLine($"prefill of {ids.Length} tokens in {started.Elapsed.TotalSeconds:F1}s; top-5 {string.Join(",", TopK(next, 5))}");
        var generated = new List<int>();
        started.Restart();
        for (int i = 0; i < 6; i++)
        {
            int token = ArgMax(next);
            generated.Add(token);
            next = model.Step(token);
        }
        string text = tokenizer.Decode(generated);
        _output.WriteLine($"ids [{string.Join(",", ids)}] greedy [{string.Join(",", generated)}] '{text}' ({started.Elapsed.TotalSeconds / 6:F1}s a token)");
        Assert.Contains("Paris", text);
    }

    private static int ArgMax(ReadOnlySpan<float> x)
    {
        int best = 0;
        for (int i = 1; i < x.Length; i++)
        {
            if (x[i] > x[best])
            {
                best = i;
            }
        }
        return best;
    }

    private static int[] TopK(float[] x, int k) =>
        Enumerable.Range(0, x.Length).OrderByDescending(i => x[i]).Take(k).ToArray();

    private static double Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double ab = 0, aa = 0, bb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            ab += (double)a[i] * b[i];
            aa += (double)a[i] * a[i];
            bb += (double)b[i] * b[i];
        }
        return ab / Math.Sqrt(aa * bb + 1e-30);
    }

    /// <summary>A NumPy <c>.npy</c> file: little-endian f4, f8 or i8 (and i4),
    /// C order, widened to f32 (ids are small integers).</summary>
    private sealed record Npy(int[] Shape, float[] Data)
    {
        public int[] AsInts() => Data.Select(v => checked((int)v)).ToArray();

        public static Npy Read(string path)
        {
            var bytes = File.ReadAllBytes(path);
            if (bytes.Length < 10 || bytes[0] != 0x93 || Encoding.ASCII.GetString(bytes, 1, 5) != "NUMPY")
            {
                throw new InvalidDataException($"{path} is not a .npy file");
            }
            int major = bytes[6];
            int headerLength = major == 1 ? BitConverter.ToUInt16(bytes, 8) : checked((int)BitConverter.ToUInt32(bytes, 8));
            int start = major == 1 ? 10 : 12;
            string header = Encoding.Latin1.GetString(bytes, start, headerLength);
            string descr = Field(header, "descr").Trim('\'', '"', ' ');
            if (Field(header, "fortran_order").Trim() != "False")
            {
                throw new InvalidDataException($"{path}: Fortran order is not read");
            }
            string shapeText = Field(header, "shape").Trim('(', ')', ' ');
            var shape = shapeText.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(int.Parse).ToArray();
            long count = shape.Aggregate(1L, (n, s) => n * s);
            int offset = start + headerLength;
            var data = new float[count];
            switch (descr)
            {
                case "<f4":
                    Buffer.BlockCopy(bytes, offset, data, 0, checked((int)(count * 4)));
                    break;
                case "<f8":
                    for (long i = 0; i < count; i++) data[i] = (float)BitConverter.ToDouble(bytes, (int)(offset + i * 8));
                    break;
                case "<i8":
                    for (long i = 0; i < count; i++) data[i] = BitConverter.ToInt64(bytes, (int)(offset + i * 8));
                    break;
                case "<i4":
                    for (long i = 0; i < count; i++) data[i] = BitConverter.ToInt32(bytes, (int)(offset + i * 4));
                    break;
                default:
                    throw new InvalidDataException($"{path}: dtype '{descr}' is not read");
            }
            return new Npy(shape, data);
        }

        private static string Field(string header, string key)
        {
            int at = header.IndexOf($"'{key}'", StringComparison.Ordinal);
            if (at < 0)
            {
                throw new InvalidDataException($".npy header has no '{key}': {header}");
            }
            int colon = header.IndexOf(':', at);
            int end = header[(colon + 1)..].TrimStart().StartsWith('(')
                ? header.IndexOf(')', colon) + 1
                : header.IndexOf(',', colon);
            return header[(colon + 1)..end].Trim();
        }
    }
}
