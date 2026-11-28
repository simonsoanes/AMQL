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
    internal static void WriteCheckpoints(string quantised, string decoded)
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

    // ── the real checkpoint against transformers ──────────────────────────────


    /// <summary>AMQL_TEST_CONTAINER, when it holds an Inkling 2 model.</summary>
    private static string? InklingContainer() => RealCheckpoint.Container(c => Inkling2Model.Serves(c, "target"));

    /// <summary>
    /// Inkling-2-Mini against transformers, when both are to hand:
    /// <c>AMQL_TEST_CONTAINER</c> (encoded with <c>--no-vision</c>) and
    /// <c>AMQL_TEST_REFERENCE</c>, the output of
    /// <c>scripts/reference/stream_reference.py</c> (see its README):
    /// <c>summary.json</c> (per prompt the ids, the top 20 logits, log-sum-exp
    /// and entropy at every position, and a greedy continuation),
    /// <c>visits.json</c> (per prompt, layer visit in run order and position:
    /// the output's RMS, first coordinates and a fingerprint on 16 fixed unit
    /// directions) and <c>{name}.last.f32</c> (the last position's logits).
    /// The directions come from torch's generator, so they are read from
    /// <c>AMQL_TEST_DIRECTIONS</c> (16 x hidden little-endian f32) when
    /// given; without them the fingerprints are not compared. Both the managed
    /// runtime and the GPU path (where it runs) are compared and every number is
    /// printed. The reference may run in bf16 and these in f32, so the
    /// assertions are on decisions: the top token at every position but one in
    /// ten, the last position's top five (four of five), and the greedy
    /// continuation's first four tokens.
    /// </summary>
    [Fact]
    public void Real_Checkpoint_Matches_Transformers_When_Available()
    {
        if (InklingContainer() is not { } containerDir || RealCheckpoint.Setting("REFERENCE") is not { } referenceDir)
        {
            return;
        }
        using var summary = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(referenceDir, "summary.json")));
        string visitsPath = Path.Combine(referenceDir, "visits.json");
        using var visits = File.Exists(visitsPath) ? JsonDocument.Parse(File.ReadAllBytes(visitsPath)) : null;
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var model = Inkling2Model.Load(container, store);
        int hidden = container.Graph!.Component("target").HiddenSize;
        float[][]? directions = null;
        if (RealCheckpoint.Setting("DIRECTIONS") is { } dirPath)
        {
            var raw = File.ReadAllBytes(dirPath);
            var flat = new float[raw.Length / 4];
            Buffer.BlockCopy(raw, 0, flat, 0, raw.Length);
            directions = Enumerable.Range(0, flat.Length / hidden).Select(i => flat.AsSpan(i * hidden, hidden).ToArray()).ToArray();
        }
        using var gpu = Inkling2Gpu.Available ? Inkling2Gpu.TryCreate(container, store, out _) : null;
        if (gpu is not null)
        {
            gpu.KvFormat = "f32";
            gpu.ExactPrefill = true;
        }
        _output.WriteLine($"reference dtype {Str(summary.RootElement, "dtype")}; experts: {string.Join(" ", model.ExpertForms.Distinct())}; " +
                          $"gpu {(gpu is null ? "absent" : "compared")}; fingerprints {(directions is null ? "not compared" : "compared")}");

        foreach (var prompt in summary.RootElement.GetProperty("prompts").EnumerateArray())
        {
            string name = prompt.GetProperty("name").GetString()!;
            var ids = prompt.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var positions = prompt.GetProperty("positions").EnumerateArray().ToList();
            var expected = prompt.GetProperty("greedy").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            Assert.Equal(ids.Length, positions.Count);
            float[]? last = null;
            string lastPath = Path.Combine(referenceDir, $"{name}.last.f32");
            if (File.Exists(lastPath))
            {
                var bytes = File.ReadAllBytes(lastPath);
                last = new float[bytes.Length / 4];
                Buffer.BlockCopy(bytes, 0, last, 0, bytes.Length);
            }
            var records = visits is not null && visits.RootElement.TryGetProperty(name, out var r) ? r.EnumerateArray().ToList() : null;

            void Judge(string who, float[][] logits, float[][][]? outputs, Func<int, float[]> step)
            {
                var lines = new StringBuilder();
                int top1 = 0;
                for (int t = 0; t < ids.Length; t++)
                {
                    var theirs = positions[t].GetProperty("top_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                    var theirLogits = positions[t].GetProperty("top_logits").EnumerateArray().Select(e => e.GetDouble()).ToArray();
                    var ours = TopK(logits[t], 5);
                    bool agree = ours[0] == theirs[0];
                    top1 += agree ? 1 : 0;
                    double worstTop = Enumerable.Range(0, theirs.Length).Max(i => Math.Abs(logits[t][theirs[i]] - theirLogits[i]));
                    double lse = Math.Abs(LogSumExp(logits[t]) - positions[t].GetProperty("logsumexp").GetDouble());
                    lines.AppendLine($"  {t,3}: argmax {(agree ? "agrees" : $"differs ({ours[0]} vs {theirs[0]})")}, top-5 overlap " +
                                     $"{ours.Intersect(theirs.Take(5)).Count()}/5, max |diff| on the top-20 {worstTop:G3}, log-sum-exp {lse:G3}");
                }
                string full = "";
                if (last is not null)
                {
                    int n = Math.Min(last.Length, logits[^1].Length);
                    double worst = 0, ab = 0, aa = 0, bb = 0;
                    for (int v = 0; v < n; v++)
                    {
                        double a = logits[^1][v], b = last[v];
                        worst = Math.Max(worst, Math.Abs(a - b));
                        ab += a * b; aa += a * a; bb += b * b;
                    }
                    full = $"; last position, all {n} logits: max |diff| {worst:G4}, cosine {ab / Math.Sqrt(aa * bb):F6}";
                }
                var generated = new List<int>();
                var next = logits[^1];
                for (int i = 0; i < expected.Length; i++)
                {
                    generated.Add(ArgMax(next));
                    next = step(generated[^1]);
                }
                int same = generated.Zip(expected).TakeWhile(p => p.First == p.Second).Count();
                _output.WriteLine($"{name} ({ids.Length} tokens), {who}: top-1 agrees at {top1}/{ids.Length}{full}; " +
                                  $"greedy agrees for {same}/{expected.Length} tokens [{string.Join(",", generated)}] vs [{string.Join(",", expected)}]");
                _output.WriteLine(lines.ToString().TrimEnd());

                if (outputs is not null && records is not null)
                {
                    var visitLines = new StringBuilder();
                    for (int v = 0; v < Math.Min(records.Count, outputs.Length); v++)
                    {
                        var rms = records[v].GetProperty("rms").EnumerateArray().Select(e => e.GetDouble()).ToArray();
                        var prints = records[v].GetProperty("fingerprint").EnumerateArray()
                            .Select(p => p.EnumerateArray().Select(e => e.GetDouble()).ToArray()).ToArray();
                        double ratioLo = double.MaxValue, ratioHi = 0, cosLo = 1;
                        for (int t = 0; t < ids.Length; t++)
                        {
                            var h = outputs[v][t];
                            double ours = Math.Sqrt(h.Average(x => (double)x * x));
                            double ratio = ours / Math.Max(rms[t], 1e-12);
                            ratioLo = Math.Min(ratioLo, ratio);
                            ratioHi = Math.Max(ratioHi, ratio);
                            if (directions is not null)
                            {
                                var mine = directions.Select(dir => Enumerable.Range(0, hidden).Sum(i => (double)dir[i] * h[i])).ToArray();
                                double ab = 0, aa = 0, bb = 0;
                                for (int i = 0; i < mine.Length; i++)
                                {
                                    ab += mine[i] * prints[t][i]; aa += mine[i] * mine[i]; bb += prints[t][i] * prints[t][i];
                                }
                                cosLo = Math.Min(cosLo, ab / Math.Sqrt(aa * bb + 1e-30));
                            }
                        }
                        visitLines.Append($"  visit {v,2} ({records[v].GetProperty("layer").GetString()}): RMS ratio {ratioLo:F4}..{ratioHi:F4}" +
                                          (directions is null ? "" : $", fingerprint cosine >= {cosLo:F5}") + Environment.NewLine);
                    }
                    _output.WriteLine($"{name}, {who}, per visit over every position:\n{visitLines.ToString().TrimEnd()}");
                }

                var lastTop = positions[^1].GetProperty("top_ids").EnumerateArray().Select(e => e.GetInt32()).Take(5).ToArray();
                Assert.Equal(lastTop[0], TopK(logits[^1], 1)[0]);
                Assert.True(TopK(logits[^1], 5).Intersect(lastTop).Count() >= 4, $"{name} {who}: last-position top-5");
                Assert.True(top1 >= ids.Length * 0.9, $"{name} {who}: top-1 agrees at {top1}/{ids.Length}");
                Assert.True(same >= Math.Min(4, expected.Length), $"{name} {who}: greedy agrees for {same} tokens");
            }

            model.Reset();
            var outputs = new Dictionary<int, List<float[]>>();
            model.OnVisitOutput = (v, h) => (outputs.TryGetValue(v, out var list) ? list : outputs[v] = new()).Add(h);
            var mine = model.Prefill(ids);
            model.OnVisitOutput = null;
            var managedVisits = outputs.OrderBy(kv => kv.Key).Select(kv => kv.Value.ToArray()).ToArray();
            Judge("managed", mine, managedVisits, model.Step);

            if (gpu is not null)
            {
                using var seq = gpu.NewSequence(ids.Length + expected.Length + 16);
                var logits = seq.PrefillAll(ids);
                var gpuVisits = ids.Length <= 256 ? seq.VisitOutputs(managedVisits.Length, ids.Length, hidden) : null;
                Judge("gpu", logits, gpuVisits, seq.Step);
            }
        }
    }

    private static string? Str(JsonElement e, string key) =>
        e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static double LogSumExp(float[] x)
    {
        double max = x.Max();
        return max + Math.Log(x.Sum(v => Math.Exp(v - max)));
    }

    /// <summary>The real checkpoint without a reference (<c>AMQL_TEST_CONTAINER</c>
    /// alone): a plain prompt whose continuation is not in doubt.</summary>
    [Fact]
    public void Real_Checkpoint_Completes_A_Plain_Prompt_When_Available()
    {
        if (InklingContainer() is not { } containerDir)
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

    /// <summary>
    /// The real checkpoint on the GPU path against the managed runtime
    /// (<c>AMQL_TEST_CONTAINER</c>, a CUDA device and amql_cuda): a few
    /// prompts, the top five at every position and a greedy continuation. f32
    /// rows and f32 prompts, so the two differ only in summation order;
    /// <c>AMQL_TEST_STEPS</c> sets the greedy length (12). Then the default
    /// BF16 prompts, held to the same top token at 95% of positions.
    /// </summary>
    [Fact]
    public void Real_Checkpoint_Gpu_Matches_Managed_When_Available()
    {
        if (InklingContainer() is not { } containerDir ||
            !Inkling2Gpu.Available)
        {
            return;
        }
        int steps = int.TryParse(RealCheckpoint.Setting("STEPS"), out var n) ? n : 12;
        var tokenizer = HfTokenizer.FromModelDir(containerDir);
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        using var gpu = Inkling2Gpu.TryCreate(container, store, out var reason);
        Assert.True(gpu is not null, reason);
        _output.WriteLine($"GPU weights {gpu.WeightBytes / (double)(1 << 30):F2} GiB, uploaded in {clock.Elapsed.TotalSeconds:F0}s");
        var managed = Inkling2Model.Load(container, store);
        gpu.KvFormat = "f32";
        gpu.ExactPrefill = true;

        string[] prompts =
        {
            "The capital of France is",
            "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
            "Water boils at a temperature of",
            // Past a prompt chunk (256 tokens), so the GPU's chunks meet inside it.
            string.Concat(Enumerable.Range(1, 24).Select(i =>
                $"Step {i}: the river carried {i * 7} boats past the old mill, and the miller counted each one. ")) + "In total the miller counted",
        };
        int positions = 0, sameTop5 = 0, sameOrder = 0, fastTop1 = 0, fastTop5 = 0;
        double worst = 0, worstFast = 0;
        foreach (string text in prompts)
        {
            var ids = tokenizer.Encode(text).Ids.ToArray();
            managed.Reset();
            var reference = managed.Prefill(ids);
            using var seq = gpu.NewSequence(256);
            var mine = seq.PrefillAll(ids);
            for (int t = 0; t < ids.Length; t++)
            {
                var a = TopK(mine[t], 5);
                var b = TopK(reference[t], 5);
                positions++;
                sameTop5 += a.ToHashSet().SetEquals(b) ? 1 : 0;
                sameOrder += a.SequenceEqual(b) ? 1 : 0;
                for (int v = 0; v < reference[t].Length; v++)
                {
                    worst = Math.Max(worst, Math.Abs(mine[t][v] - reference[t][v]));
                }
            }
            // The default prompt path: the engine's BF16 tensor cores.
            gpu.ExactPrefill = false;
            using (var tensor = gpu.NewSequence(256))
            {
                var fast = tensor.PrefillAll(ids);
                for (int t = 0; t < ids.Length; t++)
                {
                    fastTop1 += ArgMax(fast[t]) == ArgMax(reference[t]) ? 1 : 0;
                    fastTop5 += TopK(fast[t], 5).ToHashSet().SetEquals(TopK(reference[t], 5)) ? 1 : 0;
                    for (int v = 0; v < reference[t].Length; v++)
                    {
                        worstFast = Math.Max(worstFast, Math.Abs(fast[t][v] - reference[t][v]));
                    }
                }
            }
            gpu.ExactPrefill = true;

            var g = new List<int>();
            var m = new List<int>();
            float[] gl = mine[^1], ml = reference[^1];
            for (int i = 0; i < steps; i++)
            {
                g.Add(ArgMax(gl));
                m.Add(ArgMax(ml));
                gl = seq.Step(g[^1]);
                ml = managed.Step(m[^1]);
            }
            _output.WriteLine($"'{text.Replace("\n", "\\n")}' ({ids.Length} tokens): gpu '{tokenizer.Decode(g).Replace("\n", "\\n")}' " +
                              $"managed '{tokenizer.Decode(m).Replace("\n", "\\n")}'");
            Assert.Equal(m, g);
        }
        _output.WriteLine($"top-5 at {positions} positions: same set {sameTop5}, same order {sameOrder}; worst logit diff {worst:G4}");
        _output.WriteLine($"BF16 prompts: same top token at {fastTop1}/{positions}, same top-5 set at {fastTop5}; worst logit diff {worstFast:G4}");
        Assert.Equal(positions, sameTop5);
        Assert.True(fastTop1 >= positions * 0.95, $"BF16 prompts: the top token agrees at {fastTop1}/{positions}");
    }

    /// <summary>The GPU path's speeds on the real checkpoint
    /// (<c>AMQL_TEST_CONTAINER</c> and <c>AMQL_TEST_SPEED=1</c>, on a quiet
    /// machine): a prompt of each length fed in chunks, then 128 greedy steps
    /// after it, per KV format; and the device memory a sequence takes.</summary>
    [Fact]
    public void Real_Checkpoint_Gpu_Speed_When_Available()
    {
        if (InklingContainer() is not { } containerDir ||
            RealCheckpoint.Setting("SPEED") != "1" || !Inkling2Gpu.Available)
        {
            return;
        }
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        using var gpu = Inkling2Gpu.TryCreate(container, store, out var reason);
        Assert.True(gpu is not null, reason);
        var (free0, total) = Inkling2Gpu.DeviceMemory();
        _output.WriteLine($"weights {gpu.WeightBytes / (double)(1 << 30):F2} GiB; device {free0 / (double)(1 << 30):F2} of {total / (double)(1 << 30):F2} GiB free after upload");
        var clock = new System.Diagnostics.Stopwatch();
        foreach (var (format, lengths) in new[] { ("fp16", new[] { 512, 4096, 16384, 65536 }), ("nvfp4", new[] { 512, 4096, 16384, 65536 }), ("f32", new[] { 512, 4096 }) })
        {
            gpu.KvFormat = format;
            foreach (int promptLength in lengths)
            {
                using var seq = gpu.NewSequence(promptLength + 256);
                var prompt = Enumerable.Range(0, promptLength).Select(i => 1000 + (i * 7919) % 50000).ToArray();
                seq.Step(prompt[0]);
                clock.Restart();
                seq.Prefill(prompt[1..]);
                double prefill = (promptLength - 1) / clock.Elapsed.TotalSeconds;
                int token = 1000;
                seq.Step(token);
                clock.Restart();
                for (int i = 0; i < 128; i++)
                {
                    token = ArgMax(seq.Step(token));
                }
                double decode = 128 / clock.Elapsed.TotalSeconds;
                var (free, _) = Inkling2Gpu.DeviceMemory();
                _output.WriteLine($"{format} rows, {promptLength}-token prompt: prompt {prefill:F0} tok/s, then decode {decode:F1} tok/s; " +
                                  $"sequence {seq.DeviceBytes / (double)(1 << 20):F0} MiB, device used {(total - free) / (double)(1 << 30):F2} GiB");
            }
        }
        foreach (int context in new[] { 4096, 32768, 131072, 262144, 1048576 })
        {
            _output.WriteLine($"KV and histories at {context}: f32 {gpu.SequenceBytes(context, "f32") / (double)(1 << 30):F2} GiB, " +
                              $"fp16 {gpu.SequenceBytes(context, "fp16") / (double)(1 << 30):F2} GiB, nvfp4 {gpu.SequenceBytes(context, "nvfp4") / (double)(1 << 30):F2} GiB");
        }
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
}
