using Amql.Hf;
using Amql.Inference;
using Amql.Merge;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Speculative decoding on the device engine: a verify's rows give exactly the
/// logits one-token steps give (row invariance), keeping only some of its rows
/// leaves the sequence exactly where those steps would, and with an MTP
/// drafter a greedy reply drafted and verified is the same reply token for
/// token. Skips without a CUDA device.
/// </summary>
[Collection("gpu")]
public class SpeculativeDecodingTests
{
    private static string Zoo => Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo");

    private static string Encode(TempDir dir, string family, bool mtp = false)
    {
        string containerDir = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(Path.Combine(Zoo, family), containerDir, family,
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = mtp });
        return containerDir;
    }

    /// <summary>A drafter that reads the model's caches (Gemma 4's assistant) gives what transformers
    /// gives, and a reply drafted with it is the reply without it.</summary>
    [Theory]
    [InlineData(null)]
    [InlineData(Fp4Format.Nvfp4)]
    public void A_Drafter_Over_The_Models_Caches_Matches_Transformers(Fp4Format? format)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        string kv = GpuEngine.RequestedKvFormat;
        using var restore = new Restore(() => GpuEngine.RequestedKvFormat = kv);
        GpuEngine.RequestedKvFormat = format is null ? "f32" : "fp16";   // random routers make much of a cache's rounding
        using var dir = new TempDir();
        string containerDir = Encode(dir, "gemma4-moe");
        string assistant = Path.Combine(AppContext.BaseDirectory, "fixtures", "gemma4-assistant");
        var report = Attach.Drafter(containerDir, assistant, format);
        Assert.Equal(["drafter.stack"], report.Objects);
        Attach.Drafter(containerDir, assistant, format);   // again: replaced
        using var golden = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(Path.Combine(assistant, "golden.json")));
        var root = golden.RootElement;
        var ids = root.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        using var host = new DecodeSession(plan, store);
        host.Prefill(new[] { ids[0] });
        var engine = host.DeviceEngine()!;
        Assert.True(engine.HasDrafter);
        engine.ExactPrefill = true;
        var seq = engine.NewSequence();
        var last = engine.Prefill(seq, ids);
        int token = root.GetProperty("token").GetInt32();
        Assert.Equal(token, Sampler.ArgMax(last));
        var expected = root.GetProperty("logits").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetSingle()).ToArray()).ToArray();
        var (drafts, logits) = engine.DraftLogits(seq, token, expected.Length);
        if (format is null)
        {
            double scale = expected.Max(r => r.Max(Math.Abs));
            for (int i = 0; i < expected.Length; i++)
            {
                double apart = expected[i].Zip(logits[i]).Max(p => Math.Abs(p.First - p.Second)) / scale;
                Assert.True(apart < 2e-3, $"draft {i}: {apart:P3}");
            }
            Assert.Equal(root.GetProperty("drafts").EnumerateArray().Select(e => e.GetInt32()).ToArray(), drafts);
        }
        engine.ExactPrefill = false;

        // Drafted and plain greedy replies, past the window and through slides of it.
        var plain = engine.NewSequence();
        var reply = new List<int> { Sampler.ArgMax(engine.Prefill(plain, ids)) };
        while (reply.Count < 60)
        {
            reply.Add(Sampler.ArgMax(engine.Step(new[] { plain }, new[] { reply[^1] })[0]));
        }
        engine.Reset(seq);
        var drafted = new List<int> { Sampler.ArgMax(engine.Prefill(seq, ids)) };
        while (drafted.Count < 60)
        {
            var proposal = engine.Draft(seq, drafted[^1], 3);
            Assert.Equal(3, proposal.Length);
            engine.Verify(seq, [drafted[^1], .. proposal]);
            int rows = 0;
            do
            {
                drafted.Add(Sampler.ArgMax(engine.HostLogits(rows++)));
            }
            while (rows <= proposal.Length && proposal[rows - 1] == drafted[^1]);
            engine.Accept(seq, rows);
        }
        Assert.Equal(reply, drafted.Take(60));

        // A plain step passes the drafter by; the next verify brings it back.
        engine.Step(new[] { seq }, new[] { 5 });
        Assert.Empty(engine.Draft(seq, 6, 3));
        engine.Verify(seq, [6]);
        engine.Accept(seq, 1);
        Assert.Equal(3, engine.Draft(seq, 7, 3).Length);
    }

    private sealed class Restore(Action undo) : IDisposable
    {
        public void Dispose() => undo();
    }

    [Theory]
    [InlineData("qwen3")]
    [InlineData("qwen3_5-gdn")]
    public void Verify_Rows_Equal_Steps_And_Accept_Rewinds_Exactly(string family)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(Encode(dir, family));
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var (ids, _) = ArchZooTests.Golden(family);
        var prompt = Enumerable.Range(0, 23).Select(i => ids[i % ids.Length]).ToArray();

        using var host = new DecodeSession(plan, store);
        host.Prefill(new[] { ids[0] });
        var engine = host.DeviceEngine()!;
        engine.RowInvariant = true;

        // Plain greedy steps: the tokens and the logits after each.
        var plain = engine.NewSequence();
        var logits = new List<float[]> { engine.Prefill(plain, prompt) };
        var tokens = new List<int>();
        for (int i = 0; i < 16; i++)
        {
            tokens.Add(Sampler.ArgMax(logits[^1]));
            logits.Add(engine.Step(new[] { plain }, new[] { tokens[^1] })[0]);
        }

        // Verifies of 1 to 8 rows whose "drafts" are the greedy tokens with one
        // wrong one planted: each row equals the step, and accepting up to the
        // wrong one carries on exactly as the steps did.
        var spec = engine.NewSequence();
        Assert.Equal(logits[0], engine.Prefill(spec, prompt));
        int at = 0;
        foreach (int width in new[] { 1, 3, 8, 2, 5 })
        {
            int n = Math.Min(width, tokens.Count - at);
            if (n < 1)
            {
                break;
            }
            var rows = tokens.Skip(at).Take(n).ToArray();
            int wrongAt = n > 2 ? n - 1 : n;   // a planted miss in the last row
            if (wrongAt < n)
            {
                rows[wrongAt] = (rows[wrongAt] + 1) % plan.Output!.VocabSize;
            }
            engine.Verify(spec, rows);
            int keep = Math.Min(n, wrongAt);
            for (int r = 0; r < keep; r++)
            {
                Assert.Equal(logits[at + r + 1], engine.HostLogits(r).ToArray());
            }
            engine.Accept(spec, keep);
            at += keep;
            Assert.Equal(prompt.Length + at, spec.Position);
        }
        // After the last accept, a plain step from the sequence agrees too.
        var next = engine.Step(new[] { spec }, new[] { tokens[at] })[0];
        Assert.Equal(logits[at + 1], next);
    }

    [Fact]
    public void Drafted_Greedy_Reply_Equals_Plain_Greedy_Reply()
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        const string family = "qwen3_5-gdn";
        using var dir = new TempDir();
        string withMtp = Path.Combine(dir.Path, "mtp");
        GenerateMtp.Transform(Encode(dir, family, mtp: true), withMtp, Array.Empty<int>());
        using var container = Vindex3Container.Open(withMtp);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var (ids, _) = ArchZooTests.Golden(family);
        var prompt = Enumerable.Range(0, 30).Select(i => ids[(i * 5 + 1) % ids.Length]).ToArray();

        using var host = new DecodeSession(plan, store);
        host.Prefill(new[] { ids[0] });
        var engine = host.DeviceEngine()!;
        Assert.True(engine.HasDrafter);
        Assert.True(engine.RowInvariant);
        const int length = 40;

        var plain = engine.NewSequence();
        var expected = new List<int>();
        var l = engine.Prefill(plain, prompt);
        while (expected.Count < length)
        {
            expected.Add(Sampler.ArgMax(l));
            l = engine.Step(new[] { plain }, new[] { expected[^1] })[0];
        }

        foreach (int k in new[] { 1, 3, 7 })
        {
            var seq = engine.NewSequence();
            var got = new List<int>();
            int token = Sampler.ArgMax(engine.Prefill(seq, prompt));
            int drafted = 0;
            while (got.Count < length)
            {
                var drafts = engine.Draft(seq, token, k);
                Assert.Equal(k, drafts.Length);
                drafted += drafts.Length;
                var rows = new[] { token }.Concat(drafts).ToArray();
                engine.Verify(seq, rows);
                int keep = 0;
                while (true)
                {
                    got.Add(rows[keep]);
                    int choice = Sampler.ArgMax(engine.HostLogits(keep).ToArray());
                    keep++;
                    if (keep == rows.Length || choice != rows[keep])
                    {
                        token = choice;
                        break;
                    }
                }
                engine.Accept(seq, keep);
            }
            Assert.Equal(expected, got.Take(length));
            Assert.True(drafted > 0);
            engine.Release(seq);
        }
    }

    [Fact]
    public void The_Drafter_Catches_Up_As_If_It_Had_Read_Every_Token()
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        const string family = "qwen3_5-gdn";
        using var dir = new TempDir();
        string withMtp = Path.Combine(dir.Path, "mtp");
        GenerateMtp.Transform(Encode(dir, family, mtp: true), withMtp, Array.Empty<int>());
        using var container = Vindex3Container.Open(withMtp);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var (ids, _) = ArchZooTests.Golden(family);
        var tokens = Enumerable.Range(0, 24).Select(i => ids[(i * 3 + 2) % ids.Length]).ToArray();

        using var host = new DecodeSession(plan, store);
        host.Prefill(new[] { ids[0] });
        var engine = host.DeviceEngine()!;

        // One sequence takes the tokens as a prompt and verifies with some rows
        // rejected; the other takes them in verify-sized prefills. Both see the
        // same pairs, so they draft the same tokens.
        var a = engine.NewSequence();
        engine.Prefill(a, tokens.AsSpan(0, 6));
        int pos = 6;
        while (pos + 4 <= tokens.Length - 1)
        {
            var rows = tokens.AsSpan(pos, 4).ToArray();
            rows[3] = (rows[3] + 7) % plan.Output!.VocabSize;   // rejected
            engine.Verify(a, rows);
            engine.Accept(a, 3);
            pos += 3;
        }
        var b = engine.NewSequence();
        engine.Prefill(b, tokens.AsSpan(0, 6));
        for (int at = 6; at < pos; at += 3)
        {
            engine.Prefill(b, tokens.AsSpan(at, 3));
        }
        Assert.Equal(a.Position, b.Position);
        Assert.Equal(engine.Draft(b, tokens[pos], 5), engine.Draft(a, tokens[pos], 5));
    }
}
