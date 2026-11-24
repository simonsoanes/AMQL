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
