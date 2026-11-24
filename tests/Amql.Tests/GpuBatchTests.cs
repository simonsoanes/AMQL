using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Several sequences on one device engine, stepped together: each sequence's
/// logits equal what the engine gives that sequence alone, as sequences join
/// and leave the batch, for softmax attention (with a sliding window and a
/// post-norm), a Gated DeltaNet hybrid and a parallel-residual block. Skips
/// without a CUDA device.
/// </summary>
[Collection("gpu")]
public class GpuBatchTests
{
    private static string Zoo => Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo");

    [Theory]
    [InlineData("qwen3")]
    [InlineData("qwen3_5-gdn")]
    [InlineData("gemma2")]
    [InlineData("cohere")]
    public void Batched_Steps_Equal_Each_Sequence_Alone(string family)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        using var dir = new TempDir();
        string containerDir = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(Path.Combine(Zoo, family), containerDir, family,
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var (ids, _) = ArchZooTests.Golden(family);
        int[] Prompt(int length, int shift) => Enumerable.Range(0, length).Select(i => ids[(i + shift) % ids.Length]).ToArray();
        var prompts = new[] { Prompt(5, 0), Prompt(17, 3), Prompt(40, 7), Prompt(9, 11) };
        const int steps = 12;

        // Each sequence alone, greedy: the tokens it takes and the logits at every step.
        var alone = prompts.Select(p =>
        {
            using var session = new DecodeSession(plan, store);
            var logits = new List<float[]> { session.Prefill(p).FirstRow().ToArray() };
            var tokens = new List<int>();
            for (int i = 0; i < steps; i++)
            {
                tokens.Add(Sampler.ArgMax(logits[^1]));
                logits.Add(session.Step(tokens[^1]).FirstRow().ToArray());
            }
            Assert.Equal("gpu", session.Backend);
            return (Tokens: tokens, Logits: logits);
        }).ToList();

        // The same sequences on one engine: 0 and 1 start together, 2 joins at
        // step 3, 3 joins at step 5, 1 leaves at step 8 (and its state is freed).
        using var host = new DecodeSession(plan, store);
        host.Prefill(new[] { ids[0] });
        var engine = host.DeviceEngine()!;
        var seqs = new GpuSequence?[prompts.Length];
        var done = new int[prompts.Length];
        var joins = new[] { 0, 0, 3, 5 };
        for (int tick = 0; done.Any(d => d < steps); tick++)
        {
            for (int s = 0; s < prompts.Length; s++)
            {
                if (seqs[s] is null && done[s] == 0 && tick == joins[s])
                {
                    seqs[s] = engine.NewSequence();
                    AssertClose(alone[s].Logits[0], engine.Prefill(seqs[s]!, prompts[s]), $"{family} seq {s} prefill");
                }
            }
            if (tick == 8 && seqs[1] is { } leaving)
            {
                engine.Release(leaving);
                seqs[1] = null;
                done[1] = steps;
            }
            var active = Enumerable.Range(0, prompts.Length).Where(s => seqs[s] is not null && done[s] < steps).ToList();
            if (active.Count == 0)
            {
                continue;
            }
            var logits = engine.Step(active.Select(s => seqs[s]!).ToList(), active.Select(s => alone[s].Tokens[done[s]]).ToArray());
            for (int k = 0; k < active.Count; k++)
            {
                int s = active[k];
                done[s]++;
                AssertClose(alone[s].Logits[done[s]], logits[k], $"{family} seq {s} step {done[s]} in a batch of {active.Count}");
            }
        }
    }

    /// <summary>A sequence rewound to its mark picks up exactly where it was
    /// there: the rest of the prompt gives the same logits, and so do the steps
    /// after it, though the sequence went past the mark in between. A prompt that
    /// parts from the first after the mark is as if prefilled from the start in
    /// the same two pieces, and two pieces agree with one to f32 rounding.</summary>
    [Theory]
    [InlineData("qwen3")]
    [InlineData("qwen3_5-gdn")]
    public void A_Rewound_Sequence_Picks_Up_At_Its_Mark(string family)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        using var dir = new TempDir();
        string containerDir = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(Path.Combine(Zoo, family), containerDir, family,
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var (ids, _) = ArchZooTests.Golden(family);
        // Past a 512-row chunk and the chunked recurrence's 256-row threshold.
        var prompt = Enumerable.Range(0, 700).Select(i => ids[i % ids.Length]).ToArray();
        var other = Enumerable.Range(0, 90).Select(i => ids[(i * 7 + 3) % ids.Length]).ToArray();
        const int mark = 600;

        using var host = new DecodeSession(plan, store);
        host.Prefill(new[] { ids[0] });
        var engine = host.DeviceEngine()!;
        engine.ExactPrefill = true;
        float[][] Steps(GpuSequence seq, float[] from, int n)
        {
            var all = new List<float[]>();
            for (int i = 0; i < n; i++)
            {
                from = engine.Step(new[] { seq }, new[] { Sampler.ArgMax(from) })[0];
                all.Add(from);
            }
            return all.ToArray();
        }

        var seq = engine.NewSequence();
        engine.Prefill(seq, prompt.AsSpan(0, mark));
        engine.Mark(seq);
        Assert.Equal(mark, seq.MarkedAt);
        var first = engine.Prefill(seq, prompt.AsSpan(mark));
        var firstSteps = Steps(seq, first, 6);

        engine.Rewind(seq);
        Assert.Equal(mark, seq.Position);
        var again = engine.Prefill(seq, prompt.AsSpan(mark));
        Assert.Equal(first, again);
        var againSteps = Steps(seq, again, 6);
        for (int i = 0; i < firstSteps.Length; i++)
        {
            Assert.Equal(firstSteps[i], againSteps[i]);
        }

        engine.Rewind(seq);
        var parted = engine.Prefill(seq, other);
        var fresh = engine.NewSequence();
        engine.Prefill(fresh, prompt.AsSpan(0, mark));
        Assert.Equal(parted, engine.Prefill(fresh, other));

        engine.Reset(fresh);
        Assert.Equal(-1, fresh.MarkedAt);
        AssertClose(first, engine.Prefill(fresh, prompt), $"{family} two pieces against one");
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]));
        }
        Assert.True(worst < 1e-4, $"{what}: logits differ by up to {worst}");
    }
}
