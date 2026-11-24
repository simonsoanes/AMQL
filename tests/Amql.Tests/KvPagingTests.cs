using Amql.Hf;
using Amql.Inference;
using Amql.Merge;
using Amql.Server;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Paging prompt state out of GPU memory and back: a sequence restored from
/// what <see cref="GpuEngine.SaveMark"/> wrote carries on bit for bit as the
/// sequence that was saved (softmax attention, a Gated DeltaNet hybrid, and
/// with an MTP drafter); and the disk store keeps each model's states under
/// its fingerprint, oldest out first past the budget, gone after the expiry.
/// The engine tests skip without a CUDA device.
/// </summary>
[Collection("gpu")]
public class KvPagingTests
{
    private static string Zoo => Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo");

    private static string Encode(TempDir dir, string family, bool mtp)
    {
        string containerDir = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(Path.Combine(Zoo, family), containerDir, family,
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = mtp });
        if (!mtp)
        {
            return containerDir;
        }
        string withMtp = Path.Combine(dir.Path, "mtp");
        GenerateMtp.Transform(containerDir, withMtp, Array.Empty<int>());
        return withMtp;
    }

    [Theory]
    [InlineData("qwen3", false)]
    [InlineData("qwen3_5-gdn", false)]
    [InlineData("qwen3_5-gdn", true)]
    public void A_Restored_Mark_Carries_On_As_The_Saved_Sequence(string family, bool mtp)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(Encode(dir, family, mtp));
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var (ids, _) = ArchZooTests.Golden(family);
        // Past a 512-row chunk and the chunked recurrence's 256-row threshold.
        var prompt = Enumerable.Range(0, 700).Select(i => ids[i % ids.Length]).ToArray();
        const int mark = 600;

        using var host = new DecodeSession(plan, store);
        host.Prefill(new[] { ids[0] });
        var engine = host.DeviceEngine()!;
        engine.ExactPrefill = true;
        Assert.Equal(mtp, engine.HasDrafter);

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

        var saved = engine.NewSequence();
        engine.Prefill(saved, prompt.AsSpan(0, mark));
        engine.Mark(saved);
        var first = engine.Prefill(saved, prompt.AsSpan(mark));
        var firstSteps = Steps(saved, first, 6);
        int[]? firstDrafts = mtp ? engine.Draft(saved, Sampler.ArgMax(firstSteps[^1]), 3) : null;

        var state = new ChunkedBuffer();
        engine.SaveMark(saved, state);
        Assert.Equal(engine.MarkBytes(saved), state.Length);
        engine.Release(saved);

        state.Position = 0;
        var restored = engine.RestoreMark(state);
        Assert.Equal(mark, restored.Position);
        Assert.Equal(mark, restored.MarkedAt);
        var again = engine.Prefill(restored, prompt.AsSpan(mark));
        Assert.Equal(first, again);
        var againSteps = Steps(restored, again, 6);
        for (int i = 0; i < firstSteps.Length; i++)
        {
            Assert.Equal(firstSteps[i], againSteps[i]);
        }
        if (firstDrafts is not null)
        {
            Assert.Equal(firstDrafts, engine.Draft(restored, Sampler.ArgMax(againSteps[^1]), 3));
        }

        // The restored sequence keeps its mark: it rewinds there as the saved one would.
        engine.Rewind(restored);
        Assert.Equal(first, engine.Prefill(restored, prompt.AsSpan(mark)));
        Assert.True(engine.DeviceBytes(restored) > engine.MarkBytes(restored) / 2);
        engine.Release(restored);
    }

    private static ChunkedBuffer Bytes(int n, byte fill)
    {
        var b = new ChunkedBuffer();
        b.Write(Enumerable.Repeat(fill, n).ToArray());
        b.Position = 0;
        return b;
    }

    private static byte[] All(ChunkedBuffer b)
    {
        var ms = new MemoryStream();
        b.CopyTo(ms);
        return ms.ToArray();
    }

    private static void Settle(KvPageStore store) => store.Dispose();

    [Fact]
    public void The_Store_Gives_Back_The_Longest_State_The_Prompt_Starts_With()
    {
        using var dir = new TempDir();
        var store = new KvPageStore(dir.Path, "fp", 1L << 30, 0);
        Assert.True(store.Put([1, 2, 3], Bytes(1000, 7), DateTime.UtcNow));
        Assert.True(store.Put([1, 2, 3, 4, 5], Bytes(2000, 9), DateTime.UtcNow));
        Settle(store);
        Assert.Equal(2, Directory.GetFiles(dir.Path, "*.kvpage").Length);

        // A restart finds them; a prompt that only reaches the shorter takes that one.
        store = new KvPageStore(dir.Path, "fp", 1L << 30, 0);
        Assert.Equal(2, store.Count);
        Assert.Null(store.Take([1, 2, 3]));            // must go past the saved tokens
        Assert.Null(store.Take([9, 2, 3, 4, 5, 6]));
        var hit = store.Take([1, 2, 3, 4, 5, 6]);
        Assert.Equal(5, hit!.Value.Mark);
        Assert.Equal(Enumerable.Repeat((byte)9, 2000), All(hit.Value.State));
        var other = store.Take([1, 2, 3, 4, 9]);
        Assert.Equal(3, other!.Value.Mark);
        Assert.Equal(0, store.Count);
        Settle(store);
        Assert.Empty(Directory.GetFiles(dir.Path, "*.kvpage"));
    }

    [Fact]
    public void A_State_Being_Written_Can_Be_Taken()
    {
        using var dir = new TempDir();
        using var store = new KvPageStore(dir.Path, "fp", 1L << 30, 0);
        store.Put([1, 2, 3], Bytes(40 << 20, 5), DateTime.UtcNow);
        var hit = store.Take([1, 2, 3, 4]);
        Assert.Equal(40 << 20, hit!.Value.State.Length);
        Assert.All(All(hit.Value.State), b => Assert.Equal(5, b));
    }

    [Fact]
    public void Past_The_Budget_The_Oldest_Goes_And_One_Too_Big_Is_Refused()
    {
        using var dir = new TempDir();
        var store = new KvPageStore(dir.Path, "fp", 5000, 0);
        var t = DateTime.UtcNow;
        Assert.True(store.Put([1], Bytes(2000, 1), t.AddSeconds(-30)));
        Assert.True(store.Put([2], Bytes(2000, 2), t.AddSeconds(-20)));
        Assert.True(store.Put([3], Bytes(2000, 3), t.AddSeconds(-10)));
        Assert.False(store.Put([4], Bytes(6000, 4), t));
        Assert.Equal(2, store.Count);
        Assert.Null(store.Take([1, 0]));
        Assert.NotNull(store.Take([2, 0]));
        Assert.NotNull(store.Take([3, 0]));
        Settle(store);
        Assert.Empty(Directory.GetFiles(dir.Path, "*.kvpage"));
    }

    [Fact]
    public void Expired_States_And_Other_Models_States_Are_Deleted()
    {
        using var dir = new TempDir();
        var store = new KvPageStore(dir.Path, "fp", 1L << 30, 60);
        store.Put([1, 2], Bytes(100, 1), DateTime.UtcNow.AddSeconds(-120));
        store.Put([3, 4], Bytes(100, 1), DateTime.UtcNow);
        Settle(store);
        Assert.Equal(2, Directory.GetFiles(dir.Path, "*.kvpage").Length);

        store = new KvPageStore(dir.Path, "fp", 1L << 30, 60);   // the expired one goes at load
        Assert.Equal(1, store.Count);
        store.Sweep(DateTime.UtcNow.AddSeconds(120));
        Assert.Equal(0, store.Count);
        Settle(store);

        store = new KvPageStore(dir.Path, "fp", 1L << 30, 0);
        store.Put([5, 6], Bytes(100, 1), DateTime.UtcNow);
        Settle(store);
        store = new KvPageStore(dir.Path, "another model", 1L << 30, 0);
        Assert.Equal(0, store.Count);
        Settle(store);
        Assert.Empty(Directory.GetFiles(dir.Path, "*.kvpage"));
    }

    [Fact]
    public void Each_Model_Pages_To_Its_Own_Folder()
    {
        var saved = (KvPaging.DiskRoot, KvPaging.DiskEnabled);
        try
        {
            KvPaging.DiskRoot = null;
            KvPaging.DiskEnabled = false;
            Assert.Null(KvPaging.For("a", "/models/a", null).DiskDir);
            Assert.Equal(Path.GetFullPath("/elsewhere/a"), KvPaging.For("a", "/models/a", "/elsewhere/a").DiskDir);
            KvPaging.DiskEnabled = true;
            Assert.Equal(Path.GetFullPath("/models/a/kv-cache"), KvPaging.For("a", "/models/a", null).DiskDir);
            KvPaging.DiskRoot = "/kv";
            Assert.Equal(Path.GetFullPath("/kv/b"), KvPaging.For("b", "/models/b", null).DiskDir);
        }
        finally
        {
            (KvPaging.DiskRoot, KvPaging.DiskEnabled) = saved;
        }
    }
}
