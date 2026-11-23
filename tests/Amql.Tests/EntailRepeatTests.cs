using Amql.Entail;
using Amql.Hf;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// <c>repeat</c> and <c>forbid repeat</c>: a lane set when the recent tokens repeat an earlier
/// run of the text, and a mask that rules out the token that continued that run, so a text
/// that has begun to copy itself cannot go on copying.
/// </summary>
public class EntailRepeatTests
{
    [Fact]
    public void A_Run_Is_Seen_Only_After_It_Has_Occurred_And_Says_What_Followed()
    {
        var r = new EntailRepeat([3]);
        int[] text = [1, 2, 3, 4, 9, 1, 2, 3, 7, 1, 2, 3];
        var seen = new List<bool>();
        var next = new List<int?>();
        foreach (int t in text)
        {
            r.Push(t);
            seen.Add(r.Seen(3));
            next.Add(r.Continuation(3));
        }
        // 1 2 3 recurs at positions 7 and 11; each time the latest earlier run answers.
        Assert.Equal([false, false, false, false, false, false, false, true, false, false, false, true], seen);
        Assert.Equal(4, next[7]);
        Assert.Equal(7, next[11]);
        // The third run of 1 2 3 has been seen twice before; the second only once.
        Assert.True(r.Seen(3, 2));
        r.Reset();
        foreach (int t in new[] { 1, 2, 3, 9, 1, 2, 3 })
        {
            r.Push(t);
        }
        Assert.True(r.Seen(3, 1));
        Assert.False(r.Seen(3, 2));
        r.Reset();
        r.Push(1);
        Assert.False(r.Seen(3));
    }

    [Fact]
    public void Statements_Parse_And_Weights_Refuse_Them()
    {
        var p = EntailCompiler.Compile("""
            state looping
            repeat looping over 4
            forbid repeat 4 when looping
            """);
        Assert.Equal([(p.Lanes.FindIndex(l => l.Name == "looping"), 4, 1)], p.Repeats);
        Assert.Equal(3, EntailCompiler.Compile("state a\nrepeat a over 8 times 3").Repeats[0].Times);
        Assert.Single(p.RepeatBans);
        Assert.True(p.Recurrent);
        using var dir = new TempDir();
        var e = Assert.Throws<EntailException>(() => EntailWeights.WriteFresh(p, Path.Combine(dir.Path, "x")));
        Assert.Contains("repeat", e.Message);
        Assert.Throws<EntailException>(() => EntailCompiler.Compile("state a\nrepeat a over 0"));
        // A state may still be called repeat.
        EntailCompiler.Compile("state repeat, b\nentail b when repeat");
    }

    [Fact]
    public void The_Interpreter_Sets_The_Lane_On_A_Repeat()
    {
        var p = EntailCompiler.Compile("""
            state looping, flag
            repeat looping over 2
            entail flag when looping
            """);
        var rows = EntailInterpreter.Run(p, "abcab".Select(c => c.ToString()).ToList());
        int flag = p.Lanes.FindIndex(l => l.Name == "flag");
        Assert.Equal([0.0, 0, 0, 0, 1], rows.Select(r => Math.Round(r[flag])));
    }

    [Fact]
    public void An_Adapter_Bans_The_Token_That_Would_Continue_The_Copy()
    {
        using var dir = new TempDir();
        var checkpoint = Path.Combine(dir.Path, "partner.checkpoint");
        EntailWeights.WriteFresh(EntailCompiler.Compile("state a, b\ntoken \"x\" means a = 1\nentail b when a"), checkpoint);
        const string program = """
            state looping
            repeat looping over 3
            forbid repeat 3 when looping
            """;
        EntailSparseAdapter.Write(program, checkpoint, Path.Combine(dir.Path, "loop"));
        using var adapter = EntailSparseAdapter.Open(Path.Combine(dir.Path, "loop"), checkpoint);
        var tokenizer = HfTokenizer.FromModelDir(checkpoint);
        int Id(char c) => tokenizer.EncodeToIds(c.ToString())[0];

        // "abcd" then "abc" again: the copy would go on with "d", so "d" is ruled out.
        EntailVerdict verdict = EntailVerdict.Silence;
        foreach (char c in "abcdabc")
        {
            verdict = adapter.Step(Id(c));
        }
        Assert.Contains(Id('d'), verdict.Banned);
        Assert.Single(verdict.Banned);

        verdict = adapter.Step(Id('x'));
        Assert.True(verdict.Silent);
        adapter.Reset();
        Assert.True(adapter.Prefill("abc".Select(Id).ToArray()).Silent);
    }
}
