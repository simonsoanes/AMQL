using Amql.Entail;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// <c>carry</c>: a lane that starts each token holding another lane's final
/// value after the previous token. It makes a program a fold over the text, so
/// it reads any length in fixed memory.
/// </summary>
public class EntailCarryTests
{
    private const string Counter = """
        program count_x
        state is_x, is_end, n, last_n
        token "x" means is_x = 1
        token "=" means is_end = 1
        carry last_n from n
        entail n = last_n
        entail n = 1 when is_x
        state o[10]
        for d in 0 to 9
            entail o[d] when is_end, n == d
            emit "{d}" when o[d]
        end
        """;

    private static IEnumerable<string> Chars(string text) => text.Select(c => c.ToString());

    [Theory]
    [InlineData(" =", "0")]
    [InlineData(" xax=", "2")]
    [InlineData(" xxxxbxxxx=", "8")]
    public void A_Carried_Lane_Counts_Across_The_Text(string text, string answer)
    {
        var p = EntailCompiler.Compile(Counter);
        Assert.Equal(answer, EntailInterpreter.Answer(p, Chars(text), limit: 1));
        Assert.Equal(answer, new EntailStream(p).Answer(Chars(text), limit: 1));
    }

    [Fact]
    public void The_Stream_Matches_The_Interpreter_Row_For_Row()
    {
        var p = EntailCompiler.Compile(File.ReadAllText(EntailTests.RepoFile("scripts", "entail", "scan.entail.txt")));
        var rng = new Random(3);
        var text = " ?k" + string.Concat(Enumerable.Range(0, 300).Select(_ => $"{(char)('a' + rng.Next(26))}{rng.Next(10)};")) + "=";
        var rows = EntailInterpreter.Run(p, Chars(text).ToList());
        var stream = new EntailStream(p);
        int t = 0;
        foreach (var c in Chars(text))
        {
            Assert.Equal(rows[t++], stream.Step(c));
        }
        Assert.Equal(1, stream.Window);
    }

    [Theory]
    [InlineData("scan", " ?ka7;k3;b9;k5;=", "5")]
    [InlineData("scan", " ?qa7;k3;=", "0")]
    [InlineData("scan-index", " a7;k3;b9;k5;?k=", "5")]
    [InlineData("scan-index", " a7;k3;b9;k5;?b=", "9")]
    public void The_Scans_Find_The_Best_Value_Under_A_Key(string program, string text, string answer)
    {
        var p = EntailCompiler.Compile(File.ReadAllText(EntailTests.RepoFile("scripts", "entail", program + ".entail.txt")));
        Assert.Equal(answer, new EntailStream(p).Answer(Chars(text), limit: 4));
        Assert.Equal(answer, EntailInterpreter.Answer(p, Chars(text), limit: 4));
    }

    [Fact]
    public void Weights_Refuse_A_Carry()
    {
        var p = EntailCompiler.Compile(Counter);
        var dir = Path.Combine(Path.GetTempPath(), "amql-carry-" + Guid.NewGuid().ToString("N"));
        var e = Assert.Throws<EntailException>(() => EntailWeights.WriteFresh(p, dir));
        Assert.Contains("carry", e.Message);
    }

    [Fact]
    public void A_State_Called_Carry_Is_Still_A_State()
    {
        var p = EntailCompiler.Compile("""
            state carry, a
            token "1" means a = 1
            attend back 1
                carry <- a
            end
            """);
        Assert.Empty(p.Carries);
    }
}
