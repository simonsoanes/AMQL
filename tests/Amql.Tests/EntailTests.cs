using Amql.Entail;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Tests for <c>amql-cli entail</c>: the language, its meaning, and the claim
/// that the weights it writes compute that meaning with nothing learned.
/// </summary>
[Collection("gpu")]
public class EntailTests
{
    private static string RepoFile(params string[] parts)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "AMQL.slnx")))
        {
            dir = dir.Parent;
        }
        return Path.Combine(new[] { dir!.FullName }.Concat(parts).ToArray());
    }

    private static readonly Lazy<EntailProgram> Mathematics = new(() =>
        EntailCompiler.Compile(File.ReadAllText(RepoFile("scripts", "entail", "mathematics.entail.txt"))));

    private static readonly Lazy<EntailProgram> Physics = new(() =>
        EntailCompiler.Compile(File.ReadAllText(RepoFile("scripts", "entail", "physics.entail.txt"))));

    private static readonly Lazy<EntailProgram> Chemistry = new(() =>
        EntailCompiler.Compile(File.ReadAllText(RepoFile("scripts", "entail", "chemistry.entail.txt"))));

    private static IEnumerable<string> Chars(string text) => text.Select(c => c.ToString());

    [Theory]
    [InlineData(" |1|2| kg| *| |3| m|/s|^|2| =", " 36 N")]
    [InlineData(" |3| km| +| |2|0|0| m| =", " 3200 m")]
    [InlineData(" |1| m| /| |3| s| =", " 0.3333 m/s")]
    [InlineData(" |3| m| +| |2| s| =", " incompatible units")]
    [InlineData(" |3| h| *| |2| km| =", "")]
    public void Physics_Follows_From_The_Algebra_Of_Units(string tokens, string answer) =>
        Assert.Equal(answer, EntailInterpreter.Answer(Physics.Value, tokens.Split('|')));

    [Theory]
    [InlineData(" M|(H|2|O|)=", "18.015 g/mol")]
    [InlineData(" M|(C|a|(O|H|)|2|)=", "74.094 g/mol")]
    [InlineData(" mass| of| Na|Cl| is", " 58.440 g/mol")]
    [InlineData("\n|2|H|2| +| O|2| ->| |2|H|2|O| is", " balanced")]
    [InlineData("\n|H|2| +| O|2| ->| H|2|O| is", " unbalanced")]
    [InlineData(" M|(X|x|2|)=", "")]
    public void Chemistry_Adds_Up_Atoms_And_Conserves_Mass(string tokens, string answer) =>
        Assert.Equal(answer, EntailInterpreter.Answer(Chemistry.Value, tokens.Split('|')));

    private static EntailProgram Load(string name) =>
        EntailCompiler.Compile(File.ReadAllText(RepoFile("scripts", "entail", name + ".entail.txt")));

    private static readonly Lazy<EntailProgram> Dates = new(() => Load("dates"));
    private static readonly Lazy<EntailProgram> Bases = new(() => Load("bases"));
    private static readonly Lazy<EntailProgram> Logic = new(() => Load("logic"));
    private static readonly Lazy<EntailProgram> Json = new(() => Load("json"));

    [Theory]
    [InlineData(" |2|0|2|4|-|0|3|-|1|5| was| a", " Friday")]
    [InlineData(" March| |1|5|,| |2|0|2|4| falls| on| a", " Friday")]
    [InlineData(" |2|9| February| |2|0|0|0| was| a", " Tuesday")]
    [InlineData(" from| |2|0|2|4|-|0|1|-|0|1| to| |2|0|2|4|-|0|3|-|1|5| =", " 74 days")]
    [InlineData(" |2|0|2|4|-|0|3|-|1|5| +| |3|0| days| =", " 2024-04-14")]
    [InlineData(" |1|9|9|9|-|1|2|-|3|1| +| |1| day| =", " 2000-01-01")]
    [InlineData(" |3|0| days| before| |1| March| |2|0|2|4| is", " 2024-01-31")]
    [InlineData(" Is| |1|9|0|0| a| leap| year|?", " No")]
    [InlineData(" Is| |2|0|0|0| a| leap| year|?", " Yes")]
    [InlineData(" February| |2|0|2|4| has", " 29 days")]
    [InlineData(" |2|0|2|3|-|0|2|-|2|9| was| a", "")]
    public void Dates_Follow_From_The_Calendars_Law(string tokens, string answer) =>
        Assert.Equal(answer, EntailInterpreter.Answer(Dates.Value, tokens.Split('|')));

    [Theory]
    [InlineData(" |2|5|5| in| binary| is", " 11111111")]
    [InlineData(" |2|5|5| in| hex| is", " ff")]
    [InlineData(" |0|xff| in| decimal| is", " 255")]
    [InlineData(" |0|xDE|ADB|EE|F| =", " 3735928559")]
    [InlineData(" |0|b|1|0|1|1| =", " 11")]
    [InlineData(" |0|o|7|5|5| in| binary| is", " 111101101")]
    [InlineData(" |1|2|0|3| base| |7| in| decimal| is", " 444")]
    [InlineData(" |1|0|0|0| in| base| |7| is", " 2626")]
    [InlineData(" |4|2|9|4|9|6|7|2|9|5| in| hex| is", " ffffffff")]
    [InlineData(" |4|2|9|4|9|6|7|2|9|6| in| hex| is", "")]
    [InlineData(" |0|b|1|0|2| =", "")]
    public void Bases_Convert_By_Columns_And_Carries(string tokens, string answer) =>
        Assert.Equal(answer, EntailInterpreter.Answer(Bases.Value, tokens.Split('|')));

    [Theory]
    [InlineData("\n|true| and| not| false| is", " true")]
    [InlineData("\n| (|true| or| false|)| and| false| =", " false")]
    [InlineData("\n| p| or| not| p| is", " a tautology")]
    [InlineData("\n| p| and| not| p| is", " a contradiction")]
    [InlineData("\n| p| ->| q| is", " contingent")]
    [InlineData("\n| not| (|p| and| q|)| <->| (|not| p| or| not| q|)| is", " a tautology")]
    [InlineData(" Truth| table| of| p| xor| q|:", " 0110")]
    [InlineData(" Truth| table| of| (|p| or| q|):", " 0111")]
    [InlineData("\n| p| xor| q| or| r| is", "")]
    [InlineData("\n| and| or| true| is", "")]
    public void Logic_Evaluates_Every_Assignment(string tokens, string answer) =>
        Assert.Equal(answer, EntailInterpreter.Answer(Logic.Value, tokens.Split('|')));

    [Theory]
    [InlineData("\n|JSON|:| {\"|a|\":| |1|}|\n|Valid|:", " yes")]
    [InlineData("\n|JSON|:| [|1|,| |2|}|\n|Valid|:", " no")]
    [InlineData("\n|JSON|:| {\"|a|\":| \"|}|\",| \"|b|\":| [|]}|\n|Valid|:", " yes")]
    [InlineData("\n|JSON|:| {\"|a|\":| |1|,|}|\n|Valid|:", " no")]
    [InlineData("\n|JSON|:| [|0| .|5|]|\n|Valid|:", " no")]
    [InlineData("\n|JSON|:| [\"|tab|\\|there|\"]|\n|Valid|:", " yes")]
    [InlineData(" nothing| here|\n|Valid|:", "")]
    public void Json_Is_Read_Against_Its_Grammar(string tokens, string answer) =>
        Assert.Equal(answer, EntailInterpreter.Answer(Json.Value, tokens.Split('|')));

    [Fact]
    public void A_Mask_Needs_A_Host()
    {
        var p = EntailCompiler.Compile("state a, b\ntoken \"x\" means a = 1\nentail b when a\nonly \"x\", \"y\" when b\nforbid \"z\" when b");
        Assert.Equal(2, p.Masks.Count);
        Assert.True(p.Masks[0].Allow);
        Assert.False(p.Masks[1].Allow);
        var dir = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Assert.Throws<EntailException>(() => EntailWeights.WriteFresh(p, dir));
    }

    [Fact]
    public void Strings_Can_Name_Any_Token()
    {
        var p = EntailCompiler.Compile("state a, b\ntoken \"\\en\", \"\\t\", \"\\r\", \"\\\" means a = 1\nentail b when a");
        Assert.Contains("\\n", p.Tokens.Keys);
        Assert.Contains("\t", p.Tokens.Keys);
        Assert.Contains("\r", p.Tokens.Keys);
        Assert.Contains("\\", p.Tokens.Keys);
    }

    private static string Say(string expression) =>
        EntailInterpreter.Answer(Mathematics.Value, Chars(expression));

    [Theory]
    [InlineData(" 12+34=", "46")]
    [InlineData(" 999+1=", "1000")]
    [InlineData(" 99999999+1=", "100000000")]
    [InlineData(" 5-7=", "-2")]
    [InlineData(" 7-7=", "0")]
    [InlineData(" 1000-1=", "999")]
    [InlineData(" 0*5=", "0")]
    [InlineData(" 99*99=", "9801")]
    [InlineData(" 999999*999999=", "999998000001")]
    [InlineData(" 84/4=", "21")]
    [InlineData(" 7/2=", "3.5")]
    [InlineData(" 1/3=", "0.33")]
    [InlineData(" 999999/9999=", "100")]   // 100.0099..., truncated
    [InlineData(" 99/8=", "12.37")]
    [InlineData(" 12 + 34 =", "46")]
    [InlineData("Total: $12 * 3 = ", "36")]
    [InlineData(" 1,200+3=", "1203")]
    [InlineData(" 26,264 - 6,316 = ", "19948")]
    [InlineData(" 12,345,678+1=", "12345679")]
    [InlineData(" 1,000*1,000=", "1000000")]
    public void Mathematics_Means_What_It_Should(string expression, string answer) =>
        Assert.Equal(answer, Say(expression));

    /// <summary>Anything that is not exactly an expression entails nothing,
    /// which is what leaves a host free to carry on.</summary>
    [Theory]
    [InlineData(" 10/0=")]             // undefined
    [InlineData(" 1.5+2=")]            // a decimal operand
    [InlineData(" 3*2.5=")]
    [InlineData(" 1,23+4=")]           // not grouped in threes
    [InlineData(" 12,3456+1=")]
    [InlineData(" 1,234,56+1=")]
    [InlineData(" 2+3+4=")]            // a chain
    [InlineData(" (2+3)*4=")]
    [InlineData(" -5+3=")]             // a signed operand
    [InlineData(" 1234567*2=")]        // past the multiplication limit
    [InlineData(" 123456789+1=")]      // past the addition limit
    [InlineData(" x=3")]
    [InlineData(" a = b")]
    [InlineData("12+34=")]             // no token before it: see the spec
    public void Mathematics_Says_Nothing_Otherwise(string expression) =>
        Assert.Equal(string.Empty, Say(expression));

    [Fact]
    public void Mathematics_Holds_On_Random_Operands()
    {
        var random = new Random(20261104);
        for (int i = 0; i < 200; i++)
        {
            long a = random.NextInt64(0, 100_000_000), b = random.NextInt64(0, 100_000_000);
            long c = random.NextInt64(0, 1_000_000), d = random.NextInt64(0, 1_000_000);
            Assert.Equal((a + b).ToString(), Say($" {a}+{b}="));
            Assert.Equal((a - b).ToString(), Say($" {a}-{b}="));
            Assert.Equal((c * d).ToString(), Say($" {c}*{d}="));
        }
    }

    [Fact]
    public void A_State_Cannot_Be_Written_After_It_Is_Read()
    {
        const string source = """
            state x, y, z
            token "a" means x = 1
            entail y when x >= 1
            entail z when y >= 1
            entail y when x >= 1
            """;
        var e = Assert.Throws<EntailException>(() => EntailCompiler.Compile(source));
        Assert.Contains("after it has been read", e.Message);
    }

    [Theory]
    [InlineData("state x\nentail y when x >= 1", "not a declared state")]
    [InlineData("state x[2]\nentail x[2] when x[0] >= 1", "outside")]
    [InlineData("state x, y\nentail y = x * x when x >= 1", "not linear")]
    [InlineData("for i in 0 to 1\nstate x", "missing 'end'")]
    public void Mistakes_Are_Reported(string source, string message)
    {
        var e = Assert.Throws<EntailException>(() => EntailCompiler.Compile(source));
        Assert.Contains(message, e.Message);
    }

    [Fact]
    public void Layers_Follow_From_The_Dataflow()
    {
        const string source = """
            state x, y, z, w
            token "a" means x = 1
            attend back 1
                y <- x
            end
            entail z when y >= 1
            entail w when z >= 1
            emit "a" when w
            """;
        var p = EntailCompiler.Compile(source);
        var head = Assert.Single(p.Ops.OfType<Head>());
        Assert.Equal(0, head.Layer);
        // Attention of layer 0 feeds the FFN of layer 0, which feeds layer 1.
        Assert.Equal(new[] { 0, 0, 1, 1 }, p.Ops.OfType<Neuron>().Select(n => n.Layer));
        Assert.Equal(2, p.Layers);
    }

    [Fact]
    public void Every_Head_Is_A_Lookup_Over_The_Whole_Context() =>
        Assert.True(EntailWeights.AttentionGap(Mathematics.Value, 32768) > 100);

    /// <summary>The full claim, end to end: statements become a checkpoint,
    /// the checkpoint becomes a container, and AMQL's ordinary runtime
    /// produces the answers. Nothing in between is fitted.</summary>
    [Fact]
    public void A_Compiled_Empty_Model_Computes_The_Answers()
    {
        using var dir = new TempDir();
        var checkpoint = Path.Combine(dir.Path, "checkpoint");
        var containerPath = Path.Combine(dir.Path, "container");
        EntailWeights.WriteFresh(Mathematics.Value, checkpoint);
        ModelToContainer.Encode(checkpoint, containerPath, "entail-mathematics");

        var tokenizer = HfTokenizer.FromModelDir(checkpoint);
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var session = new DecodeSession(plan, store, null, WeightWorkingSet.ResidentF32);

        string Generate(string prompt) => GenerateOn(session, prompt);

        string GenerateOn(DecodeSession on, string prompt)
        {
            on.Reset();
            var logits = on.Prefill(tokenizer.EncodeToIds(prompt).ToArray());
            var reply = new List<int>();
            for (int i = 0; i < 16; i++)
            {
                var row = logits.Row(0);
                int best = 0;
                for (int t = 1; t < row.Length; t++)
                {
                    if (row[t] > row[best])
                    {
                        best = t;
                    }
                }
                if (best <= 1)
                {
                    break;   // <none> or <eos>
                }
                reply.Add(best);
                logits = on.Step(best);
            }
            return tokenizer.DecodeText(reply);
        }

        Assert.Equal("100000000", Generate(" 99999999+1="));
        Assert.Equal("-2", Generate(" 5-7="));
        Assert.Equal("80779853376", Generate(" 123456*654321="));
        Assert.Equal("3.5", Generate(" 7/2="));
        Assert.Equal("1424.17", Generate(" 967017/679="));
        Assert.Equal("12.37", Generate(" 99/8="));
        Assert.Equal(string.Empty, Generate(" 1.5+2="));

        // Sessions sharing those weights keep their own state, on their own threads.
        // On the GPU they would corrupt each other silently, so they must refuse.
        if (CudaShim.Enabled)
        {
            Assert.Throws<InvalidOperationException>(() => session.Sibling());
            return;
        }
        string[] sums = [" 99999999+1=", " 5-7=", " 123456*654321=", " 7/2=", " 967017/679=", " 99/8="];
        string[] expected = ["100000000", "-2", "80779853376", "3.5", "1424.17", "12.37"];
        var together = new string[sums.Length];
        Parallel.For(0, sums.Length, i => together[i] = GenerateOn(session.Sibling(), sums[i]));
        Assert.Equal(expected, together);
    }

    [Fact]
    public void A_Session_That_Has_Not_Run_Cannot_Share_Its_Weights()
    {
        using var dir = new TempDir();
        var checkpoint = Path.Combine(dir.Path, "checkpoint");
        var containerPath = Path.Combine(dir.Path, "container");
        EntailWeights.WriteFresh(Mathematics.Value, checkpoint);
        ModelToContainer.Encode(checkpoint, containerPath, "entail-mathematics");
        using var container = Vindex3Container.Open(containerPath);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);

        var session = new DecodeSession(plan, store, null, WeightWorkingSet.ResidentF32);
        Assert.Throws<InvalidOperationException>(() => session.Sibling());
    }
}
