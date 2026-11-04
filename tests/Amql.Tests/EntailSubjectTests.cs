using Amql.Entail;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Tests for the extract-and-optimise cycle: rules work from recorded evidence,
/// so none of this needs a model.
/// </summary>
public class EntailSubjectTests
{
    /// <summary>Rows for a two-digit relation, the way extract records them.</summary>
    private static List<EvidenceRow> Rows(string name, char op, Func<int, int, long> answer, int count,
        Func<int, int, long>? wrong = null, double margin = 0.9)
    {
        var random = new Random(7);
        var rows = new List<EvidenceRow>();
        var seen = new HashSet<(int, int)>();
        while (rows.Count < count)
        {
            int a = random.Next(10, 100), b = random.Next(10, 100);
            if (!seen.Add((a, b)))
            {
                continue;
            }
            var context = new List<string> { "\n" };
            context.AddRange($"{a}{op}{b}=".Select(c => c.ToString()));
            long value = wrong?.Invoke(a, b) ?? answer(a, b);
            var output = value.ToString().Select(c => c.ToString()).ToList();
            rows.Add(new EvidenceRow(name, new[] { a.ToString(), b.ToString() }, context, output, margin, rows.Count % 3 == 0));
        }
        return rows;
    }

    private static RelationPlan Relation(IEnumerable<EvidenceRow> rows) =>
        SubjectPlan.FromEvidence("test", rows.ToList()).Relations.Single();

    [Theory]
    [InlineData('+', 0, 1, 1, 0)]
    [InlineData('-', 0, 1, -1, 0)]
    [InlineData('*', 0, 0, 0, 1)]
    public void A_Law_Is_Derived_Not_Searched(char op, long c0, long c1, long c2, long c3)
    {
        Func<int, int, long> f = op switch
        {
            '+' => (a, b) => a + b,
            '-' => (a, b) => a - b,
            _ => (a, b) => (long)a * b,
        };
        var law = EntailOptimiser.DeriveLaw(Relation(Rows("r", op, f, 90)), 0.5);
        Assert.NotNull(law);
        Assert.Equal((c0, c1, c2, c3), (law!.C0, law.C1, law.C2, law.C3));
    }

    [Fact]
    public void A_Law_Survives_A_Model_That_Is_Sometimes_Wrong()
    {
        // One answer in five is off, as a model's multiplication is.
        var rows = Rows("mul", '*', (a, b) => (long)a * b, 120, (a, b) => (a + b) % 5 == 0 ? (long)a * b + 100 : (long)a * b);
        var law = EntailOptimiser.DeriveLaw(Relation(rows), 0.5);
        Assert.NotNull(law);
        Assert.Equal((0, 0, 0, 1), (law!.C0, law.C1, law.C2, law.C3));
        Assert.True(law.Inliers < law.Rows);
    }

    [Fact]
    public void A_Mere_Trend_Is_Not_A_Law()
    {
        var rows = Rows("div", '/', (a, b) => a / b + a % 7, 90);
        Assert.Null(EntailOptimiser.DeriveLaw(Relation(rows), 0.5));
    }

    [Fact]
    public void Undecided_Rows_Do_Not_Shape_A_Law()
    {
        var rows = Rows("add", '+', (a, b) => a + b, 90)
            .Concat(Rows("add", '+', (a, b) => a + b, 40, (a, b) => 7, margin: 0.05)).ToList();
        var law = EntailOptimiser.DeriveLaw(Relation(rows), 0.5);
        Assert.Equal((0, 1, 1, 0), (law!.C0, law.C1, law.C2, law.C3));
    }

    [Fact]
    public void An_Extracted_Table_Answers_What_It_Saw_And_Nothing_Else()
    {
        var rows = Rows("add", '+', (a, b) => a + b, 60);
        var plan = SubjectPlan.FromEvidence("test", rows);
        var standing = EntailOptimiser.Measure(EntailOptimiser.Generate(plan), rows, 0.5);
        Assert.Equal(standing.FitCount, standing.FitAgree);
        Assert.Equal(0, standing.HoldAgree);
    }

    [Theory]
    [InlineData('+')]
    [InlineData('-')]
    [InlineData('*')]
    public void A_Law_Answers_What_The_Table_Never_Saw(char op)
    {
        Func<int, int, long> f = op switch
        {
            '+' => (a, b) => a + b,
            '-' => (a, b) => a - b,
            _ => (a, b) => (long)a * b,
        };
        var rows = Rows("r", op, f, 60);
        var plan = SubjectPlan.FromEvidence("test", rows);
        plan.Relations[0].Law = EntailOptimiser.DeriveLaw(plan.Relations[0], 0.5);
        var standing = EntailOptimiser.Measure(EntailOptimiser.Generate(plan), rows, 0.5);
        Assert.Equal(standing.HoldCount, standing.HoldAgree);
        Assert.Equal(standing.FitCount, standing.FitAgree);
    }

    [Fact]
    public void A_Program_With_A_Law_Overrules_The_Models_Mistakes()
    {
        var rows = Rows("mul", '*', (a, b) => (long)a * b, 60, (a, b) => (a + b) % 7 == 0 ? (long)a * b + 10 : (long)a * b);
        var plan = SubjectPlan.FromEvidence("test", rows);
        plan.Relations[0].Law = EntailOptimiser.DeriveLaw(plan.Relations[0], 0.5);
        var standing = EntailOptimiser.Measure(EntailOptimiser.Generate(plan), rows, 0.5);
        Assert.NotEmpty(standing.Overruled);
        Assert.All(standing.Overruled, o => Assert.Equal(
            long.Parse(o.Row.Values[0]) * long.Parse(o.Row.Values[1]), long.Parse(o.Said)));
    }

    [Fact]
    public void A_Slot_That_Never_Changes_The_Answer_Is_Found()
    {
        var rows = new List<EvidenceRow>();
        foreach (var keyword in new[] { "if", "while" })
        {
            foreach (var name in new[] { " a", " b", " c" })
            {
                rows.Add(new EvidenceRow("block", new[] { keyword, name }, new[] { "\n", keyword, name }, new[] { ":" }, 0.9, false));
            }
        }
        var relation = Relation(rows);
        Assert.Equal(new[] { 0, 1 }, EntailOptimiser.IrrelevantSlots(relation, 0.5));
    }

    [Fact]
    public void A_Slot_That_Decides_The_Answer_Stays()
    {
        var rows = new[]
        {
            new EvidenceRow("not", new[] { "True" }, new[] { "\n", "not", " True", "#" }, new[] { " False" }, 0.9, false),
            new EvidenceRow("not", new[] { "False" }, new[] { "\n", "not", " False", "#" }, new[] { " True" }, 0.9, false),
        };
        Assert.Empty(EntailOptimiser.IrrelevantSlots(Relation(rows), 0.5));
    }

    [Fact]
    public void Dropping_A_Slot_Too_Far_Is_Caught_By_Ordinary_Text()
    {
        var rows = new List<EvidenceRow>();
        foreach (var keyword in new[] { "if", "while" })
        {
            foreach (var name in new[] { " a", " b", " c" })
            {
                rows.Add(new EvidenceRow("block", new[] { keyword, name }, new[] { "\n", keyword, name }, new[] { ":" }, 0.9, false));
            }
        }
        var plan = SubjectPlan.FromEvidence("test", rows);
        var background = new[] { new BackgroundLine(new[] { "x", "\n", "the", " a", " dog" }, new[] { "y", "the", " b", " dog", " ran" }) };
        var before = EntailOptimiser.Measure(EntailOptimiser.Generate(plan), rows, 0.5, background);
        plan.Relations[0].Dropped.Add(0);
        plan.Relations[0].Dropped.Add(1);
        var after = EntailOptimiser.Measure(EntailOptimiser.Generate(plan), rows, 0.5, background);
        Assert.Equal(0, before.FalseFires);
        Assert.True(after.FalseFires > 0);
        Assert.False(EntailOptimiser.Gate(before, after).Accepted);
    }

    [Theory]
    [InlineData("x\n12+34=", "46")]
    [InlineData("x\n26-50=", "-24")]
    [InlineData("x\n51*47=", "2397")]    // the model said 2407
    [InlineData("x\n99*99=", "9801")]
    [InlineData("x12+34=", "")]           // not the extracted question
    public void The_Refined_Mathematics_Program_Answers_Exactly(string text, string answer)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (!File.Exists(Path.Combine(dir!.FullName, "AMQL.slnx")))
        {
            dir = dir.Parent;
        }
        var p = EntailCompiler.Compile(File.ReadAllText(
            Path.Combine(dir.FullName, "scripts", "entail", "subjects", "mathematics.refined.entail.txt")));
        Assert.Equal(answer, EntailInterpreter.Answer(p, text.Select(c => c.ToString())));
    }

    private static Standing Standing(int hold, int fit, int neurons) =>
        new(neurons, 0, 1, 1, fit, 100, hold, 100, 0, Array.Empty<(EvidenceRow, string)>());

    [Theory]
    [InlineData(90, 90, 100, 90, 90, 90, true)]    // same agreement, cheaper
    [InlineData(90, 90, 100, 95, 80, 500, true)]   // held-out agreement is what counts
    [InlineData(90, 90, 100, 89, 100, 10, false)]  // cheaper, but it lost a row
    [InlineData(90, 90, 100, 90, 89, 10, false)]   // lost a fitted row
    [InlineData(90, 90, 100, 90, 90, 100, false)]  // no better
    public void The_Gate_Weighs_Agreement_Before_Cost(int h0, int f0, int c0, int h1, int f1, int c1, bool accepted) =>
        Assert.Equal(accepted, EntailOptimiser.Gate(Standing(h0, f0, c0), Standing(h1, f1, c1)).Accepted);

    /// <summary>The shape `entail decompile` writes: rules into a lane, votes out of it.</summary>
    private const string Voting = """
        program voting
        state sym
        token "a" means sym = 1
        token "b" means sym = 2
        token "c" means sym = 3
        token "\q" means sym = 4
        token "\{" means sym = 5
        state cat, dog
        entail cat = 2 when sym == 1
        entail dog = 1 when sym == 2
        vote "x" by 2*cat
        vote "y" by 1*cat
        vote "y" by 3*dog
        """;

    private static string? Said(EntailProgram p, string token) =>
        EntailInterpreter.Output(p, EntailInterpreter.Run(p, new[] { token }).Last());

    [Theory]
    [InlineData("a", "x")]   // x scores 4, y 2
    [InlineData("b", "y")]   // votes for one token add up across lanes
    [InlineData("c", null)]  // nothing fired, so nothing is said
    public void The_Highest_Positive_Vote_Is_Said(string token, string? said) =>
        Assert.Equal(said, Said(EntailCompiler.Compile(Voting), token));

    [Fact]
    public void A_Program_With_Votes_Does_Not_Compile_To_Weights()
    {
        var p = EntailCompiler.Compile(Voting);
        Assert.Contains("vote", Assert.Throws<EntailException>(() => EntailWeights.WriteFresh(p, Path.GetTempPath())).Message);
    }

    [Theory]
    [InlineData("\"", "\\q")]
    [InlineData("{", "\\{")]
    [InlineData("\n", "\\n")]
    public void A_Quoted_Token_Reads_Back_As_Itself(string token, string escaped)
    {
        Assert.Equal($"\"{escaped}\"", EntailOptimiser.Quoted(token));
        var p = EntailCompiler.Compile($"program q\nstate s\ntoken {EntailOptimiser.Quoted(token)} means s = 1\nstate t\nentail t when s == 1\n");
        Assert.True(p.Tokens.ContainsKey(token));
    }

    [Theory]
    [InlineData("a\\nb")]    // a literal backslash-n cannot be told from the escape
    [InlineData("\uFFFD")]   // half a character
    [InlineData("")]
    public void A_Token_That_Cannot_Be_Written_Is_Refused(string token) =>
        Assert.False(EntailOptimiser.CanQuote(token));
}
