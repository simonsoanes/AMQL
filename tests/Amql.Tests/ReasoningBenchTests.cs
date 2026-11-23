using Amql.Inference.Reasoning;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// <c>entail bench</c>'s two halves that need no model: the generated problems, which must be
/// the same for a seed and right by construction, and the reading of a reply, which must count
/// only a committed answer.
/// </summary>
public class ReasoningBenchTests
{
    [Fact]
    public void A_Seed_Always_Gives_The_Same_Problems()
    {
        var a = ProblemSuite.Generate(7, 5);
        var b = ProblemSuite.Generate(7, 5);
        Assert.Equal(a, b);
        Assert.Equal(5 * ProblemSuite.Families.Count, a.Count);
        Assert.NotEqual(a, ProblemSuite.Generate(8, 5));
    }

    [Fact]
    public void Families_Can_Be_Chosen_And_Unknown_Ones_Are_Refused()
    {
        var p = ProblemSuite.Generate(1, 3, ["plan", "order"]);
        Assert.Equal(["plan", "plan", "plan", "order", "order", "order"], p.Select(x => x.Family));
        Assert.Throws<ArgumentException>(() => ProblemSuite.Generate(1, 1, ["nope"]));
    }

    [Fact]
    public void Answers_Are_Right_By_Construction()
    {
        foreach (var p in ProblemSuite.Generate(3, 40, ["count", "counterfactual", "order"]))
        {
            var n = System.Text.RegularExpressions.Regex.Matches(p.Prompt, @"\d+").Select(m => int.Parse(m.Value)).ToArray();
            long expected = p.Family switch
            {
                "count" => Enumerable.Range(1, n[1]).Count(i => i % n[2] == 0 && i % n[3] != 0),
                // days, 1, days, days, 1, today, ahead
                "counterfactual" => (n[5] - 1 + n[6]) % n[0] + 1,
                _ => p.Answer,
            };
            Assert.Equal(expected, p.Answer);
        }
    }

    private static ReplyAnalysis Read(string[] pieces, string expected, bool thinking = true, bool budget = false) =>
        ReplyAnalysis.Read(pieces.Select(x => x.GetHashCode()).ToArray(), pieces, expected, thinking, budget);

    [Fact]
    public void Only_A_Committed_Answer_Counts()
    {
        var right = Read(["so ", "it is ", "42", ".", "</think>", "The answer is ", "42", "."], "42");
        Assert.Equal(ReplyOutcome.Correct, right.Outcome);
        Assert.Equal(4, right.ThinkingTokens);
        Assert.Equal(2, right.FoundAt);

        // The right number, written only while thinking, is not an answer.
        var open = Read(["so ", "it is ", "42", "."], "42", budget: true);
        Assert.Equal(ReplyOutcome.Unfinished, open.Outcome);
        Assert.True(open.FoundWhileThinking);
        Assert.Null(open.Answer);

        var wrong = Read(["</think>", "It is ", "41"], "42");
        Assert.Equal(ReplyOutcome.Wrong, wrong.Outcome);
        Assert.Equal(ReplyOutcome.NoNumber, Read(["</think>", "No idea."], "42").Outcome);
    }

    [Fact]
    public void Boxed_Answers_Win_And_Numbers_Are_Normalised()
    {
        Assert.Equal("1940", ReplyAnalysis.Extract(@"So \boxed{1,940}, not 3."));
        Assert.Equal("12", ReplyAnalysis.Extract("x = 12.0"));
        Assert.Equal("-3", ReplyAnalysis.Extract("The value is -3."));
        Assert.Null(ReplyAnalysis.Extract("none"));
        Assert.Equal(ReplyOutcome.Correct, Read(["It is ", "7"], "7", thinking: false).Outcome);
    }

    [Fact]
    public void A_Tool_Reply_Is_Read_For_Its_Call_And_Its_Code()
    {
        var wrote = ToolReplyAnalysis.Read(
            "<think>\n\n</think>\n\n<tool_call>\n<function=write_file>\n<parameter=path>\na.py\n</parameter>\n" +
            "<parameter=content>\nfor i in range(3):\n    print(i)\n</parameter>\n</function>\n</tool_call>", "write_file");
        Assert.Equal(ToolOutcome.Called, wrote.Outcome);
        Assert.True(wrote.OpensWithCall);
        Assert.True(wrote.CodeInCall);
        Assert.True(wrote.CallClosed);
        Assert.Equal(2, wrote.CodeLines);
        Assert.Empty(wrote.Placeholders);

        var talked = ToolReplyAnalysis.Read(
            "Here is the code:\n```python\ndef solve():\n    # TODO: implement\n    pass\n\n# ... rest of the code here\n```", "write_file");
        Assert.Equal(ToolOutcome.Talked, talked.Outcome);
        Assert.False(talked.CodeInCall);
        Assert.Equal(["todo", "bare-pass", "rest"], talked.Placeholders);
        Assert.Equal(["stand-in"], ToolReplyAnalysis.Read("```cs\nvar p = @\"C:\\Users\\YourName\\Documents\";\n```", "write_file").Placeholders);
        Assert.Empty(ToolReplyAnalysis.Read("```py\nyourself = 1\nprint(yourself)\n```", "write_file").Placeholders);

        Assert.Equal(ToolOutcome.Quiet, ToolReplyAnalysis.Read("A tuple is immutable.", "none").Outcome);
        Assert.Equal(ToolOutcome.FalseCall, ToolReplyAnalysis.Read("<tool_call>\n<function=read_file>", "none").Outcome);
        var late = ToolReplyAnalysis.Read("I'll list it.\n<tool_call>\n<function=run_terminal>", "list_directory");
        Assert.Equal(ToolOutcome.WrongTool, late.Outcome);
        Assert.False(late.OpensWithCall);
        Assert.False(late.CallClosed);
    }

    [Fact]
    public void Simulated_Tools_Answer_By_Name_And_A_Later_Call_Still_Counts()
    {
        var call = ToolSimulator.LastCallIn("Let me look.\n<tool_call>\n<function=list_directory>\n<parameter=path>\nsrc\n</parameter>\n</function>\n</tool_call>");
        Assert.Equal("list_directory", call!.Value.Tool);
        Assert.Equal("src", call.Value.Args["path"]);
        Assert.Null(ToolSimulator.LastCallIn("No call here."));
        Assert.Contains("(empty)", ToolSimulator.Respond("list_directory", call.Value.Args));
        Assert.Contains("\"success\":true", ToolSimulator.Respond("create_file", new Dictionary<string, string> { ["path"] = "a.py", ["content"] = "x\ny" }));
        Assert.Contains("not found", ToolSimulator.Respond("read_file", new Dictionary<string, string> { ["path"] = "a.py" }));
        Assert.Contains("exitCode\":0", ToolSimulator.Respond("run_terminal", new Dictionary<string, string>()));

        string reply = "<tool_call>\n<function=get_current_directory>\n</function>\n</tool_call>" +
            ToolSimulator.ResultTurn("{}", think: false) +
            "<tool_call>\n<function=create_file>\n<parameter=path>\na.py\n</parameter>\n<parameter=content>\nprint(1)\n</parameter>\n</function>\n</tool_call>";
        var a = ToolReplyAnalysis.Read(reply, "create_file");
        Assert.Equal(ToolOutcome.Called, a.Outcome);
        Assert.Equal(["get_current_directory", "create_file"], a.Tools);
        Assert.Equal("get_current_directory", a.FirstTool);
        Assert.True(a.CodeInCall);
        Assert.Equal("print(1)", a.Code);
    }

    [Fact]
    public void A_Run_Seen_Three_Times_Is_A_Loop()
    {
        var cycle = Enumerable.Range(0, ReplyAnalysis.LoopSpan).ToArray();
        var tokens = Enumerable.Range(100, 10).Concat(cycle).Concat(cycle).Concat(cycle).ToArray();
        Assert.Equal(10 + 3 * ReplyAnalysis.LoopSpan - 1, ReplyAnalysis.FindLoop(tokens));
        Assert.Null(ReplyAnalysis.FindLoop(Enumerable.Range(0, 500).ToArray()));

        var pieces = tokens.Select(t => $"w{t} ").ToArray();
        var looped = ReplyAnalysis.Read(tokens, pieces, "5", thinking: true, hitBudget: true);
        Assert.Equal(ReplyOutcome.Looped, looped.Outcome);
    }
}
