using Amql.Entail;
using Amql.Inference;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// <c>prefer</c> and <c>decide</c>: outputs that weigh the host's own probability. The
/// program says where a decision is and the threshold; the host's model supplies the
/// evidence, absolutely or against its prior for the same conversation with the latest
/// message blanked.
/// </summary>
public class EntailPreferTests
{
    private static EntailAdapterManifest Manifest(params EntailAdapterOutput[] outputs) =>
        new("test", "00", new Dictionary<int, int> { [5] = 2 }, outputs);

    /// <summary>Logits whose softmax gives token i the probability ps[i].</summary>
    private static float[] Logits(params double[] ps) => ps.Select(p => (float)Math.Log(p)).ToArray();

    private static readonly float[] Set = { 1f };

    [Fact]
    public void Prefer_And_Decide_Parse_With_Their_Thresholds()
    {
        var p = EntailCompiler.Compile("""
            state a
            token "x" means a = 1
            prefer "y" at 0.25 when a
            decide "z" at 2 times prior when a
            """);
        Assert.Equal(2, p.Prefers.Count);
        Assert.Equal(("y", 0.25, false, false), (p.Prefers[0].Text, p.Prefers[0].Threshold, p.Prefers[0].TimesPrior, p.Prefers[0].Strict));
        Assert.Equal(("z", 2.0, true, true), (p.Prefers[1].Text, p.Prefers[1].Threshold, p.Prefers[1].TimesPrior, p.Prefers[1].Strict));
    }

    [Theory]
    [InlineData("prefer \"y\" at 0 when a")]
    [InlineData("prefer \"y\" at 1.5 when a")]
    [InlineData("decide \"y\" at -1 times prior when a")]
    public void A_Threshold_Must_Be_A_Probability_Or_A_Positive_Factor(string line) =>
        Assert.Throws<EntailException>(() => EntailCompiler.Compile($"state a\ntoken \"x\" means a = 1\n{line}"));

    [Fact]
    public void A_State_May_Still_Be_Called_Prefer_Or_Decide()
    {
        var p = EntailCompiler.Compile("state prefer, decide\ntoken \"x\" means prefer = 1\nentail decide when prefer\nemit \"y\" when decide");
        Assert.Equal("y", EntailInterpreter.Answer(p, new[] { "x" }, limit: 1));
    }

    [Fact]
    public void Weights_Cannot_Hold_A_Prefer()
    {
        var p = EntailCompiler.Compile("state a\ntoken \"x\" means a = 1\nprefer \"y\" at 0.5 when a");
        var dir = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Assert.Contains("adapter", Assert.Throws<EntailException>(() => EntailWeights.WriteFresh(p, dir)).Message);
    }

    [Fact]
    public void A_Prefer_Says_Its_Token_At_Or_Above_The_Threshold()
    {
        var manifest = Manifest(new EntailAdapterOutput(0, EntailOutputKind.Prefer, new[] { 1 }, 0.2));
        var verdict = EntailVerdict.Read(manifest, Set);
        Assert.Equal((1, EntailTokenSource.Entail), verdict.Choose(Logits(0.7, 0.25, 0.05)));
        Assert.Equal((0, EntailTokenSource.Partner), verdict.Choose(Logits(0.85, 0.1, 0.05)));
    }

    [Fact]
    public void A_Prefer_Relative_To_The_Prior_Needs_One()
    {
        var manifest = Manifest(new EntailAdapterOutput(0, EntailOutputKind.Prefer, new[] { 1 }, 2, TimesPrior: true));
        Assert.True(manifest.NeedsPrior);
        var verdict = EntailVerdict.Read(manifest, Set);
        var logits = Logits(0.7, 0.1, 0.2);
        Assert.Equal((0, EntailTokenSource.Partner), verdict.Choose(logits));
        Assert.Equal((1, EntailTokenSource.Entail), verdict.Choose(logits, new[] { 0.9, 0.04, 0.06 }));
        Assert.Equal((0, EntailTokenSource.Partner), verdict.Choose(logits, new[] { 0.9, 0.06, 0.04 }));
    }

    [Fact]
    public void Decide_Also_Rules_Its_Token_Out_Below_The_Threshold()
    {
        var manifest = Manifest(new EntailAdapterOutput(0, EntailOutputKind.Prefer, new[] { 1 }, 2, TimesPrior: true, Strict: true));
        var verdict = EntailVerdict.Read(manifest, Set);
        // The partner's favourite is the token itself, but no more likely than its prior.
        Assert.Equal((2, EntailTokenSource.Narrowed), verdict.Choose(Logits(0.1, 0.6, 0.3), new[] { 0.1, 0.5, 0.4 }));
        Assert.Equal((1, EntailTokenSource.Entail), verdict.Choose(Logits(0.1, 0.6, 0.3), new[] { 0.6, 0.2, 0.2 }));
    }

    [Fact]
    public void A_Prefer_Survives_The_Disk()
    {
        using var dir = new TempDir();
        Manifest(new EntailAdapterOutput(0, EntailOutputKind.Prefer, new[] { 3 }, 2, TimesPrior: true, Strict: true)).Save(dir.Path);
        var o = EntailAdapterManifest.Load(dir.Path).Outputs[0];
        Assert.Equal((EntailOutputKind.Prefer, 2.0, true, true), (o.Kind, o.Threshold, o.TimesPrior, o.Strict));
    }

    // A tokenizer of single characters, with two control ids for ChatML's markers.
    private const int Start = 1000, End = 1001;

    private static int[] Chat(params string[] parts) => parts.SelectMany(part => part switch
    {
        "<s>" => new[] { Start },
        "</s>" => new[] { End },
        _ => part.Select(c => (int)c),
    }).ToArray();

    private static string Text(IEnumerable<int> ids) =>
        string.Concat(ids.Select(i => i switch { Start => "<s>", End => "</s>", _ => ((char)i).ToString() }));

    private static readonly Func<int[], int[]?> Blank =
        ChatPrior.ForChatMl(t => t.Select(c => (int)c).ToArray(), ids => Text(ids), Start, End);

    [Fact]
    public void The_Prior_Blanks_Only_The_Latest_User_Turn()
    {
        var prompt = Chat("<s>", "system\nTools", "</s>", "\n", "<s>", "user\nfirst", "</s>", "\n",
            "<s>", "assistant\nok", "</s>", "\n", "<s>", "user\nflying to Oslo", "</s>", "\n", "<s>", "assistant\n");
        Assert.Equal("<s>system\nTools</s>\n<s>user\nfirst</s>\n<s>assistant\nok</s>\n<s>user\nN/A</s>\n<s>assistant\n",
            Text(Blank(prompt)!));
    }

    [Fact]
    public void A_Tool_Result_Is_Blanked_Inside_Its_Wrapper()
    {
        var prompt = Chat("<s>", "user\n<tool_response>\n{\"temp\": 3}\n</tool_response>", "</s>", "\n", "<s>", "assistant\n");
        Assert.Equal("<s>user\n<tool_response>\nN/A\n</tool_response></s>\n<s>assistant\n", Text(Blank(prompt)!));
        Assert.Null(Blank(Chat("<s>", "system\nTools", "</s>")));
    }

    /// <summary>The program in the experiment marks the reply's first token, and only it, and
    /// only when the system turn showed the call format.</summary>
    [Fact]
    public void The_Call_Program_Marks_The_Reply_Start_When_Tools_Are_Offered()
    {
        var p = EntailCompiler.Compile(File.ReadAllText(EntailTests.RepoFile("scripts", "entail", "toolcall", "call-when-likely.entail.txt")));
        int lane = p.Lanes.Single(l => l.Name == "reply_start").Id;
        string[] Turns(string system) => new[]
        {
            "<|im_start|>", "system", "\n", system, "<|im_end|>", "\n",
            "<|im_start|>", "user", "\n", "hi", "<|im_end|>", "\n",
            "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n",
        };
        var offered = EntailInterpreter.Run(p, Turns("<tool_call>"));
        Assert.Equal(new[] { offered.Length - 1 }, Enumerable.Range(0, offered.Length).Where(i => offered[i][lane] > 0.5));
        var none = EntailInterpreter.Run(p, Turns("plain"));
        Assert.DoesNotContain(none, row => row[lane] > 0.5);
        Assert.Single(p.Prefers);
        Assert.True(p.Prefers[0] is { TimesPrior: true, Strict: true, Threshold: 2 });
    }
}
