using System.Text.Json;
using Amql.Server;
using Xunit;

namespace Amql.Tests;

public class LineDecorationTests
{
    private static readonly string[] Code =
    [
        "using System;", "using System.IO;", "", "var root = args.Length > 0 ? args[0] : \".\";",
        "foreach (var d in Directory.GetDirectories(root))", "{", "    Console.WriteLine(d);", "}",
    ];

    private static readonly string[] Hashes = ["f62232", "109a78", "e3b0c4", "4319bd", "021fb5", "138ca1", "0d3d2f", "abcdef"];

    private static string Decorated(string[] lines) =>
        string.Join("\n", lines.Select((l, i) => $"L{i + 1}:{Hashes[i % Hashes.Length]}│{l}"));

    private static string ReadResult(string[] lines) => JsonSerializer.Serialize(new
    {
        success = true,
        path = "Program.cs",
        content = $"# {lines.Length} lines · LF · final-newline\n" + Decorated(lines),
    });

    [Fact]
    public void LearnsMarkersAndStripsACopiedFile()
    {
        var d = LineDecoration.Learn([ReadResult(Code)]);
        Assert.NotNull(d);
        string copied = $"# {Code.Length} lines · LF · final-newline\n" + Decorated(Code);
        string clean = d!.Strip(copied, out bool stripped);
        Assert.True(stripped);
        Assert.Equal(string.Join("\n", Code), clean);
    }

    [Fact]
    public void ShowsAToolResultPlain()
    {
        string result = ReadResult(Code);
        var d = LineDecoration.Learn([result])!;
        string shown = d.Plain(result, out bool changed);
        Assert.True(changed);
        using var doc = JsonDocument.Parse(shown);
        Assert.Equal(string.Join("\n", Code), doc.RootElement.GetProperty("content").GetString());
        Assert.Equal("Program.cs", doc.RootElement.GetProperty("path").GetString());
        Assert.DoesNotContain("\\u0022", shown);   // quotes in the code stay readable
        // a result with nothing decorated is left exactly as it came
        string other = """{"success":true,"output":"Build succeeded.\n0 Warning(s)\n0 Error(s)"}""";
        Assert.Equal(other, d.Plain(other, out bool touched));
        Assert.False(touched);
    }

    [Fact]
    public void StripsAPartialCopyAndADoubleOne()
    {
        var d = LineDecoration.Learn([ReadResult(Code)])!;
        // the hash dropped, the line number kept
        string partial = string.Join("\n", Code.Select((l, i) => $"L{i + 1}:{l}"));
        Assert.Equal(string.Join("\n", Code), d.Strip(partial, out bool stripped));
        Assert.True(stripped);
        // a file that was itself written from a copy, read and copied again
        string twice = Decorated(Decorated(Code).Split('\n'));
        Assert.Equal(string.Join("\n", Code), d.Strip(twice, out stripped));
    }

    [Fact]
    public void LeavesCodeWithoutMarkersAlone()
    {
        var d = LineDecoration.Learn([ReadResult(Code)])!;
        string code = string.Join("\n", Code);
        Assert.Equal(code, d.Strip(code, out bool stripped));
        Assert.False(stripped);
        // one marked line among plain ones is not a copy
        string one = Decorated(Code[..1]) + "\n" + string.Join("\n", Code[1..]);
        Assert.Equal(one, d.Strip(one, out stripped));
        Assert.False(stripped);
    }

    [Fact]
    public void NumberedListingsAreLearnedToo()
    {
        string listing = string.Join("\n", Code.Select((l, i) => $"{i + 1,6}→{l}"));
        var d = LineDecoration.Learn([listing]);
        Assert.NotNull(d);
        Assert.Equal(string.Join("\n", Code), d!.Strip(listing, out _));
    }

    [Theory]
    [InlineData("Now let me build it", true)]
    [InlineData("The file is written. I'll run the build", true)]
    [InlineData("I’m going to fix line 3", true)]
    [InlineData("The disk usage viewer has been successfully built and tested. Here's what was done", false)]
    [InlineData("The errors are", false)]
    public void AnnouncedActions(string sentence, bool acts) =>
        Assert.Equal(acts, TextGenerator.AnnouncesAction(sentence));

    [Theory]
    [InlineData("File exists. Pass overwrite: true to replace it.", "overwrite", true)]
    [InlineData("Set 'force' to true to apply anyway", "force", true)]
    [InlineData("use background=true for long commands", "background", true)]
    [InlineData("{\"handle\":\"bg-95c9\",\"stopped\":true}", "handle", false)]
    [InlineData("NOTE: 'handle' means poll or stop an existing background command.", "handle", false)]
    [InlineData("NOTE: 'handle' means poll or stop an existing background command.", "stop", false)]
    public void ToolResultSuggestions(string result, string name, bool suggests) =>
        Assert.Equal(suggests, TextGenerator.SuggestsParameter(result, name));

    private const string Step2 = "Step 2: write a new version from scratch with create_file (overwrite true), no need to read_file the old one. "
        + "Don't use edit_file. Then run_build, fix any errors, run it once with run_terminal.";

    [Theory]
    [InlineData("create_file", true)]
    [InlineData("run_build", true)]
    [InlineData("run_terminal", true)]
    [InlineData("read_file", false)]   // "no need to"
    [InlineData("edit_file", false)]   // "Don't use"
    [InlineData("file", false)]        // part of a name, not a name
    public void ToolsAMessageNames(string tool, bool named) =>
        Assert.Equal(named, TextGenerator.NamesTool(Step2, tool));

    [Fact]
    public void NamedToolsFallDueInTheirOrder()
    {
        string[] tools = ["read_file", "run_terminal", "edit_file", "run_build", "create_file"];
        Assert.Equal(["create_file"], TextGenerator.ToolsDue(Step2, tools, null));
        Assert.Equal(["create_file", "run_build"], TextGenerator.ToolsDue(Step2, tools, new HashSet<string> { "create_file", "read_file" }));
        Assert.Equal(["create_file", "run_build", "run_terminal"],
            TextGenerator.ToolsDue(Step2, tools, new HashSet<string> { "create_file", "run_build", "run_terminal" }));
        Assert.Empty(TextGenerator.ToolsDue("How big is my disk?", tools, null));
    }

    [Fact]
    public void PlainResultsHaveNoDecoration()
    {
        Assert.Null(LineDecoration.Learn([string.Join("\n", Code)]));
        Assert.Null(LineDecoration.Learn([string.Join("\n", Enumerable.Range(0, 8).Select(i => "// note " + i))]));
        Assert.Null(LineDecoration.Learn(["{\"success\":true}"]));
    }

    [Fact]
    public void ParserStripsCallValues()
    {
        var tools = JsonDocument.Parse("""
            [{"type":"function","function":{"name":"create_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}}}}}]
            """).RootElement.EnumerateArray().ToList();
        var d = LineDecoration.Learn([ReadResult(Code)]);
        string reply = "<tool_call>\n<function=create_file>\n<parameter=path>\nProgram.cs\n</parameter>\n<parameter=content>\n"
            + $"# {Code.Length} lines · LF · final-newline\n" + Decorated(Code) + "\n</parameter>\n</function>\n</tool_call>";
        var parser = new ReplyParser(false, tools, d);
        parser.Feed(reply);
        parser.Finish();
        var args = JsonDocument.Parse(Assert.Single(parser.ToolCalls).Arguments).RootElement;
        Assert.Equal(string.Join("\n", Code), args.GetProperty("content").GetString());
        Assert.Equal(1, parser.Undecorated);
    }
}
