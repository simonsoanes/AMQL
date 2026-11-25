using System.Text.Json;
using Amql.Server;

namespace Amql.Tests;

/// <summary>
/// The tool-call grammar: once a reply opens <c>&lt;tool_call&gt;</c>, only calls the request's
/// tools declare can follow — the tool's name, its parameters (each once), the required ones
/// before <c>&lt;/function&gt;</c>, JSON-typed values that parse, string enums from their list.
/// The shapes are an agent harness's own: create_file, and edit_file with an array of edits.
/// </summary>
public sealed class ToolCallConstraintTests
{
    private static readonly ToolGrammar Grammar = ToolGrammar.Compile(JsonDocument.Parse("""
        [{"type":"function","function":{"name":"create_file","parameters":{"type":"object","properties":{
            "path":{"type":"string"},"content":{"type":"string"},"overwrite":{"type":"boolean"}},"required":["path","content"]}}},
         {"type":"function","function":{"name":"edit_file","parameters":{"type":"object","properties":{
            "path":{"type":"string"},"force":{"type":"boolean"},
            "edits":{"type":"array","items":{"type":"object","properties":{
                "op":{"type":"string","enum":["replace","delete","insert_after"]},"anchor":{"type":"string"},
                "content_lines":{"type":"array","items":{"type":"string"}}},"required":["op","anchor"]}}},
            "required":["path","edits"]}}},
         {"type":"function","function":{"name":"run_build","parameters":{"type":"object","properties":{
            "mode":{"type":"string","enum":["debug","release"]},"jobs":{"type":"integer"}},"required":["mode"]}}}]
        """).RootElement.EnumerateArray().Select(t => t.Clone()).ToList());

    private static bool Valid(string call, int piece = 1)
    {
        var c = Templates.QwenCalls.Constraint(Grammar);
        for (int i = 0; i < call.Length; i += piece)
        {
            string part = call.Substring(i, Math.Min(piece, call.Length - i));
            if (!c.Accepts(part))
            {
                return false;
            }
            c.Advance(part);
        }
        return c.Done;
    }

    private const string Create =
        "\n<function=create_file>\n<parameter=path>\nsrc/a.cs\n</parameter>\n<parameter=content>\nclass A { string s = \"<p>\"; }\n</parameter>\n</function>\n</tool_call>";

    private const string Edit =
        "\n<function=edit_file>\n<parameter=path>\nProgram.cs\n</parameter>\n<parameter=edits>\n" +
        "[{\"op\": \"replace\", \"anchor\": \"L4:4319bd\", \"content_lines\": [\"namespace Disk;\"]}]\n</parameter>\n</function>\n</tool_call>";

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(19)]
    public void Calls_The_Tools_Declare_Are_Accepted_However_They_Are_Split(int piece)
    {
        Assert.True(Valid(Create, piece));
        Assert.True(Valid(Edit, piece));
        Assert.True(Valid("\n<function=run_build>\n<parameter=jobs>\n4\n</parameter>\n<parameter=mode>\nrelease\n</parameter>\n</function>\n</tool_call>", piece));
        Assert.True(Valid("<function=run_build><parameter=mode>debug</parameter></function></tool_call>", piece));   // without the newlines
    }

    [Theory]
    [InlineData("\n<function=write_file>")]                                                                         // no such tool
    [InlineData("\n<function=create_file>\n<parameter=text>")]                                                     // no such parameter
    [InlineData("\n<function=create_file>\n<parameter=path>\na\n</parameter>\n<parameter=path>")]                // given twice
    [InlineData("\n<function=create_file>\n<parameter=path>\na\n</parameter>\n</function>")]                     // content is required
    [InlineData("\n<function=edit_file>\n<parameter=edits>\n[{\"op\": \"replace\", \"anchor\": \"L4\"},]")]      // a trailing comma
    [InlineData("\n<function=edit_file>\n<parameter=edits>\n[{\"op\": \"rename\"")]                              // op outside its enum
    [InlineData("\n<function=edit_file>\n<parameter=edits>\n'L42:a3f9c1'")]                                     // not an array
    [InlineData("\n<function=edit_file>\n<parameter=force>\nyes")]                                               // not a boolean
    [InlineData("\n<function=run_build>\n<parameter=mode>\nfast")]                                               // a string enum, bare
    [InlineData("\n<function=run_build>\n<parameter=jobs>\n2.5")]                                                // an integer
    [InlineData("\n<function=run_build>\n<parameter=jobs>\n9 9")]                                                // nothing but its close after a value
    public void Calls_The_Tools_Do_Not_Declare_Are_Refused(string prefix)
    {
        var c = Templates.QwenCalls.Constraint(Grammar);
        bool all = true;
        foreach (char ch in prefix)
        {
            if (!c.Accepts(ch.ToString()))
            {
                all = false;
                break;
            }
            c.Advance(ch.ToString());
        }
        Assert.False(all && c.Done);
        Assert.False(all, $"accepted all of {JsonSerializer.Serialize(prefix)}");
    }

    [Fact]
    public void A_Call_Cannot_End_The_Reply_And_Is_Done_Only_At_Its_Close()
    {
        var c = Templates.QwenCalls.Constraint(Grammar);
        c.Advance(Create[..^"</tool_call>".Length]);
        Assert.False(c.Done);
        Assert.False(c.CanEnd);
        c.Advance("</tool_call>");
        Assert.True(c.Done);
    }

    [Fact]
    public void What_The_Grammar_Admits_The_Reply_Parser_Reads_Back()
    {
        var tools = JsonDocument.Parse("""[{"type":"function","function":{"name":"edit_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array"}}}}}]""")
            .RootElement.EnumerateArray().Select(t => t.Clone()).ToList();
        var reply = ReplyParser.Parse("<tool_call>" + Edit, false, tools, syntax: Templates.QwenCalls);
        var args = JsonDocument.Parse(reply.ToolCalls.Single().Arguments).RootElement;
        Assert.Equal("Program.cs", args.GetProperty("path").GetString());
        Assert.Equal("replace", args.GetProperty("edits")[0].GetProperty("op").GetString());
    }

    [Fact]
    public void The_Way_To_An_Optional_Parameter_Is_Known_Wherever_One_Can_Open()
    {
        var c = Templates.QwenCalls.Constraint(Grammar);
        c.Advance("\n<function=create_file>\n");
        Assert.Equal(new[] { "overwrite" }, c.OptionalLeft());
        Assert.Equal("<parameter=overwrite>", c.PathTo("overwrite"));
        Assert.Null(c.PathTo("force"));                 // not a parameter of create_file
        c.Advance("<");
        Assert.Equal("parameter=overwrite>", c.PathTo("overwrite"));
        c.Advance("parameter=over");
        Assert.Equal("write>", c.PathTo("overwrite"));
        Assert.Null(c.PathTo("path"));                  // "over" cannot become "path"
        c.Advance("write>\ntrue\n</parameter>\n");
        Assert.Empty(c.OptionalLeft());
        Assert.Null(c.PathTo("overwrite"));             // given already
        c.Advance("<parameter=path>\na.cs\n");
        Assert.Null(c.PathTo("content"));               // inside a value nothing can open
    }
}
