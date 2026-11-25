using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Cli;
using Amql.Server;
using Xunit;

namespace Amql.Tests;

/// <summary>Gemma 4's turns, held to what transformers renders from the model's own
/// template (<c>fixtures/gemma4_chat_golden.json</c>).</summary>
public class TurnTemplateTests
{
    [Fact]
    public void Conversations_Render_As_The_Template_Does()
    {
        using var golden = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "fixtures", "gemma4_chat_golden.json")));
        foreach (var entry in golden.RootElement.EnumerateArray())
        {
            var messages = entry.GetProperty("messages").EnumerateArray().Select(m =>
                new ChatMessage(m.GetProperty("role").GetString()!, m.GetProperty("content").GetString()!)
                {
                    Reasoning = m.GetProperty("reasoning").GetString(),
                }).ToList();
            Assert.Equal(entry.GetProperty("text").GetString(),
                Templates.Gemma.Apply(messages, entry.GetProperty("thinking").GetBoolean()));
        }
    }

    [Fact]
    public void Tool_Conversations_Render_As_The_Template_Does()
    {
        using var golden = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "fixtures", "gemma4_tool_golden.json")));
        foreach (var c in golden.RootElement.EnumerateArray())
        {
            string? Text(JsonElement m, string name) => m.TryGetProperty(name, out var v) ? v.GetString() : null;
            var messages = c.GetProperty("messages").EnumerateArray().Select(m => new ChatMessage(Text(m, "role")!, Text(m, "content")!)
            {
                Reasoning = Text(m, "reasoning_content"), CallId = Text(m, "tool_call_id"), Name = Text(m, "name"),
                ToolCalls = m.TryGetProperty("tool_calls", out var calls)
                    ? calls.EnumerateArray().Select(t => new ChatToolCall(t.GetProperty("function").GetProperty("name").GetString()!,
                        t.GetProperty("function").GetProperty("arguments")) { Id = t.GetProperty("id").GetString() }).ToList()
                    : null,
            }).ToList();
            var tools = c.GetProperty("tools").ValueKind == JsonValueKind.Array ? c.GetProperty("tools").EnumerateArray().ToList() : null;
            Assert.Equal(c.GetProperty("rendered").GetString(),
                Templates.Gemma.Apply(messages, c.GetProperty("thinking").GetBoolean(), tools));
        }
    }

    [Fact]
    public void What_Was_Said_With_A_Call_Comes_Before_It_And_A_Result_Without_An_Id_Answers_In_Order()
    {
        using var args = JsonDocument.Parse("{}");
        var call = new ChatMessage("assistant", "Looking.") { ToolCalls = [new ChatToolCall("a", args.RootElement), new ChatToolCall("b", args.RootElement)] };
        string text = Templates.Gemma.Apply([new ChatMessage("user", "q"), call, new ChatMessage("tool", "1"), new ChatMessage("tool", "2")]);
        Assert.EndsWith("<|turn>model\nLooking.<|tool_call>call:a{}<tool_call|><|tool_call>call:b{}<tool_call|>" +
            "<|tool_response>response:a{value:<|\"|>1<|\"|>}<tool_response|><|tool_response>response:b{value:<|\"|>2<|\"|>}<tool_response|>", text);
    }

    [Fact]
    public void An_Image_Stands_Where_Its_Part_Does()
    {
        string text = Templates.Gemma.Apply([new ChatMessage("user", "this?") { Parts = [new ChatPart(null), new ChatPart(" this? ")] }]);
        Assert.Contains("<|turn>user\n<|image|>this?<turn|>", text);
        Assert.Throws<ChatTemplateException>(() => ChatTemplate.Parse("{% for m in messages %}<|im_start|>{{ m.content }}{% endfor %}").Apply([new ChatMessage("user", "this?") { Parts = [new ChatPart(null)] }]));
    }

    private static readonly List<JsonElement> Tools = JsonDocument.Parse("""
        [{"type":"function","function":{"name":"create_file","parameters":{"type":"object","properties":{
            "path":{"type":"string"},"content":{"type":"string"},"overwrite":{"type":"boolean"}},"required":["path","content"]}}},
         {"type":"function","function":{"name":"edit_file","parameters":{"type":"object","properties":{
            "path":{"type":"string"},"note":{"type":"string"},
            "edits":{"type":"array","items":{"type":"object","properties":{
                "op":{"type":"string","enum":["replace","it's <new>"]},"anchor":{"type":"string"},
                "lines":{"type":"array","items":{"type":"string"}}},"required":["op","anchor"]}}},
            "required":["path","edits"]}}},
         {"type":"function","function":{"name":"run_build","parameters":{"type":"object","properties":{
            "mode":{"type":"string","enum":["debug","release"]},"jobs":{"type":"integer"},"env":{"type":"object"}},"required":["mode"]}}},
         {"type":"function","function":{"name":"now"}}]
        """).RootElement.EnumerateArray().Select(t => t.Clone()).ToList();

    private static bool Valid(string call, int piece)
    {
        var c = Templates.GemmaCalls.Constraint(ToolGrammar.Compile(Tools));
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

    private const string Q = "<|\"|>";

    private static readonly string[] Calls =
    [
        $"call:create_file{{content:{Q}a = \"x\\n\" <|tool \n<|\" end{Q},path:{Q}src/é😀.py{Q}}}<tool_call|>",
        $"call:create_file{{path:{Q}{Q},content:{Q}<{Q},overwrite:true}}<tool_call|>",
        $"call:edit_file{{edits:[{{anchor:{Q}L4{Q},op:{Q}it's <new>{Q},lines:[{Q}a{Q},{Q}{Q}]}},{{op:{Q}replace{Q},anchor:{Q}b{Q}}}],path:{Q}p{Q}}}<tool_call|>",
        $"call:run_build{{jobs:4,mode:{Q}release{Q},env:{{A:1,{Q}b c{Q}:[null,-2.5e3,{{}}]}}}}<tool_call|>",
        "call:now{}<tool_call|>",
    ];

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(17)]
    [InlineData(1000)]
    public void Calls_The_Tools_Declare_Are_Accepted_However_They_Are_Split(int piece)
    {
        foreach (string call in Calls)
        {
            Assert.True(Valid(call, piece), call);
        }
    }

    [Theory]
    [InlineData("call:write_file{")]                                           // no such tool
    [InlineData("call:create_file{text")]                                      // no such parameter
    [InlineData("call:create_file{path:<|\"|>a<|\"|>}")]                       // content is required
    [InlineData("call:create_file{path:<|\"|>a<|\"|>,path")]                   // given twice
    [InlineData("call:create_file{path:a")]                                    // a string needs its marks
    [InlineData("call:create_file{path:\"a")]                                  // its own marks
    [InlineData("call:create_file{overwrite:<|\"|>")]                          // not a boolean
    [InlineData("call:edit_file{edits:[{op:<|\"|>rename")]                     // outside its enum
    [InlineData("call:edit_file{edits:[{anchor:<|\"|>a<|\"|>}")]               // op is required
    [InlineData("call:run_build{jobs:2.5")]                                    // an integer
    [InlineData("call:run_build{mode:<|\"|>debug<|\"|>},")]                    // nothing but its close
    [InlineData("call:now{a")]                                                 // takes nothing
    public void Calls_The_Tools_Do_Not_Declare_Are_Refused(string prefix)
    {
        var c = Templates.GemmaCalls.Constraint(ToolGrammar.Compile(Tools));
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
        Assert.False(all);
    }

    [Fact]
    public void The_Way_To_A_Mentioned_Parameter_And_Tool_Is_Known()
    {
        var c = Templates.GemmaCalls.Constraint(ToolGrammar.Compile(Tools));
        Assert.Equal("call:now{", c.PathToTool("now"));
        c.Advance("call:cr");
        Assert.Equal("eate_file{", c.PathToTool("create_file"));
        Assert.Null(c.PathToTool("now"));
        c.Advance("eate_file{");
        Assert.Equal(["overwrite"], c.OptionalLeft());
        Assert.Equal("overwrite:", c.PathTo("overwrite"));
        c.Advance("pa");
        Assert.Null(c.PathTo("overwrite"));
        c.Advance("th:<|\"|>a<|\"|>");
        Assert.Equal(",overwrite:", c.PathTo("overwrite"));
        Assert.Null(c.PathTo("path"));
        c.Advance(",");
        Assert.Equal("overwrite:", c.PathTo("overwrite"));
        c.Advance("content:<|\"|>a,");
        Assert.Null(c.PathTo("overwrite"));
        c.Advance("<|\"|>}<tool_call|>");
        Assert.True(c.Done);
    }

    private const string Reply = "<think>Look first.\n</think>I'll write it.<|tool_call>call:create_file{content:<|\"|>if a < b:\n  x = {\"k\": [1]}\n<|\"|>,path:<|\"|>src/a.py<|\"|>,overwrite:true}<tool_call|>" +
        "<|tool_call>call:run_build{mode:<|\"|>debug<|\"|>,jobs:<|\"|>3<|\"|>,env:{A:<|\"|>1<|\"|>,b:[1,2.0,null]}}<tool_call|>";

    [Fact]
    public void A_Reply_Splits_Into_Reasoning_Content_And_Calls_However_It_Streams()
    {
        var whole = ReplyParser.Parse(Reply, false, Tools, syntax: Templates.GemmaCalls);
        Assert.Equal("Look first.", whole.Reasoning);
        Assert.Equal("I'll write it.", whole.Content);
        Assert.Equal(["create_file", "run_build"], whole.ToolCalls.Select(c => c.Name));
        var args = JsonNode.Parse(whole.ToolCalls[0].Arguments)!;
        Assert.Equal("if a < b:\n  x = {\"k\": [1]}\n", args["content"]!.GetValue<string>());
        Assert.True(args["overwrite"]!.GetValue<bool>());
        Assert.Equal("""{"mode":"debug","jobs":3,"env":{"A":"1","b":[1,2.0,null]}}""", whole.ToolCalls[1].Arguments);   // an integer given as text is read as one

        var parser = new ReplyParser(false, Tools, null, Templates.GemmaCalls);
        var events = Reply.SelectMany(ch => parser.Feed(ch.ToString())).ToList();
        events.AddRange(parser.Finish());
        Assert.Equal(whole.Reasoning, string.Concat(events.OfType<ReplyEvent.Reasoning>().Select(e => e.Delta)));
        Assert.Equal(whole.Content, string.Concat(events.OfType<ReplyEvent.Content>().Select(e => e.Delta)));
        Assert.Equal(whole.ToolCalls, events.OfType<ReplyEvent.ToolCall>().Select(e => e.Call));
    }

    [Theory]
    [InlineData("call:f{a:b c,d:12,e:True}", """{"a":"b c","d":12,"e":true}""")]
    [InlineData("call:f{ \"a\": \"x\\\"y\", \"b\": [1, 2] }", """{"a":"x\"y","b":[1,2]}""")]
    [InlineData("call:f{a:[<|\"|>x<|\"|>,{b:<|\"|>}<|\"|>}", """{"a":["x",{"b":"}"}]}""")]
    [InlineData("call:f{}", "{}")]
    public void A_Call_Written_Freely_Is_Read_As_Far_As_It_Goes(string inner, string expected) =>
        Assert.Equal(expected, Templates.GemmaCalls.Read(inner, new Dictionary<string, JsonElement>(), s => s).Single().Arguments);

    [Fact]
    public void Text_That_Is_No_Call_Is_None() =>
        Assert.Empty(Templates.GemmaCalls.Read("get_weather Paris", new Dictionary<string, JsonElement>(), s => s));
}
