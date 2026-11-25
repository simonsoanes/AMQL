using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Cli;
using Amql.Server;
using Xunit;

namespace Amql.Tests;

/// <summary>The templates the tests render with, and how calls are written by them.</summary>
internal static class Templates
{
    private static ChatTemplate Read(string file) => ChatTemplate.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", file)), "<bos>", "");

    public static readonly ChatTemplate Qwen = Read("qwen35_chat_template.jinja");
    public static readonly ChatTemplate Gemma = Read("gemma4_chat_template.jinja");
    public static readonly CallSyntax QwenCalls = new(Qwen.Dialect.Calls!);
    public static readonly CallSyntax GemmaCalls = new(Gemma.Dialect.Calls!);
}

/// <summary>What is learned of a chat template by probing it: its calls are read back as they were
/// made and held to the tools, whatever the family.</summary>
public class ChatDialectTests
{
    /// <summary>The forms templates write calls in, in templates cut down to those.</summary>
    private static readonly (string Name, ArgumentStyle Style, string Template)[] Forms =
    [
        ("hermes", ArgumentStyle.Json, Turns(
            "{% for c in m.tool_calls %}{{ '<tool_call>\\n{\"name\": \"' + c.function.name + '\", \"arguments\": ' + (c.function.arguments | tojson) + '}\\n</tool_call>' }}{% if not loop.last %}{{ '\\n' }}{% endif %}{% endfor %}")),
        ("array in one section", ArgumentStyle.Json, Turns(
            "{{ '[TOOL_CALLS][' }}{% for c in m.tool_calls %}{{ '{\"name\": \"' + c.function.name + '\", \"arguments\": ' + (c.function.arguments | tojson) + '}' }}{% if not loop.last %}{{ ', ' }}{% endif %}{% endfor %}{{ ']' }}")),
        ("name outside the object", ArgumentStyle.Json, Turns(
            "{% for c in m.tool_calls %}{{ '[TOOL_CALLS]' + c.function.name + '[ARGS]' + (c.function.arguments | tojson) }}{% endfor %}")),
        ("a section of fenced calls", ArgumentStyle.Json, Turns(
            "{{ '<|calls|>' }}{% for c in m.tool_calls %}{{ '<|call|>function<|sep|>' + c.function.name + '\\n```json\\n' + (c.function.arguments | tojson) + '\\n```<|end_call|>' }}{% if not loop.last %}{{ '\\n' }}{% endif %}{% endfor %}{{ '<|end_calls|>' }}")),
        ("a bare object", ArgumentStyle.Json, Turns(
            "{% if m.tool_calls | length > 1 %}{{ raise_exception('one call at a time') }}{% endif %}{% set c = m.tool_calls[0] %}{{ '{\"name\": \"' + c.function.name + '\", \"parameters\": ' + (c.function.arguments | tojson) + '}' }}")),
        ("xml", ArgumentStyle.Pairs, Turns(
            "{% for c in m.tool_calls %}{{ '<tool_call>\\n<function=' + c.function.name + '>\\n' }}{% for k, v in c.function.arguments.items() %}{{ '<parameter=' + k + '>\\n' + (v if v is string else v | tojson) + '\\n</parameter>\\n' }}{% endfor %}{{ '</function>\\n</tool_call>' }}{% if not loop.last %}{{ '\\n' }}{% endif %}{% endfor %}")),
        ("attributes", ArgumentStyle.Pairs, Turns(
            "{% for c in m.tool_calls %}{{ '<function name=\"' + c.function.name + '\">' }}{% for k, v in c.function.arguments.items() %}{{ '<param name=\"' + k + '\">' }}{% if v is string and ('<' in v or '\\n' in v) %}{{ '<![CDATA[' + v + ']]>' }}{% else %}{{ v if v is string else v | tojson }}{% endif %}{{ '</param>' }}{% endfor %}{{ '</function>' }}{% if not loop.last %}{{ '\\n' }}{% endif %}{% endfor %}")),
        ("keys and values apart", ArgumentStyle.Pairs, Turns(
            "{% for c in m.tool_calls %}{{ '\\n<tool_call>' + c.function.name + '\\n' }}{% for k, v in c.function.arguments.items() %}{{ '<arg_key>' + k + '</arg_key>\\n<arg_value>' + (v if v is string else v | tojson) + '</arg_value>\\n' }}{% endfor %}{{ '</tool_call>' }}{% endfor %}")),
        ("python", ArgumentStyle.Python, Turns(
            "{{ '<|tool_call_start|>[' }}{% for c in m.tool_calls %}{{ c.function.name + '(' }}{% for k, v in c.function.arguments.items() %}{{ k + '=' + ('%r' % (v,)) }}{% if not loop.last %}{{ ', ' }}{% endif %}{% endfor %}{{ ')' }}{% if not loop.last %}{{ ', ' }}{% endif %}{% endfor %}{{ ']<|tool_call_end|>' }}")),
    ];

    private static string Turns(string calls) =>
        "{% if tools %}{{ '<|s|>system\\n' + (tools | tojson) + '<|e|>\\n' }}{% endif %}{% for m in messages %}{{ '<|s|>' + m.role + '\\n' }}" +
        "{{ m.content }}{% if m.tool_calls %}" + calls + "{% endif %}{{ '<|e|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|s|>assistant\\n' }}{% endif %}";

    private static readonly List<JsonElement> Tools = JsonDocument.Parse("""
        [{"type":"function","function":{"name":"read","description":"r","parameters":{"type":"object","properties":{
            "path":{"type":"string"},"limit":{"type":"integer"},"all":{"type":"boolean"}},"required":["path"]}}},
         {"type":"function","function":{"name":"read_lines","description":"w","parameters":{"type":"object","properties":{
            "path":{"type":"string"},"ratio":{"type":"number"},"mode":{"type":"string","enum":["fast","very slow"]},
            "edits":{"type":"array","items":{"type":"object","properties":{"op":{"type":"string"},"n":{"type":"integer"}}}}},"required":["path","mode"]}}},
         {"type":"function","function":{"name":"now","description":"n","parameters":{"type":"object","properties":{}}}}]
        """).RootElement.EnumerateArray().Select(t => t.Clone()).ToList();

    private static readonly (string Name, string Arguments)[][] Made =
    [
        [("now", "{}")],
        [("read", """{"path":"src/a b.py"}""")],
        [("read", """{"limit":12,"path":"x","all":true}""")],
        [("read_lines", """{"mode":"very slow","path":"if a < b:\n    print(\"it's\", '\\n')\n# é😀 </x> {y} [z], (w)","ratio":-0.25,"edits":[{"op":"add it's","n":3},{"op":"","n":0}]}""")],
        [("read", """{"path":"one"}"""), ("read_lines", """{"path":"two","mode":"fast"}"""), ("now", "{}")],
    ];

    public static IEnumerable<object[]> FormNames() => Forms.Select(f => new object[] { f.Name });

    [Theory]
    [MemberData(nameof(FormNames))]
    public void Calls_Come_Back_As_They_Were_Made(string form)
    {
        var (_, style, source) = Forms.Single(f => f.Name == form);
        var template = ChatTemplate.Parse(source);
        Assert.True(template.Dialect.Calls is not null, template.Dialect.CallsRefusal);
        Assert.Equal(style, template.Dialect.Calls!.Style);
        Assert.True(RoundTrips(template, out string? problem), problem);
    }

    private static string Sorted(string arguments) =>
        Sort(JsonNode.Parse(arguments))!.ToJsonString();

    private static JsonNode? Sort(JsonNode? node) => node switch
    {
        JsonObject o => new JsonObject(o.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => KeyValuePair.Create(p.Key, Sort(p.Value)))),
        JsonArray a => new JsonArray(a.Select(Sort).ToArray()),
        _ => node?.DeepClone(),
    };

    private static bool RoundTrips(ChatTemplate template, out string? problem)
    {
        problem = null;
        var syntax = new CallSyntax(template.Dialect.Calls!);
        int tried = 0;
        foreach (var made in Made)
        {
            var calls = made.Select((c, i) => new ChatToolCall(c.Name, JsonDocument.Parse(c.Arguments).RootElement.Clone()) { Id = $"call0000{i}" }).ToList();
            string text;
            try
            {
                text = template.Render([new ChatMessage("user", "QUESTION"), new ChatMessage("assistant", "") { ToolCalls = calls }], false, Tools, false);
            }
            catch (ChatTemplateException)
            {
                continue;   // a template that takes one call at a time
            }
            text = text[(text.IndexOf("QUESTION", StringComparison.Ordinal) + "QUESTION".Length)..];
            if (text.Contains("\"if a < b:\n    print(\""))
            {
                continue;   // strings written between quotes with nothing escaped, which no reader can tell apart
            }
            string body = text[text.IndexOf(syntax.Open, StringComparison.Ordinal)..];
            if (template.Dialect.AfterReply is { Length: > 0 } end && body.EndsWith(end, StringComparison.Ordinal))
            {
                body = body[..^end.Length];
            }
            tried++;

            // Read whole, and as it streams.
            var reply = ReplyParser.Parse(body, false, Tools, syntax: syntax);
            var parser = new ReplyParser(false, Tools, null, syntax);
            var streamed = body.SelectMany(c => parser.Feed(c.ToString())).ToList();
            streamed.AddRange(parser.Finish());
            foreach (var read in new[] { reply.ToolCalls, streamed.OfType<ReplyEvent.ToolCall>().Select(e => e.Call).ToList() })
            {
                string got = string.Join(" | ", read.Select(c => c.Name + Sorted(c.Arguments)));
                string want = string.Join(" | ", made.Select(c => c.Name + Sorted(c.Arguments)));
                if (got != want)
                {
                    problem = $"{body}\nread as {got}\nnot {want}";
                    return false;
                }
            }

            // The grammar takes the calls as the template writes them, however they are split.
            foreach (int piece in new[] { 1, 4, 1000 })
            {
                var grammar = ToolGrammar.Compile(Tools);
                ICallConstraint? c = null;
                string rest = body;
                while (rest.Length > 0)
                {
                    if (c is null || c.Done)
                    {
                        int open = rest.IndexOf(syntax.Open, StringComparison.Ordinal);
                        if (open < 0)
                        {
                            break;
                        }
                        rest = rest[(open + syntax.Open.Length)..];
                        c = syntax.Constraint(grammar);
                    }
                    // A piece never runs past the calls here: the server hands a token over whole only while they are open.
                    string part = rest[..Math.Min(piece, rest.Length)];
                    while (part.Length > 1 && !c.Accepts(part))
                    {
                        part = part[..^1];
                    }
                    if (!c.Accepts(part))
                    {
                        problem = $"{body}\nrefused at {rest}";
                        return false;
                    }
                    c.Advance(part);
                    rest = rest[part.Length..];
                }
                if (c is not { Done: true })
                {
                    problem = $"{body}\ndoes not end";
                    return false;
                }
            }
        }
        problem ??= tried == 0 ? "rendered nothing" : null;
        return tried > 0;
    }

    [Theory]
    [MemberData(nameof(FormNames))]
    public void Calls_The_Tools_Do_Not_Declare_Are_Refused(string form)
    {
        var template = ChatTemplate.Parse(Forms.Single(f => f.Name == form).Template);
        var syntax = new CallSyntax(template.Dialect.Calls!);
        foreach (var (name, args) in new[]
        {
            ("write", "{}"), ("read", """{"file":"x"}"""), ("read", "{}"), ("read", """{"path":"x","limit":"many"}"""),
            ("read_lines", """{"path":"x","mode":"slow"}"""), ("now", """{"path":"x"}"""), ("read_lines", """{"path":"x","mode":"fast","edits":{"op":1}}"""),
        })
        {
            var call = new ChatToolCall(name, JsonDocument.Parse(args).RootElement.Clone()) { Id = "call00000" };
            string text = template.Render([new ChatMessage("user", "Q"), new ChatMessage("assistant", "") { ToolCalls = [call] }], false, Tools, false);
            var c = syntax.Constraint(ToolGrammar.Compile(Tools));
            bool ok = true;
            foreach (char ch in text[(text.LastIndexOf(syntax.Open, StringComparison.Ordinal) + syntax.Open.Length)..])
            {
                if (c.Done)
                {
                    break;
                }
                ok = c.Accepts(ch.ToString());
                if (!ok)
                {
                    break;
                }
                c.Advance(ch.ToString());
            }
            Assert.False(ok, $"{form}: {text}");
        }
    }

    [Theory]
    [MemberData(nameof(FormNames))]
    public void The_Way_To_A_Tool_And_A_Parameter_Leads_There(string form)
    {
        var template = ChatTemplate.Parse(Forms.Single(f => f.Name == form).Template);
        var syntax = new CallSyntax(template.Dialect.Calls!);
        var c = syntax.Constraint(ToolGrammar.Compile(Tools));
        Assert.True(c.Accepts(c.PathToTool("read_lines")!), c.PathToTool("read_lines"));
        // A call to a named tool starts as the template writes it.
        string start = syntax.Start("read");
        string rest = start.TrimStart()[syntax.Open.Length..];
        Assert.True(rest.Length == 0 || c.Accepts(rest), start);
        c.Advance(rest);
        Assert.Equal(["limit", "all"], c.OptionalLeft());
        Assert.Null(c.PathToTool("now"));
        string? path = c.PathTo("limit");
        Assert.NotNull(path);
        Assert.True(c.Accepts(path), path);
        c.Advance(path);
        Assert.Null(c.PathTo("limit"));
        Assert.True(c.Accepts("12"));
        Assert.False(c.Accepts("a"));
    }

    [Fact]
    public void A_Template_Whose_Calls_Have_No_Mark_Says_Why()
    {
        var template = ChatTemplate.Parse(Turns("{% for c in m.tool_calls %}{{ ' to=' + c.function.name + ' json ' + (c.function.arguments | tojson) }}{% endfor %}"));
        Assert.Null(template.Dialect.Calls);
        Assert.Contains("mark", template.Dialect.CallsRefusal);
    }

    [Fact]
    public void Reasoning_Turns_And_Images_Are_Learned()
    {
        var gemma = Templates.Gemma.Dialect;
        Assert.Equal(("<|channel>thought", "<channel|>"), (gemma.ThinkOpenMark, gemma.ThinkCloseMark));
        Assert.Equal("<|channel>thought\n<channel|>", gemma.EmptyThinking);
        Assert.True(gemma.ThinkSwitch);
        Assert.True(gemma.SaysAfterResults);
        Assert.Equal("<|image|>", gemma.Image);
        var qwen = Templates.Qwen.Dialect;
        Assert.Equal(("<think>\n", "\n</think>\n\n"), (qwen.ThinkOpen, qwen.ThinkClose));
        Assert.Equal("<|im_end|>\n", qwen.AfterReply);
        Assert.False(qwen.SaysAfterResults);
        Assert.True(qwen.OpensThinking(Templates.Qwen.Apply("q", enableThinking: true)));
        Assert.False(qwen.OpensThinking(Templates.Qwen.Apply("q")));
        var plain = ChatTemplate.Parse("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}").Dialect;
        Assert.Equal((false, false), (plain.WritesTools, plain.WritesCalls));
        Assert.Null(plain.ThinkOpen);
        Assert.Null(plain.Image);
    }

    [Fact]
    public void More_Variables_Reach_The_Template_And_Its_Refusals_Are_Told()
    {
        var template = ChatTemplate.Parse("{% if style == 'short' %}Be brief.{% endif %}{% for m in messages %}{% if m.content == 'no' %}{{ raise_exception('not that') }}{% endif %}{{ m.content }}{% endfor %}");
        var variables = new Dictionary<string, JsonElement> { ["style"] = JsonDocument.Parse("\"short\"").RootElement };
        Assert.Equal("Be brief.hi", template.Apply([new ChatMessage("user", "hi")], variables: variables));
        Assert.Equal("hi", template.Apply("hi"));
        Assert.Contains("not that", Assert.Throws<ChatTemplateException>(() => template.Apply("no")).Message);
        Assert.Throws<ChatTemplateException>(() => ChatTemplate.Parse("{% if %}"));
    }

    /// <summary>Models' own templates, when <c>AMQL_TEMPLATES</c> names a folder of them (<c>{"template": …}</c>).</summary>
    [Fact]
    public void Real_Templates_Round_Trip()
    {
        if (Environment.GetEnvironmentVariable("AMQL_TEMPLATES") is not { Length: > 0 } dir)
        {
            return;
        }
        var problems = new List<string>();
        foreach (string file in Directory.GetFiles(dir, "*.json"))
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(file));
            string Text(string name) => doc.RootElement.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString()! : "";
            var template = ChatTemplate.Parse(Text("template"), Text("bos"), Text("eos"));
            if (template.Dialect.Calls is not null && !RoundTrips(template, out string? problem))
            {
                problems.Add(Path.GetFileName(file) + ": " + problem);
            }
        }
        Assert.True(problems.Count == 0, string.Join("\n\n", problems));
    }
}
