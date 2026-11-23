using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Cli;
using Amql.Entail;
using Amql.Hf;
using Amql.Inference;
using Amql.Server;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;

namespace Amql.Tests;

/// <summary>
/// Function calling and reasoning on amql-server: prompts with tools, calls and tool results
/// rendered exactly as transformers renders them with Qwen3.5's template
/// (fixtures/qwen35_tool_chat_golden.json, made by make_tool_chat_golden.py); replies split into
/// reasoning, content and calls, the same whether read whole or a character at a time; and the
/// chat and Responses endpoints carrying tools, forced calls and stored conversations.
/// </summary>
public sealed class ServerToolTests : IClassFixture<ServerToolTests.ToolHost>
{
    /// <summary>The tiny Qwen3.5 fixture with a 4096-token context recorded instead of 512:
    /// the tool instructions alone take more than 512 tokens. Nothing else differs.</summary>
    public sealed class ToolHost : IAsyncLifetime
    {
        private readonly TempDir _dir = new();

        public List<ModelHost> Models { get; } = new();
        public WebApplication App { get; private set; } = null!;
        public HttpClient Client { get; private set; } = null!;

        public async Task InitializeAsync()
        {
            string source = Path.Combine(_dir.Path, "src");
            Directory.CreateDirectory(source);
            foreach (string f in Directory.GetFiles(Path.Combine(ServerTests.Hosts.Fixtures, "qwen-lm-tiny")))
            {
                File.Copy(f, Path.Combine(source, Path.GetFileName(f)));
            }
            string config = Path.Combine(source, "config.json");
            File.WriteAllText(config, File.ReadAllText(config).Replace("\"max_position_embeddings\": 512", "\"max_position_embeddings\": 4096"));
            string lm = Path.Combine(_dir.Path, "lm");
            ModelToContainer.Encode(source, lm, "qwen-lm-tiny");
            Models.Add(ModelHost.Load("lm", lm));
            // The same container again under its own id and defaults: it thinks unless told not to.
            Models.Add(ModelHost.Load("lm-think", lm, defaults: new ModelDefaults(Think: true, MaxTokens: 6, Temperature: 0)));

            // A model the GPU engine runs (64-wide DeltaNet heads), so requests batch on a CUDA
            // device; and the same with an adapter that rules out a copy's next token.
            string engine = Path.Combine(_dir.Path, "engine");
            ModelToContainer.Encode(Path.Combine(ServerTests.Hosts.Fixtures, "qwen-lm-engine"), engine, "qwen-lm-engine");
            string noCopy = Path.Combine(_dir.Path, "no-copy");
            EntailSparseAdapter.Write("state again\nrepeat again over 3 times 1\nforbid repeat 3 when again", engine, noCopy);
            Models.Add(ModelHost.Load("lm-engine", engine));
            Models.Add(ModelHost.Load("lm-engine-entail", engine, new[] { noCopy }));
            App = ServerApp.Build(Models, new ServerSettings(DefaultMaxTokens: 8));
            App.Urls.Add("http://127.0.0.1:0");
            await App.StartAsync();
            string address = App.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
            Client = new HttpClient { BaseAddress = new Uri(address) };
        }

        public async Task DisposeAsync()
        {
            Client.Dispose();
            await App.StopAsync();
            await App.DisposeAsync();
            foreach (var m in Models)
            {
                m.Dispose();
            }
            _dir.Dispose();
        }
    }

    private readonly ToolHost _h;

    public ServerToolTests(ToolHost h) => _h = h;

    private static JsonElement Golden =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(ServerTests.Hosts.Fixtures, "qwen35_tool_chat_golden.json"))).RootElement;

    private static List<ChatMessage> Messages(JsonElement messages) => messages.EnumerateArray().Select(m =>
    {
        var message = new ChatMessage(m.GetProperty("role").GetString()!, m.GetProperty("content").GetString()!);
        if (m.TryGetProperty("reasoning_content", out var r))
        {
            message = message with { Reasoning = r.GetString() };
        }
        if (m.TryGetProperty("tool_calls", out var calls))
        {
            message = message with
            {
                ToolCalls = calls.EnumerateArray().Select(c => new ChatToolCall(
                    c.GetProperty("function").GetProperty("name").GetString()!,
                    c.GetProperty("function").GetProperty("arguments").Clone())).ToList(),
            };
        }
        return message;
    }).ToList();

    public static IEnumerable<object[]> Cases() =>
        Golden.GetProperty("cases").EnumerateArray().Select(c => new object[] { c.GetProperty("name").GetString()! });

    [Theory]
    [MemberData(nameof(Cases))]
    public void Tool_Conversations_Render_As_Transformers_Renders_Them(string name)
    {
        var c = Golden.GetProperty("cases").EnumerateArray().Single(x => x.GetProperty("name").GetString() == name);
        string template = File.ReadAllText(Path.Combine(ServerTests.Hosts.Fixtures, "qwen35_chat_template.jinja"));
        var tools = c.GetProperty("tools").ValueKind == JsonValueKind.Array
            ? c.GetProperty("tools").EnumerateArray().Select(t => t.Clone()).ToList()
            : null;
        string rendered = ChatTemplate.ApplyMessages(template, Messages(c.GetProperty("messages")),
            c.GetProperty("thinking").GetBoolean(), tools);
        Assert.Equal(c.GetProperty("rendered").GetString(), rendered);
    }

    [Fact]
    public void Tools_Are_Refused_For_A_Template_That_Cannot_Render_Them()
    {
        const string chatMl = "{% for m in messages %}<|im_start|>{{ m.role }}{% endfor %}";
        var tool = JsonDocument.Parse("""{"type":"function","function":{"name":"x"}}""").RootElement;
        Assert.Throws<ChatTemplateException>(() =>
            ChatTemplate.ApplyMessages(chatMl, new[] { new ChatMessage("user", "q") }, tools: new[] { tool }));
        Assert.Throws<ChatTemplateException>(() =>
            ChatTemplate.ApplyMessages(chatMl, new[] { new ChatMessage("user", "q"), new ChatMessage("tool", "r") }));
    }

    // ── reading replies ─────────────────────────────────────────────────

    private static readonly JsonElement[] Tools = JsonDocument.Parse("""
        [{"type":"function","function":{"name":"create_file","parameters":{"type":"object","properties":{
            "path":{"type":"string"},"content":{"type":"string"},"lines":{"type":"integer"},"force":{"type":"boolean"},
            "ratio":{"type":["number","null"]},"meta":{"type":"object"},"tags":{"type":"array"}}}}}]
        """).RootElement.EnumerateArray().Select(t => t.Clone()).ToArray();

    private const string AgentReply =
        "Let me think.\nThe file needs 2 lines.\n</think>\n\nI'll write it now.\n\n" +
        "<tool_call>\n<function=create_file>\n<parameter=path>\nsrc/a.py\n</parameter>\n" +
        "<parameter=content>\nprint('hi')\n\nprint(\"<tool_call>\")\n</parameter>\n" +
        "<parameter=lines>\n2\n</parameter>\n<parameter=force>\nTrue\n</parameter>\n<parameter=ratio>\n0.5\n</parameter>\n" +
        "<parameter=meta>\n{\"a\": [1, 2]}\n</parameter>\n<parameter=tags>\nnot json\n</parameter>\n" +
        "</function>\n</tool_call>\n<tool_call>\n<function=create_file>\n<parameter=path>\nb.txt\n</parameter>\n</function>\n</tool_call>\nignored suffix";

    [Fact]
    public void A_Reply_Splits_Into_Reasoning_Content_And_Typed_Calls()
    {
        var reply = ReplyParser.Parse(AgentReply, promptOpenedThink: true, Tools);
        Assert.True(reply.Thought);
        Assert.Equal("Let me think.\nThe file needs 2 lines.", reply.Reasoning);
        Assert.Equal("I'll write it now.", reply.Content);
        Assert.Equal(2, reply.ToolCalls.Count);
        Assert.Equal("create_file", reply.ToolCalls[0].Name);
        var args = JsonNode.Parse(reply.ToolCalls[0].Arguments)!.AsObject();
        Assert.Equal("src/a.py", args["path"]!.GetValue<string>());
        Assert.Equal("print('hi')\n\nprint(\"<tool_call>\")", args["content"]!.GetValue<string>());
        Assert.Equal(2, args["lines"]!.GetValue<long>());
        Assert.True(args["force"]!.GetValue<bool>());
        Assert.Equal(0.5, args["ratio"]!.GetValue<double>());
        Assert.Equal("""{"a":[1,2]}""", args["meta"]!.ToJsonString());
        Assert.Equal("not json", args["tags"]!.GetValue<string>());   // kept as text, for the client to report
        Assert.Equal("""{"path":"b.txt"}""", reply.ToolCalls[1].Arguments);
        Assert.Contains("""print(\"<tool_call>\")""", reply.ToolCalls[0].Arguments);   // "<" not escaped
    }

    [Fact]
    public void Streaming_A_Character_At_A_Time_Reads_The_Same_And_Never_Leaks_Markup()
    {
        var whole = ReplyParser.Parse(AgentReply, promptOpenedThink: true, Tools);
        var parser = new ReplyParser(true, Tools);
        var events = new List<ReplyEvent>();
        foreach (char c in AgentReply)
        {
            events.AddRange(parser.Feed(c.ToString()));
        }
        events.AddRange(parser.Finish());

        string reasoning = string.Concat(events.OfType<ReplyEvent.Reasoning>().Select(e => e.Delta));
        string content = string.Concat(events.OfType<ReplyEvent.Content>().Select(e => e.Delta));
        Assert.Equal(whole.Reasoning, reasoning);
        Assert.Equal(whole.Content, content);
        Assert.Equal(whole.ToolCalls, events.OfType<ReplyEvent.ToolCall>().Select(e => e.Call));
        Assert.Equal(new[] { 0, 1 }, events.OfType<ReplyEvent.ToolCall>().Select(e => e.Index));
        Assert.DoesNotContain("<", reasoning + content);
        // Reasoning comes first, content before the calls.
        int lastReasoning = events.FindLastIndex(e => e is ReplyEvent.Reasoning);
        int firstContent = events.FindIndex(e => e is ReplyEvent.Content);
        int firstCall = events.FindIndex(e => e is ReplyEvent.ToolCall);
        Assert.True(lastReasoning < firstContent && events.FindLastIndex(e => e is ReplyEvent.Content) < firstCall);
    }

    [Fact]
    public void A_Call_Cut_Off_By_The_Token_Limit_Is_Shown_Not_Run()
    {
        var reply = ReplyParser.Parse("Writing.\n<tool_call>\n<function=create_file>\n<parameter=content>\nwhile True:", false, Tools);
        Assert.Empty(reply.ToolCalls);
        Assert.StartsWith("Writing.\n\n<tool_call>", reply.Content);
    }

    [Fact]
    public void Without_Tools_A_Tool_Call_Is_Just_Text_And_Unclosed_Thinking_Is_All_Reasoning()
    {
        var plain = ReplyParser.Parse("Use <tool_call> tags.", false, null);
        Assert.Equal("Use <tool_call> tags.", plain.Content);
        Assert.False(plain.Thought);

        var cut = ReplyParser.Parse("still thinking about it", true, Tools);
        Assert.Equal("still thinking about it", cut.Reasoning);
        Assert.Equal("", cut.Content);

        // A model that opens its own think block although the prompt closed one.
        var own = ReplyParser.Parse("<think>\nhmm\n</think>\n\nAnswer", false, null);
        Assert.Equal("hmm", own.Reasoning);
        Assert.Equal("Answer", own.Content);
    }

    // ── HTTP ────────────────────────────────────────────────────────────

    private async Task<(HttpStatusCode Status, JsonNode Body)> Post(string path, string body)
    {
        var r = await _h.Client.PostAsync(path, new StringContent(body, Encoding.UTF8, "application/json"));
        return (r.StatusCode, JsonNode.Parse(await r.Content.ReadAsStringAsync())!);
    }

    private const string ChatTool = """{"type":"function","function":{"name":"create_file","description":"Create a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}""";

    [Fact]
    public async Task Chat_Renders_Tools_Into_The_Prompt_And_Takes_Tool_Turns()
    {
        var generator = _h.Models.Single(m => m.Id == "lm").Generator!;
        var tool = JsonDocument.Parse(ChatTool).RootElement;
        var messages = new List<ChatMessage>
        {
            new("user", "make a.txt"),
            new("assistant", "") { ToolCalls = new[] { new ChatToolCall("create_file", JsonDocument.Parse("""{"path":"a.txt"}""").RootElement) } },
            new("tool", "created"),
        };
        int expected = generator.EncodePrompt(messages, false, new[] { tool }).Length;

        var (status, body) = await Post("/v1/chat/completions", $$$"""
            {"model":"lm","max_tokens":4,"temperature":0,"tools":[{{{ChatTool}}}],"messages":[
              {"role":"user","content":"make a.txt"},
              {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"create_file","arguments":"{\"path\":\"a.txt\"}"}}]},
              {"role":"tool","tool_call_id":"call_1","content":"created"}]}
            """);
        Assert.True(status == HttpStatusCode.OK, body.ToJsonString());
        Assert.Equal(expected, body["usage"]!["prompt_tokens"]!.GetValue<int>());
    }

    [Fact]
    public async Task A_Named_Tool_Choice_Writes_The_Start_Of_The_Call_For_The_Model()
    {
        var (status, body) = await Post("/v1/chat/completions", $$$"""
            {"model":"lm","max_tokens":48,"temperature":0,"tools":[{{{ChatTool}}}],
             "tool_choice":{"type":"function","function":{"name":"create_file"}},
             "messages":[{"role":"user","content":"make a.txt"}]}
            """);
        Assert.True(status == HttpStatusCode.OK, body.ToJsonString());
        // The tiny model never closes the call, so it comes back as text: forced, not run.
        string content = body["choices"]![0]!["message"]!["content"]!.GetValue<string>();
        Assert.StartsWith("<tool_call>\n<function=create_file>", content);

        var (bad, error) = await Post("/v1/chat/completions", $$$"""
            {"model":"lm","tools":[{{{ChatTool}}}],"tool_choice":{"type":"function","function":{"name":"rm_rf"}},"messages":[{"role":"user","content":"x"}]}
            """);
        Assert.Equal(HttpStatusCode.BadRequest, bad);
        Assert.Contains("rm_rf", error["error"]!["message"]!.GetValue<string>());
    }

    [Fact]
    public async Task Thinking_Comes_Back_As_Reasoning_Content_When_Asked_For()
    {
        var (status, body) = await Post("/v1/chat/completions", """
            {"model":"lm","max_tokens":6,"temperature":0,"chat_template_kwargs":{"enable_thinking":true},
             "messages":[{"role":"user","content":"hello"}]}
            """);
        Assert.True(status == HttpStatusCode.OK, body.ToJsonString());
        var message = body["choices"]![0]!["message"]!;
        // Six tokens never close the think block: all of it is reasoning.
        Assert.False(string.IsNullOrEmpty(message["reasoning_content"]?.GetValue<string>()));
        Assert.Equal("", message["content"]!.GetValue<string>());
    }

    [Fact]
    public async Task Responses_Take_Function_Items_And_Continue_From_A_Stored_Response()
    {
        const string tool = """{"type":"function","name":"create_file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}""";
        var (s1, first) = await Post("/v1/responses", $$$"""
            {"model":"lm","input":"make a.txt","tools":[{{{tool}}}],"max_output_tokens":3,"temperature":0}
            """);
        Assert.True(s1 == HttpStatusCode.OK, first.ToJsonString());
        string id = first["id"]!.GetValue<string>();
        int firstPrompt = first["usage"]!["input_tokens"]!.GetValue<int>();

        var (s2, second) = await Post("/v1/responses", $$$"""
            {"model":"lm","previous_response_id":"{{{id}}}","tools":[{{{tool}}}],"max_output_tokens":3,"temperature":0,
             "input":[{"type":"function_call","call_id":"c1","name":"create_file","arguments":"{\"path\":\"a.txt\"}"},
                      {"type":"function_call_output","call_id":"c1","output":"created"}]}
            """);
        Assert.Equal(HttpStatusCode.OK, s2);
        Assert.Equal(id, second["previous_response_id"]!.GetValue<string>());
        Assert.True(second["usage"]!["input_tokens"]!.GetValue<int>() > firstPrompt);

        var (s3, _) = await Post("/v1/responses", """{"model":"lm","input":"x","store":false,"max_output_tokens":1}""");
        Assert.Equal(HttpStatusCode.OK, s3);
    }

    [Fact]
    public async Task Streamed_Responses_Open_And_Close_A_Reasoning_Item()
    {
        var r = await _h.Client.PostAsync("/v1/responses", new StringContent("""
            {"model":"lm","input":"hello","reasoning":{"effort":"high"},"max_output_tokens":5,"temperature":0,"stream":true}
            """, Encoding.UTF8, "application/json"));
        var events = (await r.Content.ReadAsStringAsync()).Split('\n').Where(l => l.StartsWith("event: ")).Select(l => l[7..]).ToList();
        Assert.Equal("response.created", events[0]);
        Assert.Contains("response.reasoning_text.delta", events);
        Assert.True(events.IndexOf("response.reasoning_text.done") > events.LastIndexOf("response.reasoning_text.delta"));
        Assert.Equal("response.incomplete", events[^1]);
    }

    // ── several models ──────────────────────────────────────────────────

    [Fact]
    public async Task Models_Are_Listed_With_What_They_Serve_And_Their_Defaults()
    {
        var list = JsonNode.Parse(await _h.Client.GetStringAsync("/v1/models"))!["data"]!.AsArray();
        Assert.Equal(new[] { "lm", "lm-think", "lm-engine", "lm-engine-entail" }, list.Select(m => m!["id"]!.GetValue<string>()));
        Assert.Equal(new[] { "no-copy" }, list[3]!["adapters"]!.AsArray().Select(a => a!.GetValue<string>()));
        var think = list[1]!;
        Assert.Equal(4096, think["context_length"]!.GetValue<int>());
        Assert.True(think["tools"]!.GetValue<bool>());
        Assert.True(think["defaults"]!["think"]!.GetValue<bool>());
        Assert.Equal(6, think["defaults"]!["max_tokens"]!.GetValue<int>());
        Assert.False(list[0]!["defaults"]!["think"]!.GetValue<bool>());
        Assert.Empty(list[0]!["adapters"]!.AsArray());

        var one = JsonNode.Parse(await _h.Client.GetStringAsync("/v1/models/lm-think"))!;
        Assert.Equal("lm-think", one["id"]!.GetValue<string>());
        var missing = await _h.Client.GetAsync("/v1/models/gpt-5");
        Assert.Equal(HttpStatusCode.NotFound, missing.StatusCode);
        Assert.Equal("model_not_found", JsonNode.Parse(await missing.Content.ReadAsStringAsync())!["error"]!["code"]!.GetValue<string>());
    }

    [Fact]
    public async Task Each_Model_Applies_Its_Own_Defaults_And_A_Request_Overrides_Them()
    {
        const string ask = """{"model":"lm-think","messages":[{"role":"user","content":"hello"}]""";
        var (_, thinks) = await Post("/v1/chat/completions", ask + "}");
        Assert.Equal(6, thinks["usage"]!["completion_tokens"]!.GetValue<int>());
        Assert.False(string.IsNullOrEmpty(thinks["choices"]![0]!["message"]!["reasoning_content"]?.GetValue<string>()));

        var (_, plain) = await Post("/v1/chat/completions", ask + ""","chat_template_kwargs":{"enable_thinking":false}}""");
        Assert.Null(plain["choices"]![0]!["message"]!["reasoning_content"]);

        var (status, none) = await Post("/v1/chat/completions", """{"messages":[{"role":"user","content":"x"}]}""");
        Assert.Equal(HttpStatusCode.BadRequest, status);
        Assert.Contains("name one", none["error"]!["message"]!.GetValue<string>());
    }

    [Fact]
    public async Task Two_Models_Answer_At_Once_As_They_Answer_Alone()
    {
        string Ask(string model) => $$"""{"model":"{{model}}","max_tokens":6,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"count to ten"}]}""";
        async Task<string> Text(string model) =>
            (await Post("/v1/chat/completions", Ask(model))).Body["choices"]![0]!["message"]!["content"]!.GetValue<string>();
        string alone = await Text("lm");
        var together = await Task.WhenAll(Enumerable.Range(0, 6).Select(i => Text(i % 2 == 0 ? "lm" : "lm-think")));
        Assert.All(together, t => Assert.Equal(alone, t));
    }

    [Fact]
    public void A_Config_File_Gives_Each_Model_Its_Path_Adapters_And_Defaults()
    {
        using var dir = new TempDir();
        string file = Path.Combine(dir.Path, "models.json");
        File.WriteAllText(file, """
            {
              // comments and trailing commas are fine
              "models": [
                {"id": "think", "path": "qwen", "adapters": ["adapters/stop-circling"], "think": true, "max_tokens": 4096,
                 "temperature": 0.6, "top_p": 0.95, "top_k": 20},
                {"path": "D:/models/other/"},
              ]
            }
            """);
        var models = ServerConfig.Read(file);
        Assert.Equal(2, models.Count);
        Assert.Equal("think", models[0].Id);
        Assert.Equal(Path.Combine(dir.Path, "qwen"), models[0].Path);
        Assert.Equal(new[] { Path.Combine(dir.Path, "adapters", "stop-circling") }, models[0].Adapters);
        Assert.Equal(new ModelDefaults(true, 4096, 0.6f, 0.95f, 20), models[0].Defaults);
        Assert.Equal("other", models[1].Id);
        Assert.Equal(new ModelDefaults(), models[1].Defaults);

        File.WriteAllText(file, """{"models": [{"path": "x", "temprature": 0.5}]}""");
        Assert.Contains("temprature", Assert.Throws<InvalidDataException>(() => ServerConfig.Read(file)).Message);
    }

    [Theory]
    [InlineData("lm")]
    [InlineData("lm-engine")]
    [InlineData("lm-engine-entail")]
    public async Task Many_Requests_To_One_Model_At_Once_Answer_As_Each_Does_Alone(string model)
    {
        // Different prompts, lengths, sampling and stops, so the batch changes shape as replies finish;
        // more requests than one batch holds, so some wait their turn.
        string Ask(int i) => $$$"""
            {"model":"{{{model}}}","max_tokens":{{{6 + i % 9 * 3}}},"temperature":{{{(i % 3 == 0 ? "0" : "0.8")}}},"seed":{{{i}}},
             "chat_template_kwargs":{"enable_thinking":{{{(i % 4 == 1 ? "true" : "false")}}}},
             "messages":[{"role":"user","content":"request {{{i}}}: {{{new string('x', i * 7)}}}"}]}
            """;
        async Task<string> Text(int i)
        {
            var (status, body) = await Post("/v1/chat/completions", Ask(i));
            Assert.True(status == HttpStatusCode.OK, body.ToJsonString());
            var m = body["choices"]![0]!["message"]!;
            return (m["reasoning_content"]?.GetValue<string>() ?? "") + "|" + m["content"]!.GetValue<string>();
        }
        var alone = new List<string>();
        for (int i = 0; i < 14; i++)
        {
            alone.Add(await Text(i));
        }
        var together = await Task.WhenAll(Enumerable.Range(0, 14).Select(Text));
        Assert.Equal(alone, together);

        var listed = JsonNode.Parse(await _h.Client.GetStringAsync($"/v1/models/{model}"))!;
        bool batches = model != "lm" && CudaShim.Enabled && TextGenerator.BatchingEnabled;
        Assert.True((batches ? GpuEngine.MaxBatch : 1) == listed["concurrency"]!.GetValue<int>(), listed.ToJsonString());
    }

    [Fact]
    public async Task A_Client_That_Goes_Away_Leaves_The_Batch_And_The_Others_Finish()
    {
        const string ask = """{"model":"lm-engine","max_tokens":2000,"temperature":0.8,"seed":3,"messages":[{"role":"user","content":"go"}]}""";
        using var gone = new CancellationTokenSource(TimeSpan.FromMilliseconds(30));
        var abandoned = _h.Client.PostAsync("/v1/chat/completions", new StringContent(ask, Encoding.UTF8, "application/json"), gone.Token);
        var others = Enumerable.Range(0, 3).Select(_ => Post("/v1/chat/completions", ask.Replace("2000", "20"))).ToList();
        try
        {
            await abandoned;   // the random model may well have finished first
        }
        catch (OperationCanceledException)
        {
        }
        foreach (var (status, body) in await Task.WhenAll(others))
        {
            Assert.Equal(HttpStatusCode.OK, status);
            Assert.Equal(20, body["usage"]!["completion_tokens"]!.GetValue<int>());
        }
    }
}
