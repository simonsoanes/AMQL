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

    // ── structured output ───────────────────────────────────────────────

    private const string Verdict = """{"type":"object","properties":{"ok":{"type":"boolean"},"mode":{"type":"string","enum":["fast","exact"]},"note":{"type":["string","null"],"maxLength":40}},"required":["ok","mode","note"],"additionalProperties":false}""";

    private static void AssertVerdict(string text)
    {
        using var doc = JsonDocument.Parse(text);   // a random model's reply, still valid JSON
        var o = doc.RootElement;
        Assert.Equal(new[] { "ok", "mode", "note" }, o.EnumerateObject().Select(p => p.Name));
        Assert.Contains(o.GetProperty("ok").ValueKind, new[] { JsonValueKind.True, JsonValueKind.False });
        Assert.Contains(o.GetProperty("mode").GetString(), new[] { "fast", "exact" });
        Assert.Contains(o.GetProperty("note").ValueKind, new[] { JsonValueKind.String, JsonValueKind.Null });
    }

    [Theory]
    [InlineData("lm")]
    [InlineData("lm-engine")]
    public async Task Structured_Output_Holds_Even_A_Random_Model_To_The_Schema(string model)
    {
        for (int seed = 0; seed < 4; seed++)
        {
            var (status, chat) = await Post("/v1/chat/completions", $$$"""
                {"model":"{{{model}}}","max_tokens":300,"temperature":1,"seed":{{{seed}}},"chat_template_kwargs":{"enable_thinking":true},
                 "response_format":{"type":"json_schema","json_schema":{"name":"verdict","schema":{{{Verdict}}},"strict":true}},
                 "messages":[{"role":"user","content":"judge this"}]}
                """);
            Assert.True(status == HttpStatusCode.OK, chat.ToJsonString());
            Assert.Equal("stop", chat["choices"]![0]!["finish_reason"]!.GetValue<string>());
            Assert.Null(chat["choices"]![0]!["message"]!["reasoning_content"]);   // no thinking block in a structured reply
            AssertVerdict(chat["choices"]![0]!["message"]!["content"]!.GetValue<string>());
        }

        var (s2, resp) = await Post("/v1/responses", $$$"""
            {"model":"{{{model}}}","max_output_tokens":300,"temperature":1,"seed":9,"input":"judge this",
             "text":{"format":{"type":"json_schema","name":"verdict","schema":{{{Verdict}}},"strict":true} } }
            """);
        Assert.True(s2 == HttpStatusCode.OK, resp.ToJsonString());
        Assert.Equal("completed", resp["status"]!.GetValue<string>());
        AssertVerdict(resp["output"]!.AsArray().Single(o => o!["type"]!.GetValue<string>() == "message")!["content"]![0]!["text"]!.GetValue<string>());

        var (s3, obj) = await Post("/v1/chat/completions", $$$"""
            {"model":"{{{model}}}","max_tokens":200,"temperature":0,"response_format":{"type":"json_object"},"messages":[{"role":"user","content":"x"}]}
            """);
        Assert.True(s3 == HttpStatusCode.OK, obj.ToJsonString());
        string text = obj["choices"]![0]!["message"]!["content"]!.GetValue<string>();
        if (obj["choices"]![0]!["finish_reason"]!.GetValue<string>() == "stop")
        {
            Assert.Equal(JsonValueKind.Object, JsonDocument.Parse(text).RootElement.ValueKind);
        }
        else
        {
            Assert.StartsWith("{", text.TrimStart());   // a free object may run to the token limit
        }
    }

    [Fact]
    public async Task A_Bad_Schema_Is_Refused()
    {
        var (status, body) = await Post("/v1/chat/completions", """
            {"model":"lm","response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"$ref":"https://elsewhere/s.json"}}},"messages":[{"role":"user","content":"x"}]}
            """);
        Assert.Equal(HttpStatusCode.BadRequest, status);
        Assert.Contains("$ref", body["error"]!["message"]!.GetValue<string>());
    }

    // ── the loop guard ──────────────────────────────────────────────────

    [Fact]
    public void The_Loop_Guard_Finds_A_Reply_Repeating_Itself_And_Nothing_Else()
    {
        var rng = new Random(5);
        var prefix = Enumerable.Range(0, 60).Select(_ => rng.Next(1000)).ToList();
        var line = Enumerable.Range(0, 23).Select(_ => rng.Next(1000)).ToList();   // a 23-token line

        // The same line again and again: found once 4 rounds cover 192 tokens (9 rounds, 207).
        var looping = prefix.Concat(Enumerable.Repeat(line, 12).SelectMany(x => x)).ToList();
        int first = Enumerable.Range(1, looping.Count).First(n => LoopGuard.Find(looping.Take(n).ToList()) is not null);
        Assert.Equal(60 + 192, first);
        Assert.Equal((23, 8), LoopGuard.Find(looping.Take(first).ToList()));

        // A one-token stutter is a loop too.
        Assert.Equal(1, LoopGuard.Find(prefix.Concat(Enumerable.Repeat(7, 200)).ToList())?.Period);

        // Lines that differ in one token each time (a counter, a name) are not a loop ...
        var numbered = prefix.Concat(Enumerable.Range(0, 40).SelectMany(i => line.Take(22).Append(5000 + i))).ToList();
        Assert.Null(LoopGuard.Find(numbered));
        // ... nor a stretch repeated only 3 times, nor a long one under 192 tokens in all.
        Assert.Null(LoopGuard.Find(prefix.Concat(Enumerable.Repeat(Enumerable.Range(0, 70).ToList(), 3).SelectMany(x => x)).ToList()));
        Assert.Null(LoopGuard.Find(prefix.Concat(Enumerable.Repeat(line, 8).SelectMany(x => x)).ToList()));
        // ... and nothing before `from` counts (after thinking was closed for circling).
        Assert.Null(LoopGuard.Find(looping, from: looping.Count - 100));

        // A counting loop: the numbered line keyed loosely, it is found, but only once it is long.
        var counted = numbered.Select(t => t >= 5000 ? LoopGuard.Counter : t).ToList();
        int found = Enumerable.Range(1, counted.Count).First(n => LoopGuard.FindCounting(counted.Take(n).ToList()) is not null);
        Assert.True(found >= 60 + 16 * 23, $"found at {found}");
        var shortList = counted.Take(60 + 10 * 23).ToList();   // ten numbered lines: a list, not a loop
        Assert.Null(LoopGuard.FindCounting(shortList));
        Assert.True(LoopGuard.IsCounter("2196") && LoopGuard.IsCounter("c74105") && LoopGuard.IsCounter(" 7"));
        Assert.False(LoopGuard.IsCounter("face") || LoopGuard.IsCounter("L") || LoopGuard.IsCounter(" "));
    }

    // ── the tool-call grammar over HTTP ─────────────────────────────────

    private const string SwitchTool = """{"type":"function","function":{"name":"set_switch","parameters":{"type":"object","properties":{"on":{"type":"boolean"},"mode":{"type":"string","enum":["fast","slow"]},"level":{"type":"integer"}},"required":["on","mode"]}}}""";

    [Theory]
    [InlineData("lm")]
    [InlineData("lm-engine")]
    public async Task A_Required_Call_From_A_Random_Model_Is_Still_A_Call_The_Tool_Declares(string model)
    {
        for (int seed = 0; seed < 4; seed++)
        {
            var (status, body) = await Post("/v1/chat/completions", $$$"""
                {"model":"{{{model}}}","max_tokens":200,"temperature":1,"seed":{{{seed}}},"tools":[{{{SwitchTool}}}],"tool_choice":"required",
                 "messages":[{"role":"user","content":"flip it"}]}
                """);
            Assert.True(status == HttpStatusCode.OK, body.ToJsonString());
            var choice = body["choices"]![0]!;
            Assert.True(choice["finish_reason"]!.GetValue<string>() == "tool_calls", body.ToJsonString());
            var call = choice["message"]!["tool_calls"]![0]!["function"]!;
            Assert.Equal("set_switch", call["name"]!.GetValue<string>());
            using var args = JsonDocument.Parse(call["arguments"]!.GetValue<string>());
            Assert.Contains(args.RootElement.GetProperty("on").ValueKind, new[] { JsonValueKind.True, JsonValueKind.False });
            Assert.Contains(args.RootElement.GetProperty("mode").GetString(), new[] { "fast", "slow" });
            if (args.RootElement.TryGetProperty("level", out var level))
            {
                Assert.True(level.ValueKind == JsonValueKind.Number, call["arguments"]!.GetValue<string>());
            }
        }
    }

    [Fact]
    public async Task Continue_Final_Message_Carries_On_From_A_Started_Reply()
    {
        var g = _h.Models.Single(m => m.Id == "lm").Generator!;
        // The prompt is the conversation's generation prompt plus the started reply, token for token.
        string started = "The capital of france is";
        string rendered = ChatTemplate.ApplyMessages(g.Template, new[] { new ChatMessage("user", "hello") }, false) + started;
        int expected = g.Tokenizer.EncodeToIds(rendered).Count;
        var (status, body) = await Post("/v1/chat/completions", $$$"""
            {"model":"lm","max_tokens":4,"temperature":0,"continue_final_message":true,
             "messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"{{{started}}}"}]}
            """);
        Assert.True(status == HttpStatusCode.OK, body.ToJsonString());
        Assert.Equal(expected, body["usage"]!["prompt_tokens"]!.GetValue<int>());

        var (bad, _) = await Post("/v1/chat/completions", """
            {"model":"lm","continue_final_message":true,"messages":[{"role":"user","content":"hello"}]}
            """);
        Assert.Equal(HttpStatusCode.BadRequest, bad);
    }

    [Fact]
    public async Task A_Pack_Dropped_Into_A_Running_Models_Ngram_Folder_Is_Served()
    {
        var host = _h.Models.Single(m => m.Id == "lm-engine");
        var tokenizer = host.Generator!.Tokenizer;
        var docs = new[] { tokenizer.EncodeToIds("Console.WriteLine(\"hi\"); Console.WriteLine(\"hi\");").ToArray() };
        string folder = Path.Combine(host.Path, Amql.Server.NGramLibrary.Folder);
        Directory.CreateDirectory(folder);
        string file = Path.Combine(folder, "hello.ngram");
        Amql.Inference.NGram.NGramPack.Build("hello", docs, 4, ModelHost.TokenizerFingerprint(tokenizer)).Save(file + ".tmp");
        File.Move(file + ".tmp", file);
        try
        {
            JsonNode listed = null!;
            for (int i = 0; i < 100; i++)
            {
                listed = JsonNode.Parse(await _h.Client.GetStringAsync("/v1/models/lm-engine"))!;
                if (listed["ngram_packs"]!.AsArray().Any(n => n!.GetValue<string>() == "hello"))
                {
                    break;
                }
                await Task.Delay(100);
            }
            Assert.Contains("hello", listed["ngram_packs"]!.AsArray().Select(n => n!.GetValue<string>()));
            var (status, body) = await Post("/v1/chat/completions", """
                {"model":"lm-engine","max_tokens":8,"ngram_packs":["hello"],"ngram_lambda":0.5,"ngram_scope":"all","messages":[{"role":"user","content":"x"}]}
                """);
            Assert.True(status == HttpStatusCode.OK, body.ToJsonString());
        }
        finally
        {
            File.Delete(file);
        }
        for (int i = 0; i < 100; i++)
        {
            var listed = JsonNode.Parse(await _h.Client.GetStringAsync("/v1/models/lm-engine"))!;
            if (!listed["ngram_packs"]!.AsArray().Any(n => n!.GetValue<string>() == "hello"))
            {
                return;
            }
            await Task.Delay(100);
        }
        Assert.Fail("the pack was not dropped after its file was deleted");
    }

    [Theory]
    [InlineData("lm")]
    [InlineData("lm-engine")]
    public async Task A_Reply_That_Announces_An_Action_Opens_The_Call(string model)
    {
        // The reply so far says what it will do and stops on a colon. Where the model gives a call any
        // chance, the call opens, and an opening once begun is finished (this tokenizer writes
        // "<tool_call>" a character at a time).
        // A random model gives "<" little probability, so the floor is lowered here to see the rule act.
        double floor = TextGenerator.AnnounceFloor;
        TextGenerator.AnnounceFloor = 1e-9;
        int opened = 0;
        try
        {
        for (int seed = 0; seed < 8; seed++)
        {
            var (status, body) = await Post("/v1/chat/completions", $$$"""
                {"model":"{{{model}}}","max_tokens":120,"temperature":1,"seed":{{{seed}}},"tools":[{{{SwitchTool}}}],"continue_final_message":true,
                 "messages":[{"role":"user","content":"flip it"},{"role":"assistant","content":"Now let me flip the switch:"}]}
                """);
            Assert.True(status == HttpStatusCode.OK, body.ToJsonString());
            var message = body["choices"]![0]!["message"]!;
            string content = (message["content"]?.GetValue<string>() ?? "").TrimStart();
            bool called = message["tool_calls"] is JsonArray { Count: > 0 };
            if (content.StartsWith('<'))
            {
                Assert.True(called || content.StartsWith("<tool_call>"), body.ToJsonString());
            }
            opened += called || content.StartsWith("<tool_call>") ? 1 : 0;
        }
        }
        finally
        {
            TextGenerator.AnnounceFloor = floor;
        }
        Assert.True(opened == 8, $"only {opened} of 8 replies opened the call");

        // A colon that introduces a list, not an action: no call is forced, but the reply goes on.
        TextGenerator.AnnounceFloor = 1e-9;
        try
        {
            for (int seed = 0; seed < 4; seed++)
            {
                var (status, body) = await Post("/v1/chat/completions", $$$"""
                    {"model":"{{{model}}}","max_tokens":12,"temperature":1,"seed":{{{seed}}},"tools":[{{{SwitchTool}}}],"continue_final_message":true,
                     "messages":[{"role":"user","content":"flip it"},{"role":"assistant","content":"The switch is flipped. Here's what was done:"}]}
                    """);
                Assert.True(status == HttpStatusCode.OK, body.ToJsonString());
                var message = body["choices"]![0]!["message"]!;
                string content = message["content"]?.GetValue<string>() ?? "";
                bool called = message["tool_calls"] is JsonArray { Count: > 0 };
                Assert.True(!called && content.Trim().Length > 0, body.ToJsonString());
                Assert.True(body["usage"]!["completion_tokens"]!.GetValue<int>() > 1, body.ToJsonString());
            }
        }
        finally
        {
            TextGenerator.AnnounceFloor = floor;
        }

        // A colon inside a code block is code, not an announcement.
        var (s2, plain) = await Post("/v1/chat/completions", $$$"""
            {"model":"{{{model}}}","max_tokens":3,"temperature":0,"tools":[{{{SwitchTool}}}],"continue_final_message":true,
             "messages":[{"role":"user","content":"x"},{"role":"assistant","content":"```python\ndef f():"}]}
            """);
        Assert.Equal(HttpStatusCode.OK, s2);
        Assert.False(plain["choices"]![0]!["message"]!["content"]!.GetValue<string>().TrimStart().StartsWith("<tool_call>"));
    }
}
