using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Cli;
using Amql.Hf;
using Amql.Merge;
using Amql.Server;
using Amql.Vindex3;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;

namespace Amql.Tests;

/// <summary>
/// amql-server and the engines behind it, against goldens from transformers
/// (fixtures/qwen-lm-tiny, made by make_qwen_lm_tiny.py) and the Von SDK
/// (fixtures/von-tiny): chat rendering with Qwen3.5's real template, greedy
/// generation token for token, mean-pooled embeddings, decisions — and the
/// HTTP surface itself, started on an ephemeral port.
/// </summary>
public sealed class ServerTests : IClassFixture<ServerTests.Hosts>
{
    public sealed class Hosts : IAsyncLifetime
    {
        private readonly TempDir _dir = new();

        public static string Fixtures => Path.Combine(AppContext.BaseDirectory, "fixtures");
        public JsonElement LmGolden { get; private set; }
        public JsonElement VonGolden { get; private set; }
        public List<ModelHost> Models { get; } = new();
        public WebApplication App { get; private set; } = null!;
        public HttpClient Client { get; private set; } = null!;
        public string LmPath => Path.Combine(_dir.Path, "lm");
        public string EmbPath => Path.Combine(_dir.Path, "emb");
        public string VonPath => Path.Combine(_dir.Path, "von");

        public async Task InitializeAsync()
        {
            ModelToContainer.Encode(Path.Combine(Fixtures, "qwen-lm-tiny"), LmPath, "qwen-lm-tiny");
            ModelConverter.ConvertToEmbedding(LmPath, EmbPath);
            ModelToContainer.Encode(Path.Combine(Fixtures, "von-tiny"), VonPath, "von-tiny");
            LmGolden = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "qwen-lm-tiny", "golden.json"))).RootElement.Clone();
            VonGolden = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "von-tiny", "golden.json"))).RootElement.Clone();

            Models.Add(ModelHost.Load("lm", LmPath));
            Models.Add(ModelHost.Load("emb", EmbPath));
            Models.Add(ModelHost.Load("von", VonPath));
            App = ServerApp.Build(Models, new ServerSettings(ApiKey: "sk-test", DefaultMaxTokens: 8));
            App.Urls.Add("http://127.0.0.1:0");
            await App.StartAsync();
            string address = App.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
            Client = new HttpClient { BaseAddress = new Uri(address) };
            Client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", "sk-test");
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

    private readonly Hosts _h;

    public ServerTests(Hosts h) => _h = h;

    private static List<ChatMessage> Messages(JsonElement msgs) =>
        msgs.EnumerateArray().Select(m => new ChatMessage(m.GetProperty("role").GetString()!, m.GetProperty("content").GetString()!)).ToList();

    // ── engines against transformers ────────────────────────────────────

    [Fact]
    public void Chat_Rendering_Matches_Hf_Apply_Chat_Template_On_Qwen35()
    {
        string template = File.ReadAllText(Path.Combine(Hosts.Fixtures, "qwen35_chat_template.jinja"));
        foreach (var c in _h.LmGolden.GetProperty("chat").EnumerateArray())
        {
            string rendered = ChatTemplate.ApplyMessages(template, Messages(c.GetProperty("messages")),
                c.GetProperty("enable_thinking").GetBoolean());
            Assert.Equal(c.GetProperty("text").GetString(), rendered);
        }
    }

    [Fact]
    public void Greedy_Generation_Matches_Transformers_Token_For_Token()
    {
        var generator = _h.Models.Single(m => m.Id == "lm").Generator!;
        foreach (var g in _h.LmGolden.GetProperty("greedy").EnumerateArray())
        {
            var prompt = generator.EncodePrompt(Messages(g.GetProperty("messages")), enableThinking: false);
            Assert.Equal(g.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()), prompt);

            var expected = g.GetProperty("generated").EnumerateArray().Select(e => e.GetInt32()).ToList();
            var result = generator.Generate(prompt, new GenerationParams(8, 0f, 1f, 0, Array.Empty<string>()), null, CancellationToken.None);
            // transformers includes the stop token it ended on; the server does not.
            Assert.Equal(expected.TakeWhile(t => t != generator.Tokenizer.AddedTokenId("<|im_end|>")), result.Tokens);
        }
    }

    [Fact]
    public void Embeddings_Match_Transformers_Mean_Pooled_And_Normalised()
    {
        var embedder = _h.Models.Single(m => m.Id == "emb").Embedder!;
        foreach (var g in _h.LmGolden.GetProperty("embed").EnumerateArray())
        {
            var ids = embedder.Encode(g.GetProperty("input").GetString()!);
            Assert.Equal(g.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()), ids);
            var v = embedder.Embed(ids);
            var expected = g.GetProperty("embedding").EnumerateArray().Select(e => e.GetDouble()).ToArray();
            for (int i = 0; i < v.Length; i++)
            {
                Assert.True(Math.Abs(expected[i] - v[i]) < 1e-5, $"[{i}] {v[i]} vs {expected[i]}");
            }
        }
    }

    [Fact]
    public void Generation_Honours_Stop_Strings_And_Keeps_Utf8_Whole()
    {
        var generator = _h.Models.Single(m => m.Id == "lm").Generator!;
        var prompt = generator.EncodePrompt(new[] { new ChatMessage("user", "hello") }, false);
        var full = generator.Generate(prompt, new GenerationParams(8, 0f, 1f, 0, Array.Empty<string>()), null, CancellationToken.None);
        Assert.True(full.Text.Length > 2);

        string stop = full.Text.Substring(1, 1);
        var deltas = new StringBuilder();
        var stopped = generator.Generate(prompt, new GenerationParams(8, 0f, 1f, 0, new[] { stop }), d => deltas.Append(d), CancellationToken.None);
        Assert.Equal("stop", stopped.FinishReason);
        Assert.Equal(full.Text[..full.Text.IndexOf(stop, StringComparison.Ordinal)], stopped.Text);
        Assert.Equal(stopped.Text, deltas.ToString());   // streamed text never runs past the stop

        // A multi-byte character split across tokens decodes whole.
        var tok = generator.Tokenizer;
        var ids = tok.EncodeToIds("😀");
        Assert.True(ids.Count > 1);
        Assert.Equal("😀", tok.DecodeText(ids));
    }

    // ── HTTP surface ────────────────────────────────────────────────────

    private async Task<(HttpStatusCode Status, JsonNode Body)> Post(string path, string body)
    {
        var r = await _h.Client.PostAsync(path, new StringContent(body, Encoding.UTF8, "application/json"));
        return (r.StatusCode, JsonNode.Parse(await r.Content.ReadAsStringAsync())!);
    }

    [Fact]
    public async Task Models_Lists_Each_Container_With_What_It_Serves()
    {
        var body = JsonNode.Parse(await _h.Client.GetStringAsync("/v1/models"))!;
        var caps = body["data"]!.AsArray().ToDictionary(m => m!["id"]!.GetValue<string>(),
            m => m!["capabilities"]!.AsArray().Select(c => c!.GetValue<string>()).ToArray());
        Assert.Equal(new[] { "chat.completions", "responses" }, caps["lm"]);
        Assert.Equal(new[] { "embeddings" }, caps["emb"]);
        Assert.Equal(new[] { "decisions" }, caps["von"]);
    }

    [Fact]
    public async Task Decisions_Serve_The_Von_Sdk_Answers_And_The_Jevai_Envelope()
    {
        var golden = _h.VonGolden.GetProperty("decisions")[0];
        string request = golden.GetProperty("request").GetRawText();
        var (status, body) = await Post("/v1/decisions", request);
        Assert.Equal(HttpStatusCode.OK, status);
        var expected = JsonNode.Parse(golden.GetProperty("response").GetRawText())!;
        Assert.Equal(expected["answers"]!["team"]!["choice"]!.GetValue<string>(), body["answers"]!["team"]!["choice"]!.GetValue<string>());
        Assert.Equal(expected["usage"]!.ToJsonString(), body["usage"]!.ToJsonString());

        var (s2, env) = await Post("/api/v1/decisions", request);
        Assert.Equal(HttpStatusCode.OK, s2);
        Assert.Equal(0, env["code"]!.GetValue<int>());
        Assert.Equal(body["answers"]!.ToJsonString(), env["data"]!["answers"]!.ToJsonString());

        var (s3, bad) = await Post("/api/v1/decisions", """{"state":"x","questions":{}}""");
        Assert.Equal((HttpStatusCode)422, s3);
        Assert.Equal(422, bad["code"]!.GetValue<int>());
    }

    [Fact]
    public async Task Chat_Completions_Stream_And_Non_Stream_Agree()
    {
        const string messages = """[{"role":"user","content":"hello how are you today"}]""";
        var (status, body) = await Post("/v1/chat/completions",
            $$"""{"model":"lm","messages":{{messages}},"max_tokens":8,"temperature":0}""");
        Assert.Equal(HttpStatusCode.OK, status);
        string content = body["choices"]![0]!["message"]!["content"]!.GetValue<string>();
        Assert.Equal("chat.completion", body["object"]!.GetValue<string>());
        Assert.Equal(8, body["usage"]!["completion_tokens"]!.GetValue<int>());

        var r = await _h.Client.PostAsync("/v1/chat/completions", new StringContent(
            $$$"""{"messages":{{{messages}}},"max_tokens":8,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}""",
            Encoding.UTF8, "application/json"));
        Assert.Equal("text/event-stream", r.Content.Headers.ContentType!.MediaType);
        var lines = (await r.Content.ReadAsStringAsync()).Split('\n').Where(l => l.StartsWith("data: ")).Select(l => l[6..]).ToList();
        Assert.Equal("[DONE]", lines[^1]);
        var chunks = lines[..^1].Select(l => JsonNode.Parse(l)!).ToList();
        string streamed = string.Concat(chunks.Where(c => c["choices"]!.AsArray().Count > 0)
            .Select(c => c["choices"]![0]!["delta"]!["content"]?.GetValue<string>() ?? ""));
        Assert.Equal(content, streamed);
        Assert.Equal("length", chunks.Last(c => c["choices"]!.AsArray().Count > 0)["choices"]![0]!["finish_reason"]!.GetValue<string>());
        Assert.Equal(8, chunks[^1]["usage"]!["completion_tokens"]!.GetValue<int>());
    }

    [Fact]
    public async Task Responses_Return_Output_Text_And_Stream_The_Event_Sequence()
    {
        var (status, body) = await Post("/v1/responses",
            """{"model":"lm","instructions":"be brief","input":"hello","max_output_tokens":5,"temperature":0}""");
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Equal("response", body["object"]!.GetValue<string>());
        Assert.Equal("incomplete", body["status"]!.GetValue<string>());
        string text = body["output"]![0]!["content"]![0]!["text"]!.GetValue<string>();

        var r = await _h.Client.PostAsync("/v1/responses", new StringContent(
            """{"instructions":"be brief","input":[{"role":"user","content":[{"type":"input_text","text":"hello"}]}],"max_output_tokens":5,"temperature":0,"stream":true}""",
            Encoding.UTF8, "application/json"));
        var events = (await r.Content.ReadAsStringAsync()).Split('\n').Where(l => l.StartsWith("event: ")).Select(l => l[7..]).ToList();
        Assert.Equal("response.created", events[0]);
        Assert.Equal("response.incomplete", events[^1]);
        Assert.Contains("response.output_text.done", events);
        Assert.NotEmpty(text);
    }

    [Fact]
    public async Task Embeddings_Endpoint_Returns_Floats_Or_Base64()
    {
        var (status, body) = await Post("/v1/embeddings", """{"model":"emb","input":["hello how are you today","😀"]}""");
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Equal(2, body["data"]!.AsArray().Count);
        var first = body["data"]![0]!["embedding"]!.AsArray().Select(n => n!.GetValue<float>()).ToArray();
        var golden = _h.LmGolden.GetProperty("embed")[0].GetProperty("embedding").EnumerateArray().Select(e => e.GetSingle()).ToArray();
        Assert.True(first.Zip(golden).All(p => Math.Abs(p.First - p.Second) < 1e-5));

        var (_, b64) = await Post("/v1/embeddings", """{"input":"hello how are you today","encoding_format":"base64"}""");
        var bytes = Convert.FromBase64String(b64["data"]![0]!["embedding"]!.GetValue<string>());
        Assert.Equal(first, System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(bytes).ToArray());
    }

    [Theory]
    [InlineData("/v1/chat/completions", """{"model":"gpt-4o","messages":[{"role":"user","content":"x"}]}""", 404, "model_not_found")]
    [InlineData("/v1/chat/completions", """{"model":"emb","messages":[{"role":"user","content":"x"}]}""", 400, "model_not_supported")]
    [InlineData("/v1/chat/completions", """{"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function"}]}""", 400, null)]
    [InlineData("/v1/chat/completions", """{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}""", 400, null)]
    [InlineData("/v1/chat/completions", """{"messages":[{"role":"user","content":"x"}],"n":2}""", 400, null)]
    [InlineData("/v1/responses", """{"input":"x","previous_response_id":"resp_1"}""", 400, null)]
    [InlineData("/v1/embeddings", """{"input":""}""", 400, null)]
    [InlineData("/v1/decisions", """{"state":"x","questions":{}}""", 422, null)]
    [InlineData("/v1/chat/completions", """not json""", 400, null)]
    public async Task Refusals_Use_The_OpenAI_Error_Shape(string path, string body, int status, string? code)
    {
        var (s, b) = await Post(path, body);
        Assert.Equal(status, (int)s);
        Assert.NotNull(b["error"]!["message"]);
        if (code is not null)
        {
            Assert.Equal(code, b["error"]!["code"]!.GetValue<string>());
        }
    }

    [Fact]
    public async Task Requests_Without_The_Api_Key_Are_Refused()
    {
        using var anonymous = new HttpClient { BaseAddress = _h.Client.BaseAddress };
        var r = await anonymous.GetAsync("/v1/models");
        Assert.Equal(HttpStatusCode.Unauthorized, r.StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await anonymous.GetAsync("/health")).StatusCode);
    }
}
