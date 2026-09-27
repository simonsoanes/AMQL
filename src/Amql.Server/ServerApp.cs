using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Channels;
using Amql.Cli;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;

namespace Amql.Server;

/// <summary>A request the endpoint refuses, with the HTTP status and the
/// OpenAI error <c>type</c>/<c>code</c> to report it under.</summary>
public sealed class ApiException : Exception
{
    public int Status { get; }
    public string Type { get; }
    public string? Code { get; }
    public string? Param { get; }

    public ApiException(int status, string message, string type = "invalid_request_error", string? code = null, string? param = null)
        : base(message)
    {
        Status = status;
        Type = type;
        Code = code;
        Param = param;
    }
}

public sealed record ServerSettings(string? ApiKey = null, int DefaultMaxTokens = 1024);

/// <summary>
/// The HTTP surface of amql-server. Every endpoint is offered only for models
/// whose containers can serve it:
/// <list type="bullet">
/// <item><c>POST /v1/decisions</c>, <c>/v1/systemone</c> — TypeSafe / OpenJEV decisions (Von);
/// <c>POST /api/v1/decisions</c> — the same in jevai.org's <c>{ code, message, data }</c> envelope.</item>
/// <item><c>POST /v1/embeddings</c> — OpenAI embeddings (embedding containers).</item>
/// <item><c>POST /v1/chat/completions</c>, <c>POST /v1/responses</c> — OpenAI chat and
/// Responses, streaming or not (generative decoders with a chat template).</item>
/// <item><c>GET /v1/models</c>, <c>GET /health</c>.</item>
/// </list>
/// </summary>
public static class ServerApp
{
    private static readonly JsonSerializerOptions Json = new()
    {
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    public static WebApplication Build(IReadOnlyList<ModelHost> models, ServerSettings settings, string[]? args = null)
    {
        var builder = WebApplication.CreateSlimBuilder(args ?? Array.Empty<string>());
        builder.Logging.SetMinimumLevel(LogLevel.Warning);
        var app = builder.Build();

        app.Use(async (ctx, next) =>
        {
            try
            {
                if (settings.ApiKey is { } key &&
                    (ctx.Request.Path.StartsWithSegments("/v1") || ctx.Request.Path.StartsWithSegments("/api")))
                {
                    string auth = ctx.Request.Headers.Authorization.ToString();
                    if (!System.Security.Cryptography.CryptographicOperations.FixedTimeEquals(
                            Encoding.UTF8.GetBytes(auth), Encoding.UTF8.GetBytes($"Bearer {key}")))
                    {
                        throw new ApiException(401, "Incorrect API key provided.", "invalid_request_error", "invalid_api_key");
                    }
                }
                await next(ctx);
            }
            catch (ApiException e) when (!ctx.Response.HasStarted)
            {
                await WriteError(ctx, e);
            }
            catch (JsonException e) when (!ctx.Response.HasStarted)
            {
                await WriteError(ctx, new ApiException(400, $"the request body is not valid JSON: {e.Message}"));
            }
        });

        app.MapGet("/health", () => Results.Json(new JsonObject
        {
            ["status"] = "ok",
            ["models"] = new JsonArray(models.Select(m => (JsonNode?)m.Id).ToArray()),
        }, Json));

        app.MapGet("/v1/models", () => Results.Json(new JsonObject
        {
            ["object"] = "list",
            ["data"] = new JsonArray(models.Select(m => (JsonNode?)new JsonObject
            {
                ["id"] = m.Id,
                ["object"] = "model",
                ["created"] = m.Created,
                ["owned_by"] = "amql",
                ["capabilities"] = new JsonArray(CapabilityNames(m.Capabilities).Select(c => (JsonNode?)c).ToArray()),
            }).ToArray()),
        }, Json));

        // Cast to Delegate: a lambda over HttpContext alone otherwise binds as a
        // RequestDelegate, which discards the returned IResult (an empty 200).
        app.MapPost("/v1/decisions", (Delegate)((HttpContext ctx) => Decisions(ctx, models, envelope: false)));
        app.MapPost("/v1/systemone", (Delegate)((HttpContext ctx) => Decisions(ctx, models, envelope: false)));
        app.MapPost("/api/v1/decisions", (Delegate)((HttpContext ctx) => Decisions(ctx, models, envelope: true)));
        app.MapPost("/v1/embeddings", (Delegate)((HttpContext ctx) => Embeddings(ctx, models)));
        app.MapPost("/v1/chat/completions", (HttpContext ctx) => ChatCompletions(ctx, models, settings));
        app.MapPost("/v1/responses", (HttpContext ctx) => Responses(ctx, models, settings));
        return app;
    }

    public static IEnumerable<string> CapabilityNames(Capability c)
    {
        if (c.HasFlag(Capability.Decisions))
        {
            yield return "decisions";
        }
        if (c.HasFlag(Capability.Embeddings))
        {
            yield return "embeddings";
        }
        if (c.HasFlag(Capability.Chat))
        {
            yield return "chat.completions";
            yield return "responses";
        }
    }

    // ── routing ──────────────────────────────────────────────────────────

    /// <summary>The model a request names, which must serve the endpoint. An
    /// omitted model is allowed only when exactly one loaded model serves it.
    /// For decisions the TypeSafe <c>model</c> field names the hosted Jev, not
    /// a local id, so an unknown name falls back to the sole decisions model.</summary>
    private static ModelHost Route(IReadOnlyList<ModelHost> models, JsonElement body, Capability need, bool lenientName = false)
    {
        var capable = models.Where(m => m.Capabilities.HasFlag(need)).ToList();
        string? name = body.TryGetProperty("model", out var m) && m.ValueKind == JsonValueKind.String ? m.GetString() : null;
        if (name is not null && models.FirstOrDefault(x => x.Id == name) is { } named)
        {
            return named.Capabilities.HasFlag(need)
                ? named
                : throw new ApiException(400, $"model '{name}' does not serve {CapabilityNames(need).First()}", code: "model_not_supported", param: "model");
        }
        if ((name is null || lenientName) && capable.Count == 1)
        {
            return capable[0];
        }
        if (name is null)
        {
            throw new ApiException(400, capable.Count == 0
                ? $"no loaded model serves {CapabilityNames(need).First()}"
                : $"several models serve {CapabilityNames(need).First()} — name one ({string.Join(", ", capable.Select(c => c.Id))})",
                param: "model");
        }
        throw new ApiException(404, $"The model '{name}' does not exist", code: "model_not_found", param: "model");
    }

    private static async Task<JsonDocument> ReadBody(HttpContext ctx)
    {
        var doc = await JsonDocument.ParseAsync(ctx.Request.Body, cancellationToken: ctx.RequestAborted);
        if (doc.RootElement.ValueKind != JsonValueKind.Object)
        {
            doc.Dispose();
            throw new ApiException(400, "the request body must be a JSON object");
        }
        return doc;
    }

    private static async Task<T> Locked<T>(ModelHost model, CancellationToken cancel, Func<T> work)
    {
        await model.Gate.WaitAsync(cancel);
        try
        {
            return await Task.Run(work, cancel);
        }
        finally
        {
            model.Gate.Release();
        }
    }

    // ── decisions ────────────────────────────────────────────────────────

    private static async Task<IResult> Decisions(HttpContext ctx, IReadOnlyList<ModelHost> models, bool envelope)
    {
        JsonDocument doc;
        try
        {
            doc = await ReadBody(ctx);
        }
        catch (Exception e) when (envelope && e is JsonException or ApiException)
        {
            return Results.Json(new JsonObject { ["code"] = 422, ["message"] = e.Message, ["data"] = null }, Json, statusCode: 422);
        }
        using (doc)
        {
            var model = Route(models, doc.RootElement, Capability.Decisions, lenientName: true);
            try
            {
                var response = await Locked(model, ctx.RequestAborted, () => model.Decisions!.Decide(doc.RootElement));
                return Results.Json(envelope ? new JsonObject { ["code"] = 0, ["message"] = "ok", ["data"] = response } : response, Json);
            }
            catch (DecisionRequestException e)
            {
                return envelope
                    ? Results.Json(new JsonObject { ["code"] = 422, ["message"] = e.Message, ["data"] = null }, Json, statusCode: 422)
                    : throw new ApiException(422, e.Message);
            }
        }
    }

    // ── embeddings ───────────────────────────────────────────────────────

    private static async Task<IResult> Embeddings(HttpContext ctx, IReadOnlyList<ModelHost> models)
    {
        using var doc = await ReadBody(ctx);
        var body = doc.RootElement;
        var model = Route(models, body, Capability.Embeddings);
        var embedder = model.Embedder!;

        if (!body.TryGetProperty("input", out var input))
        {
            throw new ApiException(400, "'input' is required", param: "input");
        }
        string format = body.TryGetProperty("encoding_format", out var f) && f.ValueKind == JsonValueKind.String ? f.GetString()! : "float";
        if (format is not ("float" or "base64"))
        {
            throw new ApiException(400, "'encoding_format' must be 'float' or 'base64'", param: "encoding_format");
        }
        if (body.TryGetProperty("dimensions", out var dims) && dims.ValueKind == JsonValueKind.Number && dims.GetInt32() != embedder.Dimensions)
        {
            throw new ApiException(400,
                $"this model embeds into {embedder.Dimensions} dimensions and records no Matryoshka truncation", param: "dimensions");
        }

        var inputs = EmbeddingInputs(input, embedder);
        var vectors = await Locked(model, ctx.RequestAborted, () => inputs.Select(embedder.Embed).ToList());

        var data = new JsonArray();
        for (int i = 0; i < vectors.Count; i++)
        {
            JsonNode embedding = format == "base64"
                ? JsonValue.Create(Convert.ToBase64String(System.Runtime.InteropServices.MemoryMarshal.AsBytes(vectors[i].AsSpan())))
                : new JsonArray(vectors[i].Select(v => (JsonNode?)v).ToArray());
            data.Add(new JsonObject { ["object"] = "embedding", ["index"] = i, ["embedding"] = embedding });
        }
        int tokens = inputs.Sum(x => x.Length);
        return Results.Json(new JsonObject
        {
            ["object"] = "list",
            ["data"] = data,
            ["model"] = model.Id,
            ["usage"] = new JsonObject { ["prompt_tokens"] = tokens, ["total_tokens"] = tokens },
        }, Json);
    }

    /// <summary>OpenAI's four input shapes: a string, an array of strings, an
    /// array of token ids, an array of token-id arrays.</summary>
    private static List<int[]> EmbeddingInputs(JsonElement input, Embedder embedder)
    {
        int[] Text(JsonElement s)
        {
            string text = s.GetString()!;
            if (text.Length == 0)
            {
                throw new ApiException(400, "'input' cannot contain an empty string", param: "input");
            }
            return embedder.Encode(text);
        }
        int[] Ids(JsonElement a)
        {
            var ids = a.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.Number ? e.GetInt32()
                : throw new ApiException(400, "token arrays must hold integers", param: "input")).ToArray();
            if (ids.Length == 0)
            {
                throw new ApiException(400, "'input' cannot contain an empty token array", param: "input");
            }
            return ids;
        }

        switch (input.ValueKind)
        {
            case JsonValueKind.String:
                return new List<int[]> { Text(input) };
            case JsonValueKind.Array when input.GetArrayLength() == 0:
                throw new ApiException(400, "'input' cannot be empty", param: "input");
            case JsonValueKind.Array:
            {
                var first = input[0];
                return first.ValueKind switch
                {
                    JsonValueKind.Number => new List<int[]> { Ids(input) },
                    JsonValueKind.String => input.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.String ? Text(e)
                        : throw new ApiException(400, "'input' mixes strings and other values", param: "input")).ToList(),
                    JsonValueKind.Array => input.EnumerateArray().Select(Ids).ToList(),
                    _ => throw new ApiException(400, "'input' must be a string, an array of strings, or token arrays", param: "input"),
                };
            }
            default:
                throw new ApiException(400, "'input' must be a string, an array of strings, or token arrays", param: "input");
        }
    }

    // ── chat completions ─────────────────────────────────────────────────

    private static async Task ChatCompletions(HttpContext ctx, IReadOnlyList<ModelHost> models, ServerSettings settings)
    {
        using var doc = await ReadBody(ctx);
        var body = doc.RootElement;
        var model = Route(models, body, Capability.Chat);
        RefuseUnsupported(body, "tools", "functions", "logprobs", "top_logprobs", "audio", "prediction");
        if (body.TryGetProperty("n", out var n) && n.ValueKind == JsonValueKind.Number && n.GetInt32() != 1)
        {
            throw new ApiException(400, "only n=1 is supported", param: "n");
        }
        if (body.TryGetProperty("response_format", out var rf) && rf.ValueKind == JsonValueKind.Object &&
            rf.TryGetProperty("type", out var rft) && rft.GetString() != "text")
        {
            throw new ApiException(400, "structured output (response_format) is not supported", param: "response_format");
        }
        if (!body.TryGetProperty("messages", out var msgs) || msgs.ValueKind != JsonValueKind.Array || msgs.GetArrayLength() == 0)
        {
            throw new ApiException(400, "'messages' must be a non-empty array", param: "messages");
        }
        var messages = msgs.EnumerateArray().Select((m, i) => ChatMessageFrom(m, $"messages[{i}]")).ToList();
        int maxTokens = IntField(body, "max_completion_tokens") ?? IntField(body, "max_tokens") ?? settings.DefaultMaxTokens;
        var p = Sampling(body, maxTokens);
        bool thinking = body.TryGetProperty("chat_template_kwargs", out var kw) && kw.ValueKind == JsonValueKind.Object &&
                        kw.TryGetProperty("enable_thinking", out var et) && et.ValueKind == JsonValueKind.True;
        int[] prompt = Prompt(model, messages, thinking);

        string id = "chatcmpl-" + Guid.NewGuid().ToString("N");
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        bool stream = body.TryGetProperty("stream", out var s) && s.ValueKind == JsonValueKind.True;

        if (!stream)
        {
            var result = await Locked(model, ctx.RequestAborted, () => Run(model, prompt, p, null, ctx.RequestAborted));
            await ctx.Response.WriteAsJsonAsync(new JsonObject
            {
                ["id"] = id,
                ["object"] = "chat.completion",
                ["created"] = created,
                ["model"] = model.Id,
                ["choices"] = new JsonArray(new JsonObject
                {
                    ["index"] = 0,
                    ["message"] = new JsonObject { ["role"] = "assistant", ["content"] = result.Text, ["refusal"] = null },
                    ["logprobs"] = null,
                    ["finish_reason"] = result.FinishReason,
                }),
                ["usage"] = ChatUsage(result),
            }, Json);
            return;
        }

        bool includeUsage = body.TryGetProperty("stream_options", out var so) && so.ValueKind == JsonValueKind.Object &&
                            so.TryGetProperty("include_usage", out var iu) && iu.ValueKind == JsonValueKind.True;
        JsonObject Chunk(JsonObject delta, string? finish) => new()
        {
            ["id"] = id,
            ["object"] = "chat.completion.chunk",
            ["created"] = created,
            ["model"] = model.Id,
            ["choices"] = new JsonArray(new JsonObject
            {
                ["index"] = 0, ["delta"] = delta, ["logprobs"] = null, ["finish_reason"] = finish,
            }),
        };
        await StreamSse(ctx, model, prompt, p,
            start: w => w(null, Chunk(new JsonObject { ["role"] = "assistant", ["content"] = "" }, null)),
            delta: (w, text) => w(null, Chunk(new JsonObject { ["content"] = text }, null)),
            finish: (w, result) =>
            {
                w(null, Chunk(new JsonObject(), result.FinishReason));
                if (includeUsage)
                {
                    var usage = Chunk(new JsonObject(), null);
                    usage["choices"] = new JsonArray();
                    usage["usage"] = ChatUsage(result);
                    w(null, usage);
                }
            },
            done: true);
    }

    private static JsonObject ChatUsage(GenerationResult r) => new()
    {
        ["prompt_tokens"] = r.PromptTokens,
        ["completion_tokens"] = r.CompletionTokens,
        ["total_tokens"] = r.PromptTokens + r.CompletionTokens,
    };

    // ── responses ────────────────────────────────────────────────────────

    private static async Task Responses(HttpContext ctx, IReadOnlyList<ModelHost> models, ServerSettings settings)
    {
        using var doc = await ReadBody(ctx);
        var body = doc.RootElement;
        var model = Route(models, body, Capability.Chat);
        RefuseUnsupported(body, "tools", "previous_response_id", "conversation", "prompt", "reasoning");
        if (body.TryGetProperty("text", out var textOpt) && textOpt.ValueKind == JsonValueKind.Object &&
            textOpt.TryGetProperty("format", out var fmt) && fmt.TryGetProperty("type", out var ft) && ft.GetString() != "text")
        {
            throw new ApiException(400, "structured output (text.format) is not supported", param: "text.format");
        }

        var messages = new List<ChatMessage>();
        if (body.TryGetProperty("instructions", out var ins) && ins.ValueKind == JsonValueKind.String)
        {
            messages.Add(new ChatMessage("system", ins.GetString()!));
        }
        if (!body.TryGetProperty("input", out var input))
        {
            throw new ApiException(400, "'input' is required", param: "input");
        }
        if (input.ValueKind == JsonValueKind.String)
        {
            messages.Add(new ChatMessage("user", input.GetString()!));
        }
        else if (input.ValueKind == JsonValueKind.Array)
        {
            int i = 0;
            foreach (var item in input.EnumerateArray())
            {
                if (item.TryGetProperty("type", out var type) && type.GetString() is { } t && t != "message")
                {
                    throw new ApiException(400, $"input item type '{t}' is not supported (only messages)", param: $"input[{i}]");
                }
                var m = ChatMessageFrom(item, $"input[{i}]");
                messages.Add(m.Role == "developer" ? m with { Role = "system" } : m);
                i++;
            }
        }
        else
        {
            throw new ApiException(400, "'input' must be a string or an array of messages", param: "input");
        }
        if (messages.Count(m => m.Role == "system") > 1)
        {
            // instructions + a system/developer message: one system turn, in order.
            string merged = string.Join("\n\n", messages.Where(m => m.Role == "system").Select(m => m.Content));
            messages = messages.Where(m => m.Role != "system").Prepend(new ChatMessage("system", merged)).ToList();
        }

        int maxTokens = IntField(body, "max_output_tokens") ?? settings.DefaultMaxTokens;
        var p = Sampling(body, maxTokens);
        int[] prompt = Prompt(model, messages, enableThinking: false);

        string id = "resp_" + Guid.NewGuid().ToString("N");
        string itemId = "msg_" + Guid.NewGuid().ToString("N");
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        bool stream = body.TryGetProperty("stream", out var s) && s.ValueKind == JsonValueKind.True;

        JsonObject Message(string text, string status) => new()
        {
            ["type"] = "message",
            ["id"] = itemId,
            ["status"] = status,
            ["role"] = "assistant",
            ["content"] = new JsonArray(new JsonObject
            {
                ["type"] = "output_text", ["text"] = text, ["annotations"] = new JsonArray(),
            }),
        };
        JsonObject Response(string status, GenerationResult? result) => new()
        {
            ["id"] = id,
            ["object"] = "response",
            ["created_at"] = created,
            ["status"] = status,
            ["model"] = model.Id,
            ["output"] = result is null ? new JsonArray() : new JsonArray(Message(result.Text, status == "incomplete" ? "incomplete" : "completed")),
            ["incomplete_details"] = status == "incomplete" ? new JsonObject { ["reason"] = "max_output_tokens" } : null,
            ["instructions"] = body.TryGetProperty("instructions", out var i2) && i2.ValueKind == JsonValueKind.String ? i2.GetString() : null,
            ["max_output_tokens"] = IntField(body, "max_output_tokens"),
            ["temperature"] = p.Temperature,
            ["top_p"] = p.TopP,
            ["error"] = null,
            ["usage"] = result is null ? null : new JsonObject
            {
                ["input_tokens"] = result.PromptTokens,
                ["output_tokens"] = result.CompletionTokens,
                ["total_tokens"] = result.PromptTokens + result.CompletionTokens,
            },
        };
        static string StatusOf(GenerationResult r) => r.FinishReason == "length" ? "incomplete" : "completed";

        if (!stream)
        {
            var result = await Locked(model, ctx.RequestAborted, () => Run(model, prompt, p, null, ctx.RequestAborted));
            await ctx.Response.WriteAsJsonAsync(Response(StatusOf(result), result), Json);
            return;
        }

        int sequence = 0;
        JsonObject Event(string type, JsonObject fields)
        {
            fields["type"] = type;
            fields["sequence_number"] = sequence++;
            return fields;
        }
        var text = new StringBuilder();
        await StreamSse(ctx, model, prompt, p,
            start: w =>
            {
                w("response.created", Event("response.created", new JsonObject { ["response"] = Response("in_progress", null) }));
                w("response.in_progress", Event("response.in_progress", new JsonObject { ["response"] = Response("in_progress", null) }));
                w("response.output_item.added", Event("response.output_item.added", new JsonObject
                {
                    ["output_index"] = 0, ["item"] = Message("", "in_progress"),
                }));
                w("response.content_part.added", Event("response.content_part.added", new JsonObject
                {
                    ["item_id"] = itemId, ["output_index"] = 0, ["content_index"] = 0,
                    ["part"] = new JsonObject { ["type"] = "output_text", ["text"] = "", ["annotations"] = new JsonArray() },
                }));
            },
            delta: (w, d) =>
            {
                text.Append(d);
                w("response.output_text.delta", Event("response.output_text.delta", new JsonObject
                {
                    ["item_id"] = itemId, ["output_index"] = 0, ["content_index"] = 0, ["delta"] = d,
                }));
            },
            finish: (w, result) =>
            {
                w("response.output_text.done", Event("response.output_text.done", new JsonObject
                {
                    ["item_id"] = itemId, ["output_index"] = 0, ["content_index"] = 0, ["text"] = result.Text,
                }));
                w("response.content_part.done", Event("response.content_part.done", new JsonObject
                {
                    ["item_id"] = itemId, ["output_index"] = 0, ["content_index"] = 0,
                    ["part"] = new JsonObject { ["type"] = "output_text", ["text"] = result.Text, ["annotations"] = new JsonArray() },
                }));
                string status = StatusOf(result);
                w("response.output_item.done", Event("response.output_item.done", new JsonObject
                {
                    ["output_index"] = 0, ["item"] = Message(result.Text, status == "incomplete" ? "incomplete" : "completed"),
                }));
                string type = status == "incomplete" ? "response.incomplete" : "response.completed";
                w(type, Event(type, new JsonObject { ["response"] = Response(status, result) }));
            },
            done: false);
    }

    // ── shared generation plumbing ───────────────────────────────────────

    private static int[] Prompt(ModelHost model, IReadOnlyList<ChatMessage> messages, bool enableThinking)
    {
        try
        {
            return model.Generator!.EncodePrompt(messages, enableThinking);
        }
        catch (ChatTemplateException e)
        {
            throw new ApiException(400, e.Message, param: "messages");
        }
    }

    private static GenerationResult Run(ModelHost model, int[] prompt, GenerationParams p, Action<string>? onDelta, CancellationToken cancel)
    {
        try
        {
            return model.Generator!.Generate(prompt, p, onDelta, cancel);
        }
        catch (ArgumentException e)
        {
            throw new ApiException(400, e.Message, code: "context_length_exceeded");
        }
    }

    /// <summary>
    /// Runs a generation under the model's lock and streams it as server-sent
    /// events. Generation is CPU-bound and synchronous; it pushes events into
    /// a channel that this request drains to the socket, and a client that
    /// disconnects cancels the generation at its next token.
    /// </summary>
    private static async Task StreamSse(
        HttpContext ctx, ModelHost model, int[] prompt, GenerationParams p,
        Action<Action<string?, JsonObject>> start,
        Action<Action<string?, JsonObject>, string> delta,
        Action<Action<string?, JsonObject>, GenerationResult> finish,
        bool done)
    {
        var channel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions { SingleReader = true });
        void Write(string? eventName, JsonObject data) =>
            channel.Writer.TryWrite((eventName is null ? "" : $"event: {eventName}\n") + $"data: {data.ToJsonString(Json)}\n\n");

        await model.Gate.WaitAsync(ctx.RequestAborted);
        var work = Task.Run(() =>
        {
            try
            {
                start(Write);
                var result = model.Generator!.Generate(prompt, p, d => delta(Write, d), ctx.RequestAborted);
                finish(Write, result);
                if (done)
                {
                    channel.Writer.TryWrite("data: [DONE]\n\n");
                }
                channel.Writer.TryComplete();
            }
            catch (Exception e)
            {
                channel.Writer.TryComplete(e);
            }
            finally
            {
                model.Gate.Release();
            }
        });

        ctx.Response.ContentType = "text/event-stream";
        ctx.Response.Headers.CacheControl = "no-cache";
        try
        {
            await foreach (var chunk in channel.Reader.ReadAllAsync(ctx.RequestAborted))
            {
                await ctx.Response.WriteAsync(chunk, ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            // The status line is already sent; report the failure in-band.
            var error = new JsonObject { ["error"] = new JsonObject { ["message"] = e.Message, ["type"] = "server_error" } };
            await ctx.Response.WriteAsync($"data: {error.ToJsonString(Json)}\n\n");
        }
        await work;
    }

    private static GenerationParams Sampling(JsonElement body, int maxTokens)
    {
        if (maxTokens < 1)
        {
            throw new ApiException(400, "the token limit must be at least 1", param: "max_tokens");
        }
        float temperature = FloatField(body, "temperature") ?? 1f;
        float topP = FloatField(body, "top_p") ?? 1f;
        if (temperature is < 0 or > 2)
        {
            throw new ApiException(400, "'temperature' must be between 0 and 2", param: "temperature");
        }
        if (topP is <= 0 or > 1)
        {
            throw new ApiException(400, "'top_p' must be in (0, 1]", param: "top_p");
        }
        int seed = IntField(body, "seed") ?? Random.Shared.Next();
        var stop = new List<string>();
        if (body.TryGetProperty("stop", out var st))
        {
            if (st.ValueKind == JsonValueKind.String)
            {
                stop.Add(st.GetString()!);
            }
            else if (st.ValueKind == JsonValueKind.Array)
            {
                stop.AddRange(st.EnumerateArray().Select(e => e.GetString() ?? throw new ApiException(400, "'stop' must hold strings", param: "stop")));
            }
            if (stop.Count > 4 || stop.Any(x => x.Length == 0))
            {
                throw new ApiException(400, "'stop' takes up to 4 non-empty strings", param: "stop");
            }
        }
        return new GenerationParams(maxTokens, temperature, topP, seed, stop);
    }

    /// <summary>A chat message's role and its content flattened to text:
    /// a string, or an array of text parts (<c>text</c>, <c>input_text</c>,
    /// <c>output_text</c>). Images and audio are refused, not dropped.</summary>
    private static ChatMessage ChatMessageFrom(JsonElement m, string where)
    {
        if (m.ValueKind != JsonValueKind.Object || !m.TryGetProperty("role", out var role) || role.ValueKind != JsonValueKind.String)
        {
            throw new ApiException(400, $"{where} must be an object with a 'role'", param: where);
        }
        string r = role.GetString()! switch
        {
            "developer" => "system",
            "system" or "user" or "assistant" => role.GetString()!,
            var other => throw new ApiException(400, $"{where}: role '{other}' is not supported (system, developer, user, assistant)", param: where),
        };
        if (m.TryGetProperty("tool_calls", out _))
        {
            throw new ApiException(400, $"{where}: tool calls are not supported", param: where);
        }
        if (!m.TryGetProperty("content", out var content) || content.ValueKind == JsonValueKind.Null)
        {
            return new ChatMessage(r, "");
        }
        if (content.ValueKind == JsonValueKind.String)
        {
            return new ChatMessage(r, content.GetString()!);
        }
        if (content.ValueKind != JsonValueKind.Array)
        {
            throw new ApiException(400, $"{where}.content must be a string or an array of parts", param: where);
        }
        var sb = new StringBuilder();
        foreach (var part in content.EnumerateArray())
        {
            string type = part.TryGetProperty("type", out var t) ? t.GetString() ?? "" : "";
            if (type is not ("text" or "input_text" or "output_text") || !part.TryGetProperty("text", out var text))
            {
                throw new ApiException(400, $"{where}: content part type '{type}' is not supported (text only)", param: where);
            }
            sb.Append(text.GetString());
        }
        return new ChatMessage(r, sb.ToString());
    }

    private static void RefuseUnsupported(JsonElement body, params string[] fields)
    {
        foreach (var f in fields)
        {
            if (body.TryGetProperty(f, out var v) && v.ValueKind is not (JsonValueKind.Null or JsonValueKind.False) &&
                !(v.ValueKind == JsonValueKind.Array && v.GetArrayLength() == 0))
            {
                throw new ApiException(400, $"'{f}' is not supported by this server", param: f);
            }
        }
    }

    private static int? IntField(JsonElement body, string name) =>
        body.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : null;

    private static float? FloatField(JsonElement body, string name) =>
        body.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetSingle() : null;

    private static Task WriteError(HttpContext ctx, ApiException e)
    {
        ctx.Response.StatusCode = e.Status;
        return ctx.Response.WriteAsJsonAsync(new JsonObject
        {
            ["error"] = new JsonObject
            {
                ["message"] = e.Message,
                ["type"] = e.Type,
                ["param"] = e.Param,
                ["code"] = e.Code,
            },
        }, Json);
    }
}
