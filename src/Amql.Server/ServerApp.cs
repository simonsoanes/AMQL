using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Channels;
using Amql.Cli;
using Amql.Hf;
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

/// <param name="Think">Think before answering when a request does not say.</param>
/// <param name="Temperature">Sampling defaults for requests that leave them out.</param>
public sealed record ServerSettings(string? ApiKey = null, int DefaultMaxTokens = 1024, bool Think = false,
    float? Temperature = null, float? TopP = null, int? TopK = null);

/// <summary>Responses kept for <c>previous_response_id</c>: each response's
/// conversation (its instructions excluded), the newest few hundred, in memory.</summary>
public sealed class ResponseStore
{
    private readonly int _capacity;
    private readonly Dictionary<string, IReadOnlyList<ChatMessage>> _kept = new(StringComparer.Ordinal);
    private readonly Queue<string> _order = new();

    public ResponseStore(int capacity = 512) => _capacity = capacity;

    public IReadOnlyList<ChatMessage>? Get(string id)
    {
        lock (_kept)
        {
            return _kept.GetValueOrDefault(id);
        }
    }

    public void Put(string id, IReadOnlyList<ChatMessage> conversation)
    {
        lock (_kept)
        {
            _kept[id] = conversation;
            _order.Enqueue(id);
            while (_order.Count > _capacity)
            {
                _kept.Remove(_order.Dequeue());
            }
        }
    }
}

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

        // One line per request, whatever its path: a client calling the wrong URL shows up here.
        app.Use(async (ctx, next) =>
        {
            var clock = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                await next(ctx);
            }
            finally
            {
                Log($"{ctx.Connection.RemoteIpAddress} {ctx.Request.Method} {ctx.Request.Path}{ctx.Request.QueryString} -> {ctx.Response.StatusCode} ({clock.ElapsedMilliseconds} ms)");
            }
        });

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
                Log($"{ctx.Request.Method} {ctx.Request.Path} -> {e.Status} {e.Code ?? e.Type}: {e.Message}");
                await WriteError(ctx, e);
            }
            catch (JsonException e) when (!ctx.Response.HasStarted)
            {
                Log($"{ctx.Request.Method} {ctx.Request.Path} -> 400 invalid JSON: {e.Message}");
                await WriteError(ctx, new ApiException(400, $"the request body is not valid JSON: {e.Message}"));
            }
            catch (Exception e) when (e is not OperationCanceledException && !ctx.Response.HasStarted)
            {
                // A failure of the server's own: logged whole, and reported to the client in the API's shape.
                Log($"{ctx.Request.Method} {ctx.Request.Path} -> 500{Environment.NewLine}{e}");
                await WriteError(ctx, new ApiException(500, e.Message, "server_error"));
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
            ["data"] = new JsonArray(models.Select(m => (JsonNode?)ModelJson(m, settings)).ToArray()),
        }, Json));
        app.MapGet("/v1/models/{id}", (string id) => models.FirstOrDefault(m => m.Id == id) is { } model
            ? Results.Json(ModelJson(model, settings), Json)
            : throw new ApiException(404, $"The model '{id}' does not exist", code: "model_not_found", param: "model"));

        // Cast to Delegate: a lambda over HttpContext alone otherwise binds as a
        // RequestDelegate, which discards the returned IResult (an empty 200).
        app.MapPost("/v1/decisions", (Delegate)((HttpContext ctx) => Decisions(ctx, models, envelope: false)));
        app.MapPost("/v1/systemone", (Delegate)((HttpContext ctx) => Decisions(ctx, models, envelope: false)));
        app.MapPost("/api/v1/decisions", (Delegate)((HttpContext ctx) => Decisions(ctx, models, envelope: true)));
        app.MapPost("/v1/embeddings", (Delegate)((HttpContext ctx) => Embeddings(ctx, models)));
        app.MapPost("/v1/classify", (Delegate)((HttpContext ctx) => Classify(ctx, models)));
        app.MapPost("/v1/chat/completions", (HttpContext ctx) => ChatCompletions(ctx, models, settings));
        var responses = new ResponseStore();
        app.MapPost("/v1/responses", (HttpContext ctx) => Responses(ctx, models, settings, responses));
        return app;
    }

    /// <summary>A model as <c>/v1/models</c> lists it: OpenAI's fields, then
    /// what it serves and, for a chat model, its context, whether it takes
    /// tools and thinks, its adapters, and the defaults it applies.</summary>
    private static JsonObject ModelJson(ModelHost m, ServerSettings settings)
    {
        var o = new JsonObject
        {
            ["id"] = m.Id,
            ["object"] = "model",
            ["created"] = m.Created,
            ["owned_by"] = "amql",
            ["capabilities"] = new JsonArray(CapabilityNames(m.Capabilities).Select(c => (JsonNode?)c).ToArray()),
        };
        if (m.Generator is { } g)
        {
            o["context_length"] = g.ContextLength == int.MaxValue ? null : g.ContextLength;
            o["tools"] = g.RendersTools;
            o["thinking"] = g.Thinks;
            o["adapters"] = new JsonArray(m.AdapterNames.Select(a => (JsonNode?)a).ToArray());
            o["concurrency"] = g.Concurrency;
            if (g.BatchRefusal is { } why)
            {
                o["batching"] = why;
            }
            o["defaults"] = new JsonObject
            {
                ["think"] = g.Thinks && Think(m, settings),
                ["max_tokens"] = MaxTokens(m, settings),
                ["temperature"] = m.Defaults.Temperature ?? settings.Temperature ?? 1f,
                ["top_p"] = m.Defaults.TopP ?? settings.TopP ?? 1f,
                ["top_k"] = m.Defaults.TopK ?? settings.TopK ?? 0,
            };
        }
        return o;
    }

    private static bool Think(ModelHost m, ServerSettings settings) => m.Defaults.Think ?? settings.Think;

    /// <summary>A model that thinks by default gets room to think: 4096 unless it or the server says otherwise.</summary>
    private static int MaxTokens(ModelHost m, ServerSettings settings) =>
        m.Defaults.MaxTokens ?? (m.Defaults.Think == true && !settings.Think && settings.DefaultMaxTokens < 4096 ? 4096 : settings.DefaultMaxTokens);

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
        if (c.HasFlag(Capability.Classification))
        {
            yield return "classification";
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

    /// <summary>A line on the server's error stream, timestamped.</summary>
    private static void Log(string line) => Console.Error.WriteLine($"{DateTime.Now:HH:mm:ss.fff} {line}");

    // AMQL_SERVER_DUMP=<dir>: every request body is written there, to replay a client's request exactly.
    private static readonly string? DumpDir = Environment.GetEnvironmentVariable("AMQL_SERVER_DUMP");

    private static async Task<JsonDocument> ReadBody(HttpContext ctx)
    {
        if (DumpDir is not null)
        {
            using var copy = new MemoryStream();
            await ctx.Request.Body.CopyToAsync(copy, ctx.RequestAborted);
            Directory.CreateDirectory(DumpDir);
            string file = Path.Combine(DumpDir, $"{DateTime.Now:HHmmss-fff}-{ctx.Request.Path.Value?.Trim('/').Replace('/', '-')}.json");
            await File.WriteAllBytesAsync(file, copy.ToArray());
            Log($"{ctx.Request.Method} {ctx.Request.Path}: request saved to {file}");
            copy.Position = 0;
            var dumped = await JsonDocument.ParseAsync(copy, cancellationToken: ctx.RequestAborted);
            if (dumped.RootElement.ValueKind != JsonValueKind.Object)
            {
                dumped.Dispose();
                throw new ApiException(400, "the request body must be a JSON object");
            }
            return dumped;
        }
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

    // ── classify ─────────────────────────────────────────────────────────

    private static async Task<IResult> Classify(HttpContext ctx, IReadOnlyList<ModelHost> models)
    {
        using var doc = await ReadBody(ctx);
        var body = doc.RootElement;
        var model = Route(models, body, Capability.Classification);

        if (!body.TryGetProperty("input", out var input))
        {
            throw new ApiException(400, "'input' is required", param: "input");
        }
        string format = body.TryGetProperty("format", out var f) && f.ValueKind == JsonValueKind.String ? f.GetString()! : "json";

        bool isBatch = input.ValueKind == JsonValueKind.Array &&
                       input.GetArrayLength() > 0 && input[0].ValueKind == JsonValueKind.Object;
        var inputs = isBatch
            ? input.EnumerateArray().Select(ParseClassificationInput).ToList()
            : new List<ClassificationInput> { ParseClassificationInput(input) };

        var results = await Locked(model, ctx.RequestAborted, () => inputs.Select(model.Classifier!.Classify).ToList());

        if (format == "labels")
        {
            return Results.Json(results.Select(r => (JsonNode?)JsonValue.Create(model.Classifier!.LabelLine(r))!).ToArray(), Json);
        }
        if (format == "csv")
        {
            var sb = new StringBuilder();
            sb.AppendLine(string.Join(",", model.Classifier!.Labels));
            foreach (var r in results)
            {
                sb.AppendLine(string.Join(",", r["scores"]!.AsObject().Select(kv => kv.Value!.GetValue<double>().ToString("R"))));
            }
            return Results.Text(sb.ToString(), "text/csv");
        }
        if (format == "jsonl")
        {
            var sb = new StringBuilder();
            foreach (var r in results)
            {
                sb.AppendLine(r.ToJsonString(Json));
            }
            return Results.Text(sb.ToString(), "application/x-jsonlines");
        }
        // Default: json — a single object for one input, an array for many.
        if (results.Count == 1)
        {
            return Results.Json(results[0], Json);
        }
        return Results.Json(results, Json);
    }

    private static ClassificationInput ParseClassificationInput(JsonElement e)
    {
        if (e.ValueKind == JsonValueKind.String)
        {
            return ClassificationInput.Single(e.GetString()!);
        }
        if (e.ValueKind != JsonValueKind.Object)
        {
            throw new ApiException(400, "'input' must be a string, an object with premise/hypothesis, or an array of them", param: "input");
        }
        if (e.TryGetProperty("premise", out var p) && p.ValueKind == JsonValueKind.String &&
            e.TryGetProperty("hypothesis", out var h) && h.ValueKind == JsonValueKind.String)
        {
            return ClassificationInput.Pair(p.GetString()!, h.GetString()!);
        }
        if (e.TryGetProperty("text", out var t) && t.ValueKind == JsonValueKind.String)
        {
            return ClassificationInput.Single(t.GetString()!);
        }
        throw new ApiException(400, "an input object needs 'text' or both 'premise' and 'hypothesis'", param: "input");
    }

    // ── chat completions ─────────────────────────────────────────────────

    private static async Task ChatCompletions(HttpContext ctx, IReadOnlyList<ModelHost> models, ServerSettings settings)
    {
        using var doc = await ReadBody(ctx);
        var body = doc.RootElement;
        var model = Route(models, body, Capability.Chat);
        RefuseUnsupported(body, "functions", "audio", "prediction");
        if (body.TryGetProperty("n", out var n) && n.ValueKind == JsonValueKind.Number && n.GetInt32() != 1)
        {
            throw new ApiException(400, "only n=1 is supported", param: "n");
        }
        var format = OutputFormat(body.TryGetProperty("response_format", out var rf) ? rf : default, chat: true);
        if (!body.TryGetProperty("messages", out var msgs) || msgs.ValueKind != JsonValueKind.Array || msgs.GetArrayLength() == 0)
        {
            throw new ApiException(400, "'messages' must be a non-empty array", param: "messages");
        }
        var messages = msgs.EnumerateArray().Select((m, i) => ChatMessageFrom(m, $"messages[{i}]")).ToList();
        int maxTokens = IntField(body, "max_completion_tokens") ?? IntField(body, "max_tokens") ?? MaxTokens(model, settings);

        // logprobs: accepts both { "logprobs": true } and { "logprobs": true, "top_logprobs": K }.
        int? logprobsTopK = null;
        if (body.TryGetProperty("logprobs", out var lp))
        {
            if (lp.ValueKind is JsonValueKind.True)
            {
                logprobsTopK = body.TryGetProperty("top_logprobs", out var tlp) &&
                               tlp.ValueKind == JsonValueKind.Number
                    ? Math.Clamp(tlp.GetInt32(), 1, 20)
                    : 5;
            }
            else if (lp.ValueKind == JsonValueKind.Number)
            {
                logprobsTopK = Math.Clamp(lp.GetInt32(), 1, 20);
            }
        }

        var tools = ChatTools(body, model);
        var turn = Turn(model, body, settings, messages, tools, maxTokens, ChatEffort(body), format);

        string id = "chatcmpl-" + Guid.NewGuid().ToString("N");
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        bool stream = body.TryGetProperty("stream", out var s) && s.ValueKind == JsonValueKind.True;
        var callIds = new List<string>();
        string CallId(int index)
        {
            while (callIds.Count <= index)
            {
                callIds.Add("call_" + Guid.NewGuid().ToString("N")[..24]);
            }
            return callIds[index];
        }

        if (!stream)
        {
            var (result, reply) = await Locked(model, ctx.RequestAborted, () => Run(model, turn, null, ctx.RequestAborted, logprobsTopK));
            var message = new JsonObject
            {
                ["role"] = "assistant",
                ["content"] = reply.Content.Length == 0 && reply.ToolCalls.Count > 0 ? null : reply.Content,
                ["refusal"] = null,
            };
            if (reply.Thought && reply.Reasoning.Length > 0)
            {
                message["reasoning_content"] = reply.Reasoning;
            }
            if (reply.ToolCalls.Count > 0)
            {
                message["tool_calls"] = new JsonArray(reply.ToolCalls.Select((c, i) => (JsonNode?)ToolCallJson(c, CallId(i), null)).ToArray());
            }
            await ctx.Response.WriteAsJsonAsync(new JsonObject
            {
                ["id"] = id,
                ["object"] = "chat.completion",
                ["created"] = created,
                ["model"] = model.Id,
                ["choices"] = new JsonArray(new JsonObject
                {
                    ["index"] = 0,
                    ["message"] = message,
                    ["logprobs"] = ChatLogprobsJson(result, model.Generator!.Tokenizer),
                    ["finish_reason"] = FinishReason(result, reply),
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
        await StreamSse(ctx, model, write =>
        {
            write(null, Chunk(new JsonObject { ["role"] = "assistant", ["content"] = "" }, null));
            var (result, reply) = Run(model, turn, e =>
            {
                switch (e)
                {
                    case ReplyEvent.Reasoning r:
                        write(null, Chunk(new JsonObject { ["reasoning_content"] = r.Delta }, null));
                        break;
                    case ReplyEvent.Content c:
                        write(null, Chunk(new JsonObject { ["content"] = c.Delta }, null));
                        break;
                    case ReplyEvent.ToolCall t:
                        write(null, Chunk(new JsonObject
                        {
                            ["tool_calls"] = new JsonArray(ToolCallJson(t.Call, CallId(t.Index), t.Index)),
                        }, null));
                        break;
                }
            }, ctx.RequestAborted, logprobsTopK);
            write(null, Chunk(new JsonObject(), FinishReason(result, reply)));
            if (includeUsage)
            {
                var usage = Chunk(new JsonObject(), null);
                usage["choices"] = new JsonArray();
                usage["usage"] = ChatUsage(result);
                write(null, usage);
            }
            return true;
        });
    }

    private static JsonObject ToolCallJson(ParsedToolCall call, string callId, int? index)
    {
        var o = new JsonObject();
        if (index is { } i)
        {
            o["index"] = i;
        }
        o["id"] = callId;
        o["type"] = "function";
        o["function"] = new JsonObject { ["name"] = call.Name, ["arguments"] = call.Arguments };
        return o;
    }

    private static string FinishReason(GenerationResult result, ReplyParser reply) =>
        reply.ToolCalls.Count > 0 ? "tool_calls" : result.FinishReason;

    private static JsonObject ChatUsage(GenerationResult r) => new()
    {
        ["prompt_tokens"] = r.PromptTokens,
        ["completion_tokens"] = r.CompletionTokens,
        ["total_tokens"] = r.PromptTokens + r.CompletionTokens,
    };

    /// <summary>A chat request's tools, as the template renders them: each
    /// <c>{"type":"function","function":{…}}</c> exactly as the client sent it.</summary>
    private static List<JsonElement>? ChatTools(JsonElement body, ModelHost model)
    {
        if (!body.TryGetProperty("tools", out var tools) || tools.ValueKind == JsonValueKind.Null)
        {
            return null;
        }
        if (tools.ValueKind != JsonValueKind.Array)
        {
            throw new ApiException(400, "'tools' must be an array", param: "tools");
        }
        var list = new List<JsonElement>();
        int i = 0;
        foreach (var t in tools.EnumerateArray())
        {
            if (t.ValueKind != JsonValueKind.Object || !t.TryGetProperty("type", out var type) || type.GetString() != "function"
                || !t.TryGetProperty("function", out var f) || f.ValueKind != JsonValueKind.Object
                || !f.TryGetProperty("name", out var name) || name.ValueKind != JsonValueKind.String)
            {
                throw new ApiException(400, $"tools[{i}] must be {{\"type\":\"function\",\"function\":{{\"name\":…}}}}", param: $"tools[{i}]");
            }
            list.Add(t.Clone());
            i++;
        }
        return list.Count == 0 ? null : list;
    }

    // ── responses ────────────────────────────────────────────────────────

    private static async Task Responses(HttpContext ctx, IReadOnlyList<ModelHost> models, ServerSettings settings, ResponseStore store)
    {
        using var doc = await ReadBody(ctx);
        var body = doc.RootElement;
        var model = Route(models, body, Capability.Chat);
        RefuseUnsupported(body, "conversation", "prompt");
        var format = OutputFormat(body.TryGetProperty("text", out var textOpt) && textOpt.ValueKind == JsonValueKind.Object &&
            textOpt.TryGetProperty("format", out var fmt) ? fmt : default, chat: false);

        // An earlier response's conversation continues, without its instructions.
        var messages = new List<ChatMessage>();
        string? previous = body.TryGetProperty("previous_response_id", out var prev) && prev.ValueKind == JsonValueKind.String ? prev.GetString() : null;
        if (previous is not null)
        {
            messages.AddRange(store.Get(previous)
                ?? throw new ApiException(404, $"Previous response with id '{previous}' not found.", code: "previous_response_not_found",
                    param: "previous_response_id"));
        }
        if (!body.TryGetProperty("input", out var input))
        {
            throw new ApiException(400, "'input' is required", param: "input");
        }
        messages.AddRange(ResponseInput(input));
        string? instructions = body.TryGetProperty("instructions", out var ins) && ins.ValueKind == JsonValueKind.String ? ins.GetString() : null;
        var history = messages.Where(m => m.Role != "system").ToList();
        var systems = messages.Where(m => m.Role == "system").Select(m => m.Content).ToList();
        if (instructions is not null)
        {
            systems.Insert(0, instructions);
        }
        var conversation = systems.Count == 0 ? history
            : history.Prepend(new ChatMessage("system", string.Join("\n\n", systems))).ToList();

        int maxTokens = IntField(body, "max_output_tokens") ?? MaxTokens(model, settings);
        var tools = ResponseTools(body);
        string? effort = body.TryGetProperty("reasoning", out var reasoningOpt) && reasoningOpt.ValueKind == JsonValueKind.Object
            && reasoningOpt.TryGetProperty("effort", out var eff) && eff.ValueKind == JsonValueKind.String ? eff.GetString() : null;
        var turn = Turn(model, body, settings, conversation, tools, maxTokens, effort, format);
        bool keep = !(body.TryGetProperty("store", out var st) && st.ValueKind == JsonValueKind.False);

        string id = "resp_" + Guid.NewGuid().ToString("N");
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        bool stream = body.TryGetProperty("stream", out var s) && s.ValueKind == JsonValueKind.True;
        var output = new ResponseOutput();

        JsonObject Response(string status, GenerationResult? result) => new()
        {
            ["id"] = id,
            ["object"] = "response",
            ["created_at"] = created,
            ["status"] = status,
            ["model"] = model.Id,
            ["output"] = result is null ? new JsonArray() : output.Items(),
            ["incomplete_details"] = status == "incomplete" ? new JsonObject { ["reason"] = "max_output_tokens" } : null,
            ["instructions"] = instructions,
            ["max_output_tokens"] = IntField(body, "max_output_tokens"),
            ["previous_response_id"] = previous,
            ["temperature"] = turn.Params.Temperature,
            ["top_p"] = turn.Params.TopP,
            ["parallel_tool_calls"] = turn.Parallel,
            ["tool_choice"] = body.TryGetProperty("tool_choice", out var tc) ? JsonNode.Parse(tc.GetRawText()) : "auto",
            ["tools"] = body.TryGetProperty("tools", out var tl) && tl.ValueKind == JsonValueKind.Array ? JsonNode.Parse(tl.GetRawText()) : new JsonArray(),
            ["store"] = keep,
            ["error"] = null,
            ["usage"] = result is null ? null : new JsonObject
            {
                ["input_tokens"] = result.PromptTokens,
                ["output_tokens"] = result.CompletionTokens,
                ["total_tokens"] = result.PromptTokens + result.CompletionTokens,
            },
        };
        string StatusOf(GenerationResult r, ReplyParser reply) =>
            reply.ToolCalls.Count == 0 && r.FinishReason == "length" ? "incomplete" : "completed";
        void Remember(ReplyParser reply)
        {
            if (keep)
            {
                store.Put(id, history.Append(new ChatMessage("assistant", reply.Content)
                {
                    Reasoning = reply.Thought ? reply.Reasoning : null,
                    ToolCalls = reply.ToolCalls.Select(c => new ChatToolCall(c.Name, JsonDocument.Parse(c.Arguments).RootElement.Clone())).ToList(),
                }).ToList());
            }
        }

        if (!stream)
        {
            var (result, reply) = await Locked(model, ctx.RequestAborted, () => Run(model, turn, null, ctx.RequestAborted, null));
            output.Complete(reply, StatusOf(result, reply));
            Remember(reply);
            await ctx.Response.WriteAsJsonAsync(Response(StatusOf(result, reply), result), Json);
            return;
        }

        int sequence = 0;
        await StreamSse(ctx, model, write =>
        {
            void Event(string type, JsonObject fields)
            {
                fields["type"] = type;
                fields["sequence_number"] = sequence++;
                write(type, fields);
            }
            Event("response.created", new JsonObject { ["response"] = Response("in_progress", null) });
            Event("response.in_progress", new JsonObject { ["response"] = Response("in_progress", null) });
            var (result, reply) = Run(model, turn, e => output.Stream(e, Event), ctx.RequestAborted, null);
            string status = StatusOf(result, reply);
            output.Finish(reply, status, Event);
            Remember(reply);
            string type = status == "incomplete" ? "response.incomplete" : "response.completed";
            Event(type, new JsonObject { ["response"] = Response(status, result) });
            return false;
        });
    }

    /// <summary>The Responses API's input: a string, or items — messages,
    /// <c>function_call</c>, <c>function_call_output</c> and <c>reasoning</c>
    /// — folded into chat turns: calls join the assistant turn before them,
    /// reasoning the assistant turn after it.</summary>
    private static List<ChatMessage> ResponseInput(JsonElement input)
    {
        if (input.ValueKind == JsonValueKind.String)
        {
            return new List<ChatMessage> { new("user", input.GetString()!) };
        }
        if (input.ValueKind != JsonValueKind.Array)
        {
            throw new ApiException(400, "'input' must be a string or an array of items", param: "input");
        }
        var messages = new List<ChatMessage>();
        string? reasoning = null;
        int i = 0;
        foreach (var item in input.EnumerateArray())
        {
            string where = $"input[{i++}]";
            string type = item.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String ? t.GetString()! : "message";
            switch (type)
            {
                case "message":
                {
                    var m = ChatMessageFrom(item, where);
                    if (m.Role == "assistant" && reasoning is not null)
                    {
                        m = m with { Reasoning = reasoning };
                        reasoning = null;
                    }
                    messages.Add(m);
                    break;
                }
                case "function_call":
                {
                    string name = item.TryGetProperty("name", out var nm) && nm.ValueKind == JsonValueKind.String ? nm.GetString()!
                        : throw new ApiException(400, $"{where}: a function_call needs a 'name'", param: where);
                    var call = new ChatToolCall(name, Arguments(item.TryGetProperty("arguments", out var a) ? a : default, where));
                    if (messages.Count > 0 && messages[^1].Role == "assistant")
                    {
                        var last = messages[^1];
                        messages[^1] = last with { ToolCalls = (last.ToolCalls ?? Array.Empty<ChatToolCall>()).Append(call).ToList() };
                    }
                    else
                    {
                        messages.Add(new ChatMessage("assistant", "") { ToolCalls = new[] { call }, Reasoning = reasoning });
                        reasoning = null;
                    }
                    break;
                }
                case "function_call_output":
                {
                    var o = item.TryGetProperty("output", out var ov) ? ov : default;
                    messages.Add(new ChatMessage("tool", o.ValueKind switch
                    {
                        JsonValueKind.String => o.GetString()!,
                        JsonValueKind.Array => PartsText(o, where),
                        _ => throw new ApiException(400, $"{where}: 'output' must be a string or text parts", param: where),
                    }));
                    break;
                }
                case "reasoning":
                {
                    var sb = new StringBuilder();
                    foreach (string field in new[] { "content", "summary" })
                    {
                        if (sb.Length == 0 && item.TryGetProperty(field, out var parts) && parts.ValueKind == JsonValueKind.Array)
                        {
                            foreach (var part in parts.EnumerateArray())
                            {
                                if (part.TryGetProperty("text", out var tx) && tx.ValueKind == JsonValueKind.String)
                                {
                                    sb.Append(tx.GetString());
                                }
                            }
                        }
                    }
                    reasoning = sb.ToString();
                    break;
                }
                default:
                    throw new ApiException(400, $"input item type '{type}' is not supported (message, function_call, function_call_output, reasoning)", param: where);
            }
        }
        return messages;
    }

    /// <summary>The Responses API's function tools, in the chat form the
    /// template renders.</summary>
    private static List<JsonElement>? ResponseTools(JsonElement body)
    {
        if (!body.TryGetProperty("tools", out var tools) || tools.ValueKind == JsonValueKind.Null)
        {
            return null;
        }
        if (tools.ValueKind != JsonValueKind.Array)
        {
            throw new ApiException(400, "'tools' must be an array", param: "tools");
        }
        var list = new List<JsonElement>();
        int i = 0;
        foreach (var t in tools.EnumerateArray())
        {
            string type = t.TryGetProperty("type", out var ty) && ty.ValueKind == JsonValueKind.String ? ty.GetString()! : "";
            if (type != "function")
            {
                throw new ApiException(400, $"tools[{i}]: tool type '{type}' is not supported (function only)", param: $"tools[{i}]");
            }
            if (!t.TryGetProperty("name", out var name) || name.ValueKind != JsonValueKind.String)
            {
                throw new ApiException(400, $"tools[{i}] needs a 'name'", param: $"tools[{i}]");
            }
            var function = new JsonObject { ["name"] = name.GetString() };
            if (t.TryGetProperty("description", out var d) && d.ValueKind == JsonValueKind.String)
            {
                function["description"] = d.GetString();
            }
            if (t.TryGetProperty("parameters", out var p) && p.ValueKind == JsonValueKind.Object)
            {
                function["parameters"] = JsonNode.Parse(p.GetRawText());
            }
            list.Add(JsonDocument.Parse(new JsonObject { ["type"] = "function", ["function"] = function }.ToJsonString(Json)).RootElement.Clone());
            i++;
        }
        return list.Count == 0 ? null : list;
    }

    /// <summary>The output items of one response as they are generated —
    /// reasoning, then the message, then each call — and the stream events
    /// that open, fill and close them.</summary>
    private sealed class ResponseOutput
    {
        private readonly List<JsonObject> _items = new();
        private readonly StringBuilder _reasoning = new();
        private readonly StringBuilder _text = new();
        private int _reasoningAt = -1, _messageAt = -1;
        private bool _reasoningOpen, _messageOpen;

        public JsonArray Items() => new(_items.Select(i => (JsonNode?)i.DeepClone()).ToArray());

        private static JsonObject Reasoning(string id, string text, string status) => new()
        {
            ["type"] = "reasoning", ["id"] = id, ["status"] = status, ["summary"] = new JsonArray(),
            ["content"] = new JsonArray(new JsonObject { ["type"] = "reasoning_text", ["text"] = text }),
        };

        private static JsonObject Message(string id, string text, string status) => new()
        {
            ["type"] = "message", ["id"] = id, ["status"] = status, ["role"] = "assistant",
            ["content"] = new JsonArray(new JsonObject { ["type"] = "output_text", ["text"] = text, ["annotations"] = new JsonArray() }),
        };

        private static JsonObject Call(ParsedToolCall c, string status) => new()
        {
            ["type"] = "function_call", ["id"] = "fc_" + Guid.NewGuid().ToString("N"), ["status"] = status,
            ["call_id"] = "call_" + Guid.NewGuid().ToString("N")[..24], ["name"] = c.Name, ["arguments"] = c.Arguments,
        };

        private static string Id(string prefix) => prefix + Guid.NewGuid().ToString("N");

        /// <summary>The whole output at once, for a response that is not streamed.</summary>
        public void Complete(ReplyParser reply, string status)
        {
            if (reply.Thought && reply.Reasoning.Length > 0)
            {
                _items.Add(Reasoning(Id("rs_"), reply.Reasoning, "completed"));
            }
            if (reply.Content.Length > 0 || reply.ToolCalls.Count == 0)
            {
                _items.Add(Message(Id("msg_"), reply.Content, status == "incomplete" ? "incomplete" : "completed"));
            }
            foreach (var c in reply.ToolCalls)
            {
                _items.Add(Call(c, "completed"));
            }
        }

        public void Stream(ReplyEvent e, Action<string, JsonObject> emit)
        {
            switch (e)
            {
                case ReplyEvent.Reasoning r:
                    if (!_reasoningOpen)
                    {
                        _reasoningOpen = true;
                        _reasoningAt = _items.Count;
                        _items.Add(Reasoning(Id("rs_"), "", "in_progress"));
                        emit("response.output_item.added", new JsonObject { ["output_index"] = _reasoningAt, ["item"] = _items[_reasoningAt].DeepClone() });
                        emit("response.content_part.added", Part(_reasoningAt, new JsonObject { ["type"] = "reasoning_text", ["text"] = "" }));
                    }
                    _reasoning.Append(r.Delta);
                    emit("response.reasoning_text.delta", Delta(_reasoningAt, r.Delta));
                    break;
                case ReplyEvent.Content c:
                    CloseReasoning(emit);
                    if (!_messageOpen)
                    {
                        OpenMessage(emit);
                    }
                    _text.Append(c.Delta);
                    emit("response.output_text.delta", Delta(_messageAt, c.Delta));
                    break;
                case ReplyEvent.ToolCall t:
                    CloseReasoning(emit);
                    CloseMessage(emit, "completed");
                    var call = Call(t.Call, "in_progress");
                    int at = _items.Count;
                    _items.Add(call);
                    var added = call.DeepClone().AsObject();
                    added["arguments"] = "";
                    emit("response.output_item.added", new JsonObject { ["output_index"] = at, ["item"] = added });
                    emit("response.function_call_arguments.delta", new JsonObject
                    {
                        ["item_id"] = call["id"]!.GetValue<string>(), ["output_index"] = at, ["delta"] = t.Call.Arguments,
                    });
                    emit("response.function_call_arguments.done", new JsonObject
                    {
                        ["item_id"] = call["id"]!.GetValue<string>(), ["output_index"] = at, ["name"] = t.Call.Name, ["arguments"] = t.Call.Arguments,
                    });
                    call["status"] = "completed";
                    emit("response.output_item.done", new JsonObject { ["output_index"] = at, ["item"] = call.DeepClone() });
                    break;
            }
        }

        /// <summary>Closes what is still open; a reply that said nothing still gets its (empty) message.</summary>
        public void Finish(ReplyParser reply, string status, Action<string, JsonObject> emit)
        {
            CloseReasoning(emit);
            if (_messageAt < 0 && reply.ToolCalls.Count == 0)
            {
                OpenMessage(emit);
            }
            CloseMessage(emit, status == "incomplete" ? "incomplete" : "completed");
        }

        private void OpenMessage(Action<string, JsonObject> emit)
        {
            _messageOpen = true;
            _messageAt = _items.Count;
            _items.Add(Message(Id("msg_"), "", "in_progress"));
            emit("response.output_item.added", new JsonObject { ["output_index"] = _messageAt, ["item"] = _items[_messageAt].DeepClone() });
            emit("response.content_part.added", Part(_messageAt, new JsonObject { ["type"] = "output_text", ["text"] = "", ["annotations"] = new JsonArray() }));
        }

        private void CloseReasoning(Action<string, JsonObject> emit)
        {
            if (!_reasoningOpen)
            {
                return;
            }
            _reasoningOpen = false;
            string text = _reasoning.ToString();
            var done = Part(_reasoningAt, null);
            done["text"] = text;
            emit("response.reasoning_text.done", done);
            emit("response.content_part.done", Part(_reasoningAt, new JsonObject { ["type"] = "reasoning_text", ["text"] = text }));
            _items[_reasoningAt] = Reasoning(_items[_reasoningAt]["id"]!.GetValue<string>(), text, "completed");
            emit("response.output_item.done", new JsonObject { ["output_index"] = _reasoningAt, ["item"] = _items[_reasoningAt].DeepClone() });
        }

        private void CloseMessage(Action<string, JsonObject> emit, string status)
        {
            if (!_messageOpen)
            {
                return;
            }
            _messageOpen = false;
            string text = _text.ToString();
            var done = Part(_messageAt, null);
            done["text"] = text;
            emit("response.output_text.done", done);
            emit("response.content_part.done", Part(_messageAt, new JsonObject { ["type"] = "output_text", ["text"] = text, ["annotations"] = new JsonArray() }));
            _items[_messageAt] = Message(_items[_messageAt]["id"]!.GetValue<string>(), text, status);
            emit("response.output_item.done", new JsonObject { ["output_index"] = _messageAt, ["item"] = _items[_messageAt].DeepClone() });
        }

        private JsonObject Part(int at, JsonObject? part)
        {
            var o = new JsonObject { ["item_id"] = _items[at]["id"]!.GetValue<string>(), ["output_index"] = at, ["content_index"] = 0 };
            if (part is not null)
            {
                o["part"] = part;
            }
            return o;
        }

        private JsonObject Delta(int at, string delta)
        {
            var o = Part(at, null);
            o["delta"] = delta;
            return o;
        }
    }

    // ── shared generation plumbing ───────────────────────────────────────

    /// <summary>One turn to generate: the prompt, how to sample it, and how to
    /// read the reply.</summary>
    private sealed record ChatTurn(int[] Prompt, GenerationParams Params, IReadOnlyList<JsonElement>? Tools, bool Parallel);

    /// <summary>Renders a request into a turn: thinking from
    /// <c>chat_template_kwargs.enable_thinking</c>, then the reasoning effort
    /// (<c>none</c>/<c>minimal</c> turn it off), then the server's default;
    /// tools unless <c>tool_choice</c> is <c>none</c>, with <c>required</c> or a
    /// named function enforced by writing the start of the call for the model.</summary>
    /// <summary>Structured output asked for: the schema to hold the reply to, and the instruction
    /// that tells the model about it.</summary>
    private sealed record JsonFormat(JsonSchemaNode Schema, string Instruction);

    /// <summary>
    /// A request's output format: chat's <c>response_format</c>
    /// (<c>{"type":"json_schema","json_schema":{name, schema}}</c>) or the Responses API's
    /// <c>text.format</c> (<c>{"type":"json_schema", name, schema}</c>), either also
    /// <c>{"type":"json_object"}</c>; null for plain text.
    /// </summary>
    private static JsonFormat? OutputFormat(JsonElement format, bool chat)
    {
        string param = chat ? "response_format" : "text.format";
        if (format.ValueKind != JsonValueKind.Object)
        {
            return null;
        }
        string type = format.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String ? t.GetString()! : "text";
        switch (type)
        {
            case "text":
                return null;
            case "json_object":
                return new JsonFormat(JsonSchemaNode.AnyObject(), "Respond with a single JSON object and nothing else.");
            case "json_schema":
            {
                var holder = chat && format.TryGetProperty("json_schema", out var js) ? js : format;
                if (!holder.TryGetProperty("schema", out var schema) || schema.ValueKind is not (JsonValueKind.Object or JsonValueKind.True))
                {
                    throw new ApiException(400, $"'{param}' needs a JSON schema", param: param);
                }
                try
                {
                    var node = JsonSchemaNode.Compile(schema);
                    string name = holder.TryGetProperty("name", out var nm) && nm.ValueKind == JsonValueKind.String ? $" ({nm.GetString()})" : "";
                    return new JsonFormat(node,
                        $"Respond with a single JSON value{name} that follows this JSON Schema, and nothing else:\n{JsonSerializer.Serialize(schema)}");
                }
                catch (ArgumentException e)
                {
                    throw new ApiException(400, $"'{param}': {e.Message}", param: param);
                }
            }
            default:
                throw new ApiException(400, $"unknown '{param}' type '{type}' (text, json_object, json_schema)", param: param);
        }
    }

    private static ChatTurn Turn(ModelHost model, JsonElement body, ServerSettings settings, List<ChatMessage> messages,
        List<JsonElement>? tools, int maxTokens, string? effort, JsonFormat? format = null)
    {
        var generator = model.Generator!;
        bool thinking = Think(model, settings);
        if (effort is not null)
        {
            thinking = effort is not ("none" or "minimal");
        }
        if (body.TryGetProperty("chat_template_kwargs", out var kw) && kw.ValueKind == JsonValueKind.Object &&
            kw.TryGetProperty("enable_thinking", out var et) && et.ValueKind is JsonValueKind.True or JsonValueKind.False)
        {
            thinking = et.ValueKind == JsonValueKind.True;
        }
        thinking &= generator.Thinks;

        string? forced = null;
        if (body.TryGetProperty("tool_choice", out var choice) && choice.ValueKind != JsonValueKind.Null)
        {
            string? named = choice.ValueKind == JsonValueKind.Object
                ? (choice.TryGetProperty("function", out var cf) && cf.TryGetProperty("name", out var cn) ? cn.GetString()
                    : choice.TryGetProperty("name", out var rn) ? rn.GetString() : null)
                  ?? throw new ApiException(400, "'tool_choice' names no function", param: "tool_choice")
                : null;
            string mode = choice.ValueKind == JsonValueKind.String ? choice.GetString()! : "function";
            switch (mode)
            {
                case "none":
                    tools = null;
                    break;
                case "auto":
                    break;
                case "required" when tools is not null:
                    forced = "<tool_call>\n<function=";
                    break;
                case "function" when tools is not null:
                    if (!tools.Any(t => ToolName(t) == named))
                    {
                        throw new ApiException(400, $"'tool_choice' names '{named}', which is not among the tools", param: "tool_choice");
                    }
                    forced = $"<tool_call>\n<function={named}>\n";
                    break;
                case "required" or "function":
                    throw new ApiException(400, "'tool_choice' asks for a call but the request has no tools", param: "tool_choice");
                default:
                    throw new ApiException(400, $"unknown 'tool_choice' '{mode}' (none, auto, required, or a function)", param: "tool_choice");
            }
        }
        if (tools is not null && !generator.RendersTools)
        {
            throw new ApiException(400, $"model '{model.Id}' has no chat template for tools", param: "tools");
        }
        bool parallel = !(body.TryGetProperty("parallel_tool_calls", out var pc) && pc.ValueKind == JsonValueKind.False);

        // Structured output: the reply is the JSON value, so no thinking block, no calls, and
        // the schema in the system turn, where a small model can see what it is asked to write.
        if (format is not null)
        {
            thinking = false;
            tools = null;
            forced = null;
            messages = messages.Count > 0 && messages[0].Role == "system"
                ? messages.Skip(1).Prepend(messages[0] with { Content = messages[0].Content.TrimEnd() + "\n\n" + format.Instruction }).ToList()
                : messages.Prepend(new ChatMessage("system", format.Instruction)).ToList();
        }

        // continue_final_message (vLLM's name): the last assistant message is the start of the reply,
        // which the model carries on from instead of starting a new turn. Its text is not repeated
        // in the reply, and the reply is read (calls, thinking) from the continuation alone.
        string? prefill = null;
        if (body.TryGetProperty("continue_final_message", out var cfm) && cfm.ValueKind == JsonValueKind.True)
        {
            if (messages.Count == 0 || messages[^1].Role != "assistant" || messages[^1].ToolCalls is { Count: > 0 })
            {
                throw new ApiException(400, "'continue_final_message' needs the last message to be an assistant message with text", param: "messages");
            }
            prefill = messages[^1].Content;
            messages = messages.Take(messages.Count - 1).ToList();
        }

        int[] prompt;
        try
        {
            prompt = prefill is null
                ? generator.EncodePrompt(messages, thinking, tools)
                : generator.Tokenizer.EncodeToIds(ChatTemplate.ApplyMessages(generator.Template, messages, thinking, tools) + prefill).ToArray();
        }
        catch (ChatTemplateException e)
        {
            throw new ApiException(400, e.Message, param: "messages");
        }
        // The current task, for the mentioned-parameter rule: the user's latest message and every
        // tool result since (a harness's hint such as "pass overwrite: true" may be a few calls back).
        int lastUser = messages.FindLastIndex(m => m.Role == "user");
        string recent = string.Join("\n", messages.Skip(Math.Max(lastUser, 0)).Where(m => m.Role is "user" or "tool").Select(m => m.Content));
        var p = Sampling(body, maxTokens, settings, model.Defaults) with
        {
            Forced = forced, Thinking = thinking, Json = format?.Schema, Tools = tools, Recent = recent,
        };
        return new ChatTurn(prompt, p, tools, parallel);
    }

    private static string? ToolName(JsonElement tool) =>
        tool.TryGetProperty("function", out var f) && f.TryGetProperty("name", out var n) ? n.GetString() : null;

    /// <summary>Generates a turn and reads the reply as it comes. Without
    /// parallel calls, generation stops once the first call closes.</summary>
    private static (GenerationResult Result, ReplyParser Reply) Run(ModelHost model, ChatTurn turn,
        Action<ReplyEvent>? onEvent, CancellationToken cancel, int? logprobsTopK)
    {
        var reply = new ReplyParser(turn.Params.Thinking, turn.Tools);
        using var enough = CancellationTokenSource.CreateLinkedTokenSource(cancel);
        void Read(IReadOnlyList<ReplyEvent> events)
        {
            foreach (var e in events)
            {
                if (e is ReplyEvent.ToolCall && !turn.Parallel && reply.ToolCalls.Count > 1)
                {
                    continue;
                }
                onEvent?.Invoke(e);
                if (e is ReplyEvent.ToolCall && !turn.Parallel)
                {
                    enough.Cancel();
                }
            }
        }
        GenerationResult result;
        try
        {
            result = model.Generator!.Generate(turn.Prompt, turn.Params, d => Read(reply.Feed(d)), enough.Token, logprobsTopK);
        }
        catch (ArgumentException e)
        {
            throw new ApiException(400, e.Message, code: "context_length_exceeded");
        }
        Read(reply.Finish());
        if (result.FinishReason == "cancelled" && !cancel.IsCancellationRequested)
        {
            result = result with { FinishReason = "stop" };
        }
        if (!turn.Parallel && reply.ToolCalls.Count > 1)
        {
            reply.KeepFirstCall();
        }
        return (result, reply);
    }

    /// <summary>
    /// Runs a generation under the model's lock and streams it as server-sent
    /// events. Generation is CPU-bound and synchronous; it pushes events into
    /// a channel that this request drains to the socket, and a client that
    /// disconnects cancels the generation at its next token. The work returns
    /// whether the stream ends with <c>data: [DONE]</c>.
    /// </summary>
    private static async Task StreamSse(HttpContext ctx, ModelHost model, Func<Action<string?, JsonObject>, bool> work)
    {
        var channel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions { SingleReader = true });
        void Write(string? eventName, JsonObject data) =>
            channel.Writer.TryWrite((eventName is null ? "" : $"event: {eventName}\n") + $"data: {data.ToJsonString(Json)}\n\n");

        await model.Gate.WaitAsync(ctx.RequestAborted);
        var running = Task.Run(() =>
        {
            try
            {
                if (work(Write))
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
            Log($"{ctx.Request.Method} {ctx.Request.Path} -> failed while streaming{Environment.NewLine}{e}");
            var error = new JsonObject { ["error"] = new JsonObject { ["message"] = e.Message, ["type"] = "server_error" } };
            await ctx.Response.WriteAsync($"data: {error.ToJsonString(Json)}\n\n");
        }
        await running;
    }

    private static GenerationParams Sampling(JsonElement body, int maxTokens, ServerSettings settings, ModelDefaults defaults)
    {
        if (maxTokens < 1)
        {
            throw new ApiException(400, "the token limit must be at least 1", param: "max_tokens");
        }
        float temperature = FloatField(body, "temperature") ?? defaults.Temperature ?? settings.Temperature ?? 1f;
        float topP = FloatField(body, "top_p") ?? defaults.TopP ?? settings.TopP ?? 1f;
        int topK = IntField(body, "top_k") ?? defaults.TopK ?? settings.TopK ?? 0;
        if (temperature is < 0 or > 2)
        {
            throw new ApiException(400, "'temperature' must be between 0 and 2", param: "temperature");
        }
        if (topP is <= 0 or > 1)
        {
            throw new ApiException(400, "'top_p' must be in (0, 1]", param: "top_p");
        }
        if (topK < 0)
        {
            throw new ApiException(400, "'top_k' must be 0 (off) or more", param: "top_k");
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
        return new GenerationParams(maxTokens, temperature, topP, seed, stop) { TopK = topK };
    }

    /// <summary>A chat message's role and its content flattened to text: a
    /// string, or an array of text parts (<c>text</c>, <c>input_text</c>,
    /// <c>output_text</c>). An assistant turn keeps its <c>tool_calls</c> and
    /// its <c>reasoning_content</c>; a <c>tool</c> turn is a call's result.
    /// Images and audio are refused, not dropped.</summary>
    private static ChatMessage ChatMessageFrom(JsonElement m, string where)
    {
        if (m.ValueKind != JsonValueKind.Object || !m.TryGetProperty("role", out var role) || role.ValueKind != JsonValueKind.String)
        {
            throw new ApiException(400, $"{where} must be an object with a 'role'", param: where);
        }
        string r = role.GetString()! switch
        {
            "developer" => "system",
            "system" or "user" or "assistant" or "tool" => role.GetString()!,
            var other => throw new ApiException(400, $"{where}: role '{other}' is not supported (system, developer, user, assistant, tool)", param: where),
        };
        string text;
        if (!m.TryGetProperty("content", out var content) || content.ValueKind == JsonValueKind.Null)
        {
            text = "";
        }
        else if (content.ValueKind == JsonValueKind.String)
        {
            text = content.GetString()!;
        }
        else if (content.ValueKind == JsonValueKind.Array)
        {
            text = PartsText(content, where);
        }
        else
        {
            throw new ApiException(400, $"{where}.content must be a string or an array of parts", param: where);
        }

        var message = new ChatMessage(r, text);
        if (r == "assistant")
        {
            foreach (string field in new[] { "reasoning_content", "reasoning" })
            {
                if (m.TryGetProperty(field, out var rc) && rc.ValueKind == JsonValueKind.String)
                {
                    message = message with { Reasoning = rc.GetString() };
                    break;
                }
            }
        }
        if (m.TryGetProperty("tool_calls", out var calls) && calls.ValueKind == JsonValueKind.Array && calls.GetArrayLength() > 0)
        {
            if (r != "assistant")
            {
                throw new ApiException(400, $"{where}: only an assistant message carries tool_calls", param: where);
            }
            var list = new List<ChatToolCall>();
            int i = 0;
            foreach (var call in calls.EnumerateArray())
            {
                string at = $"{where}.tool_calls[{i++}]";
                var f = call.TryGetProperty("function", out var fn) && fn.ValueKind == JsonValueKind.Object ? fn
                    : throw new ApiException(400, $"{at} needs a 'function'", param: at);
                string name = f.TryGetProperty("name", out var nm) && nm.ValueKind == JsonValueKind.String ? nm.GetString()!
                    : throw new ApiException(400, $"{at}.function needs a 'name'", param: at);
                list.Add(new ChatToolCall(name, Arguments(f.TryGetProperty("arguments", out var a) ? a : default, at)));
            }
            message = message with { ToolCalls = list };
        }
        return message;
    }

    /// <summary>A call's arguments as a JSON object: the API sends them as a
    /// JSON string, some clients as the object itself.</summary>
    private static JsonElement Arguments(JsonElement arguments, string where)
    {
        switch (arguments.ValueKind)
        {
            case JsonValueKind.Undefined or JsonValueKind.Null:
                return JsonDocument.Parse("{}").RootElement.Clone();
            case JsonValueKind.Object:
                return arguments.Clone();
            case JsonValueKind.String:
                string s = arguments.GetString()!;
                if (s.Trim().Length == 0)
                {
                    return JsonDocument.Parse("{}").RootElement.Clone();
                }
                try
                {
                    using var parsed = JsonDocument.Parse(s);
                    if (parsed.RootElement.ValueKind == JsonValueKind.Object)
                    {
                        return parsed.RootElement.Clone();
                    }
                }
                catch (JsonException)
                {
                }
                throw new ApiException(400, $"{where}: 'arguments' must be a JSON object", param: where);
            default:
                throw new ApiException(400, $"{where}: 'arguments' must be a JSON object", param: where);
        }
    }

    private static string PartsText(JsonElement parts, string where)
    {
        var sb = new StringBuilder();
        foreach (var part in parts.EnumerateArray())
        {
            if (part.ValueKind == JsonValueKind.String)
            {
                sb.Append(part.GetString());
                continue;
            }
            string type = part.TryGetProperty("type", out var t) ? t.GetString() ?? "" : "";
            if (type is not ("text" or "input_text" or "output_text") || !part.TryGetProperty("text", out var text))
            {
                throw new ApiException(400, $"{where}: content part type '{type}' is not supported (text only)", param: where);
            }
            sb.Append(text.GetString());
        }
        return sb.ToString();
    }

    /// <summary>Effort for chat completions: <c>reasoning_effort</c>, or <c>reasoning.effort</c>.</summary>
    private static string? ChatEffort(JsonElement body) =>
        body.TryGetProperty("reasoning_effort", out var e) && e.ValueKind == JsonValueKind.String ? e.GetString()
        : body.TryGetProperty("reasoning", out var r) && r.ValueKind == JsonValueKind.Object && r.TryGetProperty("effort", out var re)
          && re.ValueKind == JsonValueKind.String ? re.GetString() : null;

    /// <summary>Builds the OpenAI-format <c>logprobs</c> object from the
    /// generation result's per-token statistics. Returns null (JSON null)
    /// when no logprobs were requested.</summary>
    private static JsonNode? ChatLogprobsJson(GenerationResult result, HfTokenizer tokenizer)
    {
        if (result.Logprobs is not { Count: > 0 })
        {
            return null;
        }
        var content = new JsonArray();
        foreach (var lp in result.Logprobs)
        {
            var topArray = new JsonArray();
            foreach (var tlp in lp.TopLogprobs)
            {
                string? tokenText = tokenizer.TokenInfo(tlp.Token).DecodedText;
                var bytesArray = new JsonArray();
                if (tokenText is not null)
                {
                    foreach (byte b in Encoding.UTF8.GetBytes(tokenText))
                        bytesArray.Add(b);
                }
                topArray.Add(new JsonObject
                {
                    ["token"] = tokenText ?? $"<{tlp.Token}>",
                    ["logprob"] = float.IsNegativeInfinity(tlp.Logprob) ? -9999.0f : tlp.Logprob,
                    ["bytes"] = bytesArray,
                });
            }
            // The selected token is always TopLogprobs[0] since it has the
            // highest probability among the sampled distribution.
            var first = lp.TopLogprobs[0];
            string? firstText = tokenizer.TokenInfo(first.Token).DecodedText;
            var firstBytes = new JsonArray();
            if (firstText is not null)
            {
                foreach (byte b in Encoding.UTF8.GetBytes(firstText))
                    firstBytes.Add(b);
            }
            content.Add(new JsonObject
            {
                ["token"] = firstText ?? $"<{first.Token}>",
                ["logprob"] = float.IsNegativeInfinity(lp.TokenLogprob) ? -9999.0f : lp.TokenLogprob,
                ["bytes"] = firstBytes,
                ["top_logprobs"] = topArray,
            });
        }
        return new JsonObject { ["content"] = content };
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
