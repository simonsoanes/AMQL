using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace Amql.Gui.Services;

/// <summary>Talks to a running amql-server: health, status and chat.</summary>
public sealed class ServerClient
{
    private static readonly HttpClient Http = new() { Timeout = Timeout.InfiniteTimeSpan };

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        NumberHandling = JsonNumberHandling.AllowReadingFromString,
    };

    private readonly string _baseUrl;
    private readonly string _apiKey;

    public ServerClient(string baseUrl, string apiKey)
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _apiKey = apiKey?.Trim() ?? string.Empty;
    }

    public static ServerClient ForRunningServer() =>
        new(ServerController.Instance.BaseUrl, ServerController.Instance.ApiKey);

    private HttpRequestMessage Request(HttpMethod method, string path)
    {
        var request = new HttpRequestMessage(method, _baseUrl + path);
        if (_apiKey.Length > 0)
        {
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
        }
        return request;
    }

    public static async Task<bool> IsHealthyAsync(string baseUrl, string apiKey)
    {
        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(1.5));
            using var response = await Http.SendAsync(new ServerClient(baseUrl, apiKey).Request(HttpMethod.Get, "/health"), cts.Token);
            return response.IsSuccessStatusCode;
        }
        catch (Exception e) when (e is HttpRequestException or TaskCanceledException or OperationCanceledException)
        {
            return false;
        }
    }

    /// <summary>The server's /amql/status, or null when it does not answer (or predates the endpoint).</summary>
    public async Task<ServerStatus?> GetStatusAsync(CancellationToken cancel = default)
    {
        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancel);
            cts.CancelAfter(TimeSpan.FromSeconds(3));
            using var response = await Http.SendAsync(Request(HttpMethod.Get, "/amql/status"), cts.Token);
            if (!response.IsSuccessStatusCode)
            {
                return null;
            }
            await using var stream = await response.Content.ReadAsStreamAsync(cts.Token);
            return await JsonSerializer.DeserializeAsync<ServerStatus>(stream, JsonOptions, cts.Token);
        }
        catch (Exception e) when (e is HttpRequestException or TaskCanceledException or OperationCanceledException or JsonException)
        {
            return null;
        }
    }

    /// <summary>The ids and capabilities of the loaded models.</summary>
    public async Task<List<(string Id, string[] Capabilities)>> GetModelsAsync(CancellationToken cancel = default)
    {
        using var response = await Http.SendAsync(Request(HttpMethod.Get, "/v1/models"), cancel);
        response.EnsureSuccessStatusCode();
        var doc = JsonNode.Parse(await response.Content.ReadAsStringAsync(cancel));
        return doc?["data"]?.AsArray()
            .Select(m => (m!["id"]!.GetValue<string>(),
                m["capabilities"]?.AsArray().Select(c => c!.GetValue<string>()).ToArray() ?? Array.Empty<string>()))
            .ToList() ?? new();
    }

    /// <summary>
    /// Streams one chat completion. <paramref name="onDelta"/> gets each piece
    /// of the reply as it arrives (reasoning true for thinking), on a
    /// thread-pool thread. Returns the usage the server reported, if any.
    /// </summary>
    public async Task<ChatUsage> StreamChatAsync(JsonObject body, Action<string, bool> onDelta, CancellationToken cancel)
    {
        body["stream"] = true;
        body["stream_options"] = new JsonObject { ["include_usage"] = true };
        var request = Request(HttpMethod.Post, "/v1/chat/completions");
        request.Content = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json");
        using var response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancel);
        if (!response.IsSuccessStatusCode)
        {
            string text = await response.Content.ReadAsStringAsync(cancel);
            string message = text;
            try
            {
                message = JsonNode.Parse(text)?["error"]?["message"]?.GetValue<string>() ?? text;
            }
            catch (JsonException)
            {
                // not JSON — show it as it is
            }
            throw new InvalidOperationException($"{(int)response.StatusCode}: {message}");
        }

        ChatUsage? usage = null;
        string? finish = null;
        await using var stream = await response.Content.ReadAsStreamAsync(cancel);
        using var reader = new StreamReader(stream, Encoding.UTF8);
        while (await reader.ReadLineAsync(cancel) is { } line)
        {
            if (!line.StartsWith("data:", StringComparison.Ordinal))
            {
                continue;
            }
            string data = line[5..].Trim();
            if (data == "[DONE]")
            {
                break;
            }
            var chunk = JsonNode.Parse(data);
            if (chunk?["error"]?["message"]?.GetValue<string>() is { } error)
            {
                throw new InvalidOperationException(error);
            }
            if (chunk?["usage"] is JsonObject u)
            {
                usage = new ChatUsage(u["prompt_tokens"]?.GetValue<int>() ?? 0, u["completion_tokens"]?.GetValue<int>() ?? 0);
            }
            if (chunk?["choices"]?.AsArray() is { Count: > 0 } finished && finished[0]?["finish_reason"]?.GetValue<string>() is { } reason)
            {
                finish = reason;
            }
            if (chunk?["choices"]?.AsArray() is { Count: > 0 } choices && choices[0]?["delta"] is JsonObject delta)
            {
                if (delta["reasoning_content"]?.GetValue<string>() is { Length: > 0 } r)
                {
                    onDelta(r, true);
                }
                if (delta["content"]?.GetValue<string>() is { Length: > 0 } c)
                {
                    onDelta(c, false);
                }
            }
        }
        return (usage ?? new ChatUsage(0, 0)) with { FinishReason = finish };
    }

    /// <summary>POSTs a JSON body and returns the pretty-printed answer (embeddings, classify, decisions).</summary>
    public async Task<string> PostJsonAsync(string path, string json, CancellationToken cancel)
    {
        var request = Request(HttpMethod.Post, path);
        request.Content = new StringContent(json, Encoding.UTF8, "application/json");
        using var response = await Http.SendAsync(request, cancel);
        string text = await response.Content.ReadAsStringAsync(cancel);
        try
        {
            text = JsonNode.Parse(text)!.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
        }
        catch (JsonException)
        {
            // not JSON — show it as it is
        }
        return $"HTTP {(int)response.StatusCode}\n{text}";
    }
}

/// <summary>What a streamed reply ended with: its usage (when the server sent it) and finish reason.</summary>
public sealed record ChatUsage(int PromptTokens, int CompletionTokens, string? FinishReason = null);

// /amql/status (src/Amql.Server/ServerStats.cs), snake_case on the wire.

public sealed class ServerStatus
{
    public DateTime Started { get; set; }
    public double UptimeSeconds { get; set; }
    public ProcessStatus Process { get; set; } = new();
    public TotalsStatus Totals { get; set; } = new();
    public List<ModelStatus> Models { get; set; } = new();
    public List<RequestStatus> Active { get; set; } = new();
    public List<RequestStatus> Recent { get; set; } = new();
}

public sealed class ProcessStatus
{
    public int Pid { get; set; }
    public long WorkingSetBytes { get; set; }
    public long PrivateBytes { get; set; }
    public long ManagedHeapBytes { get; set; }
    public double CpuSeconds { get; set; }
    public int Threads { get; set; }
    public bool Gpu { get; set; }
}

public sealed class TotalsStatus
{
    public long Requests { get; set; }
    public long Failures { get; set; }
    public int RequestsLastMinute { get; set; }
    public long CompletionTokensLastMinute { get; set; }
}

public sealed class ModelStatus
{
    public string Id { get; set; } = "";
    public string Path { get; set; } = "";
    public List<string> Capabilities { get; set; } = new();
    public long Requests { get; set; }
    public long Errors { get; set; }
    public long PromptTokens { get; set; }
    public long CompletionTokens { get; set; }
    public double? MeanDecodeTps { get; set; }
    public DateTime? LastUsed { get; set; }
    public int Active { get; set; }
    public int? ContextLength { get; set; }
    public List<string> Adapters { get; set; } = new();
    public List<string> Notes { get; set; } = new();
}

public sealed class RequestStatus
{
    public long Id { get; set; }
    public string Method { get; set; } = "";
    public string Path { get; set; } = "";
    public string? Model { get; set; }
    public DateTime Started { get; set; }
    public double Seconds { get; set; }
    public int? Status { get; set; }
    public int PromptTokens { get; set; }
    public int CompletionTokens { get; set; }
    public double? TtftSeconds { get; set; }
    public double? DecodeTps { get; set; }
    public double? PrefillTps { get; set; }
    public string? FinishReason { get; set; }
}
