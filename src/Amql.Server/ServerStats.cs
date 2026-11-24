using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text.Json.Nodes;

namespace Amql.Server;

/// <summary>
/// What the server is doing, kept in memory for <c>GET /amql/status</c>: the
/// requests in flight, the most recent finished ones (model, tokens, time to
/// first token, decode speed) and running totals per model. A monitor polls
/// it instead of reading the request log.
/// </summary>
public static class ServerStats
{
    private const int RecentCapacity = 200;

    private static readonly DateTime StartedUtc = DateTime.UtcNow;
    private static readonly AsyncLocal<RequestTrace?> CurrentTrace = new();
    private static readonly ConcurrentDictionary<long, RequestTrace> Active = new();
    private static readonly ConcurrentDictionary<string, ModelCounters> Models = new(StringComparer.Ordinal);
    private static readonly LinkedList<RequestTrace> Recent = new();
    private static long _nextId;
    private static long _requests, _failures;

    /// <summary>The request this async flow serves, if it came through the server's pipeline.</summary>
    public static RequestTrace? Current => CurrentTrace.Value;

    public static RequestTrace Begin(string method, string path)
    {
        var trace = new RequestTrace(Interlocked.Increment(ref _nextId), method, path);
        Active[trace.Id] = trace;
        CurrentTrace.Value = trace;
        return trace;
    }

    public static void End(RequestTrace trace, int status)
    {
        trace.Finish(status);
        Active.TryRemove(trace.Id, out _);
        // Health and status polls are not activity worth counting or listing.
        if (IsPoll(trace.Path))
        {
            return;
        }
        Interlocked.Increment(ref _requests);
        if (status >= 500)
        {
            Interlocked.Increment(ref _failures);
        }
        if (trace.Model is { } id)
        {
            var c = Models.GetOrAdd(id, _ => new ModelCounters());
            lock (c)
            {
                c.Requests++;
                c.PromptTokens += trace.PromptTokens;
                c.CompletionTokens += trace.CompletionTokens;
                c.GenerateSeconds += trace.GenerateSeconds;
                if (status >= 400) c.Errors++;
                c.LastUsedUtc = DateTime.UtcNow;
            }
        }
        lock (Recent)
        {
            Recent.AddFirst(trace);
            while (Recent.Count > RecentCapacity)
            {
                Recent.RemoveLast();
            }
        }
    }

    /// <summary>The status document: process, per-model totals, in-flight and recent requests.</summary>
    public static JsonObject Snapshot(IReadOnlyList<ModelHost> models)
    {
        var process = Process.GetCurrentProcess();
        var gc = GC.GetGCMemoryInfo();
        var now = DateTime.UtcNow;
        RequestTrace[] recent;
        lock (Recent)
        {
            recent = Recent.ToArray();
        }
        var active = Active.Values.Where(t => !IsPoll(t.Path))
            .OrderBy(t => t.StartedUtc).ToArray();

        // Throughput over the last minute: completion tokens of requests that finished in it.
        var lastMinute = recent.Where(t => (now - t.FinishedUtc!.Value).TotalSeconds <= 60).ToArray();

        var modelArray = new JsonArray();
        foreach (var m in models)
        {
            var c = Models.GetOrAdd(m.Id, _ => new ModelCounters());
            JsonObject o;
            lock (c)
            {
                o = new JsonObject
                {
                    ["id"] = m.Id,
                    ["path"] = m.Path,
                    ["capabilities"] = new JsonArray(ServerApp.CapabilityNames(m.Capabilities).Select(x => (JsonNode?)x).ToArray()),
                    ["requests"] = c.Requests,
                    ["errors"] = c.Errors,
                    ["prompt_tokens"] = c.PromptTokens,
                    ["completion_tokens"] = c.CompletionTokens,
                    ["mean_decode_tps"] = c.GenerateSeconds > 0 ? Math.Round(c.CompletionTokens / c.GenerateSeconds, 1) : null,
                    ["last_used"] = c.LastUsedUtc?.ToString("O"),
                };
            }
            o["active"] = active.Count(t => t.Model == m.Id);
            o["context_length"] = m.Generator is { } g && g.ContextLength != int.MaxValue ? g.ContextLength : null;
            o["adapters"] = new JsonArray(m.AdapterNames.Select(a => (JsonNode?)a).ToArray());
            o["notes"] = new JsonArray(m.Notes.Select(n => (JsonNode?)n).ToArray());
            modelArray.Add(o);
        }

        return new JsonObject
        {
            ["started"] = StartedUtc.ToString("O"),
            ["uptime_seconds"] = Math.Round((now - StartedUtc).TotalSeconds),
            ["process"] = new JsonObject
            {
                ["pid"] = Environment.ProcessId,
                ["working_set_bytes"] = process.WorkingSet64,
                ["private_bytes"] = process.PrivateMemorySize64,
                ["managed_heap_bytes"] = gc.HeapSizeBytes,
                ["cpu_seconds"] = Math.Round(process.TotalProcessorTime.TotalSeconds, 1),
                ["threads"] = process.Threads.Count,
                ["gpu"] = Amql.Inference.CudaShim.Enabled,
            },
            ["totals"] = new JsonObject
            {
                ["requests"] = Interlocked.Read(ref _requests),
                ["failures"] = Interlocked.Read(ref _failures),
                ["requests_last_minute"] = lastMinute.Length,
                ["completion_tokens_last_minute"] = lastMinute.Sum(t => (long)t.CompletionTokens),
            },
            ["models"] = modelArray,
            ["active"] = new JsonArray(active.Select(t => (JsonNode?)t.ToJson(now)).ToArray()),
            ["recent"] = new JsonArray(recent.Take(50).Select(t => (JsonNode?)t.ToJson(now)).ToArray()),
        };
    }

    private static bool IsPoll(string path) =>
        path.StartsWith("/amql", StringComparison.Ordinal) || path == "/health" || path == "/v1/models";

    private sealed class ModelCounters
    {
        public long Requests, Errors, PromptTokens, CompletionTokens;
        public double GenerateSeconds;
        public DateTime? LastUsedUtc;
    }
}

/// <summary>One request as the monitor sees it.</summary>
public sealed class RequestTrace
{
    private readonly Stopwatch _clock = Stopwatch.StartNew();
    private double? _firstTokenSeconds, _generateStartSeconds, _generateEndSeconds;
    private int _streamed;

    public RequestTrace(long id, string method, string path)
    {
        Id = id;
        Method = method;
        Path = path;
    }

    public long Id { get; }
    public string Method { get; }
    public string Path { get; }
    public DateTime StartedUtc { get; } = DateTime.UtcNow;
    public DateTime? FinishedUtc { get; private set; }
    public int Status { get; private set; }
    public double Seconds { get; private set; }

    /// <summary>The model the request was routed to.</summary>
    public string? Model { get; set; }
    public int PromptTokens { get; set; }
    public int CompletionTokens { get; set; }
    public string? FinishReason { get; set; }

    /// <summary>Seconds spent generating (first prompt token in to last token out).</summary>
    public double GenerateSeconds =>
        _generateStartSeconds is { } s && _generateEndSeconds is { } e ? e - s : 0;

    public void GenerateStarted() => _generateStartSeconds ??= _clock.Elapsed.TotalSeconds;

    /// <summary>A piece of the reply came out: the first marks time to first token.</summary>
    public void Delta()
    {
        _firstTokenSeconds ??= _clock.Elapsed.TotalSeconds;
        Interlocked.Increment(ref _streamed);
    }

    public void GenerateFinished(int promptTokens, int completionTokens, string finish)
    {
        _generateEndSeconds = _clock.Elapsed.TotalSeconds;
        PromptTokens += promptTokens;
        CompletionTokens += completionTokens;
        FinishReason = finish;
    }

    internal void Finish(int status)
    {
        Status = status;
        Seconds = _clock.Elapsed.TotalSeconds;
        FinishedUtc = DateTime.UtcNow;
        _generateEndSeconds ??= _generateStartSeconds is null ? null : Seconds;
    }

    internal JsonObject ToJson(DateTime now)
    {
        double elapsed = FinishedUtc is null ? _clock.Elapsed.TotalSeconds : Seconds;
        int tokens = FinishedUtc is null ? Math.Max(CompletionTokens, _streamed) : CompletionTokens;
        // Decode speed counts the tokens after the first over the time after it, so prefill is not in it.
        double? decodeTps = null;
        double? end = _generateEndSeconds ?? (FinishedUtc is null ? _clock.Elapsed.TotalSeconds : null);
        if (_firstTokenSeconds is { } first && end is { } last && last - first > 0.05 && tokens > 1)
        {
            decodeTps = Math.Round((tokens - 1) / (last - first), 1);
        }
        double? prefillTps = _firstTokenSeconds is { } f && _generateStartSeconds is { } g && f - g > 0.001 && PromptTokens > 0
            ? Math.Round(PromptTokens / (f - g), 0)
            : null;
        return new JsonObject
        {
            ["id"] = Id,
            ["method"] = Method,
            ["path"] = Path,
            ["model"] = Model,
            ["started"] = StartedUtc.ToString("O"),
            ["seconds"] = Math.Round(elapsed, 3),
            ["status"] = FinishedUtc is null ? null : Status,
            ["prompt_tokens"] = PromptTokens,
            ["completion_tokens"] = tokens,
            ["ttft_seconds"] = _firstTokenSeconds is { } t && _generateStartSeconds is { } s ? Math.Round(t - s, 3) : null,
            ["decode_tps"] = decodeTps,
            ["prefill_tps"] = prefillTps,
            ["finish_reason"] = FinishReason,
        };
    }
}
