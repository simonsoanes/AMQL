using System.Diagnostics;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Amql.Cli;

/// <summary>
/// <c>amql-cli tool-eval</c>: validates a model's tool use through an
/// OpenAI-compatible <c>/v1/chat/completions</c> endpoint — llama-server,
/// vLLM, amql-server, anything that speaks the protocol — so the check is
/// the one a deployment actually depends on: does the server hand back a
/// well-formed call to the right tool with the right arguments, and does the
/// model use the tool's result.
/// <para>
/// Each case (one JSON object per line) carries <c>messages</c>, <c>tools</c>
/// and an <c>expect</c> block: <c>{"tool": "name", "args": {…}}</c> (argument
/// values compared case-insensitively as strings; a value starting with
/// <c>~</c> is a regex), <c>{"tool": null}</c> for "answer without a tool",
/// optionally <c>"result"</c> (fed back as the tool message) and
/// <c>"answer_contains"</c> (checked against the follow-up reply). When the
/// server returns raw text instead of structured <c>tool_calls</c>, the
/// <c>&lt;tool_call&gt;</c> block is parsed here (Qwen/LFM3 XML, or a JSON
/// body) and the case is marked <c>raw</c> — the model was right but the
/// server's parser did not surface it.
/// </para>
/// </summary>
internal static class ToolEval
{
    public sealed record CaseResult(string Id, bool NameOk, bool ArgsOk, bool? AnswerOk, bool Raw, string Detail, double Seconds)
    {
        public bool Passed => NameOk && ArgsOk && AnswerOk != false;
    }

    public static int Run(string[] args, Func<string[], string, string?> option, Func<string[], string, bool> has)
    {
        string endpoint = (option(args, "--endpoint") ?? "http://127.0.0.1:8080/v1").TrimEnd('/');
        string casesPath = option(args, "--cases") ?? throw new CliException("tool-eval requires '--cases <cases.jsonl>'");
        string model = option(args, "--model") ?? "default";
        int maxTokens = int.Parse(option(args, "--max-tokens") ?? "2048");
        bool noThink = has(args, "--no-think");
        string? reportPath = option(args, "--report");
        int repeat = int.Parse(option(args, "--repeat") ?? "1");

        var cases = File.ReadAllLines(casesPath)
            .Where(l => l.Trim().Length > 0 && !l.TrimStart().StartsWith("//", StringComparison.Ordinal))
            .Select(l => JsonNode.Parse(l)!.AsObject())
            .ToList();
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        var results = new List<CaseResult>();
        for (int r = 0; r < repeat; r++)
        {
            foreach (var testCase in cases)
            {
                var result = RunCase(http, endpoint, model, testCase, maxTokens, noThink);
                results.Add(result);
                Console.WriteLine($"{(result.Passed ? "PASS" : "FAIL")}{(result.Raw ? " (raw)" : "")}  {result.Id,-28} " +
                                  $"{result.Seconds,6:0.0}s  {result.Detail}");
            }
        }

        int passed = results.Count(x => x.Passed);
        int nameOk = results.Count(x => x.NameOk);
        int raw = results.Count(x => x.Raw);
        Console.WriteLine();
        Console.WriteLine($"tool-eval: {passed}/{results.Count} passed  (right tool or no-tool: {nameOk}/{results.Count}; " +
                          $"tool calls only found in raw text: {raw})");
        if (reportPath is not null)
        {
            File.WriteAllText(reportPath, JsonSerializer.Serialize(new
            {
                endpoint,
                model,
                passed,
                total = results.Count,
                rawOnly = raw,
                cases = results,
            }, new JsonSerializerOptions { WriteIndented = true }));
        }
        CliProgress.Result(new { command = "tool-eval", passed, total = results.Count, raw });
        return passed == results.Count ? 0 : 1;
    }

    private static CaseResult RunCase(HttpClient http, string endpoint, string model, JsonObject testCase, int maxTokens, bool noThink)
    {
        string id = testCase["id"]?.GetValue<string>() ?? "case";
        var expect = testCase["expect"]!.AsObject();
        string? expectedTool = expect["tool"]?.GetValue<string>();
        var messages = testCase["messages"]!.DeepClone().AsArray();
        var tools = testCase["tools"]?.DeepClone();
        var clock = Stopwatch.StartNew();

        JsonObject reply;
        try
        {
            reply = Complete(http, endpoint, model, messages, tools, maxTokens, noThink);
        }
        catch (Exception e) when (e is HttpRequestException or TaskCanceledException or JsonException)
        {
            return new CaseResult(id, false, false, null, false, $"request failed: {e.Message}", clock.Elapsed.TotalSeconds);
        }

        var (calls, raw) = ExtractCalls(reply);
        string content = reply["content"]?.GetValue<string>() ?? string.Empty;
        if (expectedTool is null)
        {
            bool ok = calls.Count == 0 && content.Trim().Length > 0;
            return new CaseResult(id, ok, ok, null, false,
                ok ? "answered without a tool" : $"called {string.Join(",", calls.Select(c => c.Name))} (no tool expected)",
                clock.Elapsed.TotalSeconds);
        }
        if (calls.Count == 0)
        {
            return new CaseResult(id, false, false, null, false, $"no tool call; said: {Clip(content)}", clock.Elapsed.TotalSeconds);
        }

        var call = calls.FirstOrDefault(c => c.Name == expectedTool) ?? calls[0];
        bool nameOk = call.Name == expectedTool;
        var argProblems = new List<string>();
        if (expect["args"] is JsonObject expectedArgs)
        {
            foreach (var (key, value) in expectedArgs)
            {
                string want = value is JsonValue v && v.TryGetValue<string>(out var s) ? s : value?.ToJsonString() ?? "null";
                if (!call.Args.TryGetValue(key, out var got))
                {
                    argProblems.Add($"missing {key}");
                }
                else if (!ArgMatches(want, got))
                {
                    argProblems.Add($"{key}={got} (want {want})");
                }
            }
        }
        bool argsOk = argProblems.Count == 0;
        string detail = $"{call.Name}({string.Join(", ", call.Args.Select(a => $"{a.Key}={Clip(a.Value, 30)}"))})" +
                        (argsOk ? string.Empty : "  ✗ " + string.Join("; ", argProblems));

        bool? answerOk = null;
        if (nameOk && expect["result"] is { } result && expect["answer_contains"] is { } needle)
        {
            // Second turn: the assistant's call, then the tool's result.
            string callId = call.Id ?? "call_0";
            messages.Add(new JsonObject
            {
                ["role"] = "assistant",
                ["content"] = "",
                ["tool_calls"] = new JsonArray(new JsonObject
                {
                    ["id"] = callId,
                    ["type"] = "function",
                    ["function"] = new JsonObject
                    {
                        ["name"] = call.Name,
                        ["arguments"] = JsonSerializer.Serialize(call.Args),
                    },
                }),
            });
            messages.Add(new JsonObject
            {
                ["role"] = "tool",
                ["tool_call_id"] = callId,
                ["content"] = result is JsonValue rv && rv.TryGetValue<string>(out var rs) ? rs : result.ToJsonString(),
            });
            try
            {
                var follow = Complete(http, endpoint, model, messages, tools, maxTokens, noThink);
                string answer = follow["content"]?.GetValue<string>() ?? string.Empty;
                string want = needle.GetValue<string>();
                // "7,006,652" answers "7006652": digit grouping is presentation.
                answerOk = answer.Contains(want, StringComparison.OrdinalIgnoreCase) ||
                           Regex.Replace(answer, @"(?<=\d)[,  ](?=\d{3})", "").Contains(want, StringComparison.OrdinalIgnoreCase);
                detail += answerOk.Value ? $"  → used result" : $"  → answer lacks '{want}': {Clip(answer)}";
            }
            catch (Exception e) when (e is HttpRequestException or TaskCanceledException or JsonException)
            {
                answerOk = false;
                detail += $"  → follow-up failed: {e.Message}";
            }
        }
        return new CaseResult(id, nameOk, argsOk, answerOk, raw, detail, clock.Elapsed.TotalSeconds);
    }

    private static JsonObject Complete(HttpClient http, string endpoint, string model, JsonArray messages,
        JsonNode? tools, int maxTokens, bool noThink)
    {
        var body = new JsonObject
        {
            ["model"] = model,
            ["messages"] = messages.DeepClone(),
            ["temperature"] = 0,
            ["max_tokens"] = maxTokens,
        };
        if (tools is not null)
        {
            body["tools"] = tools.DeepClone();
        }
        if (noThink)
        {
            body["chat_template_kwargs"] = new JsonObject { ["enable_thinking"] = false };
        }
        using var response = http.PostAsync($"{endpoint}/chat/completions",
            new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json")).GetAwaiter().GetResult();
        string text = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException($"{(int)response.StatusCode}: {Clip(text, 200)}");
        }
        var root = JsonNode.Parse(text)!;
        return root["choices"]![0]!["message"]!.AsObject();
    }

    private sealed record Call(string Name, Dictionary<string, string> Args, string? Id);

    /// <summary>Structured <c>tool_calls</c> first; otherwise the raw
    /// <c>&lt;tool_call&gt;</c> blocks in the content (raw = true).</summary>
    private static (List<Call> Calls, bool Raw) ExtractCalls(JsonObject message)
    {
        var calls = new List<Call>();
        if (message["tool_calls"] is JsonArray structured && structured.Count > 0)
        {
            foreach (var item in structured)
            {
                var fn = item!["function"]!;
                calls.Add(new Call(fn["name"]!.GetValue<string>(), ParseArgs(fn["arguments"]), item["id"]?.GetValue<string>()));
            }
            return (calls, false);
        }
        string content = message["content"]?.GetValue<string>() ?? string.Empty;
        foreach (Match block in Regex.Matches(content, @"<tool_call>(.*?)</tool_call>", RegexOptions.Singleline))
        {
            string inner = block.Groups[1].Value;
            var fn = Regex.Match(inner, @"<function=([^>\s]+)>(.*?)</function>", RegexOptions.Singleline);
            if (fn.Success)
            {
                var args = new Dictionary<string, string>(StringComparer.Ordinal);
                foreach (Match p in Regex.Matches(fn.Groups[2].Value, @"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", RegexOptions.Singleline))
                {
                    args[p.Groups[1].Value] = p.Groups[2].Value;
                }
                calls.Add(new Call(fn.Groups[1].Value, args, null));
                continue;
            }
            try
            {
                var json = JsonNode.Parse(inner.Trim())!;
                calls.Add(new Call(json["name"]!.GetValue<string>(), ParseArgs(json["arguments"]), null));
            }
            catch (Exception e) when (e is JsonException or NullReferenceException or InvalidOperationException)
            {
                // Not a call we can read — leave it out; the case fails on "no tool call".
            }
        }
        return (calls, calls.Count > 0);
    }

    private static Dictionary<string, string> ParseArgs(JsonNode? node)
    {
        var args = new Dictionary<string, string>(StringComparer.Ordinal);
        if (node is JsonValue v && v.TryGetValue<string>(out var s))
        {
            try
            {
                node = JsonNode.Parse(s);
            }
            catch (JsonException)
            {
                return args;
            }
        }
        if (node is JsonObject obj)
        {
            foreach (var (key, value) in obj)
            {
                args[key] = value is JsonValue jv && jv.TryGetValue<string>(out var str) ? str : value?.ToJsonString() ?? "null";
            }
        }
        return args;
    }

    private static bool ArgMatches(string want, string got)
    {
        if (want.StartsWith('~'))
        {
            return Regex.IsMatch(got, want[1..], RegexOptions.IgnoreCase);
        }
        string Norm(string x) => x.Trim().Trim('"').ToLowerInvariant();
        if (Norm(want) == Norm(got))
        {
            return true;
        }
        return double.TryParse(Norm(want), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var a) &&
               double.TryParse(Norm(got), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var b) &&
               Math.Abs(a - b) < 1e-9;
    }

    private static string Clip(string s, int max = 80)
    {
        s = s.Replace('\n', ' ').Trim();
        return s.Length <= max ? s : s[..max] + "…";
    }
}
