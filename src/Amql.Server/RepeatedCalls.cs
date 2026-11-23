using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Amql.Cli;

namespace Amql.Server;

/// <summary>
/// Calls that have stopped telling the agent anything: the same call (tool and arguments) made at
/// least twice already, with the same result each of the last two times. A small model at
/// temperature 0 copies its own previous call from the conversation and goes round and round
/// ("run it again", the same output, "run it again"); a third identical call cannot return
/// anything new. A reply that makes one is generated again with calls closed, so it answers from
/// the results it already has.
/// </summary>
public static class RepeatedCalls
{
    // Results differ in what changes every time (build ids, timestamps, durations); those are not news.
    private static readonly Regex Volatile = new(@"[0-9a-fA-F]*\d[0-9a-fA-F]*", RegexOptions.Compiled);

    /// <summary>The stale calls of a conversation, as <see cref="Key"/>s.</summary>
    public static HashSet<string> Stale(IReadOnlyList<ChatMessage> messages)
    {
        var results = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        for (int i = 0; i < messages.Count; i++)
        {
            if (messages[i].Role != "assistant" || messages[i].ToolCalls is not { Count: > 0 } calls)
            {
                continue;
            }
            // The calls' results follow the turn, in order.
            for (int c = 0; c < calls.Count && i + 1 + c < messages.Count && messages[i + 1 + c].Role == "tool"; c++)
            {
                string key = Key(calls[c].Name, calls[c].Arguments.ValueKind == JsonValueKind.Undefined ? "{}" : calls[c].Arguments.GetRawText());
                (results.TryGetValue(key, out var list) ? list : results[key] = new List<string>())
                    .Add(Volatile.Replace(messages[i + 1 + c].Content, "#"));
            }
        }
        var stale = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (key, list) in results)
        {
            if (list.Count >= 2 && list[^1] == list[^2])
            {
                stale.Add(key);
            }
        }
        return stale;
    }

    /// <summary>A call as a comparable string: the tool's name and its arguments with keys sorted.</summary>
    public static string Key(string name, string arguments)
    {
        JsonNode? args;
        try
        {
            args = JsonNode.Parse(arguments);
        }
        catch (JsonException)
        {
            return name + " " + arguments;
        }
        return name + " " + Canonical(args);
    }

    private static string Canonical(JsonNode? node) => node switch
    {
        JsonObject o => "{" + string.Join(",", o.OrderBy(p => p.Key, StringComparer.Ordinal)
            .Select(p => JsonSerializer.Serialize(p.Key) + ":" + Canonical(p.Value))) + "}",
        JsonArray a => "[" + string.Join(",", a.Select(Canonical)) + "]",
        null => "null",
        _ => node.ToJsonString(),
    };
}
