using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Amql.Server;

/// <summary>One call a reply made: the function and its arguments as a JSON
/// object string, as the OpenAI APIs carry them.</summary>
public sealed record ParsedToolCall(string Name, string Arguments);

/// <summary>Something a reply said, as it is read: reasoning or content text
/// (deltas), or a whole tool call once its <c>&lt;/tool_call&gt;</c> arrives.</summary>
public abstract record ReplyEvent
{
    public sealed record Reasoning(string Delta) : ReplyEvent;

    public sealed record Content(string Delta) : ReplyEvent;

    public sealed record ToolCall(int Index, ParsedToolCall Call) : ReplyEvent;
}

/// <summary>
/// Reads a Qwen3.5 reply as it is generated and splits it the way the
/// OpenAI APIs report it: the thinking block becomes reasoning, the text
/// after it content, and each
/// <c>&lt;tool_call&gt;&lt;function=NAME&gt;&lt;parameter=K&gt;V&lt;/parameter&gt;…</c>
/// block a call whose arguments are a JSON object. Nothing is sent before it
/// is certain: text that could be the start of <c>&lt;/think&gt;</c> or
/// <c>&lt;tool_call&gt;</c>, and trailing whitespace, wait for the next delta.
/// Text after the last call is dropped, since the template tells the model to
/// write nothing there; a call that never closed is returned as content, so a
/// reply cut off by the token limit never writes a truncated file.
/// </summary>
public sealed class ReplyParser
{
    private const string ThinkOpen = "<think>";
    private const string ThinkClose = "</think>";
    private const string CallOpen = "<tool_call>";
    private const string CallClose = "</tool_call>";

    private static readonly Regex Function = new(@"^\s*<function=([^>\n]+)>(.*?)(?:</function>\s*)?$", RegexOptions.Singleline);
    private static readonly Regex Parameter = new(@"<parameter=([^>\n]+)>(.*?)</parameter>", RegexOptions.Singleline);

    // Arguments as a client would write them: "<", ">" and "&" stay as they are (still valid JSON).
    private static readonly JsonSerializerOptions Relaxed = new()
    {
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    private readonly bool _promptOpenedThink;
    private readonly bool _parseTools;
    private readonly IReadOnlyDictionary<string, JsonElement> _schemas;
    private string _text = string.Empty;
    private int _reasoningSent;
    private int _contentSent;
    private readonly List<ParsedToolCall> _calls = new();

    /// <param name="promptOpenedThink">The prompt ended inside a think block
    /// (thinking enabled), so the reply starts with reasoning.</param>
    /// <param name="tools">The request's tools in chat form
    /// (<c>{"type":"function","function":{name, parameters}}</c>), whose
    /// parameter types decide how each value is read; null when the request
    /// offered none, and then a <c>&lt;tool_call&gt;</c> is ordinary text.</param>
    public ReplyParser(bool promptOpenedThink, IReadOnlyList<JsonElement>? tools)
    {
        _promptOpenedThink = promptOpenedThink;
        _parseTools = tools is { Count: > 0 };
        var schemas = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (var t in tools ?? Array.Empty<JsonElement>())
        {
            var f = t.TryGetProperty("function", out var fn) ? fn : t;
            if (f.TryGetProperty("name", out var n) && n.ValueKind == JsonValueKind.String)
            {
                schemas[n.GetString()!] = f.TryGetProperty("parameters", out var p) ? p : default;
            }
        }
        _schemas = schemas;
    }

    /// <summary>The reasoning, content and calls of the whole reply, once <see cref="Finish"/> has run.</summary>
    public string Reasoning { get; private set; } = string.Empty;

    public string Content { get; private set; } = string.Empty;

    public IReadOnlyList<ParsedToolCall> ToolCalls => _calls;

    /// <summary>Whether the reply had a thinking block at all.</summary>
    public bool Thought { get; private set; }

    /// <summary>Drops every call after the first (<c>parallel_tool_calls: false</c>).</summary>
    public void KeepFirstCall()
    {
        if (_calls.Count > 1)
        {
            _calls.RemoveRange(1, _calls.Count - 1);
        }
    }

    /// <summary>Reads the next piece of the reply.</summary>
    public IReadOnlyList<ReplyEvent> Feed(string delta)
    {
        _text += delta;
        return Read(final: false);
    }

    /// <summary>Reads what was held back, at the end of the reply.</summary>
    public IReadOnlyList<ReplyEvent> Finish() => Read(final: true);

    /// <summary>Parses a whole reply at once.</summary>
    public static ReplyParser Parse(string text, bool promptOpenedThink, IReadOnlyList<JsonElement>? tools)
    {
        var parser = new ReplyParser(promptOpenedThink, tools);
        parser.Feed(text);
        parser.Finish();
        return parser;
    }

    private List<ReplyEvent> Read(bool final)
    {
        var events = new List<ReplyEvent>();
        string text = _text;

        // ── reasoning ──
        int bodyStart = 0;
        bool thinking = _promptOpenedThink;
        if (!thinking)
        {
            string lead = text.TrimStart();
            if (lead.StartsWith(ThinkOpen, StringComparison.Ordinal))
            {
                thinking = true;
            }
            else if (!final && lead.Length < ThinkOpen.Length && ThinkOpen.StartsWith(lead, StringComparison.Ordinal))
            {
                return events;     // could still become "<think>"
            }
        }
        if (thinking)
        {
            Thought = true;
            int from = 0;
            if (!_promptOpenedThink || text.TrimStart().StartsWith(ThinkOpen, StringComparison.Ordinal))
            {
                int open = text.IndexOf(ThinkOpen, StringComparison.Ordinal);
                from = open >= 0 ? open + ThinkOpen.Length : 0;
            }
            int close = text.IndexOf(ThinkClose, from, StringComparison.Ordinal);
            string reasoning;
            if (close < 0)
            {
                reasoning = text[from..];
                if (!final)
                {
                    reasoning = HoldBack(reasoning, ThinkClose);
                }
            }
            else
            {
                reasoning = text[from..close];
                bodyStart = close + ThinkClose.Length;
            }
            reasoning = reasoning.TrimStart('\n');
            reasoning = reasoning.TrimEnd();
            if (close >= 0 || final)
            {
                Reasoning = reasoning;
            }
            if (reasoning.Length > _reasoningSent)
            {
                events.Add(new ReplyEvent.Reasoning(reasoning[_reasoningSent..]));
                _reasoningSent = reasoning.Length;
            }
            if (close < 0)
            {
                if (final)
                {
                    Content = string.Empty;
                }
                return events;
            }
        }

        // ── content, then calls ──
        string body = text[bodyStart..];
        int call = _parseTools ? body.IndexOf(CallOpen, StringComparison.Ordinal) : -1;
        string content = call >= 0 ? body[..call] : body;
        if (call < 0 && _parseTools && !final)
        {
            content = HoldBack(content, CallOpen);
        }
        content = content.TrimStart();
        // Trailing whitespace waits: it may be the gap before a call or the end.
        content = content.TrimEnd();

        if (call >= 0)
        {
            int at = call;
            while (at >= 0)
            {
                int end = body.IndexOf(CallClose, at, StringComparison.Ordinal);
                if (end < 0)
                {
                    if (final && _calls.Count == 0)
                    {
                        // A call cut off before it closed: shown, never run.
                        content = (content + (content.Length > 0 ? "\n\n" : "") + body[at..].TrimEnd()).Trim();
                    }
                    break;
                }
                string inner = body[(at + CallOpen.Length)..end];
                int index = CountCalls(body, call, at);
                if (index >= _calls.Count && ReadCall(inner) is { } parsed)
                {
                    _calls.Add(parsed);
                    events.Add(new ReplyEvent.ToolCall(_calls.Count - 1, parsed));
                }
                at = body.IndexOf(CallOpen, end + CallClose.Length, StringComparison.Ordinal);
            }
        }

        if (content.Length > _contentSent)
        {
            // Content is only ever sent before the first call; once a call
            // starts, nothing more is added to it (unless it never closed).
            events.Insert(events.FindIndex(e => e is ReplyEvent.ToolCall) is var i and >= 0 ? i : events.Count,
                new ReplyEvent.Content(content[_contentSent..]));
            _contentSent = content.Length;
        }
        if (final || call >= 0)
        {
            Content = content;
        }
        return events;
    }

    /// <summary>How many complete call blocks precede the one at <paramref name="at"/>.</summary>
    private static int CountCalls(string body, int first, int at)
    {
        int n = 0;
        for (int i = first; i < at;)
        {
            int end = body.IndexOf(CallClose, i, StringComparison.Ordinal);
            if (end < 0 || end >= at)
            {
                break;
            }
            n++;
            i = body.IndexOf(CallOpen, end, StringComparison.Ordinal);
            if (i < 0)
            {
                break;
            }
        }
        return n;
    }

    private ParsedToolCall? ReadCall(string inner)
    {
        var m = Function.Match(inner);
        if (!m.Success)
        {
            return null;
        }
        string name = m.Groups[1].Value.Trim();
        var properties = _schemas.TryGetValue(name, out var schema) && schema.ValueKind == JsonValueKind.Object
            && schema.TryGetProperty("properties", out var props) && props.ValueKind == JsonValueKind.Object
            ? props
            : default;
        var args = new JsonObject();
        foreach (Match p in Parameter.Matches(m.Groups[2].Value))
        {
            string key = p.Groups[1].Value.Trim();
            string value = p.Groups[2].Value;
            // The template writes "<parameter=K>\nVALUE\n</parameter>".
            if (value.StartsWith('\n'))
            {
                value = value[1..];
            }
            if (value.EndsWith('\n'))
            {
                value = value[..^1];
            }
            string? type = properties.ValueKind == JsonValueKind.Object && properties.TryGetProperty(key, out var ps)
                ? TypeOf(ps)
                : null;
            args[key] = Value(value, type);
        }
        return new ParsedToolCall(name, args.ToJsonString(Relaxed));
    }

    private static string? TypeOf(JsonElement schema)
    {
        if (!schema.TryGetProperty("type", out var t))
        {
            return null;
        }
        if (t.ValueKind == JsonValueKind.String)
        {
            return t.GetString();
        }
        // ["integer", "null"] and the like: the first non-null type.
        return t.ValueKind == JsonValueKind.Array
            ? t.EnumerateArray().Select(x => x.GetString()).FirstOrDefault(x => x is not null and not "null")
            : null;
    }

    /// <summary>A parameter's text as the JSON value its schema asks for. A
    /// value that does not read as that type stays a string, and the client's
    /// own validation reports it.</summary>
    private static JsonNode? Value(string text, string? type)
    {
        string t = text.Trim();
        switch (type)
        {
            case "integer" when long.TryParse(t, NumberStyles.Integer, CultureInfo.InvariantCulture, out long l):
                return JsonValue.Create(l);
            case "integer" or "number" when JsonNumber(t) is { } big:
                return big;   // a valid JSON number too large for a long or double, kept as written
            case "number" when double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out double d) && double.IsFinite(d):
                return t.Contains('.') || t.Contains('e') || t.Contains('E') ? JsonValue.Create(d)
                    : long.TryParse(t, NumberStyles.Integer, CultureInfo.InvariantCulture, out long whole) ? JsonValue.Create(whole) : JsonValue.Create(d);
            case "boolean" when t.Equals("true", StringComparison.OrdinalIgnoreCase) || t.Equals("false", StringComparison.OrdinalIgnoreCase):
                return JsonValue.Create(t.Equals("true", StringComparison.OrdinalIgnoreCase));
            case "object" or "array":
                try
                {
                    var node = JsonNode.Parse(t);
                    if (node is JsonObject or JsonArray)
                    {
                        return node;
                    }
                }
                catch (JsonException)
                {
                }
                return JsonValue.Create(text);
            case "null" when t == "null" || t == "None":
                return null;
            default:
                return JsonValue.Create(text);
        }
    }

    private static JsonNode? JsonNumber(string t)
    {
        try
        {
            return JsonNode.Parse(t) is JsonValue v && v.GetValueKind() == JsonValueKind.Number ? v : null;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    /// <summary>The text without a suffix that could be the start of <paramref name="marker"/>.</summary>
    private static string HoldBack(string text, string marker)
    {
        for (int n = Math.Min(marker.Length - 1, text.Length); n > 0; n--)
        {
            if (text.EndsWith(marker[..n], StringComparison.Ordinal))
            {
                return text[..^n];
            }
        }
        return text;
    }
}
