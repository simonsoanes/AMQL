using System.Text.Json;

namespace Amql.Server;

/// <summary>One call a reply made: the function and its arguments as a JSON
/// object string, as the OpenAI APIs carry them.</summary>
public sealed record ParsedToolCall(string Name, string Arguments);

/// <summary>Something a reply said, as it is read: reasoning or content text
/// (deltas), or a whole tool call once it closes.</summary>
public abstract record ReplyEvent
{
    public sealed record Reasoning(string Delta) : ReplyEvent;

    public sealed record Content(string Delta) : ReplyEvent;

    public sealed record ToolCall(int Index, ParsedToolCall Call) : ReplyEvent;
}

/// <summary>
/// Reads a reply as it is generated and splits it the way the
/// OpenAI APIs report it: the thinking block becomes reasoning, the text
/// after it content, and what its model's <see cref="CallSyntax"/> reads as calls
/// calls whose arguments are a JSON object. Nothing is sent before it
/// is certain: text that could be the start of <c>&lt;/think&gt;</c> or of a
/// call, and trailing whitespace, wait for the next delta.
/// Text after the last call is dropped, since the template tells the model to
/// write nothing there; a call that never closed is returned as content, so a
/// reply cut off by the token limit never writes a truncated file.
/// </summary>
public sealed class ReplyParser
{
    private const string ThinkOpen = "<think>";
    private const string ThinkClose = "</think>";

    private readonly CallSyntax? _calling;
    private readonly bool _promptOpenedThink;
    private readonly bool _parseTools;
    private readonly IReadOnlyDictionary<string, JsonElement> _schemas;
    private readonly LineDecoration? _decoration;
    private string _text = string.Empty;
    private int _reasoningSent;
    private int _contentSent;
    private int _blocksRead;
    private readonly List<ParsedToolCall> _calls = new();

    /// <param name="promptOpenedThink">The prompt ended inside a think block
    /// (thinking enabled), so the reply starts with reasoning.</param>
    /// <param name="tools">The request's tools in chat form
    /// (<c>{"type":"function","function":{name, parameters}}</c>), whose
    /// parameter types decide how each value is read; null when the request
    /// offered none, and then a call is ordinary text.</param>
    /// <param name="decoration">The tool results' per-line decoration, removed from a value that
    /// copied it (null: values are read as written).</param>
    /// <param name="syntax">How the model writes its calls (null: it writes none).</param>
    public ReplyParser(bool promptOpenedThink, IReadOnlyList<JsonElement>? tools, LineDecoration? decoration = null, CallSyntax? syntax = null)
    {
        _calling = syntax;
        _promptOpenedThink = promptOpenedThink;
        _decoration = decoration;
        _parseTools = syntax is not null && tools is { Count: > 0 };
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

    /// <summary>How many call values had a tool result's line decoration removed.</summary>
    public int Undecorated { get; private set; }

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
    public static ReplyParser Parse(string text, bool promptOpenedThink, IReadOnlyList<JsonElement>? tools, LineDecoration? decoration = null,
        CallSyntax? syntax = null)
    {
        var parser = new ReplyParser(promptOpenedThink, tools, decoration, syntax);
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
        int call = _parseTools ? NextCall(body, 0) : -1;
        string content = call >= 0 ? body[..call] : body;
        if (call < 0 && _parseTools && !final)
        {
            content = !_calling!.AtStartOnly ? HoldBack(content, _calling.Open)
                : _calling.Open.StartsWith(body.TrimStart(), StringComparison.Ordinal) ? "" : content;
        }
        content = content.TrimStart();
        // Trailing whitespace waits: it may be the gap before a call or the end.
        content = content.TrimEnd();

        if (call >= 0)
        {
            int at = call, blocks = 0;
            while (at >= 0)
            {
                if (_calling!.Block(body, at, final) is not { } block)
                {
                    if (final && _calls.Count == 0)
                    {
                        // A call cut off before it closed: shown, never run.
                        content = (content + (content.Length > 0 ? "\n\n" : "") + body[at..].TrimEnd()).Trim();
                    }
                    break;
                }
                if (blocks++ == _blocksRead)
                {
                    _blocksRead++;
                    foreach (var parsed in ReadCalls(body[block.From..block.To]))
                    {
                        _calls.Add(parsed);
                        events.Add(new ReplyEvent.ToolCall(_calls.Count - 1, parsed));
                    }
                }
                at = NextCall(body, block.Next);
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

    /// <summary>Where the next calls open, or −1.</summary>
    private int NextCall(string body, int from)
    {
        if (!_calling!.AtStartOnly)
        {
            return body.IndexOf(_calling.Open, from, StringComparison.Ordinal);
        }
        while (from < body.Length && (char.IsWhiteSpace(body[from]) || (from > 0 && body[from] is ';' or ',')))
        {
            from++;
        }
        return string.CompareOrdinal(body, from, _calling.Open, 0, _calling.Open.Length) == 0 ? from : -1;
    }

    private IReadOnlyList<ParsedToolCall> ReadCalls(string inner) => _calling!.Read(inner, _schemas, value =>
    {
        if (_decoration is null)
        {
            return value;
        }
        value = _decoration.Strip(value, out bool stripped);
        Undecorated += stripped ? 1 : 0;
        return value;
    });

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
