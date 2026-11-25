using System.Text.Json;

namespace Amql.Cli;

/// <summary>
/// What a chat template does, learned by rendering probe conversations and comparing: whether it writes
/// tools, calls, results and images, how it marks reasoning, how it writes calls (<see cref="CallShape"/>),
/// and what opens and closes turns. So a model is served by its own template, whatever its family.
/// </summary>
public sealed class ChatDialect
{
    public bool WritesTools { get; private init; }
    public bool WritesCalls { get; private init; }
    public bool WritesResults { get; private init; }

    /// <summary>How calls are written, or null with <see cref="CallsRefusal"/> when they cannot be read back.</summary>
    public CallShape? Calls { get; private init; }
    public string? CallsRefusal { get; private init; }

    /// <summary>What an assistant says along with its calls would come after their results.</summary>
    public bool SaysAfterResults { get; private init; }

    /// <summary>What stands for an image, or null when images are not written.</summary>
    public string? Image { get; private init; }

    /// <summary>What opens and closes reasoning, with the space around it, or null when the model has none.</summary>
    public string? ThinkOpen { get; private init; }
    public string? ThinkClose { get; private init; }

    /// <summary>The marks alone.</summary>
    public string ThinkOpenMark => ThinkOpen?.Trim() ?? "";
    public string ThinkCloseMark => ThinkClose?.Trim() ?? "";

    /// <summary>The template takes thinking as a switch.</summary>
    public bool ThinkSwitch { get; private init; }

    /// <summary>What a prompt that is not to think ends with, beyond one that may; empty when nothing does.</summary>
    public string EmptyThinking { get; private init; } = "";

    /// <summary>What follows an assistant's words (its turn's end), and stands between them and the next user's.</summary>
    public string AfterReply { get; private init; } = "";
    public string BetweenTurns { get; private init; } = "";

    /// <summary>What stands between a user's words and the assistant's.</summary>
    public string BeforeReply { get; private init; } = "";

    /// <summary>What follows a turn's last call, before its result.</summary>
    public string BeforeResult { get; private init; } = "";

    /// <summary>Whether a rendered prompt ends inside the reasoning.</summary>
    public bool OpensThinking(string rendered) => ThinkOpenMark.Length > 0 && rendered.TrimEnd().EndsWith(ThinkOpenMark, StringComparison.Ordinal);

    private const string User = "qzuserq", Second = "qzsecondq", Said = "Qzsaidq", Why = "qzwhyq", Result = "qzresultq";

    internal static ChatDialect Learn(ChatTemplate template)
    {
        string? Try(bool thinking, bool generation, IReadOnlyList<JsonElement>? tools, params ChatMessage[] messages)
        {
            try
            {
                return template.Render(messages, thinking, tools, generation);
            }
            catch (ChatTemplateException)
            {
                return null;
            }
        }
        static string? After(string? s, string what) => s is not null && s.LastIndexOf(what, StringComparison.Ordinal) is var i and >= 0 ? s[(i + what.Length)..] : null;
        static string? Before(string? s, string what) => s is not null && s.IndexOf(what, StringComparison.Ordinal) is var i and >= 0 ? s[..i] : null;

        var user = new ChatMessage("user", User);
        var said = new ChatMessage("assistant", Said);
        var tools = CallShape.ProbeTools;

        // ── turns ──
        string afterReply = After(Try(false, false, null, user, said), Said) ?? "";
        string between = Before(After(Try(false, false, null, user, said, new ChatMessage("user", Second)), Said), Second) ?? "";

        // ── reasoning ──
        string? off = Try(false, true, null, user), on = Try(true, true, null, user);
        string offEnd = After(off, User) ?? "", onEnd = After(on, User) ?? "";
        string? open = null, close = null;
        if (After(Try(true, false, null, user, said with { Reasoning = Why }), User) is { } reasoned &&
            reasoned.IndexOf(Why, StringComparison.Ordinal) is var w and >= 0 && reasoned.IndexOf(Said, w, StringComparison.Ordinal) is var s and >= 0)
        {
            string pre = reasoned[..w];
            close = reasoned[(w + Why.Length)..s];
            string mark = close.Trim();
            if (mark.StartsWith("</", StringComparison.Ordinal) && pre.LastIndexOf("<" + mark[2..], StringComparison.Ordinal) is var at and >= 0)
            {
                open = pre[at..];   // <x> … </x>
            }
            else
            {
                // What the turn starts with when there is no reasoning, the shortest way it starts, is not the mark.
                string? header = new[] { Before(After(Try(false, false, null, user, said, new ChatMessage("user", Second)), User), Said), onEnd, offEnd }
                    .Where(h => h is { Length: > 0 } && pre.StartsWith(h, StringComparison.Ordinal) && h.Length < pre.Length).MinBy(h => h!.Length);
                open = header is null ? null : pre[header.Length..];
            }
            if (open is null || open.Trim().Length == 0 || mark.Length == 0)
            {
                open = close = null;
            }
        }
        if (open is null && off is not null && onEnd.Length < offEnd.Length && offEnd.StartsWith(onEnd, StringComparison.Ordinal) &&
            System.Text.RegularExpressions.Regex.Matches(offEnd[onEnd.Length..], @"<[^<>\s]+>|\[[^\[\]\s]+\]") is [var first, var second])
        {
            // Thinking that is off: an empty block, its two marks.
            string blank = offEnd[onEnd.Length..];
            (open, close) = (blank[first.Index..second.Index], blank[second.Index..]);
        }
        if (open is null && template.ResponseTemplate is { } response && response.TryGetProperty("fields", out var fields) && fields.TryGetProperty("thinking", out var t) &&
            t.TryGetProperty("open", out var o) && t.TryGetProperty("close", out var c) && o.ValueKind == JsonValueKind.String && c.ValueKind == JsonValueKind.String)
        {
            (open, close) = (o.GetString(), c.GetString());
        }
        if (open is null && template.Source.Contains("</think>", StringComparison.Ordinal))
        {
            (open, close) = ("<think>\n", "\n</think>\n\n");   // written by the model, taken off by the template
        }
        string empty = "";
        if (open is not null && offEnd.LastIndexOf(open.Trim(), StringComparison.Ordinal) is var e and >= 0 && offEnd.IndexOf(close!.Trim(), e, StringComparison.Ordinal) >= 0)
        {
            empty = offEnd[e..];
        }

        // ── images ──
        string? image = null;
        if (Try(false, false, null, new ChatMessage("user", Second) { Parts = [new ChatPart(null), new ChatPart(Second)] }) is { } pictured &&
            Try(false, false, null, new ChatMessage("user", Second)) is { } plain && Before(pictured, Second) is { } withImage && Before(plain, Second) is { } without &&
            withImage.Length > without.Length && withImage.StartsWith(without, StringComparison.Ordinal) && !System.Text.RegularExpressions.Regex.IsMatch(withImage[without.Length..], "['\"]type['\"]"))
        {
            image = withImage[without.Length..];
        }

        // ── tools ──
        var called = new ChatMessage("assistant", "") { ToolCalls = [CallShape.Probe(CallShape.ProbeName, "{}")] };
        var result = new ChatMessage("tool", Result) { CallId = called.ToolCalls[0].Id, Name = CallShape.ProbeName };
        string? Body(ChatToolCall[] calls)
        {
            // The turn's own text: past what a turn that says something starts with, and before what it ends with.
            string? words = After(Try(false, false, tools, user, said), User), written = After(Try(false, false, tools, user, called with { ToolCalls = calls }), User);
            if (words is null || written is null)
            {
                return null;
            }
            int from = CallShape.CommonStart(written, Before(words, Said) ?? "");
            int opened = from > 0 ? written.LastIndexOfAny(['<', '['], from - 1) : -1;
            if (opened >= 0 && written.IndexOfAny(['>', ']'], opened, from - opened) < 0)
            {
                from = opened;   // both texts go on with a mark: it belongs to what follows
            }
            string ends = After(words, Said) ?? "";
            string body = written[from..];
            return ends.Length > 0 && body.EndsWith(ends, StringComparison.Ordinal) ? body[..^ends.Length] : body;
        }
        bool writesTools = Try(false, true, tools, user)?.Contains(CallShape.ProbeName) == true;
        bool writesCalls = Body(called.ToolCalls.ToArray())?.Contains(CallShape.ProbeName) == true;
        string? answered = Try(false, true, tools, user, called, result);
        bool writesResults = answered?.Contains(Result) == true;
        string? refusal = null;
        var shape = !writesCalls ? null : CallShape.Learn(Body, out refusal);
        bool late = writesCalls && writesResults &&
            Try(false, true, tools, user, called with { Content = Said }, result) is { } spoken && spoken.IndexOf(Said, StringComparison.Ordinal) > spoken.IndexOf(Result, StringComparison.Ordinal) &&
            Try(false, true, tools, user, said, called, result) is { } apart && apart.IndexOf(Said, StringComparison.Ordinal) < apart.IndexOf(Result, StringComparison.Ordinal);
        string beforeResult = "";
        if (writesResults && After(answered, User) is { } exchange && Before(exchange, Result) is { } upTo)
        {
            string closer = shape is null ? CallShape.ProbeName : shape.Closer;
            beforeResult = closer.Length > 0 && After(upTo, closer) is { } rest ? rest : upTo;
        }
        return new ChatDialect
        {
            WritesTools = writesTools, WritesCalls = writesCalls, WritesResults = writesResults,
            Calls = shape, CallsRefusal = shape is null ? refusal ?? "its chat template does not write calls" : null, SaysAfterResults = late,
            Image = image, ThinkOpen = open, ThinkClose = close, ThinkSwitch = off is not null && on is not null && off != on, EmptyThinking = empty,
            AfterReply = afterReply, BetweenTurns = between, BeforeReply = Before(After(Try(false, false, null, user, said), User), Said) ?? "", BeforeResult = beforeResult,
        };
    }
}
