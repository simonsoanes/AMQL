using System.Text.RegularExpressions;
using Amql.Hf;

namespace Amql.Cli;

/// <summary>The chat marks an Entail program may name instead of spelling them, so one program
/// reads any model's turns: <c>token "{mark:think_close}" means …</c>, <c>emit "{mark:call_open}" when …</c>.
/// <see cref="Fill"/> replaces each with the partner's own token, learned from its chat template
/// as the server learns it (<see cref="ChatDialect"/>), so no family is named.</summary>
/// <remarks>
/// <list type="table">
/// <item><term>think_open, think_close</term><description>The tokens that open and close reasoning: <c>&lt;think&gt;</c>, <c>&lt;/think&gt;</c>; <c>&lt;|channel&gt;</c>, <c>&lt;channel|&gt;</c>.</description></item>
/// <item><term>reply_open</term><description>The last token the template writes between reasoning and the reply: <c>\n\n</c> after Qwen's <c>&lt;/think&gt;</c>, Gemma's <c>&lt;channel|&gt;</c> itself. The reply's first token comes next.</description></item>
/// <item><term>call_open</term><description>The token a call starts with: <c>&lt;tool_call&gt;</c>, <c>&lt;|tool_call&gt;</c>.</description></item>
/// <item><term>turn_open</term><description>The token a turn starts with, before its role: <c>&lt;|im_start|&gt;</c>, <c>&lt;|turn&gt;</c>.</description></item>
/// <item><term>turn_close, eos</term><description>What ends a reply: <c>&lt;|im_end|&gt;</c>, <c>&lt;turn|&gt;</c>; and the tokenizer's end of text.</description></item>
/// <item><term>system, user, assistant</term><description>The role tokens that follow it: <c>assistant</c>, <c>model</c>.</description></item>
/// </list>
/// </remarks>
public static class EntailMarks
{
    private static readonly Regex Name = new(@"(?<!\\)\{mark:(think_open|think_close|reply_open|call_open|turn_open|turn_close|eos|system|user|assistant)\}", RegexOptions.CultureInvariant);

    private const string SystemProbe = "qzsystemq", UserProbe = "qzuserq", SaidProbe = "Qzsaidq";

    /// <summary>The program with its marks spelt in the partner's tokens; unchanged when it names none.</summary>
    public static string Fill(string source, string partnerDir)
    {
        if (!Name.IsMatch(source))
        {
            return source;
        }
        var template = ChatTemplate.Load(partnerDir)
            ?? throw new CliException($"the program names chat marks, and '{partnerDir}' has no chat template to learn them from");
        var marks = new Dictionary<string, string>(Learn(template, HfTokenizer.FromModelDir(partnerDir)), StringComparer.Ordinal);
        if (EndOfText(partnerDir) is { } eos)
        {
            marks["eos"] = eos;
        }
        return Name.Replace(source, m => marks.TryGetValue(m.Groups[1].Value, out var v) ? v
            : throw new CliException($"the program names {{mark:{m.Groups[1].Value}}}, and '{partnerDir}''s chat template writes none"));
    }

    /// <summary>The tokenizer's end-of-text token, from <c>tokenizer_config.json</c>.</summary>
    private static string? EndOfText(string dir)
    {
        string path = Path.Combine(dir, "tokenizer_config.json");
        if (!File.Exists(path))
        {
            return null;
        }
        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(path));
        if (!doc.RootElement.TryGetProperty("eos_token", out var eos))
        {
            return null;
        }
        string? text = eos.ValueKind == System.Text.Json.JsonValueKind.String ? eos.GetString()
            : eos.ValueKind == System.Text.Json.JsonValueKind.Object && eos.TryGetProperty("content", out var c) ? c.GetString() : null;
        return text is { Length: > 0 } && !text.Contains('"') && !text.Contains('\\') ? text : null;
    }

    /// <summary>Each mark the template writes, as one of the partner's tokens.</summary>
    public static IReadOnlyDictionary<string, string> Learn(ChatTemplate template, HfTokenizer tokenizer) =>
        Learn(template, text => tokenizer.EncodeToIds(text).Select(id => tokenizer.Decode([id])).ToArray());

    /// <summary>Each mark the template writes, as one of the pieces <paramref name="cut"/> makes of its text.</summary>
    public static IReadOnlyDictionary<string, string> Learn(ChatTemplate template, Func<string, string[]> cut)
    {
        string[] Pieces(string text) => cut(text);
        var marks = new Dictionary<string, string>(StringComparer.Ordinal);
        void Put(string name, string? token)
        {
            if (token is { Length: > 0 } && !token.Contains('"') && !token.Contains('\\'))
            {
                marks[name] = token;
            }
        }
        string? Render(bool generation, params ChatMessage[] messages)
        {
            try
            {
                return template.Render(messages, thinking: false, tools: null, generation);
            }
            catch (ChatTemplateException)
            {
                return null;
            }
        }

        var dialect = template.Dialect;
        if (dialect.ThinkOpenMark.Length > 0 && dialect.ThinkCloseMark.Length > 0)
        {
            Put("think_open", Pieces(dialect.ThinkOpenMark).FirstOrDefault());
            Put("think_close", Pieces(dialect.ThinkCloseMark).LastOrDefault());
            // What the template writes after the close mark, newlines included, ends with the reply's lead.
            string close = dialect.ThinkClose!;
            string after = close[(close.IndexOf(dialect.ThinkCloseMark, StringComparison.Ordinal) + dialect.ThinkCloseMark.Length)..];
            Put("reply_open", after.Length > 0 ? Pieces(after).LastOrDefault() : marks.GetValueOrDefault("think_close"));
        }
        if (dialect.AfterReply.Trim() is { Length: > 0 } ends)
        {
            Put("turn_close", Pieces(ends).FirstOrDefault());
        }
        if (dialect.Calls?.Opener is { Length: > 0 } opener)
        {
            Put("call_open", Pieces(opener).FirstOrDefault());
        }

        // A turn's header is its opening token, its role, then space: read from the text before a user's words.
        var user = new ChatMessage("user", UserProbe);
        if (Render(false, user) is { } alone && alone.IndexOf(UserProbe, StringComparison.Ordinal) is var u and > 0)
        {
            var header = Pieces(alone[..u]).Reverse().SkipWhile(p => p.Trim().Length == 0).ToArray();
            if (header.Length >= 2)
            {
                string open = header[1];
                Put("turn_open", open);
                Put("user", header[0]);
                string? Role(string? text)
                {
                    if (text is null)
                    {
                        return null;
                    }
                    var pieces = Pieces(text);
                    int at = Array.LastIndexOf(pieces, open);
                    return at >= 0 && at + 1 < pieces.Length ? pieces[at + 1] : null;
                }
                if (Render(false, user, new ChatMessage("assistant", SaidProbe)) is { } replied &&
                    replied.IndexOf(UserProbe, StringComparison.Ordinal) is var a and >= 0 && replied.IndexOf(SaidProbe, a, StringComparison.Ordinal) is var s and > 0)
                {
                    Put("assistant", Role(replied[(a + UserProbe.Length)..s]));
                }
                if (Render(false, new ChatMessage("system", SystemProbe), user) is { } told && told.IndexOf(SystemProbe, StringComparison.Ordinal) is var t and > 0)
                {
                    Put("system", Role(told[..t]));
                }
            }
        }
        return marks;
    }
}
