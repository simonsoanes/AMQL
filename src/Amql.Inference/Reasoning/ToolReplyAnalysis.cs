using System.Text.RegularExpressions;

namespace Amql.Inference.Reasoning;

/// <summary>What a reply did with the tools it was offered.</summary>
public enum ToolOutcome
{
    /// <summary>It called the expected tool (first, or after others in a simulated conversation).</summary>
    Called,
    /// <summary>It called tools, but never the expected one.</summary>
    WrongTool,
    /// <summary>A tool was expected and it answered in text instead.</summary>
    Talked,
    /// <summary>No tool was expected and it called none.</summary>
    Quiet,
    /// <summary>No tool was expected and it called one.</summary>
    FalseCall,
}

/// <summary>
/// Reads a reply in Qwen's XML call format (<c>&lt;tool_call&gt;&lt;function=NAME&gt;
/// &lt;parameter=KEY&gt;VALUE&lt;/parameter&gt;…</c>): whether and when it calls, which tool,
/// and the code it produced, whether written through the call's <c>content</c> or shown in a
/// fenced block. Nothing is executed. Placeholders are the marks of an unfinished solution a
/// small model leaves in code: "...", "TODO", "your code here", a bare <c>pass</c>, and the like.
/// </summary>
public sealed record ToolReplyAnalysis(
    ToolOutcome Outcome,
    IReadOnlyList<string> Tools,
    string? FirstTool,
    bool OpensWithCall,
    int CallAt,
    string Code,
    bool CodeInCall,
    IReadOnlyList<string> Placeholders,
    bool CallClosed,
    bool Written)
{
    private static readonly Regex Call = new(@"<tool_call>\s*<function=([\w.\-]+)>", RegexOptions.Compiled);
    private static readonly Regex Content = new(@"<parameter=content>\r?\n?(.*?)(?:\r?\n?</parameter>|$)",
        RegexOptions.Compiled | RegexOptions.Singleline);
    private static readonly Regex Thinking = new(@"<think>.*?</think>", RegexOptions.Compiled | RegexOptions.Singleline);
    private static readonly Regex Fence = new(@"```[\w#+.\-]*\r?\n(.*?)(?:```|$)", RegexOptions.Compiled | RegexOptions.Singleline);

    /// <summary>Marks of code left unfinished. Each is matched per line of code.</summary>
    public static readonly (string Name, Regex Pattern)[] PlaceholderPatterns =
    [
        ("ellipsis", new(@"^\s*(?:#|//|/\*|<!--)?\s*(?:\.\.\.|…)\s*(?:\*/|-->)?\s*$|\(\s*\.\.\.\s*\)|\{\s*\.\.\.\s*\}|\[\s*\.\.\.\s*\]", RegexOptions.Compiled)),
        ("todo", new(@"\b(?:TODO|FIXME|TBD)\b", RegexOptions.Compiled)),
        ("your-code", new(@"(?i)\b(?:your|add|put|insert|implement)\b[^\n]{0,30}\b(?:code|logic|implementation)\b[^\n]{0,12}\bhere\b", RegexOptions.Compiled)),
        ("placeholder", new(@"(?i)\bplaceholder\b", RegexOptions.Compiled)),
        ("rest", new(@"(?i)(?:#|//)\s*(?:(?:\.\.\.|…)\s*)?(?:the\s+)?(?:rest|remaining|more|other)\b[^\n]{0,40}\b(?:here|code|methods?|logic|similar|omitted)", RegexOptions.Compiled)),
        ("not-implemented", new(@"NotImplementedException|raise NotImplementedError|throw new Error\(['""]not implemented", RegexOptions.Compiled)),
        ("bare-pass", new(@"^\s*pass\s*(?:#.*)?$", RegexOptions.Compiled)),
        ("stand-in", new(@"(?i)\byour[_\- ]?(?:name|user(?:name)?|path|folder|file(?:name)?|api[_\- ]?key|token|password|directory)\b|path/to/|<(?:your|insert)[^>]*>", RegexOptions.Compiled)),
    ];

    /// <summary>Reads a reply; <paramref name="expect"/> is the tool to call first, or "none".</summary>
    public static ToolReplyAnalysis Read(string reply, string expect)
    {
        // Reasoning is not the reply: drop every block, and the first one when the prompt
        // opened it and the reply only closes it.
        int close = reply.IndexOf("</think>", StringComparison.Ordinal);
        int open = reply.IndexOf("<think>", StringComparison.Ordinal);
        string body = close >= 0 && (open < 0 || open > close) ? reply[(close + 8)..] : reply;
        body = Thinking.Replace(body, string.Empty);
        var calls = Call.Matches(body);
        var tools = calls.Select(m => m.Groups[1].Value).ToList();
        var call = calls.Count > 0 ? calls[0] : Call.Match(string.Empty);
        string? tool = tools.FirstOrDefault();
        bool opens = call.Success && body[..call.Index].Trim().Length == 0;

        // The code is what the expected tool was given, or else what was shown in fences.
        var target = calls.FirstOrDefault(m => m.Groups[1].Value == expect);
        var content = target is null ? Match.Empty : Content.Match(body, target.Index);
        bool inCall = content.Success;
        string code = inCall
            ? content.Groups[1].Value
            : string.Join("\n", Fence.Matches(body).Select(m => m.Groups[1].Value));
        var marks = new List<string>();
        foreach (string line in code.Split('\n'))
        {
            foreach (var (name, pattern) in PlaceholderPatterns)
            {
                if (pattern.IsMatch(line))
                {
                    marks.Add(name);
                }
            }
        }

        bool none = expect == "none";
        var outcome = tool is null
            ? (none ? ToolOutcome.Quiet : ToolOutcome.Talked)
            : none ? ToolOutcome.FalseCall
            : tools.Contains(expect) ? ToolOutcome.Called : ToolOutcome.WrongTool;
        return new(outcome, tools, tool, opens, call.Success ? call.Index : -1, code, inCall, marks,
            call.Success && body.IndexOf("</tool_call>", call.Index, StringComparison.Ordinal) >= 0,
            target is not null && body.IndexOf("</tool_call>", target.Index, StringComparison.Ordinal) >= 0);
    }

    /// <summary>Non-blank lines of code produced.</summary>
    public int CodeLines => Code.Split('\n').Count(l => l.Trim().Length > 0);
}
