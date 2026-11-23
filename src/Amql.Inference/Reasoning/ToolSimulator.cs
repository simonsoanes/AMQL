using System.Text.RegularExpressions;

namespace Amql.Inference.Reasoning;

/// <summary>
/// Plays the part of a harness's tools so a conversation can go on without running anything.
/// Each call gets a plausible, uneventful result chosen from its tool's name: writes succeed,
/// reads find nothing, listings are empty, commands exit cleanly with no output, and a
/// question to the user is told to go ahead. So a model is never stopped by a tool, and what
/// it does next is its own choice. Nothing is executed, written or fetched.
/// </summary>
public static class ToolSimulator
{
    private static readonly Regex LastCall = new(
        @"<tool_call>\s*<function=([\w.\-]+)>(.*?)(?:</function>|$)", RegexOptions.Compiled | RegexOptions.Singleline);
    private static readonly Regex Parameter = new(
        @"<parameter=([\w\-]+)>\r?\n?(.*?)\r?\n?</parameter>", RegexOptions.Compiled | RegexOptions.Singleline);

    public const string Workspace = @"D:\work";

    /// <summary>The last call in an assistant turn, or null when it makes none.</summary>
    public static (string Tool, IReadOnlyDictionary<string, string> Args)? LastCallIn(string turn)
    {
        var calls = LastCall.Matches(turn);
        if (calls.Count == 0)
        {
            return null;
        }
        var call = calls[^1];
        var args = Parameter.Matches(call.Groups[2].Value)
            .GroupBy(m => m.Groups[1].Value).ToDictionary(g => g.Key, g => g.Last().Groups[2].Value);
        return (call.Groups[1].Value, args);
    }

    /// <summary>The simulated result of a call.</summary>
    public static string Respond(string tool, IReadOnlyDictionary<string, string> args)
    {
        string name = tool.ToLowerInvariant();
        string path = args.GetValueOrDefault("path") ?? args.GetValueOrDefault("file") ?? "";
        if (name.Contains("current") || name.Contains("cwd") || name.Contains("pwd"))
        {
            return $"{{\"success\":true,\"path\":\"{Escape(Workspace)}\",\"permitted_directories\":[\"{Escape(Workspace)}\"]}}";
        }
        if (name.Contains("create") || name.Contains("write") || name.Contains("edit") || name.Contains("save"))
        {
            int lines = (args.GetValueOrDefault("content") ?? args.GetValueOrDefault("new_text") ?? "").Split('\n').Length;
            return name.Contains("director") || name.Contains("folder") || name.Contains("mkdir")
                ? $"{{\"success\":true,\"message\":\"Directory created: {Escape(path)}\"}}"
                : $"{{\"success\":true,\"path\":\"{Escape(path)}\",\"lines\":{lines},\"message\":\"Wrote {Escape(path)} ({lines} lines)\"}}";
        }
        if (name.Contains("read") || name.Contains("open") || name.Contains("cat"))
        {
            return $"{{\"success\":false,\"error\":\"File not found: {Escape(path)}\"}}";
        }
        if (name.Contains("list") || name.Contains("glob") || name.Contains("search") || name.Contains("find"))
        {
            return "{\"success\":true,\"files\":[],\"count\":0,\"listing\":\"(empty)\"}";
        }
        if (name.Contains("run") || name.Contains("terminal") || name.Contains("shell") || name.Contains("exec") || name.Contains("command"))
        {
            return "{\"success\":true,\"exitCode\":0,\"output\":\"\"}";
        }
        if (name.Contains("ask") || name.Contains("question") || name.Contains("user"))
        {
            return "{\"success\":true,\"answer\":\"Please just go ahead with your best judgement.\"}";
        }
        if (name.Contains("web") || name.Contains("fetch") || name.Contains("url") || name.Contains("http"))
        {
            return "{\"success\":true,\"results\":[]}";
        }
        return "{\"success\":true}";
    }

    /// <summary>The text a ChatML template puts between an assistant turn that ended in a call
    /// and the next assistant turn: the tool's result as a user turn, as Qwen's template renders
    /// it, then the generation prompt.</summary>
    public static string ResultTurn(string result, bool think) =>
        $"<|im_end|>\n<|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n<|im_start|>assistant\n" +
        (think ? "<think>\n" : "<think>\n\n</think>\n\n");

    private static string Escape(string s) => s.Replace("\\", "\\\\").Replace("\"", "\\\"");
}
