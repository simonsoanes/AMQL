using System.Text.Json;

namespace Amql.Cli;

/// <summary>One call an assistant turn made: the function's name and its
/// arguments as a JSON object, in the order the caller gave them.</summary>
public sealed record ChatToolCall(string Name, JsonElement Arguments);

/// <summary>One chat turn, already flattened to text. An assistant turn may
/// carry the calls it made and its reasoning; a <c>tool</c> turn carries a
/// call's result as its content.</summary>
public sealed record ChatMessage(string Role, string Content)
{
    public IReadOnlyList<ChatToolCall>? ToolCalls { get; init; }

    /// <summary>The assistant's reasoning, when the client sends it apart
    /// from the content (<c>reasoning_content</c>).</summary>
    public string? Reasoning { get; init; }
}

/// <summary>
/// Minimal chat-template renderer for the <c>generate --chat</c> path and the
/// server. Full Jinja2 is out of scope; this recognises the two most common
/// families — ChatML (Granite, Qwen, MiniCPM) and Llama 3 — plus a
/// generic fallback that applies token substitution on the raw template
/// for less common formats.
/// </summary>
public static class ChatTemplate
{
    /// <summary>
    /// Renders a whole conversation with the generation prompt appended — what
    /// HF's <c>apply_chat_template(messages, tools=tools, add_generation_prompt=True)</c>
    /// produces for the template families recognised here:
    /// <list type="bullet">
    /// <item>Qwen3-style ChatML (the template tracks <c>last_query_index</c>):
    /// contents trimmed; <c>&lt;think&gt;</c> blocks stripped from assistant
    /// turns before the last user query and re-rendered after it; tools in the
    /// system turn, calls as <c>&lt;tool_call&gt;</c> XML, results as
    /// <c>&lt;tool_response&gt;</c> in a user turn; the prompt ends with an
    /// empty think block unless thinking is enabled.</item>
    /// <item>Plain ChatML (<c>&lt;|im_start|&gt;</c>).</item>
    /// <item>Llama 3 (<c>&lt;|start_header_id|&gt;</c>).</item>
    /// </list>
    /// Any other template is refused rather than approximated: a mis-rendered
    /// prompt silently degrades every answer. Tools, calls and tool turns are
    /// rendered for the Qwen3 family only, and refused for the others.
    /// </summary>
    public static string ApplyMessages(string template, IReadOnlyList<ChatMessage> messages, bool enableThinking = false,
        IReadOnlyList<JsonElement>? tools = null)
    {
        if (messages.Count == 0)
        {
            throw new ChatTemplateException("no messages provided");
        }
        bool qwen3 = template.Contains("<|im_start|>") && template.Contains("last_query_index");
        bool rendersTools = RendersTools(template);
        foreach (var m in messages)
        {
            if (m.Role is not ("system" or "user" or "assistant" or "tool"))
            {
                throw new ChatTemplateException($"message role '{m.Role}' is not supported (system, user, assistant, tool)");
            }
            if (!rendersTools && (m.Role == "tool" || m.ToolCalls is { Count: > 0 }))
            {
                throw new ChatTemplateException("this model's chat template does not render tool calls");
            }
        }
        if (!rendersTools && tools is { Count: > 0 })
        {
            throw new ChatTemplateException("this model's chat template does not render tools");
        }
        for (int i = 1; i < messages.Count; i++)
        {
            if (messages[i].Role == "system")
            {
                throw new ChatTemplateException("a system message must be the first message");
            }
        }

        if (template.Contains("<|im_start|>"))
        {
            return qwen3
                ? RenderQwen3(messages, enableThinking, tools)
                : RenderChatMl(template, messages);
        }
        if (template.Contains("<|start_header_id|>"))
        {
            string bos = ExtractSpecial(template, "bos_token", "<|begin_of_text|>") ?? "<|begin_of_text|>";
            string eot = ExtractSpecial(template, "eos_token", "<|eot_id|>") ?? "<|eot_id|>";
            var sb = new System.Text.StringBuilder(bos);
            foreach (var m in messages)
            {
                sb.Append("<|start_header_id|>").Append(m.Role).Append("<|end_header_id|>\n\n")
                  .Append(m.Content.Trim()).Append(eot);
            }
            return sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n").ToString();
        }
        throw new ChatTemplateException(
            "the container's chat template is not a family this build renders (Qwen3 ChatML, ChatML, Llama 3)");
    }

    /// <summary>True for a template this renderer can give tools: the Qwen3.5
    /// family, which writes calls as <c>&lt;function=…&gt;</c> XML.</summary>
    public static bool RendersTools(string template) =>
        template.Contains("<|im_start|>") && template.Contains("last_query_index") && template.Contains("<function=");

    private static string RenderChatMl(string template, IReadOnlyList<ChatMessage> messages)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var m in messages)
        {
            sb.Append("<|im_start|>").Append(m.Role).Append('\n').Append(m.Content).Append("<|im_end|>\n");
        }
        sb.Append("<|im_start|>assistant\n");
        if (template.Contains(" thinking\n"))
        {
            sb.Append(" thinking\n");
        }
        return sb.ToString();
    }

    private const string ToolsPreamble = "# Tools\n\nYou have access to the following functions:\n\n<tools>";

    private const string ToolsInstructions =
        "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n" +
        "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n" +
        "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" +
        "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
        "- Required parameters MUST be specified\n" +
        "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
        "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
        "</IMPORTANT>";

    /// <summary>The Qwen3 / Qwen3.5 template, statement for statement (vision
    /// content excluded — the server refuses it upstream).</summary>
    private static string RenderQwen3(IReadOnlyList<ChatMessage> messages, bool enableThinking, IReadOnlyList<JsonElement>? tools)
    {
        static bool IsResponse(string c) =>
            c.StartsWith("<tool_response>", StringComparison.Ordinal) && c.EndsWith("</tool_response>", StringComparison.Ordinal);

        int lastQuery = -1;
        for (int i = messages.Count - 1; i >= 0; i--)
        {
            if (messages[i].Role == "user" && !IsResponse(PyFormat.Strip(messages[i].Content)))
            {
                lastQuery = i;
                break;
            }
        }
        if (lastQuery < 0)
        {
            throw new ChatTemplateException("no user query found in messages");
        }

        var sb = new System.Text.StringBuilder();
        if (tools is { Count: > 0 })
        {
            sb.Append("<|im_start|>system\n").Append(ToolsPreamble);
            foreach (var tool in tools)
            {
                sb.Append('\n').Append(PyFormat.JsonDumps(tool, sortKeys: false, ensureAscii: false));
            }
            sb.Append("\n</tools>").Append(ToolsInstructions);
            if (messages[0].Role == "system" && PyFormat.Strip(messages[0].Content) is { Length: > 0 } system)
            {
                sb.Append("\n\n").Append(system);
            }
            sb.Append("<|im_end|>\n");
        }
        else if (messages[0].Role == "system")
        {
            sb.Append("<|im_start|>system\n").Append(PyFormat.Strip(messages[0].Content)).Append("<|im_end|>\n");
        }

        for (int i = 0; i < messages.Count; i++)
        {
            var m = messages[i];
            string content = PyFormat.Strip(m.Content);
            switch (m.Role)
            {
                case "user":
                    sb.Append("<|im_start|>user\n").Append(content).Append("<|im_end|>\n");
                    break;
                case "assistant":
                {
                    string reasoning = string.Empty;
                    if (m.Reasoning is { } given)
                    {
                        reasoning = given;
                    }
                    else if (content.Contains("</think>", StringComparison.Ordinal))
                    {
                        string before = content[..content.IndexOf("</think>", StringComparison.Ordinal)].TrimEnd('\n');
                        int open = before.LastIndexOf("<think>", StringComparison.Ordinal);
                        reasoning = (open >= 0 ? before[(open + "<think>".Length)..] : before).TrimStart('\n');
                        content = content[(content.LastIndexOf("</think>", StringComparison.Ordinal) + "</think>".Length)..].TrimStart('\n');
                    }
                    reasoning = PyFormat.Strip(reasoning);
                    sb.Append("<|im_start|>assistant\n");
                    if (i > lastQuery)
                    {
                        sb.Append("<think>\n").Append(reasoning).Append("\n</think>\n\n");
                    }
                    sb.Append(content);
                    if (m.ToolCalls is { Count: > 0 } calls)
                    {
                        for (int c = 0; c < calls.Count; c++)
                        {
                            sb.Append(c > 0 ? "\n<tool_call>\n<function="
                                    : PyFormat.Strip(content).Length > 0 ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function=")
                              .Append(calls[c].Name).Append(">\n");
                            if (calls[c].Arguments.ValueKind == JsonValueKind.Object)
                            {
                                foreach (var (name, value) in PyFormat.LastWins(calls[c].Arguments))
                                {
                                    // Objects and lists as JSON; everything else as Python's str().
                                    sb.Append("<parameter=").Append(name).Append(">\n")
                                      .Append(value.ValueKind is JsonValueKind.Object or JsonValueKind.Array
                                          ? PyFormat.JsonDumps(value, sortKeys: false, ensureAscii: false)
                                          : PyFormat.Str(value))
                                      .Append("\n</parameter>\n");
                                }
                            }
                            sb.Append("</function>\n</tool_call>");
                        }
                    }
                    sb.Append("<|im_end|>\n");
                    break;
                }
                case "tool":
                    if (i > 0 && messages[i - 1].Role != "tool")
                    {
                        sb.Append("<|im_start|>user");
                    }
                    sb.Append("\n<tool_response>\n").Append(content).Append("\n</tool_response>");
                    if (i == messages.Count - 1 || messages[i + 1].Role != "tool")
                    {
                        sb.Append("<|im_end|>\n");
                    }
                    break;
            }
        }
        sb.Append("<|im_start|>assistant\n");
        sb.Append(enableThinking ? "<think>\n" : "<think>\n\n</think>\n\n");
        return sb.ToString();
    }

    /// <summary>Loads the chat template from a container or checkpoint
    /// directory: first <c>chat_template.jinja</c>, then the
    /// <c>chat_template</c> key inside <c>tokenizer_config.json</c>.</summary>
    public static string? Load(string dir)
    {
        string ctPath = Path.Combine(dir, "chat_template.jinja");
        if (File.Exists(ctPath))
            return File.ReadAllText(ctPath);

        string tcPath = Path.Combine(dir, "tokenizer_config.json");
        if (!File.Exists(tcPath))
            return null;

        using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(tcPath));
        if (doc.RootElement.TryGetProperty("chat_template", out var ct) && ct.ValueKind == System.Text.Json.JsonValueKind.String)
            return ct.GetString();

        return null;
    }

    /// <summary>Renders the chat template for a single-turn user message.
    /// <paramref name="prompt"/> becomes the user content; an optional
    /// <paramref name="systemPrompt"/> prepends a system message when the
    /// template supports it.</summary>
    public static string Apply(string template, string prompt, string? systemPrompt)
    {
        // ── ChatML family ──────────────────────────────────────────
        if (template.Contains("<|im_start|>") || template.Contains("<|im_end|>"))
        {
            var sb = new System.Text.StringBuilder();
            if (systemPrompt is { } sys)
            {
                sb.Append("<|im_start|>system\n");
                sb.Append(sys);
                sb.Append("<|im_end|>\n");
            }
            sb.Append("<|im_start|>user\n");
            sb.Append(prompt);
            sb.Append("<|im_end|>\n");
            sb.Append("<|im_start|>assistant\n");
            // Granite's thinking tag
            if (template.Contains(" thinking\n"))
                sb.Append(" thinking\n");
            return sb.ToString();
        }

        // ── Llama 3 family ─────────────────────────────────────────
        if (template.Contains("<|start_header_id|>"))
        {
            string bos = ExtractSpecial(template, "bos_token", "<|begin_of_text|>") ?? "<|begin_of_text|>";
            string eot = ExtractSpecial(template, "eos_token", "<|eot_id|>") ?? "<|eot_id|>";
            var sb = new System.Text.StringBuilder();
            sb.Append(bos);
            if (systemPrompt is { } sys)
            {
                sb.Append("<|start_header_id|>system<|end_header_id|>\n\n");
                sb.Append(sys);
                sb.Append(eot);
            }
            sb.Append("<|start_header_id|>user<|end_header_id|>\n\n");
            sb.Append(prompt);
            sb.Append(eot);
            sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
            return sb.ToString();
        }

        // ── Generic Jinja2-lite ────────────────────────────────────
        // For templates we don't explicitly recognise, do a simple
        // token-substitution pass: replace {{ bos_token }} etc.
        string rendered = template;
        rendered = rendered.Replace("{{ bos_token }}", "<s>");

        // Try a simple for-messages loop
        if (template.Contains("{% for message in messages %}"))
        {
            // Extract the indentation and apply per-message rules
            // This is approximate but works for most templates
            var sb = new System.Text.StringBuilder();
            int loopStart = template.IndexOf("{% for message in messages %}");
            int loopEnd = template.IndexOf("{% endfor %}", loopStart);
            if (loopStart >= 0 && loopEnd > loopStart)
            {
                string before = template[..loopStart];
                string loopBody = template[(loopStart + "{% for message in messages %}".Length)..loopEnd];
                string after = template[(loopEnd + "{% endfor %}".Length)..];

                sb.Append(RenderText(before));

                if (systemPrompt is { } sys)
                    sb.Append(RenderMessage(loopBody, "system", sys));

                sb.Append(RenderMessage(loopBody, "user", prompt));

                if (after.Contains("add_generation_prompt") && after.Contains("{% if add_generation_prompt %}"))
                {
                    int agpStart = after.IndexOf("{% if add_generation_prompt %}");
                    int agpEnd = after.IndexOf("{% endif %}", agpStart);
                    if (agpStart >= 0 && agpEnd > agpStart)
                    {
                        sb.Append(RenderText(after[..agpStart]));
                        string agpBody = after[(agpStart + "{% if add_generation_prompt %}".Length)..agpEnd];
                        sb.Append(RenderText(agpBody));
                        sb.Append(RenderText(after[(agpEnd + "{% endif %}".Length)..]));
                    }
                    else
                    {
                        sb.Append(RenderText(after));
                    }
                }
                else
                {
                    sb.Append(RenderText(after));
                }
                return sb.ToString();
            }
        }

        // Last resort: just append the prompt to the raw template
        return rendered + "\n" + prompt;
    }

    private static string RenderMessage(string body, string role, string content)
    {
        string result = body;
        result = result.Replace("{{ message['role'] }}", role);
        result = result.Replace("{{ message[\"role\"] }}", role);
        result = result.Replace("{{ message['content'] }}", content);
        result = result.Replace("{{ message[\"content\"] }}", content);
        result = StripConditional(result, role);
        return result;
    }

    private static string StripConditional(string text, string role)
    {
        // Remove {% if message['role'] == 'other_role' %} ... {% endif %}
        // leaving only the branch matching our current role.
        var sb = new System.Text.StringBuilder();
        int pos = 0;
        while (pos < text.Length)
        {
            int ifStart = text.IndexOf("{% if", pos);
            if (ifStart < 0) { sb.Append(text[pos..]); break; }

            sb.Append(text[pos..ifStart]);
            int ifEnd = text.IndexOf("%}", ifStart);
            if (ifEnd < 0) { sb.Append(text[ifStart..]); break; }
            string condition = text[(ifStart + "{% if".Length)..ifEnd].Trim();
            int endif = text.IndexOf("{% endif %}", ifEnd);
            if (endif < 0) { sb.Append(text[ifStart..]); break; }

            // Check for elif/else branches
            int nextBranch = FindNextBranch(text, ifEnd + 2);
            bool ourBranch = ConditionMatches(condition, role);

            if (ourBranch)
            {
                int bodyEnd = nextBranch >= 0 ? nextBranch : endif;
                sb.Append(text[(ifEnd + 2)..bodyEnd]);
            }
            else if (nextBranch >= 0)
            {
                // Try elif/else branches
                while (nextBranch >= 0 && nextBranch < endif)
                {
                    int branchTagEnd = text.IndexOf("%}", nextBranch);
                    if (branchTagEnd < 0) break;
                    string branchCond = text[(nextBranch + 2)..branchTagEnd].Trim();
                    int nextNext = FindNextBranch(text, branchTagEnd + 2);
                    int branchBodyEnd = nextNext >= 0 ? nextNext : endif;

                    if (branchCond.StartsWith("else") && !branchCond.StartsWith("elif"))
                    {
                        // {% else %} — our fallback
                        sb.Append(text[(branchTagEnd + 2)..branchBodyEnd]);
                        break;
                    }
                    if (branchCond.StartsWith("elif") && ConditionMatches(branchCond[4..].Trim(), role))
                    {
                        sb.Append(text[(branchTagEnd + 2)..branchBodyEnd]);
                        break;
                    }
                    nextBranch = nextNext;
                }
            }
            pos = endif + "{% endif %}".Length;
        }
        return sb.ToString();
    }

    private static int FindNextBranch(string text, int start)
    {
        int elif = text.IndexOf("{% elif", start);
        int els = text.IndexOf("{% else %}", start);
        int result = -1;
        if (elif >= 0 && (result < 0 || elif < result)) result = elif;
        if (els >= 0 && (result < 0 || els < result)) result = els;
        return result;
    }

    private static bool ConditionMatches(string condition, string role)
    {
        // Strip leading "not " — we don't handle negation yet
        bool negated = condition.StartsWith("not ");
        if (negated) condition = condition[4..];

        // message['role'] == 'system', etc.
        bool matches =
            condition.Contains($"'role'] == '{role}'") ||
            condition.Contains($"\"role\"] == \"{role}\"");
        return negated ? !matches : matches;
    }

    private static string RenderText(string text)
    {
        // Strip remaining Jinja2 markers that don't apply
        // (e.g., {% endif %} remnants, unsupported tags)
        var sb = new System.Text.StringBuilder();
        int pos = 0;
        while (pos < text.Length)
        {
            int tagStart = text.IndexOf("{%", pos);
            if (tagStart < 0) { sb.Append(text[pos..]); break; }
            sb.Append(text[pos..tagStart]);
            int tagEnd = text.IndexOf("%}", tagStart);
            if (tagEnd < 0) { sb.Append(text[tagStart..]); break; }
            pos = tagEnd + 2;
        }
        return sb.ToString();
    }

    private static string? ExtractSpecial(string template, string varName, string defaultValue)
    {
        // Look for {% set varName = "value" %} or just use default
        return defaultValue;
    }
}

/// <summary>A conversation the container's chat template cannot render.</summary>
public sealed class ChatTemplateException : Exception
{
    public ChatTemplateException(string message) : base(message) { }
}
