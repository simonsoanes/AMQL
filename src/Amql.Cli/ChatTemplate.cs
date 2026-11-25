using System.Text.Json;
using Amql.Cli.Jinja;

namespace Amql.Cli;

/// <summary>One call an assistant turn made: the function's name and its
/// arguments as a JSON object, in the order the caller gave them.</summary>
public sealed record ChatToolCall(string Name, JsonElement Arguments)
{
    /// <summary>The id a result names the call by, when the client gave one.</summary>
    public string? Id { get; init; }
}

/// <summary>A piece of a turn's content: text, or (without any) an image.</summary>
public sealed record ChatPart(string? Text)
{
    /// <summary>The image's file, and the most tokens it may take (null: the model's own).</summary>
    public byte[]? Image { get; init; }

    public int? ImageTokens { get; init; }
}

/// <summary>One chat turn, already flattened to text. An assistant turn may
/// carry the calls it made and its reasoning; a <c>tool</c> turn carries a
/// call's result as its content.</summary>
public sealed record ChatMessage(string Role, string Content)
{
    public IReadOnlyList<ChatToolCall>? ToolCalls { get; init; }

    /// <summary>The assistant's reasoning, when the client sends it apart
    /// from the content (<c>reasoning_content</c>).</summary>
    public string? Reasoning { get; init; }

    /// <summary>A tool turn's call: the id it answers, or the function's name.</summary>
    public string? CallId { get; init; }

    public string? Name { get; init; }

    /// <summary>The content in its pieces, when some of it is images
    /// (<see cref="Content"/> is then its text alone).</summary>
    public IReadOnlyList<ChatPart>? Parts { get; init; }
}

/// <summary>A conversation the model's chat template does not render.</summary>
public sealed class ChatTemplateException : Exception
{
    public ChatTemplateException(string message) : base(message) { }
}

/// <summary>
/// A model's own chat template, rendered as transformers' <c>apply_chat_template</c> renders it
/// (<see cref="JinjaTemplate"/>). Nothing here knows a family: what a template does with tools,
/// reasoning and images is learned from what it makes of probes (<see cref="Dialect"/>), and what it
/// would drop in silence is refused.
/// </summary>
public sealed class ChatTemplate
{
    private readonly JinjaTemplate _template;
    private readonly Dictionary<string, string> _tokens;
    private readonly bool _shortIds;
    private ChatDialect? _dialect;

    private ChatTemplate(string source, Dictionary<string, string> tokens, JsonElement? response)
    {
        try
        {
            _template = JinjaTemplate.Parse(source);
        }
        catch (JinjaException e)
        {
            throw new ChatTemplateException($"the chat template cannot be read: {e.Message}");
        }
        _tokens = tokens;
        ResponseTemplate = response;
        // Mistral's templates take ids of nine characters and no others.
        _shortIds = System.Text.RegularExpressions.Regex.IsMatch(source, @"(length|len)\W+9\b");
    }

    public string Source => _template.Source;

    /// <summary>The tokenizer's own description of a reply, when it ships one (<c>response_template</c>).</summary>
    internal JsonElement? ResponseTemplate { get; }

    /// <summary>What the template does, learned by probing it.</summary>
    public ChatDialect Dialect => _dialect ??= ChatDialect.Learn(this);

    /// <summary>The names the template can be given.</summary>
    internal IReadOnlySet<string> Reads => _template.Reads;

    /// <summary>A template with the tokens it names (<c>bos_token</c>, <c>eos_token</c>, …).</summary>
    public static ChatTemplate Parse(string source, string bos = "", string eos = "") =>
        new(source, new Dictionary<string, string> { ["bos_token"] = bos, ["eos_token"] = eos }, null);

    /// <summary>The chat template of a container or checkpoint directory (<c>chat_template.jinja</c>, or
    /// <c>chat_template</c> in <c>tokenizer_config.json</c>), or null when it has none.</summary>
    public static ChatTemplate? Load(string dir)
    {
        string path = Path.Combine(dir, "chat_template.jinja"), config = Path.Combine(dir, "tokenizer_config.json");
        string? source = File.Exists(path) ? File.ReadAllText(path) : null;
        var tokens = new Dictionary<string, string>(StringComparer.Ordinal);
        JsonElement? response = null;
        if (File.Exists(config))
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(config));
            foreach (var p in doc.RootElement.EnumerateObject())
            {
                if (p.Name.EndsWith("_token", StringComparison.Ordinal) && Content(p.Value) is { } token)
                {
                    tokens[p.Name] = token;
                }
            }
            if (source is null && doc.RootElement.TryGetProperty("chat_template", out var t))
            {
                // A string, or several by name.
                source = t.ValueKind == JsonValueKind.String ? t.GetString()
                    : t.ValueKind == JsonValueKind.Array ? t.EnumerateArray().FirstOrDefault(x => x.GetProperty("name").GetString() == "default") is { ValueKind: JsonValueKind.Object } d
                        ? d.GetProperty("template").GetString() : null
                    : null;
            }
            if (doc.RootElement.TryGetProperty("response_template", out var r) && r.ValueKind == JsonValueKind.Object)
            {
                response = r.Clone();
            }
        }
        return source is null ? null : new ChatTemplate(source, tokens, response);

        static string? Content(JsonElement e) => e.ValueKind == JsonValueKind.String ? e.GetString()
            : e.ValueKind == JsonValueKind.Object && e.TryGetProperty("content", out var c) ? c.GetString() : null;
    }

    /// <summary>
    /// A conversation with the generation prompt appended — what HF's
    /// <c>apply_chat_template(messages, tools=tools, add_generation_prompt=True)</c> produces, with three
    /// differences. A result that names no call answers the calls in order. Where a template writes what an
    /// assistant said along with its calls after the calls' results, so that the reply would start outside
    /// any turn, it is written as a turn before them. And a reply that goes on from an image, where the
    /// template opens no turn for it, starts with the empty thinking its other replies start with.
    /// </summary>
    /// <param name="variables">More for the template (<c>chat_template_kwargs</c>).</param>
    public string Apply(IReadOnlyList<ChatMessage> messages, bool enableThinking = false, IReadOnlyList<JsonElement>? tools = null,
        IReadOnlyDictionary<string, JsonElement>? variables = null)
    {
        if (messages.Count == 0)
        {
            throw new ChatTemplateException("no messages provided");
        }
        var dialect = Dialect;
        for (int i = 0; i < messages.Count; i++)
        {
            var m = messages[i];
            if (m.Role is not ("system" or "user" or "assistant" or "tool"))
            {
                throw new ChatTemplateException($"message role '{m.Role}' is not supported (system, user, assistant, tool)");
            }
            if (i > 0 && m.Role == "system")
            {
                throw new ChatTemplateException("a system message must be the first message");
            }
            if (m.ToolCalls is { Count: > 0 } && !dialect.WritesCalls)
            {
                throw new ChatTemplateException("this model's chat template does not render tool calls");
            }
            if (m.Role == "tool" && !dialect.WritesResults)
            {
                throw new ChatTemplateException("this model's chat template does not render tool results");
            }
            if (m.Parts?.Any(p => p.Text is null) == true)
            {
                if (dialect.Image is null)
                {
                    throw new ChatTemplateException("this model's chat template does not render images");
                }
                if (m.Role == "system")
                {
                    throw new ChatTemplateException("a system message holds no images");
                }
            }
        }
        if (tools is { Count: > 0 } && !dialect.WritesTools)
        {
            throw new ChatTemplateException("this model's chat template does not render tools");
        }
        if (dialect.SaysAfterResults)
        {
            messages = messages.SelectMany(m => m is { Role: "assistant", ToolCalls.Count: > 0 } && m.Content.Trim().Length > 0
                ? new[] { m with { ToolCalls = null }, m with { Content = "", Parts = null, Reasoning = null } } : new[] { m }).ToList();
        }
        string text = Render(messages, enableThinking, tools, true, variables);
        return messages[^1].Role == "tool" && messages[^1].Parts?.Any(p => p.Text is null) == true && !enableThinking
               && dialect.EmptyThinking.Length > 0 && text.TrimEnd().EndsWith(dialect.Image!.Trim(), StringComparison.Ordinal)
            ? text + dialect.EmptyThinking : text;
    }

    /// <summary>One user message, with the generation prompt.</summary>
    public string Apply(string prompt, string? system = null, bool enableThinking = false) =>
        Apply(system is null ? [new ChatMessage("user", prompt)] : [new ChatMessage("system", system), new ChatMessage("user", prompt)], enableThinking);

    /// <summary>The template's text for these messages as they are.</summary>
    public string Render(IReadOnlyList<ChatMessage> messages, bool thinking, IReadOnlyList<JsonElement>? tools, bool generation,
        IReadOnlyDictionary<string, JsonElement>? variables = null)
    {
        var values = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var (name, token) in _tokens)
        {
            values[name] = token;
        }
        values["messages"] = Messages(messages);
        values["tools"] = tools is { Count: > 0 } ? new JList(tools.Select(Py.FromJson).ToList()) : null;
        values["documents"] = null;
        values["add_generation_prompt"] = generation;
        values["enable_thinking"] = thinking;
        if (Reads.Contains("thinking"))
        {
            values["thinking"] = thinking;
        }
        foreach (var (name, value) in variables ?? new Dictionary<string, JsonElement>())
        {
            if (name is not ("messages" or "tools" or "add_generation_prompt" or "bos_token" or "eos_token"))
            {
                values[name] = Py.FromJson(value);
            }
        }
        try
        {
            return _template.Render(values);
        }
        catch (JinjaException e)
        {
            throw new ChatTemplateException($"the chat template does not render this: {e.Message}");
        }
    }

    private JList Messages(IReadOnlyList<ChatMessage> messages)
    {
        var list = new List<object?>();
        var waiting = new List<(string Id, string Name)>();   // calls of the last turn, as they are answered
        int made = 0;
        string Id(string? given) => given is null ? $"call{made++:D5}"
            : _shortIds && !(given.Length == 9 && given.All(char.IsAsciiLetterOrDigit))
                ? Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(given)))[..9] : given;
        foreach (var m in messages)
        {
            var d = new JDict();
            d.Set("role", m.Role);
            d.Set("content", m.Parts?.Any(p => p.Text is null) == true
                ? new JList(m.Parts.Select(p =>
                {
                    var part = new JDict();
                    part.Set("type", p.Text is null ? "image" : "text");
                    if (p.Text is not null)
                    {
                        part.Set("text", p.Text);
                    }
                    return (object?)part;
                }).ToList())
                : m.Content);
            if (m.Reasoning is { Length: > 0 } reasoning)
            {
                foreach (string name in new[] { "reasoning_content", "reasoning", "thinking" })
                {
                    if (name == "reasoning_content" || Source.Contains($".{name}") || Source.Contains($"'{name}'") || Source.Contains($"\"{name}\""))
                    {
                        d.Set(name, reasoning);
                    }
                }
            }
            if (m.ToolCalls is { Count: > 0 } calls)
            {
                waiting.Clear();
                d.Set("tool_calls", new JList(calls.Select(c =>
                {
                    string id = Id(c.Id);
                    waiting.Add((id, c.Name));
                    var function = new JDict();
                    function.Set("name", c.Name);
                    function.Set("arguments", c.Arguments.ValueKind == JsonValueKind.Object ? Py.FromJson(c.Arguments) : new JDict());
                    var call = new JDict();
                    call.Set("id", id);
                    call.Set("type", "function");
                    call.Set("function", function);
                    return (object?)call;
                }).ToList()));
            }
            if (m.Role == "tool")
            {
                int at = m.CallId is null ? (waiting.Count > 0 ? 0 : -1) : waiting.FindIndex(w => w.Id == Id(m.CallId));
                string? id = m.CallId is not null ? Id(m.CallId) : at >= 0 ? waiting[at].Id : null;
                string? name = m.Name ?? (at >= 0 ? waiting[at].Name : null);
                if (id is not null)
                {
                    d.Set("tool_call_id", id);
                }
                if (name is not null)
                {
                    d.Set("name", name);
                }
                if (at >= 0)
                {
                    waiting.RemoveAt(at);
                }
            }
            list.Add(d);
        }
        return new JList(list);
    }
}
