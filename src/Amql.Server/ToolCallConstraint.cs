using System.Text.Json;

namespace Amql.Server;

/// <summary>What constrained decoding asks of a grammar: whether a piece of text may come
/// next, taking it, and whether the reply may end where it stands.</summary>
public interface ITokenConstraint
{
    bool Accepts(string text);

    void Advance(string text);

    bool CanEnd { get; }
}

/// <summary>The tools a request offers, compiled for <see cref="ToolCallConstraint"/>: each
/// tool's parameters, which are required, and the schema of those that are not plain text.</summary>
public sealed class ToolGrammar
{
    /// <param name="Choices">For a string enum: the values, written bare as the template writes strings.</param>
    public sealed record Parameter(string Name, bool Required, JsonSchemaNode? Json, IReadOnlyList<string>? Choices = null);

    public sealed record Tool(string Name, IReadOnlyList<Parameter> Parameters);

    public IReadOnlyList<Tool> Tools { get; }

    private ToolGrammar(IReadOnlyList<Tool> tools) => Tools = tools;

    /// <summary>Compiles chat-form tools (<c>{"type":"function","function":{name, parameters}}</c>).
    /// A parameter typed string (or not typed) is free text; any other is held to its schema
    /// as JSON. A schema this build cannot compile leaves that parameter free text.</summary>
    public static ToolGrammar Compile(IReadOnlyList<JsonElement> tools)
    {
        var list = new List<Tool>();
        foreach (var t in tools)
        {
            var f = t.TryGetProperty("function", out var fn) ? fn : t;
            if (!f.TryGetProperty("name", out var n) || n.GetString() is not { Length: > 0 } name)
            {
                continue;
            }
            var parameters = new List<Parameter>();
            if (f.TryGetProperty("parameters", out var ps) && ps.ValueKind == JsonValueKind.Object &&
                ps.TryGetProperty("properties", out var props) && props.ValueKind == JsonValueKind.Object)
            {
                var required = ps.TryGetProperty("required", out var req) && req.ValueKind == JsonValueKind.Array
                    ? req.EnumerateArray().Select(x => x.GetString()).ToHashSet()
                    : new HashSet<string?>();
                foreach (var p in props.EnumerateObject())
                {
                    JsonSchemaNode? json = null;
                    List<string>? choices = null;
                    try
                    {
                        var node = JsonSchemaNode.Compile(p.Value);
                        if (node.Enum is { } values && values.All(e => e.StartsWith('"')))
                        {
                            // A string enum: one of its values, bare.
                            choices = values.Select(e => JsonSerializer.Deserialize<string>(e)!).ToList();
                        }
                        else if (!node.Str)
                        {
                            json = node;   // not text: held to its schema as JSON
                        }
                    }
                    catch (ArgumentException)
                    {
                    }
                    parameters.Add(new Parameter(p.Name, required.Contains(p.Name), json, choices));
                }
            }
            list.Add(new Tool(name, parameters));
        }
        return new ToolGrammar(list);
    }
}

/// <summary>
/// A tool call as Qwen3.5's template writes it, held to the request's tools:
/// <c>&lt;function=NAME&gt;</c> names a tool that exists, each
/// <c>&lt;parameter=K&gt;</c> one of its parameters not given yet, a parameter whose
/// schema is not text is valid JSON for that schema, and <c>&lt;/function&gt;</c> comes only
/// once every required parameter is there. Starts after <c>&lt;tool_call&gt;</c>; done at
/// <c>&lt;/tool_call&gt;</c>. A small model's calls are then always ones the harness can run:
/// the right tool, the right parameter names, arrays and objects that parse.
/// </summary>
public sealed class ToolCallConstraint : ITokenConstraint
{
    private const string FunctionOpen = "<function=", ParameterOpen = "<parameter=", ParameterClose = "</parameter>",
        FunctionClose = "</function>", CallClose = "</tool_call>";

    private const int MaxWhitespace = 8;

    private State _state;

    public ToolCallConstraint(ToolGrammar grammar) => _state = new State(grammar, Phase.BeforeFunction, "", null, null, null, 0, 0);

    /// <summary>The call has closed (<c>&lt;/tool_call&gt;</c>).</summary>
    public bool Done => _state.Phase == Phase.Done;

    /// <summary>
    /// What is left to write to open <paramref name="name"/> as the next parameter, from here:
    /// null when that cannot come next (it is not a parameter of this tool, it was given
    /// already, or the call is not between parameters).
    /// </summary>
    public string? PathTo(string name)
    {
        var s = _state;
        if (s.Tool is null || !s.Tool.Parameters.Any(p => p.Name == name) || s.Given!.Contains(name))
        {
            return null;
        }
        return s.Phase switch
        {
            Phase.BetweenParameters => "<parameter=" + name + ">",
            Phase.AfterLessThan => "parameter=" + name + ">",
            Phase.Literal when s.Next == Phase.ParameterName => s.Pending + name + ">",
            Phase.ParameterName when name.StartsWith(s.Typed, StringComparison.Ordinal) => name[s.Typed.Length..] + ">",
            _ => null,
        };
    }

    /// <summary>
    /// What is left to write to name <paramref name="tool"/> as the tool called, from here: null when
    /// the call's tool is chosen already or the text typed so far names another.
    /// </summary>
    public string? PathToTool(string tool)
    {
        var s = _state;
        if (!s.Grammar.Tools.Any(t => t.Name == tool))
        {
            return null;
        }
        return s.Phase switch
        {
            Phase.BeforeFunction => FunctionOpen + tool + ">",
            Phase.Literal when s.Next == Phase.FunctionName => s.Pending + tool + ">",
            Phase.FunctionName when tool.StartsWith(s.Typed, StringComparison.Ordinal) => tool[s.Typed.Length..] + ">",
            _ => null,
        };
    }

    /// <summary>The names of the tools the call may name.</summary>
    public IEnumerable<string> ToolNames => _state.Grammar.Tools.Select(t => t.Name);

    /// <summary>The optional parameters of the tool being called that are not given yet.</summary>
    public IEnumerable<string> OptionalLeft() =>
        _state.Tool?.Parameters.Where(p => !p.Required && !_state.Given!.Contains(p.Name)).Select(p => p.Name) ?? Enumerable.Empty<string>();

    /// <summary>A reply cannot end inside a call.</summary>
    public bool CanEnd => false;

    public bool Accepts(string text) => text.Length > 0 && Run(_state, text) is not null;

    public void Advance(string text) =>
        _state = Run(_state, text) ?? throw new InvalidOperationException($"'{text}' does not continue the tool call");

    private static State? Run(State? s, string text)
    {
        foreach (char c in text)
        {
            s = s!.Feed(c);
            if (s is null)
            {
                return null;
            }
        }
        return s;
    }

    private enum Phase
    {
        BeforeFunction,     // whitespace, then "<function="
        Literal,            // typing a fixed text (Pending), then Next
        FunctionName,       // typing a tool's name, then ">"
        BetweenParameters,  // whitespace, then "<parameter=" or "</function>"
        AfterLessThan,      // "<" between parameters: "parameter=" or "/function>" follows
        ParameterName,      // typing a parameter's name, then ">"
        TextValue,          // free text up to "</parameter>"
        ChoiceValue,        // one of a string enum's values, then "</parameter>"
        ChoiceDone,         // a newline after the value: "</parameter>" next
        JsonValue,          // a JSON value, then whitespace and "</parameter>"
        AfterJson,          // the value is whole: whitespace, then "</parameter>"
        AfterFunction,      // whitespace, then "</tool_call>"
        Done,
    }

    /// <summary>One point in the call. Immutable: each character makes a new one.</summary>
    private sealed record State(ToolGrammar Grammar, Phase Phase, string Typed, ToolGrammar.Tool? Tool,
        ImmutableNames? Given, object? Json, int Match, int Whitespace)
    {
        // For Phase.Literal: the text still to type and the phase after it.
        public string Pending { get; init; } = "";
        public Phase Next { get; init; }
        public ToolGrammar.Parameter? Parameter { get; init; }

        private static bool IsWhitespace(char c) => c is ' ' or '\n' or '\r' or '\t';

        private State? Space() => Whitespace < MaxWhitespace ? this with { Whitespace = Whitespace + 1 } : null;

        private State Expect(string literal, Phase next) => this with { Phase = Phase.Literal, Pending = literal, Next = next, Whitespace = 0 };

        private bool Owed => Tool!.Parameters.Any(p => p.Required && !(Given?.Contains(p.Name) ?? false));

        public State? Feed(char c)
        {
            switch (Phase)
            {
                case Phase.BeforeFunction:
                    if (IsWhitespace(c))
                    {
                        return Space();
                    }
                    return c == FunctionOpen[0] ? Expect(FunctionOpen[1..], Phase.FunctionName) with { Typed = "" } : null;

                case Phase.Literal:
                    if (c != Pending[0])
                    {
                        return null;
                    }
                    return Pending.Length == 1 ? this with { Phase = Next, Pending = "", Typed = "", Whitespace = 0 } : this with { Pending = Pending[1..] };

                case Phase.FunctionName:
                {
                    if (c == '>')
                    {
                        var tool = Grammar.Tools.FirstOrDefault(t => t.Name == Typed);
                        return tool is null ? null : this with { Phase = Phase.BetweenParameters, Tool = tool, Given = ImmutableNames.Empty, Typed = "", Whitespace = 0 };
                    }
                    string typed = Typed + c;
                    return Grammar.Tools.Any(t => t.Name.StartsWith(typed, StringComparison.Ordinal)) ? this with { Typed = typed } : null;
                }

                case Phase.BetweenParameters:
                    if (IsWhitespace(c))
                    {
                        return Space();
                    }
                    return c == '<' ? this with { Phase = Phase.AfterLessThan, Whitespace = 0 } : null;

                case Phase.AfterLessThan:
                    // "<parameter=" while a parameter is left to give, "</function>" once none is owed.
                    if (c == 'p' && Tool!.Parameters.Any(p => !Given!.Contains(p.Name)))
                    {
                        return Expect("arameter=", Phase.ParameterName);
                    }
                    return c == '/' && !Owed ? Expect("function>", Phase.AfterFunction) : null;

                case Phase.ParameterName:
                {
                    if (c == '>')
                    {
                        var p = Tool!.Parameters.FirstOrDefault(x => x.Name == Typed && !Given!.Contains(x.Name));
                        if (p is null)
                        {
                            return null;
                        }
                        var given = Given!.With(p.Name);
                        return p.Json is { } schema
                            ? this with { Phase = Phase.JsonValue, Given = given, Parameter = p, Json = JsonConstraint.Begin(schema), Typed = "", Whitespace = 0 }
                            : p.Choices is not null
                                ? this with { Phase = Phase.ChoiceValue, Given = given, Parameter = p, Typed = "", Whitespace = 0 }
                                : this with { Phase = Phase.TextValue, Given = given, Parameter = p, Match = 0, Typed = "", Whitespace = 0 };
                    }
                    string typed = Typed + c;
                    return Tool!.Parameters.Any(x => !Given!.Contains(x.Name) && x.Name.StartsWith(typed, StringComparison.Ordinal))
                        ? this with { Typed = typed }
                        : null;
                }

                case Phase.TextValue:
                {
                    // Anything, until the text closes the parameter.
                    int match = c == ParameterClose[Match] ? Match + 1 : c == ParameterClose[0] ? 1 : 0;
                    return match == ParameterClose.Length
                        ? this with { Phase = Phase.BetweenParameters, Match = 0, Whitespace = 0 }
                        : this with { Match = match };
                }

                case Phase.ChoiceValue:
                {
                    var choices = Parameter!.Choices!;
                    if (c == '\n' && Typed.Length == 0 && Whitespace == 0)
                    {
                        return this with { Whitespace = 1 };   // the template's newline before the value
                    }
                    bool whole = choices.Contains(Typed);
                    if (whole && c == '\n')
                    {
                        return this with { Phase = Phase.ChoiceDone };
                    }
                    if (whole && c == '<' && !choices.Any(x => x.Length > Typed.Length && x.StartsWith(Typed + c, StringComparison.Ordinal)))
                    {
                        return Expect("/parameter>", Phase.BetweenParameters);
                    }
                    string typed = Typed + c;
                    return choices.Any(x => x.StartsWith(typed, StringComparison.Ordinal)) ? this with { Typed = typed } : null;
                }

                case Phase.ChoiceDone:
                    return c == '<' ? Expect("/parameter>", Phase.BetweenParameters) : null;

                case Phase.JsonValue:
                {
                    if (JsonConstraint.Step(Json!, c) is { } next)
                    {
                        return this with { Json = next };
                    }
                    // The value is whole: whitespace, then the parameter closes.
                    if (!JsonConstraint.Complete(Json!))
                    {
                        return null;
                    }
                    return (this with { Phase = Phase.AfterJson, Whitespace = 0 }).Feed(c);
                }

                case Phase.AfterJson:
                    if (IsWhitespace(c))
                    {
                        return Space();
                    }
                    return c == '<' ? Expect("/parameter>", Phase.BetweenParameters) : null;

                case Phase.AfterFunction:
                    if (IsWhitespace(c))
                    {
                        return Space();
                    }
                    return c == '<' ? Expect("/tool_call>", Phase.Done) : null;

                default:
                    return null;
            }
        }
    }

    /// <summary>The parameter names a call has given, as a small immutable set.</summary>
    private sealed class ImmutableNames
    {
        public static readonly ImmutableNames Empty = new(Array.Empty<string>());

        private readonly string[] _names;

        private ImmutableNames(string[] names) => _names = names;

        public bool Contains(string name) => Array.IndexOf(_names, name) >= 0;

        public ImmutableNames With(string name) => new(_names.Append(name).ToArray());
    }
}
