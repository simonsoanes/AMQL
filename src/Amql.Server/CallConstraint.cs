using Amql.Cli;

namespace Amql.Server;

/// <summary>A call being written under a grammar (<see cref="CallSyntax.Constraint"/>): where it
/// stands, for the rules that steer it.</summary>
public interface ICallConstraint : ITokenConstraint
{
    /// <summary>The call has closed.</summary>
    bool Done { get; }

    /// <summary>
    /// What is left to write to open <paramref name="name"/> as the next parameter, from here:
    /// null when that cannot come next (it is not a parameter of this tool, it was given
    /// already, or the call is not between parameters).
    /// </summary>
    string? PathTo(string name);

    /// <summary>
    /// What is left to write to name <paramref name="tool"/> as the tool called, from here: null when
    /// the call's tool is chosen already or the text typed so far names another.
    /// </summary>
    string? PathToTool(string tool);

    /// <summary>The names of the tools the call may name.</summary>
    IEnumerable<string> ToolNames { get; }

    /// <summary>The optional parameters of the tool being called that are not given yet.</summary>
    IEnumerable<string> OptionalLeft();
}

/// <summary>
/// A tool call as the model's chat template writes it (<see cref="CallShape"/>), held to the request's
/// tools: it names a tool that exists, gives each of its parameters once and the required ones before it
/// closes, and every value is one the parameter's schema allows. A small model's calls are then always
/// ones the harness can run. Blank space the template writes may be written differently. Starts after
/// the shape's opener; done at the end of the call, or of the section where the template puts several
/// calls in one.
/// </summary>
public sealed class CallConstraint : ICallConstraint
{
    private const int MaxWhitespace = 8;

    private readonly ToolGrammar _grammar;
    private readonly CallShape _shape;
    private readonly string _start, _next, _afterName, _end, _close, _pairOpen, _nextPair, _keyEnd, _pairClose;
    private State _state;

    public CallConstraint(CallShape shape, ToolGrammar grammar)
    {
        _grammar = grammar;
        _shape = shape;
        _start = shape.AfterOpener.Trim();
        _next = (shape.Separator + shape.Lead).Trim();
        _afterName = shape.AfterName.Trim();
        _end = shape.End.Trim();
        _close = shape.SectionOpen.Length > 0 ? shape.SectionClose.Trim() : "";
        _pairOpen = shape.PairOpen.Trim();
        _nextPair = shape.Style == ArgumentStyle.Python ? shape.PairSeparator.Trim() : (shape.PairSeparator + shape.PairOpen).Trim();
        _keyEnd = shape.KeyEnd.Trim();
        _pairClose = shape.PairClose.Trim();
        _state = new State(this).After(_start, Go.Name);
    }

    public bool Done => _state.Phase == Phase.Done;

    /// <summary>A reply cannot end inside a call.</summary>
    public bool CanEnd => false;

    public IEnumerable<string> ToolNames => _grammar.Tools.Select(t => t.Name);

    public bool Accepts(string text) => text.Length > 0 && Run(_state, text) is not null;

    public void Advance(string text) =>
        _state = Run(_state, text) ?? throw new InvalidOperationException($"'{text}' does not continue the tool call");

    public IEnumerable<string> OptionalLeft() => _state.Tool is { } tool
        ? tool.Parameters.Where(p => !p.Required && !_state.Given.Contains(p.Name)).Select(p => p.Name)
        : Enumerable.Empty<string>();

    public string? PathTo(string name)
    {
        var s = _state;
        if (s.Tool is null || !s.Tool.Parameters.Any(p => p.Name == name) || s.Given.Contains(name))
        {
            return null;
        }
        switch (s.Phase)
        {
            case Phase.Literals:
                return s.Options.FirstOrDefault(o => o.Then is Go.Parameter or Go.Key) is { Text: not null } o ? o.Text[o.At..] + name + _keyEnd : null;
            case Phase.Parameter or Phase.Key:
                return name.StartsWith(s.Typed, StringComparison.Ordinal) ? name[s.Typed.Length..] + _keyEnd : null;
            case Phase.PythonValue when _shape.Style == ArgumentStyle.Python && s.Skip.Length == 0 && Whole(s.Json!):
                return _nextPair + " " + name + _keyEnd;
            case Phase.Object when s.Nest.Length == 1:
                string open = _shape.Style == ArgumentStyle.Json ? "\"" : "", close = open + ":";
                if (_shape.Style == ArgumentStyle.Json && s.InString)
                {
                    return s.StringIsKey && name.StartsWith(s.Key, StringComparison.Ordinal) ? name[s.Key.Length..] + close : null;
                }
                if (s.InString || s.Match > 0)
                {
                    return null;
                }
                if (s.Key.Length > 0)
                {
                    return name.StartsWith(s.Key, StringComparison.Ordinal) ? name[s.Key.Length..] + close : null;
                }
                return s.KeyNext ? open + name + close : Whole(s.Json!) ? (open.Length > 0 ? ", " : ",") + open + name + close : null;
            default:
                return null;
        }

        // After a whole value (not a number, which may go on).
        static bool Whole(object json) => JsonConstraint.Step(json, '0') is null && JsonConstraint.Step(json, '.') is null && JsonConstraint.Step(json, 'e') is null &&
            (JsonConstraint.Step(json, ',') is not null || JsonConstraint.Complete(json));
    }

    public string? PathToTool(string tool)
    {
        var s = _state;
        if (!_grammar.Tools.Any(t => t.Name == tool))
        {
            return null;
        }
        string end = _afterName.Length > 0 ? _afterName[..1] : _shape.Style is ArgumentStyle.Json or ArgumentStyle.Marked ? "{" : "";
        return s.Phase switch
        {
            Phase.Literals when s.Options.FirstOrDefault(o => o.Then == Go.Name) is { Text: not null } o => o.Text[o.At..] + tool + end,
            Phase.Name when tool.StartsWith(s.Typed, StringComparison.Ordinal) => tool[s.Typed.Length..] + end,
            _ => null,
        };
    }

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
        Literals,      // one of some fixed texts
        Name,          // typing a tool's name
        Object,        // the arguments as one object (JSON, or its marked form)
        Parameter,     // typing a parameter's name
        Text,          // a parameter's value as it is, up to what closes it
        Choice,        // one of a string enum's values
        Value,         // a parameter's value as JSON
        Key,           // a Python keyword, or the end of the call
        PythonValue,   // a Python literal
        Done,
    }

    /// <summary>Where a fixed text leads.</summary>
    private enum Go { Name, Arguments, Pairs, Parameter, Key, Ended, Done }

    private readonly record struct Option(string Text, int At, int Whitespace, Go Then);

    /// <summary>One point in the call. Immutable: each character makes a new one.</summary>
    private sealed record State(CallConstraint Owner)
    {
        public Phase Phase { get; init; }
        public Option[] Options { get; init; } = [];
        public string Typed { get; init; } = "";
        public ToolGrammar.Tool? Tool { get; init; }
        public NameSet Given { get; init; } = NameSet.Empty;
        public ToolGrammar.Parameter? Parameter { get; init; }
        public object? Json { get; init; }              // the value (or all the arguments) so far, as JSON
        public int Match { get; init; }                 // how much of a closing text or quote mark has been read
        public bool Cdata { get; init; }

        // The object's (Phase.Object) and, partly, a Python value's progress.
        public string Nest { get; init; } = "";         // the objects and arrays open
        public bool InString { get; init; }
        public bool Escape { get; init; }
        public bool StringIsKey { get; init; }
        public bool KeyNext { get; init; }              // an object waits for a key
        public string Key { get; init; } = "";          // a key being written
        public char QuoteChar { get; init; }
        public string Skip { get; init; } = "";         // text that must follow and says nothing more
        public string Hex { get; init; } = "";          // an escape's digits still to come, as '?'

        private bool Owed => Tool!.Parameters.Any(p => p.Required && !Given.Contains(p.Name));
        private bool Left => Tool!.Parameters.Any(p => !Given.Contains(p.Name));

        // ── moving on ──

        /// <summary>Past a fixed text (empty: at once).</summary>
        public State After(string text, Go then) => text.Length == 0 ? Enter(then) : Literal([new Option(text, 0, 0, then)]);

        private State Literal(Option[] options) => this with { Phase = Phase.Literals, Options = options, Typed = "", Match = 0, Cdata = false, Parameter = null, Json = null };

        private State Enter(Go go)
        {
            switch (go)
            {
                case Go.Name:
                    return this with { Phase = Phase.Name, Typed = "", Match = 0, Tool = null, Given = NameSet.Empty };
                case Go.Arguments when Owner._shape.Style is ArgumentStyle.Json or ArgumentStyle.Marked:
                    return this with { Phase = Phase.Object, Json = JsonConstraint.Begin(Tool!.Arguments), Nest = "", InString = false, KeyNext = false, Key = "", Match = 0 };
                case Go.Arguments when Owner._shape.Style == ArgumentStyle.Python:
                case Go.Key:
                    return this with { Phase = Phase.Key, Typed = "", Json = null };
                case Go.Arguments or Go.Pairs:
                {
                    var options = new List<Option>();
                    if (Left)
                    {
                        options.Add(new Option(Given.IsEmpty ? Owner._pairOpen : Owner._nextPair, 0, 0, Go.Parameter));
                    }
                    if (!Owed)
                    {
                        options.AddRange(Ends());
                    }
                    return Literal(options.ToArray());
                }
                case Go.Parameter:
                    return this with { Phase = Phase.Parameter, Typed = "" };
                case Go.Ended when Owner._close.Length > 0:
                    return Literal(Ends(afterEnd: true).ToArray());
                default:
                    return this with { Phase = Phase.Done, Options = [] };
            }
        }

        /// <summary>What may stand where a call's arguments are whole.</summary>
        private IEnumerable<Option> Ends(bool afterEnd = false) =>
            !afterEnd && Owner._end.Length > 0 ? [new Option(Owner._end, 0, 0, Go.Ended)]
            : [new Option(Owner._next, 0, 0, Go.Name), new Option(Owner._close, 0, 0, Go.Done)];

        private State? Json1(string text)
        {
            object? json = Json;
            foreach (char c in text)
            {
                json = JsonConstraint.Step(json!, c);
                if (json is null)
                {
                    return null;
                }
            }
            return this with { Json = json };
        }

        public State? Feed(char c)
        {
            switch (Phase)
            {
                case Phase.Literals:
                    return Fixed(c);

                case Phase.Name:
                {
                    if (Typed.Length == 0 && char.IsWhiteSpace(c))
                    {
                        return Match < MaxWhitespace ? this with { Match = Match + 1 } : null;
                    }
                    if (Owner._grammar.Tools.FirstOrDefault(t => t.Name == Typed) is { } tool)
                    {
                        // The name is whole when what comes after it starts (it holds no blank space).
                        var named = this with { Tool = tool, Given = NameSet.Empty };
                        var next = Owner._afterName.Length == 0 ? named.Enter(Go.Arguments)
                            : named.Literal([new Option(Owner._afterName, 0, char.IsWhiteSpace(Owner._shape.AfterName[0]) ? 0 : MaxWhitespace, Go.Arguments)]);
                        if (next.Feed(c) is { } on)
                        {
                            return on;
                        }
                    }
                    string typed = Typed + c;
                    return Owner._grammar.Tools.Any(t => t.Name.StartsWith(typed, StringComparison.Ordinal)) ? this with { Typed = typed } : null;
                }

                case Phase.Object:
                    return Owner._shape.Style == ArgumentStyle.Json ? Plain(c) : InString ? Inside(c) : Outside(c);

                case Phase.Parameter:
                {
                    if (c == Owner._keyEnd[0] && Tool!.Parameters.FirstOrDefault(p => p.Name == Typed && !Given.Contains(p.Name)) is { } p &&
                        (this with { Parameter = p, Given = Given.With(p.Name) }).KeyEnd(c, 0) is { } on)
                    {
                        return on;
                    }
                    string typed = Typed + c;
                    return Tool!.Parameters.Any(x => !Given.Contains(x.Name) && x.Name.StartsWith(typed, StringComparison.Ordinal)) ? this with { Typed = typed } : null;
                }

                case Phase.Text when Skip.Length > 0:
                    return KeyEnd(c, Owner._keyEnd.Length - Skip.Length);

                case Phase.Text:
                {
                    // Anything, until the text closes the parameter (or the CDATA it is in, first).
                    string close = Cdata ? "]]>" : Owner._pairClose;
                    int match = c == close[Match] ? Match + 1 : Cdata && c == ']' && Match == 2 ? 2 : c == close[0] ? 1 : 0;
                    if (match == close.Length)
                    {
                        return Cdata ? Literal([new Option(Owner._pairClose, 0, 0, Go.Pairs)]) : Enter(Go.Pairs);
                    }
                    const string cdata = "<![CDATA[";
                    string typed = Typed.Length >= cdata.Length || Cdata ? Typed : (Typed + c).TrimStart();
                    return this with { Match = match, Typed = typed, Cdata = Cdata || typed == cdata };
                }

                case Phase.Choice:
                {
                    var choices = Parameter!.Choices!;
                    if (c == '\n' && Typed.Length == 0 && Match == 0)
                    {
                        return this with { Match = 1 };   // the template's newline before the value
                    }
                    if (choices.Contains(Typed) && !choices.Any(x => x.Length > Typed.Length && x.StartsWith(Typed + c, StringComparison.Ordinal)))
                    {
                        var closing = Literal([new Option(Owner._pairClose, 0, MaxWhitespace, Go.Pairs)]);
                        return c == '\n' ? closing : closing.Feed(c);
                    }
                    string typed = Typed + c;
                    return choices.Any(x => x.StartsWith(typed, StringComparison.Ordinal)) ? this with { Typed = typed } : null;
                }

                case Phase.Value when Skip.Length > 0:
                    return c == Skip[0] ? this with { Skip = Skip[1..], Match = 1 } : null;

                case Phase.Value when Match == 0 && Owner._shape.PythonNames && c is 'T' or 'F' or 'N':
                    // True, False and None, which some templates write so.
                    return c == 'N' ? (this with { Skip = "one", Match = 1 }).Json1("null") : (this with { Match = 1 }).Json1(char.ToLowerInvariant(c).ToString());

                case Phase.Value:
                    return JsonConstraint.Step(Json!, c) is { } next1 ? this with { Json = next1, Match = char.IsWhiteSpace(c) ? Match : 1 }
                        : JsonConstraint.Complete(Json!) ? Literal([new Option(Owner._pairClose, 0, 0, Go.Pairs)]).Feed(c) : null;

                case Phase.Key:
                    return Keyword(c);

                case Phase.PythonValue:
                    return Python(c);

                default:
                    return null;
            }
        }

        private State? Fixed(char c)
        {
            var left = new List<Option>(Options.Length);
            foreach (var o in Options)
            {
                int at = o.At;
                bool space = char.IsWhiteSpace(c);
                if (space && (at == 0 || char.IsWhiteSpace(o.Text[at])))
                {
                    if (o.Whitespace < MaxWhitespace)
                    {
                        left.Add(o with { Whitespace = o.Whitespace + 1 });
                    }
                    continue;
                }
                while (char.IsWhiteSpace(o.Text[at]))
                {
                    at++;
                }
                if (o.Text[at] != c)
                {
                    continue;
                }
                if (at + 1 == o.Text.Length)
                {
                    return Enter(o.Then);
                }
                left.Add(o with { At = at + 1, Whitespace = 0 });
            }
            return left.Count == 0 ? null : this with { Options = left.ToArray() };
        }

        /// <summary>Between a parameter's name and its value.</summary>
        private State? KeyEnd(char c, int at)
        {
            string end = Owner._keyEnd;
            while (char.IsWhiteSpace(end[at]) && !char.IsWhiteSpace(c))
            {
                at++;
            }
            if (end[at] != c && !(char.IsWhiteSpace(end[at]) && char.IsWhiteSpace(c)))
            {
                return null;
            }
            if (at + 1 < end.Length)
            {
                return this with { Phase = Phase.Text, Skip = end[(at + 1)..], Typed = "" };
            }
            var begun = this with { Skip = "", Typed = "", Match = 0, Cdata = false };
            if (Owner._shape.PythonObjects && Parameter!.Json is { } python && (python.Obj || python.Arr))
            {
                return begun with { Phase = Phase.PythonValue, Json = JsonConstraint.Begin(python), InString = false };
            }
            return Parameter!.Json is { } schema ? begun with { Phase = Phase.Value, Json = JsonConstraint.Begin(schema) }
                : begun with { Phase = Parameter.Choices is null ? Phase.Text : Phase.Choice };
        }

        // ── one JSON object ──

        private State? Plain(char c)
        {
            if (InString)
            {
                if (Json1(c.ToString()) is not { } s)
                {
                    return null;
                }
                if (Escape)
                {
                    return s with { Escape = false, Key = StringIsKey ? Key + c : Key };
                }
                return c switch
                {
                    '\\' => s with { Escape = true },
                    '"' => s with { InString = false, StringIsKey = false, Key = "", Given = StringIsKey && Nest.Length == 1 ? Given.With(Key) : Given },
                    _ => s with { Key = StringIsKey ? Key + c : Key },
                };
            }
            if (Json1(c.ToString()) is not { } next)
            {
                return null;
            }
            return c switch
            {
                '"' => next with { InString = true, Escape = false, StringIsKey = KeyNext, KeyNext = false, Key = "" },
                _ => next.Structure(c),
            };
        }

        /// <summary>The object's nesting after a character outside strings.</summary>
        private State Structure(char c)
        {
            switch (c)
            {
                case '{' or '[':
                    return this with { Nest = Nest + c, KeyNext = c == '{' };
                case '}' or ']':
                    if (Nest.Length > 1)
                    {
                        return this with { Nest = Nest[..^1], KeyNext = false };
                    }
                    var whole = this with { Nest = "", KeyNext = false };
                    return Owner._end.Length > 0 || Owner._close.Length > 0 ? whole.Literal(Ends().ToArray()) : whole.Enter(Go.Done);
                case ',':
                    return this with { KeyNext = Nest[^1] == '{' };
                default:
                    return this;
            }
        }

        // ── one object with bare keys, strings between marks and nothing escaped in them, checked as the JSON it stands for ──

        private State? Inside(char c)
        {
            string quote = Owner._shape.Quote;
            if (c == quote[Match])
            {
                if (Match + 1 < quote.Length)
                {
                    return this with { Match = Match + 1 };
                }
                var closed = (this with { InString = false, Match = 0, StringIsKey = false }).Json1("\"");
                return closed is null || !StringIsKey ? closed : closed with { Key = "", Given = Nest.Length == 1 ? Given.With(Key) : Given };
            }
            // Not the quote mark after all: what was read of it is text.
            string held = quote[..Match];
            var flushed = held.Length == 0 ? this : (this with { Match = 0 }).Json1(Escaped(held));
            if (flushed is null)
            {
                return null;
            }
            if (c == quote[0])
            {
                return flushed with { Match = 1 };
            }
            return flushed.Json1(Escaped(c.ToString())) is { } s ? s with { Key = StringIsKey ? Key + held + c : Key } : null;
        }

        private State? Outside(char c)
        {
            string quote = Owner._shape.Quote;
            if (Match > 0)
            {
                if (c != quote[Match])
                {
                    return null;
                }
                return Match + 1 < quote.Length ? this with { Match = Match + 1 }
                    : (this with { Match = 0, InString = true, StringIsKey = KeyNext, KeyNext = false, Key = "" }).Json1("\"");
            }
            if (c == quote[0])
            {
                return Key.Length == 0 && Json1("\"") is not null ? this with { Match = 1 } : null;
            }
            if (Key.Length > 0)
            {
                if (c == ':')
                {
                    return (this with { Key = "", Given = Nest.Length == 1 ? Given.With(Key) : Given }).Json1("\":");
                }
                return c is '"' or '{' or '}' or '[' or ']' or ',' ? null : (this with { Key = Key + c }).Json1(Escaped(c.ToString()));
            }
            if (KeyNext && !char.IsWhiteSpace(c) && c != '}')
            {
                return c is '"' or '{' or '[' or ']' or ',' or ':' ? null
                    : (this with { KeyNext = false, Key = c.ToString() }).Json1("\"" + Escaped(c.ToString()));
            }
            // A string stands between quote marks here.
            return c == '"' ? null : Json1(c.ToString())?.Structure(c);
        }

        // ── a Python call ──

        private State? Keyword(char c)
        {
            if (Typed.Length == 0)
            {
                if (char.IsWhiteSpace(c))
                {
                    return Match < MaxWhitespace ? this with { Match = Match + 1 } : null;
                }
                if (Given.IsEmpty && !Owed && Literal(Ends().ToArray()).Feed(c) is { } ended)
                {
                    return ended;
                }
            }
            if (c == Owner._keyEnd[0] && Tool!.Parameters.FirstOrDefault(p => p.Name == Typed && !Given.Contains(p.Name)) is { } p)
            {
                var schema = Tool.Arguments.Properties?.FirstOrDefault(x => x.Name == p.Name).Node ?? JsonSchemaNode.Any();
                return this with { Phase = Phase.PythonValue, Parameter = p, Given = Given.With(p.Name), Json = JsonConstraint.Begin(schema), Typed = "", Match = 0, InString = false, Skip = "" };
            }
            string typed = Typed + c;
            return Tool!.Parameters.Any(x => !Given.Contains(x.Name) && x.Name.StartsWith(typed, StringComparison.Ordinal)) ? this with { Typed = typed } : null;
        }

        private State? Python(char c)
        {
            if (Skip.Length > 0)
            {
                return c == Skip[0] ? this with { Skip = Skip[1..] } : null;
            }
            if (InString)
            {
                if (Hex.Length > 0)
                {
                    if (!char.IsAsciiHexDigit(c))
                    {
                        return null;
                    }
                    int at = Hex.IndexOf('?');
                    string hex = Hex[..at] + c + Hex[(at + 1)..];
                    return hex.Contains('?') ? this with { Hex = hex }
                        : (this with { Hex = "" }).Json1(Escaped(((char)Convert.ToInt32(hex, 16)).ToString()));
                }
                if (Escape)
                {
                    var s = this with { Escape = false };
                    return c switch
                    {
                        'x' => s with { Hex = "??" },
                        'u' => s with { Hex = "????" },
                        'n' or 't' or 'r' or 'b' or 'f' or '\\' => s.Json1("\\" + c),
                        '\'' or '"' => s.Json1(Escaped(c.ToString())),
                        _ => s.Json1("\\\\" + Escaped(c.ToString())),
                    };
                }
                return c == '\\' ? this with { Escape = true }
                    : c == QuoteChar ? (this with { InString = false }).Json1("\"") : Json1(Escaped(c.ToString()));
            }
            var taken = c switch
            {
                '\'' or '"' => (this with { InString = true, QuoteChar = c, Escape = false, Hex = "" }).Json1("\""),
                'T' => Json1("t"),
                'F' => Json1("f"),
                'N' => (this with { Skip = "one" }).Json1("null"),
                _ => Json1(c.ToString()),
            };
            if (taken is not null || !JsonConstraint.Complete(Json!))
            {
                return taken;
            }
            if (Owner._shape.Style == ArgumentStyle.Pairs)
            {
                return Literal([new Option(Owner._pairClose, 0, 0, Go.Pairs)]).Feed(c);
            }
            var options = new List<Option>();
            if (Left)
            {
                options.Add(new Option(Owner._nextPair, 0, 0, Go.Key));
            }
            if (!Owed)
            {
                options.AddRange(Ends());
            }
            return Literal(options.ToArray()).Feed(c);
        }

        /// <summary>A string's characters as <see cref="System.Text.Json.JsonSerializer"/> escapes them, which is how
        /// a schema's enum values are held.</summary>
        private static string Escaped(string text)
        {
            var sb = new System.Text.StringBuilder();
            foreach (char c in text)
            {
                sb.Append(c switch
                {
                    '\n' => "\\n",
                    '\r' => "\\r",
                    '\t' => "\\t",
                    '\b' => "\\b",
                    '\f' => "\\f",
                    '\\' => "\\\\",
                    < ' ' or > '~' or '"' or '&' or '\'' or '+' or '<' or '>' or '`' => "\\u" + ((int)c).ToString("X4"),
                    _ => c.ToString(),
                });
            }
            return sb.ToString();
        }
    }

    /// <summary>The parameter names a call has given, as a small immutable set.</summary>
    private sealed class NameSet
    {
        public static readonly NameSet Empty = new(Array.Empty<string>());

        private readonly string[] _names;

        private NameSet(string[] names) => _names = names;

        public bool IsEmpty => _names.Length == 0;

        public bool Contains(string name) => Array.IndexOf(_names, name) >= 0;

        public NameSet With(string name) => new(_names.Append(name).ToArray());
    }
}
