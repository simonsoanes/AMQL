using System.Globalization;
using System.Text;

namespace Amql.Cli.Jinja;

/// <summary>A template that cannot be read, or that fails as it renders.</summary>
public sealed class JinjaException(string message, int line = 0)
    : Exception(line > 0 ? $"{message} (line {line} of the template)" : message);

internal enum TokenKind { Text, VarBegin, VarEnd, BlockBegin, BlockEnd, Name, String, Integer, Float, Op, End }

internal readonly record struct Token(TokenKind Kind, string Text, int Line)
{
    public bool Is(string op) => Kind == TokenKind.Op && Text == op;

    public bool IsName(string name) => Kind == TokenKind.Name && Text == name;
}

/// <summary>
/// A template's text as tokens, with the whitespace rules transformers compiles chat templates
/// under: a newline after a block tag is dropped (<c>trim_blocks</c>), and blanks before one on
/// its line (<c>lstrip_blocks</c>); <c>-</c> strips more and <c>+</c> strips nothing.
/// </summary>
internal sealed class JinjaLexer(string source)
{
    private readonly string _s = Prepared(source);
    private readonly List<Token> _tokens = new();
    private int _at, _line = 1;

    private static string Prepared(string s)
    {
        s = s.Replace("\r\n", "\n").Replace('\r', '\n');
        return s.EndsWith('\n') ? s[..^1] : s;   // one newline at the end is no part of it
    }

    private static readonly string[] Operators =
    [
        "//", "**", "==", "!=", "<=", ">=", "+", "-", "/", "*", "%", "~", "[", "]", "(", ")", "{", "}", "<", ">", "=", ".", ":", "|", ",", ";",
    ];

    public List<Token> Read()
    {
        bool lineStart = true, stripLeft = false;
        while (_at <= _s.Length)
        {
            int open = NextTag(_at);
            string text = open < 0 ? _s[_at..] : _s[_at..open];
            int consumed = text.Length;
            if (stripLeft)
            {
                string trimmed = text.TrimStart(Whitespace);
                text = trimmed;
            }
            if (open < 0)
            {
                Text(text, _s[_at..]);
                break;
            }
            char kind = _s[open + 1];
            char sign = open + 2 < _s.Length ? _s[open + 2] : '\0';
            if (sign == '-')
            {
                text = text.TrimEnd(Whitespace);
            }
            else if (sign != '+' && kind != '{')
            {
                int nl = text.LastIndexOf('\n') + 1;
                bool whole = !stripLeft || consumed == text.Length;
                if ((nl > 0 || (lineStart && whole)) && text.AsSpan(nl).IndexOfAnyExcept(" \t\f\v") < 0)
                {
                    text = text[..nl];
                }
            }
            Text(text, _s.Substring(_at, consumed));
            _at = open + 2 + (sign is '-' or '+' ? 1 : 0);
            stripLeft = false;
            lineStart = false;
            if (kind == '#')
            {
                int end = _s.IndexOf("#}", _at, StringComparison.Ordinal);
                if (end < 0)
                {
                    throw new JinjaException("a comment does not end", _line);
                }
                bool strip = end > 0 && _s[end - 1] == '-';
                bool keep = end > 0 && _s[end - 1] == '+';
                Count(_s.AsSpan(_at, end + 2 - _at));
                _at = end + 2;
                (stripLeft, lineStart) = After(strip, keep, block: true);
                continue;
            }
            if (kind == '%' && Raw(out bool rawStrip, out bool rawKeep))
            {
                (stripLeft, lineStart) = After(rawStrip, rawKeep, block: true);
                continue;
            }
            bool isBlock = kind == '%';
            _tokens.Add(new Token(isBlock ? TokenKind.BlockBegin : TokenKind.VarBegin, "", _line));
            (bool dash, bool plus) = Inside(isBlock);
            _tokens.Add(new Token(isBlock ? TokenKind.BlockEnd : TokenKind.VarEnd, "", _line));
            (stripLeft, lineStart) = After(dash, plus, isBlock);
        }
        _tokens.Add(new Token(TokenKind.End, "", _line));
        return _tokens;
    }

    private static readonly char[] Whitespace = [' ', '\t', '\n', '\r', '\f', '\v'];

    private (bool Strip, bool LineStart) After(bool strip, bool keep, bool block)
    {
        if (!strip && !keep && block && _at < _s.Length && _s[_at] == '\n')
        {
            _at++;
            _line++;
            return (false, true);
        }
        return (strip, false);
    }

    private void Text(string text, string consumed)
    {
        if (text.Length > 0)
        {
            _tokens.Add(new Token(TokenKind.Text, text, _line));
        }
        Count(consumed);
        _at += consumed.Length;
    }

    private void Count(ReadOnlySpan<char> text) => _line += text.Count('\n');

    private int NextTag(int from)
    {
        while (true)
        {
            int i = _s.IndexOf('{', from);
            if (i < 0 || i + 1 >= _s.Length)
            {
                return -1;
            }
            if (_s[i + 1] is '{' or '%' or '#')
            {
                return i;
            }
            from = i + 1;
        }
    }

    /// <summary>A raw block at this tag, taken whole.</summary>
    private bool Raw(out bool strip, out bool keep)
    {
        strip = keep = false;
        int i = _at;
        while (i < _s.Length && char.IsWhiteSpace(_s[i]))
        {
            i++;
        }
        if (string.CompareOrdinal(_s, i, "raw", 0, 3) != 0)
        {
            return false;
        }
        i += 3;
        while (i < _s.Length && char.IsWhiteSpace(_s[i]))
        {
            i++;
        }
        bool bodyStrip = i < _s.Length && _s[i] == '-', bodyKeep = i < _s.Length && _s[i] == '+';
        i += bodyStrip || bodyKeep ? 1 : 0;
        if (string.CompareOrdinal(_s, i, "%}", 0, 2) != 0)
        {
            return false;
        }
        int body = i + 2;
        var m = System.Text.RegularExpressions.Regex.Match(_s[body..], @"\{%([-+]?)\s*endraw\s*([-+]?)%\}");
        if (!m.Success)
        {
            throw new JinjaException("a raw block does not end", _line);
        }
        string text = _s.Substring(body, m.Index);
        if (bodyStrip)
        {
            text = text.TrimStart(Whitespace);
        }
        else if (!bodyKeep && text.StartsWith('\n'))
        {
            text = text[1..];
        }
        if (m.Groups[1].Value == "-")
        {
            text = text.TrimEnd(Whitespace);
        }
        else if (m.Groups[1].Value != "+" && text.LastIndexOf('\n') + 1 is var nl && text.AsSpan(nl).IndexOfAnyExcept(' ', '\t') < 0)
        {
            text = text[..nl];
        }
        if (text.Length > 0)
        {
            _tokens.Add(new Token(TokenKind.Text, text, _line));
        }
        int end = body + m.Index + m.Length;
        Count(_s.AsSpan(_at, end - _at));
        _at = end;
        strip = m.Groups[2].Value == "-";
        keep = m.Groups[2].Value == "+";
        return true;
    }

    /// <summary>The tokens of one tag, up to its end: whether it ended with <c>-</c> or <c>+</c>.</summary>
    private (bool Dash, bool Plus) Inside(bool block)
    {
        string close = block ? "%}" : "}}";
        int depth = 0;
        while (true)
        {
            while (_at < _s.Length && char.IsWhiteSpace(_s[_at]))
            {
                _line += _s[_at] == '\n' ? 1 : 0;
                _at++;
            }
            if (_at >= _s.Length)
            {
                throw new JinjaException("a tag does not end", _line);
            }
            char c = _s[_at];
            if (depth == 0 && At(close))
            {
                _at += 2;
                return (false, false);
            }
            if (depth == 0 && (c == '-' || (c == '+' && block)) && string.CompareOrdinal(_s, _at + 1, close, 0, 2) == 0)
            {
                _at += 3;
                return (c == '-', c == '+');
            }
            if (c is '"' or '\'')
            {
                _tokens.Add(new Token(TokenKind.String, Literal(c), _line));
                continue;
            }
            if (char.IsDigit(c))
            {
                Number();
                continue;
            }
            if (char.IsLetter(c) || c == '_')
            {
                int from = _at;
                while (_at < _s.Length && (char.IsLetterOrDigit(_s[_at]) || _s[_at] == '_'))
                {
                    _at++;
                }
                _tokens.Add(new Token(TokenKind.Name, _s[from.._at], _line));
                continue;
            }
            string? op = Operators.FirstOrDefault(At);
            if (op is null)
            {
                throw new JinjaException($"unexpected character '{c}'", _line);
            }
            depth += op is "(" or "[" or "{" ? 1 : op is ")" or "]" or "}" ? -1 : 0;
            _tokens.Add(new Token(TokenKind.Op, op, _line));
            _at += op.Length;
        }
    }

    private bool At(string text) => string.CompareOrdinal(_s, _at, text, 0, text.Length) == 0;

    private void Number()
    {
        int from = _at;
        bool real = false;
        if (_s[_at] == '0' && _at + 1 < _s.Length && _s[_at + 1] is 'x' or 'X' or 'b' or 'B' or 'o' or 'O')
        {
            int radix = char.ToLowerInvariant(_s[_at + 1]) switch { 'x' => 16, 'b' => 2, _ => 8 };
            _at += 2;
            int digits = _at;
            while (_at < _s.Length && (char.IsAsciiHexDigit(_s[_at]) || _s[_at] == '_'))
            {
                _at++;
            }
            _tokens.Add(new Token(TokenKind.Integer, Convert.ToInt64(_s[digits.._at].Replace("_", ""), radix).ToString(CultureInfo.InvariantCulture), _line));
            return;
        }
        while (_at < _s.Length && (char.IsDigit(_s[_at]) || _s[_at] == '_'))
        {
            _at++;
        }
        if (_at + 1 < _s.Length && _s[_at] == '.' && char.IsDigit(_s[_at + 1]))
        {
            real = true;
            _at++;
            while (_at < _s.Length && (char.IsDigit(_s[_at]) || _s[_at] == '_'))
            {
                _at++;
            }
        }
        if (_at < _s.Length && _s[_at] is 'e' or 'E')
        {
            int mark = _at + 1;
            if (mark < _s.Length && _s[mark] is '+' or '-')
            {
                mark++;
            }
            if (mark < _s.Length && char.IsDigit(_s[mark]))
            {
                real = true;
                _at = mark;
                while (_at < _s.Length && char.IsDigit(_s[_at]))
                {
                    _at++;
                }
            }
        }
        _tokens.Add(new Token(real ? TokenKind.Float : TokenKind.Integer, _s[from.._at].Replace("_", ""), _line));
    }

    private string Literal(char quote)
    {
        int line = _line;
        var sb = new StringBuilder();
        for (_at++; ; _at++)
        {
            if (_at >= _s.Length)
            {
                throw new JinjaException("a string does not end", line);
            }
            char c = _s[_at];
            if (c == quote)
            {
                _at++;
                return sb.ToString();
            }
            _line += c == '\n' ? 1 : 0;
            if (c != '\\' || _at + 1 >= _s.Length)
            {
                sb.Append(c);
                continue;
            }
            char e = _s[++_at];
            switch (e)
            {
                case 'n': sb.Append('\n'); break;
                case 't': sb.Append('\t'); break;
                case 'r': sb.Append('\r'); break;
                case 'b': sb.Append('\b'); break;
                case 'f': sb.Append('\f'); break;
                case 'v': sb.Append('\v'); break;
                case 'a': sb.Append('\a'); break;
                case '\n': _line++; break;
                case '\\' or '\'' or '"': sb.Append(e); break;
                case 'x' or 'u' or 'U':
                {
                    int n = e == 'x' ? 2 : e == 'u' ? 4 : 8;
                    if (_at + n >= _s.Length || !uint.TryParse(_s.AsSpan(_at + 1, n), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out uint code) || code > 0x10FFFF)
                    {
                        throw new JinjaException("a string holds a bad escape", _line);
                    }
                    sb.Append(code < 0x10000 ? ((char)code).ToString() : char.ConvertFromUtf32((int)code));
                    _at += n;
                    break;
                }
                case >= '0' and <= '7':
                {
                    int v = 0, n = 0;
                    while (n < 3 && _at + n < _s.Length && _s[_at + n] is >= '0' and <= '7')
                    {
                        v = 8 * v + (_s[_at + n++] - '0');
                    }
                    sb.Append((char)v);
                    _at += n - 1;
                    break;
                }
                default: sb.Append('\\').Append(e); break;
            }
        }
    }
}

// ── the tree ──────────────────────────────────────────────────────────────

internal abstract record Node(int Line);

internal abstract record Expr(int Line) : Node(Line);

internal sealed record Const(object? Value, int Line) : Expr(Line);

internal sealed record NameRef(string Name, int Line) : Expr(Line);

internal sealed record GetAttr(Expr Target, string Name, int Line) : Expr(Line);

internal sealed record GetItem(Expr Target, Expr Index, int Line) : Expr(Line);

internal sealed record Slice(Expr? From, Expr? To, Expr? Step, int Line) : Expr(Line);

internal sealed record Call(Expr Target, List<Expr> Args, List<(string Name, Expr Value)> Named, int Line) : Expr(Line);

internal sealed record FilterUse(Expr? Target, string Name, List<Expr> Args, List<(string Name, Expr Value)> Named, int Line) : Expr(Line);

internal sealed record TestUse(Expr Target, string Name, List<Expr> Args, bool Negated, int Line) : Expr(Line);

internal sealed record Unary(string Op, Expr Value, int Line) : Expr(Line);

internal sealed record Binary(string Op, Expr Left, Expr Right, int Line) : Expr(Line);

internal sealed record Compare(Expr First, List<(string Op, Expr Value)> Rest, int Line) : Expr(Line);

internal sealed record Conditional(Expr Test, Expr Then, Expr? Else, int Line) : Expr(Line);

internal sealed record ListOf(List<Expr> Items, bool Tuple, int Line) : Expr(Line);

internal sealed record DictOf(List<(Expr Key, Expr Value)> Items, int Line) : Expr(Line);

internal sealed record TextNode(string Text, int Line) : Node(Line);

internal sealed record Output(Expr Value, int Line) : Node(Line);

internal sealed record If(List<(Expr Test, List<Node> Body)> Branches, List<Node>? Else, int Line) : Node(Line);

internal sealed record For(Expr Target, Expr Source, Expr? Filter, List<Node> Body, List<Node>? Else, bool Recursive, int Line) : Node(Line);

internal sealed record Set(Expr Target, Expr Value, int Line) : Node(Line);

internal sealed record SetBlock(Expr Target, List<Node> Body, Expr? Filter, int Line) : Node(Line);

internal sealed record Macro(string Name, List<(string Name, Expr? Default)> Parameters, List<Node> Body, int Line) : Node(Line);

internal sealed record CallBlock(Call Call, List<(string Name, Expr? Default)> Parameters, List<Node> Body, int Line) : Node(Line);

internal sealed record FilterBlock(Expr Filter, List<Node> Body, int Line) : Node(Line);

internal sealed record With(List<(Expr Target, Expr Value)> Bindings, List<Node> Body, int Line) : Node(Line);

internal sealed record Group(List<Node> Body, int Line) : Node(Line);

internal sealed record Jump(bool Break, int Line) : Node(Line);

internal sealed class JinjaParser(string source)
{
    private readonly List<Token> _t = new JinjaLexer(source).Read();
    private int _i, _depth;

    private Token Here => _t[_i];

    private Token Next() => _t[_i++];

    public List<Node> Parse()
    {
        var body = Body();
        if (Here.Kind != TokenKind.End)
        {
            throw Unexpected();
        }
        return body;
    }

    private JinjaException Unexpected(string? wanted = null) =>
        new($"unexpected {(Here.Kind is TokenKind.Op or TokenKind.Name ? $"'{Here.Text}'" : Here.Kind.ToString())}" + (wanted is null ? "" : $", expected {wanted}"), Here.Line);

    private void Expect(string op)
    {
        if (!Here.Is(op))
        {
            throw Unexpected($"'{op}'");
        }
        _i++;
    }

    private bool Take(string op)
    {
        if (!Here.Is(op))
        {
            return false;
        }
        _i++;
        return true;
    }

    private bool TakeName(string name)
    {
        if (!Here.IsName(name))
        {
            return false;
        }
        _i++;
        return true;
    }

    private string Name() => Here.Kind == TokenKind.Name ? Next().Text : throw Unexpected("a name");

    private void EndTag()
    {
        if (Here.Kind != TokenKind.BlockEnd)
        {
            throw Unexpected("the end of the tag");
        }
        _i++;
    }

    /// <summary>Nodes up to a block tag named one of <paramref name="ends"/>, which is entered (its name is next).</summary>
    private List<Node> Body(params string[] ends)
    {
        if (++_depth > 200)
        {
            throw new JinjaException("the template nests too deeply", Here.Line);
        }
        var nodes = new List<Node>();
        while (Here.Kind != TokenKind.End)
        {
            switch (Here.Kind)
            {
                case TokenKind.Text:
                    nodes.Add(new TextNode(Here.Text, Next().Line));
                    break;
                case TokenKind.VarBegin:
                {
                    int line = Next().Line;
                    nodes.Add(new Output(TupleExpr(), line));
                    if (Here.Kind != TokenKind.VarEnd)
                    {
                        throw Unexpected("'}}'");
                    }
                    _i++;
                    break;
                }
                case TokenKind.BlockBegin:
                    if (_t[_i + 1].Kind == TokenKind.Name && ends.Contains(_t[_i + 1].Text))
                    {
                        _i++;
                        _depth--;
                        return nodes;
                    }
                    _i++;
                    if (Statement() is { } node)
                    {
                        nodes.Add(node);
                    }
                    break;
                default:
                    throw Unexpected();
            }
        }
        if (ends.Length > 0)
        {
            throw new JinjaException($"the template ends before '{ends[^1]}'", Here.Line);
        }
        _depth--;
        return nodes;
    }

    private List<Node> Block(string end)
    {
        EndTag();
        var body = Body(end);
        _i++;   // its name
        if (Here.Kind == TokenKind.Name)
        {
            _i++;   // endblock name, endmacro name
        }
        EndTag();
        return body;
    }

    private Node? Statement()
    {
        int line = Here.Line;
        string name = Name();
        switch (name)
        {
            case "if":
            {
                var branches = new List<(Expr, List<Node>)>();
                var test = TupleExpr();
                while (true)
                {
                    EndTag();
                    branches.Add((test, Body("elif", "else", "endif")));
                    string next = Name();
                    if (next == "elif")
                    {
                        test = TupleExpr();
                        continue;
                    }
                    List<Node>? otherwise = null;
                    if (next == "else")
                    {
                        EndTag();
                        otherwise = Body("endif");
                        _i++;
                    }
                    EndTag();
                    return new If(branches, otherwise, line);
                }
            }
            case "for":
            {
                var target = Target();
                if (!TakeName("in"))
                {
                    throw Unexpected("'in'");
                }
                var from = TupleExpr(conditional: false);
                Expr? filter = TakeName("if") ? Expression() : null;
                bool recursive = TakeName("recursive");
                EndTag();
                var body = Body("else", "endfor");
                List<Node>? otherwise = null;
                if (Name() == "else")
                {
                    EndTag();
                    otherwise = Body("endfor");
                    _i++;
                }
                EndTag();
                return new For(target, from, filter, body, otherwise, recursive, line);
            }
            case "set":
            {
                var target = Target();
                if (Take("="))
                {
                    var value = TupleExpr();
                    EndTag();
                    return new Set(target, value, line);
                }
                Expr? filter = Here.Is("|") ? Filters(null) : null;
                return new SetBlock(target, Block("endset"), filter, line);
            }
            case "macro":
            {
                string macro = Name();
                var parameters = Parameters();
                return new Macro(macro, parameters, Block("endmacro"), line);
            }
            case "call":
            {
                var parameters = Here.Is("(") ? Parameters() : [];
                if (Expression() is not Call call)
                {
                    throw new JinjaException("a call block calls something", line);
                }
                return new CallBlock(call, parameters, Block("endcall"), line);
            }
            case "filter":
            {
                var filter = Filters(null, bare: true);
                return new FilterBlock(filter!, Block("endfilter"), line);
            }
            case "with":
            {
                var bindings = new List<(Expr, Expr)>();
                while (Here.Kind != TokenKind.BlockEnd)
                {
                    var target = Target(single: true);
                    Expect("=");
                    bindings.Add((target, Expression()));
                    Take(",");
                }
                return new With(bindings, Block("endwith"), line);
            }
            case "generation":
                return new Group(Block("endgeneration"), line);
            case "block":
                Name();
                return new Group(Block("endblock"), line);
            case "autoescape":
                Expression();
                return new Group(Block("endautoescape"), line);
            case "break" or "continue":
                EndTag();
                return new Jump(name == "break", line);
            case "do":
            {
                var value = TupleExpr();
                EndTag();
                return new Set(new NameRef("", line), value, line);
            }
            default:
                throw new JinjaException($"'{name}' is not a statement this reads", line);
        }
    }

    private List<(string, Expr?)> Parameters()
    {
        Expect("(");
        var list = new List<(string, Expr?)>();
        while (!Take(")"))
        {
            string name = Name();
            list.Add((name, Take("=") ? Expression() : null));
            if (!Here.Is(")"))
            {
                Expect(",");
            }
        }
        return list;
    }

    /// <summary>What a loop or an assignment binds: names, an attribute of a namespace, or a tuple of them.</summary>
    private Expr Target(bool single = false)
    {
        Expr One()
        {
            if (Take("("))
            {
                var inner = Target();
                Expect(")");
                return inner;
            }
            int line = Here.Line;
            string name = Name();
            if (Take("."))
            {
                return new GetAttr(new NameRef(name, line), Name(), line);
            }
            return new NameRef(name, line);
        }
        var first = One();
        if (single || !Here.Is(","))
        {
            return first;
        }
        var items = new List<Expr> { first };
        while (Take(","))
        {
            if (Here.Is("=") || Here.IsName("in") || Here.Is(")"))
            {
                break;
            }
            items.Add(One());
        }
        return new ListOf(items, true, first.Line);
    }

    // ── expressions ──

    private Expr TupleExpr(bool conditional = true)
    {
        var first = conditional ? Expression() : Or();
        if (!Here.Is(","))
        {
            return first;
        }
        var items = new List<Expr> { first };
        while (Take(","))
        {
            if (Here.Kind is TokenKind.BlockEnd or TokenKind.VarEnd || Here.Is(")"))
            {
                break;
            }
            items.Add(conditional ? Expression() : Or());
        }
        return new ListOf(items, true, first.Line);
    }

    public Expr Expression()
    {
        if (++_depth > 200)
        {
            throw new JinjaException("an expression nests too deeply", Here.Line);
        }
        var value = Or();
        while (Here.IsName("if"))
        {
            int line = Next().Line;
            var test = Or();
            value = new Conditional(test, value, TakeName("else") ? Expression() : null, line);
        }
        _depth--;
        return value;
    }

    private Expr Or()
    {
        var left = And();
        while (Here.IsName("or"))
        {
            int line = Next().Line;
            left = new Binary("or", left, And(), line);
        }
        return left;
    }

    private Expr And()
    {
        var left = Not();
        while (Here.IsName("and"))
        {
            int line = Next().Line;
            left = new Binary("and", left, Not(), line);
        }
        return left;
    }

    private Expr Not()
    {
        if (Here.IsName("not"))
        {
            int line = Next().Line;
            return new Unary("not", Not(), line);
        }
        return Comparison();
    }

    private Expr Comparison()
    {
        var first = Sum();
        List<(string, Expr)>? rest = null;
        while (true)
        {
            string? op = null;
            if (Here.Kind == TokenKind.Op && Here.Text is "==" or "!=" or "<" or "<=" or ">" or ">=")
            {
                op = Next().Text;
            }
            else if (Here.IsName("in"))
            {
                _i++;
                op = "in";
            }
            else if (Here.IsName("not") && _t[_i + 1].IsName("in"))
            {
                _i += 2;
                op = "not in";
            }
            if (op is null)
            {
                return rest is null ? first : new Compare(first, rest, first.Line);
            }
            (rest ??= new()).Add((op, Sum()));
        }
    }

    private Expr Sum()
    {
        var left = Concat();
        while (Here.Is("+") || Here.Is("-"))
        {
            var op = Next();
            left = new Binary(op.Text, left, Concat(), op.Line);
        }
        return left;
    }

    private Expr Concat()
    {
        var left = Product();
        while (Here.Is("~"))
        {
            int line = Next().Line;
            left = new Binary("~", left, Product(), line);
        }
        return left;
    }

    private Expr Product()
    {
        var left = Power();
        while (Here.Is("*") || Here.Is("/") || Here.Is("//") || Here.Is("%"))
        {
            var op = Next();
            left = new Binary(op.Text, left, Power(), op.Line);
        }
        return left;
    }

    private Expr Power()
    {
        var left = UnaryExpr(true);
        while (Here.Is("**"))
        {
            int line = Next().Line;
            left = new Binary("**", left, UnaryExpr(true), line);
        }
        return left;
    }

    private Expr UnaryExpr(bool filters)
    {
        Expr value;
        if (Here.Is("-") || Here.Is("+"))
        {
            var op = Next();
            value = new Unary(op.Text, UnaryExpr(false), op.Line);
        }
        else
        {
            value = Postfix(Primary());
        }
        return filters ? Filters(value)! : value;
    }

    private Expr Primary()
    {
        var t = Next();
        switch (t.Kind)
        {
            case TokenKind.Name:
                return t.Text switch
                {
                    "true" or "True" => new Const(true, t.Line),
                    "false" or "False" => new Const(false, t.Line),
                    "none" or "None" => new Const(null, t.Line),
                    _ => new NameRef(t.Text, t.Line),
                };
            case TokenKind.String:
            {
                string s = t.Text;
                while (Here.Kind == TokenKind.String)
                {
                    s += Next().Text;
                }
                return new Const(s, t.Line);
            }
            case TokenKind.Integer:
                return new Const(long.Parse(t.Text, CultureInfo.InvariantCulture), t.Line);
            case TokenKind.Float:
                return new Const(double.Parse(t.Text, CultureInfo.InvariantCulture), t.Line);
            case TokenKind.Op when t.Text == "(":
            {
                if (Take(")"))
                {
                    return new ListOf([], true, t.Line);
                }
                var inner = TupleExpr();
                Expect(")");
                return inner;
            }
            case TokenKind.Op when t.Text == "[":
            {
                var items = new List<Expr>();
                while (!Take("]"))
                {
                    items.Add(Expression());
                    if (!Here.Is("]"))
                    {
                        Expect(",");
                    }
                }
                return new ListOf(items, false, t.Line);
            }
            case TokenKind.Op when t.Text == "{":
            {
                var items = new List<(Expr, Expr)>();
                while (!Take("}"))
                {
                    var key = Expression();
                    Expect(":");
                    items.Add((key, Expression()));
                    if (!Here.Is("}"))
                    {
                        Expect(",");
                    }
                }
                return new DictOf(items, t.Line);
            }
            default:
                _i--;
                throw Unexpected("a value");
        }
    }

    private Expr Postfix(Expr value)
    {
        while (true)
        {
            int line = Here.Line;
            if (Take("."))
            {
                if (Here.Kind == TokenKind.Integer)
                {
                    value = new GetItem(value, new Const(long.Parse(Next().Text, CultureInfo.InvariantCulture), line), line);
                    continue;
                }
                value = new GetAttr(value, Name(), line);
            }
            else if (Take("["))
            {
                value = new GetItem(value, Subscript(), line);
                Expect("]");
            }
            else if (Here.Is("("))
            {
                var (args, named) = Arguments();
                value = new Call(value, args, named, line);
            }
            else
            {
                return value;
            }
        }
    }

    private Expr Subscript()
    {
        int line = Here.Line;
        Expr? from = Here.Is(":") ? null : Expression();
        if (!Take(":"))
        {
            return from!;
        }
        Expr? to = Here.Is(":") || Here.Is("]") ? null : Expression();
        Expr? step = Take(":") && !Here.Is("]") ? Expression() : null;
        return new Slice(from, to, step, line);
    }

    private (List<Expr>, List<(string, Expr)>) Arguments()
    {
        Expect("(");
        var args = new List<Expr>();
        var named = new List<(string, Expr)>();
        while (!Take(")"))
        {
            if (Here.Is("*") || Here.Is("**"))
            {
                throw new JinjaException("a call that spreads its arguments is not read", Here.Line);
            }
            if (Here.Kind == TokenKind.Name && _t[_i + 1].Is("="))
            {
                string name = Next().Text;
                _i++;
                named.Add((name, Expression()));
            }
            else
            {
                args.Add(Expression());
            }
            if (!Here.Is(")"))
            {
                Expect(",");
            }
        }
        return (args, named);
    }

    private Expr? Filters(Expr? value, bool bare = false)
    {
        while (true)
        {
            int line = Here.Line;
            if (bare || Take("|"))
            {
                bare = false;
                string name = Name();
                while (Take("."))
                {
                    name += "." + Name();
                }
                var (args, named) = Here.Is("(") ? Arguments() : ([], []);
                value = new FilterUse(value, name, args, named, line);
            }
            else if (value is not null && TakeName("is"))
            {
                bool negated = TakeName("not");
                string name = Name();
                var args = new List<Expr>();
                if (Here.Is("("))
                {
                    args = Arguments().Item1;
                }
                else if (Here.Kind is TokenKind.Name or TokenKind.String or TokenKind.Integer or TokenKind.Float && Here.Text is not ("else" or "or" or "and" or "is" or "if" or "not" or "in")
                         || Here.Is("[") || Here.Is("{"))
                {
                    args.Add(Postfix(Primary()));
                }
                value = new TestUse(value, name, args, negated, line);
            }
            else if (value is not null && Here.Is("("))
            {
                var (args, named) = Arguments();
                value = Postfix(new Call(value, args, named, line));
            }
            else
            {
                return value;
            }
        }
    }
}
