using System.Globalization;

namespace Amql.Entail;

/// <summary>A source error, with the line it came from.</summary>
public sealed class EntailException : Exception
{
    public EntailException(string message) : base(message) { }
    public EntailException(int line, string message) : base($"line {line}: {message}") { }
}

internal enum Tok { Id, Num, Str, Sym, Newline, Eof }

internal readonly record struct Token(Tok Kind, string Text, int Line);

/// <summary>Splits Entail source into tokens. Identifiers may carry
/// <c>{param}</c> templates, which expand when a <c>def</c> is called.</summary>
internal static class EntailLexer
{
    private static readonly string[] Symbols =
    {
        "<-", ">=", "<=", "==", "!=", "+", "-", "*", "/", "^", "(", ")", "[", "]", ",", ":", "=", "<", ">",
    };

    public static List<Token> Lex(string source)
    {
        var tokens = new List<Token>();
        int line = 1, i = 0;
        while (i < source.Length)
        {
            char c = source[i];
            if (c == '\n')
            {
                tokens.Add(new Token(Tok.Newline, "\n", line++));
                i++;
            }
            else if (char.IsWhiteSpace(c))
            {
                i++;
            }
            else if (c == '#')
            {
                while (i < source.Length && source[i] != '\n')
                {
                    i++;
                }
            }
            else if (c == '"')
            {
                int end = source.IndexOf('"', i + 1);
                if (end < 0)
                {
                    throw new EntailException(line, "unterminated string");
                }
                tokens.Add(new Token(Tok.Str, source[(i + 1)..end], line));
                i = end + 1;
            }
            else if (char.IsDigit(c))
            {
                int start = i;
                while (i < source.Length && (char.IsDigit(source[i]) ||
                       (source[i] == '.' && i + 1 < source.Length && char.IsDigit(source[i + 1]))))
                {
                    i++;
                }
                tokens.Add(new Token(Tok.Num, source[start..i], line));
            }
            else if (char.IsLetter(c) || c == '_' || c == '{')
            {
                int start = i;
                while (i < source.Length && (char.IsLetterOrDigit(source[i]) || source[i] is '_' or '{' or '}'))
                {
                    i++;
                }
                tokens.Add(new Token(Tok.Id, source[start..i], line));
            }
            else
            {
                int at = i;
                var symbol = Symbols.FirstOrDefault(s => string.CompareOrdinal(source, at, s, 0, s.Length) == 0)
                    ?? throw new EntailException(line, $"unexpected character '{c}'");
                tokens.Add(new Token(Tok.Sym, symbol, line));
                i += symbol.Length;
            }
        }
        tokens.Add(new Token(Tok.Newline, "\n", line));
        tokens.Add(new Token(Tok.Eof, string.Empty, line));
        return tokens;
    }
}

// ── syntax tree ─────────────────────────────────────────────────────────────

internal abstract record Expr(int Line);
internal sealed record NumExpr(double Value, int Line) : Expr(Line);
internal sealed record RefExpr(string Name, IReadOnlyList<Expr> Index, int Line) : Expr(Line);
internal sealed record BinExpr(char Op, Expr Left, Expr Right, int Line) : Expr(Line);
internal sealed record NegExpr(Expr Inner, int Line) : Expr(Line);
internal sealed record SumExpr(string Var, Expr From, Expr To, Expr Body, int Line) : Expr(Line);
internal sealed record MinExpr(Expr Left, Expr Right, int Line) : Expr(Line);
internal sealed record FloorExpr(Expr Inner, int Line) : Expr(Line);

/// <summary>One item of a <c>when</c> list: a fact, its negation, or a
/// threshold on a whole-number sum.</summary>
internal sealed record Cond(Expr Value, string? Compare, Expr? Bound, bool Negated, int Line);

internal abstract record Stmt(int Line);
internal sealed record ProgramStmt(string Name, int Line) : Stmt(Line);
internal sealed record ConstStmt(string Name, Expr Value, int Line) : Stmt(Line);
internal sealed record StateDecl(string Name, IReadOnlyList<Expr> Dims, Expr? Scale);
internal sealed record StateStmt(IReadOnlyList<StateDecl> Decls, int Line) : Stmt(Line);
internal sealed record LetStmt(RefExpr Target, Expr Value, int Line) : Stmt(Line);
internal sealed record TokenStmt(IReadOnlyList<string> Texts, IReadOnlyList<(RefExpr Lane, Expr Value)> Means, int Line) : Stmt(Line);
internal sealed record ForStmt(string Var, Expr From, Expr To, IReadOnlyList<Stmt> Body, int Line) : Stmt(Line);
internal sealed record IfStmt(Expr Left, string Compare, Expr Right, IReadOnlyList<Stmt> Then, IReadOnlyList<Stmt> Else, int Line) : Stmt(Line);
internal sealed record DefStmt(string Name, IReadOnlyList<string> Params, IReadOnlyList<Stmt> Body, int Line) : Stmt(Line);
internal sealed record CallStmt(string Name, IReadOnlyList<Expr> Args, int Line) : Stmt(Line);
internal sealed record AttendStmt(Expr? Offset, string? Pointer, IReadOnlyList<Stmt> Body, int Line) : Stmt(Line);
internal sealed record CopyStmt(RefExpr Target, Expr Source, int Line) : Stmt(Line);
internal sealed record EntailStmt(RefExpr Target, Expr? Value, IReadOnlyList<Cond> When, int Line) : Stmt(Line);
internal sealed record EmitStmt(string? Text, RefExpr Lane, int Line) : Stmt(Line);

/// <summary>Recursive-descent parser for the statement forms of Entail.</summary>
internal sealed class EntailParser
{
    private readonly List<Token> _tokens;
    private int _at;

    private EntailParser(List<Token> tokens) => _tokens = tokens;

    public static IReadOnlyList<Stmt> Parse(string source) =>
        new EntailParser(EntailLexer.Lex(source)).Block(topLevel: true);

    private Token Peek => _tokens[_at];
    private Token Next() => _tokens[_at++];
    private bool IsSym(string s) => Peek.Kind == Tok.Sym && Peek.Text == s;
    private bool IsWord(string s) => Peek.Kind == Tok.Id && Peek.Text == s;

    private bool AcceptSym(string s)
    {
        if (IsSym(s))
        {
            _at++;
            return true;
        }
        return false;
    }

    private bool AcceptWord(string s)
    {
        if (IsWord(s))
        {
            _at++;
            return true;
        }
        return false;
    }

    private void ExpectSym(string s)
    {
        if (!AcceptSym(s))
        {
            throw new EntailException(Peek.Line, $"expected '{s}', found '{Peek.Text.Trim()}'");
        }
    }

    private void ExpectWord(string s)
    {
        if (!AcceptWord(s))
        {
            throw new EntailException(Peek.Line, $"expected '{s}', found '{Peek.Text.Trim()}'");
        }
    }

    private string Identifier()
    {
        if (Peek.Kind != Tok.Id)
        {
            throw new EntailException(Peek.Line, $"expected a name, found '{Peek.Text.Trim()}'");
        }
        return Next().Text;
    }

    private void EndOfLine()
    {
        if (Peek.Kind != Tok.Newline && Peek.Kind != Tok.Eof)
        {
            throw new EntailException(Peek.Line, $"unexpected '{Peek.Text}' at the end of a statement");
        }
    }

    private void SkipNewlines()
    {
        while (Peek.Kind == Tok.Newline)
        {
            _at++;
        }
    }

    private List<Stmt> Block(bool topLevel = false, bool stopAtElse = false)
    {
        var body = new List<Stmt>();
        while (true)
        {
            SkipNewlines();
            if (Peek.Kind == Tok.Eof)
            {
                if (!topLevel)
                {
                    throw new EntailException(Peek.Line, "missing 'end'");
                }
                return body;
            }
            if (IsWord("end") && !topLevel)
            {
                return body;
            }
            if (stopAtElse && IsWord("else"))
            {
                return body;
            }
            body.Add(Statement());
        }
    }

    private Stmt Statement()
    {
        int line = Peek.Line;
        if (AcceptWord("program"))
        {
            var name = Identifier();
            EndOfLine();
            return new ProgramStmt(name, line);
        }
        if (AcceptWord("const"))
        {
            var name = Identifier();
            ExpectSym("=");
            var value = Expression();
            EndOfLine();
            return new ConstStmt(name, value, line);
        }
        if (AcceptWord("state"))
        {
            var decls = new List<StateDecl>();
            do
            {
                SkipNewlines();
                var name = Identifier();
                var dims = new List<Expr>();
                while (AcceptSym("["))
                {
                    dims.Add(Expression());
                    ExpectSym("]");
                }
                Expr? scale = AcceptWord("scale") ? Expression() : null;
                decls.Add(new StateDecl(name, dims, scale));
            }
            while (AcceptSym(","));
            EndOfLine();
            return new StateStmt(decls, line);
        }
        if (AcceptWord("let"))
        {
            var target = Reference();
            ExpectSym("=");
            var value = Expression();
            EndOfLine();
            return new LetStmt(target, value, line);
        }
        if (AcceptWord("token"))
        {
            var texts = new List<string>();
            do
            {
                SkipNewlines();
                if (Peek.Kind != Tok.Str)
                {
                    throw new EntailException(line, "token expects a quoted string");
                }
                texts.Add(Next().Text);
            }
            while (AcceptSym(","));
            ExpectWord("means");
            var means = new List<(RefExpr, Expr)>();
            do
            {
                var lane = Reference();
                ExpectSym("=");
                means.Add((lane, Expression()));
            }
            while (AcceptSym(","));
            EndOfLine();
            return new TokenStmt(texts, means, line);
        }
        if (AcceptWord("for"))
        {
            var name = Identifier();
            ExpectWord("in");
            var from = Expression();
            ExpectWord("to");
            var to = Expression();
            EndOfLine();
            var body = Block();
            ExpectWord("end");
            return new ForStmt(name, from, to, body, line);
        }
        if (AcceptWord("if"))
        {
            var left = Expression();
            if (Peek.Kind != Tok.Sym || Peek.Text is not ("<" or ">" or "<=" or ">=" or "==" or "!="))
            {
                throw new EntailException(line, "if expects a comparison");
            }
            var compare = Next().Text;
            var right = Expression();
            EndOfLine();
            var then = Block(stopAtElse: true);
            var otherwise = new List<Stmt>();
            if (AcceptWord("else"))
            {
                otherwise = Block();
            }
            ExpectWord("end");
            return new IfStmt(left, compare, right, then, otherwise, line);
        }
        if (AcceptWord("def"))
        {
            var name = Identifier();
            ExpectSym("(");
            var parameters = new List<string>();
            if (!IsSym(")"))
            {
                do
                {
                    parameters.Add(Identifier());
                }
                while (AcceptSym(","));
            }
            ExpectSym(")");
            EndOfLine();
            var body = Block();
            ExpectWord("end");
            return new DefStmt(name, parameters, body, line);
        }
        if (AcceptWord("call"))
        {
            var name = Identifier();
            ExpectSym("(");
            var args = new List<Expr>();
            if (!IsSym(")"))
            {
                do
                {
                    args.Add(Expression());
                }
                while (AcceptSym(","));
            }
            ExpectSym(")");
            EndOfLine();
            return new CallStmt(name, args, line);
        }
        if (AcceptWord("attend"))
        {
            ExpectWord("back");
            Expr? offset = null;
            string? pointer = null;
            if (AcceptWord("by"))
            {
                pointer = Identifier();
            }
            else
            {
                offset = Expression();
            }
            EndOfLine();
            var body = Block();
            ExpectWord("end");
            return new AttendStmt(offset, pointer, body, line);
        }
        if (AcceptWord("entail"))
        {
            var target = Reference();
            Expr? value = AcceptSym("=") ? Expression() : null;
            var when = new List<Cond>();
            if (AcceptWord("when"))
            {
                do
                {
                    SkipNewlines();
                    when.Add(Condition());
                }
                while (AcceptSym(","));
            }
            EndOfLine();
            if (value is null && when.Count == 0)
            {
                throw new EntailException(line, "entail needs a value or a 'when' list");
            }
            return new EntailStmt(target, value, when, line);
        }
        if (AcceptWord("emit"))
        {
            string? text = null;
            if (!AcceptWord("end"))
            {
                if (Peek.Kind != Tok.Str)
                {
                    throw new EntailException(line, "emit expects a quoted token or 'end'");
                }
                text = Next().Text;
            }
            ExpectWord("when");
            var lane = Reference();
            EndOfLine();
            return new EmitStmt(text, lane, line);
        }

        // Inside an attend block: dst <- source.
        var dst = Reference();
        ExpectSym("<-");
        var source = Expression();
        EndOfLine();
        return new CopyStmt(dst, source, line);
    }

    private Cond Condition()
    {
        int line = Peek.Line;
        bool negated = AcceptWord("not");
        var value = Expression();
        if (Peek.Kind == Tok.Sym && Peek.Text is ">=" or "<=" or "==" or "<" or ">")
        {
            var compare = Next().Text;
            var bound = Expression();
            return new Cond(value, compare, bound, negated, line);
        }
        return new Cond(value, null, null, negated, line);
    }

    private RefExpr Reference()
    {
        int line = Peek.Line;
        var name = Identifier();
        var index = new List<Expr>();
        while (AcceptSym("["))
        {
            index.Add(Expression());
            ExpectSym("]");
        }
        return new RefExpr(name, index, line);
    }

    private Expr Expression()
    {
        var left = Product();
        while (IsSym("+") || IsSym("-"))
        {
            var op = Next();
            left = new BinExpr(op.Text[0], left, Product(), op.Line);
        }
        return left;
    }

    private Expr Product()
    {
        var left = Unary();
        while (IsSym("*") || IsSym("/"))
        {
            var op = Next();
            left = new BinExpr(op.Text[0], left, Unary(), op.Line);
        }
        return left;
    }

    private Expr Unary()
    {
        if (IsSym("-"))
        {
            int line = Next().Line;
            return new NegExpr(Unary(), line);
        }
        return Power();
    }

    private Expr Power()
    {
        var left = Atom();
        if (IsSym("^"))
        {
            var op = Next();
            return new BinExpr('^', left, Unary(), op.Line);
        }
        return left;
    }

    private Expr Atom()
    {
        int line = Peek.Line;
        if (Peek.Kind == Tok.Num)
        {
            return new NumExpr(double.Parse(Next().Text, CultureInfo.InvariantCulture), line);
        }
        if (AcceptSym("("))
        {
            if (AcceptWord("sum"))
            {
                var name = Identifier();
                ExpectWord("in");
                var from = Expression();
                ExpectWord("to");
                var to = Expression();
                ExpectSym(":");
                var body = Expression();
                ExpectSym(")");
                return new SumExpr(name, from, to, body, line);
            }
            var inner = Expression();
            ExpectSym(")");
            return inner;
        }
        if (IsWord("min"))
        {
            Next();
            ExpectSym("(");
            var left = Expression();
            ExpectSym(",");
            var right = Expression();
            ExpectSym(")");
            return new MinExpr(left, right, line);
        }
        if (IsWord("floor"))
        {
            Next();
            ExpectSym("(");
            var inner = Expression();
            ExpectSym(")");
            return new FloorExpr(inner, line);
        }
        return Reference();
    }
}
