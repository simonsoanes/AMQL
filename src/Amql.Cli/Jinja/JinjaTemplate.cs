using System.Globalization;
using System.Text;
using System.Text.Json;

namespace Amql.Cli.Jinja;

/// <summary>
/// A Jinja template, rendered as transformers renders chat templates: the same whitespace rules, filters,
/// tests and globals (<c>raise_exception</c>, <c>strftime_now</c>, <c>tojson</c> without HTML escapes,
/// <c>generation</c> blocks, loop controls), and nothing that reaches outside the template. Templates come
/// with models, so how much one may do is bounded.
/// </summary>
public sealed class JinjaTemplate
{
    private readonly List<Node> _body;

    public string Source { get; }

    private JinjaTemplate(string source)
    {
        Source = source;
        _body = new JinjaParser(source).Parse();
    }

    /// <exception cref="JinjaException">The template cannot be read.</exception>
    public static JinjaTemplate Parse(string source) => new(source);

    /// <summary>The names the template reads and does not set itself: what it can be given.</summary>
    public IReadOnlySet<string> Reads => _reads ??= Names();

    private HashSet<string>? _reads;

    /// <summary>The template's text for these variables: JSON values and .NET's own (null, bool, numbers, strings).</summary>
    /// <exception cref="JinjaException">It failed, or raised an exception of its own.</exception>
    public string Render(IReadOnlyDictionary<string, object?> variables, DateTime? now = null)
    {
        var scope = new Scope(null);
        foreach (var (name, value) in variables)
        {
            scope.Vars[name] = value switch
            {
                JsonElement e => Py.FromJson(e),
                int i => (long)i,
                float f => (double)f,
                _ => value,
            };
        }
        var run = new Run(now ?? DateTime.Now);
        try
        {
            run.Nodes(_body, scope);
        }
        catch (InsufficientExecutionStackException)
        {
            throw new JinjaException("the template recurses too deeply");
        }
        return run.Output;
    }

    private HashSet<string> Names()
    {
        var read = new HashSet<string>(StringComparer.Ordinal);
        var set = new HashSet<string>(StringComparer.Ordinal) { "loop", "caller", "varargs", "kwargs" };
        void Visit(object? o)
        {
            switch (o)
            {
                case null or string or bool or int or TokenKind:
                    return;
                case NameRef n:
                    read.Add(n.Name);
                    return;
                case Set s:
                    Bind(s.Target);
                    Visit(s.Value);
                    return;
                case SetBlock s:
                    Bind(s.Target);
                    break;
                case For f:
                    Bind(f.Target);
                    break;
                case Macro m:
                    set.Add(m.Name);
                    m.Parameters.ForEach(p => set.Add(p.Name));
                    break;
                case With w:
                    w.Bindings.ForEach(b => Bind(b.Target));
                    break;
            }
            if (o is System.Collections.IEnumerable list)
            {
                foreach (var x in list)
                {
                    Visit(x);
                }
                return;
            }
            if (o is Node || o is System.Runtime.CompilerServices.ITuple)
            {
                if (o is System.Runtime.CompilerServices.ITuple tuple and not Node)
                {
                    for (int i = 0; i < tuple.Length; i++)
                    {
                        Visit(tuple[i]);
                    }
                    return;
                }
                if (o is Const)
                {
                    return;
                }
                foreach (var p in o.GetType().GetProperties())
                {
                    if (p.Name is not "EqualityContract" and not "Line")
                    {
                        Visit(p.GetValue(o));
                    }
                }
            }
        }
        void Bind(Expr target)
        {
            if (target is NameRef n)
            {
                set.Add(n.Name);
            }
            else if (target is ListOf l)
            {
                l.Items.ForEach(Bind);
            }
        }
        Visit(_body);
        read.ExceptWith(set);
        return read;
    }

    // ── rendering ─────────────────────────────────────────────────────────

    private sealed class Scope(Scope? parent)
    {
        public readonly Dictionary<string, object?> Vars = new(StringComparer.Ordinal);
        public readonly Scope? Parent = parent;

        public bool TryGet(string name, out object? value)
        {
            for (var s = this; s is not null; s = s.Parent)
            {
                if (s.Vars.TryGetValue(name, out value))
                {
                    return true;
                }
            }
            value = null;
            return false;
        }
    }

    private sealed class JumpException(bool isBreak) : Exception
    {
        public bool Break { get; } = isBreak;
    }

    private sealed class Loop
    {
        public List<object?> Items = [];
        public int Index, Depth;
        public Callable? Again;
        public object? Last;
        public bool Compared;
    }

    private sealed class Run(DateTime now)
    {
        private const long MostSteps = 20_000_000;
        private const int MostText = 1 << 28;

        private StringBuilder _out = new();
        private long _steps;

        public string Output => _out.ToString();

        private void Step(int line)
        {
            if (++_steps > MostSteps)
            {
                throw new JinjaException("the template does too much", line);
            }
        }

        private void Write(string text, int line)
        {
            if (_out.Length + text.Length > MostText)
            {
                throw new JinjaException("the template writes too much", line);
            }
            _out.Append(text);
        }

        private string Capture(List<Node> body, Scope scope)
        {
            var outer = _out;
            _out = new StringBuilder();
            try
            {
                Nodes(body, scope);
                return _out.ToString();
            }
            finally
            {
                _out = outer;
            }
        }

        public void Nodes(List<Node> nodes, Scope scope)
        {
            System.Runtime.CompilerServices.RuntimeHelpers.EnsureSufficientExecutionStack();
            foreach (var node in nodes)
            {
                Step(node.Line);
                switch (node)
                {
                    case TextNode t:
                        Write(t.Text, t.Line);
                        break;
                    case Output o:
                        Write(Py.Str(Eval(o.Value, scope)), o.Line);
                        break;
                    case If f:
                    {
                        var branch = f.Branches.FirstOrDefault(b => Py.Truthy(Eval(b.Test, scope)));
                        if ((branch.Body ?? f.Else) is { } body)
                        {
                            Nodes(body, scope);
                        }
                        break;
                    }
                    case For f:
                        ForLoop(f, scope, Eval(f.Source, scope), 1);
                        break;
                    case Set s:
                        Assign(s.Target, Eval(s.Value, scope), scope);
                        break;
                    case SetBlock s:
                    {
                        object? text = Capture(s.Body, new Scope(scope));
                        Assign(s.Target, s.Filter is null ? text : Chain(s.Filter, text, scope), scope);
                        break;
                    }
                    case Macro m:
                        scope.Vars[m.Name] = MacroOf(m.Name, m.Parameters, m.Body, scope);
                        break;
                    case CallBlock c:
                    {
                        var named = c.Call.Named.ToDictionary(n => n.Name, n => Eval(n.Value, scope), StringComparer.Ordinal);
                        named["caller"] = MacroOf("caller", c.Parameters, c.Body, scope);
                        Write(Py.Str(Invoke(Eval(c.Call.Target, scope), c.Call.Args.Select(a => Eval(a, scope)).ToList(), named, c.Line)), c.Line);
                        break;
                    }
                    case FilterBlock b:
                        Write(Py.Str(Chain(b.Filter, Capture(b.Body, new Scope(scope)), scope)), b.Line);
                        break;
                    case With w:
                    {
                        var inner = new Scope(scope);
                        foreach (var (target, value) in w.Bindings)
                        {
                            Assign(target, Eval(value, inner), inner);
                        }
                        Nodes(w.Body, inner);
                        break;
                    }
                    case Group g:
                        Nodes(g.Body, scope);
                        break;
                    case Jump j:
                        throw new JumpException(j.Break);
                }
            }
        }

        private void ForLoop(For f, Scope scope, object? source, int depth)
        {
            if (source is Undefined)
            {
                source = new JList([]);
            }
            var items = Py.Iterate(source).ToList();
            if (f.Filter is not null)
            {
                items = items.Where(item =>
                {
                    var test = new Scope(scope);
                    Assign(f.Target, item, test);
                    return Py.Truthy(Eval(f.Filter, test));
                }).ToList();
            }
            if (items.Count == 0)
            {
                if (f.Else is not null)
                {
                    Nodes(f.Else, scope);
                }
                return;
            }
            var loop = new Loop { Items = items, Depth = depth };
            loop.Again = f.Recursive
                ? (args, _) =>
                {
                    var outer = _out;
                    _out = new StringBuilder();
                    try
                    {
                        ForLoop(f, scope, args[0], depth + 1);
                        return _out.ToString();
                    }
                    finally
                    {
                        _out = outer;
                    }
                }
            : null;
            for (int i = 0; i < items.Count; i++)
            {
                loop.Index = i;
                var inner = new Scope(scope);
                inner.Vars["loop"] = loop;
                Assign(f.Target, items[i], inner);
                try
                {
                    Nodes(f.Body, inner);
                }
                catch (JumpException j)
                {
                    if (j.Break)
                    {
                        break;
                    }
                }
            }
        }

        private static void Assign(Expr target, object? value, Scope scope)
        {
            switch (target)
            {
                case NameRef { Name: "" }:
                    return;
                case NameRef n:
                    scope.Vars[n.Name] = value;
                    return;
                case GetAttr { Target: NameRef owner } a:
                    if (!scope.TryGet(owner.Name, out var holder) || holder is not JNamespace ns)
                    {
                        throw new JinjaException($"'{owner.Name}' is no namespace to assign into", a.Line);
                    }
                    ns.Members[a.Name] = value;
                    return;
                case ListOf l:
                {
                    var items = Py.Iterate(value).ToList();
                    if (items.Count != l.Items.Count)
                    {
                        throw new JinjaException($"{items.Count} values cannot be taken as {l.Items.Count}", l.Line);
                    }
                    for (int i = 0; i < items.Count; i++)
                    {
                        Assign(l.Items[i], items[i], scope);
                    }
                    return;
                }
            }
        }

        private Callable MacroOf(string name, List<(string Name, Expr? Default)> parameters, List<Node> body, Scope defined) => (args, named) =>
        {
            System.Runtime.CompilerServices.RuntimeHelpers.EnsureSufficientExecutionStack();
            var scope = new Scope(defined);
            for (int i = 0; i < parameters.Count; i++)
            {
                var (p, fallback) = parameters[i];
                scope.Vars[p] = i < args.Count ? args[i]
                    : named.TryGetValue(p, out var given) ? given
                    : fallback is not null ? Eval(fallback, scope) : new Undefined(p);
            }
            scope.Vars["varargs"] = new JList(args.Skip(parameters.Count).ToList(), true);
            var rest = new JDict();
            foreach (var (k, v) in named.Where(n => n.Key != "caller" && parameters.All(p => p.Name != n.Key)))
            {
                rest.Set(k, v);
            }
            scope.Vars["kwargs"] = rest;
            if (named.TryGetValue("caller", out var caller))
            {
                scope.Vars["caller"] = caller;
            }
            return Capture(body, scope);
        };

        // ── expressions ──

        private object? Eval(Expr e, Scope scope)
        {
            Step(e.Line);
            switch (e)
            {
                case Const c:
                    return c.Value;
                case NameRef n:
                    return scope.TryGet(n.Name, out var v) ? v : Global(n.Name) ?? new Undefined(n.Name);
                case GetAttr a:
                    return Attribute(Eval(a.Target, scope), a.Name, a.Line);
                case GetItem { Index: Slice s } g:
                    return At(g.Line, () => Py.Slice(Eval(g.Target, scope), s.From is null ? null : Eval(s.From, scope),
                        s.To is null ? null : Eval(s.To, scope), s.Step is null ? null : Eval(s.Step, scope)));
                case GetItem g:
                    return Item(Eval(g.Target, scope), Eval(g.Index, scope), g.Line);
                case Call c:
                    return Invoke(Eval(c.Target, scope), c.Args.Select(a => Eval(a, scope)).ToList(),
                        c.Named.ToDictionary(x => x.Name, x => Eval(x.Value, scope), StringComparer.Ordinal), c.Line);
                case FilterUse f:
                    return Filtered(f, Eval(f.Target!, scope), scope);
                case TestUse t:
                    return At(t.Line, () => Builtins.Test(t.Name, Eval(t.Target, scope), t.Args.Select(a => Eval(a, scope)).ToList())) != t.Negated;
                case Unary u:
                {
                    var value = Eval(u.Value, scope);
                    return u.Op switch
                    {
                        "not" => !Py.Truthy(value),
                        "-" => At<object?>(u.Line, () => value is double d ? (object)-d : -Py.Int(value)),
                        _ => value,
                    };
                }
                case Binary { Op: "or" } b:
                {
                    var left = Eval(b.Left, scope);
                    return Py.Truthy(left) ? left : Eval(b.Right, scope);
                }
                case Binary { Op: "and" } b:
                {
                    var left = Eval(b.Left, scope);
                    return Py.Truthy(left) ? Eval(b.Right, scope) : left;
                }
                case Binary b:
                    return At(b.Line, () => Operate(b.Op, Eval(b.Left, scope), Eval(b.Right, scope)));
                case Compare c:
                {
                    var left = Eval(c.First, scope);
                    foreach (var (op, right) in c.Rest)
                    {
                        var r = Eval(right, scope);
                        var l = left;
                        if (!At(c.Line, () => Holds(op, l, r)))
                        {
                            return false;
                        }
                        left = r;
                    }
                    return true;
                }
                case Conditional c:
                    return Py.Truthy(Eval(c.Test, scope)) ? Eval(c.Then, scope) : c.Else is null ? new Undefined("") : Eval(c.Else, scope);
                case ListOf l:
                    return new JList(l.Items.Select(i => Eval(i, scope)).ToList(), l.Tuple);
                case DictOf d:
                {
                    var dict = new JDict();
                    foreach (var (k, value) in d.Items)
                    {
                        dict.Set(Eval(k, scope), Eval(value, scope));
                    }
                    return dict;
                }
                default:
                    throw new JinjaException("an expression of a kind this does not read", e.Line);
            }
        }

        /// <summary>Runs something, with the template's line on what it throws.</summary>
        private static T At<T>(int line, Func<T> work)
        {
            try
            {
                return work();
            }
            catch (JinjaException e) when (!e.Message.EndsWith("of the template)"))
            {
                throw new JinjaException(e.Message, line);
            }
        }

        private static bool Holds(string op, object? a, object? b) => op switch
        {
            "==" => Py.AreEqual(a, b),
            "!=" => !Py.AreEqual(a, b),
            "in" => Py.Contains(b, a),
            "not in" => !Py.Contains(b, a),
            "<" => Py.CompareValues(a, b) < 0,
            "<=" => Py.CompareValues(a, b) <= 0,
            ">" => Py.CompareValues(a, b) > 0,
            _ => Py.CompareValues(a, b) >= 0,
        };

        private static object? Operate(string op, object? a, object? b)
        {
            if (op == "~")
            {
                return Py.Str(a) + Py.Str(b);
            }
            if (op == "+")
            {
                if (a is string x && b is string y)
                {
                    return x + y;
                }
                if (a is JList l && b is JList m && l.Tuple == m.Tuple)
                {
                    return new JList([.. l.Items, .. m.Items], l.Tuple);
                }
            }
            if (op == "*" && ((a is string or JList && b is long) || (a is long && b is string or JList)))
            {
                (object? many, long times) = a is long t ? (b, t) : (a, (long)b!);
                times = Math.Clamp(times, 0, 1 << 20);
                return many is string s ? string.Concat(Enumerable.Repeat(s, (int)times))
                    : new JList(Enumerable.Repeat(((JList)many!).Items, (int)times).SelectMany(i => i).ToList(), ((JList)many!).Tuple);
            }
            if (op == "%" && a is string format)
            {
                return Builtins.Percent(format, b);
            }
            if (!Py.IsNumber(a) || !Py.IsNumber(b))
            {
                throw new JinjaException(a is Undefined or null || b is Undefined or null
                    ? $"'{op}' of {Py.TypeName(a)} and {Py.TypeName(b)}"
                    : $"'{op}' does not join {Py.TypeName(a)} and {Py.TypeName(b)}");
            }
            if (a is double || b is double || op == "/")
            {
                double p = Py.Num(a), q = Py.Num(b);
                if (op is "/" or "//" or "%" && q == 0)
                {
                    throw new JinjaException("division by zero");
                }
                return op switch
                {
                    "+" => p + q,
                    "-" => p - q,
                    "*" => p * q,
                    "/" => p / q,
                    "//" => Math.Floor(p / q),
                    "%" => p - q * Math.Floor(p / q),
                    _ => Math.Pow(p, q),
                };
            }
            long i = Py.Int(a), j = Py.Int(b);
            if (op is "//" or "%" && j == 0)
            {
                throw new JinjaException("division by zero");
            }
            try
            {
                checked
                {
                    switch (op)
                    {
                        case "+": return i + j;
                        case "-": return i - j;
                        case "*": return i * j;
                        case "//": return i / j - (i % j != 0 && (i < 0) != (j < 0) ? 1 : 0);
                        case "%": return i % j is var r && r != 0 && (r < 0) != (j < 0) ? r + j : i % j;
                        default:
                            if (j < 0)
                            {
                                return Math.Pow(i, j);
                            }
                            long power = 1;
                            for (long k = 0; k < j && k < 4096; k++)
                            {
                                power *= i;
                            }
                            return power;
                    }
                }
            }
            catch (OverflowException)
            {
                throw new JinjaException("a number is too large");
            }
        }

        private object? Invoke(object? target, List<object?> args, Dictionary<string, object?> named, int line)
        {
            if (target is Loop { Again: { } again })
            {
                return again(args, named);
            }
            return target is Callable call
                ? At(line, () => call(args, named))
                : throw new JinjaException($"{Py.TypeName(target)} cannot be called", line);
        }

        private object? Attribute(object? owner, string name, int line)
        {
            switch (owner)
            {
                case Undefined u:
                    throw new JinjaException($"'{u.What}' is undefined, so it has no '{name}'", line);
                case JNamespace n:
                    return n.Members.TryGetValue(name, out var member) ? member : new Undefined(name);
                case Loop loop:
                    return LoopMember(loop, name);
            }
            if (Builtins.Method(owner, name) is { } method)
            {
                return method;
            }
            return owner is JDict d && d.TryGet(name, out var v) ? v : new Undefined(name);
        }

        private object? Item(object? owner, object? index, int line)
        {
            switch (owner)
            {
                case Undefined u:
                    throw new JinjaException($"'{u.What}' is undefined, so it has no items", line);
                case JDict d when d.TryGet(index, out var v):
                    return v;
                case JList l when index is long or bool:
                {
                    long i = Py.Int(index);
                    i = i < 0 ? i + l.Items.Count : i;
                    return i >= 0 && i < l.Items.Count ? l.Items[(int)i] : new Undefined($"item {index}");
                }
                case string s when index is long or bool:
                {
                    long i = Py.Int(index), n = Py.Length(s);
                    i = i < 0 ? i + n : i;
                    return i < 0 || i >= n ? new Undefined($"character {index}") : n == s.Length ? s[(int)i].ToString() : Py.Characters(s).ElementAt((int)i);
                }
                case JNamespace:
                    return index is string member ? Attribute(owner, member, line) : new Undefined("item");
            }
            return index is string name && Builtins.Method(owner, name) is { } method ? method : new Undefined(index as string ?? "item");
        }

        private static object? LoopMember(Loop loop, string name) => name switch
        {
            "index" => (long)loop.Index + 1,
            "index0" => (long)loop.Index,
            "revindex" => (long)loop.Items.Count - loop.Index,
            "revindex0" => (long)loop.Items.Count - loop.Index - 1,
            "first" => loop.Index == 0,
            "last" => loop.Index == loop.Items.Count - 1,
            "length" => (long)loop.Items.Count,
            "depth" => (long)loop.Depth,
            "depth0" => (long)loop.Depth - 1,
            "previtem" => loop.Index > 0 ? loop.Items[loop.Index - 1] : new Undefined("previtem"),
            "nextitem" => loop.Index + 1 < loop.Items.Count ? loop.Items[loop.Index + 1] : new Undefined("nextitem"),
            "cycle" => (Callable)((args, _) => args.Count == 0 ? throw new JinjaException("cycle goes through nothing") : args[loop.Index % args.Count]),
            "changed" => (Callable)((args, _) =>
            {
                var value = new JList(args, true);
                bool changed = !loop.Compared || !Py.AreEqual(loop.Last, value);
                (loop.Last, loop.Compared) = (value, true);
                return changed;
            }),
            _ => new Undefined(name),
        };

        /// <summary>A block's filters, the first over the block's text.</summary>
        private object? Chain(Expr filter, object? text, Scope scope) =>
            filter is FilterUse f ? Filtered(f, f.Target is null ? text : Chain(f.Target, text, scope), scope) : text;

        private object? Filtered(FilterUse f, object? input, Scope scope)
        {
            var args = f.Args.Select(a => Eval(a, scope)).ToList();
            var named = f.Named.ToDictionary(x => x.Name, x => Eval(x.Value, scope), StringComparer.Ordinal);
            return At(f.Line, () => Builtins.Filter(f.Name, input, args, named));
        }

        private object? Global(string name) => name switch
        {
            "range" => (Callable)Builtins.Range,
            "dict" => (Callable)((args, named) =>
            {
                var d = new JDict();
                if (args.Count > 0 && args[0] is JDict from)
                {
                    from.Items.ForEach(p => d.Set(p.Key, p.Value));
                }
                foreach (var (k, v) in named)
                {
                    d.Set(k, v);
                }
                return d;
            }),
            "namespace" => (Callable)((args, named) =>
            {
                var n = new JNamespace();
                if (args.Count > 0 && args[0] is JDict from)
                {
                    from.Items.ForEach(p => n.Members[Py.Str(p.Key)] = p.Value);
                }
                foreach (var (k, v) in named)
                {
                    n.Members[k] = v;
                }
                return n;
            }),
            "raise_exception" => (Callable)((args, _) => throw new JinjaException(args.Count > 0 ? Py.Str(args[0]) : "the template raised an exception")),
            "strftime_now" => (Callable)((args, _) => Builtins.StrfTime(args.Count > 0 ? Py.Str(args[0]) : "%d %b %Y", now)),
            "cycler" => (Callable)((args, _) =>
            {
                var n = new JNamespace();
                int at = 0;
                n.Members["next"] = (Callable)((_, _) => args[at++ % args.Count]);
                return n;
            }),
            "joiner" => (Callable)((args, _) =>
            {
                bool used = false;
                return (Callable)((_, _) =>
                {
                    string s = used ? args.Count > 0 ? Py.Str(args[0]) : ", " : "";
                    used = true;
                    return s;
                });
            }),
            _ => null,
        };
    }
}
