using System.Globalization;
using System.Text;

namespace Amql.Entail;

/// <summary>One residual lane: a named interim fact. <see cref="Scale"/> is
/// the raw value a lane stores per unit of meaning — large counts are stored
/// small so they barely move the RMS norm, and read back through the
/// weights.</summary>
public sealed class Lane
{
    public required int Id { get; init; }
    public required string Name { get; init; }
    public double Scale { get; set; } = 1;
    /// <summary>Compile time (2·layer attention, 2·layer+1 FFN) of the last
    /// write; -1 for lanes only the embedding writes.</summary>
    public int Ready { get; set; } = -1;
    public bool Closed { get; set; }
    public bool Emits { get; set; }
}

/// <summary>A linear form over lanes plus a constant, in units of meaning.</summary>
public sealed class Lin
{
    public Dictionary<int, double> Terms { get; } = new();
    public double Const { get; set; }

    public static Lin Of(double value) => new() { Const = value };

    public static Lin OfLane(int lane)
    {
        var l = new Lin();
        l.Terms[lane] = 1;
        return l;
    }

    public bool IsConstant => Terms.Count == 0;

    public Lin Add(Lin other, double factor = 1)
    {
        var result = Scaled(1);
        result.Const += other.Const * factor;
        foreach (var (lane, c) in other.Terms)
        {
            double sum = result.Terms.GetValueOrDefault(lane) + c * factor;
            if (sum == 0)
            {
                result.Terms.Remove(lane);
            }
            else
            {
                result.Terms[lane] = sum;
            }
        }
        return result;
    }

    public Lin Scaled(double factor)
    {
        var result = new Lin { Const = Const * factor };
        foreach (var (lane, c) in Terms)
        {
            result.Terms[lane] = c * factor;
        }
        return result;
    }

    public double Eval(double[] lanes)
    {
        double sum = Const;
        foreach (var (lane, c) in Terms)
        {
            sum += c * lanes[lane];
        }
        return sum;
    }

    public string Key()
    {
        var sb = new StringBuilder(Const.ToString("R", CultureInfo.InvariantCulture));
        foreach (var (lane, c) in Terms.OrderBy(t => t.Key))
        {
            sb.Append('|').Append(lane).Append(':').Append(c.ToString("R", CultureInfo.InvariantCulture));
        }
        return sb.ToString();
    }
}

public abstract record EntailOp
{
    /// <summary>The block layer the scheduler placed this op in.</summary>
    public int Layer { get; set; }
}

/// <summary>One SwiGLU neuron: <c>lane += coef · silu(gate) · up</c>.</summary>
public sealed record Neuron(Lin Gate, Lin Up, int Dest, double Coef) : EntailOp;

/// <summary>One attention head. A fixed head looks <see cref="Offset"/>
/// tokens back; a pointer head looks back by whichever of
/// <see cref="Pointer"/>'s lanes is set.</summary>
public sealed record Head(int Offset, IReadOnlyList<(int Lane, int Offset)>? Pointer,
    IReadOnlyList<(int Dest, Lin Source)> Copies) : EntailOp;

/// <summary>The compiled meaning of a program: lanes, what each token writes
/// into them, the ops, and the outputs — everything except a target's
/// weights.</summary>
public sealed class EntailProgram
{
    public const int HeadsPerLayer = 8;
    public const int KvGroups = 2;
    public const int HeadDim = 256;

    public string Name { get; set; } = "entail";
    public List<Lane> Lanes { get; } = new();
    /// <summary>Token text → lane values written by the embedding row.</summary>
    public Dictionary<string, Dictionary<int, double>> Tokens { get; } = new(StringComparer.Ordinal);
    public List<EntailOp> Ops { get; } = new();
    /// <summary>(token text, or null for "end of answer", lane).</summary>
    public List<(string? Text, int Lane)> Emits { get; } = new();

    /// <summary>Token sets a lane rules out: every token but these
    /// (<c>only</c>), or these (<c>forbid</c>). They need a host, whose
    /// predictions they narrow.</summary>
    public List<(bool Allow, IReadOnlyList<string> Texts, int Lane)> Masks { get; } = new();
    /// <summary>Soft output: token text → what adds to its score. The highest
    /// positive score wins, as a softmax's would. An optimised program has
    /// none: votes are what `emit` replaces.</summary>
    public Dictionary<string, Lin> Votes { get; } = new(StringComparer.Ordinal);
    public int Layers { get; set; }
    public int MaxNeurons { get; set; }
    public int Statements { get; set; }
}

/// <summary>Expands a parsed program — constants, loops, macros, lets — into
/// neurons and heads, then schedules them. Every construct is a pure
/// rewriting: there is nothing to fit.</summary>
public static class EntailCompiler
{
    /// <summary>Gain of a threshold: the ramp is 1/K wide around c - ½.</summary>
    public const double K = 4;
    /// <summary>Sharpness that makes silu act as relu at a ramp's corners.</summary>
    public const double Beta = 32;
    /// <summary>Gate level of a decided fact: silu(±G) is ±G or 0.</summary>
    public const double G = 32;
    /// <summary>A gated value is a pair of neurons, G apart: both on, they
    /// differ by exactly G however far above threshold the gate stands, so a
    /// fact a few percent off its height still gates a value exactly.</summary>
    public const double Lift = 2;

    public static EntailProgram Compile(string source, int neuronsPerLayer = 3584)
    {
        var expander = new Expander();
        expander.Run(EntailParser.Parse(source));
        var program = expander.Program;
        Schedule(program, neuronsPerLayer);
        return program;
    }

    private static void Schedule(EntailProgram p, int capacity)
    {
        var neurons = new Dictionary<int, int>();
        var heads = new Dictionary<int, int>();
        int Read(Lin l, int ready)
        {
            foreach (var lane in l.Terms.Keys)
            {
                ready = Math.Max(ready, p.Lanes[lane].Ready);
                p.Lanes[lane].Closed = true;
            }
            return ready;
        }

        foreach (var op in p.Ops)
        {
            if (op is Neuron n)
            {
                int ready = Read(n.Up, Read(n.Gate, -1));
                // The FFN of layer L runs at time 2L+1, after `ready`.
                int layer = ready < 0 ? 0 : ready / 2 + ready % 2;
                while (neurons.GetValueOrDefault(layer) >= capacity)
                {
                    layer++;
                }
                neurons[layer] = neurons.GetValueOrDefault(layer) + 1;
                n.Layer = layer;
                Write(p, n.Dest, 2 * layer + 1);
            }
            else if (op is Head h)
            {
                int ready = -1;
                foreach (var (_, source) in h.Copies)
                {
                    ready = Read(source, ready);
                }
                if (h.Pointer is { } pointer)
                {
                    foreach (var (lane, _) in pointer)
                    {
                        ready = Math.Max(ready, p.Lanes[lane].Ready);
                        p.Lanes[lane].Closed = true;
                    }
                }
                // The attention of layer L runs at time 2L, after `ready`.
                int layer = ready < 0 ? 0 : ready / 2 + 1;
                while (heads.GetValueOrDefault(layer) >= EntailProgram.HeadsPerLayer)
                {
                    layer++;
                }
                heads[layer] = heads.GetValueOrDefault(layer) + 1;
                h.Layer = layer;
                foreach (var (dest, _) in h.Copies)
                {
                    Write(p, dest, 2 * layer);
                }
            }
        }
        foreach (var vote in p.Votes.Values)
        {
            Read(vote, -1);
        }
        p.Layers = p.Ops.Max(o => o.Layer) + 1;
        p.MaxNeurons = neurons.Values.DefaultIfEmpty(0).Max();
    }

    private static void Write(EntailProgram p, int lane, int time)
    {
        if (p.Lanes[lane].Closed)
        {
            throw new EntailException($"'{p.Lanes[lane].Name}' is written after it has been read");
        }
        p.Lanes[lane].Ready = Math.Max(p.Lanes[lane].Ready, time);
    }

    private sealed class Expander
    {
        public EntailProgram Program { get; } = new();

        private readonly Dictionary<string, double> _consts = new();
        private readonly Dictionary<string, int> _lanes = new();
        private readonly Dictionary<string, int[]> _shapes = new();
        private readonly Dictionary<string, double> _scales = new();
        private readonly Dictionary<string, Lin> _lets = new();
        private readonly Dictionary<string, DefStmt> _defs = new();
        private readonly Dictionary<string, int> _temps = new();
        private Dictionary<string, object> _env = new();
        private List<(int, Lin)>? _copies;

        public void Run(IReadOnlyList<Stmt> body)
        {
            foreach (var stmt in body)
            {
                Exec(stmt);
            }
        }

        private string Name(string template)
        {
            if (!template.Contains('{'))
            {
                return _env.TryGetValue(template, out var v) && v is string s ? s : template;
            }
            var sb = new StringBuilder();
            int i = 0;
            while (i < template.Length)
            {
                if (template[i] == '{')
                {
                    int end = template.IndexOf('}', i);
                    var key = template[(i + 1)..end];
                    sb.Append(_env.TryGetValue(key, out var v)
                        ? Convert.ToString(v, CultureInfo.InvariantCulture)
                        : throw new EntailException($"'{{{key}}}' is not defined"));
                    i = end + 1;
                }
                else
                {
                    sb.Append(template[i++]);
                }
            }
            return sb.ToString();
        }

        private int Int(Expr e)
        {
            var value = Eval(e);
            if (!value.IsConstant)
            {
                throw new EntailException(e.Line, "this must be a constant");
            }
            return checked((int)Math.Round(value.Const));
        }

        private string Key(RefExpr r) =>
            Name(r.Name) + string.Concat(r.Index.Select(i => $"[{Int(i)}]"));

        private int LaneFor(RefExpr r)
        {
            var key = Key(r);
            if (_lanes.TryGetValue(key, out int lane))
            {
                return lane;
            }
            var name = Name(r.Name);
            if (!_shapes.TryGetValue(name, out var shape))
            {
                throw new EntailException(r.Line, $"'{name}' is not a declared state");
            }
            if (shape.Length != r.Index.Count)
            {
                throw new EntailException(r.Line, $"'{name}' has {shape.Length} index(es)");
            }
            for (int i = 0; i < shape.Length; i++)
            {
                int index = Int(r.Index[i]);
                if (index < 0 || index >= shape[i])
                {
                    throw new EntailException(r.Line, $"index {index} is outside {name}[{shape[i]}]");
                }
            }
            return NewLane(key, _scales[name]);
        }

        private int NewLane(string key, double scale)
        {
            var lane = new Lane { Id = Program.Lanes.Count, Name = key, Scale = scale };
            Program.Lanes.Add(lane);
            _lanes[key] = lane.Id;
            return lane.Id;
        }

        private Lin Eval(Expr e)
        {
            switch (e)
            {
                case NumExpr n:
                    return Lin.Of(n.Value);
                case NegExpr n:
                    return Eval(n.Inner).Scaled(-1);
                case MinExpr m:
                    return Lin.Of(Math.Min(Int(m.Left), Int(m.Right)));
                case FloorExpr f:
                    return Lin.Of(Math.Floor(Eval(f.Inner).Const));
                case SumExpr s:
                    {
                        int from = Int(s.From), to = Int(s.To);
                        var total = Lin.Of(0);
                        var saved = _env;
                        for (int i = from; i <= to; i++)
                        {
                            _env = new Dictionary<string, object>(saved) { [s.Var] = (double)i };
                            total = total.Add(Eval(s.Body));
                        }
                        _env = saved;
                        return total;
                    }
                case BinExpr b:
                    {
                        var l = Eval(b.Left);
                        var r = Eval(b.Right);
                        switch (b.Op)
                        {
                            case '+': return l.Add(r);
                            case '-': return l.Add(r, -1);
                            case '*':
                                if (l.IsConstant) return r.Scaled(l.Const);
                                if (r.IsConstant) return l.Scaled(r.Const);
                                throw new EntailException(b.Line, "a product of two states is not linear: gate one by the other with 'when'");
                            case '/':
                                if (r.IsConstant) return l.Scaled(1 / r.Const);
                                throw new EntailException(b.Line, "cannot divide by a state");
                            default:
                                if (l.IsConstant && r.IsConstant) return Lin.Of(Math.Pow(l.Const, r.Const));
                                throw new EntailException(b.Line, "'^' takes constants");
                        }
                    }
                case RefExpr r:
                    {
                        if (r.Index.Count == 0 && _env.TryGetValue(r.Name, out var bound) && bound is double d)
                        {
                            return Lin.Of(d);
                        }
                        if (r.Index.Count == 0 && _consts.TryGetValue(r.Name, out var c))
                        {
                            return Lin.Of(c);
                        }
                        var key = Key(r);
                        if (_lets.TryGetValue(key, out var let))
                        {
                            return let;
                        }
                        return Lin.OfLane(LaneFor(r));
                    }
                default:
                    throw new EntailException(e.Line, "unknown expression");
            }
        }

        private static bool Holds(double l, string compare, double r) => compare switch
        {
            "<" => l < r,
            ">" => l > r,
            "<=" => l <= r,
            ">=" => l >= r,
            "==" => l == r,
            _ => l != r,
        };

        private void Exec(Stmt stmt)
        {
            Program.Statements++;
            switch (stmt)
            {
                case ProgramStmt s:
                    Program.Name = s.Name;
                    break;
                case ConstStmt s:
                    _consts[s.Name] = Eval(s.Value).Const;
                    break;
                case StateStmt s:
                    foreach (var decl in s.Decls)
                    {
                        var name = Name(decl.Name);
                        _shapes[name] = decl.Dims.Select(Int).ToArray();
                        _scales[name] = decl.Scale is null ? 1 : Eval(decl.Scale).Const;
                        if (decl.Dims.Count == 0)
                        {
                            NewLane(name, _scales[name]);
                        }
                    }
                    break;
                case LetStmt s:
                    _lets[Key(s.Target)] = Eval(s.Value);
                    break;
                case TokenStmt s:
                    foreach (var text in s.Texts)
                    {
                        var writes = new Dictionary<int, double>();
                        foreach (var (lane, value) in s.Means)
                        {
                            writes[LaneFor(lane)] = Eval(value).Const;
                        }
                        Program.Tokens[Literal(Name(text))] = writes;
                    }
                    break;
                case ForStmt s:
                    {
                        int from = Int(s.From), to = Int(s.To);
                        var saved = _env;
                        for (int i = from; i <= to; i++)
                        {
                            _env = new Dictionary<string, object>(saved) { [s.Var] = (double)i };
                            foreach (var inner in s.Body)
                            {
                                Exec(inner);
                            }
                        }
                        _env = saved;
                    }
                    break;
                case IfStmt s:
                    foreach (var inner in Holds(Eval(s.Left).Const, s.Compare, Eval(s.Right).Const) ? s.Then : s.Else)
                    {
                        Exec(inner);
                    }
                    break;
                case DefStmt s:
                    _defs[s.Name] = s;
                    break;
                case CallStmt s:
                    Call(s);
                    break;
                case AttendStmt s:
                    Attend(s);
                    break;
                case CopyStmt s:
                    if (_copies is null)
                    {
                        throw new EntailException(s.Line, "'<-' belongs inside an attend block");
                    }
                    _copies.Add((LaneFor(s.Target), Eval(s.Source)));
                    break;
                case EntailStmt s:
                    Entail(s);
                    break;
                case EmitStmt s:
                    {
                        int lane = LaneFor(s.Lane);
                        Program.Lanes[lane].Emits = true;
                        Program.Emits.Add((s.Text is null ? null : Literal(Name(s.Text)), lane));
                    }
                    break;
                case MaskStmt s:
                    {
                        int lane = LaneFor(s.Lane);
                        Program.Lanes[lane].Emits = true;
                        Program.Masks.Add((s.Allow, s.Texts.Select(t => Literal(Name(t))).ToList(), lane));
                    }
                    break;
                case VoteStmt s:
                    {
                        var key = Literal(Name(s.Text));
                        Program.Votes[key] = Program.Votes.TryGetValue(key, out var sum)
                            ? sum.Add(Eval(s.Value))
                            : Eval(s.Value);
                    }
                    break;
            }
        }

        private static string Literal(string text) => text.Replace(EntailLexer.Brace, '{');

        private void Call(CallStmt s)
        {
            if (!_defs.TryGetValue(s.Name, out var def))
            {
                throw new EntailException(s.Line, $"'{s.Name}' is not defined");
            }
            if (def.Params.Count != s.Args.Count)
            {
                throw new EntailException(s.Line, $"'{s.Name}' takes {def.Params.Count} argument(s)");
            }
            var env = new Dictionary<string, object>(_env);
            for (int i = 0; i < s.Args.Count; i++)
            {
                // A bare name that is not a number binds as a name.
                if (s.Args[i] is RefExpr { Index.Count: 0 } r &&
                    !_consts.ContainsKey(r.Name) &&
                    !(_env.TryGetValue(r.Name, out var v) && v is double))
                {
                    env[def.Params[i]] = Name(r.Name);
                }
                else
                {
                    env[def.Params[i]] = (double)Int(s.Args[i]);
                }
            }
            var saved = _env;
            _env = env;
            foreach (var inner in def.Body)
            {
                Exec(inner);
            }
            _env = saved;
        }

        private void Attend(AttendStmt s)
        {
            _copies = new List<(int, Lin)>();
            foreach (var inner in s.Body)
            {
                Exec(inner);
            }
            var copies = _copies;
            _copies = null;
            if (s.Pointer is { } pointerName)
            {
                var name = Name(pointerName);
                if (!_shapes.TryGetValue(name, out var shape) || shape.Length != 1)
                {
                    throw new EntailException(s.Line, $"'attend back by' needs a one-index state, not '{name}'");
                }
                var pointer = new List<(int, int)>();
                for (int i = 0; i < shape[0]; i++)
                {
                    if (_lanes.TryGetValue($"{name}[{i}]", out int lane))
                    {
                        pointer.Add((lane, i));
                    }
                }
                Program.Ops.Add(new Head(0, pointer, copies));
            }
            else
            {
                Program.Ops.Add(new Head(Int(s.Offset!), null, copies));
            }
        }

        // ── consequences → neurons ──────────────────────────────────────────

        private void Step(Lin e, double threshold, int dest, double coef)
        {
            // [e >= c] on whole numbers: relu(h + ½) - relu(h - ½) with
            // h = K(e - c + ½); exact at every integer, ±3/8 of noise margin.
            var h = e.Add(Lin.Of(0.5 - threshold)).Scaled(K);
            Program.Ops.Add(new Neuron(h.Add(Lin.Of(0.5)).Scaled(Beta), Lin.Of(1 / Beta), dest, coef));
            Program.Ops.Add(new Neuron(h.Add(Lin.Of(-0.5)).Scaled(Beta), Lin.Of(1 / Beta), dest, -coef));
        }

        private void Threshold(Lin e, string compare, double bound, int dest, double coef)
        {
            switch (compare)
            {
                case ">=": Step(e, bound, dest, coef); break;
                case ">": Step(e, bound + 1, dest, coef); break;
                case "<=": Step(e.Scaled(-1), -bound, dest, coef); break;
                case "<": Step(e.Scaled(-1), -bound + 1, dest, coef); break;
                default:
                    Step(e, bound, dest, coef);
                    Step(e, bound + 1, dest, -coef);
                    break;
            }
        }

        /// <summary>A threshold used inside a conjunction gets a lane of its
        /// own, shared by every statement that asks the same question.</summary>
        private Lin Fact(Cond c)
        {
            var value = Eval(c.Value);
            if (c.Compare is null)
            {
                return c.Negated ? Lin.Of(1).Add(value, -1) : value;
            }
            var e = value.Add(Eval(c.Bound!), -1);
            var key = c.Compare + "?" + e.Key();
            if (!_temps.TryGetValue(key, out int lane))
            {
                lane = NewLane($"$fact{_temps.Count}", 1);
                _temps[key] = lane;
                Threshold(e, c.Compare, 0, lane, 1);
            }
            var fact = Lin.OfLane(lane);
            return c.Negated ? Lin.Of(1).Add(fact, -1) : fact;
        }

        private void Entail(EntailStmt s)
        {
            int dest = LaneFor(s.Target);
            if (s.Value is null && s.When.Count == 1 && s.When[0] is { Compare: { } compare, Negated: false } only)
            {
                Threshold(Eval(only.Value).Add(Eval(only.Bound!), -1), compare, 0, dest, 1);
                return;
            }
            var value = s.Value is null ? Lin.Of(1) : Eval(s.Value);
            var gate = Lin.Of(1 - 2.0 * s.When.Count);
            foreach (var c in s.When)
            {
                gate = gate.Add(Fact(c), 2);
            }
            var high = gate.Scaled(G * Lift);
            Program.Ops.Add(new Neuron(high, value.Scaled(1 / G), dest, 1));
            Program.Ops.Add(new Neuron(high.Add(Lin.Of(-G)), value.Scaled(1 / G), dest, -1));
        }
    }
}

/// <summary>Executes a program's ops exactly as written, with no weights: the
/// meaning the compiled model has to reproduce.</summary>
public static class EntailInterpreter
{
    private static double Silu(double x) => x / (1 + Math.Exp(-x));

    /// <summary>Lane values at each position of a token sequence (tokens the
    /// program does not know write nothing).</summary>
    public static double[][] Run(EntailProgram p, IReadOnlyList<string> tokens)
    {
        var x = new double[tokens.Count][];
        for (int t = 0; t < tokens.Count; t++)
        {
            x[t] = new double[p.Lanes.Count];
            if (p.Tokens.TryGetValue(tokens[t], out var writes))
            {
                foreach (var (lane, value) in writes)
                {
                    x[t][lane] = value;
                }
            }
        }

        // Time order: every op in a group sees the state before the group,
        // exactly as a layer does.
        foreach (var group in p.Ops
            .GroupBy(o => 2 * o.Layer + (o is Neuron ? 1 : 0))
            .OrderBy(g => g.Key))
        {
            var delta = new List<(int T, int Lane, double V)>();
            foreach (var op in group)
            {
                for (int t = 0; t < tokens.Count; t++)
                {
                    if (op is Neuron n)
                    {
                        double up = n.Up.Eval(x[t]);
                        double gate = n.Gate.Eval(x[t]);
                        // A closed gate adds nothing a double could hold.
                        if (up != 0 && gate > -40)
                        {
                            delta.Add((t, n.Dest, n.Coef * Silu(gate) * up));
                        }
                    }
                    else if (op is Head h)
                    {
                        int offset = h.Offset;
                        if (h.Pointer is { } pointer)
                        {
                            offset = -1;
                            foreach (var (lane, o) in pointer)
                            {
                                if (x[t][lane] > 0.5)
                                {
                                    offset = o;
                                }
                            }
                            if (offset < 0)
                            {
                                continue;
                            }
                        }
                        // Past the start of the text a head finds the first token.
                        int source = Math.Max(t - offset, 0);
                        foreach (var (dest, lin) in h.Copies)
                        {
                            delta.Add((t, dest, lin.Eval(x[source])));
                        }
                    }
                }
            }
            foreach (var (t, lane, v) in delta)
            {
                x[t][lane] += v;
            }
        }
        return x;
    }

    /// <summary>The output at one position: the token text, "" for end of
    /// answer, or null when the program says nothing.</summary>
    public static string? Output(EntailProgram p, double[] lanes)
    {
        foreach (var (text, lane) in p.Emits)
        {
            if (lanes[lane] > 0.5)
            {
                return text ?? string.Empty;
            }
        }
        string? best = null;
        double top = 0;
        foreach (var (text, score) in p.Votes)
        {
            double value = score.Eval(lanes);
            if (value > top)
            {
                top = value;
                best = text;
            }
        }
        return best;
    }

    /// <summary>Writes out the answer to an expression by feeding each output
    /// back as the next token, the way a decoder would.</summary>
    public static string Answer(EntailProgram p, IEnumerable<string> tokens, int limit = 20)
    {
        var sequence = tokens.ToList();
        var answer = new StringBuilder();
        for (int i = 0; i < limit; i++)
        {
            var output = Output(p, Run(p, sequence)[^1]);
            if (string.IsNullOrEmpty(output))
            {
                break;
            }
            answer.Append(output);
            sequence.Add(output);
        }
        return answer.ToString();
    }
}
