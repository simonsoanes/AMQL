namespace Amql.Entail;

/// <summary>A program running beside a partner, fed the partner's token ids.</summary>
public interface IEntailAdapter : IDisposable
{
    EntailAdapterManifest Manifest { get; }

    /// <summary>A separate reader of the same program, for another thread.</summary>
    IEntailAdapter Sibling();

    void Reset();

    /// <summary>Reads a text and says what must follow it.</summary>
    EntailVerdict Prefill(int[] partnerTokens);

    /// <summary>Reads one more token and says what must follow it.</summary>
    EntailVerdict Step(int partnerToken);

    /// <summary>How many of the latest tokens <see cref="Rewind"/> can un-read now; 0 when it cannot.</summary>
    int RewindLimit => 0;

    /// <summary>Whether, stepped through <paramref name="drafts"/> more tokens
    /// than will be kept, it can take every one of them back.</summary>
    bool CanRewind(int drafts) => false;

    /// <summary>Un-reads the latest <paramref name="tokens"/> tokens, at most
    /// <see cref="RewindLimit"/>, as if they had never been read: speculation
    /// steps an adapter through drafts and takes back those not kept.</summary>
    void Rewind(int tokens) => throw new NotSupportedException($"adapter '{Manifest.Program}' cannot rewind");
}

/// <summary>
/// A program run as what it is: only 0.06% of a compiled adapter's weights are
/// not zero, and those are its own statements. So this executes the statements,
/// in doubles, one token at a time. A head reads a place in a ring of recent
/// tokens rather than attending to it, so the length of a text doesn't matter,
/// and memory is one row of lanes for each token the program can see back.
/// </summary>
/// <remarks>
/// A row kept from an earlier token holds that token's final values, which are
/// the values a head would have read at its layer: a state is closed once read.
/// </remarks>
public sealed class EntailSparseAdapter : IEntailAdapter
{
    public const string ProgramFile = "program.entail.txt";

    private readonly Compiled _plan;
    private readonly double[][] _ring;
    private readonly double[] _first;
    private readonly float[] _outputs;
    private readonly List<(int Lane, double Value)> _delta = new();
    private int _position;
    private readonly EntailRepeat _repeat;

    private EntailSparseAdapter(Compiled plan)
    {
        _plan = plan;
        _repeat = new EntailRepeat(plan.RepeatSpans, MaxRewind);
        // Rows beyond the window let the latest MaxRewind tokens be taken back:
        // their steps overwrite no row a later step at the rewound position reads.
        _ring = new double[plan.Window + 1 + MaxRewind][];
        for (int i = 0; i < _ring.Length; i++)
        {
            _ring[i] = new double[plan.Lanes];
        }
        _first = new double[plan.Lanes];
        _outputs = new float[plan.Manifest.Outputs.Count];
    }

    public EntailAdapterManifest Manifest => _plan.Manifest;

    /// <summary>Writes an adapter: the program and how it joins its partner.</summary>
    public static EntailAdapterManifest Write(string source, string partnerTokenizerDir, string outDir)
    {
        Directory.CreateDirectory(outDir);
        var binding = EntailWeights.BindAdapter(EntailCompiler.Compile(source), partnerTokenizerDir);
        File.WriteAllText(Path.Combine(outDir, ProgramFile), source);
        binding.Manifest.Save(outDir);
        return binding.Manifest;
    }

    /// <summary>Loads an adapter, refusing one built for another tokenizer.</summary>
    public static EntailSparseAdapter Open(string dir, string partnerTokenizerDir)
    {
        EntailAdapterManifest.Load(dir).RequirePartner(partnerTokenizerDir);
        var program = EntailCompiler.Compile(File.ReadAllText(Path.Combine(dir, ProgramFile)));
        return new EntailSparseAdapter(new Compiled(program, EntailWeights.BindAdapter(program, partnerTokenizerDir)));
    }

    public IEntailAdapter Sibling() => new EntailSparseAdapter(_plan);

    /// <summary>How many of the latest tokens can be taken back: more than a
    /// speculative step drafts (<c>GpuEngine.MaxDrafts</c>, 7).</summary>
    public const int MaxRewind = 16;

    // The furthest position stepped: a row is safe to read again only while no
    // step since has run more than MaxRewind positions past it.
    private int _furthest;

    public int RewindLimit => Math.Max(0, Math.Min(_position, Math.Min(MaxRewind - (_furthest - _position), _repeat.Undoable)));

    public bool CanRewind(int drafts)
    {
        int after = _position + drafts + 1;
        int furthest = Math.Max(_furthest, after);
        long undoable = _repeat.Any ? Math.Min(MaxRewind, (long)_repeat.Undoable + drafts + 1) : long.MaxValue;
        return drafts >= 0 && MaxRewind - (furthest - after) >= drafts && undoable >= drafts;
    }

    public void Reset()
    {
        _position = 0;
        _furthest = 0;
        _repeat.Reset();
    }

    /// <summary>Un-reads the latest tokens. A row is rebuilt whole when its
    /// position is stepped again, and the ring keeps <see cref="MaxRewind"/>
    /// rows past the window, so moving the position back is the whole of it,
    /// with the repeat table's own undo.</summary>
    public void Rewind(int tokens)
    {
        if (tokens < 0 || tokens > RewindLimit)
        {
            throw new ArgumentOutOfRangeException(nameof(tokens), $"can rewind 0 to {RewindLimit} tokens, not {tokens}");
        }
        _position -= tokens;
        _repeat.Rewind(tokens);
    }

    /// <summary>How many tokens back the program's state can depend on, or null when a
    /// carry or a repeat lets it depend on any of them (see <see cref="Compiled.Reach"/>).</summary>
    public int? Reach => _plan.Reach;

    public EntailVerdict Prefill(int[] partnerTokens)
    {
        // A program that cannot see further back than its reach reads a long text from where it
        // can: every row from there on, and so the verdict, is what reading all of it gives.
        int from = 0;
        if (_position == 0 && _plan.Reach is { } reach && partnerTokens.Length > reach + _plan.Window + 2)
        {
            from = partnerTokens.Length - (reach + _plan.Window + 2);
        }
        var verdict = EntailVerdict.Silence;
        for (int i = from; i < partnerTokens.Length; i++)
        {
            verdict = Step(partnerTokens[i]);
        }
        return verdict;
    }

    public EntailVerdict Step(int partnerToken)
    {
        var x = _ring[_position % _ring.Length];
        Array.Clear(x);
        if (_plan.Tokens.TryGetValue(partnerToken, out var writes))
        {
            foreach (var (lane, value) in writes)
            {
                x[lane] = value;
            }
        }
        if (_position > 0)
        {
            var previous = _ring[(_position - 1) % _ring.Length];
            foreach (var (from, to) in _plan.Carries)
            {
                x[to] += previous[from];
            }
        }
        if (_repeat.Any)
        {
            _repeat.Push(partnerToken);
            foreach (var (lane, span, times) in _plan.Repeats)
            {
                x[lane] = _repeat.Seen(span, times) ? 1 : 0;
            }
        }

        // Every op in a stage sees the state before the stage, as a layer does.
        foreach (var stage in _plan.Stages)
        {
            _delta.Clear();
            foreach (var head in stage.Heads)
            {
                Read(head, x);
            }
            foreach (var n in stage.Neurons)
            {
                double up = n.Up.Eval(x);
                double gate = n.Gate.Eval(x);
                // A closed gate adds nothing a double could hold.
                if (up != 0 && gate > -40)
                {
                    _delta.Add((n.Dest, n.Coef * gate / (1 + Math.Exp(-gate)) * up));
                }
            }
            foreach (var (lane, value) in _delta)
            {
                x[lane] += value;
            }
        }

        if (_position == 0)
        {
            x.CopyTo(_first, 0);
        }
        _position++;
        _furthest = Math.Max(_furthest, _position);

        for (int i = 0; i < _outputs.Length; i++)
        {
            _outputs[i] = (float)x[_plan.Outputs[i]];
        }
        var verdict = EntailVerdict.Read(_plan.Manifest, _outputs);
        // A repeat ban names no token of its own: it rules out whatever continued the
        // earlier run, which only the text so far can say.
        foreach (var o in _plan.Manifest.Outputs)
        {
            if (o.Kind == EntailOutputKind.ForbidRepeat && _outputs[o.Row] > 0.5f && _repeat.Continuation(o.Span) is { } next)
            {
                verdict = verdict.Banning(next);
            }
        }
        return verdict;
    }

    private void Read(CompiledHead head, double[] x)
    {
        int offset = head.Offset;
        if (head.Pointer is { } pointer)
        {
            offset = -1;
            foreach (var (lane, o) in pointer)
            {
                if (x[lane] > 0.5)
                {
                    offset = o;
                }
            }
            if (offset < 0)
            {
                return;
            }
        }
        // Past the start of the text a head finds the first token.
        var source = offset == 0 ? x : offset >= _position ? (_position == 0 ? x : _first) : _ring[(_position - offset) % _ring.Length];
        foreach (var (dest, form) in head.Copies)
        {
            _delta.Add((dest, form.Eval(source)));
        }
    }

    public void Dispose()
    {
    }

    private sealed record CompiledHead(int Offset, (int Lane, int Offset)[]? Pointer, (int Dest, Form Form)[] Copies);

    private sealed record CompiledNeuron(Form Gate, Form Up, int Dest, double Coef);

    private sealed record Stage(CompiledHead[] Heads, CompiledNeuron[] Neurons);

    /// <summary>A linear form flattened for reading.</summary>
    private sealed class Form
    {
        private readonly double _constant;
        private readonly int[] _lanes;
        private readonly double[] _factors;

        public Form(Lin lin)
        {
            _constant = lin.Const;
            _lanes = lin.Terms.Keys.ToArray();
            _factors = lin.Terms.Values.ToArray();
        }

        public double Eval(double[] x)
        {
            double sum = _constant;
            for (int i = 0; i < _lanes.Length; i++)
            {
                sum += _factors[i] * x[_lanes[i]];
            }
            return sum;
        }
    }

    /// <summary>What every reader of a program shares.</summary>
    private sealed class Compiled
    {
        public Compiled(EntailProgram p, EntailAdapterBinding binding)
        {
            Manifest = binding.Manifest;
            Lanes = p.Lanes.Count;
            Outputs = binding.Outputs.OrderBy(o => o.Row).Select(o => o.Lane).ToArray();
            var texts = binding.Inputs.ToDictionary(i => i.Row, i => i.Text);
            Tokens = binding.Manifest.Rows.ToDictionary(r => r.Key, r =>
                p.Tokens[texts[r.Value]].Select(w => (w.Key, w.Value)).ToArray());

            // Heads run before the neurons of their layer.
            var stages = new List<Stage>();
            foreach (var layer in p.Ops.GroupBy(o => o.Layer).OrderBy(g => g.Key))
            {
                var heads = layer.OfType<Head>().Select(h => new CompiledHead(h.Offset, h.Pointer?.ToArray(),
                    h.Copies.Select(c => (c.Item1, new Form(c.Item2))).ToArray())).ToArray();
                var neurons = layer.OfType<Neuron>().Select(n => new CompiledNeuron(new Form(n.Gate), new Form(n.Up), n.Dest, n.Coef)).ToArray();
                if (heads.Length > 0)
                {
                    stages.Add(new Stage(heads, Array.Empty<CompiledNeuron>()));
                }
                if (neurons.Length > 0)
                {
                    stages.Add(new Stage(Array.Empty<CompiledHead>(), neurons));
                }
            }
            Stages = stages.ToArray();
            Window = p.Ops.OfType<Head>().Select(h => Math.Max(h.Offset, h.Pointer?.Max(x => x.Offset) ?? 0)).DefaultIfEmpty(0).Max();
            Carries = p.Carries.ToArray();
            Repeats = p.Repeats.ToArray();
            RepeatSpans = p.RepeatSpans.Distinct().ToArray();
            // A carry reads the previous row, so the ring keeps at least one.
            if (Carries.Length > 0)
            {
                Window = Math.Max(Window, 1);
            }
            Reach = ReachOf(p);
        }

        /// <summary>
        /// How far back any lane of a row can depend on the text, in tokens: what a token writes
        /// reaches 0; a carry one more than the lane it copies; a head its offset more than what it
        /// reads (its pointer's lanes too); a neuron as far as its inputs. Worked to a fixed point,
        /// since a head reads the finished rows of earlier tokens. A cycle through a carry, such as
        /// a count, never settles: such a program, and one with a repeat, depends on the whole text.
        /// </summary>
        public int? Reach { get; }

        private static int? ReachOf(EntailProgram p)
        {
            if (p.Repeats.Count > 0)
            {
                return null;
            }
            const int Unbounded = 1 << 16;
            var reach = new int[p.Lanes.Count];
            bool changed = true;
            for (int pass = 0; changed; pass++)
            {
                changed = false;
                void Raise(int lane, int to)
                {
                    if (to > reach[lane])
                    {
                        reach[lane] = Math.Min(to, Unbounded);
                        changed = true;
                    }
                }
                int Max(Lin lin) => lin.Terms.Keys.Select(l => reach[l]).DefaultIfEmpty(0).Max();
                foreach (var (from, to) in p.Carries)
                {
                    Raise(to, reach[from] + 1);
                }
                foreach (var op in p.Ops.OrderBy(o => o.Layer))
                {
                    switch (op)
                    {
                        case Neuron n:
                            Raise(n.Dest, Math.Max(Max(n.Gate), Max(n.Up)));
                            break;
                        case Head h:
                            int back = Math.Max(h.Offset, h.Pointer?.Max(x => x.Offset) ?? 0);
                            int choose = h.Pointer?.Select(x => reach[x.Lane]).DefaultIfEmpty(0).Max() ?? 0;
                            foreach (var (dest, source) in h.Copies)
                            {
                                Raise(dest, Math.Max(choose, back + Max(source)));
                            }
                            break;
                    }
                }
                // A chain settles within a pass for each head or carry along it; one that has not
                // settled by then runs round a cycle.
                if (reach.Any(r => r >= Unbounded) || pass > 256)
                {
                    return null;
                }
            }
            return reach.DefaultIfEmpty(0).Max();
        }

        public (int From, int To)[] Carries { get; }

        public (int Lane, int Span, int Times)[] Repeats { get; }

        public int[] RepeatSpans { get; }

        public EntailAdapterManifest Manifest { get; }

        public int Lanes { get; }

        public int[] Outputs { get; }

        public Dictionary<int, (int Key, double Value)[]> Tokens { get; }

        public Stage[] Stages { get; }

        public int Window { get; private set; }
    }
}
