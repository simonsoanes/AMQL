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

    private EntailSparseAdapter(Compiled plan)
    {
        _plan = plan;
        _ring = new double[plan.Window + 1][];
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

    public void Reset() => _position = 0;

    public EntailVerdict Prefill(int[] partnerTokens)
    {
        var verdict = EntailVerdict.Silence;
        foreach (int token in partnerTokens)
        {
            verdict = Step(token);
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

        for (int i = 0; i < _outputs.Length; i++)
        {
            _outputs[i] = (float)x[_plan.Outputs[i]];
        }
        return EntailVerdict.Read(_plan.Manifest, _outputs);
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
        }

        public EntailAdapterManifest Manifest { get; }

        public int Lanes { get; }

        public int[] Outputs { get; }

        public Dictionary<int, (int Key, double Value)[]> Tokens { get; }

        public Stage[] Stages { get; }

        public int Window { get; }
    }
}
