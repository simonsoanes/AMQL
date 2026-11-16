using System.Text;

namespace Amql.Entail;

/// <summary>
/// Runs a program over a text of any length in fixed memory: one row of lanes
/// per token a head can see back (the program's window), plus one for the
/// row being built. Each token runs every layer before the next begins, the
/// order <c>carry</c> needs, and the result at every position is the one
/// <see cref="EntailInterpreter.Run"/> gives.
/// </summary>
public sealed class EntailStream
{
    private readonly EntailProgram _p;
    private readonly EntailOp[][] _groups;
    private readonly double[][] _ring;
    private readonly List<(int Lane, double V)> _delta = new();
    private readonly double[] _first;
    private long _position;

    public EntailStream(EntailProgram p)
    {
        _p = p;
        _groups = p.Ops.GroupBy(o => 2 * o.Layer + (o is Neuron ? 1 : 0)).OrderBy(g => g.Key)
            .Select(g => g.ToArray()).ToArray();
        Window = p.Ops.OfType<Head>().Select(h => Math.Max(h.Offset, h.Pointer?.Max(x => x.Offset) ?? 0))
            .DefaultIfEmpty(0).Max();
        if (p.Carries.Count > 0)
            Window = Math.Max(Window, 1);
        _ring = new double[Window + 1][];
        for (int i = 0; i < _ring.Length; i++)
            _ring[i] = new double[p.Lanes.Count];
        _first = new double[p.Lanes.Count];
    }

    /// <summary>How many earlier rows are kept: the memory, in rows.</summary>
    public int Window { get; }

    public long Position => _position;

    /// <summary>Bytes of lane state held, whatever the length of the text.</summary>
    public long StateBytes => (long)(_ring.Length + 1) * _p.Lanes.Count * sizeof(double);

    /// <summary>Reads one token and returns its finished row (valid until the
    /// next call).</summary>
    public double[] Step(string token)
    {
        var row = _ring[_position % _ring.Length];
        Array.Clear(row);
        if (_p.Tokens.TryGetValue(token, out var writes))
        {
            foreach (var (lane, value) in writes)
                row[lane] = value;
        }
        if (_position > 0)
        {
            var previous = _ring[(_position - 1) % _ring.Length];
            foreach (var (from, to) in _p.Carries)
                row[to] += previous[from];
        }
        foreach (var group in _groups)
        {
            _delta.Clear();
            foreach (var op in group)
            {
                if (op is Neuron n)
                {
                    double up = n.Up.Eval(row);
                    double gate = n.Gate.Eval(row);
                    if (up != 0 && gate > -40)
                        _delta.Add((n.Dest, n.Coef * gate / (1 + Math.Exp(-gate)) * up));
                }
                else if (op is Head h)
                {
                    long offset = h.Offset;
                    if (h.Pointer is { } pointer)
                    {
                        offset = -1;
                        foreach (var (lane, o) in pointer)
                        {
                            if (row[lane] > 0.5)
                                offset = o;
                        }
                        if (offset < 0)
                            continue;
                    }
                    // Past the start of the text a head finds the first token,
                    // which only a window reaching back to it still holds.
                    long at = Math.Max(_position - offset, 0);
                    var source = offset == 0 ? row : at == 0 ? (_position == 0 ? row : _first) : _ring[at % _ring.Length];
                    foreach (var (dest, lin) in h.Copies)
                        _delta.Add((dest, lin.Eval(source)));
                }
            }
            foreach (var (lane, v) in _delta)
                row[lane] += v;
        }
        if (_position == 0)
            row.CopyTo(_first, 0);
        _position++;
        return row;
    }

    /// <summary>Reads a text, then writes the answer the way a decoder would,
    /// feeding each output back in.</summary>
    public string Answer(IEnumerable<string> tokens, int limit = 20)
    {
        double[]? last = null;
        foreach (var token in tokens)
            last = Step(token);
        var answer = new StringBuilder();
        for (int i = 0; i < limit && last is not null; i++)
        {
            var output = EntailInterpreter.Output(_p, last);
            if (string.IsNullOrEmpty(output))
                break;
            answer.Append(output);
            last = Step(output);
        }
        return answer.ToString();
    }
}
