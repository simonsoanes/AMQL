namespace Amql.Entail;

/// <summary>
/// What <c>repeat</c> and <c>forbid repeat</c> read: for each run length a program names, how
/// many times the run of that many tokens ending at the current one has been seen earlier in
/// the text, and which token followed it the latest time. This is what an induction head
/// computes, a match of the recent tokens against everything before them, so it needs the
/// whole text rather than a fixed window: memory grows with the text, by one id and one
/// table entry a span.
/// </summary>
public sealed class EntailRepeat
{
    private readonly int[] _spans;
    private readonly Dictionary<ulong, (int End, int Count)>[] _seen;
    private readonly int[] _count;
    private readonly int?[] _continuation;
    private readonly List<int> _tokens = new();

    public EntailRepeat(IEnumerable<int> spans)
    {
        _spans = spans.Distinct().Order().ToArray();
        _seen = _spans.Select(_ => new Dictionary<ulong, (int, int)>()).ToArray();
        _count = new int[_spans.Length];
        _continuation = new int?[_spans.Length];
    }

    public bool Any => _spans.Length > 0;

    public void Reset()
    {
        _tokens.Clear();
        foreach (var s in _seen)
        {
            s.Clear();
        }
        Array.Clear(_count);
        Array.Clear(_continuation);
    }

    /// <summary>Reads one more token.</summary>
    public void Push(int token)
    {
        _tokens.Add(token);
        int end = _tokens.Count - 1;
        for (int k = 0; k < _spans.Length; k++)
        {
            int span = _spans[k];
            _count[k] = 0;
            _continuation[k] = null;
            if (_tokens.Count < span)
            {
                continue;
            }
            ulong h = Hash(end - span + 1, span);
            // A hash that matches is checked token by token: a collision counts as a new run.
            if (_seen[k].TryGetValue(h, out var earlier) && Same(earlier.End - span + 1, end - span + 1, span))
            {
                _count[k] = earlier.Count;
                _continuation[k] = _tokens[earlier.End + 1];
            }
            _seen[k][h] = (end, _count[k] + 1);
        }
    }

    /// <summary>The run of <paramref name="span"/> tokens ending here was seen at least
    /// <paramref name="times"/> times before.</summary>
    public bool Seen(int span, int times = 1) => _count[Array.IndexOf(_spans, span)] >= times;

    /// <summary>The token that followed the latest earlier run of <paramref name="span"/>
    /// tokens equal to the one ending here, or null when there is none.</summary>
    public int? Continuation(int span) => _continuation[Array.IndexOf(_spans, span)];

    private ulong Hash(int start, int span)
    {
        ulong h = 1469598103934665603UL;
        for (int i = start; i < start + span; i++)
        {
            h = (h ^ (uint)_tokens[i]) * 1099511628211UL;
        }
        return h;
    }

    private bool Same(int a, int b, int span)
    {
        for (int i = 0; i < span; i++)
        {
            if (_tokens[a + i] != _tokens[b + i])
            {
                return false;
            }
        }
        return true;
    }
}
