namespace Amql.Inference;

/// <summary>
/// Which passages of a long context a question is about, read off the model's
/// own attention as it reads the question (<see cref="GpuEngine.ReadAttention"/>).
/// A model loses its way among similar passages far back long before it stops
/// attending to them, so the passages it attends to most, read again beside the
/// question, put the answer at close range.
/// </summary>
public static class AttentionFocus
{
    /// <summary>Tokens a passage holds.</summary>
    public const int PassageTokens = 256;

    /// <summary>The <paramref name="count"/> passages that drew the most
    /// attention, in document order, none overlapping.</summary>
    /// <param name="mass">Attention by window of <see cref="GpuEngine.AttentionWindow"/> positions.</param>
    /// <param name="idle">The same for a question about nothing in particular, taken
    /// off first: some windows draw attention whatever is asked. Null takes nothing off.</param>
    /// <param name="from">The first position that may be in a passage.</param>
    /// <param name="to">The position after the last that may be.</param>
    public static List<(int From, int To)> Passages(float[] mass, float[]? idle, int from, int to, int count) =>
        Passages(mass, idle, from, to, count, GpuEngine.AttentionWindow);

    public static List<(int From, int To)> Passages(float[] mass, float[]? idle, int from, int to, int count, int window)
    {
        int per = Math.Max(1, PassageTokens / window);
        int first = (from + window - 1) / window, last = Math.Min(mass.Length, to / window);   // [first, last)
        var net = new float[Math.Max(0, last)];
        for (int i = first; i < last; i++)
        {
            net[i] = idle is not null && i < idle.Length ? Math.Max(0, mass[i] - idle[i]) : mass[i];
        }
        // Each passage is centred on the window that drew most, so a fact is not cut by its edge.
        var chosen = new List<(int From, int To)>();
        foreach (int w in Enumerable.Range(first, Math.Max(0, last - first)).Where(i => net[i] > 0).OrderByDescending(i => net[i]).ThenBy(i => i))
        {
            int a = Math.Clamp(w - per / 2, first, Math.Max(first, last - per)) * window;
            int b = Math.Min(a + per * window, last * window);
            if (chosen.Any(c => c.From <= w * window && w * window < c.To))
            {
                continue;   // already inside one
            }
            // Beside another: only the part it does not already hold.
            foreach (var c in chosen)
            {
                if (a < c.To && c.From < b)
                {
                    (a, b) = c.From <= a ? (c.To, b) : (a, c.From);
                }
            }
            if (b - a < window)
            {
                continue;
            }
            chosen.Add((a, b));
            if (chosen.Count == count)
            {
                break;
            }
        }
        chosen.Sort();
        return chosen;
    }
}
