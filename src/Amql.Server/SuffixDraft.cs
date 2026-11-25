namespace Amql.Server;

/// <summary>
/// Draft tokens from the context itself: when the reply's latest tokens
/// already appeared earlier (in the prompt or the reply), what followed them
/// there is a likely continuation. An agent copies paths, names and code from
/// its context, and these drafts cost nothing to make.
/// </summary>
public static class SuffixDraft
{
    private const int LongestMatch = 32;

    /// <summary>What followed the latest earlier occurrence of the context's
    /// longest repeated ending (at least <paramref name="minMatch"/> tokens, up to
    /// 32), at most <paramref name="max"/> tokens; null when no ending that long repeats.</summary>
    public static int[]? Find(IReadOnlyList<int> context, int minMatch, int max)
    {
        int n = context.Count;
        if (minMatch <= 0 || max <= 0 || n <= minMatch)
        {
            return null;
        }
        int last = context[n - 1], best = 0, bestAt = -1;
        for (int i = n - 2; i >= minMatch - 1; i--)
        {
            if (context[i] != last)
            {
                continue;
            }
            int m = 1;
            while (m < LongestMatch && i - m >= 0 && context[i - m] == context[n - 1 - m])
            {
                m++;
            }
            if (m > best)
            {
                best = m;
                bestAt = i;
                if (m == LongestMatch)
                {
                    break;
                }
            }
        }
        if (best < minMatch)
        {
            return null;
        }
        int count = 0;
        while (count < Math.Min(max, n - 1 - bestAt) && context[bestAt + 1 + count] >= 0)   // an image's ids are no tokens
        {
            count++;
        }
        return count < 1 ? null : Enumerable.Range(bestAt + 1, count).Select(i => context[i]).ToArray();
    }
}
