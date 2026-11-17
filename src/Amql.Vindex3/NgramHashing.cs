namespace Amql.Vindex3;

/// <summary>
/// The arithmetic of Qwen4's hashed n-gram table (transformers
/// <c>Qwen4ExpTextNGramEmbedding</c>): which prime-sized slice of the table
/// each hash head owns, the odd multipliers the token ids are hashed with, and
/// the ids a token's n-grams look up. Pure functions of the surface and the
/// token history, shared by import (to check the shards hold the table the
/// config describes) and the runtime (to look rows up).
/// </summary>
public static class NgramHashing
{
    private const ulong SplitmixGamma = 0x9E3779B97F4A7C15UL;
    private const ulong SplitmixM1 = 0xBF58476D1CE4E5B9UL;
    private const ulong SplitmixM2 = 0x94D049BB133111EBUL;
    private const long Prime1 = 10007;

    /// <summary>One PLE layer's table layout: each head's vocabulary (a prime)
    /// and offset into the table, and the padded row count.</summary>
    public sealed record Layout(long[] HeadVocabSizes, long[] HeadOffsets, long PaddedRows, long[] Multipliers);

    /// <summary>The layout of the <paramref name="pleIndex"/>-th PLE layer (its
    /// position in the sorted <c>ple_layer_ids</c>, not its layer number).</summary>
    public static Layout For(NgramEmbeddingSurface surface, int pleIndex)
    {
        int heads = surface.NgramHeads;
        var sizes = new long[heads];
        var offsets = new long[heads];
        long total = 0;
        // _find_nth_prime_after(base - 1, global + 1) for each head in turn:
        // the heads of every PLE layer take consecutive primes after the base.
        long prime = surface.VocabSizeBase - 1;
        int skip = pleIndex * heads;
        for (int i = 0; i < skip; i++)
        {
            prime = NextPrime(prime);
        }
        for (int h = 0; h < heads; h++)
        {
            prime = NextPrime(prime);
            sizes[h] = prime;
            offsets[h] = total;
            total += prime;
        }
        long divisor = surface.VocabSizeDivisor;
        long padded = (total + divisor - 1) / divisor * divisor;
        return new Layout(sizes, offsets, padded, Multipliers(surface, pleIndex));
    }

    /// <summary><c>_build_layer_multipliers</c>: one odd multiplier per n-gram
    /// position, small enough that <c>token · multiplier</c> never overflows a
    /// signed 64-bit integer.</summary>
    public static long[] Multipliers(NgramEmbeddingSurface surface, int pleIndex)
    {
        const long maxLong = long.MaxValue;
        long multiplierMax = maxLong / Math.Max(surface.VocabSize, 1);
        long halfBound = Math.Max(1, multiplierMax / 2);
        ulong baseSeed = unchecked((ulong)(surface.Seed + Prime1 * pleIndex));
        var result = new long[surface.NgramSize];
        for (int index = 0; index < surface.NgramSize; index++)
        {
            ulong value = unchecked(baseSeed + SplitmixGamma * (ulong)(index + 1));
            result[index] = 2 * (long)(Splitmix64(value) % (ulong)halfBound) + 1;
        }
        return result;
    }

    public static ulong Splitmix64(ulong value)
    {
        unchecked
        {
            value += SplitmixGamma;
            value = (value ^ (value >> 30)) * SplitmixM1;
            value = (value ^ (value >> 27)) * SplitmixM2;
            return value ^ (value >> 31);
        }
    }

    /// <summary>
    /// The table rows one token looks up, head by head (2-grams first):
    /// <paramref name="history"/> holds the tokens before it (most recent last,
    /// at least <c>NgramSize − 1</c> of them; the start of the sequence reads as
    /// end-of-sequence tokens), <paramref name="token"/> is the token itself.
    /// An n-gram never reaches back across an end-of-sequence token: the
    /// earlier tokens it would have read are replaced by it.
    /// </summary>
    public static void Rows(Layout layout, NgramEmbeddingSurface surface, ReadOnlySpan<int> history, int token, Span<long> rows)
    {
        int n = surface.NgramSize;
        int eos = surface.EosTokenId;
        // shifted[s]: the token s back, or eos when one of the s tokens before
        // this one (or the position itself) runs past a document boundary
        // (_shift_right_ignore_eos).
        Span<long> shifted = stackalloc long[n];
        shifted[0] = token;
        bool crossed = false;
        for (int s = 1; s < n; s++)
        {
            int previous = history[history.Length - s];
            crossed |= previous == eos;
            shifted[s] = crossed ? eos : previous;
        }
        var m = layout.Multipliers;
        int perNgram = surface.HeadsPerNgram;
        for (int ngram = 2; ngram <= n; ngram++)
        {
            long mixed = shifted[0] * m[0];
            for (int position = 1; position < ngram; position++)
            {
                mixed ^= shifted[position] * m[position];
            }
            int start = (ngram - 2) * perNgram;
            for (int h = 0; h < perNgram; h++)
            {
                long size = layout.HeadVocabSizes[start + h];
                long id = mixed % size;
                if (id < 0)
                {
                    id += size; // torch.remainder takes the divisor's sign
                }
                rows[start + h] = id + layout.HeadOffsets[start + h];
            }
        }
    }

    private static long NextPrime(long after)
    {
        long candidate = after + 1;
        while (!IsPrime(candidate))
        {
            candidate++;
        }
        return candidate;
    }

    public static bool IsPrime(long value)
    {
        if (value < 2)
        {
            return false;
        }
        if (value % 2 == 0)
        {
            return value == 2;
        }
        for (long divisor = 3; divisor * divisor <= value; divisor += 2)
        {
            if (value % divisor == 0)
            {
                return false;
            }
        }
        return true;
    }
}
