namespace Amql.Inference;

/// <summary>How the positions committed at each step are chosen among the
/// still-masked ones in the current block.</summary>
public enum Remasking
{
    /// <summary>The positions whose predicted token is most probable.</summary>
    LowConfidence,

    /// <summary>Uniformly random positions (seeded).</summary>
    Random,
}

public sealed record DiffusionOptions
{
    /// <summary>Masked positions appended after the prompt.</summary>
    public int GenLength { get; init; } = 128;

    /// <summary>Denoising passes over the whole generation (split evenly per block).</summary>
    public int Steps { get; init; } = 128;

    /// <summary>Blocks are denoised left to right; each finishes before the next starts.</summary>
    public int BlockLength { get; init; } = 32;

    /// <summary>0 is greedy; above 0 the token is drawn by Gumbel-max at this temperature.</summary>
    public float Temperature { get; init; }

    public Remasking Remasking { get; init; } = Remasking.LowConfidence;

    public int Seed { get; init; } = 42;

    /// <summary>Tokens that end the answer. When a finished block holds one,
    /// later blocks are skipped: committed tokens never change, so the text up
    /// to the stop token is already final.</summary>
    public IReadOnlySet<int>? StopTokens { get; init; }
}

/// <summary>One denoising pass: which positions (absolute) were committed.</summary>
public sealed record DiffusionStep(int Block, int Step, IReadOnlyList<int> Committed, IReadOnlyList<int> Sequence);

public sealed record DiffusionResult(
    int[] Generated,
    int Forwards,
    bool StoppedEarly,
    IReadOnlyList<IReadOnlyList<int>> CommitTrace,
    TimeSpan StackTime,
    TimeSpan HeadTime);

/// <summary>
/// Masked diffusion (MDLM) generation over a bidirectional decoder: the
/// prompt is followed by <see cref="DiffusionOptions.GenLength"/> mask tokens;
/// block by block, each pass predicts every masked position at once and
/// commits a scheduled number of them, until none remain. This mirrors the
/// low-confidence / random remasking sampler published with the dLLM A2D
/// checkpoints. Logits at a position predict the token at that position.
/// </summary>
public sealed class MaskedDiffusionSampler
{
    private readonly GenericRuntime _runtime;
    private readonly int _maskId;

    public MaskedDiffusionSampler(GenericRuntime runtime)
    {
        _runtime = runtime;
        _maskId = runtime.Plan.Diffusion?.MaskTokenId
            ?? throw new UnsupportedOperatorException("the plan declares no masked diffusion objective");
    }

    public int MaskTokenId => _maskId;

    private long _stackTicks, _headTicks;

    /// <summary>One full bidirectional pass over <paramref name="tokens"/>,
    /// returning the logits of rows [rowStart, rowStart + rowCount) only —
    /// the vocabulary head is the costliest GEMM, and only the rows being
    /// denoised are read.</summary>
    public Tensor2D Logits(int[] tokens, int rowStart, int rowCount)
    {
        if (rowStart < 0 || rowCount < 1 || rowStart + rowCount > tokens.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(rowCount), "row range outside the sequence");
        }
        long started = System.Diagnostics.Stopwatch.GetTimestamp();
        _runtime.Kv.Reset();
        _runtime.ResetSession();
        var hidden = _runtime.Embed(tokens);
        var positions = Enumerable.Range(0, tokens.Length).ToArray();
        for (int layer = 0; layer < _runtime.Plan.Layers.Count; layer++)
        {
            hidden = _runtime.RunLayerInternal(hidden, layer, positions, positions, appendKv: true);
        }
        var rows = new float[rowCount * hidden.Cols];
        Array.Copy(hidden.Data, rowStart * hidden.Cols, rows, 0, rows.Length);
        long stackDone = System.Diagnostics.Stopwatch.GetTimestamp();
        var logits = _runtime.FinalNormAndHead(new Tensor2D(rows, rowCount, hidden.Cols));
        _stackTicks += stackDone - started;
        _headTicks += System.Diagnostics.Stopwatch.GetTimestamp() - stackDone;
        return logits;
    }

    public DiffusionResult Generate(int[] prompt, DiffusionOptions options, Action<DiffusionStep>? onStep = null)
    {
        int gen = options.GenLength;
        int block = options.BlockLength;
        if (prompt.Length == 0)
        {
            throw new ArgumentException("diffusion needs at least one prompt token", nameof(prompt));
        }
        if (gen < 1 || block < 1 || gen % block != 0)
        {
            throw new ArgumentException($"gen length {gen} must be a positive multiple of block length {block}");
        }
        int blocks = gen / block;
        if (options.Steps < blocks || options.Steps % blocks != 0)
        {
            throw new ArgumentException($"steps {options.Steps} must be a positive multiple of the block count {blocks}");
        }
        int stepsPerBlock = options.Steps / blocks;

        var x = new int[prompt.Length + gen];
        prompt.CopyTo(x, 0);
        Array.Fill(x, _maskId, prompt.Length, gen);
        var rng = new Random(options.Seed);
        _stackTicks = _headTicks = 0;
        var trace = new List<IReadOnlyList<int>>();
        int forwards = 0;
        bool stoppedEarly = false;

        for (int b = 0; b < blocks; b++)
        {
            int lo = prompt.Length + b * block;
            int masked = 0;
            for (int i = lo; i < lo + block; i++)
            {
                if (x[i] == _maskId)
                {
                    masked++;
                }
            }
            // The reference's schedule: an even split, the remainder going to
            // the earliest steps.
            int baseCount = masked / stepsPerBlock, remainder = masked % stepsPerBlock;

            for (int s = 0; s < stepsPerBlock; s++)
            {
                int k = baseCount + (s < remainder ? 1 : 0);
                if (k == 0)
                {
                    continue;
                }
                var logits = Logits(x, lo, block);
                forwards++;
                var candidates = new List<(int Pos, int Token, double Confidence)>(block);
                for (int r = 0; r < block; r++)
                {
                    int pos = lo + r;
                    if (x[pos] != _maskId)
                    {
                        continue;
                    }
                    var row = logits.Row(r);
                    int token = options.Temperature > 0f ? GumbelArgmax(row, options.Temperature, rng) : Argmax(row);
                    double confidence = options.Remasking == Remasking.LowConfidence
                        ? Probability(row, token)
                        : rng.NextDouble();
                    candidates.Add((pos, token, confidence));
                }
                // Highest confidence first; equal confidence keeps the earlier
                // position, which is what topk returns for a stable sort.
                var chosen = candidates
                    .OrderByDescending(c => c.Confidence)
                    .ThenBy(c => c.Pos)
                    .Take(k)
                    .ToList();
                foreach (var c in chosen)
                {
                    x[c.Pos] = c.Token;
                }
                var committed = chosen.Select(c => c.Pos).OrderBy(p => p).ToList();
                trace.Add(committed);
                onStep?.Invoke(new DiffusionStep(b, s, committed, x));
            }

            if (options.StopTokens is { Count: > 0 } stops && b < blocks - 1)
            {
                bool hasStop = false;
                for (int i = lo; i < lo + block && !hasStop; i++)
                {
                    hasStop = stops.Contains(x[i]);
                }
                if (hasStop)
                {
                    stoppedEarly = true;
                    break;
                }
            }
        }

        return new DiffusionResult(x[prompt.Length..], forwards, stoppedEarly, trace,
            System.Diagnostics.Stopwatch.GetElapsedTime(0, _stackTicks),
            System.Diagnostics.Stopwatch.GetElapsedTime(0, _headTicks));
    }

    private static int Argmax(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
        {
            if (row[i] > row[best])
            {
                best = i;
            }
        }
        return best;
    }

    /// <summary>softmax(row)[token], computed stably in double.</summary>
    private static double Probability(ReadOnlySpan<float> row, int token)
    {
        float max = row[0];
        foreach (var v in row)
        {
            if (v > max)
            {
                max = v;
            }
        }
        double sum = 0;
        foreach (var v in row)
        {
            sum += Math.Exp(v - max);
        }
        return Math.Exp(row[token] - max) / sum;
    }

    /// <summary>argmax(logits + T·Gumbel) — the reference's
    /// <c>exp(logits) / (−log u)^T</c>, taken in log space.</summary>
    private static int GumbelArgmax(ReadOnlySpan<float> row, float temperature, Random rng)
    {
        int best = 0;
        double bestScore = double.NegativeInfinity;
        for (int i = 0; i < row.Length; i++)
        {
            double u = rng.NextDouble();
            double score = row[i] - temperature * Math.Log(-Math.Log(u + double.Epsilon));
            if (score > bestScore)
            {
                bestScore = score;
                best = i;
            }
        }
        return best;
    }
}
