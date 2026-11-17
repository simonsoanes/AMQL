using System.Diagnostics;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Cli;

// ── bench: single-request prefill / decode throughput ─────────────────────

internal static partial class Program
{
    /// <summary>
    /// <c>bench &lt;container&gt; [--prompt-tokens N] [--steps M] [--repeat R] [--weights mode] [--profile] [--compare]</c>.
    /// Loads the model, runs one untimed warm-up request (so weights are
    /// resident and the kernels compiled — "once loaded"), then R timed
    /// requests of an N-token prefill followed by M greedy decode steps.
    /// Reports prefill and decode tokens per second, comparable to llama.cpp's
    /// pp/tg figures, and a checksum over the bit patterns of the logits, so
    /// a change that claims not to alter the numerics can be checked against
    /// the checksum from before it. <c>--profile</c> adds a per-operator split
    /// of one prefill and one decode run into time inside the parallel
    /// kernels and serial time.
    /// </summary>
    private static int Bench(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("bench requires a container directory");
        int promptTokens = IntOption(args, "--prompt-tokens", 32);
        int steps = IntOption(args, "--steps", 64);
        int repeat = IntOption(args, "--repeat", 3);
        if (promptTokens < 1 || steps < 1 || repeat < 1)
        {
            throw new CliException("bench needs --prompt-tokens, --steps and --repeat of at least 1");
        }
        string weightMode = OptionValue(args, "--weights") ?? string.Empty;
        WeightWorkingSet? workingSet = weightMode switch
        {
            "f32" or "fp32" => WeightWorkingSet.ResidentF32,
            "bf16" => WeightWorkingSet.OnDemandBf16,
            "mxfp4" or "fp4" => WeightWorkingSet.Mxfp4,
            "disk" => WeightWorkingSet.Disk,
            "lean" => WeightWorkingSet.Lean,
            "" => null,
            _ => throw new CliException($"unknown weight mode '{weightMode}' — use f32, bf16, mxfp4, disk or lean"),
        };

        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var load = Stopwatch.StartNew();
        using var session = new DecodeSession(plan, store, null, workingSet);

        // A deterministic pseudo-prompt inside the vocabulary.
        int vocab = plan.Embedding?.VocabSize ?? 1000;
        var prompt = new int[promptTokens];
        for (int i = 0; i < promptTokens; i++)
        {
            prompt[i] = Math.Min(1000, vocab / 2) + (i * 7919) % Math.Max(1, vocab - Math.Min(1000, vocab / 2));
        }

        // Warm-up: loads the weights and compiles the kernels.
        RunOnce(session, prompt, Math.Min(steps, 4), out _, out _);
        Console.WriteLine($"bench:     {plan.Layers.Count} layers, hidden {plan.HiddenSize}, " +
                          $"{ComputeBudget.Describe()}, load + warm-up {load.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine(session.GpuRefusal is { } refusal
            ? $"backend:   {session.Backend} (device engine not used: {refusal})"
            : $"backend:   {session.Backend}");

        if (HasOption(args, "--compare"))
        {
            Compare(session, plan, store, workingSet, prompt, steps);
        }

        if (HasOption(args, "--profile"))
        {
            Profile(session, prompt, steps);
        }

        var prefillRates = new List<double>();
        var decodeRates = new List<double>();
        ulong checksum = 0;
        for (int r = 0; r < repeat; r++)
        {
            checksum = RunOnce(session, prompt, steps, out double prefillSeconds, out double decodeSeconds);
            prefillRates.Add(promptTokens / prefillSeconds);
            decodeRates.Add(steps / decodeSeconds);
            Console.WriteLine($"run {r + 1}:     pp{promptTokens} {promptTokens / prefillSeconds,8:F1} tok/s   " +
                              $"tg{steps} {steps / decodeSeconds,8:F1} tok/s   ({decodeSeconds * 1000.0 / steps:F2} ms/token)");
        }
        Console.WriteLine($"checksum:  {checksum:x16} (bit patterns of the prefill and final-step logits)");
        Console.WriteLine($"best:      pp{promptTokens} {prefillRates.Max():F1} tok/s   tg{steps} {decodeRates.Max():F1} tok/s");
        if (session.Runtime.Weights.WorkingSet == WeightWorkingSet.Lean)
        {
            Console.WriteLine($"lean:      {LeanLine(session.Runtime.Weights.Lean)}");
        }
        Console.WriteLine($"memory:    {ProcessMemoryLine()}");
        return 0;
    }

    /// <summary>One prefill and one decode run with each layer operator's
    /// time split into time inside <see cref="KernelPool"/> loops (parallel)
    /// and the rest (serial) — where a single request still runs on one core.</summary>
    private static void Profile(DecodeSession session, int[] prompt, int steps)
    {
        var perOp = new Dictionary<string, double>();
        var perOpPool = new Dictionary<string, double>();
        long lastPool = 0;
        KernelPool.Profiling = true;
        session.Runtime.OpTrace = o =>
        {
            if (o.Ms > 0)
            {
                perOp[o.Op] = perOp.GetValueOrDefault(o.Op) + o.Ms;
                long nowPool = KernelPool.InsideTicks;
                perOpPool[o.Op] = perOpPool.GetValueOrDefault(o.Op) + (nowPool - lastPool) * 1000.0 / Stopwatch.Frequency;
                lastPool = nowPool;
            }
        };
        void Report(string title, double scale, string unit)
        {
            Console.WriteLine(title);
            foreach (var (op, ms) in perOp.OrderByDescending(kv => kv.Value))
            {
                double serial = ms - perOpPool.GetValueOrDefault(op);
                Console.WriteLine($"  {op,-16} {ms * scale,9:F2} {unit}  (serial {Math.Max(0, serial) * scale:F2})");
            }
            perOp.Clear();
            perOpPool.Clear();
        }
        try
        {
            // Prefill alone (no decode steps), then decode alone (a one-token
            // prompt), so neither phase's operators blur into the other's.
            session.Reset();
            lastPool = KernelPool.InsideTicks;
            var sw = Stopwatch.StartNew();
            session.Prefill(prompt);
            Report($"profile:   prefill of {prompt.Length} tokens, {sw.Elapsed.TotalMilliseconds:F1} ms", 1.0, "ms");

            session.Reset();
            session.Prefill(new[] { prompt[0] });
            perOp.Clear();
            perOpPool.Clear();
            long poolStart = KernelPool.InsideTicks;
            long loopsStart = KernelPool.Loops;
            lastPool = poolStart;
            sw.Restart();
            int next = ArgMax(session.LastLogits.FirstRow());
            for (int i = 0; i < steps; i++)
            {
                next = ArgMax(session.Step(next).FirstRow());
            }
            double ms = sw.Elapsed.TotalMilliseconds / steps;
            double inside = (KernelPool.InsideTicks - poolStart) * 1000.0 / Stopwatch.Frequency / steps;
            Report($"profile:   decode {ms:F2} ms/token, {inside:F2} inside parallel loops " +
                   $"({(KernelPool.Loops - loopsStart) / (double)steps:F0} loops/token)", 1.0 / steps, "ms/token");
        }
        finally
        {
            session.Runtime.OpTrace = null;
            KernelPool.Profiling = false;
        }
    }

    /// <summary>The same greedy request on this session and on a managed-path
    /// session: where the two first choose different tokens, and how far apart
    /// their logits are while they agree. Checks the device engine, whose
    /// reduction order differs from the managed kernels'.</summary>
    private static void Compare(DecodeSession session, ComponentOpPlan plan, OperandStore store,
        WeightWorkingSet? workingSet, int[] prompt, int steps)
    {
        using var reference = new DecodeSession(plan, store, null, workingSet) { AllowGpuEngine = false };
        session.Reset();
        var a = session.Prefill(prompt).FirstRow().ToArray();
        var b = reference.Prefill(prompt).FirstRow().ToArray();
        double worst = 0, worstRel = 0;
        int agreed = 0;
        for (int i = 0; ; i++)
        {
            double maxAbs = 0, scale = 0;
            for (int j = 0; j < a.Length; j++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(a[j] - b[j]));
                scale = Math.Max(scale, Math.Abs(b[j]));
            }
            worst = Math.Max(worst, maxAbs);
            worstRel = Math.Max(worstRel, maxAbs / Math.Max(scale, 1e-9));
            int ta = ArgMax(a), tb = ArgMax(b);
            if (ta != tb)
            {
                Console.WriteLine($"compare:   tokens differ at step {i} ({ta} vs {tb}; logit gap there {b[tb] - b[ta]:G3})");
                break;
            }
            agreed++;
            if (i == steps)
            {
                break;
            }
            a = session.Step(ta).FirstRow().ToArray();
            b = reference.Step(tb).FirstRow().ToArray();
        }
        Console.WriteLine($"compare:   {session.Backend} vs {reference.Backend}: {agreed} greedy tokens agree, " +
                          $"max logit diff {worst:G3} ({worstRel:P3} of the largest logit)");
    }

    /// <summary>One request: prefill, then greedy steps. Returns FNV-1a over
    /// the bit patterns of the prefill logits and the final step's logits —
    /// any numeric change to either path, however small, changes it.</summary>
    private static ulong RunOnce(DecodeSession session, int[] prompt, int steps, out double prefillSeconds, out double decodeSeconds)
    {
        ulong checksum = 14695981039346656037UL;
        void Mix(ReadOnlySpan<float> row)
        {
            foreach (var v in row)
            {
                checksum = (checksum ^ (uint)BitConverter.SingleToInt32Bits(v)) * 1099511628211UL;
            }
        }

        session.Reset();
        var sw = Stopwatch.StartNew();
        var logits = session.Prefill(prompt);
        prefillSeconds = sw.Elapsed.TotalSeconds;
        int next = ArgMax(logits.FirstRow());
        Mix(logits.FirstRow());
        sw.Restart();
        for (int i = 0; i < steps; i++)
        {
            logits = session.Step(next);
            next = ArgMax(logits.FirstRow());
        }
        decodeSeconds = sw.Elapsed.TotalSeconds;
        Mix(logits.FirstRow());
        return checksum ^ (ulong)next;
    }

    private static int ArgMax(ReadOnlySpan<float> row) => Sampler.ArgMax(row);
}
