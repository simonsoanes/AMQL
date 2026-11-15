namespace Amql.Inference;

/// <summary>
/// A persistent set of compute threads for the hot kernels. A single-token
/// decode issues a few hundred small parallel loops per token (every
/// projection of every layer), and <see cref="Parallel.For(int,int,Action{int})"/>
/// pays thread-pool dispatch and wake-up on each one — at decode sizes that
/// overhead is a large share of the token. These workers spin briefly
/// between loops, so the next loop of the same token starts in
/// microseconds, and park on a monitor once the process goes quiet.
/// <para>
/// The pool changes scheduling only: a loop body computes the same values
/// whichever thread runs which index, so results are identical to the
/// sequential order. One loop runs at a time; a caller that finds the pool
/// busy (another session on another thread) or that is itself running
/// inside a pool loop falls back to <see cref="Parallel.For(int,int,Action{int})"/>.
/// </para>
/// </summary>
public static class KernelPool
{
    /// <summary>How long an idle worker spins for the next loop before it
    /// parks. Decode loops follow each other within microseconds; anything
    /// longer is a pause in the work and not worth burning a core on.</summary>
    private const int SpinIterations = 200_000;

    private static readonly object Gate = new();
    private static readonly int WorkerCount = Math.Max(0, ComputeBudget.Cores - 1);
    private static Thread[]? _threads;

    /// <summary>One loop: its own counters, so a worker that wakes late and
    /// picks up a finished job finds no indices left instead of mixing two
    /// loops' counters.</summary>
    private sealed class Job(Action<int> body, int count)
    {
        public readonly Action<int> Body = body;
        public readonly int Count = count;
        // The claim counter and the completion counter are each hit by every
        // thread; on separate cache lines they do not invalidate each other.
        public PaddedInt Next;
        public PaddedInt Remaining = new() { Value = count };
        public Exception? Failure;
    }

    /// <summary>An int alone on its cache line (and its neighbours').</summary>
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 128)]
    private struct PaddedInt
    {
        [System.Runtime.InteropServices.FieldOffset(64)]
        public int Value;
    }

    private static volatile Job? _job;
    private static volatile int _generation;
    private static int _busy;

    [ThreadStatic]
    private static bool _insideLoop;

    /// <summary>Total threads a loop can use (workers plus the caller).</summary>
    public static int Parallelism => WorkerCount + 1;

    /// <summary>Runs <paramref name="body"/> for every index in
    /// [0, <paramref name="count"/>), spread over the workers and the calling
    /// thread; returns once every index has completed.</summary>
    /// <summary>When set, <see cref="For"/> accumulates its wall time in
    /// <see cref="InsideTicks"/> and counts <see cref="Loops"/> — the
    /// <c>bench --profile</c> split between parallel and serial time. Off,
    /// it costs one flag test per loop.</summary>
    public static bool Profiling { get; set; }

    /// <summary>Stopwatch ticks spent inside <see cref="For"/> while <see cref="Profiling"/>.</summary>
    public static long InsideTicks;

    /// <summary>Loops run while <see cref="Profiling"/>.</summary>
    public static long Loops;

    public static void For(int count, Action<int> body)
    {
        if (!Profiling)
        {
            Run(count, body);
            return;
        }
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        try
        {
            Run(count, body);
        }
        finally
        {
            InsideTicks += System.Diagnostics.Stopwatch.GetTimestamp() - start;
            Loops++;
        }
    }

    private static void Run(int count, Action<int> body)
    {
        if (count <= 0)
        {
            return;
        }
        if (count == 1 || WorkerCount == 0)
        {
            for (int i = 0; i < count; i++)
            {
                body(i);
            }
            return;
        }
        if (_insideLoop || Interlocked.CompareExchange(ref _busy, 1, 0) != 0)
        {
            Parallel.For(0, count, body);
            return;
        }

        try
        {
            EnsureStarted();
            var job = new Job(body, count);
            _job = job;
            lock (Gate)
            {
                _generation++;
                Monitor.PulseAll(Gate);
            }

            _insideLoop = true;
            try
            {
                Drain(job);
            }
            finally
            {
                _insideLoop = false;
            }

            var spin = new SpinWait();
            while (Volatile.Read(ref job.Remaining.Value) > 0)
            {
                spin.SpinOnce(-1);
            }
            _job = null;
            if (job.Failure is { } failure)
            {
                throw new AggregateException(failure);
            }
        }
        finally
        {
            Volatile.Write(ref _busy, 0);
        }
    }

    private static void Drain(Job job)
    {
        // Completions are reported once, after this thread's last claim, so a
        // chunk costs one contended atomic (the claim) rather than two.
        int done = 0;
        try
        {
            while (true)
            {
                int index = Interlocked.Increment(ref job.Next.Value) - 1;
                if (index >= job.Count)
                {
                    return;
                }
                try
                {
                    job.Body(index);
                }
                catch (Exception e)
                {
                    Interlocked.CompareExchange(ref job.Failure, e, null);
                }
                done++;
            }
        }
        finally
        {
            if (done > 0)
            {
                Interlocked.Add(ref job.Remaining.Value, -done);
            }
        }
    }

    private static void EnsureStarted()
    {
        if (_threads is not null)
        {
            return;
        }
        lock (Gate)
        {
            if (_threads is not null)
            {
                return;
            }
            var threads = new Thread[WorkerCount];
            for (int i = 0; i < threads.Length; i++)
            {
                threads[i] = new Thread(WorkerLoop)
                {
                    IsBackground = true,
                    Name = $"amql-kernel-{i}",
                };
                threads[i].Start();
            }
            _threads = threads;
        }
    }

    private static void WorkerLoop()
    {
        _insideLoop = true;
        int seen = 0;
        while (true)
        {
            int spins = 0;
            while (_generation == seen && spins < SpinIterations)
            {
                Thread.SpinWait(8);
                spins++;
            }
            if (_generation == seen)
            {
                lock (Gate)
                {
                    while (_generation == seen)
                    {
                        Monitor.Wait(Gate);
                    }
                }
            }
            seen = _generation;
            if (_job is { } job)
            {
                Drain(job);
            }
        }
    }
}
