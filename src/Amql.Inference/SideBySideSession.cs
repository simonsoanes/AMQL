using System.Diagnostics;
using Amql.Entail;
using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// A program compiled into dense weights, for a runtime that can only run a
/// model. It holds them at f32 whatever precision its partner runs at. Where
/// nothing requires this form, <see cref="EntailSparseAdapter"/> does the same
/// work in doubles with a fraction of the memory.
/// </summary>
public sealed class EntailAdapterRunner : IEntailAdapter
{
    private readonly Vindex3Container? _container;
    private readonly OperandStore? _store;
    private readonly DecodeSession _session;

    private EntailAdapterRunner(EntailAdapterManifest manifest, DecodeSession session, Vindex3Container? container, OperandStore? store)
    {
        Manifest = manifest;
        _session = session;
        _container = container;
        _store = store;
    }

    public EntailAdapterManifest Manifest { get; }

    /// <summary>Loads an adapter of either form, refusing one built for another tokenizer.</summary>
    public static IEntailAdapter Load(string dir, string partnerTokenizerDir) =>
        File.Exists(Path.Combine(dir, EntailSparseAdapter.ProgramFile))
            ? EntailSparseAdapter.Open(dir, partnerTokenizerDir)
            : Open(dir, partnerTokenizerDir);

    public static EntailAdapterRunner Open(string containerDir, string partnerTokenizerDir)
    {
        var manifest = EntailAdapterManifest.Load(containerDir);
        manifest.RequirePartner(partnerTokenizerDir);
        var container = Vindex3Container.Open(containerDir);
        var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        return new EntailAdapterRunner(manifest, new DecodeSession(plan, store, null, WeightWorkingSet.ResidentF32), container, store);
    }

    public IEntailAdapter Sibling() => new EntailAdapterRunner(Manifest, _session.Sibling(), null, null);

    public void Reset() => _session.Reset();

    public EntailVerdict Prefill(int[] partnerTokens) =>
        EntailVerdict.Read(Manifest, _session.Prefill(partnerTokens.Select(Manifest.Row).ToArray()).Row(0));

    public EntailVerdict Step(int partnerToken) =>
        EntailVerdict.Read(Manifest, _session.Step(Manifest.Row(partnerToken)).Row(0));

    public void Dispose()
    {
        _store?.Dispose();
        _container?.Dispose();
    }
}

/// <summary>
/// Runs a partner model and its adapters side by side. Every token goes to all
/// of them, whoever chose it, so they always read the same text. The partner is
/// never changed: an adapter's verdict acts on the choice of the next token.
/// With no adapters this is the partner alone.
/// </summary>
public sealed class SideBySideSession
{
    private readonly DecodeSession _partner;
    private readonly IReadOnlyList<IEntailAdapter> _adapters;
    private readonly Stopwatch _partnerClock = new();
    private readonly Stopwatch _adapterClock = new();

    private double[]? _prior;
    private bool _fresh = true;

    public SideBySideSession(DecodeSession partner, IReadOnlyList<IEntailAdapter> adapters)
    {
        _partner = partner;
        _adapters = adapters;
    }

    /// <summary>
    /// Turns a prompt into its blank: the same conversation with the latest message's
    /// content replaced (see <see cref="ChatPrior"/>), or null when it has none.
    /// Set it when an adapter weighs a token against the partner's prior: a fresh prefill
    /// then runs the blank first, and its next-token distribution is that prior.
    /// </summary>
    public Func<int[], int[]?>? PriorPrompt { get; set; }

    /// <summary>The partner's next-token distribution for the blank of the last prompt, if
    /// one was measured.</summary>
    public IReadOnlyList<double>? Prior => _prior;

    /// <summary>The partner's logits for the next token, unchanged.</summary>
    public Tensor2D Logits { get; private set; } = null!;

    /// <summary>What the adapters require of the next token.</summary>
    public EntailVerdict Verdict { get; private set; } = EntailVerdict.Silence;

    /// <summary>Time spent in the partner and in the adapters so far.</summary>
    public (TimeSpan Partner, TimeSpan Adapters) Spent => (_partnerClock.Elapsed, _adapterClock.Elapsed);

    /// <summary>A session sharing these weights, for another thread. Run this
    /// one first: see <see cref="DecodeSession.Sibling"/>.</summary>
    public SideBySideSession Sibling() =>
        new(_partner.Sibling(), _adapters.Select(a => a.Sibling()).ToList()) { PriorPrompt = PriorPrompt };

    public void Reset()
    {
        _partner.Reset();
        foreach (var adapter in _adapters)
        {
            adapter.Reset();
        }
        Verdict = EntailVerdict.Silence;
        _prior = null;
        _fresh = true;
    }

    public void Prefill(int[] tokens)
    {
        _partnerClock.Start();
        if (_fresh && PriorPrompt is not null && _adapters.Any(a => a.Manifest.NeedsPrior)
            && PriorPrompt(tokens) is { } blank)
        {
            _prior = Softmax(_partner.Prefill(blank).Row(0));
            _partner.Reset();
        }
        _fresh = false;
        // The adapters read the same tokens and need nothing of the partner's: they run meanwhile.
        var ruling = _adapters.Count == 0 ? null : Task.Run(() => Rule(_adapters, a => a.Prefill(tokens)));
        Logits = _partner.Prefill(tokens);
        _partnerClock.Stop();
        _adapterClock.Start();
        Verdict = ruling?.GetAwaiter().GetResult() ?? EntailVerdict.Silence;
        _adapterClock.Stop();
    }

    public void Step(int token)
    {
        _partnerClock.Start();
        var ruling = _adapters.Count == 0 ? null : Task.Run(() => Rule(_adapters, a => a.Step(token)));
        Logits = _partner.Step(token);
        _partnerClock.Stop();
        _adapterClock.Start();   // what is left of the adapters' time once the partner is done
        Verdict = ruling?.GetAwaiter().GetResult() ?? EntailVerdict.Silence;
        _adapterClock.Stop();
    }

    /// <summary>Every adapter's verdict on a token, merged in their order. Each holds its own
    /// state, so more than one runs on cores of their own.</summary>
    public static EntailVerdict Rule(IReadOnlyList<IEntailAdapter> adapters, Func<IEntailAdapter, EntailVerdict> step)
    {
        if (adapters.Count <= 1)
        {
            return EntailVerdict.Merge(adapters.Select(step).ToList());
        }
        var verdicts = new EntailVerdict[adapters.Count];
        Parallel.For(0, adapters.Count, i => verdicts[i] = step(adapters[i]));
        return EntailVerdict.Merge(verdicts);
    }

    /// <summary>Chooses the next token greedily, and says who decided it.</summary>
    public (int Token, EntailTokenSource Source) Choose() => Verdict.Choose(Logits.Row(0), _prior);

    /// <summary>Draws the next token with <paramref name="config"/>, from what the adapters
    /// permit; a token they say or prefer comes first, as in <see cref="Choose()"/>.</summary>
    public (int Token, EntailTokenSource Source) Sample(SamplingConfig config, Random rng) =>
        Verdict.Choose(Logits.Row(0), _prior, row => Sampler.Sample(row, config, rng));

    private static double[] Softmax(ReadOnlySpan<float> logits)
    {
        float max = float.NegativeInfinity;
        foreach (float v in logits)
        {
            max = Math.Max(max, v);
        }
        var p = new double[logits.Length];
        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            sum += p[i] = Math.Exp(logits[i] - max);
        }
        for (int i = 0; i < p.Length; i++)
        {
            p[i] /= sum;
        }
        return p;
    }
}
