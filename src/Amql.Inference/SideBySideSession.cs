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

    public SideBySideSession(DecodeSession partner, IReadOnlyList<IEntailAdapter> adapters)
    {
        _partner = partner;
        _adapters = adapters;
    }

    /// <summary>The partner's logits for the next token, unchanged.</summary>
    public Tensor2D Logits { get; private set; } = null!;

    /// <summary>What the adapters require of the next token.</summary>
    public EntailVerdict Verdict { get; private set; } = EntailVerdict.Silence;

    /// <summary>Time spent in the partner and in the adapters so far.</summary>
    public (TimeSpan Partner, TimeSpan Adapters) Spent => (_partnerClock.Elapsed, _adapterClock.Elapsed);

    /// <summary>A session sharing these weights, for another thread. Run this
    /// one first: see <see cref="DecodeSession.Sibling"/>.</summary>
    public SideBySideSession Sibling() => new(_partner.Sibling(), _adapters.Select(a => a.Sibling()).ToList());

    public void Reset()
    {
        _partner.Reset();
        foreach (var adapter in _adapters)
        {
            adapter.Reset();
        }
        Verdict = EntailVerdict.Silence;
    }

    public void Prefill(int[] tokens)
    {
        _partnerClock.Start();
        Logits = _partner.Prefill(tokens);
        _partnerClock.Stop();
        _adapterClock.Start();
        Verdict = EntailVerdict.Merge(_adapters.Select(a => a.Prefill(tokens)).ToList());
        _adapterClock.Stop();
    }

    public void Step(int token)
    {
        _partnerClock.Start();
        Logits = _partner.Step(token);
        _partnerClock.Stop();
        _adapterClock.Start();
        Verdict = EntailVerdict.Merge(_adapters.Select(a => a.Step(token)).ToList());
        _adapterClock.Stop();
    }

    /// <summary>Chooses the next token greedily, and says who decided it.</summary>
    public (int Token, EntailTokenSource Source) Choose() => Verdict.Choose(Logits.Row(0));
}
