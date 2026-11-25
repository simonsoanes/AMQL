using System.Runtime.InteropServices;
using Amql.Vindex3;

namespace Amql.Inference;

public sealed partial class GpuEngine
{
    // A drafter that reads the model's own caches (null without one).
    private readonly DrafterDevice? _drafter;

    /// <summary>
    /// <see cref="DrafterPlan"/> on the device, and the scratch of its one-row passes. It has
    /// no state: what it needs of a sequence is the final-normed hidden state of the last row
    /// run (<c>MtpPending</c>), so it drafts again after any prompt or verify.
    /// </summary>
    private sealed class DrafterDevice
    {
        public required DrafterPlan Plan;
        public required DeviceMatrix Pre, Post, Head;
        public required (IntPtr W, IntPtr B, float Off, float Eps, int Kind) Norm;
        public required LayerOf[] Layers;
        public required int Act;
        public IntPtr Cat, X, Proj, Q, Ctx, Ffn, Branch, H, Raw, Pos, WinPos;

        public sealed class LayerOf
        {
            public required DrafterLayer Plan;
            public required (IntPtr W, IntPtr B, float Off, float Eps, int Kind) PreAttn, PostAttn, PreFfn, PostFfn, QNorm;
            public required DeviceMatrix Q, O, Up, Down;
            public required float OutScale;
        }

        public static DrafterDevice? TryUpload(GpuEngine e, WeightLoader weights, DrafterPlan? plan)
        {
            if (plan is null || plan.FinalNorm.Kind != NormType.RmsNorm || plan.VocabSize != e._vocab)
            {
                return null;
            }
            int h = plan.HiddenSize, hidden = e._hidden, inter = plan.IntermediateSize;
            var layers = new LayerOf[plan.Layers.Count];
            int qWidth = 1;
            for (int i = 0; i < layers.Length; i++)
            {
                var l = plan.Layers[i];
                var attn = e._plan.Layers[l.Source].Attention!;
                int qDim = l.Heads * attn.HeadDim;
                qWidth = Math.Max(qWidth, qDim);
                e._maxHeads = Math.Max(e._maxHeads, l.Heads);
                layers[i] = new LayerOf
                {
                    Plan = l,
                    PreAttn = e.Norm(weights, l.PreAttention), PostAttn = e.Norm(weights, l.PostAttention),
                    PreFfn = e.Norm(weights, l.PreFfn), PostFfn = e.Norm(weights, l.PostFfn), QNorm = e.Norm(weights, l.QNorm),
                    Q = e.Matrix(weights, l.QProj, qDim, h, project: true),
                    O = e.Matrix(weights, l.OProj, h, qDim, project: true),
                    Up = e.Concat(weights, h, (l.Gate, inter), (l.Up, inter)),
                    Down = e.Matrix(weights, l.Down, h, inter, project: true),
                    OutScale = l.OutputScale is { } s ? weights.Vector(s, 1)[0] : 1f,
                };
            }
            return new DrafterDevice
            {
                Plan = plan,
                Pre = e.Matrix(weights, plan.PreProjection, h, 2 * hidden, project: true),
                Post = e.Matrix(weights, plan.PostProjection, hidden, h, project: true),
                Head = e.Matrix(weights, plan.Head, plan.VocabSize, h, project: true),
                Norm = e.Norm(weights, plan.FinalNorm),
                Layers = layers,
                Act = ActivationCode(plan.Activation),
                Cat = e.Alloc(2L * hidden * 4), X = e.Alloc(h * 4L), Proj = e.Alloc(qWidth * 4L), Q = e.Alloc(qWidth * 4L), Ctx = e.Alloc(qWidth * 4L),
                Ffn = e.Alloc(inter * 4L), Branch = e.Alloc(h * 4L), H = e.Alloc(h * 4L), Raw = e.Alloc(Math.Max(qWidth, 2 * inter) * 4L),
                Pos = e.Alloc(16), WinPos = e.Alloc(16),
            };
        }
    }

    private const int DraftTokenAt = 2 * MaxDrafts * 4;

    /// <summary>The drafter's probability of each of the last drafts, given those before it.</summary>
    public IReadOnlyList<float> LastDraftShares => DraftShares;

    private float[] DraftShares = [];

    /// <summary>How many of the last <see cref="Draft"/>'s drafts are worth a verify's row: those
    /// the drafter gives at least <paramref name="floor"/> of holding along with all before them.</summary>
    public int WorthVerifying(float floor)
    {
        float held = 1f;
        int n = 0;
        while (n < DraftShares.Length && (held *= DraftShares[n]) >= floor)
        {
            n++;
        }
        return n;
    }

    /// <summary>The most drafts a step that pay with the drafter there is.</summary>
    public int DraftsThatPay => _drafter is not null ? 4 : 3;

    // Passes captured for one sequence, by what they do: verifies of n rows, runs of n drafts.
    private readonly Dictionary<int, IntPtr> _passes = new();
    private GpuSequence? _passesFor;

    /// <summary>Runs a pass whose launches depend on nothing but the sequence and the key:
    /// captured the first time, replayed after.</summary>
    private void Pass(GpuSequence seq, int key, Action launch)
    {
        if (!ReferenceEquals(_passesFor, seq))
        {
            DropPasses();
            _passesFor = seq;
        }
        if (!_passes.TryGetValue(key, out var pass))
        {
            Check(Native.amqe_pass_begin(_engine), "capture");
            try
            {
                launch();
            }
            catch
            {
                Native.amqe_pass_end(_engine, out var broken);
                Native.amqe_pass_free(broken);
                throw;
            }
            Check(Native.amqe_pass_end(_engine, out pass), "capture");
            _passes[key] = pass;
        }
        Check(Native.amqe_pass_run(_engine, pass), "pass");
    }

    private void DropPasses()
    {
        _passesFor = null;
        if (_passes is not { Count: > 0 })
        {
            return;
        }
        Native.amqe_sync(_engine);
        foreach (var pass in _passes.Values)
        {
            Native.amqe_pass_free(pass);
        }
        _passes.Clear();
    }

    /// <summary><paramref name="count"/> drafts from the drafter that reads the model's caches, after
    /// the token on the host, into <c>_drafts</c>.</summary>
    private void DraftShared(GpuSequence seq, int count)
    {
        var d = _drafter!;
        // The last row the caches hold: the query stands one position after it.
        Check(Native.amqe_copy_async(_engine, d.Pos, seq.Pos, 16, 3), "draft position");
        Check(Native.amqe_advance(_engine, d.Pos, -1), "draft position");
        if (seq.WinPos != IntPtr.Zero)
        {
            Check(Native.amqe_copy_async(_engine, d.WinPos, seq.WinPos, 16, 3), "draft position");
            Check(Native.amqe_advance(_engine, d.WinPos, -1), "draft position");
        }
        Check(Native.amqe_copy_async(_engine, _mtpTok, _hostDrafts + DraftTokenAt, 4, 1), "draft");
        for (int i = 0; i < count; i++)
        {
            DrafterPass(seq, _mtpTok + i * 4, i == 0 ? seq.MtpPending : _mtpH, _drafts + i * 4, _mtpTok + (i + 1) * 4, more: i + 1 < count);
        }
    }

    private void DrafterPass(GpuSequence seq, IntPtr token, IntPtr hidden, IntPtr draftTo, IntPtr nextToken, bool more)
    {
        var d = _drafter!;
        int h = d.Plan.HiddenSize, inter = d.Plan.IntermediateSize;
        Check(Native.amqe_embed(_engine, _embed.Ptr, _embed.Type, _embed.Scales, _embed.RowScale, token, 1, _hidden, _embedScale, _hasEmbedScale ? 1 : 0, d.Cat), "draft embed");
        Check(Native.amqe_copy_async(_engine, d.Cat + _hidden * 4, hidden, (nuint)(_hidden * 4), 3), "draft hidden");
        DraftLinear(d, d.Pre, d.Cat, 2 * _hidden, d.X);
        foreach (var l in d.Layers)
        {
            var src = _layers[l.Plan.Source];
            var attn = src.Plan.Attention!;
            int qDim = l.Plan.Heads * attn.HeadDim;
            bool windowed = Windowed(l.Plan.Source);
            DraftLinear(d, l.Q, d.X, h, d.Proj, norm: l.PreAttn);
            Check(Native.amqe_attn_query(_engine, d.Proj, l.Plan.Heads, attn.HeadDim, d.Q, PosOf(seq, l.Plan.Source), l.QNorm.W, l.QNorm.Off, l.QNorm.Eps,
                InvFreq(src, seq), src.RopePairs, src.RopeScale, attn.RopeInterleaved ? 1 : 0, src.InvFreq != IntPtr.Zero ? 1 : 0), "draft query");
            Check(Native.amqe_attention(_engine, d.Q, seq.K[l.Plan.Source], seq.V[l.Plan.Source], IntPtr.Zero, d.Ctx, windowed ? d.WinPos : d.Pos,
                1, l.Plan.Heads, attn.NumKvHeads, attn.HeadDim, (float)attn.ScoreScale, 0f, l.Plan.Window is { } w ? (int)w + 1 : 0,
                _attnPart, Splits(seq, l.Plan.Source)), "draft attention");
            DraftLinear(d, l.O, d.Ctx, qDim, d.Branch);
            Check(Native.amqe_norm_join(_engine, d.X, d.Branch, 1, h, l.PostAttn.W, l.PostAttn.Off, l.PostAttn.Eps, 1f, 1f), "draft residual");
            DraftLinear(d, l.Up, d.X, h, d.Ffn, act: d.Act, glu: true, n: inter, norm: l.PreFfn);
            DraftLinear(d, l.Down, d.Ffn, inter, d.Branch);
            Check(Native.amqe_norm_join(_engine, d.X, d.Branch, 1, h, l.PostFfn.W, l.PostFfn.Off, l.PostFfn.Eps, 1f, l.OutScale), "draft residual");
        }
        Check(Native.amqe_norm(_engine, d.X, h, d.H, h, 1, h, d.Norm.W, d.Norm.Off, d.Norm.B, d.Norm.Eps, d.Norm.Kind), "draft norm");
        if (more)
        {
            DraftLinear(d, d.Post, d.H, h, _mtpH);
        }
        DraftLinear(d, d.Head, d.H, h, _mtpLogits, n: DraftVocab);
        Check(Native.amqe_draft_pick(_engine, _mtpLogits, DraftVocab, draftTo, nextToken, draftTo + MaxDrafts * 4), "draft pick");
    }

    private void DraftLinear(DrafterDevice d, DeviceMatrix w, IntPtr x, int ldx, IntPtr y, int act = 0, bool glu = false, int n = -1,
        (IntPtr W, IntPtr B, float Off, float Eps, int Kind)? norm = null)
    {
        int outputs = n < 0 ? w.Rows : n;
        var nm = norm ?? default;
        Check(Native.amqe_linear(_engine, w.Ptr, w.Type, outputs, w.Cols, x, ldx, 1, y, outputs, IntPtr.Zero, IntPtr.Zero, outputs, 1f,
            act, glu ? 1 : 0, d.Raw, nm.W, nm.Off, nm.B, nm.Eps, nm.Kind, w.Scales, w.RowScale), "draft linear");
    }

    /// <summary>For tests: the drafter's logits for each of <paramref name="count"/> drafts after
    /// <paramref name="token"/>, and the drafts.</summary>
    public (int[] Drafts, float[][] Logits) DraftLogits(GpuSequence seq, int token, int count)
    {
        var logits = new float[count][];
        var drafts = new int[count];
        int vocab = DraftVocab;
        DraftVocab = _vocab;
        try
        {
            for (int i = 0; i < count; i++)
            {
                Marshal.WriteInt32(_hostDrafts, DraftTokenAt, token);
                DraftShared(seq, i + 1);
                logits[i] = new float[_vocab];
                Download(logits[i], _mtpLogits);
                var host = new int[i + 1];
                Download(host, _drafts);
                drafts[i] = host[i];
            }
        }
        finally
        {
            DraftVocab = vocab;
        }
        return (drafts, logits);
    }
}
