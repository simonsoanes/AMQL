using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// The token-level driver over a planned component. Execution mode follows
/// the plan: a fully softmax component prefills in one batched pass (every
/// position through every layer), while a component containing stateful
/// (linear-attention) layers prefills position-major — one token forward
/// through all layers at a time — because recurrent state must advance
/// sequentially. Both modes expose the same
/// <c>prefill → step → position</c> contract.
/// </summary>
public sealed class DecodeSession
{
    private readonly GenericRuntime _runtime;

    public DecodeSession(ComponentOpPlan plan, OperandStore store, WeightPatch? patch = null, WeightWorkingSet? workingSet = null)
    {
        _runtime = new GenericRuntime(plan, store, patch, workingSet);
    }

    public GenericRuntime Runtime => _runtime;

    /// <summary>Tokens consumed so far — the next token's absolute position.
    /// Independent of the KV cache shape (mixed plans may have no key/value
    /// rows at layer 0).</summary>
    public int Position => _runtime.SessionPosition;

    public Tensor2D LastLogits { get; private set; } = null!;

    /// <summary>
    /// Forward pass over a token sequence. Batched when every layer is
    /// stateless; position-major otherwise. Returns the logits of the last
    /// position.
    /// </summary>
    public Tensor2D Prefill(int[] tokens)
    {
        var lastHidden = RunPrefill(tokens);
        // The logits-session contract (mirroring the reference) returns the
        // last position's row — the one `Step` continues from.
        var allLogits = _runtime.FinalNormAndHead(lastHidden);
        LastLogits = SliceLastRow(allLogits);
        return LastLogits;
    }

    /// <summary>
    /// Forward pass over a token sequence that stops at the final norm: the
    /// post-norm hidden state of the LAST position (1 × hidden), the row a
    /// last-token pooled head reads. No vocabulary projection is computed.
    /// The session is left advanced, as after <see cref="Prefill"/>.
    /// </summary>
    public Tensor2D PrefillHidden(int[] tokens)
    {
        var hidden = RunPrefill(tokens);
        var last = new float[hidden.Cols];
        hidden.Row(hidden.Rows - 1).CopyTo(last);
        return _runtime.FinalNorm(new Tensor2D(last, 1, hidden.Cols));
    }

    private Tensor2D RunPrefill(int[] tokens)
    {
        if (tokens.Length == 0)
        {
            throw new ArgumentException("prefill requires at least one token", nameof(tokens));
        }
        if (_runtime.SessionPosition != 0)
        {
            throw new InvalidOperationException(
                $"cannot prefill a session that already consumed {_runtime.SessionPosition} tokens");
        }

        Tensor2D lastHidden;
        // GatedDeltaNet layers process a whole sequence in order inside the
        // layer (the conv and the recurrence both loop over positions), so a
        // plan whose only state is theirs runs layer-major like a softmax
        // plan: the projections become real GEMMs instead of one matrix-vector
        // product per token. Verified bit-identical to position-major on
        // Qwen3.5. Other stateful operators keep the position-major path.
        bool layerMajor = _runtime.Plan.Layers.All(l => l.Conv is null);
        if (!layerMajor)
        {
            // Position-major: one token through every layer at a time, so
            // recurrent state advances in sequence. The softmax layers'
            // KV rows accumulate per step exactly as they would in the
            // batched pass.
            lastHidden = null!;
            foreach (var token in tokens)
            {
                lastHidden = _runtime.StepForward(token);
            }
        }
        else
        {
            // Batched: every position through every layer, layer by layer.
            var hidden = _runtime.Embed(tokens);
            var positions = Enumerable.Range(0, tokens.Length).ToArray();
            for (int layer = 0; layer < _runtime.Plan.Layers.Count; layer++)
            {
                hidden = _runtime.RunLayerInternal(hidden, layer, positions, positions, appendKv: true);
            }
            _runtime.SessionPosition = tokens.Length;
            lastHidden = hidden;
        }
        return lastHidden;
    }

    private static Tensor2D SliceLastRow(Tensor2D logits)
    {
        var row = new float[logits.Cols];
        logits.Row(logits.Rows - 1).CopyTo(row);
        return new Tensor2D(row, 1, logits.Cols);
    }

    /// <summary>
    /// One-token forward pass at the current position: the new key/value
    /// rows are cached and recurrent state advances, then the token attends
    /// to its span. A session resumed from a prefill continues exactly
    /// where prefill's last logits left off.
    /// </summary>
    public Tensor2D Step(int token)
    {
        var hidden = _runtime.StepForward(token);
        LastLogits = _runtime.FinalNormAndHead(hidden);
        return LastLogits;
    }

    /// <summary>Resets the KV cache and the recurrent state, starting a
    /// fresh session.</summary>
    public void Reset()
    {
        _runtime.Kv.Reset();
        _runtime.ResetSession();
        LastLogits = null!;
    }
}