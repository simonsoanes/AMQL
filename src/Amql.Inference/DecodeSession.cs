using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>
/// The token-level driver over a planned component: distinguishes prefill
/// (batch every position through every layer, cache all rows) from decode
/// (one token, cache rows, resume at position). Exposes the logits-session
/// contract <c>prefill → step → position</c> that sampling consumes.
/// </summary>
public sealed class DecodeSession
{
    private readonly GenericRuntime _runtime;

    public DecodeSession(ComponentOpPlan plan, OperandStore store)
    {
        _runtime = new GenericRuntime(plan, store);
    }

    public GenericRuntime Runtime => _runtime;

    /// <summary>Rows in the cache (equivalently, the next token's absolute
    /// position).</summary>
    public int Position => _runtime.Kv.Position;

    public Tensor2D LastLogits { get; private set; } = null!;

    /// <summary>
    /// Batched forward pass over a token sequence: every position through
    /// every layer, all key/value rows cached in position order. Returns
    /// the logits of the last position.
    /// </summary>
    public Tensor2D Prefill(int[] tokens)
    {
        if (tokens.Length == 0)
        {
            throw new ArgumentException("prefill requires at least one token", nameof(tokens));
        }
        if (_runtime.Kv.Position != 0)
        {
            throw new InvalidOperationException(
                $"cannot prefill a session that already holds {_runtime.Kv.Position} cached rows");
        }

        // Embed all positions; absolute positions are 0..T-1 for both
        // queries and keys (position-ordered cache).
        var hidden = _runtime.Embed(tokens);
        var positions = Enumerable.Range(0, tokens.Length).ToArray();

        for (int layer = 0; layer < _runtime.Plan.Layers.Count; layer++)
        {
            hidden = _runtime.RunLayerInternal(hidden, layer, positions, positions, appendKv: true);
        }

        // The logits-session contract (mirroring the reference) returns the
        // last position's row — the one `Step` continues from.
        var allLogits = _runtime.FinalNormAndHead(hidden);
        LastLogits = SliceLastRow(allLogits);
        return LastLogits;
    }

    private static Tensor2D SliceLastRow(Tensor2D logits)
    {
        var row = new float[logits.Cols];
        logits.Row(logits.Rows - 1).CopyTo(row);
        return new Tensor2D(row, 1, logits.Cols);
    }

    /// <summary>
    /// One-token forward pass at the current position: the new key/value
    /// rows are cached, then the token attends to every cached row. A
    /// session resumed from a prefill continues exactly where prefill's
    /// last logits left off.
    /// </summary>
    public Tensor2D Step(int token)
    {
        var position = this.Position;
        var hidden = _runtime.Embed(new[] { token });
        var queryPositions = new[] { position };
        var kvPositions = Enumerable.Range(0, position + 1).ToArray();

        for (int layer = 0; layer < _runtime.Plan.Layers.Count; layer++)
        {
            hidden = _runtime.RunLayerInternal(hidden, layer, queryPositions, kvPositions, appendKv: true);
        }

        LastLogits = _runtime.FinalNormAndHead(hidden);
        return LastLogits;
    }

    /// <summary>Resets the KV cache, starting a fresh session.</summary>
    public void Reset()
    {
        _runtime.Kv.Reset();
        LastLogits = null!;
    }
}