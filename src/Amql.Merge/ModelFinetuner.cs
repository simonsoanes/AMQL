using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>Outcome of one fine-tune run: what was adjusted and the patch
/// that carries the deltas.</summary>
public sealed record FineTuneReport(
    string PatchPath,
    string ComponentId,
    string HeadObjectId,
    string HeadTensorName,
    int VocabSize,
    int HiddenSize,
    int Pairs,
    int Steps,
    bool HeadReusesEmbedding,
    double LearningRate,
    int Epochs,
    IReadOnlyList<string> Notes);

/// <summary>
/// Supervised fine-tuning via teacher-forced output-head adaptation:
/// given (prompt, completion) text pairs, runs the model forward and
/// accumulates per-token head deltas that push the output toward the
/// desired completions. The result is a standard AMQL weight-patch file
/// — no gradient infrastructure, no autograd, deterministic given the
/// same inputs. The container itself is never rewritten.
/// </summary>
public static class ModelFinetuner
{
    /// <summary>
    /// Runs fine-tuning over a TSV data file (one <c>prompt\tcompletion</c>
    /// per line) and writes the accumulated head deltas to
    /// <c>outPatchPath</c>.
    /// </summary>
    public static FineTuneReport FineTune(
        Vindex3Container container,
        string componentId,
        string dataPath,
        string outPatchPath,
        HfTokenizer tokenizer,
        double lr = 1e-4,
        int epochs = 1,
        WeightPatch? basePatch = null)
    {
        if (lr <= 0)
        {
            throw new MergeException("learning rate must be positive");
        }
        if (epochs <= 0)
        {
            throw new MergeException("epochs must be positive");
        }

        var lines = File.ReadAllLines(dataPath);
        var pairs = ParsePairs(lines, dataPath);
        if (pairs.Count == 0)
        {
            throw new MergeException($"'{dataPath}' contains no valid training pairs");
        }

        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, componentId, store);
        if (plan.Output is null)
        {
            throw new MergeException(
                $"component '{componentId}' carries no output head — fine-tuning needs one");
        }

        int vocabSize = plan.Output.VocabSize;
        int hiddenSize = plan.Output.HiddenSize;
        string headObjectId = plan.Output.Projection.ObjectId;
        string headTensorName = plan.Output.Projection.TensorName;
        bool headReusesEmbedding = plan.Output.ReusesEmbedding;

        // Accumulate deltas in double precision to avoid cancellation.
        var accum = new double[vocabSize * hiddenSize];
        int totalSteps = 0;
        var notes = new List<string>();

        // Resolve the final-norm weight once (same for every forward).
        var finalNormW = store.ResolveWidened(
            plan.FinalNorm.Weight).Values;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            foreach (var (prompt, completion) in pairs)
            {
                var promptIds = tokenizer.EncodeToIds(prompt).ToArray();
                var completionIds = tokenizer.EncodeToIds(completion).ToArray();
                if (promptIds.Length == 0 || completionIds.Length == 0)
                {
                    continue;
                }

                // Fresh runtime per pair — the KV cache and recurrent
                // state are local to one example.
                var rt = new GenericRuntime(plan, store, basePatch);

                // ── prefill the prompt ──────────────────────────────────
                var hidden = rt.Embed(promptIds);
                var positions = Enumerable.Range(0, promptIds.Length).ToArray();
                for (int layer = 0; layer < plan.Layers.Count; layer++)
                {
                    hidden = rt.RunLayerInternal(
                        hidden, layer, positions, positions, appendKv: true);
                }
                int pos = promptIds.Length; // position of the next token

                // ── teacher-force each completion token ─────────────────
                foreach (var targetId in completionIds)
                {
                    if (targetId < 0 || targetId >= vocabSize)
                    {
                        notes.Add(
                            $"token {targetId} outside vocabulary [0, {vocabSize}) — skipped");
                        break;
                    }

                    // Clone before applying the in-place final norm.
                    var hNorm = hidden.Clone();
                    Norms.ApplyInPlace(
                        hNorm, plan.FinalNorm.Kind, plan.FinalNorm.Eps,
                        finalNormW, plan.FinalNorm.WeightOffset);

                    // Δhead[target, :] += lr · h_norm
                    var row = hNorm.Row(hNorm.Rows - 1);
                    int baseOffset = targetId * hiddenSize;
                    for (int d = 0; d < hiddenSize; d++)
                    {
                        accum[baseOffset + d] += lr * row[d];
                    }

                    totalSteps++;

                    // Step forward with the target token (teacher forcing).
                    // Embed one token, run all layers, append KV.
                    hidden = rt.Embed(new[] { targetId });
                    var qpos = new[] { pos };
                    var kvpos = Enumerable.Range(0, pos + 1).ToArray();
                    for (int layer = 0; layer < plan.Layers.Count; layer++)
                    {
                        hidden = rt.RunLayerInternal(
                            hidden, layer, qpos, kvpos, appendKv: true);
                    }
                    pos++;
                }
            }
        }

        if (totalSteps == 0)
        {
            throw new MergeException(
                "no training steps were taken — check that the data file contains " +
                "valid prompt/completion pairs whose tokens are in vocabulary");
        }

        // Build the patch entry: one dense F32 delta array for the head
        // tensor, shaped [vocabSize, hiddenSize].
        var delta = new float[vocabSize * hiddenSize];
        bool hasNonZero = false;
        for (int i = 0; i < delta.Length; i++)
        {
            if (accum[i] != 0.0)
            {
                delta[i] = (float)accum[i];
                hasNonZero = true;
            }
        }

        if (!hasNonZero)
        {
            throw new MergeException(
                "accumulated deltas are all zero — the learning rate may be too small " +
                "or the model already predicts the targets perfectly");
        }

        var entry = new WeightPatchEntry(
            headObjectId, headTensorName,
            new long[] { vocabSize, hiddenSize }, delta);

        WeightPatch.Save(outPatchPath, new[] { entry }, container.Index.Model);

        string reuseNote = headReusesEmbedding
            ? "the head reuses the embedding table — the embedding is also adjusted"
            : string.Empty;
        var allNotes = new List<string>();
        if (reuseNote.Length > 0)
        {
            allNotes.Add(reuseNote);
        }
        allNotes.AddRange(notes);

        return new FineTuneReport(
            outPatchPath, componentId, headObjectId, headTensorName,
            vocabSize, hiddenSize, pairs.Count, totalSteps,
            headReusesEmbedding, lr, epochs, allNotes);
    }

    /// <summary>
    /// Parses a TSV file: each line is <c>prompt\tcompletion</c>.
    /// Blank lines and lines starting with '#' are skipped.
    /// Lines without a tab are treated as prompt-only (skipped).
    /// </summary>
    private static List<(string Prompt, string Completion)> ParsePairs(
        string[] lines, string path)
    {
        var pairs = new List<(string, string)>();
        foreach (var raw in lines)
        {
            var line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#'))
            {
                continue;
            }
            int tab = line.IndexOf('\t');
            if (tab <= 0)
            {
                continue;
            }
            string prompt = line[..tab].Trim();
            string completion = line[(tab + 1)..].Trim();
            if (prompt.Length > 0 && completion.Length > 0)
            {
                pairs.Add((prompt, completion));
            }
        }
        return pairs;
    }
}