using System.Security.Cryptography;
using System.Text.Json;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>Where the collected continuation pairs landed, plus the
/// determinism marker.</summary>
public sealed record MtpPairData(
    string Directory,
    int FitCount,
    int GateCount,
    int Hidden,
    string DeterminismHashHex,
    IReadOnlyList<string> Notes);

/// <summary>
/// Phase 0 of the calculated MTP drafter: the <b>data contract</b>. One
/// deterministic forward pass over the frozen model collects, per
/// position t:
///
/// <list type="bullet">
/// <item><c>x_t ∈ R^{2h}</c> — the drafter's input:
/// concat(pre_fc_norm_hidden(h_t), pre_fc_norm_embedding(e_{t+1}))
/// where h_t is the model's FINAL-normed hidden and e_{t+1} the shared
/// embedding of the next token (the pre-fc halves normalised with the
/// drafter's pre-fc norms — the final-norm copy, or the model's own final
/// norm before the module exists);</item>
/// <item><c>y_t ∈ R^h</c> — the regression target: the model's
/// PRE-final-norm residual hidden at t+1 (what the trunk input should
/// approximate);</item>
/// <item>the token window (including token_{t+2} for the held-out gate).</item>
/// </list>
///
/// The first <c>fitPositions</c> positions are the fit split; the last
/// <c>gatePositions</c> are held out for acceptance (never used by the
/// fit, per the design). The whole run has no randomness, so re-running
/// produces byte-identical sinks — the determinism hash is the marker.
/// </summary>
public static class PairCollector
{
    public static MtpPairData Collect(
        string containerDir,
        string outDir,
        IReadOnlyList<int> tokens,
        int fitPositions,
        int gatePositions)
    {
        if (Directory.Exists(outDir))
        {
            throw new MergeException($"pair sink '{outDir}' already exists");
        }
        if (fitPositions < 1 || gatePositions < 1)
        {
            throw new MergeException($"fit {fitPositions} and gate {gatePositions} positions must both be positive");
        }
        int total = fitPositions + gatePositions + 2; // +2: e at t+1 and the final token_{t+2}
        if (tokens.Count < total)
        {
            throw new MergeException($"need at least {total} corpus tokens, got {tokens.Count}");
        }

        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        int hidden = plan.HiddenSize;

        // The pre-fc norms: the drafter's copies when the module EXISTS (is
        // materialised — a carried mtp.stack has no tensors to probe), else
        // the model's final norm (what generate-mtp would copy).
        float[] preFcWeight;
        float[] finalNormWeight = Widen(store, plan.FinalNorm.Weight);
        using (var mtp = container.CreateOperandStore())
        {
            bool moduleMaterialised = container.Graph?.Objects
                    .FirstOrDefault(o => o.Id == "mtp.stack") is { Representations.Count: > 0 } &&
                mtp.ContainsTensor("mtp.stack", "pre_fc_norm_hidden.weight");
            preFcWeight = moduleMaterialised
                ? Widen(mtp, new OperandRef("mtp.stack", "pre_fc_norm_hidden.weight"))
                : finalNormWeight;
        }

        var dense = new GenericRuntime(plan, store);
        var table = dense.Weights.Matrix(plan.Embedding!.Table, plan.Embedding.VocabSize, plan.Embedding.HiddenSize);

        int nFit = fitPositions;
        using var sink = new PairSink(outDir, nFit, hidden);

        // One forward pass; y at position t is the hidden produced AFTER
        // consuming token t (i.e. the residual h(t+1) for x_t).
        for (int t = 0; t < total; t++)
        {
            var h = dense.StepForward(tokens[t]);
            if (t > 0 && t - 1 < nFit)
            {
                sink.WriteTarget(h.Data); // y[t-1] = pre-final-norm residual at t
            }
            if (t < nFit)
            {
                var n = (float[])h.Data.Clone();
                Norms.ApplyInPlace(new Tensor2D(n, 1, hidden), plan.FinalNorm.Kind,
                    plan.FinalNorm.Eps, finalNormWeight, plan.FinalNorm.WeightOffset);
                var halfH = (float[])n.Clone();
                Norms.ApplyInPlace(new Tensor2D(halfH, 1, hidden), plan.FinalNorm.Kind,
                    plan.FinalNorm.Eps, preFcWeight, plan.FinalNorm.WeightOffset);
                var halfE = (float[])TensorOps.GatherRows(table, new[] { tokens[t + 1] }).Data.Clone();
                Norms.ApplyInPlace(new Tensor2D(halfE, 1, hidden), plan.FinalNorm.Kind,
                    plan.FinalNorm.Eps, preFcWeight, plan.FinalNorm.WeightOffset);
                sink.WriteInput(halfH, halfE);
            }
        }

        var window = tokens.Take(total).ToArray();
        sink.Finish(window, container.Index.Model, gatePositions);
        return new MtpPairData(
            outDir,
            nFit,
            gatePositions,
            hidden,
            sink.DeterminismHashHex,
            new List<string>
            {
                $"collected {nFit} fit + {gatePositions} held-out pairs over {total} tokens; " +
                $"determinism hash {sink.DeterminismHashHex}",
            });
    }

    private static float[] Widen(OperandStore store, OperandRef r)
    {
        var resolution = store.Resolve(r.ObjectId, r.TensorName);
        return BitPattern.WidenToF32(resolution.Dtype, resolution.Payload);
    }

    /// <summary>The pair sink: f32 row-major arrays + a JSON manifest + a
    /// SHA-256 determinism marker over the exact bytes written.</summary>
    private sealed class PairSink : IDisposable
    {
        private readonly string _dir;
        private readonly int _nFit;
        private readonly int _hidden;
        private int _written;
        private readonly IncrementalHash _hash;
        private readonly List<string> _noteTargets = new();

        public PairSink(string dir, int nFit, int hidden)
        {
            _dir = dir;
            _nFit = nFit;
            _hidden = hidden;
            _hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            Directory.CreateDirectory(dir);
        }

        public void WriteInput(float[] halfH, float[] halfE)
        {
            var row = new byte[2 * _hidden * 4];
            Buffer.BlockCopy(halfH, 0, row, 0, _hidden * 4);
            Buffer.BlockCopy(halfE, 0, row, _hidden * 4, _hidden * 4);
            Append("fit.x.bin", row);
        }

        public void WriteTarget(float[] hidden)
        {
            var row = new byte[_hidden * 4];
            Buffer.BlockCopy(hidden, 0, row, 0, _hidden * 4);
            Append("fit.y.bin", row);
        }

        private void Append(string fileName, byte[] row)
        {
            using var file = File.OpenWrite(Path.Combine(_dir, fileName));
            file.Seek(0, SeekOrigin.End);
            file.Write(row);
            _hash.AppendData(row);
            _written++;
        }

        public void Finish(int[] window, string model, int gateCount)
        {
            var tokens = new byte[window.Length * 4];
            Buffer.BlockCopy(window, 0, tokens, 0, tokens.Length);
            using (var file = File.OpenWrite(Path.Combine(_dir, "tokens.bin")))
            {
                file.Write(tokens);
            }
            _hash.AppendData(tokens);

            var manifest = new
            {
                version = 1,
                model,
                hidden = _hidden,
                fit_count = _nFit,
                gate_count = gateCount,
                split = new[] { 0, _nFit, _nFit + gateCount },
                dtype = "F32",
                x_columns = 2 * _hidden,
                y_columns = _hidden,
            };
            File.WriteAllText(Path.Combine(_dir, "manifest.json"),
                JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }));
            _hash.AppendData(System.Text.Encoding.UTF8.GetBytes(File.ReadAllText(Path.Combine(_dir, "manifest.json"))));

            DeterminismHashHex = Convert.ToHexStringLower(_hash.GetHashAndReset());
        }

        public string DeterminismHashHex { get; private set; } = string.Empty;

        public void Dispose()
        {
            _hash.Dispose();
        }
    }
}