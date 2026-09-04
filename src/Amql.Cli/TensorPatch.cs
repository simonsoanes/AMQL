using System.Text.Json;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Cli;

/// <summary>The edit a <c>change-tensor</c> call applies to one cell of a
/// weight.</summary>
public enum TensorEditOp
{
    Set,
    Add,
    Scale,
}

/// <summary>Outcome of one manual tensor edit: the resulting patch entry
/// list (deltas over the container's base weights) plus the before/after
/// values and whether the edit left anything to record.</summary>
public sealed record EditResult(
    IReadOnlyList<WeightPatchEntry> Entries,
    string ObjectId,
    string TensorName,
    long[] Shape,
    string DtypeLabel,
    long FlatIndex,
    float Before,
    float After,
    bool Removed);

/// <summary>
/// The manual tensor-editing engine behind <c>amql-cli change-tensor</c>:
/// read a weight from a container, apply a single-cell edit in f32,
/// recompute the delta against the base, and merge it into a patch entry
/// list. Re-running with the same patch file accumulates edits — each call
/// applies against base + existing delta, so edits compose.
/// </summary>
public static class TensorPatchTools
{
    /// <summary>The tensor's stored shape (refused unless 1-D or 2-D —
    /// patches edit vectors and matrices).</summary>
    public static long[] ResolveShape(Vindex3Container container, string objectId, string tensorName)
    {
        using var store = container.CreateOperandStore();
        var resolution = store.Resolve(objectId, tensorName);
        if (resolution.Shape.Length is not (1 or 2))
        {
            throw new CliException(
                $"tensor '{objectId}/{tensorName}' is {resolution.Shape.Length}-D " +
                $"([{string.Join("x", resolution.Shape)}]) — patches edit 1-D vectors and 2-D matrices");
        }
        return resolution.Shape;
    }

    /// <summary>
    /// Applies one edit and returns the updated patch entry list. The edit
    /// targets the CURRENT value (base + existing delta), so repeated
    /// <c>--add</c>/<c>--scale</c> calls compose. An edit that lands exactly
    /// on the base value removes the entry (zero deltas carry no
    /// information and are dropped at write time).
    /// </summary>
    public static EditResult ApplyEdit(
        Vindex3Container container,
        string objectId,
        string tensorName,
        TensorEditOp op,
        float value,
        long flatIndex,
        IReadOnlyList<WeightPatchEntry> existing)
    {
        using var store = container.CreateOperandStore();
        var resolution = store.Resolve(objectId, tensorName);
        if (resolution.Shape.Length is not (1 or 2))
        {
            throw new CliException(
                $"tensor '{objectId}/{tensorName}' is {resolution.Shape.Length}-D " +
                $"([{string.Join("x", resolution.Shape)}]) — patches edit 1-D vectors and 2-D matrices");
        }
        if (!resolution.Dtype.IsWidenableToF32())
        {
            throw new CliException(
                $"tensor '{objectId}/{tensorName}' dtype {resolution.Dtype.Label()} has no f32 widening path");
        }

        long count = WeightPatch.ElementCount(resolution.Shape);
        if (flatIndex < 0 || flatIndex >= count)
        {
            throw new CliException(
                $"cell {flatIndex} is outside tensor '{objectId}/{tensorName}' " +
                $"[{string.Join("x", resolution.Shape)}] ({count} elements)");
        }

        var baseValues = BitPattern.WidenToF32(resolution.Dtype, resolution.Payload);
        var current = (float[])baseValues.Clone();
        var entries = existing.ToList();
        string key = objectId + "/" + tensorName;
        int found = entries.FindIndex(e => e.Key == key);
        if (found >= 0)
        {
            var entry = entries[found];
            if (entry.Delta.Length != current.Length)
            {
                throw new CliException(
                    $"patch entry '{entry.Key}' holds {entry.Delta.Length} deltas but the tensor has " +
                    $"{current.Length} elements — the patch was made for a different container");
            }
            for (int i = 0; i < current.Length; i++)
            {
                current[i] += entry.Delta[i];
            }
        }

        float before = current[flatIndex];
        float after = op switch
        {
            TensorEditOp.Set => value,
            TensorEditOp.Add => before + value,
            TensorEditOp.Scale => before * value,
            _ => throw new CliException($"unknown edit operation '{op}'"),
        };
        current[flatIndex] = after;

        for (int i = 0; i < current.Length; i++)
        {
            current[i] -= baseValues[i];
        }

        bool removed;
        if (WeightPatch.HasNonZero(current))
        {
            if (found >= 0)
            {
                entries[found] = new WeightPatchEntry(objectId, tensorName, resolution.Shape, current);
            }
            else
            {
                entries.Add(new WeightPatchEntry(objectId, tensorName, resolution.Shape, current));
            }
            removed = false;
        }
        else if (found >= 0)
        {
            entries.RemoveAt(found);
            removed = true;
        }
        else
        {
            removed = true; // the edit left no trace
        }

        return new EditResult(entries, objectId, tensorName, resolution.Shape,
            resolution.Dtype.Label(), flatIndex, before, after, removed);
    }

    /// <summary>Loads an existing patch file, or an empty entry list when
    /// none exists yet.</summary>
    public static IReadOnlyList<WeightPatchEntry> LoadOrEmpty(string? patchPath)
    {
        if (patchPath is null || !File.Exists(patchPath))
        {
            return Array.Empty<WeightPatchEntry>();
        }
        return WeightPatch.Load(patchPath).Entries;
    }
}

/// <summary>One factored target: the low-rank A/B pair for one edited
/// weight, with its reconstruction error.</summary>
public sealed record LoraTarget(
    string ObjectId,
    string TensorName,
    long[] Shape,
    int Rank,
    double ReconstructionError,
    string AName,
    string BName);

/// <summary>Everything saved by <c>amql-cli save-lora</c>: the adapter
/// directory, its metadata, the factored targets, and the entries skipped
/// (1-D norm scales cannot be factored into linear-layer LoRA pairs).</summary>
public sealed record LoraReport(
    string OutDir,
    int Rank,
    double Alpha,
    double Scale,
    string? Model,
    IReadOnlyList<LoraTarget> Targets,
    IReadOnlyList<string> Skipped);

/// <summary>
/// Writes a weight patch as a LoRA adapter for the ORIGINAL model: every
/// 2-D delta is factored into <c>lora_A</c> (r x in) and <c>lora_B</c>
/// (out x r) with the standard <c>alpha / r</c> scaling, so applying
/// <c>scale · lora_B · lora_A</c> to the unpatched weights reproduces the
/// patch's behaviour — that is the point of the low-rank form. The
/// factorization is a randomized truncated SVD; alpha/r is treated as the
/// scaling constant, so the stored matrices reconstruct the delta
/// EXACTLY when the delta's rank does not exceed r.
/// </summary>
public static class LoraWriter
{
    public const string LoraFormat = "amql-lora-v1";

    public static LoraReport SaveAsLora(string patchPath, string outDir, int rank, double alpha)
    {
        var patch = WeightPatch.Load(patchPath);
        if (rank <= 0)
        {
            throw new CliException($"--rank must be positive (got {rank})");
        }
        if (alpha <= 0)
        {
            throw new CliException($"--alpha must be positive (got {alpha})");
        }

        Directory.CreateDirectory(outDir);
        var tensors = new List<TensorPayload>();
        var targets = new List<LoraTarget>();
        var skipped = new List<string>();

        int index = 0;
        foreach (var entry in patch.Entries)
        {
            if (entry.Shape.Length != 2)
            {
                skipped.Add($"{entry.Key} [{string.Join("x", entry.Shape)}] — 1-D norm scale, not a LoRA target");
                continue;
            }
            int rows = checked((int)entry.Shape[0]);
            int cols = checked((int)entry.Shape[1]);
            if (entry.Delta.Length != (long)rows * cols)
            {
                throw new CliException(
                    $"patch entry '{entry.Key}' declares shape [{rows} x {cols}] but holds " +
                    $"{entry.Delta.Length} deltas — corrupt patch file");
            }
            foreach (var v in entry.Delta)
            {
                if (!float.IsFinite(v))
                {
                    throw new CliException(
                        $"patch entry '{entry.Key}' contains a non-finite delta ({v}) — cannot factor");
                }
            }

            int r = Math.Min(rank, Math.Min(rows, cols));
            if (r > LowRankSvd.MaxRank)
            {
                throw new CliException(
                    $"requested rank {rank} is not low-rank for [{rows} x {cols}] — " +
                    $"LoRA factors rank up to {LowRankSvd.MaxRank}");
            }

            // The stored pair factors delta/scale, so the applied update
            // scale · lora_B · lora_A reproduces the raw delta exactly
            // whenever the delta's rank does not exceed r.
            double scale = alpha / r;
            var target = new float[entry.Delta.Length];
            for (int i = 0; i < target.Length; i++)
            {
                target[i] = entry.Delta[i] / (float)scale;
            }

            LowRankSvd.Factorize(target, rows, cols, r, out var b, out var aMat, out double error);

            string aName = $"lora_A.{index}";
            string bName = $"lora_B.{index}";
            tensors.Add(new TensorPayload { Name = aName, Dtype = Dtype.F32, Shape = new long[] { r, cols }, Data = F32Bytes(aMat) });
            tensors.Add(new TensorPayload { Name = bName, Dtype = Dtype.F32, Shape = new long[] { rows, r }, Data = F32Bytes(b) });
            targets.Add(new LoraTarget(entry.ObjectId, entry.TensorName, entry.Shape, r, error, aName, bName));
            index++;
        }

        if (targets.Count == 0)
        {
            throw new CliException(
                $"'{patchPath}' contains no 2-D tensor deltas — LoRA factors linear-layer weights only. " +
                (skipped.Count == 0 ? string.Empty : $"Skipped: {string.Join("; ", skipped)}"));
        }

        string modelFile = Path.Combine(outDir, "adapter_model.safetensors");
        SafetensorsWriter.Write(modelFile, tensors, new Dictionary<string, string>
        {
            ["format"] = LoraFormat,
            ["model"] = patch.Model ?? string.Empty,
        });

        double scaleUsed = alpha / targets[0].Rank;
        var config = new AdapterConfig
        {
            Format = LoraFormat,
            Model = patch.Model,
            Rank = rank,
            Alpha = alpha,
            Scale = scaleUsed,
            Targets = targets.Select(t => new AdapterTarget
            {
                ObjectId = t.ObjectId,
                TensorName = t.TensorName,
                Shape = t.Shape,
                Rank = t.Rank,
                Scale = alpha / t.Rank,
                ReconstructionError = t.ReconstructionError,
                LoraA = t.AName,
                LoraB = t.BName,
            }).ToList(),
        };
        File.WriteAllText(Path.Combine(outDir, "adapter_config.json"),
            JsonSerializer.Serialize(config, new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            }));

        return new LoraReport(outDir, rank, alpha, scaleUsed, patch.Model, targets, skipped);
    }

    private sealed record AdapterConfig
    {
        public required string Format { get; init; }
        public string? Model { get; init; }
        public required int Rank { get; init; }
        public required double Alpha { get; init; }
        public required double Scale { get; init; }
        public required List<AdapterTarget> Targets { get; init; }
    }

    private sealed record AdapterTarget
    {
        public required string ObjectId { get; init; }
        public required string TensorName { get; init; }
        public required long[] Shape { get; init; }
        public required int Rank { get; init; }
        public required double Scale { get; init; }
        public required double ReconstructionError { get; init; }
        public required string LoraA { get; init; }
        public required string LoraB { get; init; }
    }

    private static byte[] F32Bytes(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}

/// <summary>
/// Rank-r truncated SVD of a dense row-major matrix
/// <c>A ≈ B · Aᵣ</c> (B rows x r, Aᵣ r x cols) by randomized projection:
/// an oversampled Gaussian sketch, a few power iterations to concentrate
/// the sketch onto the dominant singular subspace, and an exact small SVD
/// of the projected matrix via its (tiny) Gram matrix. The sqrt split
/// keeps both factors balanced; the returned SVD must be scaled by
/// <c>alpha / r</c> to become a LoRA update.
/// </summary>
internal static class LowRankSvd
{
    private const int Oversample = 8;
    private const int PowerIterations = 4;
    private const int MaxJacobiSweeps = 80;
    private const int Seed = 0x7A0B1;

    /// <summary>Largest rank the factorization will attempt (the small
    /// eigenproblem below it stays cheap).</summary>
    public const int MaxRank = 256;

    public static void Factorize(float[] a, int rows, int cols, int rank,
        out float[] b, out float[] aRank, out double relativeError)
    {
        if (rows <= 0 || cols <= 0)
        {
            throw new CliException($"cannot factor a [{rows} x {cols}] matrix");
        }
        if (a.Length != rows * cols)
        {
            throw new CliException(
                $"matrix has {rows * cols} elements but the caller supplied {a.Length}");
        }
        if (rank <= 0)
        {
            throw new CliException("rank must be positive");
        }

        var A = new double[rows * (long)cols];
        for (int i = 0; i < a.Length; i++)
        {
            A[i] = a[i];
        }
        double norm = Frobenius(A);
        if (norm == 0)
        {
            b = new float[rows * rank];
            aRank = new float[rank * cols];
            relativeError = 0;
            return;
        }

        int p = Math.Min(rank + Oversample, Math.Min(rows, cols));

        // Gaussian sketch, deterministic so the factorization is stable.
        var rng = new Random(Seed);
        var omega = new double[(long)cols * p];
        for (int i = 0; i < omega.Length; i++)
        {
            omega[i] = NextGaussian(rng);
        }

        // Y = A·Ω (rows x p), then power iterations Y ← A·(Aᵀ·Y).
        var y = new double[(long)rows * p];
        MatTimes(A, rows, cols, omega, cols, p, y);
        var t = new double[(long)cols * p];
        for (int it = 0; it < PowerIterations; it++)
        {
            MatTransposedTimes(A, rows, cols, y, rows, p, t);
            MatTimes(A, rows, cols, t, cols, p, y);
        }

        // Q = orthonormal basis of Y's column space.
        var q = new double[(long)rows * p];
        Orthogonalize(y, rows, p, q);

        // B = Qᵀ·A (p x cols); C = B·Bᵀ (p x p) symmetric PSD.
        var bl = new double[(long)p * cols];
        for (int j = 0; j < p; j++)
        {
            for (int k = 0; k < cols; k++)
            {
                double sum = 0;
                for (int m = 0; m < rows; m++)
                {
                    sum += q[(long)m * p + j] * A[(long)m * cols + k];
                }
                bl[(long)j * cols + k] = sum;
            }
        }
        var c = new double[(long)p * p];
        for (int i = 0; i < p; i++)
        {
            for (int j = i; j < p; j++)
            {
                double sum = 0;
                for (int k = 0; k < cols; k++)
                {
                    sum += bl[(long)i * cols + k] * bl[(long)j * cols + k];
                }
                c[(long)i * p + j] = sum;
                c[(long)j * p + i] = sum;
            }
        }

        SymmetricJacobi(c, p, out var evalues, out var evectors);

        int r = Math.Min(rank, p);
        var sigma = new double[r];
        double sigmaMax = 1e-30;
        for (int i = 0; i < r; i++)
        {
            sigma[i] = evalues[i] > 0 ? Math.Sqrt(evalues[i]) : 0;
            sigmaMax = Math.Max(sigmaMax, sigma[i]);
        }

        // Ũ_r = first r eigenvectors (p x r); V_r = Bᵀ·Ũ_r·Σ⁻¹ (cols x r);
        // Q̂ = Q·Ũ_r (rows x r). Balanced split:
        //   b = Q̂·√Σ (rows x r), aRank = √Σ·V_rᵀ (r x cols).
        b = new float[(long)rows * r];
        aRank = new float[(long)r * cols];
        for (int i = 0; i < r; i++)
        {
            double s = sigma[i];
            if (s <= sigmaMax * 1e-12)
            {
                continue; // numerically zero direction — leaves a zero factor column
            }
            double invS = 1.0 / s;
            double sqrtS = Math.Sqrt(s);
            // V_r column i = Σ_j B[j,k]·Ũ[j,i] / s.
            for (int k = 0; k < cols; k++)
            {
                double sum = 0;
                for (int j = 0; j < p; j++)
                {
                    sum += bl[(long)j * cols + k] * evectors[(long)j * p + i];
                }
                aRank[(long)i * cols + k] = (float)(sqrtS * invS * sum);
            }
            // Q̂ column i, scaled by √Σ.
            for (int m = 0; m < rows; m++)
            {
                double sum = 0;
                for (int j = 0; j < p; j++)
                {
                    sum += q[(long)m * p + j] * evectors[(long)j * p + i];
                }
                b[(long)m * r + i] = (float)(sqrtS * sum);
            }
        }

        // ||A − B·Aᵣ||_F / ||A||_F.
        double err = 0;
        for (int m = 0; m < rows; m++)
        {
            for (int k = 0; k < cols; k++)
            {
                double rec = 0;
                for (int i = 0; i < r; i++)
                {
                    rec += (double)b[(long)m * r + i] * aRank[(long)i * cols + k];
                }
                double diff = A[(long)m * cols + k] - rec;
                err += diff * diff;
            }
        }
        relativeError = Math.Sqrt(err) / norm;
    }

    // ── linear algebra helpers (double) ────────────────────────────────────

    private static void MatTimes(double[] a, int rows, int cols,
        double[] x, int xCols, int p, double[] y)
        // y (rows x p) = a (rows x cols) · x (cols x p)
    {
        for (int m = 0; m < rows; m++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int k = 0; k < cols; k++)
                {
                    sum += a[(long)m * cols + k] * x[(long)k * p + j];
                }
                y[(long)m * p + j] = sum;
            }
        }
    }

    private static void MatTransposedTimes(double[] a, int rows, int cols,
        double[] x, int xRows, int p, double[] y)
        // y (cols x p) = aᵀ (cols x rows) · x (rows x p)
    {
        for (int k = 0; k < cols; k++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int m = 0; m < rows; m++)
                {
                    sum += a[(long)m * cols + k] * x[(long)m * p + j];
                }
                y[(long)k * p + j] = sum;
            }
        }
    }

    private static double Frobenius(double[] a)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * a[i];
        }
        return Math.Sqrt(sum);
    }

    private static double NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    /// <summary>Modified Gram-Schmidt in place, the surviving basis in
    /// <c>q</c> (zero columns beyond the numerical rank).</summary>
    private static void Orthogonalize(double[] y, int rows, int p, double[] q)
    {
        for (int j = 0; j < p; j++)
        {
            int start = checked(j * rows);
            for (int i = 0; i < j; i++)
            {
                int prevStart = checked(i * rows);
                double dot = 0;
                for (int m = 0; m < rows; m++)
                {
                    dot += y[start + m] * q[prevStart + m];
                }
                for (int m = 0; m < rows; m++)
                {
                    y[start + m] -= dot * q[prevStart + m];
                }
            }
            double norm = 0;
            for (int m = 0; m < rows; m++)
            {
                norm += y[start + m] * y[start + m];
            }
            double scale = Math.Sqrt(norm);
            if (scale > 1e-14)
            {
                for (int m = 0; m < rows; m++)
                {
                    q[start + m] = y[start + m] / scale;
                }
            }
            else
            {
                Array.Clear(q, start, rows);
            }
        }
    }

    /// <summary>Cyclic Jacobi eigensolver for a symmetric matrix (n x n,
    /// row-major, destroyed on exit). Returns eigenvalues sorted
    /// descending with matching eigenvectors as columns.</summary>
    private static void SymmetricJacobi(double[] a, int n, out double[] evalues, out double[] evectors)
    {
        var v = new double[(long)n * n];
        for (int i = 0; i < n; i++)
        {
            v[(long)i * n + i] = 1.0;
        }

        for (int sweep = 0; sweep < MaxJacobiSweeps; sweep++)
        {
            double off = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    double x = a[(long)i * n + j];
                    off += x * x;
                }
            }
            if (off < 1e-24 || !double.IsFinite(off))
            {
                break;
            }

            for (int p = 0; p < n - 1; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    double apq = a[(long)p * n + q];
                    if (apq == 0 || double.IsNaN(apq))
                    {
                        continue;
                    }
                    double app = a[(long)p * n + p];
                    double aqq = a[(long)q * n + q];
                    double theta = (aqq - app) / (2.0 * apq);
                    double t = theta == 0
                        ? 1.0
                        : Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1.0));
                    double c = 1.0 / Math.Sqrt(t * t + 1.0);
                    double s = t * c;

                    for (int k = 0; k < n; k++)
                    {
                        if (k == p || k == q)
                        {
                            continue;
                        }
                        double akp = a[(long)k * n + p];
                        double akq = a[(long)k * n + q];
                        a[(long)k * n + p] = c * akp - s * akq;
                        a[(long)k * n + q] = s * akp + c * akq;
                        a[(long)p * n + k] = a[(long)k * n + p];
                        a[(long)q * n + k] = a[(long)k * n + q];
                    }
                    a[(long)p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                    a[(long)q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                    a[(long)p * n + q] = 0.0;
                    a[(long)q * n + p] = 0.0;

                    for (int k = 0; k < n; k++)
                    {
                        double vkp = v[(long)k * n + p];
                        double vkq = v[(long)k * n + q];
                        v[(long)k * n + p] = c * vkp - s * vkq;
                        v[(long)k * n + q] = s * vkp + c * vkq;
                    }
                }
            }
        }

        evalues = new double[n];
        for (int i = 0; i < n; i++)
        {
            evalues[i] = a[(long)i * n + i];
        }
        var sorted = evalues.ToArray();
        var order = Enumerable.Range(0, n)
            .OrderByDescending(i => sorted[i])
            .ToArray();
        for (int i = 0; i < n; i++)
        {
            evalues[i] = sorted[order[i]];
        }
        evectors = new double[(long)n * n];
        for (int col = 0; col < n; col++)
        {
            int src = order[col];
            for (int row = 0; row < n; row++)
            {
                evectors[(long)row * n + col] = v[(long)row * n + src];
            }
        }
    }
}