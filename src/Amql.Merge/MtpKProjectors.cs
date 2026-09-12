using System.Security.Cryptography;
using System.Text.Json;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>One candidate of the K-sweep: the cluster count, the fitted
/// block's in-sample residual, and its held-out draft acceptance measured
/// through the runtime path (in-memory, same math as the container).</summary>
public sealed record MtpSweepEntry(int Clusters, double R2, double GateAcceptance, string WeightsHash);

/// <summary>The outcome of the K-projector mixture fit (Phase 2 of the
/// calculated MTP drafter): the K-sweep table, the shipped cluster count
/// the acceptance gate selected, and the held-out acceptance through the
/// runtime path before and after the module rewrite.</summary>
public sealed record MtpKFitReport(
    string ContainerDir,
    int ShippedClusters,
    IReadOnlyList<MtpSweepEntry> Sweep,
    double GateAcceptanceBefore,
    double GateAcceptanceAfter,
    string WeightsHashHex,
    IReadOnlyList<string> Notes);

/// <summary>
/// Phase 2 of the calculated MTP drafter: the K-projector mixture. The
/// fit split's <c>x_t</c> rows are clustered into <c>K</c> balanced groups
/// (deterministic farthest-first k-means, no random init — two runs assign
/// identically), one ridge projector is fit per cluster on its members,
/// and a linear router (the L2-normalised cluster-mean rows, the moe-ify
/// convention) picks the expert per row. The candidates <c>K</c> are
/// measured on the held-out split through the runtime path with one shared
/// dense forward; the acceptance gate ships the winner (ties prefer the
/// smaller K). No search, no random seeds.
/// </summary>
public static class MtpKProjectors
{
    /// <param name="clusters">The candidate cluster counts to measure,
    /// e.g. {1, 4, 8}; the gate ships the acceptance winner.</param>
    public static MtpKFitReport SweepAndFit(
        string containerDir, string pairsDir, IReadOnlyList<int> clusters, double ridgeRel = 1e-4)
    {
        var kSet = clusters.Distinct().OrderBy(c => c).ToArray();
        if (kSet.Length == 0 || kSet[0] < 1)
        {
            throw new MergeException("need at least one cluster count >= 1");
        }

        // ── the pairs (phase 0 contract) ────────────────────────────────
        int fitCount, gateCount, hidden, xColumns, yColumns;
        int[] split;
        using (var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(pairsDir, "manifest.json"))))
        {
            var root = doc.RootElement;
            fitCount = root.GetProperty("fit_count").GetInt32();
            gateCount = root.GetProperty("gate_count").GetInt32();
            hidden = root.GetProperty("hidden").GetInt32();
            xColumns = root.GetProperty("x_columns").GetInt32();
            yColumns = root.GetProperty("y_columns").GetInt32();
            split = root.GetProperty("split").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        }
        if (xColumns != 2 * hidden || yColumns != hidden)
        {
            throw new MergeException(
                $"pair manifest implies x [{fitCount}, {xColumns}] → y [{fitCount}, {yColumns}] — expected 2h→h for hidden {hidden}");
        }
        var x = MtpFitter.ReadF32(Path.Combine(pairsDir, "fit.x.bin"));
        var y = MtpFitter.ReadF32(Path.Combine(pairsDir, "fit.y.bin"));
        if (x.Length != fitCount * xColumns || y.Length != fitCount * yColumns)
        {
            throw new MergeException("the pair sink does not match its manifest");
        }
        if (kSet[^1] > fitCount)
        {
            throw new MergeException(
                $"cannot fit {kSet[^1]} balanced clusters from {fitCount} fit pairs");
        }

        // ── the container must already carry the bootstrapped skeleton ──
        using var container = Vindex3Container.Open(containerDir);
        if (container.Graph?.Objects.FirstOrDefault(o => o.Id == "mtp.stack") is not { Representations.Count: > 0 })
        {
            throw new MergeException(
                "the container carries no materialised MTP drafter — run 'amql-cli generate-mtp' first");
        }
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        if (plan.HiddenSize != hidden)
        {
            throw new MergeException(
                $"pairs were collected for hidden {hidden} but the container is hidden {plan.HiddenSize}");
        }

        // ── held-out gate tokens from the collected window ───────────────
        var window = MtpFitter.ReadI32(Path.Combine(pairsDir, "tokens.bin"));
        int gateStart = split[1];
        int gateEnd = split[2];
        var gateTokens = window.Skip(gateStart).Take(gateEnd - gateStart + 2).ToArray();

        double before = GenerateMtp.MeasureDraftAcceptance(containerDir, gateTokens, gateCount + 2);

        // ── the sweep: one shared dense forward, one trunk pass per K ────
        var candidates = new List<(int K, double R2, double Acceptance, DraftProjector Block)>();
        using (var ctx = new DraftContext(containerDir, gateTokens, gateCount + 2))
        {
            foreach (int k in kSet)
            {
                DraftProjector block;
                if (k == 1)
                {
                    var w = LeastSquares.FitProjection(y, x, fitCount, hidden, xColumns, ridgeRel);
                    block = DraftProjector.Single(w, hidden);
                }
                else
                {
                    var assign = ClusterRows(x, fitCount, xColumns, k);
                    var router = RouterRows(x, fitCount, xColumns, k, assign);
                    var experts = new float[k][];
                    var members = Enumerable.Range(0, k).Select(_ => new List<int>()).ToArray();
                    for (int t = 0; t < fitCount; t++)
                    {
                        members[assign[t]].Add(t);
                    }
                    for (int e = 0; e < k; e++)
                    {
                        var idx = members[e];
                        var xe = new float[idx.Count * xColumns];
                        var ye = new float[idx.Count * hidden];
                        for (int r = 0; r < idx.Count; r++)
                        {
                            Array.Copy(x, idx[r] * xColumns, xe, r * xColumns, xColumns);
                            Array.Copy(y, idx[r] * hidden, ye, r * hidden, hidden);
                        }
                        experts[e] = LeastSquares.FitProjection(ye, xe, idx.Count, hidden, xColumns, ridgeRel);
                    }
                    block = DraftProjector.Routed(router, experts, hidden);
                }
                double acceptance = ctx.Score(block);
                double r2 = ResidualR2(y, x, fitCount, hidden, xColumns, block);
                candidates.Add((k, r2, acceptance, block));
            }
        }

        // The gate ships the acceptance winner; ties prefer fewer clusters.
        var winner = candidates
            .OrderByDescending(c => c.Acceptance)
            .ThenBy(c => c.K)
            .First();
        var shipped = winner.Block;
        if (shipped.W is not null)
        {
            MtpFitter.ReplaceFc(container, shipped.W);
        }
        else
        {
            MtpFitter.ReplaceFcRouted(container, shipped.Router!, shipped.Experts!);
        }
        double after = GenerateMtp.MeasureDraftAcceptance(containerDir, gateTokens, gateCount + 2);

        var table = candidates
            .OrderBy(c => c.K)
            .Select(c => new MtpSweepEntry(c.K, c.R2, c.Acceptance, Hash(c.Block)))
            .ToList();
        string shippedHash = table.First(e => e.Clusters == winner.K).WeightsHash;
        var notes = new List<string>
        {
            $"swept K ∈ {{{string.Join(", ", kSet)}}}: " +
            string.Join("; ", table.Select(e => $"K={e.Clusters} R² {e.R2:0.000} acc {e.GateAcceptance:0.0%}")),
            $"shipped K={winner.K} (the acceptance gate's winner; ties prefer fewer clusters); " +
            $"weights {shippedHash}",
            $"held-out draft acceptance over {gateCount} positions: " +
            $"boot {before:0.0%} → fitted {after:0.0%}",
        };
        return new MtpKFitReport(
            containerDir,
            winner.K,
            table,
            before,
            after,
            shippedHash,
            notes);
    }

    /// <summary>1 − SSE/SST of the block over the fit split (the routed
    /// prediction uses the same argmax the container will serve).</summary>
    private static double ResidualR2(
        float[] y, float[] x, int n, int hidden, int xColumns, DraftProjector block)
    {
        var predicted = block.Project(x, n);
        double mean = 0;
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < hidden; c++)
            {
                mean += y[i * hidden + c];
            }
        }
        mean /= n * hidden;
        double sse = 0, sst = 0;
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < hidden; c++)
            {
                double err = y[i * hidden + c] - predicted[i * hidden + c];
                sse += err * err;
                double diff = y[i * hidden + c] - mean;
                sst += diff * diff;
            }
        }
        return sst > 0 ? 1 - sse / sst : double.NaN;
    }

    private static string Hash(DraftProjector block)
    {
        using var stream = new MemoryStream();
        void Append(float[] values)
        {
            var bytes = new byte[values.Length * 4];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            stream.Write(bytes);
        }
        if (block.W is not null)
        {
            Append(block.W);
        }
        else
        {
            Append(block.Router!);
            foreach (var expert in block.Experts!)
            {
                Append(expert);
            }
        }
        return Convert.ToHexStringLower(SHA256.HashData(stream.ToArray()));
    }

    /// <summary>Balanced k-means over the fit rows, mirroring the moe-ify
    /// clustering exactly: L2-normalised rows, cosine-distance assignment,
    /// farthest-first seeding from row 0 (no randomness — two runs assign
    /// identically), then force-balance to <c>n/k</c> rows per cluster.</summary>
    internal static int[] ClusterRows(float[] x, int n, int dim, int k)
    {
        int target = n / k;
        var rows = new float[n * dim];
        for (int j = 0; j < n; j++)
        {
            double norm = 0;
            for (int t = 0; t < dim; t++)
            {
                double v = x[j * dim + t];
                norm += v * v;
            }
            if (norm > 0)
            {
                double inv = 1.0 / Math.Sqrt(norm);
                for (int t = 0; t < dim; t++)
                {
                    rows[j * dim + t] = (float)(x[j * dim + t] * inv);
                }
            }
        }

        var centroids = new float[k * dim];
        Array.Copy(rows, 0, centroids, 0, dim);
        var best = new double[n];
        for (int c = 1; c < k; c++)
        {
            double farthest = -1;
            int farthestRow = 0;
            for (int j = 0; j < n; j++)
            {
                double d = DistanceSquared(rows, j, centroids, 0, dim);
                if (d > best[j])
                {
                    best[j] = d;
                }
                if (best[j] > farthest)
                {
                    farthest = best[j];
                    farthestRow = j;
                }
            }
            Array.Copy(rows, farthestRow * dim, centroids, c * dim, dim);
        }

        var assign = new int[n];
        var counts = new int[k];
        for (int iter = 0; iter < 40; iter++)
        {
            bool changed = false;
            Array.Clear(counts);
            for (int j = 0; j < n; j++)
            {
                int bestCluster = 0;
                double bestD = double.PositiveInfinity;
                for (int c = 0; c < k; c++)
                {
                    if (counts[c] >= target && iter < 39)
                    {
                        continue;
                    }
                    double d = DistanceSquared(rows, j, centroids, c * dim, dim);
                    if (d < bestD)
                    {
                        bestD = d;
                        bestCluster = c;
                    }
                }
                if (assign[j] != bestCluster)
                {
                    changed = true;
                    assign[j] = bestCluster;
                }
                counts[bestCluster]++;
            }
            RecomputedCentroids(rows, assign, counts, centroids, n, k, dim);
            if (!changed)
            {
                break;
            }
        }

        // Enforce the exact balance: move the worst-fitting row from every
        // overfull cluster into the underfull cluster it fits best.
        for (int pass = 0; pass < n; pass++)
        {
            var underfull = Enumerable.Range(0, k).Where(c => counts[c] < target).ToArray();
            if (underfull.Length == 0)
            {
                break;
            }
            int targetCluster = underfull[0];
            var candidates = Enumerable.Range(0, n)
                .Where(j => counts[assign[j]] > target)
                .ToArray();
            if (candidates.Length == 0)
            {
                break;
            }
            double bestFit = double.PositiveInfinity;
            int bestRow = candidates[0];
            foreach (int j in candidates)
            {
                double d = DistanceSquared(rows, j, centroids, targetCluster * dim, dim);
                if (d < bestFit)
                {
                    bestFit = d;
                    bestRow = j;
                }
            }
            counts[assign[bestRow]]--;
            counts[targetCluster]++;
            assign[bestRow] = targetCluster;
        }
        return assign;
    }

    private static double DistanceSquared(float[] rows, int row, float[] centroids, int centroidStart, int dim)
    {
        double sum = 0;
        for (int t = 0; t < dim; t++)
        {
            double d = rows[row * dim + t] - centroids[centroidStart + t];
            sum += d * d;
        }
        return sum;
    }

    private static void RecomputedCentroids(
        float[] rows, int[] assign, int[] counts, float[] centroids, int n, int k, int dim)
    {
        Array.Clear(centroids);
        var sums = new float[k * dim];
        for (int j = 0; j < n; j++)
        {
            int cluster = assign[j];
            for (int t = 0; t < dim; t++)
            {
                sums[cluster * dim + t] += rows[j * dim + t];
            }
        }
        for (int c = 0; c < k; c++)
        {
            if (counts[c] > 0)
            {
                float inv = 1f / counts[c];
                for (int t = 0; t < dim; t++)
                {
                    centroids[c * dim + t] = sums[c * dim + t] * inv;
                }
            }
        }
    }

    /// <summary>The linear router: each expert's row is the L2-normalised
    /// mean of its cluster's raw <c>x_t</c> rows — token affinity
    /// <c>x · routerᵀ</c>, the moe-ify convention.</summary>
    private static float[] RouterRows(float[] x, int n, int dim, int k, int[] assign)
    {
        var router = new float[k * dim];
        var counts = new int[k];
        for (int t = 0; t < n; t++)
        {
            int cluster = assign[t];
            counts[cluster]++;
            for (int c = 0; c < dim; c++)
            {
                router[cluster * dim + c] += x[t * dim + c];
            }
        }
        for (int cluster = 0; cluster < k; cluster++)
        {
            if (counts[cluster] == 0)
            {
                continue;
            }
            double norm = 0;
            for (int c = 0; c < dim; c++)
            {
                double v = router[cluster * dim + c] / counts[cluster];
                router[cluster * dim + c] = (float)v;
                norm += v * v;
            }
            if (norm > 0)
            {
                double inv = 1.0 / Math.Sqrt(norm);
                for (int c = 0; c < dim; c++)
                {
                    router[cluster * dim + c] = (float)(router[cluster * dim + c] * inv);
                }
            }
        }
        return router;
    }
}