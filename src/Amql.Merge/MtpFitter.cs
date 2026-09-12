using System.Security.Cryptography;
using System.Text.Json;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>The outcome of the calculated MTP projector fit (Phase 1): the
/// fitted block's diagnostics and the held-out acceptance through the
/// runtime path, before and after the module rewrite.</summary>
public sealed record MtpFitReport(
    string ContainerDir,
    int FitCount,
    int Hidden,
    double LambdaRel,
    double ResidualR2,
    string WeightsHashHex,
    double GateAcceptanceBefore,
    double GateAcceptanceAfter,
    IReadOnlyList<string> Notes);

/// <summary>
/// Phase 1 of the calculated MTP drafter: fit the free block of the
/// bootstrapped skeleton with a closed-form ridge least-squares
/// projection onto the model's own continuation pairs
/// (<see cref="PairCollector"/>), replace <c>fc.weight</c> in the
/// container's <c>mtp.stack</c>, and re-measure draft acceptance through
/// the runtime gate on the HELD-OUT split. Closed-form and
/// order-independent: two fits on the same pairs produce byte-identical
/// weights (the determinism hash is the marker).
/// </summary>
public static class MtpFitter
{
    public static MtpFitReport FitAndAssemble(
        string containerDir, string pairsDir, double ridgeRel = 1e-4)
    {
        // ── the pairs (Phase 0 contract) ────────────────────────────────
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

        var x = ReadF32(Path.Combine(pairsDir, "fit.x.bin"));
        var y = ReadF32(Path.Combine(pairsDir, "fit.y.bin"));
        if (x.Length != fitCount * xColumns || y.Length != fitCount * yColumns)
        {
            throw new MergeException("the pair sink does not match its manifest");
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

        // ── the closed-form fit ──────────────────────────────────────────
        var w = LeastSquares.FitProjection(y, x, fitCount, dOut: hidden, dIn: xColumns, ridgeRel);

        // diagnostics: R² = 1 − SSE/SST over the fit pairs.
        double sse = 0, sst = 0, meanY = 0;
        for (int i = 0; i < fitCount; i++)
        {
            double mean = 0;
            for (int c = 0; c < hidden; c++)
            {
                mean += y[i * hidden + c];
            }
            mean /= hidden;
            meanY += mean;
        }
        meanY /= fitCount;
        for (int i = 0; i < fitCount; i++)
        {
            for (int c = 0; c < hidden; c++)
            {
                double diff = y[i * hidden + c] - meanY;
                sst += diff * diff;
            }
            for (int o = 0; o < hidden; o++)
            {
                double predicted = 0;
                for (int j = 0; j < xColumns; j++)
                {
                    predicted += w[o * xColumns + j] * x[i * xColumns + j];
                }
                double err = y[i * hidden + o] - predicted;
                sse += err * err;
            }
        }
        double r2 = sst > 0 ? 1 - sse / sst : double.NaN;

        var weightBytes = new byte[w.Length * 4];
        Buffer.BlockCopy(w, 0, weightBytes, 0, weightBytes.Length);
        string weightsHash = Convert.ToHexStringLower(SHA256.HashData(weightBytes));

        // ── held-out gate tokens from the collected window ───────────────
        var window = ReadI32(Path.Combine(pairsDir, "tokens.bin"));
        int gateStart = split[1];
        int gateEnd = split[2];
        // The gate consumes gateCount+2 window tokens (h_t, e_{t+1}, the
        // final target) and reports over the gateCount positions.
        var gateTokens = window.Skip(gateStart).Take(gateEnd - gateStart + 2).ToArray();

        double before = GenerateMtp.MeasureDraftAcceptance(containerDir, gateTokens, gateCount + 2);

        // ── assemble: replace fc.weight with the fitted block ────────────
        ReplaceFc(container, w);

        double after = GenerateMtp.MeasureDraftAcceptance(containerDir, gateTokens, gateCount + 2);

        return new MtpFitReport(
            containerDir,
            fitCount,
            hidden,
            ridgeRel,
            double.IsNaN(r2) ? double.NaN : r2,
            weightsHash,
            before,
            after,
            new List<string>
            {
                $"ridge {ridgeRel:g3}: R² {r2:0.000} over {fitCount} fit pairs; weights {weightsHash}",
                $"held-out draft acceptance over {gateCount} positions: " +
                $"boot {before:0.0%} → fitted {after:0.0%}",
            });
    }

    /// <summary>Replaces <c>fc.weight</c> in the container's mtp.stack with
    /// the fitted block (re-encoded to the fc's stored dtype), rewriting
    /// the segment and its index entry hashes. The source mapping is
    /// released before the rewrite so the segment can be recreated.</summary>
    private static void ReplaceFc(Vindex3Container container, float[] w)
    {
        string repId = container.CanonicalRepresentationId("mtp.stack");
        var entry = container.Index.Representations[repId];
        var segmentPath = Path.Combine(container.Root, entry.Segment);

        List<NamedTensorData> rebuilt;
        Dtype fcDtype;
        using (var segment = SegmentFile.Open(segmentPath))
        {
            fcDtype = Dtype.F32;
            rebuilt = new List<NamedTensorData>(segment.Header.Tensors.Count);
            foreach (var tensor in segment.Header.Tensors)
            {
                if (tensor.Name == "fc.weight")
                {
                    fcDtype = DtypeExtensions.FromLabel(tensor.Dtype);
                }
                rebuilt.Add(new NamedTensorData
                {
                    Name = tensor.Name,
                    Dtype = DtypeExtensions.FromLabel(tensor.Dtype),
                    Shape = tensor.Shape,
                    Data = tensor.Name == "fc.weight" ? EncodeToDtype(fcDtype, w) : segment.ReadBytes(tensor.Name),
                });
            }
        }

        var result = SegmentWriter.Write(segmentPath, $"mtp.stack@{entry.Encoding}", rebuilt);
        UpdateIndexEntry(container, repId, entry, rebuilt, result);
    }

    private static void UpdateIndexEntry(
        Vindex3Container container, string repId, RepresentationEntry entry,
        List<NamedTensorData> rebuilt, SegmentWriteResult result)
    {
        var representations = new Dictionary<string, RepresentationEntry>(container.Index.Representations)
        {
            [repId] = new RepresentationEntry
            {
                Object = entry.Object,
                Encoding = entry.Encoding,
                Segment = entry.Segment,
                TensorCount = rebuilt.Count,
                PayloadBytes = result.PayloadBytes,
                PayloadSha256 = result.PayloadSha256Hex,
                SegmentSha256 = result.SegmentSha256Hex,
            },
        };
        var index = new Vindex3Index
        {
            Version = container.Index.Version,
            Model = container.Index.Model,
            Family = container.Index.Family,
            HiddenSize = container.Index.HiddenSize,
            NumLayers = container.Index.NumLayers,
            SystemGraph = container.Index.SystemGraph,
            Representations = representations,
            Profiles = container.Index.Profiles,
            Segments = container.Index.Segments,
            Authority = container.Index.Authority,
            DerivedFromModel = container.Index.DerivedFromModel,
            PrecisionMap = container.Index.PrecisionMap,
            TokenMap = container.Index.TokenMap,
        };
        File.WriteAllText(Path.Combine(container.Root, "index.json"),
            JsonSerializer.Serialize(index, ViJson.Options));
    }

    private static byte[] EncodeToDtype(Dtype dtype, float[] values) => dtype switch
    {
        Dtype.F32 => ToF32Bytes(values),
        Dtype.BF16 => ToBf16Bytes(values),
        Dtype.F16 => ToF16Bytes(values),
        _ => throw new MergeException(
            $"cannot store the fitted projector as '{dtype.Label()}' — F32/BF16/F16 only"),
    };

    private static byte[] ToF32Bytes(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static byte[] ToBf16Bytes(float[] values)
    {
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            ushort bits = BitPattern.EncodeBf16(values[i]);
            bytes[2 * i] = (byte)(bits & 0xFF);
            bytes[2 * i + 1] = (byte)(bits >> 8);
        }
        return bytes;
    }

    private static byte[] ToF16Bytes(float[] values)
    {
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            ushort bits = BitPattern.EncodeF16(values[i]);
            bytes[2 * i] = (byte)(bits & 0xFF);
            bytes[2 * i + 1] = (byte)(bits >> 8);
        }
        return bytes;
    }

    private static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static int[] ReadI32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new int[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }
}