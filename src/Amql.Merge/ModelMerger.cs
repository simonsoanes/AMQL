using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Merge;

/// <summary>The outcome of one import/merge: what the merged container
/// holds, how the vocabulary was combined, and what was kept.</summary>
public sealed record MergeReport(
    string OutDir,
    string ResultModel,
    string BaseModel,
    string ImportedModel,
    string Scaffold,
    int BaseVocab,
    int ImportedVocab,
    int MergedVocab,
    int Blended,
    int AlignedNew,
    int BaseOnly,
    int HiddenSize,
    int Layers,
    string Dtype,
    bool HeadTied,
    int PreservedSegments,
    IReadOnlyList<string> Notes);

/// <summary>
/// Imports a second model into an existing container, producing a merged
/// container that export materialises as one model:
///
/// <list type="bullet">
/// <item>the tokenization mapping layer relates the two vocabularies —
/// identical token strings relate; everything else is new;</item>
/// <item>the token-interface tensors (embedding and, when not tied, the
/// output head) grow to the union vocabulary: shared tokens blend the
/// base row with the imported row after a ridge least-squares alignment
/// fits the imported space into the base space on those shared anchors;
/// imported-only tokens are appended through the same map; base-only
/// rows stay verbatim (zero-extended when the imported model is
/// wider);</item>
/// <item>the transformer stack evolves to the union shape — the larger
/// model (hidden size, then layers) scaffolds the result, its tensors
/// are copied byte-identically (intermediate layers are never averaged —
/// differently shaped stacks cannot be blended), and per-layer
/// provenance records the source of every tensor;</item>
/// <item>the replaced (non-scaffold) stack is preserved inside the
/// container under <c>segments/source/</c>, and the manifest
/// (<c>token-map.json</c>) is the relationship tracker for every token
/// and layer linkage.</item>
/// </list>
/// </summary>
public static class ModelMerger
{
    /// <summary>Class-level clock for progress lines emitted from static
    /// helpers (AlignRows) where the Merge method's Phase local function
    /// is out of scope; mirrors the Phase format so the log stays uniform.</summary>
    private static readonly System.Diagnostics.Stopwatch ProgressClock =
        System.Diagnostics.Stopwatch.StartNew();

    public const string ManifestName = "token-map.json";

    /// <summary>Cosine-agreement gate for shared tokens: rows whose two
    /// models' aligned representations agree at or above this cosine are
    /// blended (then restored to the scaffold's energy); below it the
    /// scaffold's own row wins, because averaging disagreeing directions
    /// fabricates rows neither model holds.</summary>
    public const double AgreementThreshold = 0.5;

    private const string PreservedRoot = "segments/source";

    public static MergeReport Import(
        string baseContainerDir,
        string importedPath,
        string outDir,
        bool importedIsContainer)
    {
        if (Directory.Exists(outDir))
        {
            throw new MergeException($"merge output '{outDir}' already exists");
        }

        var stopwatch = Stopwatch.StartNew();
        void Phase(string label)
        {
            Console.Error.WriteLine($"[{stopwatch.Elapsed.TotalSeconds,7:0.0}s] {label}");
            Console.Error.Flush();
        }

        using var imported = OpenImported(importedPath, importedIsContainer);
        using var baseContainer = Vindex3Container.Open(baseContainerDir);

        var baseView = ModelView.Open(baseContainer, "base");
        var importedView = ModelView.Open(imported.Container, "imported");

        // ── Tokenization mapping layer ────────────────────────────────────
        var mapping = TokenMapping.Build(
            baseView.Vocab, importedView.Vocab, baseView.Model, importedView.Model);
        if (mapping.BlendedCount == 0)
        {
            throw new MergeException(
                $"the vocabularies share no token strings ('{baseView.Model}' vs '{importedView.Model}') — " +
                "no anchors to align the spaces; refusing to append unaligned rows");
        }
        Phase($"token mapping: {mapping.Count} merged tokens ({mapping.BlendedCount} anchors)");

        // ── Shape union: the larger model scaffolds the stack ─────────────
        var scaffold = importedView.Hidden > baseView.Hidden ||
                       (importedView.Hidden == baseView.Hidden && importedView.Layers > baseView.Layers)
            ? importedView
            : baseView;
        var other = ReferenceEquals(scaffold, baseView) ? importedView : baseView;
        bool scaffoldIsImported = ReferenceEquals(scaffold, importedView);
        int width = Math.Max(baseView.Width, importedView.Width);
        int layers = Math.Max(baseView.Layers, importedView.Layers);
        var mergedDtype = scaffold.Embedding.DtypeEnum;
        bool mergedTied = baseView.HeadTied;
        bool headUntied = !baseView.HeadTied;

        var notes = new List<string>();
        if (baseView.Width != width)
        {
            notes.Add($"base embedding {baseView.Width}→{width}: rows zero-extended to the merged width");
        }
        if (importedView.Width != width)
        {
            notes.Add($"imported embedding {importedView.Width}→{width}: rows zero-extended to the merged width");
        }
        notes.Add($"stack scaffold: {scaffold.Model} (hidden {scaffold.Hidden}, layers {scaffold.Layers}); " +
                  $"the replaced stack of '{other.Model}' is preserved under {PreservedRoot}/");

        // ── Token-interface merge: the union lives in the SCAFFOLD model's
        // space (the model whose stack runs). The other model's rows are
        // mapped into it with the alignment fit; shared tokens are blended
        // only where the two models agree (cosine gate) and the blend is
        // then restored to the scaffold's energy — disagreements defer to
        // the scaffold's own row instead of fabricating a direction. ──────
        var scaffoldEmbedRows = ReadWidenedRows(scaffold.Embedding, width);
        var otherEmbedRows = ReadWidenedRows(other.Embedding, width);
        Phase("widened embedding tables");
        var embeddingAlignment = AlignRows(
            scaffoldEmbedRows, otherEmbedRows, mapping, scaffoldIsImported, width, AgreementThreshold);
        Phase($"fitted embedding alignment ({embeddingAlignment.Anchors} anchors)");
        notes.Add($"alignment: {embeddingAlignment.Anchors} shared-token anchors into {scaffold.Model}'s space, " +
                  $"residual L² {embeddingAlignment.ResidualL2:g4} (ridge {embeddingAlignment.RidgeRel:g3}, " +
                  $"agreement threshold {AgreementThreshold:g2})");
        var embeddingMerge = MergeTable(
            scaffoldEmbedRows, otherEmbedRows, mapping, scaffoldIsImported, embeddingAlignment, width, mergedDtype);
        Phase($"merged embedding rows (mean agreement {embeddingMerge.MeanAgreement:0.000}, " +
              $"{embeddingMerge.ShareAboveThreshold * 100:0.0}% above threshold)");
        notes.Add(AgreementNote(embeddingMerge, scaffold.Model));

        IReadOnlyList<byte[]>? headChunks = null;
        Dtype? headDtype = null;
        if (headUntied)
        {
            var scaffoldHead = scaffold.OutputHead ?? scaffold.Embedding;
            var otherHead = other.OutputHead ?? other.Embedding;
            headDtype = scaffoldHead.DtypeEnum;
            var scaffoldHeadRows = ReadWidenedRows(scaffoldHead, width);
            var otherHeadRows = ReadWidenedRows(otherHead, width);
            var headAlignment = AlignRows(
                scaffoldHeadRows, otherHeadRows, mapping, scaffoldIsImported, width, AgreementThreshold);
            var headMerge = MergeTable(
                scaffoldHeadRows, otherHeadRows, mapping, scaffoldIsImported, headAlignment, width, headDtype.Value);
            headChunks = headMerge.Chunks;
            notes.Add(AgreementNote(headMerge, scaffold.Model, "head"));
            Phase("merged output head rows");
        }

        // ── Stack scaffold + per-layer provenance ─────────────────────────
        var stack = PlanStack(baseView, importedView, scaffold);
        Phase(stack.NeedsRebuild ? "stack scaffold (grown tensors present)" : "stack scaffold (byte copies)");

        // ── Write the merged container ────────────────────────────────────
        Directory.CreateDirectory(outDir);

        var representations = new Dictionary<string, RepresentationEntry>(StringComparer.Ordinal);
        var segments = new Dictionary<string, int>(StringComparer.Ordinal);

        string embeddingEncoding = mergedDtype.Label();
        var embeddingResult = SegmentWriter.Write(
            Path.Combine(outDir, "segments", "target.embedding.bin"),
            $"target.embedding@{embeddingEncoding}",
            new[] { new NamedTensorData
            {
                Name = "weight",
                Dtype = mergedDtype,
                Shape = new long[] { mapping.Count, width },
                Chunks = embeddingMerge.Chunks,
            } });
        representations[$"target.embedding@{embeddingEncoding}"] = new RepresentationEntry
        {
            Object = "target.embedding",
            Encoding = embeddingEncoding,
            Segment = "segments/target.embedding.bin",
            TensorCount = 1,
            PayloadBytes = embeddingResult.PayloadBytes,
            PayloadSha256 = embeddingResult.PayloadSha256Hex,
            SegmentSha256 = embeddingResult.SegmentSha256Hex,
        };
        segments["segments/target.embedding"] = 1;

        CopyMaterialised(outDir, scaffold, "target.decoder_stack", stack.StackEncoding, stack.RebuiltEncoding,
            stack.Rebuilt, representations, segments);
        CopyMaterialised(outDir, scaffold, "target.final_norm", stack.FinalNormEncoding, null, null,
            representations, segments);

        if (headUntied)
        {
            string headEncoding = headDtype!.Value.Label();
            var headResult = SegmentWriter.Write(
                Path.Combine(outDir, "segments", "target.output_head.bin"),
                $"target.output_head@{headEncoding}",
                new[] { new NamedTensorData
                {
                    Name = "weight",
                    Dtype = headDtype.Value,
                    Shape = new long[] { mapping.Count, width },
                    Chunks = headChunks!,
                } });
            representations[$"target.output_head@{headEncoding}"] = new RepresentationEntry
            {
                Object = "target.output_head",
                Encoding = headEncoding,
                Segment = "segments/target.output_head.bin",
                TensorCount = 1,
                PayloadBytes = headResult.PayloadBytes,
                PayloadSha256 = headResult.PayloadSha256Hex,
                SegmentSha256 = headResult.SegmentSha256Hex,
            };
            segments["segments/target.output_head"] = 1;
        }

        // The merged container keeps every scaffold segment the merge does
        // not rebuild — the vision tower, a carried MTP drafter, anything
        // else the encoder materialised. The output graph references those
        // objects unchanged; index entries and segment files must follow
        // or the merged container is structurally incomplete.
        foreach (var (repId, entry) in scaffold.Index.Representations)
        {
            if (representations.ContainsKey(repId))
            {
                continue;
            }
            Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(outDir, entry.Segment))!);
            File.Copy(Path.Combine(scaffold.Container.Root, entry.Segment), Path.Combine(outDir, entry.Segment), overwrite: true);
            representations[repId] = entry;
            segments[entry.Segment[..^4]] = 1;
        }

        // The replaced stack is preserved inside the same container; the
        // manifest documents it. Deliberately outside the graph/index so
        // the merged model stays the single authority for the container.
        var preservedSegments = new List<string>();
        if (!ReferenceEquals(other, scaffold))
        {
            foreach (var stem in new[] { "target.embedding", "target.decoder_stack", "target.final_norm" })
            {
                var entry = other.Index.Representations.Values.FirstOrDefault(e => e.Object == stem);
                if (entry is null)
                {
                    continue;
                }
                string destRel = $"{PreservedRoot}/{Sanitize(other.Model)}/{Path.GetFileName(entry.Segment)}";
                Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(outDir, destRel))!);
                File.Copy(Path.Combine(other.Container.Root, entry.Segment), Path.Combine(outDir, destRel));
                preservedSegments.Add(destRel);
            }
        }

        var manifest = BuildManifest(baseView, importedView, mapping, width, mergedDtype, headDtype,
            mergedTied, stack, embeddingAlignment, embeddingMerge, other.Model, preservedSegments);
        File.WriteAllText(Path.Combine(outDir, ManifestName), manifest.ToJsonString(ViJson.Options));

        var appended = mapping.Entries
            .Where(e => e.Kind == TokenMergeKind.AlignedNew)
            .Select(e => e.Token)
            .ToList();
        File.WriteAllText(Path.Combine(outDir, "tokenizer.json"),
            TokenVocabulary.WithAppendedTokens(
                Path.Combine(baseView.Container.Root, "tokenizer.json"), appended));

        WriteGraph(outDir, baseView, scaffold, mapping.Count, width, layers, mergedTied, mergedDtype,
            headUntied, headDtype);
        WriteIndex(outDir, baseView, importedView, scaffold, mapping.Count, width, layers,
            scaffold.Index.PrecisionMap, representations, segments);
        Phase("wrote container metadata");

        return new MergeReport(
            OutDir: outDir,
            ResultModel: $"{baseView.Model}+{importedView.Model}",
            BaseModel: baseView.Model,
            ImportedModel: importedView.Model,
            Scaffold: scaffold.Model,
            BaseVocab: mapping.BaseVocab,
            ImportedVocab: mapping.ImportedVocab,
            MergedVocab: mapping.Count,
            Blended: mapping.BlendedCount,
            AlignedNew: mapping.Count - mapping.BlendedCount - (mapping.BaseVocab - mapping.BlendedCount),
            BaseOnly: mapping.BaseVocab - mapping.BlendedCount,
            HiddenSize: width,
            Layers: layers,
            Dtype: mergedDtype.Label(),
            HeadTied: mergedTied,
            PreservedSegments: preservedSegments.Count,
            Notes: notes);
    }

    // ── imported source handling ──────────────────────────────────────────

    private sealed class ImportedSource : IDisposable
    {
        public Vindex3Container Container { get; }
        private readonly string? _tempDir;

        public ImportedSource(Vindex3Container container, string? tempDir)
        {
            Container = container;
            _tempDir = tempDir;
        }

        public void Dispose()
        {
            Container.Dispose();
            if (_tempDir is not null && Directory.Exists(_tempDir))
            {
                Directory.Delete(_tempDir, recursive: true);
            }
        }
    }

    private static ImportedSource OpenImported(string path, bool isContainer)
    {
        var indexPath = Path.Combine(path, "index.json");
        bool looksLikeContainer = File.Exists(indexPath);
        if (!isContainer && !looksLikeContainer)
        {
            // A plain checkpoint directory: encode it into a scratch
            // container first, so both sides of the merge are containers.
            string temp = Path.Combine(Path.GetTempPath(), $"amql-merge-{Guid.NewGuid():N}");
            string modelName = Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            ModelToContainer.Encode(path, temp, modelName);
            return new ImportedSource(Vindex3Container.Open(temp), temp);
        }
        if (isContainer && !looksLikeContainer)
        {
            throw new MergeException($"'{path}' has no index.json — not a container");
        }
        return new ImportedSource(Vindex3Container.Open(path), null);
    }

    // ── per-model view ────────────────────────────────────────────────────

    private sealed record TableView(
        string ObjectId,
        string Dtype,
        long[] Shape,
        Vindex3Container Container)
    {
        public int Rows => (int)Shape[0];
        public int Width => (int)Shape[1];
        public Dtype DtypeEnum => DtypeExtensions.FromLabel(Dtype);
    }

    private sealed record ModelView(
        string Model,
        Vindex3Container Container,
        SystemGraph Graph,
        Component Component,
        TokenVocabulary Vocab,
        TableView Embedding,
        TableView? OutputHead,
        RepresentationEntry StackEntry,
        RepresentationEntry FinalNormEntry,
        string StackEncoding,
        string FinalNormEncoding,
        int Hidden,
        int Layers)
    {
        public bool HeadTied => OutputHead is null;
        public int Width => Embedding.Width;
        public Vindex3Index Index => Container.Index;

        public static ModelView Open(Vindex3Container container, string which)
        {
            var graph = container.Graph
                ?? throw new MergeException($"the {which} container records no system graph — nothing to merge");
            var component = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.PrimaryText)
                ?? throw new MergeException($"the {which} container records no primary text component");
            var embedding = graph.Objects.FirstOrDefault(o => o.Component == component.Id && o.Kind == ObjectKind.Embedding)
                ?? throw new MergeException($"the {which} container records no embedding object");
            if (embedding.Representations.Count == 0)
            {
                throw new MergeException($"the {which} container's embedding is carried without tensors — nothing to merge");
            }

            var embedEntry = SegmentEntry(container, embedding.Id, which);
            var embedTensor = SingleTensor(container, embedEntry, embedding.Id, which);
            var stackEntry = SegmentEntry(container, "target.decoder_stack", which);
            var finalNormEntry = SegmentEntry(container, "target.final_norm", which);

            TableView? head = null;
            var headObject = graph.Objects.FirstOrDefault(o => o.Component == component.Id && o.Kind == ObjectKind.OutputHead);
            if (headObject is { Representations.Count: > 0 })
            {
                var headEntry = SegmentEntry(container, headObject.Id, which);
                var headTensor = SingleTensor(container, headEntry, headObject.Id, which);
                head = new TableView(headObject.Id, headTensor.Dtype, headTensor.Shape, container);
            }

            return new ModelView(
                container.Index.Model,
                container,
                graph,
                component,
                TokenVocabulary.ReadFromContainer(container),
                new TableView(embedding.Id, embedTensor.Dtype, embedTensor.Shape, container),
                head,
                stackEntry,
                finalNormEntry,
                stackEntry.Encoding,
                finalNormEntry.Encoding,
                component.HiddenSize,
                component.NumLayers);
        }

        private static RepresentationEntry SegmentEntry(Vindex3Container container, string objectId, string which)
        {
            string repId = container.CanonicalRepresentationId(objectId);
            if (!container.Index.Representations.TryGetValue(repId, out var entry))
            {
                throw new MergeException(
                    $"the {which} container records no representation '{repId}' — cannot read '{objectId}'");
            }
            return entry;
        }

        private static SegmentTensor SingleTensor(Vindex3Container container, RepresentationEntry entry, string objectId, string which)
        {
            using var segment = SegmentFile.Open(Path.Combine(container.Root, entry.Segment));
            if (segment.Header.Tensors.Count != 1)
            {
                throw new MergeException(
                    $"the {which} container's '{objectId}' segment has {segment.Header.Tensors.Count} tensors — expected one");
            }
            return segment.Header.Tensors[0];
        }
    }

    // ── anchored alignment + consensus table merge ────────────────────────

    private sealed record Alignment(
        float[] Map,
        int Anchors,
        double ResidualL2,
        double RidgeRel,
        double Threshold);

    private sealed record MergeTableResult(
        IReadOnlyList<byte[]> Chunks,
        double MeanAgreement,
        double ShareAboveThreshold);

    /// <summary>Fits the least-squares map from the OTHER model's table
    /// into the SCAFFOLD model's table on the shared-token anchors; both
    /// tables are already widened to the merged width. The scaffold model
    /// is the one whose stack runs, so its space is where the merged row
    /// must live.</summary>
    private static Alignment AlignRows(
        float[] scaffoldRows,
        float[] otherRows,
        TokenMapping mapping,
        bool scaffoldIsImported,
        int width,
        double threshold)
    {
        var anchors = mapping.Anchors;
        var a = new float[anchors.Count * width];
        var b = new float[anchors.Count * width];

        for (int k = 0; k < anchors.Count; k++)
        {
            var anchor = anchors[k];
            Array.Copy(scaffoldRows, (scaffoldIsImported ? anchor.ImportedId : anchor.BaseId)!.Value * width,
                a, k * width, width);
            Array.Copy(otherRows, (scaffoldIsImported ? anchor.BaseId : anchor.ImportedId)!.Value * width,
                b, k * width, width);
        }

        if (anchors.Count == 0)
        {
            Console.Error.WriteLine($"[{ProgressClock.Elapsed.TotalSeconds,7:0.0}s] alignment fit: no shared-token anchors");
            Console.Error.Flush();
        }
        else if (MergeGpu.Enabled)
        {
            Console.Error.WriteLine($"[{ProgressClock.Elapsed.TotalSeconds,7:0.0}s] alignment fit on GPU ({anchors.Count} anchors x {width})");
            Console.Error.Flush();
        }
        else
        {
            Console.Error.WriteLine($"[{ProgressClock.Elapsed.TotalSeconds,7:0.0}s] alignment fit on CPU ({anchors.Count} anchors x {width})");
            Console.Error.Flush();
        }
        var map = LeastSquares.Fit(a, b, anchors.Count, width);
        return new Alignment(map, anchors.Count, LeastSquares.ResidualL2(a, b, anchors.Count, width, map), 1e-4, threshold);
    }

    /// <summary>Builds the merged table for one token-interface tensor in
    /// the scaffold's space: shared tokens pass a cosine-agreement gate —
    /// agree → blend the rows and restore the scaffold's energy (reinforce
    /// consensus), disagree → keep the scaffold's row verbatim (never
    /// fabricate a direction) — scaffold-only rows are copied verbatim and
    /// other-only rows come through the alignment map. Every row is
    /// independent, so the rows merge in parallel.</summary>
    private static MergeTableResult MergeTable(
        float[] scaffoldRows,
        float[] otherRows,
        TokenMapping mapping,
        bool scaffoldIsImported,
        Alignment alignment,
        int width,
        Dtype mergedDtype)
    {
        var m = alignment.Map;
        var merged = new float[mapping.Count * width];
        var entries = mapping.Entries;
        var agreement = new double[mapping.Count]; // NaN where not a shared token

        // CUDA path: the per-token ApplyMap (M·other, a d×d matvec per
        // shared token ≈ O(n·d²) — the other ~40-minute CPU phase at 27B
        // scale) is one blocked GEMM aligned = otherRows @ Mᵀ over all
        // rows; the token loop below then copies the precomputed aligned
        // row instead of running the matvec. Falls back to the managed
        // ApplyMap on any GPU failure.
        float[]? alignedRows = null;
        if (MergeGpu.TryMapApply(otherRows, otherRows.Length / width, width, m, out alignedRows))
        {
            alignedRows ??= Array.Empty<float>();
        }

        Parallel.For(
            0,
            mapping.Count,
            new ParallelOptions { MaxDegreeOfParallelism = ComputeBudget.Cores },
            i =>
        {
            var entry = entries[i];
            int? scaffoldId = scaffoldIsImported ? entry.ImportedId : entry.BaseId;
            int? otherId = scaffoldIsImported ? entry.BaseId : entry.ImportedId;
            int dest = i * width;

            if (scaffoldId is { } s)
            {
                if (otherId is { } o)
                {
                    if (alignedRows is { Length: > 0 })
                    {
                        Array.Copy(alignedRows, o * width, merged, dest, width);
                    }
                    else
                    {
                        ApplyMap(otherRows, o, merged, i, m, width);
                    }
                    double c = Cosine(merged, dest, scaffoldRows, s * width, width);
                    agreement[i] = c;
                    if (double.IsFinite(c) && c >= alignment.Threshold)
                    {
                        for (int k = 0; k < width; k++)
                        {
                            merged[dest + k] = 0.5f * (scaffoldRows[s * width + k] + merged[dest + k]);
                        }
                        RestoreEnergy(merged, dest, scaffoldRows, s * width, width);
                    }
                    else
                    {
                        Array.Copy(scaffoldRows, s * width, merged, dest, width);
                    }
                }
                else
                {
                    Array.Copy(scaffoldRows, s * width, merged, dest, width);
                }
            }
            else
            {
                if (alignedRows is { Length: > 0 })
                {
                    Array.Copy(alignedRows, otherId!.Value * width, merged, dest, width);
                }
                else
                {
                    ApplyMap(otherRows, otherId!.Value, merged, i, m, width);
                }
            }
        });

        int shared = 0;
        double sum = 0;
        int above = 0;
        foreach (double c in agreement)
        {
            if (double.IsFinite(c))
            {
                shared++;
                sum += c;
                if (c >= alignment.Threshold)
                {
                    above++;
                }
            }
        }

        return new MergeTableResult(
            EncodeChunks(mergedDtype, merged),
            shared == 0 ? 0 : sum / shared,
            shared == 0 ? 0 : (double)above / shared);
    }

    /// <summary>Re-encodes the merged f32 table in pages so no encoded
    /// chunk ever approaches the 2 GiB single-array ceiling: the merged
    /// embedding/head can exceed 2.5 GiB of BF16 (5 GiB f32) and a whole
    /// payload would overflow the byte[] length arithmetic. Mirrors the
    /// exporter's page size (2²⁶ elements ≈ 256 MiB of stored bytes).</summary>
    private static IReadOnlyList<byte[]> EncodeChunks(Dtype dtype, float[] values, int pageElements = 1 << 26)
    {
        var chunks = new List<byte[]>();
        for (int start = 0; start < values.Length; start += pageElements)
        {
            int count = Math.Min(pageElements, values.Length - start);
            chunks.Add(EncodeToDtype(dtype, values.AsSpan(start, count).ToArray()));
        }
        return chunks;
    }

    /// <summary>M·row: the other model's row mapped into the scaffold's
    /// space. Writes into <c>dest[destRow]</c>.</summary>
    private static void ApplyMap(float[] source, int row, float[] dest, int destRow, float[] m, int width)
    {
        int src = row * width;
        for (int r = 0; r < width; r++)
        {
            double sum = 0;
            for (int c = 0; c < width; c++)
            {
                sum += m[r * width + c] * source[src + c];
            }
            dest[destRow * width + r] = (float)sum;
        }
    }

    private static double Cosine(float[] a, int aOff, float[] b, int bOff, int width)
    {
        double dot = 0, na = 0, nb = 0;
        for (int c = 0; c < width; c++)
        {
            float av = a[aOff + c];
            float bv = b[bOff + c];
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
        }
        if (na == 0 || nb == 0)
        {
            return double.NaN;
        }
        return dot / Math.Sqrt(na * nb);
    }

    /// <summary>Scales <c>row</c> so its energy equals the reference row's —
    /// the consensus reinforcement: agreeing blends keep the scaffold's
    /// full logit magnitude instead of the averaging collapse.</summary>
    private static void RestoreEnergy(float[] row, int off, float[] reference, int refOff, int width)
    {
        double rowEnergy = 0, refEnergy = 0;
        for (int c = 0; c < width; c++)
        {
            rowEnergy += (double)row[off + c] * row[off + c];
            refEnergy += (double)reference[refOff + c] * reference[refOff + c];
        }
        if (rowEnergy <= 0 || refEnergy <= 0)
        {
            return;
        }
        float scale = (float)Math.Sqrt(refEnergy / rowEnergy);
        for (int c = 0; c < width; c++)
        {
            row[off + c] *= scale;
        }
    }

    private static string AgreementNote(MergeTableResult result, string scaffoldModel, string what = "embedding")
    {
        if (result.ShareAboveThreshold >= 0.999)
        {
            return $"{what} consensus: {result.ShareAboveThreshold * 100:0.0}% of shared tokens agree " +
                   $"(mean {result.MeanAgreement:0.000}) — blended and reinforced toward {scaffoldModel}";
        }
        return $"{what} consensus: {result.ShareAboveThreshold * 100:0.0}% of shared tokens agree " +
               $"(mean {result.MeanAgreement:0.000}) — agreeing rows blend and restore {scaffoldModel}'s energy, " +
               "disagreements defer to the scaffold's own row";
    }

    /// <summary>All rows of a token-interface table, widened to the merged
    /// width (rows narrower than the target are zero-extended).</summary>
    private static float[] ReadWidenedRows(TableView table, int width)
    {
        using var store = table.Container.CreateOperandStore();
        var resolution = store.ResolveWidened(table.ObjectId, "weight");
        var values = resolution.Values;
        if (table.Width == width)
        {
            return values;
        }
        var padded = new float[table.Rows * width];
        for (int r = 0; r < table.Rows; r++)
        {
            Array.Copy(values, r * table.Width, padded, r * width, table.Width);
        }
        return padded;
    }

    // ── stack scaffold + provenance ───────────────────────────────────────

    private sealed record TensorProvenance(string Model, string Operation);

    private sealed record StackPlan(
        string ScaffoldModel,
        int LayerCount,
        string StackEncoding,
        string FinalNormEncoding,
        string? RebuiltEncoding,
        List<NamedTensorData>? Rebuilt,
        IReadOnlyDictionary<string, TensorProvenance> Tensors)
    {
        public bool NeedsRebuild => Rebuilt is not null;
    }

    /// <summary>Copies the scaffold's stack verbatim (byte-identical, the
    /// ordinary case). Only when the scaffold lacks a tensor kind the other
    /// model has is the stack rebuilt in memory, with the missing tensors
    /// zero-padded to the scaffold's shapes and marked "grown".</summary>
    private static StackPlan PlanStack(ModelView baseView, ModelView importedView, ModelView scaffold)
    {
        var other = ReferenceEquals(scaffold, baseView) ? importedView : baseView;

        var scaffoldTensors = StackTensors(scaffold);
        var otherTensors = StackTensors(other);
        var provenance = new Dictionary<string, TensorProvenance>(StringComparer.Ordinal);
        foreach (var tensor in scaffoldTensors)
        {
            provenance[tensor.Name] = new TensorProvenance(scaffold.Model, "copied");
        }

        var grown = otherTensors.Where(t => !provenance.ContainsKey(t.Name)).ToList();
        if (grown.Count == 0)
        {
            return new StackPlan(scaffold.Model, scaffold.Layers, scaffold.StackEncoding,
                scaffold.FinalNormEncoding, null, null, provenance);
        }

        // Rare path: rebuild the stack segment in memory with the extra
        // tensors grown (zero-padded) from the other model.
        var bySuffix = scaffoldTensors
            .GroupBy(t => Suffix(t.Name))
            .ToDictionary(g => g.Key, g => g.First().Shape);
        var rebuilt = new List<NamedTensorData>();
        using (var scaffoldSegment = SegmentFile.Open(Path.Combine(scaffold.Container.Root, scaffold.StackEntry.Segment)))
        {
            foreach (var tensor in scaffoldTensors)
            {
                rebuilt.Add(new NamedTensorData
                {
                    Name = tensor.Name,
                    Dtype = DtypeExtensions.FromLabel(tensor.Dtype),
                    Shape = tensor.Shape,
                    Data = scaffoldSegment.ReadBytes(tensor.Name),
                });
            }
        }

        using (var otherSegment = SegmentFile.Open(Path.Combine(other.Container.Root, other.StackEntry.Segment)))
        {
            foreach (var tensor in grown)
            {
                var dtype = DtypeExtensions.FromLabel(tensor.Dtype);
                var grownShape = bySuffix.TryGetValue(Suffix(tensor.Name), out var targetShape)
                    ? targetShape
                    : tensor.Shape;
                var data = otherSegment.ReadBytes(tensor.Name);
                if (!ShapesEqual(grownShape, tensor.Shape))
                {
                    data = GrowToShape(tensor.Shape, grownShape, data, dtype);
                }
                provenance[tensor.Name] = new TensorProvenance(other.Model, "grown");
                rebuilt.Add(new NamedTensorData
                {
                    Name = tensor.Name,
                    Dtype = dtype,
                    Shape = grownShape,
                    Data = data,
                });
            }
        }

        return new StackPlan(scaffold.Model, scaffold.Layers, scaffold.StackEncoding,
            scaffold.FinalNormEncoding, scaffold.StackEncoding, rebuilt, provenance);
    }

    private static List<SegmentTensor> StackTensors(ModelView view)
    {
        using var segment = SegmentFile.Open(Path.Combine(view.Container.Root, view.StackEntry.Segment));
        return segment.Header.Tensors.ToList();
    }

    private static string Suffix(string tensorName) =>
        tensorName.Contains('.') ? tensorName[(tensorName.IndexOf('.') + 1)..] : tensorName;

    private static bool ShapesEqual(long[] a, long[] b) => a.SequenceEqual(b);

    /// <summary>Zero-pads a stored tensor to a larger shape (elementwise,
    /// top-left anchored — the standard deterministic growth operator).</summary>
    private static byte[] GrowToShape(long[] from, long[] to, byte[] data, Dtype dtype)
    {
        if (from.Length != to.Length)
        {
            throw new MergeException("cannot grow a tensor across rank");
        }
        int elemSize = dtype.ElementSize();
        long toElements = 1;
        foreach (var dim in to)
        {
            toElements = checked(toElements * dim);
        }
        var result = new byte[toElements * elemSize];
        // Walk the source index space; every source element lands at the
        // same dest index (dest dims ≥ source dims), dest is zero-filled.
        var strides = new long[from.Length];
        long stride = 1;
        for (int d = from.Length - 1; d >= 0; d--)
        {
            strides[d] = stride;
            stride *= to[d];
        }
        var index = new int[from.Length];
        for (long src = 0; src < data.Length / elemSize; src++)
        {
            // decompose src into per-dim indices
            long rem = src;
            for (int d = from.Length - 1; d >= 0; d--)
            {
                index[d] = (int)(rem % from[d]);
                rem /= from[d];
            }
            long dst = 0;
            for (int d = 0; d < from.Length; d++)
            {
                dst += index[d] * strides[d];
            }
            Array.Copy(data, src * elemSize, result, dst * elemSize, elemSize);
        }
        return result;
    }

    // ── segment copying / writing ─────────────────────────────────────────

    private static void CopyMaterialised(
        string outDir,
        ModelView scaffold,
        string objectId,
        string encoding,
        string? rebuiltEncoding,
        List<NamedTensorData>? rebuilt,
        Dictionary<string, RepresentationEntry> representations,
        Dictionary<string, int> segments)
    {
        if (rebuilt is not null && objectId == "target.decoder_stack")
        {
            // Rebuilt stack (rare grown path): write a fresh segment whose
            // hashes are recomputed by the writer.
            var result = SegmentWriter.Write(
                Path.Combine(outDir, "segments", "target.decoder_stack.bin"),
                $"target.decoder_stack@{rebuiltEncoding}",
                rebuilt);
            representations[$"target.decoder_stack@{rebuiltEncoding}"] = new RepresentationEntry
            {
                Object = "target.decoder_stack",
                Encoding = rebuiltEncoding!,
                Segment = "segments/target.decoder_stack.bin",
                TensorCount = rebuilt.Count,
                PayloadBytes = result.PayloadBytes,
                PayloadSha256 = result.PayloadSha256Hex,
                SegmentSha256 = result.SegmentSha256Hex,
            };
            segments["segments/target.decoder_stack"] = 1;
            return;
        }

        string repId = scaffold.Container.CanonicalRepresentationId(objectId);
        var entry = scaffold.Index.Representations[repId];
        Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(outDir, entry.Segment))!);
        File.Copy(Path.Combine(scaffold.Container.Root, entry.Segment), Path.Combine(outDir, entry.Segment));
        // Byte-identical copy: the source's hashes remain valid.
        representations[$"{entry.Object}@{encoding}"] = new RepresentationEntry
        {
            Object = entry.Object,
            Encoding = encoding,
            Segment = entry.Segment,
            TensorCount = entry.TensorCount,
            PayloadBytes = entry.PayloadBytes,
            PayloadSha256 = entry.PayloadSha256,
            SegmentSha256 = entry.SegmentSha256,
        };
        segments[entry.Segment[..^4]] = 1;
    }

    // ── container files ───────────────────────────────────────────────────

    private static void WriteGraph(
        string outDir,
        ModelView baseView,
        ModelView scaffold,
        int vocab,
        int width,
        int layers,
        bool mergedTied,
        Dtype mergedDtype,
        bool headUntied,
        Dtype? headDtype)
    {
        var baseGraph = baseView.Graph;
        var scaffoldSurface = scaffold.Component.Execution!;
        var surface = new ExecutionSurface
        {
            ContextLength = scaffoldSurface.ContextLength,
            Attention = scaffoldSurface.Attention,
            Ffn = scaffoldSurface.Ffn,
            Norm = scaffoldSurface.Norm,
            Head = scaffoldSurface.Head is null
                ? null
                : new HeadSurface
                {
                    VocabSize = vocab,
                    EmbeddingNorm = scaffoldSurface.Head.EmbeddingNorm,
                    EmbedScale = scaffoldSurface.Head.EmbedScale,
                    OutputMultiplier = scaffoldSurface.Head.OutputMultiplier,
                    FinalLogitSoftcapping = scaffoldSurface.Head.FinalLogitSoftcapping,
                    HeadReusesEmbedding = mergedTied,
                },
            ResidualScale = scaffoldSurface.ResidualScale,
            LinearAttention = scaffoldSurface.LinearAttention,
            Kda = scaffoldSurface.Kda,
            KdaGateLowerBound = scaffoldSurface.KdaGateLowerBound,
            Mla = scaffoldSurface.Mla,
            Mamba2 = scaffoldSurface.Mamba2,
            ConvQkv = scaffoldSurface.ConvQkv,
            ResidualInFp32 = scaffoldSurface.ResidualInFp32,
        };

        var components = new List<Component>(baseGraph.Components.Count);
        foreach (var component in baseGraph.Components)
        {
            if (component.Role == ComponentRole.PrimaryText)
            {
                components.Add(new Component
                {
                    Id = component.Id,
                    Role = ComponentRole.PrimaryText,
                    SourceArtifact = scaffold.Component.SourceArtifact,
                    NumLayers = layers,
                    HiddenSize = width,
                    Attention = scaffold.Component.Attention,
                    Execution = surface,
                    Perception = null,
                });
            }
            else
            {
                components.Add(component);
            }
        }

        var scaffoldStackReps = scaffold.Graph.Objects
            .First(o => o.Kind == ObjectKind.DecoderStack).Representations;
        var scaffoldNormReps = scaffold.Graph.Objects
            .First(o => o.Kind == ObjectKind.FinalNorm).Representations;

        var objects = new List<LogicalObject>();
        objects.AddRange(baseGraph.Objects.Where(o => o.Component != baseView.Component.Id));
        objects.Add(RebuildTargetObject(baseGraph, baseView.Component.Id, ObjectKind.Embedding,
            reps: new List<Representation>
            {
                new() { Encoding = mergedDtype.Label(), Fidelity = Fidelity.Canonical },
            }, boundBytes: checked((long)vocab * width * mergedDtype.ElementSize())));
        objects.Add(RebuildTargetObject(baseGraph, baseView.Component.Id, ObjectKind.DecoderStack,
            reps: new List<Representation>(scaffoldStackReps), boundBytes: null));
        objects.Add(RebuildTargetObject(baseGraph, baseView.Component.Id, ObjectKind.FinalNorm,
            reps: new List<Representation>(scaffoldNormReps), boundBytes: null));
        objects.Add(RebuildTargetObject(baseGraph, baseView.Component.Id, ObjectKind.OutputHead,
            reps: headUntied
                ? new List<Representation>
                {
                    new() { Encoding = headDtype!.Value.Label(), Fidelity = Fidelity.Canonical },
                }
                : new List<Representation>(),
            boundBytes: headUntied ? checked((long)vocab * width * headDtype!.Value.ElementSize()) : null));

        var graph = new SystemGraph
        {
            Schema = baseGraph.Schema,
            Components = components,
            Objects = objects,
            Edges = baseGraph.Edges,
        };
        File.WriteAllText(Path.Combine(outDir, "system_graph.json"),
            JsonSerializer.Serialize(graph, ViJson.Options));
    }

    private static LogicalObject RebuildTargetObject(
        SystemGraph graph,
        string componentId,
        ObjectKind kind,
        List<Representation> reps,
        long? boundBytes)
    {
        var source = graph.Objects.First(o => o.Component == componentId && o.Kind == kind);
        var bindings = new List<SourceBinding>(source.SourceBindings.Count);
        for (int i = 0; i < source.SourceBindings.Count; i++)
        {
            var binding = source.SourceBindings[i];
            bindings.Add(i == 0 && boundBytes is { } bytes
                ? new SourceBinding
                {
                    Artifact = binding.Artifact,
                    TensorPrefix = binding.TensorPrefix,
                    Tensors = binding.Tensors,
                    Bytes = bytes,
                }
                : binding);
        }
        return new LogicalObject
        {
            Id = source.Id,
            Component = source.Component,
            Kind = source.Kind,
            SourceBindings = bindings,
            Representations = reps,
        };
    }

    private static void WriteIndex(
        string outDir,
        ModelView baseView,
        ModelView importedView,
        ModelView scaffold,
        int vocab,
        int width,
        int layers,
        PrecisionMap? precisionMap,
        Dictionary<string, RepresentationEntry> representations,
        Dictionary<string, int> segments)
    {
        var index = new Vindex3Index
        {
            Version = Vindex3Index.CurrentSchema,
            Model = $"{baseView.Model}+{importedView.Model}",
            Family = scaffold.Index.Family,
            HiddenSize = width,
            NumLayers = layers,
            SystemGraph = "system_graph.json",
            Representations = representations,
            Profiles = new List<Profile> { Profile.Exact() },
            Segments = segments,
            Authority = ContainerAuthority.Derived,
            DerivedFromModel = baseView.Model,
            PrecisionMap = precisionMap,
            TokenMap = ManifestName,
        };
        File.WriteAllText(Path.Combine(outDir, "index.json"),
            JsonSerializer.Serialize(index, ViJson.Options));
    }

    private static JsonObject BuildManifest(
        ModelView baseView,
        ModelView importedView,
        TokenMapping mapping,
        int width,
        Dtype mergedDtype,
        Dtype? headDtype,
        bool mergedTied,
        StackPlan stack,
        Alignment alignment,
        MergeTableResult embeddingMerge,
        string? preservedModel,
        IReadOnlyList<string> preservedSegments)
    {
        var json = new JsonObject
        {
            ["schema"] = 1,
            ["result_model"] = $"{baseView.Model}+{importedView.Model}",
            ["base"] = new JsonObject { ["model"] = baseView.Model, ["vocab"] = mapping.BaseVocab },
            ["imported"] = new JsonObject { ["model"] = importedView.Model, ["vocab"] = mapping.ImportedVocab },
            ["anchors"] = new JsonObject
            {
                ["count"] = alignment.Anchors,
                ["ridge_rel"] = alignment.RidgeRel,
                ["residual_l2"] = alignment.ResidualL2,
                ["space"] = stack.ScaffoldModel,
                ["agreement_threshold"] = alignment.Threshold,
                ["mean_agreement"] = embeddingMerge.MeanAgreement,
                ["share_above_threshold"] = embeddingMerge.ShareAboveThreshold,
            },
            ["embedding"] = new JsonObject
            {
                ["dtype"] = mergedDtype.Label(),
                ["width"] = width,
                ["rows"] = mapping.Count,
            },
            ["head"] = new JsonObject
            {
                ["tied"] = mergedTied,
                ["dtype"] = headDtype?.Label() ?? mergedDtype.Label(),
                ["rows"] = mapping.Count,
            },
            ["scaffold"] = new JsonObject
            {
                ["model"] = stack.ScaffoldModel,
                ["layers"] = stack.LayerCount,
                ["hidden"] = width,
            },
        };

        var vocab = new JsonArray();
        foreach (var entry in mapping.Entries)
        {
            var item = new JsonObject
            {
                ["id"] = entry.Id,
                ["token"] = entry.Token,
                ["kind"] = entry.KindLabel,
            };
            if (entry.BaseId is { } baseId)
            {
                item["base_id"] = baseId;
            }
            if (entry.ImportedId is { } importedId)
            {
                item["imported_id"] = importedId;
            }
            vocab.Add(item);
        }
        json["vocab"] = vocab;

        var layerTable = new JsonArray();
        foreach (var group in stack.Tensors.GroupBy(kv => LayerOf(kv.Key)).OrderBy(g => g.Key))
        {
            var tensors = new JsonObject();
            foreach (var (name, provenance) in group.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            {
                tensors[name] = new JsonObject
                {
                    ["model"] = provenance.Model,
                    ["operation"] = provenance.Operation,
                };
            }
            layerTable.Add(new JsonObject
            {
                ["layer"] = group.Key,
                ["tensors"] = tensors,
            });
        }
        json["layer_table"] = layerTable;

        if (preservedModel is not null)
        {
            var preserved = new JsonArray();
            foreach (var segment in preservedSegments)
            {
                preserved.Add(segment);
            }
            json["preserved"] = new JsonObject { ["model"] = preservedModel, ["segments"] = preserved };
        }
        return json;
    }

    private static int LayerOf(string tensorName)
    {
        int dot = tensorName.IndexOf('.');
        return dot > 0 && int.TryParse(tensorName[..dot], out int layer) ? layer : -1;
    }

    // ── storage dtype encoding (f32 → stored bytes) ───────────────────────

    private static byte[] EncodeToDtype(Dtype dtype, float[] values) => dtype switch
    {
        Dtype.F32 => ToF32Bytes(values),
        Dtype.BF16 => ToBf16Bytes(values),
        Dtype.F16 => ToF16Bytes(values),
        _ => throw new MergeException(
            $"the merged token-interface table would need '{dtype.Label()}' storage — " +
            "this build merges into F32/BF16/F16 only"),
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

    private static string Sanitize(string model) =>
        string.Concat(model.Select(c => Path.GetInvalidFileNameChars().Contains(c) ? '_' : c));
}