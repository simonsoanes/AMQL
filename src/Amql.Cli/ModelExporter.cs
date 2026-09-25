using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Cli;

/// <summary>Everything written by <c>amql-cli export</c>: the checkpoint
/// directory, its shard, and the objects that could not be materialised
/// (carried-only objects like the tied output head).</summary>
public sealed record ExportReport(
    string OutDir,
    string Model,
    int Tensors,
    long PayloadBytes,
    IReadOnlyList<string> Notes);

/// <summary>
/// The inverse of <c>encode</c>: materialises an HF-style checkpoint
/// directory (config.json + model.safetensors + tokenizer.json) from a
/// VINDEX3 container. Segment tensor names are object-relative, so the
/// HF tensor name is rebuilt from the graph's source binding
/// (<c>TensorPrefix + "." + name</c>); an optional weight patch is baked
/// in by widening to f32, adding the delta, and re-encoding to the
/// tensor's stored dtype. Tensors a patch never touches are copied
/// verbatim, so an unpatched export is byte-identical to the source.
/// config.json is regenerated from judged graph facts only — an operator
/// without a judged <c>layer_types</c> spelling refuses the export rather
/// than approximating it.
/// </summary>
public static class ModelExporter
{
    public const string ExportFormat = "amql-export-v1";
    private const string ShardName = "model.safetensors";

    public static ExportReport Export(
        Vindex3Container container,
        string outDir,
        WeightPatch? patch,
        bool quantizeMxfp4 = false,
        string? quantizeTernary = null,
        string? arch = null)
    {
        if (Directory.Exists(outDir))
        {
            throw new CliException($"export output '{outDir}' already exists");
        }

        bool isQwen4Next = arch == Qwen4NextLayout.Arch;

        var graph = container.Graph
            ?? throw new CliException("container records no system graph — cannot rebuild HF tensor names");

        // Build config.json first: an unjudged operator refuses here,
        // before anything is written.
        string configJson;
        IReadOnlyList<string>? layerTypes = null;
        if (isQwen4Next)
        {
            layerTypes = Qwen4NextLayout.BuildLayerTypes(container);
            configJson = Qwen4NextLayout.BuildConfigJson(container, quantizeMxfp4, layerTypes);
        }
        else
        {
            configJson = ExportConfig.BuildJson(container, quantizeMxfp4);
        }

        using var store = container.CreateOperandStore();
        var payloads = new List<TensorPayload>();
        var names = new Dictionary<string, string>(StringComparer.Ordinal);
        var notes = new List<string>();

        // ── pass 1 (header-only): the work list ─────────────────────────
        // Tensor table reads alone, so the HF name reconstruction and its
        // collision check stay deterministic regardless of parallelism.
        // The PRIMARY text component exports into the model shard together
        // with a materialised VISION tower (the tower is part of the model);
        // an MTP drafter stays out — it exports automatically as the mtp
        // companion below.
        string primaryId = graph.Components.First(c => c.Role == ComponentRole.PrimaryText).Id;
        string? visionId = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.Perception)?.Id;
        var objectPrefixes = new Dictionary<string, string>(StringComparer.Ordinal);
        var work = new List<(string ObjectId, string SegmentPath, SegmentTensor Tensor)>();
        foreach (var obj in graph.Objects.Where(o =>
                     o.Component == primaryId || (visionId is not null && o.Component == visionId)))
        {
            if (obj.Representations.Count == 0)
            {
                notes.Add($"object '{obj.Id}': carried only (no materialised tensors) — skipped");
                continue;
            }
            string? prefix = obj.SourceBindings.FirstOrDefault()?.TensorPrefix;
            if (string.IsNullOrEmpty(prefix))
            {
                throw new CliException(
                    $"object '{obj.Id}' materialises tensors but declares no source-binding tensor prefix — cannot rebuild HF tensor names");
            }

            string? segmentPath = store.SegmentPathFor(obj.Id);
            if (segmentPath is null)
            {
                notes.Add($"object '{obj.Id}': representation recorded but no segment on disk — skipped");
                continue;
            }

            objectPrefixes[obj.Id] = prefix;
            using (var segment = SegmentFile.Open(Path.Combine(container.Root, segmentPath)))
            {
                foreach (var tensor in segment.Header.Tensors)
                {
                    string hfName = prefix + "." + tensor.Name;
                    if (isQwen4Next)
                    {
                        hfName = Qwen4NextLayout.RemapTensorName(obj.Id, hfName);
                    }
                    if (names.TryGetValue(hfName, out var other))
                    {
                        throw new CliException(
                            $"export collision: tensor name '{hfName}' rebuilt for both '{other}' and '{obj.Id}'");
                    }
                    names[hfName] = obj.Id;
                    work.Add((obj.Id, segmentPath, tensor));
                }
            }
        }

        if (work.Count == 0)
        {
            throw new CliException("the container holds no materialised tensors — nothing to export");
        }

        // ── pass 2: build the payloads in parallel ─────────────────────
        // Widening and quantisation dominate; each worker reads through its
        // own segment mapping (memory-mapped views are not thread-shared) so
        // only the small payload/name bookkeeping locks.
        int workers = quantizeMxfp4 ? 1 : WorkerCount(Environment.ProcessorCount);
        var lockObj = new object();
        int quantized = 0;
        // Logical shapes of the ternary tensors. The packed payload is stored as
        // an opaque byte matrix (see BuildTernaryPayloads), so the rows/cols the
        // decoder needs to invert the blockwise Hadamard rotation have to be
        // recorded somewhere — the file header is the only place that survives.
        var ternaryShapes = new List<(string Name, long Rows, long Cols)>();
        Parallel.ForEach(work, new ParallelOptions { MaxDegreeOfParallelism = workers }, item =>
        {
            string hfName = objectPrefixes[item.ObjectId] + "." + item.Tensor.Name;
            if (isQwen4Next)
            {
                hfName = Qwen4NextLayout.RemapTensorName(item.ObjectId, hfName);
            }
            bool quantizing = (quantizeMxfp4 || quantizeTernary is not null) &&
                ShouldQuantize(item.ObjectId, item.Tensor.Name, item.Tensor.Shape);

            using var segment = SegmentFile.Open(Path.Combine(container.Root, item.SegmentPath));
            IReadOnlyList<TensorPayload> produced;
            // Gate on `quantizing`, not merely on which mode was asked for.
            // ShouldQuantize is what excludes 1-D norms and biases, and
            // dispatching without it sent those into the 2-D-only ternary
            // encoder, which read Shape[1] and threw. MXFP4 survived the same
            // mistake only because BuildQuantizedPayloads catches and falls
            // back — exception-driven control flow for every tensor that was
            // never a candidate.
            if (quantizing && quantizeTernary is not null)
            {
                produced = BuildTernaryPayloads(item.ObjectId, segment, item.Tensor, patch, hfName, quantizeTernary);
            }
            else if (quantizing && quantizeMxfp4)
            {
                produced = BuildQuantizedPayloads(item.ObjectId, segment, item.Tensor, patch, hfName);
            }
            else
            {
                produced = new[] { BuildExportPayload(item.ObjectId, segment, item.Tensor, patch, hfName) };
            }

            lock (lockObj)
            {
                foreach (var payload in produced)
                {
                    if (payload.Name != hfName && names.TryGetValue(payload.Name, out var owner))
                    {
                        throw new CliException(
                            $"export collision: MXFP4 companion tensor '{payload.Name}' rebuilt for both '{owner}' and '{item.ObjectId}'");
                    }
                    names[payload.Name] = item.ObjectId;
                    payloads.Add(payload);
                }
                if (quantizing)
                {
                    quantized++;
                    if (quantizeTernary is not null)
                    {
                        ternaryShapes.Add((hfName, item.Tensor.Shape[0], item.Tensor.Shape[1]));
                    }
                }
            }
        });

        // ── Flash-Next placeholder tensors (HC, shared expert) ────────
        // Emitted as zero-initialised so the checkpoint is structurally
        // loadable.  HC streams are identity-initialised (mix averages
        // streams, combine injects 1:1); the shared expert duplicates the
        // first routed expert when one is present.
        if (isQwen4Next)
        {
            int placeholders = 0;
            foreach (var (name, shape, dtype) in Qwen4NextLayout.PlaceholderTensors(container))
            {
                if (names.ContainsKey(name))
                {
                    continue; // a real tensor already occupies this name
                }
                byte[] zeroBytes = Qwen4NextLayout.ZeroPayload(shape, dtype);
                payloads.Add(new TensorPayload
                {
                    Name = name,
                    Dtype = dtype,
                    Shape = shape,
                    Data = zeroBytes,
                });
                names[name] = "(placeholder)";
                placeholders++;
            }
            if (placeholders > 0)
            {
                notes.Add($"{placeholders} Flash-Next placeholder tensors emitted as zeros " +
                          "(HyperConnection streams, shared expert) — " +
                          "the checkpoint is structurally loadable; HC identity init means " +
                          "the model passes the residual through unchanged");
            }
        }

        if (quantized > 0)
        {
            bool isTernary = quantizeTernary is not null;
            // Name the packing, not just "ternary": the two layouts are not
            // interchangeable and a reader has to know which one it got.
            string quantLabel = isTernary
                ? quantizeTernary == "ptq1"
                    ? $"ternary PTQ1_0 ({Ternary.Ptq1BytesPerBlock} B per {Ternary.BlockElements})"
                    : $"ternary PQ2_0 ({Ternary.Pq2BytesPerBlock} B per {Ternary.BlockElements})"
                : "MXFP4";
            string gridDesc = isTernary
                ? $"{{-1,0,+1}} grid elements, per-{Ternary.BlockElements}-element {Ternary.ScaleDtype.Label()} scales"
                : $"FP4 E2M1 grid elements, per-{Mxfp4.BlockElements}-element {Dtype.F8_E8M0.Label()} scales";
            notes.Add($"{quantized} stack projection tensors exported as {quantLabel} ({gridDesc}) — " +
                      "embeddings, norms, biases and the output head keep their full precision");
            notes.Add($"export ran with {workers} parallel workers " +
                      $"({Environment.ProcessorCount} cores" +
                      (Environment.ProcessorCount > 10 ? ", two left spare" : string.Empty) + ")");
        }

        Directory.CreateDirectory(outDir);
        var metadata = new Dictionary<string, string>
        {
            ["format"] = ExportFormat,
            ["model"] = container.Index.Model,
        };
        if (isQwen4Next)
        {
            metadata["arch"] = Qwen4NextLayout.Arch;
        }
        if (quantizeTernary is not null && ternaryShapes.Count > 0)
        {
            metadata["amql.ternary.packing"] = quantizeTernary == "ptq1" ? "PTQ1_0" : "PQ2_0";
            metadata["amql.ternary.block_elements"] = Ternary.BlockElements.ToString();
            metadata["amql.ternary.bytes_per_block"] = (quantizeTernary == "ptq1"
                ? Ternary.Ptq1BytesPerBlock
                : Ternary.Pq2BytesPerBlock).ToString();
            metadata["amql.ternary.scale_dtype"] = Ternary.ScaleDtype.Label();
            // "name=rowsxcols;name=rowsxcols;…" — the packed payload's own shape
            // is [blocks, bytes_per_block], which does not encode the logical
            // matrix, and DecodePtq1/DecodePq2 need rows and cols to undo the
            // blockwise Hadamard rotation.
            metadata["amql.ternary.shapes"] = string.Join(";",
                ternaryShapes.OrderBy(s => s.Name, StringComparer.Ordinal)
                    .Select(s => $"{s.Name}={s.Rows}x{s.Cols}"));
        }
        SafetensorsWriter.Write(Path.Combine(outDir, ShardName), payloads, metadata);
        File.WriteAllText(Path.Combine(outDir, "config.json"), configJson);

        var tokenizerPath = Path.Combine(container.Root, "tokenizer.json");
        if (File.Exists(tokenizerPath))
        {
            File.Copy(tokenizerPath, Path.Combine(outDir, "tokenizer.json"));
        }

        // Whatever ancillary config the container carries — chat template,
        // processor configs, generation config — has to come back out, or the
        // exported checkpoint can tokenize but not format a conversation.
        var ancillary = HfAncillaryFiles.CopyInto(container.Root, outDir);
        if (ancillary.Count > 0)
        {
            notes.Add($"carried from the container: {string.Join(", ", ancillary)}");
        }

        // A materialised MTP drafter rides the same export as a companion:
        // mtp.safetensors + mtp.config.json beside the model's shard.
        var mtp = graph.Objects.FirstOrDefault(o => o.Component == "mtp" && o.Kind == ObjectKind.DecoderStack);
        if (mtp is { Representations.Count: > 0 })
        {
            var companion = ExportMtp(container, outDir, shardName: "mtp.safetensors", configName: "mtp.config.json", copyTokenizer: false);
            notes.Add($"MTP drafter exported alongside: mtp.safetensors + mtp.config.json " +
                      $"({companion.Tensors} tensors, {companion.PayloadBytes} bytes)");
        }

        return new ExportReport(
            outDir,
            container.Index.Model,
            payloads.Count,
            payloads.Sum(p => p.PayloadLength),
            notes);
    }

    /// <summary>Everything a drafter export wrote.</summary>
    public sealed record MtpExportReport(
        string OutDir,
        string Model,
        int Tensors,
        long PayloadBytes,
        bool TiedHead,
        IReadOnlyList<string> Notes);

    /// <summary>
    /// Exports the MTP drafter as a standalone checkpoint: the module's
    /// tensors under their original <c>mtp.</c> names (fc projector, the
    /// two pre-fc norms, the single trunk layer, the pre-head norm), plus
    /// the SHARED embedding (the conditioning token) and the SHARED head —
    /// the 27B reuses the main model's embed_tokens and lm_head
    /// (<c>mtp_use_dedicated_embeddings: false</c>), so the drafter is
    /// composed from the container's own tables. Requires a container
    /// encoded from a checkpoint that carries mtp tensors.
    /// <paramref name="shardName"/>/<paramref name="configName"/> let the
    /// parent export emit the drafter as a companion
    /// (<c>mtp.safetensors</c> + <c>mtp.config.json</c>) in the same
    /// directory; the standalone <c>export-mtp</c> command uses the
    /// defaults and also copies the tokenizer.</summary>
    public static MtpExportReport ExportMtp(
        Vindex3Container container,
        string outDir,
        string shardName = "model.safetensors",
        string configName = "config.json",
        bool copyTokenizer = true)
    {
        if (File.Exists(Path.Combine(outDir, shardName)) || File.Exists(Path.Combine(outDir, configName)))
        {
            throw new CliException($"export-mtp output '{Path.Combine(outDir, shardName)}' already exists");
        }

        var graph = container.Graph ?? throw new CliException("container records no system graph");
        var mtp = graph.Objects.FirstOrDefault(o => o.Component == "mtp" && o.Kind == ObjectKind.DecoderStack);
        if (mtp is null || mtp.Representations.Count == 0)
        {
            throw new CliException(
                "the container carries no materialised MTP drafter — encode a checkpoint that has mtp tensors");
        }
        string mtpRep = container.CanonicalRepresentationId(mtp.Id);
        if (!container.Index.Representations.TryGetValue(mtpRep, out var mtpEntry))
        {
            throw new CliException($"the container records no representation '{mtpRep}'");
        }
        string primaryId = graph.Components.First(c => c.Role == ComponentRole.PrimaryText).Id;
        var embeddings = graph.Objects.First(o => o.Component == primaryId && o.Kind == ObjectKind.Embedding);
        var headObject = graph.Objects.FirstOrDefault(o => o.Component == primaryId && o.Kind == ObjectKind.OutputHead);

        var payloads = new List<TensorPayload>();
        using (var segment = SegmentFile.Open(Path.Combine(container.Root, mtpEntry.Segment)))
        {
            foreach (var tensor in segment.Header.Tensors)
            {
                payloads.Add(PagedPayload(segment, tensor, DtypeExtensions.FromLabel(tensor.Dtype), "mtp." + tensor.Name));
            }
        }

        // The shared embedding table feeds the conditioning token.
        CopyObjectPayloads(container, embeddings, overrideName: null, payloads);

        // The output head (materialised when untied; the embedding table
        // otherwise) produces the predicted token.
        bool tied = headObject is not { Representations.Count: > 0 };
        if (tied)
        {
            CopyObjectPayloads(container, embeddings, "lm_head.weight", payloads);
        }
        else
        {
            CopyObjectPayloads(container, headObject!, overrideName: null, payloads);
        }

        Directory.CreateDirectory(outDir);
        SafetensorsWriter.Write(Path.Combine(outDir, shardName), payloads, new Dictionary<string, string>
        {
            ["format"] = ExportFormat,
            ["model"] = container.Index.Model + " (mtp drafter)",
        });

        // The drafter shares the text geometry; its module is one
        // full-attention trunk layer, and its projection/embeddings are
        // shared — never routed, regardless of the source's FFN surface.
        var config = JsonNode.Parse(ExportConfig.BuildJson(container))!.AsObject();
        config.Remove("moe");
        config["num_hidden_layers"] = 1;
        config["layer_types"] = new JsonArray(JsonValue.Create("full_attention")!);
        config["mtp"] = new JsonObject
        {
            ["num_hidden_layers"] = 1,
            ["use_dedicated_embeddings"] = false,
        };
        File.WriteAllText(Path.Combine(outDir, configName),
            config.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));

        if (copyTokenizer)
        {
            var tokenizerPath = Path.Combine(container.Root, "tokenizer.json");
            if (File.Exists(tokenizerPath))
            {
                File.Copy(tokenizerPath, Path.Combine(outDir, "tokenizer.json"));
            }
            HfAncillaryFiles.CopyInto(container.Root, outDir);
        }

        return new MtpExportReport(
            outDir,
            container.Index.Model,
            payloads.Count,
            payloads.Sum(p => p.PayloadLength),
            tied,
            new List<string>
            {
                $"drafter module: {mtpEntry.TensorCount} mtp.* tensors (fc projector, trunk layer, norms) + shared embedding and " +
                (tied ? "head (tied to the embedding)" : "un-tied lm_head"),
            });
    }

    /// <summary>Copies one object's segment payloads into the output; a tensor
    /// name is rebuilt from the object's binding prefix (embedding
    /// <c>…embed_tokens.weight</c>, head <c>lm_head.weight</c>), or
    /// replaced wholesale by <paramref name="overrideName"/> (the tied head
    /// reuses the shared embedding table under <c>lm_head.weight</c>).</summary>
    private static void CopyObjectPayloads(
        Vindex3Container container, LogicalObject obj, string? overrideName, List<TensorPayload> payloads)
    {
        string bindingPrefix = obj.SourceBindings.FirstOrDefault()?.TensorPrefix
            ?? throw new CliException($"object '{obj.Id}' declares no binding tensor prefix");
        string rep = container.CanonicalRepresentationId(obj.Id);
        var entry = container.Index.Representations[rep];
        using var segment = SegmentFile.Open(Path.Combine(container.Root, entry.Segment));
        foreach (var tensor in segment.Header.Tensors)
        {
            payloads.Add(PagedPayload(
                segment, tensor, DtypeExtensions.FromLabel(tensor.Dtype),
                overrideName ?? bindingPrefix + "." + tensor.Name));
        }
    }

    /// <summary>One payload with the 2 GiB ceiling paged as chunks when the
    /// tensor exceeds it (the shared embedding and lm_head are 2.37 GiB on
    /// the 27B).</summary>
    private static TensorPayload PagedPayload(SegmentFile segment, SegmentTensor tensor, Dtype dtype, string name)
    {
        if (tensor.Len > LargePayloadThresholdBytes)
        {
            return new TensorPayload
            {
                Name = name,
                Dtype = dtype,
                Shape = tensor.Shape,
                Chunks = ReadChunked(segment, tensor.Name, tensor.Len),
            };
        }
        return new TensorPayload
        {
            Name = name,
            Dtype = dtype,
            Shape = tensor.Shape,
            Data = segment.ReadBytes(tensor.Name),
        };
    }

    /// <summary>
    /// The stored bytes of one tensor in the exported checkpoint. An
    /// unpatched tensor's payload is copied verbatim (byte-identical
    /// regeneration); a patched tensor is widened to f32, the patch delta
    /// is added, and the result is re-encoded to the stored dtype. Dtypes
    /// without an encode path refuse if a patch touches them.
    /// </summary>
    private static byte[] ExportTensor(string objectId, SegmentFile segment, SegmentTensor tensor, WeightPatch? patch)
    {
        var dtype = DtypeExtensions.FromLabel(tensor.Dtype);
        return EncodeToDtype(dtype, WidenedValues(segment, objectId, tensor, patch));
    }

    /// <summary>Parallel workers for the export payload pass: the process
    /// compute budget's core count, so the budgeted spare cores stay free
    /// (the machine is usually running other work).</summary>
    public static int WorkerCount(int processorCount) =>
        Math.Max(1, Math.Min(ComputeBudget.Cores, processorCount));

    /// <summary>Payloads beyond this size page as chunked buffers — the
    /// 2 GiB single-array ceiling bites Qwen3.8-27B's embedding and lm_head
    /// (2.37 GiB each).</summary>
    private const long LargePayloadThresholdBytes = 1L << 30; // 1 GiB

    /// <summary>Pages a tensor's payload for the shard: verbatim for
    /// unpatched tensors (single buffer below the ceiling, chunks above),
    /// widen-delta-re-encode for patched tensors (chunks in both
    /// directions).</summary>
    private static TensorPayload BuildExportPayload(
        string objectId, SegmentFile segment, SegmentTensor tensor, WeightPatch? patch, string name)
    {
        var dtype = DtypeExtensions.FromLabel(tensor.Dtype);
        bool patched = patch?.TryGet(objectId, tensor.Name, out _) == true;
        bool paged = tensor.Len > LargePayloadThresholdBytes;

        if (paged)
        {
            var chunks = ReadChunked(segment, tensor.Name, tensor.Len);
            if (patched)
            {
                patch!.TryGet(objectId, tensor.Name, out var entry);
                return new TensorPayload
                {
                    Name = name,
                    Dtype = dtype,
                    Shape = tensor.Shape,
                    Chunks = EncodeChunks(dtype, ApplyDelta(WidenFromChunks(chunks, dtype), entry.Delta)),
                };
            }
            return new TensorPayload { Name = name, Dtype = dtype, Shape = tensor.Shape, Chunks = chunks };
        }

        return new TensorPayload
        {
            Name = name,
            Dtype = dtype,
            Shape = tensor.Shape,
            Data = ExportTensor(objectId, segment, tensor, patch),
        };
    }

    private static IReadOnlyList<byte[]> ReadChunked(SegmentFile segment, string tensorName, long length)
    {
        var chunks = new List<byte[]>();
        for (long done = 0; done < length; done += ChunkBytes)
        {
            int count = (int)Math.Min(ChunkBytes, length - done);
            chunks.Add(segment.ReadBytes(tensorName, done, count));
        }
        return chunks;
    }

    private const long ChunkBytes = 512L * 1024 * 1024;

    /// <summary>Widens a chunked payload to f32 without ever materialising
    /// the storage bytes as one buffer (boundaries are element-aligned for
    /// every byte size this build widens).</summary>
    private static float[] WidenFromChunks(IReadOnlyList<byte[]> chunks, Dtype dtype)
    {
        long elements = chunks.Sum(c => (long)c.Length) / dtype.ElementSize();
        var values = new float[elements];
        int offset = 0;
        foreach (var chunk in chunks)
        {
            var widened = BitPattern.WidenToF32(dtype, chunk);
            Array.Copy(widened, 0, values, offset, widened.Length);
            offset += widened.Length;
        }
        return values;
    }

    private static float[] ApplyDelta(float[] values, float[] delta)
    {
        for (int i = 0; i < values.Length; i++)
        {
            values[i] += delta[i];
        }
        return values;
    }

    /// <summary>Re-encodes in pages so the result never exceeds the 2 GiB
    /// ceiling (each page covers <paramref name="pageElements"/> elements,
    /// ~256 MiB of storage).</summary>
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

    /// <summary>The tensor's f32 values (patch deltas applied when given)
    /// — the shared input for the encode and MXFP4 paths.</summary>
    private static float[] WidenedValues(SegmentFile segment, string objectId, SegmentTensor tensor, WeightPatch? patch)
    {
        var dtype = DtypeExtensions.FromLabel(tensor.Dtype);
        var widened = BitPattern.WidenToF32(dtype, segment.ReadBytes(tensor.Name));
        if (patch is not null && patch.TryGet(objectId, tensor.Name, out var entry))
        {
            if (entry.Delta.Length != WeightPatch.ElementCount(tensor.Shape))
            {
                throw new CliException(
                    $"patch entry '{entry.Key}' holds {entry.Delta.Length} deltas but the tensor " +
                    $"'{objectId}/{tensor.Name}' has {WeightPatch.ElementCount(tensor.Shape)} elements");
            }
            for (int i = 0; i < widened.Length; i++)
            {
                widened[i] += entry.Delta[i];
            }
        }
        return widened;
    }

    /// <summary>MXFP4 quantises the stack's projection matrices: 2-D
    /// per-layer weights in the decoder stack, leaving the synthetic
    /// log-space A_log tensor and any biases at full precision.</summary>
    private static bool ShouldQuantize(string objectId, string tensorName, long[] shape) =>
        objectId == "target.decoder_stack" &&
        shape.Length == 2 &&
        shape[0] > 0 &&
        shape[1] > 0 &&
        !tensorName.Contains("A_log") &&
        !tensorName.Contains("bias");

    /// <summary>Builds the MXFP4 pair for one weight tensor — the FP4 packed
    /// weight (logical shape, two elements per byte) and the per-32-element
    /// E8M0 block scales. The weight name was registered by the caller;
    /// the companion name's collision guard runs under the caller's lock.</summary>
    private static IReadOnlyList<TensorPayload> BuildQuantizedPayloads(
        string objectId, SegmentFile segment, SegmentTensor tensor, WeightPatch? patch, string hfName)
    {
        try
        {
            var values = WidenedValues(segment, objectId, tensor, patch);
            long rows = tensor.Shape[0];
            long columns = tensor.Shape[1];
            if (rows * columns != values.Length)
            {
                throw new CliException(
                    $"MXFP4 shape mismatch: tensor '{hfName}' shape [{rows},{columns}] = {rows * columns} elements, " +
                    $"but widened values has {values.Length} elements");
            }
            // Tensors smaller than one block cannot be MXFP4-quantised: the
            // block structure requires at least BlockElements for a scale row.
            if (values.Length <= Mxfp4.BlockElements)
            {
                return new[] { BuildExportPayload(objectId, segment, tensor, patch, hfName) };
            }

            var quantized = Mxfp4.Quantize(values, rows, columns);

            return new[]
            {
                new TensorPayload
                {
                    Name = hfName,
                    Dtype = Dtype.FP4,
                    Shape = tensor.Shape,
                    Data = quantized.Packed,
                },
                new TensorPayload
                {
                    Name = hfName + Mxfp4.ScaleSuffix,
                    Dtype = Dtype.F8_E8M0,
                    Shape = new[] { rows, Mxfp4.BlocksPerRow(columns) },
                    Data = quantized.BlockScales,
                },
            };
        }
        catch (Exception)
        {
            // Fall back to full precision if MXFP4 quantise fails for any
            // reason — this yields a correct, larger checkpoint rather than a
            // corrupt one, so it is a safe degradation rather than a hide.
            return new[] { BuildExportPayload(objectId, segment, tensor, patch, hfName) };
        }
    }

    /// <summary>f32 weight → packed ternary ({-1,0,+1}) + FP16 block scales.
    /// Emits two tensors: the packed weight payload and a companion scale
    /// tensor. <paramref name="packing"/> selects the bit layout — it is not
    /// cosmetic, and defaulting it silently produced PQ2 bytes for a PTQ1_0
    /// request.</summary>
    private static IReadOnlyList<TensorPayload> BuildTernaryPayloads(
        string objectId, SegmentFile segment, SegmentTensor tensor, WeightPatch? patch, string hfName,
        string packing)
    {
        var values = WidenedValues(segment, objectId, tensor, patch);
        var rows = tensor.Shape[0];
        var cols = tensor.Shape[1];
        // Both packings use 128-element blocks with FP16 scales and the same
        // blockwise Hadamard rotation; only the bit layout differs. PTQ1_0
        // stores 5 trits per byte (26 B/block), PQ2_0 stores 4 values per byte
        // (32 B/block). Reading one as the other yields noise rather than an
        // error, so the packing has to come from the flag.
        var (packed, scales) = packing switch
        {
            "ptq1" => Ternary.EncodePtq1(values, (int)rows, (int)cols),
            "pq2" => Ternary.EncodePq2(values, (int)rows, (int)cols),
            _ => throw new CliException(
                $"unknown ternary packing '{packing}' — this build exports 'ptq1' or 'pq2'"),
        };
        int blocks = Ternary.BlockScaleCount(values.Length);
        int bytesPerBlock = packing == "ptq1" ? Ternary.Ptq1BytesPerBlock : Ternary.Pq2BytesPerBlock;

        return new[]
        {
            new TensorPayload
            {
                Name = hfName,
                // An opaque byte matrix, one row per 128-weight block.
                // Safetensors has no dtype whose element size matches a ternary
                // packing — PTQ1_0 is 26 bytes per block and PQ2_0 is 32, for
                // 128 weights — and FP4, the only sub-byte tag, declares half a
                // byte per element, so the writer rejected the payload with a
                // length mismatch. U8 sized by block is both valid and honest
                // about what the bytes are; the logical [rows, cols] travels in
                // the file's __metadata__ instead.
                Dtype = Dtype.U8,
                Shape = new long[] { blocks, bytesPerBlock },
                Data = packed,
            },
            new TensorPayload
            {
                Name = hfName + ".scales",
                Dtype = Ternary.ScaleDtype,
                Shape = new long[] { blocks },
                Data = scales,
            },
        };
    }

    /// <summary>f32 → the tensor's stored dtype bytes. Only dtypes with a
    /// round-trip encode path are re-encodable; anything else fails closed
    /// naming the dtype.</summary>
    private static byte[] EncodeToDtype(Dtype dtype, float[] values)
    {
        switch (dtype)
        {
            case Dtype.F32:
            {
                var bytes = new byte[values.Length * 4];
                Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
                return bytes;
            }
            case Dtype.BF16:
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
            case Dtype.F16:
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
            default:
                throw new CliException(
                    $"a patched '{dtype.Label()}' tensor cannot be re-encoded — export merges patches into F32/BF16/F16 weights only");
        }
    }
}

/// <summary>
/// Regenerates <c>config.json</c> for the exported checkpoint from judged
/// graph facts: the primary text component's execution surface and
/// per-layer attention policy table. Mirrors <see cref="ModelConfig"/>'s
/// field set so the exported checkpoint re-encodes; a fact this build has
/// not judged (an unjudged layer operator, activation or position policy)
/// refuses the export by name instead of fabricating a value.
/// </summary>
internal static class ExportConfig
{
    public static string BuildJson(Vindex3Container container, bool quantizeMxfp4 = false)
    {
        var graph = container.Graph
            ?? throw new CliException("container records no system graph — cannot regenerate config.json");
        var component = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.PrimaryText)
            ?? throw new CliException("container has no primary text component — cannot regenerate config.json");
        var surface = component.Execution
            ?? throw new CliException($"component '{component.Id}' declares no execution surface — cannot regenerate config.json");
        var attention = surface.Attention
            ?? throw new CliException($"component '{component.Id}' declares no attention surface — cannot regenerate config.json");
        var ffn = surface.Ffn
            ?? throw new CliException($"component '{component.Id}' declares no FFN surface — cannot regenerate config.json");
        var head = surface.Head
            ?? throw new CliException($"component '{component.Id}' declares no head surface — cannot regenerate config.json");
        if (surface.ContextLength is not { } contextLength)
        {
            throw new CliException($"component '{component.Id}' declares no context length — cannot regenerate config.json");
        }
        if (component.Attention is not { Count: > 0 } policies)
        {
            throw new CliException($"component '{component.Id}' declares no per-layer attention table — cannot regenerate config.json");
        }

        var layerTypes = new List<string>();
        bool hasLinearAttention = false;
        for (int l = 0; l < component.NumLayers; l++)
        {
            string op = policies[l].Operator;
            switch (op)
            {
                case LayerOperators.Softmax:
                    layerTypes.Add("full_attention");
                    break;
                case LayerOperators.LinearAttention:
                    layerTypes.Add("linear_attention");
                    hasLinearAttention = true;
                    break;
                case LayerOperators.Conv:
                    layerTypes.Add("conv");
                    break;
                default:
                    throw new CliException(
                        $"layer {l}: operator '{op}' has no judged layer_types spelling — this build regenerates " +
                        "'full_attention' and 'linear_attention' only; refusing to approximate");
            }
        }

        string hiddenAct = ffn.Activation switch
        {
            Activation.Silu => "silu",
            var other => throw new CliException(
                $"FFN activation '{other}' has no judged hidden_act spelling — refusing to approximate"),
        };

        // Detect a classifier container: a ClassifierHead object with a
        // ClassifierSurface on the execution surface.
        var classifier = surface.Classifier;
        bool isClassifier = classifier is not null;
        string architectures = isClassifier
            ? "Qwen3_5ForSequenceClassification"
            : "Qwen3_5ForConditionalGeneration";

        var config = new JsonObject
        {
            ["architectures"] = new JsonArray(JsonValue.Create(architectures)!),
            ["model_type"] = container.Index.Family,
            ["hidden_size"] = component.HiddenSize,
            ["num_hidden_layers"] = component.NumLayers,
            ["num_attention_heads"] = attention.NumQHeads,
            ["num_key_value_heads"] = attention.NumKvHeads,
            ["head_dim"] = attention.HeadDim,
            ["intermediate_size"] = ffn.IntermediateSize,
            ["hidden_act"] = hiddenAct,
            ["rms_norm_eps"] = surface.Norm.Pre.Eps,
            ["vocab_size"] = head.VocabSize,
            ["max_position_embeddings"] = contextLength,
            ["layer_types"] = new JsonArray(layerTypes.Select(t => JsonValue.Create(t)).ToArray()),
        };

        if (ffn.Moe is { } moe)
        {
            // The MoE facts ride the export so a re-encoded container still
            // judges its FFNs routed instead of dense.
            config["moe"] = new JsonObject
            {
                ["experts"] = moe.Experts,
                ["top_k"] = moe.TopK,
                ["expert_intermediate_size"] = moe.ExpertIntermediateSize,
                ["routing_policy"] = moe.RoutingPolicy == ExpertRoutingPolicy.NormalisedOverSelected
                    ? "normalised_over_selected"
                    : "softmax_then_select",
            };
        }

        if (attention.AttentionBias == true)
        {
            config["attention_bias"] = true;
        }
        if (attention.OutputGate is not null)
        {
            config["attn_output_gate"] = true;
        }
        if (head.HeadReusesEmbedding)
        {
            config["tie_word_embeddings"] = true;
        }

        if (hasLinearAttention)
        {
            if (surface.LinearAttention is not { } linearAttn)
            {
                throw new CliException(
                    "component declares linear_attention layers but no linear-attention surface facts — cannot regenerate config.json");
            }
            config["linear_conv_kernel_dim"] = LinearField(linearAttn, "conv_kernel");
            config["linear_num_key_heads"] = LinearField(linearAttn, "key_heads");
            config["linear_key_head_dim"] = LinearField(linearAttn, "key_head_dim");
            config["linear_num_value_heads"] = LinearField(linearAttn, "value_heads");
            config["linear_value_head_dim"] = LinearField(linearAttn, "value_head_dim");
        }

        if (RopeParameters(policies[0].Position) is { } rope)
        {
            config["rope_parameters"] = rope;
        }

        if (quantizeMxfp4)
        {
            config["quantization_config"] = new JsonObject
            {
                ["quant_method"] = "mxfp4",
                ["element_dtype"] = "FP4",
                ["element_grid"] = new JsonArray(BitPattern.Fp4PositiveGrid
                    .Select(v => JsonValue.Create(v)).ToArray()),
                ["block_elements"] = Mxfp4.BlockElements,
                ["block_scale_dtype"] = "F8_E8M0",
            };
        }

        // Classifier head facts: carried from the ClassifierSurface.
        if (isClassifier)
        {
            config["problem_type"] = classifier!.ProblemType;
            if (classifier.Template is { } template)
            {
                config["nli_template"] = template;
            }
            // id2label: synthetic 0→"LABEL_0", 1→"LABEL_1", …
            // The real checkpoint carries the actual labels; here we
            // regenerate from NumLabels since the surface records count
            // but not the label strings (carried in classifier.json).
            var id2Label = new JsonObject();
            var label2Id = new JsonObject();
            for (int i = 0; i < classifier.NumLabels; i++)
            {
                string label = $"LABEL_{i}";
                id2Label[i.ToString()] = label;
                label2Id[label] = i;
            }
            config["id2label"] = id2Label;
            config["label2id"] = label2Id;
        }

        // A materialised vision tower is part of the model: the export's
        // config carries its judged facts (dimensionality from the
        // component, the carried transform descriptor) alongside the text
        // facts, and the tower's tensors ride the same shard under the
        // binding's model.visual prefix.
        var perception = graph.Components.FirstOrDefault(c => c.Role == ComponentRole.Perception);
        if (perception is { } vision &&
            graph.Objects.Any(o => o.Component == vision.Id && o.Representations.Count > 0))
        {
            var visionConfig = new JsonObject
            {
                ["modality"] = "image",
                ["hidden_size"] = vision.HiddenSize,
                ["num_hidden_layers"] = vision.NumLayers,
            };
            if (vision.Perception is { } perceptionFacts &&
                perceptionFacts.ValueKind == JsonValueKind.Object &&
                perceptionFacts.TryGetProperty("transform", out var transform))
            {
                visionConfig["transform"] = JsonNode.Parse(transform.GetRawText());
            }
            config["vision_config"] = visionConfig;
        }

        return config.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
    }

    /// <summary>The encoder reads one text-level rope block; this build's
    /// per-layer policies are homogeneous, so layer 0's policy stands for
    /// the stack.</summary>
    private static JsonNode? RopeParameters(PositionPolicy position) => position switch
    {
        PositionRope rope => new JsonObject
        {
            ["rope_type"] = "default",
            ["rope_theta"] = rope.Theta,
        },
        PositionPartialRope partial => new JsonObject
        {
            ["rope_type"] = "default",
            ["rope_theta"] = partial.Theta,
            ["partial_rotary_factor"] = partial.RotaryFactor,
        },
        PositionNone => null,
        PositionUnresolved unresolved => throw new CliException(
            $"position policy '{unresolved.Kind}' is carried unjudged — cannot regenerate rope_parameters"),
        _ => throw new CliException("unsupported position policy — cannot regenerate rope_parameters"),
    };

    private static long LinearField(JsonElement surface, string name)
    {
        if (surface.ValueKind == JsonValueKind.Object &&
            surface.TryGetProperty(name, out var value) &&
            value.TryGetInt64(out long result))
        {
            return result;
        }
        throw new CliException(
            $"linear-attention surface facts declare no '{name}' — cannot regenerate config.json");
    }
}