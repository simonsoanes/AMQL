using System.Text.Json;
using Amql.Gguf;
using Amql.Hf;
using Amql.Inference;
using Amql.Merge;
using Amql.Onnx;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Cli;

/// <summary>
/// amql-cli — the G0→G3 loader front-end: turn a raw HF checkpoint
/// (Qwen3.5 and similar text stacks) into a canonical VINDEX3 container,
/// then verify the container's byte equivalence from disk alone.
///
///   amql-cli encode &lt;model-dir&gt; --out &lt;container-dir&gt;
///   amql-cli verify &lt;container-dir&gt;
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            PrintHelp();
            return args.Length == 0 ? ExitUsage : ExitOk;
        }

        // Parse global --cpu / --gpu before any command runs so the probe
        // honours the forced mode on the very first CudaShim.Enabled read.
        // --progress turns on the machine-readable protocol (also via
        // AMQL_PROGRESS=1); --verbose restores full stack traces on errors.
        bool forceCpu = false;
        bool forceGpu = false;
        bool progress = Environment.GetEnvironmentVariable("AMQL_PROGRESS") is "1" or "true";
        bool verbose = false;
        var filtered = new List<string>();
        foreach (var a in args)
        {
            if (a == "--cpu") { forceCpu = true; continue; }
            if (a == "--gpu") { forceGpu = true; continue; }
            if (a == "--progress") { progress = true; continue; }
            if (a == "--verbose") { verbose = true; continue; }
            filtered.Add(a);
        }
        if (forceCpu && forceGpu)
        {
            Console.Error.WriteLine("error: --cpu and --gpu are mutually exclusive");
            return ExitUsage;
        }
        if (forceCpu)
        {
            CudaShim.ForceDisable();
            MergeGpu.ForceDisable();
        }
        else if (forceGpu)
        {
            CudaShim.ForceEnable();
            MergeGpu.ForceEnable();
        }
        // When neither flag is given, CudaShim auto-detects (the default).

        args = filtered.ToArray();
        CliProgress.Configure(args[0], progress, verbose);

        int result;
        try
        {
            result = args[0] switch
            {
                "encode" => Encode(args[1..]),
                "verify" => Verify(args[1..]),
                "synth-model" => SynthModel(args[1..]),
                "tokens" => Tokens(args[1..]),
                "decode" => Decode(args[1..]),
                "route" => Route(args[1..]),
                "path" => PathCmd(args[1..]),
                "generate" => Generate(args[1..]),
                "inspect-token" => InspectToken(args[1..]),
                "change-tensor" => ChangeTensor(args[1..]),
                "edit-tensor" => EditTensor(args[1..]),
                "save-lora" => SaveLora(args[1..]),
                "export" => Export(args[1..]),
                "to-gguf" => ToGguf(args[1..]),
                "export-mtp" => ExportMtp(args[1..]),
                "generate-mtp" => GenerateMtp(args[1..]),
                "collect-mtp" => CollectMtp(args[1..]),
                "fit-mtp" => FitMtp(args[1..]),
                "layers" => Layers(args[1..]),
                "import" => Import(args[1..]),
                "moe-ify" => MoeIfy(args[1..]),
                "prune" => Prune(args[1..]),
                "fine-tune" => FineTune(args[1..]),
                "classify" => Classify(args[1..]),
                "convert-to-classifier" => ConvertToClassifier(args[1..]),
                "convert-to-embedding" => ConvertToEmbedding(args[1..]),
                "export-onnx" => ExportOnnx(args[1..]),
                "create-model" => CreateModel(args[1..]),
                _ => throw new CliException($"unknown command '{args[0]}'"),
            };
        }
        catch (CliException e)
        {
            Console.Error.WriteLine($"error: {e.Message}");
            Console.Error.WriteLine("run 'amql-cli help' for usage");
            result = ExitUsage;
        }
        catch (MergeException e)
        {
            Console.Error.WriteLine($"error: {e.Message}");
            result = ExitUsage;
        }
        catch (Exception e)
        {
            // A clean single-line error by default; the stack trace is a
            // debugging aid, so it is reserved for --verbose (TODO.md item 2).
            Console.Error.WriteLine($"error: {e.Message}");
            if (CliProgress.Verbose)
            {
                Console.Error.WriteLine(e.ToString());
            }
            else
            {
                Console.Error.WriteLine("re-run with --verbose for the full stack trace");
            }
            result = ExitUsage;
        }
        CliProgress.EndFragmentLine();
        return result;
    }

    /// <summary>Documented exit codes (TODO.md item 6):</summary>
    /// <remarks>
    ///   0 — success (the command produced its answer, whatever it was);
    ///   1 — a legitimate negative result: the command ran fine and the
    ///       answer is 'not found' / 'not satisfied' (path found no chain
    ///       within budget, verify's integrity check failed). Emit a
    ///       ##amql-result line so front-ends can read the reason;
    ///   2 — usage or runtime error (bad arguments, unreadable input, an
    ///       unsupported operator, an exception).
    /// </remarks>
    private const int ExitOk = 0;
    private const int ExitNegativeResult = 1;
    private const int ExitUsage = 2;


    // ── encode ──────────────────────────────────────────────────────────────

    private static int Encode(string[] args)
    {
        var modelDir = Arg(args, 0) ?? throw new CliException("encode requires a model directory");
        string? outDir = OptionValue(args, "--out");
        if (outDir is null)
        {
            throw new CliException("encode requires '--out <container-dir>'");
        }

        Console.WriteLine($"encoding '{modelDir}' → '{outDir}'");
        CliProgress.Phase("encode");
        var report = ModelToContainer.Encode(modelDir, outDir);
        CliProgress.Complete($"{report.Tensors} tensors, {report.PayloadBytes} bytes");

        Console.WriteLine();
        Console.WriteLine($"model:        {report.ModelId}");
        Console.WriteLine($"encoding:     {report.Encoding}");
        Console.WriteLine($"tensors:      {report.Tensors}");
        Console.WriteLine($"payload:      {FormatBytes(report.PayloadBytes)}");
        Console.WriteLine($"tokenizer:    {(report.TokenizerCopied ? "tokenizer.json copied" : "not found")}");
        if (report.AncillaryCopied.Count > 0)
        {
            Console.WriteLine($"ancillary:    {string.Join(", ", report.AncillaryCopied)}");
        }
        foreach (var (repId, write) in report.Segments.OrderBy(s => s.Key))
        {
            Console.WriteLine($"  {repId}: {write.PayloadBytes} bytes  payload_sha={Short(write.PayloadSha256Hex)}  segment_sha={Short(write.SegmentSha256Hex)}");
        }
        Console.WriteLine("done. run 'amql-cli verify <container-dir>' for integrity + runtime readiness.");
        return 0;
    }

    // ── verify ─────────────────────────────────────────────────────────────

    private static int Verify(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("verify requires a container directory");

        using var container = Vindex3Container.Open(containerDir);
        var index = container.Index;

        Console.WriteLine($"container:  {containerDir}");
        Console.WriteLine($"index:      schema {index.Version} authority {index.Authority} model '{index.Model}' family '{index.Family}'");
        Console.WriteLine($"graph:      schema {container.Graph?.Schema} components={container.Graph?.Components.Count} objects={container.Graph?.Objects.Count}");

        // Byte equivalence: recompute all hashes from disk alone.
        var report = container.VerifyIntegrity();
        Console.WriteLine("integrity:");
        foreach (var check in report.Checks)
        {
            Console.WriteLine($"  [{(check.Ok ? "ok" : "FAIL")}] {check.Representation}{(check.Detail is null ? string.Empty : $" — {check.Detail}")}");
        }
        if (!report.Ok)
        {
            Console.Error.WriteLine("integrity verification FAILED — the container diverges from its index");
            CliProgress.Result(new { integrity = false, failed = report.Checks.Count(c => !c.Ok), checks = report.Checks.Count });
            return ExitNegativeResult; // a real answer: the container is diverged
        }

        // Operand resolution on real weights: tensor tables, shapes, stored
        // dtypes and payload widening all exercised from the container alone.
        using var store = container.CreateOperandStore();
        Console.WriteLine("operands:");
        // The probes are shape-specific (a Qwen3.5-style stack with at least
        // four layers and linear attention). Containers that legitimately do
        // not have those tensors — the 2-layer synth-model, a merged or pruned
        // stack — skip the probes instead of failing verification (TODO.md item 4).
        int probed = 0;
        void Probe(string objectId, string tensorName, string? note = null)
        {
            try
            {
                var operand = store.Resolve(objectId, tensorName);
                Console.WriteLine($"  {objectId}/{tensorName}: {operand.Dtype.Label()} shape=[{string.Join("x", operand.Shape)}]{(note is null ? string.Empty : $" ({note})")}");
                probed++;
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                Console.WriteLine($"  {objectId}/{tensorName}: [skipped — {e.Message}]");
            }
        }
        Probe("target.decoder_stack", "3.self_attn.q_proj.weight");
        Probe("target.decoder_stack", "0.linear_attn.in_proj_qkv.weight");
        Probe("target.decoder_stack", "0.linear_attn.A_log", "precision exception");
        try
        {
            var norms = store.Resolve("target.final_norm", "weight");
            var widened = Amql.Safetensors.BitPattern.WidenToF32(norms.Dtype, norms.Payload);
            float firstNorm = widened[0];
            Console.WriteLine($"  final_norm/weight: {widened.Length} f32 values (first={firstNorm:F4}, dtype {norms.Dtype.Label()})");
            probed++;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Console.WriteLine($"  final_norm/weight: [skipped — {e.Message}]");
        }
        Console.WriteLine(probed == 0
            ? "operand store: no shape-specific probe matched this container (integrity above is unaffected)"
            : $"operand store: {store.TouchedObjects.Count} objects touched, {store.Loads} loads ({probed} probes)");

        // Runtime readiness: plan the primary text component and report the
        // operator boundary — which persisted primitives this build serves
        // and which it refuses (fail-closed, by name).
        Console.WriteLine("runtime readiness:");
        Console.WriteLine(Census(container));
        try
        {
            var plan = Planner.Plan(container, "target", store);
            Console.WriteLine("  [served] the primary text decoder plans and executes.");
            Console.WriteLine($"  layers: {plan.Layers.Count}, hidden: {plan.HiddenSize}, embedding: {(plan.Embedding is null ? "none" : plan.Embedding.VocabSize.ToString())}, head: {(plan.Output is null ? "none (tied?)" : plan.Output.VocabSize.ToString())}");
        }
        catch (UnsupportedOperatorException e)
        {
            Console.WriteLine($"  [refused] {e.Message}");
        }
        return 0;
    }

    private static string Census(Vindex3Container container)
    {
        var graph = container.Graph;
        if (graph is null)
        {
            return "  (no system graph recorded)";
        }
        var lines = new List<string>();
        foreach (var component in graph.Components)
        {
            lines.Add($"  component '{component.Id}' role={component.Role} layers={component.NumLayers} hidden={component.HiddenSize}");
            if (component.Attention is not null)
            {
                foreach (var group in component.Attention.GroupBy(p => p.Operator).OrderBy(g => g.Key))
                {
                    lines.Add($"    operator '{group.Key}': {group.Count()} layers");
                }
            }
            foreach (var obj in graph.Objects.Where(o => o.Component == component.Id))
            {
                string reps = obj.Representations.Count == 0 ? "carried" : string.Join(",", obj.Representations.Select(r => r.Encoding));
                string bindings = string.Join(", ", obj.SourceBindings.Select(b => $"{b.TensorPrefix}.*({b.Tensors})"));
                lines.Add($"    object '{obj.Id}' kind={obj.Kind} reps=[{reps}] bindings=[{(bindings.Length == 0 ? "-" : bindings)}]");
            }
        }
        return string.Join(Environment.NewLine, lines);
    }

    // ── synth-model ────────────────────────────────────────────────────────

    private static int SynthModel(string[] args)
    {
        var dir = Arg(args, 0) ?? throw new CliException("synth-model requires an output directory");
        Amql.Hf.SyntheticCheckpoint.Write(dir);
        Console.WriteLine($"wrote an executable 2-layer Qwen3.5-shaped demo checkpoint to '{dir}'\n" +
                          "encode it:  amql-cli encode <dir> --out <container>\n" +
                          "then run:   amql-cli generate <container> --tokens 0,1 --steps 8\n" +
                          "            amql-cli inspect-token <container> 2 --tokens 0,1");
        return 0;
    }

    // ── tokens / decode (text ↔ ids via the model's tokenizer) ─────────────

    private static int Tokens(string[] args)
    {
        var modelDir = RequiredModelDir(args);
        string text = FirstPositional(args, "--model-dir", "--tokenizer", "--patch") ?? throw new CliException("tokens requires a text argument (quote it)");
        var tokenizer = Tokenizer(modelDir);
        LoadTokenizerPatch(args);

        var result = tokenizer.Encode(text);
        Console.WriteLine($"text:    {text}");
        Console.WriteLine($"tokens:  {result.Ids.Count}");
        foreach (var piece in result.Pieces)
        {
            Console.WriteLine($"  {piece.Id,6}  {(piece.IsSpecial ? "special " : "        ")}{(piece.Representation ?? "-")}  →  {piece.DecodedText ?? "-"}");
        }
        Console.WriteLine($"decoded: {result.ToDecodedText()}");
        return 0;
    }

    private static int Decode(string[] args)
    {
        var modelDir = RequiredModelDir(args);
        var ids = ParseIntList(FirstPositional(args, "--model-dir", "--tokenizer", "--patch"), fallback: Array.Empty<int>());
        if (ids.Length == 0)
        {
            throw new CliException("decode requires token ids, e.g. 'amql-cli decode --tokenizer <checkpoint-dir> 9419,11'");
        }
        var tokenizer = Tokenizer(modelDir);
        LoadTokenizerPatch(args);
        var text = tokenizer.Decode(ids);
        Console.WriteLine($"ids {string.Join(",", ids)} → \"{text}\"");
        foreach (var id in ids)
        {
            var info = tokenizer.TokenInfo(id);
            Console.WriteLine($"  {id,6}  {(info.IsSpecial ? "special " : "        ")}{info.Representation ?? "-"}");
        }
        return 0;
    }

    /// <summary>tokens/decode are tokenizer-only — they never load weights,
    /// so a patch is accepted (and parsed, so typos surface) but cannot
    /// influence the output.</summary>
    private static void LoadTokenizerPatch(string[] args)
    {
        string? path = OptionValue(args, "--patch");
        if (path is null)
        {
            return;
        }
        var patch = WeightPatch.Load(path);
        Console.WriteLine($"patch: {path} ({patch.Entries.Count} tensor{(patch.Entries.Count == 1 ? string.Empty : "s")}) — " +
                          "the tokenizer path is unaffected by weight patches");
    }

    /// <summary>The checkpoint directory whose tokenizer.json converts text ↔
    /// ids. The prime name is <c>--tokenizer</c> (a checkpoint dir, NOT the
    /// container dir — containers hold weights only); <c>--model-dir</c> is
    /// kept as an alias.</summary>
    private static string? TokenizerDir(string[] args) =>
        OptionValue(args, "--tokenizer") ?? OptionValue(args, "--model-dir");

    /// <summary>Resolves the tokenizer source for a command that already
    /// opened a container: an explicit flag wins, then the container's own
    /// tokenizer.json (copied in at encode time), otherwise a typed
    /// error.</summary>
    private static string ResolveTokenizerDir(string? containerDir, string[] args)
    {
        var dir = TokenizerDir(args);
        if (dir is not null)
        {
            return dir;
        }
        if (containerDir is not null && File.Exists(Path.Combine(containerDir, "tokenizer.json")))
        {
            return containerDir;
        }
        throw new CliException(
            "this command needs text, which requires a tokenizer — pass '--tokenizer <checkpoint-dir>' " +
            "(alias: --model-dir), or use a container that was encoded with a tokenizer.json beside it. " +
            "The positional argument is the VINDEX3 container directory (encode output); it carries the " +
            "tokenizer only when encode found one in the checkpoint.");
    }

    private static string RequiredModelDir(string[] args)
    {
        var modelDir = TokenizerDir(args);
        if (modelDir is null)
        {
            throw new CliException(
                "this command needs text, which requires the checkpoint directory containing the " +
                "model's tokenizer.json — pass '--tokenizer <checkpoint-dir>' (alias: --model-dir). " +
                "The positional argument is the VINDEX3 container directory (encode output); it " +
                "holds weights only, no tokenizer.");
        }
        return modelDir;
    }

    private static HfTokenizer Tokenizer(string modelDir) => HfTokenizer.FromModelDir(modelDir);

    // ── route: relationship probing between two tokens ─────────────────────

    /// <summary>First token id of a word in its in-context form: the leading
    /// space is merged into the token ("ĠFrance"), which is what the model
    /// actually continues with; falls back to the standalone spelling.</summary>
    private static int FirstContinuationId(HfTokenizer tokenizer, string word)
    {
        var spaced = tokenizer.EncodeToIds(" " + word);
        if (spaced.Count > 0)
        {
            return spaced[0];
        }
        return tokenizer.EncodeToIds(word).FirstOrDefault(-1);
    }

    private static int Route(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("route requires a container directory");
        // A and B are positionals 2 and 3 — the container already consumed.
        var rest = args.Skip(1).ToArray();
        string a = FirstPositional(rest, "--tokenizer", "--model-dir", "--top", "--templates",
            "--trace-layer-start", "--trace-layer-end", "--corrupt", "--component", "--patch") ??
            throw new CliException("route requires two tokens, e.g. 'amql-cli route <container> France Paris --tokenizer <checkpoint-dir>'");
        string b = SecondPositional(rest, "--tokenizer", "--model-dir", "--top", "--templates",
            "--trace-layer-start", "--trace-layer-end", "--corrupt", "--component", "--patch") ??
            throw new CliException("route requires two tokens: 'amql-cli route <container> <A> <B>'");
        var modelDir = ResolveTokenizerDir(containerDir, args);

        var options = new RouteOptions(
            Top: IntOption(args, "--top", 5),
            MaxTemplates: IntOption(args, "--templates", 8),
            TraceLayerStart: IntOption(args, "--trace-layer-start", 8),
            TraceLayerEnd: IntOption(args, "--trace-layer-end", 24),
            NoTrace: args.Contains("--no-trace"),
            CorruptToken: OptionValue(args, "--corrupt") ?? "the");
        string component = OptionValue(args, "--component") ?? "target";

        using var container = Vindex3Container.Open(containerDir);
        var patch = LoadPatch(args, container);
        Console.WriteLine($"container: {containerDir} (weights)   tokenizer: {modelDir} (checkpoint)");
        var tokenizer = Tokenizer(modelDir);

        CliProgress.Phase("probe", options.MaxTemplates);
        var (links, notes) = RelationRouter.Route(container, component, tokenizer, a, b, options, CliProgress.FragmentWriter, patch);
        foreach (var note in notes)
        {
            Console.WriteLine($"note: {note}");
        }
        Console.WriteLine();
        foreach (var link in links.Take(options.Top))
        {
            var topCoord = link.Coordinates.FirstOrDefault();
            string coordTag = topCoord is null
                ? string.Empty
                : $" @ {topCoord.Layer},{topCoord.Head},{topCoord.QueryPos},{topCoord.KeyPos}";
            Console.WriteLine($"{a} -> {link.Relation} ({link.Score:0.00}{coordTag}) -> {b}");
            foreach (var c in link.Coordinates.Skip(1).Take(3))
            {
                Console.WriteLine($"     @ L{c.Layer} H{c.Head} ({c.QueryPos}->{c.KeyPos}) {c.Weight:0.00}");
            }
            if (!options.NoTrace && link.Attribution is { } attr)
            {
                Console.WriteLine($"     causal weights (patch targets), P({b}) clean={attr.CleanProbability:0.###} corrupt={attr.CorruptProbability:0.###}:");
                var strong = Enumerable.Range(0, attr.LayerDelta.Length)
                    .Select((l, i) => (Layer: i, Delta: attr.LayerDelta[i]))
                    .Where(x => x.Delta > 0f)
                    .OrderByDescending(x => x.Delta)
                    .Take(8);
                foreach (var (layer, delta) in strong)
                {
                    Console.WriteLine($"       L{layer,2}: Δ {delta:0.0000} ({attr.LayerShare[layer] * 100,4:0.0}% of effect)");
                }
            }
        }
        Console.WriteLine();
        Console.WriteLine("scores = P(B) after template(A); coords = (layer, head, queryPos, keyPos) of the final-row attention onto A;");
        Console.WriteLine("causal Δ = P(B) restored by reinstating that layer's clean residual (corrupt → clean) — the tensors to patch/LoRA.");
        Console.WriteLine($"progress: {links.Count} templates probed{(options.NoTrace ? " (no attribution)" : $", attribution on top link over layers {options.TraceLayerStart}..{options.TraceLayerEnd}")}.");
        return 0;
    }

    // ── path: bidirectional best-first search between two tokens ────────────

    private static int PathCmd(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("path requires a container directory");
        var rest = args.Skip(1).ToArray();
        string a = FirstPositional(rest, "--tokenizer", "--model-dir", "--topk", "--max-nodes", "--max-depth", "--component", "--patch") ??
            throw new CliException("path requires two tokens, e.g. 'amql-cli path <container> France Paris'");
        string b = SecondPositional(rest, "--tokenizer", "--model-dir", "--topk", "--max-nodes", "--max-depth", "--component", "--patch") ??
            throw new CliException("path requires two tokens: 'amql-cli path <container> <A> <B>'");
        var modelDir = ResolveTokenizerDir(containerDir, args);
        var tokenizer = Tokenizer(modelDir);

        // The model continues with the space-merged form ("ĠFrance") —
        // search that spelling, falling back to the standalone token.
        int aId = FirstContinuationId(tokenizer, a);
        int bId = FirstContinuationId(tokenizer, b);
        if (aId < 0 || bId < 0)
        {
            throw new CliException($"cannot tokenize '{a}' or '{b}'");
        }

        var options = new PathSearchOptions(
            TopK: IntOption(args, "--topk", 6),
            MaxNodes: IntOption(args, "--max-nodes", 48),
            MaxDepth: IntOption(args, "--max-depth", 6),
            Debug: args.Contains("--debug"));
        string component = OptionValue(args, "--component") ?? "target";

        using var container = Vindex3Container.Open(containerDir);
        var patch = LoadPatch(args, container);
        bool inContainer = modelDir.Equals(containerDir, StringComparison.OrdinalIgnoreCase);
        Console.WriteLine($"container: {containerDir} (weights)   tokenizer: {modelDir} ({(inContainer ? "in container" : "checkpoint")})");
        Console.WriteLine($"searching from '{a}' (id {aId}) toward '{b}' (id {bId}) — edges = top-{options.TopK} continuations (cost −log P) …");

        CliProgress.Phase("search", options.MaxNodes);
        var result = PathFinder.Search(container, component, tokenizer, aId, bId, options, CliProgress.FragmentWriter, patch);
        Console.WriteLine();

        if (!result.Found)
        {
            Console.WriteLine($"no path found within the budget ({options.MaxNodes} expansions, depth {options.MaxDepth}).");
            // Exit 1 = a legitimate negative result, not an error (TODO.md item 6);
            // the result line lets a front-end read the reason without parsing prose.
            CliProgress.Result(new { found = false, nodes = result.NodesVisited, forwards = result.Forwards, budgetNodes = options.MaxNodes, budgetDepth = options.MaxDepth });
            return ExitNegativeResult;
        }

        foreach (var hop in result.Hops)
        {
            string costTag = hop.EdgeCost <= 0 ? "start" : $"+{hop.EdgeCost:0.00}";
            Console.WriteLine($"  {hop.TokenId,7}  {hop.TokenText,-24} {costTag}");
        }
        Console.WriteLine();
        Console.WriteLine($"meeting point: '{result.Hops[^1].TokenText}' — fwd {result.MeetingForwardCost:0.00}, bwd {result.MeetingBackwardCost:0.00}");
        Console.WriteLine($"total cost {result.TotalCost:0.00} · {result.Forwards} model forwards · {result.NodesVisited} nodes");
        Console.WriteLine("path = token chain only (no relation names); costs are −log P of each continuation edge.");
        CliProgress.Result(new { found = true, cost = Math.Round(result.TotalCost, 4), hops = result.Hops.Count, nodes = result.NodesVisited, forwards = result.Forwards });
        CliProgress.Complete();
        return ExitOk;
    }

    // ── generate ───────────────────────────────────────────────────────────

    private static int Generate(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("generate requires a container directory");
        string? prompt = OptionValue(args, "--prompt");

        // Prompt mode: the tokenizer comes from --tokenizer, or from the
        // container when encode placed a tokenizer.json beside it.
        HfTokenizer? tokenizer = null;
        string? tokenizerSource = null;
        int[] tokens;
        if (prompt is not null)
        {
            tokenizerSource = ResolveTokenizerDir(containerDir, args);
            tokenizer = Tokenizer(tokenizerSource);
            tokens = tokenizer.EncodeToIds(prompt).ToArray();
            if (tokens.Length == 0)
            {
                throw new CliException("the prompt encoded to zero tokens");
            }
        }
        else
        {
            tokens = ParseIntList(OptionValue(args, "--tokens"), fallback: new[] { 0 });
        }

        int steps = IntOption(args, "--steps", 8);
        var config = new Amql.Inference.SamplingConfig(
            Seed: IntOption(args, "--seed", 42),
            Temperature: FloatOption(args, "--temperature", 0f),
            TopK: IntOption(args, "--top-k", 0),
            TopP: FloatOption(args, "--top-p", 0f));
        string component = OptionValue(args, "--component") ?? "target";
        int? showTopK = IntOptionOrNull(args, "--logits");
        bool sampling = config.Temperature > 0f || config.TopK > 0 || config.TopP > 0;
        bool trace = HasOption(args, "--trace");
        bool traceTensors = HasOption(args, "--trace-tensors");
        string weightMode = OptionValue(args, "--weights") ?? string.Empty;
        WeightWorkingSet? workingSet = weightMode switch
        {
            "f32" => WeightWorkingSet.ResidentF32,
            "fp32" => WeightWorkingSet.ResidentF32,
            "bf16" => WeightWorkingSet.OnDemandBf16,
            "mxfp4" => WeightWorkingSet.Mxfp4,
            "fp4" => WeightWorkingSet.Mxfp4,
            "" or null => null, // use env var default
            _ => throw new CliException($"unknown weight mode '{weightMode}' — use f32, bf16, or mxfp4"),
        };

        using var container = Vindex3Container.Open(containerDir);
        var patch = LoadPatch(args, container);
        if (tokenizer is not null)
        {
            bool inContainer = tokenizerSource!.Equals(containerDir, StringComparison.OrdinalIgnoreCase);
            Console.WriteLine($"container: {containerDir} (weights)   tokenizer: {tokenizerSource} ({(inContainer ? "in container" : "checkpoint")})");
        }
        var workingSetEffective = workingSet ?? WeightWorkingSetExtensions.FromEnv();
        if (workingSetEffective == WeightWorkingSet.Mxfp4 && CudaShim.Enabled)
        {
            Console.WriteLine("cuda:      MXFP4 packs resident on device — GEMMs run on the GPU (FP16 tensor cores, FP32 accumulate)");
        }
        else if (workingSetEffective == WeightWorkingSet.Mxfp4)
        {
            Console.WriteLine("weights:   MXFP4 working set (dequantised to f32 on CPU)");
        }
        else if (workingSetEffective == WeightWorkingSet.OnDemandBf16)
        {
            Console.WriteLine("weights:   BF16 on-demand (LRU-bounded, widened on access)");
        }
        else
        {
            Console.WriteLine("weights:   f32 (full-precision resident)");
        }

        string? traceJson = OptionValue(args, "--trace-json");
        bool attribute = HasOption(args, "--attribute");
        int corruptId = -1;
        if (attribute)
        {
            string corruptText = OptionValue(args, "--attribute-corrupt") ?? throw new CliException(
                "--attribute needs --attribute-corrupt <text>: the token that replaces the source "
                + "position in the corrupted runs, e.g. --attribute-corrupt the");
            if (tokenizer is null)
            {
                throw new CliException(
                    "--attribute resolves --attribute-corrupt through the tokenizer, so it also needs "
                    + "--tokenizer <checkpoint-dir>");
            }
            corruptId = FirstContinuationId(tokenizer, corruptText);
            if (corruptId < 0)
            {
                throw new CliException($"--attribute-corrupt '{corruptText}' is not in the vocabulary");
            }
        }
        int attributeSource = IntOption(args, "--attribute-source", -1);
        int attributeLayerEnd = IntOption(args, "--attribute-layers", -1);

        var tensorLoads = traceTensors ? new List<TensorTraceLine>() : null;
        var genOpts = (trace || traceTensors || traceJson is not null || workingSet is not null || attribute)
            ? new InferenceRunner.GenerateOptions(Trace: trace, TraceTensors: traceTensors,
                WeightWorkingSet: workingSet, TraceJsonPath: traceJson,
                // Resolve token text at capture time so a saved trace reads as
                // words when reopened, rather than depending on the tokenizer
                // still being to hand.
                TokenText: tokenizer is not null ? id => tokenizer.Decode(new[] { id }) : null,
                OnTensorLoad: tensorLoads is not null ? line => tensorLoads.Add(line) : null,
                Attribute: attribute,
                AttributeSourceRow: attributeSource,
                AttributeCorruptTokenId: corruptId,
                AttributeLayerEnd: attributeLayerEnd)
            : null;

        var (prefill, steps2) = InferenceRunner.Generate(
            container, component, tokens, steps, config, showTopK, patch, genOpts);

        string prefillText = tokenizer is null ? string.Empty : tokenizer.Decode(prefill);
        string mode = sampling ? "sampled" : "greedy";
        Console.WriteLine($"prefill [{string.Join(",", prefill)}] → position {prefill.Length} during [{mode}]" +
                          (prefillText.Length > 0 ? $"  ({prefillText})" : string.Empty));
        foreach (var outcome in steps2)
        {
            string? text = tokenizer?.TokenInfo(outcome.Token).DecodedText;
            Console.Write($"{outcome.Token}");
            if (text is { Length: > 0 })
            {
                Console.Write($"  ({text})");
            }
            if (outcome.Candidates is { } candidates)
            {
                Console.Write("   " + string.Join("  ",
                    candidates.Select(c => $"{c.Token} {c.Logit:0.####}({c.Probability * 100:0.###}%)")));
            }
            Console.WriteLine();

            if (outcome.Trace is { } traceLines && traceLines.Count > 0)
            {
                Console.WriteLine($"   trace ({traceLines.Count} layers):");
                foreach (var t in traceLines)
                {
                    Console.WriteLine($"     L{t.Layer}: residual |h|={t.ResidualNorm:F2}  Δ={t.DeltaNorm:F4}");
                }
            }
        }

        if (tensorLoads is not null)
        {
            // Grouped rather than listed raw: a decode step reloads every
            // weight it touched on the previous one, so the interesting number
            // is how many distinct tensors were pulled and how often each was
            // cold, not the several thousand individual cache hits.
            var distinct = tensorLoads
                .GroupBy(l => (l.ObjectId, l.TensorName))
                .Select(g => (g.Key.ObjectId, g.Key.TensorName, Shape: g.First().Shape,
                    Loads: g.Count(), Cold: g.Count(l => !l.CacheHit)))
                .OrderByDescending(t => t.Cold)
                .ThenBy(t => t.TensorName, StringComparer.Ordinal)
                .ToList();

            Console.WriteLine();
            Console.WriteLine($"tensor loads: {tensorLoads.Count:N0} total, {distinct.Count} distinct, "
                + $"{tensorLoads.Count(l => !l.CacheHit)} cold");
            const int shown = 40;
            foreach (var t in distinct.Take(shown))
            {
                Console.WriteLine($"  {t.ObjectId}/{t.TensorName} [{string.Join("x", t.Shape)}]"
                    + $"  {t.Loads} load{(t.Loads == 1 ? string.Empty : "s")}, {t.Cold} cold");
            }
            if (distinct.Count > shown)
            {
                Console.WriteLine($"  … and {distinct.Count - shown} more");
            }
        }

        if (tokenizer is not null)
        {
            var generatedIds = prefill.Concat(steps2.Select(s => s.Token)).ToArray();
            Console.WriteLine($"text:      {tokenizer.Decode(generatedIds)}");
        }
        Console.WriteLine($"position: {prefill.Length + steps}");

        // Read the trace back rather than trusting the write: it proves the
        // JSON round-trips before anything downstream depends on the format.
        if (traceJson is not null)
        {
            var written = Amql.Inference.Tracing.TraceRecorder.ReadJson(traceJson);
            Console.WriteLine($"trace:     {written.Steps.Count} steps over {written.Nodes.Count} operator nodes, "
                + $"{written.Steps.Sum(s => s.Ops.Count)} observations → {traceJson}");
            var top = written.Aggregate()
                .OrderByDescending(kv => kv.Value.MeanL2)
                .Take(5)
                .Select(kv =>
                {
                    var node = written.Nodes[kv.Key];
                    return $"{node.Op}@L{node.Layer} (mean ‖·‖ {kv.Value.MeanL2:F3})";
                });
            Console.WriteLine($"           busiest operators: {string.Join(", ", top)}");

            if (written.Causal is { } causal)
            {
                string Label(int id) => tokenizer is not null ? tokenizer.Decode(new[] { id }) : id.ToString();
                Console.WriteLine($"attribution: P({Label(causal.TargetTokenId)}) "
                    + $"{causal.CleanProbability:P2} clean → {causal.CorruptProbability:P2} with "
                    + $"{Label(causal.SourceTokenId)} at row {causal.SourceRow} replaced by "
                    + $"{Label(causal.CorruptTokenId)}");
                Console.WriteLine($"           total effect {causal.TotalEffect:P2}; "
                    + $"{causal.LayerDelta.Count} layers traced");
                var peaks = causal.LayerShare
                    .Select((share, layer) => (share, layer))
                    .OrderByDescending(x => x.share)
                    .Take(5)
                    .Where(x => x.share > 0.001f)
                    .Select(x => $"L{x.layer} {x.share * 100:F1}%");
                Console.WriteLine($"           largest shares: {string.Join(", ", peaks)}");
            }
        }
        return 0;
    }

    // ── edit-tensor: whole-tensor scale / zero / offset ────────────────────

    /// <summary>
    /// The whole-tensor counterpart to <c>change-tensor</c>. A single-cell edit
    /// to a multi-million-element matrix is unmeasurable downstream, so it
    /// cannot answer the question the inference visualiser exists to raise:
    /// if I turn this tensor down, what happens?
    /// </summary>
    private static int EditTensor(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException(
            "edit-tensor requires a container directory, e.g. 'amql-cli edit-tensor <container> target.decoder_stack 3.self_attn.q_proj.weight --scale 0.5 --out patch.safetensors'");
        var pos = Positionals(args.Skip(1).ToArray(), "--out", "--set", "--add", "--scale", "--zero", "--patch");
        var problems = new List<string>();
        string? objectId = pos.Length > 0 ? pos[0] : null;
        string? tensorName = pos.Length > 1 ? pos[1] : null;
        if (objectId is null) problems.Add("missing object id (e.g. target.decoder_stack)");
        if (tensorName is null) problems.Add("missing tensor name (e.g. 3.self_attn.q_proj.weight)");
        if (pos.Length > 2)
        {
            problems.Add($"unexpected argument '{pos[2]}' — edit-tensor takes no cell index; "
                + "use change-tensor to edit a single cell");
        }
        string? outPatch = OptionValue(args, "--out");
        if (outPatch is null) problems.Add("missing '--out <patch.safetensors>'");
        TensorEditOp op = TensorEditOp.Set;
        float value = 0f;
        try
        {
            op = ParseEditOp(args);
            value = ParseEditValue(args, op);
        }
        catch (CliException e)
        {
            problems.Add(e.Message);
        }
        if (problems.Count > 0)
        {
            throw new CliException("edit-tensor is missing or rejects arguments:" + Environment.NewLine +
                string.Join(Environment.NewLine, problems.Select(p => "  - " + p)));
        }

        string obj = objectId!;
        string tensor = tensorName!;
        string patchOut = outPatch!;
        string? existingPath = File.Exists(patchOut) ? patchOut : null;
        var existing = TensorPatchTools.LoadOrEmpty(existingPath);

        using var container = Vindex3Container.Open(containerDir);
        var result = TensorPatchTools.ApplyTensorEdit(container, obj, tensor, op, value, existing);

        string describe = op switch
        {
            TensorEditOp.Scale => $"× {value:0.######}",
            TensorEditOp.Add => $"+ {value:0.######}",
            _ => $"set to {value:0.######}",
        };

        if (result.Removed)
        {
            if (existingPath is not null)
            {
                File.Delete(existingPath);
                Console.WriteLine($"patch {existingPath}: '{obj}/{tensor}' is back at its base values; patch cleared.");
            }
            else
            {
                Console.WriteLine($"'{obj}/{tensor}': {describe} leaves it unchanged; nothing written.");
            }
            return 0;
        }

        WeightPatch.Save(patchOut, result.Entries, container.Index.Model);
        double normChange = result.NormBefore == 0f
            ? 0.0
            : (result.NormAfter - result.NormBefore) / result.NormBefore * 100.0;
        Console.WriteLine($"'{obj}/{tensor}' [{string.Join("x", result.Shape)}] {result.DtypeLabel}  "
            + $"{result.ElementCount:N0} elements, every one edited ({describe})");
        Console.WriteLine($"  mean element: {result.MeanBefore:0.######} → {result.MeanAfter:0.######}");
        Console.WriteLine($"  tensor norm:  {result.NormBefore:0.###} → {result.NormAfter:0.###}  ({normChange:+0.##;-0.##;0}%)");
        Console.WriteLine($"patch: {patchOut} ({result.Entries.Count} tensor{(result.Entries.Count == 1 ? string.Empty : "s")})");
        Console.WriteLine("re-run and diff it against the baseline trace:");
        Console.WriteLine($"  amql-cli generate {containerDir} --prompt \"…\" --patch {patchOut} --trace-json after.json");
        return 0;
    }

    // ── inspect-token ──────────────────────────────────────────────────────

    private static int InspectToken(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("inspect-token requires a container directory");
        int token = IntOption(args, 1, -1);
        if (token < 0)
        {
            throw new CliException("inspect-token requires a token id");
        }
        string component = OptionValue(args, "--component") ?? "target";
        int neighbors = IntOption(args, "--neighbors", TokenInspector.DefaultNeighbors);
        int? logitsK = IntOptionOrNull(args, "--logits");
        int[]? context = ParseOptionalIntList(OptionValue(args, "--tokens"));
        string? modelDir = TokenizerDir(args);
        if (modelDir is null && File.Exists(Path.Combine(containerDir, "tokenizer.json")))
        {
            modelDir = containerDir; // the container carries its own tokenizer
        }
        if (modelDir is not null)
        {
            bool inContainer = modelDir.Equals(containerDir, StringComparison.OrdinalIgnoreCase);
            Console.WriteLine($"container: {containerDir} (weights)   tokenizer: {modelDir} ({(inContainer ? "in container" : "checkpoint")})");
        }

        using var container = Vindex3Container.Open(containerDir);
        var patch = LoadPatch(args, container);
        var profile = TokenInspector.InspectEmbedding(container, component, token, neighbors, patch);

        Console.WriteLine($"token {profile.Token} — vocab {profile.Vocab}, dim {profile.Dim}, stored {profile.StoredDtype}");
        if (modelDir is not null)
        {
            var tokenizer = Tokenizer(modelDir);
            var info = tokenizer.TokenInfo(token);
            Console.WriteLine($"  text:    \"{info.DecodedText ?? "-"}\"{(info.IsSpecial ? " (special)" : string.Empty)}");
            Console.WriteLine($"  repr:    {info.Representation ?? "-"}");
        }
        Console.WriteLine($"  row: [{string.Join(", ", profile.Row.Take(Math.Min(8, profile.Dim)).Select(v => $"{v:0.###}"))}{(profile.Dim > 8 ? ", …" : string.Empty)}]");
        Console.WriteLine($"  min {profile.Min:0.###}  max {profile.Max:0.###}  mean {profile.Mean:0.###}  L2 {profile.Norm:0.###}");
        Console.WriteLine("  nearest neighbours (cosine):");
        foreach (var neighbor in profile.Neighbors)
        {
            Console.WriteLine($"    token {neighbor.Token}: {neighbor.Cosine:0.###}");
        }

        if (context is not null)
        {
            try
            {
                var report = TokenInspector.InspectLogits(container, component, token, context, logitsK ?? 5, patch);
                if (report is not null)
                {
                    Console.WriteLine($"logits after prefill [{string.Join(",", context)}]: token {token} rank {report.Rank} logit {report.Logit:0.####} (p {report.Probability * 100:0.###}%)");
                    foreach (var candidate in report.Top)
                    {
                        Console.WriteLine($"  top: {candidate.Token} {candidate.Logit:0.####}");
                    }
                }
            }
            catch (Amql.Inference.UnsupportedOperatorException e)
            {
                Console.WriteLine($"logits inspection unavailable: {e.Message}");
            }
        }
        return 0;
    }

    // ── change-tensor: manually edit one weight cell into a patch ──────────

    private static int ChangeTensor(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException(
            "change-tensor requires a container directory, e.g. 'amql-cli change-tensor <container> target.embedding weight 3,1 --set 0.5 --out patch.safetensors'");
        var pos = Positionals(args.Skip(1).ToArray(), "--out", "--set", "--add", "--scale", "--zero", "--patch");
        // Collect EVERY problem before throwing, so one run surfaces all of
        // them instead of one fix per invocation (TODO.md item 3).
        var problems = new List<string>();
        string? objectId = pos.Length > 0 ? pos[0] : null;
        string? tensorName = pos.Length > 1 ? pos[1] : null;
        string? cell = pos.Length > 2 ? pos[2] : null;
        if (objectId is null) problems.Add("missing object id (e.g. target.embedding)");
        if (tensorName is null) problems.Add("missing tensor name (e.g. weight, 0.self_attn.q_proj.weight)");
        if (cell is null) problems.Add("missing cell: 'row,col' for a 2-D tensor, a flat index otherwise");
        string? outPatch = OptionValue(args, "--out");
        if (outPatch is null) problems.Add("missing '--out <patch.safetensors>'");
        TensorEditOp op = TensorEditOp.Set;
        float value = 0f;
        try
        {
            op = ParseEditOp(args);
            value = ParseEditValue(args, op);
        }
        catch (CliException e)
        {
            problems.Add(e.Message);
        }
        if (problems.Count > 0)
        {
            throw new CliException("change-tensor is missing or rejects arguments:" + Environment.NewLine +
                string.Join(Environment.NewLine, problems.Select(p => "  - " + p)));
        }
        // Validation above guarantees these are non-null; copy into non-nullable
        // locals so the flow analysis agrees (and the call sites stay clean).
        string obj = objectId!;
        string tensor = tensorName!;
        string cellArg = cell!;
        string patchOut = outPatch!;
        string? existingPath = File.Exists(patchOut) ? patchOut : null;
        var existing = TensorPatchTools.LoadOrEmpty(existingPath);

        using var container = Vindex3Container.Open(containerDir);
        var shape = TensorPatchTools.ResolveShape(container, obj, tensor);
        long flat = ParseCell(cellArg, shape, obj, tensor);

        var result = TensorPatchTools.ApplyEdit(container, obj, tensor, op, value, flat, existing);
        if (result.Removed)
        {
            if (existingPath is not null)
            {
                File.Delete(existingPath);
                Console.WriteLine($"patch {existingPath}: '{obj}/{tensor}'[{cellArg}]\n  {result.Before:0.######} → {result.After:0.######} — back at the base value; patch cleared (no changes remain).");
            }
            else
            {
                Console.WriteLine($"'{obj}/{tensor}'[{cellArg}]: {result.Before:0.######} → {result.After:0.######} — no change, nothing written.");
            }
            return 0;
        }

        WeightPatch.Save(patchOut, result.Entries, container.Index.Model);
        Console.WriteLine($"'{obj}/{tensor}' [{string.Join("x", result.Shape)}] {result.DtypeLabel} [{cellArg}] {result.Before:0.######} → {result.After:0.######} (Δ {result.After - result.Before:0.######})");
        Console.WriteLine($"patch: {patchOut} ({result.Entries.Count} tensor{(result.Entries.Count == 1 ? string.Empty : "s")})");
        Console.WriteLine("run a pathway with it: amql-cli route <container> A B --tokenizer <checkpoint> --patch " + patchOut);
        return 0;
    }

    private static TensorEditOp ParseEditOp(string[] args)
    {
        bool set = args.Contains("--set");
        bool add = args.Contains("--add");
        bool scale = args.Contains("--scale");
        bool zero = args.Contains("--zero");
        if ((set ? 1 : 0) + (add ? 1 : 0) + (scale ? 1 : 0) + (zero ? 1 : 0) != 1)
        {
            throw new CliException("change-tensor requires exactly one of '--set <value>', '--add <value>', '--scale <factor>', '--zero'");
        }
        return zero ? TensorEditOp.Set : (set ? TensorEditOp.Set : (add ? TensorEditOp.Add : TensorEditOp.Scale));
    }

    private static float ParseEditValue(string[] args, TensorEditOp op)
    {
        if (op == TensorEditOp.Set && args.Contains("--zero"))
        {
            return 0f;
        }
        string name = op switch
        {
            TensorEditOp.Set => "--set",
            TensorEditOp.Add => "--add",
            TensorEditOp.Scale => "--scale",
            _ => throw new CliException("unknown edit operation"),
        };
        if (!float.TryParse(OptionValue(args, name), System.Globalization.CultureInfo.InvariantCulture, out float value))
        {
            throw new CliException($"{name} requires a numeric value");
        }
        return value;
    }

    private static long ParseCell(string cell, long[] shape, string objectId, string tensorName)
    {
        if (shape.Length == 2 && cell.Contains(','))
        {
            var parts = cell.Split(',', StringSplitOptions.TrimEntries);
            if (parts.Length != 2 || !long.TryParse(parts[0], out long row) || !long.TryParse(parts[1], out long col))
            {
                throw new CliException($"cell '{cell}' is not 'row,col'");
            }
            if (row < 0 || row >= shape[0] || col < 0 || col >= shape[1])
            {
                throw new CliException($"cell ({row},{col}) is outside '{objectId}/{tensorName}' [{string.Join("x", shape)}]");
            }
            return checked(row * shape[1] + col);
        }
        if (!long.TryParse(cell, out long flat))
        {
            throw new CliException($"cell '{cell}' is not an index");
        }
        return flat; // bounds are re-checked against the tensor in ApplyEdit
    }

    // ── save-lora: factor a patch into a LoRA for the original model ───────

    private static int SaveLora(string[] args)
    {
        string patchPath = Arg(args, 0) ?? throw new CliException("save-lora requires a patch file, e.g. 'amql-cli save-lora patch.safetensors --out lora --rank 8 --alpha 16'");
        string outDir = OptionValue(args, "--out") ?? throw new CliException("save-lora requires '--out <lora-dir>'");
        int rank = IntOption(args, "--rank", 8);
        double alpha = DoubleOption(args, "--alpha", 16);
        string? containerDir = OptionValue(args, "--container");

        if (containerDir is not null)
        {
            using var container = Vindex3Container.Open(containerDir);
            WeightPatch.Load(patchPath).ValidateAgainst(container);
        }

        var report = LoraWriter.SaveAsLora(patchPath, outDir, rank, alpha);
        Console.WriteLine($"LoRA: {report.OutDir}   rank {report.Rank} (scale alpha/r = {report.Alpha}/{report.Rank} = {report.Scale:0.###})   model {report.Model ?? "-"}");
        foreach (var target in report.Targets)
        {
            Console.WriteLine($"  {target.ObjectId}/{target.TensorName} [{string.Join("x", target.Shape)}] → r={target.Rank}  {target.AName} {target.BName}  (reconstruction error {target.ReconstructionError:0.###e+00})");
        }
        foreach (var note in report.Skipped)
        {
            Console.WriteLine($"  skipped: {note}");
        }
        Console.WriteLine("apply to the base container: for each target, add scale · lora_B · lora_A to the tensor.");
        return 0;
    }

    // ── export: materialise an HF checkpoint from the container ──────────

    private static int Export(string[] args)
    {
        if (HasOption(args, "--list-architectures"))
        {
            Console.WriteLine("qwen3.x  (default) — Qwen3.5 / Qwen3-Next hybrid checkpoint format");
            Console.WriteLine($"{Qwen4NextLayout.Arch}  — Qwen3.8-Flash-Next (qwen4_exp) architecture");
            return 0;
        }

        var containerDir = Arg(args, 0) ?? throw new CliException(
            "export requires a container directory, e.g. 'amql-cli export <container-dir> --out <checkpoint-dir>'");
        string outDir = OptionValue(args, "--out") ?? throw new CliException("export requires '--out <checkpoint-dir>'");
        string quant = OptionValue(args, "--quant") ?? "none";
        if (quant != "none" && quant != "mxfp4" && quant != "ptq1" && quant != "pq2")
        {
            throw new CliException($"unknown quantization '{quant}' — this build exports 'none' (full precision), 'mxfp4', 'ptq1' (ternary dense), or 'pq2' (ternary 2-bit)");
        }
        string arch = OptionValue(args, "--arch") ?? "qwen3.x";
        if (arch != "qwen3.x" && arch != Qwen4NextLayout.Arch)
        {
            throw new CliException($"unknown architecture '{arch}' — this build exports 'qwen3.x' (default) or '{Qwen4NextLayout.Arch}'");
        }
        string? archParam = arch == Qwen4NextLayout.Arch ? arch : null;

        using var container = Vindex3Container.Open(containerDir);
        var patch = LoadPatch(args, container);
        var report = ModelExporter.Export(container, outDir, patch,
            quantizeMxfp4: quant == "mxfp4",
            quantizeTernary: quant == "ptq1" ? "ptq1" : quant == "pq2" ? "pq2" : null,
            arch: archParam);

        Console.WriteLine($"exported:  {report.OutDir}");
        Console.WriteLine($"model:      {report.Model}");
        Console.WriteLine($"tensors:    {report.Tensors}  ({FormatBytes(report.PayloadBytes)})");
        foreach (var note in report.Notes)
        {
            Console.WriteLine($"note:       {note}");
        }
        string files = "model.safetensors, config.json" +
                       (File.Exists(Path.Combine(outDir, "tokenizer.json")) ? ", tokenizer.json" : string.Empty);
        string archLabel = archParam is not null ? $", architecture: {archParam}" : string.Empty;
        Console.WriteLine($"wrote:      {files}  (quantization: {quant}{archLabel})");
        if (quant == "mxfp4")
        {
            Console.WriteLine("the MXFP4 checkpoint is a terminal artifact — the encoder reads full-precision dtypes; the Mxfp4 codec is the reference for consumer runtimes");
        }
        else
        {
            Console.WriteLine("the checkpoint is the original model with any patch deltas baked in — encode it to move back into a container:");
            Console.WriteLine($"  amql-cli encode {outDir} --out <new-container>");
        }
        return 0;
    }

    // ── to-gguf: convert an exported HF checkpoint to GGUF ──────────────

    private static int ToGguf(string[] args)
    {
        var checkpointDir = Arg(args, 0) ?? throw new CliException(
            "to-gguf requires a checkpoint directory, e.g. 'amql-cli to-gguf <checkpoint-dir> --out <model.gguf>'");
        string outFile = OptionValue(args, "--out") ?? throw new CliException("to-gguf requires '--out <file.gguf>'");
        string quantization = OptionValue(args, "--quant") ?? "none";
        if (quantization == "f16")
        {
            // 'f16' is the user-facing alias for "do not quantize".
            quantization = "none";
        }
        if (!GgufConverter.Quantizations.Contains(quantization))
        {
            throw new CliException(
                $"unknown quantization '{quantization}' — this build supports "
                + string.Join(", ", GgufConverter.Quantizations.Select(q => $"'{q}'"))
                + ", plus 'f16' as an alias for 'none'");
        }
        if (File.Exists(outFile))
        {
            // --force makes re-runs idempotent (TODO.md item 7).
            if (!HasOption(args, "--force"))
            {
                throw new CliException($"output '{outFile}' already exists — pass --force to overwrite");
            }
            File.Delete(outFile);
        }

        var report = GgufConverter.Convert(checkpointDir, outFile, quantization);

        Console.WriteLine($"converted: {report.OutputPath}");
        Console.WriteLine($"arch:       {report.Architecture}");
        Console.WriteLine($"tensors:    {report.TensorsWritten}  ({FormatBytes(report.OutputBytes)})");
        foreach (var note in report.Notes)
        {
            Console.WriteLine($"note:       {note}");
        }
        Console.WriteLine("load it with llama.cpp / LM Studio, e.g. 'llama-cli -m <file.gguf> -p \"hello\"' — validate the hybrid arch on the target build");
        return 0;
    }

    // ── layers: describe the per-layer policy table and tensors ──────────

    private static int Layers(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("layers requires a container directory");
        string componentId = OptionValue(args, "--component") ?? "target";

        using var container = Vindex3Container.Open(containerDir);
        var graph = container.Graph ?? throw new CliException("container records no system graph — nothing to describe");

        Console.WriteLine($"container: {containerDir}   model '{container.Index.Model}' ({container.Index.Family})");
        foreach (var component in graph.Components)
        {
            Console.WriteLine();
            Console.WriteLine($"component '{component.Id}' role={component.Role} source={component.SourceArtifact} layers={component.NumLayers} hidden={component.HiddenSize}");
            if (component.Attention is { Count: > 0 } policies)
            {
                string census = string.Join(", ", policies.GroupBy(p => p.Operator)
                    .Select(g => $"{g.Key} × {g.Count()}")
                    .OrderBy(x => x, StringComparer.Ordinal));
                Console.WriteLine($"  attention:  {census}");
            }
            else
            {
                Console.WriteLine("  attention:  no per-layer table recorded");
            }
            if (component.Perception is { } perception)
            {
                Console.WriteLine($"  perception: {perception}");
            }
        }

        if (container.Index.TokenMap is { } tokenMapPath)
        {
            try
            {
                using var manifest = JsonDocument.Parse(
                    File.ReadAllBytes(Path.Combine(container.Root, tokenMapPath)));
                var root = manifest.RootElement;
                string baseModel = root.GetProperty("base").GetProperty("model").GetString() ?? "?";
                string importedModel = root.GetProperty("imported").GetProperty("model").GetString() ?? "?";
                string scaffold = root.GetProperty("scaffold").GetProperty("model").GetString() ?? "?";
                var kinds = root.GetProperty("vocab").EnumerateArray()
                    .GroupBy(e => e.GetProperty("kind").GetString())
                    .ToDictionary(g => g.Key ?? "?", g => g.Count());
                int preserved = root.TryGetProperty("preserved", out var preservedProp)
                    ? preservedProp.GetProperty("segments").GetArrayLength()
                    : 0;
                Console.WriteLine();
                Console.WriteLine($"merged:     {baseModel} + {importedModel} (scaffold {scaffold}) — " +
                                  $"{root.GetProperty("vocab").GetArrayLength()} tokens " +
                                  $"({kinds.GetValueOrDefault("blended")} blended, " +
                                  $"{kinds.GetValueOrDefault("aligned_new")} aligned-new, " +
                                  $"{kinds.GetValueOrDefault("base_only")} base-only)" +
                                  (preserved > 0 ? $", {preserved} preserved segments under segments/source/" : string.Empty));
            }
            catch (Exception e) when (e is IOException or JsonException or KeyNotFoundException)
            {
                Console.WriteLine($"  note: token-map.json is present but unreadable: {e.Message}");
            }
        }

        var target = graph.Components.FirstOrDefault(c => c.Id == componentId)
            ?? throw new CliException($"system graph has no component '{componentId}'");
        Console.WriteLine();
        Console.WriteLine($"stack '{componentId}': {target.NumLayers} layers, hidden {target.HiddenSize}");

        if (target.Execution is { } surface)
        {
            Console.WriteLine("  surface:");
            if (surface.ContextLength is { } context)
            {
                Console.WriteLine($"    context: {context}");
            }
            if (surface.Attention is { } attn)
            {
                Console.WriteLine($"    attention: {attn.NumQHeads} q heads — {attn.NumKvHeads} kv heads, head_dim {attn.HeadDim}" +
                                  (attn.AttentionBias == true ? ", bias" : string.Empty) +
                                  (attn.OutputGate is not null ? ", output gate" : string.Empty));
            }
            if (surface.Ffn is { } ffn)
            {
                Console.WriteLine($"    ffn: intermediate {ffn.IntermediateSize}, {ffn.Activation}" +
                                  (ffn.FfnType == FfnType.Gated ? " (gated)" : string.Empty));
            }
            if (surface.Head is { } head)
            {
                Console.WriteLine($"    head: vocab {head.VocabSize}" + (head.HeadReusesEmbedding ? " (tied to the embedding)" : string.Empty));
            }
            if (surface.LinearAttention is not null)
            {
                Console.WriteLine("    linear_attention: carried surface facts (declared, refused by the planner)");
            }
        }

        if (target.Attention is { Count: > 0 } table)
        {
            Console.WriteLine("  layers:");
            for (int l = 0; l < target.NumLayers; l++)
            {
                var policy = table[l];
                string span = policy.Span is { } spanKind
                    ? spanKind.ToString().ToLowerInvariant()
                    : "declared:" + (policy.DeclaredSpan ?? "?");
                string window = policy.Window is { } w ? $" window={w}" : string.Empty;
                string geometry = policy.Geometry is { } geo ? $"  heads {geo.NumKvHeads}×{geo.HeadDim}" : string.Empty;
                string vFromK = policy.VFromK ? "  (V from K)" : string.Empty;
                Console.WriteLine($"    L{l,2}: {policy.Operator,-18} {span,-10}{window}  position {DescribePosition(policy.Position)}{geometry}{vFromK}");
            }
        }

        // Tensor inventory per layer, from the component's decoder-stack object.
        var stack = graph.Objects.FirstOrDefault(o => o.Component == componentId && o.Kind == ObjectKind.DecoderStack);
        if (stack is not null)
        {
            using var store = container.CreateOperandStore();
            if (store.SegmentPathFor(stack.Id) is { } segmentPath)
            {
                using var segment = SegmentFile.Open(Path.Combine(container.Root, segmentPath));
                var grouped = segment.Header.Tensors
                    .GroupBy(t => t.Name.Split('.')[0])
                    .OrderBy(g => int.TryParse(g.Key, out int n) ? n : int.MaxValue)
                    .ThenBy(g => g.Key, StringComparer.Ordinal);
                Console.WriteLine($"  tensors of '{stack.Id}':");
                foreach (var group in grouped)
                {
                    string details = string.Join("; ", group.OrderBy(t => t.Name, StringComparer.Ordinal)
                        .Select(t => $"{t.Name[(group.Key.Length + 1)..]} {t.Dtype} [{string.Join("x", t.Shape)}]"));
                    Console.WriteLine($"    L{group.Key,2}: {details}");
                }
            }
        }

        using (var store = container.CreateOperandStore())
        {
            try
            {
                var plan = Planner.Plan(container, componentId, store);
                Console.WriteLine($"  runtime: [served] plans and executes — {plan.Layers.Count} layers, hidden {plan.HiddenSize}" +
                                  $"{(plan.Embedding is null ? ", embedding none" : $", embedding {plan.Embedding.VocabSize}")}" +
                                  $"{(plan.Output is null ? ", head none (tied?)" : $", head {plan.Output.VocabSize}")}");
            }
            catch (UnsupportedOperatorException e)
            {
                Console.WriteLine($"  runtime: [refused] {e.Message}");
            }
        }
        return 0;
    }

    private static string DescribePosition(PositionPolicy position) => position switch
    {
        PositionNone => "none",
        PositionRope rope => $"rope θ={rope.Theta:0.###}",
        PositionPartialRope partial => $"partial rope θ={partial.Theta:0.###} f={partial.RotaryFactor:0.###}",
        PositionUnresolved unresolved => $"{unresolved.Kind} (unresolved)",
        _ => "?",
    };

    // ── export-mtp: emit the MTP drafter as a standalone checkpoint ─────────

    private static int ExportMtp(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException(
            "export-mtp requires a container directory, e.g. 'amql-cli export-mtp <container-dir> --out <drafter-dir>'");
        string outDir = OptionValue(args, "--out") ?? throw new CliException("export-mtp requires '--out <drafter-dir>'");

        using var container = Vindex3Container.Open(containerDir);
        var report = ModelExporter.ExportMtp(container, outDir);

        Console.WriteLine($"drafter:    {report.Model}");
        Console.WriteLine($"tensors:    {report.Tensors}  ({FormatBytes(report.PayloadBytes)})");
        foreach (var note in report.Notes)
        {
            Console.WriteLine($"note:       {note}");
        }
        Console.WriteLine("wrote:      model.safetensors, config.json" +
                          (File.Exists(Path.Combine(outDir, "tokenizer.json")) ? ", tokenizer.json" : string.Empty));
        Console.WriteLine("the drafter is the MTP module composed with the shared embedding and head — a standalone checkpoint for speculative decoding");
        return 0;
    }

    // ── generate-mtp: bootstrap an MTP drafter for a model without one ──────

    private static int GenerateMtp(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException(
            "generate-mtp requires a container directory, e.g. 'amql-cli generate-mtp <container-dir> --out <out>'");
        string outDir = OptionValue(args, "--out") ?? throw new CliException("generate-mtp requires '--out <out>'");
        int sample = IntOption(args, "--sample", 4096);
        int use = sample == int.MaxValue ? 4096 : sample;
        int eval = IntOption(args, "--eval", 1024);
        double ridge = DoubleOption(args, "--ridge", 1e-4);
        int? clusters = IntOptionOrNull(args, "--clusters");
        string? sweep = OptionValue(args, "--sweep");
        bool fitMode = HasOption(args, "--fit");

        IReadOnlyList<int> tokens = Array.Empty<int>();
        if (OptionValue(args, "--text") is { } textPath)
        {
            string text;
            try
            {
                text = File.ReadAllText(textPath);
            }
            catch (IOException e)
            {
                throw new CliException($"cannot read corpus '{textPath}': {e.Message}");
            }
            tokens = HfTokenizer.FromModelDir(containerDir).EncodeToIds(text);
        }

        var report = Amql.Merge.GenerateMtp.Transform(containerDir, outDir, tokens, use);

        Console.WriteLine($"generated:  {report.Model} + MTP drafter");
        Console.WriteLine($"trunk:      a copy of full-attention layer {report.TrunkLayer} " +
                          $"({report.Tensors} module tensors)");
        foreach (var note in report.Notes)
        {
            Console.WriteLine($"note:       {note}");
        }
        Console.WriteLine("wrote:      index.json, system_graph.json, segments/, tokenizer.json");

        // --fit runs the calculated path (phases 0–2) in one command —
        // "pre-trained" by calculation, no training: collect the model's
        // continuation pairs from the boot the drafter just got, then fit
        // the free block (single projector, or the K-projector mixture
        // {1,4,8} whose acceptance gate decides the shipped K).
        if (fitMode)
        {
            if (tokens.Count == 0)
            {
                throw new CliException("generate-mtp --fit requires '--text <corpus.txt>'");
            }
            int fitCount = use - eval - 2;
            if (fitCount < 1)
            {
                throw new CliException(
                    $"--sample {use} with --eval {eval} leaves no fit pairs — need sample >= eval + 3");
            }
            if (tokens.Count < use)
            {
                throw new CliException(
                    $"the corpus has {tokens.Count} tokens but --sample {use} needs {fitCount + eval + 2} — pass a smaller --sample");
            }
            var pairsDir = $"{outDir}-pairs";
            Amql.Merge.PairCollector.Collect(outDir, pairsDir, tokens, fitCount, eval);

            var candidates = sweep is not null
                ? sweep.Split(',').Select(s => int.Parse(s.Trim())).ToArray()
                : clusters is { } c
                    ? new[] { c }
                    : new[] { 1, 4, 8 };
            var kReport = Amql.Merge.MtpKProjectors.SweepAndFit(outDir, pairsDir, candidates, ridge);
            Console.WriteLine($"fit:        K ∈ {{{string.Join(", ", kReport.Sweep.Select(e => e.Clusters))}}}: " +
                string.Join("; ", kReport.Sweep.Select(e => $"K={e.Clusters} R² {e.R2:0.000} acc {e.GateAcceptance:0.0%}")));
            foreach (var note in kReport.Notes)
            {
                Console.WriteLine($"note:       {note}");
            }
            Console.WriteLine("the fitted free block has replaced the boot projector in the container.");
        }

        Console.WriteLine("the drafter is part of the container now — export emits it automatically:");
        Console.WriteLine($"  amql-cli export {outDir} --out <checkpoint-dir>");
        Console.WriteLine($"  amql-cli verify {outDir}");
        return 0;
    }

    // ── collect-mtp: the calculated drafter's data contract (phase 0) ────

    private static int CollectMtp(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException(
            "collect-mtp requires a container directory, e.g. 'amql-cli collect-mtp <container-dir> --out <pairs-dir> --text <corpus.txt>'");
        string outDir = OptionValue(args, "--out") ?? throw new CliException("collect-mtp requires '--out <pairs-dir>'");
        string textPath = OptionValue(args, "--text") ?? throw new CliException("collect-mtp requires '--text <corpus.txt>'");
        int fit = IntOption(args, "--fit", 512);
        int gate = IntOption(args, "--gate", 256);

        string text;
        try
        {
            text = File.ReadAllText(textPath);
        }
        catch (IOException e)
        {
            throw new CliException($"cannot read corpus '{textPath}': {e.Message}");
        }
        var ids = HfTokenizer.FromModelDir(containerDir).EncodeToIds(text);

        var data = Amql.Merge.PairCollector.Collect(containerDir, outDir, ids, fit, gate);

        Console.WriteLine($"pairs:      {data.FitCount} fit + {data.GateCount} held-out (shape [n, {2 * data.Hidden}] → [n, {data.Hidden}])");
        foreach (var note in data.Notes)
        {
            Console.WriteLine($"note:       {note}");
        }
        Console.WriteLine("wrote:      manifest.json, fit.x.bin, fit.y.bin, tokens.bin");
        return 0;
    }

    // ── fit-mtp: the calculated drafter's projector fit (phases 1–2) ──────

    private static int FitMtp(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException(
            "fit-mtp requires a container directory (a generate-mtp output), e.g. 'amql-cli fit-mtp <container> --pairs <pairs-dir>'");
        string pairsDir = OptionValue(args, "--pairs") ?? throw new CliException("fit-mtp requires '--pairs <pairs-dir>'");
        double ridge = DoubleOption(args, "--ridge", 1e-4);
        string? sweep = OptionValue(args, "--sweep");
        int? clusters = IntOptionOrNull(args, "--clusters");

        // Phase 2: the K-projector mixture. --sweep 1,4,8 measures the
        // candidates and ships the acceptance winner; --clusters K fits and
        // ships a single routed block. The plain fit (phase 1) stays the
        // single ridge projector.
        if (sweep is not null || clusters is > 1)
        {
            var candidates = sweep is not null
                ? sweep.Split(',').Select(s => int.Parse(s.Trim())).ToArray()
                : new[] { clusters!.Value };
            var kReport = Amql.Merge.MtpKProjectors.SweepAndFit(containerDir, pairsDir, candidates, ridge);
            Console.WriteLine($"sweep:      K ∈ {{{string.Join(", ", kReport.Sweep.Select(e => e.Clusters))}}}: " +
                string.Join("; ", kReport.Sweep.Select(e => $"K={e.Clusters} R² {e.R2:0.000} acc {e.GateAcceptance:0.0%}")));
            foreach (var note in kReport.Notes)
            {
                Console.WriteLine($"note:       {note}");
            }
            Console.WriteLine("the fitted free block has replaced the projector in the container — export emits the drafter companion:");
            Console.WriteLine($"  amql-cli export {containerDir} --out <checkpoint-dir>");
            return 0;
        }

        var report = Amql.Merge.MtpFitter.FitAndAssemble(containerDir, pairsDir, ridge);

        Console.WriteLine($"fit:        {report.FitCount} pairs, hidden {report.Hidden}, ridge {report.LambdaRel:g3}");
        foreach (var note in report.Notes)
        {
            Console.WriteLine($"note:       {note}");
        }
        Console.WriteLine("the fitted projector has replaced fc.weight in the container — export emits the drafter companion:");
        Console.WriteLine($"  amql-cli export {containerDir} --out <checkpoint-dir>");
        return 0;
    }

    // ── import: merge a second model into the container ──────────────────

    private static int Import(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException(
            "import requires a container directory first, e.g. 'amql-cli import <container-dir> <model> --out <merged>'");
        var imported = Arg(args, 1) ?? throw new CliException("import requires the model to import");
        string outDir = OptionValue(args, "--out") ?? throw new CliException("import requires '--out <merged-dir>'");
        bool importedIsContainer = HasOption(args, "--container");

        var report = ModelMerger.Import(containerDir, imported, outDir, importedIsContainer);

        Console.WriteLine($"imported:   {report.ImportedModel} into {report.BaseModel}");
        Console.WriteLine($"result:     {report.ResultModel}");
        Console.WriteLine($"scaffold:   {report.Scaffold}  (hidden {report.HiddenSize}, layers {report.Layers})");
        Console.WriteLine($"vocab:      {report.BaseVocab} + {report.ImportedVocab} → {report.MergedVocab} " +
                          $"({report.Blended} blended, {report.AlignedNew} aligned-new, {report.BaseOnly} base-only)");
        Console.WriteLine($"storage:    {report.Dtype} × {report.HiddenSize}" + (report.HeadTied ? ", head tied" : ", head materialised"));
        if (report.PreservedSegments > 0)
        {
            Console.WriteLine($"preserved:  {report.PreservedSegments} segments of the replaced stack under segments/source/");
        }
        foreach (var note in report.Notes)
        {
            Console.WriteLine($"note:       {note}");
        }
        Console.WriteLine("wrote:      index.json, system_graph.json, token-map.json, tokenizer.json, segments/");
        Console.WriteLine("the merged container tracks every token relationship in token-map.json and exports as one model:");
        Console.WriteLine($"  amql-cli export {outDir} --out <checkpoint-dir>");
        return 0;
    }

    // ── moe-ify: restructure a dense container into a routed MoE ──────────

    private static int MoeIfy(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException(
            "moe-ify requires a container directory, e.g. 'amql-cli moe-ify <container-dir> --out <moe-dir> --text <corpus.txt>'");
        string outDir = OptionValue(args, "--out") ?? throw new CliException("moe-ify requires '--out <moe-dir>'");
        string textPath = OptionValue(args, "--text") ?? throw new CliException("moe-ify requires '--text <corpus.txt>'");
        int experts = IntOption(args, "--experts", 8);
        int topK = IntOption(args, "--top-k", 2);
        int sample = IntOption(args, "--sample", 4096);
        int eval = IntOption(args, "--eval", 1024);
        bool renormalise = (OptionValue(args, "--policy") ?? "softmax") == "renormalise";

        string text;
        try
        {
            text = File.ReadAllText(textPath);
        }
        catch (IOException e)
        {
            throw new CliException($"cannot read corpus '{textPath}': {e.Message}");
        }
        var tokenizer = HfTokenizer.FromModelDir(containerDir);
        var ids = tokenizer.EncodeToIds(text);
        if (ids.Count < sample + eval)
        {
            throw new CliException(
                $"corpus encodes to {ids.Count} tokens — need at least {sample + eval} for sampling and evaluation");
        }
        var clusterTokens = ids.Take(sample).ToList();
        var evalTokens = ids.Skip(sample).Take(eval).ToList();

        var policy = renormalise
            ? ExpertRoutingPolicy.NormalisedOverSelected
            : ExpertRoutingPolicy.SoftmaxThenSelect;
        var report = Amql.Merge.MoeIfy.Transform(
            containerDir, outDir, clusterTokens, evalTokens, experts, topK, policy);

        Console.WriteLine($"moе-ified:  {report.Model}");
        Console.WriteLine($"routing:    {report.Experts} experts × top-{report.TopK} " +
                          $"(expert intermediate {report.ExpertIntermediateSize}) over {report.Layers} layers");
        foreach (var note in report.Notes)
        {
            Console.WriteLine($"note:       {note}");
        }
        Console.WriteLine("wrote:      index.json, system_graph.json, segments/, tokenizer.json");
        Console.WriteLine("every FFN is now judged routed and runs the top-k expert kernel — verify and export as usual:");
        Console.WriteLine($"  amql-cli verify {outDir}");
        Console.WriteLine($"  amql-cli export {outDir} --out <checkpoint-dir> [--quant mxfp4]");
        return 0;
    }

    // ── prune: drop decoder layers to meet a byte budget ──────────────────

    private static int Prune(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException(
            "prune requires a container directory, e.g. 'amql-cli prune <container-dir> --out <pruned> --target-bytes 2GiB'");
        string outDir = OptionValue(args, "--out") ?? throw new CliException("prune requires '--out <pruned-dir>'");
        string targetRaw = OptionValue(args, "--target-bytes") ?? OptionValue(args, "--target")
            ?? throw new CliException("prune requires '--target-bytes <size>' (bare bytes, or with B/KB/MB/GB/TB, KiB/MiB/GiB/TiB)");
        long targetBytes = ParseByteSize(targetRaw, "--target-bytes");
        string approachRaw = OptionValue(args, "--approach") ?? "provenance";
        var approach = approachRaw switch
        {
            "provenance" => PruneApproach.Provenance,
            "corpus" => PruneApproach.Corpus,
            "random" => PruneApproach.Random,
            _ => throw new CliException($"unknown pruning approach '{approachRaw}' — use provenance, corpus, or random"),
        };
        int seed = IntOption(args, "--seed", 42);
        int sample = IntOption(args, "--sample", 8192);
        int minLayers = IntOption(args, "--min-layers", 1);
        string? text = null;
        if (OptionValue(args, "--text") is { } textPath)
        {
            try
            {
                text = File.ReadAllText(textPath);
            }
            catch (IOException e)
            {
                throw new CliException($"cannot read corpus '{textPath}': {e.Message}");
            }
        }

        var report = LayerPruner.Prune(containerDir, outDir, new PruneOptions(
            TargetBytes: targetBytes,
            Approach: approach,
            Seed: seed,
            CorpusText: text,
            SampleCap: sample,
            MinLayers: minLayers));

        Console.WriteLine($"pruned:    {report.OutDir}");
        Console.WriteLine($"model:     {report.Model}");
        Console.WriteLine($"approach:  {report.Approach}     seed: {seed}");
        Console.WriteLine($"layers:    {report.OriginalLayers} → {report.KeptLayers}  (dropped {report.DroppedLayers.Count}: {CompactList(report.DroppedLayers)})");
        Console.WriteLine($"bytes:     {FormatBytes(report.OriginalBytes)} → {FormatBytes(report.FinalBytes)}  " +
                          $"(target {FormatBytes(report.TargetBytes)}, saved {FormatBytes(report.OriginalBytes - report.FinalBytes)})");
        foreach (var note in report.Notes)
        {
            Console.WriteLine($"note:      {note}");
        }
        Console.WriteLine("the pruned container is a complete model — verify, export and import it as usual:");
        Console.WriteLine($"  amql-cli verify {outDir}");
        Console.WriteLine($"  amql-cli import {outDir} <smaller-model> --out <result>");
        return 0;
    }

    /// <summary>Byte size from user spelling: a bare count or with an
    /// IEC/SI suffix (case-insensitive), e.g. <c>2GiB</c>, <c>512MiB</c>,
    /// <c>1.5GB</c>, <c>800KB</c>, <c>10B</c>.</summary>
    private static long ParseByteSize(string raw, string flag)
    {
        string text = raw.Trim();
        int digits = 0;
        while (digits < text.Length && (char.IsDigit(text[digits]) || text[digits] == '.'))
        {
            digits++;
        }
        string number = text[..digits];
        string unit = text[digits..].Trim().ToUpperInvariant();
        if (!double.TryParse(number, System.Globalization.CultureInfo.InvariantCulture, out double value) || value <= 0)
        {
            throw new CliException($"{flag} '{raw}' is not a positive byte size");
        }
        long multiplier = unit switch
        {
            "" or "B" => 1,
            "K" or "KB" or "KIB" => 1024,
            "M" or "MB" or "MIB" => 1024L * 1024,
            "G" or "GB" or "GIB" => 1024L * 1024 * 1024,
            "T" or "TB" or "TIB" => 1024L * 1024 * 1024 * 1024,
            _ => throw new CliException($"{flag} '{raw}': unit '{unit}' is not recognized " +
                                        "(use B/KB/MB/GB/TB, KiB/MiB/GiB/TiB, or a bare byte count)"),
        };
        double bytes = value * multiplier;
        if (bytes > long.MaxValue)
        {
            throw new CliException($"{flag} '{raw}' overflows a 64-bit byte count");
        }
        return (long)bytes;
    }

    private static string CompactList(IReadOnlyList<int> values) =>
        values.Count <= 24 ? string.Join(",", values) : string.Join(",", values.Take(24)) + ",…";

    // ── patch option plumbing ──────────────────────────────────────────────

    private static double DoubleOption(string[] args, string name, double fallback) =>
        double.TryParse(OptionValue(args, name), System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : fallback;

    // ── fine-tune: teacher-forced output-head adaptation ───────────────────

    private static int FineTune(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException(
            "fine-tune requires a container directory");
        string dataPath = OptionValue(args, "--data") ?? throw new CliException(
            "fine-tune requires '--data <pairs.tsv>' — a tab-separated file of prompt<TAB>completion lines");
        string outPatch = OptionValue(args, "--out") ?? throw new CliException(
            "fine-tune requires '--out <patch.safetensors>'");
        string? modelDir = TokenizerDir(args);
        if (modelDir is null && File.Exists(Path.Combine(containerDir, "tokenizer.json")))
        {
            modelDir = containerDir;
        }
        if (modelDir is null)
        {
            throw new CliException(
                "fine-tune needs a tokenizer — pass '--tokenizer <checkpoint-dir>' or " +
                "use a container that was encoded with a tokenizer.json beside it");
        }
        string component = OptionValue(args, "--component") ?? "target";
        double lr = DoubleOption(args, "--lr", 1e-4);
        int epochs = IntOption(args, "--epochs", 1);

        using var container = Vindex3Container.Open(containerDir);
        var tokenizer = HfTokenizer.FromModelDir(modelDir);

        bool inContainer = modelDir.Equals(containerDir, StringComparison.OrdinalIgnoreCase);
        Console.WriteLine($"container: {containerDir} (weights)   tokenizer: {modelDir} ({(inContainer ? "in container" : "checkpoint")})");
        Console.WriteLine($"data:      {dataPath}");

        var workingSet = WeightWorkingSetExtensions.FromEnv();
        if (workingSet == WeightWorkingSet.Mxfp4 && CudaShim.Enabled)
        {
            Console.WriteLine("cuda:      MXFP4 packs resident on device — GEMMs run on the GPU (FP16 tensor cores, FP32 accumulate)");
        }
        else if (workingSet == WeightWorkingSet.Mxfp4)
        {
            Console.WriteLine("weights:   MXFP4 working set (dequantised to f32 on CPU)");
        }
        else
        {
            Console.WriteLine($"weights:   {workingSet} (managed f32 path)");
        }

        var report = ModelFinetuner.FineTune(
            container, component, dataPath, outPatch, tokenizer, lr, epochs);

        Console.WriteLine();
        Console.WriteLine($"component:  {report.ComponentId}");
        Console.WriteLine($"head:       {report.HeadObjectId}/{report.HeadTensorName}  [{report.VocabSize}×{report.HiddenSize}]" +
                          (report.HeadReusesEmbedding ? " (reuses embedding)" : string.Empty));
        Console.WriteLine($"pairs:      {report.Pairs} training pairs → {report.Steps} teacher-forced steps");
        Console.WriteLine($"lr:         {report.LearningRate}  epochs: {report.Epochs}");
        Console.WriteLine($"patch:      {report.PatchPath}  (1 tensor)");
        foreach (var note in report.Notes)
        {
            Console.WriteLine($"note:       {note}");
        }
        Console.WriteLine();
        Console.WriteLine("apply it:   amql-cli generate <container> --prompt \"...\" --tokenizer <checkpoint> --patch " + outPatch);
        Console.WriteLine("bake it:    amql-cli export <container> --out <checkpoint> --patch " + outPatch);
        return 0;
    }

    // ── plumbing ───────────────────────────────────────────────────────────

    /// <summary>Loads and (when a container is at hand) shape-validates the
    /// <c>--patch</c> file. Returns null when no patch was given.</summary>
    private static WeightPatch? LoadPatch(string[] args, Vindex3Container? container)
    {
        string? path = OptionValue(args, "--patch");
        if (path is null)
        {
            return null;
        }
        var patch = WeightPatch.Load(path);
        if (container is not null)
        {
            patch.ValidateAgainst(container);
        }
        Console.WriteLine($"patch: {path} ({patch.Entries.Count} tensor{(patch.Entries.Count == 1 ? string.Empty : "s")})");
        return patch;
    }

    // ── classify: run a classifier (Jev/NLI-style) ──────────────────────

    private static int Classify(string[] args)
    {
        Console.WriteLine("classify is not yet implemented (Phase B — serve).");
        Console.WriteLine("Classifier containers can be created via 'encode' and exported back via 'export'.");
        Console.WriteLine("Use 'inspect <container> --classifier' to view the classifier surface.");
        return 0;
    }

    // ── convert-to-classifier: generative → classifier container ────────

    private static int ConvertToClassifier(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException(
            "convert-to-classifier requires a container directory, e.g. 'amql-cli convert-to-classifier <container> --num-labels 3 --out <dir>'");
        string outDir = OptionValue(args, "--out") ?? throw new CliException("convert-to-classifier requires '--out <dir>'");
        int numLabels = int.Parse(OptionValue(args, "--num-labels") ?? "2");

        using var container = Vindex3Container.Open(containerDir);
        var report = ModelConverter.ConvertToClassifier(containerDir, outDir, numLabels);

        Console.WriteLine($"converted: {report.OutDir}");
        Console.WriteLine($"model:     {report.Model}");
        Console.WriteLine($"labels:    {numLabels}");
        foreach (var note in report.Notes)
        {
            Console.WriteLine($"note:      {note}");
        }
        Console.WriteLine("apply classification training data to the score head, then export:");
        Console.WriteLine($"  amql-cli export {outDir} --out <checkpoint>");
        return 0;
    }

    private static int ConvertToEmbedding(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException(
            "convert-to-embedding requires a container directory, e.g. 'amql-cli convert-to-embedding <container> --out <dir>'");
        string outDir = OptionValue(args, "--out") ?? throw new CliException("convert-to-embedding requires '--out <dir>'");

        using var container = Vindex3Container.Open(containerDir);
        var report = ModelConverter.ConvertToEmbedding(containerDir, outDir);

        Console.WriteLine($"converted: {report.OutDir}");
        Console.WriteLine($"model:     {report.Model}");
        foreach (var note in report.Notes)
        {
            Console.WriteLine($"note:      {note}");
        }
        Console.WriteLine("the embedding model can be exported back:");
        Console.WriteLine($"  amql-cli export {outDir} --out <checkpoint>");
        return 0;
    }

    // ── export-onnx: container → ONNX model ─────────────────────────────

    private static int ExportOnnx(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException(
            "export-onnx requires a container directory, e.g. 'amql-cli export-onnx <container> --out <model.onnx>'");
        string outPath = OptionValue(args, "--out") ?? throw new CliException("export-onnx requires '--out <model.onnx>'");
        string component = OptionValue(args, "--component") ?? "target";

        using var container = Vindex3Container.Open(containerDir);
        var patch = LoadPatch(args, container);
        var result = OnnxExporter.Export(container, component, outPath, patch);

        Console.WriteLine($"onnx:      {result.Path}");
        Console.WriteLine($"model:     {result.Model}");
        Console.WriteLine($"nodes:     {result.NodeCount}");
        Console.WriteLine($"weights:   {result.InitializerCount}");
        Console.WriteLine("the ONNX graph uses standard ops only (no custom ops) — compatible with ONNX Runtime 1.21+");
        return 0;
    }

    // ── create-model: new empty container with random weights ────────────

    private static int CreateModel(string[] args)
    {
        string outDir = OptionValue(args, "--out") ?? throw new CliException("create-model requires '--out <dir>'");
        int numLayers = IntOption(args, "--layers", 12);
        var spec = new ModelSpec(
            ModelName: OptionValue(args, "--name") ?? "untitled-model",
            Family: OptionValue(args, "--family") ?? "qwen3_5",
            HiddenSize: IntOption(args, "--hidden", 768),
            NumLayers: numLayers,
            NumQHeads: IntOption(args, "--heads", 12),
            NumKvHeads: IntOption(args, "--kv-heads", 2),
            HeadDim: IntOption(args, "--head-dim", 64),
            IntermediateSize: IntOption(args, "--intermediate", 2048),
            VocabSize: IntOption(args, "--vocab", 32000),
            ContextLength: IntOption(args, "--context", 2048),
            NormEps: DoubleOption(args, "--norm-eps", 1e-5),
            TieEmbeddings: !HasOption(args, "--untied"),
            LayerTypes: ParseLayerTypes(args, numLayers)
        );

        ModelInitializer.Create(spec, outDir);
        Console.WriteLine($"model:  {spec.ModelName}");
        Console.WriteLine($"arch:   {spec.NumLayers}L × {spec.HiddenSize}H, {spec.NumQHeads}Q/{spec.NumKvHeads}KV heads × {spec.HeadDim}D");
        Console.WriteLine($"vocab:  {spec.VocabSize}, context: {spec.ContextLength}, tied: {spec.TieEmbeddings}");
        Console.WriteLine($"init:   Xavier-uniform weights, norm weights = 1.0");
        Console.WriteLine($"out:    {outDir}");
        Console.WriteLine("the container is ready for training — use 'amql-cli fine-tune' to apply training data");
        return 0;
    }

    private static IReadOnlyList<string> ParseLayerTypes(string[] args, int numLayers)
    {
        string raw = OptionValue(args, "--layer-types") ?? string.Empty;
        if (string.IsNullOrWhiteSpace(raw))
        {
            return Enumerable.Repeat("full_attention", numLayers).ToList();
        }
        var parts = raw.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length == 0) return Enumerable.Repeat("full_attention", numLayers).ToList();
        if (parts.Length == numLayers) return parts;
        if (parts.Length < numLayers)
        {
            var repeated = new List<string>(numLayers);
            while (repeated.Count < numLayers) repeated.AddRange(parts);
            return repeated.Take(numLayers).ToList();
        }
        throw new CliException(
            $"--layer-types has {parts.Length} entries but model has {numLayers} layers");
    }

    // ── plumbing ───────────────────────────────────────────────────────────

    private static void PrintHelp()
    {
        Console.WriteLine("""
            amql-cli — load an HF checkpoint into a canonical VINDEX3 container,
            then run and inspect inference against it

            USAGE:
              amql-cli [--cpu | --gpu] [--progress] [--verbose] <command> [args]

            Commands:
              amql-cli encode <model-dir> --out <container-dir>   map + materialise
              amql-cli verify <container-dir>                     integrity + readiness
              amql-cli synth-model <dir>                          write an executable demo checkpoint
              amql-cli tokens --tokenizer <checkpoint-dir> "text"
              amql-cli decode --tokenizer <checkpoint-dir> <id,id,…>
              amql-cli route <container-dir> <A> <B> --tokenizer <checkpoint-dir>
                              [--top 5] [--templates 8] [--trace-layer-start 8]
                              [--trace-layer-end 24] [--no-trace] [--corrupt the]
                              [--patch <patch.safetensors>]
              amql-cli path <container-dir> <A> <B>
                              [--topk 6] [--max-nodes 48] [--max-depth 6]
                              [--patch <patch.safetensors>]
              amql-cli generate <container-dir>
                              --prompt "text" --tokenizer <checkpoint-dir>
                              [--steps 8] [--temperature 0] [--top-k 0] [--top-p 0]
                              [--seed 42] [--logits K] [--component target]
                              [--patch <patch.safetensors>]
                              [--trace] [--trace-tensors] [--weights f32|bf16|mxfp4]
                              [--trace-json <trace.json>]
                              [--attribute --attribute-corrupt <text>]
                              [--attribute-source <row>] [--attribute-layers <end>]
              amql-cli inspect-token <container-dir> <token>
                              [--tokens ctx,ids] [--neighbors 5] [--logits K]
                              [--tokenizer <checkpoint-dir>] [--component target]
                              [--patch <patch.safetensors>]
              amql-cli change-tensor <container-dir> <object> <tensor> <cell>
                              (--set V | --add V | --scale F | --zero)
                              --out <patch.safetensors>
              amql-cli edit-tensor <container-dir> <object> <tensor>
                              (--set V | --add V | --scale F | --zero)
                              --out <patch.safetensors>
              amql-cli save-lora <patch.safetensors> --out <lora-dir>
                              [--rank 8] [--alpha 16] [--container <container-dir>]
              amql-cli export <container-dir> --out <checkpoint-dir>
                              [--patch <patch.safetensors>] [--quant mxfp4]
                              [--arch qwen4-next]
              amql-cli to-gguf <checkpoint-dir> --out <file.gguf> [--force]
                    [--quant none|f16|q4_0|mxfp4|mxfp4_moe]
              amql-cli export-mtp <container-dir> --out <drafter-dir>
              amql-cli generate-mtp <container-dir> --out <out>
                              [--text <corpus.txt>] [--sample 4096] [--fit]
                              [--eval 1024] [--clusters K | --sweep K1,K2,K3]
                              [--ridge 1e-4]
              amql-cli collect-mtp <container-dir> --out <pairs-dir> --text <corpus.txt>
                              [--fit 512] [--gate 256]
              amql-cli fit-mtp <container-dir> --pairs <pairs-dir>
                              [--ridge 1e-4] [--clusters K] [--sweep K1,K2,K3]
              amql-cli layers <container-dir> [--component target]
              amql-cli import <container-dir> <model> --out <merged-dir>
                              [--container]
              amql-cli moe-ify <container-dir> --out <moe-dir> --text <corpus.txt>
                              [--experts 8] [--top-k 2] [--sample 4096] [--eval 1024]
                              [--policy softmax|renormalise]
              amql-cli prune <container-dir> --out <pruned-dir> --target-bytes <size>
                              [--approach provenance|corpus|random] [--seed 42]
                              [--text <corpus.txt>] [--sample 8192] [--min-layers 1]
              amql-cli fine-tune <container-dir> --data <pairs.tsv>
                              --out <patch.safetensors>
                              [--tokenizer <checkpoint-dir>] [--component target]
                              [--lr 1e-4] [--epochs 1]
              amql-cli help

            Example:
              amql-cli synth-model demo-model
              amql-cli encode demo-model --out demo-container
              amql-cli generate demo-container --prompt "hi" --tokenizer demo-model
              amql-cli route demo-container France Paris --tokenizer demo-model --top 5

            route probes relationships between two tokens: template-scored
            links (capital, language, contains, …) each with (layer, head,
            position) attention coordinates, and — for the strongest link —
            causal-tracing layer weights naming exactly which residual
            tensors to adjust (patch/LoRA) to change the propensity.
            path searches the token-continuation graph bidirectionally
            (Dijkstra-style, meet in the middle) and returns the token chain
            without relation names.
            change-tensor edits one cell of a weight and records the f32
            delta in a patch file ("--add"/"--scale" compose across runs);
            save-lora factors a patch's 2-D deltas into lora_A/lora_B with
            alpha/r scaling for the ORIGINAL (unpatched) model weights.
            export materialises an HF checkpoint directory (config.json +
            model.safetensors + tokenizer.json, plus whatever ancillary
            config the container carries — chat template, processor and
            generation configs) from the container — the
            inverse of encode — with patch deltas baked into the stored
            tensors (unpatched tensors are copied byte-identically), so the
            result is a plain original model again. Pass --arch qwen4-next
            to target the Qwen3.8-Flash-Next (qwen4_exp) architecture:
            tensor names and config.json follow the Flash-Next checkpoint
            contract, with HyperConnection placeholder tensors emitted as
            zeros so the checkpoint is structurally loadable by vLLM/SGLang.
            Pass --quant mxfp4 to export the stack's projection matrices in
            the OCP MXFP4 form (FP4 E2M1 elements, two per byte,
            per-32-element E8M0 scales); embeddings, norms, biases and the
            output head keep their full precision, and the quantized
            checkpoint is terminal for this build (the encoder reads
            full-precision dtypes). A materialised VISION tower rides the
            same shard under model.visual.* (it is part of the model), and
            a materialised MTP drafter exports automatically alongside as
            mtp.safetensors + mtp.config.json — or standalone via
            export-mtp.
            to-gguf converts an exported HF checkpoint directory
            (config.json + model.safetensors + tokenizer.json) into a
            GGUF v3 file for llama.cpp / LM Studio. A chat template is
            embedded as tokenizer.chat_template when the directory carries
            chat_template.jinja (or one inside tokenizer_config.json);
            without it the GGUF is completion-only and the run says so.
            --quant picks the weight encoding: none/f16 writes F16, q4_0
            writes ggml's Q4_0 (ftype 2), mxfp4 writes OCP MXFP4 for every
            quantizable weight (ggml type 39: 32 E2M1 values sharing one
            E8M0 exponent per 17-byte block), and mxfp4_moe follows
            llama.cpp's MXFP4_MOE recipe (ftype 38) — 3-D MoE expert stacks
            in MXFP4 and every other quantizable weight in Q8_0. Norms,
            embeddings, the output head, MoE routers and all 1-D tensors stay
            full precision in every mode, matching llama-quantize's skip list.
            Qwen3.5-family
            checkpoints (hybrid linear/full attention, optional MoE) are
            emitted as the qwen35 / qwen35moe architectures following
            llama.cpp's converter: linear-attention tensors map to
            attn_qkv / attn_gate / ssm_* with the V-head tiled reorder and
            the value transforms (-exp A_log, norms +1, conv squeeze),
            MoE experts stack into 3-D ffn_*_exps with the transposed
            router, full-attention layers keep attn_q/k/v/output, and the
            required Qwen3.5 MRoPE section + ssm/recurrent metadata are
            written. Weights are F16 (BF16 sources, lossless in the normal
            range); the vision tower is skipped and the MTP drafter stays
            a separate shard. Validate by loading the file in the target
            llama.cpp build.
            layers lists every component and, for the selected one, the
            per-layer attention policy table and tensor inventory, then
            whether the planner serves the stack or refuses it by name.
            import merges a second model into the container: the token
            vocabularies are related by token string (the tokenization
            mapping layer), the embedding/head grow to the union
            vocabulary via an anchored least-squares alignment of the
            imported space into the base space, and the stack evolves to
            the larger shape (the bigger model scaffolds it; intermediate
            layers are never averaged). token-map.json tracks every token
            relationship and layer provenance; the replaced stack is
            preserved under segments/source/. Pass --container when the
            model to import is itself a container rather than a checkpoint
            directory.
            moe-ify restructures a DENSE container into a routed mixture
            of experts: it samples FFN inputs through the model, clusters
            each layer's intermediate units by co-activation into
            balanced --experts, slices the gate/up rows and down columns
            into per-expert tensors (each expert its own segment — the
            "separate enough to run as such" guarantee), materialises a
            linear per-layer router (--policy softmax|renormalise), and
            rebuilds the container so every FFN is judged routed and runs
            the top-k expert kernel. The held-out perplexity gate prints
            dense → moе for the chosen --eval tokens; --sample tokens
            drive the clustering.
            prune shrinks a container down to a byte budget by dropping
            whole decoder layers (the vocabulary and every non-layer file
            are untouched, the kept layers are renumbered, and the result
            is a complete, mergeable container). --target-bytes accepts
            bare bytes or B/KB/MB/GB/TB, KiB/MiB/GiB/TiB suffixes. The
            approach ranks which layers drop first: "provenance" uses
            token-map.json layer provenance (grown tensors first, then
            top of the stack — deterministic, no corpus), "corpus" scores
            each layer's residual delta on one forward pass over --text
            (needs a tokenizer beside the container), and "random" is a
            seeded uniform order (--seed) for ablation baselines. The
            prune drops exactly as many layers as needed to get under the
            budget (never below --min-layers), verifies the exact rebuilt
            sizes before writing, and refuses when the target is below
            the floor.
            fine-tune runs supervised fine-tuning via teacher-forced
            output-head adaptation. The --data file is a TSV of
            prompt<TAB>completion lines; the model runs forward on each
            prompt, then teacher-forces the completion tokens and
            accumulates per-token deltas Δhead[target] += lr · h (the
            post-norm hidden state) — a single SGD step per position,
            no autograd. The result is a standard AMQL weight patch
            (one F32 tensor: the head delta) that works with --patch
            on any command; export --patch bakes it into the checkpoint.
            When the head reuses the embedding table the embedding is
            also adjusted. The learning rate and epochs are optional;
            papers typically use lr ∈ [1e-5, 1e-4] for LoRA-style
            fine-tuning.
            generate-mtp boots an MTP drafter for a dense model that has
            none: the trunk is a copy of the model's last full-attention
            layer, the drafter/final norms copy the final norm, the fc
            projector boots as the mean-combination 0.5·(norm(h)+norm(e)),
            and the shared embedding/head compose the rest. The module is
            materialised into the container (mtp.stack), so export emits
            the companion automatically; --text provides the corpus for
            the zero-shot draft-acceptance gate, --sample bounds it.
            --fit runs the calculated path in one command: collect the
            model's continuation pairs (fit = --sample − --eval − 2,
            held-out = --eval) and fit the free block by closed-form ridge
            least squares — the single projector, or the K-projector
            mixture {1,4,8} whose held-out acceptance gate ships the
            winner (--clusters K / --sweep override the candidates).
            Any pathway (route, path, generate, inspect-token, and
            tokens/decode, which parse but cannot be affected) accepts
            --patch to run with the patch's deltas merged into the loaded
            weights; the container is never rewritten.
            --tokenizer is optional when the container was encoded with a
            tokenizer.json beside it (encode copies it in).
            --cpu forces the CPU path even when a GPU is present.
            --gpu requires the GPU and fails when the device or the
            native DLL is unavailable.  Without either flag the device is
            auto-detected: a CUDA-capable GPU with the native DLL present
            enables itself automatically (set AMQL_WEIGHTS=mxfp4 to put
            weights on the device).  AMQL_GPU=0 is equivalent to --cpu;
            AMQL_GPU=1 is the same auto-probe as the default.  --cpu and
            --gpu take priority over the env var.
            --progress turns on the machine-readable protocol (also via
            AMQL_PROGRESS=1): one JSON object per line on stdout, prefixed
            '##amql-progress ' for phase/done/total/percent updates and
            '##amql-result ' for the command's terminal outcome, so a
            front-end can bind progress exactly instead of scraping text.
            --verbose prints full stack traces for unexpected errors;
            without it a failure is one 'error: …' line on stderr.

            EXIT CODES:
              0  success — the command produced its answer, whatever it was;
              1  a legitimate negative result: the command ran fine and the
                 answer is 'not found'/'not satisfied' (path found no chain
                 within budget, verify's integrity check failed). A
                 ##amql-result line carries the reason when --progress is on;
              2  usage or runtime error — bad arguments, unreadable input,
                 an unsupported operator, an exception.

            Two kinds of directory are involved: the CONTAINER (<container-dir>,
            encode output, holds weights only) and the CHECKPOINT
            (--tokenizer, the original HF model directory whose tokenizer.json
            converts text to ids; --model-dir is an accepted alias).
            The encoder runs the G0→G3 pipeline: shard inventory, config facts,
            system graph + execution surface, canonical (unquantised) segments.
            Operators this build has not judged are recorded verbatim and refused
            at plan time by name — never approximated.
            """);
    }

    private static string? Arg(string[] args, int index) => index < args.Length ? args[index] : null;

    private static bool HasOption(string[] args, string name) =>
        args.Any(a => a == name);

    /// <summary>All non-option arguments in order; options that take a
    /// value are skipped together with their value, so positionals land in
    /// the same slots regardless of where the options sit.</summary>
    private static string[] Positionals(string[] args, params string[] valueOptions)
    {
        var result = new List<string>();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i].StartsWith("--", StringComparison.Ordinal))
            {
                if (valueOptions.Contains(args[i]))
                {
                    i++; // skip the option's value
                }
                continue;
            }
            result.Add(args[i]);
        }
        return result.ToArray();
    }

    /// <summary>First non-option argument; options that take a value are
    /// skipped together with their value, so <c>tokens --model-dir d "text"</c>
    /// and <c>tokens "text" --model-dir d</c> both find "text".</summary>
    private static string? FirstPositional(string[] args, params string[] valueOptions)
    {
        var pos = Positionals(args, valueOptions);
        return pos.Length > 0 ? pos[0] : null;
    }

    /// <summary>Second non-option argument (route's B token).</summary>
    private static string? SecondPositional(string[] args, params string[] valueOptions)
    {
        var pos = Positionals(args, valueOptions);
        return pos.Length > 1 ? pos[1] : null;
    }

    private static string? OptionValue(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == name)
            {
                return args[i + 1];
            }
        }
        return null;
    }

    private static int? IntOptionOrNull(string[] args, string name) =>
        int.TryParse(OptionValue(args, name), out var v) ? v : null;

    private static int IntOption(string[] args, string name, int fallback) =>
        IntOptionOrNull(args, name) ?? fallback;

    private static float FloatOption(string[] args, string name, float fallback) =>
        float.TryParse(OptionValue(args, name), System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : fallback;

    private static int IntOption(string[] args, int positionalIndex, int fallback) =>
        Arg(args, positionalIndex) is { } raw && int.TryParse(raw, out var v) ? v : fallback;

    private static int[] ParseIntList(string? raw, int[] fallback)
    {
        if (raw is null)
        {
            return fallback;
        }
        var parts = raw.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length == 0)
        {
            return fallback;
        }
        return parts.Select(p => int.Parse(p, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
    }

    private static int[]? ParseOptionalIntList(string? raw) => raw is null ? null : ParseIntList(raw, Array.Empty<int>());

    private static string FormatBytes(long bytes) => bytes switch
    {
        >= 1 << 30 => $"{bytes / 1073741824.0:0.00} GiB",
        >= 1 << 20 => $"{bytes / 1048576.0:0.00} MiB",
        >= 1 << 10 => $"{bytes / 1024.0:0.00} KiB",
        _ => $"{bytes} B",
    };

    private static string Short(string hex) => hex[..12];
}

internal sealed class CliException : Exception
{
    public CliException(string message) : base(message) { }
}