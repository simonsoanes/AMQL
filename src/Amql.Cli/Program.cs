using Amql.Hf;
using Amql.Inference;
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
            return args.Length == 0 ? 1 : 0;
        }

        try
        {
            return args[0] switch
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
                "save-lora" => SaveLora(args[1..]),
                _ => throw new CliException($"unknown command '{args[0]}'"),
            };
        }
        catch (CliException e)
        {
            Console.Error.WriteLine($"error: {e.Message}");
            Console.Error.WriteLine("run 'amql-cli help' for usage");
            return 2;
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"error: {e.Message}");
            return 2;
        }
    }

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
        var report = ModelToContainer.Encode(modelDir, outDir);

        Console.WriteLine();
        Console.WriteLine($"model:        {report.ModelId}");
        Console.WriteLine($"encoding:     {report.Encoding}");
        Console.WriteLine($"tensors:      {report.Tensors}");
        Console.WriteLine($"payload:      {FormatBytes(report.PayloadBytes)}");
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
            return 1;
        }

        // Operand resolution on real weights: tensor tables, shapes, stored
        // dtypes and payload widening all exercised from the container alone.
        using var store = container.CreateOperandStore();
        Console.WriteLine("operands:");
        var qProj = store.Resolve("target.decoder_stack", "3.self_attn.q_proj.weight");
        Console.WriteLine($"  3.self_attn.q_proj.weight: {qProj.Dtype.Label()} shape=[{string.Join("x", qProj.Shape)}]");
        var kvIn = store.Resolve("target.decoder_stack", "0.linear_attn.in_proj_qkv.weight");
        Console.WriteLine($"  0.linear_attn.in_proj_qkv.weight: {kvIn.Dtype.Label()} shape=[{string.Join("x", kvIn.Shape)}]");
        var aLog = store.Resolve("target.decoder_stack", "0.linear_attn.A_log");
        Console.WriteLine($"  0.linear_attn.A_log: {aLog.Dtype.Label()} shape=[{string.Join("x", aLog.Shape)}] (precision exception)");
        var norms = store.Resolve("target.final_norm", "weight");
        var widened = Amql.Safetensors.BitPattern.WidenToF32(norms.Dtype, norms.Payload);
        float firstNorm = widened[0];
        Console.WriteLine($"  final_norm/weight: {widened.Length} f32 values (first={firstNorm:F4}, dtype {norms.Dtype.Label()})");
        Console.WriteLine($"operand store: {store.TouchedObjects.Count} objects touched, {store.Loads} loads");

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

        var (links, notes) = RelationRouter.Route(container, component, tokenizer, a, b, options, Console.Write, patch);
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

        var result = PathFinder.Search(container, component, tokenizer, aId, bId, options, Console.Write, patch);
        Console.WriteLine();

        if (!result.Found)
        {
            Console.WriteLine($"no path found within the budget ({options.MaxNodes} expansions, depth {options.MaxDepth}).");
            return 1;
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
        return 0;
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

        using var container = Vindex3Container.Open(containerDir);
        var patch = LoadPatch(args, container);
        if (tokenizer is not null)
        {
            bool inContainer = tokenizerSource!.Equals(containerDir, StringComparison.OrdinalIgnoreCase);
            Console.WriteLine($"container: {containerDir} (weights)   tokenizer: {tokenizerSource} ({(inContainer ? "in container" : "checkpoint")})");
        }
        var (prefill, steps2) = InferenceRunner.Generate(
            container, component, tokens, steps, config, showTopK, patch);

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
                Console.WriteLine("   " + string.Join("  ",
                    candidates.Select(c => $"{c.Token} {c.Logit:0.####}({c.Probability * 100:0.###}%)")));
            }
            else
            {
                Console.WriteLine();
            }
        }

        if (tokenizer is not null)
        {
            var generatedIds = prefill.Concat(steps2.Select(s => s.Token)).ToArray();
            Console.WriteLine($"text:      {tokenizer.Decode(generatedIds)}");
        }
        Console.WriteLine($"position: {prefill.Length + steps}");
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
        string objectId = pos.Length > 0 ? pos[0] : throw new CliException("change-tensor requires an object id (e.g. target.embedding)");
        string tensorName = pos.Length > 1 ? pos[1] : throw new CliException("change-tensor requires a tensor name (e.g. weight, 0.self_attn.q_proj.weight)");
        string cell = pos.Length > 2 ? pos[2] : throw new CliException("change-tensor requires a cell: 'row,col' for a 2-D tensor, a flat index otherwise");

        var op = ParseEditOp(args);
        float value = ParseEditValue(args, op);

        string outPatch = OptionValue(args, "--out") ?? throw new CliException("change-tensor requires '--out <patch.safetensors>'");
        string? existingPath = File.Exists(outPatch) ? outPatch : null;
        var existing = TensorPatchTools.LoadOrEmpty(existingPath);

        using var container = Vindex3Container.Open(containerDir);
        var shape = TensorPatchTools.ResolveShape(container, objectId, tensorName);
        long flat = ParseCell(cell, shape, objectId, tensorName);

        var result = TensorPatchTools.ApplyEdit(container, objectId, tensorName, op, value, flat, existing);
        if (result.Removed)
        {
            if (existingPath is not null)
            {
                File.Delete(existingPath);
                Console.WriteLine($"patch {existingPath}: '{objectId}/{tensorName}'[{cell}]\n  {result.Before:0.######} → {result.After:0.######} — back at the base value; patch cleared (no changes remain).");
            }
            else
            {
                Console.WriteLine($"'{objectId}/{tensorName}'[{cell}]: {result.Before:0.######} → {result.After:0.######} — no change, nothing written.");
            }
            return 0;
        }

        WeightPatch.Save(outPatch, result.Entries, container.Index.Model);
        Console.WriteLine($"'{objectId}/{tensorName}' [{string.Join("x", result.Shape)}] {result.DtypeLabel} [{cell}] {result.Before:0.######} → {result.After:0.######} (Δ {result.After - result.Before:0.######})");
        Console.WriteLine($"patch: {outPatch} ({result.Entries.Count} tensor{(result.Entries.Count == 1 ? string.Empty : "s")})");
        Console.WriteLine("run a pathway with it: amql-cli route <container> A B --tokenizer <checkpoint> --patch " + outPatch);
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

    // ── patch option plumbing ──────────────────────────────────────────────

    private static double DoubleOption(string[] args, string name, double fallback) =>
        double.TryParse(OptionValue(args, name), System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : fallback;

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

    // ── plumbing ───────────────────────────────────────────────────────────

    private static void PrintHelp()
    {
        Console.WriteLine("""
            amql-cli — load an HF checkpoint into a canonical VINDEX3 container,
            then run and inspect inference against it

            USAGE:
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
              amql-cli inspect-token <container-dir> <token>
                              [--tokens ctx,ids] [--neighbors 5] [--logits K]
                              [--tokenizer <checkpoint-dir>] [--component target]
                              [--patch <patch.safetensors>]
              amql-cli change-tensor <container-dir> <object> <tensor> <cell>
                              (--set V | --add V | --scale F | --zero)
                              --out <patch.safetensors>
              amql-cli save-lora <patch.safetensors> --out <lora-dir>
                              [--rank 8] [--alpha 16] [--container <container-dir>]
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
            Any pathway (route, path, generate, inspect-token, and
            tokens/decode, which parse but cannot be affected) accepts
            --patch to run with the patch's deltas merged into the loaded
            weights; the container is never rewritten.
            --tokenizer is optional when the container was encoded with a
            tokenizer.json beside it (encode copies it in).

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