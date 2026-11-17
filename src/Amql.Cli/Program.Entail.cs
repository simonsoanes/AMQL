using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Amql.Entail;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Cli;

// ── entail: compile a statement of what must follow from what into weights ──

internal static partial class Program
{
    private static int EntailCommand(string[] args)
    {
        string verb = Arg(args, 0) ?? throw new CliException(
            "entail requires a verb: check | run | compile | graft | adapter | side | eval | find | extract | optimise | review | accept");
        var rest = args[1..];
        return verb switch
        {
            "check" => EntailCheck(rest),
            "run" => EntailRun(rest),
            "compile" => EntailCompile(rest),
            "graft" => EntailGraft(rest),
            "adapter" => EntailAdapter(rest),
            "side" => EntailSide(rest),
            "eval" => EntailEval(rest),
            "find" => EntailFind(rest),
            "decompile" => EntailDecompile(rest),
            "from-lora" => EntailFromLora(rest),
            "fidelity" => EntailFidelity(rest),
            "extract" => EntailExtract(rest),
            "background" => EntailBackground(rest),
            "optimise" => EntailOptimise(rest),
            "review" => EntailReview(rest),
            "accept" => EntailAccept(rest),
            _ => throw new CliException($"unknown entail verb '{verb}'"),
        };
    }

    private static EntailProgram LoadEntail(string[] args, int? neuronsPerLayer = null, int headsPerLayer = EntailProgram.HeadsPerLayer)
    {
        var path = Arg(args, 0) ?? throw new CliException("entail needs a program file");
        try
        {
            return EntailCompiler.Compile(File.ReadAllText(path), IntOption(args, "--neurons-per-layer", neuronsPerLayer ?? 3584), headsPerLayer);
        }
        catch (EntailException e)
        {
            throw new CliException($"{path}: {e.Message}");
        }
    }

    private static void PrintEntail(EntailProgram p)
    {
        Console.WriteLine($"program:    {p.Name}");
        Console.WriteLine($"statements: {p.Statements} (after expansion)");
        Console.WriteLine($"lanes:      {p.Lanes.Count}");
        Console.WriteLine($"neurons:    {p.Ops.Count(o => o is Neuron)}");
        Console.WriteLine($"heads:      {p.Ops.Count(o => o is Head)}");
        Console.WriteLine($"depth:      {p.Layers} layers (widest {p.MaxNeurons} neurons)");
    }

    private static int EntailCheck(string[] args)
    {
        var p = LoadEntail(args);
        PrintEntail(p);
        Console.WriteLine($"attention:  every head beats every other token by at least " +
                          $"{EntailWeights.AttentionGap(p, IntOption(args, "--context", 8192)):F1} logits");
        return 0;
    }

    /// <summary>Runs text through the program's meaning alone — no weights — one
    /// character per token, or as a model's tokenizer would split it. With
    /// --cases, answers each case and reports the replies.</summary>
    private static int EntailRun(string[] args)
    {
        var p = LoadEntail(args);
        Func<string, IEnumerable<string>> split = text => text.Select(c => c.ToString());
        if (OptionValue(args, "--tokenizer") is { } dir)
        {
            var tokenizer = Tokenizer(dir);
            // Special tokens are kept: a program may read the chat template's markers.
            split = text => tokenizer.EncodeToIds(text).Select(id => tokenizer.DecodeText(new List<int> { id }, skipSpecialTokens: false));
        }

        int limit = IntOption(args, "--max-tokens", 32);
        if (OptionValue(args, "--cases") is { } casesPath)
        {
            var cases = File.ReadAllLines(casesPath).Where(l => l.Trim().Length > 0)
                .Select(l => JsonNode.Parse(l)!.AsObject()).ToList();
            var replies = new string[cases.Count];
            Parallel.For(0, cases.Count, i =>
                replies[i] = EntailInterpreter.Answer(p, split(cases[i]["prompt"]!.GetValue<string>()), limit));
            var details = new JsonArray();
            for (int i = 0; i < cases.Count; i++)
            {
                var c = (JsonObject)cases[i].DeepClone();
                c["reply"] = replies[i];
                details.Add(c);
            }
            var report = new JsonObject { ["cases"] = details }.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
            if (OptionValue(args, "--report") is { } reportPath)
            {
                File.WriteAllText(reportPath, report);
            }
            else
            {
                Console.WriteLine(report);
            }
            return 0;
        }

        // --file streams a text of any length through the program in fixed
        // memory: the rows its heads can see back, nothing else.
        if (OptionValue(args, "--file") is { } textPath)
        {
            var stream = new EntailStream(p);
            var watch = System.Diagnostics.Stopwatch.StartNew();
            string text = File.ReadAllText(textPath);
            string answer = stream.Answer(split(text), limit);
            Console.WriteLine($"read:   {text.Length:N0} characters, {stream.Position:N0} tokens in {watch.Elapsed.TotalSeconds:F1}s");
            Console.WriteLine($"state:  {stream.Window} row(s) kept back, {stream.StateBytes:N0} bytes, whatever the length");
            Console.WriteLine($"answer: {answer}");
            return 0;
        }

        var expression = OptionValue(args, "--expr") ?? throw new CliException("entail run requires '--expr \"12+34=\"', '--file <text>' or '--cases <cases.jsonl>'");
        Console.WriteLine(expression + EntailInterpreter.Answer(p, split(expression), limit));
        return 0;
    }

    private static int EntailCompile(string[] args)
    {
        var p = LoadEntail(args);
        string outDir = OptionValue(args, "--out") ?? throw new CliException("entail compile requires '--out <checkpoint-dir>'");
        PrintEntail(p);
        var report = EntailWeights.WriteFresh(p, outDir, IntOption(args, "--context", 8192));
        PrintEntailBuild(report);
        return 0;
    }

    /// <summary>Grafts into any host the importer can judge: the host is
    /// judged first, so the program is scheduled for the host's own heads and
    /// FFN width, and the block is written in the host's layer shape.</summary>
    private static int EntailGraft(string[] args)
    {
        string host = OptionValue(args, "--host") ?? throw new CliException("entail graft requires '--host <checkpoint-dir>'");
        string outDir = OptionValue(args, "--out") ?? throw new CliException("entail graft requires '--out <checkpoint-dir>'");
        EntailHost judged;
        try
        {
            judged = EntailHost.Judge(host);
        }
        catch (EntailException e)
        {
            throw new CliException(e.Message);
        }
        var t = judged.Target;
        var p = LoadEntail(args, Math.Min(3584, t.Intermediate), t.ProgramHeads);
        PrintEntail(p);
        Console.WriteLine($"host:       {t.Heads} heads ({t.ProgramHeads} for the program) over {t.KvGroups} groups of {t.HeadDim}, " +
                          $"{t.Intermediate} neurons per layer, {t.Ffn.ToString().ToLowerInvariant()} FFN" +
                          (t.PostNorms ? ", post-norms" : string.Empty) + (t.QkNorm != QkNormKind.None ? $", q/k norm {t.QkNorm.ToString().ToLowerInvariant()}" : string.Empty));
        try
        {
            var report = EntailWeights.WriteGraft(p, judged, outDir, IntOption(args, "--context", 8192),
                DoubleOption(args, "--output-norm", EntailWeights.DefaultHostOutputNorm));
            PrintEntailBuild(report);
        }
        catch (EntailException e)
        {
            throw new CliException(e.Message);
        }
        return 0;
    }

    /// <summary>Builds a program to run beside a partner, reading the partner's
    /// token ids. With --dense it is a container of f32 weights, for a runtime
    /// that can only run a model.</summary>
    private static int EntailAdapter(string[] args)
    {
        var p = LoadEntail(args);
        string tokenizer = OptionValue(args, "--tokenizer") ?? throw new CliException("entail adapter requires '--tokenizer <partner-dir>'");
        string outDir = OptionValue(args, "--out") ?? throw new CliException("entail adapter requires '--out <dir>'");
        PrintEntail(p);
        if (!args.Contains("--dense"))
        {
            var written = EntailSparseAdapter.Write(File.ReadAllText(args[0]), tokenizer, outDir);
            Console.WriteLine($"outputs:    {written.Outputs.Count} ({written.Rows.Count} partner tokens read)");
            Console.WriteLine($"wrote:      {outDir}");
            return 0;
        }
        string checkpoint = outDir.TrimEnd('/', '\\') + ".checkpoint";
        var (report, manifest) = EntailWeights.WriteAdapter(p, tokenizer, checkpoint, IntOption(args, "--context", 8192));
        PrintEntailBuild(report);
        Console.WriteLine($"outputs:    {manifest.Outputs.Count} ({manifest.Rows.Count} partner tokens read)");
        ModelToContainer.Encode(checkpoint, outDir, $"entail-{p.Name}");
        manifest.Save(outDir);
        Directory.Delete(checkpoint, recursive: true);
        return 0;
    }

    /// <summary>Grafts are exact only in f32 on the CPU, so a graft runs
    /// either resident in f32 or from disk. <c>--weights disk</c> (or
    /// AMQL_WEIGHTS=disk) streams the weights from the container; with
    /// neither, disk is chosen when the model's f32 weights would not fit in
    /// the memory budget, which a graft of a large host easily exceeds.</summary>
    private static WeightWorkingSet EntailWorkingSet(string[] args, Vindex3Container container)
    {
        string? mode = OptionValue(args, "--weights")
            ?? (WeightWorkingSetExtensions.FromEnv() == WeightWorkingSet.Disk ? "disk" : null);
        switch (mode)
        {
            case "f32" or "fp32":
                return WeightWorkingSet.ResidentF32;
            case "disk":
                Console.Error.WriteLine($"weights:    disk (exact f32, {FormatGiB(ComputeBudget.DiskCacheBytes)} kept in memory)");
                return WeightWorkingSet.Disk;
            case null:
                long need = container.WidenedF32Bytes();
                if (need <= ComputeBudget.MemoryBudgetBytes)
                {
                    return WeightWorkingSet.ResidentF32;
                }
                Console.Error.WriteLine($"weights:    disk: f32 weights need {FormatGiB(need)}, over the " +
                    $"{FormatGiB(ComputeBudget.MemoryBudgetBytes)} budget; {FormatGiB(ComputeBudget.DiskCacheBytes)} kept in memory, " +
                    "the rest read on every pass");
                return WeightWorkingSet.Disk;
            default:
                throw new CliException($"unknown weight mode '{mode}' for entail: use f32 or disk (grafts need exact f32)");
        }
    }

    private static string FormatGiB(long bytes) => $"{bytes / (double)(1L << 30):0.0} GiB";

    private static List<IEntailAdapter> OpenAdapters(string[] args, string tokenizerDir)
    {
        try
        {
            return (OptionValue(args, "--adapter") ?? string.Empty).Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(dir => EntailAdapterRunner.Load(dir, tokenizerDir)).ToList();
        }
        catch (EntailException e)
        {
            throw new CliException(e.Message);
        }
    }

    private static string EntailPrompt(string prompt, string? template, bool chat, bool think)
    {
        if (!chat)
        {
            return prompt;
        }
        if (template is null)
        {
            throw new CliException("this model has no chat template");
        }
        string rendered = ChatTemplate.Apply(template, prompt, null);
        if (template.Contains("<think>", StringComparison.Ordinal))
        {
            rendered += think ? "<think>\n" : "<think>\n\n</think>\n\n";
        }
        return rendered;
    }

    /// <summary>Decodes one prompt with a partner and its adapters, marking
    /// [[what a program said]] and {{what a rule steered}}.</summary>
    private static int EntailSide(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("entail side requires a partner container");
        string prompt = OptionValue(args, "--prompt") ?? throw new CliException("entail side requires '--prompt <text>'");
        string tokenizerDir = ResolveTokenizerDir(containerDir, args);
        var tokenizer = Tokenizer(tokenizerDir);
        var stop = ResolveStopTokens(tokenizerDir, tokenizer);
        string? template = ChatTemplate.Load(tokenizerDir);

        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, OptionValue(args, "--component") ?? "target", store);
        var adapters = OpenAdapters(args, tokenizerDir);
        var side = new SideBySideSession(new DecodeSession(plan, store, null, EntailWorkingSet(args, container)), adapters)
        {
            PriorPrompt = ChatPriorFor(tokenizer),
        };

        side.Prefill(tokenizer.EncodeToIds(EntailPrompt(prompt, template, args.Contains("--chat"), args.Contains("--think"))).ToArray());
        var spans = new List<(EntailTokenSource Source, List<int> Tokens)>();
        var counts = new Dictionary<EntailTokenSource, int>();
        for (int i = 0; i < IntOption(args, "--max-tokens", 128); i++)
        {
            var (token, source) = side.Choose();
            if (stop.Contains(token))
            {
                break;
            }
            counts[source] = counts.GetValueOrDefault(source) + 1;
            if (spans.Count == 0 || spans[^1].Source != source)
            {
                spans.Add((source, new List<int>()));
            }
            spans[^1].Tokens.Add(token);
            side.Step(token);
        }

        foreach (var (source, tokens) in spans)
        {
            string text = tokenizer.DecodeText(tokens);
            Console.Write(source switch
            {
                EntailTokenSource.Entail => $"[[{text}]]",
                EntailTokenSource.Narrowed => $"{{{{{text}}}}}",
                _ => text,
            });
        }
        Console.WriteLine();
        Console.WriteLine($"tokens:     partner {counts.GetValueOrDefault(EntailTokenSource.Partner)}, " +
                          $"steered {counts.GetValueOrDefault(EntailTokenSource.Narrowed)}, " +
                          $"said by a program {counts.GetValueOrDefault(EntailTokenSource.Entail)}");
        adapters.ForEach(a => a.Dispose());
        return 0;
    }

    private static void PrintEntailBuild(EntailBuildReport r)
    {
        Console.WriteLine($"hidden:     {r.Hidden} ({r.LanesUsed} lane dims, {r.LanesPeak} live at once)");
        Console.WriteLine($"layers:     {r.Layers} ({r.BlockLayers} compiled in front)");
        Console.WriteLine($"attention:  margin {r.MinAttentionGap:F1} logits over the whole context");
        Console.WriteLine($"tokens:     {r.TokensBound} bound" +
                          (r.TokensMissing.Count > 0 ? $", not in this vocabulary: {string.Join(" ", r.TokensMissing.Select(t => $"'{t}'"))}" : string.Empty));
        Console.WriteLine($"wrote:      {r.Directory}");
    }

    /// <summary>
    /// Greedy-decodes each case of a JSONL file and scores it. A case is
    /// <c>{"category","prompt","answer","chat"?}</c>; the case passes when the
    /// last number in the reply equals <c>answer</c>. With
    /// <c>--text &lt;file&gt;</c> it also measures next-token loss, so a model
    /// and its graft can be compared on ordinary prose.
    /// </summary>
    /// <summary>The blank of a ChatML prompt, for an adapter that weighs a token against the
    /// partner's prior; null for a tokenizer without ChatML's turn markers.</summary>
    private static Func<int[], int[]?>? ChatPriorFor(HfTokenizer tokenizer) =>
        tokenizer.AddedTokenId("<|im_start|>") is { } start && tokenizer.AddedTokenId("<|im_end|>") is { } end
            ? ChatPrior.ForChatMl(t => tokenizer.EncodeToIds(t), ids => tokenizer.DecodeText(ids, skipSpecialTokens: false), start, end)
            : null;

    /// <summary>The k most probable tokens of a row, as [text, probability] pairs.</summary>
    private static JsonArray TopTokens(ReadOnlySpan<float> logits, int k, HfTokenizer tokenizer)
    {
        var best = new List<int>();
        for (int i = 0; i < logits.Length; i++)
        {
            int at = best.Count;
            while (at > 0 && logits[i] > logits[best[at - 1]])
            {
                at--;
            }
            if (at < k)
            {
                best.Insert(at, i);
                if (best.Count > k)
                {
                    best.RemoveAt(k);
                }
            }
        }
        var top = new JsonArray();
        foreach (int id in best)
        {
            top.Add(new JsonArray(tokenizer.DecodeText(new List<int> { id }, skipSpecialTokens: false), Math.Round(Softmax(logits, id), 5)));
        }
        return top;
    }

    /// <summary>The softmax probability of one entry of a row of logits.</summary>
    private static double Softmax(ReadOnlySpan<float> logits, int index)
    {
        float max = float.NegativeInfinity;
        foreach (float v in logits)
        {
            max = Math.Max(max, v);
        }
        double sum = 0;
        foreach (float v in logits)
        {
            sum += Math.Exp(v - max);
        }
        return Math.Exp(logits[index] - max) / sum;
    }

    private static int EntailEval(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("entail eval requires a container directory");
        string tokenizerDir = ResolveTokenizerDir(containerDir, args);
        var tokenizer = Tokenizer(tokenizerDir);
        var stop = ResolveStopTokens(tokenizerDir, tokenizer);
        string? template = ChatTemplate.Load(tokenizerDir);
        int maxTokens = IntOption(args, "--max-tokens", 24);
        string? reportPath = OptionValue(args, "--report");

        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, OptionValue(args, "--component") ?? "target", store);
        var adapters = OpenAdapters(args, tokenizerDir);
        var session = new SideBySideSession(new DecodeSession(plan, store, null, EntailWorkingSet(args, container)), adapters)
        {
            PriorPrompt = ChatPriorFor(tokenizer),
        };

        static int ArgMax(ReadOnlySpan<float> row)
        {
            int best = 0;
            for (int i = 1; i < row.Length; i++)
            {
                if (row[i] > row[best])
                {
                    best = i;
                }
            }
            return best;
        }

        var report = new JsonObject();
        if (OptionValue(args, "--cases") is { } casesPath)
        {
            var cases = File.ReadAllLines(casesPath).Where(l => l.Trim().Length > 0)
                .Select(l => JsonNode.Parse(l)!.AsObject()).ToList();
            int limit = IntOption(args, "--limit", cases.Count);
            var totals = new SortedDictionary<string, (int Pass, int Count)>(StringComparer.Ordinal);
            var details = new JsonArray();
            var clock = Stopwatch.StartNew();
            var chosen = cases.Take(limit).ToList();

            // --watch TOKEN: the probability the partner gives TOKEN on the reply's first
            // row, before any adapter acts, so a decision can be read as a score.
            int? watched = null;
            if (OptionValue(args, "--watch") is { } watchText)
            {
                var ids = tokenizer.EncodeToIds(watchText);
                watched = ids.Count == 1 ? ids[0] : throw new CliException($"--watch '{watchText}' is {ids.Count} tokens, not one");
            }

            string? defaultPrefix = OptionValue(args, "--prefix")?.Replace("\\n", "\n");
            bool logprobs = args.Contains("--logprobs");

            (string Text, int Said, int Steered, double? Watch, double? PriorWatch, JsonArray? Top, JsonObject? Probs) Answer(SideBySideSession on, JsonObject c)
            {
                string rendered = EntailPrompt(c["prompt"]!.GetValue<string>(), template,
                    c["chat"]?.GetValue<bool>() ?? false, c["think"]?.GetValue<bool>() ?? false);
                on.Reset();
                // A prefix (--prefix, or the case's "prefix") is the reply's forced opening: it is
                // prefilled with the prompt and is the start of the reply that is scored.
                string prefix = c["prefix"]?.GetValue<string>() ?? defaultPrefix ?? string.Empty;
                var forced = prefix.Length > 0 ? tokenizer.EncodeToIds(prefix).ToArray() : Array.Empty<int>();
                on.Prefill(tokenizer.EncodeToIds(rendered).ToArray());
                double? watch = watched is { } w ? Softmax(on.Logits.Row(0), w) : null;
                double? priorWatch = watched is { } pw0 && on.Prior is { } prior ? prior[pw0] : null;
                JsonArray? top = watched is null ? null : TopTokens(on.Logits.Row(0), 5, tokenizer);
                foreach (int f in forced)
                {
                    on.Step(f);
                }
                var reply = new List<int>(forced);
                var probs = new JsonArray();
                int said = 0, steered = 0;
                for (int i = 0; i < (c["max_tokens"]?.GetValue<int>() ?? maxTokens); i++)
                {
                    var (next, source) = on.Choose();
                    if (logprobs)
                    {
                        probs.Add(Math.Round(Softmax(on.Logits.Row(0), next), 5));
                    }
                    if (stop.Contains(next))
                    {
                        break;
                    }
                    said += source == EntailTokenSource.Entail ? 1 : 0;
                    steered += source == EntailTokenSource.Narrowed ? 1 : 0;
                    reply.Add(next);
                    on.Step(next);
                }
                var tokens = logprobs ? new JsonArray(reply.Skip(forced.Length)
                    .Select(t => (JsonNode)tokenizer.DecodeText(new List<int> { t }, skipSpecialTokens: false)).ToArray()) : null;
                return (tokenizer.DecodeText(reply), said, steered, watch, priorWatch, top, tokens is null ? null : new JsonObject { ["tokens"] = tokens, ["p"] = probs });
            }

            // Cases are independent, so each thread takes the next one on its own session.
            // The first runs alone, because it is what loads the weights the others share.
            var replies = new (string Text, int Said, int Steered, double? Watch, double? PriorWatch, JsonArray? Top, JsonObject? Probs)[chosen.Count];
            int batch = Math.Clamp(IntOption(args, "--batch", 1), 1, Math.Max(chosen.Count, 1));
            if (batch > 1)
            {
                if (CudaShim.Enabled)
                {
                    throw new CliException("--batch runs on the CPU: add --cpu");
                }
                session.Reset();
                session.Prefill(tokenizer.EncodeToIds("a b").ToArray());
                session.Step(tokenizer.EncodeToIds("a")[0]);
                var sessions = new ConcurrentBag<SideBySideSession>(Enumerable.Range(0, batch).Select(_ => session.Sibling()));
                Parallel.For(0, chosen.Count, new ParallelOptions { MaxDegreeOfParallelism = batch }, i =>
                {
                    sessions.TryTake(out var on);
                    replies[i] = Answer(on!, chosen[i]);
                    sessions.Add(on!);
                });
            }
            else
            {
                for (int i = 0; i < chosen.Count; i++)
                {
                    replies[i] = Answer(session, chosen[i]);
                }
            }

            for (int i = 0; i < chosen.Count; i++)
            {
                var c = chosen[i];
                string category = c["category"]!.GetValue<string>();
                string prompt = c["prompt"]!.GetValue<string>();
                string answer = c["answer"]!.GetValue<string>();
                string text = replies[i].Text;
                bool pass = EntailScore(text, answer, c["first"]?.GetValue<bool>() ?? false);
                var (p, n) = totals.GetValueOrDefault(category);
                totals[category] = (p + (pass ? 1 : 0), n + 1);
                var detail = new JsonObject
                {
                    ["category"] = category, ["prompt"] = prompt, ["answer"] = answer,
                    ["reply"] = text, ["pass"] = pass,
                };
                if (replies[i].Watch is { } pw)
                {
                    detail["watch"] = pw;
                    detail["first_top"] = replies[i].Top;
                    if (replies[i].PriorWatch is { } prw)
                    {
                        detail["prior_watch"] = prw;
                    }
                }
                if (replies[i].Probs is { } pr)
                {
                    detail["logprobs"] = pr;
                }
                if (adapters.Count > 0)
                {
                    detail["said"] = replies[i].Said;
                    detail["steered"] = replies[i].Steered;
                }
                details.Add(detail);
            }
            Console.WriteLine($"{"category",-22} {"exact",12}");
            int allPass = 0, allCount = 0;
            var summary = new JsonObject();
            foreach (var (category, (pass, count)) in totals)
            {
                Console.WriteLine($"{category,-22} {pass,5}/{count,-5} {100.0 * pass / count,5:F1}%");
                allPass += pass;
                allCount += count;
                summary[category] = new JsonObject { ["pass"] = pass, ["count"] = count };
            }
            Console.WriteLine($"{"overall",-22} {allPass,5}/{allCount,-5} {100.0 * allPass / Math.Max(allCount, 1),5:F1}%   ({clock.Elapsed.TotalSeconds:F0}s)");
            report["summary"] = summary;
            report["cases"] = details;
        }

        if (OptionValue(args, "--text") is { } textPath)
        {
            int count = IntOption(args, "--text-tokens", 1024);
            var ids = tokenizer.EncodeToIds(File.ReadAllText(textPath)[..(count * 12)]).Take(count).ToArray();
            session.Reset();
            session.Prefill(ids[..1]);
            double nll = 0;
            int spoke = 0, overruled = 0;
            var top = new JsonArray();
            for (int i = 1; i < ids.Length; i++)
            {
                var row = session.Logits.Row(0);
                if (!session.Verdict.Silent)
                {
                    spoke++;
                    overruled += session.Choose().Token != ArgMax(row) ? 1 : 0;
                }
                double max = double.NegativeInfinity;
                foreach (var v in row)
                {
                    max = Math.Max(max, v);
                }
                double sum = 0;
                foreach (var v in row)
                {
                    sum += Math.Exp(v - max);
                }
                nll += Math.Log(sum) + max - row[ids[i]];
                top.Add(ArgMax(row));
                session.Step(ids[i]);
            }
            double mean = nll / (ids.Length - 1);
            var (partnerTime, adapterTime) = session.Spent;
            Console.WriteLine($"text:       {ids.Length} tokens, loss {mean:F5} nats/token, perplexity {Math.Exp(mean):F3}");
            report["text"] = new JsonObject { ["tokens"] = ids.Length, ["loss"] = mean, ["top1"] = top };
            if (adapters.Count > 0)
            {
                Console.WriteLine($"adapters:   spoke at {spoke} of {ids.Length - 1} positions, overruling the partner at {overruled}");
                Console.WriteLine($"time:       {adapterTime.TotalMilliseconds / ids.Length:F2} ms a token in adapters, " +
                                  $"{partnerTime.TotalMilliseconds / ids.Length:F2} ms in the partner");
                report["text"]!["spoke"] = spoke;
                report["text"]!["overruled"] = overruled;
            }
        }

        if (reportPath is not null)
        {
            File.WriteAllText(reportPath, report.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        }
        adapters.ForEach(a => a.Dispose());
        return 0;
    }

    private static readonly Regex EntailNumber = new(@"-?\d[\d,]*(?:\.\d+)?", RegexOptions.Compiled);

    private static bool EntailScore(string reply, string answer, bool first)
    {
        int think = reply.LastIndexOf("</think>", StringComparison.Ordinal);
        if (think >= 0)
        {
            reply = reply[(think + 8)..];
        }
        var numbers = EntailNumber.Matches(reply);
        if (numbers.Count == 0)
        {
            return false;
        }
        var chosen = first ? numbers[0].Value : numbers[^1].Value;
        return chosen.Replace(",", string.Empty, StringComparison.Ordinal) == answer;
    }
}
