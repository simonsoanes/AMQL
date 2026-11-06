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
            "fidelity" => EntailFidelity(rest),
            "extract" => EntailExtract(rest),
            "background" => EntailBackground(rest),
            "optimise" => EntailOptimise(rest),
            "review" => EntailReview(rest),
            "accept" => EntailAccept(rest),
            _ => throw new CliException($"unknown entail verb '{verb}'"),
        };
    }

    private static EntailProgram LoadEntail(string[] args)
    {
        var path = Arg(args, 0) ?? throw new CliException("entail needs a program file");
        try
        {
            return EntailCompiler.Compile(File.ReadAllText(path), IntOption(args, "--neurons-per-layer", 3584));
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
            split = text => tokenizer.EncodeToIds(text).Select(id => tokenizer.DecodeText(new List<int> { id }));
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

        var expression = OptionValue(args, "--expr") ?? throw new CliException("entail run requires '--expr \"12+34=\"' or '--cases <cases.jsonl>'");
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

    private static int EntailGraft(string[] args)
    {
        var p = LoadEntail(args);
        string host = OptionValue(args, "--host") ?? throw new CliException("entail graft requires '--host <checkpoint-dir>'");
        string outDir = OptionValue(args, "--out") ?? throw new CliException("entail graft requires '--out <checkpoint-dir>'");
        PrintEntail(p);
        var report = EntailWeights.WriteGraft(p, host, outDir, IntOption(args, "--context", 8192));
        PrintEntailBuild(report);
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
        var side = new SideBySideSession(new DecodeSession(plan, store, null, WeightWorkingSet.ResidentF32), adapters);

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
        var session = new SideBySideSession(new DecodeSession(plan, store, null, WeightWorkingSet.ResidentF32), adapters);

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

            (string Text, int Said, int Steered) Answer(SideBySideSession on, JsonObject c)
            {
                string rendered = EntailPrompt(c["prompt"]!.GetValue<string>(), template,
                    c["chat"]?.GetValue<bool>() ?? false, c["think"]?.GetValue<bool>() ?? false);
                on.Reset();
                on.Prefill(tokenizer.EncodeToIds(rendered).ToArray());
                var reply = new List<int>();
                int said = 0, steered = 0;
                for (int i = 0; i < (c["max_tokens"]?.GetValue<int>() ?? maxTokens); i++)
                {
                    var (next, source) = on.Choose();
                    if (stop.Contains(next))
                    {
                        break;
                    }
                    said += source == EntailTokenSource.Entail ? 1 : 0;
                    steered += source == EntailTokenSource.Narrowed ? 1 : 0;
                    reply.Add(next);
                    on.Step(next);
                }
                return (tokenizer.DecodeText(reply), said, steered);
            }

            // Cases are independent, so each thread takes the next one on its own session.
            // The first runs alone, because it is what loads the weights the others share.
            var replies = new (string Text, int Said, int Steered)[chosen.Count];
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
