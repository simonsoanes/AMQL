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
            "entail requires a verb: check | run | compile | graft | eval | find | extract | optimise | review | accept");
        var rest = args[1..];
        return verb switch
        {
            "check" => EntailCheck(rest),
            "run" => EntailRun(rest),
            "compile" => EntailCompile(rest),
            "graft" => EntailGraft(rest),
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

    /// <summary>Runs an expression through the program's meaning alone — no
    /// weights — character by character.</summary>
    private static int EntailRun(string[] args)
    {
        var p = LoadEntail(args);
        var text = OptionValue(args, "--expr") ?? throw new CliException("entail run requires '--expr \"12+34=\"'");
        Console.WriteLine(text + EntailInterpreter.Answer(p, text.Select(c => c.ToString())));
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
        var session = new DecodeSession(plan, store, null, WeightWorkingSet.ResidentF32);

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
            foreach (var c in cases.Take(limit))
            {
                string category = c["category"]!.GetValue<string>();
                string prompt = c["prompt"]!.GetValue<string>();
                string answer = c["answer"]!.GetValue<string>();
                bool chat = c["chat"]?.GetValue<bool>() ?? false;
                string rendered = prompt;
                if (chat)
                {
                    if (template is null)
                    {
                        throw new CliException("this model has no chat template");
                    }
                    rendered = ChatTemplate.Apply(template, prompt, null);
                    if (template.Contains("<think>", StringComparison.Ordinal))
                    {
                        rendered += "<think>\n\n</think>\n\n";
                    }
                }

                session.Reset();
                var logits = session.Prefill(tokenizer.EncodeToIds(rendered).ToArray());
                var reply = new List<int>();
                for (int i = 0; i < (c["max_tokens"]?.GetValue<int>() ?? maxTokens); i++)
                {
                    int next = ArgMax(logits.Row(0));
                    if (stop.Contains(next))
                    {
                        break;
                    }
                    reply.Add(next);
                    logits = session.Step(next);
                }
                string text = tokenizer.DecodeText(reply);
                bool pass = EntailScore(text, answer, c["first"]?.GetValue<bool>() ?? false);
                var (p, n) = totals.GetValueOrDefault(category);
                totals[category] = (p + (pass ? 1 : 0), n + 1);
                details.Add(new JsonObject
                {
                    ["category"] = category, ["prompt"] = prompt, ["answer"] = answer,
                    ["reply"] = text, ["pass"] = pass,
                });
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
            var logits = session.Prefill(ids[..1]);
            double nll = 0;
            var top = new JsonArray();
            for (int i = 1; i < ids.Length; i++)
            {
                var row = logits.Row(0);
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
                logits = session.Step(ids[i]);
            }
            double mean = nll / (ids.Length - 1);
            Console.WriteLine($"text:       {ids.Length} tokens, loss {mean:F5} nats/token, perplexity {Math.Exp(mean):F3}");
            report["text"] = new JsonObject { ["tokens"] = ids.Length, ["loss"] = mean, ["top1"] = top };
        }

        if (reportPath is not null)
        {
            File.WriteAllText(reportPath, report.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        }
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
