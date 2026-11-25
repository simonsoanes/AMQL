using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Entail;
using Amql.Inference;
using Amql.Inference.Reasoning;
using Amql.Vindex3;

namespace Amql.Cli;

// ── entail bench: problem solving on generated, never-seen problems ──

internal static partial class Program
{
    /// <summary>
    /// <c>entail bench</c> measures problem solving: it generates novel problems
    /// (<see cref="ProblemSuite"/>) or reads a case file, lets the model reason with or without
    /// adapters, and scores only a committed answer. Each reply is classed as correct, wrong,
    /// looped, unfinished or without a number, and the report records how long it thought and
    /// whether the right number appeared in its thinking without being committed. With
    /// <c>--compare</c> it prints the change from an earlier report, case by case.
    /// </summary>
    private static int EntailBench(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("entail bench requires a container directory");
        string tokenizerDir = ResolveTokenizerDir(containerDir, args);
        var tokenizer = Tokenizer(tokenizerDir);
        var stop = ResolveStopTokens(tokenizerDir, tokenizer);
        var template = ChatTemplate.Load(tokenizerDir);
        bool think = !args.Contains("--no-think");
        int maxTokens = IntOption(args, "--max-tokens", think ? 2048 : 512);
        string? reportPath = OptionValue(args, "--report");
        // --score answer (the default) reads a committed number; --score tools reads what a
        // reply did with the tools in its prompt, and the code it produced.
        string score = OptionValue(args, "--score") ?? "answer";
        // --simulate N: answer up to N tool calls with simulated results (ToolSimulator) and let
        // the model go on, so a reply that looks around first can still reach its goal.
        int simulate = IntOption(args, "--simulate", 0);
        int imEnd = tokenizer.AddedTokenId("<|im_end|>") ?? -1;
        if (score is not ("answer" or "tools"))
        {
            throw new CliException($"unknown --score '{score}': use answer or tools");
        }
        // Greedy by default. Qwen's model cards ask for temperature 0.6, top-p 0.95 and
        // top-k 20 when thinking; each case draws from its own seed, so a run repeats.
        float temperature = float.Parse(OptionValue(args, "--temperature") ?? "0", System.Globalization.CultureInfo.InvariantCulture);
        var sampling = new SamplingConfig(
            Temperature: temperature,
            TopK: IntOption(args, "--top-k", 0),
            TopP: float.Parse(OptionValue(args, "--top-p") ?? "0", System.Globalization.CultureInfo.InvariantCulture));
        int sampleSeed = IntOption(args, "--sample-seed", 1);

        var cases = BenchCases(args);
        if (OptionValue(args, "--write-cases") is { } writePath)
        {
            File.WriteAllLines(writePath, cases.Select(c => new JsonObject
            {
                ["category"] = c.Family, ["prompt"] = c.Prompt, ["answer"] = c.Answer.ToString(),
            }.ToJsonString()));
            Console.WriteLine($"wrote {cases.Count} cases to {writePath}");
        }
        cases = cases.Take(IntOption(args, "--limit", cases.Count)).ToList();

        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, OptionValue(args, "--component") ?? "target", store);
        var adapters = OpenAdapters(args, tokenizerDir);
        var session = new SideBySideSession(new DecodeSession(plan, store, null, EntailWorkingSet(args, container)), adapters)
        {
            PriorPrompt = ChatPriorFor(tokenizer),
        };

        // --watch TOKEN: the partner's probability of TOKEN at every step of the reply, before
        // any adapter acts, kept where it reaches 0.001. A threshold rule can be tuned from it.
        int? watched = null;
        if (OptionValue(args, "--watch") is { } watchText)
        {
            var ids = tokenizer.EncodeToIds(watchText);
            watched = ids.Count == 1 ? ids[0] : throw new CliException($"--watch '{watchText}' is {ids.Count} tokens, not one");
        }

        var details = new JsonArray();
        var rows = new List<(string Family, ReplyAnalysis A)>();
        var toolRows = new List<(string Family, ToolReplyAnalysis A)>();
        var clock = Stopwatch.StartNew();
        long allTokens = 0;
        for (int n = 0; n < cases.Count; n++)
        {
            var problem = cases[n];
            bool thinking = problem.Think ?? think;
            string rendered = problem.Raw ? problem.Prompt : EntailPrompt(problem.Prompt, template, chat: true, thinking);
            session.Reset();
            session.Prefill(tokenizer.EncodeToIds(rendered).ToArray());
            var reply = new List<int>();
            var pieces = new List<string>();
            int said = 0, steered = 0;
            bool hitBudget = true;
            var watch = watched is null ? null : new JsonArray();
            var rng = new Random(sampleSeed * 100003 + n);
            int turnStart = 0, simulated = 0;
            for (int i = 0; i < (problem.MaxTokens ?? maxTokens); i++)
            {
                if (watched is { } w && Softmax(session.Logits.Row(0), w) is var pw and >= 0.001)
                {
                    watch!.Add(new JsonArray(i, Math.Round(pw, 4)));
                }
                var (next, source) = temperature > 0 ? session.Sample(sampling, rng) : session.Choose();
                if (stop.Contains(next))
                {
                    // A turn that ends in a call gets its simulated result, and the model goes on.
                    if (simulated < simulate && next == imEnd
                        && ToolSimulator.LastCallIn(string.Concat(pieces.Skip(turnStart))) is { } lastCall)
                    {
                        simulated++;
                        string result = ToolSimulator.ResultTurn(ToolSimulator.Respond(lastCall.Tool, lastCall.Args), thinking);
                        var resultIds = tokenizer.EncodeToIds(result);
                        session.Step(next);
                        foreach (int id in resultIds.Skip(1))
                        {
                            session.Step(id);
                        }
                        reply.AddRange(resultIds);
                        pieces.Add(result);
                        for (int k = 1; k < resultIds.Count; k++)
                        {
                            pieces.Add(string.Empty);
                        }
                        turnStart = pieces.Count;
                        continue;
                    }
                    hitBudget = false;
                    break;
                }
                said += source == EntailTokenSource.Entail ? 1 : 0;
                steered += source == EntailTokenSource.Narrowed ? 1 : 0;
                reply.Add(next);
                pieces.Add(tokenizer.DecodeText(new List<int> { next }, skipSpecialTokens: false));
                session.Step(next);
            }
            allTokens += reply.Count;
            string expected = problem.Answer.ToString();
            string text = string.Concat(pieces);
            var a = ReplyAnalysis.Read(reply, pieces, expected, thinking, hitBudget);
            JsonObject detail;
            if (score == "tools")
            {
                var t = ToolReplyAnalysis.Read(text, problem.Expect ?? "none");
                toolRows.Add((problem.Family, t));
                detail = new JsonObject
                {
                    ["id"] = problem.Id ?? n.ToString(), ["category"] = problem.Family, ["prompt"] = problem.Prompt,
                    ["expect"] = problem.Expect, ["outcome"] = t.Outcome.ToString().ToLowerInvariant(),
                    ["tool"] = t.FirstTool, ["tools"] = new JsonArray(t.Tools.Select(x => (JsonNode?)x).ToArray()),
                    ["simulated"] = simulated, ["opens_with_call"] = t.OpensWithCall, ["call_closed"] = t.CallClosed,
                    ["code_in_call"] = t.CodeInCall, ["written"] = t.Written, ["code_lines"] = t.CodeLines,
                    ["placeholders"] = new JsonArray(t.Placeholders.Select(p => (JsonNode?)p).ToArray()),
                    ["tokens"] = a.Tokens, ["loop_at"] = a.LoopAt, ["hit_budget"] = hitBudget, ["reply"] = text,
                };
                Console.Error.WriteLine($"[{n + 1}/{cases.Count}] {problem.Id ?? problem.Family,-15} {t.Outcome,-10} " +
                    $"{a.Tokens,5} tokens  tool {t.FirstTool ?? "-"}, {t.CodeLines} lines, {t.Placeholders.Count} placeholders");
            }
            else
            {
                rows.Add((problem.Family, a));
                detail = new JsonObject
                {
                    ["id"] = n, ["category"] = problem.Family, ["prompt"] = problem.Prompt, ["answer"] = expected,
                    ["outcome"] = a.Outcome.ToString().ToLowerInvariant(), ["given"] = a.Answer,
                    ["tokens"] = a.Tokens, ["thinking_tokens"] = a.ThinkingTokens,
                    ["loop_at"] = a.LoopAt, ["found_at"] = a.FoundAt,
                    ["reply"] = text,
                };
                Console.Error.WriteLine($"[{n + 1}/{cases.Count}] {problem.Family,-15} {a.Outcome,-10} " +
                    $"{a.Tokens,5} tokens  expected {expected}, gave {a.Answer ?? "-"}");
            }
            if (watch is not null)
            {
                detail["watch"] = watch;
            }
            if (adapters.Count > 0)
            {
                detail["said"] = said;
                detail["steered"] = steered;
            }
            details.Add(detail);
        }

        var summary = score == "tools" ? PrintToolSummary(toolRows) : PrintBenchSummary(rows);
        double seconds = clock.Elapsed.TotalSeconds;
        Console.WriteLine($"{allTokens} tokens in {seconds:F0}s ({allTokens / Math.Max(seconds, 1e-9):F0} tokens/s)");
        var report = new JsonObject
        {
            ["think"] = think, ["max_tokens"] = maxTokens, ["score"] = score,
            ["temperature"] = temperature, ["top_p"] = sampling.TopP, ["top_k"] = sampling.TopK, ["sample_seed"] = sampleSeed,
            ["adapters"] = OptionValue(args, "--adapter") ?? string.Empty,
            ["summary"] = summary, ["cases"] = details,
        };

        if (OptionValue(args, "--compare") is { } comparePath)
        {
            PrintBenchComparison(JsonNode.Parse(File.ReadAllText(comparePath))!.AsObject(), report);
        }
        if (reportPath is not null)
        {
            File.WriteAllText(reportPath, report.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        }
        adapters.ForEach(x => x.Dispose());
        return 0;
    }

    /// <summary>The problems to run: a case file's (<c>--cases</c>, lines of
    /// <c>{"category","prompt","answer"}</c>), or a generated suite.</summary>
    private static List<Problem> BenchCases(string[] args)
    {
        if (OptionValue(args, "--cases") is { } casesPath)
        {
            return File.ReadAllLines(casesPath).Where(l => l.Trim().Length > 0).Select(l =>
            {
                var o = JsonNode.Parse(l)!.AsObject();
                string answer = o["answer"]?.ToString() ?? "0";
                return new Problem(o["category"]?.GetValue<string>() ?? "cases", o["prompt"]!.GetValue<string>(),
                    long.TryParse(answer, out long v) ? v : throw new CliException($"answer '{answer}' is not a whole number"),
                    o["raw"]?.GetValue<bool>() ?? false, o["expect"]?.GetValue<string>(), o["max_tokens"]?.GetValue<int>(),
                    o["think"]?.GetValue<bool>(), o["id"]?.GetValue<string>());
            }).ToList();
        }
        string suite = OptionValue(args, "--suite") ?? "reasoning";
        var families = OptionValue(args, "--families")?.Split(',', StringSplitOptions.RemoveEmptyEntries);
        try
        {
            var problems = ProblemSuite.Generate(IntOption(args, "--seed", 7), IntOption(args, "--per", 10), families, suite);
            // The hard suite always asks for an answer line. The first suite's prompts stay as its
            // published results had them unless it is asked for.
            return suite == "hard" || args.Contains("--answer-line")
                ? problems.Select(p => p with { Prompt = p.Prompt + ProblemSuite.AnswerLine }).ToList()
                : problems;
        }
        catch (ArgumentException e)
        {
            throw new CliException(e.Message);
        }
    }

    private static readonly ReplyOutcome[] Outcomes = Enum.GetValues<ReplyOutcome>();

    private static JsonObject PrintToolSummary(List<(string Family, ToolReplyAnalysis A)> rows)
    {
        Console.WriteLine($"{"category",-10} {"right",9} {"written",8} {"talked",7} {"wrong",6} {"false",6} {"opens",6} {"code",6} {"with marks",11} {"marks",6}");
        var summary = new JsonObject();
        foreach (var group in rows.GroupBy(r => r.Family).Append(rows.GroupBy(_ => "overall").First()))
        {
            var g = group.Select(r => r.A).ToList();
            int Count(ToolOutcome o) => g.Count(a => a.Outcome == o);
            int right = Count(ToolOutcome.Called) + Count(ToolOutcome.Quiet);
            int withCode = g.Count(a => a.CodeLines > 0);
            int marked = g.Count(a => a.Placeholders.Count > 0);
            Console.WriteLine($"{group.Key,-10} {right,3}/{g.Count,-3} {100.0 * right / g.Count,3:F0}% {g.Count(a => a.Written),8} {Count(ToolOutcome.Talked),7} " +
                $"{Count(ToolOutcome.WrongTool),6} {Count(ToolOutcome.FalseCall),6} {g.Count(a => a.OpensWithCall),6} {withCode,6} " +
                $"{marked,11} {g.Sum(a => a.Placeholders.Count),6}");
            var s = new JsonObject
            {
                ["count"] = g.Count, ["right"] = right, ["written"] = g.Count(a => a.Written), ["opens_with_call"] = g.Count(a => a.OpensWithCall),
                ["with_code"] = withCode, ["with_placeholders"] = marked, ["placeholders"] = g.Sum(a => a.Placeholders.Count),
            };
            foreach (var o in Enum.GetValues<ToolOutcome>())
            {
                s[o.ToString().ToLowerInvariant()] = Count(o);
            }
            summary[group.Key] = s;
        }
        return summary;
    }

    private static JsonObject PrintBenchSummary(List<(string Family, ReplyAnalysis A)> rows)
    {
        Console.WriteLine($"{"family",-15} {"correct",9} {"wrong",6} {"none",5} {"looped",7} {"unfin",6} {"uncommitted",12} {"think",6}");
        var summary = new JsonObject();
        foreach (var group in rows.GroupBy(r => r.Family).Append(rows.GroupBy(_ => "overall").First()))
        {
            var g = group.Select(r => r.A).ToList();
            int Count(ReplyOutcome o) => g.Count(a => a.Outcome == o);
            // Right number written while thinking, but not committed as the answer.
            int uncommitted = g.Count(a => a.Outcome != ReplyOutcome.Correct && a.FoundWhileThinking);
            double meanThink = g.Average(a => (double)a.ThinkingTokens);
            int correct = Count(ReplyOutcome.Correct);
            Console.WriteLine($"{group.Key,-15} {correct,3}/{g.Count,-3} {100.0 * correct / g.Count,3:F0}% " +
                $"{Count(ReplyOutcome.Wrong),3} {Count(ReplyOutcome.NoNumber),5} {Count(ReplyOutcome.Looped),7} " +
                $"{Count(ReplyOutcome.Unfinished),6} {uncommitted,12} {meanThink,6:F0}");
            var s = new JsonObject { ["count"] = g.Count, ["uncommitted"] = uncommitted, ["mean_thinking"] = Math.Round(meanThink, 1) };
            foreach (var o in Outcomes)
            {
                s[o.ToString().ToLowerInvariant()] = Count(o);
            }
            summary[group.Key] = s;
        }
        return summary;
    }

    /// <summary>Prints, per family, the correct count before and after, and the cases that
    /// were won and lost, matched by prompt.</summary>
    private static void PrintBenchComparison(JsonObject before, JsonObject after)
    {
        var old = before["cases"]!.AsArray().Select(c => c!.AsObject())
            .GroupBy(c => c["prompt"]!.GetValue<string>()).ToDictionary(g => g.Key, g => g.First());
        int won = 0, lost = 0, matched = 0;
        var perFamily = new SortedDictionary<string, (int Before, int After, int Count)>(StringComparer.Ordinal);
        foreach (var c in after["cases"]!.AsArray().Select(c => c!.AsObject()))
        {
            if (!old.TryGetValue(c["prompt"]!.GetValue<string>(), out var o))
            {
                continue;
            }
            matched++;
            bool was = o["outcome"]!.GetValue<string>() == "correct", now = c["outcome"]!.GetValue<string>() == "correct";
            won += !was && now ? 1 : 0;
            lost += was && !now ? 1 : 0;
            string family = c["category"]!.GetValue<string>();
            var (b, a, n) = perFamily.GetValueOrDefault(family);
            perFamily[family] = (b + (was ? 1 : 0), a + (now ? 1 : 0), n + 1);
        }
        Console.WriteLine();
        Console.WriteLine($"compared with the earlier report on {matched} matching cases:");
        foreach (var (family, (b, a, n)) in perFamily)
        {
            Console.WriteLine($"  {family,-15} {b,3} -> {a,3} of {n} ({a - b:+0;-0;0})");
        }
        Console.WriteLine($"  won {won}, lost {lost}, net {won - lost:+0;-0;0}");
    }
}
