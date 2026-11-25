using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Inference;
using Amql.Inference.Reasoning;
using Amql.Vindex3;

namespace Amql.Cli;

// ── entail settle: has the model made up its mind? ─────────────────────────
//
// A small reasoning model writes its answer long before it stops thinking, and
// gives the closing token no probability while it doubts. Nothing in its next
// token says it is done; but what it would answer can simply be read: every so
// often, the block is closed on the sequence, the answer's first words are taken
// greedily, and the sequence is rewound to where it was, which costs a handful
// of tokens. Probes that agree mean the thinking no longer moves the answer.
//
// The problems run in a batch (greedy), each with its own sequence.

internal static partial class Program
{
    private sealed class SettleRun(Problem problem, GpuSequence seq, float[] logits, Random rng)
    {
        public readonly Random Rng = rng;   // each case draws from its own seed, whatever shares its batch
        public readonly Problem Problem = problem;
        public readonly GpuSequence Seq = seq;
        public float[] Logits = logits;
        public readonly List<int> Reply = new();
        public readonly List<(int At, string Answer)> Probes = new();
        public bool Thinking = true;
        public int? ClosedAt;       // where thinking was ended for it
        public string ClosedBy = "";
        public int ProbeTokens;
        public bool HitBudget;
    }

    private static int EntailSettle(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("entail settle requires a container directory");
        string tokenizerDir = ResolveTokenizerDir(containerDir, args);
        var tokenizer = Tokenizer(tokenizerDir);
        var stop = ResolveStopTokens(tokenizerDir, tokenizer);
        string? template = ChatTemplate.Load(tokenizerDir);
        int every = IntOption(args, "--every", 64);
        int agree = IntOption(args, "--agree", 0);          // 0: probe and record, never close
        int least = IntOption(args, "--least", 0);          // thinking tokens before the first probe counts
        int budget = IntOption(args, "--budget", 0);        // thinking tokens after which it is closed whatever it says (0: never)
        int maxTokens = IntOption(args, "--max-tokens", 4096);
        int lanes = Math.Clamp(IntOption(args, "--batch", GpuEngine.MaxBatch), 1, GpuEngine.MaxBatch);
        // Greedy by default; each case samples from its own seed.
        var sampling = new SamplingConfig(Temperature: FloatOption(args, "--temperature", 0f), TopK: IntOption(args, "--top-k", 0), TopP: FloatOption(args, "--top-p", 0f));
        int sampleSeed = IntOption(args, "--sample-seed", 1);
        var cases = BenchCases(args);
        cases = cases.Take(IntOption(args, "--limit", cases.Count)).ToList();

        int closeToken = tokenizer.AddedTokenId("</think>") ?? throw new CliException("entail settle needs a model that thinks in <think> blocks");
        int[] probe = tokenizer.EncodeToIds("\n</think>\n\nAnswer:").ToArray();
        int[] closing = tokenizer.EncodeToIds("\n</think>\n\n").ToArray();

        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, OptionValue(args, "--component") ?? "target", store);
        using var session = new DecodeSession(plan, store);
        session.Prefill(new[] { closing[0] });
        var engine = session.DeviceEngine() ?? throw new CliException($"entail settle runs on the GPU engine ({session.GpuRefusal})");

        var clock = Stopwatch.StartNew();
        var waiting = new Queue<(int Index, Problem Problem)>(cases.Select((c, i) => (i, c)));
        var running = new List<(int Index, SettleRun Run)>();
        var finished = new SortedDictionary<int, SettleRun>();
        long stepped = 0;
        while (waiting.Count > 0 || running.Count > 0)
        {
            while (running.Count < lanes && waiting.TryDequeue(out var next))
            {
                string rendered = EntailPrompt(next.Problem.Prompt, template, chat: true, think: true);
                var seq = engine.NewSequence();
                running.Add((next.Index, new SettleRun(next.Problem, seq, engine.Prefill(seq, tokenizer.EncodeToIds(rendered).ToArray()), new Random(sampleSeed * 100003 + next.Index))));
            }

            // The next token of each, greedily; a reply that ends leaves the batch.
            for (int i = running.Count - 1; i >= 0; i--)
            {
                var run = running[i].Run;
                int token = sampling.Temperature > 0 ? Sampler.Sample(run.Logits, sampling, run.Rng) : Sampler.ArgMax(run.Logits);
                bool over = run.Reply.Count >= maxTokens;
                if (stop.Contains(token) || over)
                {
                    run.HitBudget = over && !stop.Contains(token);
                    finished[running[i].Index] = run;
                    running.RemoveAt(i);
                    engine.Release(run.Seq);
                    Console.Error.WriteLine($"[{finished.Count}/{cases.Count}] {run.Problem.Family,-12} {run.Reply.Count,5} tokens, {run.Probes.Count} probes{(run.ClosedAt is { } at ? $", closed at {at} ({run.ClosedBy})" : "")}");
                    continue;
                }
                run.Reply.Add(token);
                if (token == closeToken)
                {
                    run.Thinking = false;
                }
            }
            if (running.Count == 0)
            {
                continue;
            }
            var rows = engine.Step(running.Select(r => r.Run.Seq).ToList(), running.Select(r => r.Run.Reply[^1]).ToArray());
            stepped += running.Count;
            for (int i = 0; i < running.Count; i++)
            {
                running[i].Run.Logits = rows[i];
            }

            foreach (var (_, run) in running)
            {
                if (!run.Thinking)
                {
                    continue;
                }
                int at = run.Reply.Count;
                if (at % every == 0)
                {
                    engine.Mark(run.Seq);
                    var logits = engine.Prefill(run.Seq, probe);
                    var said = new List<int>();
                    for (int t = 0; t < 16; t++)
                    {
                        int token = Sampler.ArgMax(logits);
                        string piece = tokenizer.DecodeText(new[] { token }, skipSpecialTokens: false);
                        if (stop.Contains(token) || (said.Count > 0 && piece.Contains('\n')))
                        {
                            break;
                        }
                        said.Add(token);
                        logits = engine.Prefill(run.Seq, new[] { token });
                    }
                    run.ProbeTokens += probe.Length + said.Count;
                    engine.Rewind(run.Seq);
                    run.Probes.Add((at, SettledAnswer(tokenizer.DecodeText(said.ToArray()))));
                }
                bool settled = agree > 0 && run.Probes.Count >= agree && run.Probes[^1].At == at && run.Probes[^1].Answer.Length > 0 &&
                               run.Probes.TakeLast(agree).All(p => p.At >= least && p.Answer == run.Probes[^1].Answer);
                bool spent = budget > 0 && at >= budget;
                if (settled || spent)
                {
                    run.ClosedAt = at;
                    run.ClosedBy = settled ? "settled" : "budget";
                    run.Thinking = false;
                    run.Logits = engine.Prefill(run.Seq, closing);
                    run.Reply.AddRange(closing);
                }
            }
        }

        double seconds = clock.Elapsed.TotalSeconds;
        var analysed = new List<(string Family, ReplyAnalysis A)>();
        var details = new JsonArray();
        foreach (var (index, run) in finished)
        {
            var pieces = run.Reply.Select(t => tokenizer.DecodeText(new[] { t }, skipSpecialTokens: false)).ToList();
            var a = ReplyAnalysis.Read(run.Reply, pieces, run.Problem.Answer.ToString(), true, run.HitBudget);
            analysed.Add((run.Problem.Family, a));
            details.Add(new JsonObject
            {
                ["id"] = index, ["category"] = run.Problem.Family, ["prompt"] = run.Problem.Prompt, ["answer"] = run.Problem.Answer.ToString(),
                ["outcome"] = a.Outcome.ToString().ToLowerInvariant(), ["given"] = a.Answer, ["tokens"] = a.Tokens,
                ["thinking_tokens"] = a.ThinkingTokens, ["loop_at"] = a.LoopAt, ["found_at"] = a.FoundAt,
                ["closed_at"] = run.ClosedAt, ["closed_by"] = run.ClosedBy, ["probe_tokens"] = run.ProbeTokens,
                ["probes"] = new JsonArray(run.Probes.Select(p => (JsonNode)new JsonArray(p.At, p.Answer)).ToArray()),
                ["reply"] = string.Concat(pieces),
            });
        }
        var summary = PrintBenchSummary(analysed);
        long tokens = finished.Values.Sum(r => (long)r.Reply.Count), probeTokens = finished.Values.Sum(r => (long)r.ProbeTokens);
        Console.WriteLine($"{tokens} tokens ({tokens / (double)Math.Max(1, finished.Count):F0} a reply) and {probeTokens} probe tokens in {seconds:F0}s; " +
            $"settled {finished.Values.Count(r => r.ClosedBy == "settled")}, cut at the budget {finished.Values.Count(r => r.ClosedBy == "budget")}");
        var report = new JsonObject
        {
            ["think"] = true, ["max_tokens"] = maxTokens, ["every"] = every, ["agree"] = agree, ["least"] = least, ["budget"] = budget,
            ["temperature"] = sampling.Temperature, ["top_p"] = sampling.TopP, ["top_k"] = sampling.TopK, ["sample_seed"] = sampleSeed, ["seconds"] = seconds, ["tokens"] = tokens, ["probe_tokens"] = probeTokens, ["summary"] = summary, ["cases"] = details,
        };
        if (OptionValue(args, "--compare") is { } comparePath)
        {
            PrintBenchComparison(JsonNode.Parse(File.ReadAllText(comparePath))!.AsObject(), report);
        }
        if (OptionValue(args, "--report") is { } reportPath)
        {
            File.WriteAllText(reportPath, report.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        }
        return ExitOk;
    }

    /// <summary>A probe's answer in a form two probes can be compared by: the number in it, or its words.</summary>
    private static string SettledAnswer(string text) =>
        ReplyAnalysis.Extract(text) ?? string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)).Trim('*', '.', '$', ' ').ToLowerInvariant();
}
