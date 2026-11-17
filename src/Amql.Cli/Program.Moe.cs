using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Hf;
using Amql.Merge;
using Amql.Vindex3;

namespace Amql.Cli;

// ── moe-eval / moe-router / moe-merge: branch-adapt-route over topic LoRAs ──

internal static partial class Program
{
    private sealed record TopicSet(List<LoraExample> Examples, List<string> Topics);

    private static TopicSet ReadTopicExamples(string path, HfTokenizer tokenizer)
    {
        var examples = ReadLoraExamples(path, tokenizer);
        var topics = File.ReadAllLines(path).Where(l => l.Trim().Length > 0)
            .Select(l => (string?)JsonNode.Parse(l)!["topic"] ?? "").ToList();
        return new TopicSet(examples, topics);
    }

    /// <summary>--experts name=dir,name=dir,…, in router order.</summary>
    private static List<(string Name, string Dir)> ExpertSpecs(string[] args, string command)
    {
        var raw = OptionValue(args, "--experts")
            ?? throw new CliException($"{command}: missing '--experts name=<lora-dir>,name=<lora-dir>,…'");
        return raw.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(s =>
        {
            int eq = s.IndexOf('=');
            if (eq <= 0)
                throw new CliException($"{command}: expert '{s}' is not name=<lora-dir>");
            return (s[..eq], s[(eq + 1)..]);
        }).ToList();
    }

    private static MoeBench OpenBench(string containerDir, string[] args, string command, out Vindex3Container container)
    {
        // moe-eval with no experts scores the container as it is (a merged MoE, say).
        var specs = command == "moe-eval" && OptionValue(args, "--experts") is null
            ? new List<(string Name, string Dir)>()
            : ExpertSpecs(args, command);
        container = Vindex3Container.Open(containerDir);
        return new MoeBench(container, OptionValue(args, "--component") ?? "target",
            plan => specs.Select(s => LoraExpert.Load(s.Dir, s.Name, plan)).ToList());
    }

    private static void JsonLog(StreamWriter? writer, object entry)
    {
        var line = JsonSerializer.Serialize(entry);
        writer?.WriteLine(line);
        Console.WriteLine("  " + line);
    }

    /// <summary>
    /// <c>moe-eval</c>: mean completion loss per topic under each mixture of the
    /// experts: the base, each expert alone, their average (weight averaging), their
    /// sum (task arithmetic), the oracle (each example's own topic expert), and any
    /// routers given, each at the top-k values given.
    /// </summary>
    private static int MoeEval(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("moe-eval requires a container directory");
        var dataPath = OptionValue(args, "--data") ?? throw new CliException("moe-eval: missing '--data <eval.jsonl>' (with a \"topic\" per line)");
        var tokenizer = Tokenizer(ResolveTokenizerDir(containerDir, args));
        var set = ReadTopicExamples(dataPath, tokenizer);
        using var bench = OpenBench(containerDir, args, "moe-eval", out var container);
        using var _ = container;
        int n = bench.Names.Count;

        var mixes = new List<(string Name, Func<int, Mixing> For)>();
        var wanted = (OptionValue(args, "--mix") ?? "base,experts,average,sum,oracle").Split(',', StringSplitOptions.TrimEntries);
        foreach (var w in wanted)
        {
            switch (w)
            {
                case "base":
                    mixes.Add(("base", _ => Mixing.Of()));
                    break;
                case "experts":
                    for (int e = 1; e < n; e++)
                    {
                        var one = new double[n];
                        one[e] = 1;
                        mixes.Add(($"only:{bench.Names[e]}", _ => Mixing.Of(one)));
                    }
                    break;
                case "average":
                    var avg = Enumerable.Range(0, n).Select(e => e == 0 ? 0.0 : 1.0 / (n - 1)).ToArray();
                    mixes.Add(("average", _ => Mixing.Of(avg)));
                    break;
                case "sum":
                    var sum = Enumerable.Range(0, n).Select(e => e == 0 ? 0.0 : 1.0).ToArray();
                    mixes.Add(("sum", _ => Mixing.Of(sum)));
                    break;
                case "oracle":
                    mixes.Add(("oracle", i =>
                    {
                        var one = new double[n];
                        one[Math.Max(0, bench.Names.ToList().IndexOf(set.Topics[i]))] = 1;
                        return Mixing.Of(one);
                    }));
                    break;
                case "":
                    break;
                default:
                    throw new CliException($"moe-eval: unknown mix '{w}' (base, experts, average, sum, oracle)");
            }
        }
        var topKs = ParseIntList(OptionValue(args, "--top-k"), new[] { 0 });
        foreach (var spec in (OptionValue(args, "--router") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int eq = spec.IndexOf('=');
            string name = eq > 0 ? spec[..eq] : Path.GetFileNameWithoutExtension(spec);
            var router = MoeRouter.Load(eq > 0 ? spec[(eq + 1)..] : spec);
            if (!router.Experts.SequenceEqual(bench.Names))
                throw new CliException($"moe-eval: router '{name}' routes {string.Join(",", router.Experts)}, the bench has {string.Join(",", bench.Names)}");
            foreach (var k in topKs)
                mixes.Add(($"{name}@{(k <= 0 || k >= n ? "all" : k.ToString())}", _ => Mixing.Routed(router, k)));
        }

        Console.WriteLine($"bench:  {string.Join(", ", bench.Names)} over layers {string.Join(",", bench.Layers)}");
        Console.WriteLine($"data:   {set.Examples.Count} examples, topics {string.Join(", ", set.Topics.Distinct())}");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var prefixes = set.Examples.Select(bench.Prefix).ToList();
        Console.WriteLine($"prefix: {sw.Elapsed.TotalSeconds:F1}s");

        var topicNames = set.Topics.Distinct().ToList();
        var report = new JsonObject();
        Console.WriteLine();
        Console.WriteLine($"{"mix",-22}" + string.Join("", topicNames.Select(t => $"{t,12}")) + $"{"mean",10}");
        foreach (var (name, forExample) in mixes)
        {
            var losses = new double[set.Examples.Count];
            for (int i = 0; i < losses.Length; i++)
                losses[i] = bench.Loss(set.Examples[i], prefixes[i], forExample(i));
            var byTopic = topicNames.ToDictionary(t => t,
                t => Enumerable.Range(0, losses.Length).Where(i => set.Topics[i] == t).Average(i => losses[i]));
            double mean = byTopic.Values.Average();
            Console.WriteLine($"{name,-22}" + string.Join("", topicNames.Select(t => $"{byTopic[t],12:F4}")) + $"{mean,10:F4}");
            report[name] = new JsonObject
            {
                ["topics"] = new JsonObject(byTopic.Select(kv => KeyValuePair.Create(kv.Key, (JsonNode?)kv.Value))),
                ["mean"] = mean,
                ["examples"] = new JsonArray(losses.Select((l, i) => (JsonNode)new JsonObject
                {
                    ["id"] = set.Examples[i].Id, ["topic"] = set.Topics[i], ["loss"] = l,
                }).ToArray()),
            };
        }
        Console.WriteLine($"\n{sw.Elapsed.TotalSeconds:F0}s");
        if (OptionValue(args, "--report") is { } reportPath)
            File.WriteAllText(reportPath, report.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        return 0;
    }

    /// <summary><c>moe-router</c>: trains a per-layer linear router over anchor +
    /// experts, everything else frozen: <c>--method loss</c> (BAR: the LM loss) or
    /// <c>--method label</c> (the topic labels).</summary>
    private static int MoeRouterCmd(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("moe-router requires a container directory");
        var dataPath = OptionValue(args, "--data") ?? throw new CliException("moe-router: missing '--data <route.jsonl>' (with a \"topic\" per line)");
        var outPath = OptionValue(args, "--out") ?? throw new CliException("moe-router: missing '--out <router.safetensors>'");
        string method = OptionValue(args, "--method") ?? "loss";
        var tokenizer = Tokenizer(ResolveTokenizerDir(containerDir, args));
        var set = ReadTopicExamples(dataPath, tokenizer);
        using var bench = OpenBench(containerDir, args, "moe-router", out var container);
        using var _ = container;
        var options = new RouterTrainOptions
        {
            Steps = IntOption(args, "--steps", method == "label" ? 200 : 30),
            LearningRate = DoubleOption(args, "--lr", 1e-2),
            WeightDecay = DoubleOption(args, "--weight-decay", 0),
            ClipNorm = DoubleOption(args, "--clip", 1.0),
            RowStride = IntOption(args, "--row-stride", 2),
            HoldPrompt = (OptionValue(args, "--hold") ?? "token") == "prompt",
            Init = OptionValue(args, "--init") is { } initPath ? MoeRouter.Load(initPath) : null,
        };
        Console.WriteLine($"bench:  {string.Join(", ", bench.Names)} over layers {string.Join(",", bench.Layers)}");
        Console.WriteLine($"router: {method}, {(options.HoldPrompt ? "held from the prompt" : "per token")}, {options.Steps} steps, lr {options.LearningRate}");
        var prefixes = set.Examples.Select(bench.Prefix).ToList();
        using var log = OptionValue(args, "--log") is { } logPath ? new StreamWriter(logPath) { AutoFlush = true } : null;
        var router = method switch
        {
            "loss" => bench.TrainByLoss(set.Examples, prefixes, options, e => JsonLog(log, e)),
            "label" => bench.TrainByLabel(set.Examples, prefixes, set.Topics, options, e => JsonLog(log, e)),
            _ => throw new CliException($"moe-router: unknown method '{method}' (loss, label)"),
        };
        router.Save(outPath);
        Console.WriteLine($"router: {outPath}");
        return 0;
    }

    /// <summary><c>moe-merge</c>: writes anchor + experts + router as one routed container.</summary>
    private static int MoeMerge(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("moe-merge requires a container directory");
        var outDir = OptionValue(args, "--out") ?? throw new CliException("moe-merge: missing '--out <dir>'");
        var routerPath = OptionValue(args, "--router") ?? throw new CliException("moe-merge: missing '--router <router.safetensors>'");
        var specs = ExpertSpecs(args, "moe-merge");
        List<LoraExpert> experts;
        using (var container = Vindex3Container.Open(containerDir))
        using (var store = container.CreateOperandStore())
        {
            var plan = Amql.Inference.Planner.Plan(container, OptionValue(args, "--component") ?? "target", store);
            experts = specs.Select(s => LoraExpert.Load(s.Dir, s.Name, plan)).ToList();
        }
        var router = MoeRouter.Load(routerPath);
        string model = MoeBench.WriteMoe(containerDir, outDir, experts, router, IntOption(args, "--top-k", 0));
        Console.WriteLine($"wrote {outDir}: {model}, {router.Experts.Count} experts on layers {string.Join(",", router.Layers)}");
        return 0;
    }
}
