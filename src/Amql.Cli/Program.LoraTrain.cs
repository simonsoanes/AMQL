using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Hf;
using Amql.Inference;
using Amql.Merge;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Cli;

// ── lora-train: supervised LoRA on FFN down-projections, monitored ─────────

internal static partial class Program
{
    private static int LoraTrain(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("lora-train requires a container directory");
        var problems = new List<string>();
        string? dataPath = OptionValue(args, "--data");
        if (dataPath is null)
            problems.Add("missing '--data <train.jsonl>' — one {\"id\",\"prompt\",\"completion\"} object per line, prompt already rendered");
        string? outDir = OptionValue(args, "--out");
        if (outDir is null)
            problems.Add("missing '--out <lora-dir>'");
        if (problems.Count > 0)
            throw new CliException("lora-train:\n" + string.Join("\n", problems.Select(p => "  - " + p)));

        string component = OptionValue(args, "--component") ?? "target";
        string tokenizerDir = ResolveTokenizerDir(containerDir, args);
        var tokenizer = Tokenizer(tokenizerDir);
        var train = ReadLoraExamples(dataPath!, tokenizer);
        var eval = OptionValue(args, "--eval") is { } evalPath ? ReadLoraExamples(evalPath, tokenizer) : new List<LoraExample>();
        Directory.CreateDirectory(outDir!);

        var options = new LoraTrainOptions
        {
            Layers = OptionValue(args, "--layers") is { } layers ? ParseIntList(layers, Array.Empty<int>()) : null,
            Rank = IntOption(args, "--rank", 8),
            Alpha = DoubleOption(args, "--alpha", 16),
            LearningRate = DoubleOption(args, "--lr", 2e-3),
            Steps = IntOption(args, "--steps", 60),
            EvalEvery = IntOption(args, "--eval-every", 10),
            ClipNorm = DoubleOption(args, "--clip", 1.0),
            WeightDecay = DoubleOption(args, "--weight-decay", 0),
            TargetLoss = DoubleOption(args, "--target-loss", 0.02),
            GradChecks = IntOption(args, "--grad-checks", 2),
            Seed = IntOption(args, "--seed", 1234),
            AnchorTokens = IntOption(args, "--anchor-tokens", 40),
            AnchorWeight = DoubleOption(args, "--anchor-weight", 1.0),
            StopTokens = ResolveStopTokens(tokenizerDir, tokenizer),
        };

        Console.WriteLine($"container: {containerDir}   tokenizer: {tokenizerDir}");
        Console.WriteLine($"train:     {train.Count} examples, {train.Sum(e => e.Completion.Length)} supervised tokens");
        Console.WriteLine($"eval:      {eval.Count} examples");
        Console.WriteLine($"lora:      rank {options.Rank}, alpha {options.Alpha}, lr {options.LearningRate}, up to {options.Steps} steps");

        string logPath = Path.Combine(outDir!, "train.log.jsonl");
        using var logWriter = new StreamWriter(logPath) { AutoFlush = true };
        var jsonOptions = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase, NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals };
        void Log(object entry)
        {
            string line = JsonSerializer.Serialize(entry, jsonOptions);
            logWriter.WriteLine(line);
            var node = JsonNode.Parse(line)!.AsObject();
            switch ((string?)node["kind"])
            {
                case "step":
                    Console.WriteLine($"step {node["step"],4}  loss {(double)node["loss"]!,9:F5}  lr {(double)node["lr"]!:0.####e+0}  " +
                                      $"|g| {(double)node["gradNorm"]!:F4}  {(double)node["seconds"]!:F2}s");
                    break;
                case "eval":
                    Console.WriteLine($"  eval @{node["step"]} {node["set"],-5}: {node["exact"]}/{node["of"]} exact, loss {(double)node["loss"]!:F4}  " +
                                      string.Join(" ", node["cases"]!.AsArray().Select(c =>
                                          $"{c!["id"]}:{((bool)c["exact"]! ? "ok" : "@" + c["firstWrong"])}")));
                    break;
                default:
                    Console.WriteLine($"  {line}");
                    break;
            }
        }

        using var container = Vindex3Container.Open(containerDir);
        var anchors = OptionValue(args, "--anchors") is { } anchorPath
            ? File.ReadAllLines(anchorPath).Where(l => l.Trim().Length > 0).Select((l, i) =>
            {
                var o = JsonNode.Parse(l)!.AsObject();
                return new LoraExample((string?)o["id"] ?? $"a{i}", tokenizer.EncodeToIds((string)o["prompt"]!).ToArray(), Array.Empty<int>());
            }).ToList()
            : null;
        var report = LoraTrainer.Train(container, component, train, eval, options, Log, anchors);

        SaveTrainedLora(outDir!, report.Layers, container.Index.Model);
        var patchEntries = report.Layers.Select(l => new WeightPatchEntry(l.Down.ObjectId, l.Down.TensorName,
            new long[] { l.Hidden, l.Inter }, l.Delta())).ToList();
        if (report.StepsRun > 0)
            WeightPatch.Save(Path.Combine(outDir!, "patch.safetensors"), patchEntries, container.Index.Model);

        var summary = new JsonObject
        {
            ["layers"] = new JsonArray(report.Layers.Select(l => (JsonNode)l.Layer).ToArray()),
            ["rank"] = options.Rank,
            ["steps"] = report.StepsRun,
            ["initialLoss"] = double.IsFinite(report.InitialLoss) ? report.InitialLoss : null,
            ["finalLoss"] = report.FinalLoss,
        };
        void Cases(string name, IReadOnlyList<LoraEvalCase> before, IReadOnlyList<LoraEvalCase> after, List<LoraExample> examples)
        {
            var array = new JsonArray();
            for (int i = 0; i < before.Count && i < examples.Count; i++)
            {
                array.Add(new JsonObject
                {
                    ["id"] = before[i].Id,
                    ["lossBefore"] = before[i].Loss,
                    ["lossAfter"] = after[i].Loss,
                    ["exactBefore"] = before[i].Exact,
                    ["exactAfter"] = after[i].Exact,
                    ["expected"] = tokenizer.DecodeText(examples[i].Completion),
                    ["before"] = tokenizer.DecodeText(before[i].Output),
                    ["after"] = tokenizer.DecodeText(after[i].Output),
                });
            }
            summary[name] = array;
        }
        Cases("train", report.TrainBefore, report.TrainAfter, train);
        Cases("eval", report.EvalBefore, report.EvalAfter, eval);
        File.WriteAllText(Path.Combine(outDir!, "report.json"), summary.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));

        Console.WriteLine();
        Console.WriteLine($"loss {report.InitialLoss:F4} → {report.FinalLoss:F4} in {report.StepsRun} steps");
        for (int i = 0; i < eval.Count; i++)
        {
            Console.WriteLine($"{eval[i].Id,-12} {(report.EvalBefore[i].Exact ? "ok " : "-- ")}→ {(report.EvalAfter[i].Exact ? "ok " : "-- ")} " +
                              OneLine(tokenizer.DecodeText(report.EvalAfter[i].Output)));
        }
        Console.WriteLine($"adapter:   {outDir}  (adapter_model.safetensors, adapter_config.json, patch.safetensors, train.log.jsonl, report.json)");
        return 0;
    }

    private static List<LoraExample> ReadLoraExamples(string path, HfTokenizer tokenizer) =>
        File.ReadAllLines(path)
            .Where(l => l.Trim().Length > 0)
            .Select((l, i) =>
            {
                var o = JsonNode.Parse(l)!.AsObject();
                string id = (string?)o["id"] ?? $"case{i}";
                var prompt = tokenizer.EncodeToIds((string)o["prompt"]!).ToArray();
                var completion = tokenizer.EncodeToIds((string)o["completion"]!).ToArray();
                if (prompt.Length == 0 || completion.Length == 0)
                    throw new CliException($"{path}: '{id}' has an empty prompt or completion");
                return new LoraExample(id, prompt, completion);
            })
            .ToList();

    /// <summary>Writes the trained factors in the amql-lora-v1 layout that
    /// <c>save-lora</c> also writes, with each target's layer recorded.</summary>
    private static void SaveTrainedLora(string outDir, IReadOnlyList<LoraLayer> layers, string? model)
    {
        var tensors = new List<TensorPayload>();
        var targets = new JsonArray();
        for (int i = 0; i < layers.Count; i++)
        {
            var l = layers[i];
            tensors.Add(new TensorPayload { Name = $"lora_A.{i}", Dtype = Dtype.F32, Shape = new long[] { l.Rank, l.Inter }, Data = Bytes(l.A) });
            tensors.Add(new TensorPayload { Name = $"lora_B.{i}", Dtype = Dtype.F32, Shape = new long[] { l.Hidden, l.Rank }, Data = Bytes(l.B) });
            targets.Add(new JsonObject
            {
                ["object_id"] = l.Down.ObjectId,
                ["tensor_name"] = l.Down.TensorName,
                ["layer"] = l.Layer,
                ["shape"] = new JsonArray(l.Hidden, l.Inter),
                ["rank"] = l.Rank,
                ["scale"] = l.Scale,
                ["lora_a"] = $"lora_A.{i}",
                ["lora_b"] = $"lora_B.{i}",
            });
        }
        SafetensorsWriter.Write(Path.Combine(outDir, "adapter_model.safetensors"), tensors,
            new Dictionary<string, string> { ["format"] = LoraWriter.LoraFormat, ["model"] = model ?? string.Empty });
        var config = new JsonObject
        {
            ["format"] = LoraWriter.LoraFormat,
            ["model"] = model,
            ["rank"] = layers.Count == 0 ? 0 : layers[0].Rank,
            ["scale"] = layers.Count == 0 ? 0 : layers[0].Scale,
            ["trained_by"] = "amql-cli lora-train",
            ["targets"] = targets,
        };
        File.WriteAllText(Path.Combine(outDir, "adapter_config.json"),
            config.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));

        static byte[] Bytes(float[] values)
        {
            var bytes = new byte[values.Length * 4];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            return bytes;
        }
    }
}
