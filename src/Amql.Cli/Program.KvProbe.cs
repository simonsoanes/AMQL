using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Cli;

// ── kv-probe: what a conversation's later questions attend to ─────────────
//
// A research dump for choosing which KV rows to keep. The conversation is
// prefilled once (the shared prefix, up to the questions' turn) on the GPU
// engine with f32 caches; then, for each question, the engine rewinds to that
// prefix, reads the question and writes its own greedy answer. Every softmax
// layer's real query rows are recorded as they run (the prefix's last
// --window rows, and every row of each question and answer), with the prefix's
// keys and values and each question's own, so attention over the prefix can be
// computed exactly offline: how much of it any choice of kept rows captures.

internal static partial class Program
{
    private static int KvProbe(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("kv-probe requires a container directory");
        string convPath = OptionValue(args, "--conversation") ?? throw new CliException(
            "kv-probe requires '--conversation <json>' ({\"messages\": [...], \"questions\": [...]})");
        string outDir = OptionValue(args, "--out") ?? throw new CliException("kv-probe requires '--out <dir>'");
        int answerTokens = IntOption(args, "--answer-tokens", 48);
        int window = IntOption(args, "--window", 32);
        Directory.CreateDirectory(outDir);

        var conv = JsonNode.Parse(File.ReadAllText(convPath))!.AsObject();
        var messages = conv["messages"]!.AsArray().Select(m => new ChatMessage((string)m!["role"]!, (string)m["content"]!)).ToList();
        var questions = conv["questions"]!.AsArray().Select(q => (string)q!).ToList();
        var tokenizer = Tokenizer(containerDir);
        var stop = ResolveStopTokens(containerDir, tokenizer);
        var template = ChatTemplate.Load(containerDir) ?? throw new CliException("the container has no chat template");
        int turn = tokenizer.AddedTokenId("<|im_start|>") ?? throw new CliException("kv-probe needs <|im_start|> turns");
        int[] PromptFor(string q) => tokenizer.EncodeToIds(template.Apply(
            messages.Append(new ChatMessage("user", q)).ToList(), enableThinking: false)).ToArray();
        var prompts = questions.Select(PromptFor).ToList();
        // The prefix: everything before the question's own turn (as the server marks a prompt).
        int last = Array.LastIndexOf(prompts[0], turn);
        int prefix = Array.LastIndexOf(prompts[0], turn, last - 1);
        if (prefix <= 0 || prompts.Any(p => p.Length <= prefix || !p.AsSpan(0, prefix).SequenceEqual(prompts[0].AsSpan(0, prefix))))
        {
            throw new CliException("the questions do not share the conversation as a prefix");
        }

        string savedFormat = GpuEngine.RequestedKvFormat;
        GpuEngine.RequestedKvFormat = "f32";
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, OptionValue(args, "--component") ?? "target", store);
        using var session = new DecodeSession(plan, store);
        session.Prefill(new[] { prompts[0][0] });
        var engine = session.DeviceEngine() ?? throw new CliException($"kv-probe runs on the GPU engine ({session.GpuRefusal})");
        GpuEngine.RequestedKvFormat = savedFormat;
        var layers = Enumerable.Range(0, plan.Layers.Count).Where(i => plan.Layers[i].Attention is not null).ToList();
        var a0 = plan.Layers[layers[0]].Attention!;

        // Query rows by layer, kept from position `keepFrom` on.
        var rowsByLayer = layers.ToDictionary(l => l, _ => new List<float>());
        int keepFrom = 0;
        engine.QueryProbe = (layer, from, rows, q) =>
        {
            int width = q.Length / rows;
            int skip = Math.Clamp(keepFrom - from, 0, rows);
            rowsByLayer[layer].AddRange(q.AsSpan(skip * width).ToArray());
        };
        void FlushQueries(string name)
        {
            foreach (var l in layers)
            {
                WriteFloats(Path.Combine(outDir, $"{name}_q_L{l}.f32"), rowsByLayer[l]);
                rowsByLayer[l].Clear();
            }
        }

        var seq = engine.NewSequence();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        keepFrom = prefix - window;
        engine.Prefill(seq, prompts[0].AsSpan(0, prefix));
        engine.Mark(seq);
        Console.WriteLine($"prefix:     {prefix} tokens prefilled in {sw.ElapsedMilliseconds} ms; {layers.Count} softmax layers");
        FlushQueries("prefix");
        foreach (var l in layers)
        {
            WriteFloats(Path.Combine(outDir, $"prefix_k_L{l}.f32"), engine.DownloadKv(seq, l, false, prefix));
            WriteFloats(Path.Combine(outDir, $"prefix_v_L{l}.f32"), engine.DownloadKv(seq, l, true, prefix));
        }

        var qs = new JsonArray();
        for (int qi = 0; qi < prompts.Count; qi++)
        {
            engine.Rewind(seq);
            keepFrom = prefix;
            var tail = prompts[qi].AsSpan(prefix).ToArray();
            var logits = engine.Prefill(seq, tail);
            var answer = new List<int>();
            for (int t = 0; t < answerTokens; t++)
            {
                int next = Sampler.ArgMax(logits);
                if (stop.Contains(next))
                {
                    break;
                }
                answer.Add(next);
                logits = engine.Prefill(seq, new[] { next });
            }
            int end = seq.Position;
            FlushQueries($"q{qi}");
            foreach (var l in layers)
            {
                var k = engine.DownloadKv(seq, l, false, end);
                var v = engine.DownloadKv(seq, l, true, end);
                int kvDim = plan.Layers[l].Attention!.KvDim;
                WriteFloats(Path.Combine(outDir, $"q{qi}_k_L{l}.f32"), k.AsSpan(prefix * kvDim).ToArray());
                WriteFloats(Path.Combine(outDir, $"q{qi}_v_L{l}.f32"), v.AsSpan(prefix * kvDim).ToArray());
            }
            string text = tokenizer.DecodeText(answer.ToArray());
            Console.WriteLine($"q{qi}:         {questions[qi]}\n            -> {OneLine(text)}");
            qs.Add(new JsonObject
            {
                ["question"] = questions[qi],
                ["tail_tokens"] = tail.Length,
                ["answer_tokens"] = answer.Count,
                ["rows"] = end - prefix,
                ["answer"] = text,
                ["answer_ids"] = new JsonArray(answer.Select(x => (JsonNode)x).ToArray()),
            });
        }
        engine.Release(seq);

        var manifest = new JsonObject
        {
            ["model"] = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(containerDir))),
            ["layers"] = new JsonArray(layers.Select(l => (JsonNode)l).ToArray()),
            ["q_heads"] = a0.NumQHeads,
            ["kv_heads"] = a0.NumKvHeads,
            ["head_dim"] = a0.HeadDim,
            ["score_scale"] = a0.ScoreScale,
            ["prefix"] = prefix,
            ["window"] = window,
            ["turn_token"] = turn,
            ["prefix_ids"] = new JsonArray(prompts[0].Take(prefix).Select(x => (JsonNode)x).ToArray()),
            ["prefix_pieces"] = new JsonArray(prompts[0].Take(prefix).Select(x => (JsonNode)tokenizer.DecodeText(new[] { x }, skipSpecialTokens: false)).ToArray()),
            ["questions"] = qs,
        };
        File.WriteAllText(Path.Combine(outDir, "manifest.json"), manifest.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"wrote:      {outDir}");
        return 0;
    }

    private static void WriteFloats(string path, IReadOnlyList<float> values)
    {
        var arr = values as float[] ?? values.ToArray();
        using var fs = File.Create(path);
        fs.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(arr.AsSpan()));
    }
}
