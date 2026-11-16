using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Amql.Inference;
using Amql.Merge;
using Amql.Vindex3;

namespace Amql.Cli;

// ── tool-trace: how a request becomes a tool and its arguments ─────────────
//
// For each case the model writes its own reply (greedy), and three decisions in
// it are traced: whether to call (the first reply token), which tool (the first
// token of the function name) and each argument (the first token of its value).
// At each, the softmax layers' attention is summed over the prompt's parts: each
// tool's name and description, the request, the format instructions, the rest.
// Separately, the request and each tool description are pooled at every few
// layers, to test whether the chosen tool is simply the nearest in meaning.

internal static partial class Program
{
    private sealed record Span(string Label, int Start, int End); // token rows [Start, End)

    private static int ToolTrace(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("tool-trace requires a container directory");
        string casesPath = OptionValue(args, "--cases") ?? throw new CliException("tool-trace requires '--cases <jsonl>'");
        string outPath = OptionValue(args, "--out") ?? throw new CliException("tool-trace requires '--out <report.json>'");
        string tokenizerDir = ResolveTokenizerDir(containerDir, args);
        var tokenizer = Tokenizer(tokenizerDir);
        var stop = ResolveStopTokens(tokenizerDir, tokenizer);
        int[] poolLayers = ParseIntList(OptionValue(args, "--pool-layers") ?? "-1,3,7,11,15,19,23", Array.Empty<int>());

        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, OptionValue(args, "--component") ?? "target", store);
        var session = new DecodeSession(plan, store, workingSet: WeightWorkingSet.ResidentF32) { AllowGpuEngine = false };
        var rt = session.Runtime;
        int hidden = plan.HiddenSize;

        var report = new JsonArray();
        foreach (var line in File.ReadAllLines(casesPath).Where(l => l.Trim().Length > 0))
        {
            var c = JsonNode.Parse(line)!.AsObject();
            string id = (string)c["id"]!, promptText = (string)c["prompt"]!;
            string? request = (string?)c["request"];
            var prompt = tokenizer.EncodeToIds(promptText).ToArray();
            int RowAt(int charIndex) => tokenizer.EncodeToIds(promptText[..charIndex]).Count();

            // The prompt's parts, as token spans.
            var spans = new List<Span>();
            int at = 0;
            foreach (var l in promptText.Split('\n'))
            {
                if (l.StartsWith("{\"type\": \"function\"", StringComparison.Ordinal))
                {
                    var name = Regex.Match(l, "\"name\": \"([^\"]+)\"");
                    var desc = Regex.Match(l, "\"description\": \"([^\"]+)\"");
                    string tool = name.Groups[1].Value;
                    spans.Add(new Span("name:" + tool, RowAt(at + name.Groups[1].Index), RowAt(at + name.Groups[1].Index + name.Groups[1].Length)));
                    if (desc.Success)
                        spans.Add(new Span("desc:" + tool, RowAt(at + desc.Groups[1].Index), RowAt(at + desc.Groups[1].Index + desc.Groups[1].Length)));
                    spans.Add(new Span("schema:" + tool, RowAt(at), RowAt(at + l.Length)));
                }
                at += l.Length + 1;
            }
            int fmt = promptText.IndexOf("If you choose to call", StringComparison.Ordinal);
            int fmtEnd = promptText.IndexOf("</IMPORTANT>", StringComparison.Ordinal);
            if (fmt >= 0 && fmtEnd > fmt)
                spans.Add(new Span("format", RowAt(fmt), RowAt(fmtEnd + "</IMPORTANT>".Length)));
            int pol = promptText.IndexOf("Policy:", StringComparison.Ordinal);
            if (pol >= 0)
                spans.Add(new Span("policy", RowAt(pol), RowAt(promptText.IndexOf("<|im_end|>", pol, StringComparison.Ordinal))));
            int req = request is null ? -1 : promptText.LastIndexOf(request, StringComparison.Ordinal);
            if (req >= 0)
                spans.Add(new Span("request", RowAt(req), RowAt(req + request!.Length)));

            // The model's own reply, and the rows that decide it.
            var reply = LoraTrainer.Greedy(session, prompt, 64, stop);
            var pieces = reply.Select(t => tokenizer.DecodeText(new[] { t })).ToList();
            string replyText = string.Concat(pieces);
            var decisions = new List<(string What, int Row)> { ("call", 0) };
            int RowOfChar(int ch)
            {
                int acc = 0;
                for (int i = 0; i < pieces.Count; i++)
                {
                    if (acc + pieces[i].Length > ch)
                        return i;
                    acc += pieces[i].Length;
                }
                return -1;
            }
            var fn = Regex.Match(replyText, "<function=([^>\n]+)>");
            if (fn.Success)
                decisions.Add(("tool " + fn.Groups[1].Value, RowOfChar(fn.Groups[1].Index)));
            foreach (Match pm in Regex.Matches(replyText, "<parameter=([^>\n]+)>\n([^\n]*)"))
                decisions.Add(($"arg {pm.Groups[1].Value}={pm.Groups[2].Value}", RowOfChar(pm.Groups[2].Index)));

            Console.WriteLine($"{id}: {OneLine(replyText)}");
            var decisionJson = new JsonArray();
            foreach (var (what, row) in decisions.Where(d => d.Row >= 0))
            {
                var tokens = prompt.Concat(reply.Take(row)).ToArray();
                rt.ResetKvCache();
                rt.AttentionTrace = new List<GenericRuntime.LayerHeadAttention>();
                Tensor2D logits;
                try
                {
                    var x = rt.Embed(tokens);
                    var pos = Enumerable.Range(0, tokens.Length).ToArray();
                    for (int layer = 0; layer < plan.Layers.Count; layer++)
                        x = rt.RunLayerInternal(x, layer, pos, pos, appendKv: true);
                    logits = rt.FinalNormAndHead(new Tensor2D(x.Row(x.Rows - 1).ToArray(), 1, hidden));
                }
                finally
                {
                    rt.ResetKvCache();
                }
                var trace = rt.AttentionTrace;
                rt.AttentionTrace = null;

                var probs = Softmax(logits.Row(0));
                var top = Enumerable.Range(0, probs.Length).OrderByDescending(v => probs[v]).Take(5)
                    .Select(v => new JsonObject { ["token"] = tokenizer.DecodeText(new[] { v }), ["p"] = Math.Round(probs[v], 4) }).ToArray();

                // Attention mass per span, per layer: mean over heads and the strongest head.
                var layersJson = new JsonObject();
                foreach (var byLayer in trace.GroupBy(t => t.Layer))
                {
                    var spanJson = new JsonObject();
                    foreach (var s in spans)
                    {
                        var masses = byLayer.Select(h =>
                        {
                            double m = 0;
                            for (int i = s.Start; i < Math.Min(s.End, h.Weights.Length); i++)
                                m += h.Weights[i];
                            return m;
                        }).ToList();
                        spanJson[s.Label] = new JsonObject { ["mean"] = Math.Round(masses.Average(), 4), ["max"] = Math.Round(masses.Max(), 4) };
                    }
                    layersJson["L" + byLayer.Key] = spanJson;
                }
                decisionJson.Add(new JsonObject
                {
                    ["what"] = what, ["row"] = row, ["said"] = pieces[row], ["top"] = new JsonArray(top), ["attention"] = layersJson,
                });
                // A short line per decision: the two parts each layer attends to most.
                var strongest = layersJson.Select(kv => $"{kv.Key}:" + string.Join("/", kv.Value!.AsObject()
                    .Where(s => !s.Key.StartsWith("schema:", StringComparison.Ordinal))
                    .OrderByDescending(s => (double)s.Value!["max"]!).Take(2).Select(s => $"{s.Key}={(double)s.Value!["max"]!:F2}")));
                Console.WriteLine($"  {what,-34} p={probs[ArgMaxRow(logits.Row(0))]:F2}  {string.Join("  ", strongest)}");
            }

            // Pooled meaning: request against each tool's description and name.
            var pooled = PoolLayers(rt, plan, prompt, poolLayers);
            var simJson = new JsonObject();
            var reqSpan = spans.FirstOrDefault(s => s.Label == "request");
            if (reqSpan is not null)
            {
                foreach (var (layer, rows) in pooled)
                {
                    // Centred on the prompt's mean row: the residual's few huge
                    // dimensions are shared by every row and would make all
                    // cosines near 1.
                    var centre = Mean(rows, new Span("all", 0, rows.Rows), hidden);
                    double[] Centred(Span s) => Mean(rows, s, hidden).Select((v, i) => v - centre[i]).ToArray();
                    var r = Centred(reqSpan);
                    var sims = new JsonObject();
                    foreach (var s in spans.Where(s => s.Label.StartsWith("desc:") || s.Label.StartsWith("name:")))
                        sims[s.Label] = Math.Round(Cosine(r, Centred(s)), 4);
                    simJson[layer < 0 ? "embedding" : "L" + layer] = sims;
                }
                string Nearest(string kind, JsonObject sims) => sims.Where(s => s.Key.StartsWith(kind))
                    .OrderByDescending(s => (double)s.Value!).First().Key[kind.Length..];
                Console.WriteLine("  nearest by meaning: " + string.Join("  ", simJson.Select(kv =>
                    $"{kv.Key}:{Nearest("desc:", kv.Value!.AsObject())}")));
            }
            report.Add(new JsonObject
            {
                ["id"] = id, ["reply"] = replyText, ["spans"] = new JsonArray(spans.Select(s => (JsonNode)new JsonObject
                {
                    ["label"] = s.Label, ["start"] = s.Start, ["end"] = s.End,
                }).ToArray()),
                ["decisions"] = decisionJson, ["similarity"] = simJson,
            });
        }
        File.WriteAllText(outPath, report.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"report: {outPath}");
        session.Dispose();
        return 0;
    }

    /// <summary>The residual of every prompt row after each requested layer
    /// (-1 is the embedding).</summary>
    private static List<(int Layer, Tensor2D Rows)> PoolLayers(GenericRuntime rt, ComponentOpPlan plan, int[] tokens, int[] layers)
    {
        var result = new List<(int, Tensor2D)>();
        rt.ResetKvCache();
        try
        {
            var x = rt.Embed(tokens);
            if (layers.Contains(-1))
                result.Add((-1, x.Clone()));
            var pos = Enumerable.Range(0, tokens.Length).ToArray();
            for (int layer = 0; layer < plan.Layers.Count; layer++)
            {
                x = rt.RunLayerInternal(x, layer, pos, pos, appendKv: true);
                if (layers.Contains(layer))
                    result.Add((layer, x.Clone()));
            }
        }
        finally
        {
            rt.ResetKvCache();
        }
        return result;
    }

    private static double[] Mean(Tensor2D rows, Span s, int hidden)
    {
        var m = new double[hidden];
        int n = 0;
        for (int r = s.Start; r < s.End && r < rows.Rows; r++, n++)
        {
            var row = rows.Row(r);
            for (int h = 0; h < hidden; h++)
                m[h] += row[h];
        }
        for (int h = 0; h < hidden; h++)
            m[h] /= Math.Max(n, 1);
        return m;
    }

    private static double Cosine(double[] a, double[] b)
    {
        double ab = 0, aa = 0, bb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            ab += a[i] * b[i];
            aa += a[i] * a[i];
            bb += b[i] * b[i];
        }
        return ab / Math.Sqrt(aa * bb + 1e-30);
    }

    private static double[] Softmax(ReadOnlySpan<float> logits)
    {
        float max = float.NegativeInfinity;
        foreach (var v in logits)
            max = Math.Max(max, v);
        var p = new double[logits.Length];
        double sum = 0;
        for (int i = 0; i < p.Length; i++)
            sum += p[i] = Math.Exp(logits[i] - max);
        for (int i = 0; i < p.Length; i++)
            p[i] /= sum;
        return p;
    }
}
