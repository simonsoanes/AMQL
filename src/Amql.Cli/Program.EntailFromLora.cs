using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Inference;
using Amql.Merge;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Cli;

// ── entail from-lora: what a LoRA adds, written as Entail ──────────────────
//
// A decompile reads the whole model, where a subject is superposed with
// everything else. A LoRA is only what training changed: r directions per
// layer. Each rank component is one virtual unit, a reader (a row of A over
// the FFN activation) and a writer (a column of B into the residual), so the
// whole delta decompiles into r states per layer. This writes, per component:
//
//   * where it fires, as token rules like decompile's, from teacher-forced cases
//   * what it says: its column read through the final norm and the head (exact
//     for the last layer, the direct path for earlier ones), as votes
//   * what in the context it depends on, by occlusion at the decision rows,
//     because a decision whose local tokens are always the same cannot be told
//     apart by token rules at all
//
// and an evidence file with every number behind those lines.

internal static partial class Program
{
    private sealed record LoadedLora(List<LoraLayer> Layers, string Dir);

    private static LoadedLora LoadLora(string dir, ComponentOpPlan plan)
    {
        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(dir, "adapter_config.json")))!.AsObject();
        using var file = SafetensorsFile.Open(Path.Combine(dir, "adapter_model.safetensors"));
        float[] Read(string name)
        {
            var info = file.GetTensor(name);
            return BitPattern.WidenToF32(info.Dtype, file.ReadBytes(info));
        }
        var layers = new List<LoraLayer>();
        foreach (var t in config["targets"]!.AsArray())
        {
            string objectId = (string)t!["object_id"]!, tensor = (string)t["tensor_name"]!;
            int layer = t["layer"] is { } l ? (int)l : -1;
            if (layer < 0)
            {
                layer = Enumerable.Range(0, plan.Layers.Count).FirstOrDefault(i =>
                    plan.Layers[i].Ffn?.Dense?.Down is { } d && d.ObjectId == objectId && d.TensorName == tensor, -1);
            }
            if (layer < 0 || plan.Layers[layer].Ffn?.Dense is not { } dense ||
                dense.Down.ObjectId != objectId || dense.Down.TensorName != tensor)
            {
                throw new CliException($"LoRA target {objectId}/{tensor} is not a dense FFN down-projection of this model — " +
                                       "from-lora reads down-projection adapters");
            }
            var shape = t["shape"]!.AsArray().Select(v => (int)v!).ToArray();
            int rank = (int)t["rank"]!;
            layers.Add(new LoraLayer
            {
                Layer = layer, Down = dense.Down, Hidden = shape[0], Inter = shape[1], Rank = rank,
                Scale = (double)t["scale"]!, A = Read((string)t["lora_a"]!), B = Read((string)t["lora_b"]!),
            });
        }
        return new LoadedLora(layers.OrderBy(l => l.Layer).ToList(), dir);
    }

    private sealed class ComponentRecord
    {
        public int Layer, Index;
        public double Max, SumAbs;
        public long Rows;
        public readonly Dictionary<long, (double Sum, int Count, string Where)> Pairs = new();
    }

    private static int EntailFromLora(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("entail from-lora requires a container directory");
        string loraDir = OptionValue(args, "--lora") ?? throw new CliException("entail from-lora requires '--lora <dir>'");
        string casesPath = OptionValue(args, "--cases") ?? throw new CliException(
            "entail from-lora requires '--cases <jsonl>' — {\"id\",\"prompt\",\"completion\"[,\"request\"]} per line");
        string outPath = OptionValue(args, "--out") ?? throw new CliException("entail from-lora requires '--out <file.entail.txt>'");
        int occlude = IntOption(args, "--occlude", 3);
        int topVotes = IntOption(args, "--votes", 6);
        double fireFraction = DoubleOption(args, "--fire", 0.25);
        string tokenizerDir = ResolveTokenizerDir(containerDir, args);
        var tokenizer = Tokenizer(tokenizerDir);

        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, OptionValue(args, "--component") ?? "target", store);
        var lora = LoadLora(loraDir, plan);
        var session = new DecodeSession(plan, store, workingSet: WeightWorkingSet.ResidentF32) { AllowGpuEngine = false };
        var rt = session.Runtime;
        int hidden = plan.HiddenSize, last = plan.Layers.Count - 1;
        var output = plan.Output ?? throw new CliException("the model has no output head");
        var head = rt.Weights.Matrix(output.Projection, output.VocabSize, output.HiddenSize);
        var finalW = rt.Weights.Vector(plan.FinalNorm.Weight, hidden);
        string Tok(int id) => Show(tokenizer.DecodeText(new[] { id }));

        var cases = File.ReadAllLines(casesPath).Where(l => l.Trim().Length > 0)
            .Select(l => JsonNode.Parse(l)!.AsObject()).ToList();
        Console.WriteLine($"lora:       {loraDir}  layers [{string.Join(",", lora.Layers.Select(l => l.Layer))}], rank {lora.Layers[0].Rank}");
        Console.WriteLine($"cases:      {cases.Count} from {casesPath}");

        var deltas = lora.Layers.Select(l => l.Delta()).ToList();
        void SetLora(bool on)
        {
            for (int i = 0; i < lora.Layers.Count; i++)
            {
                var l = lora.Layers[i];
                var d = on ? deltas[i] : deltas[i].Select(v => -v).ToArray();
                rt.Weights.AddDelta(l.Down, l.Hidden, l.Inter, d);
            }
        }

        // u[layer][row][k] = scale · A_k · act(row): each component's activation.
        (float[][][] U, Tensor2D Logits) Run(int[] tokens, int firstRow, int rows)
        {
            var u = new float[lora.Layers.Count][][];
            rt.ResetKvCache();
            rt.FfnInputCapture = (layer, hf) =>
            {
                int i = lora.Layers.FindIndex(l => l.Layer == layer);
                if (i < 0)
                    return;
                var l = lora.Layers[i];
                var act = LoraTrainer.Intermediate(rt, plan.Layers[layer].Ffn!.Dense!, hf, hidden);
                u[i] = new float[act.Rows][];
                for (int r = 0; r < act.Rows; r++)
                {
                    var a = act.Row(r);
                    var row = u[i][r] = new float[l.Rank];
                    for (int k = 0; k < l.Rank; k++)
                    {
                        double s = 0;
                        var ak = l.A.AsSpan(k * l.Inter, l.Inter);
                        for (int j = 0; j < l.Inter; j++)
                            s += ak[j] * a[j];
                        row[k] = (float)(l.Scale * s);
                    }
                }
            };
            try
            {
                var x = rt.Embed(tokens);
                var positions = Enumerable.Range(0, tokens.Length).ToArray();
                for (int layer = 0; layer < plan.Layers.Count; layer++)
                    x = rt.RunLayerInternal(x, layer, positions, positions, appendKv: true);
                var slice = new Tensor2D(x.Data.AsSpan(firstRow * hidden, rows * hidden).ToArray(), rows, hidden);
                return (u, rt.FinalNormAndHead(slice));
            }
            finally
            {
                rt.FfnInputCapture = null;
                rt.ResetKvCache();
            }
        }

        static double LogProb(ReadOnlySpan<float> row, int target)
        {
            float max = float.NegativeInfinity;
            foreach (var v in row)
                max = Math.Max(max, v);
            double sum = 0;
            foreach (var v in row)
                sum += Math.Exp(v - max);
            return row[target] - max - Math.Log(sum);
        }

        // ── 1. what each component says: its column through norm and head ──
        var readouts = new List<(int Layer, int K, List<(int Token, double W)> Up, List<(int Token, double W)> Down)>();
        foreach (var l in lora.Layers)
        {
            for (int k = 0; k < l.Rank; k++)
            {
                var col = new float[hidden];
                for (int h = 0; h < hidden; h++)
                    col[h] = l.B[h * l.Rank + k] * (finalW[h] + plan.FinalNorm.WeightOffset);
                var scores = new double[output.VocabSize];
                for (int v = 0; v < scores.Length; v++)
                {
                    var w = head.Data.AsSpan(v * hidden, hidden);
                    double s = 0;
                    for (int h = 0; h < hidden; h++)
                        s += w[h] * col[h];
                    scores[v] = s;
                }
                var order = Enumerable.Range(0, scores.Length).OrderByDescending(v => scores[v]).ToList();
                readouts.Add((l.Layer, k,
                    order.Take(topVotes).Select(v => (v, scores[v])).ToList(),
                    order.TakeLast(topVotes).Reverse().Select(v => (v, scores[v])).ToList()));
            }
        }

        // ── 2. where each component fires, over teacher-forced cases ──
        var records = new Dictionary<(int, int), ComponentRecord>();
        foreach (var l in lora.Layers)
            for (int k = 0; k < l.Rank; k++)
                records[(l.Layer, k)] = new ComponentRecord { Layer = l.Layer, Index = k };
        var evidence = new JsonArray();
        var decisions = new List<(string Id, int[] Prompt, int Target, float[][] U, double LpBase, double LpLora)>();
        foreach (var c in cases)
        {
            string id = (string?)c["id"] ?? "case";
            var prompt = tokenizer.EncodeToIds((string)c["prompt"]!).ToArray();
            var completion = tokenizer.EncodeToIds((string)c["completion"]!).ToArray();
            var tokens = prompt.Concat(completion.Take(completion.Length - 1)).ToArray();
            int first = prompt.Length - 1, rows = completion.Length;
            var (_, baseLogits) = Run(tokens, first, rows);
            SetLora(true);
            var (u, loraLogits) = Run(tokens, first, rows);
            SetLora(false);

            var rowsJson = new JsonArray();
            for (int r = 0; r < rows; r++)
            {
                int t = first + r;
                int sym = tokens[t], prev = t > 0 ? tokens[t - 1] : -1;
                double lpb = LogProb(baseLogits.Row(r), completion[r]), lpl = LogProb(loraLogits.Row(r), completion[r]);
                var comps = new JsonArray();
                for (int i = 0; i < lora.Layers.Count; i++)
                {
                    for (int k = 0; k < lora.Layers[i].Rank; k++)
                    {
                        double val = u[i][t][k];
                        comps.Add(Math.Round(val, 4));
                        var rec = records[(lora.Layers[i].Layer, k)];
                        rec.Rows++;
                        rec.SumAbs += Math.Abs(val);
                        rec.Max = Math.Max(rec.Max, Math.Abs(val));
                        long key = (long)prev << 32 | (uint)sym;
                        var cur = rec.Pairs.GetValueOrDefault(key, (0.0, 0, $"{id}@{r}"));
                        rec.Pairs[key] = (cur.Item1 + val, cur.Item2 + 1, cur.Item3);
                    }
                }
                rowsJson.Add(new JsonObject
                {
                    ["row"] = r, ["prev"] = tokenizer.DecodeText(new[] { prev }), ["sym"] = tokenizer.DecodeText(new[] { sym }),
                    ["target"] = tokenizer.DecodeText(new[] { completion[r] }),
                    ["logpBase"] = Math.Round(lpb, 4), ["logpLora"] = Math.Round(lpl, 4),
                    ["topBase"] = tokenizer.DecodeText(new[] { ArgMaxRow(baseLogits.Row(r)) }),
                    ["topLora"] = tokenizer.DecodeText(new[] { ArgMaxRow(loraLogits.Row(r)) }),
                    ["u"] = comps,
                });
            }
            evidence.Add(new JsonObject { ["id"] = id, ["rows"] = rowsJson });
            decisions.Add((id, prompt, completion[0], u.Select(x => x[first]).ToArray(),
                LogProb(baseLogits.Row(0), completion[0]), LogProb(loraLogits.Row(0), completion[0])));
            Console.WriteLine($"  {id,-24} decision: p({Tok(completion[0])}) {Math.Exp(LogProb(baseLogits.Row(0), completion[0])):F3} → " +
                              $"{Math.Exp(LogProb(loraLogits.Row(0), completion[0])):F3}");
        }

        // ── 3. what the decision depends on: occlusion at the first reply row ──
        var occlusion = new JsonArray();
        foreach (var c in cases.Take(occlude))
        {
            string id = (string?)c["id"] ?? "case";
            string promptText = (string)c["prompt"]!;
            int target = tokenizer.EncodeToIds((string)c["completion"]!).First();
            var segments = OcclusionSegments(promptText, (string?)c["request"]);
            double Decide(string text, bool on, out float[] uRow)
            {
                var ids = tokenizer.EncodeToIds(text).ToArray();
                if (on)
                    SetLora(true);
                try
                {
                    var (u, logits) = Run(ids, ids.Length - 1, 1);
                    uRow = on ? u.SelectMany(x => x[ids.Length - 1]).ToArray() : Array.Empty<float>();
                    return Math.Exp(LogProb(logits.Row(0), target));
                }
                finally
                {
                    if (on)
                        SetLora(false);
                }
            }
            double pBase = Decide(promptText, false, out _), pLora = Decide(promptText, true, out var uFull);
            var segJson = new JsonArray();
            Console.WriteLine($"  occlude {id}: {segments.Count} segments, p({Tok(target)}) base {pBase:F3}, lora {pLora:F3}");
            foreach (var (label, start, length) in segments)
            {
                string text = promptText.Remove(start, length);
                double pb = Decide(text, false, out _), pl = Decide(text, true, out var uRow);
                segJson.Add(new JsonObject
                {
                    ["segment"] = label, ["pBase"] = Math.Round(pb, 4), ["pLora"] = Math.Round(pl, 4),
                    ["du"] = new JsonArray(uRow.Select((v, i) => (JsonNode)Math.Round(v - uFull[i], 4)).ToArray()),
                });
                Console.WriteLine($"    - {Show(label),-40} base {pBase:F3}→{pb:F3}  lora {pLora:F3}→{pl:F3}");
            }
            occlusion.Add(new JsonObject
            {
                ["id"] = id, ["target"] = tokenizer.DecodeText(new[] { target }), ["pBase"] = pBase, ["pLora"] = pLora,
                ["u"] = new JsonArray(uFull.Select(v => (JsonNode)Math.Round(v, 4)).ToArray()), ["segments"] = segJson,
            });
        }

        // ── 4. write the program ──
        var sb = new StringBuilder();
        string name = OptionValue(args, "--name") ?? "lora_" + Path.GetFileName(Path.GetFullPath(loraDir).TrimEnd('\\', '/')).Replace('-', '_');
        sb.AppendLine($"program {name}");
        sb.AppendLine($"# From the LoRA in {loraDir}: layers {string.Join(", ", lora.Layers.Select(l => l.Layer))}, " +
                      $"rank {lora.Layers[0].Rank}, scale {lora.Layers[0].Scale.ToString(CultureInfo.InvariantCulture)}.");
        sb.AppendLine("# Each rank component is one state. Its value is scale · A_k · act, its votes its column read");
        sb.AppendLine("# through the final norm and the head" + (lora.Layers.Any(l => l.Layer != last)
            ? " (exact for the last layer; for earlier layers the direct path only)." : ", which is exact for the last layer."));
        sb.AppendLine("# Rules are measured on teacher-forced cases and see only the current and previous token,");
        sb.AppendLine("# the same ceiling as decompile. What a rule cannot see is in the context notes below.");
        sb.AppendLine();
        foreach (var (layer, k, upVotes, downVotes) in readouts)
        {
            var rec = records[(layer, k)];
            string state = $"lora_L{layer}_{k}";
            double unit = Math.Max(rec.Max / 8, 1e-6);
            sb.AppendLine($"# L{layer} component {k}: peak |u| {rec.Max:F3}, mean |u| {(rec.Rows == 0 ? 0 : rec.SumAbs / rec.Rows):F3} over {rec.Rows} rows");
            sb.AppendLine($"# column promotes {string.Join(" ", upVotes.Select(v => Tok(v.Token)))}");
            sb.AppendLine($"# column demotes  {string.Join(" ", downVotes.Select(v => Tok(v.Token)))}");
            sb.AppendLine($"state {state}");
            foreach (var (key, (sum, count, where)) in rec.Pairs
                         .Where(p => Math.Abs(p.Value.Sum / p.Value.Count) >= fireFraction * rec.Max)
                         .OrderByDescending(p => Math.Abs(p.Value.Sum / p.Value.Count)))
            {
                int prev = (int)(key >> 32), sym = (int)(key & 0xffffffff);
                double mean = sum / count;
                long level = (long)Math.Round(mean / unit);
                if (level == 0)
                    continue;
                sb.AppendLine($"entail {state} = {level} when prev == {prev}, sym == {sym}   # {Tok(prev)} {Tok(sym)} x{count} mean u {mean:F3}, e.g. {where}");
            }
            foreach (var (token, w) in upVotes)
                sb.AppendLine($"vote {Show(tokenizer.DecodeText(new[] { token }))} by {(w * unit).ToString("0.####", CultureInfo.InvariantCulture)}*{state}");
            sb.AppendLine();
        }

        sb.AppendLine("# ── Context: what the decision at the first reply row depends on ──");
        sb.AppendLine("# The first reply row always reads the same local tokens, so no rule above can tell a call");
        sb.AppendLine("# from prose there. Occlusion removes one segment of the prompt and re-measures.");
        foreach (var o in occlusion)
        {
            sb.AppendLine($"# {o!["id"]}: p({Show((string)o["target"]!)}) base {(double)o["pBase"]!:F3}, with LoRA {(double)o["pLora"]!:F3}");
            foreach (var s in o["segments"]!.AsArray()
                         .OrderBy(s => (double)s!["pLora"]!))
            {
                sb.AppendLine($"#   without {Show((string)s!["segment"]!),-44} base {(double)s["pBase"]!:F3}  lora {(double)s["pLora"]!:F3}");
            }
        }
        File.WriteAllText(outPath, sb.ToString());
        string evidencePath = Path.ChangeExtension(outPath, null).Replace(".entail", "") + ".evidence.json";
        File.WriteAllText(evidencePath, new JsonObject
        {
            ["lora"] = loraDir,
            ["readouts"] = new JsonArray(readouts.Select(r => (JsonNode)new JsonObject
            {
                ["layer"] = r.Layer, ["k"] = r.K,
                ["promotes"] = new JsonArray(r.Up.Select(v => (JsonNode)new JsonObject { ["token"] = tokenizer.DecodeText(new[] { v.Token }), ["w"] = v.W }).ToArray()),
                ["demotes"] = new JsonArray(r.Down.Select(v => (JsonNode)new JsonObject { ["token"] = tokenizer.DecodeText(new[] { v.Token }), ["w"] = v.W }).ToArray()),
            }).ToArray()),
            ["cases"] = evidence,
            ["occlusion"] = occlusion,
        }.ToJsonString(new JsonSerializerOptions { WriteIndented = false }));
        Console.WriteLine($"program:    {outPath}");
        Console.WriteLine($"evidence:   {evidencePath}");
        session.Dispose();
        return 0;
    }

    /// <summary>The parts of a chat prompt worth removing one at a time: each
    /// tool definition, each line of a system turn, and each word of the request.</summary>
    private static List<(string Label, int Start, int Length)> OcclusionSegments(string prompt, string? request)
    {
        var segments = new List<(string, int, int)>();
        int at = 0;
        foreach (var line in prompt.Split('\n'))
        {
            if (line.StartsWith("{\"type\": \"function\"", StringComparison.Ordinal))
            {
                var m = System.Text.RegularExpressions.Regex.Match(line, "\"name\": \"([^\"]+)\"");
                segments.Add(("tool " + (m.Success ? m.Groups[1].Value : "?"), at, line.Length + 1));
            }
            else if (line.StartsWith("Policy:", StringComparison.Ordinal) || line.StartsWith("If you choose to call", StringComparison.Ordinal))
            {
                segments.Add((line.Length > 40 ? line[..40] + "…" : line, at, line.Length + 1));
            }
            at += line.Length + 1;
        }
        int important = prompt.IndexOf("<IMPORTANT>", StringComparison.Ordinal);
        int importantEnd = prompt.IndexOf("</IMPORTANT>", StringComparison.Ordinal);
        if (important >= 0 && importantEnd > important)
            segments.Add(("<IMPORTANT> block", important, importantEnd + "</IMPORTANT>".Length - important));
        if (request is not null)
        {
            int r = prompt.LastIndexOf(request, StringComparison.Ordinal);
            if (r >= 0)
            {
                foreach (System.Text.RegularExpressions.Match w in System.Text.RegularExpressions.Regex.Matches(request, @"\S+"))
                {
                    // remove the word and the space before it, so the rest reads naturally
                    int start = r + w.Index, length = w.Length;
                    if (w.Index > 0)
                    {
                        start--;
                        length++;
                    }
                    segments.Add(("word " + w.Value, start, length));
                }
            }
        }
        return segments;
    }

    private static int ArgMaxRow(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
            if (row[i] > row[best])
                best = i;
        return best;
    }
}
