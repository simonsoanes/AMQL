using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Amql.Entail;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Cli;

// ── entail find / extract / optimise / review / accept ──────────────────────
//
// From a trained model to a subject's Entail, and from there to something
// exact. docs/entail-optimisation.md describes the cycle.

internal static partial class Program
{
    private static readonly JsonSerializerOptions EntailJson = new() { WriteIndented = true };

    private sealed class SubjectModel : IDisposable
    {
        public required Vindex3Container Container { get; init; }
        public required OperandStore Store { get; init; }
        public required ComponentOpPlan Plan { get; init; }
        public required DecodeSession Session { get; init; }
        public required HfTokenizer Tokenizer { get; init; }
        public required IReadOnlySet<int> Stop { get; init; }

        public void Dispose()
        {
            Store.Dispose();
            Container.Dispose();
        }
    }

    private static SubjectModel OpenSubjectModel(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("this entail verb requires a container directory");
        string tokenizerDir = ResolveTokenizerDir(containerDir, args);
        var tokenizer = Tokenizer(tokenizerDir);
        var container = Vindex3Container.Open(containerDir);
        var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, OptionValue(args, "--component") ?? "target", store);
        return new SubjectModel
        {
            Container = container,
            Store = store,
            Plan = plan,
            Session = new DecodeSession(plan, store, null, WeightWorkingSet.ResidentF32),
            Tokenizer = tokenizer,
            Stop = ResolveStopTokens(tokenizerDir, tokenizer),
        };
    }

    // ── find ────────────────────────────────────────────────────────────────

    /// <summary>One candidate carrier of the subject: a neuron of a dense FFN,
    /// or an expert of a routed one.</summary>
    private sealed record SubjectUnit(int Layer, int Index, bool Expert, double Score, double Subject, double Contrast, string Token);

    private sealed class LayerStats
    {
        public required double[] Sum, Square, ContrastSum, ContrastSquare, Best;
        public required string[] BestToken;

        public static LayerStats Of(int n) => new()
        {
            Sum = new double[n], Square = new double[n], ContrastSum = new double[n], ContrastSquare = new double[n],
            Best = new double[n], BestToken = new string[n],
        };
    }

    /// <summary>
    /// Which units carry a subject. A dense unit is measured by what it adds
    /// to the residual (|activation| × the norm of its output column), an
    /// expert by how often the router picks it; a unit's score is how far
    /// subject text moves that measure from contrast text, in standard
    /// deviations. With <c>--verify</c> the claim is tested causally: remove
    /// the top units, and an equal number at random, and compare the damage.
    /// </summary>
    private static int EntailFind(string[] args)
    {
        var subject = Subject.Load(OptionValue(args, "--subject") ?? throw new CliException("entail find requires '--subject <file>'"));
        string outDir = OptionValue(args, "--out") ?? throw new CliException("entail find requires '--out <dir>'");
        int top = IntOption(args, "--top", 64);
        Directory.CreateDirectory(outDir);
        using var model = OpenSubjectModel(args);
        var runtime = model.Session.Runtime;
        var plan = model.Plan;

        var stats = new LayerStats?[plan.Layers.Count];
        var outputNorms = new float[plan.Layers.Count][];
        for (int l = 0; l < plan.Layers.Count; l++)
        {
            if (plan.Layers[l].Ffn is { Dense: { } dense })
            {
                stats[l] = LayerStats.Of(dense.IntermediateSize);
                var down = runtime.Weights.Matrix(dense.Down, dense.HiddenSize, dense.IntermediateSize);
                var norms = new double[dense.IntermediateSize];
                for (int r = 0; r < dense.HiddenSize; r++)
                {
                    for (int c = 0; c < dense.IntermediateSize; c++)
                    {
                        double v = down.Data[r * dense.IntermediateSize + c];
                        norms[c] += v * v;
                    }
                }
                outputNorms[l] = norms.Select(x => (float)Math.Sqrt(x)).ToArray();
            }
            else if (plan.Layers[l].Ffn is { Routed: { } routed })
            {
                stats[l] = LayerStats.Of(routed.NumExperts);
            }
        }

        bool inSubject = true;
        string currentToken = string.Empty;
        long subjectTokens = 0, contrastTokens = 0;
        runtime.FfnInputCapture = (layer, input) =>
        {
            var s = stats[layer];
            if (s is null)
            {
                return;
            }
            var measure = Measure(runtime, plan.Layers[layer].Ffn!, input, outputNorms[layer]);
            for (int i = 0; i < measure.Length; i++)
            {
                double v = measure[i];
                if (inSubject)
                {
                    s.Sum[i] += v;
                    s.Square[i] += v * v;
                    if (v > s.Best[i])
                    {
                        s.Best[i] = v;
                        s.BestToken[i] = currentToken;
                    }
                }
                else
                {
                    s.ContrastSum[i] += v;
                    s.ContrastSquare[i] += v * v;
                }
            }
        };

        void Read(IReadOnlyList<string> lines)
        {
            foreach (var line in lines)
            {
                model.Session.Reset();
                foreach (int id in model.Tokenizer.EncodeToIds(line))
                {
                    currentToken = model.Tokenizer.TokenInfo(id).DecodedText ?? string.Empty;
                    runtime.StepForward(id);
                    if (inSubject)
                    {
                        subjectTokens++;
                    }
                    else
                    {
                        contrastTokens++;
                    }
                }
            }
        }

        CliProgress.Phase("entail find");
        inSubject = true;
        Read(subject.Lines);
        inSubject = false;
        Read(subject.Contrast);
        runtime.FfnInputCapture = null;

        var units = new List<SubjectUnit>();
        for (int l = 0; l < stats.Length; l++)
        {
            if (stats[l] is not { } s)
            {
                continue;
            }
            bool expert = plan.Layers[l].Ffn!.Routed is not null;
            for (int i = 0; i < s.Sum.Length; i++)
            {
                double ms = s.Sum[i] / subjectTokens, mc = s.ContrastSum[i] / contrastTokens;
                double vs = s.Square[i] / subjectTokens - ms * ms, vc = s.ContrastSquare[i] / contrastTokens - mc * mc;
                double score = (ms - mc) / Math.Sqrt(0.5 * (vs + vc) + 1e-9);
                units.Add(new SubjectUnit(l, i, expert, score, ms, mc, s.BestToken[i] ?? string.Empty));
            }
        }
        units.Sort((a, b) => b.Score.CompareTo(a.Score));
        var chosen = units.Take(top).ToList();

        Console.WriteLine($"subject:    {subject.Name}: {subjectTokens} tokens, contrast {contrastTokens} tokens");
        Console.WriteLine($"units:      {units.Count} measured, {(units.Any(u => u.Expert) ? "experts" : "neurons")}");
        Console.WriteLine("by layer:   " + string.Join(" ", chosen.GroupBy(u => u.Layer).OrderBy(g => g.Key).Select(g => $"L{g.Key}:{g.Count()}")));
        foreach (var unit in chosen.Take(12))
        {
            // An expert is on or off, so it has no strongest token.
            Console.WriteLine($"  L{unit.Layer,-2} {(unit.Expert ? "expert" : "neuron")} {unit.Index,-5} score {unit.Score,5:F2}  " +
                              $"subject {unit.Subject:F3} contrast {unit.Contrast:F3}" +
                              (unit.Expert ? string.Empty : $"  strongest on {JsonSerializer.Serialize(unit.Token)}"));
        }

        var map = new JsonObject
        {
            ["subject"] = subject.Name,
            ["subject_tokens"] = subjectTokens,
            ["contrast_tokens"] = contrastTokens,
            ["units"] = new JsonArray(chosen.Select(u => (JsonNode?)new JsonObject
            {
                ["layer"] = u.Layer, ["index"] = u.Index, ["kind"] = u.Expert ? "expert" : "neuron",
                ["score"] = Math.Round(u.Score, 4), ["subject"] = Math.Round(u.Subject, 5),
                ["contrast"] = Math.Round(u.Contrast, 5), ["strongest_on"] = u.Token,
            }).ToArray()),
        };

        if (HasOption(args, "--verify"))
        {
            var random = new Random(IntOption(args, "--seed", 1));
            var layerUnits = units.GroupBy(u => u.Layer).ToDictionary(g => g.Key, g => g.ToList());
            // Same layers, same counts: only which units differs.
            var control = chosen.GroupBy(u => u.Layer)
                .SelectMany(g => layerUnits[g.Key].OrderBy(_ => random.Next()).Where(u => !chosen.Contains(u)).Take(g.Count()))
                .ToList();
            var results = new JsonObject();
            (double S, double C) Loss(IReadOnlyList<SubjectUnit> removed)
            {
                var session = removed.Count == 0
                    ? model.Session
                    : new DecodeSession(plan, model.Store, Ablation(model, removed), WeightWorkingSet.ResidentF32);
                return (TextLoss(session, model.Tokenizer, subject.Lines), TextLoss(session, model.Tokenizer, subject.Contrast));
            }
            var baseLoss = Loss(Array.Empty<SubjectUnit>());
            var topLoss = Loss(chosen);
            var controlLoss = Loss(control);
            Console.WriteLine();
            Console.WriteLine($"{"removed",-24} {"subject loss",14} {"contrast loss",14}");
            Console.WriteLine($"{"nothing",-24} {baseLoss.S,14:F4} {baseLoss.C,14:F4}");
            Console.WriteLine($"{$"top {chosen.Count} units",-24} {topLoss.S,14:F4} {topLoss.C,14:F4}");
            Console.WriteLine($"{$"{control.Count} random units",-24} {controlLoss.S,14:F4} {controlLoss.C,14:F4}");
            double selective = (topLoss.S - baseLoss.S) - (topLoss.C - baseLoss.C);
            Console.WriteLine($"selectivity: removing the top units costs the subject {selective:F4} nats/token more than the contrast");
            map["verify"] = new JsonObject
            {
                ["base"] = new JsonArray(baseLoss.S, baseLoss.C),
                ["top"] = new JsonArray(topLoss.S, topLoss.C),
                ["random"] = new JsonArray(controlLoss.S, controlLoss.C),
                ["selectivity"] = selective,
            };
        }

        File.WriteAllText(Path.Combine(outDir, "map.json"), map.ToJsonString(EntailJson));
        Console.WriteLine($"wrote:      {Path.Combine(outDir, "map.json")}");
        return 0;
    }

    private static float[] Measure(GenericRuntime runtime, LayerFfn ffn, Tensor2D input, float[]? outputNorms)
    {
        var row = new Tensor2D(input.Row(input.Rows - 1).ToArray(), 1, input.Cols);
        if (ffn.Dense is { } dense)
        {
            var up = TensorOps.MatMulTransposedB(row, runtime.Weights.Matrix(dense.Up, dense.IntermediateSize, dense.HiddenSize));
            var gate = dense.Gate is null ? null
                : TensorOps.MatMulTransposedB(row, runtime.Weights.Matrix(dense.Gate, dense.IntermediateSize, dense.HiddenSize));
            var result = new float[dense.IntermediateSize];
            for (int i = 0; i < result.Length; i++)
            {
                float a = up.Data[i];
                if (gate is not null)
                {
                    float g = gate.Data[i];
                    a *= g / (1f + MathF.Exp(-g));
                }
                result[i] = MathF.Abs(a) * outputNorms![i];
            }
            return result;
        }
        var routed = ffn.Routed!;
        var logits = TensorOps.MatMulTransposedB(row, runtime.Weights.Matrix(routed.Router, routed.NumExperts, routed.HiddenSize));
        var picked = new float[routed.NumExperts];
        foreach (int e in Enumerable.Range(0, routed.NumExperts).OrderByDescending(i => logits.Data[i]).Take(routed.TopK))
        {
            picked[e] = 1f;
        }
        return picked;
    }

    /// <summary>A patch that zeroes the units' contribution to the residual.</summary>
    private static WeightPatch Ablation(SubjectModel model, IReadOnlyList<SubjectUnit> units)
    {
        var entries = new List<WeightPatchEntry>();
        foreach (var layer in units.GroupBy(u => u.Layer))
        {
            var ffn = model.Plan.Layers[layer.Key].Ffn!;
            if (ffn.Dense is { } dense)
            {
                var down = model.Session.Runtime.Weights.Matrix(dense.Down, dense.HiddenSize, dense.IntermediateSize);
                var delta = new float[down.Data.Length];
                foreach (var unit in layer)
                {
                    for (int r = 0; r < dense.HiddenSize; r++)
                    {
                        int at = r * dense.IntermediateSize + unit.Index;
                        delta[at] = -down.Data[at];
                    }
                }
                entries.Add(new WeightPatchEntry(dense.Down.ObjectId, dense.Down.TensorName,
                    new long[] { dense.HiddenSize, dense.IntermediateSize }, delta));
            }
            else
            {
                var routed = ffn.Routed!;
                foreach (var unit in layer)
                {
                    var operand = new OperandRef(routed.Router.ObjectId, $"{routed.ExpertGatePrefix}{unit.Index}.down_proj.weight");
                    var down = model.Session.Runtime.Weights.Matrix(operand, routed.HiddenSize, routed.ExpertIntermediateSize);
                    entries.Add(new WeightPatchEntry(operand.ObjectId, operand.TensorName,
                        new long[] { routed.HiddenSize, routed.ExpertIntermediateSize }, down.Data.Select(v => -v).ToArray()));
                }
            }
        }
        return WeightPatch.FromEntries(entries);
    }

    private static double TextLoss(DecodeSession session, HfTokenizer tokenizer, IReadOnlyList<string> lines)
    {
        double total = 0;
        long count = 0;
        foreach (var line in lines)
        {
            var ids = tokenizer.EncodeToIds(line).ToArray();
            if (ids.Length < 2)
            {
                continue;
            }
            session.Reset();
            var logits = session.Prefill(ids[..1]);
            for (int i = 1; i < ids.Length; i++)
            {
                var row = logits.Row(0);
                double max = double.NegativeInfinity, sum = 0;
                foreach (var v in row)
                {
                    max = Math.Max(max, v);
                }
                foreach (var v in row)
                {
                    sum += Math.Exp(v - max);
                }
                total += Math.Log(sum) + max - row[ids[i]];
                count++;
                logits = session.Step(ids[i]);
            }
        }
        return total / Math.Max(count, 1);
    }

    // ── extract ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Asks the model for every relation of a subject across its domain,
    /// records what it said and how decided the softmax was, and writes the
    /// result out as Entail: a faithful, unoptimised table. Roughly a third of
    /// the instances are held out; no later stage may fit to them.
    /// </summary>
    private static int EntailExtract(string[] args)
    {
        var subject = Subject.Load(OptionValue(args, "--subject") ?? throw new CliException("entail extract requires '--subject <file>'"));
        string outDir = OptionValue(args, "--out") ?? throw new CliException("entail extract requires '--out <dir>'");
        int seed = IntOption(args, "--seed", 1);
        Directory.CreateDirectory(outDir);
        using var model = OpenSubjectModel(args);
        var rows = new List<EvidenceRow>();
        var split = new Random(seed + 1);

        foreach (var relation in subject.Relations)
        {
            int shotTokens = relation.Shots.Length == 0 ? 0 : model.Tokenizer.EncodeToIds(relation.Shots).Count;
            int done = 0;
            foreach (var values in relation.Instances(seed))
            {
                var ids = model.Tokenizer.EncodeToIds(relation.Shots + relation.Render(values)).ToArray();
                var context = ids.Skip(Math.Max(shotTokens - 1, 0))
                    .Select(id => model.Tokenizer.TokenInfo(id).DecodedText ?? string.Empty).ToList();

                model.Session.Reset();
                var logits = model.Session.Prefill(ids);
                var output = new List<string>();
                double margin = 1;
                for (int step = 0; step < relation.MaxTokens; step++)
                {
                    var (best, gap) = Decide(logits.Row(0));
                    string text = model.Tokenizer.TokenInfo(best).DecodedText ?? string.Empty;
                    margin = Math.Min(margin, gap);
                    if (model.Stop.Contains(best) || text.Contains(relation.Until, StringComparison.Ordinal))
                    {
                        break;
                    }
                    output.Add(text);
                    logits = model.Session.Step(best);
                }
                rows.Add(new EvidenceRow(relation.Name, values, context, output, margin, split.NextDouble() < 0.33));
                if (++done % 50 == 0)
                {
                    Console.WriteLine($"  {relation.Name}: {done}");
                }
            }
        }

        File.WriteAllLines(Path.Combine(outDir, "evidence.jsonl"), rows.Select(r => r.ToJson().ToJsonString()));
        WriteBackground(model, subject, outDir);
        var plan = SubjectPlan.FromEvidence(subject.Name, rows);
        plan.Confidence = 0;
        string program = EntailOptimiser.Generate(plan, $"Subject '{subject.Name}', extracted from {model.Container.Index.Model}.");
        File.WriteAllText(Path.Combine(outDir, "v0.entail.txt"), program);
        File.WriteAllText(Path.Combine(outDir, "current.entail.txt"), program);
        var standing = EntailOptimiser.Measure(program, rows, 0.5);
        Console.WriteLine($"relations:  {plan.Relations.Count}, {rows.Count} instances ({rows.Count(r => r.Holdout)} held out)");
        Console.WriteLine($"decided:    {rows.Count(r => r.Margin >= 0.5)} of {rows.Count} had a softmax margin of at least 0.5");
        Console.WriteLine($"program:    {standing}");
        Console.WriteLine($"wrote:      {Path.Combine(outDir, "v0.entail.txt")}");
        return 0;
    }

    /// <summary>What the model predicts at each position of ordinary text,
    /// for the gate's false-fire check.</summary>
    private static void WriteBackground(SubjectModel model, Subject subject, string outDir)
    {
        var lines = new List<string>();
        foreach (var line in subject.Lines.Concat(subject.Contrast))
        {
            var ids = model.Tokenizer.EncodeToIds(line).ToArray();
            var tokens = ids.Select(id => model.Tokenizer.TokenInfo(id).DecodedText ?? string.Empty).ToList();
            var next = new List<string>();
            model.Session.Reset();
            var logits = model.Session.Prefill(ids[..1]);
            for (int i = 0; i < ids.Length; i++)
            {
                next.Add(model.Tokenizer.TokenInfo(Decide(logits.Row(0)).Best).DecodedText ?? string.Empty);
                if (i + 1 < ids.Length)
                {
                    logits = model.Session.Step(ids[i + 1]);
                }
            }
            lines.Add(new BackgroundLine(tokens, next).ToJson().ToJsonString());
        }
        File.WriteAllLines(Path.Combine(outDir, "background.jsonl"), lines);
    }

    /// <summary>Records the negative evidence on its own, for a work directory
    /// extracted before it existed or against a different corpus.</summary>
    private static int EntailBackground(string[] args)
    {
        var subject = Subject.Load(OptionValue(args, "--subject") ?? throw new CliException("entail background requires '--subject <file>'"));
        string outDir = OptionValue(args, "--out") ?? throw new CliException("entail background requires '--out <dir>'");
        using var model = OpenSubjectModel(args);
        WriteBackground(model, subject, outDir);
        return 0;
    }

    /// <summary>The argmax, and the gap between the top two probabilities.</summary>
    private static (int Best, double Gap) Decide(ReadOnlySpan<float> row)
    {
        int best = 0, second = -1;
        for (int i = 1; i < row.Length; i++)
        {
            if (row[i] > row[best])
            {
                second = best;
                best = i;
            }
            else if (second < 0 || row[i] > row[second])
            {
                second = i;
            }
        }
        double max = row[best], sum = 0;
        foreach (var v in row)
        {
            sum += Math.Exp(v - max);
        }
        return (best, (1 - Math.Exp(row[second] - max)) / sum);
    }

    // ── optimise ────────────────────────────────────────────────────────────

    /// <summary>
    /// Applies the mechanical rewrites of docs/entail-optimisation.md in
    /// order. Each is measured, passed through the gate, and journalled; one
    /// the gate refuses is undone. No step looks at held-out rows except the
    /// gate itself.
    /// </summary>
    private static int EntailOptimise(string[] args)
    {
        string dir = Arg(args, 0) ?? throw new CliException("entail optimise requires a work directory");
        double confidence = DoubleOption(args, "--confidence", 0.5);
        var rows = EvidenceRow.Load(Path.Combine(dir, "evidence.jsonl"));
        var background = BackgroundLine.Load(Path.Combine(dir, "background.jsonl"));
        string name = Regex.Match(File.ReadAllText(Path.Combine(dir, "v0.entail.txt")), @"^program (\S+)", RegexOptions.Multiline).Groups[1].Value;
        var plan = SubjectPlan.FromEvidence(name, rows);
        plan.Confidence = 0;
        string current = EntailOptimiser.Generate(plan);
        var standing = EntailOptimiser.Measure(current, rows, confidence, background);
        Console.WriteLine($"start:      {standing}");
        var journal = new List<JsonObject>();

        bool Try(string rule, string detail, Action apply, Action undo)
        {
            apply();
            var candidate = EntailOptimiser.Generate(plan);
            var after = EntailOptimiser.Measure(candidate, rows, confidence, background);
            var (accepted, why) = EntailOptimiser.Gate(standing, after);
            Console.WriteLine($"{rule,-4} {(accepted ? "kept   " : "refused")} {detail}: {why}");
            Journal(journal, rule, detail, "mechanical", accepted, why, standing, after);
            if (accepted)
            {
                current = candidate;
                standing = after;
            }
            else
            {
                undo();
            }
            return accepted;
        }

        Try("R1", $"keep rows with softmax margin of at least {confidence}",
            () => plan.Confidence = confidence, () => plan.Confidence = 0);
        foreach (var relation in plan.Relations)
        {
            Try("R2", $"{relation.Name}: hoist literal conditions",
                () => relation.Factored = true, () => relation.Factored = false);
        }
        foreach (var relation in plan.Relations)
        {
            var irrelevant = EntailOptimiser.IrrelevantSlots(relation, confidence);
            foreach (int slot in irrelevant)
            {
                Try("R3", $"{relation.Name}: slot {slot} never changes the answer",
                    () => relation.Dropped.Add(slot), () => relation.Dropped.Remove(slot));
            }
        }
        foreach (var relation in plan.Relations)
        {
            if (EntailOptimiser.DeriveLaw(relation, confidence) is { } law)
            {
                Try("R4", $"{relation.Name}: value = {law} ({law.Inliers} of {law.Rows} rows)",
                    () => relation.Law = law, () => relation.Law = null);
            }
            else
            {
                Console.WriteLine($"R4   none    {relation.Name}: no whole-number law in the basis");
            }
        }

        File.WriteAllText(Path.Combine(dir, "current.entail.txt"), EntailOptimiser.Generate(plan, Provenance(journal)));
        File.WriteAllLines(Path.Combine(dir, "journal.jsonl"), journal.Select(j => j.ToJsonString()));
        WriteOverruled(dir, standing);
        Console.WriteLine($"end:        {standing}");
        Console.WriteLine($"overruled:  {standing.Overruled.Count} rows where the program and the model differ " +
                          $"({standing.Overruled.Count(o => o.Row.Margin < confidence)} of them undecided by the softmax)");
        return 0;
    }

    private static string Provenance(IEnumerable<JsonObject> journal) => string.Join('\n',
        journal.Where(j => j["accepted"]!.GetValue<bool>())
            .Select(j => $"{j["rule"]}: {j["detail"]}"));

    private static void Journal(List<JsonObject> journal, string rule, string detail, string author, bool accepted,
        string why, Standing before, Standing after) =>
        journal.Add(new JsonObject
        {
            ["rule"] = rule, ["detail"] = detail, ["author"] = author, ["accepted"] = accepted, ["why"] = why,
            ["before"] = before.ToString(), ["after"] = after.ToString(),
        });

    private static void WriteOverruled(string dir, Standing standing) =>
        File.WriteAllLines(Path.Combine(dir, "overruled.jsonl"), standing.Overruled
            .OrderByDescending(o => o.Row.Margin)
            .Select(o => new JsonObject
            {
                ["relation"] = o.Row.Relation,
                ["values"] = new JsonArray(o.Row.Values.Select(v => (JsonNode?)v).ToArray()),
                ["model"] = o.Row.Answer, ["program"] = o.Said, ["margin"] = Math.Round(o.Row.Margin, 4),
            }.ToJsonString()));

    // ── review / accept ─────────────────────────────────────────────────────

    /// <summary>
    /// Puts the documentation, the program and its standing in front of a
    /// reviewer. Any model will do, because everything it needs is in the
    /// prompt and nothing it says is trusted: with <c>--endpoint</c> this asks
    /// an OpenAI-compatible server and gates each answer; without, it writes
    /// the prompt for any other reviewer to answer through <c>entail accept</c>.
    /// </summary>
    private static int EntailReview(string[] args)
    {
        string dir = Arg(args, 0) ?? throw new CliException("entail review requires a work directory");
        double confidence = DoubleOption(args, "--confidence", 0.5);
        var rows = EvidenceRow.Load(Path.Combine(dir, "evidence.jsonl"));
        int rounds = IntOption(args, "--rounds", 3);
        string? endpoint = OptionValue(args, "--endpoint");
        for (int round = 1; round <= (endpoint is null ? 1 : rounds); round++)
        {
            string current = File.ReadAllText(Path.Combine(dir, "current.entail.txt"));
            var standing = EntailOptimiser.Measure(current, rows, confidence, BackgroundLine.Load(Path.Combine(dir, "background.jsonl")));
            string prompt = ReviewPrompt(dir, current, standing, args);
            File.WriteAllText(Path.Combine(dir, "review.prompt.md"), prompt);
            if (endpoint is null)
            {
                Console.WriteLine($"wrote:      {Path.Combine(dir, "review.prompt.md")}");
                Console.WriteLine("then:       amql-cli entail accept <dir> --candidate <file> --rule <rule> --why \"<reason>\"");
                return 0;
            }
            using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(20) };
            // Buffered, not streamed: not every inference server takes chunked requests.
            var body = JsonSerializer.Serialize(new
            {
                model = OptionValue(args, "--model") ?? "default",
                messages = new[] { new { role = "user", content = prompt } },
                temperature = 0,
            });
            var response = http.PostAsync(endpoint.TrimEnd('/') + "/chat/completions",
                new StringContent(body, Encoding.UTF8, "application/json")).GetAwaiter().GetResult();
            response.EnsureSuccessStatusCode();
            var answer = JsonNode.Parse(response.Content.ReadAsStringAsync().GetAwaiter().GetResult())!["choices"]![0]!["message"]!["content"]!.GetValue<string>();
            File.WriteAllText(Path.Combine(dir, $"review.{round}.answer.md"), answer);
            var block = Regex.Match(answer, "```[a-z]*\\r?\\n(.*?)```", RegexOptions.Singleline);
            if (!block.Success)
            {
                Console.WriteLine($"round {round}: the reviewer returned no program");
                continue;
            }
            var rule = Regex.Match(answer, @"\bR\d+\b");
            Accept(dir, block.Groups[1].Value, rule.Success ? rule.Value : "R?", "reviewer", endpoint, rows, confidence);
        }
        return 0;
    }

    private static string ReviewPrompt(string dir, string current, Standing standing, string[] args)
    {
        string Doc(string name)
        {
            var folder = new DirectoryInfo(OptionValue(args, "--docs") ?? Directory.GetCurrentDirectory());
            while (folder is not null)
            {
                var path = Path.Combine(folder.FullName, "docs", name);
                if (File.Exists(path))
                {
                    return File.ReadAllText(path);
                }
                folder = folder.Parent;
            }
            throw new CliException($"cannot find docs/{name}: pass --docs <repository>");
        }

        var text = new StringBuilder();
        text.AppendLine("You are reviewing a program in the Entail language. Everything you need is below.");
        text.AppendLine("Improve it by applying a rule from the optimisation guide. Reply with the rule you");
        text.AppendLine("applied, why, and the complete revised program in one fenced code block.");
        text.AppendLine();
        text.AppendLine(Doc("entail-language.md"));
        text.AppendLine(Doc("entail-optimisation.md"));
        text.AppendLine("# The program");
        text.AppendLine("```");
        text.AppendLine(current.TrimEnd());
        text.AppendLine("```");
        text.AppendLine();
        text.AppendLine("# Its standing");
        text.AppendLine(standing.ToString());
        text.AppendLine();
        text.AppendLine("# Rules already applied");
        var journal = Path.Combine(dir, "journal.jsonl");
        if (File.Exists(journal))
        {
            foreach (var line in File.ReadAllLines(journal))
            {
                var j = JsonNode.Parse(line)!;
                text.AppendLine($"- {j["rule"]} ({(j["accepted"]!.GetValue<bool>() ? "kept" : "refused")}): {j["detail"]}: {j["why"]}");
            }
        }
        text.AppendLine();
        text.AppendLine("# Where the program and the model differ (most decided first)");
        foreach (var (row, said) in standing.Overruled.OrderByDescending(o => o.Row.Margin).Take(20))
        {
            text.AppendLine($"- {row.Relation} {string.Join(" ", row.Values)}: model {row.Answer}, program {said}, margin {row.Margin:F2}");
        }
        return text.ToString();
    }

    private static int EntailAccept(string[] args)
    {
        string dir = Arg(args, 0) ?? throw new CliException("entail accept requires a work directory");
        string candidate = OptionValue(args, "--candidate") ?? throw new CliException("entail accept requires '--candidate <file>'");
        string rule = OptionValue(args, "--rule") ?? throw new CliException("entail accept requires '--rule <rule>'");
        var rows = EvidenceRow.Load(Path.Combine(dir, "evidence.jsonl"));
        return Accept(dir, File.ReadAllText(candidate), rule, OptionValue(args, "--author") ?? "reviewer",
            OptionValue(args, "--why") ?? string.Empty, rows, DoubleOption(args, "--confidence", 0.5)) ? 0 : 1;
    }

    private static bool Accept(string dir, string candidate, string rule, string author, string detail,
        IReadOnlyList<EvidenceRow> rows, double confidence)
    {
        var journal = new List<JsonObject>();
        var current = File.ReadAllText(Path.Combine(dir, "current.entail.txt"));
        var background = BackgroundLine.Load(Path.Combine(dir, "background.jsonl"));
        var before = EntailOptimiser.Measure(current, rows, confidence, background);
        bool accepted;
        string why;
        Standing after = before;
        try
        {
            after = EntailOptimiser.Measure(candidate, rows, confidence, background);
            (accepted, why) = EntailOptimiser.Gate(before, after);
        }
        catch (EntailException e)
        {
            accepted = false;
            why = "does not compile: " + e.Message;
        }
        Console.WriteLine($"{rule}: {(accepted ? "kept" : "refused")}: {why}");
        Console.WriteLine($"  before: {before}");
        Console.WriteLine($"  after:  {after}");
        Journal(journal, rule, detail, author, accepted, why, before, after);
        File.AppendAllLines(Path.Combine(dir, "journal.jsonl"), journal.Select(j => j.ToJsonString()));
        if (accepted)
        {
            File.WriteAllText(Path.Combine(dir, "current.entail.txt"), candidate);
            WriteOverruled(dir, after);
        }
        return accepted;
    }
}
