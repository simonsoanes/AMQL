using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Entail;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Cli;

// ── entail decompile: a large, lossy, unoptimised translation of a model ────
//
// The opposite of `extract`, which asks narrow questions. This reads what the
// model's units actually do across a corpus and writes all of it down, trained-
// in noise included, for the optimisation cycle to work on.

internal static partial class Program
{
    /// <summary>What one FFN neuron was seen doing.</summary>
    private sealed class UnitRecord
    {
        public int Layer, Index;
        public long Hits, ContrastHits;
        public double Mass, Signed, Best;
        public readonly Dictionary<int, (double Sum, int Count)> Tokens = new();
        public readonly Dictionary<long, (double Sum, int Count)> Pairs = new();
        /// <summary>What the model went on to say when this unit fired.</summary>
        public readonly Dictionary<int, int> Follows = new();
    }

    /// <summary>What a unit's firing says about a token, given how often the
    /// model says that token anyway. Share alone only finds common tokens.</summary>
    private static double VoteWeight(string rule, double given, double anyway) => rule switch
    {
        "share" => given,
        "lift" => given * Math.Log(given / anyway),
        "bayes" => Math.Log(given / anyway),
        _ => throw new CliException($"unknown --vote '{rule}': use share, lift or bayes"),
    };

    private static void SaveStudy(string path, Dictionary<long, UnitRecord> units, Dictionary<(int, int), HeadRecord> heads,
        Dictionary<int, long> overall, long studied, long contrasted)
    {
        using var writer = new BinaryWriter(new BufferedStream(File.Create(path), 1 << 20));
        writer.Write(studied);
        writer.Write(contrasted);
        writer.Write(overall.Count);
        foreach (var (token, count) in overall)
        {
            writer.Write(token);
            writer.Write(count);
        }
        writer.Write(units.Count);
        foreach (var unit in units.Values)
        {
            writer.Write(unit.Layer);
            writer.Write(unit.Index);
            writer.Write(unit.Hits);
            writer.Write(unit.ContrastHits);
            writer.Write(unit.Mass);
            writer.Write(unit.Signed);
            writer.Write(unit.Best);
            writer.Write(unit.Tokens.Count);
            foreach (var (token, value) in unit.Tokens)
            {
                writer.Write(token);
                writer.Write(value.Sum);
                writer.Write(value.Count);
            }
            writer.Write(unit.Pairs.Count);
            foreach (var (pair, value) in unit.Pairs)
            {
                writer.Write(pair);
                writer.Write(value.Sum);
                writer.Write(value.Count);
            }
            writer.Write(unit.Follows.Count);
            foreach (var (token, count) in unit.Follows)
            {
                writer.Write(token);
                writer.Write(count);
            }
        }
        writer.Write(heads.Count);
        foreach (var ((layer, index), head) in heads)
        {
            writer.Write(layer);
            writer.Write(index);
            writer.Write(head.First);
            writer.Write(head.Total);
            foreach (double offset in head.Offsets)
            {
                writer.Write(offset);
            }
        }
    }

    private static (long Studied, long Contrasted) LoadStudy(string path, Dictionary<long, UnitRecord> units,
        Dictionary<(int, int), HeadRecord> heads, Dictionary<int, long> overall)
    {
        using var reader = new BinaryReader(new BufferedStream(File.OpenRead(path), 1 << 20));
        long studied = reader.ReadInt64(), contrasted = reader.ReadInt64();
        for (int n = reader.ReadInt32(); n > 0; n--)
        {
            int token = reader.ReadInt32();
            overall[token] = reader.ReadInt64();
        }
        for (int n = reader.ReadInt32(); n > 0; n--)
        {
            var unit = new UnitRecord { Layer = reader.ReadInt32(), Index = reader.ReadInt32() };
            unit.Hits = reader.ReadInt64();
            unit.ContrastHits = reader.ReadInt64();
            unit.Mass = reader.ReadDouble();
            unit.Signed = reader.ReadDouble();
            unit.Best = reader.ReadDouble();
            for (int k = reader.ReadInt32(); k > 0; k--)
            {
                int token = reader.ReadInt32();
                unit.Tokens[token] = (reader.ReadDouble(), reader.ReadInt32());
            }
            for (int k = reader.ReadInt32(); k > 0; k--)
            {
                long pair = reader.ReadInt64();
                unit.Pairs[pair] = (reader.ReadDouble(), reader.ReadInt32());
            }
            for (int k = reader.ReadInt32(); k > 0; k--)
            {
                int token = reader.ReadInt32();
                unit.Follows[token] = reader.ReadInt32();
            }
            units[(long)unit.Layer << 32 | (uint)unit.Index] = unit;
        }
        for (int n = reader.ReadInt32(); n > 0; n--)
        {
            int layer = reader.ReadInt32(), index = reader.ReadInt32();
            var head = new HeadRecord { First = reader.ReadDouble(), Total = reader.ReadDouble() };
            for (int k = 0; k < head.Offsets.Length; k++)
            {
                head.Offsets[k] = reader.ReadDouble();
            }
            heads[(layer, index)] = head;
        }
        return (studied, contrasted);
    }

    private sealed class HeadRecord
    {
        public readonly double[] Offsets = new double[33];
        public double First, Total;
    }

    private static readonly JsonSerializerOptions ShowOptions = new()
    {
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    /// <summary>A token as a comment shows it.</summary>
    private static string Show(string token) => JsonSerializer.Serialize(token, ShowOptions);

    private static int EntailDecompile(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("entail decompile requires a container directory");
        string corpusDir = OptionValue(args, "--corpus") ?? throw new CliException("entail decompile requires '--corpus <dir>'");
        string outDir = OptionValue(args, "--out") ?? throw new CliException("entail decompile requires '--out <dir>'");
        string name = OptionValue(args, "--name") ?? "subject_raw";
        string? contrastPath = OptionValue(args, "--contrast");
        string? extraPath = OptionValue(args, "--extra");
        int corpusTokens = IntOption(args, "--corpus-tokens", 60000);
        int contrastTokens = IntOption(args, "--contrast-tokens", 8000);
        int perLayer = IntOption(args, "--per-layer", 24);
        int keep = IntOption(args, "--units", 4000);
        int seed = IntOption(args, "--seed", 1);
        Directory.CreateDirectory(outDir);

        string tokenizerDir = ResolveTokenizerDir(containerDir, args);
        var tokenizer = Tokenizer(tokenizerDir);
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, OptionValue(args, "--component") ?? "target", store);
        var session = new DecodeSession(plan, store, null,
            CudaShim.Enabled ? WeightWorkingSet.OnDemandBf16 : WeightWorkingSet.ResidentF32);
        var host = new DecodeSession(plan, store, null, WeightWorkingSet.ResidentF32);
        var runtime = session.Runtime;
        Console.WriteLine($"compute:    {(CudaShim.Enabled ? "CUDA" : "CPU")}");

        // The corpus, as chunks of whole files, plus any targeted text.
        var random = new Random(seed);
        var files = Directory.EnumerateFiles(corpusDir, "*.py", SearchOption.AllDirectories)
            .Where(f => !f.Contains("test", StringComparison.OrdinalIgnoreCase))
            .OrderBy(f => f, StringComparer.Ordinal).ToList();
        files = files.OrderBy(_ => random.Next()).ToList();
        var chunks = new List<int[]>();
        if (extraPath is not null)
        {
            var extra = File.ReadAllText(extraPath).ReplaceLineEndings("\n");
            for (int at = 0; at < extra.Length; at += 1000)
            {
                chunks.Add(tokenizer.EncodeToIds(extra.Substring(at, Math.Min(1000, extra.Length - at))).ToArray());
            }
        }
        // Targeted text comes on top of the budget, or it would crowd out the code.
        long planned = 0;
        foreach (var file in files)
        {
            if (planned >= corpusTokens)
            {
                break;
            }
            var text = File.ReadAllText(file).ReplaceLineEndings("\n");
            for (int at = 0; at + 200 < text.Length && planned < corpusTokens; at += Math.Max(1000, text.Length / 6))
            {
                var ids = tokenizer.EncodeToIds(text.Substring(at, Math.Min(1000, text.Length - at))).ToArray();
                chunks.Add(ids);
                planned += ids.Length;
            }
        }
        chunks = chunks.OrderBy(_ => random.Next()).ToList();
        // `--only-holdout` records what the model says and studies nothing: a test set.
        int held = HasOption(args, "--only-holdout") ? chunks.Count : Math.Max(4, chunks.Count / 20);
        var holdout = chunks.Take(held).ToList();
        var study = chunks.Skip(held).ToList();
        Console.WriteLine($"corpus:     {study.Sum(c => c.Length)} tokens to study, {holdout.Sum(c => c.Length)} held out, from {files.Count} files");

        var units = new Dictionary<long, UnitRecord>();
        var heads = new Dictionary<(int, int), HeadRecord>();
        int layers = plan.Layers.Count;
        var runningMean = new double[layers][];
        var seen = new long[layers];
        bool contrast = false;
        // One chunk goes through the model at once, so the capture sees every row.
        int[] reading = Array.Empty<int>();
        var fired = Array.Empty<List<UnitRecord>>();
        var tokenTable = new Dictionary<int, Dictionary<int, int>>();
        var pairTable = new Dictionary<long, Dictionary<int, int>>();
        var overall = new Dictionary<int, long>();

        runtime.FfnInputCapture = (layer, input) =>
        {
            if (plan.Layers[layer].Ffn?.Dense is not { } dense)
            {
                return;
            }
            var up = TensorOps.MatMulTransposedB(input, runtime.Weights.Matrix(dense.Up, dense.IntermediateSize, dense.HiddenSize));
            var gate = TensorOps.MatMulTransposedB(input, runtime.Weights.Matrix(dense.Gate!, dense.IntermediateSize, dense.HiddenSize));
            int n = dense.IntermediateSize;
            var act = new float[n];
            var mean = runningMean[layer] ??= new double[n];
            var excess = new double[n];
            for (int r = 0; r < input.Rows; r++)
            {
                long count = ++seen[layer];
                var upRow = up.Row(r);
                var gateRow = gate.Row(r);
                for (int i = 0; i < n; i++)
                {
                    float g = gateRow[i];
                    act[i] = upRow[i] * (g / (1f + MathF.Exp(-g)));
                    double magnitude = Math.Abs(act[i]);
                    // What is unusual for this neuron, not what it always does.
                    excess[i] = magnitude - mean[i];
                    mean[i] += (magnitude - mean[i]) / count;
                }
                var strongest = new PriorityQueue<int, double>();
                for (int i = 0; i < n; i++)
                {
                    if (strongest.Count < perLayer)
                    {
                        strongest.Enqueue(i, excess[i]);
                    }
                    else if (strongest.TryPeek(out _, out double least) && excess[i] > least)
                    {
                        strongest.EnqueueDequeue(i, excess[i]);
                    }
                }
                int current = reading[r], previous = r > 0 ? reading[r - 1] : -1;
                foreach (var (i, _) in strongest.UnorderedItems)
                {
                    long key = (long)layer << 32 | (uint)i;
                    if (!units.TryGetValue(key, out var unit))
                    {
                        units[key] = unit = new UnitRecord { Layer = layer, Index = i };
                    }
                    if (contrast)
                    {
                        unit.ContrastHits++;
                        continue;
                    }
                    double level = Math.Abs(act[i]);
                    fired[r].Add(unit);
                    unit.Hits++;
                    unit.Mass += level;
                    unit.Signed += act[i];
                    unit.Best = Math.Max(unit.Best, level);
                    var t = unit.Tokens.GetValueOrDefault(current);
                    unit.Tokens[current] = (t.Sum + level, t.Count + 1);
                    if (previous >= 0)
                    {
                        long pair = (long)previous << 32 | (uint)current;
                        var p = unit.Pairs.GetValueOrDefault(pair);
                        unit.Pairs[pair] = (p.Sum + level, p.Count + 1);
                    }
                }
            }
        };

        var trace = new List<GenericRuntime.LayerHeadAttention>();
        runtime.AttentionTrace = trace;
        void Consume()
        {
            foreach (var entry in trace)
            {
                if (!heads.TryGetValue((entry.Layer, entry.Head), out var head))
                {
                    heads[(entry.Layer, entry.Head)] = head = new HeadRecord();
                }
                int last = entry.Weights.Length - 1;
                head.Total += 1;
                for (int k = 0; k <= Math.Min(32, last); k++)
                {
                    head.Offsets[k] += entry.Weights[last - k];
                }
                head.First += entry.Weights[0];
            }
            trace.Clear();
        }

        static void Count<TKey>(Dictionary<TKey, Dictionary<int, int>> table, TKey key, int said) where TKey : notnull
        {
            if (!table.TryGetValue(key, out var row))
            {
                table[key] = row = new Dictionary<int, int>();
            }
            row[said] = row.GetValueOrDefault(said) + 1;
        }

        static int[] Said(Tensor2D logits)
        {
            var said = new int[logits.Rows];
            Parallel.For(0, logits.Rows, r => said[r] = Decide(logits.Row(r)).Best);
            return said;
        }

        var clock = Stopwatch.StartNew();
        long done = 0;
        void Read(IEnumerable<int[]> list, long limit)
        {
            long read = 0;
            foreach (var ids in list)
            {
                session.Reset();
                reading = ids;
                fired = ids.Select(_ => new List<UnitRecord>()).ToArray();
                if (contrast)
                {
                    session.PrefillHidden(ids);
                }
                else
                {
                    var said = Said(session.PrefillLogits(ids));
                    for (int i = 0; i < ids.Length; i++)
                    {
                        foreach (var unit in fired[i])
                        {
                            unit.Follows[said[i]] = unit.Follows.GetValueOrDefault(said[i]) + 1;
                        }
                        overall[said[i]] = overall.GetValueOrDefault(said[i]) + 1;
                        Count(tokenTable, ids[i], said[i]);
                        if (i > 0)
                        {
                            Count(pairTable, (long)ids[i - 1] << 32 | (uint)ids[i], said[i]);
                        }
                    }
                }
                // The trace holds each chunk's last row only: a sample, for comments.
                Consume();
                read += ids.Length;
                done += ids.Length;
                if (done % 5000 < ids.Length)
                {
                    Console.WriteLine($"  {done} tokens, {done / Math.Max(clock.Elapsed.TotalSeconds, 1):F0}/s, {units.Count} units seen");
                }
                if (read >= limit)
                {
                    break;
                }
            }
        }

        // A study is the slow part and writing it down is not, so it is kept:
        // `--from` writes a program from a saved study without reading anything.
        long studied, contrasted;
        string? savedStudy = OptionValue(args, "--from");
        if (savedStudy is not null)
        {
            (studied, contrasted) = LoadStudy(savedStudy, units, heads, overall);
            Console.WriteLine($"study:      {savedStudy}, {studied} tokens, {units.Count} units");
        }
        else
        {
            CliProgress.Phase("entail decompile");
            Read(study, long.MaxValue);
            studied = done;
            if (contrastPath is not null)
            {
                contrast = true;
                var text = File.ReadAllText(contrastPath);
                var slices = Enumerable.Range(0, contrastTokens / 200 + 1)
                    .Select(i => tokenizer.EncodeToIds(text.Substring(random.Next(text.Length - 1100), 1000)).ToArray());
                Read(slices, contrastTokens);
            }
            contrasted = done - studied;
            runtime.FfnInputCapture = null;
            runtime.AttentionTrace = null;

            // What the model predicts on held-out text: the measure of fidelity.
            var heldLines = new List<string>();
            int heldTotal = 0, tokenRight = 0, pairRight = 0;
            static int Likeliest(Dictionary<int, int> row) => row.MaxBy(x => x.Value).Key;
            foreach (var ids in holdout)
            {
                session.Reset();
                var tokens = ids.Select(id => tokenizer.TokenInfo(id).DecodedText ?? string.Empty).ToList();
                var next = new List<string>();
                var saidHeld = Said(session.PrefillLogits(ids));
                for (int i = 0; i < ids.Length; i++)
                {
                    int said = saidHeld[i];
                    next.Add(tokenizer.TokenInfo(said).DecodedText ?? string.Empty);
                    heldTotal++;
                    int byToken = tokenTable.TryGetValue(ids[i], out var seenToken) ? Likeliest(seenToken) : -1;
                    int byPair = i > 0 && pairTable.TryGetValue((long)ids[i - 1] << 32 | (uint)ids[i], out var seenPair)
                        ? Likeliest(seenPair) : byToken;
                    tokenRight += byToken == said ? 1 : 0;
                    pairRight += byPair == said ? 1 : 0;
                }
                heldLines.Add(new BackgroundLine(tokens, next).ToJson().ToJsonString());
            }
            File.WriteAllLines(Path.Combine(outDir, "holdout.jsonl"), heldLines);
            // The most any rule on this token and the one before could score, near enough.
            Console.WriteLine($"reference:  next token by token {100.0 * tokenRight / Math.Max(heldTotal, 1):F1}%, by token pair {100.0 * pairRight / Math.Max(heldTotal, 1):F1}%, of {heldTotal} held-out positions");
            SaveStudy(Path.Combine(outDir, "study.bin"), units, heads, overall, studied, contrasted);
        }

        string voteRule = OptionValue(args, "--vote") ?? "lift";
        double allSaid = Math.Max(overall.Values.Sum(), 1);

        // Rank by how much of the text each unit accounts for. Selectivity is
        // reported, not used: unrelated trained-in units stay in on purpose.
        var selected = units.Values.Where(u => u.Hits >= 4)
            .OrderByDescending(u => u.Mass).Take(keep).ToList();
        if (HasOption(args, "--cover"))
        {
            // Mass favours what is common. Also keep, for every token seen,
            // the unit that responds to it most, so that rare names survive.
            var strongest = new Dictionary<int, (UnitRecord Unit, double Sum)>();
            foreach (var unit in units.Values)
            {
                foreach (var (token, value) in unit.Tokens)
                {
                    if (!strongest.TryGetValue(token, out var best) || value.Sum > best.Sum)
                    {
                        strongest[token] = (unit, value.Sum);
                    }
                }
            }
            var chosen = new HashSet<UnitRecord>(selected);
            foreach (var (unit, _) in strongest.Values)
            {
                if (chosen.Add(unit))
                {
                    selected.Add(unit);
                }
            }
        }
        Console.WriteLine($"units:      {units.Count} seen, {selected.Count} written");

        // Effects: each unit's output column read through the final norm and
        // the output head, as logits per unit of activation.
        var head0 = host.Runtime;
        var finalNorm = plan.FinalNorm ?? throw new CliException("this plan has no final norm");
        var gamma = head0.Weights.Vector(finalNorm.Weight, plan.HiddenSize);
        float offset = finalNorm.WeightOffset;
        var table = head0.Weights.Matrix(plan.Embedding!.Table, plan.Embedding.VocabSize, plan.HiddenSize);
        int hidden = plan.HiddenSize;
        const double FinalRms = 3.6;     // the residual's RMS at the final norm, measured

        (int Id, double Logit)[] Effect(float[] direction, int count)
        {
            var scaled = new float[hidden];
            for (int d = 0; d < hidden; d++)
            {
                scaled[d] = direction[d] * (gamma[d] + offset);
            }
            var logits = new float[table.Rows];
            Parallel.For(0, table.Rows, t =>
            {
                float sum = 0;
                var row = table.Data.AsSpan(t * hidden, hidden);
                for (int d = 0; d < hidden; d++)
                {
                    sum += row[d] * scaled[d];
                }
                logits[t] = sum;
            });
            var best = new PriorityQueue<int, float>();
            for (int t = 0; t < logits.Length; t++)
            {
                best.Enqueue(t, logits[t]);
                if (best.Count > count)
                {
                    best.Dequeue();
                }
            }
            var top = new List<(int, double)>();
            while (best.TryDequeue(out int id, out float logit))
            {
                top.Add((id, logit / FinalRms));
            }
            top.Reverse();
            return top.ToArray();
        }

        var text2 = new StringBuilder();
        var vocabulary = new SortedDictionary<string, int>(StringComparer.Ordinal);
        string Piece(int id) => tokenizer.TokenInfo(id).DecodedText ?? string.Empty;
        int Code(string token)
        {
            if (!EntailOptimiser.CanQuote(token))
            {
                return 0;
            }
            if (!vocabulary.TryGetValue(token, out int code))
            {
                vocabulary[token] = code = vocabulary.Count + 1;
            }
            return code;
        }

        int statements = 0;
        var body = new StringBuilder();
        var summary = new JsonArray();
        foreach (var layerGroup in selected.GroupBy(u => u.Layer).OrderBy(g => g.Key))
        {
            var dense = plan.Layers[layerGroup.Key].Ffn!.Dense!;
            var down = head0.Weights.Matrix(dense.Down, dense.HiddenSize, dense.IntermediateSize);
            body.AppendLine();
            body.AppendLine($"# ═══ Layer {layerGroup.Key}: {layerGroup.Count()} units ═══");
            foreach (var unit in layerGroup.OrderByDescending(u => u.Mass))
            {
                double subjectRate = (double)unit.Hits / Math.Max(studied, 1);
                double contrastRate = (unit.ContrastHits + 0.5) / Math.Max(contrasted, 1);
                double ratio = subjectRate / contrastRate;
                string lane = $"L{unit.Layer}_n{unit.Index}";
                double peak = unit.Tokens.Values.Where(t => t.Count >= 2).Select(t => t.Sum / t.Count).DefaultIfEmpty(unit.Best).Max();
                int least = HasOption(args, "--cover") ? 1 : 2;
                var triggers = unit.Tokens.Where(t => t.Value.Count >= least && t.Value.Sum / t.Value.Count >= 0.3 * peak)
                    .OrderByDescending(t => t.Value.Sum).Take(12).ToList();
                var pairs = unit.Pairs.Where(p => p.Value.Count >= 2 && p.Value.Sum / p.Value.Count >= 0.5 * peak)
                    .OrderByDescending(p => p.Value.Sum).Take(6).ToList();
                if (triggers.Count + pairs.Count == 0)
                {
                    continue;
                }

                var direction = new float[hidden];
                float sign = unit.Signed >= 0 ? 1f : -1f;
                for (int d = 0; d < hidden; d++)
                {
                    direction[d] = sign * down.Data[d * dense.IntermediateSize + unit.Index];
                }
                var promotes = HasOption(args, "--no-columns") ? Array.Empty<(int Id, double Logit)>() : Effect(direction, 6);

                body.AppendLine();
                body.AppendLine($"# L{unit.Layer} neuron {unit.Index}: fired {unit.Hits}x, {(ratio >= 3 ? "subject-specific" : ratio >= 1.2 ? "leans subject" : "general")} ({ratio:F1}x prose)");
                body.AppendLine($"state {lane}");
                int written = 0;
                foreach (var (token, value) in triggers)
                {
                    int code = Code(Piece(token));
                    if (code == 0)
                    {
                        continue;
                    }
                    int level = Math.Clamp((int)Math.Round(8 * value.Sum / value.Count / peak), 1, 8);
                    body.AppendLine($"entail {lane} = {level} when sym == {code}   # {Show(Piece(token))} x{value.Count}");
                    written++;
                }
                foreach (var (pair, value) in pairs)
                {
                    int a = Code(Piece((int)(pair >> 32))), b = Code(Piece((int)(pair & 0xFFFFFFFF)));
                    if (a == 0 || b == 0)
                    {
                        continue;
                    }
                    int level = Math.Clamp((int)Math.Round(8 * value.Sum / value.Count / peak), 1, 8);
                    body.AppendLine($"entail {lane} = {level} when prev == {a}, sym == {b}   # {Show(Piece((int)(pair >> 32)))} {Show(Piece((int)(pair & 0xFFFFFFFF)))} x{value.Count}");
                    written++;
                }
                statements += written;
                // The vote is what the model was seen to say when this unit fired.
                // Its output column alone says little before the last few layers:
                // that is kept as a comment, because it is what the weights hold.
                if (promotes.Length > 0)
                {
                    body.AppendLine($"# column promotes {string.Join(" ", promotes.Select(x => Show(Piece(x.Id))))}");
                }
                double firings = Math.Max(unit.Follows.Values.Sum(), 1);
                var weighed = unit.Follows.Where(f => f.Value >= least)
                    .Select(f => (Id: f.Key, Times: f.Value, Weight: VoteWeight(voteRule, f.Value / firings, overall.GetValueOrDefault(f.Key, 1) / allSaid)))
                    .Where(v => v.Weight > 0).OrderByDescending(v => v.Weight).Take(6);
                foreach (var (id, times, weight) in weighed)
                {
                    string token = Piece(id);
                    if (Code(token) != 0)
                    {
                        body.AppendLine($"vote {EntailOptimiser.Quoted(token)} by {(weight / 8).ToString("F4", CultureInfo.InvariantCulture)}*{lane}   # x{times}");
                        statements++;
                    }
                }
                summary.Add(new JsonObject
                {
                    ["unit"] = lane, ["hits"] = unit.Hits, ["ratio"] = Math.Round(ratio, 2),
                    ["triggers"] = new JsonArray(triggers.Take(6).Select(t => (JsonNode?)Piece(t.Key)).ToArray()),
                    ["promotes"] = new JsonArray(promotes.Select(p => (JsonNode?)Piece(p.Id)).ToArray()),
                });
            }
        }

        // Heads: only where a head mostly looks at one offset can it be
        // written as `attend back`. The rest have no Entail form yet.
        var headText = new StringBuilder();
        var used = new HashSet<int> { 1 };
        foreach (var ((layer, index), head) in heads.OrderBy(h => h.Key.Item1).ThenBy(h => h.Key.Item2))
        {
            int best = Enumerable.Range(0, 33).OrderByDescending(k => head.Offsets[k]).First();
            double share = head.Offsets[best] / Math.Max(head.Total, 1);
            double first = head.First / Math.Max(head.Total, 1);
            string pattern = first >= 0.5 ? $"mostly the first token ({first:P0})" : $"offset {best} ({share:P0})";
            headText.AppendLine($"# L{layer} head {index}: {pattern}");
            if (share >= 0.3 && best > 0 && first < 0.5)
            {
                used.Add(best);
            }
        }

        var linear = plan.Layers.Count(l => l.LinearAttention is not null);
        text2.AppendLine($"# RAW EXTRACT of {container.Index.Model}: not optimised, and not meant to be read as correct.");
        text2.AppendLine($"# {studied} tokens studied, {selected.Count} of {units.Count} neurons seen, {statements} rule lines.");
        text2.AppendLine("# Each unit is one FFN neuron, reduced to the tokens seen to fire it and what");
        text2.AppendLine("# the model said next. `vote` is soft, like the softmax it came from.");
        text2.AppendLine($"# Not represented: {linear} linear-attention layers, and how neurons read each other.");
        text2.AppendLine();
        text2.AppendLine($"program {name}");
        text2.AppendLine();
        text2.AppendLine("state sym");
        var header = new StringBuilder();
        foreach (var (token, code) in vocabulary)
        {
            header.AppendLine($"token {EntailOptimiser.Quoted(token)} means sym = {code}");
        }
        text2.Append(header);
        text2.AppendLine();
        text2.AppendLine("# Attention, as far as a fixed offset can say it.");
        text2.Append(headText);
        text2.AppendLine("state prev");
        text2.AppendLine("attend back 1");
        text2.AppendLine("    prev <- sym");
        text2.AppendLine("end");
        text2.Append(body);
        string path = Path.Combine(outDir, "raw.entail.txt");
        File.WriteAllText(path, text2.ToString());
        File.WriteAllText(Path.Combine(outDir, "units.json"), summary.ToJsonString(EntailJson));
        Console.WriteLine($"wrote:      {path}: {text2.Length / 1024} KB, {statements} rule lines, {vocabulary.Count} tokens");
        return 0;
    }

    /// <summary>How much of a model's behaviour a soft program reproduces: at
    /// positions of held-out text, does its top vote match the model's next
    /// token, compared with simply repeating the commonest answer?</summary>
    private static int EntailFidelity(string[] args)
    {
        var path = Arg(args, 0) ?? throw new CliException("entail fidelity requires a program");
        var lines = BackgroundLine.Load(OptionValue(args, "--holdout") ?? throw new CliException("entail fidelity requires '--holdout <file>'"));
        int limit = IntOption(args, "--lines", 40);
        var clock = Stopwatch.StartNew();
        var p = EntailCompiler.Compile(File.ReadAllText(path), int.MaxValue);
        Console.WriteLine($"compiled:   {p.Statements} statements, {p.Ops.Count(o => o is Neuron)} neurons, {p.Lanes.Count} lanes, {p.Layers} layers ({clock.Elapsed.TotalSeconds:F0}s)");
        int positions = 0, spoke = 0, right = 0;
        var answers = new System.Collections.Concurrent.ConcurrentBag<(int Total, int Spoke, int Right)>();
        var rightTokens = new System.Collections.Concurrent.ConcurrentBag<string>();
        Parallel.ForEach(lines.Take(limit), line =>
        {
            var lanes = EntailInterpreter.Run(p, line.Tokens);
            int s = 0, r = 0;
            for (int t = 0; t < lanes.Length; t++)
            {
                var said = EntailInterpreter.Output(p, lanes[t]);
                if (!string.IsNullOrEmpty(said))
                {
                    s++;
                    if (said == line.Next[t])
                    {
                        r++;
                        rightTokens.Add(said);
                    }
                }
            }
            answers.Add((lanes.Length, s, r));
        });
        foreach (var (total, s, r) in answers)
        {
            positions += total;
            spoke += s;
            right += r;
        }
        var common = lines.SelectMany(l => l.Next).GroupBy(x => x).OrderByDescending(g => g.Count()).First();
        int commonRight = lines.Take(limit).SelectMany(l => l.Next).Count(x => x == common.Key);
        Console.WriteLine($"positions:  {positions}, program spoke at {spoke} ({100.0 * spoke / positions:F1}%)");
        Console.WriteLine($"agreement:  {right} of {positions} ({100.0 * right / positions:F1}%), {100.0 * right / Math.Max(spoke, 1):F1}% where it spoke");
        Console.WriteLine($"baseline:   always {Show(common.Key)}: {100.0 * commonRight / positions:F1}%");

        // A program can score well on newlines and commas alone, so also say
        // how it does on everything but the commonest things the model says.
        var usual = lines.Take(limit).SelectMany(l => l.Next).GroupBy(x => x)
            .OrderByDescending(g => g.Count()).Take(20).Select(g => g.Key).ToHashSet();
        int rare = lines.Take(limit).SelectMany(l => l.Next).Count(x => !usual.Contains(x));
        int rareRight = rightTokens.Count(x => !usual.Contains(x));
        Console.WriteLine($"past the 20 commonest: {rareRight} of {rare} ({100.0 * rareRight / Math.Max(rare, 1):F1}%), {rightTokens.Select(x => x).Where(x => !usual.Contains(x)).Distinct().Count()} different tokens right");
        Console.WriteLine($"right most: {string.Join(" ", rightTokens.GroupBy(x => x).OrderByDescending(g => g.Count()).Take(14).Select(g => $"{Show(g.Key)}x{g.Count()}"))}");
        return 0;
    }
}
