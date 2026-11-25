using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Cli;

// ── context bench: how well a model uses a context of a given length ───────
//
// One long document (real text, with a fact planted every few thousand tokens)
// is read once, stopping at each length on the way. At each stop, from the same
// state: questions about facts spread evenly over everything read so far, the
// perplexity of a held-out passage, and how well the model repeats passages it
// read earlier. Reading once and rewinding is what makes a million tokens
// affordable; the numbers at 8K and at 1M come from the same run.
//
// Everything is asked of the assistant's turn: a chat model's predictions inside
// a user turn were never trained, so text scored there measures nothing (a
// passage it can quote perfectly costs several nats a token there). --plain
// reads the document as bare text instead, and scores the held-out passage's
// perplexity and the repeats as continuations of it.

internal static partial class Program
{
    private static int ContextCommand(string[] args) => Arg(args, 0) switch
    {
        "bench" => ContextBench(args[1..]),
        _ => throw new CliException("context takes 'bench' (see --help)"),
    };

    private sealed record Needle(string Name, string Code, int At);

    private static int ContextBench(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("context bench requires a container directory");
        string textSpec = OptionValue(args, "--text") ?? throw new CliException("context bench requires '--text <file or folder>[;...]' to read");
        var lengths = (OptionValue(args, "--lengths") ?? "8K,32K,128K,256K").Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
            .Select(t => ContextExtension.TryParseLength(t, out long n) ? (int)n : throw new CliException($"'{t}' is not a length")).Order().ToList();
        bool plain = HasOption(args, "--plain");
        int spacing = IntOption(args, "--needle-every", 2048);
        int asked = IntOption(args, "--questions", 24);
        int score = IntOption(args, "--score-tokens", 1024);
        // --focus N: also answer each question after reading again the N passages of the
        // document the model's own attention went to as it read the question.
        int focus = IntOption(args, "--focus", 0);
        int seed = IntOption(args, "--seed", 1);
        string? outFile = OptionValue(args, "--out");
        int from = OptionValue(args, "--text-offset") is { } off && ContextExtension.TryParseLength(off, out long o0) ? (int)o0 : 0;

        var tokenizer = Tokenizer(containerDir);
        var stop = ResolveStopTokens(containerDir, tokenizer);
        string template = ChatTemplate.Load(containerDir) ?? (plain ? "" : throw new CliException("the container has no chat template; use --plain"));

        // The prompt around the document: rendered once, cut where the document goes.
        const string Cut = "DOC", Ask = "ASK";
        string frame = plain ? Cut + Ask : ChatTemplate.ApplyMessages(template, new[]
        {
            new ChatMessage("user", $"Read the document, then answer the question after it.\n\n<document>\n{Cut}\n</document>\n\n{Ask}"),
        }, enableThinking: false);
        int cut = frame.IndexOf(Cut, StringComparison.Ordinal), ask = frame.IndexOf(Ask, StringComparison.Ordinal);
        if (cut < 0 || ask < cut)
        {
            throw new CliException("the chat template does not carry the document through");
        }
        int[] head = plain ? Array.Empty<int>() : tokenizer.EncodeToIds(frame[..cut]).ToArray();
        string middle = frame[(cut + Cut.Length)..ask], tail = frame[(ask + Ask.Length)..];

        // Document tokens, with a fact at every `spacing` tokens.
        int longest = lengths[^1];
        int reserve = 4096;   // the frame, a question and its answer, the scored passages
        var corpus = ReadCorpus(textSpec, tokenizer, (long)from + longest + score);
        if (corpus.Length < from + longest + score)
        {
            throw new CliException($"--text holds {corpus.Length:N0} tokens; {from + longest + score:N0} are needed (with --text-offset)");
        }
        int[] heldOut = corpus.AsSpan(corpus.Length - score).ToArray();   // never in the document
        var random = new Random(seed);
        var names = NeedleNames(random);
        var doc = new List<int>(longest);
        doc.AddRange(head);
        var needles = new List<Needle>();
        int used = from;
        while (doc.Count < longest - reserve && !plain)
        {
            int until = Math.Min(longest - reserve, head.Length + (needles.Count + 1) * spacing - spacing / 2);
            int take = Math.Max(0, until - doc.Count);
            doc.AddRange(corpus.AsSpan(used, take));
            used += take;
            if (doc.Count >= longest - reserve)
            {
                break;
            }
            string code = random.Next(10000, 100000).ToString(CultureInfo.InvariantCulture);
            if (needles.Count == names.Count)
            {
                throw new CliException("too many needles; raise --needle-every");
            }
            string name = names[needles.Count];
            needles.Add(new Needle(name, code, doc.Count));
            doc.AddRange(tokenizer.EncodeToIds($"\n\nNote: the maintenance code for the {name} is {code}.\n\n"));
        }
        if (plain)
        {
            doc.AddRange(corpus.AsSpan(from, longest - reserve));
        }
        var all = doc.ToArray();

        using var container = Vindex3Container.Open(containerDir);
        foreach (var note in container.ContextNotes)
        {
            Console.WriteLine($"context:    {note}");
        }
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, OptionValue(args, "--component") ?? "target", store);
        using var session = new DecodeSession(plan, store);
        session.Prefill(new[] { all[0] });
        var engine = session.DeviceEngine() ?? throw new CliException($"context bench runs on the GPU engine ({session.GpuRefusal})");
        Console.WriteLine($"model:      {containerDir}, KV {engine.KvFormat}, {needles.Count} facts in {all.Length:N0} tokens");

        var rows = new JsonArray();
        var seq = engine.NewSequence();
        // One reading passes every length, so it is planned for the longest: an extension's
        // frequencies, when the longest needs them, are measured at the short lengths too.
        engine.Plan(seq, longest);
        Console.WriteLine(plain
            ? "  length     ppl   repeat NLL/token (mean of 8 depths)   read s    tok/s"
            : "  length  recall  by depth (quarters)      quote NLL/token (by depth quarter)       read s    tok/s   decode tok/s");
        foreach (int length in lengths)
        {
            int target = Math.Min(length - reserve, all.Length);
            var sw = Stopwatch.StartNew();
            int before = seq.Position;
            if (target > before)
            {
                engine.Prefill(seq, all.AsSpan(before, target - before));
            }
            double readSeconds = sw.Elapsed.TotalSeconds;
            engine.Mark(seq);

            // Facts spread evenly over what was read.
            var seen = needles.Where(n => n.At + 64 < target).ToList();
            var picks = plain ? new List<Needle>() : Enumerable.Range(0, Math.Min(asked, seen.Count)).Select(i => seen[(int)((i + 0.5) * seen.Count / Math.Min(asked, seen.Count))]).ToList();
            var quarters = new int[4];
            var quarterOf = new int[4];
            int right = 0, decoded = 0, rightFocused = 0, rightFresh = 0, foundRaw = 0, foundAbove = 0;
            float[]? idle = null;
            double decodeSeconds = 0;
            var misses = new JsonArray();
            foreach (var needle in picks)
            {
                engine.Rewind(seq);
                var question = tokenizer.EncodeToIds(middle + $"What is the maintenance code for the {needle.Name}? Reply with the code only." + tail).ToArray();
                var logits = engine.Prefill(seq, question);
                var answer = new List<int>();
                sw.Restart();
                for (int t = 0; t < 16; t++)
                {
                    int next = Sampler.ArgMax(logits);
                    if (stop.Contains(next))
                    {
                        break;
                    }
                    answer.Add(next);
                    logits = engine.Prefill(seq, new[] { next });
                }
                decodeSeconds += sw.Elapsed.TotalSeconds;
                decoded += answer.Count;
                string said = tokenizer.DecodeText(answer.ToArray());
                string? saidFocused = null;
                if (focus > 0)
                {
                    if (idle is null)
                    {
                        // Where attention goes whatever is asked (the text's start, its landmarks).
                        engine.Rewind(seq);
                        idle = engine.ReadAttention(seq, tokenizer.EncodeToIds(middle + "Reply with OK." + tail).ToArray()).Mass;
                    }
                    engine.Rewind(seq);
                    var mass = engine.ReadAttention(seq, question).Mass;
                    var raw = AttentionFocus.Passages(mass, null, head.Length, target, focus);
                    var above = AttentionFocus.Passages(mass, idle, head.Length, target, focus);
                    bool Holds(List<(int From, int To)> ps) => ps.Any(x => x.From <= needle.At + 12 && needle.At + 12 < x.To);
                    foundRaw += Holds(raw) ? 1 : 0;
                    foundAbove += Holds(above) ? 1 : 0;
                    engine.Rewind(seq);
                    var again = new List<int>(tokenizer.EncodeToIds(middle + "Passages of the document that may matter for the question, in order:\n"));
                    foreach (var (a, b) in above)
                    {
                        again.AddRange(tokenizer.EncodeToIds("\n[...]\n"));
                        again.AddRange(all.AsSpan(a, b - a));
                    }
                    again.AddRange(tokenizer.EncodeToIds($"\n[...]\n\nWhat is the maintenance code for the {needle.Name}? Reply with the code only." + tail));
                    logits = engine.Prefill(seq, again.ToArray());
                    var focused = new List<int>();
                    for (int t = 0; t < 16; t++)
                    {
                        int next = Sampler.ArgMax(logits);
                        if (stop.Contains(next))
                        {
                            break;
                        }
                        focused.Add(next);
                        logits = engine.Prefill(seq, new[] { next });
                    }
                    saidFocused = tokenizer.DecodeText(focused.ToArray());
                    rightFocused += saidFocused.Contains(needle.Code, StringComparison.Ordinal) ? 1 : 0;
                    // The same passages in a fresh conversation of their own, without the long context.
                    var alone = engine.NewSequence();
                    var freshPrompt = new List<int>(head);
                    foreach (var (a, b) in above)
                    {
                        freshPrompt.AddRange(tokenizer.EncodeToIds("\n[...]\n"));
                        freshPrompt.AddRange(all.AsSpan(a, b - a));
                    }
                    freshPrompt.AddRange(tokenizer.EncodeToIds("\n[...]\n"));
                    freshPrompt.AddRange(question);
                    logits = engine.Prefill(alone, freshPrompt.ToArray());
                    var fresh = new List<int>();
                    for (int t = 0; t < 16; t++)
                    {
                        int next = Sampler.ArgMax(logits);
                        if (stop.Contains(next))
                        {
                            break;
                        }
                        fresh.Add(next);
                        logits = engine.Prefill(alone, new[] { next });
                    }
                    engine.Release(alone);
                    string saidFresh = tokenizer.DecodeText(fresh.ToArray());
                    rightFresh += saidFresh.Contains(needle.Code, StringComparison.Ordinal) ? 1 : 0;
                }
                int quarter = Math.Min(3, (int)(4L * needle.At / target));
                quarterOf[quarter]++;
                if (said.Contains(needle.Code, StringComparison.Ordinal))
                {
                    right++;
                    quarters[quarter]++;
                }
                else
                {
                    misses.Add(new JsonObject { ["name"] = needle.Name, ["code"] = needle.Code, ["at"] = needle.At, ["said"] = said });
                }
            }

            // Passages from 8 depths: 64 tokens to find each by, 128 scored.
            double nll = 0;
            if (plain)
            {
                engine.Rewind(seq);
                nll = ScorePassage(engine, seq, heldOut, 0);
            }
            var repeats = new double[8];
            for (int d = 0; d < repeats.Length; d++)
            {
                engine.Rewind(seq);
                int at = Math.Max(head.Length, Math.Min(target - 256, (int)((d + 0.5) / repeats.Length * target)));
                var passage = all.AsSpan(at, 192).ToArray();
                if (plain)
                {
                    repeats[d] = ScorePassage(engine, seq, passage, 64);
                    continue;
                }
                // The excerpt is cut at a token, so the quote carries straight on from it.
                engine.Prefill(seq, tokenizer.EncodeToIds(middle + "Quote the document from this excerpt on, word for word. Excerpt:\n\n").ToArray()
                    .Concat(passage.AsSpan(0, 64).ToArray()).Concat(tokenizer.EncodeToIds(tail)).ToArray());
                repeats[d] = ScorePassage(engine, seq, passage, 64);
            }
            engine.Rewind(seq);
            double repeat = repeats.Average();
            var byQuarter = Enumerable.Range(0, 4).Select(q => (repeats[2 * q] + repeats[2 * q + 1]) / 2).ToArray();
            var row = new JsonObject
            {
                ["length"] = target, ["asked"] = picks.Count, ["right"] = right,
                ["by_quarter"] = new JsonArray(quarters.Select((q, i) => (JsonNode)$"{q}/{quarterOf[i]}").ToArray()),
                ["ppl"] = plain ? Math.Exp(nll) : null,
                ["repeat_nll"] = new JsonArray(repeats.Select(r => (JsonNode)r).ToArray()),
                ["read_seconds"] = readSeconds, ["read_tokens"] = target - before,
                ["decode_tokens_per_second"] = decodeSeconds > 0 ? decoded / decodeSeconds : null,
                ["misses"] = misses,
            };
            if (focus > 0)
            {
                row["focus"] = focus;
                row["right_focused"] = rightFocused;
                row["right_fresh"] = rightFresh;
                row["passage_found"] = foundAbove;
                row["passage_found_raw"] = foundRaw;
            }
            rows.Add(row);
            Console.WriteLine(plain
                ? string.Create(CultureInfo.InvariantCulture, $"{target,8} {Math.Exp(nll),7:F3}   {repeat,8:F4}   {string.Join(' ', repeats.Select(r => r.ToString("F3", CultureInfo.InvariantCulture)))} {readSeconds,8:F1} {(target - before) / Math.Max(readSeconds, 1e-9),8:F0}")
                : string.Create(CultureInfo.InvariantCulture, $"{target,8} {right,4}/{picks.Count,-3} {string.Join(' ', quarters.Select((q, i) => $"{q}/{quarterOf[i]}")),-24} {repeat,8:F4}  ({string.Join(' ', byQuarter.Select(r => r.ToString("F3", CultureInfo.InvariantCulture)))}) {readSeconds,10:F1} {(target - before) / Math.Max(readSeconds, 1e-9),8:F0} {decoded / Math.Max(decodeSeconds, 1e-9),8:F1}"));
            if (focus > 0)
            {
                Console.WriteLine($"         focused on {focus} passages: {rightFocused}/{picks.Count} right read again in place, {rightFresh}/{picks.Count} in a fresh conversation; the fact's passage was among them {foundAbove}/{picks.Count} ({foundRaw} without taking off idle attention)");
            }
            if (outFile is not null)
            {
                File.WriteAllText(outFile, new JsonObject
                {
                    ["model"] = containerDir, ["kv"] = engine.KvFormat,
                    ["extension"] = ContextExtension.Requested is { } e ? $"{e.Length} {e.Method}" : null,
                    ["seed"] = seed, ["rows"] = rows.DeepClone(),
                }.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
            }
        }
        engine.Release(seq);
        return ExitOk;
    }

    /// <summary>Mean negative log-likelihood (nats a token) of a passage read on
    /// from where the sequence stands, after its first <paramref name="skip"/> + 1 tokens.
    /// </summary>
    private static double ScorePassage(GpuEngine engine, GpuSequence seq, int[] ids, int skip)
    {
        double sum = 0;
        int n = 0;
        for (int at = 0; at + 1 < ids.Length; at += 8)
        {
            int rows = Math.Min(8, ids.Length - 1 - at);
            engine.Verify(seq, ids.AsSpan(at, rows));
            for (int r = 0; r < rows; r++)
            {
                if (at + r < skip)
                {
                    continue;
                }
                var row = engine.HostLogits(r).ToArray();
                float max = row.Max();
                double z = 0;
                foreach (float v in row)
                {
                    z += Math.Exp(v - max);
                }
                sum -= row[ids[at + r + 1]] - max - Math.Log(z);
                n++;
            }
            engine.Accept(seq, rows);
        }
        return sum / Math.Max(1, n);
    }

    /// <summary>Text files' tokens, in path order, up to <paramref name="want"/>.</summary>
    private static int[] ReadCorpus(string spec, Amql.Hf.HfTokenizer tokenizer, long want)
    {
        var files = new List<string>();
        foreach (var part in spec.Split(';', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
        {
            if (Directory.Exists(part))
            {
                files.AddRange(Directory.EnumerateFiles(part, "*", SearchOption.AllDirectories)
                    .Where(f => Path.GetExtension(f).ToLowerInvariant() is ".md" or ".cs" or ".txt" or ".cu" or ".xaml")
                    .Where(f => !f.Split('/', '\\').Any(d => d is "bin" or "obj"))
                    .OrderBy(f => f, StringComparer.OrdinalIgnoreCase));
            }
            else if (File.Exists(part))
            {
                files.Add(part);
            }
            else
            {
                throw new CliException($"--text: '{part}' does not exist");
            }
        }
        var ids = new List<int>();
        foreach (var f in files)
        {
            if (ids.Count >= want)
            {
                break;
            }
            ids.AddRange(tokenizer.EncodeToIds($"\n\n=== {Path.GetFileName(f)} ===\n\n" + File.ReadAllText(f)));
        }
        return ids.ToArray();
    }

    private static List<string> NeedleNames(Random random)
    {
        string[] first = ["amber", "brass", "cobalt", "copper", "coral", "crimson", "ebony", "emerald", "golden", "granite", "hazel", "indigo",
            "ivory", "jade", "lilac", "marble", "maroon", "mossy", "ochre", "olive", "opal", "pearl", "quartz", "russet", "sable", "scarlet",
            "silver", "slate", "teal", "umber", "violet", "walnut"];
        string[] second = ["anchor", "beacon", "bellows", "boiler", "bridge", "chimney", "cistern", "clock", "compass", "crane", "drill",
            "elevator", "engine", "furnace", "gate", "generator", "kiln", "lantern", "lathe", "loom", "pulley", "pump", "reactor", "shutter",
            "siphon", "spindle", "turbine", "turret", "valve", "vault", "winch", "windmill"];
        var names = first.SelectMany(a => second.Select(b => $"{a} {b}")).ToList();
        for (int i = names.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (names[i], names[j]) = (names[j], names[i]);
        }
        return names;
    }
}
