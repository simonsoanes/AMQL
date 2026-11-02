using Amql.Hf;
using Amql.Inference;
using Amql.Merge;
using Amql.Vindex3;

namespace Amql.Cli;

// ── bake-predicate: a predicate the model reads, solved into its weights ──

internal static partial class Program
{
    /// <summary>Generic probe turns, mixed in with the model's own probes so
    /// the update is fitted on ordinary requests as well as the ones the
    /// predicate obviously touches.</summary>
    private static readonly string[] GenericProbes =
    {
        "Hello! Who are you?",
        "What is the capital of France?",
        "Explain how a rainbow forms.",
        "Write a short poem about the sea.",
        "Give me three tips for learning a new language.",
        "What is 17 times 23?",
        "Summarise the plot of a fairy tale in two sentences.",
        "How do I boil an egg?",
    };

    private static int BakePredicate(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException(
            "bake-predicate requires a container directory");
        var problems = new List<string>();
        string? predicate = OptionValue(args, "--predicate");
        if (OptionValue(args, "--predicate-file") is { } predicateFile)
            predicate = File.ReadAllText(predicateFile).Trim();
        if (string.IsNullOrWhiteSpace(predicate))
            problems.Add("missing '--predicate \"<text>\"' (or '--predicate-file <path>') — the rule to bake in");
        string? outPatch = OptionValue(args, "--out");
        if (outPatch is null)
            problems.Add("missing '--out <patch.safetensors>'");
        if (problems.Count > 0)
            throw new CliException("bake-predicate:\n" + string.Join("\n", problems.Select(p => "  - " + p)));

        string component = OptionValue(args, "--component") ?? "target";
        string tokenizerDir = ResolveTokenizerDir(containerDir, args);
        var tokenizer = Tokenizer(tokenizerDir);
        string? template = HasOption(args, "--raw") ? null : ChatTemplate.Load(tokenizerDir);
        var stop = ResolveStopTokens(tokenizerDir, tokenizer);
        int maxTokens = IntOption(args, "--max-tokens", 48);

        // Teacher and student see the same user turn; only the teacher is
        // given the predicate — as the system message under a chat
        // template, as a preamble otherwise.
        string Render(string prompt, bool withPredicate) => template is not null
            ? ChatTemplate.Apply(template, prompt, withPredicate ? predicate : null)
            : withPredicate ? predicate + "\n\n" + prompt : prompt;

        using var container = Vindex3Container.Open(containerDir);
        Console.WriteLine($"container: {containerDir}   tokenizer: {tokenizerDir}");
        Console.WriteLine($"predicate: {predicate}");
        Console.WriteLine($"format:    {(template is not null ? "chat template (predicate as system message)" : "raw text (predicate as preamble)")}");

        var prompts = new List<string>();
        if (OptionValue(args, "--prompts") is { } promptsPath)
        {
            prompts.AddRange(File.ReadAllLines(promptsPath)
                .Select(l => l.Trim())
                .Where(l => l.Length > 0 && !l.StartsWith('#')));
            Console.WriteLine($"probes:    {prompts.Count} from {promptsPath}");
        }
        else
        {
            int selfProbes = IntOption(args, "--self-probes", 8);
            var own = selfProbes > 0
                ? SelfProbes(container, component, tokenizer, template, stop, predicate!, selfProbes)
                : new List<string>();
            Console.WriteLine($"probes:    {own.Count} written by the model for this predicate + {GenericProbes.Length} generic");
            foreach (var p in own)
                Console.WriteLine($"           · {p}");
            prompts.AddRange(own);
            prompts.AddRange(GenericProbes);
        }
        if (prompts.Count == 0)
            throw new CliException("no probe prompts — pass '--prompts <file>' with one user turn per line");

        var probes = prompts
            .Select(p => new PredicateProbe(
                tokenizer.EncodeToIds(Render(p, withPredicate: true)).ToArray(),
                tokenizer.EncodeToIds(Render(p, withPredicate: false)).ToArray()))
            .Where(p => p.TeacherPrompt.Length > 0 && p.StudentPrompt.Length > 0)
            .ToList();

        var options = new PredicateBakeOptions
        {
            MaxNewTokens = maxTokens,
            Layers = OptionValue(args, "--layers") is { } layers ? ParseIntList(layers, Array.Empty<int>()) : null,
            LayerCount = IntOption(args, "--layer-count", 4),
            Ridge = DoubleOption(args, "--ridge", 1e-2),
            StopTokens = stop,
            Samples = IntOption(args, "--samples", 3),
        };

        CliProgress.Phase("bake-predicate");
        var report = PredicateBaker.Bake(container, component, probes, outPatch!, options);
        CliProgress.Complete($"{report.Layers.Count} layers, KL {report.KlBefore:F3} → {report.KlAfter:F3}");

        Console.WriteLine();
        Console.WriteLine($"component: {report.ComponentId}");
        Console.WriteLine($"aligned:   {report.Probes} probes → {report.AlignedRows} teacher-generated positions");
        foreach (var fit in report.Layers)
        {
            Console.WriteLine($"layer {fit.Layer,3}: {fit.ObjectId}/{fit.TensorName}  gap rms {fit.GapNorm:F3}  " +
                              $"closed {fit.Explained:P1} of its share");
        }
        Console.WriteLine($"KL(teacher‖model):  {report.KlBefore:F4} → {report.KlAfter:F4} nats/token");
        Console.WriteLine($"top-1 agreement:    {report.Top1Before:P1} → {report.Top1After:P1}");
        foreach (var sample in report.Samples)
        {
            Console.WriteLine();
            Console.WriteLine($"probe:   {prompts[sample.ProbeIndex]}");
            Console.WriteLine($"  with predicate:    {OneLine(tokenizer.DecodeText(sample.Teacher))}");
            Console.WriteLine($"  bare, before:      {OneLine(tokenizer.DecodeText(sample.StudentBefore))}");
            Console.WriteLine($"  bare, patched:     {OneLine(tokenizer.DecodeText(sample.StudentAfter))}");
        }
        foreach (var note in report.Notes)
            Console.WriteLine($"note:      {note}");
        Console.WriteLine();
        Console.WriteLine($"patch:     {report.PatchPath}  ({report.Layers.Count} tensors)");
        Console.WriteLine("apply it:  amql-cli generate <container> --chat --prompt \"...\" --patch " + report.PatchPath);
        Console.WriteLine("bake it:   amql-cli export <container> --out <checkpoint> --patch " + report.PatchPath);
        CliProgress.Result(new
        {
            command = "bake-predicate",
            patch = report.PatchPath,
            layers = report.Layers.Select(l => l.Layer).ToArray(),
            klBefore = report.KlBefore,
            klAfter = report.KlAfter,
            top1Before = report.Top1Before,
            top1After = report.Top1After,
        });
        return 0;
    }

    /// <summary>Asks the model itself which requests its predicate would
    /// change the answer to — the probes then cover the predicate as the
    /// model reads it, not just as the caller guessed.</summary>
    private static List<string> SelfProbes(
        Vindex3Container container, string component, HfTokenizer tokenizer,
        string? template, IReadOnlySet<int> stop, string predicate, int count)
    {
        // The rule is quoted as the subject of the request, not given as an
        // instruction — a small model otherwise just starts obeying it.
        string ask =
            $"Another assistant has been given this rule:\n\"\"\"\n{predicate}\n\"\"\"\n\n" +
            $"Do not follow the rule yourself. Write {count} different short messages, in English, " +
            "that a user might send to that assistant, where the rule would change its reply. " +
            "Write one message per line. No numbering, no answers, nothing else.";
        string rendered = template is not null ? ChatTemplate.Apply(template, ask, null) : ask + "\n\n";
        // A thinking model would otherwise spend the budget reasoning about
        // the request; an empty think block is how its template turns that off.
        if (template is not null && template.Contains("<think>"))
            rendered += "<think>\n\n</think>\n\n";
        var tokens = tokenizer.EncodeToIds(rendered).ToArray();
        var (_, steps) = InferenceRunner.Generate(
            container, component, tokens, steps: 40 * count,
            new SamplingConfig(Temperature: 0f), stopTokens: stop);
        string text = tokenizer.DecodeText(steps.Select(s => s.Token).ToList());
        int thinkEnd = text.LastIndexOf("</think>", StringComparison.Ordinal);
        if (thinkEnd >= 0)
            text = text[(thinkEnd + "</think>".Length)..];
        else if (text.Contains("<think>", StringComparison.Ordinal))
            return new List<string>();  // never finished thinking — no messages to take
        return text.Split('\n')
            .Select(l => l.Trim().TrimStart('-', '*', '•', ' ').TrimStart("0123456789.) ".ToCharArray()).Trim().Trim('"'))
            .Where(l => l.Length >= 8 && !l.StartsWith('<'))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(count)
            .ToList();
    }

    private static string OneLine(string text) => text.ReplaceLineEndings(" ⏎ ").Trim();
}
