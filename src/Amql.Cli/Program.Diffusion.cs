using System.Diagnostics;
using System.Text;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Cli;

// ── generate, masked diffusion models ─────────────────────────────────────

internal static partial class Program
{
    /// <summary>Whether the container's text component is a masked diffusion
    /// LM, which <c>generate</c> must hand to the diffusion sampler: stepping
    /// it one token at a time would run a bidirectional stack as if causal.</summary>
    private static bool IsDiffusionContainer(string containerDir, string component)
    {
        using var container = Vindex3Container.Open(containerDir);
        return container.Graph?.Components.FirstOrDefault(c => c.Id == component)?.Execution?.Diffusion is not null;
    }

    /// <summary>
    /// <c>generate</c> for a masked diffusion container. <c>--gen</c> masked
    /// positions follow the prompt and are unmasked over <c>--steps</c> full
    /// passes, <c>--block</c> positions at a time, left to right.
    /// </summary>
    private static int GenerateDiffusion(string containerDir, string[] args, string component)
    {
        string? prompt = OptionValue(args, "--prompt");
        string tokenizerDir = ResolveTokenizerDir(containerDir, args);
        var tokenizer = Tokenizer(tokenizerDir);
        int[] tokens;
        if (prompt is not null)
        {
            if (HasOption(args, "--chat"))
            {
                string template = ChatTemplate.Load(tokenizerDir) ?? throw new CliException(
                    "--chat requires a chat template, but none was found in chat_template.jinja or tokenizer_config.json");
                prompt = ChatTemplate.Apply(template, prompt, OptionValue(args, "--system"));
                // The A2D checkpoints are tuned with thinking off: the template's
                // enable_thinking=false branch pre-fills an empty think block.
                if (template.Contains("enable_thinking") && !HasOption(args, "--think"))
                {
                    prompt += "<think>\n\n</think>\n\n";
                }
                Console.WriteLine("chat:      prompt formatted via chat template" +
                                  (HasOption(args, "--think") ? "" : " (thinking off)"));
            }
            tokens = tokenizer.EncodeToIds(prompt).ToArray();
        }
        else
        {
            tokens = ParseIntList(OptionValue(args, "--tokens"), fallback: Array.Empty<int>());
        }
        if (tokens.Length == 0)
        {
            throw new CliException("diffusion generation needs a prompt (--prompt or --tokens)");
        }

        var stopTokens = ResolveStopTokens(tokenizerDir, tokenizer);
        int gen = IntOption(args, "--gen", 128);
        string remasking = OptionValue(args, "--remasking") ?? "low_confidence";
        var options = new DiffusionOptions
        {
            GenLength = gen,
            Steps = IntOption(args, "--steps", gen),
            BlockLength = IntOption(args, "--block", Math.Min(32, gen)),
            Temperature = FloatOption(args, "--temperature", 0f),
            Seed = IntOption(args, "--seed", 42),
            Remasking = remasking switch
            {
                "low_confidence" => Remasking.LowConfidence,
                "random" => Remasking.Random,
                _ => throw new CliException($"unknown --remasking '{remasking}' — use low_confidence or random"),
            },
            StopTokens = HasOption(args, "--no-early-stop") ? null : stopTokens,
        };
        bool showSteps = HasOption(args, "--show-steps");
        WeightWorkingSet? workingSet = (OptionValue(args, "--weights") ?? "") switch
        {
            "f32" or "fp32" => WeightWorkingSet.ResidentF32,
            "bf16" => WeightWorkingSet.OnDemandBf16,
            "mxfp4" or "fp4" => WeightWorkingSet.Mxfp4,
            "" => null,
            var other => throw new CliException($"unknown weight mode '{other}' — use f32, bf16, or mxfp4"),
        };

        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, component, store);
        var runtime = new GenericRuntime(plan, store, LoadPatch(args, container), workingSet);
        var sampler = new MaskedDiffusionSampler(runtime);

        Console.WriteLine($"diffusion: masked (MDLM), mask token {sampler.MaskTokenId}, " +
                          $"{options.GenLength} positions, {options.Steps} steps, block {options.BlockLength}, " +
                          $"remasking {remasking}, temperature {options.Temperature}");
        Console.WriteLine($"prompt:    {tokens.Length} tokens");
        Console.WriteLine($"weights:   {workingSet ?? WeightWorkingSetExtensions.FromEnv()}" +
                          (CudaShim.Enabled ? " (CUDA available)" : " (CPU)"));

        string Canvas(IReadOnlyList<int> sequence)
        {
            var sb = new StringBuilder();
            for (int i = tokens.Length; i < sequence.Count; i++)
            {
                sb.Append(sequence[i] == sampler.MaskTokenId ? "_" : tokenizer.Decode(new[] { sequence[i] }));
            }
            return sb.ToString().Replace("\n", "⏎");
        }

        var clock = Stopwatch.StartNew();
        var result = sampler.Generate(tokens, options, showSteps
            ? step => Console.WriteLine($"  [b{step.Block} s{step.Step}] {Canvas(step.Sequence)}")
            : null);
        clock.Stop();

        // The answer ends at the first stop token; what follows it is padding
        // the sampler filled because every position must be committed.
        int end = Array.FindIndex(result.Generated, stopTokens.Contains);
        var answer = end < 0 ? result.Generated : result.Generated[..end];
        var text = tokenizer.DecodeText(answer.Where(t => t != sampler.MaskTokenId).ToArray());

        Console.WriteLine();
        Console.WriteLine(text);
        Console.WriteLine();
        double seconds = clock.Elapsed.TotalSeconds;
        Console.WriteLine($"stats:     {result.Forwards} forward passes in {seconds:F2}s " +
                          $"({result.Forwards / Math.Max(seconds, 1e-9):F1} passes/s), " +
                          $"{answer.Length} answer tokens ({answer.Length / Math.Max(seconds, 1e-9):F1} tok/s)" +
                          (result.StoppedEarly ? ", stopped after the block holding the stop token" : ""));
        Console.WriteLine($"           decoder stack {result.StackTime.TotalSeconds:F2}s, vocabulary head " +
                          $"{result.HeadTime.TotalSeconds:F2}s, sampling {seconds - result.StackTime.TotalSeconds - result.HeadTime.TotalSeconds:F2}s");
        return 0;
    }
}
