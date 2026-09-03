using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;

namespace Amql.Cli;

/// <summary>
/// amql-cli — the G0→G3 loader front-end: turn a raw HF checkpoint
/// (Qwen3.5 and similar text stacks) into a canonical VINDEX3 container,
/// then verify the container's byte equivalence from disk alone.
///
///   amql-cli encode &lt;model-dir&gt; --out &lt;container-dir&gt;
///   amql-cli verify &lt;container-dir&gt;
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            PrintHelp();
            return args.Length == 0 ? 1 : 0;
        }

        try
        {
            return args[0] switch
            {
                "encode" => Encode(args[1..]),
                "verify" => Verify(args[1..]),
                _ => throw new CliException($"unknown command '{args[0]}'"),
            };
        }
        catch (CliException e)
        {
            Console.Error.WriteLine($"error: {e.Message}");
            Console.Error.WriteLine("run 'amql-cli help' for usage");
            return 2;
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"error: {e.Message}");
            return 2;
        }
    }

    // ── encode ──────────────────────────────────────────────────────────────

    private static int Encode(string[] args)
    {
        var modelDir = Arg(args, 0) ?? throw new CliException("encode requires a model directory");
        string? outDir = OptionValue(args, "--out");
        if (outDir is null)
        {
            throw new CliException("encode requires '--out <container-dir>'");
        }

        Console.WriteLine($"encoding '{modelDir}' → '{outDir}'");
        var report = ModelToContainer.Encode(modelDir, outDir);

        Console.WriteLine();
        Console.WriteLine($"model:        {report.ModelId}");
        Console.WriteLine($"encoding:     {report.Encoding}");
        Console.WriteLine($"tensors:      {report.Tensors}");
        Console.WriteLine($"payload:      {FormatBytes(report.PayloadBytes)}");
        foreach (var (repId, write) in report.Segments.OrderBy(s => s.Key))
        {
            Console.WriteLine($"  {repId}: {write.PayloadBytes} bytes  payload_sha={Short(write.PayloadSha256Hex)}  segment_sha={Short(write.SegmentSha256Hex)}");
        }
        Console.WriteLine("done. run 'amql-cli verify <container-dir>' for integrity + runtime readiness.");
        return 0;
    }

    // ── verify ─────────────────────────────────────────────────────────────

    private static int Verify(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException("verify requires a container directory");

        using var container = Vindex3Container.Open(containerDir);
        var index = container.Index;

        Console.WriteLine($"container:  {containerDir}");
        Console.WriteLine($"index:      schema {index.Version} authority {index.Authority} model '{index.Model}' family '{index.Family}'");
        Console.WriteLine($"graph:      schema {container.Graph?.Schema} components={container.Graph?.Components.Count} objects={container.Graph?.Objects.Count}");

        // Byte equivalence: recompute all hashes from disk alone.
        var report = container.VerifyIntegrity();
        Console.WriteLine("integrity:");
        foreach (var check in report.Checks)
        {
            Console.WriteLine($"  [{(check.Ok ? "ok" : "FAIL")}] {check.Representation}{(check.Detail is null ? string.Empty : $" — {check.Detail}")}");
        }
        if (!report.Ok)
        {
            Console.Error.WriteLine("integrity verification FAILED — the container diverges from its index");
            return 1;
        }

        // Operand resolution on real weights: tensor tables, shapes, stored
        // dtypes and payload widening all exercised from the container alone.
        using var store = container.CreateOperandStore();
        Console.WriteLine("operands:");
        var qProj = store.Resolve("target.decoder_stack", "3.self_attn.q_proj.weight");
        Console.WriteLine($"  3.self_attn.q_proj.weight: {qProj.Dtype.Label()} shape=[{string.Join("x", qProj.Shape)}]");
        var kvIn = store.Resolve("target.decoder_stack", "0.linear_attn.in_proj_qkv.weight");
        Console.WriteLine($"  0.linear_attn.in_proj_qkv.weight: {kvIn.Dtype.Label()} shape=[{string.Join("x", kvIn.Shape)}]");
        var aLog = store.Resolve("target.decoder_stack", "0.linear_attn.A_log");
        Console.WriteLine($"  0.linear_attn.A_log: {aLog.Dtype.Label()} shape=[{string.Join("x", aLog.Shape)}] (precision exception)");
        var norms = store.Resolve("target.final_norm", "weight");
        var widened = Amql.Safetensors.BitPattern.WidenToF32(norms.Dtype, norms.Payload);
        float firstNorm = widened[0];
        Console.WriteLine($"  final_norm/weight: {widened.Length} f32 values (first={firstNorm:F4}, dtype {norms.Dtype.Label()})");
        Console.WriteLine($"operand store: {store.TouchedObjects.Count} objects touched, {store.Loads} loads");

        // Runtime readiness: plan the primary text component and report the
        // operator boundary — which persisted primitives this build serves
        // and which it refuses (fail-closed, by name).
        Console.WriteLine("runtime readiness:");
        Console.WriteLine(Census(container));
        try
        {
            var plan = Planner.Plan(container, "target", store);
            Console.WriteLine("  [served] the primary text decoder plans and executes.");
            Console.WriteLine($"  layers: {plan.Layers.Count}, hidden: {plan.HiddenSize}, embedding: {(plan.Embedding is null ? "none" : plan.Embedding.VocabSize.ToString())}, head: {(plan.Output is null ? "none (tied?)" : plan.Output.VocabSize.ToString())}");
        }
        catch (UnsupportedOperatorException e)
        {
            Console.WriteLine($"  [refused] {e.Message}");
        }
        return 0;
    }

    private static string Census(Vindex3Container container)
    {
        var graph = container.Graph;
        if (graph is null)
        {
            return "  (no system graph recorded)";
        }
        var lines = new List<string>();
        foreach (var component in graph.Components)
        {
            lines.Add($"  component '{component.Id}' role={component.Role} layers={component.NumLayers} hidden={component.HiddenSize}");
            if (component.Attention is not null)
            {
                foreach (var group in component.Attention.GroupBy(p => p.Operator).OrderBy(g => g.Key))
                {
                    lines.Add($"    operator '{group.Key}': {group.Count()} layers");
                }
            }
            foreach (var obj in graph.Objects.Where(o => o.Component == component.Id))
            {
                string reps = obj.Representations.Count == 0 ? "carried" : string.Join(",", obj.Representations.Select(r => r.Encoding));
                string bindings = string.Join(", ", obj.SourceBindings.Select(b => $"{b.TensorPrefix}.*({b.Tensors})"));
                lines.Add($"    object '{obj.Id}' kind={obj.Kind} reps=[{reps}] bindings=[{(bindings.Length == 0 ? "-" : bindings)}]");
            }
        }
        return string.Join(Environment.NewLine, lines);
    }

    // ── plumbing ───────────────────────────────────────────────────────────

    private static void PrintHelp()
    {
        Console.WriteLine("""
            amql-cli — load an HF checkpoint into a canonical VINDEX3 container

            USAGE:
              amql-cli encode <model-dir> --out <container-dir>   map + materialise
              amql-cli verify <container-dir>                     integrity + readiness
              amql-cli help

            The encoder runs the G0→G3 pipeline: shard inventory, config facts,
            system graph + execution surface, canonical (unquantised) segments.
            Unknown operators/facts are recorded verbatim and refused at plan
            time by name — never approximated.
            """);
    }

    private static string? Arg(string[] args, int index) => index < args.Length ? args[index] : null;

    private static string? OptionValue(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == name)
            {
                return args[i + 1];
            }
        }
        return null;
    }

    private static string FormatBytes(long bytes) => bytes switch
    {
        >= 1 << 30 => $"{bytes / 1073741824.0:0.00} GiB",
        >= 1 << 20 => $"{bytes / 1048576.0:0.00} MiB",
        >= 1 << 10 => $"{bytes / 1024.0:0.00} KiB",
        _ => $"{bytes} B",
    };

    private static string Short(string hex) => hex[..12];
}

internal sealed class CliException : Exception
{
    public CliException(string message) : base(message) { }
}