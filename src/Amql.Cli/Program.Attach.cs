using System.Diagnostics;
using Amql.Hf;
using Amql.Inference;
using Amql.Merge;

namespace Amql.Cli;

// ── attach: a drafter, a vision tower or Entail adapters, added to a container ──

internal static partial class Program
{
    /// <summary><c>attach &lt;container&gt; [--drafter &lt;checkpoint&gt;] [--vision &lt;checkpoint&gt;] [--adapter &lt;dir&gt;,…] [--format nvfp4|mxfp4|none]</c>:
    /// adds the part to the container as it stands. The format is the container's own unless given.
    /// Adapters are copied in, to run beside the model wherever it is served.</summary>
    private static int AttachPart(string[] args)
    {
        var containerDir = Arg(args, 0) ?? throw new CliException(
            "attach requires a container directory, e.g. 'amql-cli attach <container> --drafter <checkpoint>'");
        string? drafter = OptionValue(args, "--drafter"), vision = OptionValue(args, "--vision"), adapters = OptionValue(args, "--adapter");
        if (drafter is null && vision is null && adapters is null)
        {
            throw new CliException("attach requires '--drafter <checkpoint>', '--vision <checkpoint>' or '--adapter <dir>,…'");
        }
        if (adapters is not null)
        {
            try
            {
                var order = Amql.Entail.ShippedAdapters.Attach(containerDir, adapters.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
                Console.WriteLine($"attached:   adapters {string.Join(", ", order)} (run in that order beside the model)");
            }
            catch (Amql.Entail.EntailException e)
            {
                throw new CliException(e.Message);
            }
            if (drafter is null && vision is null)
            {
                return 0;
            }
        }
        Fp4Format? format = (OptionValue(args, "--format") ?? ContainerFormat(containerDir)).ToLowerInvariant() switch
        {
            "nvfp4" => Fp4Format.Nvfp4,
            "mxfp4" => Fp4Format.Mxfp4,
            "none" => null,
            var other => throw new CliException($"unknown format '{other}': use nvfp4, mxfp4 or none"),
        };
        if (drafter is not null)
        {
            Report(Attach.Drafter(containerDir, CheckpointDir(drafter, args), format));
        }
        if (vision is not null)
        {
            Report(Attach.Vision(containerDir, CheckpointDir(vision, args)));
        }
        return 0;

        static void Report(AttachReport report)
        {
            foreach (string note in report.Notes)
            {
                Console.WriteLine($"attached:   {note}");
            }
            Console.WriteLine($"wrote:      {string.Join(", ", report.Objects)} ({report.PayloadBytes / 1e9:F2} GB) into {report.Container}");
        }
    }

    /// <summary>The 4-bit format a container was quantised to, or "none".</summary>
    private static string ContainerFormat(string containerDir)
    {
        using var index = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(Path.Combine(containerDir, "index.json")));
        return index.RootElement.TryGetProperty("quantisation", out var q) && q.TryGetProperty("format", out var f) ? f.GetString() ?? "none" : "none";
    }

    /// <summary>A checkpoint's folder: the one named, or a Hub model downloaded.</summary>
    private static string CheckpointDir(string source, string[] args)
    {
        if (Directory.Exists(source))
        {
            return source;
        }
        return HfDownloader.TryParse(source, OptionValue(args, "--revision")) is { } hub
            ? DownloadFromHub(hub, args)
            : throw new CliException($"'{source}' is neither a local checkpoint directory nor a Hugging Face URL or owner/name");
    }

    /// <summary><c>bench --draft N</c>: greedy decode with up to N drafts a step, checked against
    /// plain decode, over the model's own reply to <c>--chat</c> text or the bench's prompt.</summary>
    private static int DraftBench(DecodeSession session, int[] prompt, int steps, int drafts, string[] args)
    {
        var engine = session.DeviceEngine() ?? throw new CliException($"--draft needs the GPU engine ({session.GpuRefusal})");
        if (!engine.HasDrafter)
        {
            throw new CliException("the container has no drafter (amql-cli attach --drafter)");
        }
        var plain = engine.NewSequence();
        var expected = new List<int> { Sampler.ArgMax(engine.Prefill(plain, prompt)) };
        engine.StepInPlace(new[] { plain }, new[] { expected[0] });   // capture outside the timing
        expected.Add(Sampler.ArgMax(engine.HostLogits(0)));
        var sw = Stopwatch.StartNew();
        while (expected.Count < steps)
        {
            engine.StepInPlace(new[] { plain }, new[] { expected[^1] });
            expected.Add(Sampler.ArgMax(engine.HostLogits(0)));
        }
        double plainRate = (steps - 2) / sw.Elapsed.TotalSeconds;
        engine.Release(plain);
        Console.WriteLine($"plain:     {plainRate,8:F1} tok/s");
        float floor = FloatOption(args, "--draft-floor", 0f);
        for (int n = 0; n <= drafts + 1; n++)
        {
            var pace = n > drafts ? new DraftPace(engine.DraftsThatPay) : null;   // last, as the server does it
            if (pace is not null)
            {
                floor = DraftPace.Floor;
            }
            var seq = engine.NewSequence();
            var reply = new List<int> { Sampler.ArgMax(engine.Prefill(seq, prompt)) };
            int cycles = 0, drafted = 0, kept = 0;
            long tDraft = 0, tVerify = 0;
            var held = new int[GpuEngine.MaxDrafts];
            sw.Restart();
            while (reply.Count < steps)
            {
                long t0 = Stopwatch.GetTimestamp();
                var proposal = n == 0 ? [] : engine.Draft(seq, reply[^1], pace?.Next ?? n);
                proposal = floor > 0 && n > 0 ? proposal[..Math.Min(proposal.Length, engine.WorthVerifying(floor))] : proposal;
                long t1 = Stopwatch.GetTimestamp();
                engine.Verify(seq, [reply[^1], .. proposal]);
                tVerify += Stopwatch.GetTimestamp() - t1;
                tDraft += t1 - t0;
                int rows = 0;
                while (true)
                {
                    int choice = Sampler.ArgMax(engine.HostLogits(rows));
                    reply.Add(choice);
                    rows++;
                    if (rows > proposal.Length || proposal[rows - 1] != choice)
                    {
                        break;
                    }
                    held[rows - 1]++;
                }
                engine.Accept(seq, rows);
                pace?.Saw(proposal.Length, rows - 1);
                cycles++;
                drafted += proposal.Length;
                kept += rows - 1;
            }
            double rate = (reply.Count - 1) / sw.Elapsed.TotalSeconds;
            int same = 0;
            while (same < Math.Min(steps, reply.Count) && reply[same] == expected[same])
            {
                same++;
            }
            double ms(long t) => t * 1000.0 / Stopwatch.Frequency / cycles;
            Console.WriteLine($"draft {(pace is null ? n.ToString() : "auto")}:   {rate,8:F1} tok/s   {(double)(reply.Count - 1) / cycles:F2} tokens a step, {100.0 * kept / Math.Max(1, drafted):F0}% of drafts kept" +
                              $" [{string.Join(" ", held.Take(n).Select(x => $"{100.0 * x / cycles:F0}"))}]" +
                              $", draft {ms(tDraft):F2} ms, verify {ms(tVerify):F2} ms; {same} of {steps} tokens as plain decode");
            engine.Release(seq);
        }
        return 0;
    }
}
