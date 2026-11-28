using System.Text.Json;
using Amql.Vindex3;

namespace Amql.Cli;

// ── graph: rewrite a container's system graph without re-encoding ──────────
//
// 'graph upgrade' rewrites system_graph.json with the generic surfaces the
// planner reads, built from facts an older build recorded that this one no
// longer reads (a config adapter's: see Amql.Hf.GraphUpgrade). The tensors,
// the index and every other file are untouched, so it needs only the graph:
// the segments need not be present.

internal static partial class Program
{
    private static int GraphCommand(string[] args) => Arg(args, 0) switch
    {
        "upgrade" => GraphUpgrade(args[1..]),
        _ => throw new CliException("graph takes 'upgrade' (see --help)"),
    };

    private static int GraphUpgrade(string[] args)
    {
        string dir = Arg(args, 0) ?? throw new CliException("graph upgrade requires a container directory");
        string? outDir = OptionValue(args, "--out");
        bool dryRun = HasOption(args, "--dry-run");

        string graphName = "system_graph.json";
        string indexPath = Path.Combine(dir, "index.json");
        if (File.Exists(indexPath))
        {
            using var index = JsonDocument.Parse(File.ReadAllBytes(indexPath));
            if (index.RootElement.TryGetProperty("system_graph", out var named) && named.ValueKind == JsonValueKind.String)
            {
                graphName = named.GetString()!;
            }
        }
        string graphPath = Path.Combine(dir, graphName);
        if (!File.Exists(graphPath))
        {
            throw new CliException($"'{dir}' has no {graphName}");
        }
        var graph = JsonSerializer.Deserialize<SystemGraph>(File.ReadAllBytes(graphPath), ViJson.Options)
                    ?? throw new CliException($"{graphPath} is empty");
        var (upgraded, components) = Amql.Hf.GraphUpgrade.Apply(graph);
        if (components.Count == 0)
        {
            Console.WriteLine($"{graphPath}: already current (no component to upgrade)");
            return 0;
        }
        Console.WriteLine($"{graphPath}: components {string.Join(", ", components)} gain the generic surfaces " +
                          "(depth loop, block residual, message gates, residual convs, attention conditioning, packed experts, roles)");
        if (dryRun)
        {
            Console.WriteLine(JsonSerializer.Serialize(upgraded, ViJson.Options));
            return 0;
        }
        string target;
        if (outDir is not null)
        {
            Directory.CreateDirectory(outDir);
            target = Path.Combine(outDir, graphName);
        }
        else
        {
            string backup = graphPath + ".pre-upgrade";
            File.Copy(graphPath, backup, overwrite: true);
            Console.WriteLine($"kept the old graph as {backup}");
            target = graphPath;
        }
        File.WriteAllBytes(target, JsonSerializer.SerializeToUtf8Bytes(upgraded, ViJson.Options));
        Console.WriteLine($"wrote {target}");
        return 0;
    }
}
