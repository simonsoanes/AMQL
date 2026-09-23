namespace Amql.Gui.Commands;

public enum ParamKind
{
    /// <summary>Positional argument, in order.</summary>
    Positional,
    /// <summary>--flag &lt;value&gt; (string/int/float as typed by the user).</summary>
    Option,
    /// <summary>Bare presence flag (--no-trace, --fit, --debug, --container).</summary>
    Switch,
}

public enum EditorKind
{
    Text,
    MultiLineText,
    Int,
    Float,
    Directory,
    File,
    Choice,
    IntList,
}

/// <summary>One declared parameter of a CLI command.</summary>
public sealed record ParamDef(
    string Key,
    string Label,
    ParamKind Kind,
    EditorKind Editor,
    string? Flag = null,
    string DefaultValue = "",
    string? Help = null,
    string[]? Choices = null,
    /// <summary>Project-default key this parameter can be stamped from (container/tokenizer/patch/component).</summary>
    string? DefaultFrom = null);

/// <summary>One declared amql-cli command and its full parameter surface.</summary>
public sealed record CommandDef(
    string Name,
    string Category,
    string Summary,
    IReadOnlyList<ParamDef> Params)
{
    /// <summary>Builds the argv (after the command name) from the user's values.</summary>
    public List<string> BuildArguments(IReadOnlyDictionary<string, string> values)
    {
        string Get(string key) => values.TryGetValue(key, out var v) ? v.Trim() : string.Empty;
        var args = new List<string>();

        // Positionals first, in declared order.
        foreach (var p in Params.Where(p => p.Kind == ParamKind.Positional))
        {
            var v = Get(p.Key);
            if (v.Length > 0)
            {
                args.Add(v);
            }
        }

        foreach (var p in Params.Where(p => p.Kind != ParamKind.Positional))
        {
            var v = Get(p.Key);
            if (p.Kind == ParamKind.Switch)
            {
                if (v is "true" or "1" or "on")
                {
                    args.Add(p.Flag!);
                }
                continue;
            }
            if (p.Flag is null)
            {
                // change-tensor's op/value pair: the op choice IS the flag
                // ('--set 0.5'), and '--zero' takes no value. The 'value'
                // param is consumed here, never emitted on its own.
                if (p.Key == "op" && v.Length > 0)
                {
                    args.Add(v);
                    if (v != "--zero" && Get("value") is { Length: > 0 } value)
                    {
                        args.Add(value);
                    }
                }
                continue;
            }
            if (v.Length > 0)
            {
                args.Add(p.Flag);
                args.Add(v);
            }
        }
        return args;
    }
}

/// <summary>
/// Declarative catalogue of every amql-cli command and option — the single
/// source of truth for the UI forms and the argv builder. Kept in lockstep
/// with src/Amql.Cli/Program.cs PrintHelp (2026-09).
/// </summary>
public static class CommandCatalog
{
    public const string CatIngest = "1. Ingest & Verify";
    public const string CatText = "2. Text & Tokenizer";
    public const string CatPathways = "3. Pathways & Probing";
    public const string CatPatching = "4. Patching & LoRA";
    public const string CatExport = "5. Export & Conversion";
    public const string CatMtp = "6. MTP Drafter";
    public const string CatTransform = "7. Model Transformation";
    public const string CatTraining = "8. Fine-Tuning";

    private static readonly ParamDef PatchParam = new(
        "patch", "--patch file", ParamKind.Option, EditorKind.File,
        Flag: "--patch", DefaultFrom: "patch",
        Help: "Weight patch (.safetensors) merged into the loaded weights; the container is never rewritten.");

    private static readonly ParamDef ComponentParam = new(
        "component", "--component", ParamKind.Option, EditorKind.Text,
        Flag: "--component", DefaultValue: "target", DefaultFrom: "component",
        Help: "System-graph component id to run against.");

    private static readonly ParamDef TokenizerParam = new(
        "tokenizer", "--tokenizer dir", ParamKind.Option, EditorKind.Directory,
        Flag: "--tokenizer", DefaultFrom: "tokenizer",
        Help: "HF checkpoint dir with tokenizer.json (alias --model-dir). Optional when the container carries its own.");

    private static readonly ParamDef ContainerParam = new(
        "container", "container dir", ParamKind.Positional, EditorKind.Directory,
        DefaultFrom: "container", Help: "VINDEX3 container directory (encode output).");

    private static readonly ParamDef OutDirParam = new(
        "out", "--out dir", ParamKind.Option, EditorKind.Directory,
        Flag: "--out", Help: "Output directory.");

    private static readonly ParamDef TextParam = new(
        "text", "--text corpus", ParamKind.Option, EditorKind.File,
        Flag: "--text", Help: "Corpus text file.");

    public static IReadOnlyList<CommandDef> All { get; } = new List<CommandDef>
    {
        // ── 1. Ingest & Verify ───────────────────────────────────────────────
        new("encode", CatIngest, "Map + materialise an HF checkpoint into a VINDEX3 container.", new[]
        {
            new ParamDef("modelDir", "model dir", ParamKind.Positional, EditorKind.Directory,
                Help: "Raw HF checkpoint directory (Qwen3.5, nomic-bert, Jev classifier)."),
            new("out", "--out container dir", ParamKind.Option, EditorKind.Directory, Flag: "--out",
                Help: "Container directory to write (index.json, system_graph.json, segments/)."),
        }),
        new("verify", CatIngest, "Integrity + runtime readiness of a container.", new[]
        {
            new ParamDef("containerDir", "container dir", ParamKind.Positional, EditorKind.Directory, DefaultFrom: "container"),
        }),
        new("synth-model", CatIngest, "Write an executable 2-layer demo checkpoint.", new[]
        {
            new ParamDef("dir", "output dir", ParamKind.Positional, EditorKind.Directory),
        }),
        new("layers", CatIngest, "Per-layer attention policy table + tensor inventory.", new[]
        {
            ContainerParam,
            ComponentParam,
        }),

        // ── 2. Text & Tokenizer ─────────────────────────────────────────────
        new("tokens", CatText, "Encode text to token ids via the model's tokenizer.", new[]
        {
            new ParamDef("text", "text", ParamKind.Positional, EditorKind.MultiLineText,
                Help: "The text to encode (quoted automatically)."),
            TokenizerParam,
            PatchParam,
        }),
        new("decode", CatText, "Decode token ids back to text.", new[]
        {
            new ParamDef("ids", "token ids", ParamKind.Positional, EditorKind.IntList,
                Help: "Comma-separated ids, e.g. 9419,11"),
            TokenizerParam,
            PatchParam,
        }),

        // ── 3. Pathways & Probing ───────────────────────────────────────────
        new("route", CatPathways, "Relationship probing between two tokens with causal tracing.", new[]
        {
            ContainerParam,
            new ParamDef("tokenA", "token A", ParamKind.Positional, EditorKind.Text),
            new ParamDef("tokenB", "token B", ParamKind.Positional, EditorKind.Text),
            TokenizerParam,
            new ParamDef("top", "--top", ParamKind.Option, EditorKind.Int, Flag: "--top", DefaultValue: "5"),
            new ParamDef("templates", "--templates", ParamKind.Option, EditorKind.Int, Flag: "--templates", DefaultValue: "8"),
            new ParamDef("traceStart", "--trace-layer-start", ParamKind.Option, EditorKind.Int, Flag: "--trace-layer-start", DefaultValue: "8"),
            new ParamDef("traceEnd", "--trace-layer-end", ParamKind.Option, EditorKind.Int, Flag: "--trace-layer-end", DefaultValue: "24"),
            new ParamDef("noTrace", "--no-trace", ParamKind.Switch, EditorKind.Text, Flag: "--no-trace",
                Help: "Skip the causal-attribution trace."),
            new ParamDef("corrupt", "--corrupt", ParamKind.Option, EditorKind.Text, Flag: "--corrupt", DefaultValue: "the"),
            ComponentParam,
            PatchParam,
        }),
        new("path", CatPathways, "Bidirectional best-first search between two tokens.", new[]
        {
            ContainerParam,
            new ParamDef("tokenA", "token A", ParamKind.Positional, EditorKind.Text),
            new ParamDef("tokenB", "token B", ParamKind.Positional, EditorKind.Text),
            TokenizerParam,
            new ParamDef("topk", "--topk", ParamKind.Option, EditorKind.Int, Flag: "--topk", DefaultValue: "6"),
            new ParamDef("maxNodes", "--max-nodes", ParamKind.Option, EditorKind.Int, Flag: "--max-nodes", DefaultValue: "48"),
            new ParamDef("maxDepth", "--max-depth", ParamKind.Option, EditorKind.Int, Flag: "--max-depth", DefaultValue: "6"),
            new ParamDef("debug", "--debug", ParamKind.Switch, EditorKind.Text, Flag: "--debug"),
            ComponentParam,
            PatchParam,
        }),
        new("generate", CatPathways, "Run inference: prompt or raw token ids.", new[]
        {
            ContainerParam,
            new ParamDef("prompt", "--prompt", ParamKind.Option, EditorKind.MultiLineText, Flag: "--prompt"),
            new ParamDef("tokens", "--tokens", ParamKind.Option, EditorKind.IntList, Flag: "--tokens",
                Help: "Comma-separated start ids (used when --prompt is absent; default 0)."),
            TokenizerParam,
            new ParamDef("steps", "--steps", ParamKind.Option, EditorKind.Int, Flag: "--steps", DefaultValue: "8"),
            new ParamDef("temperature", "--temperature", ParamKind.Option, EditorKind.Float, Flag: "--temperature", DefaultValue: "0"),
            new ParamDef("topK", "--top-k", ParamKind.Option, EditorKind.Int, Flag: "--top-k", DefaultValue: "0"),
            new ParamDef("topP", "--top-p", ParamKind.Option, EditorKind.Float, Flag: "--top-p", DefaultValue: "0"),
            new ParamDef("seed", "--seed", ParamKind.Option, EditorKind.Int, Flag: "--seed", DefaultValue: "42"),
            new ParamDef("logits", "--logits K", ParamKind.Option, EditorKind.Int, Flag: "--logits",
                Help: "Show top-K logits per step (leave empty to hide)."),
            ComponentParam,
            PatchParam,
        }),
        new("inspect-token", CatPathways, "Embedding profile, neighbours and logits for one token.", new[]
        {
            ContainerParam,
            new ParamDef("token", "token id", ParamKind.Positional, EditorKind.Int),
            new ParamDef("tokens", "--tokens ctx ids", ParamKind.Option, EditorKind.IntList, Flag: "--tokens",
                Help: "Context ids for the logits inspection."),
            new ParamDef("neighbors", "--neighbors", ParamKind.Option, EditorKind.Int, Flag: "--neighbors", DefaultValue: "5"),
            new ParamDef("logits", "--logits K", ParamKind.Option, EditorKind.Int, Flag: "--logits", DefaultValue: "5"),
            TokenizerParam,
            ComponentParam,
            PatchParam,
        }),

        // ── 4. Patching & LoRA ──────────────────────────────────────────────
        new("change-tensor", CatPatching, "Edit one weight cell into a patch file.", new[]
        {
            ContainerParam,
            new ParamDef("objectId", "object id", ParamKind.Positional, EditorKind.Text, Help: "e.g. target.embedding"),
            new ParamDef("tensorName", "tensor name", ParamKind.Positional, EditorKind.Text, Help: "e.g. weight, 0.self_attn.q_proj.weight"),
            new ParamDef("cell", "cell", ParamKind.Positional, EditorKind.Text, Help: "'row,col' for 2-D, flat index otherwise"),
            new ParamDef("op", "operation", ParamKind.Option, EditorKind.Choice, Flag: null, DefaultValue: "--set",
                Choices: new[] { "--set", "--add", "--scale", "--zero" },
                Help: "Exactly one edit operation."),
            new ParamDef("value", "value", ParamKind.Option, EditorKind.Float, Flag: null,
                Help: "Value for --set/--add/--scale (ignored for --zero)."),
            new ParamDef("out", "--out patch", ParamKind.Option, EditorKind.File, Flag: "--out",
                Help: "patch.safetensors to write (existing patch is loaded and composed)."),
            PatchParam,
        }),
        new("save-lora", CatPatching, "Factor a patch's 2-D deltas into lora_A/lora_B.", new[]
        {
            new ParamDef("patchFile", "patch file", ParamKind.Positional, EditorKind.File),
            OutDirParam with { Label = "--out lora dir" },
            new ParamDef("rank", "--rank", ParamKind.Option, EditorKind.Int, Flag: "--rank", DefaultValue: "8"),
            new ParamDef("alpha", "--alpha", ParamKind.Option, EditorKind.Float, Flag: "--alpha", DefaultValue: "16"),
            new ParamDef("container", "--container dir", ParamKind.Option, EditorKind.Directory, Flag: "--container",
                DefaultFrom: "container", Help: "Validate the patch against this container before writing."),
        }),
        new("fine-tune", CatTraining, "Teacher-forced output-head adaptation → weight patch.", new[]
        {
            ContainerParam,
            new ParamDef("data", "--data pairs.tsv", ParamKind.Option, EditorKind.File, Flag: "--data",
                Help: "TSV of prompt<TAB>completion lines."),
            new ParamDef("out", "--out patch", ParamKind.Option, EditorKind.File, Flag: "--out"),
            TokenizerParam,
            ComponentParam,
            new ParamDef("lr", "--lr", ParamKind.Option, EditorKind.Float, Flag: "--lr", DefaultValue: "1e-4"),
            new ParamDef("epochs", "--epochs", ParamKind.Option, EditorKind.Int, Flag: "--epochs", DefaultValue: "1"),
        }),

        // ── 5. Export & Conversion ──────────────────────────────────────────
        new("export", CatExport, "Materialise an HF checkpoint from the container.", new[]
        {
            ContainerParam,
            OutDirParam with { Label = "--out checkpoint dir" },
            PatchParam,
            new ParamDef("quant", "--quant", ParamKind.Option, EditorKind.Choice, Flag: "--quant", DefaultValue: "none",
                Choices: new[] { "none", "mxfp4" }),
            new ParamDef("arch", "--arch", ParamKind.Option, EditorKind.Choice, Flag: "--arch", DefaultValue: "qwen3.x",
                Choices: new[] { "qwen3.x", "qwen4-next" },
                Help: "Target architecture: qwen3.x (default Qwen3.5) or qwen4-next (Flash-Next)."),
        }),
        new("to-gguf", CatExport, "Convert an exported HF checkpoint to GGUF v3.", new[]
        {
            new ParamDef("checkpointDir", "checkpoint dir", ParamKind.Positional, EditorKind.Directory),
            new ParamDef("out", "--out file.gguf", ParamKind.Option, EditorKind.File, Flag: "--out"),
        }),

        // ── 8. Classify ─────────────────────────────────────────────────────
        new("classify", CatPathways, "Run a classifier model (Jev/NLI-style) on premise-hypothesis pairs.", new[]
        {
            ContainerParam,
            new ParamDef("text", "--text premise|hypothesis", ParamKind.Option, EditorKind.Text, Flag: "--text",
                Help: "Single string split on first | (premise | hypothesis)."),
            new ParamDef("premise", "--premise", ParamKind.Option, EditorKind.Text, Flag: "--premise"),
            new ParamDef("hypothesis", "--hypothesis", ParamKind.Option, EditorKind.Text, Flag: "--hypothesis"),
            new ParamDef("format", "--format", ParamKind.Option, EditorKind.Choice, Flag: "--format", DefaultValue: "labels",
                Choices: new[] { "labels", "json", "jsonl", "csv" }),
            PatchParam,
        }),

        // ── 6. MTP Drafter ──────────────────────────────────────────────────
        new("export-mtp", CatMtp, "Emit the MTP drafter as a standalone checkpoint.", new[]
        {
            ContainerParam,
            OutDirParam with { Label = "--out drafter dir" },
        }),
        new("generate-mtp", CatMtp, "Bootstrap (and optionally fit) an MTP drafter.", new[]
        {
            ContainerParam,
            OutDirParam,
            TextParam,
            new ParamDef("sample", "--sample", ParamKind.Option, EditorKind.Int, Flag: "--sample", DefaultValue: "4096"),
            new ParamDef("fit", "--fit", ParamKind.Switch, EditorKind.Text, Flag: "--fit",
                Help: "Run the calculated path in one command: collect pairs + ridge fit."),
            new ParamDef("eval", "--eval", ParamKind.Option, EditorKind.Int, Flag: "--eval", DefaultValue: "1024"),
            new ParamDef("clusters", "--clusters K", ParamKind.Option, EditorKind.Int, Flag: "--clusters"),
            new ParamDef("sweep", "--sweep K1,K2,K3", ParamKind.Option, EditorKind.Text, Flag: "--sweep"),
            new ParamDef("ridge", "--ridge", ParamKind.Option, EditorKind.Float, Flag: "--ridge", DefaultValue: "1e-4"),
        }),
        new("collect-mtp", CatMtp, "Collect the calculated drafter's continuation pairs.", new[]
        {
            ContainerParam,
            OutDirParam with { Label = "--out pairs dir" },
            TextParam,
            new ParamDef("fit", "--fit", ParamKind.Option, EditorKind.Int, Flag: "--fit", DefaultValue: "512"),
            new ParamDef("gate", "--gate", ParamKind.Option, EditorKind.Int, Flag: "--gate", DefaultValue: "256"),
        }),
        new("fit-mtp", CatMtp, "Fit the drafter projector by closed-form ridge least squares.", new[]
        {
            ContainerParam,
            new ParamDef("pairs", "--pairs dir", ParamKind.Option, EditorKind.Directory, Flag: "--pairs"),
            new ParamDef("ridge", "--ridge", ParamKind.Option, EditorKind.Float, Flag: "--ridge", DefaultValue: "1e-4"),
            new ParamDef("clusters", "--clusters K", ParamKind.Option, EditorKind.Int, Flag: "--clusters"),
            new ParamDef("sweep", "--sweep K1,K2,K3", ParamKind.Option, EditorKind.Text, Flag: "--sweep"),
        }),

        // ── 7. Model Transformation ─────────────────────────────────────────
        new("import", CatTransform, "Merge a second model into the container (vocab union).", new[]
        {
            ContainerParam,
            new ParamDef("model", "model to import", ParamKind.Positional, EditorKind.Directory,
                Help: "Checkpoint dir, or a container dir when --container is set."),
            OutDirParam with { Label = "--out merged dir" },
            new ParamDef("asContainer", "--container", ParamKind.Switch, EditorKind.Text, Flag: "--container",
                Help: "The model to import is itself a container."),
        }),
        new("moe-ify", CatTransform, "Restructure a dense container into a routed MoE.", new[]
        {
            ContainerParam,
            OutDirParam with { Label = "--out moe dir" },
            TextParam,
            new ParamDef("experts", "--experts", ParamKind.Option, EditorKind.Int, Flag: "--experts", DefaultValue: "8"),
            new ParamDef("topK", "--top-k", ParamKind.Option, EditorKind.Int, Flag: "--top-k", DefaultValue: "2"),
            new ParamDef("sample", "--sample", ParamKind.Option, EditorKind.Int, Flag: "--sample", DefaultValue: "4096"),
            new ParamDef("eval", "--eval", ParamKind.Option, EditorKind.Int, Flag: "--eval", DefaultValue: "1024"),
            new ParamDef("policy", "--policy", ParamKind.Option, EditorKind.Choice, Flag: "--policy", DefaultValue: "softmax",
                Choices: new[] { "softmax", "renormalise" }),
        }),
        new("prune", CatTransform, "Drop decoder layers to meet a byte budget.", new[]
        {
            ContainerParam,
            OutDirParam with { Label = "--out pruned dir" },
            new ParamDef("targetBytes", "--target-bytes", ParamKind.Option, EditorKind.Text, Flag: "--target-bytes",
                Help: "Bare bytes or B/KB/MB/GB/TB, KiB/MiB/GiB/TiB, e.g. 2GiB"),
            new ParamDef("approach", "--approach", ParamKind.Option, EditorKind.Choice, Flag: "--approach", DefaultValue: "provenance",
                Choices: new[] { "provenance", "corpus", "random" }),
            new ParamDef("seed", "--seed", ParamKind.Option, EditorKind.Int, Flag: "--seed", DefaultValue: "42"),
            TextParam,
            new ParamDef("sample", "--sample", ParamKind.Option, EditorKind.Int, Flag: "--sample", DefaultValue: "8192"),
            new ParamDef("minLayers", "--min-layers", ParamKind.Option, EditorKind.Int, Flag: "--min-layers", DefaultValue: "1"),
        }),
    };

    public static CommandDef? Find(string name) =>
        All.FirstOrDefault(c => string.Equals(c.Name, name, StringComparison.OrdinalIgnoreCase));

    public static IEnumerable<IGrouping<string, CommandDef>> ByCategory() =>
        All.GroupBy(c => c.Category).OrderBy(g => g.Key, StringComparer.Ordinal);
}
