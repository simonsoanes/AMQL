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
    /// <summary>A checkpoint: a local folder or a Hugging Face URL / owner/name, chosen in the form.</summary>
    ModelSource,
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
    string? DefaultFrom = null,
    /// <summary>Required to run: the CLI refuses the command without it. The GUI shows
    /// required parameters up front and puts the rest under "Advanced".</summary>
    bool Required = false);

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
/// with src/Amql.Cli/Program.cs PrintHelp and the commands' own option
/// parsing (2026-11). Subcommand-style commands ("entail bench", "ngram
/// build") carry the full words in <see cref="CommandDef.Name"/>; the runner
/// splits the name on spaces into separate argv entries.
/// </summary>
public static class CommandCatalog
{
    public const string CatIngest = "1. Ingest & Verify";
    public const string CatText = "2. Text & Tokenizer";
    public const string CatPathways = "3. Pathways & Probing";
    public const string CatBench = "3a. Inference & Benchmarks";
    public const string CatPatching = "4. Patching & LoRA";
    public const string CatExport = "5. Export & Conversion";
    public const string CatQuant = "5a. Quantisation";
    public const string CatMtp = "6. MTP Drafter";
    public const string CatTransform = "7. Model Transformation";
    public const string CatMoe = "7a. MoE Experts";
    public const string CatTraining = "8. Fine-Tuning";
    public const string CatEntail = "9. Entail";
    public const string CatNGram = "9a. N-gram Packs";

    private static readonly string[] OnOff = { "", "1" };

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
        DefaultFrom: "container", Help: "VINDEX3 container directory (encode output).", Required: true);

    private static readonly ParamDef OutDirParam = new(
        "out", "--out dir", ParamKind.Option, EditorKind.Directory,
        Flag: "--out", Help: "Output directory.", Required: true);

    private static readonly ParamDef TextParam = new(
        "text", "--text corpus", ParamKind.Option, EditorKind.File,
        Flag: "--text", Help: "Corpus text file.");

    private static readonly ParamDef EntailProgramParam = new(
        "program", "program file", ParamKind.Positional, EditorKind.File,
        Help: "Entail program (.entail.txt).", Required: true);

    private static readonly ParamDef NeuronsPerLayerParam = Opt(
        "neuronsPerLayer", "--neurons-per-layer", EditorKind.Int, "",
        "Widest layer the compiler may build (default 3584).");

    private static readonly ParamDef EntailContextParam = Opt(
        "context", "--context", EditorKind.Int, "8192", "Context length the attention heads must hold over.");

    private static readonly ParamDef EntailWeightsParam = new(
        "weights", "--weights", ParamKind.Option, EditorKind.Choice, Flag: "--weights",
        Choices: new[] { "", "f32", "disk" },
        Help: "Grafts are exact only in f32: f32 keeps weights resident, disk streams them. "
              + "Empty = disk when f32 would not fit in memory.");

    private static readonly ParamDef AdapterParam = Opt(
        "adapter", "--adapter dir[,dir]", EditorKind.Text, "",
        "Entail adapter directories run beside the model, comma-separated.");

    private static readonly ParamDef ModelWeightsParam = new(
        "weights", "--weights", ParamKind.Option, EditorKind.Choice, Flag: "--weights",
        DefaultValue: "", Choices: new[] { "", "f32", "bf16", "mxfp4", "disk", "lean" },
        Help: "Weight working set (empty = env var default). lean keeps stored bits and reads routed experts on demand.");

    /// <summary>An option that takes a value.</summary>
    private static ParamDef Opt(string key, string flag, EditorKind editor, string def = "", string? help = null,
        bool required = false, string[]? choices = null) =>
        new(key, flag, ParamKind.Option, editor, Flag: flag, DefaultValue: def, Help: help,
            Choices: choices, Required: required);

    /// <summary>A bare presence flag, shown as an empty/1 choice.</summary>
    private static ParamDef Flag(string key, string flag, string? help = null) =>
        new(key, flag, ParamKind.Switch, EditorKind.Choice, Flag: flag, DefaultValue: "", Choices: OnOff, Help: help);

    /// <summary>A positional argument.</summary>
    private static ParamDef Pos(string key, string label, EditorKind editor, string? help = null, bool required = true) =>
        new(key, label, ParamKind.Positional, editor, Help: help, Required: required);

    public static IReadOnlyList<CommandDef> All { get; } = new List<CommandDef>
    {
        // ── 1. Ingest & Verify ───────────────────────────────────────────────
        new("encode", CatIngest, "Turn a Hugging Face checkpoint (a local folder, or downloaded from the Hub) into a VINDEX3 container.", new[]
        {
            new ParamDef("modelDir", "checkpoint", ParamKind.Positional, EditorKind.ModelSource,
                Help: "Qwen3.5, nomic-bert, ModernBERT / Von, Jev classifier and other transformers checkpoints.",
                Required: true),
            new("out", "--out container dir", ParamKind.Option, EditorKind.Directory, Flag: "--out",
                Help: "Container directory to write (index.json, system_graph.json, segments/).", Required: true),
            new("revision", "--revision", ParamKind.Option, EditorKind.Text, Flag: "--revision",
                Help: "Hugging Face branch, tag or commit to download (default main)."),
            new("downloadDir", "--download-dir", ParamKind.Option, EditorKind.Directory, Flag: "--download-dir",
                Help: "Where the Hugging Face download goes (default: a folder per model under %LOCALAPPDATA%\\AMQL\\hf, or AMQL_HF_CACHE). Kept, so encoding again does not download again."),
            new("allFiles", "--all-files", ParamKind.Switch, EditorKind.Text, Flag: "--all-files",
                Help: "Download every file in the repository, including weights in formats the encoder does not read."),
            new("noVision", "--no-vision", ParamKind.Switch, EditorKind.Text, Flag: "--no-vision",
                Help: "Leave a multimodal checkpoint's vision tower out: a smaller container, for text alone."),
        }),
        new("attach", CatIngest, "Add a drafter or a vision tower to a container, beside what it holds.", new[]
        {
            ContainerParam,
            Opt("drafter", "--drafter checkpoint", EditorKind.Directory, "", "A drafter's checkpoint (Gemma 4's assistant), or its Hugging Face name."),
            Opt("vision", "--vision checkpoint", EditorKind.Directory, "", "The model's own checkpoint, for the vision tower a --no-vision encode left out."),
            Opt("format", "--format", EditorKind.Choice, "", "How the drafter's weights are held. Empty = as the container's.",
                choices: new[] { "", "nvfp4", "mxfp4", "none" }),
        }),
        new("download", CatIngest, "Download a checkpoint from the Hugging Face Hub (resumable, SHA-256 checked).", new[]
        {
            new ParamDef("source", "model", ParamKind.Positional, EditorKind.Text,
                Help: "The model's page URL (https://huggingface.co/owner/name) or owner/name; @revision may follow.", Required: true),
            new("out", "--out dir", ParamKind.Option, EditorKind.Directory, Flag: "--out",
                Help: "Folder to download into (default: a folder per model under %LOCALAPPDATA%\\AMQL\\hf)."),
            new("revision", "--revision", ParamKind.Option, EditorKind.Text, Flag: "--revision",
                Help: "Branch, tag or commit (default main)."),
            new("allFiles", "--all-files", ParamKind.Switch, EditorKind.Text, Flag: "--all-files",
                Help: "Download every file, including weights in formats the encoder does not read."),
        }),
        new("verify", CatIngest, "Integrity + runtime readiness of a container.", new[]
        {
            new ParamDef("containerDir", "container dir", ParamKind.Positional, EditorKind.Directory,
                DefaultFrom: "container", Required: true),
        }),
        new("synth-model", CatIngest, "Write an executable demo checkpoint.", new[]
        {
            new ParamDef("dir", "output dir", ParamKind.Positional, EditorKind.Directory, Required: true),
            Opt("family", "--family", EditorKind.Choice, "qwen3_5",
                "Shape of the demo: 2-layer Qwen3.5, 4-layer LFM3, or a 2-layer A2D-Qwen3 masked diffusion model.",
                choices: new[] { "qwen3_5", "lfm3", "a2d-qwen3" }),
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
                Help: "The text to encode (quoted automatically).", Required: true),
            TokenizerParam with { Required = true, Help = "HF checkpoint dir with tokenizer.json (alias --model-dir)." },
            PatchParam,
        }),
        new("decode", CatText, "Decode token ids back to text.", new[]
        {
            new ParamDef("ids", "token ids", ParamKind.Positional, EditorKind.IntList,
                Help: "Comma-separated ids, e.g. 9419,11", Required: true),
            TokenizerParam with { Required = true, Help = "HF checkpoint dir with tokenizer.json (alias --model-dir)." },
            PatchParam,
        }),

        // ── 3. Pathways & Probing ───────────────────────────────────────────
        new("route", CatPathways, "Relationship probing between two tokens with causal tracing.", new[]
        {
            ContainerParam,
            new ParamDef("tokenA", "token A", ParamKind.Positional, EditorKind.Text, Required: true),
            new ParamDef("tokenB", "token B", ParamKind.Positional, EditorKind.Text, Required: true),
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
            new ParamDef("tokenA", "token A", ParamKind.Positional, EditorKind.Text, Required: true),
            new ParamDef("tokenB", "token B", ParamKind.Positional, EditorKind.Text, Required: true),
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
            new ParamDef("prompt", "--prompt", ParamKind.Option, EditorKind.MultiLineText, Flag: "--prompt",
                Help: "The prompt text. Leave empty to start from --tokens instead.", Required: true),
            new ParamDef("tokens", "--tokens", ParamKind.Option, EditorKind.IntList, Flag: "--tokens",
                Help: "Comma-separated start ids (used when --prompt is absent; default 0)."),
            TokenizerParam,
            new ParamDef("steps", "--steps", ParamKind.Option, EditorKind.Int, Flag: "--steps", DefaultValue: "",
                Help: "Tokens to generate. Empty = until an end token (8 for raw --tokens without a tokenizer); "
                      + "-1 also means until an end token. For a diffusion model, the number of denoising steps."),
            Opt("maxContext", "--max-context", EditorKind.Text, "", "Run at this context length instead of the model's own, e.g. 1M."),
            Opt("kvFormat", "--kv-format", EditorKind.Choice, "", "How the conversation is held in GPU memory (nvfp4 for long contexts).",
                choices: new[] { "", "fp16", "nvfp4", "f32" }),
            new ParamDef("temperature", "--temperature", ParamKind.Option, EditorKind.Float, Flag: "--temperature", DefaultValue: "0"),
            new ParamDef("topK", "--top-k", ParamKind.Option, EditorKind.Int, Flag: "--top-k", DefaultValue: "0"),
            new ParamDef("topP", "--top-p", ParamKind.Option, EditorKind.Float, Flag: "--top-p", DefaultValue: "0"),
            new ParamDef("seed", "--seed", ParamKind.Option, EditorKind.Int, Flag: "--seed", DefaultValue: "42"),
            new ParamDef("chat", "--chat", ParamKind.Switch, EditorKind.Choice, Flag: "--chat",
                DefaultValue: "", Choices: new[] { "", "1" },
                Help: "Wrap the prompt in the model's own chat template before "
                      + "encoding. Requires a tokenizer with a chat_template.jinja or "
                      + "tokenizer_config.json."),
            Opt("system", "--system text", EditorKind.MultiLineText, "",
                "System message placed in the chat template. Requires --chat."),
            Opt("image", "--image file", EditorKind.File, "",
                "PNG or JPEG the prompt is about (';' separates several). Requires --chat, the GPU and a container with a vision tower."),
            new ParamDef("logits", "--logits K", ParamKind.Option, EditorKind.Int, Flag: "--logits",
                Help: "Show top-K logits per step (leave empty to hide)."),
            Opt("logprobs", "--logprobs K", EditorKind.Int, "",
                "Report the log-probability of each generated token and its top-K alternatives. "
                + "Leave empty to hide; 5 matches the bare flag."),
            Flag("uncertainty", "--uncertainty",
                "Summarise how unsure the model was over the run: perplexity, mean entropy and mean "
                + "top-1 margin. Needs --logprobs."),
            ModelWeightsParam,
            new ParamDef("trace", "--trace", ParamKind.Switch, EditorKind.Choice, Flag: "--trace",
                DefaultValue: "", Choices: new[] { "", "1" },
                Help: "Per-layer residual norm trace for each generated token."),
            new ParamDef("traceTensors", "--trace-tensors", ParamKind.Switch, EditorKind.Choice, Flag: "--trace-tensors",
                DefaultValue: "", Choices: new[] { "", "1" },
                Help: "Weight-load trace: distinct tensors pulled, load counts and how many were "
                      + "cold, including 1-D norm weights."),
            new ParamDef("traceJson", "--trace-json file.json", ParamKind.Option, EditorKind.File, Flag: "--trace-json",
                Help: "Writes a per-operator trace of every generated token: which operators ran, "
                      + "their output magnitude, which experts a MoE layer routed to, and the "
                      + "resulting top-k distribution. This is the file the inference visualiser opens."),
            new ParamDef("traceStream", "--trace-stream file.jsonl", ParamKind.Option, EditorKind.File,
                Flag: "--trace-stream",
                Help: "Like --trace-json but line-delimited and flushed per record, so it can be "
                      + "tailed while the run is still going. This is what the visualiser's live "
                      + "view reads."),
            new ParamDef("logitLens", "--logit-lens", ParamKind.Switch, EditorKind.Choice, Flag: "--logit-lens",
                DefaultValue: "", Choices: new[] { "", "1" },
                Help: "Also project each layer's residual through the final norm and output head, "
                      + "recording what that layer would have emitted. Shows the depth at which the "
                      + "prediction forms. Costs a full head GEMM per layer per step, so it is slow "
                      + "on a large model. Requires --trace-json."),
            new ParamDef("traceAttention", "--trace-attention", ParamKind.Switch, EditorKind.Choice,
                Flag: "--trace-attention", DefaultValue: "", Choices: new[] { "", "1" },
                Help: "Record each softmax-attention head's post-softmax weights for the last query "
                      + "row, per step. Recurrent layers produce none. Requires --trace-json."),
            new ParamDef("attribute", "--attribute", ParamKind.Switch, EditorKind.Choice, Flag: "--attribute",
                DefaultValue: "", Choices: new[] { "", "1" },
                Help: "Run ROME-style causal attribution after generating: corrupt one prompt token, "
                      + "then restore each layer's clean residual in turn and re-measure the target's "
                      + "probability. One extra forward per layer. Requires --trace-json and "
                      + "--attribute-corrupt."),
            new ParamDef("attributeCorrupt", "--attribute-corrupt text", ParamKind.Option, EditorKind.Text,
                Flag: "--attribute-corrupt",
                Help: "The token that replaces the source position in the corrupted runs, e.g. 'the'. "
                      + "There is no default: the replacement defines the question being asked."),
            new ParamDef("attributeSource", "--attribute-source row", ParamKind.Option, EditorKind.Int,
                Flag: "--attribute-source", Help: "Prompt row to corrupt. Empty = the last prompt token."),
            new ParamDef("attributeLayers", "--attribute-layers end", ParamKind.Option, EditorKind.Int,
                Flag: "--attribute-layers", Help: "Trace only layers below this index. Empty = all."),
            Flag("think", "--think", "With --chat: turn the chat template's thinking on."),
            Opt("gen", "--gen", EditorKind.Int, "", "Diffusion models: how many tokens to generate (default 128)."),
            Opt("block", "--block", EditorKind.Int, "", "Diffusion models: block length denoised at a time (default 32)."),
            Opt("remasking", "--remasking", EditorKind.Choice, "",
                "Diffusion models: which tokens are re-masked each step (default low_confidence).",
                choices: new[] { "", "low_confidence", "random" }),
            Flag("showSteps", "--show-steps", "Diffusion models: print the canvas after every denoising step."),
            Flag("noEarlyStop", "--no-early-stop", "Diffusion models: keep denoising after an end token appears."),
            ComponentParam,
            PatchParam,
        }),
        new("inspect-token", CatPathways, "Embedding profile, neighbours and logits for one token.", new[]
        {
            ContainerParam,
            new ParamDef("token", "token id", ParamKind.Positional, EditorKind.Int, Required: true),
            new ParamDef("tokens", "--tokens ctx ids", ParamKind.Option, EditorKind.IntList, Flag: "--tokens",
                Help: "Context ids for the logits inspection."),
            new ParamDef("neighbors", "--neighbors", ParamKind.Option, EditorKind.Int, Flag: "--neighbors", DefaultValue: "5"),
            new ParamDef("logits", "--logits K", ParamKind.Option, EditorKind.Int, Flag: "--logits", DefaultValue: "5"),
            TokenizerParam,
            ComponentParam,
            PatchParam,
        }),

        // ── 3a. Inference & Benchmarks ──────────────────────────────────────
        new("bench", CatBench, "Measure prefill and decode speed (tokens per second) on a container.", new[]
        {
            ContainerParam,
            Opt("promptTokens", "--prompt-tokens", EditorKind.Int, "32", "Length of the synthetic prompt."),
            Opt("steps", "--steps", EditorKind.Int, "64", "Tokens decoded per run."),
            Opt("repeat", "--repeat", EditorKind.Int, "3", "Timed runs after the warm-up."),
            ModelWeightsParam,
            Flag("profile", "--profile", "Break the time down per operator."),
            Flag("compare", "--compare", "Also check the device's logits against the CPU reference."),
            Opt("batch", "--batch N", EditorKind.Int, "", "Decode N sequences together and report the batched rate."),
            Opt("draft", "--draft N", EditorKind.Int, "", "Decode with up to N drafts a step from the container's drafter, checked against plain decode."),
            Opt("chat", "--chat text", EditorKind.Text, "", "With --draft: the message to reply to (text or a file)."),
        }),
        new("context bench", CatBench, "Measure how well a model uses a long context: facts recalled, passages quoted, speed.", new[]
        {
            ContainerParam,
            Opt("text", "--text file or folder", EditorKind.File, "", "Text the long document is made of (';' separates several).", required: true),
            Opt("lengths", "--lengths", EditorKind.Text, "8K,32K,128K,256K", "Context lengths to measure at, in one reading."),
            Opt("questions", "--questions", EditorKind.Int, "24", "Questions asked at each length."),
            Opt("needleEvery", "--needle-every", EditorKind.Int, "2048", "Tokens between the facts planted in the document."),
            Flag("plain", "--plain", "Read the document as bare text and measure perplexity instead of questions."),
            Opt("maxContext", "--max-context", EditorKind.Text, "", "Run the model at this context length, e.g. 1M."),
            Opt("ropeExtend", "--rope-extend", EditorKind.Choice, "auto", "How positions past the trained length rotate.",
                choices: new[] { "auto", "none", "yarn", "ntk", "linear" }),
            Opt("out", "--out report.json", EditorKind.File, "", "Report to write."),
        }),
        new("tool-trace", CatBench, "Trace how a model reads the tools in its prompt, case by case.", new[]
        {
            ContainerParam,
            Opt("cases", "--cases file.jsonl", EditorKind.File, "", "Tool-use cases, one JSON object per line.", required: true),
            Opt("out", "--out report.json", EditorKind.File, "", "Report to write.", required: true),
            Opt("poolLayers", "--pool-layers", EditorKind.Text, "-1,3,7,11,15,19,23",
                "Comma-separated layers whose hidden states are pooled (-1 = embeddings)."),
            TokenizerParam,
            ComponentParam,
        }),
        new("tool-eval", CatBench, "Check a running server's tool calls against expected answers.", new[]
        {
            Opt("cases", "--cases file.jsonl", EditorKind.File, "",
                "Cases: messages, tools and the expected tool call, one JSON object per line.", required: true),
            Opt("endpoint", "--endpoint url", EditorKind.Text, "http://127.0.0.1:8080/v1",
                "OpenAI-compatible base URL (amql-server, llama-server, vLLM…)."),
            Opt("model", "--model", EditorKind.Text, "default", "Model name sent with each request."),
            Opt("maxTokens", "--max-tokens", EditorKind.Int, "2048"),
            Flag("noThink", "--no-think", "Ask the server to turn thinking off."),
            Opt("repeat", "--repeat", EditorKind.Int, "1", "Run every case this many times."),
            Opt("report", "--report file.json", EditorKind.File, "", "Write the per-case results here."),
        }),

        // ── 4. Patching & LoRA ──────────────────────────────────────────────
        new("change-tensor", CatPatching, "Edit one weight cell into a patch file.", new[]
        {
            ContainerParam,
            new ParamDef("objectId", "object id", ParamKind.Positional, EditorKind.Text, Help: "e.g. target.embedding", Required: true),
            new ParamDef("tensorName", "tensor name", ParamKind.Positional, EditorKind.Text, Help: "e.g. weight, 0.self_attn.q_proj.weight", Required: true),
            new ParamDef("cell", "cell", ParamKind.Positional, EditorKind.Text, Help: "'row,col' for 2-D, flat index otherwise", Required: true),
            new ParamDef("op", "operation", ParamKind.Option, EditorKind.Choice, Flag: null, DefaultValue: "--set",
                Choices: new[] { "--set", "--add", "--scale", "--zero" },
                Help: "Exactly one edit operation.", Required: true),
            new ParamDef("value", "value", ParamKind.Option, EditorKind.Float, Flag: null,
                Help: "Value for --set/--add/--scale (ignored for --zero).", Required: true),
            new ParamDef("out", "--out patch", ParamKind.Option, EditorKind.File, Flag: "--out",
                Help: "patch.safetensors to write (existing patch is loaded and composed).", Required: true),
            PatchParam,
        }),
        new("edit-tensor", CatPatching, "Scale, zero or offset an entire weight tensor into a patch file.", new[]
        {
            ContainerParam,
            new ParamDef("objectId", "object id", ParamKind.Positional, EditorKind.Text, Help: "e.g. target.decoder_stack", Required: true),
            new ParamDef("tensorName", "tensor name", ParamKind.Positional, EditorKind.Text, Help: "e.g. 3.self_attn.q_proj.weight", Required: true),
            new ParamDef("op", "operation", ParamKind.Option, EditorKind.Choice, Flag: null, DefaultValue: "--scale",
                Choices: new[] { "--scale", "--zero", "--add", "--set" },
                Help: "Applied to every element. --scale 0.5 halves the tensor; --zero removes its "
                      + "contribution entirely. To edit a single cell use change-tensor, though one "
                      + "cell of a large matrix is too small to measure downstream.", Required: true),
            new ParamDef("value", "value", ParamKind.Option, EditorKind.Float, Flag: null,
                Help: "Value for --scale/--add/--set (ignored for --zero).", Required: true),
            new ParamDef("out", "--out patch", ParamKind.Option, EditorKind.File, Flag: "--out",
                Help: "patch.safetensors to write (existing patch is loaded and composed).", Required: true),
            PatchParam,
        }),
        new("save-lora", CatPatching, "Factor a patch's 2-D deltas into lora_A/lora_B.", new[]
        {
            new ParamDef("patchFile", "patch file", ParamKind.Positional, EditorKind.File, Required: true),
            OutDirParam with { Label = "--out lora dir" },
            new ParamDef("rank", "--rank", ParamKind.Option, EditorKind.Int, Flag: "--rank", DefaultValue: "8"),
            new ParamDef("alpha", "--alpha", ParamKind.Option, EditorKind.Float, Flag: "--alpha", DefaultValue: "16"),
            new ParamDef("container", "--container dir", ParamKind.Option, EditorKind.Directory, Flag: "--container",
                DefaultFrom: "container", Help: "Validate the patch against this container before writing."),
        }),

        // ── 8. Fine-Tuning ──────────────────────────────────────────────────
        new("fine-tune", CatTraining, "Teacher-forced output-head adaptation → weight patch.", new[]
        {
            ContainerParam,
            new ParamDef("data", "--data pairs.tsv", ParamKind.Option, EditorKind.File, Flag: "--data",
                Help: "TSV of prompt<TAB>completion lines.", Required: true),
            new ParamDef("out", "--out patch", ParamKind.Option, EditorKind.File, Flag: "--out", Required: true),
            TokenizerParam,
            ComponentParam,
            new ParamDef("lr", "--lr", ParamKind.Option, EditorKind.Float, Flag: "--lr", DefaultValue: "1e-4"),
            new ParamDef("epochs", "--epochs", ParamKind.Option, EditorKind.Int, Flag: "--epochs", DefaultValue: "1"),
        }),
        new("bake-predicate", CatTraining, "Write a rule the model reads in context into its weights → weight patch.", new[]
        {
            ContainerParam,
            new ParamDef("predicate", "--predicate", ParamKind.Option, EditorKind.MultiLineText, Flag: "--predicate",
                Help: "The rule, as you would put it in a system prompt. The model interprets it itself.",
                Required: true),
            Opt("predicateFile", "--predicate-file", EditorKind.File, "",
                "Read the rule from a file instead of --predicate."),
            new ParamDef("out", "--out patch", ParamKind.Option, EditorKind.File, Flag: "--out", Required: true),
            new ParamDef("prompts", "--prompts file", ParamKind.Option, EditorKind.File, Flag: "--prompts",
                Help: "Probe user turns, one per line. Omit to let the model write its own."),
            new ParamDef("selfProbes", "--self-probes", ParamKind.Option, EditorKind.Int, Flag: "--self-probes", DefaultValue: "8"),
            new ParamDef("layers", "--layers", ParamKind.Option, EditorKind.Text, Flag: "--layers",
                Help: "Comma-separated layers to edit. Omit for evenly spaced middle layers."),
            Opt("layerCount", "--layer-count", EditorKind.Int, "4",
                "How many evenly spaced layers to edit when --layers is not given."),
            new ParamDef("ridge", "--ridge", ParamKind.Option, EditorKind.Float, Flag: "--ridge", DefaultValue: "1e-2",
                Help: "Larger keeps the change smaller; smaller follows the predicated model more tightly."),
            new ParamDef("maxTokens", "--max-tokens", ParamKind.Option, EditorKind.Int, Flag: "--max-tokens", DefaultValue: "48"),
            Opt("samples", "--samples", EditorKind.Int, "3", "Sample answers printed in the report."),
            Flag("raw", "--raw", "Put the rule in as a plain preamble instead of a chat-template system message."),
            TokenizerParam,
            ComponentParam,
        }),
        new("lora-train", CatTraining, "Train a LoRA adapter on prompt/completion examples.", new[]
        {
            ContainerParam,
            Opt("data", "--data train.jsonl", EditorKind.File, "",
                "One {\"id\",\"prompt\",\"completion\"} object per line, prompt already rendered.", required: true),
            Opt("out", "--out lora dir", EditorKind.Directory, "", "Adapter directory to write.", required: true),
            Opt("eval", "--eval file.jsonl", EditorKind.File, "", "Held-out examples scored during training."),
            Opt("anchors", "--anchors file.jsonl", EditorKind.File, "",
                "Examples whose original behaviour must be kept (guards against forgetting)."),
            Opt("layers", "--layers", EditorKind.Text, "", "Comma-separated layers to adapt. Empty = the default set."),
            Opt("rank", "--rank", EditorKind.Int, "8"),
            Opt("alpha", "--alpha", EditorKind.Float, "16"),
            Opt("lr", "--lr", EditorKind.Float, "2e-3"),
            Opt("steps", "--steps", EditorKind.Int, "60"),
            Opt("evalEvery", "--eval-every", EditorKind.Int, "10"),
            Opt("clip", "--clip", EditorKind.Float, "1.0", "Gradient-norm clip."),
            Opt("weightDecay", "--weight-decay", EditorKind.Float, "0"),
            Opt("targetLoss", "--target-loss", EditorKind.Float, "0.02", "Stop once the training loss is below this."),
            Opt("gradChecks", "--grad-checks", EditorKind.Int, "2", "Finite-difference gradient checks before training."),
            Opt("seed", "--seed", EditorKind.Int, "1234"),
            Opt("anchorTokens", "--anchor-tokens", EditorKind.Int, "40"),
            Opt("anchorWeight", "--anchor-weight", EditorKind.Float, "1.0"),
            Opt("batch", "--batch", EditorKind.Int, "0", "Examples per step (0 = all)."),
            Flag("noDecode", "--no-decode", "Skip the sample generations during evaluation."),
            Flag("keepBest", "--keep-best", "Keep the adapter from the best evaluation, not the last step."),
            TokenizerParam,
            ComponentParam,
        }),

        // ── 5. Export & Conversion ──────────────────────────────────────────
        new("export", CatExport, "Materialise an HF checkpoint (or GGUF / ONNX file) from the container.", new[]
        {
            ContainerParam,
            OutDirParam with
            {
                Label = "--out checkpoint dir / file",
                Help = "Checkpoint directory for hf; the .gguf or .onnx file for those formats.",
            },
            Opt("format", "--format", EditorKind.Choice, "hf",
                "hf: safetensors + config.json. gguf: llama.cpp layout chosen from the model's properties, "
                + "weights at their stored precision. onnx: an ONNX model.",
                choices: new[] { "hf", "gguf", "onnx" }),
            PatchParam,
            new ParamDef("quant", "--quant", ParamKind.Option, EditorKind.Choice, Flag: "--quant", DefaultValue: "none",
                Choices: new[] { "none", "mxfp4", "ptq1", "pq2" },
                Help: "Weight format for the exported checkpoint. 'none' keeps F16; 'mxfp4' is OCP "
                      + "MXFP4 (FP4 E2M1 with E8M0 block scales, 32-element blocks). 'ptq1' is ternary "
                      + "PTQ1_0 (5 trits/byte, ~1.75 bpw) and 'pq2' is ternary PQ2_0 (2 bits/trit, "
                      + "~2.13 bpw); both use 128-element blocks with FP16 scales and a blockwise "
                      + "Hadamard rotation, and produce PrismML Bonsai-compatible checkpoints for the "
                      + "Bonsai llama.cpp fork. Their 142/143 type ids are that fork's own numbering, "
                      + "not stock ggml's. The list is shorter than to-gguf's on purpose: safetensors "
                      + "has no dtype for Q4_0, and mxfp4_moe is a GGUF recipe, so those live only in "
                      + "to-gguf. The CLI takes the packing explicitly — there is no bare 'ternary' "
                      + "option. Not used with --format gguf."),
            new ParamDef("arch", "--arch", ParamKind.Option, EditorKind.Choice, Flag: "--arch", DefaultValue: "qwen3.x",
                Choices: new[] { "qwen3.x", "qwen4-next" },
                Help: "Target architecture: qwen3.x (default Qwen3.5) or qwen4-next (Flash-Next)."),
            Flag("listArchitectures", "--list-architectures",
                "Only list the architectures this build can export, then stop."),
        }),
        new("to-gguf", CatExport, "Convert an exported HF checkpoint to GGUF v3.", new[]
        {
            new ParamDef("checkpointDir", "checkpoint or container dir", ParamKind.Positional, EditorKind.Directory,
                Help: "An exported HF checkpoint, or a VINDEX3 container — a container is bridged "
                      + "through a temporary checkpoint beside the output file and cleaned up after, "
                      + "so one command goes straight from container to GGUF.", Required: true),
            new ParamDef("out", "--out file.gguf", ParamKind.Option, EditorKind.File, Flag: "--out", Required: true),
            new ParamDef("quant", "--quant", ParamKind.Option, EditorKind.Choice, Flag: "--quant", DefaultValue: "none",
                Choices: new[] { "none", "f16", "q4_0", "mxfp4", "mxfp4_moe", "ptq1", "pq2" },
                Help: "Weight encoding: none/f16 (full precision), q4_0 (ggml 4-bit, ftype 2), "
                      + "mxfp4 (OCP MXFP4, ggml type 39) for every quantizable weight, mxfp4_moe "
                      + "(llama.cpp ftype 38) which puts 3-D MoE expert stacks in MXFP4 and every "
                      + "other quantizable weight in Q8_0, or ptq1/pq2 — the PrismML Bonsai ternary "
                      + "packings at ggml type ids 143/142, which only the Bonsai fork loads. "
                      + "Norms, embeddings, the output head, "
                      + "routers and 1-D tensors stay full precision in every mode."),
            Opt("recipe", "--recipe types.txt", EditorKind.File, "",
                "Per-tensor type list (lfm3, a2d-qwen3 checkpoints): which ggml type each tensor is written as."),
            Opt("imatrix", "--imatrix file.gguf", EditorKind.File, "",
                "Importance matrix that guides the quantisation chosen by --recipe."),
            Opt("ggml", "--ggml dir", EditorKind.Directory, "",
                "ggml-base build directory used to quantise the --recipe types."),
            Flag("force", "--force", "Overwrite the output file if it already exists."),
        }),
        new("from-gguf", CatIngest, "Import a GGUF file into a VINDEX3 container.", new[]
        {
            new ParamDef("ggufFile", ".gguf file", ParamKind.Positional, EditorKind.File,
                Help: "GGUF v3 file to import (Qwen3.5-family architectures).", Required: true),
            OutDirParam with { Label = "--out container dir", Help = "Container directory to write." },
        }),
        new("export-onnx", CatExport, "Export a container as an ONNX model (.onnx).", new[]
        {
            ContainerParam,
            new ParamDef("out", "--out model.onnx", ParamKind.Option, EditorKind.File, Flag: "--out",
                Help: "Output .onnx file path.", Required: true),
            Flag("int4", "--int4", "Store the weights as 4-bit integers."),
            ComponentParam,
            PatchParam,
        }),
        new("create-model", CatIngest, "Create a new empty container with random weights.", new[]
        {
            new ParamDef("out", "--out container dir", ParamKind.Option, EditorKind.Directory, Flag: "--out",
                Help: "Container directory to write.", Required: true),
            new ParamDef("name", "--name", ParamKind.Option, EditorKind.Text, Flag: "--name", DefaultValue: "untitled-model"),
            Opt("family", "--family", EditorKind.Text, "qwen3_5", "Model family recorded in the container."),
            new ParamDef("hidden", "--hidden", ParamKind.Option, EditorKind.Int, Flag: "--hidden", DefaultValue: "768"),
            new ParamDef("layers", "--layers", ParamKind.Option, EditorKind.Int, Flag: "--layers", DefaultValue: "12"),
            new ParamDef("heads", "--heads", ParamKind.Option, EditorKind.Int, Flag: "--heads", DefaultValue: "12"),
            new ParamDef("kvHeads", "--kv-heads", ParamKind.Option, EditorKind.Int, Flag: "--kv-heads", DefaultValue: "2"),
            new ParamDef("headDim", "--head-dim", ParamKind.Option, EditorKind.Int, Flag: "--head-dim", DefaultValue: "64"),
            new ParamDef("intermediate", "--intermediate", ParamKind.Option, EditorKind.Int, Flag: "--intermediate", DefaultValue: "2048"),
            new ParamDef("vocab", "--vocab", ParamKind.Option, EditorKind.Int, Flag: "--vocab", DefaultValue: "32000"),
            new ParamDef("context", "--context", ParamKind.Option, EditorKind.Int, Flag: "--context", DefaultValue: "2048"),
            Opt("normEps", "--norm-eps", EditorKind.Float, "1e-5"),
            new ParamDef("layerTypes", "--layer-types", ParamKind.Option, EditorKind.Text, Flag: "--layer-types",
                Help: "Comma-separated list of 'full_attention'/'linear_attention' per layer (e.g. 'full,full,linear' or a repeated pattern 'full,linear')."),
            new ParamDef("untied", "--untied", ParamKind.Switch, EditorKind.Choice, Flag: "--untied",
                DefaultValue: "", Choices: new[] { "", "1" },
                Help: "Untied output head (separate from embedding)."),
        }),

        // ── 5a. Quantisation ────────────────────────────────────────────────
        new("quantise", CatQuant, "Round a container's weights to 4-bit NVFP4 or MXFP4 (GPTQ by default).", new[]
        {
            ContainerParam,
            OutDirParam with { Label = "--out quantised dir", Help = "Container directory to write." },
            Opt("format", "--format", EditorKind.Choice, "nvfp4", "4-bit format to write.",
                choices: new[] { "nvfp4", "mxfp4" }),
            Opt("rounding", "--rounding", EditorKind.Choice, "gptq",
                "gptq: error-correcting rounding calibrated on text (best quality). search: per-block scale search. "
                + "nearest: plain round to nearest (fastest).",
                choices: new[] { "gptq", "search", "nearest" }),
            Opt("calibration", "--calibration file or folder", EditorKind.File, "",
                "Calibration text for GPTQ. Empty = a built-in text."),
            Flag("embedding", "--embedding", "Also the token embedding and the head. Smaller and faster, at some cost in quality."),
            Opt("calibrateWith", "--calibrate-with container", EditorKind.Directory, "",
                "For a model too large for the GPU at full precision: a 4-bit copy of it (its own nearest-rounded one, say) that reads the calibration text. Empty = the model itself."),
            Opt("calibrationTokens", "--calibration-tokens", EditorKind.Int, "32768"),
            Opt("sequenceTokens", "--sequence-tokens", EditorKind.Int, "1024"),
            Opt("damp", "--damp", EditorKind.Float, "0.01", "GPTQ dampening, as a fraction of the mean diagonal."),
            Opt("keep", "--keep regex", EditorKind.Text, "",
                "Projections whose names match stay at full precision. Keep a layer's q, k and v together."),
            Opt("reuse", "--reuse container", EditorKind.Directory, "",
                "A quantised copy of the same model: its 4-bit weights are copied rather than rounded again, so trying another --keep or --scale takes minutes."),
            Opt("scale", "--scale regex=factor", EditorKind.Text, "",
                "Multiply the tensors that match, an edit with no training. A norm read as x·(1 + w) has 1 + w multiplied. Gemma 4 26B: '^(5|11|17|23|29)\\.self_attn\\.q_norm\\.weight$=1.5' sharpens its global attention for long contexts."),
        }),
        new("quantise compare", CatQuant, "Compare quantised copies against the original: KL, perplexity, agreement.", new[]
        {
            Pos("reference", "reference container", EditorKind.Directory, "The full-precision container."),
            Pos("candidate", "candidate container", EditorKind.Directory, "A quantised copy to compare against it."),
            Opt("text", "--text file or folder", EditorKind.File, "", "Text the models are scored on.", required: true),
            Opt("tokens", "--tokens", EditorKind.Int, "4096", "How many tokens of the text to score."),
        }),

        // ── 8. Classify ─────────────────────────────────────────────────────
        new("classify", CatPathways,
            "Run a sequence-classification head: format the input by the recorded template, pool the last "
            + "real token, score it and name the label.", new[]
        {
            ContainerParam,
            new ParamDef("premise", "--premise", ParamKind.Option, EditorKind.Text, Flag: "--premise"),
            new ParamDef("hypothesis", "--hypothesis", ParamKind.Option, EditorKind.Text, Flag: "--hypothesis"),
            new ParamDef("text", "--text premise|hypothesis", ParamKind.Option, EditorKind.Text, Flag: "--text",
                Help: "One string split on the first | into premise and hypothesis (a single text when no template is recorded)."),
            new ParamDef("input", "--input text", ParamKind.Option, EditorKind.MultiLineText, Flag: "--input",
                Help: "A single text used verbatim, for classifiers without a pair template."),
            new ParamDef("jsonl", "--jsonl file", ParamKind.Option, EditorKind.File, Flag: "--jsonl",
                Help: "One {\"premise\",\"hypothesis\"} or {\"text\"} object per line."),
            new ParamDef("format", "--format", ParamKind.Option, EditorKind.Choice, Flag: "--format", DefaultValue: "labels",
                Choices: new[] { "labels", "json", "jsonl", "csv" }),
            new ParamDef("maxTokens", "--max-tokens", ParamKind.Option, EditorKind.Int, Flag: "--max-tokens", DefaultValue: "4096"),
            PatchParam,
        }),

        // ── 8b. Decide (Von) ───────────────────────────────────────────────
        new("decide", CatPathways,
            "Answer a TypeSafe /v1/decisions request (choice / score / noul questions over one state) "
            + "with a Von decision-model container, on the in-process ModernBERT encoder.", new[]
        {
            ContainerParam,
            new ParamDef("request", "--request json | @file | -", ParamKind.Option, EditorKind.MultiLineText, Flag: "--request",
                Help: "The /v1/decisions JSON body: {\"state\": …, \"questions\": {id: {\"type\", \"instructions\", \"criteria\"}}}. "
                      + "Prefix a path with @ to read it from a file.", Required: true),
            new ParamDef("envelope", "--envelope", ParamKind.Option, EditorKind.Choice, Flag: "--envelope", DefaultValue: "none",
                Choices: new[] { "none", "jevai" },
                Help: "none: the OpenJEV / Von body { model, answers, usage }. jevai: wrapped as { code, message, data }."),
            new ParamDef("attention", "--attention", ParamKind.Option, EditorKind.Choice, Flag: "--attention", DefaultValue: "auto",
                Choices: new[] { "auto", "independent", "full" },
                Help: "auto uses the mode the checkpoint was trained with (marker_calibration.json)."),
            Flag("compact", "--compact", "Print the answer JSON on one line."),
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
            TextParam with { Help = "Corpus text file. Required with --fit." },
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
            TextParam with { Required = true },
            new ParamDef("fit", "--fit", ParamKind.Option, EditorKind.Int, Flag: "--fit", DefaultValue: "512"),
            new ParamDef("gate", "--gate", ParamKind.Option, EditorKind.Int, Flag: "--gate", DefaultValue: "256"),
        }),
        new("fit-mtp", CatMtp, "Fit the drafter projector by closed-form ridge least squares.", new[]
        {
            ContainerParam,
            new ParamDef("pairs", "--pairs dir", ParamKind.Option, EditorKind.Directory, Flag: "--pairs", Required: true),
            new ParamDef("ridge", "--ridge", ParamKind.Option, EditorKind.Float, Flag: "--ridge", DefaultValue: "1e-4"),
            new ParamDef("clusters", "--clusters K", ParamKind.Option, EditorKind.Int, Flag: "--clusters"),
            new ParamDef("sweep", "--sweep K1,K2,K3", ParamKind.Option, EditorKind.Text, Flag: "--sweep"),
        }),

        // ── 7. Model Transformation ─────────────────────────────────────────
        new("import", CatTransform, "Merge a second model into the container (vocab union).", new[]
        {
            ContainerParam,
            new ParamDef("model", "model to import", ParamKind.Positional, EditorKind.Directory,
                Help: "Checkpoint dir, or a container dir when --container is set.", Required: true),
            OutDirParam with { Label = "--out merged dir" },
            new ParamDef("asContainer", "--container", ParamKind.Switch, EditorKind.Text, Flag: "--container",
                Help: "The model to import is itself a container."),
            Opt("blend", "--blend", EditorKind.Float, "0.5", "How much of the imported model to mix in (0 to 1)."),
            Opt("vocab", "--vocab", EditorKind.Choice, "union",
                "union: grow to both vocabularies. base: keep only the base model's vocabulary.",
                choices: new[] { "union", "base" }),
        }),
        new("moe-ify", CatTransform, "Restructure a dense container into a routed MoE.", new[]
        {
            ContainerParam,
            OutDirParam with { Label = "--out moe dir" },
            TextParam with { Required = true },
            new ParamDef("experts", "--experts", ParamKind.Option, EditorKind.Int, Flag: "--experts", DefaultValue: "8"),
            new ParamDef("topK", "--top-k", ParamKind.Option, EditorKind.Int, Flag: "--top-k", DefaultValue: "2"),
            new ParamDef("sample", "--sample", ParamKind.Option, EditorKind.Int, Flag: "--sample", DefaultValue: "4096"),
            new ParamDef("eval", "--eval", ParamKind.Option, EditorKind.Int, Flag: "--eval", DefaultValue: "1024"),
            new ParamDef("policy", "--policy", ParamKind.Option, EditorKind.Choice, Flag: "--policy", DefaultValue: "softmax",
                Choices: new[] { "softmax", "renormalise" }),
        }),
        new("prune", CatTransform, "Shrink a container to a byte budget by dropping layers or narrowing them.", new[]
        {
            ContainerParam,
            OutDirParam with { Label = "--out pruned dir" },
            new ParamDef("targetBytes", "--target-bytes", ParamKind.Option, EditorKind.Text, Flag: "--target-bytes",
                Help: "Bare bytes or B/KB/MB/GB/TB, KiB/MiB/GiB/TiB, e.g. 2GiB", Required: true),
            new ParamDef("approach", "--approach", ParamKind.Option, EditorKind.Choice, Flag: "--approach", DefaultValue: "provenance",
                Choices: new[] { "provenance", "corpus", "random", "star-elastic" },
                Help: "provenance/corpus/random choose which whole layers to drop. star-elastic instead narrows "
                      + "the FFN and attention heads, ranked on --text (needs --text and a tokenizer beside the container)."),
            new ParamDef("seed", "--seed", ParamKind.Option, EditorKind.Int, Flag: "--seed", DefaultValue: "42"),
            TextParam,
            new ParamDef("sample", "--sample", ParamKind.Option, EditorKind.Int, Flag: "--sample", DefaultValue: "8192"),
            new ParamDef("minLayers", "--min-layers", ParamKind.Option, EditorKind.Int, Flag: "--min-layers", DefaultValue: "1"),
            Opt("minIntermediate", "--min-intermediate", EditorKind.Int, "1",
                "star-elastic: the FFN is never narrowed below this many channels."),
        }),
        new("convert-to-classifier", CatTransform, "Turn a text-generating model into a classifier with N labels.", new[]
        {
            ContainerParam,
            OutDirParam with { Label = "--out classifier dir", Help = "Container directory to write." },
            Opt("numLabels", "--num-labels", EditorKind.Int, "2", "How many labels the new score head has."),
        }),
        new("convert-to-embedding", CatTransform, "Turn a text-generating model into an embedding model.", new[]
        {
            ContainerParam,
            OutDirParam with { Label = "--out embedding dir", Help = "Container directory to write." },
        }),
        new("strip", "2a. Slim Containers", "Keep one encoding, drop the rest — single-precision slim copy.", new[]
        {
            ContainerParam,
            new ParamDef("keep", "--keep encoding", ParamKind.Option, EditorKind.Choice, Flag: "--keep",
                DefaultValue: "Q4_0", Choices: new[] { "Q4_0", "Q8_0", "FP4", "BF16", "FP32" },
                Help: "Encoding to keep (all others discarded).", Required: true),
            new ParamDef("out", "--out slim dir", ParamKind.Option, EditorKind.Directory, Flag: "--out",
                Help: "Output directory (default: <container>-<encoding>)."),
        }),

        // ── 7a. MoE Experts ─────────────────────────────────────────────────
        new("moe-eval", CatMoe, "Score LoRA experts per topic, alone and mixed in different ways.", new[]
        {
            ContainerParam,
            Opt("data", "--data eval.jsonl", EditorKind.File, "", "Examples with a \"topic\" per line.", required: true),
            Opt("experts", "--experts name=dir,…", EditorKind.Text, "",
                "LoRA experts, comma-separated name=<lora-dir> pairs. Empty = score the container as it is."),
            Opt("mix", "--mix", EditorKind.Text, "base,experts,average,sum,oracle",
                "Mixtures to score, comma-separated: base, experts, average, sum, oracle."),
            Opt("router", "--router name=file,…", EditorKind.Text, "",
                "Trained routers to score too, comma-separated name=<router.safetensors> pairs."),
            Opt("topK", "--top-k", EditorKind.IntList, "0", "Router top-k values to try, comma-separated (0 = all experts)."),
            Opt("report", "--report file.json", EditorKind.File, "", "Write the results as JSON."),
            TokenizerParam,
            ComponentParam,
        }),
        new("moe-router", CatMoe, "Train a router that picks which LoRA expert to use per token.", new[]
        {
            ContainerParam,
            Opt("experts", "--experts name=dir,…", EditorKind.Text, "",
                "LoRA experts, comma-separated name=<lora-dir> pairs, in router order.", required: true),
            Opt("data", "--data route.jsonl", EditorKind.File, "", "Examples with a \"topic\" per line.", required: true),
            Opt("out", "--out router.safetensors", EditorKind.File, "", "Router file to write.", required: true),
            Opt("method", "--method", EditorKind.Choice, "loss",
                "loss: learn from which expert lowers the loss. label: learn from the topic labels.",
                choices: new[] { "loss", "label" }),
            Opt("hold", "--hold", EditorKind.Choice, "token",
                "token: route every token. prompt: choose once from the prompt and hold it.",
                choices: new[] { "token", "prompt" }),
            Opt("init", "--init router", EditorKind.File, "", "Start from an existing router."),
            Opt("steps", "--steps", EditorKind.Int, "", "Training steps (default 30 for loss, 200 for label)."),
            Opt("lr", "--lr", EditorKind.Float, "1e-2"),
            Opt("weightDecay", "--weight-decay", EditorKind.Float, "0"),
            Opt("clip", "--clip", EditorKind.Float, "1.0"),
            Opt("rowStride", "--row-stride", EditorKind.Int, "2"),
            Opt("log", "--log file.jsonl", EditorKind.File, "", "Write the training log here."),
            TokenizerParam,
            ComponentParam,
        }),
        new("moe-merge", CatMoe, "Merge LoRA experts and their router into one MoE container.", new[]
        {
            ContainerParam,
            Opt("experts", "--experts name=dir,…", EditorKind.Text, "",
                "LoRA experts, comma-separated name=<lora-dir> pairs, in router order.", required: true),
            Opt("router", "--router file", EditorKind.File, "", "Router from moe-router.", required: true),
            OutDirParam with { Label = "--out moe dir", Help = "Container directory to write." },
            Opt("topK", "--top-k", EditorKind.Int, "0", "Experts used per token (0 = all)."),
            ComponentParam,
        }),

        // ── 9. Entail ───────────────────────────────────────────────────────
        new("entail check", CatEntail, "Compile an Entail program and report its size, without writing anything.", new[]
        {
            EntailProgramParam,
            EntailContextParam,
            NeuronsPerLayerParam,
        }),
        new("entail run", CatEntail, "Run text through an Entail program's logic directly, no model weights.", new[]
        {
            EntailProgramParam,
            Opt("expr", "--expr text", EditorKind.Text, "", "Text to run, e.g. 12+34= (or use --file / --cases)."),
            Opt("file", "--file text", EditorKind.File, "", "Run the program over a text file."),
            Opt("cases", "--cases file.jsonl", EditorKind.File, "", "Test cases, one JSON object per line."),
            TokenizerParam with { Help = "Split text as this model's tokenizer would. Empty = one character per token." },
            Opt("maxTokens", "--max-tokens", EditorKind.Int, "32"),
            Opt("report", "--report file.json", EditorKind.File, "", "Write the case results as JSON."),
            NeuronsPerLayerParam,
        }),
        new("entail compile", CatEntail, "Compile an Entail program into a standalone checkpoint.", new[]
        {
            EntailProgramParam,
            OutDirParam with { Label = "--out checkpoint dir" },
            EntailContextParam,
            NeuronsPerLayerParam,
        }),
        new("entail graft", CatEntail, "Graft an Entail program into a host model's checkpoint.", new[]
        {
            EntailProgramParam,
            Opt("host", "--host checkpoint dir", EditorKind.Directory, "", "HF checkpoint to graft into.", required: true),
            OutDirParam with { Label = "--out checkpoint dir" },
            EntailContextParam,
            Opt("outputNorm", "--output-norm", EditorKind.Float, "115", "Scale of the program's output written into the host."),
            NeuronsPerLayerParam,
        }),
        new("entail adapter", CatEntail, "Build an Entail program as an adapter that runs beside a partner model.", new[]
        {
            EntailProgramParam,
            TokenizerParam with
            {
                Label = "--tokenizer partner dir", Required = true,
                Help = "The partner model whose tokens the adapter reads.",
            },
            OutDirParam with { Label = "--out adapter dir" },
            Flag("dense", "--dense", "Build dense weights (a container) instead of the sparse adapter."),
            EntailContextParam,
            NeuronsPerLayerParam,
        }),
        new("entail side", CatEntail, "Generate from a model with Entail adapters running beside it.", new[]
        {
            ContainerParam,
            Opt("prompt", "--prompt text", EditorKind.MultiLineText, "", required: true),
            AdapterParam,
            Flag("chat", "--chat", "Wrap the prompt in the model's chat template."),
            Flag("think", "--think", "Leave thinking on in the chat template."),
            Opt("maxTokens", "--max-tokens", EditorKind.Int, "128"),
            EntailWeightsParam,
            TokenizerParam,
            ComponentParam,
        }),
        new("entail eval", CatEntail, "Score a model (with optional Entail adapters) on test cases or a text.", new[]
        {
            ContainerParam,
            Opt("cases", "--cases file.jsonl", EditorKind.File, "", "Test cases, one JSON object per line."),
            AdapterParam,
            Opt("limit", "--limit", EditorKind.Int, "", "Run only the first N cases."),
            Opt("batch", "--batch", EditorKind.Int, "1", "Cases run together (CPU only: add --cpu)."),
            Opt("watch", "--watch token", EditorKind.Text, "", "Report the probability of this one token."),
            Opt("prefix", "--prefix text", EditorKind.Text, "", "Text placed before every answer (\\n for a newline)."),
            Flag("logprobs", "--logprobs", "Report log-probabilities."),
            Opt("maxTokens", "--max-tokens", EditorKind.Int, "24"),
            Opt("text", "--text file", EditorKind.File, "", "Also measure perplexity on this text."),
            Opt("textTokens", "--text-tokens", EditorKind.Int, "1024"),
            Opt("report", "--report file.json", EditorKind.File, "", "Write the results as JSON."),
            EntailWeightsParam,
            TokenizerParam,
            ComponentParam,
        }),
        new("entail bench", CatEntail, "Run a reasoning benchmark on a model and score its answers.", new[]
        {
            ContainerParam,
            Opt("suite", "--suite", EditorKind.Choice, "reasoning", "Built-in generated problem suite.",
                choices: new[] { "reasoning", "hard" }),
            Opt("per", "--per", EditorKind.Int, "10", "Problems per family."),
            Opt("seed", "--seed", EditorKind.Int, "7", "Seed for generating the suite."),
            Opt("families", "--families a,b", EditorKind.Text, "", "Only these problem families."),
            Opt("cases", "--cases file.jsonl", EditorKind.File, "", "Use these cases instead of the generated suite."),
            Opt("limit", "--limit", EditorKind.Int, "", "Run only the first N cases."),
            Flag("noThink", "--no-think", "Turn thinking off."),
            Opt("maxTokens", "--max-tokens", EditorKind.Int, "", "Default 2048 with thinking, 512 without."),
            Opt("temperature", "--temperature", EditorKind.Float, "0"),
            Opt("topP", "--top-p", EditorKind.Float, "0"),
            Opt("topK", "--top-k", EditorKind.Int, "0"),
            Opt("sampleSeed", "--sample-seed", EditorKind.Int, "1"),
            Opt("watch", "--watch token", EditorKind.Text, "", "Report the probability of this one token."),
            Opt("score", "--score", EditorKind.Choice, "answer",
                "answer: check the final number. tools: check what the reply did with the tools in its prompt.",
                choices: new[] { "answer", "tools" }),
            Opt("simulate", "--simulate N", EditorKind.Int, "0", "Answer up to N tool calls with simulated results."),
            AdapterParam,
            Opt("report", "--report file.json", EditorKind.File, "", "Write the results as JSON."),
            Opt("compare", "--compare file.json", EditorKind.File, "", "Compare against an earlier report."),
            Opt("writeCases", "--write-cases file.jsonl", EditorKind.File, "", "Save the generated cases."),
            EntailWeightsParam,
            TokenizerParam,
            ComponentParam,
        }),
        new("entail find", CatEntail, "Find the units in a model that respond to a subject.", new[]
        {
            ContainerParam,
            Opt("subject", "--subject file", EditorKind.File, "", "Subject description file.", required: true),
            OutDirParam,
            Opt("top", "--top", EditorKind.Int, "64", "How many units to keep."),
            Flag("verify", "--verify", "Check the found units against random ones."),
            Opt("seed", "--seed", EditorKind.Int, "1"),
            TokenizerParam,
            ComponentParam,
        }),
        new("entail extract", CatEntail, "Extract what a model knows about a subject as Entail rules.", new[]
        {
            ContainerParam,
            Opt("subject", "--subject file", EditorKind.File, "", "Subject description file.", required: true),
            OutDirParam,
            Opt("seed", "--seed", EditorKind.Int, "1"),
            TokenizerParam,
            ComponentParam,
        }),
        new("entail background", CatEntail, "Collect background lines for a subject from a model.", new[]
        {
            ContainerParam,
            Opt("subject", "--subject file", EditorKind.File, "", "Subject description file.", required: true),
            OutDirParam,
            TokenizerParam,
            ComponentParam,
        }),
        new("entail from-lora", CatEntail, "Turn a trained LoRA adapter into a readable Entail program.", new[]
        {
            ContainerParam,
            Opt("lora", "--lora dir", EditorKind.Directory, "", "The LoRA adapter.", required: true),
            Opt("cases", "--cases file.jsonl", EditorKind.File, "", "Cases the adapter was trained for.", required: true),
            Opt("out", "--out file.entail.txt", EditorKind.File, "", "Program to write.", required: true),
            Opt("occlude", "--occlude", EditorKind.Int, "3"),
            Opt("votes", "--votes", EditorKind.Int, "6"),
            Opt("fire", "--fire", EditorKind.Float, "0.25"),
            Opt("name", "--name", EditorKind.Text, "", "Program name. Empty = lora_<adapter folder>."),
            TokenizerParam,
            ComponentParam,
        }),
        new("entail decompile", CatEntail, "Recover Entail rules from what a model learned on a corpus.", new[]
        {
            ContainerParam,
            Opt("corpus", "--corpus dir", EditorKind.Directory, "", "Text the model is studied on.", required: true),
            OutDirParam,
            Opt("name", "--name", EditorKind.Text, "subject_raw"),
            Opt("contrast", "--contrast file", EditorKind.File, "", "Unrelated text to contrast against."),
            Opt("extra", "--extra file", EditorKind.File, ""),
            Opt("corpusTokens", "--corpus-tokens", EditorKind.Int, "60000"),
            Opt("contrastTokens", "--contrast-tokens", EditorKind.Int, "8000"),
            Opt("perLayer", "--per-layer", EditorKind.Int, "24"),
            Opt("units", "--units", EditorKind.Int, "4000", "How many units to keep."),
            Opt("vote", "--vote", EditorKind.Choice, "lift", "How units vote for a rule.",
                choices: new[] { "lift", "share", "bayes" }),
            Opt("from", "--from study.bin", EditorKind.File, "", "Reuse a saved study instead of re-reading the corpus."),
            Opt("seed", "--seed", EditorKind.Int, "1"),
            Flag("cover", "--cover", "Cover the text with rules, keeping rules seen only once."),
            Flag("onlyHoldout", "--only-holdout", "Use the whole corpus as held-out text."),
            Flag("noColumns", "--no-columns", "Skip the output-column effects."),
            TokenizerParam,
            ComponentParam,
        }),
        new("entail fidelity", CatEntail, "Check how well an Entail program predicts held-out text.", new[]
        {
            EntailProgramParam,
            Opt("holdout", "--holdout file", EditorKind.File, "", "Held-out lines to check against.", required: true),
            Opt("lines", "--lines", EditorKind.Int, "40"),
        }),
        new("entail optimise", CatEntail, "Simplify the rules in an Entail work directory.", new[]
        {
            Pos("dir", "work dir", EditorKind.Directory),
            Opt("confidence", "--confidence", EditorKind.Float, "0.5"),
        }),
        new("entail review", CatEntail, "Have a model review candidate rules in an Entail work directory.", new[]
        {
            Pos("dir", "work dir", EditorKind.Directory),
            Opt("endpoint", "--endpoint url", EditorKind.Text, "", "OpenAI-compatible server that does the review."),
            Opt("model", "--model", EditorKind.Text, "default"),
            Opt("rounds", "--rounds", EditorKind.Int, "3"),
            Opt("docs", "--docs dir", EditorKind.Directory, "", "Reference documents. Empty = the current folder."),
            Opt("confidence", "--confidence", EditorKind.Float, "0.5"),
        }),
        new("entail accept", CatEntail, "Accept a reviewed candidate rule into an Entail work directory.", new[]
        {
            Pos("dir", "work dir", EditorKind.Directory),
            Opt("candidate", "--candidate file", EditorKind.File, "", required: true),
            Opt("rule", "--rule", EditorKind.Text, "", required: true),
            Opt("why", "--why reason", EditorKind.Text, "", "Why the rule is accepted."),
            Opt("author", "--author", EditorKind.Text, "reviewer"),
            Opt("confidence", "--confidence", EditorKind.Float, "0.5"),
        }),

        // ── 9a. N-gram Packs ────────────────────────────────────────────────
        new("ngram build", CatNGram, "Build an n-gram pack from text files to steer a model's predictions.", new[]
        {
            Pos("inputs", "input file or folder", EditorKind.Directory,
                "Text file or folder (searched for --ext files) to learn from."),
            TokenizerParam with
            {
                Label = "--tokenizer model or container dir", Required = true,
                Help = "The model the pack is built for.",
            },
            Opt("out", "--out pack file", EditorKind.File, "", "Pack file to write.", required: true),
            Opt("name", "--name", EditorKind.Text, "", "Pack name. Empty = the output file name."),
            Opt("order", "--order", EditorKind.Int, "4", "Longest context, in tokens."),
            Opt("keywords", "--keywords a,b", EditorKind.Text, "", "Words that switch the pack on."),
            Opt("minCount", "--min-count", EditorKind.Int, "2"),
            Opt("topK", "--top-k", EditorKind.Int, "8", "Next tokens kept per context."),
            Opt("ext", "--ext", EditorKind.Text, ".cs,.txt,.md", "File extensions read from folders."),
            Opt("vocab", "--vocab", EditorKind.Int, "", "Vocabulary size. Empty = the tokenizer's."),
        }),
        new("ngram info", CatNGram, "Show what is in an n-gram pack.", new[]
        {
            Pos("pack", "pack file", EditorKind.File),
        }),
        new("ngram eval", CatNGram, "Measure how much n-gram packs improve a model's predictions on text.", new[]
        {
            ContainerParam,
            Pos("inputs", "input file or folder", EditorKind.Directory, "Text file or folder to score on."),
            Opt("pack", "--pack file[,file]", EditorKind.File, "", "Pack files, comma-separated.", required: true),
            Opt("lambda", "--lambda", EditorKind.Text, "0.1,0.2,0.3,0.5", "Pack weights to try, comma-separated."),
            Opt("maxTokens", "--max-tokens", EditorKind.Int, "20000"),
            Opt("minContext", "--min-context", EditorKind.Int, "2"),
            Opt("halfCount", "--half-count", EditorKind.Float, "4"),
            Opt("modelConfident", "--model-confident", EditorKind.Float, "1"),
            Opt("packConfident", "--pack-confident", EditorKind.Float, "0"),
            Opt("ext", "--ext", EditorKind.Text, ".cs,.txt,.md", "File extensions read from folders."),
            TokenizerParam,
        }),
    };

    public static CommandDef? Find(string name) =>
        All.FirstOrDefault(c => string.Equals(c.Name, name, StringComparison.OrdinalIgnoreCase));

    public static IEnumerable<IGrouping<string, CommandDef>> ByCategory() =>
        All.GroupBy(c => c.Category).OrderBy(g => g.Key, StringComparer.Ordinal);
}
