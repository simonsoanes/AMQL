using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Amql.Cli;
using Amql.Entail;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Server;

/// <summary>What a loaded container can answer. Decided once, from the
/// container's recorded facts — an endpoint is only offered for a container
/// that can actually serve it.</summary>
[Flags]
public enum Capability
{
    None = 0,
    Decisions = 1,
    Chat = 2,
    Embeddings = 4,
    Classification = 8,
}

/// <summary>
/// One container held open by the server, with the engines its capabilities
/// need. The runtimes keep per-sequence state (KV cache, recurrent state), so
/// every request takes <see cref="Gate"/> — one request per model at a time.
/// </summary>
public sealed class ModelHost : IDisposable
{
    private readonly Vindex3Container _container;
    private readonly OperandStore? _store;

    public string Id { get; }
    public string Path { get; }
    public Capability Capabilities { get; }
    public long Created { get; }
    public SemaphoreSlim Gate { get; }

    /// <summary>Requests a batching model accepts at once: a batch running and
    /// the rest waiting their turn inside the scheduler.</summary>
    public const int BatchQueue = 64;

    /// <summary>One request at a time across every model while CUDA is on: the
    /// native kernels stage through process-wide device buffers, so two
    /// sessions on the GPU at once would overwrite each other's activations.
    /// On the CPU each model has its own gate and different models run in parallel.</summary>
    public static SemaphoreSlim DeviceGate { get; } = new(1, 1);

    /// <summary>Whether this model waits behind the others (see <see cref="DeviceGate"/>).</summary>
    public bool SharesDevice => ReferenceEquals(Gate, DeviceGate);

    /// <summary>This model's own defaults for requests that leave a setting
    /// out; what it leaves unset falls back to the server's.</summary>
    public ModelDefaults Defaults { get; init; } = new();

    /// <summary>The names of the Entail adapters running beside the model.</summary>
    public IReadOnlyList<string> AdapterNames { get; private init; } = Array.Empty<string>();

    /// <summary>The n-gram packs in the container's <c>ngram/</c> folder, kept current while the
    /// server runs (null for a model that does not generate).</summary>
    public NGramLibrary? NGram { get; private init; }

    /// <summary>A tokenizer's fingerprint as n-gram packs record it (as <c>amql-cli ngram build</c> computes it).</summary>
    public static string TokenizerFingerprint(HfTokenizer tokenizer) =>
        Amql.Inference.NGram.NGramPack.Fingerprint(tokenizer.VocabSize, id =>
        {
            try
            {
                return tokenizer.DecodeText(new[] { id }, skipSpecialTokens: false);
            }
            catch (TokenizerException)
            {
                return "";
            }
        });

    public VonDecisionEngine? Decisions { get; }
    public TextGenerator? Generator { get; }
    public Embedder? Embedder { get; }
    public ClassificationService? Classifier { get; }

    /// <summary>Why a capability a container looked like it might have was
    /// not offered — reported at startup instead of failing a request later.</summary>
    public IReadOnlyList<string> Notes { get; }

    private ModelHost(string id, string path, Vindex3Container container, OperandStore? store, VonDecisionEngine? decisions,
        TextGenerator? generator, Embedder? embedder, ClassificationService? classifier, List<string> notes)
    {
        Id = id;
        Path = path;
        _container = container;
        _store = store;
        Decisions = decisions;
        Generator = generator;
        Embedder = embedder;
        Classifier = classifier;
        Notes = notes;
        Capabilities = (decisions is null ? Capability.None : Capability.Decisions) |
                       (generator is null ? Capability.None : Capability.Chat) |
                       (embedder is null ? Capability.None : Capability.Embeddings) |
                       (classifier is null ? Capability.None : Capability.Classification);
        Created = new DateTimeOffset(Directory.GetCreationTimeUtc(path)).ToUnixTimeSeconds();
        // A batching model takes many requests and schedules the device itself;
        // anything else on the GPU waits its turn behind every other model.
        Gate = generator?.Concurrency > 1 ? new SemaphoreSlim(BatchQueue, BatchQueue) : CudaShim.Enabled ? DeviceGate : new SemaphoreSlim(1, 1);
    }

    /// <param name="adapters">Entail adapter directories to run beside a
    /// generative model; each must have been built for this model's tokenizer.</param>
    /// <param name="kvDir">This model's own folder for paged-out prompt state
    /// (its <c>kv_dir</c> in <c>--config</c>); see <see cref="KvPaging.For"/>.</param>
    /// <param name="context">A context length other than the model's own; null is
    /// <see cref="ContextExtension.Requested"/> (<c>--max-context</c>), or none.</param>
    public static ModelHost Load(string id, string path, IReadOnlyList<string>? adapters = null, ModelDefaults? defaults = null, string? kvDir = null,
        ContextExtension? context = null)
    {
        var host = LoadEngines(id, path, adapters, context);
        void Log(string line) => Console.Error.WriteLine($"{DateTime.Now:HH:mm:ss.fff} {id}: {line}");
        var library = host.Generator is { } g
            ? new NGramLibrary(path, TokenizerFingerprint(g.Tokenizer), Log)
            : null;
        if (host.Generator?.Batch is { } batch)
        {
            var paging = KvPaging.For(id, path, kvDir);
            var store = paging.DiskDir is { } dir
                ? new KvPageStore(dir, KvPageStore.Fingerprint(path, batch.MarkGeometry), paging.DiskBudgetBytes, paging.ExpireSeconds, Log)
                : null;
            batch.Configure(paging, store, Log);
            if (host.Notes is List<string> kvNotes && (batch.KvFormat.Format != "f32" || batch.KvFormat.Refusal is not null))
            {
                kvNotes.Add(batch.KvFormat.Refusal is { } why ? $"KV cache stays f32: {why}"
                    : $"KV cache stored as {batch.KvFormat.Format}" + (batch.KvFormat.Format != "nvfp4" ? "" : paging.ExactPrefix switch
                    {
                        < 0 => $", each conversation's system prompt and first user message at {(paging.ExactF16 ? "FP16" : "f32")}",
                        0 => "",
                        var n => $", each conversation's first {n} tokens at {(paging.ExactF16 ? "FP16" : "f32")}",
                    }));
            }
            if (store is not null && host.Notes is List<string> notes)
            {
                notes.Add($"prompt state pages out to {store.Folder} after {paging.IdleSeconds:F0} s idle " +
                               $"(up to {paging.DiskBudgetBytes / (1024.0 * 1024 * 1024):F1} GiB, deleted after {paging.ExpireSeconds:F0} s)");
            }
        }
        return new ModelHost(host, defaults ?? new ModelDefaults(),
            (adapters ?? Array.Empty<string>()).Select(d => System.IO.Path.GetFileName(System.IO.Path.GetFullPath(d).TrimEnd('/', '\\'))).ToList())
        {
            NGram = library,
        };
    }

    private ModelHost(ModelHost loaded, ModelDefaults defaults, IReadOnlyList<string> adapterNames)
        : this(loaded.Id, loaded.Path, loaded._container, loaded._store, loaded.Decisions, loaded.Generator, loaded.Embedder,
            loaded.Classifier, loaded.Notes.ToList())
    {
        Defaults = defaults;
        AdapterNames = loaded.Generator is null ? Array.Empty<string>() : adapterNames;
    }

    private static ModelHost LoadEngines(string id, string path, IReadOnlyList<string>? adapters, ContextExtension? context)
    {
        var container = Vindex3Container.Open(path, context);
        var notes = new List<string>(container.ContextNotes);
        try
        {
            var component = container.Graph?.Components.FirstOrDefault(c => c.Role == ComponentRole.PrimaryText)
                ?? throw new ContainerException("container records no primary text component");
            var surface = component.Execution;

            // Von: a ModernBERT encoder with an option-marker head.
            if (ModernBertEncoder.Serves(container))
            {
                if (surface?.OptionMarker is null)
                {
                    notes.Add("ModernBERT encoder without an option-marker head: no endpoint serves a bare encoder");
                    return new ModelHost(id, path, container, null, null, null, null, null, notes);
                }
                return new ModelHost(id, path, container, null, VonDecisionEngine.Load(container), null, null, null, notes);
            }

            if (surface?.Classifier is not null)
            {
                string tokenizerPathCls = System.IO.Path.Combine(path, "tokenizer.json");
                if (!File.Exists(tokenizerPathCls))
                {
                    notes.Add("no tokenizer.json: classification endpoint needs the container's own tokenizer");
                    return new ModelHost(id, path, container, null, null, null, null, null, notes);
                }
                var storeCls = container.CreateOperandStore();
                try
                {
                    var classifier = ClassificationService.Load(container, storeCls);
                    return new ModelHost(id, path, container, storeCls, null, null, null, classifier, notes);
                }
                catch (Exception e)
                {
                    storeCls.Dispose();
                    notes.Add($"classifier could not be loaded: {e.Message}");
                    return new ModelHost(id, path, container, null, null, null, null, null, notes);
                }
            }

            string tokenizerPath = System.IO.Path.Combine(path, "tokenizer.json");
            if (!File.Exists(tokenizerPath))
            {
                notes.Add("no tokenizer.json: text endpoints need the container's own tokenizer");
                return new ModelHost(id, path, container, null, null, null, null, null, notes);
            }
            var tokenizer = HfTokenizer.FromTokenizerFile(tokenizerPath);
            var store = container.CreateOperandStore();
            ComponentOpPlan plan;
            try
            {
                plan = Planner.Plan(container, component.Id, store);
            }
            catch (UnsupportedOperatorException e)
            {
                store.Dispose();
                notes.Add($"the runtime does not serve this stack: {e.Message}");
                return new ModelHost(id, path, container, null, null, null, null, null, notes);
            }

            if (surface?.Embedding is { } embedding)
            {
                var embedder = new Embedder(new DecodeSession(plan, store), tokenizer, EmbeddingPooling.From(embedding),
                    (int)(surface.ContextLength ?? int.MaxValue));
                return new ModelHost(id, path, container, store, null, null, embedder, null, notes);
            }

            if (plan.Output is null)
            {
                notes.Add("the plan has no output head, so it cannot generate");
                return new ModelHost(id, path, container, store, null, null, null, null, notes);
            }
            ChatTemplate? template;
            try
            {
                template = ChatTemplate.Load(path);
            }
            catch (ChatTemplateException e)
            {
                notes.Add(e.Message);
                return new ModelHost(id, path, container, store, null, null, null, null, notes);
            }
            if (template is null)
            {
                notes.Add("no chat template (chat_template.jinja / tokenizer_config.json): chat endpoints need one");
                return new ModelHost(id, path, container, store, null, null, null, null, notes);
            }
            try
            {
                template.Apply("probe");
            }
            catch (ChatTemplateException e)
            {
                notes.Add($"chat template not rendered by this build: {e.Message}");
                return new ModelHost(id, path, container, store, null, null, null, null, notes);
            }
            var loaded = new List<IEntailAdapter>();
            foreach (string dir in adapters ?? Array.Empty<string>())
            {
                loaded.Add(EntailAdapterRunner.Load(dir, path));
                notes.Add($"entail adapter {System.IO.Path.GetFileName(System.IO.Path.GetFullPath(dir).TrimEnd('/', '\\'))} runs beside the model");
            }
            var prior = tokenizer.AddedTokenId("<|im_start|>") is { } start && tokenizer.AddedTokenId("<|im_end|>") is { } end
                ? ChatPrior.ForChatMl(t => tokenizer.EncodeToIds(t), ids => tokenizer.DecodeText(ids, skipSpecialTokens: false), start, end)
                : null;
            var generator = new TextGenerator(new DecodeSession(plan, store), loaded, prior, tokenizer, template,
                StopTokens.Resolve(path, tokenizer, template), (int)(surface?.ContextLength ?? int.MaxValue));
            if (VisionPlan.Build(container, store, out string? noVision) is { } vision)
            {
                notes.Add(generator.UseVision(vision, store)
                    ? $"images: up to {vision.SoftTokens} tokens each"
                    : "images are not read: the vision tower runs on the GPU engine, without adapters");
            }
            else if (container.Graph?.Components.Any(c => c.Role == ComponentRole.Perception) == true)
            {
                notes.Add($"images are not read: {noVision}");
            }
            notes.Add(generator.RendersTools ? "tools: calls are read as the chat template writes them"
                : $"no tools: {template.Dialect.CallsRefusal ?? "the chat template does not render tools or their results"}");
            notes.Add(generator.BatchRefusal is null
                ? $"continuous batching: up to {generator.Concurrency} requests generate together on the GPU"
                : $"one request at a time: {generator.BatchRefusal}");
            return new ModelHost(id, path, container, store, null, generator, null, null, notes);
        }
        catch
        {
            container.Dispose();
            throw;
        }
    }

    /// <summary>Runs one short forward pass so weights are resident before
    /// the first request, instead of widening a gigabyte inside it.</summary>
    public void Warm()
    {
        Generator?.Warm();
        Embedder?.Warm();
    }

    public void Dispose()
    {
        NGram?.Dispose();
        Generator?.Dispose();
        foreach (var adapter in Generator?.Adapters ?? Array.Empty<IEntailAdapter>())
        {
            adapter.Dispose();
        }
        _store?.Dispose();
        _container.Dispose();
        if (!SharesDevice)
        {
            Gate.Dispose();
        }
    }
}

/// <summary>The stop tokens a generation ends on: <c>generation_config.json</c>'s
/// <c>eos_token_id</c>, the tokenizer config's <c>eos_token</c>, and the
/// template's end-of-turn marker.</summary>
public static class StopTokens
{
    public static IReadOnlySet<int> Resolve(string dir, HfTokenizer tokenizer, ChatTemplate template)
    {
        var ids = new HashSet<int>();
        void AddContent(string? content)
        {
            if (content is not null && tokenizer.AddedTokenId(content) is { } id)
            {
                ids.Add(id);
            }
        }

        string generation = System.IO.Path.Combine(dir, "generation_config.json");
        if (File.Exists(generation))
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(generation));
            if (doc.RootElement.TryGetProperty("eos_token_id", out var eos))
            {
                if (eos.ValueKind == JsonValueKind.Number)
                {
                    ids.Add(eos.GetInt32());
                }
                else if (eos.ValueKind == JsonValueKind.Array)
                {
                    foreach (var e in eos.EnumerateArray())
                    {
                        ids.Add(e.GetInt32());
                    }
                }
            }
        }
        string tokenizerConfig = System.IO.Path.Combine(dir, "tokenizer_config.json");
        if (File.Exists(tokenizerConfig))
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(tokenizerConfig));
            if (doc.RootElement.TryGetProperty("eos_token", out var eos))
            {
                AddContent(eos.ValueKind == JsonValueKind.String ? eos.GetString()
                    : eos.ValueKind == JsonValueKind.Object && eos.TryGetProperty("content", out var c) ? c.GetString() : null);
            }
        }
        // What the template ends a turn with, after words and after calls.
        var dialect = template.Dialect;
        foreach (string end in dialect.Calls is null ? new[] { dialect.AfterReply } : [dialect.AfterReply, dialect.BeforeResult])
        {
            if (TurnMarks.First(tokenizer, end) is { } id)
            {
                ids.Add(id);
            }
        }
        return ids;
    }
}

/// <summary>Defaults one model applies to requests that leave them out
/// (null: the server's default).</summary>
public sealed record ModelDefaults(bool? Think = null, int? MaxTokens = null, float? Temperature = null, float? TopP = null, int? TopK = null)
{
    /// <summary>The n-gram packs this model blends by default (names of loaded packs).</summary>
    public IReadOnlyList<string>? NGramPacks { get; init; }

    /// <summary>Thinking tokens after which the block is closed (null: the server's, 0: never).</summary>
    public int? ThinkingBudget { get; init; }
}

/// <summary>Where a reply with a thinking budget closes its block.</summary>
public static class ThinkingLimit
{
    /// <summary>How long past its budget thinking may run to reach the end of a line.</summary>
    public const int LineWait = 64;

    /// <summary>The thinking token at which a reply of at most
    /// <paramref name="maxTokens"/> closes its block: the budget, or sooner when
    /// the reply's own limit would leave no room for an answer (a quarter of it,
    /// at most 512 tokens, is kept).</summary>
    public static int For(int budget, int maxTokens) =>
        Math.Max(1, Math.Min(budget, maxTokens - Math.Min(maxTokens / 4, 512) - LineWait));
}

public sealed record GenerationParams(
    int MaxTokens,
    float Temperature,
    float TopP,
    int Seed,
    IReadOnlyList<string> Stop)
{
    /// <summary>Keep only the k most probable tokens before sampling (0: all).</summary>
    public int TopK { get; init; }

    /// <summary>Text written for the model at the start of its answer — after
    /// its thinking block when it thinks — and continued from there: how
    /// <c>tool_choice</c> required or a named function is enforced.</summary>
    public string? Forced { get; init; }

    /// <summary>The reply thinks first: the prompt ended inside an open think block, or the
    /// model opens one itself.</summary>
    public bool Thinking { get; init; }

    /// <summary>The prompt ended inside an open think block.</summary>
    public bool ThinkOpened { get; init; }

    /// <summary>Thinking tokens after which the block is closed and the reply
    /// goes on to its answer (0: never). See <see cref="ThinkingLimit"/>.</summary>
    public int ThinkingBudget { get; init; }

    /// <summary>Structured output: the reply must be a JSON value this schema allows.</summary>
    public JsonSchemaNode? Json { get; init; }

    /// <summary>The request's tools (chat form): a call the reply opens is held to them.</summary>
    public IReadOnlyList<System.Text.Json.JsonElement>? Tools { get; init; }

    /// <summary>The reply may not open a call (it must answer in words from what it has).</summary>
    public bool NoCalls { get; init; }

    /// <summary>The user's latest message, for the mentioned-parameter rule.</summary>
    public string? Recent { get; init; }

    /// <summary>The tool results since the user's latest message, for the mentioned-parameter rule,
    /// where only a suggestion counts ("Pass overwrite: true"), not a field or a warning.</summary>
    public string? RecentResults { get; init; }

    /// <summary>The user's own message that began the turn (not the notes after it), for the mentioned-tool rule.</summary>
    public string? Task { get; init; }

    /// <summary>The tools called since the user's latest message, for the mentioned-tool rule.</summary>
    public IReadOnlySet<string>? CalledSince { get; init; }

    /// <summary>The reply's start the client wrote (<c>continue_final_message</c>): part of the
    /// reply for the rules that read it, though not repeated in the output.</summary>
    public string? Prefill { get; init; }

    /// <summary>The prompt's images, where they stand in it.</summary>
    public IReadOnlyList<PromptImage>? Images { get; init; }

    /// <summary>N-gram packs blended into the logits outside thinking (null: none).</summary>
    public Amql.Inference.NGram.NGramPrior? NGram { get; init; }
}

/// <summary>An image made ready, and what the tower made of it once it has been read.</summary>
public sealed class ServedImage(PreparedImage image)
{
    public PreparedImage Image { get; } = image;

    /// <summary>Its tokens, once a request has needed them (written by the batch's thread alone).</summary>
    public float[]? Features { get; set; }
}

/// <summary>An image's place in a prompt.</summary>
public sealed record PromptImage(int Start, ServedImage Image);

/// <summary>The images lately sent, by file and token limit, so that a conversation about a picture reads it once.</summary>
public sealed class ImageCache
{
    private const int Most = 32;

    private readonly Dictionary<string, (ServedImage Image, long Used)> _images = new(StringComparer.Ordinal);
    private long _clock;

    public ServedImage Get(VisionPlan vision, byte[] file, int? tokens)
    {
        string key = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(file)) + ":" + tokens;
        lock (_images)
        {
            if (_images.TryGetValue(key, out var known))
            {
                _images[key] = (known.Image, ++_clock);
                return known.Image;
            }
        }
        var image = new ServedImage(vision.Prepare(file, tokens));
        lock (_images)
        {
            if (_images.Count >= Most)
            {
                _images.Remove(_images.MinBy(p => p.Value.Used).Key);
            }
            _images[key] = (image, ++_clock);
        }
        return image;
    }
}

public sealed record GenerationResult(string Text, string FinishReason, int PromptTokens, int CompletionTokens)
{
    /// <summary>When the loop guard stopped the reply: its text with the repeats after the first
    /// round removed (what the reply had to say, once).</summary>
    public string? Looped { get; init; }

    /// <summary>The generated token ids (stop token excluded).</summary>
    public IReadOnlyList<int> Tokens { get; init; } = Array.Empty<int>();

    /// <summary>Per-token logprobs for the completion, when requested.</summary>
    public IReadOnlyList<LogprobsResult>? Logprobs { get; init; }

    /// <summary>Perplexity over the completion tokens (exp of mean NLL).</summary>
    public float? Perplexity { get; init; }

    /// <summary>Mean Shannon entropy (nats) over the completion tokens.</summary>
    public float? MeanEntropy { get; init; }
}

/// <summary>
/// Autoregressive generation over a decoder container: prefill the rendered
/// prompt, then sample until a stop token, a stop string, the token budget or
/// the context runs out. Text is decoded from the accumulated bytes each step,
/// so a character split across tokens is emitted once, whole. Entail adapters,
/// when the server was given any, run beside the model and rule on each next
/// token; with none, every token is the model's own draw.
/// <para>
/// On the GPU engine, requests are batched continuously (<see cref="BatchScheduler"/>):
/// each joins the running batch when it arrives, leaves it when it finishes, and
/// every step advances up to <see cref="GpuEngine.MaxBatch"/> of them in one
/// pass over the weights. Elsewhere one request runs at a time.
/// </para>
/// </summary>
public sealed class TextGenerator : IDisposable
{
    private readonly SideBySideSession _session;
    private readonly IReadOnlySet<int> _stopTokens;
    private readonly BatchScheduler? _batch;

    /// <summary>The share of the model's probability the way to a call must have for the
    /// announced-call rule to open it (default 1e-4, as for mentioned parameters).</summary>
    public static double AnnounceFloor { get; set; } = 1e-4;

    private static readonly Regex Intent = new(
        @"\b(let me|let's|i'll|i will|i'm going to|i am going to|i need to|now i|next i|i'm now|i shall)\b",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled);

    /// <summary>Whether a tool result tells the next call to pass a parameter: "Pass overwrite:
    /// true", "set 'force' to true", "use background=true"; not a field of the result
    /// (<c>"handle": "bg-95c9"</c>) nor a warning ("'handle' means poll or stop").</summary>
    public static bool SuggestsParameter(string results, string name) => Regex.IsMatch(results,
        $@"\b(?:pass|set|use|add|include|specify|with)\s+(?:the\s+)?[`'""]?{Regex.Escape(name)}[`'""]?\s*(?::|=|to\b)",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

    private static readonly Regex Negation = new(
        @"\b(?:not|don't|do not|never|no need|no|without|avoid|instead of|rather than|stop)\b",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled);

    /// <summary>Whether a message names a tool to use: the name as a whole word ("write it with
    /// create_file"), outside a clause that says not to ("no need to read_file it", "don't use
    /// edit_file").</summary>
    public static bool NamesTool(string text, string name) => ToolMention(text, name) >= 0;

    /// <summary>Where a message first names a tool to use (see <see cref="NamesTool"/>), or -1.</summary>
    public static int ToolMention(string text, string name)
    {
        foreach (Match m in Regex.Matches(text, $@"(?<![\w-]){Regex.Escape(name)}(?![\w-])"))
        {
            int start = text.LastIndexOfAny(['.', '!', '?', '\n', ';', ',', '('], Math.Max(m.Index - 1, 0)) + 1;
            if (!Negation.IsMatch(text[start..m.Index].Replace('’', '\'')))
            {
                return m.Index;
            }
        }
        return -1;
    }

    /// <summary>
    /// The tools a message names that a call may name now: those called already since it (a step
    /// may be done again, a file written again after a failed build), and the first one not called
    /// yet, since instructions name their tools in the order they are to be used.
    /// </summary>
    public static IReadOnlyList<string> ToolsDue(string text, IEnumerable<string> tools, IReadOnlySet<string>? called)
    {
        var named = tools.Select(t => (Tool: t, At: ToolMention(text, t))).Where(x => x.At >= 0).OrderBy(x => x.At).ToList();
        var due = named.Where(x => called?.Contains(x.Tool) == true).Select(x => x.Tool).ToList();
        if (named.FirstOrDefault(x => called?.Contains(x.Tool) != true) is { Tool: { } next })
        {
            due.Add(next);
        }
        return due;
    }

    /// <summary>Whether the sentence that ends at a colon says the reply is about to act
    /// ("Now let me build it", not "Here's what was done").</summary>
    public static bool AnnouncesAction(string beforeColon)
    {
        int start = beforeColon.LastIndexOfAny(['.', '!', '?', '\n']) + 1;
        return Intent.IsMatch(beforeColon[start..].Replace('’', '\''));
    }

    /// <summary>Set false (<c>--no-mention-rule</c>) to let calls leave out optional parameters the
    /// reply or the newest messages name.</summary>
    public static bool MentionRuleEnabled { get; set; } = true;

    /// <summary>Set false (<c>--free-tool-calls</c>) to let a reply write calls the tools do not declare.</summary>
    public static bool ToolGrammarEnabled { get; set; } = true;

    /// <summary>Set false (<c>--no-loop-guard</c>) to let a reply repeat itself until its token limit.</summary>
    public static bool LoopGuardEnabled { get; set; } = true;

    /// <summary>Set false (<c>--no-batch</c>) to run one request at a time even on the GPU engine.</summary>
    public static bool BatchingEnabled { get; set; } =
        Environment.GetEnvironmentVariable("AMQL_SERVER_BATCH")?.Trim() is not ("0" or "off" or "false");

    public HfTokenizer Tokenizer { get; }
    public ChatTemplate Template { get; }

    /// <summary>What the template does.</summary>
    internal ChatDialect Dialect => Template.Dialect;

    public int ContextLength { get; }

    /// <summary>The tokens that close and open a thinking block (the first of each mark), where they
    /// are tokens of their own.</summary>
    internal int? ThinkClose { get; }

    internal int? ThinkOpen { get; }

    /// <summary>What opens a thinking block that is written for the model.</summary>
    internal string OpenThinking => Dialect.ThinkOpen ?? "";

    /// <summary>What closes a thinking block from outside, after a line's end or within one.</summary>
    internal string CloseThinking(bool lineEnded) => Dialect.ThinkClose is not { } close ? ""
        : lineEnded ? close.TrimStart('\n') : close;

    /// <summary>Between a closed thinking block and a call that is written for the model.</summary>
    internal string AfterThinking => Dialect.ThinkClose is { } close ? close[close.TrimEnd().Length..] : "";

    /// <summary>A reply's text as a prompt would carry it: what <see cref="DecodeReply"/> undoes.</summary>
    internal string AsWritten(string reply) => _thinkMarks.Count == 0 ? reply
        : reply.Replace("<think>\n", OpenThinking).Replace("<think>", OpenThinking).Replace("</think>", Dialect.ThinkCloseMark);

    /// <summary>How the model writes its calls; null when the template's calls cannot be read.</summary>
    internal CallSyntax? Calls { get; }

    /// <summary>The model's marks around thinking, as tokens, and the marks every reply is read by;
    /// empty when the model writes those itself.</summary>
    private readonly List<(int[] Ids, string Text)> _thinkMarks = new();

    /// <summary>The special tokens that are part of a call's text.</summary>
    private readonly Dictionary<int, string> _callMarks = new();

    /// <summary>A token's text as it stands in a reply: <see cref="TokenText"/>, and a call's marks.</summary>
    internal string ReplyPiece(int id, int vocab) => _callMarks.TryGetValue(id, out var mark) ? mark : TokenText(id, vocab);

    /// <summary>A reply's tokens as text: special tokens dropped, but for a model that
    /// marks its thinking or its calls with special tokens, those as the marks every reply
    /// is read by.</summary>
    internal string DecodeReply(IReadOnlyList<int> ids)
    {
        if (_thinkMarks.Count == 0 && _callMarks.Count == 0)
        {
            return Tokenizer.DecodeText(ids);
        }
        var sb = new StringBuilder();
        int from = 0;
        void Plain(int to)
        {
            if (to > from)
            {
                sb.Append(Tokenizer.DecodeText(ids.Skip(from).Take(to - from).ToList()));
            }
        }
        for (int i = 0; i < ids.Count; i++)
        {
            var think = _thinkMarks.FirstOrDefault(m => Holds(ids, i, m.Ids));
            string? mark = think.Text ?? _callMarks.GetValueOrDefault(ids[i]);
            if (mark is not null)
            {
                Plain(i);
                sb.Append(mark);
                i += (think.Ids?.Length ?? 1) - 1;
                from = i + 1;
            }
        }
        Plain(ids.Count);
        return sb.ToString();

        static bool Holds(IReadOnlyList<int> ids, int at, int[] mark)
        {
            for (int k = 0; k < mark.Length; k++)
            {
                if (at + k >= ids.Count || ids[at + k] != mark[k])
                {
                    return false;
                }
            }
            return true;
        }
    }

    /// <summary>The tokens end inside a thinking mark of several, so what they say is not settled yet.</summary>
    internal bool EndsInsideMark(IReadOnlyList<int> ids)
    {
        foreach (var (mark, _) in _thinkMarks)
        {
            for (int n = Math.Min(mark.Length - 1, ids.Count); n > 0; n--)
            {
                bool same = true;
                for (int k = 0; k < n && same; k++)
                {
                    same = ids[ids.Count - n + k] == mark[k];
                }
                if (same)
                {
                    return true;
                }
            }
        }
        return false;
    }

    internal IReadOnlySet<int> StopTokenIds => _stopTokens;

    private string?[]? _tokenTexts;

    /// <summary>A token's own text, for structured output to check; empty for a special token
    /// (it would vanish from the decoded reply, so it is never allowed there) or a padding row.</summary>
    /// <summary>A token's text, from the cache <see cref="TokenText"/> fills when it is there.</summary>
    internal string TokenPiece(int id)
    {
        var texts = _tokenTexts;
        if (texts is not null && (uint)id < (uint)texts.Length)
        {
            return TokenText(id, texts.Length);
        }
        try
        {
            return Tokenizer.IsSpecial(id) ? "" : Tokenizer.DecodeText(new[] { id });
        }
        catch (TokenizerException)
        {
            return "";
        }
    }

    internal string TokenText(int id, int vocab)
    {
        var texts = _tokenTexts;
        if (texts is null || texts.Length < vocab)
        {
            texts = _tokenTexts = new string?[vocab];
        }
        if ((uint)id >= (uint)texts.Length)
        {
            return "";
        }
        if (texts[id] is { } known)
        {
            return known;
        }
        string text;
        try
        {
            text = Tokenizer.IsSpecial(id) ? "" : Tokenizer.DecodeText(new[] { id });
        }
        catch (TokenizerException)
        {
            text = "";   // a padding row of the head, no token at all
        }
        return texts[id] = text;
    }

    /// <summary>The adapters running beside the model.</summary>
    public IReadOnlyList<IEntailAdapter> Adapters { get; }

    /// <summary>Whether the template renders tools, calls and tool results.</summary>
    public bool RendersTools => Dialect is { WritesTools: true, WritesCalls: true, WritesResults: true } && Calls is not null;

    /// <summary>Whether the template has a thinking switch.</summary>
    public bool Thinks => Dialect.ThinkSwitch && Dialect.ThinkOpen is not null;

    /// <summary>Requests that can generate at once: the batch size on the GPU engine, else 1.</summary>
    public int Concurrency => _batch is null ? 1 : GpuEngine.MaxBatch;

    /// <summary>The batch scheduler, when requests are batched on the GPU engine.</summary>
    internal BatchScheduler? Batch => _batch;

    /// <summary>How the model reads images, when it does.</summary>
    public VisionPlan? Vision { get; private set; }

    internal ImageCache Images { get; } = new();

    /// <summary>Lets requests carry images: the container's tower runs on the GPU engine.</summary>
    internal bool UseVision(VisionPlan plan, OperandStore store)
    {
        if (_batch is null || Adapters.Count > 0)
        {
            return false;
        }
        _batch.UseVision(plan, store);
        Vision = plan;
        return true;
    }

    /// <summary>A rendered prompt as tokens, each image's mark replaced by the tokens it takes.</summary>
    internal (int[] Prompt, IReadOnlyList<PromptImage>? Images) EncodeWithImages(string rendered, IReadOnlyList<ChatPart> images)
    {
        var ids = Tokenizer.EncodeToIds(rendered);
        if (images.Count == 0)
        {
            return (ids.ToArray(), null);
        }
        var vision = Vision!;
        if (ids.Count(i => i == vision.ImageToken) != images.Count)
        {
            throw new ArgumentException("the text of a message holds the mark that stands for an image");
        }
        var prompt = new List<int>(ids.Count + images.Count * vision.SoftTokens);
        var placed = new List<PromptImage>();
        foreach (int id in ids)
        {
            if (id != vision.ImageToken)
            {
                prompt.Add(id);
                continue;
            }
            var part = images[placed.Count];
            var image = Images.Get(vision, part.Image!, part.ImageTokens);
            prompt.Add(vision.BeginToken);
            placed.Add(new PromptImage(prompt.Count, image));
            prompt.AddRange(image.Image.Ids());
            prompt.Add(vision.EndToken);
        }
        return (prompt.ToArray(), placed);
    }

    /// <summary>Why requests are not batched, when they are not.</summary>
    public string? BatchRefusal { get; }

    public TextGenerator(DecodeSession session, HfTokenizer tokenizer, ChatTemplate template, IReadOnlySet<int> stopTokens, int contextLength)
        : this(session, Array.Empty<IEntailAdapter>(), null, tokenizer, template, stopTokens, contextLength)
    {
    }

    public TextGenerator(DecodeSession partner, IReadOnlyList<IEntailAdapter> adapters, Func<int[], int[]?>? priorPrompt,
        HfTokenizer tokenizer, ChatTemplate template, IReadOnlySet<int> stopTokens, int contextLength)
    {
        _session = new SideBySideSession(partner, adapters) { PriorPrompt = priorPrompt };
        Adapters = adapters;
        Tokenizer = tokenizer;
        Template = template;
        _stopTokens = stopTokens;
        ContextLength = contextLength;
        var dialect = template.Dialect;
        if (dialect.ThinkOpenMark.Length > 0)
        {
            foreach (var (mark, text) in new[] { (dialect.ThinkOpenMark, "<think>"), (dialect.ThinkCloseMark, "</think>") })
            {
                var ids = tokenizer.EncodeToIds(mark).ToArray();
                int? token = ids.Length > 0 && TurnMarks.First(tokenizer, mark) == ids[0] ? ids[0] : null;
                if (text == "<think>")
                {
                    ThinkOpen = token;
                }
                else
                {
                    ThinkClose = token;
                }
                if (token is not null && (mark != text || ids.Any(tokenizer.IsSpecial)))
                {
                    _thinkMarks.Add((ids, text));
                }
            }
        }
        Calls = dialect.Calls is { } shape ? new CallSyntax(shape) : null;
        foreach (int id in (Calls?.Pieces ?? []).SelectMany(piece => TurnMarks.All(tokenizer, piece)).Where(tokenizer.IsSpecial))
        {
            _callMarks[id] = tokenizer.DecodeText(new[] { id }, skipSpecialTokens: false);
        }

        if (!BatchingEnabled)
        {
            BatchRefusal = "batching is off (--no-batch)";
        }
        else if (!CudaShim.Enabled)
        {
            BatchRefusal = "no CUDA device";
        }
        else if (partner.DeviceEngine() is not { } engine)
        {
            BatchRefusal = $"the GPU engine does not run this model ({partner.GpuRefusal})";
        }
        else
        {
            try
            {
                foreach (var adapter in adapters)
                {
                    adapter.Sibling().Dispose();
                }
                _batch = new BatchScheduler(engine, this, adapters, priorPrompt);
            }
            catch (InvalidOperationException e)
            {
                BatchRefusal = $"an adapter cannot run once per request: {e.Message}";
            }
        }
    }

    /// <summary>The prompt as token ids. The rendered template already carries
    /// every special token it needs, so no post-processor template is added —
    /// HF tokenises chat prompts with <c>add_special_tokens=False</c>.</summary>
    public int[] EncodePrompt(IReadOnlyList<ChatMessage> messages, bool enableThinking, IReadOnlyList<System.Text.Json.JsonElement>? tools = null) =>
        Tokenizer.EncodeToIds(Template.Apply(messages, enableThinking, tools)).ToArray();

    public void Warm()
    {
        _session.Reset();
        _session.Prefill(new[] { 0 });
        _session.Reset();
    }

    /// <summary>Generates a reply. Safe to call from several threads at once
    /// when requests are batched; otherwise the caller runs one at a time.</summary>
    public GenerationResult Generate(int[] prompt, GenerationParams p, Action<string>? onDelta,
        CancellationToken cancel, int? logprobsTopK = null)
    {
        var run = new GenerationRun(this, prompt, p, onDelta, cancel, logprobsTopK);
        if (_batch is not null)
        {
            return _batch.Run(run);
        }
        _session.Reset();
        _session.Prefill(prompt);
        LogitsRow logits = () => _session.Logits.Row(0);
        run.Begin(logits, _session.Verdict, _session.Prior);
        while (run.Next() is { } token)
        {
            _session.Step(token);
            run.Fed(logits, _session.Verdict);
        }
        return run.Result!;
    }

    public void Dispose() => _batch?.Dispose();
}

/// <summary>
/// One reply being generated, step by step: given the next-token logits (and
/// what the adapters require), it picks the next token, decodes and emits the
/// text, and says when the reply is over — a stop token, a stop string, the
/// token budget or cancellation. Who runs the model (one session, or a batch of
/// sequences) is not its concern.
/// </summary>
/// <summary>
/// Whether a reply's own tail is one stretch said again and again, back to back: at least
/// <see cref="Repeats"/> times and <see cref="Tokens"/> tokens in all, for a stretch of up to
/// <see cref="LongestPeriod"/> tokens. Only the reply is read, never the prompt, so copying a
/// path or a command from the conversation is never mistaken for a loop; and code that merely
/// resembles itself line to line does not repeat token for token for that long.
/// </summary>
public static class LoopGuard
{
    public const int Repeats = 4, Tokens = 192, LongestPeriod = 256;

    /// <summary>A counting loop (lines that differ only in a number or hash) must be held much
    /// longer before it counts: a numbered list is fine, the same line numbered 2,000 times is not.</summary>
    public const int CountingRepeats = 16, CountingTokens = 512;

    /// <summary>The key a number or hash token has in a loose sequence.</summary>
    public const int Counter = -1;

    /// <summary>Whether a token's text is a number or a piece of a hash (hex with a digit).</summary>
    public static bool IsCounter(string piece)
    {
        string t = piece.Trim();
        return t.Length > 0 && t.Any(char.IsDigit) && t.All(Uri.IsHexDigit);
    }

    /// <summary>The loop a loose sequence (number and hash tokens keyed <see cref="Counter"/>)
    /// is in: "L2196:c74105│ //", "L2197:…" over and over. Needs <see cref="CountingRepeats"/>
    /// rounds and <see cref="CountingTokens"/> tokens.</summary>
    public static (int Period, int Repeats)? FindCounting(IReadOnlyList<int> loose, int from = 0) =>
        Find(loose, from, CountingRepeats, CountingTokens);

    /// <summary>The loop the tail of <paramref name="reply"/> is in, looking no further back
    /// than <paramref name="from"/>; null when it is in none.</summary>
    public static (int Period, int Repeats)? Find(IReadOnlyList<int> reply, int from = 0) =>
        Find(reply, from, Repeats, Tokens);

    private static (int Period, int Repeats)? Find(IReadOnlyList<int> reply, int from, int Repeats, int Tokens)
    {
        int n = reply.Count;
        if (n - from < Tokens)
        {
            return null;
        }
        for (int period = 1; period <= LongestPeriod && period * Repeats <= n - from; period++)
        {
            // How far back the tail keeps its period.
            int k = 0;
            while (n - 1 - k - period >= from && reply[n - 1 - k] == reply[n - 1 - k - period])
            {
                k++;
            }
            int span = k + period;
            if (span >= Tokens && span / period >= Repeats)
            {
                return (period, span / period);
            }
        }
        return null;
    }
}

/// <summary>The current next-token logits of one reply, wherever they are.</summary>
internal delegate ReadOnlySpan<float> LogitsRow();

internal sealed class GenerationRun
{
    private readonly TextGenerator _g;
    private readonly GenerationParams _p;
    private readonly Action<string>? _onDelta;
    private readonly CancellationToken _cancel;

    /// <summary>Whether the request was given up.</summary>
    public bool Cancelled => _cancel.IsCancellationRequested;

    /// <summary>The most tokens the reply may take.</summary>
    public int Budget => _budget;
    private readonly int? _logprobsTopK;
    private readonly SamplingConfig _config;
    private readonly Random _rng;
    private readonly int _budget;
    private readonly int _holdBack;
    private readonly Queue<int> _forced = new();
    private readonly List<int> _generated = new();
    private readonly List<int> _loose = new();   // _generated with number and hash tokens as one
    private readonly List<LogprobsResult>? _logprobs;
    private bool _forcePending;
    private string _emitted = string.Empty;
    private readonly JsonConstraint? _json;
    private readonly ToolGrammar? _toolGrammar;
    private ICallConstraint? _call;
    private bool _said;   // something but blank space, thinking aside
    private int _openMatch;   // how much of a call's opening the reply's text ends with
    private bool _thinkClosed;
    private int _loopFrom;   // the loop guard looks only at tokens from here on
    private float[]? _masked;
    private LogitsRow _logits = null!;
    private float[]? _blended;      // this step's logits with the n-gram packs blended in
    private bool _blendedNow;
    private EntailVerdict _verdict = EntailVerdict.Silence;
    private IReadOnlyList<double>? _prior;

    public int[] Prompt { get; }

    /// <summary>The prompt's images.</summary>
    public IReadOnlyList<PromptImage> Images => _p.Images ?? [];

    /// <summary>The finished reply, once <see cref="Next"/> has returned null.</summary>
    public GenerationResult? Result { get; private set; }

    public GenerationRun(TextGenerator g, int[] prompt, GenerationParams p, Action<string>? onDelta,
        CancellationToken cancel, int? logprobsTopK)
    {
        if (prompt.Length == 0)
        {
            throw new ArgumentException("the prompt renders to no tokens");
        }
        _budget = Math.Min(p.MaxTokens, g.ContextLength - prompt.Length);
        if (_budget <= 0)
        {
            throw new ArgumentException($"the prompt is {prompt.Length} tokens, which fills the {g.ContextLength}-token context");
        }
        _g = g;
        Prompt = prompt;
        _p = p;
        _onDelta = onDelta;
        _cancel = cancel;
        _logprobsTopK = logprobsTopK;
        _config = new SamplingConfig(p.Seed, p.Temperature, TopK: p.TopK, TopP: p.TopP >= 1f ? 0f : p.TopP);
        _rng = new Random(p.Seed);
        _logprobs = logprobsTopK is > 0 ? new List<LogprobsResult>() : null;
        _holdBack = p.Stop.Count == 0 ? 0 : p.Stop.Max(s => s.Length) - 1;
        _json = p.Json is null ? null : new JsonConstraint(p.Json);
        if (p.Tools is { Count: > 0 } tools && TextGenerator.ToolGrammarEnabled && g.Calls is not null)
        {
            _toolGrammar = ToolGrammar.Compile(tools);
            if (p.Prefill is { Length: > 0 } started && !p.Thinking)
            {
                WatchForCall(started);   // a started reply may already have opened a call
            }
        }
        // Forced text goes in after the thinking block, or at once without one.
        _forcePending = p.Forced is not null;
        if (_forcePending && !p.Thinking)
        {
            Enqueue(p.Forced!);
        }
        else if (_forcePending && !p.ThinkOpened)
        {
            // The call follows the block, so there must be one.
            foreach (int id in g.Tokenizer.EncodeToIds(g.OpenThinking))
            {
                _forced.Enqueue(id);
            }
        }
    }

    /// <summary>The logits after the prompt, and what the adapters require of the first token.</summary>
    public void Begin(LogitsRow logits, EntailVerdict verdict, IReadOnlyList<double>? prior)
    {
        _logits = logits;
        _verdict = verdict;
        _prior = prior;
    }

    /// <summary>The logits after the token <see cref="Next"/> returned.</summary>
    public void Fed(LogitsRow logits, EntailVerdict verdict)
    {
        _logits = logits;
        _verdict = verdict;
    }

    /// <summary>The next token to run the model on, or null when the reply is over.</summary>
    public int? Next()
    {
        if (Result is not null)
        {
            return null;
        }
        if (_generated.Count >= _budget || _cancel.IsCancellationRequested)
        {
            return Finish("length");
        }
        _blendedNow = false;
        if (_p.NGram is { } prior && _forced.Count == 0 && (!_p.Thinking || _thinkClosed) &&
            (prior.Scope == "all" || _call is not null || Amql.Inference.NGram.NGramPrior.InFencedCode((_p.Prefill ?? "") + _stableText)))
        {
            var raw = _logits();
            _blended ??= new float[raw.Length];
            _blendedNow = prior.Apply(raw, Context(prior), _blended);
        }

        int token;
        if (_forced.Count > 0)
        {
            token = _forced.Dequeue();
        }
        else if (_json is not null)
        {
            token = Constrained(_json);
        }
        else if (_call is not null)
        {
            token = Constrained(_call);
        }
        else if (AnnouncedCall() is { } announced)
        {
            token = Constrained(announced);
        }
        else if (_p.NoCalls && _g.Calls is { } calls && (!_p.Thinking || _thinkClosed))
        {
            token = Constrained(NoCallOpen.After(calls, (_p.Prefill ?? "") + _stableText));
        }
        else if (_verdict.Silent)
        {
            LogprobsResult? lp;
            (token, lp) = Sampler.SampleWithLogprobs(Row(), _config, _rng, _logprobsTopK);
            if (lp is not null)
            {
                _logprobs!.Add(lp);
            }
        }
        else
        {
            // An adapter rules on this token; logprobs still describe the model's own distribution.
            (token, _) = _verdict.Choose(Row(), _prior, r => Sampler.Sample(r, _config, _rng));
            if (_logprobs is not null && Sampler.SampleWithLogprobs(Row(), _config, new Random(0), _logprobsTopK).Logprobs is { } lp)
            {
                _logprobs.Add(lp);
            }
        }
        if (_g.StopTokenIds.Contains(token))
        {
            return Finish("stop");
        }
        if (_generated.Count == 0 && _p.Thinking && !_p.ThinkOpened && _forced.Count == 0 &&
            (_g.ThinkOpen is { } opens ? token != opens : !_g.Dialect.ThinkOpenMark.StartsWith(_g.TokenPiece(token).Trim(), StringComparison.Ordinal)))
        {
            _thinkClosed = true;   // it chose not to think
        }
        string piece = _g.ReplyPiece(token, _logits().Length);
        _json?.Advance(piece);
        if (_call is not null)
        {
            _call.Advance(piece);
            if (_call.Done)
            {
                _call = null;
            }
        }
        else if (_toolGrammar is not null && (!_p.Thinking || _thinkClosed))
        {
            WatchForCall(piece);
        }
        _generated.Add(token);
        _loose.Add(LoopGuard.IsCounter(piece) ? LoopGuard.Counter : token);
        if (token == _g.ThinkClose)
        {
            _thinkClosed = true;
        }
        if (_forcePending && _p.Thinking && token == _g.ThinkClose)
        {
            Enqueue(_g.AfterThinking + _p.Forced);
        }

        if (_p.ThinkingBudget > 0 && _p.Thinking && !_thinkClosed && _g.ThinkClose is not null && _json is null && _call is null && _forced.Count == 0
            && ThinkingLimit.For(_p.ThinkingBudget, _budget) is var limit && _generated.Count >= limit
            && (piece.EndsWith('\n') || _generated.Count >= limit + ThinkingLimit.LineWait))
        {
            // Out of thinking budget: close the block, at the end of a line when one comes soon.
            foreach (int id in _g.Tokenizer.EncodeToIds(_g.CloseThinking(piece.EndsWith('\n'))))
            {
                _forced.Enqueue(id);
            }
            _loopFrom = _generated.Count;
        }

        string text = Text();
        int stopAt = FirstStop(text, _p.Stop);
        if (stopAt >= 0)
        {
            Emit(text[..stopAt]);
            _emitted = text[..stopAt];
            Result = Build(_emitted, "stop");
            return null;
        }
        // Hold back a trailing partial character, and enough text that a
        // stop string straddling the next token is never half-sent.
        int safe = text.Length;
        while (safe > 0 && text[safe - 1] == '�')
        {
            safe--;
        }
        safe = Math.Max(_emitted.Length, safe - _holdBack);
        Emit(text[..safe]);
        if (TextGenerator.LoopGuardEnabled && _json is null && _generated.Count % 8 == 0 &&
            (LoopGuard.Find(_generated, _loopFrom) ?? LoopGuard.FindCounting(_loose, _loopFrom)) is { } loop)
        {
            if (_p.Thinking && !_thinkClosed && _g.ThinkClose is not null && _forced.Count == 0)
            {
                // Thinking that circles is closed, and the answer goes on from there.
                Console.Error.WriteLine($"{DateTime.Now:HH:mm:ss.fff} loop guard: thinking repeats a {loop.Period}-token stretch {loop.Repeats} times; closing it");
                foreach (int id in _g.Tokenizer.EncodeToIds(_g.CloseThinking(lineEnded: false)))
                {
                    _forced.Enqueue(id);
                }
                _loopFrom = _generated.Count;
            }
            else
            {
                Console.Error.WriteLine($"{DateTime.Now:HH:mm:ss.fff} loop guard: the reply repeats a {loop.Period}-token stretch {loop.Repeats} times; " +
                                        $"stopped at {_generated.Count} tokens");
                int keep = _generated.Count - loop.Period * (loop.Repeats - 1);
                string once = _g.DecodeReply(_generated.GetRange(0, keep)).TrimEnd('�');
                Finish("length");
                Result = Result! with { Looped = once };
                return null;
            }
        }
        return _generated.Count >= _budget ? Finish("length") : token;
    }

    // The reply's text so far, decoded incrementally: the text of the first
    // _stableTokens tokens (which end on a whole character) is kept, and only
    // the tokens after it are decoded each step. UTF-8 cut at a character
    // boundary decodes the same in pieces as whole, so this is the full decode.
    private string _stableText = string.Empty;
    private int _stableTokens;

    private string Text()
    {
        var tail = _generated.GetRange(_stableTokens, _generated.Count - _stableTokens);
        string text = _stableText + _g.DecodeReply(tail);
        // A tail ending mid-character waits for its next token; one that never
        // completes (invalid bytes) is kept as decoded after 64 tokens.
        if (tail.Count > 0 && !_g.EndsInsideMark(tail) && (text.Length == 0 || text[^1] != '�' || tail.Count > 64))
        {
            _stableText = text;
            _stableTokens = _generated.Count;
        }
        return text;
    }

    private string CallOpen => _g.Calls!.Open;

    /// <summary>Set false (<c>--no-announce-rule</c>) to let a reply end on "…let me build it:".</summary>
    internal static bool AnnounceRuleEnabled = true;

    /// <summary>
    /// The announced-call rule. A reply with tools on offer whose text, outside thinking, ends with
    /// a colon ("Now let me build the project:") has said that something follows; ending the turn
    /// there leaves the harness a promise and no call, and a small model then goes on to claim the
    /// build succeeded. So there, when the model gives opening a call at least
    /// <see cref="MentionFloor"/>, the call opens (whitespace may come first); when it does not,
    /// nothing is forced. Returns the grammar for this step, or null when the rule does not act.
    /// </summary>
    private ITokenConstraint? AnnouncedCall()
    {
        if (!AnnounceRuleEnabled || _toolGrammar is null || _g.Calls!.AtStartOnly || _json is not null || (_p.Thinking && !_thinkClosed))
        {
            return null;
        }
        string text = (_p.Prefill ?? "") + _stableText;
        int close = text.LastIndexOf("</think>", StringComparison.Ordinal);
        string visible = (close >= 0 ? text[(close + "</think>".Length)..] : text).TrimEnd();
        // A colon in prose announces; in a code block it is code ("def f():", "case 1:"). After the
        // colon: whitespace, then perhaps the call's opening already begun (a tokenizer without a
        // "<tool_call>" token writes it a piece at a time), which the rule sees through to the end.
        if (Amql.Inference.NGram.NGramPrior.InFencedCode(visible))
        {
            return null;
        }
        int colon = visible.LastIndexOf(':');
        if (colon < 0)
        {
            return null;
        }
        string after = visible[(colon + 1)..];
        string begun = after.TrimStart(' ', '\n', '\r', '\t');
        if (begun.Length >= CallOpen.Length || !CallOpen.StartsWith(begun, StringComparison.Ordinal))
        {
            return null;
        }
        var way = new OpensCall(CallOpen[begun.Length..], begun.Length == 0);
        if (begun.Length > 0)
        {
            return way;   // the opening is under way: finish it
        }
        // Only an announced action of the reply's own ("let me build it:", "I'll run the tests:")
        // promises a call. "Here's what was done:" promises the list that follows, and forcing a
        // call there sends the agent round again; the reply may only not end on it.
        if (!TextGenerator.AnnouncesAction(visible[..colon]))
        {
            return NotTheEnd.Instance;
        }
        var row = Row();
        float max = float.NegativeInfinity;
        foreach (float v in row)
        {
            max = Math.Max(max, v);
        }
        double sum = 0, on = 0;
        for (int id = 0; id < row.Length; id++)
        {
            double e = Math.Exp(row[id] - max);
            sum += e;
            if (e > 1e-12 && !_g.StopTokenIds.Contains(id) && _g.ReplyPiece(id, row.Length) is { Length: > 0 } t && way.Accepts(t) && t.Trim().Length > 0)
            {
                on += e;
            }
        }
        if (Environment.GetEnvironmentVariable("AMQL_DEBUG_RULES") is "1")
        {
            Console.Error.WriteLine($"announce rule: '{visible[^Math.Min(30, visible.Length)..]}' p(call) = {on / sum:G3}");
        }
        return on / sum >= TextGenerator.AnnounceFloor ? way : null;
    }

    /// <summary>Anything but a call's markup, whole or written a piece at a time (a reply that must
    /// answer from what it has would otherwise write the call out as text).</summary>
    private sealed class NoCallOpen(CallSyntax calls, string tail) : ITokenConstraint
    {
        public static NoCallOpen After(CallSyntax calls, string text) => new(calls, text.Length > Longest ? text[^Longest..] : text);

        private const int Longest = 40;   // longer than any markup

        public bool Accepts(string t)
        {
            string s = tail + t;
            return !calls.Markup.Any(m => s.IndexOf(m, StringComparison.Ordinal) is var i && i >= 0 && i + m.Length > tail.Length);
        }

        public void Advance(string text)
        {
        }

        public bool CanEnd => true;
    }

    /// <summary>Anything but the end of the reply.</summary>
    private sealed class NotTheEnd : ITokenConstraint
    {
        public static readonly NotTheEnd Instance = new();

        public bool Accepts(string t) => true;

        public void Advance(string text)
        {
        }

        public bool CanEnd => false;
    }

    /// <summary>The rest of the way to a call's opening (whitespace first, if the opening has not
    /// begun); never the end of the reply.</summary>
    private sealed class OpensCall(string remaining, bool spaceFirst) : ITokenConstraint
    {
        public bool Accepts(string t)
        {
            string rest = spaceFirst ? t.TrimStart(' ', '\n', '\r', '\t') : t;
            return rest.Length == 0 ? spaceFirst
                : remaining.StartsWith(rest, StringComparison.Ordinal) || rest.StartsWith(remaining, StringComparison.Ordinal);
        }

        public void Advance(string text)
        {
        }

        public bool CanEnd => false;
    }

    /// <summary>The reply opens a call when its text, outside thinking, reaches a call's opening:
    /// from there to its close it is held to the tools. What the same token says after
    /// the opening goes to the call; if that already breaks it, the call is left unheld.</summary>
    private void WatchForCall(string piece)
    {
        if (_g.Calls!.AtStartOnly && _openMatch == 0 && _said)
        {
            return;
        }
        for (int i = 0; i < piece.Length; i++)
        {
            char c = piece[i];
            _said |= !char.IsWhiteSpace(c);
            _openMatch = c == CallOpen[_openMatch] ? _openMatch + 1 : c == CallOpen[0] ? 1 : 0;
            if (_openMatch < CallOpen.Length)
            {
                continue;
            }
            _openMatch = 0;
            var call = _g.Calls!.Constraint(_toolGrammar!);
            string rest = piece[(i + 1)..];
            if (rest.Length == 0 || call.Accepts(rest))
            {
                if (rest.Length > 0)
                {
                    call.Advance(rest);
                }
                _call = call.Done ? null : call;
            }
            return;
        }
    }

    /// <summary>
    /// The next token under a grammar (structured output, or a tool call): the request's
    /// own sampling over just the tokens the grammar allows. The 256 likeliest tokens are
    /// checked first, the whole vocabulary only when none of them fits; a stop token is
    /// valid where the grammar may end.
    /// </summary>
    private int Constrained(ITokenConstraint grammar)
    {
        var row = Row();
        int n = row.Length;
        _masked ??= new float[n];
        Array.Fill(_masked, float.NegativeInfinity);
        Func<string, bool>? onPath = grammar is ICallConstraint call ? MentionedPath(call) : null;
        // A call's marks are its text; in JSON they would be no one's.
        bool Valid(int id) => _g.StopTokenIds.Contains(id)
            ? grammar.CanEnd && onPath is null
            : (grammar is JsonConstraint ? _g.TokenText(id, n) : _g.ReplyPiece(id, n)) is { Length: > 0 } t && grammar.Accepts(t) && (onPath is null || onPath(t));

        bool any = false;
        foreach (int id in TopIndices(row, 256))
        {
            if (Valid(id))
            {
                _masked[id] = row[id];
                any = true;
            }
        }
        if (!any)
        {
            for (int id = 0; id < n; id++)
            {
                if (!float.IsNegativeInfinity(row[id]) && Valid(id))
                {
                    _masked[id] = row[id];
                    any = true;
                }
            }
        }
        if (!any)
        {
            throw new InvalidOperationException("no token continues what the grammar allows");
        }
        return Sampler.Sample(_masked, _config, _rng);
    }

    /// <summary>The smallest share of the model's probability that the tokens towards a mentioned
    /// parameter must have for the rule to act: it follows the model, never overrides it outright.</summary>
    private const double MentionFloor = 1e-4;

    /// <summary>
    /// The mentioned-parameter rule. A small model plans in words ("I'll set overwrite to true")
    /// and then writes the call by habit: the required parameters in schema order, then
    /// &lt;/function&gt;, at 0.99+ even when the tool's last error said "Pass overwrite: true".
    /// So where the call can open a parameter, an optional parameter not given yet whose name
    /// the reply's own thinking or the newest user or tool message mentions (as a whole word) is
    /// opened next, provided the model gives the way there at least <see cref="MentionFloor"/>.
    /// Its value is the model's own, under the grammar. Returns a filter for the tokens on that
    /// way, or null when the rule does not act here.
    /// </summary>
    private Func<string, bool>? MentionedPath(ICallConstraint call)
    {
        if (!TextGenerator.MentionRuleEnabled)
        {
            return null;
        }
        if (MentionedToolPath(call) is { } toTool)
        {
            return toTool;
        }
        string context = (_p.Recent ?? "") + "\n" + Reasoning();
        foreach (string name in call.OptionalLeft())
        {
            string word = Regex.Escape(name);
            // A tool result names many fields ("handle": "bg-95c9") and warns about some ("'handle'
            // means poll or stop"); only one that tells the call what to pass counts.
            bool mentioned = Regex.IsMatch(context, $@"(?<![\w-]){word}(?![\w-])") ||
                (_p.RecentResults is { } results && TextGenerator.SuggestsParameter(results, name));
            if (call.PathTo(name) is not { } path || !mentioned)
            {
                continue;
            }
            // Whitespace may come first; then the text must stay on the way to "<parameter=NAME>".
            bool OnPath(string t)
            {
                string rest = t.TrimStart(' ', '\n', '\r', '\t');
                return rest.Length == 0 || path.StartsWith(rest, StringComparison.Ordinal) || rest.StartsWith(path, StringComparison.Ordinal);
            }
            var row = Row();
            float max = float.NegativeInfinity;
            foreach (float v in row)
            {
                max = Math.Max(max, v);
            }
            double sum = 0, on = 0;
            for (int id = 0; id < row.Length; id++)
            {
                double e = Math.Exp(row[id] - max);
                sum += e;
                if (e > 1e-12 && !_g.StopTokenIds.Contains(id) && _g.ReplyPiece(id, row.Length) is { Length: > 0 } t && OnPath(t) && t.Trim().Length > 0)
                {
                    on += e;
                }
            }
            if (on / sum >= MentionFloor)
            {
                return OnPath;
            }
        }
        return null;
    }

    /// <summary>
    /// The mentioned-tool rule, the same idea for the tool a call names. Told "write a new version
    /// with create_file, no need to read the old one", Qwen 3.5 0.8B read the file and patched it with
    /// edit_file, whose anchors it cannot write. So where the first call after the user's newest
    /// message names its tool and the message names some of the tools (not in a clause saying not
    /// to), the call names the first of them, provided the model gives it at least
    /// <see cref="MentionFloor"/>. A request whose message names no tool is untouched.
    /// </summary>
    private Func<string, bool>? MentionedToolPath(ICallConstraint call)
    {
        // Only the first call after the message: later, the model is working from results, and a
        // redirect there turned a reply that meant to read the file it had to fix into a third build.
        if (_p.Task is not { Length: > 0 } recent || _p.CalledSince is { Count: > 0 })
        {
            return null;
        }
        var paths = TextGenerator.ToolsDue(recent, call.ToolNames, _p.CalledSince)
            .Select(call.PathToTool).OfType<string>().ToList();
        if (paths.Count == 0)
        {
            return null;
        }
        bool OnPath(string t)
        {
            string rest = t.TrimStart(' ', '\n', '\r', '\t');
            return rest.Length == 0 || paths.Any(p => p.StartsWith(rest, StringComparison.Ordinal) || rest.StartsWith(p, StringComparison.Ordinal));
        }
        return Share(OnPath) >= MentionFloor ? OnPath : null;
    }

    /// <summary>The share of the model's probability on the tokens (other than stops and
    /// whitespace) whose text passes <paramref name="on"/>.</summary>
    private double Share(Func<string, bool> on)
    {
        var row = Row();
        float max = float.NegativeInfinity;
        foreach (float v in row)
        {
            max = Math.Max(max, v);
        }
        double sum = 0, share = 0;
        for (int id = 0; id < row.Length; id++)
        {
            double e = Math.Exp(row[id] - max);
            sum += e;
            if (e > 1e-12 && !_g.StopTokenIds.Contains(id) && _g.ReplyPiece(id, row.Length) is { Length: > 0 } t && t.Trim().Length > 0 && on(t))
            {
                share += e;
            }
        }
        return share / sum;
    }

    /// <summary>The logits this step chooses from: the model's, or with the packs blended in.</summary>
    private ReadOnlySpan<float> Row() => _blendedNow ? _blended : _logits();

    private readonly List<int> _context = new();

    /// <summary>The last tokens of prompt and reply together, as the n-gram packs read them.</summary>
    private IReadOnlyList<int> Context(Amql.Inference.NGram.NGramPrior prior)
    {
        int need = prior.Packs.Max(p => p.Order) - 1;
        _context.Clear();
        int fromPrompt = Math.Max(0, need - _generated.Count);
        for (int i = Math.Max(0, Prompt.Length - fromPrompt); i < Prompt.Length; i++)
        {
            _context.Add(Prompt[i]);
        }
        for (int i = Math.Max(0, _generated.Count - need); i < _generated.Count; i++)
        {
            _context.Add(_generated[i]);
        }
        return _context;
    }

    /// <summary>The reply's thinking so far (all of it, when the prompt opened a think block).</summary>
    private string Reasoning()
    {
        if (!_p.Thinking)
        {
            return "";
        }
        string text = _stableText;
        int close = text.IndexOf("</think>", StringComparison.Ordinal);
        return close >= 0 ? text[..close] : text;
    }

    private static int[] TopIndices(ReadOnlySpan<float> row, int k)
    {
        var heap = new PriorityQueue<int, float>(k + 1);
        for (int i = 0; i < row.Length; i++)
        {
            if (heap.Count < k)
            {
                heap.Enqueue(i, row[i]);
            }
            else if (row[i] > row[heap.Peek()])
            {
                heap.EnqueueDequeue(i, row[i]);
            }
        }
        var top = new int[heap.Count];
        for (int i = top.Length - 1; i >= 0; i--)
        {
            top[i] = heap.Dequeue();
        }
        return top;
    }

    /// <summary>Ends the reply now, for a failure in the model under it.</summary>
    public void Abort(string reason) => Result ??= Build(Text(), reason);

    private int? Finish(string finish)
    {
        string final = Text();
        Emit(final);
        Result = Build(final, _cancel.IsCancellationRequested && finish != "stop" ? "cancelled" : finish);
        return null;
    }

    private void Emit(string upTo)
    {
        if (upTo.Length > _emitted.Length && upTo.StartsWith(_emitted, StringComparison.Ordinal))
        {
            _onDelta?.Invoke(upTo[_emitted.Length..]);
            _emitted = upTo;
        }
    }

    private void Enqueue(string text)
    {
        _forcePending = false;
        foreach (int id in _g.Tokenizer.EncodeToIds(text))
        {
            _forced.Enqueue(id);
        }
    }

    private static int FirstStop(string text, IReadOnlyList<string> stops)
    {
        int best = -1;
        foreach (var s in stops)
        {
            int at = text.IndexOf(s, StringComparison.Ordinal);
            if (at >= 0 && (best < 0 || at < best))
            {
                best = at;
            }
        }
        return best;
    }

    private GenerationResult Build(string text, string finish)
    {
        float? perplexity = null;
        float? meanEntropy = null;
        if (_logprobs is { Count: > 0 })
        {
            double nllSum = _logprobs.Sum(lp => -(double)lp.TokenLogprob);
            perplexity = (float)Math.Exp(nllSum / _logprobs.Count);
            meanEntropy = _logprobs.Average(lp => lp.Entropy);
        }
        return new GenerationResult(text, finish, Prompt.Length, _generated.Count)
        {
            Tokens = _generated.ToList(),
            Logprobs = _logprobs,
            Perplexity = perplexity,
            MeanEntropy = meanEntropy,
        };
    }
}

/// <summary>Where the batch scheduler saves a prompt's state for the next request.</summary>
public static class PromptMark
{
    /// <summary>Where a prompt's state is saved: at its second-last turn marker.
    /// The last opens the reply's own turn, and the message before it is the
    /// newest, which an agent may not keep in the next call's history (Hydra
    /// sends its notes as a user message after the newest one, and moves it on
    /// every call). Everything before stays the same from one call to the next. Where a call's
    /// results stay in the reply's own turn (<paramref name="resultStart"/>), a prompt
    /// that ends on one is saved before it. <paramref name="replyStart"/> opens the reply's turn where
    /// that is another token.</summary>
    public static int At(int[] prompt, int? turnStart, int? resultStart = null, int? replyStart = null)
    {
        if (resultStart is { } result && Array.LastIndexOf(prompt, result) is var at and > 0 &&
            (turnStart is not { } t || at > Array.LastIndexOf(prompt, t)))
        {
            return at;
        }
        if (turnStart is { } start)
        {
            int last = Math.Max(Array.LastIndexOf(prompt, start), replyStart is { } reply ? Array.LastIndexOf(prompt, reply) : -1);
            int before = last > 0 ? Array.LastIndexOf(prompt, start, last - 1) : -1;
            if (before > 0)
            {
                return before;
            }
            if (last > 0)
            {
                return last;
            }
        }
        return prompt.Length - 1;
    }

    /// <summary>The most first positions auto keeps exact.</summary>
    public const int AutoExactMost = 16384;

    /// <summary>How many first positions to keep exact: <paramref name="mode"/>
    /// tokens when positive, none at 0, and at -1 (auto) everything through the
    /// first user message, up to the turn after it: the system prompt, the tools
    /// and the first instruction, up to <see cref="AutoExactMost"/> tokens (a first
    /// message that is a whole document is not all instruction, and at FP16 a
    /// million tokens of it would not fit a GPU). None when the prompt has no
    /// user turn.</summary>
    public static int Exact(int[] prompt, int mode, int? turnStart, int? userRole)
    {
        if (mode >= 0)
        {
            return Math.Min(mode, prompt.Length);
        }
        if (turnStart is not { } start || userRole is not { } user)
        {
            return 0;
        }
        for (int i = 0; i + 1 < prompt.Length; i++)
        {
            if (prompt[i] == start && prompt[i + 1] == user)
            {
                int next = Array.IndexOf(prompt, start, i + 2);
                return Math.Min(next < 0 ? prompt.Length : next, AutoExactMost);
            }
        }
        return 0;
    }
}

/// <summary>
/// Continuous batching on the GPU engine. One thread owns the engine: it admits
/// waiting requests (each gets a sequence of its own and is prefilled), then
/// steps every running request together, one token each, and repeats; a request
/// leaves the batch the moment it finishes, and the next waiting one takes its
/// place. Each request has its own copy of the adapters. Every call into the
/// device holds <see cref="ModelHost.DeviceGate"/>, so other models on the GPU
/// interleave with this one a step at a time.
/// </summary>
internal sealed class BatchScheduler : IDisposable
{
    private sealed class Job(GenerationRun run)
    {
        public readonly GenerationRun Run = run;
        public readonly TaskCompletionSource<GenerationResult> Done = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public GpuSequence? Seq;
        public IEntailAdapter[] Adapters = Array.Empty<IEntailAdapter>();
        public int Token;
        public int Row;     // its row in the last step
        public DraftPace? Pace;   // how many drafts its next step asks for
        public int? Pending;   // a token already chosen (a verify's first miss), run next
        public List<int>? Context;   // the prompt and every token run since, for suffix drafts
        public (int Mark, ChunkedBuffer State)? Paged;   // a state taken from disk for its prompt
    }

    private readonly GpuEngine _engine;
    private readonly TextGenerator _g;
    private readonly IReadOnlyList<IEntailAdapter> _adapters;
    private readonly Func<int[], int[]?>? _priorPrompt;
    private readonly System.Collections.Concurrent.BlockingCollection<Job> _waiting = new();
    // Finished sequences kept for the next requests, least recently used first,
    // each with the prompt it last ran and when: one whose prompt the next
    // shares up to its mark is rewound there rather than prefilled again (an
    // agent's calls share all but their last few hundred tokens). They leave
    // GPU memory when unused for the idle time or past the GPU budget, oldest
    // first, paged to disk when the model has a disk tier. Only the batch
    // thread touches these (and Dispose, once it has stopped).
    private readonly List<GpuSequence> _free = new();
    private readonly Dictionary<GpuSequence, int[]> _held = new();
    private readonly Dictionary<GpuSequence, DateTime> _usedAt = new();
    private readonly int? _turnStart, _userRole, _resultStart, _replyStart;
    private GpuSequence? _priorSeq;
    private KvPaging _paging = KvPaging.Defaults;
    private KvPageStore? _store;
    private Action<string>? _log;
    private long _lastHousekeep;

    /// <summary>Set false (<c>--no-prefix-reuse</c>) to prefill every prompt whole.</summary>
    internal static bool PrefixReuse = true;

    public long ReusedTokens, PrefilledTokens, RestoredTokens, Released;

    /// <summary>The disk tier, when the model has one.</summary>
    public KvPageStore? Store => _store;

    /// <summary>The engine's geometry for a saved state (see <see cref="GpuEngine.MarkGeometry"/>).</summary>
    public string MarkGeometry => _engine.MarkGeometry;

    /// <summary>The KV cache's format, and why NVFP4 was refused when asked for.</summary>
    public (string Format, string? Refusal) KvFormat => (_engine.KvFormat, _engine.KvFormatRefusal);

    /// <summary>How long kept state stays, how much GPU memory it may hold,
    /// and the disk tier. Set before the first request.</summary>
    public void Configure(KvPaging paging, KvPageStore? store, Action<string>? log)
    {
        _paging = paging;
        _store = store;
        _log = log;
    }
    private readonly Thread _thread;

    // AMQL_SERVER_TIMING=1: where a batch step's time goes, printed every 500 steps.
    private static readonly bool Timing = Environment.GetEnvironmentVariable("AMQL_SERVER_TIMING") is "1";
    private long _tChoose, _tDevice, _tFeed, _steps, _rows;

    /// <summary>Drafts per step when one reply runs alone on a model with an
    /// MTP drafter (<c>AMQL_MTP_DRAFTS</c>, default 3; 0 turns speculation off).</summary>
    internal static int? DraftsSet =
        int.TryParse(Environment.GetEnvironmentVariable("AMQL_MTP_DRAFTS"), out int d) ? Math.Clamp(d, 0, GpuEngine.MaxDrafts) : null;

    private int Drafts => DraftsSet ?? _engine.DraftsThatPay;
    /// <summary>Suffix drafts: when the last <c>AMQL_SUFFIX_DRAFT</c> tokens
    /// (default 8; 0 turns it off) appeared earlier in the prompt or the reply,
    /// what followed them there is drafted instead of asking the drafter. An
    /// agent's replies repeat paths, names and code from its context. Shorter
    /// matches fire on prose's common phrases, where a quarter of such drafts
    /// held and the longer verify cost more than they saved.</summary>
    internal static int SuffixMatch =
        int.TryParse(Environment.GetEnvironmentVariable("AMQL_SUFFIX_DRAFT"), out int m) ? Math.Max(0, m) : 8;
    private long _specCycles, _specDrafted, _specKept, _specTokens, _suffixCycles, _suffixDrafted, _suffixKept;
    private long _tDraft, _tVerify, _tSpecFeed;

    public BatchScheduler(GpuEngine engine, TextGenerator g, IReadOnlyList<IEntailAdapter> adapters, Func<int[], int[]?>? priorPrompt)
    {
        _engine = engine;
        _g = g;
        _adapters = adapters;
        _priorPrompt = priorPrompt;
        // The tokens turns open with, as the template writes them.
        (_turnStart, _userRole, _replyStart, _resultStart) = TurnMarks.Openers(g.Tokenizer, g.Dialect);
        _thread = new Thread(Loop) { IsBackground = true, Name = "amql batch" };
        _thread.Start();
    }

    /// <summary>Gives the engine the container's vision tower.</summary>
    public void UseVision(VisionPlan plan, OperandStore store) => Device(() =>
    {
        _engine.UseVision(plan, store);
        return 0;
    });

    /// <summary>What the tower made of the images that are still to be read, from <paramref name="reused"/> on.</summary>
    private IReadOnlyList<ImageSpan>? Spans(GenerationRun run, int reused)
    {
        List<ImageSpan>? spans = null;
        foreach (var image in run.Images)
        {
            if (image.Start + image.Image.Image.Tokens > reused)
            {
                var features = image.Image.Features ??= Device(() => _engine.EncodeImage(image.Image.Image));
                (spans ??= new()).Add(new ImageSpan(image.Start, image.Image.Image.Tokens, features));
            }
        }
        return spans;
    }

    /// <summary>Queues a reply and waits for it.</summary>
    public GenerationResult Run(GenerationRun run)
    {
        var job = new Job(run);
        // Read from disk here, on the request's thread, so the batch keeps stepping meanwhile.
        if (PrefixReuse && _store is { } store)
        {
            job.Paged = store.Take(run.Prompt);
        }
        _waiting.Add(job);
        return job.Done.Task.GetAwaiter().GetResult();
    }

    private static T Device<T>(Func<T> work)
    {
        ModelHost.DeviceGate.Wait();
        try
        {
            return work();
        }
        finally
        {
            ModelHost.DeviceGate.Release();
        }
    }

    private void Loop()
    {
        var active = new List<Job>();
        while (!_waiting.IsCompleted)
        {
            try
            {
                Housekeep();
                // Idle with state kept: wake now and then to release what has sat too long.
                if (active.Count == 0 && _waiting.TryTake(out var first, _free.Count > 0 ? 1000 : Timeout.Infinite))
                {
                    Admit(first, active);
                }
                if (active.Count == 0)
                {
                    continue;
                }
                while (active.Count < GpuEngine.MaxBatch && _waiting.TryTake(out var waiting))
                {
                    Admit(waiting, active);
                }

                if (active.Count == 1 && Drafts > 0 && _engine.HasDrafter && _waiting.Count == 0)
                {
                    if (!Speculate(active[0]))
                    {
                        var done = active[0];
                        active.Clear();
                        Complete(done);
                    }
                    continue;
                }

                StepAll(active);
            }
            catch (InvalidOperationException) when (_waiting.IsAddingCompleted)
            {
                break;
            }
            catch (Exception e)
            {
                FailAll(active, e);
            }
        }
        foreach (var job in active)
        {
            job.Run.Abort("cancelled");
            Complete(job);
        }
    }

    /// <summary>The device failed under the batch: every running reply ends with the error.</summary>
    private void FailAll(List<Job> active, Exception e)
    {
        foreach (var job in active)
        {
            job.Run.Abort("error");
            if (job.Seq is { } failed)
            {
                _held.Remove(failed);   // its state is not to be trusted
            }
            Release(job);
            job.Done.TrySetException(e);
        }
        active.Clear();
    }

    /// <summary>One token for every running reply; those that end leave the list.</summary>
    private void StepAll(List<Job> active)
    {
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        // Each reply picks its next token on its own core: sampling reads the
        // whole vocabulary, and the replies share nothing but the device.
        var next = new int?[active.Count];
        int? NextOf(Job job)
        {
            var pending = job.Pending;
            job.Pending = null;
            return pending ?? job.Run.Next();
        }
        if (active.Count > 1)
        {
            Parallel.For(0, active.Count, i => next[i] = NextOf(active[i]));
        }
        else
        {
            next[0] = NextOf(active[0]);
        }
        for (int i = active.Count - 1; i >= 0; i--)
        {
            var job = active[i];
            if (next[i] is { } token)
            {
                job.Token = token;
            }
            else
            {
                active.RemoveAt(i);
                Complete(job);
            }
        }
        if (active.Count == 0)
        {
            return;
        }

        var seqs = active.Select(j => j.Seq!).ToList();
        var tokens = active.Select(j => j.Token).ToArray();
        long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
        Device(() =>
        {
            _engine.StepInPlace(seqs, tokens);
            return 0;
        });
        long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
        void Feed(int i)
        {
            var job = active[i];
            job.Row = i;
            job.Context?.Add(job.Token);
            var verdict = EntailVerdict.Merge(job.Adapters.Select(a => a.Step(job.Token)).ToList());
            // Read in place, before the next step overwrites it: no copy of the vocabulary per row.
            job.Run.Fed(() => _engine.HostLogits(job.Row), verdict);
        }
        if (active.Count > 1 && active[0].Adapters.Length > 0)
        {
            Parallel.For(0, active.Count, Feed);
        }
        else
        {
            for (int i = 0; i < active.Count; i++)
            {
                Feed(i);
            }
        }
        if (Timing)
        {
            _tChoose += t1 - t0;
            _tDevice += t2 - t1;
            _tFeed += System.Diagnostics.Stopwatch.GetTimestamp() - t2;
            _rows += active.Count;
            if (++_steps % 500 == 0)
            {
                double us(long t) => t * 1e6 / System.Diagnostics.Stopwatch.Frequency / _steps;
                Console.Error.WriteLine($"batch: {_steps} steps, {(double)_rows / _steps:F1} rows/step; per step choose {us(_tChoose):F0} us, " +
                    $"device {us(_tDevice):F0} us, feed {us(_tFeed):F0} us");
                _tChoose = _tDevice = _tFeed = _steps = _rows = 0;
            }
        }
    }

    /// <summary>
    /// One speculative step for a reply running alone: the drafter proposes
    /// tokens to follow the one chosen, the model runs them all as rows of one
    /// pass, and the reply takes each row as a plain step would (adapters, the
    /// sampler and every constraint, in order) for as long as what it chooses
    /// is the next draft. Its first other choice is the next step's token. The
    /// reply is exactly what plain steps give, sampled or greedy: a draft only
    /// saves a pass when the reply would have chosen it anyway. False when the
    /// reply is over.
    /// </summary>
    private bool Speculate(Job job)
    {
        var pending = job.Pending;
        job.Pending = null;
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        if ((pending ?? job.Run.Next()) is not { } token)
        {
            return false;
        }
        var seq = job.Seq!;
        job.Pace ??= new DraftPace(Drafts);
        job.Context ??= new List<int>(job.Run.Prompt);
        job.Context.Add(token);
        var suffix = SuffixDraft.Find(job.Context, SuffixMatch, GpuEngine.MaxDrafts);
        int[] rows = Device(() =>
        {
            var drafts = suffix ?? _engine.Draft(seq, token, DraftsSet ?? job.Pace!.Next);
            if (suffix is null && DraftsSet is null && drafts.Length > 0)
            {
                drafts = drafts[..Math.Min(drafts.Length, _engine.WorthVerifying(DraftPace.Floor))];
            }
            var all = new int[drafts.Length + 1];
            all[0] = token;
            drafts.CopyTo(all, 1);
            long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
            _tDraft += t1 - t0;
            _engine.Verify(seq, all);
            _tVerify += System.Diagnostics.Stopwatch.GetTimestamp() - t1;
            return all;
        });
        long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
        int kept = 0;
        bool going = true;
        while (true)
        {
            int row = kept;
            job.Row = row;
            var verdict = EntailVerdict.Merge(job.Adapters.Select(a => a.Step(rows[row])).ToList());
            job.Run.Fed(() => _engine.HostLogits(job.Row), verdict);
            kept++;
            if (job.Run.Next() is not { } choice)
            {
                going = false;
                break;
            }
            if (kept == rows.Length || choice != rows[kept])
            {
                job.Pending = choice;
                break;
            }
            job.Context.Add(choice);
        }
        Device(() =>
        {
            _engine.Accept(seq, kept);
            return 0;
        });
        _tSpecFeed += System.Diagnostics.Stopwatch.GetTimestamp() - t2;
        _specCycles++;
        _specDrafted += rows.Length - 1;
        _specKept += kept - 1;
        _specTokens += kept;
        if (suffix is null)
        {
            job.Pace.Saw(rows.Length - 1, kept - 1);
        }
        else
        {
            _suffixCycles++;
            _suffixDrafted += rows.Length - 1;
            _suffixKept += kept - 1;
        }
        if (Timing && _specCycles % 200 == 0)
        {
            double ms(long t) => t * 1e3 / System.Diagnostics.Stopwatch.Frequency / _specCycles;
            Console.Error.WriteLine($"speculative: {_specCycles} steps, {(double)_specTokens / _specCycles:F2} tokens/step, " +
                $"{100.0 * _specKept / Math.Max(1, _specDrafted):F0}% of drafts kept; per step draft {ms(_tDraft):F2} ms, " +
                $"verify {ms(_tVerify):F2} ms, choose {ms(_tSpecFeed):F2} ms; {_suffixCycles} from the context, " +
                $"{100.0 * _suffixKept / Math.Max(1, _suffixDrafted):F0}% of theirs kept");
            _specCycles = _specDrafted = _specKept = _specTokens = _suffixCycles = _suffixDrafted = _suffixKept = 0;
            _tDraft = _tVerify = _tSpecFeed = 0;
        }
        return going;
    }

    /// <summary>Gives a waiting reply a sequence and prefills its prompt (and,
    /// for an adapter that weighs tokens against the prior, the blank prompt first).</summary>
    private void Admit(Job job, List<Job> active)
    {
        try
        {
            int[] prompt = job.Run.Prompt;
            int mark = MarkFor(prompt);
            // How far the conversation may get this turn: a model with a context extension
            // keeps its trained frequencies for one that stays inside its trained length.
            long plannedLength = (long)prompt.Length + Math.Min(job.Run.Budget, 16384);
            int reused = 0;
            bool fromDisk = false;
            var paged = job.Paged;
            job.Paged = null;
            job.Seq = Device(() =>
            {
                if (Timing)
                {
                    foreach (var kept in _free)
                    {
                        int common = _held.TryGetValue(kept, out var h) ? h.AsSpan().CommonPrefixLength(prompt) : -1;
                        Console.Error.WriteLine($"reuse: kept sequence marked at {kept.MarkedAt}, agrees for {common} tokens");
                    }
                }
                var seq = PrefixReuse ? Reusable(prompt, _engine.Extends(plannedLength)) : null;
                if (paged is { } p && (seq is null || p.Mark > seq.MarkedAt))
                {
                    try
                    {
                        var restored = _engine.RestoreMark(p.State);
                        if (restored.MarkedAt != p.Mark)
                        {
                            _engine.Release(restored);
                            throw new InvalidDataException($"the state was saved at {restored.MarkedAt}, not {p.Mark}");
                        }
                        if (restored.Extended != _engine.Extends(plannedLength))
                        {
                            _engine.Release(restored);
                            throw new InvalidDataException("the conversation has passed the model's trained length since it was saved");
                        }
                        seq = restored;
                        reused = seq.Position;
                        fromDisk = true;
                        paged = null;
                    }
                    catch (Exception e) when (e is InvalidDataException or EndOfStreamException or InvalidOperationException)
                    {
                        _log?.Invoke($"kv: could not restore a paged-out state ({e.Message}); prefilling");
                        paged = null;
                    }
                }
                if (!fromDisk && seq is not null)
                {
                    _free.Remove(seq);
                    _usedAt.Remove(seq);
                    _engine.Rewind(seq);
                    reused = seq.Position;
                }
                else if (!fromDisk)
                {
                    seq = _engine.NewSequence();
                    _engine.SetExactRows(seq, ExactFor(prompt), _paging.ExactF16);
                }
                _held.Remove(seq!);
                return seq!;
            });
            if (paged is { } unused && _store is { } store)
            {
                // A kept sequence went further: the state read from disk goes back.
                store.Put(prompt[..unused.Mark], unused.State.View(), DateTime.UtcNow);
            }
            job.Adapters = _adapters.Select(a => a.Sibling()).ToArray();
            double[]? prior = null;
            if (job.Adapters.Any(a => a.Manifest.NeedsPrior) && _priorPrompt?.Invoke(prompt) is { } blank)
            {
                // On a sequence of its own, so the reply's sequence keeps what it reused.
                prior = Softmax(Device(() =>
                {
                    _priorSeq ??= _engine.NewSequence();
                    _engine.Reset(_priorSeq);
                    return _engine.Prefill(_priorSeq, blank);
                }));
            }
            long a0 = System.Diagnostics.Stopwatch.GetTimestamp();
            var seqNow = job.Seq;
            Reserve(seqNow, prompt.Length + 1, plannedLength);
            seqNow.Images = Spans(job.Run, reused);
            // Up to the mark, save the state there, then the rest of the prompt.
            if (PrefixReuse && mark > reused)
            {
                Read(job, seqNow, prompt, mark, active);
                Device(() =>
                {
                    _engine.Mark(seqNow);
                    return 0;
                });
            }
            var logits = Read(job, seqNow, prompt, prompt.Length, active);
            seqNow.Images = null;
            _held[seqNow] = prompt;
            ReusedTokens += reused;
            PrefilledTokens += prompt.Length - reused;
            if (fromDisk)
            {
                RestoredTokens += reused;
                _log?.Invoke($"kv: restored {reused} tokens from disk, prefilled {prompt.Length - reused}");
            }
            long a1 = System.Diagnostics.Stopwatch.GetTimestamp();
            var verdict = EntailVerdict.Merge(job.Adapters.Select(a =>
            {
                a.Reset();
                return a.Prefill(prompt);
            }).ToList());
            if (Timing)
            {
                double ms(long t) => t * 1e3 / System.Diagnostics.Stopwatch.Frequency;
                Console.Error.WriteLine($"admit: {prompt.Length} prompt tokens, {reused} reused, prefill {ms(a1 - a0):F0} ms, adapters {ms(System.Diagnostics.Stopwatch.GetTimestamp() - a1):F0} ms");
            }
            job.Run.Begin(() => logits, verdict, prior);
            active.Add(job);
        }
        catch (Exception e)
        {
            if (job.Seq is { } failed)
            {
                _held.Remove(failed);
            }
            Release(job);
            job.Done.TrySetException(e);
        }
    }

    /// <summary>Tokens of a long prompt read before the running replies get a
    /// turn (whole prompt chunks, so reading in slices changes nothing).</summary>
    private const int ReadSlice = 4096;

    /// <summary>How long the running replies generate between two slices of a long prompt.</summary>
    private const double TurnMilliseconds = 250;

    /// <summary>Reads the prompt up to <paramref name="to"/>, in slices while
    /// other replies are running so that they go on between them, and gives up
    /// if the request is cancelled.</summary>
    private float[] Read(Job job, GpuSequence seq, int[] prompt, int to, List<Job> active)
    {
        while (true)
        {
            int at = seq.Position;
            // Slices end where the chunks of one read of the whole would, and cut no image.
            int n = to - at <= ReadSlice ? to - at : ReadSlice;
            if (seq.Images?.FirstOrDefault(i => i.Start > at && i.Start < at + n && i.Start + i.Tokens > at + n) is { } cut)
            {
                n = cut.Start - at;
            }
            var logits = Device(() => _engine.Prefill(seq, prompt.AsSpan(at, n)));
            if (seq.Position >= to)
            {
                return logits;
            }
            if (job.Run.Cancelled)
            {
                throw new OperationCanceledException("the request was cancelled while its prompt was read");
            }
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            while (active.Count > 0 && System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds < TurnMilliseconds)
            {
                try
                {
                    StepAll(active);
                }
                catch (Exception e)
                {
                    FailAll(active, e);
                }
            }
        }
    }

    /// <summary>Makes the sequence's cache long enough for the whole prompt at
    /// once. Without the memory for it, kept sequences leave the device, least
    /// recently used first, before the request fails.</summary>
    private void Reserve(GpuSequence seq, int positions, long planned)
    {
        while (true)
        {
            try
            {
                Device(() =>
                {
                    _engine.Plan(seq, planned);
                    _engine.Reserve(seq, positions);
                    return 0;
                });
                return;
            }
            catch (InvalidOperationException) when (_free.Count > 0)
            {
                Evict(_free[0], "making room for a longer prompt");
            }
        }
    }

    private int MarkFor(int[] prompt) => PromptMark.At(prompt, _turnStart, _resultStart, _replyStart);

    /// <summary>How many of a new conversation's first positions stay f32 in an NVFP4 cache.</summary>
    private int ExactFor(int[] prompt) => PromptMark.Exact(prompt, _paging.ExactPrefix, _turnStart, _userRole);

    /// <summary>The kept sequence that the prompt can rewind into: its mark is
    /// short of the prompt's end and the prompt agrees with what it held up to
    /// there. The latest mark wins.</summary>
    private GpuSequence? Reusable(int[] prompt, bool extended)
    {
        GpuSequence? best = null;
        foreach (var seq in _free)
        {
            int at = seq.MarkedAt;
            if (at > 0 && at < prompt.Length && at > (best?.MarkedAt ?? 0) && seq.Extended == extended && _held.TryGetValue(seq, out var held)
                && held.Length >= at && held.AsSpan(0, at).SequenceEqual(prompt.AsSpan(0, at)))
            {
                best = seq;
            }
        }
        return best;
    }

    private void Complete(Job job)
    {
        Release(job);
        job.Done.TrySetResult(job.Run.Result!);
    }

    private void Release(Job job)
    {
        foreach (var a in job.Adapters)
        {
            a.Dispose();
        }
        job.Adapters = Array.Empty<IEntailAdapter>();
        if (job.Seq is { } seq)
        {
            job.Seq = null;
            if (PrefixReuse && _held.ContainsKey(seq) && seq.MarkedAt > 0)
            {
                // Kept for the next requests while it fits the GPU budget.
                _free.Add(seq);
                _usedAt[seq] = DateTime.UtcNow;
                TrimToBudget();
            }
            else
            {
                _held.Remove(seq);
                Device(() =>
                {
                    _engine.Release(seq);
                    return 0;
                });
            }
        }
    }

    /// <summary>Releases kept sequences unused for the idle time. At most once a second.</summary>
    private void Housekeep()
    {
        long now = Environment.TickCount64;
        if (now - _lastHousekeep < 1000 || _paging.IdleSeconds <= 0)
        {
            return;
        }
        _lastHousekeep = now;
        var cutoff = DateTime.UtcNow - TimeSpan.FromSeconds(_paging.IdleSeconds);
        foreach (var seq in _free.Where(s => _usedAt[s] < cutoff).ToList())
        {
            Evict(seq, "idle");
        }
    }

    /// <summary>Releases the least recently used kept sequences until the rest fit the GPU budget.</summary>
    private void TrimToBudget()
    {
        long total = _free.Sum(_engine.DeviceBytes);
        while (_free.Count > 0 && total > _paging.GpuBudgetBytes)
        {
            var oldest = _free[0];
            total -= _engine.DeviceBytes(oldest);
            Evict(oldest, "over the GPU budget");
        }
    }

    /// <summary>Takes a kept sequence out of GPU memory: to disk first when
    /// the model has a disk tier and the state is worth loading back.</summary>
    private void Evict(GpuSequence seq, string why)
    {
        _free.Remove(seq);
        var usedAt = _usedAt.Remove(seq, out var t) ? t : DateTime.UtcNow;
        _held.Remove(seq, out var held);
        long bytes = _engine.DeviceBytes(seq);
        string where = "released";
        try
        {
            if (_store is { } store && held is not null && seq.MarkedAt >= _paging.MinPageTokens && held.Length >= seq.MarkedAt)
            {
                var state = new ChunkedBuffer();
                Device(() =>
                {
                    _engine.SaveMark(seq, state);
                    return 0;
                });
                if (store.Put(held[..seq.MarkedAt], state, usedAt))
                {
                    where = "paged out";
                }
            }
        }
        catch (Exception e)
        {
            _log?.Invoke($"kv: could not page out a sequence: {e.Message}");
        }
        finally
        {
            Device(() =>
            {
                _engine.Release(seq);
                return 0;
            });
            Released++;
        }
        _log?.Invoke($"kv: {where} a sequence ({why}; {seq.MarkedAt} tokens, {bytes / (1024.0 * 1024):F0} MiB of GPU memory); {_free.Count} kept");
    }

    private static double[] Softmax(float[] logits)
    {
        float max = logits.Max();
        var p = new double[logits.Length];
        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            sum += p[i] = Math.Exp(logits[i] - max);
        }
        for (int i = 0; i < p.Length; i++)
        {
            p[i] /= sum;
        }
        return p;
    }

    public void Dispose()
    {
        _waiting.CompleteAdding();
        if (_thread.Join(TimeSpan.FromSeconds(10)))
        {
            // The batch thread has stopped: what is kept goes to disk, for after a restart.
            foreach (var seq in _free.ToList())
            {
                Evict(seq, "shutting down");
            }
        }
        _store?.Dispose();
    }
}

/// <summary>How an embedding container turns hidden states into one vector,
/// from its recorded <c>embedding.pooling</c> facts.</summary>
public sealed record EmbeddingPooling(string Kind, bool L2Normalise)
{
    public static EmbeddingPooling From(JsonElement embedding)
    {
        if (!embedding.TryGetProperty("pooling", out var pooling) || pooling.ValueKind != JsonValueKind.Object)
        {
            throw new ContainerException("the embedding surface records no pooling rule");
        }
        string kind = pooling.TryGetProperty("kind", out var k) ? k.GetString() ?? "" : "";
        if (kind is not ("mean" or "last" or "cls"))
        {
            throw new UnsupportedOperatorException($"embedding pooling '{kind}' is not served (mean, last, cls)");
        }
        bool l2 = pooling.TryGetProperty("l2_normalise", out var n) && n.ValueKind == JsonValueKind.True;
        return new EmbeddingPooling(kind, l2);
    }
}

/// <summary>Embeds text with an embedding container: forward, take every
/// position's post-final-norm hidden state, pool by the recorded rule,
/// optionally L2-normalise.</summary>
public sealed class Embedder
{
    private readonly DecodeSession _session;

    public HfTokenizer Tokenizer { get; }
    public EmbeddingPooling Pooling { get; }
    public int ContextLength { get; }
    public int Dimensions => _session.Runtime.Plan.HiddenSize;

    public Embedder(DecodeSession session, HfTokenizer tokenizer, EmbeddingPooling pooling, int contextLength)
    {
        _session = session;
        Tokenizer = tokenizer;
        Pooling = pooling;
        ContextLength = contextLength;
    }

    /// <summary>Text as HF's <c>tokenizer(text)</c> encodes it (post-processor
    /// template included), truncated to the context.</summary>
    public int[] Encode(string text) => Tokenizer.EncodeForModel(text, ContextLength, out _);

    public void Warm() => Embed(new[] { 0 });

    public float[] Embed(int[] ids)
    {
        if (ids.Length == 0)
        {
            throw new ArgumentException("cannot embed an empty input");
        }
        _session.Reset();
        var hidden = _session.PrefillHiddenStates(ids);
        int h = hidden.Cols;
        var v = new float[h];
        switch (Pooling.Kind)
        {
            case "mean":
                for (int r = 0; r < hidden.Rows; r++)
                {
                    var row = hidden.Row(r);
                    for (int i = 0; i < h; i++)
                    {
                        v[i] += row[i];
                    }
                }
                for (int i = 0; i < h; i++)
                {
                    v[i] /= hidden.Rows;
                }
                break;
            case "last":
                hidden.Row(hidden.Rows - 1).CopyTo(v);
                break;
            case "cls":
                hidden.Row(0).CopyTo(v);
                break;
        }
        if (Pooling.L2Normalise)
        {
            double norm = Math.Sqrt(v.Sum(x => (double)x * x));
            if (norm > 0)
            {
                for (int i = 0; i < h; i++)
                {
                    v[i] = (float)(v[i] / norm);
                }
            }
        }
        return v;
    }
}
