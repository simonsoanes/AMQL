using System.Text;
using System.Text.Json;
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
    public static ModelHost Load(string id, string path, IReadOnlyList<string>? adapters = null, ModelDefaults? defaults = null)
    {
        var host = LoadEngines(id, path, adapters);
        return new ModelHost(host, defaults ?? new ModelDefaults(),
            (adapters ?? Array.Empty<string>()).Select(d => System.IO.Path.GetFileName(System.IO.Path.GetFullPath(d).TrimEnd('/', '\\'))).ToList());
    }

    private ModelHost(ModelHost loaded, ModelDefaults defaults, IReadOnlyList<string> adapterNames)
        : this(loaded.Id, loaded.Path, loaded._container, loaded._store, loaded.Decisions, loaded.Generator, loaded.Embedder,
            loaded.Classifier, loaded.Notes.ToList())
    {
        Defaults = defaults;
        AdapterNames = loaded.Generator is null ? Array.Empty<string>() : adapterNames;
    }

    private static ModelHost LoadEngines(string id, string path, IReadOnlyList<string>? adapters)
    {
        var container = Vindex3Container.Open(path);
        var notes = new List<string>();
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
            string? template = ChatTemplate.Load(path);
            if (template is null)
            {
                notes.Add("no chat template (chat_template.jinja / tokenizer_config.json): chat endpoints need one");
                return new ModelHost(id, path, container, store, null, null, null, null, notes);
            }
            try
            {
                ChatTemplate.ApplyMessages(template, new[] { new ChatMessage("user", "probe") });
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
    public static IReadOnlySet<int> Resolve(string dir, HfTokenizer tokenizer, string template)
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
        if (template.Contains("<|im_end|>"))
        {
            AddContent("<|im_end|>");
        }
        if (template.Contains("<|eot_id|>"))
        {
            AddContent("<|eot_id|>");
        }
        return ids;
    }
}

/// <summary>Defaults one model applies to requests that leave them out
/// (null: the server's default).</summary>
public sealed record ModelDefaults(bool? Think = null, int? MaxTokens = null, float? Temperature = null, float? TopP = null, int? TopK = null);

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

    /// <summary>The prompt ended inside an open think block.</summary>
    public bool Thinking { get; init; }

    /// <summary>Structured output: the reply must be a JSON value this schema allows.</summary>
    public JsonSchemaNode? Json { get; init; }

    /// <summary>The request's tools (chat form): a call the reply opens is held to them.</summary>
    public IReadOnlyList<System.Text.Json.JsonElement>? Tools { get; init; }

    /// <summary>The user's latest message and the tool results since, for the mentioned-parameter rule.</summary>
    public string? Recent { get; init; }
}

public sealed record GenerationResult(string Text, string FinishReason, int PromptTokens, int CompletionTokens)
{
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
    public string Template { get; }
    public int ContextLength { get; }
    internal int? ThinkClose { get; }
    internal IReadOnlySet<int> StopTokenIds => _stopTokens;

    private string?[]? _tokenTexts;

    /// <summary>A token's own text, for structured output to check; empty for a special token
    /// (it would vanish from the decoded reply, so it is never allowed there) or a padding row.</summary>
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
    public bool RendersTools => ChatTemplate.RendersTools(Template);

    /// <summary>Whether the template has a thinking switch.</summary>
    public bool Thinks => Template.Contains("enable_thinking", StringComparison.Ordinal);

    /// <summary>Requests that can generate at once: the batch size on the GPU engine, else 1.</summary>
    public int Concurrency => _batch is null ? 1 : GpuEngine.MaxBatch;

    /// <summary>Why requests are not batched, when they are not.</summary>
    public string? BatchRefusal { get; }

    public TextGenerator(DecodeSession session, HfTokenizer tokenizer, string template, IReadOnlySet<int> stopTokens, int contextLength)
        : this(session, Array.Empty<IEntailAdapter>(), null, tokenizer, template, stopTokens, contextLength)
    {
    }

    public TextGenerator(DecodeSession partner, IReadOnlyList<IEntailAdapter> adapters, Func<int[], int[]?>? priorPrompt,
        HfTokenizer tokenizer, string template, IReadOnlySet<int> stopTokens, int contextLength)
    {
        _session = new SideBySideSession(partner, adapters) { PriorPrompt = priorPrompt };
        Adapters = adapters;
        Tokenizer = tokenizer;
        Template = template;
        _stopTokens = stopTokens;
        ContextLength = contextLength;
        ThinkClose = tokenizer.AddedTokenId("</think>");

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
        Tokenizer.EncodeToIds(ChatTemplate.ApplyMessages(Template, messages, enableThinking, tools)).ToArray();

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

    /// <summary>The loop the tail of <paramref name="reply"/> is in, looking no further back
    /// than <paramref name="from"/>; null when it is in none.</summary>
    public static (int Period, int Repeats)? Find(IReadOnlyList<int> reply, int from = 0)
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
    private readonly int? _logprobsTopK;
    private readonly SamplingConfig _config;
    private readonly Random _rng;
    private readonly int _budget;
    private readonly int _holdBack;
    private readonly Queue<int> _forced = new();
    private readonly List<int> _generated = new();
    private readonly List<LogprobsResult>? _logprobs;
    private bool _forcePending;
    private string _emitted = string.Empty;
    private readonly JsonConstraint? _json;
    private readonly ToolGrammar? _toolGrammar;
    private ToolCallConstraint? _call;
    private int _openMatch;   // how much of "<tool_call>" the reply's text ends with
    private bool _thinkClosed;
    private int _loopFrom;   // the loop guard looks only at tokens from here on
    private float[]? _masked;
    private LogitsRow _logits = null!;
    private EntailVerdict _verdict = EntailVerdict.Silence;
    private IReadOnlyList<double>? _prior;

    public int[] Prompt { get; }

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
        if (p.Tools is { Count: > 0 } tools && TextGenerator.ToolGrammarEnabled)
        {
            _toolGrammar = ToolGrammar.Compile(tools);
        }
        // Forced text goes in after the thinking block, or at once without one.
        _forcePending = p.Forced is not null;
        if (_forcePending && !p.Thinking)
        {
            Enqueue(p.Forced!);
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
        else if (_verdict.Silent)
        {
            LogprobsResult? lp;
            (token, lp) = Sampler.SampleWithLogprobs(_logits(), _config, _rng, _logprobsTopK);
            if (lp is not null)
            {
                _logprobs!.Add(lp);
            }
        }
        else
        {
            // An adapter rules on this token; logprobs still describe the model's own distribution.
            (token, _) = _verdict.Choose(_logits(), _prior, r => Sampler.Sample(r, _config, _rng));
            if (_logprobs is not null && Sampler.SampleWithLogprobs(_logits(), _config, new Random(0), _logprobsTopK).Logprobs is { } lp)
            {
                _logprobs.Add(lp);
            }
        }
        if (_g.StopTokenIds.Contains(token))
        {
            return Finish("stop");
        }
        string piece = _g.TokenText(token, _logits().Length);
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
        if (token == _g.ThinkClose)
        {
            _thinkClosed = true;
        }
        if (_forcePending && _p.Thinking && token == _g.ThinkClose)
        {
            Enqueue("\n\n" + _p.Forced);
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
        if (TextGenerator.LoopGuardEnabled && _json is null && _generated.Count % 8 == 0 && LoopGuard.Find(_generated, _loopFrom) is { } loop)
        {
            if (_p.Thinking && !_thinkClosed && _g.ThinkClose is not null && _forced.Count == 0)
            {
                // Thinking that circles is closed, and the answer goes on from there.
                Console.Error.WriteLine($"{DateTime.Now:HH:mm:ss.fff} loop guard: thinking repeats a {loop.Period}-token stretch {loop.Repeats} times; closing it");
                foreach (int id in _g.Tokenizer.EncodeToIds("\n</think>\n\n"))
                {
                    _forced.Enqueue(id);
                }
                _loopFrom = _generated.Count;
            }
            else
            {
                Console.Error.WriteLine($"{DateTime.Now:HH:mm:ss.fff} loop guard: the reply repeats a {loop.Period}-token stretch {loop.Repeats} times; " +
                                        $"stopped at {_generated.Count} tokens");
                return Finish("length");
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
        string text = _stableText + _g.Tokenizer.DecodeText(tail);
        // A tail ending mid-character waits for its next token; one that never
        // completes (invalid bytes) is kept as decoded after 64 tokens.
        if (tail.Count > 0 && (text.Length == 0 || text[^1] != '�' || tail.Count > 64))
        {
            _stableText = text;
            _stableTokens = _generated.Count;
        }
        return text;
    }

    private const string CallOpen = "<tool_call>";

    /// <summary>The reply opens a call when its text, outside thinking, reaches "&lt;tool_call&gt;":
    /// from there to "&lt;/tool_call&gt;" it is held to the tools. What the same token says after
    /// the opening goes to the call; if that already breaks it, the call is left unheld.</summary>
    private void WatchForCall(string piece)
    {
        for (int i = 0; i < piece.Length; i++)
        {
            char c = piece[i];
            _openMatch = c == CallOpen[_openMatch] ? _openMatch + 1 : c == CallOpen[0] ? 1 : 0;
            if (_openMatch < CallOpen.Length)
            {
                continue;
            }
            _openMatch = 0;
            var call = new ToolCallConstraint(_toolGrammar!);
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
        var row = _logits();
        int n = row.Length;
        _masked ??= new float[n];
        Array.Fill(_masked, float.NegativeInfinity);
        Func<string, bool>? onPath = grammar is ToolCallConstraint call ? MentionedPath(call) : null;
        bool Valid(int id) => _g.StopTokenIds.Contains(id)
            ? grammar.CanEnd && onPath is null
            : _g.TokenText(id, n) is { Length: > 0 } t && grammar.Accepts(t) && (onPath is null || onPath(t));

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
    private Func<string, bool>? MentionedPath(ToolCallConstraint call)
    {
        if (!TextGenerator.MentionRuleEnabled)
        {
            return null;
        }
        string context = (_p.Recent ?? "") + "\n" + Reasoning();
        foreach (string name in call.OptionalLeft())
        {
            if (call.PathTo(name) is not { } path ||
                !System.Text.RegularExpressions.Regex.IsMatch(context, $@"(?<![\w-]){System.Text.RegularExpressions.Regex.Escape(name)}(?![\w-])"))
            {
                continue;
            }
            // Whitespace may come first; then the text must stay on the way to "<parameter=NAME>".
            bool OnPath(string t)
            {
                string rest = t.TrimStart(' ', '\n', '\r', '\t');
                return rest.Length == 0 || path.StartsWith(rest, StringComparison.Ordinal) || rest.StartsWith(path, StringComparison.Ordinal);
            }
            var row = _logits();
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
                if (e > 1e-12 && !_g.StopTokenIds.Contains(id) && _g.TokenText(id, row.Length) is { Length: > 0 } t && OnPath(t) && t.Trim().Length > 0)
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
    }

    private readonly GpuEngine _engine;
    private readonly TextGenerator _g;
    private readonly IReadOnlyList<IEntailAdapter> _adapters;
    private readonly Func<int[], int[]?>? _priorPrompt;
    private readonly System.Collections.Concurrent.BlockingCollection<Job> _waiting = new();
    private readonly Stack<GpuSequence> _free = new();
    private readonly Thread _thread;

    // AMQL_SERVER_TIMING=1: where a batch step's time goes, printed every 500 steps.
    private static readonly bool Timing = Environment.GetEnvironmentVariable("AMQL_SERVER_TIMING") is "1";
    private long _tChoose, _tDevice, _tFeed, _steps, _rows;

    public BatchScheduler(GpuEngine engine, TextGenerator g, IReadOnlyList<IEntailAdapter> adapters, Func<int[], int[]?>? priorPrompt)
    {
        _engine = engine;
        _g = g;
        _adapters = adapters;
        _priorPrompt = priorPrompt;
        _thread = new Thread(Loop) { IsBackground = true, Name = "amql batch" };
        _thread.Start();
    }

    /// <summary>Queues a reply and waits for it.</summary>
    public GenerationResult Run(GenerationRun run)
    {
        var job = new Job(run);
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
                if (active.Count == 0 && _waiting.TryTake(out var first, Timeout.Infinite))
                {
                    Admit(first, active);
                }
                while (active.Count < GpuEngine.MaxBatch && _waiting.TryTake(out var waiting))
                {
                    Admit(waiting, active);
                }

                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                // Each reply picks its next token on its own core: sampling reads the
                // whole vocabulary, and the replies share nothing but the device.
                var next = new int?[active.Count];
                if (active.Count > 1)
                {
                    Parallel.For(0, active.Count, i => next[i] = active[i].Run.Next());
                }
                else
                {
                    next[0] = active[0].Run.Next();
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
                    continue;
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
            catch (InvalidOperationException) when (_waiting.IsAddingCompleted)
            {
                break;
            }
            catch (Exception e)
            {
                // The device failed under the batch: every running reply ends with the error.
                foreach (var job in active)
                {
                    job.Run.Abort("error");
                    Release(job);
                    job.Done.TrySetException(e);
                }
                active.Clear();
            }
        }
        foreach (var job in active)
        {
            job.Run.Abort("cancelled");
            Complete(job);
        }
    }

    /// <summary>Gives a waiting reply a sequence and prefills its prompt (and,
    /// for an adapter that weighs tokens against the prior, the blank prompt first).</summary>
    private void Admit(Job job, List<Job> active)
    {
        try
        {
            job.Seq = Device(() =>
            {
                var seq = _free.Count > 0 ? _free.Pop() : _engine.NewSequence();
                _engine.Reset(seq);
                return seq;
            });
            job.Adapters = _adapters.Select(a => a.Sibling()).ToArray();
            int[] prompt = job.Run.Prompt;
            double[]? prior = null;
            if (job.Adapters.Any(a => a.Manifest.NeedsPrior) && _priorPrompt?.Invoke(prompt) is { } blank)
            {
                prior = Softmax(Device(() =>
                {
                    var l = _engine.Prefill(job.Seq, blank);
                    _engine.Reset(job.Seq);
                    return l;
                }));
            }
            var logits = Device(() => _engine.Prefill(job.Seq, prompt));
            var verdict = EntailVerdict.Merge(job.Adapters.Select(a =>
            {
                a.Reset();
                return a.Prefill(prompt);
            }).ToList());
            job.Run.Begin(() => logits, verdict, prior);
            active.Add(job);
        }
        catch (Exception e)
        {
            Release(job);
            job.Done.TrySetException(e);
        }
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
            // Keep a batch's worth of sequences for the next requests; free the rest.
            if (_free.Count < GpuEngine.MaxBatch)
            {
                _free.Push(seq);
            }
            else
            {
                Device(() =>
                {
                    _engine.Release(seq);
                    return 0;
                });
            }
        }
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
        _thread.Join(TimeSpan.FromSeconds(10));
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
