using System.Text;
using System.Text.Json;
using Amql.Cli;
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
    public SemaphoreSlim Gate { get; } = new(1, 1);

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
    }

    public static ModelHost Load(string id, string path)
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
            var generator = new TextGenerator(new DecodeSession(plan, store), tokenizer, template,
                StopTokens.Resolve(path, tokenizer, template), (int)(surface?.ContextLength ?? int.MaxValue));
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
        _store?.Dispose();
        _container.Dispose();
        Gate.Dispose();
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

public sealed record GenerationParams(
    int MaxTokens,
    float Temperature,
    float TopP,
    int Seed,
    IReadOnlyList<string> Stop);

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
/// so a character split across tokens is emitted once, whole.
/// </summary>
public sealed class TextGenerator
{
    private readonly DecodeSession _session;
    private readonly IReadOnlySet<int> _stopTokens;

    public HfTokenizer Tokenizer { get; }
    public string Template { get; }
    public int ContextLength { get; }

    public TextGenerator(DecodeSession session, HfTokenizer tokenizer, string template, IReadOnlySet<int> stopTokens, int contextLength)
    {
        _session = session;
        Tokenizer = tokenizer;
        Template = template;
        _stopTokens = stopTokens;
        ContextLength = contextLength;
    }

    /// <summary>The prompt as token ids. The rendered template already carries
    /// every special token it needs, so no post-processor template is added —
    /// HF tokenises chat prompts with <c>add_special_tokens=False</c>.</summary>
    public int[] EncodePrompt(IReadOnlyList<ChatMessage> messages, bool enableThinking) =>
        Tokenizer.EncodeToIds(ChatTemplate.ApplyMessages(Template, messages, enableThinking)).ToArray();

    public void Warm()
    {
        _session.Reset();
        _session.Prefill(new[] { 0 });
        _session.Reset();
    }

    public GenerationResult Generate(int[] prompt, GenerationParams p, Action<string>? onDelta,
        CancellationToken cancel, int? logprobsTopK = null)
    {
        if (prompt.Length == 0)
        {
            throw new ArgumentException("the prompt renders to no tokens");
        }
        int budget = Math.Min(p.MaxTokens, ContextLength - prompt.Length);
        if (budget <= 0)
        {
            throw new ArgumentException($"the prompt is {prompt.Length} tokens, which fills the {ContextLength}-token context");
        }

        var config = new SamplingConfig(p.Seed, p.Temperature, TopK: 0, TopP: p.TopP >= 1f ? 0f : p.TopP);
        var rng = new Random(p.Seed);
        _session.Reset();
        var logits = _session.Prefill(prompt);

        var generated = new List<int>();
        var logprobsList = logprobsTopK is { } lpTk && lpTk > 0
            ? new List<LogprobsResult>()
            : null;
        string emitted = string.Empty;
        string finish = "length";
        int holdBack = p.Stop.Count == 0 ? 0 : p.Stop.Max(s => s.Length) - 1;

        while (generated.Count < budget && !cancel.IsCancellationRequested)
        {
            int token;
            LogprobsResult? stepLp = null;

            if (logprobsTopK is { } lpTk2 && lpTk2 > 0)
            {
                (token, stepLp) = Sampler.SampleWithLogprobs(logits, config, rng, lpTk2);
                logprobsList!.Add(stepLp!);
            }
            else
            {
                token = Sampler.Sample(logits, config, rng);
            }
            if (_stopTokens.Contains(token))
            {
                finish = "stop";
                break;
            }
            generated.Add(token);

            string text = Tokenizer.DecodeText(generated);
            int stopAt = FirstStop(text, p.Stop);
            if (stopAt >= 0)
            {
                Emit(text[..stopAt]);
                finish = "stop";
                emitted = text[..stopAt];
                return BuildResult(emitted, finish, prompt.Length, generated, logprobsList);
            }
            // Hold back a trailing partial character, and enough text that a
            // stop string straddling the next token is never half-sent.
            int safe = text.Length;
            while (safe > 0 && text[safe - 1] == '�')
            {
                safe--;
            }
            safe = Math.Max(emitted.Length, safe - holdBack);
            Emit(text[..safe]);

            if (generated.Count < budget)
            {
                logits = _session.Step(token);
            }
        }

        string final = Tokenizer.DecodeText(generated);
        Emit(final);
        return BuildResult(final, cancel.IsCancellationRequested ? "cancelled" : finish, prompt.Length, generated, logprobsList);

        void Emit(string upTo)
        {
            if (upTo.Length > emitted.Length && upTo.StartsWith(emitted, StringComparison.Ordinal))
            {
                onDelta?.Invoke(upTo[emitted.Length..]);
                emitted = upTo;
            }
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

    private static GenerationResult BuildResult(string text, string finish, int promptTokens,
        List<int> generated, List<LogprobsResult>? logprobsList)
    {
        float? perplexity = null;
        float? meanEntropy = null;
        if (logprobsList is { Count: > 0 })
        {
            double nllSum = logprobsList.Sum(lp => -(double)lp.TokenLogprob);
            perplexity = (float)Math.Exp(nllSum / logprobsList.Count);
            meanEntropy = logprobsList.Average(lp => lp.Entropy);
        }
        return new GenerationResult(text, finish, promptTokens, generated.Count)
        {
            Tokens = generated,
            Logprobs = logprobsList,
            Perplexity = perplexity,
            MeanEntropy = meanEntropy,
        };
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
