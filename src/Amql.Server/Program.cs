using Amql.Server;

// amql-server --model [id=]<container-dir> [--model …] [--urls http://127.0.0.1:8000]
//             [--api-key KEY | AMQL_API_KEY] [--max-tokens N] [--no-warm]
//             [--adapter [id=]<dir>[,<dir>…]] [--think] [--temperature T] [--top-p P] [--top-k K]
//             [--config <models.json>]
using System.Globalization;
using System.Text.Json;

var containers = new List<(string Id, string Path, List<string> Adapters, ModelDefaults Defaults)>();
var adapterSpecs = new List<(string? Id, string[] Dirs)>();
string urls = "http://127.0.0.1:8000";
string? apiKey = Environment.GetEnvironmentVariable("AMQL_API_KEY");
int? maxTokens = null;
bool warm = true, think = false;
float? temperature = null, topP = null;
int? topK = null;
var ngramPacks = new Dictionary<string, Amql.Inference.NGram.NGramPack>();
var ngram = new Amql.Inference.NGram.NGramPrior(Array.Empty<Amql.Inference.NGram.NGramPack>());
double Num(string s) => double.Parse(s, CultureInfo.InvariantCulture);
for (int i = 0; i < args.Length; i++)
{
    string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value");
    switch (args[i])
    {
        case "--model":
        {
            string spec = Next();
            int eq = spec.IndexOf('=');
            string path = eq > 0 ? spec[(eq + 1)..] : spec;
            string id = eq > 0 ? spec[..eq] : Path.GetFileName(Path.GetFullPath(path).TrimEnd('/', '\\'));
            containers.Add((id, path, new List<string>(), new ModelDefaults()));
            break;
        }
        case "--config":
            try
            {
                containers.AddRange(ServerConfig.Read(Next()).Select(m => (m.Id, m.Path, m.Adapters.ToList(), m.Defaults)));
            }
            catch (Exception e) when (e is JsonException or InvalidDataException or IOException)
            {
                Console.Error.WriteLine($"--config: {e.Message}");
                return 2;
            }
            break;
        case "--urls": urls = Next(); break;
        case "--api-key": apiKey = Next(); break;
        case "--max-tokens": maxTokens = int.Parse(Next()); break;
        case "--no-warm": warm = false; break;
        case "--no-batch": TextGenerator.BatchingEnabled = false; break;
        case "--no-prefix-reuse": BatchScheduler.PrefixReuse = false; break;
        case "--no-loop-guard": TextGenerator.LoopGuardEnabled = false; break;
        case "--free-tool-calls": TextGenerator.ToolGrammarEnabled = false; break;
        case "--no-mention-rule": TextGenerator.MentionRuleEnabled = false; break;
        case "--no-announce-rule": GenerationRun.AnnounceRuleEnabled = false; break;
        case "--allow-repeated-calls": ServerApp.RefuseStaleCalls = false; break;
        case "--keep-line-marks": ServerApp.PlainToolResults = false; break;
        case "--ngram-pack":
        {
            var pack = Amql.Inference.NGram.NGramPack.Load(Next());
            ngramPacks[pack.Name] = pack;
            break;
        }
        case "--ngram-lambda": ngram = new(ngram.Packs) { Lambda = Num(Next()), HalfCount = ngram.HalfCount, MinContext = ngram.MinContext, ModelConfident = ngram.ModelConfident, PackConfident = ngram.PackConfident }; break;
        case "--ngram-min-context": ngram = new(ngram.Packs) { Lambda = ngram.Lambda, HalfCount = ngram.HalfCount, MinContext = int.Parse(Next()), ModelConfident = ngram.ModelConfident, PackConfident = ngram.PackConfident }; break;
        case "--ngram-model-confident": ngram = new(ngram.Packs) { Lambda = ngram.Lambda, HalfCount = ngram.HalfCount, MinContext = ngram.MinContext, ModelConfident = Num(Next()), PackConfident = ngram.PackConfident }; break;
        case "--ngram-members-only": ngram = new(ngram.Packs) { Lambda = ngram.Lambda, HalfCount = ngram.HalfCount, MinContext = ngram.MinContext, ModelConfident = ngram.ModelConfident, PackConfident = ngram.PackConfident, MembersOnly = _ => "" }; break;
        case "--ngram-pack-confident": ngram = new(ngram.Packs) { Lambda = ngram.Lambda, HalfCount = ngram.HalfCount, MinContext = ngram.MinContext, ModelConfident = ngram.ModelConfident, PackConfident = Num(Next()) }; break;
        case "--batch-bf16":
            // Batches of 4 or more run their projections on tensor cores (activations
            // rounded to BF16), as prompts already do: ~1.6x the throughput, but a reply
            // then depends slightly on what shares its batch.
            Environment.SetEnvironmentVariable("AMQE_GEMM_FROM", "4");
            break;
        case "--think": think = true; break;
        case "--temperature": temperature = float.Parse(Next(), CultureInfo.InvariantCulture); break;
        case "--top-p": topP = float.Parse(Next(), CultureInfo.InvariantCulture); break;
        case "--top-k": topK = int.Parse(Next()); break;
        case "--adapter":
        {
            string spec = Next();
            int eq = spec.IndexOf('=');
            // "id=dir" names a model; a bare list goes to every chat model.
            bool named = eq > 0;
            adapterSpecs.Add((named ? spec[..eq] : null, (named ? spec[(eq + 1)..] : spec).Split(',', StringSplitOptions.RemoveEmptyEntries)));
            break;
        }
        case "--help" or "-h":
            Console.WriteLine("""
                amql-server --model [id=]<container-dir> [--model …] [--urls http://127.0.0.1:8000]
                            [--api-key KEY] [--max-tokens N] [--no-warm]
                            [--adapter [id=]<dir>[,<dir>…]] [--think] [--temperature T] [--top-p P] [--top-k K]
                            [--config <models.json>] [--no-batch] [--no-prefix-reuse] [--batch-bf16] [--no-loop-guard] [--free-tool-calls] [--no-mention-rule]

                Serves each container on the endpoints it can answer:
                  Von decision model          POST /v1/decisions, /v1/systemone, /api/v1/decisions (jevai envelope)
                  embedding container         POST /v1/embeddings
                  classifier container        POST /v1/classify
                  generative decoder + chat   POST /v1/chat/completions, /v1/responses (stream or not)
                  all                         GET  /v1/models, /v1/models/{id}, /health
                The API key may also come from AMQL_API_KEY; without one, requests are not authenticated.

                Chat serves tools (OpenAI function calling) for Qwen3.5-family templates, returns thinking
                as reasoning_content (chat) or reasoning items (responses), and keeps responses for
                previous_response_id.
                  --adapter     Entail adapters to run beside a chat model (id= picks the model; bare: all)
                  --think       think when a request does not say (enable_thinking / reasoning effort)
                  --max-tokens  default token limit per reply: 1024, or 4096 with --think
                  --temperature, --top-p, --top-k   sampling defaults for requests that leave them out
                  --no-batch    one request at a time per model, even on the GPU engine (which otherwise
                                batches up to 8 requests per step)
                  --no-prefix-reuse  prefill every prompt whole (by default a prompt that starts like an
                                earlier one, up to that one's last turn, picks up from the state saved there)
                  --free-tool-calls  do not hold calls to the request's tools (by default a call names a
                                declared tool and its parameters, gives the required ones, and typed values parse)
                  --no-mention-rule  do not open an optional parameter the reply's thinking or the newest
                                user/tool message names (by default it is opened when the model gives it any chance)
                  --ngram-pack FILE  load an n-gram pack (amql-cli ngram build); a request names packs with
                                "ngram_packs", a model with "ngram_packs" in --config, else a pack is chosen when
                                one of its keywords is in the current task
                  --ngram-lambda, --ngram-min-context, --ngram-model-confident, --ngram-pack-confident
                                how a chosen pack is blended (a request may set "ngram_lambda")
                  --allow-repeated-calls  let a reply make a call that has already returned the same result twice
                                (by default it is generated again with calls closed, and answers)
                  --keep-line-marks       show tool results with their line decoration (line numbers, hashes) instead of plain
                  --no-announce-rule  let a reply with tools end on a colon ("now let me build it:") with no
                                call (by default the call opens there when the model gives it any chance)
                  --no-loop-guard  let a reply repeat itself to its token limit (by default a reply whose
                                own tail repeats a stretch 4+ times, 192+ tokens, is stopped, or its thinking closed)
                  --batch-bf16  batches of 4+ on tensor cores, activations rounded to BF16: ~1.6x the
                                throughput, but a reply then depends slightly on what shares its batch

                Several models are served at once, each under its own id; a request names one with "model".
                --config takes a JSON file of models, each with its own adapters and defaults (paths are
                relative to the file); the flags above are the defaults for what a model leaves out:
                  {"models": [{"id": "qwen-think", "path": "qwen3.5-0.8b", "adapters": ["stop-circling"],
                               "think": true, "max_tokens": 4096, "temperature": 0.6, "top_p": 0.95, "top_k": 20},
                              {"id": "qwen", "path": "qwen3.5-0.8b"}]}
                """);
            return 0;
        default:
            Console.Error.WriteLine($"unknown argument '{args[i]}' (see --help)");
            return 2;
    }
}
if (containers.Count == 0)
{
    Console.Error.WriteLine("amql-server needs at least one --model <container-dir> or --config <models.json> (see --help)");
    return 2;
}
if (containers.GroupBy(c => c.Id).FirstOrDefault(g => g.Count() > 1) is { } dup)
{
    Console.Error.WriteLine($"two containers share the model id '{dup.Key}' — name them with --model id=<dir>");
    return 2;
}

var models = new List<ModelHost>();
foreach (var (id, path, own, defaults) in containers)
{
    Console.WriteLine($"loading {id} from {path} …");
    var adapters = own.Concat(adapterSpecs.Where(a => a.Id is null || a.Id == id).SelectMany(a => a.Dirs)).ToList();
    var host = ModelHost.Load(id, path, adapters, defaults);
    if (warm)
    {
        host.Warm();
    }
    Console.WriteLine($"  {id}: {(host.Capabilities == Capability.None ? "no endpoint" : string.Join(", ", ServerApp.CapabilityNames(host.Capabilities)))}");
    foreach (var note in host.Notes)
    {
        Console.WriteLine($"    note: {note}");
    }
    models.Add(host);
}
if (models.All(m => m.Capabilities == Capability.None))
{
    Console.Error.WriteLine("none of the containers can serve an endpoint");
    return 2;
}

// CUDA status — same diagnostic the CLI prints, so the operator knows
// whether GPU acceleration is engaged before the first request lands.
{
    var ws = Amql.Inference.WeightWorkingSetExtensions.FromEnv();
    if (Amql.Inference.CudaShim.Enabled)
    {
        string memNote = Amql.Inference.CudaShim.IsUnifiedMemory
            ? " (unified memory — zero-copy, scratch-buffer staging skipped)"
            : "";
        if (ws == Amql.Inference.WeightWorkingSet.Mxfp4)
        {
            Console.WriteLine($"cuda:      MXFP4 packs resident on device — GEMMs run on the GPU (FP16 tensor cores, FP32 accumulate){memNote}");
        }
        else
        {
            Console.WriteLine($"cuda:      GPU available — {ws} weights uploaded as FP16 for device GEMMs{memNote}");
        }
        if (models.Count(m => m.Capabilities != Capability.None) > 1)
        {
            Console.WriteLine("cuda:      models share the GPU: batching models interleave a step at a time, others a request at a time");
        }
    }
    else
    {
        Console.WriteLine("cuda:      GPU not available or disabled (AMQL_GPU) — all GEMMs run on the CPU");
    }
}

if (adapterSpecs.FirstOrDefault(a => a.Id is not null && containers.All(c => c.Id != a.Id)) is { Id: { } unknown })
{
    Console.Error.WriteLine($"--adapter names the model '{unknown}', which is not loaded");
    return 2;
}
foreach (var pack in ngramPacks.Values)
{
    Console.WriteLine($"ngram:     pack {pack.Name}, order {pack.Order}, {pack.ContextCount:N0} contexts from {pack.CorpusTokens:N0} tokens; " +
                      $"chosen for {(pack.Keywords.Count > 0 ? string.Join(", ", pack.Keywords) : "requests that name it")}");
}
var app = ServerApp.Build(models, new ServerSettings(apiKey, maxTokens ?? (think ? 4096 : 1024), think, temperature, topP, topK)
{
    NGramPacks = ngramPacks,
    NGramDefaults = ngram,
});
app.Urls.Add(urls);
Console.WriteLine($"listening on {urls}{(apiKey is null ? " (no API key — requests are not authenticated)" : "")}");
await app.RunAsync();
foreach (var m in models)
{
    m.Dispose();
}
return 0;
