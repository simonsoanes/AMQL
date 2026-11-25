using Amql.Server;

// amql-server --model [id=]<container-dir> [--model …] [--urls http://127.0.0.1:8000]
//             [--api-key KEY | AMQL_API_KEY] [--max-tokens N] [--no-warm]
//             [--adapter [id=]<dir>[,<dir>…]] [--think] [--thinking-budget N] [--temperature T] [--top-p P] [--top-k K]
//             [--config <models.json>] [--gpu-format nvfp4|mxfp4|fp16]
using System.Globalization;
using System.Text.Json;

var containers = new List<(string Id, string Path, List<string> Adapters, ModelDefaults Defaults, string? Format, string? KvDir, Amql.Vindex3.ContextExtension? Context)>();
string? maxContext = null, ropeExtend = null;
int thinkingBudget = 0;
var adapterSpecs = new List<(string? Id, string[] Dirs)>();
string urls = "http://127.0.0.1:8000";
// A 4-bit format serves each container's GPTQ copy (Fp4Cache), not weights rounded to nearest as they load.
string? gpuFormat = Environment.GetEnvironmentVariable("AMQL_GPU_FORMAT")?.Trim().ToLowerInvariant() switch
{
    "nvfp4" or "fp4" => "nvfp4",
    "mxfp4" => "mxfp4",
    _ => null,
};
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
            containers.Add((id, path, new List<string>(), new ModelDefaults(), null, null, null));
            break;
        }
        case "--config":
            try
            {
                containers.AddRange(ServerConfig.Read(Next()).Select(m => (m.Id, m.Path, m.Adapters.ToList(), m.Defaults, m.Format, m.KvDir, m.Context)));
            }
            catch (Exception e) when (e is JsonException or InvalidDataException or IOException)
            {
                Console.Error.WriteLine($"--config: {e.Message}");
                return 2;
            }
            break;
        case "--gpu-format":
        {
            string f = Next().ToLowerInvariant();
            if (f is not ("nvfp4" or "mxfp4" or "fp16"))
            {
                Console.Error.WriteLine("--gpu-format needs nvfp4, mxfp4 or fp16");
                return 2;
            }
            gpuFormat = f;
            break;
        }
        case "--max-ram" or "--gpu-memory":
        {
            // As amql-cli: GiB targets for the lean working set; a RAM ceiling selects it
            // unless a weight mode was chosen explicitly.
            string flag = args[i];
            if (!double.TryParse(Next(), NumberStyles.Float, CultureInfo.InvariantCulture, out double gb) || gb < 0 ||
                (flag == "--max-ram" && gb == 0))
            {
                Console.Error.WriteLine($"{flag} needs a size in GiB, e.g. {flag} 10");
                return 2;
            }
            if (flag == "--max-ram")
            {
                Amql.Inference.ComputeBudget.LeanRamBytes = (long)(gb * (1L << 30));
                if (Environment.GetEnvironmentVariable("AMQL_WEIGHTS") is null)
                {
                    Environment.SetEnvironmentVariable("AMQL_WEIGHTS", "lean");
                }
            }
            else
            {
                Amql.Inference.ComputeBudget.GpuMemoryBytes = (long)(gb * (1L << 30));
            }
            break;
        }
        case "--cpu":
            Amql.Inference.CudaShim.ForceDisable();
            break;
        case "--urls": urls = Next(); break;
        case "--api-key": apiKey = Next(); break;
        case "--max-tokens": maxTokens = int.Parse(Next()); break;
        case "--no-warm": warm = false; break;
        case "--no-batch": TextGenerator.BatchingEnabled = false; break;
        case "--no-prefix-reuse": BatchScheduler.PrefixReuse = false; break;
        case "--kv-idle" or "--kv-expire" or "--kv-gpu-budget" or "--kv-disk-budget":
        {
            string flag = args[i];
            if (!double.TryParse(Next(), NumberStyles.Float, CultureInfo.InvariantCulture, out double v) || v < 0)
            {
                Console.Error.WriteLine($"{flag} needs a number ({(flag.EndsWith("budget") ? "GiB" : "seconds, 0 for never")})");
                return 2;
            }
            KvPaging.Defaults = flag switch
            {
                "--kv-idle" => KvPaging.Defaults with { IdleSeconds = v },
                "--kv-expire" => KvPaging.Defaults with { ExpireSeconds = v },
                "--kv-gpu-budget" => KvPaging.Defaults with { GpuBudgetBytes = (long)(v * (1L << 30)) },
                _ => KvPaging.Defaults with { DiskBudgetBytes = (long)(v * (1L << 30)) },
            };
            break;
        }
        case "--kv-disk": KvPaging.DiskEnabled = true; break;
        case "--kv-exact-format":
        {
            string v = Next().ToLowerInvariant();
            if (v is not ("f32" or "f16" or "fp16"))
            {
                Console.Error.WriteLine("--kv-exact-format needs f32 or f16");
                return 2;
            }
            KvPaging.Defaults = KvPaging.Defaults with { ExactF16 = v != "f32" };
            break;
        }
        case "--kv-exact-prefix":
        {
            string v = Next().ToLowerInvariant();
            int n = 0;
            if (v != "auto" && (!int.TryParse(v, out n) || n < 0))
            {
                Console.Error.WriteLine("--kv-exact-prefix needs auto, 0 or a token count");
                return 2;
            }
            KvPaging.Defaults = KvPaging.Defaults with { ExactPrefix = v == "auto" ? -1 : n };
            break;
        }
        case "--kv-format":
        {
            string f = Next().ToLowerInvariant();
            if (f is not ("nvfp4" or "fp16" or "f32"))
            {
                Console.Error.WriteLine("--kv-format needs f32, fp16 or nvfp4");
                return 2;
            }
            Amql.Inference.GpuEngine.RequestedKvFormat = f;
            break;
        }
        case "--max-context": maxContext = Next(); break;
        case "--rope-extend": ropeExtend = Next(); break;
        case "--kv-dir": KvPaging.DiskRoot = Next(); KvPaging.DiskEnabled = true; break;
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
        case "--thinking-budget":
            if (!int.TryParse(Next(), out thinkingBudget) || thinkingBudget < 0)
            {
                Console.Error.WriteLine("--thinking-budget needs a token count (0: no limit)");
                return 2;
            }
            break;
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
                            [--adapter [id=]<dir>[,<dir>…]] [--think] [--thinking-budget N] [--temperature T] [--top-p P] [--top-k K]
                            [--config <models.json>] [--gpu-format nvfp4|mxfp4|fp16] [--max-context N] [--rope-extend auto|none|yarn|ntk|linear] [--max-ram GB] [--gpu-memory GB] [--cpu] [--no-batch] [--no-prefix-reuse] [--kv-format f32|fp16|nvfp4] [--kv-exact-prefix auto|N|0] [--kv-exact-format f16|f32] [--kv-idle S] [--kv-gpu-budget GB] [--kv-disk] [--kv-dir DIR] [--kv-disk-budget GB] [--kv-expire S] [--batch-bf16] [--no-loop-guard] [--free-tool-calls] [--no-mention-rule]

                Serves each container on the endpoints it can answer:
                  Von decision model          POST /v1/decisions, /v1/systemone, /api/v1/decisions (jevai envelope)
                  embedding container         POST /v1/embeddings
                  classifier container        POST /v1/classify
                  generative decoder + chat   POST /v1/chat/completions, /v1/responses (stream or not)
                  all                         GET  /v1/models, /v1/models/{id}, /health
                The API key may also come from AMQL_API_KEY; without one, requests are not authenticated.

                Chat runs the model's own chat template and serves tools (OpenAI function calling) in the
                form it writes calls, learned from it. It returns thinking as reasoning_content (chat) or reasoning items (responses), and keeps responses for
                previous_response_id.
                  --adapter     Entail adapters to run beside a chat model (id= picks the model; bare: all)
                  --gpu-format  nvfp4 or mxfp4 serves each container's GPTQ-rounded 4-bit copy, built beside it
                                ({dir}-nvfp4-gptq) the first time and reused; a container stored at 4 bits is served
                                as it is (also AMQL_GPU_FORMAT, or "format" per model in --config)
                  --max-ram     GiB the process aims to stay under: weights stream through a cache sized to fit
                                (the lean working set, as amql-cli --max-ram)
                  --gpu-memory  GiB of GPU memory the weights may take (0 keeps every weight on the host)
                  --cpu         run on the CPU even when a GPU is available
                  --think       think when a request does not say (enable_thinking / reasoning effort)
                  --thinking-budget N  close a reply's thinking after N tokens (at the end of a line) so that it
                                commits to an answer, and before its token limit would leave no room for one
                                (default 0, no limit; "thinking_budget" per model in --config and per request)
                  --max-tokens  default token limit per reply: 1024, or 4096 with --think
                  --temperature, --top-p, --top-k   sampling defaults for requests that leave them out
                  --no-batch    one request at a time per model, even on the GPU engine (which otherwise
                                batches up to 8 requests per step)
                  --max-context N  serve the models at this context length instead of their own (1048576, 512K,
                                1M; "max_context" per model in --config). Longer than a model was trained to needs
                                the KV memory too: --kv-format nvfp4 holds 1M tokens of Qwen 3.5 4B in about 11 GB
                  --rope-extend M  how positions past the trained length rotate: auto (the default), none (as
                                trained), yarn, ntk or linear ("rope_extend" per model in --config)
                  --no-prefix-reuse  prefill every prompt whole (by default a prompt that starts like an
                                earlier one, up to that one's last turn, picks up from the state saved there)
                  --kv-format F fp16 (the default) stores the GPU engine's KV cache as FP16: half f32's memory, a
                                little faster decode at long contexts, and no measurable difference on Qwen 3.5 4B.
                                f32 keeps the values as computed. nvfp4 (E2M1 values, an E4M3 scale per 16, an f32
                                scale per head) takes ~7x less memory a token than f32, with replies that differ
                                slightly from f32's (also AMQL_KV_FORMAT)
                  --kv-exact-prefix auto|N|0  with --kv-format nvfp4, keep each conversation's first positions
                                out of the 4-bit rounding: auto (the default) through its first user message (system
                                prompt, tools, the first instruction), N tokens, or 0 for none
                  --kv-exact-format f16|f32  those positions at FP16 (the default: half f32's memory, ~2e-4
                                relative rounding) or f32 (bit for bit an f32 cache's); K and V are computed in f32
                                whatever the weights' format
                  --kv-idle S   seconds a finished conversation's state stays in GPU memory unused (default 300;
                                0 keeps it until the budget needs the room); then it is released, or paged out
                  --kv-gpu-budget GB  GPU memory unused kept state may hold (default 4); oldest leave first
                  --kv-disk     page released state out to disk (default off), each model to kv-cache in its
                                container; a later prompt that starts the same way loads it instead of prefilling
                  --kv-dir DIR  page out under DIR/<model id> instead (turns --kv-disk on; "kv_dir" per model
                                in --config names a model's own folder)
                  --kv-disk-budget GB  disk each model's paged-out state may take (default 1); oldest deleted first
                  --kv-expire S seconds after its last use paged-out state is deleted (default 1800; 0 never)
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
if (ropeExtend is not null && maxContext is null)
{
    Console.Error.WriteLine("--rope-extend needs --max-context");
    return 2;
}
if (maxContext is not null)
{
    try
    {
        Amql.Vindex3.ContextExtension.Requested = Amql.Vindex3.ContextExtension.Parse(maxContext, ropeExtend);
    }
    catch (FormatException e)
    {
        Console.Error.WriteLine(e.Message);
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
foreach (var (id, given, own, defaults, format, kvDir, context) in containers)
{
    string path = (format ?? gpuFormat) switch
    {
        "nvfp4" => Amql.Merge.Fp4Cache.Resolve(given, Amql.Merge.Fp4Format.Nvfp4, line => Console.WriteLine($"  {id}: {line}")),
        "mxfp4" => Amql.Merge.Fp4Cache.Resolve(given, Amql.Merge.Fp4Format.Mxfp4, line => Console.WriteLine($"  {id}: {line}")),
        _ => given,
    };
    Console.WriteLine($"loading {id} from {path} …");
    var adapters = own.Concat(adapterSpecs.Where(a => a.Id is null || a.Id == id).SelectMany(a => a.Dirs)).ToList();
    var host = ModelHost.Load(id, path, adapters, defaults, kvDir, context);
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
    ThinkingBudget = thinkingBudget,
});
app.Urls.Add(urls);
Console.WriteLine($"listening on {urls}{(apiKey is null ? " (no API key — requests are not authenticated)" : "")}");
await app.RunAsync();
foreach (var m in models)
{
    m.Dispose();
}
return 0;
