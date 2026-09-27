using Amql.Server;

// amql-server --model [id=]<container-dir> [--model …] [--urls http://127.0.0.1:8000]
//             [--api-key KEY | AMQL_API_KEY] [--max-tokens 1024] [--no-warm]
var containers = new List<(string Id, string Path)>();
string urls = "http://127.0.0.1:8000";
string? apiKey = Environment.GetEnvironmentVariable("AMQL_API_KEY");
int maxTokens = 1024;
bool warm = true;
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
            containers.Add((id, path));
            break;
        }
        case "--urls": urls = Next(); break;
        case "--api-key": apiKey = Next(); break;
        case "--max-tokens": maxTokens = int.Parse(Next()); break;
        case "--no-warm": warm = false; break;
        case "--help" or "-h":
            Console.WriteLine("""
                amql-server --model [id=]<container-dir> [--model …] [--urls http://127.0.0.1:8000]
                            [--api-key KEY] [--max-tokens 1024] [--no-warm]

                Serves each container on the endpoints it can answer:
                  Von decision model          POST /v1/decisions, /v1/systemone, /api/v1/decisions (jevai envelope)
                  embedding container         POST /v1/embeddings
                  generative decoder + chat   POST /v1/chat/completions, /v1/responses (stream or not)
                  all                         GET  /v1/models, /health
                The API key may also come from AMQL_API_KEY; without one, requests are not authenticated.
                """);
            return 0;
        default:
            Console.Error.WriteLine($"unknown argument '{args[i]}' (see --help)");
            return 2;
    }
}
if (containers.Count == 0)
{
    Console.Error.WriteLine("amql-server needs at least one --model <container-dir> (see --help)");
    return 2;
}
if (containers.GroupBy(c => c.Id).FirstOrDefault(g => g.Count() > 1) is { } dup)
{
    Console.Error.WriteLine($"two containers share the model id '{dup.Key}' — name them with --model id=<dir>");
    return 2;
}

var models = new List<ModelHost>();
foreach (var (id, path) in containers)
{
    Console.WriteLine($"loading {id} from {path} …");
    var host = ModelHost.Load(id, path);
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
        if (ws == Amql.Inference.WeightWorkingSet.Mxfp4)
        {
            Console.WriteLine("cuda:      MXFP4 packs resident on device — GEMMs run on the GPU (FP16 tensor cores, FP32 accumulate)");
        }
        else
        {
            Console.WriteLine($"cuda:      GPU available — {ws} weights uploaded as FP16 for device GEMMs");
        }
    }
    else
    {
        Console.WriteLine("cuda:      GPU not available or disabled (AMQL_GPU) — all GEMMs run on the CPU");
    }
}

var app = ServerApp.Build(models, new ServerSettings(apiKey, maxTokens));
app.Urls.Add(urls);
Console.WriteLine($"listening on {urls}{(apiKey is null ? " (no API key — requests are not authenticated)" : "")}");
await app.RunAsync();
foreach (var m in models)
{
    m.Dispose();
}
return 0;
