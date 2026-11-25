using System.Text.Json;

namespace Amql.Server;

/// <summary>One model of a <c>--config</c> file: where it is, the adapters beside it, its defaults.</summary>
public sealed record ConfiguredModel(string Id, string Path, IReadOnlyList<string> Adapters, ModelDefaults Defaults)
{
    /// <summary>"nvfp4" or "mxfp4" to serve the container's GPTQ 4-bit copy
    /// (built beside it the first time), "fp16" for its weights as stored;
    /// null leaves it to --gpu-format / AMQL_GPU_FORMAT.</summary>
    public string? Format { get; init; }

    /// <summary>The model's own folder for paged-out prompt state, which turns
    /// its disk tier on; null leaves it to --kv-disk / --kv-dir.</summary>
    public string? KvDir { get; init; }

    /// <summary>A context length other than the model's own (<c>"max_context"</c>,
    /// with <c>"rope_extend"</c>); null leaves it to --max-context.</summary>
    public Amql.Vindex3.ContextExtension? Context { get; init; }
}

/// <summary>amql-server's <c>--config</c> file, for serving several models each with its own settings.</summary>
public static class ServerConfig
{
    /// <summary>The models of a <c>--config</c> file: <c>{"models": [{id, path, adapters?, think?, max_tokens?,
    /// temperature?, top_p?, top_k?}]}</c>, with paths relative to the file.</summary>
    public static IReadOnlyList<ConfiguredModel> Read(string file)
    {
        string root = Path.GetDirectoryName(Path.GetFullPath(file))!;
        string Resolve(string p) => Path.IsPathRooted(p) ? p : Path.GetFullPath(Path.Combine(root, p));
        using var doc = JsonDocument.Parse(File.ReadAllText(file), new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true });
        if (!doc.RootElement.TryGetProperty("models", out var models) || models.ValueKind != JsonValueKind.Array)
        {
            throw new InvalidDataException($"{file} needs a \"models\" array");
        }
        var known = new HashSet<string> { "id", "path", "adapters", "think", "max_tokens", "temperature", "top_p", "top_k", "ngram_packs", "format", "kv_dir", "max_context", "rope_extend", "thinking_budget" };
        var list = new List<ConfiguredModel>();
        int i = 0;
        foreach (var m in models.EnumerateArray())
        {
            string where = $"models[{i++}]";
            if (m.EnumerateObject().Select(p => p.Name).FirstOrDefault(n => !known.Contains(n)) is { } unknown)
            {
                throw new InvalidDataException($"{where}: unknown setting '{unknown}' ({string.Join(", ", known)})");
            }
            string path = m.TryGetProperty("path", out var pv) && pv.ValueKind == JsonValueKind.String ? Resolve(pv.GetString()!)
                : throw new InvalidDataException($"{where} needs a \"path\"");
            string id = m.TryGetProperty("id", out var iv) && iv.ValueKind == JsonValueKind.String ? iv.GetString()!
                : Path.GetFileName(path.TrimEnd('/', '\\'));
            var adapters = m.TryGetProperty("adapters", out var av) && av.ValueKind == JsonValueKind.Array
                ? av.EnumerateArray().Select(a => Resolve(a.GetString() ?? throw new InvalidDataException($"{where}: adapters are paths"))).ToList()
                : new List<string>();
            float? Float(string name) => m.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetSingle() : null;
            int? Int(string name) => m.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : null;
            bool? think = m.TryGetProperty("think", out var tv) && tv.ValueKind is JsonValueKind.True or JsonValueKind.False ? tv.GetBoolean() : null;
            var packs = m.TryGetProperty("ngram_packs", out var np) && np.ValueKind == JsonValueKind.Array
                ? np.EnumerateArray().Select(x => x.GetString()!).ToList()
                : null;
            string? format = m.TryGetProperty("format", out var fv) && fv.ValueKind == JsonValueKind.String ? fv.GetString()!.ToLowerInvariant() : null;
            if (format is not (null or "nvfp4" or "mxfp4" or "fp16"))
            {
                throw new InvalidDataException($"{where}: format is nvfp4, mxfp4 or fp16, not '{format}'");
            }
            string? Text(string name) => !m.TryGetProperty(name, out var v) ? null : v.ValueKind switch
            {
                JsonValueKind.String => v.GetString(),
                JsonValueKind.Number => v.GetRawText(),
                _ => throw new InvalidDataException($"{where}: {name} is a string or a number"),
            };
            Amql.Vindex3.ContextExtension? context = null;
            if (Text("max_context") is { } length)
            {
                try
                {
                    context = Amql.Vindex3.ContextExtension.Parse(length, Text("rope_extend"));
                }
                catch (FormatException e)
                {
                    throw new InvalidDataException($"{where}: {e.Message}");
                }
            }
            else if (m.TryGetProperty("rope_extend", out _))
            {
                throw new InvalidDataException($"{where}: rope_extend needs max_context");
            }
            list.Add(new ConfiguredModel(id, path, adapters, new ModelDefaults(think, Int("max_tokens"), Float("temperature"), Float("top_p"), Int("top_k"))
            {
                NGramPacks = packs,
                ThinkingBudget = Int("thinking_budget") is { } tb ? tb >= 0 ? tb : throw new InvalidDataException($"{where}: thinking_budget is 0 (no limit) or more") : null,
            })
            {
                Format = format,
                Context = context,
                KvDir = m.TryGetProperty("kv_dir", out var kv) && kv.ValueKind == JsonValueKind.String ? Resolve(kv.GetString()!) : null,
            });
        }
        return list;
    }
}
