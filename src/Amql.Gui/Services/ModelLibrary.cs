using System.IO;
using Amql.Vindex3;

namespace Amql.Gui.Services;

/// <summary>What a container can be served as — mirrors amql-server's choice of endpoint.</summary>
public enum ModelKind
{
    Chat,
    Embedding,
    Classifier,
    Decision,
    Diffusion,
    Unservable,
}

/// <summary>A container found on disk, and what the server would make of it.</summary>
public sealed class ModelEntry
{
    public required string Path { get; init; }
    public required string Name { get; init; }
    public ModelKind Kind { get; init; }
    public long SizeBytes { get; init; }
    public long? ContextLength { get; init; }

    /// <summary>"nvfp4" / "mxfp4" when the folder name says it is a 4-bit copy.</summary>
    public string? StoredFormat { get; init; }

    /// <summary>Why the server could not serve it, when <see cref="Kind"/> is Unservable.</summary>
    public string? Problem { get; init; }

    public string KindLabel => Kind switch
    {
        ModelKind.Chat => "Chat",
        ModelKind.Embedding => "Embeddings",
        ModelKind.Classifier => "Classifier",
        ModelKind.Decision => "Decisions",
        ModelKind.Diffusion => "Diffusion",
        _ => "Not servable",
    };

    public string SizeLabel => Format.Bytes(SizeBytes);

    public string Summary =>
        $"{KindLabel} · {SizeLabel}" +
        (ContextLength is { } c ? $" · {c / 1024}K context" : "") +
        (StoredFormat is { } f ? $" · {f.ToUpperInvariant()}" : "");

    public override string ToString() => Name;
}

/// <summary>
/// Finds containers (folders holding an index.json) under the configured model
/// folders and reads each one's system graph to tell chat, embedding,
/// classifier and decision models apart, the way amql-server does.
/// </summary>
public static class ModelLibrary
{
    private static readonly string[] SkipFolders = { "bin", "obj", "src", "segments", "native", ".git", "node_modules", "tests", "docs" };

    public static Task<List<ModelEntry>> ScanAsync(AppSettings settings) => Task.Run(() =>
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var found = new List<ModelEntry>();
        void Consider(string dir)
        {
            string full;
            try
            {
                full = System.IO.Path.GetFullPath(dir).TrimEnd('\\', '/');
            }
            catch (Exception e) when (e is ArgumentException or NotSupportedException or PathTooLongException)
            {
                return;
            }
            if (seen.Add(full) && File.Exists(System.IO.Path.Combine(full, "index.json")))
            {
                found.Add(Describe(full));
            }
        }

        foreach (var root in settings.ModelFolders)
        {
            Walk(root, 0, Consider);
        }
        foreach (var extra in settings.ExtraModels)
        {
            Consider(extra);
        }
        return found.OrderBy(m => m.Kind == ModelKind.Unservable).ThenBy(m => m.Name, StringComparer.OrdinalIgnoreCase).ToList();
    });

    private static void Walk(string dir, int depth, Action<string> consider)
    {
        if (depth > 2 || !Directory.Exists(dir))
        {
            return;
        }
        if (File.Exists(System.IO.Path.Combine(dir, "index.json")))
        {
            consider(dir);
            return;     // a container's own folders hold no further containers
        }
        try
        {
            foreach (var sub in Directory.EnumerateDirectories(dir))
            {
                string name = System.IO.Path.GetFileName(sub);
                if (name.StartsWith('.') || SkipFolders.Contains(name, StringComparer.OrdinalIgnoreCase))
                {
                    continue;
                }
                Walk(sub, depth + 1, consider);
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            // unreadable folder
        }
    }

    /// <summary>Reads one container's facts. Never throws: a broken container is listed as not servable.</summary>
    public static ModelEntry Describe(string path)
    {
        string name = System.IO.Path.GetFileName(path.TrimEnd('\\', '/'));
        long size = FolderSize(path);
        string? stored = name.Contains("nvfp4", StringComparison.OrdinalIgnoreCase) ? "nvfp4"
            : name.Contains("mxfp4", StringComparison.OrdinalIgnoreCase) ? "mxfp4" : null;
        try
        {
            using var container = Vindex3Container.Open(path);
            var component = container.Graph?.Components.FirstOrDefault(c => c.Role == ComponentRole.PrimaryText);
            if (component is null)
            {
                return new ModelEntry { Path = path, Name = name, Kind = ModelKind.Unservable, SizeBytes = size, Problem = "no primary text component" };
            }
            var surface = component.Execution;
            bool tokenizer = File.Exists(System.IO.Path.Combine(path, "tokenizer.json"));
            (ModelKind kind, string? problem) = surface switch
            {
                { Encoder.Architecture: "ModernBertModel", OptionMarker: not null } => (ModelKind.Decision, null),
                { Encoder.Architecture: "ModernBertModel" } => (ModelKind.Unservable, "encoder without an option-marker head"),
                { Classifier: not null } => tokenizer ? (ModelKind.Classifier, null) : (ModelKind.Unservable, "no tokenizer.json"),
                _ when !tokenizer => (ModelKind.Unservable, "no tokenizer.json"),
                { Embedding: not null } => (ModelKind.Embedding, null),
                { Diffusion: not null } => (ModelKind.Diffusion, (string?)"masked diffusion runs from the CLI, not the server"),
                _ when !HasChatTemplate(path) => (ModelKind.Unservable, "no chat template"),
                _ => (ModelKind.Chat, null),
            };
            return new ModelEntry
            {
                Path = path, Name = name, Kind = kind, Problem = problem, SizeBytes = size,
                ContextLength = surface?.ContextLength, StoredFormat = stored,
            };
        }
        catch (Exception e)
        {
            return new ModelEntry { Path = path, Name = name, Kind = ModelKind.Unservable, SizeBytes = size, Problem = e.Message };
        }
    }

    private static bool HasChatTemplate(string path) =>
        File.Exists(System.IO.Path.Combine(path, "chat_template.jinja")) ||
        (File.Exists(System.IO.Path.Combine(path, "tokenizer_config.json")) &&
         File.ReadAllText(System.IO.Path.Combine(path, "tokenizer_config.json")).Contains("chat_template", StringComparison.Ordinal));

    private static long FolderSize(string path)
    {
        try
        {
            return new DirectoryInfo(path).EnumerateFiles("*", SearchOption.AllDirectories).Sum(f => f.Length);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return 0;
        }
    }
}

/// <summary>Small display helpers.</summary>
public static class Format
{
    public static string Bytes(long bytes) => bytes switch
    {
        >= 1L << 30 => $"{bytes / (double)(1L << 30):0.0} GB",
        >= 1L << 20 => $"{bytes / (double)(1L << 20):0} MB",
        >= 1L << 10 => $"{bytes / 1024.0:0} KB",
        _ => $"{bytes} B",
    };

    public static string Duration(TimeSpan t) => t.TotalSeconds switch
    {
        < 1 => $"{t.TotalMilliseconds:0} ms",
        < 60 => $"{t.TotalSeconds:0.0} s",
        < 3600 => $"{(int)t.TotalMinutes}m {t.Seconds:00}s",
        _ => $"{(int)t.TotalHours}h {t.Minutes:00}m",
    };
}
