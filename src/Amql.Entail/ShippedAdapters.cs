using System.Text.Json.Nodes;

namespace Amql.Entail;

/// <summary>
/// Adapters a container carries for its own model, in <c>adapters/&lt;name&gt;/</c>, run
/// in the order <c>index.json</c>'s <c>"adapters"</c> lists them (the first <c>Say</c>
/// wins). A server runs them unless it is told which adapters to run.
/// </summary>
public static class ShippedAdapters
{
    public const string Folder = "adapters";

    /// <summary>The container's adapter folders, in order; none when it carries none.</summary>
    public static IReadOnlyList<string> List(string containerDir)
    {
        string index = Path.Combine(containerDir, "index.json");
        if (!File.Exists(index) || JsonNode.Parse(File.ReadAllText(index))?["adapters"] is not JsonArray names)
        {
            return [];
        }
        return names.Select(n => Path.Combine(containerDir, Folder, (string)n!)).ToList();
    }

    /// <summary>A folder as given, or else the container's own adapter of that name.</summary>
    public static string Resolve(string dir, string containerDir)
    {
        if (Directory.Exists(dir))
        {
            return dir;
        }
        string name = Path.GetFileName(Path.TrimEndingDirectorySeparator(dir));
        return List(containerDir).FirstOrDefault(d => Path.GetFileName(d) == name)
            ?? throw new EntailException($"no adapter '{dir}', and '{containerDir}' carries none named '{name}'");
    }

    /// <summary>Copies adapters into the container, after any it carries (one of the same
    /// name is replaced in its place). Each must have been built for the container's tokenizer.</summary>
    public static IReadOnlyList<string> Attach(string containerDir, IEnumerable<string> adapterDirs)
    {
        string indexPath = Path.Combine(containerDir, "index.json");
        var index = JsonNode.Parse(File.ReadAllText(indexPath))!.AsObject();
        var names = (index["adapters"] as JsonArray)?.Select(n => (string)n!).ToList() ?? [];
        foreach (string dir in adapterDirs)
        {
            EntailAdapterManifest.Load(dir).RequirePartner(containerDir);
            if (!File.Exists(Path.Combine(dir, EntailSparseAdapter.ProgramFile)))
            {
                throw new EntailException($"'{dir}' is a dense adapter: a container carries the program form (built without --dense)");
            }
            string name = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(dir)));
            string to = Path.Combine(containerDir, Folder, name);
            if (Directory.Exists(to))
            {
                Directory.Delete(to, recursive: true);
            }
            Directory.CreateDirectory(to);
            foreach (string file in Directory.EnumerateFiles(dir))
            {
                File.Copy(file, Path.Combine(to, Path.GetFileName(file)));
            }
            if (!names.Contains(name))
            {
                names.Add(name);
            }
        }
        index["adapters"] = new JsonArray(names.Select(n => (JsonNode)n).ToArray());
        File.WriteAllText(indexPath, index.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = false }));
        return names;
    }
}
