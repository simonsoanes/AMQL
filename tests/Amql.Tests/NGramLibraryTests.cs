using Amql.Inference.NGram;
using Amql.Server;

namespace Amql.Tests;

/// <summary>
/// A container's <c>ngram/</c> folder, served live: packs present at start are loaded; a pack
/// added, replaced or deleted while running is picked up, reloaded or dropped; the folder may
/// appear later; a pack in another tokenizer, or a file that is not a pack, is refused.
/// </summary>
public sealed class NGramLibraryTests
{
    private const string Tokenizer = "10:test";

    private static void WritePack(string path, string name, int next, string tokenizer = Tokenizer)
    {
        var docs = new[] { new[] { 1, 2, 3, next, 1, 2, 3, next } };
        NGramPack.Build(name, docs, 4, tokenizer).Save(path + ".tmp");
        File.Move(path + ".tmp", path, overwrite: true);   // whole, as a copy tool would land it
    }

    private static void WaitFor(Func<bool> condition, string what)
    {
        var until = DateTime.UtcNow.AddSeconds(10);
        while (!condition())
        {
            Assert.True(DateTime.UtcNow < until, $"timed out waiting for {what}");
            Thread.Sleep(50);
        }
    }

    private static int Next(NGramLibrary lib, string name) => lib.Packs[name].Lookup(new[] { 1, 2, 3 })!.Value.Entry.Tokens[0];

    [Fact]
    public void Packs_Are_Loaded_Added_Replaced_And_Dropped_As_Their_Files_Are()
    {
        using var dir = new TempDir();
        string folder = Path.Combine(dir.Path, NGramLibrary.Folder);
        Directory.CreateDirectory(folder);
        WritePack(Path.Combine(folder, "a.ngram"), "alpha", 4);
        var log = new System.Collections.Concurrent.ConcurrentQueue<string>();
        using var lib = new NGramLibrary(dir.Path, Tokenizer, log.Enqueue) { Settle = TimeSpan.FromMilliseconds(50) };
        Assert.Equal(new[] { "alpha" }, lib.Packs.Keys);

        WritePack(Path.Combine(folder, "b.ngram"), "beta", 5);
        WaitFor(() => lib.Packs.ContainsKey("beta"), "beta to load");
        Assert.Equal(5, Next(lib, "beta"));

        WritePack(Path.Combine(folder, "b.ngram"), "beta", 6);
        WaitFor(() => Next(lib, "beta") == 6, "beta to reload");

        File.Delete(Path.Combine(folder, "a.ngram"));
        WaitFor(() => !lib.Packs.ContainsKey("alpha"), "alpha to drop");
        Assert.Contains(log, l => l.Contains("dropped pack alpha"));
    }

    [Fact]
    public void A_Folder_Made_Later_Is_Watched_And_Foreign_Or_Broken_Files_Are_Refused()
    {
        using var dir = new TempDir();
        var log = new System.Collections.Concurrent.ConcurrentQueue<string>();
        using var lib = new NGramLibrary(dir.Path, Tokenizer, log.Enqueue) { Settle = TimeSpan.FromMilliseconds(50) };
        Assert.Empty(lib.Packs);

        string folder = Path.Combine(dir.Path, NGramLibrary.Folder);
        Directory.CreateDirectory(folder);
        WritePack(Path.Combine(folder, "late.ngram"), "late", 4);
        WaitFor(() => lib.Packs.ContainsKey("late"), "the later folder's pack");

        WritePack(Path.Combine(folder, "other.ngram"), "other", 4, tokenizer: "99:elsewhere");
        File.WriteAllText(Path.Combine(folder, "junk.ngram"), "not a pack");
        WaitFor(() => log.Count(l => l.Contains("refused")) >= 2, "both refusals");
        Assert.Contains(log, l => l.Contains("other.ngram") && l.Contains("99:elsewhere"));
        Assert.Equal(new[] { "late" }, lib.Packs.Keys);

        // A pack beside the folder, not in it, is not the model's.
        WritePack(Path.Combine(dir.Path, "stray.ngram"), "stray", 4);
        Thread.Sleep(300);
        Assert.DoesNotContain("stray", lib.Packs.Keys);
    }
}
