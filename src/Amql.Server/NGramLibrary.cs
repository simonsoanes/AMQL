using System.Collections.Immutable;
using Amql.Inference.NGram;

namespace Amql.Server;

/// <summary>
/// The n-gram packs a model's container holds in its <c>ngram/</c> folder, kept current while the
/// server runs: a pack is loaded when its file is added or replaced and dropped when the file is
/// deleted, with no restart. Requests read an immutable snapshot, so a pack never changes under a
/// reply. A file still being written is retried until it reads whole; a pack counted in another
/// tokenizer is refused, since its token ids would mean other tokens to this model. The folder need
/// not exist when the server starts: the container is watched for it.
/// </summary>
public sealed class NGramLibrary : IDisposable
{
    public const string Folder = "ngram";
    public const string Extension = ".ngram";

    private readonly string _root;
    private readonly string _folder;
    private readonly string _tokenizer;
    private readonly Action<string> _log;
    private readonly FileSystemWatcher? _watcher;
    private readonly object _gate = new();
    private readonly Dictionary<string, CancellationTokenSource> _pending = new(StringComparer.OrdinalIgnoreCase);
    private ImmutableDictionary<string, NGramPack> _byFile = ImmutableDictionary.Create<string, NGramPack>(StringComparer.OrdinalIgnoreCase);
    private bool _disposed;

    /// <summary>How long a file must be quiet before it is read (a copy fires several events).</summary>
    public TimeSpan Settle { get; init; } = TimeSpan.FromMilliseconds(300);

    /// <param name="containerDir">The model's container; packs live in its <c>ngram/</c> folder.</param>
    /// <param name="tokenizer">The model's tokenizer fingerprint (<see cref="NGramPack.Fingerprint"/>).</param>
    public NGramLibrary(string containerDir, string tokenizer, Action<string>? log = null, bool watch = true)
    {
        _root = Path.GetFullPath(containerDir);
        _folder = Path.Combine(_root, Folder);
        _tokenizer = tokenizer;
        _log = log ?? (_ => { });
        if (Directory.Exists(_folder))
        {
            foreach (string file in Directory.EnumerateFiles(_folder, "*" + Extension))
            {
                Load(file);
            }
        }
        if (watch)
        {
            // The container, not the folder: the folder may be created later, or replaced whole.
            _watcher = new FileSystemWatcher(_root)
            {
                IncludeSubdirectories = true,
                NotifyFilter = NotifyFilters.FileName | NotifyFilters.DirectoryName | NotifyFilters.LastWrite | NotifyFilters.Size,
            };
            _watcher.Created += (_, e) => Changed(e.FullPath);
            _watcher.Changed += (_, e) => Changed(e.FullPath);
            _watcher.Deleted += (_, e) => Changed(e.FullPath);
            _watcher.Renamed += (_, e) =>
            {
                Changed(e.OldFullPath);
                Changed(e.FullPath);
            };
            _watcher.Error += (_, e) =>
            {
                _log($"ngram: watching {_folder} failed ({e.GetException().Message}); rescanning");
                Rescan();
            };
            _watcher.EnableRaisingEvents = true;
        }
    }

    /// <summary>The packs loaded now, by name.</summary>
    public IReadOnlyDictionary<string, NGramPack> Packs
    {
        get
        {
            var byFile = _byFile;
            var byName = new Dictionary<string, NGramPack>(StringComparer.Ordinal);
            foreach (var (file, pack) in byFile.OrderBy(kv => kv.Key, StringComparer.OrdinalIgnoreCase))
            {
                byName[pack.Name] = pack;
            }
            return byName;
        }
    }

    private bool InFolder(string path)
    {
        string? dir = Path.GetDirectoryName(Path.GetFullPath(path));
        return string.Equals(dir, _folder, StringComparison.OrdinalIgnoreCase);
    }

    private void Changed(string path)
    {
        if (_disposed)
        {
            return;
        }
        string full = Path.GetFullPath(path);
        if (string.Equals(full, _folder, StringComparison.OrdinalIgnoreCase))
        {
            Rescan();   // the folder itself appeared, vanished or was renamed
            return;
        }
        if (!InFolder(full) || !full.EndsWith(Extension, StringComparison.OrdinalIgnoreCase))
        {
            return;
        }
        // Wait for the file to settle: a copy is a create and several writes.
        CancellationTokenSource cts;
        lock (_gate)
        {
            if (_pending.Remove(full, out var earlier))
            {
                earlier.Cancel();
            }
            cts = new CancellationTokenSource();
            _pending[full] = cts;
        }
        _ = Task.Delay(Settle, cts.Token).ContinueWith(t =>
        {
            if (t.IsCanceled)
            {
                return;
            }
            lock (_gate)
            {
                _pending.Remove(full);
            }
            if (File.Exists(full))
            {
                Load(full);
            }
            else
            {
                Drop(full);
            }
        }, TaskScheduler.Default);
    }

    private void Rescan()
    {
        var present = Directory.Exists(_folder)
            ? Directory.EnumerateFiles(_folder, "*" + Extension).Select(Path.GetFullPath).ToHashSet(StringComparer.OrdinalIgnoreCase)
            : new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string gone in _byFile.Keys.Where(f => !present.Contains(f)).ToList())
        {
            Drop(gone);
        }
        foreach (string file in present)
        {
            Changed(file);
        }
    }

    /// <summary>Reads a pack file, retrying while it is still locked or short (being written).</summary>
    private void Load(string file)
    {
        string full = Path.GetFullPath(file);
        for (int attempt = 0; attempt < 20 && !_disposed; attempt++)
        {
            try
            {
                var pack = NGramPack.Load(full);
                if (pack.Tokenizer != _tokenizer)
                {
                    _log($"ngram: refused {full}: built for tokenizer {pack.Tokenizer}, the model's is {_tokenizer}");
                    Drop(full, quiet: true);
                    return;
                }
                bool replaced = _byFile.ContainsKey(full);
                ImmutableInterlocked.Update(ref _byFile, d => d.SetItem(full, pack));
                _log($"ngram: {(replaced ? "reloaded" : "loaded")} pack {pack.Name} from {Path.GetFileName(full)} " +
                     $"({pack.ContextCount:N0} contexts, order {pack.Order})");
                return;
            }
            catch (Exception e) when (e is IOException or EndOfStreamException or UnauthorizedAccessException)
            {
                Thread.Sleep(250);   // still being copied
            }
            catch (Exception e) when (e is InvalidDataException or System.Text.Json.JsonException or ArgumentException or KeyNotFoundException
                                          or FormatException or OverflowException)
            {
                _log($"ngram: refused {full}: not a pack this build reads ({e.Message})");
                Drop(full, quiet: true);
                return;
            }
        }
        _log($"ngram: could not read {full}: it stayed locked or incomplete");
    }

    private void Drop(string file, bool quiet = false)
    {
        string full = Path.GetFullPath(file);
        if (_byFile.TryGetValue(full, out var pack))
        {
            ImmutableInterlocked.Update(ref _byFile, d => d.Remove(full));
            if (!quiet)
            {
                _log($"ngram: dropped pack {pack.Name} ({Path.GetFileName(full)} was removed)");
            }
        }
    }

    public void Dispose()
    {
        _disposed = true;
        _watcher?.Dispose();
        lock (_gate)
        {
            foreach (var cts in _pending.Values)
            {
                cts.Cancel();
            }
            _pending.Clear();
        }
    }
}
