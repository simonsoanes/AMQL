using System.Text;

namespace Amql.Server;

/// <summary>
/// How a batching model keeps prompt state between requests: in GPU memory
/// while it is in use, then (when a folder is given) on disk. Set from the
/// server's flags and each model's <c>--config</c> entry.
/// </summary>
public sealed record KvPaging
{
    /// <summary>Seconds a kept sequence may sit unused before its GPU memory
    /// is released (paged to disk first when the disk tier is on); 0 never.</summary>
    public double IdleSeconds { get; init; } = 300;

    /// <summary>Bytes of GPU memory unused kept sequences may hold; past it
    /// the least recently used leave first.</summary>
    public long GpuBudgetBytes { get; init; } = 4L << 30;

    /// <summary>The disk tier's folder for this model, or null with the disk tier off.</summary>
    public string? DiskDir { get; init; }

    /// <summary>Bytes the disk tier may hold; past it the oldest files are deleted.</summary>
    public long DiskBudgetBytes { get; init; } = 1L << 30;

    /// <summary>Seconds after its last use a paged-out state is deleted; 0 never.</summary>
    public double ExpireSeconds { get; init; } = 1800;

    /// <summary>With an NVFP4 KV cache, how many of a conversation's first
    /// positions stay at f32: -1 (auto) through the end of its first user
    /// message (system prompt, tools and the first instruction: the intent),
    /// 0 none, or that many tokens.</summary>
    public int ExactPrefix { get; init; } = -1;

    /// <summary>Whether those positions are FP16 (the default: half f32's
    /// memory, ~2e-4 relative rounding, no measurable difference on Qwen 3.5 4B)
    /// rather than f32 (<c>--kv-exact-format f32</c>).</summary>
    public bool ExactF16 { get; init; } = true;

    /// <summary>Prompts saved up to fewer tokens than this are not paged out:
    /// they prefill again faster than they load.</summary>
    public int MinPageTokens { get; init; } = 512;

    /// <summary>The server's defaults (<c>--kv-*</c> flags), which a model's <c>kv_dir</c> may override.</summary>
    public static KvPaging Defaults { get; set; } = new();

    /// <summary>The server-wide root for the disk tier (<c>--kv-dir</c>); each model gets a folder under it.</summary>
    public static string? DiskRoot { get; set; }

    /// <summary>Whether the disk tier is on for models that name no folder of their own (<c>--kv-disk</c>).</summary>
    public static bool DiskEnabled { get; set; }

    /// <summary>The paging for one model: its own folder when it names one,
    /// else <c>&lt;--kv-dir&gt;/&lt;id&gt;</c>, else <c>kv-cache</c> under its container.</summary>
    public static KvPaging For(string id, string containerPath, string? ownDir)
    {
        string? dir = ownDir ?? (DiskRoot is { } root ? Path.Combine(root, id)
            : DiskEnabled ? Path.Combine(containerPath, "kv-cache") : null);
        return Defaults with { DiskDir = dir is null ? null : Path.GetFullPath(dir) };
    }
}

/// <summary>
/// A model's paged-out prompt states on disk: what a GPU engine's
/// <c>SaveMark</c> wrote, under the prompt tokens it was saved after. Each
/// model has its own folder; a file records the model's fingerprint (its
/// container and engine geometry), so a state is only restored under the
/// weights and layout that made it, and others are deleted when found. The
/// oldest go first past the size budget, and every one after the expiry time.
/// Writes run off the caller's thread; a state being written can still be taken.
/// Thread-safe.
/// </summary>
public sealed class KvPageStore : IDisposable
{
    private const string Magic = "AMQLKV01";
    private const string Extension = ".kvpage";

    private sealed class Entry
    {
        public required int[] Tokens;
        public required long Bytes;
        public required DateTime LastUsed;
        public string? File;           // once written
        public ChunkedBuffer? Pending; // while being written
    }

    private readonly string _dir;
    private readonly string _fingerprint;
    private readonly long _budget;
    private readonly TimeSpan? _expire;
    private readonly Action<string>? _log;
    private readonly List<Entry> _entries = new();
    private readonly object _lock = new();
    private readonly Timer _sweep;
    private Task _writes = Task.CompletedTask;
    public long PagedOut, Restored, Expired, Evicted;

    public KvPageStore(string dir, string fingerprint, long budgetBytes, double expireSeconds, Action<string>? log = null)
    {
        _dir = Path.GetFullPath(dir);
        _fingerprint = fingerprint;
        _budget = budgetBytes;
        _expire = expireSeconds > 0 ? TimeSpan.FromSeconds(expireSeconds) : null;
        _log = log;
        Directory.CreateDirectory(_dir);
        Load();
        _sweep = new Timer(_ => Sweep(DateTime.UtcNow), null, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
    }

    public string Folder => _dir;

    /// <summary>Bytes held on disk and being written.</summary>
    public long Bytes
    {
        get
        {
            lock (_lock)
            {
                return _entries.Sum(e => e.Bytes);
            }
        }
    }

    public int Count
    {
        get
        {
            lock (_lock)
            {
                return _entries.Count;
            }
        }
    }

    /// <summary>A model's fingerprint: its engine geometry and its container's
    /// files (names, sizes, times), so a rebuilt container never restores
    /// another's state.</summary>
    public static string Fingerprint(string containerPath, string geometry)
    {
        var sb = new StringBuilder(geometry).Append('|');
        foreach (var f in new DirectoryInfo(containerPath).GetFiles().OrderBy(f => f.Name, StringComparer.Ordinal))
        {
            sb.Append(f.Name).Append(':').Append(f.Length).Append(':').Append(f.LastWriteTimeUtc.Ticks).Append(';');
        }
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString())));
    }

    /// <summary>Keeps a state saved after <paramref name="tokens"/>; it is
    /// written to disk in the background. Refused (false) when it alone is
    /// over the budget.</summary>
    public bool Put(int[] tokens, ChunkedBuffer state, DateTime lastUsed)
    {
        long bytes = HeaderBytes(tokens.Length) + state.Length;
        if (bytes > _budget)
        {
            _log?.Invoke($"kv: not paged out ({tokens.Length} tokens, {Mb(bytes)} is over the {Mb(_budget)} disk budget)");
            return false;
        }
        var entry = new Entry { Tokens = tokens, Bytes = bytes, LastUsed = lastUsed, Pending = state };
        lock (_lock)
        {
            // A state saved after the same tokens replaces the older one.
            foreach (var same in _entries.Where(e => e.Tokens.AsSpan().SequenceEqual(tokens)).ToList())
            {
                Remove(same);
            }
            _entries.Add(entry);
            Trim();
            PagedOut++;
            _writes = _writes.ContinueWith(_ => Write(entry), TaskScheduler.Default);
        }
        _log?.Invoke($"kv: paged out {tokens.Length} tokens ({Mb(bytes)}); {Count} on disk, {Mb(Bytes)}");
        return true;
    }

    /// <summary>Takes the state whose tokens the prompt starts with and goes
    /// on past, the longest such: its tokens and its bytes, read here, and
    /// gone from the store. Null when none matches.</summary>
    public (int Mark, ChunkedBuffer State)? Take(int[] prompt)
    {
        Entry? best;
        lock (_lock)
        {
            best = _entries.Where(e => e.Tokens.Length < prompt.Length && prompt.AsSpan(0, e.Tokens.Length).SequenceEqual(e.Tokens))
                .MaxBy(e => e.Tokens.Length);
            if (best is null)
            {
                return null;
            }
            _entries.Remove(best);
        }
        try
        {
            // One being written is read through a view of its own: the writer reads it too.
            var state = best.Pending?.View() ?? ReadState(best.File!, best.Tokens.Length);
            Delete(best.File);
            Interlocked.Increment(ref Restored);
            return (best.Tokens.Length, state);
        }
        catch (Exception e) when (e is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            _log?.Invoke($"kv: could not read a paged-out state: {e.Message}");
            Delete(best.File);
            return null;
        }
    }

    /// <summary>Deletes the states unused for longer than the expiry time.</summary>
    public void Sweep(DateTime now)
    {
        if (_expire is not { } expire)
        {
            return;
        }
        List<Entry> old;
        lock (_lock)
        {
            old = _entries.Where(e => now - e.LastUsed > expire).ToList();
            foreach (var e in old)
            {
                Remove(e);
                Expired++;
            }
        }
        if (old.Count > 0)
        {
            _log?.Invoke($"kv: {old.Count} paged-out state(s) expired; {Count} on disk, {Mb(Bytes)}");
        }
    }

    // Past the budget, the least recently used go first. Under the lock.
    private void Trim()
    {
        long total = _entries.Sum(e => e.Bytes);
        foreach (var e in _entries.OrderBy(e => e.LastUsed).ToList())
        {
            if (total <= _budget)
            {
                break;
            }
            total -= e.Bytes;
            Remove(e);
            Evicted++;
        }
    }

    // Under the lock: the file goes now, or once it is written.
    private void Remove(Entry e)
    {
        _entries.Remove(e);
        e.Pending = null;
        Delete(e.File);
    }

    private void Write(Entry entry)
    {
        ChunkedBuffer? state;
        lock (_lock)
        {
            state = entry.Pending;
            if (state is null || !_entries.Contains(entry))
            {
                return;   // taken or removed before it was written
            }
        }
        string file = Path.Combine(_dir, Guid.NewGuid().ToString("N") + Extension);
        string temp = file + ".tmp";
        try
        {
            using (var fs = new FileStream(temp, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1 << 20, FileOptions.SequentialScan))
            {
                WriteHeader(fs, entry.Tokens, entry.LastUsed);
                state.View().CopyTo(fs, 4 << 20);
            }
            File.Move(temp, file);
            lock (_lock)
            {
                if (_entries.Contains(entry))
                {
                    entry.File = file;
                    entry.Pending = null;
                    return;
                }
            }
            Delete(file);   // taken or removed while it was written
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            _log?.Invoke($"kv: could not write a paged-out state: {e.Message}");
            Delete(temp);
            lock (_lock)
            {
                if (_entries.Contains(entry))
                {
                    Remove(entry);
                }
            }
        }
    }

    private void Load()
    {
        foreach (var temp in Directory.EnumerateFiles(_dir, "*" + Extension + ".tmp"))
        {
            Delete(temp);
        }
        int kept = 0, dropped = 0;
        var now = DateTime.UtcNow;
        foreach (var file in Directory.EnumerateFiles(_dir, "*" + Extension))
        {
            try
            {
                using var fs = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read);
                var (tokens, lastUsed, fingerprint) = ReadHeader(fs);
                if (fingerprint != _fingerprint || (_expire is { } expire && now - lastUsed > expire))
                {
                    fs.Dispose();
                    Delete(file);
                    dropped++;
                    continue;
                }
                _entries.Add(new Entry { Tokens = tokens, Bytes = fs.Length, LastUsed = lastUsed, File = file });
                kept++;
            }
            catch (Exception e) when (e is IOException or InvalidDataException or EndOfStreamException)
            {
                Delete(file);
                dropped++;
            }
        }
        lock (_lock)
        {
            Trim();
        }
        if (kept + dropped > 0)
        {
            _log?.Invoke($"kv: {_entries.Count} paged-out state(s) kept from before ({Mb(Bytes)}), {dropped} stale deleted");
        }
    }

    private static long HeaderBytes(int tokens) => Magic.Length + 4 + 64 + 8 + 4 + 4L * tokens;

    private void WriteHeader(Stream s, int[] tokens, DateTime lastUsed)
    {
        using var w = new BinaryWriter(s, Encoding.UTF8, leaveOpen: true);
        w.Write(Encoding.ASCII.GetBytes(Magic));
        var fp = Encoding.ASCII.GetBytes(_fingerprint.PadRight(64)[..64]);
        w.Write(fp.Length);
        w.Write(fp);
        w.Write(lastUsed.ToUniversalTime().Ticks);
        w.Write(tokens.Length);
        foreach (int t in tokens)
        {
            w.Write(t);
        }
    }

    private static (int[] Tokens, DateTime LastUsed, string Fingerprint) ReadHeader(Stream s)
    {
        using var r = new BinaryReader(s, Encoding.UTF8, leaveOpen: true);
        if (Encoding.ASCII.GetString(r.ReadBytes(Magic.Length)) != Magic)
        {
            throw new InvalidDataException("not a paged-out state");
        }
        int fpLength = r.ReadInt32();
        if (fpLength is < 0 or > 256)
        {
            throw new InvalidDataException("bad fingerprint");
        }
        string fp = Encoding.ASCII.GetString(r.ReadBytes(fpLength)).TrimEnd();
        var lastUsed = new DateTime(r.ReadInt64(), DateTimeKind.Utc);
        int count = r.ReadInt32();
        if (count <= 0 || count > (s.Length - s.Position) / 4)
        {
            throw new InvalidDataException("bad token count");
        }
        var tokens = new int[count];
        for (int i = 0; i < count; i++)
        {
            tokens[i] = r.ReadInt32();
        }
        return (tokens, lastUsed, fp);
    }

    private ChunkedBuffer ReadState(string file, int tokens)
    {
        using var fs = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);
        var (stored, _, fingerprint) = ReadHeader(fs);
        if (fingerprint != _fingerprint || stored.Length != tokens)
        {
            throw new InvalidDataException("the file changed under the store");
        }
        var state = new ChunkedBuffer();
        fs.CopyTo(state, 4 << 20);
        state.Position = 0;
        return state;
    }

    private static void Delete(string? file)
    {
        if (file is null)
        {
            return;
        }
        try
        {
            File.Delete(file);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
        }
    }

    private static string Mb(long bytes) => $"{bytes / (1024.0 * 1024):F0} MiB";

    public void Dispose()
    {
        _sweep.Dispose();
        Task writes;
        lock (_lock)
        {
            writes = _writes;
        }
        writes.Wait(TimeSpan.FromSeconds(30));
    }
}

/// <summary>A growable in-memory stream of 16 MiB chunks: holds a saved state
/// past the 2 GiB a <see cref="MemoryStream"/> can.</summary>
public sealed class ChunkedBuffer : Stream
{
    private const int Chunk = 16 << 20;
    private readonly List<byte[]> _chunks;
    private long _length, _position;

    public ChunkedBuffer()
    {
        _chunks = new();
    }

    private ChunkedBuffer(List<byte[]> chunks, long length)
    {
        _chunks = chunks;
        _length = length;
    }

    /// <summary>A reader over the same bytes from the start, with a position
    /// of its own (for reading on another thread; neither may write then).</summary>
    public ChunkedBuffer View() => new(_chunks, _length);

    public override bool CanRead => true;
    public override bool CanSeek => true;
    public override bool CanWrite => true;
    public override long Length => _length;

    public override long Position
    {
        get => _position;
        set => _position = Math.Clamp(value, 0, _length);
    }

    public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

    public override int Read(Span<byte> buffer)
    {
        int total = 0;
        while (buffer.Length > 0 && _position < _length)
        {
            int at = (int)(_position % Chunk);
            int n = (int)Math.Min(Math.Min(buffer.Length, Chunk - at), _length - _position);
            _chunks[(int)(_position / Chunk)].AsSpan(at, n).CopyTo(buffer);
            buffer = buffer[n..];
            _position += n;
            total += n;
        }
        return total;
    }

    public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));

    public override void Write(ReadOnlySpan<byte> buffer)
    {
        while (buffer.Length > 0)
        {
            int index = (int)(_position / Chunk), at = (int)(_position % Chunk);
            while (_chunks.Count <= index)
            {
                _chunks.Add(new byte[Chunk]);
            }
            int n = Math.Min(buffer.Length, Chunk - at);
            buffer[..n].CopyTo(_chunks[index].AsSpan(at));
            buffer = buffer[n..];
            _position += n;
            _length = Math.Max(_length, _position);
        }
    }

    public override long Seek(long offset, SeekOrigin origin) => Position = origin switch
    {
        SeekOrigin.Begin => offset,
        SeekOrigin.Current => _position + offset,
        _ => _length + offset,
    };

    public override void SetLength(long value) => throw new NotSupportedException();

    public override void Flush()
    {
    }
}
