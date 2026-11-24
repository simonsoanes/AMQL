using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Amql.Hf;

/// <summary>A Hugging Face model repository and revision, parsed from what a user typed.</summary>
public sealed record HfModelRef(string Repo, string Revision)
{
    /// <summary>The repository's name without its owner ("Qwen/Qwen3.5-0.8B" → "Qwen3.5-0.8B").</summary>
    public string Name => Repo[(Repo.IndexOf('/') + 1)..];

    public string PageUrl => $"https://huggingface.co/{Repo}" + (Revision == "main" ? "" : $"/tree/{Revision}");

    public override string ToString() => Revision == "main" ? Repo : $"{Repo}@{Revision}";
}

/// <summary>One file being fetched, for progress reports.</summary>
public sealed record HfDownloadProgress(string File, int FileIndex, int FileCount, long FileDone, long FileTotal,
    long Done, long Total, double BytesPerSecond);

/// <summary>What a download fetched and where it lives.</summary>
public sealed record HfDownloadResult(string Directory, HfModelRef Model, string CommitSha, int Files, int Fetched, long Bytes);

/// <summary>
/// Downloads a model checkpoint from the Hugging Face Hub into a local folder
/// that <see cref="ModelToContainer.Encode"/> reads like any other checkpoint
/// directory.
/// <list type="bullet">
/// <item>The file list comes from the Hub's model API. Weights in formats the
/// encoder does not read (PyTorch <c>.bin</c> when safetensors exist, TF, Flax,
/// ONNX, GGUF, OpenVINO, Core ML, <c>original/</c> consolidated checkpoints) are
/// skipped unless <c>allFiles</c> is set. A <c>.pt</c> beside safetensors, such
/// as Von's <c>option_marker.pt</c> head, is kept.</item>
/// <item>Large (LFS) files are checked against the SHA-256 the Hub publishes. A
/// file already verified is not fetched again, and a partly downloaded file
/// resumes where it stopped.</item>
/// <item>A token for gated or private models comes from the caller,
/// <c>HF_TOKEN</c> / <c>HUGGING_FACE_HUB_TOKEN</c>, or the token file
/// <c>huggingface-cli login</c> writes.</item>
/// </list>
/// </summary>
public static class HfDownloader
{
    private const string Endpoint = "https://huggingface.co";
    private const string StateFile = ".amql-download.json";

    private static readonly Regex RepoId = new(@"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$", RegexOptions.Compiled);

    /// <summary>The folder downloads go to by default: <c>AMQL_HF_CACHE</c>, else
    /// <c>%LOCALAPPDATA%\AMQL\hf</c> (<c>~/.cache/amql/hf</c> elsewhere).</summary>
    public static string DefaultCacheRoot =>
        Environment.GetEnvironmentVariable("AMQL_HF_CACHE") is { Length: > 0 } custom
            ? custom
            : OperatingSystem.IsWindows()
                ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "AMQL", "hf")
                : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "amql", "hf");

    /// <summary>
    /// Reads a model reference from a Hub URL (<c>https://huggingface.co/owner/name</c>, optionally
    /// <c>/tree/&lt;revision&gt;</c>), <c>hf://owner/name</c>, or a bare <c>owner/name</c> that is not
    /// an existing local path. <c>@revision</c> may follow either of the last two.
    /// Returns null for anything else, so callers can treat it as a local directory.
    /// </summary>
    public static HfModelRef? TryParse(string source, string? revision = null)
    {
        string s = source.Trim();
        string? rev = revision;
        if (s.StartsWith("https://huggingface.co/", StringComparison.OrdinalIgnoreCase) ||
            s.StartsWith("http://huggingface.co/", StringComparison.OrdinalIgnoreCase) ||
            s.StartsWith("https://hf.co/", StringComparison.OrdinalIgnoreCase))
        {
            var parts = new Uri(s).AbsolutePath.Trim('/').Split('/', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length >= 1 && parts[0] is "models")
            {
                parts = parts[1..];
            }
            if (parts.Length < 2 || parts[0] is "datasets" or "spaces")
            {
                return null;
            }
            // owner/name/tree/<rev> or owner/name/resolve/<rev>/<file>
            if (parts.Length >= 4 && parts[2] is "tree" or "blob" or "resolve")
            {
                rev ??= Uri.UnescapeDataString(parts[3]);
            }
            return new HfModelRef($"{parts[0]}/{parts[1]}", rev ?? "main");
        }
        bool explicitHub = s.StartsWith("hf://", StringComparison.OrdinalIgnoreCase) || s.StartsWith("hf:", StringComparison.OrdinalIgnoreCase);
        if (explicitHub)
        {
            s = s[(s.IndexOf(':') + 1)..].TrimStart('/');
        }
        int at = s.LastIndexOf('@');
        if (at > 0)
        {
            rev ??= s[(at + 1)..];
            s = s[..at];
        }
        if (!RepoId.IsMatch(s) || (!explicitHub && (Directory.Exists(source) || File.Exists(source))))
        {
            return null;
        }
        return new HfModelRef(s, rev ?? "main");
    }

    /// <summary>The token to send: the given one, the environment, or the Hugging Face CLI's token file.</summary>
    public static string? ResolveToken(string? given = null)
    {
        if (!string.IsNullOrWhiteSpace(given))
        {
            return given.Trim();
        }
        foreach (var name in new[] { "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN" })
        {
            if (Environment.GetEnvironmentVariable(name) is { Length: > 0 } t)
            {
                return t.Trim();
            }
        }
        string? home = Environment.GetEnvironmentVariable("HF_HOME");
        var candidates = new List<string>();
        if (home is { Length: > 0 })
        {
            candidates.Add(Path.Combine(home, "token"));
        }
        candidates.Add(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface", "token"));
        foreach (var file in candidates)
        {
            try
            {
                if (File.Exists(file) && File.ReadAllText(file).Trim() is { Length: > 0 } t)
                {
                    return t;
                }
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                // unreadable token file: carry on without it
            }
        }
        return null;
    }

    /// <summary>Whether a file belongs in the download (see the class summary).</summary>
    public static bool Wanted(string path, IReadOnlyCollection<string> allFiles)
    {
        string p = path.Replace('\\', '/');
        string lower = p.ToLowerInvariant();
        string[] skipFolders = { "original/", "onnx/", "openvino/", "coreml/", "tflite/", "gguf/", "metal/" };
        if (skipFolders.Any(f => lower.StartsWith(f, StringComparison.Ordinal) || lower.Contains("/" + f, StringComparison.Ordinal)))
        {
            return false;
        }
        string[] skipExtensions = { ".h5", ".msgpack", ".onnx", ".onnx_data", ".gguf", ".ot", ".mlmodel", ".tflite", ".pth", ".ckpt", ".mlpackage" };
        if (skipExtensions.Any(e => lower.EndsWith(e, StringComparison.Ordinal)))
        {
            return false;
        }
        bool haveSafetensors = allFiles.Any(f => f.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase));
        if (haveSafetensors && (lower.EndsWith(".bin", StringComparison.Ordinal) || lower.EndsWith(".bin.index.json", StringComparison.Ordinal)))
        {
            return false;   // the same weights again as PyTorch pickles
        }
        return true;
    }

    /// <summary>
    /// Downloads <paramref name="model"/> into <paramref name="directory"/> (default: a folder under
    /// <see cref="DefaultCacheRoot"/> named after the repository and revision) and returns it.
    /// </summary>
    public static async Task<HfDownloadResult> DownloadAsync(
        HfModelRef model,
        string? directory = null,
        string? token = null,
        bool allFiles = false,
        Action<string>? log = null,
        Action<HfDownloadProgress>? progress = null,
        CancellationToken cancel = default)
    {
        token = ResolveToken(token);
        using var http = new HttpClient(new SocketsHttpHandler
        {
            AutomaticDecompression = DecompressionMethods.None,
            PooledConnectionLifetime = TimeSpan.FromMinutes(5),
        })
        {
            Timeout = Timeout.InfiniteTimeSpan,
        };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("amql/0.x");

        var (commit, files) = await ListFilesAsync(http, model, token, cancel);
        var wanted = files.Where(f => allFiles || Wanted(f.Path, files.Select(x => x.Path).ToList())).ToList();
        if (!wanted.Any(f => f.Path.Equals("config.json", StringComparison.OrdinalIgnoreCase)))
        {
            log?.Invoke($"warning: {model} has no config.json at its root; it may not be a transformers checkpoint");
        }
        directory ??= DefaultDirectory(model);
        Directory.CreateDirectory(directory);

        var state = LoadState(directory);
        long total = wanted.Sum(f => f.Size);
        long doneBefore = 0;
        int fetched = 0;
        var clock = System.Diagnostics.Stopwatch.StartNew();
        long bytesThisRun = 0;
        log?.Invoke($"{model}: {wanted.Count} files, {FormatBytes(total)} (commit {commit[..Math.Min(8, commit.Length)]}) → {directory}");

        for (int i = 0; i < wanted.Count; i++)
        {
            cancel.ThrowIfCancellationRequested();
            var file = wanted[i];
            string target = Path.Combine(directory, file.Path.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            string stamp = file.Sha256 ?? file.Oid ?? "";
            if (File.Exists(target) && new FileInfo(target).Length == file.Size &&
                state.TryGetValue(file.Path, out var known) && known == stamp)
            {
                doneBefore += file.Size;
                continue;   // already here and verified
            }

            long baseline = doneBefore;
            int index = i;
            await FetchAsync(http, model, commit, file, target, token, (fileDone, rate) =>
                progress?.Invoke(new HfDownloadProgress(file.Path, index + 1, wanted.Count, fileDone, file.Size,
                    baseline + fileDone, total, rate)), cancel);
            bytesThisRun += file.Size;
            doneBefore += file.Size;
            fetched++;
            state[file.Path] = stamp;
            SaveState(directory, state);
        }

        double seconds = Math.Max(clock.Elapsed.TotalSeconds, 0.001);
        log?.Invoke(fetched == 0
            ? $"{model}: all {wanted.Count} files already downloaded"
            : $"{model}: fetched {fetched} of {wanted.Count} files, {FormatBytes(bytesThisRun)} in {seconds:0.0} s ({FormatBytes((long)(bytesThisRun / seconds))}/s)");
        return new HfDownloadResult(directory, model, commit, wanted.Count, fetched, total);
    }

    private sealed record HubFile(string Path, long Size, string? Sha256, string? Oid);

    private static async Task<(string Commit, List<HubFile> Files)> ListFilesAsync(HttpClient http, HfModelRef model, string? token, CancellationToken cancel)
    {
        string url = $"{Endpoint}/api/models/{model.Repo}/revision/{Uri.EscapeDataString(model.Revision)}?blobs=true";
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (token is not null)
        {
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
        }
        using var response = await http.SendAsync(request, cancel);
        if (!response.IsSuccessStatusCode)
        {
            throw Refusal(response.StatusCode, model, token is not null);
        }
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(cancel));
        var root = doc.RootElement;
        if (root.TryGetProperty("gated", out var gated) && gated.ValueKind == JsonValueKind.String && token is null)
        {
            // Listing works without a token; the files will not. Say so before the first fetch fails.
            throw new HfDownloadException(
                $"{model.Repo} is gated: accept its terms at {model.PageUrl}, then set HF_TOKEN (or run 'huggingface-cli login').");
        }
        string commit = root.TryGetProperty("sha", out var sha) && sha.ValueKind == JsonValueKind.String ? sha.GetString()! : model.Revision;
        var files = new List<HubFile>();
        if (root.TryGetProperty("siblings", out var siblings) && siblings.ValueKind == JsonValueKind.Array)
        {
            foreach (var s in siblings.EnumerateArray())
            {
                string name = s.GetProperty("rfilename").GetString()!;
                long size = s.TryGetProperty("size", out var sz) && sz.ValueKind == JsonValueKind.Number ? sz.GetInt64() : -1;
                string? sha256 = null;
                if (s.TryGetProperty("lfs", out var lfs) && lfs.ValueKind == JsonValueKind.Object)
                {
                    if (lfs.TryGetProperty("sha256", out var h)) sha256 = h.GetString();
                    if (size < 0 && lfs.TryGetProperty("size", out var ls)) size = ls.GetInt64();
                }
                string? oid = s.TryGetProperty("blobId", out var b) ? b.GetString() : null;
                files.Add(new HubFile(name, Math.Max(size, 0), sha256, oid));
            }
        }
        if (files.Count == 0)
        {
            throw new HfDownloadException($"{model} lists no files");
        }
        return (commit, files);
    }

    private static async Task FetchAsync(HttpClient http, HfModelRef model, string commit, HubFile file, string target,
        string? token, Action<long, double> onProgress, CancellationToken cancel)
    {
        string part = target + ".part";
        long have = File.Exists(part) ? new FileInfo(part).Length : 0;
        if (have > file.Size && file.Size > 0)
        {
            File.Delete(part);
            have = 0;
        }
        string url = $"{Endpoint}/{model.Repo}/resolve/{commit}/{string.Join('/', file.Path.Split('/').Select(Uri.EscapeDataString))}";

        for (int attempt = 1; ; attempt++)
        {
            try
            {
                using var request = new HttpRequestMessage(HttpMethod.Get, url);
                if (token is not null)
                {
                    // .NET drops this header when the Hub redirects to its CDN on another host.
                    request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                }
                if (have > 0)
                {
                    request.Headers.Range = new RangeHeaderValue(have, null);
                }
                using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancel);
                if (response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable && have == file.Size)
                {
                    break;      // the part file is already whole
                }
                if (!response.IsSuccessStatusCode)
                {
                    throw Refusal(response.StatusCode, model, token is not null);
                }
                if (have > 0 && response.StatusCode != HttpStatusCode.PartialContent)
                {
                    have = 0;   // the server ignored the range: start over
                }
                await using (var output = new FileStream(part, have > 0 ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20))
                await using (var input = await response.Content.ReadAsStreamAsync(cancel))
                {
                    var buffer = new byte[1 << 20];
                    var clock = System.Diagnostics.Stopwatch.StartNew();
                    long startedAt = have;
                    long lastReport = 0;
                    int n;
                    while ((n = await input.ReadAsync(buffer, cancel)) > 0)
                    {
                        await output.WriteAsync(buffer.AsMemory(0, n), cancel);
                        have += n;
                        if (clock.ElapsedMilliseconds - lastReport >= 250)
                        {
                            lastReport = clock.ElapsedMilliseconds;
                            onProgress(have, (have - startedAt) / Math.Max(clock.Elapsed.TotalSeconds, 0.001));
                        }
                    }
                }
                break;
            }
            catch (Exception e) when (attempt < 5 && !cancel.IsCancellationRequested &&
                                      e is HttpRequestException or IOException or TaskCanceledException)
            {
                // A dropped connection resumes from what is on disk.
                await Task.Delay(TimeSpan.FromSeconds(2 * attempt), cancel);
                have = File.Exists(part) ? new FileInfo(part).Length : 0;
            }
        }

        if (file.Size > 0 && new FileInfo(part).Length != file.Size)
        {
            throw new HfDownloadException($"{file.Path}: got {new FileInfo(part).Length:N0} bytes, expected {file.Size:N0}");
        }
        if (file.Sha256 is { } expected)
        {
            string actual;
            await using (var check = new FileStream(part, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20))
            {
                actual = Convert.ToHexStringLower(await SHA256.HashDataAsync(check, cancel));
            }
            if (!string.Equals(actual, expected, StringComparison.OrdinalIgnoreCase))
            {
                File.Delete(part);
                throw new HfDownloadException($"{file.Path}: SHA-256 {actual} does not match the Hub's {expected}; the download was discarded");
            }
        }
        File.Move(part, target, overwrite: true);
        onProgress(file.Size, 0);
    }

    private static HfDownloadException Refusal(HttpStatusCode status, HfModelRef model, bool hadToken) => status switch
    {
        HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden => new HfDownloadException(hadToken
            ? $"the Hub refused access to {model.Repo} ({(int)status}). If it is gated, accept its terms at {model.PageUrl}; check the token can read it."
            : $"{model.Repo} needs a Hugging Face token ({(int)status}): it is private or gated. Accept its terms at {model.PageUrl}, then set HF_TOKEN (or run 'huggingface-cli login')."),
        HttpStatusCode.NotFound => new HfDownloadException($"{model} was not found on the Hub (check the name and revision; a private repo also answers 404 without a token)"),
        _ => new HfDownloadException($"the Hub answered {(int)status} {status} for {model}"),
    };

    private static Dictionary<string, string> LoadState(string directory)
    {
        try
        {
            string path = Path.Combine(directory, StateFile);
            return File.Exists(path)
                ? JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(path)) ?? new()
                : new();
        }
        catch (Exception e) when (e is IOException or JsonException)
        {
            return new();
        }
    }

    private static void SaveState(string directory, Dictionary<string, string> state) =>
        File.WriteAllText(Path.Combine(directory, StateFile), JsonSerializer.Serialize(state, new JsonSerializerOptions { WriteIndented = true }));

    /// <summary>Where a model downloads to by default: <c>{cache}/{owner}--{name}/{revision}</c>.</summary>
    public static string DefaultDirectory(HfModelRef model) =>
        Path.Combine(DefaultCacheRoot, model.Repo.Replace("/", "--"), Sanitise(model.Revision));

    private static string Sanitise(string s) => string.Concat(s.Select(c => Path.GetInvalidFileNameChars().Contains(c) || c == '/' ? '_' : c));

    public static string FormatBytes(long b) => b switch
    {
        >= 1L << 30 => $"{b / (double)(1L << 30):0.00} GB",
        >= 1L << 20 => $"{b / (double)(1L << 20):0.0} MB",
        >= 1L << 10 => $"{b / 1024.0:0.0} KB",
        _ => $"{b} B",
    };
}

/// <summary>A download the Hub refused or that arrived damaged; the message says what to do.</summary>
public sealed class HfDownloadException : Exception
{
    public HfDownloadException(string message) : base(message)
    {
    }
}
