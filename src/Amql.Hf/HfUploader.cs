using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Amql.Hf;

/// <summary>What <see cref="HfUploader.UploadAsync"/> is asked to do.</summary>
public sealed record HfUploadOptions
{
    /// <summary>The model's id on the Hub, <c>owner/name</c>.</summary>
    public required string Repo { get; init; }

    public string Revision { get; init; } = "main";

    /// <summary>A repository that does not exist yet is made private unless this is false.</summary>
    public bool Private { get; init; } = true;

    public string? Message { get; init; }

    /// <summary>The token to write with; null reads it as downloads do (<see cref="HfDownloader.ResolveToken"/>).</summary>
    public string? Token { get; init; }
}

/// <summary>One file of an upload, and what became of it.</summary>
public sealed record HfUploadedFile(string Path, long Size, string Sha256, bool Lfs, bool Sent);

public sealed record HfUploadResult(string Repo, string Revision, string? CommitUrl, IReadOnlyList<HfUploadedFile> Files)
{
    public string PageUrl => $"https://huggingface.co/{Repo}";
}

/// <summary>
/// Publishes a folder (a container, adapters and its card) as a model on the Hugging Face Hub
/// in one commit, by the Hub's own HTTP protocol, the one <c>huggingface_hub</c> uses:
/// <list type="number">
/// <item>The repository is made if it is not there (private unless told otherwise).</item>
/// <item><c>preupload</c> says which files are stored by LFS.</item>
/// <item>Their bytes go by the LFS batch API, in parts when the Hub asks for parts.
/// Bytes the Hub already holds are not sent again.</item>
/// <item>One commit names every file: small ones inline, LFS ones by SHA-256.</item>
/// </list>
/// Files the revision already holds unchanged are left out of the commit, so running it
/// again sends only what changed. Nothing on the Hub is deleted.
/// </summary>
public static class HfUploader
{
    public const string Endpoint = "https://huggingface.co";
    private const string LfsJson = "application/vnd.git-lfs+json";

    /// <summary>The files of a folder that are published: all of them, but hidden ones and download state.</summary>
    public static IReadOnlyList<(string Path, string Full, long Size)> Files(string dir)
    {
        string root = System.IO.Path.GetFullPath(dir);
        return Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories)
            .Select(f => (Path: System.IO.Path.GetRelativePath(root, f).Replace('\\', '/'), Full: f))
            .Where(f => !f.Path.Split('/').Any(part => part.StartsWith('.')))
            .OrderBy(f => f.Path, StringComparer.Ordinal)
            .Select(f => (f.Path, f.Full, new FileInfo(f.Full).Length))
            .ToList();
    }

    public static async Task<HfUploadResult> UploadAsync(string dir, HfUploadOptions options, Action<string>? log = null,
        HttpMessageHandler? handler = null, string endpoint = Endpoint, CancellationToken cancel = default)
    {
        string token = HfDownloader.ResolveToken(options.Token)
            ?? throw new HfDownloadException("uploading needs a Hugging Face token with write access: set HF_TOKEN (or run 'huggingface-cli login')");
        var parts = options.Repo.Split('/');
        if (parts.Length != 2 || parts.Any(p => p.Length == 0))
        {
            throw new HfDownloadException($"'{options.Repo}' is not a model id (owner/name)");
        }
        using var http = new HttpClient(handler ?? new SocketsHttpHandler { AutomaticDecompression = DecompressionMethods.All }, disposeHandler: handler is null)
        {
            Timeout = Timeout.InfiniteTimeSpan,
        };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("amql/1.0");
        string api = $"{endpoint}/api/models/{options.Repo}";
        string rev = Uri.EscapeDataString(options.Revision);
        HttpRequestMessage Authorised(HttpMethod method, string url, HttpContent? content = null)
        {
            var request = new HttpRequestMessage(method, url) { Content = content };
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
            return request;
        }

        // 1. The repository.
        using (var made = await http.SendAsync(Authorised(HttpMethod.Post, $"{endpoint}/api/repos/create",
            Json(new JsonObject { ["name"] = parts[1], ["organization"] = parts[0], ["private"] = options.Private, ["type"] = "model" })), cancel))
        {
            if (made.StatusCode == HttpStatusCode.Conflict)
            {
                log?.Invoke($"{options.Repo} exists: adding to it");
            }
            else
            {
                await Require(made, "making the repository", cancel);
                log?.Invoke($"made {options.Repo} ({(options.Private ? "private" : "public")})");
            }
        }

        // 2. What is there already, and how each file is stored.
        var files = Files(dir);
        if (files.Count == 0)
        {
            throw new HfDownloadException($"'{dir}' has no files to upload");
        }
        var remote = await RemoteAsync(http, Authorised, $"{api}/tree/{rev}?recursive=true", cancel);
        var lfs = new HashSet<string>(StringComparer.Ordinal);
        foreach (var batch in files.Chunk(100))
        {
            var request = new JsonArray();
            foreach (var f in batch)
            {
                request.Add(new JsonObject { ["path"] = f.Path, ["size"] = f.Size, ["sample"] = Convert.ToBase64String(Sample(f.Full)) });
            }
            using var answer = await http.SendAsync(Authorised(HttpMethod.Post, $"{api}/preupload/{rev}", Json(new JsonObject { ["files"] = request })), cancel);
            await Require(answer, "asking how files are stored", cancel);
            var modes = JsonNode.Parse(await answer.Content.ReadAsStringAsync(cancel))?["files"]?.AsArray() ?? [];
            foreach (var m in modes)
            {
                if ((string?)m?["uploadMode"] == "lfs")
                {
                    lfs.Add((string)m!["path"]!);
                }
            }
        }

        // 3. Each file's identity; LFS bytes the Hub lacks.
        var results = new List<HfUploadedFile>();
        var commit = new List<JsonObject>();
        foreach (var f in files)
        {
            bool isLfs = lfs.Contains(f.Path);
            log?.Invoke($"hashing {f.Path} ({f.Size / 1e9:F2} GB)");
            string sha256 = await Sha256Async(f.Full, cancel);
            bool same = remote.TryGetValue(f.Path, out var there) &&
                (isLfs ? there.Lfs == sha256 : there.Blob == GitBlobSha1(f.Full, f.Size));
            results.Add(new HfUploadedFile(f.Path, f.Size, sha256, isLfs, !same));
            if (same)
            {
                continue;
            }
            if (isLfs)
            {
                await SendLfsAsync(http, Authorised, endpoint, options, f.Full, f.Path, f.Size, sha256, log, cancel);
                commit.Add(new JsonObject { ["key"] = "lfsFile", ["value"] = new JsonObject { ["path"] = f.Path, ["algo"] = "sha256", ["oid"] = sha256 } });
            }
            else
            {
                commit.Add(new JsonObject { ["key"] = "file", ["value"] = new JsonObject
                {
                    ["content"] = Convert.ToBase64String(await File.ReadAllBytesAsync(f.Full, cancel)), ["path"] = f.Path, ["encoding"] = "base64",
                } });
            }
        }
        if (commit.Count == 0)
        {
            log?.Invoke($"{options.Repo}@{options.Revision} already holds every file as it is here: nothing to commit");
            return new HfUploadResult(options.Repo, options.Revision, null, results);
        }

        // 4. One commit.
        var lines = new StringBuilder();
        var header = new JsonObject { ["key"] = "header", ["value"] = new JsonObject
        {
            ["summary"] = options.Message ?? $"Upload {commit.Count} files with AMQL", ["description"] = "",
        } };
        foreach (var line in commit.Prepend(header))
        {
            lines.Append(line.ToJsonString()).Append('\n');
        }
        using var committed = await http.SendAsync(Authorised(HttpMethod.Post, $"{api}/commit/{rev}",
            new StringContent(lines.ToString(), Encoding.UTF8, "application/x-ndjson")), cancel);
        await Require(committed, "committing", cancel);
        string? url = (string?)JsonNode.Parse(await committed.Content.ReadAsStringAsync(cancel))?["commitUrl"];
        log?.Invoke($"committed {commit.Count} files: {url}");
        return new HfUploadResult(options.Repo, options.Revision, url, results);
    }

    /// <summary>What the revision holds: each file's git blob id and, stored by LFS, its SHA-256.</summary>
    private static async Task<Dictionary<string, (string? Blob, string? Lfs)>> RemoteAsync(HttpClient http,
        Func<HttpMethod, string, HttpContent?, HttpRequestMessage> authorised, string url, CancellationToken cancel)
    {
        var found = new Dictionary<string, (string?, string?)>(StringComparer.Ordinal);
        string? next = url;
        while (next is not null)
        {
            using var answer = await http.SendAsync(authorised(HttpMethod.Get, next, null), cancel);
            if (answer.StatusCode == HttpStatusCode.NotFound)
            {
                return found;   // a new repository, or a revision with nothing in it
            }
            await Require(answer, "reading what the repository holds", cancel);
            foreach (var entry in JsonNode.Parse(await answer.Content.ReadAsStringAsync(cancel))?.AsArray() ?? [])
            {
                if ((string?)entry?["type"] == "file")
                {
                    found[(string)entry!["path"]!] = ((string?)entry["oid"], (string?)entry["lfs"]?["oid"]);
                }
            }
            next = NextPage(answer);
        }
        return found;
    }

    private static string? NextPage(HttpResponseMessage answer)
    {
        if (!answer.Headers.TryGetValues("Link", out var links))
        {
            return null;
        }
        foreach (var link in string.Join(",", links).Split(','))
        {
            var bits = link.Split(';');
            if (bits.Length > 1 && bits.Skip(1).Any(b => b.Trim() == "rel=\"next\""))
            {
                return bits[0].Trim().TrimStart('<').TrimEnd('>');
            }
        }
        return null;
    }

    private static async Task SendLfsAsync(HttpClient http, Func<HttpMethod, string, HttpContent?, HttpRequestMessage> authorised,
        string endpoint, HfUploadOptions options, string full, string path, long size, string sha256, Action<string>? log, CancellationToken cancel)
    {
        var body = new JsonObject
        {
            ["operation"] = "upload",
            ["transfers"] = new JsonArray("basic", "multipart"),
            ["objects"] = new JsonArray(new JsonObject { ["oid"] = sha256, ["size"] = size }),
            ["hash_algo"] = "sha256",
            ["ref"] = new JsonObject { ["name"] = $"refs/heads/{options.Revision}" },
        };
        var content = new StringContent(body.ToJsonString(), Encoding.UTF8);
        content.Headers.ContentType = new MediaTypeHeaderValue(LfsJson);
        var request = authorised(HttpMethod.Post, $"{endpoint}/{options.Repo}.git/info/lfs/objects/batch", content);
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue(LfsJson));
        using var answer = await http.SendAsync(request, cancel);
        await Require(answer, $"asking where to send {path}", cancel);
        var reply = JsonNode.Parse(await answer.Content.ReadAsStringAsync(cancel))!;
        var obj = reply["objects"]?.AsArray().FirstOrDefault() ?? throw new HfDownloadException($"the Hub said nothing of {path}");
        if (obj["error"] is { } error)
        {
            throw new HfDownloadException($"the Hub refused {path}: {error["message"] ?? error.ToJsonString()}");
        }
        if (obj["actions"]?["upload"] is not JsonObject upload)
        {
            log?.Invoke($"{path}: the Hub has its bytes already");
            return;
        }
        string href = (string)upload["href"]!;
        var headers = upload["header"] as JsonObject ?? [];
        if (headers["chunk_size"] is { } chunkNode)
        {
            long chunk = long.Parse(chunkNode.ToString(), System.Globalization.CultureInfo.InvariantCulture);
            var urls = headers.Where(h => int.TryParse(h.Key, out _)).OrderBy(h => int.Parse(h.Key)).Select(h => (Part: int.Parse(h.Key), Url: (string)h.Value!)).ToList();
            var etags = new JsonArray();
            await using var stream = File.OpenRead(full);
            foreach (var (part, partUrl) in urls)
            {
                long from = (long)(part - 1) * chunk;
                int length = (int)Math.Min(chunk, size - from);
                var buffer = new byte[length];
                stream.Position = from;
                await stream.ReadExactlyAsync(buffer, cancel);
                using var put = await http.PutAsync(partUrl, new ByteArrayContent(buffer), cancel);
                await Require(put, $"sending part {part} of {urls.Count} of {path}", cancel);
                string etag = put.Headers.ETag?.Tag ?? (put.Headers.TryGetValues("ETag", out var e) ? e.First() : "");
                if (etag.Length == 0)
                {
                    throw new HfDownloadException($"part {part} of {path} came back with no ETag");
                }
                etags.Add(new JsonObject { ["partNumber"] = part, ["etag"] = etag });
                log?.Invoke($"{path}: part {part} of {urls.Count} sent");
            }
            using var done = await http.PostAsync(href, Json(new JsonObject { ["oid"] = sha256, ["parts"] = etags }), cancel);
            await Require(done, $"completing {path}", cancel);
        }
        else
        {
            await using var stream = File.OpenRead(full);
            var put = new HttpRequestMessage(HttpMethod.Put, href) { Content = new StreamContent(stream) };
            put.Content.Headers.ContentLength = size;
            foreach (var (name, value) in headers)
            {
                put.Headers.TryAddWithoutValidation(name, value?.ToString());
            }
            using var sent = await http.SendAsync(put, cancel);
            await Require(sent, $"sending {path}", cancel);
            log?.Invoke($"{path}: sent");
        }
        if (obj["actions"]?["verify"] is JsonObject verify)
        {
            var check = new HttpRequestMessage(HttpMethod.Post, (string)verify["href"]!)
            {
                Content = new StringContent(new JsonObject { ["oid"] = sha256, ["size"] = size }.ToJsonString(), Encoding.UTF8),
            };
            check.Content.Headers.ContentType = new MediaTypeHeaderValue(LfsJson);
            foreach (var (name, value) in verify["header"] as JsonObject ?? [])
            {
                check.Headers.TryAddWithoutValidation(name, value?.ToString());
            }
            using var verified = await http.SendAsync(check, cancel);
            await Require(verified, $"verifying {path}", cancel);
        }
    }

    private static StringContent Json(JsonObject body) => new(body.ToJsonString(), Encoding.UTF8, "application/json");

    private static async Task Require(HttpResponseMessage answer, string what, CancellationToken cancel)
    {
        if (answer.IsSuccessStatusCode)
        {
            return;
        }
        string text = await answer.Content.ReadAsStringAsync(cancel);
        string why = (int)answer.StatusCode switch
        {
            401 => "the token was refused",
            403 => "the token may not write here (it needs write access, and the owner must be you or an organisation you write to)",
            _ => answer.ReasonPhrase ?? "",
        };
        throw new HfDownloadException($"{what}: {(int)answer.StatusCode} {why}{(text.Length > 0 ? $": {text[..Math.Min(text.Length, 300)]}" : "")}");
    }

    private static byte[] Sample(string full)
    {
        using var f = File.OpenRead(full);
        var buffer = new byte[Math.Min(512, f.Length)];
        f.ReadExactly(buffer);
        return buffer;
    }

    private static async Task<string> Sha256Async(string full, CancellationToken cancel)
    {
        await using var f = new FileStream(full, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(f, cancel));
    }

    /// <summary>A file's git blob id: SHA-1 of "blob &lt;size&gt;\0" and its bytes.</summary>
    internal static string GitBlobSha1(string full, long size)
    {
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA1);
        sha.AppendData(Encoding.ASCII.GetBytes($"blob {size}\0"));
        using var f = File.OpenRead(full);
        var buffer = new byte[1 << 20];
        int n;
        while ((n = f.Read(buffer)) > 0)
        {
            sha.AppendData(buffer, 0, n);
        }
        return Convert.ToHexStringLower(sha.GetHashAndReset());
    }
}
