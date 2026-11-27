using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Amql.Hf;
using Xunit;

namespace Amql.Tests;

/// <summary>Publishing a folder to the Hub, against a stand-in that speaks the Hub's protocol.</summary>
public class HfUploaderTests
{
    /// <summary>The Hub as far as an upload sees it: it keeps what is committed, and the LFS bytes sent.</summary>
    private sealed class FakeHub : HttpMessageHandler
    {
        public const long Chunk = 1000;
        public readonly List<string> Calls = [];
        public readonly Dictionary<string, (string Blob, string? Lfs)> Tree = new(StringComparer.Ordinal);
        public readonly Dictionary<string, byte[]> Lfs = new(StringComparer.Ordinal);
        public readonly Dictionary<int, byte[]> Parts = [];
        public bool Exists;
        public string? Created;
        public string? Token;

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancel)
        {
            string url = request.RequestUri!.ToString();
            string body = request.Content is null ? "" : request.Method == HttpMethod.Put ? "" : await request.Content.ReadAsStringAsync(cancel);
            Calls.Add($"{request.Method} {request.RequestUri.AbsolutePath}");
            if (request.Headers.Authorization is { } auth)
            {
                Token = auth.Parameter;
            }
            HttpResponseMessage Reply(HttpStatusCode code, JsonNode? json = null) =>
                new(code) { Content = new StringContent(json?.ToJsonString() ?? "", Encoding.UTF8, "application/json") };

            if (url.EndsWith("/api/repos/create"))
            {
                Created = body;
                return Reply(Exists ? HttpStatusCode.Conflict : HttpStatusCode.OK, new JsonObject { ["url"] = "x" });
            }
            if (url.Contains("/tree/"))
            {
                if (!Exists)
                {
                    return Reply(HttpStatusCode.NotFound);
                }
                var list = new JsonArray();
                foreach (var (path, (blob, lfs)) in Tree)
                {
                    var entry = new JsonObject { ["type"] = "file", ["path"] = path, ["oid"] = blob };
                    if (lfs is not null)
                    {
                        entry["lfs"] = new JsonObject { ["oid"] = lfs };
                    }
                    list.Add(entry);
                }
                return Reply(HttpStatusCode.OK, list);
            }
            if (url.Contains("/preupload/"))
            {
                var files = new JsonArray();
                foreach (var f in JsonNode.Parse(body)!["files"]!.AsArray())
                {
                    files.Add(new JsonObject { ["path"] = (string)f!["path"]!, ["uploadMode"] = (long)f["size"]! > 100 ? "lfs" : "regular" });
                }
                return Reply(HttpStatusCode.OK, new JsonObject { ["files"] = files });
            }
            if (url.EndsWith(".git/info/lfs/objects/batch"))
            {
                var o = JsonNode.Parse(body)!["objects"]![0]!;
                string oid = (string)o["oid"]!;
                long size = (long)o["size"]!;
                var obj = new JsonObject { ["oid"] = oid, ["size"] = size };
                if (!Lfs.ContainsKey(oid))
                {
                    var header = new JsonObject { ["chunk_size"] = Chunk.ToString() };
                    for (int p = 1; (p - 1) * Chunk < size; p++)
                    {
                        header[p.ToString()] = $"https://store.test/part/{p}";
                    }
                    obj["actions"] = new JsonObject { ["upload"] = new JsonObject { ["href"] = $"https://store.test/complete/{oid}", ["header"] = header } };
                }
                return Reply(HttpStatusCode.OK, new JsonObject { ["transfer"] = "multipart", ["objects"] = new JsonArray(obj) });
            }
            if (request.Method == HttpMethod.Put && url.StartsWith("https://store.test/part/"))
            {
                Parts[int.Parse(url[(url.LastIndexOf('/') + 1)..])] = await request.Content!.ReadAsByteArrayAsync(cancel);
                var ok = Reply(HttpStatusCode.OK);
                ok.Headers.ETag = new System.Net.Http.Headers.EntityTagHeaderValue($"\"e{Parts.Count}\"");
                return ok;
            }
            if (url.StartsWith("https://store.test/complete/"))
            {
                var done = JsonNode.Parse(body)!;
                Assert.Equal(Parts.Count, done["parts"]!.AsArray().Count);
                var bytes = Parts.OrderBy(p => p.Key).SelectMany(p => p.Value).ToArray();
                Assert.Equal((string)done["oid"]!, Convert.ToHexStringLower(SHA256.HashData(bytes)));
                Lfs[(string)done["oid"]!] = bytes;
                Parts.Clear();
                return Reply(HttpStatusCode.OK);
            }
            if (url.Contains("/commit/"))
            {
                var lines = body.Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(l => JsonNode.Parse(l)!).ToList();
                Assert.Equal("header", (string)lines[0]["key"]!);
                foreach (var line in lines.Skip(1))
                {
                    var v = line["value"]!;
                    string path = (string)v["path"]!;
                    if ((string)line["key"]! == "lfsFile")
                    {
                        Assert.True(Lfs.ContainsKey((string)v["oid"]!), $"{path} committed before its bytes were sent");
                        Tree[path] = ("pointer", (string)v["oid"]!);
                    }
                    else
                    {
                        var content = Convert.FromBase64String((string)v["content"]!);
                        Tree[path] = (Blob(content), null);
                    }
                }
                Exists = true;
                return Reply(HttpStatusCode.OK, new JsonObject { ["commitUrl"] = "https://huggingface.co/me/m/commit/1" });
            }
            return Reply(HttpStatusCode.NotFound);
        }

        public static string Blob(byte[] content) =>
            Convert.ToHexStringLower(SHA1.HashData([.. Encoding.ASCII.GetBytes($"blob {content.Length}\0"), .. content]));
    }

    [Fact]
    public async Task A_Folder_Goes_Up_In_One_Commit_And_Only_What_Changed_Goes_Again()
    {
        using var dir = new TempDir();
        var big = new byte[2500];
        new Random(5).NextBytes(big);
        Directory.CreateDirectory(Path.Combine(dir.Path, "segments"));
        File.WriteAllBytes(Path.Combine(dir.Path, "segments", "stack.bin"), big);
        File.WriteAllText(Path.Combine(dir.Path, "index.json"), "{\"a\":1}");
        File.WriteAllText(Path.Combine(dir.Path, "README.md"), "# model\n");
        File.WriteAllText(Path.Combine(dir.Path, ".amql-download.json"), "{}");   // hidden: not published

        var hub = new FakeHub();
        var options = new HfUploadOptions { Repo = "me/m", Token = "secret" };
        var first = await HfUploader.UploadAsync(dir.Path, options, handler: hub, endpoint: "https://hub.test");
        Assert.Equal("secret", hub.Token);
        Assert.Contains("\"private\":true", hub.Created);
        Assert.Equal(["README.md", "index.json", "segments/stack.bin"], first.Files.Select(f => f.Path));
        Assert.All(first.Files, f => Assert.True(f.Sent));
        Assert.True(first.Files.Single(f => f.Path == "segments/stack.bin").Lfs);
        Assert.Equal(big, hub.Lfs.Values.Single());   // in three parts of 1000, 1000 and 500
        Assert.Equal(FakeHub.Blob(Encoding.UTF8.GetBytes("{\"a\":1}")), hub.Tree["index.json"].Blob);

        // Again, with one file changed: only it is committed, and no bytes are sent twice.
        File.WriteAllText(Path.Combine(dir.Path, "README.md"), "# model, described\n");
        hub.Calls.Clear();
        var second = await HfUploader.UploadAsync(dir.Path, options, handler: hub, endpoint: "https://hub.test");
        Assert.Equal(["README.md"], second.Files.Where(f => f.Sent).Select(f => f.Path));
        Assert.DoesNotContain(hub.Calls, c => c.Contains("lfs/objects/batch") || c.StartsWith("PUT"));

        // And with nothing changed, no commit at all.
        hub.Calls.Clear();
        var third = await HfUploader.UploadAsync(dir.Path, options, handler: hub, endpoint: "https://hub.test");
        Assert.Null(third.CommitUrl);
        Assert.DoesNotContain(hub.Calls, c => c.Contains("/commit/"));
    }
}
