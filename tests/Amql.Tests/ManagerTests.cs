using System.IO.Compression;
using System.Net;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Amql.Manager;
using Xunit;

namespace Amql.Tests;

/// <summary>The manager's pieces that need no model: versions, its config, reading the CLI's
/// help and nvidia-smi, and the updater against a fake GitHub, which must install a release
/// only when its archive matches its checksum and holds an install.</summary>
public class ManagerTests
{
    [Theory]
    [InlineData("v1.2.3", "1.2.2", 1)]
    [InlineData("1.10.0", "1.9.9", 1)]
    [InlineData("1.2.0", "1.2.0-rc.1", 1)]
    [InlineData("1.2.0-rc.2", "1.2.0-rc.1", 1)]
    [InlineData("1.2", "1.2.0", 0)]
    [InlineData("1.0.0+04b9a6a", "1.0.0", 0)]
    public void Versions_Compare_As_Releases(string a, string b, int sign)
    {
        Assert.Equal(sign, Math.Sign(ReleaseVersion.Parse(a)!.CompareTo(ReleaseVersion.Parse(b))));
    }

    [Fact]
    public void A_Config_That_Listens_Beyond_The_Machine_Needs_A_Key()
    {
        using var dir = new TempDir();
        string path = Path.Combine(dir.Path, "manager.json");
        File.WriteAllText(path, """{ "urls": "http://0.0.0.0:8100" }""");
        string? saved = Environment.GetEnvironmentVariable("AMQL_MANAGER_API_KEY");
        Environment.SetEnvironmentVariable("AMQL_MANAGER_API_KEY", null);
        try
        {
            Assert.Contains("api_key", Assert.Throws<ManagerException>(() => ManagerConfig.Load(path)).Message);
            File.WriteAllText(path, """{ "urls": "http://0.0.0.0:8100", "api_key": "k", /* a comment */ }""");
            Assert.Equal("k", ManagerConfig.Load(path).ApiKey);
            File.WriteAllText(path, """{ "urls": "http://127.0.0.1:8100", "modles": [] }""");
            Assert.Throws<ManagerException>(() => ManagerConfig.Load(path));
        }
        finally
        {
            Environment.SetEnvironmentVariable("AMQL_MANAGER_API_KEY", saved);
        }
        Assert.True(ManagerConfig.LoopbackOnly("http://127.0.0.1:1;http://localhost:2"));
        Assert.False(ManagerConfig.LoopbackOnly("http://+:8100"));
    }

    [Fact]
    public void The_Cli_Help_Gives_Its_Commands()
    {
        var help = new[]
        {
            "Commands:",
            "  amql-cli encode <model-dir | hf-url | owner/name> --out <container-dir> [--no-vision]   map + materialise",
            "                  [--dtype Q4_0|Q8_0|FP4|BF16|FP32]   single-encoding slim import",
            "  amql-cli context bench <container> --text <file|dir>[;.] [--lengths 8K,32K,128K,256K]",
            "                  how well the model uses a context of each length",
            "  amql-cli tokens --tokenizer <checkpoint-dir> \"text\"",
            "  amql-cli verify <container-dir>                     integrity + readiness",
        };
        var commands = CliJobs.ParseHelp(help);
        Assert.Equal(new[] { "encode", "context bench", "tokens", "verify" }, commands.Select(c => c.Name));
        Assert.Equal(2, commands[0].Usage.Count);
    }

    [Fact]
    public void Nvidia_Smi_Rows_Are_Read()
    {
        var gpus = ModelHostApi.ParseGpus("0, NVIDIA GeForce RTX 5090, 32607, 4230, 27958, 3\n1, NVIDIA GB10, [N/A], [N/A], [N/A], 0\n");
        Assert.Equal(2, gpus.Count);
        Assert.Equal(32607, gpus[0].MemoryTotalMb);
        Assert.Null(gpus[1].MemoryTotalMb);
        Assert.Equal("NVIDIA GB10", gpus[1].Name);
    }

    [Fact]
    public void Sums_Are_Read_In_Sha256sum_Form()
    {
        string hash = new string('a', 64);
        Assert.Equal(hash, Updater.Checksum($"{new string('b', 64)}  other.zip\n{hash} *amql-1.2.0-linux-arm64.tar.gz\n", "amql-1.2.0-linux-arm64.tar.gz"));
        Assert.Null(Updater.Checksum($"{hash}  x.zip", "y.zip"));
    }

    [Fact]
    public async Task An_Update_Installs_A_Checked_Release_And_Keeps_The_One_Before()
    {
        using var dir = new TempDir();
        string root = dir.Path;
        Directory.CreateDirectory(Layout.VersionDir(root, "1.0.0"));
        Layout.MakeCurrent(root, "1.0.0");
        var hub = new FakeGitHub(Archive(Updater.Executables), "1.1.0");
        var updater = new Updater(new UpdateSettings { ApiBase = FakeGitHub.Base }, hub, root);
        Assert.Equal("1.0.0", updater.Status.Current);

        var release = await updater.CheckAsync(CancellationToken.None);
        Assert.NotNull(release);
        Assert.True(updater.Status.Available);
        Assert.Equal("1.1.0", release!.Version.ToString());

        Assert.Equal("1.1.0", await updater.ApplyAsync(release, CancellationToken.None));
        Assert.Equal("1.1.0", Layout.ReadCurrent(root));
        Assert.Equal("1.0.0", Layout.ReadPrevious(root));
        Assert.True(File.Exists(Layout.Executable(Layout.VersionDir(root, "1.1.0"), "amql-server")));

        Assert.Equal("1.0.0", updater.Rollback());
        Assert.Equal("1.0.0", Layout.ReadCurrent(root));
    }

    [Fact]
    public async Task A_Release_Whose_Archive_Does_Not_Match_Its_Sum_Is_Not_Installed()
    {
        using var dir = new TempDir();
        Layout.MakeCurrent(dir.Path, "1.0.0");
        var hub = new FakeGitHub(Archive(Updater.Executables), "1.1.0") { Tamper = true };
        var updater = new Updater(new UpdateSettings { ApiBase = FakeGitHub.Base }, hub, dir.Path);
        var release = await updater.CheckAsync(CancellationToken.None);
        var refused = await Assert.ThrowsAsync<ManagerException>(() => updater.ApplyAsync(release!, CancellationToken.None));
        Assert.Contains("SHA-256", refused.Message);
        Assert.Equal("1.0.0", Layout.ReadCurrent(dir.Path));
        Assert.False(Directory.Exists(Layout.VersionDir(dir.Path, "1.1.0")));
    }

    [Fact]
    public async Task A_Release_Without_Sums_Or_Without_An_Install_Is_Not_Installed()
    {
        using var dir = new TempDir();
        Layout.MakeCurrent(dir.Path, "1.0.0");
        var noSums = new Updater(new UpdateSettings { ApiBase = FakeGitHub.Base },
            new FakeGitHub(Archive(Updater.Executables), "1.1.0") { WithoutSums = true }, dir.Path);
        var release = await noSums.CheckAsync(CancellationToken.None);
        Assert.Contains("SHA256SUMS", (await Assert.ThrowsAsync<ManagerException>(() => noSums.ApplyAsync(release!, CancellationToken.None))).Message);

        var partial = new Updater(new UpdateSettings { ApiBase = FakeGitHub.Base },
            new FakeGitHub(Archive(new[] { "amql-cli" }), "1.1.0"), dir.Path);
        release = await partial.CheckAsync(CancellationToken.None);
        Assert.Contains("holds no", (await Assert.ThrowsAsync<ManagerException>(() => partial.ApplyAsync(release!, CancellationToken.None))).Message);
        Assert.Equal("1.0.0", Layout.ReadCurrent(dir.Path));
    }

    [Fact]
    public async Task No_Newer_Release_Means_No_Update()
    {
        using var dir = new TempDir();
        Layout.MakeCurrent(dir.Path, "1.1.0");
        var updater = new Updater(new UpdateSettings { ApiBase = FakeGitHub.Base }, new FakeGitHub(Archive(Updater.Executables), "1.1.0"), dir.Path);
        Assert.Null(await updater.CheckAsync(CancellationToken.None));
        Assert.False(updater.Status.Available);
        Assert.Equal("1.1.0", updater.Status.Latest);
    }

    [Fact]
    public void Pruning_Keeps_The_Current_And_Previous_Versions()
    {
        using var dir = new TempDir();
        foreach (string v in new[] { "1.0.0", "1.1.0", "1.2.0" })
        {
            Directory.CreateDirectory(Layout.VersionDir(dir.Path, v));
            Layout.MakeCurrent(dir.Path, v);
        }
        Layout.Prune(dir.Path);
        Assert.Equal(new[] { "1.1.0", "1.2.0" }, Directory.GetDirectories(Path.Combine(dir.Path, "versions")).Select(Path.GetFileName).Order());
    }

    /// <summary>A zip of an install wrapped in one folder, as the packaging script makes it.</summary>
    private static byte[] Archive(IEnumerable<string> executables)
    {
        using var buffer = new MemoryStream();
        using (var zip = new ZipArchive(buffer, ZipArchiveMode.Create, leaveOpen: true))
        {
            foreach (string exe in executables)
            {
                var entry = zip.CreateEntry($"amql/{(OperatingSystem.IsWindows() ? exe + ".exe" : exe)}");
                using var w = new StreamWriter(entry.Open());
                w.Write("binary " + exe);
            }
            using (var w = new StreamWriter(zip.CreateEntry("amql/README.md").Open()))
            {
                w.Write("readme");
            }
        }
        return buffer.ToArray();
    }

    /// <summary>GitHub's releases API and its asset downloads, for one release.</summary>
    private sealed class FakeGitHub(byte[] archive, string version) : HttpMessageHandler
    {
        public const string Base = "https://github.test/api";
        public bool Tamper { get; init; }
        public bool WithoutSums { get; init; }

        private string AssetName => $"amql-{version}-{RuntimeInformation.RuntimeIdentifier}.zip";

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancel)
        {
            string url = request.RequestUri!.ToString();
            if (url.Contains("/releases?"))
            {
                var assets = new List<object>
                {
                    new { name = AssetName, url = $"{Base}/assets/1" },
                    new { name = $"amql-{version}-other-rid.zip", url = $"{Base}/assets/9" },
                };
                if (!WithoutSums)
                {
                    assets.Add(new { name = "SHA256SUMS", url = $"{Base}/assets/2" });
                }
                var releases = new object[]
                {
                    new { tag_name = $"v{version}", draft = false, prerelease = false, published_at = "2026-11-27T10:00:00Z", assets },
                    new { tag_name = "v9.0.0", draft = true, prerelease = false, published_at = "2026-11-27T10:00:00Z", assets },
                    new { tag_name = "v8.0.0-rc.1", draft = false, prerelease = true, published_at = "2026-11-27T10:00:00Z", assets },
                };
                return Task.FromResult(Json(JsonSerializer.Serialize(releases)));
            }
            if (url.EndsWith("/assets/1", StringComparison.Ordinal))
            {
                var bytes = Tamper ? archive.Concat(new byte[] { 0 }).ToArray() : archive;
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(bytes) });
            }
            if (url.EndsWith("/assets/2", StringComparison.Ordinal))
            {
                string sum = Convert.ToHexStringLower(SHA256.HashData(archive));
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent($"{sum}  {AssetName}\n{new string('0', 64)}  amql-{version}-other-rid.zip\n"),
                });
            }
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }

        private static HttpResponseMessage Json(string body) =>
            new(HttpStatusCode.OK) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
    }
}
