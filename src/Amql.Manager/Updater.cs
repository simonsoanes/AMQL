using System.Formats.Tar;
using System.IO.Compression;
using System.Net.Http.Headers;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;

namespace Amql.Manager;

/// <summary>A release version: numbers, then an optional prerelease label, as in <c>v1.4.0-rc.1</c>.</summary>
public sealed record ReleaseVersion(int[] Numbers, string? Label) : IComparable<ReleaseVersion>
{
    public static ReleaseVersion? Parse(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return null;
        }
        string s = text.Trim().TrimStart('v', 'V');
        int plus = s.IndexOf('+');
        if (plus >= 0)
        {
            s = s[..plus];
        }
        string? label = null;
        int dash = s.IndexOf('-');
        if (dash >= 0)
        {
            label = s[(dash + 1)..];
            s = s[..dash];
        }
        var parts = s.Split('.');
        var numbers = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            if (!int.TryParse(parts[i], out numbers[i]))
            {
                return null;
            }
        }
        return new ReleaseVersion(numbers, string.IsNullOrEmpty(label) ? null : label);
    }

    public int CompareTo(ReleaseVersion? other)
    {
        if (other is null)
        {
            return 1;
        }
        for (int i = 0; i < Math.Max(Numbers.Length, other.Numbers.Length); i++)
        {
            int a = i < Numbers.Length ? Numbers[i] : 0, b = i < other.Numbers.Length ? other.Numbers[i] : 0;
            if (a != b)
            {
                return a.CompareTo(b);
            }
        }
        // A release outranks its prereleases; prereleases compare by label.
        return (Label, other.Label) switch
        {
            (null, null) => 0,
            (null, _) => 1,
            (_, null) => -1,
            _ => string.CompareOrdinal(Label, other.Label),
        };
    }

    public override string ToString() => string.Join('.', Numbers) + (Label is null ? "" : "-" + Label);
}

public sealed record ReleaseInfo(string Tag, ReleaseVersion Version, bool Prerelease, string AssetName, string AssetUrl,
    string? ChecksumsUrl, DateTimeOffset? Published);

public sealed class UpdateStatus
{
    public string Current { get; set; } = "";
    public string? Latest { get; set; }
    public bool Available { get; set; }
    public DateTimeOffset? LastCheck { get; set; }
    public string? LastError { get; set; }
    public string? Applied { get; set; }
    public string? Previous { get; set; }
    public bool Supervised { get; set; }
    public string? Root { get; set; }
    public string Runtime { get; set; } = "";
}

/// <summary>
/// Updates an install from its GitHub releases. A release carries one archive per
/// platform, <c>amql-&lt;version&gt;-&lt;rid&gt;.zip</c> (or <c>.tar.gz</c>), and a
/// <c>SHA256SUMS</c> file listing them (<c>scripts/release/package.ps1</c> makes both).
/// The archive is checked against that list, unpacked beside the running version
/// in <c>&lt;root&gt;/versions/&lt;version&gt;</c>, and made current; the service
/// supervisor then restarts the manager from it, and keeps the version before for
/// a rollback. Nothing outside the root is written, and only an archive whose
/// checksum matches is used.
/// </summary>
public sealed class Updater
{
    public const int RestartExitCode = 75;
    public static readonly string[] Executables = { "amql-manager", "amql-server", "amql-cli" };

    private readonly UpdateSettings _settings;
    private readonly HttpClient _http;
    private readonly string? _root;
    private readonly SemaphoreSlim _gate = new(1, 1);

    public UpdateStatus Status { get; } = new();

    public Updater(UpdateSettings settings, HttpMessageHandler? handler = null, string? root = null, string? current = null)
    {
        _settings = settings;
        _http = handler is null ? new HttpClient() : new HttpClient(handler);
        _http.Timeout = TimeSpan.FromMinutes(30);
        _http.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("amql-manager", "1"));
        if (Environment.GetEnvironmentVariable(settings.TokenVariable) is { Length: > 0 } token)
        {
            _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
        }
        _root = root ?? Environment.GetEnvironmentVariable("AMQL_ROOT");
        Status.Root = _root;
        Status.Supervised = Environment.GetEnvironmentVariable("AMQL_SUPERVISED") == "1";
        Status.Runtime = RuntimeInformation.RuntimeIdentifier;
        Status.Current = current ?? (_root is not null && Layout.ReadCurrent(_root) is { } v ? v : RunningVersion());
        Status.Previous = _root is null ? null : Layout.ReadPrevious(_root);
    }

    public static string RunningVersion()
    {
        string? info = Assembly.GetEntryAssembly()?.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        return ReleaseVersion.Parse(info)?.ToString() ?? "0.0.0";
    }

    /// <summary>Asks GitHub for the newest release with a build for this platform.</summary>
    public async Task<ReleaseInfo?> CheckAsync(CancellationToken cancel)
    {
        await _gate.WaitAsync(cancel).ConfigureAwait(false);
        try
        {
            Status.LastCheck = DateTimeOffset.UtcNow;
            Status.LastError = null;
            var release = await LatestAsync(cancel).ConfigureAwait(false);
            Status.Latest = release?.Version.ToString();
            Status.Available = release is not null && release.Version.CompareTo(ReleaseVersion.Parse(Status.Current)) > 0;
            return Status.Available ? release : null;
        }
        catch (Exception e) when (e is HttpRequestException or JsonException or TaskCanceledException or ManagerException)
        {
            Status.LastError = e.Message;
            return null;
        }
        finally
        {
            _gate.Release();
        }
    }

    private async Task<ReleaseInfo?> LatestAsync(CancellationToken cancel)
    {
        string url = $"{_settings.ApiBase.TrimEnd('/')}/repos/{_settings.Repository}/releases?per_page=20";
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/vnd.github+json"));
        using var response = await _http.SendAsync(request, cancel).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            throw new ManagerException($"GitHub answered {(int)response.StatusCode} for {_settings.Repository}'s releases" +
                (response.StatusCode == System.Net.HttpStatusCode.NotFound ? $" (a private repository needs a token in {_settings.TokenVariable})" : ""));
        }
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(cancel).ConfigureAwait(false));
        string rid = Status.Runtime;
        ReleaseInfo? best = null;
        foreach (var r in doc.RootElement.EnumerateArray())
        {
            if (r.TryGetProperty("draft", out var draft) && draft.GetBoolean())
            {
                continue;
            }
            bool pre = r.TryGetProperty("prerelease", out var p) && p.GetBoolean();
            if (pre && !_settings.Prerelease)
            {
                continue;
            }
            string tag = r.GetProperty("tag_name").GetString() ?? "";
            if (ReleaseVersion.Parse(tag) is not { } version)
            {
                continue;
            }
            string? assetName = null, assetUrl = null, sums = null;
            foreach (var a in r.GetProperty("assets").EnumerateArray())
            {
                string name = a.GetProperty("name").GetString() ?? "";
                string api = a.GetProperty("url").GetString() ?? "";
                if (name == "SHA256SUMS")
                {
                    sums = api;
                }
                else if (name == $"amql-{version}-{rid}.zip" || name == $"amql-{version}-{rid}.tar.gz")
                {
                    (assetName, assetUrl) = (name, api);
                }
            }
            if (assetName is null)
            {
                continue;
            }
            var info = new ReleaseInfo(tag, version, pre, assetName, assetUrl!, sums,
                r.TryGetProperty("published_at", out var at) && at.ValueKind == JsonValueKind.String ? at.GetDateTimeOffset() : null);
            if (best is null || info.Version.CompareTo(best.Version) > 0)
            {
                best = info;
            }
        }
        return best;
    }

    /// <summary>Downloads, checks and unpacks a release and makes it current. The caller
    /// then restarts the manager (under the supervisor, by exiting with <see cref="RestartExitCode"/>).</summary>
    public async Task<string> ApplyAsync(ReleaseInfo release, CancellationToken cancel)
    {
        if (_root is null)
        {
            throw new ManagerException("updates are installed by the service: run under 'amql-manager supervise --root <dir>' (see docs/manager.md)");
        }
        await _gate.WaitAsync(cancel).ConfigureAwait(false);
        try
        {
            if (release.ChecksumsUrl is null)
            {
                throw new ManagerException($"release {release.Tag} has no SHA256SUMS: refusing an archive that cannot be checked");
            }
            string staging = Path.Combine(_root, "staging");
            Directory.CreateDirectory(staging);
            string archive = Path.Combine(staging, release.AssetName);
            string sums = await DownloadTextAsync(release.ChecksumsUrl, cancel).ConfigureAwait(false);
            string expected = Checksum(sums, release.AssetName)
                ?? throw new ManagerException($"SHA256SUMS of {release.Tag} does not list {release.AssetName}");
            await DownloadFileAsync(release.AssetUrl, archive, cancel).ConfigureAwait(false);
            string actual = await HashAsync(archive, cancel).ConfigureAwait(false);
            if (!string.Equals(actual, expected, StringComparison.OrdinalIgnoreCase))
            {
                File.Delete(archive);
                throw new ManagerException($"{release.AssetName}: SHA-256 {actual} is not the release's {expected}; not installed");
            }

            string version = release.Version.ToString();
            string target = Layout.VersionDir(_root, version);
            string partial = target + ".partial";
            if (Directory.Exists(partial))
            {
                Directory.Delete(partial, recursive: true);
            }
            Directory.CreateDirectory(partial);
            Extract(archive, partial);
            string content = SingleTopFolder(partial);
            foreach (string exe in Executables)
            {
                string path = Path.Combine(content, OperatingSystem.IsWindows() ? exe + ".exe" : exe);
                if (!File.Exists(path))
                {
                    Directory.Delete(partial, recursive: true);
                    throw new ManagerException($"{release.AssetName} holds no {Path.GetFileName(path)}; not installed");
                }
                if (!OperatingSystem.IsWindows())
                {
                    File.SetUnixFileMode(path, File.GetUnixFileMode(path) | UnixFileMode.UserExecute | UnixFileMode.GroupExecute | UnixFileMode.OtherExecute);
                }
            }
            if (Directory.Exists(target))
            {
                Directory.Delete(target, recursive: true);
            }
            Directory.Move(content, target);
            if (Directory.Exists(partial))
            {
                Directory.Delete(partial, recursive: true);
            }
            File.Delete(archive);

            Layout.MakeCurrent(_root, version);
            Status.Previous = Layout.ReadPrevious(_root);
            Status.Applied = version;
            Layout.Prune(_root);
            return version;
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>Makes the version before current again; the restart that follows runs it.</summary>
    public string Rollback()
    {
        if (_root is null)
        {
            throw new ManagerException("not running under the supervisor: nothing to roll back");
        }
        string previous = Layout.ReadPrevious(_root) ?? throw new ManagerException("no earlier version is kept");
        if (!Directory.Exists(Layout.VersionDir(_root, previous)))
        {
            throw new ManagerException($"version {previous} is no longer installed");
        }
        Layout.MakeCurrent(_root, previous);
        Status.Applied = previous;
        return previous;
    }

    internal static string? Checksum(string sums, string name)
    {
        foreach (string line in sums.Split('\n'))
        {
            var parts = line.Trim().Split((char[]?)null, 2, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 2 && parts[1].TrimStart('*') == name && parts[0].Length == 64)
            {
                return parts[0];
            }
        }
        return null;
    }

    private async Task<string> DownloadTextAsync(string url, CancellationToken cancel)
    {
        using var request = Asset(url);
        using var response = await _http.SendAsync(request, cancel).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync(cancel).ConfigureAwait(false);
    }

    private async Task DownloadFileAsync(string url, string path, CancellationToken cancel)
    {
        using var request = Asset(url);
        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancel).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        await using var source = await response.Content.ReadAsStreamAsync(cancel).ConfigureAwait(false);
        await using var file = File.Create(path);
        await source.CopyToAsync(file, cancel).ConfigureAwait(false);
    }

    /// <summary>A release asset by its API URL: GitHub sends the bytes for octet-stream
    /// (redirecting to storage), which also works for a private repository with a token.</summary>
    private HttpRequestMessage Asset(string url)
    {
        if (!url.StartsWith("https://", StringComparison.OrdinalIgnoreCase) && !url.StartsWith(_settings.ApiBase, StringComparison.OrdinalIgnoreCase))
        {
            throw new ManagerException($"refusing to download from '{url}': not HTTPS");
        }
        var request = new HttpRequestMessage(HttpMethod.Get, url);
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/octet-stream"));
        return request;
    }

    private static async Task<string> HashAsync(string path, CancellationToken cancel)
    {
        await using var file = File.OpenRead(path);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(file, cancel).ConfigureAwait(false));
    }

    private static void Extract(string archive, string into)
    {
        string root = Path.GetFullPath(into) + Path.DirectorySeparatorChar;
        if (archive.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
        {
            using var zip = ZipFile.OpenRead(archive);
            foreach (var entry in zip.Entries)
            {
                string dest = Path.GetFullPath(Path.Combine(into, entry.FullName));
                if (!dest.StartsWith(root, StringComparison.Ordinal))
                {
                    throw new ManagerException($"archive entry '{entry.FullName}' escapes the install folder");
                }
            }
            zip.ExtractToDirectory(into);
        }
        else
        {
            using var file = File.OpenRead(archive);
            using var gz = new GZipStream(file, CompressionMode.Decompress);
            using var tar = new TarReader(gz);
            while (tar.GetNextEntry() is { } entry)
            {
                string dest = Path.GetFullPath(Path.Combine(into, entry.Name));
                if (!dest.StartsWith(root, StringComparison.Ordinal))
                {
                    throw new ManagerException($"archive entry '{entry.Name}' escapes the install folder");
                }
                if (entry.EntryType is TarEntryType.Directory)
                {
                    Directory.CreateDirectory(dest);
                }
                else if (entry.EntryType is TarEntryType.RegularFile or TarEntryType.V7RegularFile)
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
                    entry.ExtractToFile(dest, overwrite: true);
                }
                else
                {
                    throw new ManagerException($"archive entry '{entry.Name}' is a {entry.EntryType}, which an install does not hold");
                }
            }
        }
    }

    /// <summary>An archive may wrap everything in one folder; the install is what is inside it.</summary>
    private static string SingleTopFolder(string dir)
    {
        var entries = Directory.GetFileSystemEntries(dir);
        return entries.Length == 1 && Directory.Exists(entries[0]) ? entries[0] : dir;
    }
}

/// <summary>
/// The service's folder:
/// <code>
///   &lt;root&gt;/manager.json             the manager's config
///   &lt;root&gt;/current, previous        the version run, and the one before it
///   &lt;root&gt;/versions/&lt;version&gt;/     each install: amql-manager, amql-server, amql-cli, …
///   &lt;root&gt;/supervisor/              the service's own copy of amql-manager
///   &lt;root&gt;/data/                    job logs, model logs, which models were running
/// </code>
/// </summary>
public static class Layout
{
    public static string VersionDir(string root, string version) => Path.Combine(root, "versions", version);

    public static string? ReadCurrent(string root) => Read(Path.Combine(root, "current"));

    public static string? ReadPrevious(string root) => Read(Path.Combine(root, "previous"));

    private static string? Read(string path) => File.Exists(path) ? File.ReadAllText(path).Trim() is { Length: > 0 } v ? v : null : null;

    public static void MakeCurrent(string root, string version)
    {
        if (ReadCurrent(root) is { } current && current != version)
        {
            Write(Path.Combine(root, "previous"), current);
        }
        Write(Path.Combine(root, "current"), version);
    }

    /// <summary>Writes through a temporary file and a rename, so a reader never sees half a version.</summary>
    private static void Write(string path, string text)
    {
        string tmp = path + ".tmp";
        File.WriteAllText(tmp, text);
        File.Move(tmp, path, overwrite: true);
    }

    /// <summary>Keeps the current and previous versions and removes the rest.</summary>
    public static void Prune(string root)
    {
        string versions = Path.Combine(root, "versions");
        if (!Directory.Exists(versions))
        {
            return;
        }
        var keep = new[] { ReadCurrent(root), ReadPrevious(root) };
        foreach (string dir in Directory.GetDirectories(versions))
        {
            string name = Path.GetFileName(dir);
            if (!keep.Contains(name) && !name.EndsWith(".partial", StringComparison.Ordinal))
            {
                try
                {
                    Directory.Delete(dir, recursive: true);
                }
                catch (IOException)
                {
                    // Still in use (a process from it is running); the next prune takes it.
                }
                catch (UnauthorizedAccessException)
                {
                }
            }
        }
    }

    public static string Executable(string dir, string name) => Path.Combine(dir, OperatingSystem.IsWindows() ? name + ".exe" : name);
}
