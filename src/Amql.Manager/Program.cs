// amql-manager serve [--config manager.json] [--urls URL] [--api-key KEY]
// amql-manager supervise --root <dir>
// amql-manager version
//
// See docs/manager.md.

using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Amql.Manager;
using Microsoft.Extensions.Hosting;

string command = args.Length > 0 && !args[0].StartsWith('-') ? args[0] : "serve";
string[] rest = args.Length > 0 && !args[0].StartsWith('-') ? args[1..] : args;

string? Option(string name)
{
    int i = Array.IndexOf(rest, name);
    return i >= 0 && i + 1 < rest.Length ? rest[i + 1] : null;
}

try
{
    switch (command)
    {
        case "version" or "--version" or "-V":
            Console.WriteLine(Updater.RunningVersion());
            return 0;
        case "supervise":
        {
            string root = Path.GetFullPath(Option("--root") ?? Environment.GetEnvironmentVariable("AMQL_ROOT")
                ?? throw new ManagerException("supervise needs --root <dir> (the folder holding manager.json and versions/)"));
            var builder = Host.CreateApplicationBuilder(rest);
            builder.Services.AddWindowsService(o => o.ServiceName = "AMQL");
            builder.Services.AddSystemd();
            builder.Services.AddSingleton(root);
            builder.Services.AddHostedService(sp => new Supervisor(root, sp.GetRequiredService<ILogger<Supervisor>>()));
            await builder.Build().RunAsync();
            return 0;
        }
        case "serve":
        {
            var config = ManagerConfig.Load(Option("--config")
                ?? (Environment.GetEnvironmentVariable("AMQL_ROOT") is { } root ? Path.Combine(root, "manager.json") : null)
                ?? (File.Exists("manager.json") ? "manager.json" : null));
            if (Option("--urls") is { } urls)
            {
                config.Urls = urls;
            }
            if (Option("--api-key") is { } key)
            {
                config.ApiKey = key;
            }
            await using var app = ManagerApp.Build(config, rest);
            await app.RunAsync();
            return Environment.ExitCode;
        }
        default:
            Console.Error.WriteLine("usage: amql-manager serve [--config manager.json] [--urls URL] [--api-key KEY]\n" +
                                    "       amql-manager supervise --root <dir>\n" +
                                    "       amql-manager version");
            return 2;
    }
}
catch (ManagerException e)
{
    Console.Error.WriteLine($"amql-manager: {e.Message}");
    return 1;
}

namespace Amql.Manager
{
    /// <summary>The management API and the model endpoint, on one listener.</summary>
    public static class ManagerApp
    {
        internal static readonly JsonSerializerOptions Json = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            Converters = { new System.Text.Json.Serialization.JsonStringEnumConverter(JsonNamingPolicy.SnakeCaseLower) },
        };

        private static readonly string[] Forwarded = { "chat/completions", "completions", "responses", "embeddings", "decisions" };

        /// <summary>Set on a request authorised by one instance's own key.</summary>
        private const string InstanceKey = "amql.instance";

        private static bool Same(string? presented, string key) =>
            presented is not null && CryptographicOperations.FixedTimeEquals(Encoding.UTF8.GetBytes(presented), Encoding.UTF8.GetBytes(key));

        public static WebApplication Build(ManagerConfig config, string[]? args = null, HttpMessageHandler? updateHandler = null)
        {
            var builder = WebApplication.CreateSlimBuilder(args ?? Array.Empty<string>());
            builder.WebHost.UseUrls(config.Urls);
            builder.Services.AddWindowsService(o => o.ServiceName = "AMQL Manager");
            builder.Services.AddSystemd();
            builder.Services.ConfigureHttpJsonOptions(o =>
            {
                o.SerializerOptions.PropertyNamingPolicy = Json.PropertyNamingPolicy;
                foreach (var c in Json.Converters)
                {
                    o.SerializerOptions.Converters.Add(c);
                }
            });
            var pool = new ModelPool(config);
            var jobs = new CliJobs(config);
            var updater = new Updater(config.Update, updateHandler);
            builder.Services.AddSingleton(config).AddSingleton(pool).AddSingleton(jobs).AddSingleton(updater);
            builder.Services.AddHostedService(sp => new UpdateSchedule(config.Update, updater, jobs,
                sp.GetRequiredService<IHostApplicationLifetime>(), pool, sp.GetRequiredService<ILogger<UpdateSchedule>>()));
            var app = builder.Build();
            var started = DateTimeOffset.UtcNow;
            var forward = new HttpClient(new SocketsHttpHandler { PooledConnectionLifetime = TimeSpan.FromMinutes(5) })
            {
                Timeout = Timeout.InfiniteTimeSpan,
            };

            app.Lifetime.ApplicationStarted.Register(pool.StartRemembered);
            app.Lifetime.ApplicationStopping.Register(() =>
            {
                jobs.Dispose();
                pool.DisposeAsync().AsTask().GetAwaiter().GetResult();
            });

            // The manager's key (as a bearer token, or x-api-key as Ariadne.Router sends it) opens
            // everything; an instance's own key opens only the model endpoint, for that instance's
            // model (Forward checks which). /health and /healthz need none.
            app.Use(async (ctx, next) =>
            {
                string path = ctx.Request.Path.Value ?? "";
                if (path is "/health" or "/healthz")
                {
                    await next(ctx);
                    return;
                }
                string? presented = ctx.Request.Headers.Authorization.ToString() is { } auth && auth.StartsWith("Bearer ", StringComparison.Ordinal)
                    ? auth["Bearer ".Length..]
                    : ctx.Request.Headers["x-api-key"].ToString() is { Length: > 0 } x ? x : null;
                bool admin = config.ApiKey is not { Length: > 0 } key || Same(presented, key);
                if (!admin)
                {
                    bool modelEndpoint = path == "/v1/models" || Forwarded.Any(f => path == "/v1/" + f);
                    var instance = modelEndpoint ? pool.All.FirstOrDefault(m => m.ApiKey is { } k && Same(presented, k)) : null;
                    if (instance is null)
                    {
                        ctx.Response.StatusCode = 401;
                        await ctx.Response.WriteAsJsonAsync(Error("an API key is required (Authorization: Bearer, or x-api-key)", "unauthorized"), Json);
                        return;
                    }
                    ctx.Items[InstanceKey] = instance;
                }
                await next(ctx);
            });

            app.MapGet("/health", () => Results.Ok(new { status = "ok" }));
            ModelHostApi.Map(app, config, pool, jobs, updater, release => ApplyAndRestart(release));

            // ── status ──────────────────────────────────────────────────────
            var manage = app.MapGroup("/v1/manage");
            manage.MapGet("/status", () => Results.Json(new
            {
                version = Updater.RunningVersion(),
                started,
                uptime_seconds = (long)(DateTimeOffset.UtcNow - started).TotalSeconds,
                os = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
                runtime = System.Runtime.InteropServices.RuntimeInformation.RuntimeIdentifier,
                bin_dir = config.BinDirectory,
                models = pool.All.Select(m => Describe(m)),
                jobs_running = jobs.All.Count(j => j.State == JobState.Running),
                jobs_queued = jobs.All.Count(j => j.State == JobState.Queued),
                update = updater.Status,
            }, Json));

            // ── models ──────────────────────────────────────────────────────
            manage.MapGet("/models", () => Results.Json(new { data = pool.All.Select(m => Describe(m)) }, Json));
            manage.MapGet("/models/{id}", (string id) =>
                pool.Get(id) is { } m ? Results.Json(Describe(m), Json) : NotFound($"no model '{id}' is configured"));
            manage.MapPost("/models/{id}/start", async (string id, bool? wait, CancellationToken cancel) =>
            {
                if (pool.Get(id) is not { } m)
                {
                    return NotFound($"no model '{id}' is configured");
                }
                var start = pool.StartAsync(id, CancellationToken.None);
                if (wait == false)
                {
                    return Results.Json(Describe(m), Json, statusCode: 202);
                }
                bool ready = await start.WaitAsync(cancel);
                return Results.Json(Describe(m), Json, statusCode: ready ? 200 : 502);
            });
            manage.MapPost("/models/{id}/stop", async (string id) =>
            {
                if (pool.Get(id) is not { } m)
                {
                    return NotFound($"no model '{id}' is configured");
                }
                await pool.StopAsync(id);
                return Results.Json(Describe(m), Json);
            });
            manage.MapPost("/models/{id}/restart", async (string id, CancellationToken cancel) =>
            {
                if (pool.Get(id) is not { } m)
                {
                    return NotFound($"no model '{id}' is configured");
                }
                await pool.StopAsync(id);
                bool ready = await pool.StartAsync(id, cancel);
                return Results.Json(Describe(m), Json, statusCode: ready ? 200 : 502);
            });
            manage.MapGet("/models/{id}/log", (string id, long? from, int? max) =>
                pool.Get(id) is { Process: { } p } ? Lines(p, from, max) : NotFound($"model '{id}' has not been started"));

            // ── CLI jobs ────────────────────────────────────────────────────
            manage.MapGet("/cli/commands", async () => Results.Json(new { data = await jobs.CommandsAsync() }, Json));
            manage.MapPost("/jobs", (JobRequest request) =>
            {
                var job = jobs.Submit(request.Args ?? new List<string>(), request.WorkingDirectory is { } wd ? config.Resolve(wd) : null);
                return Results.Json(Describe(job), Json, statusCode: 202);
            });
            manage.MapGet("/jobs", () => Results.Json(new { data = jobs.All.Select(Describe) }, Json));
            manage.MapGet("/jobs/{id}", (string id) => jobs.Get(id) is { } j ? Results.Json(Describe(j), Json) : NotFound($"no job '{id}'"));
            manage.MapGet("/jobs/{id}/log", (string id, long? from, int? max) =>
                jobs.Get(id) is { Process: { } p } ? Lines(p, from, max)
                : jobs.Get(id) is { } queued ? Results.Json(new { lines = Array.Empty<string>(), first = 0, next = 0, state = queued.State }, Json)
                : NotFound($"no job '{id}'"));
            manage.MapDelete("/jobs/{id}", (string id) =>
                jobs.CancelJob(id) ? Results.Json(Describe(jobs.Get(id)!), Json) : NotFound($"no queued or running job '{id}'"));

            // ── updates ─────────────────────────────────────────────────────
            manage.MapGet("/update", () => Results.Json(updater.Status, Json));
            manage.MapPost("/update/check", async (bool? apply, CancellationToken cancel) =>
            {
                var release = await updater.CheckAsync(cancel);
                if (release is not null && (apply ?? false))
                {
                    return await ApplyAndRestart(release);
                }
                return Results.Json(updater.Status, Json);
            });
            manage.MapPost("/update/apply", async (CancellationToken cancel) =>
            {
                var release = await updater.CheckAsync(cancel);
                if (release is null)
                {
                    return Results.Json(updater.Status, Json, statusCode: updater.Status.LastError is null ? 200 : 502);
                }
                return await ApplyAndRestart(release);
            });
            manage.MapPost("/update/rollback", async () =>
            {
                string version = updater.Rollback();
                await RestartSoon();
                return Results.Json(new { rolled_back_to = version, restarting = true }, Json, statusCode: 202);
            });
            manage.MapPost("/restart", async () =>
            {
                if (!updater.Status.Supervised)
                {
                    return Results.Json(Error("not running under the supervisor, so nothing would start it again", "not_supervised"), Json, statusCode: 409);
                }
                await RestartSoon();
                return Results.Json(new { restarting = true }, Json, statusCode: 202);
            });

            async Task<IResult> ApplyAndRestart(ReleaseInfo release)
            {
                if (jobs.Busy)
                {
                    return Results.Json(Error("a CLI job is running: the update waits for it (or cancel it first)", "busy"), Json, statusCode: 409);
                }
                try
                {
                    string version = await updater.ApplyAsync(release, CancellationToken.None);
                    await RestartSoon();
                    return Results.Json(new { installed = version, restarting = updater.Status.Supervised }, Json, statusCode: 202);
                }
                catch (ManagerException e)
                {
                    return Results.Json(Error(e.Message, "update_failed"), Json, statusCode: 502);
                }
            }

            async Task RestartSoon()
            {
                await pool.StopAllForRestartAsync();
                Environment.ExitCode = Updater.RestartExitCode;
                _ = Task.Delay(500).ContinueWith(_ => app.Lifetime.StopApplication());
            }

            // ── the model endpoint: OpenAI-compatible, routed by "model" ─────
            app.MapGet("/v1/models", () => Results.Json(new
            {
                @object = "list",
                data = pool.All.SelectMany(m => (m.Served.Count > 0 ? m.Served : new[] { m.Entry.Id }).Select(id => new
                {
                    id,
                    @object = "model",
                    owned_by = "amql",
                    status = m.State,
                    instance = m.Entry.Id,
                })),
            }, Json));
            foreach (string path in Forwarded)
            {
                app.MapPost("/v1/" + path, (HttpContext ctx) => Forward(ctx, path, pool, forward));
            }

            return app;
        }

        private static async Task Forward(HttpContext ctx, string path, ModelPool pool, HttpClient http)
        {
            using var buffer = new MemoryStream();
            await ctx.Request.Body.CopyToAsync(buffer, ctx.RequestAborted);
            string? model = null;
            try
            {
                using var doc = JsonDocument.Parse(buffer.ToArray());
                if (doc.RootElement.TryGetProperty("model", out var m) && m.ValueKind == JsonValueKind.String)
                {
                    model = m.GetString();
                }
            }
            catch (JsonException)
            {
            }
            var keyed = ctx.Items[InstanceKey] as ModelInstance;
            var instance = model is null ? keyed ?? (pool.All.Count == 1 ? pool.All.First() : null) : pool.Resolve(model);
            if (instance is null)
            {
                ctx.Response.StatusCode = 404;
                await ctx.Response.WriteAsJsonAsync(Error(model is null ? "the request names no 'model'" : $"no model '{model}'", "model_not_found"), Json);
                return;
            }
            if (keyed is not null && keyed != instance)
            {
                ctx.Response.StatusCode = 403;
                await ctx.Response.WriteAsJsonAsync(Error($"this key is for '{keyed.Entry.ServedName ?? keyed.Entry.Id}', not '{model}'", "forbidden"), Json);
                return;
            }
            if (instance.State != ModelState.Ready)
            {
                if (!instance.Entry.StartOnDemand)
                {
                    ctx.Response.StatusCode = 503;
                    await ctx.Response.WriteAsJsonAsync(Error($"model '{instance.Entry.Id}' is {instance.State.ToString().ToLowerInvariant()} and does not start on demand", "model_not_running"), Json);
                    return;
                }
                if (!await pool.StartAsync(instance.Entry.Id, ctx.RequestAborted))
                {
                    ctx.Response.StatusCode = 503;
                    await ctx.Response.WriteAsJsonAsync(Error($"model '{instance.Entry.Id}' failed to start: {instance.Error}", "model_failed"), Json);
                    return;
                }
            }

            using var use = pool.Use(instance);
            using var request = new HttpRequestMessage(HttpMethod.Post, $"http://127.0.0.1:{instance.Port}/v1/{path}{ctx.Request.QueryString}")
            {
                Content = new ByteArrayContent(buffer.ToArray()),
            };
            request.Content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/json");
            using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ctx.RequestAborted);
            ctx.Response.StatusCode = (int)response.StatusCode;
            foreach (var h in response.Content.Headers.Concat(response.Headers))
            {
                if (h.Key is "Transfer-Encoding" or "Content-Length" or "Connection")
                {
                    continue;
                }
                ctx.Response.Headers[h.Key] = h.Value.ToArray();
            }
            await using var body = await response.Content.ReadAsStreamAsync(ctx.RequestAborted);
            var chunk = new byte[16384];
            int n;
            while ((n = await body.ReadAsync(chunk, ctx.RequestAborted)) > 0)
            {
                await ctx.Response.Body.WriteAsync(chunk.AsMemory(0, n), ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }
        }

        private static object Describe(ModelInstance m) => new
        {
            id = m.Entry.Id,
            state = m.State,
            served = m.Served,
            path = m.Entry.Path,
            port = m.Port > 0 ? m.Port : (int?)null,
            instance_url = m.BaseUrl,
            pid = m.Process is { Running: true } p ? p.Id : (int?)null,
            ready_at = m.ReadyAt,
            last_used = m.LastUsed,
            in_flight = Volatile.Read(ref m.InFlight),
            autostart = m.Entry.Autostart,
            start_on_demand = m.Entry.StartOnDemand,
            idle_minutes = m.Entry.IdleMinutes,
            error = m.Error,
        };

        private static object Describe(CliJob j) => new
        {
            id = j.Id,
            args = j.Args,
            working_directory = j.WorkingDirectory,
            state = j.State,
            queued = j.Queued,
            started = j.Process?.Started,
            exited = j.Process?.Exited,
            exit_code = j.Process?.ExitCode,
            error = j.Error,
            tail = j.Process?.Tail(10),
            log = j.LogPath,
        };

        private static IResult Lines(ChildProcess p, long? from, int? max)
        {
            var (lines, first, next) = p.Read(from ?? 0, Math.Clamp(max ?? 1000, 1, ChildProcess.Keep));
            return Results.Json(new { lines, first, next, running = p.Running, exit_code = p.ExitCode }, Json);
        }

        private static IResult NotFound(string message) => Results.Json(Error(message, "not_found"), Json, statusCode: 404);

        internal static object Error(string message, string code) => new { error = new { message, type = code, code } };
    }

    public sealed record JobRequest(List<string>? Args, string? WorkingDirectory);

    /// <summary>Checks for an update every <see cref="UpdateSettings.IntervalHours"/> (the first a few
    /// minutes after starting) and, with <see cref="UpdateSettings.AutoApply"/>, installs it when no job runs.</summary>
    public sealed class UpdateSchedule(UpdateSettings settings, Updater updater, CliJobs jobs, IHostApplicationLifetime life,
        ModelPool pool, ILogger<UpdateSchedule> log) : BackgroundService
    {
        protected override async Task ExecuteAsync(CancellationToken stop)
        {
            if (!settings.Enabled)
            {
                return;
            }
            try
            {
                await Task.Delay(TimeSpan.FromMinutes(3), stop);
                while (!stop.IsCancellationRequested)
                {
                    var release = await updater.CheckAsync(stop);
                    if (release is not null)
                    {
                        log.LogInformation("AMQL {Version} is available (running {Current})", release.Version, updater.Status.Current);
                        if (settings.AutoApply && updater.Status.Supervised && !jobs.Busy)
                        {
                            try
                            {
                                string version = await updater.ApplyAsync(release, stop);
                                log.LogInformation("installed {Version}; restarting into it", version);
                                await pool.StopAllForRestartAsync();
                                Environment.ExitCode = Updater.RestartExitCode;
                                life.StopApplication();
                                return;
                            }
                            catch (ManagerException e)
                            {
                                log.LogError("update {Version} not installed: {Error}", release.Version, e.Message);
                                updater.Status.LastError = e.Message;
                            }
                        }
                    }
                    else if (updater.Status.LastError is { } error)
                    {
                        log.LogWarning("update check failed: {Error}", error);
                    }
                    await Task.Delay(TimeSpan.FromHours(settings.IntervalHours), stop);
                }
            }
            catch (OperationCanceledException)
            {
            }
        }
    }
}
