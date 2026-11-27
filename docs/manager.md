# amql-manager: headless management, a service, and updates

`amql-manager` runs an AMQL install with no one at it:
- **Models:** each in its own `amql-server`, started and stopped on demand behind one OpenAI-compatible endpoint.
- **CLI jobs:** every `amql-cli` command, run as a job.
- **Service:** it runs as a Windows service or a systemd unit.
- **Updates:** it installs new versions from GitHub releases after checking them.

It speaks Ariadne ModelHost's desired-state API, so Ariadne.Router can manage it as a deployment location.

```bash
amql-manager serve [--config manager.json] [--urls http://127.0.0.1:8100] [--api-key KEY]
amql-manager supervise --root <dir>      # what the service runs
amql-manager version
```

## Configuration

`manager.json` (comments and trailing commas allowed; an unknown setting is refused):

```jsonc
{
  "urls": "http://0.0.0.0:8100",
  "api_key": "amql-…",                // or AMQL_MANAGER_API_KEY; required unless it listens on loopback only
  "public_url": "http://spark.local:8100",  // what clients reach it at (default: the address a request came in on)
  "bin_dir": null,                      // amql-cli and amql-server; default the manager's own folder
  "containers_dir": "containers",       // where a Hugging Face model asked for by the router is encoded
  "first_model_port": 8201,             // each model's server takes a loopback port from here
  "server_args": ["--kv-format", "nvfp4"],
  "models": [
    { "id": "kaja", "path": "containers/kaja-gem-26b-a4b", "autostart": true },
    { "id": "ink", "path": "containers/inkling-2-mini", "idle_minutes": 30 },
    { "id": "pair", "path": "servers/pair.jsonc" }   // an amql-server --config file: several models in one process
  ],
  "jobs": { "max_concurrent": 1, "low_priority": true, "keep_finished": 200 },
  "update": { "enabled": true, "repository": "simonsoanes/AMQL", "prerelease": false,
              "interval_hours": 24, "auto_apply": true, "token_variable": "AMQL_UPDATE_TOKEN" }
}
```

A model's settings:

| Setting | Effect | Default |
|---|---|---|
| `autostart` | starts with the manager | off |
| `start_on_demand` | a request for a stopped model starts it, and is answered once the model is ready | on |
| `idle_minutes` | stops the model after that long without a request | 0, keep running |
| `start_timeout_seconds` | how long a start may take before it counts as failed | |
| `served_name` | the name requests use | the model's id |
| `args` | arguments added to its server's command line | none |
| `port` | a fixed port | the next free one from `first_model_port` |

When the manager restarts, after an update for instance, the models of `manager.json` that were running start again.

## Keys

- **Manager key:** given as `Authorization: Bearer <key>`, or as `x-api-key: <key>` the way Ariadne.Router sends it. It opens everything, and is compared in constant time.
- **Instance key:** each model the router asks for may come with its own key. That key opens only the model endpoint (`/v1/models` and the request routes), and only for that model: another model answers 403, the management API 401.
- **No key:** `/health` and `/healthz`.
- **Listening:** the manager refuses to listen beyond loopback without a key, since the API runs commands and installs software. Each model's server listens on loopback only, so models are reached through the manager.

## The model endpoint

OpenAI-compatible, routed by the request's `model`. The routed paths are:
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/responses`
- `POST /v1/embeddings`
- `POST /v1/decisions`

The request goes to the server of that model, which is started first if it is stopped and starts on demand. Streams pass through unbuffered. `GET /v1/models` lists every model with its state.

## The management API (`/v1/manage`, snake_case JSON)

| Request | Does |
|---|---|
| `GET /status` | version, uptime, platform, the models, jobs running and queued, the update state |
| `GET /models`, `GET /models/{id}` | each model: state (`stopped`, `starting`, `ready`, `stopping`, `failed`), served names, port, process, last use, error |
| `POST /models/{id}/start[?wait=false]` | start; waits until ready (200) or failed (502) unless `wait=false` (202) |
| `POST /models/{id}/stop`, `/restart` | |
| `GET /models/{id}/log?from=N&max=M` | its server's output, numbered lines from `N`; `next` is where the next read starts |
| `GET /cli/commands` | every amql-cli command and its usage, read from the CLI's own help |
| `POST /jobs` `{"args": ["verify", "containers/x"], "working_directory": null}` | runs `amql-cli` with those arguments (no shell is involved); 202 with the job |
| `GET /jobs`, `GET /jobs/{id}` | state (`queued`, `running`, `succeeded`, `failed`, `cancelled`), exit code, the last lines |
| `GET /jobs/{id}/log?from=N` | the job's output, incrementally; the whole of it is in its log file |
| `DELETE /jobs/{id}` | cancel |
| `GET /update` | running version, newest release seen, last check and error, supervised or not |
| `POST /update/check[?apply=true]` | the nudge: check now, and with `apply=true` install what it finds |
| `POST /update/apply` | install the newest release; refused (409) while a job runs |
| `POST /update/rollback` | make the version before current again, and restart into it |
| `POST /restart` | restart the manager (only under the supervisor, which starts it again) |

Jobs run one at a time by default, below normal priority, so the machine stays usable.

## Ariadne.Router: the ModelHost v2 contract

The manager speaks the desired-state API that Ariadne.Router uses for Ariadne.ModelHost (camelCase JSON, `x-api-key`), so an AMQL machine is a deployment location of kind `AriadneModelHost`:

| Request | Does |
|---|---|
| `GET /healthz` | `{status, apiVersion: 2}`, no key |
| `GET /api/v2/capacity` | the host: CPUs, memory, GPUs (from `nvidia-smi`), free disk, and the engine `amql` with its version |
| `GET /api/v2/instances` | the instances the router asked for, each `pending`, `downloading`, `ready` or `failed` |
| `PUT /api/v2/desired` `{instances: [...]}` | the whole set the router wants. New instances start, changed ones restart, missing ones stop. Answers `{instances, rejected}` at once and converges in the background. |
| `GET /api/v2/instances/{name}/logs?lines=N` | an instance's output, as text |
| `POST /api/v2/engines/amql/install` | the router's "install engine": update AMQL to the newest release |

An instance's `engine` must be `amql`. Its `model` can be:
- a model of `manager.json`, by id;
- an AMQL container folder, by path or by name under `containers_dir`;
- a Hugging Face repo. It is downloaded and encoded into `containers_dir` with `amql-cli encode` first, and reported as `downloading` meanwhile.

The other fields:

| Field | Effect |
|---|---|
| `servedModelName` | the name requests use |
| `apiKey` | the instance's own key (`apiKeyEnforced: true`) |
| `contextLength` | becomes `--max-context` |
| `extraArgs` | go to its server |

Every instance's `url` is the manager's (`public_url`): the router sends the instance's traffic to the manager, which routes it by `servedModelName`.

**What the router itself still needs:** the engine name `amql` in its `ManagedEngineCatalog` (`Deployments/ManagedDeploymentModels.cs`), with `SupportsApiKey: true` and `LoadsRepository: true`. The test that keeps that catalogue in step with Ariadne.ModelHost's engine names needs the same change.

## Running as a service

```powershell
# Windows, elevated
scripts/service/install-windows.ps1 -Source dist/amql-1.1.0-win-x64.zip [-Urls http://0.0.0.0:8100] [-Root C:\ProgramData\AMQL]
```

```bash
# Linux (systemd)
sudo scripts/service/install-linux.sh dist/amql-1.1.0-linux-arm64.tar.gz        # ROOT=/opt/amql URLS=... API_KEY=...
```

Both install into a root folder:

```
<root>/versions/<version>/   each install (amql-manager, amql-server, amql-cli, amql-gui, the CUDA libraries)
<root>/current, previous     the version run, and the one before it
<root>/supervisor/           the service's own copy of amql-manager
<root>/manager.json          the config (written once, readable only by administrators or the service user)
<root>/data/                 manager.log, job and model logs, which models were running
```

What the service does:
- **What runs:** the service runs `amql-manager supervise --root <root>`, which runs the current version's manager and keeps it running.
- **Restarts:** a manager that exits for a restart (after an update or a rollback) starts again at once, from whichever version is current. One that crashes restarts after a pause that grows to a minute.
- **Rollback:** a version that fails within a minute of starting three times running, or cannot start at all, is rolled back to the version before it. A version that has failed is never rolled back to.
- **Supervisor:** the supervisor is its own copy, so an update never replaces a running executable. To update the supervisor itself, run the install script again.

## Updates

A release is a GitHub release tagged `v<version>`. It carries:
- one archive per platform: `amql-<version>-<rid>.zip` for Windows, `.tar.gz` for Linux, with everything inside one `amql/` folder;
- `SHA256SUMS`, listing those archives.

`scripts/release/package.ps1 -Version 1.1.0` builds all of them into `dist/`, for `win-x64`, `linux-x64` and `linux-arm64` by default. Publish them with:

```bash
gh release create v1.1.0 dist/* --title "AMQL 1.1.0"
```

The manager checks a few minutes after it starts, then every `interval_hours`, or at once on `POST /v1/manage/update/check`. When a newer release has an archive for its platform, it installs it. With `auto_apply`, that happens when no job is running; otherwise it waits for `POST /v1/manage/update/apply`. Installing a release:
1. Downloads the archive and `SHA256SUMS` from the release, over HTTPS. A private repository needs a token in `AMQL_UPDATE_TOKEN`.
2. Refuses a release without `SHA256SUMS`, or an archive whose SHA-256 is not the one listed.
3. Unpacks it into `<root>/versions/<version>.partial`. It refuses archive entries that would land outside that folder, links, and an install missing `amql-manager`, `amql-server` or `amql-cli`.
4. Renames the folder into place and makes it current, keeping the version before as `previous`.
5. Stops the models, remembering which were running, and exits for a restart. The supervisor starts the new version, which brings those models back.
6. Keeps the current and previous versions, and removes older ones.

**What the checksum does and doesn't protect.** It proves the archive is the one the release lists, which guards against a corrupted or swapped download. It does not prove who made the release. That rests on control of the GitHub repository (and of the token, for a private one).

## Not done

- **Signatures:** releases are not signed, so a release is trusted as far as the repository is.
- **Supervisor:** an update does not replace the supervisor; rerun the install script to do that.
- **Batching:** one model's server serves its requests itself. The manager does no batching or queueing of its own.
