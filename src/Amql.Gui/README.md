# AMQL Studio (Amql.Gui)

The WPF desktop app: it starts and watches `amql-server`, offers a chat playground and an API tester, puts
every `amql-cli` command behind a form, and hosts the container explorer and inference visualiser. The user
guide, with screenshots, is [docs/amql-studio.md](../../docs/amql-studio.md). This file covers how the app is
built.

```powershell
dotnet run --project src/Amql.Gui
```

## Layout

| Folder | What is there |
|---|---|
| `MainWindow` | The shell: a navigation rail with the server's state always in view, and a page host. Pages are built on first visit and kept. |
| `Pages/` | `HomePage` (getting started), `ServerPage` (models, memory, start/stop, advanced), `PlaygroundPage` (streamed chat, raw API requests), `ToolsPage` (the command forms, run history and project), `ExplorerPage` (wraps `Explorer/ContainerExplorer`). |
| `Windows/` | `ServerMonitorWindow` (the live dashboard) and `Sparkline` (its single-series trend chart). |
| `Services/` | `ToolLocator` (finds amql-cli and amql-server), `AppSettings` (per-machine settings), `ModelLibrary` (finds containers and reads their kind), `ServerController` (starts and stops the server and follows its start-up), `ServerClient` (health, `/amql/status`, streamed chat), `SystemInfo` (RAM, and the GPU through nvidia-smi), `WheelScrolling`. |
| `Commands/CommandCatalog.cs` | One record per CLI command and parameter. It drives both the forms and the argv builder. |
| `Run/CliRunner.cs` | Runs one CLI command as a child process and reads its `--progress` protocol. |
| `Explorer/`, `Visualiser/` | The container explorer and the inference visualiser. |
| `Program.cs` | The entry point: the WPF window, or the text UI (`--tui`; always on Linux). |
| `Tui/` | The text UI (Terminal.Gui): `Shell` (page list, server box, status bar), a page per desktop page plus `MonitorPageTui` and `VisualiserPageTui`, `FieldList` (forms as a list of settings) and `TuiKit` (prompts, path picker, chooser, OSC 52 copy, sparklines). |
| `Services/ServerOptions.cs`, `Commands/CommandForms.cs`, `Model/ProjectSession.cs` | What both front ends offer and how forms, runs and projects behave, without UI, so the two cannot drift. `UiThread` is where services hand events back to whichever UI runs. |

The app uses WPF's Fluent theme (`Application.ThemeMode="System"`), so it follows Windows' light or dark
setting. Page colours come from the theme's brushes, and no standard control is restyled implicitly. The
explorer's tensor grid and the visualiser were drawn for a light background, so they keep
`ThemeMode="Light"`.

## Two builds, one project

On Windows the project builds `net11.0-windows` with WPF, and `amql-gui.exe` opens the window (or the text UI
with `--tui`, in a console of its own). Built on Linux, or published with `-r linux-arm64` / `-r linux-x64`, it
builds `net11.0` without the WPF files and is the text UI only (`AmqlDesktop` in the project file chooses).
`scripts/publish-linux.sh` (or `.ps1` from Windows) publishes it with `amql-server` and `amql-cli` as one
self-contained folder. A cross-build goes to `bin/<Configuration>/<rid>/` so it does not overwrite the Windows
build.

## Finding the executables

There is no setting for where `amql-cli` or `amql-server` live. `ToolLocator` looks at every candidate and
uses the most recently built one that is complete. The candidates are: beside the app, the folder above it,
the repository's `bin/<runtime>/Release` and `bin/<runtime>/Debug` (`bin/win-x64/Release` on Windows), `bin-dev`, and `PATH`.

A candidate is complete when its own dll and every project library in its `.deps.json` sit beside it, so a
partial build output is skipped. With nothing built but the repository present, the tool runs with
`dotnet run --project src/<Project>`. Discovery is cached and is redone before every server start and from the
Tools page's ⋯ menu.

## The server

`ServerController` turns a `ServerProfile` into an amql-server command line. The enabled models, with their
per-model options, go into a `--config` file in the settings folder. Memory, device, request defaults and the
optional behaviours become flags. The server's output is read only to follow start-up, using its
`loading {id} from …`, `  {id}: …` and `listening on …` lines, and is kept in a 600-line buffer for the Advanced
view.

Once the server runs, what it is doing comes from `GET /amql/status` (`src/Amql.Server/ServerStats.cs`):
in-flight and recent requests with tokens, time to first token and decode speed, plus per-model totals and
process memory. Stopping kills the process tree. When the app starts, a healthy server already on the
configured port is adopted as *External*: it can be watched and used, but not stopped.

## Command forms

`CommandCatalog` lists every command the CLI dispatches, subcommands included. A name with a space, such as
`entail bench`, becomes separate argv words. Parameters marked `Required` are shown first, and the rest go
under "Advanced options". The CLI's global flags (`--cpu`/`--gpu`, `--max-ram`, `--gpu-memory`,
`--gpu-format`, `--fp4-prefill`, `--verbose`) come from the project's `CliSettings`, along with
`AMQL_WEIGHTS`. Keep the catalogue in step with `src/Amql.Cli/Program.cs` when commands change.

## Projects

A project (`.amqlproj`, JSON, `formatVersion` 1, additive) holds the CLI run settings, project defaults, each
command's last form values and the run history (status, exit code, progress, an output tail).

A default project in `%LOCALAPPDATA%\AMQL Studio` is opened when no other has been. Projects save
automatically after each run, on switching and on exit. Runs that were active when the app exited reopen as
`Interrupted`. Older project files still load: their `cli.cliExePath` and `cli.repoPath` are ignored.

The CLI's exit codes are recorded as `Succeeded` (0), `Negative` (1: an answer, such as `path` finding nothing)
or `Failed`.

## Keyboard

F5 runs the selected command on the Tools page. In the playground, Enter sends and Shift+Enter starts a new
line.
