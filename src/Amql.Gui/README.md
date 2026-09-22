# AMQL Studio (Amql.Gui)

A project-based WPF front-end for `amql-cli`. Every command and option of the
CLI is exposed as a form; projects are stored in a single `.amqlproj` file that
holds all parameters **and** the status/progress/output of every run.

## Design principle: contained by construction

`Amql.Gui` references **no AMQL library**. It drives `amql-cli` as a child
process (`Process` + `ArgumentList`, stdout/stderr streamed), so:

- feature parity with the command line is structural — same binary, same flags;
- concurrent work on the AMQL core cannot break the GUI (or vice versa);
- the global `--cpu` / `--gpu` flags and the `AMQL_WEIGHTS` env var are
  surfaced as project settings.

Three ways to reach the CLI (in priority order):

1. **CLI exe path** — a built `amql-cli.exe` (fastest startup);
2. **Repo path** — runs `dotnet run --no-launch-profile --project <repo>/src/Amql.Cli --`;
3. neither set — `amql-cli` on `PATH`.

## The `.amqlproj` format

JSON, versioned (`formatVersion`, currently 1), additive — unknown properties
are ignored on load. Sections:

| Section      | Contents |
|--------------|----------|
| `cli`        | exe path, repo path, device (`auto`/`cpu`/`gpu`), `AMQL_WEIGHTS` value |
| `defaults`   | project-wide container dir, tokenizer dir, patch file, component — stamped into forms via *Apply Project Defaults* (never overrides typed values) |
| `commands`   | per-command last-entered parameter values, keyed by command then parameter |
| `runs`       | the run history: id, command, argv, start/finish UTC, **status** (`Running`/`Succeeded`/`Failed`/`Cancelled`/`Interrupted`), exit code, output line count, last parsed **progress percent**, a bounded output tail (400 lines), and the error message |
| `lastCommand`| command selected when saved |

Saves are atomic (temp file + replace). The project is autosaved after every
run, so status and progress survive a crash or a mid-run save. Runs persisted
while `Running` are re-opened as `Interrupted`.

## Commands covered

All 22 CLI commands, grouped into 8 categories: ingest & verify
(`encode`, `verify`, `synth-model`, `layers`), text & tokenizer
(`tokens`, `decode`), pathways & probing (`route`, `path`, `generate`,
`inspect-token`), patching & LoRA (`change-tensor`, `save-lora`), export &
conversion (`export`, `to-gguf`), MTP drafter (`export-mtp`, `generate-mtp`,
`collect-mtp`, `fit-mtp`), model transformation (`import`, `moe-ify`,
`prune`), and fine-tuning (`fine-tune`).

The declarative catalogue lives in `Commands/CommandCatalog.cs` — one record per
command with its parameters (positional / option / switch, editor kind,
defaults, choices). `BuildArguments` produces the argv, and the same catalogue
drives both the dynamic form and the live command-line preview.

## Progress

The GUI asks the CLI for its machine-readable progress protocol (`--progress`,
or `AMQL_PROGRESS=1`): newline-delimited `##amql-progress` / `##amql-result`
JSON lines on stdout (`src/Amql.Cli/CliProgress.cs`). Progress lines feed the
bar exactly and are kept out of the output pane; result lines are echoed as a
compact `[result] key=value, …` summary.

`CliRunner.ParseProgress` (heuristic `NN%` / `N/M` scraping) remains as the
fallback for output the protocol does not cover and for CLI builds that predate
`--progress`.

## Exit codes

The CLI documents its exit codes (`amql-cli help`): 0 = success, 1 = a
legitimate negative result (e.g. `path` found no chain within budget, `verify`
integrity failed), 2 = usage/runtime error. The GUI records exit 0 as
`Succeeded`, exit 1 as `Negative` (an answer, not a fault — the bar completes),
and anything else as `Failed` (`CliExitCodes`).

## Run it

```
dotnet run --project src/Amql.Gui
```

Keyboard: `F5` runs the selected command, `Ctrl+S` saves the project. The
Cancel button kills the CLI process tree.
