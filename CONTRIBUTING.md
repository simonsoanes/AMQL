# Contributing to AMQL

First off, thanks for taking the time to contribute — whether that's reporting a bug, improving documentation, or submitting code.

## Getting Started

### Prerequisites

- [.NET 10.0 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) (latest LTS)
- Git

### Building

```bash
git clone https://github.com/simonsoanes/AMQL.git
cd AMQL
dotnet build AMQL.slnx -c Release
```

The CLI can be run directly from the built output:

```bash
dotnet run --project src/Amql.Cli -- help
```

Or build the published standalone executable:

```bash
.\scripts\publish-exe.cmd
```

### Shared output folder

Every project under `src/` writes to a single `bin/<Configuration>/` at the
repository root, rather than to a per-project `bin/`. So `bin/Release/` holds
`amql-cli.exe`, `amql-gui.exe` and all the `Amql.*` libraries side by side:

```bash
.\bin\Release\amql-cli.exe help
.\bin\Release\amql-gui.exe
```

This is configured in `src/Directory.Build.props` and is deliberately scoped to
`src/` — `tests/Amql.Tests` keeps its own output folder so xunit and its runner
do not land in the directory people launch things from. It also means the GUI
finds the CLI without configuration: `CliAutoDetect` looks beside itself first.
Because the two target frameworks in play (`net10.0` and the GUI's
`net10.0-windows`) share the folder, `AppendTargetFrameworkToOutputPath` is off;
each entry point still carries its own `runtimeconfig.json`.

`bin/` is gitignored. If you have an older checkout, the stale per-project
`src/*/bin/` folders are no longer written to and can be deleted.

### Project Structure

```
AMQL/
├── src/
│   ├── Directory.Build.props  # shared bin/<Configuration>/ output for all of src/
│   ├── Amql.Cli/        # CLI front-end (amql-cli)
│   ├── Amql.Gui/        # WPF desktop GUI (amql-gui): explorer, command runner, visualiser
│   ├── Amql.Safetensors/ # Safetensors I/O, MXFP4 and ternary codecs
│   ├── Amql.Vindex3/     # VIndex3 container graph & schema
│   ├── Amql.Inference/   # Tensor inference engine & tracing
│   ├── Amql.Gguf/        # GGUF v3 reader/writer and HF-checkpoint conversion
│   ├── Amql.Hf/          # Hugging Face checkpoint integration
│   ├── Amql.Merge/       # Model merging (token alignment, provenance)
│   └── Amql.Onnx/        # ONNX export
├── tests/
│   └── Amql.Tests/       # Unit & integration tests
├── scripts/
│   └── publish-exe.cmd   # Windows standalone publish script
├── bin/                  # shared build output (gitignored)
├── README.md
└── LICENSE
```

### Running Tests

```bash
dotnet test AMQL.slnx -c Release
```

## How to Contribute

### Reporting Bugs

- Use the [GitHub Issues](https://github.com/simonsoanes/AMQL/issues) page.
- Include steps to reproduce, the relevant checkpoint/model, and any error output.
- Mention the `.NET` version and platform.

### Suggesting Features

- Open an issue with the `enhancement` label.
- Describe the use case — AMQL is research-driven, so the broader context helps.

### Submitting Code

1. Fork the repository.
2. Create a feature branch from `main` (e.g., `feat/add-new-command`).
3. Make your changes — keep commits focused and messages descriptive.
4. Ensure the build is clean: `dotnet build AMQL.slnx -c Release` produces 0 warnings.
5. Open a pull request targeting `main`.

### Code Style

- C# 12+ idioms (primary constructors, pattern matching, etc.).
- Nullable reference types enabled (`<Nullable>enable</Nullable>`).
- `ImplicitUsings` enabled across all projects.
- Keep dependencies minimal; this is a research codebase, not a library.

## Research Note

AMQL is research code developed in the open. Some internals are intentionally opinionated or tightly coupled to the VIndex3 model. PRs that clarify, generalise, or fix are welcome — PRs that refactor for elegance at the cost of correctness will be treated with suspicion. When in doubt, discuss first.

---

This project is maintained by [Simon Soanes](https://github.com/simonsoanes). Questions? Open an issue or reach out via the [Fizl](https://www.fizl.co.uk) channels.
