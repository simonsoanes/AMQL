# AMQL Studio

AMQL Studio (`amql-gui`) is the desktop app for running models locally. It starts and stops
[`amql-server`](server.md) with the models you choose, shows what the server is doing while it runs, lets you
chat with a model or send any API request to it, and puts every `amql-cli` command behind a form.

It needs nothing set up first. It finds `amql-cli` and `amql-server` itself, finds the containers on disk, and
remembers what you chose for next time. Everyday choices are up front. Everything else is under **Advanced**.

```powershell
dotnet build src/Amql.Gui
```

```powershell
.\bin\amql-gui.exe
```

(The output folder follows `src/Directory.Build.props`. `dotnet run --project src/Amql.Gui` also works.)

## Home

![Home](../screenshots/AMQL%20Studio%20-%20Home.png)

The three steps to a running model: pick models, start the server, use it. Below them are shortcuts to common
jobs (import from Hugging Face or GGUF, benchmark, generate, make a 4-bit copy, export). Each one opens that
command's form on the Tools page. **This computer** shows the GPU and its free memory, installed RAM, and
which `amql-cli` and `amql-server` builds the app will run.

The server's state is always in the bottom-left corner, with **Start**/**Stop** and **Monitor** buttons, on
every page. **About AMQL Studio**, below it, shows the version, copyright and a link to the project.

## Server

![Server](../screenshots/AMQL%20Studio%20-%20Server.png)

**Models to load** lists every container found in the model folders. The repository is searched by default;
**Add folder** adds more, and **Add model** adds a single container. Each container's kind is read from its
system graph, the way the server decides what to serve it on:

| Badge | Served on |
|---|---|
| Chat | `/v1/chat/completions`, `/v1/responses` |
| Embeddings | `/v1/embeddings` |
| Classifier | `/v1/classify` |
| Decisions | `/v1/decisions`, `/v1/systemone`, `/api/v1/decisions` |
| Not servable | nothing. The reason is shown (no tokenizer, no chat template, a bare encoder…) |

Tick any mix of them. They load together, each under its own name, behind one endpoint, and requests choose
one with `"model"`. **Start server** writes them to a `--config` file and starts `amql-server`. The status
card shows each model as it loads, then the endpoint URL to copy into any OpenAI-compatible client. If the
server stops on its own, the card says why and the server's output opens under Advanced.

The ⚙ beside a model sets its options. The search icon opens it in the Explorer.

![Model options and memory](../screenshots/AMQL%20Studio%20-%20Server%20options.png)

- **Name**: the id requests use (default: the folder name).
- **GPU format**: serve the container's GPTQ-rounded NVFP4 or MXFP4 copy, built beside it the first time.
- **Reasoning, max reply tokens, temperature, top-p, top-k**: this model's defaults for requests that leave
  them out.
- **Entail adapters**: adapter folders run beside a chat model.

**Memory and device**:

- **Limit system memory (RAM)**: a slider from 2 GB up to the installed RAM. It sets `--max-ram`, so the
  server streams weights through a cache sized to fit. That is slower, but runs models larger than memory or
  leaves room for other work. With no limit, models are held in memory whole. The page warns when the chosen
  models will not fit comfortably.
- **Limit GPU memory**: caps how much of the GPU the weights take (`--gpu-memory`). The rest stay on the
  host. At 0, every weight stays on the host and the GPU only computes. With no limit, the server takes the
  GPU's free memory, less a margin.
- **CPU only**: runs on the CPU even when a GPU is present (`--cpu`).

**Advanced** holds the port, listening on the network, the API key, stopping the server when the app closes,
the server-wide request defaults and GPU format, every optional serving behaviour (batching, prefix reuse,
BF16 batches, the loop guard, tool-call rules…), n-gram packs, extra arguments, the exact command line, and the
server's own output for diagnosing a start that fails.

Changes are saved straight away. While the server runs, the page notes that they take effect on restart.

## Monitor

![Server Monitor](../screenshots/Server%20Monitor.png)

The Monitor is a separate window, so it can sit on another screen. It shows what the server is doing now,
not a log to scroll through. It reads the server's `GET /amql/status` once a second and the GPU through
`nvidia-smi`:

- **Headline numbers**: requests generating now (per model), output speed in tokens per second (now and over
  the last minute), request count, the server's memory, and GPU utilisation and memory.
- **Trends**: tokens per second and GPU utilisation over the last five minutes (server CPU use when there is no
  GPU). Hover over a chart to read a point.
- **Models**: per model: busy or idle, requests, tokens out, average decode speed, last used.
- **In progress**: each running request, with elapsed time, tokens so far, live decode speed and time to
  first token.
- **Recent requests**: the last 50: model, endpoint, status and finish reason, prompt and reply tokens,
  time to first token, decode speed and total time.

**Pause** freezes the view. **Keep on top** keeps it over other windows. If a server is already answering on
the port when the app starts (one started outside it, say), the app watches that one. It can be monitored and
chatted with, but not stopped.

## Playground

![Playground](../screenshots/AMQL%20Studio%20-%20Playground.png)

**Chat** talks to a chat model on the running server. Replies stream in, and under each one are its token
count, decode speed, time to first token and prompt size. **Think before answering** turns on reasoning and
gives the reply a 4,096-token budget, since thinking takes far more tokens than an answer. The thinking is
shown in italics unless **Show the thinking** is off. A reply cut off by the token limit says so. Under
**Advanced** are the temperature, max tokens, and an option to show each request's JSON.

**API request** sends any body to `/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/classify` or
`/v1/decisions`, and shows the response and how long it took. Choosing a model or an endpoint fills in a
working example. A model that does not chat opens this tab.

## Tools

![Tools](../screenshots/AMQL%20Studio%20-%20Tools.png)

Every `amql-cli` command is a form, grouped by job, with a search box over names and descriptions. This
includes the subcommands (`entail …`, `ngram …`, `quantise compare`). Required fields (marked `*`) come first.
The rest are under **Advanced options**, which opens by itself when one of them is set. **Run** (F5) checks the
required fields, then streams the output. The progress bar follows the CLI's `--progress` protocol.

**Run settings** holds the CLI's global options: device, a RAM limit slider (`--max-ram`), a GPU memory slider
(`--gpu-memory`), `--gpu-format`, `--fp4-prefill`, `--verbose` and `AMQL_WEIGHTS`. The exact command line is
shown below the form.

Form values and the run history (status, exit code, duration and an output tail per run) live in a project. A
default project is opened and saved automatically. The ⋯ menu creates, opens or saves projects as
`.amqlproj` files, and sets project defaults (container, tokenizer, patch, component) that fill empty fields.

### Importing a model from Hugging Face

![Import from Hugging Face](../screenshots/AMQL%20Studio%20-%20Import%20from%20Hugging%20Face.png)

**encode** (Home ▸ *Import from Hugging Face*) takes a checkpoint from either of two places:

- **Folder on this computer**: a checkpoint folder holding `config.json`, `*.safetensors` and the tokenizer files.
- **Hugging Face**: paste the model page's URL, or type `owner/name`. Add `@revision` for a branch, tag or
  commit; `/tree/<revision>` URLs work too.

For a Hugging Face model, the form says where the download will go, and fills an empty output folder with one
in the repository's `containers` folder, so the new container shows on the Server page. **Run** downloads the
checkpoint, then encodes it. The download is resumable and checked against the Hub's SHA-256 for every large
file. It is kept, so encoding again does not fetch it again.

The download takes the safetensors weights, config, tokenizer and chat-template files. It skips copies of the
weights in formats the encoder does not read (PyTorch `.bin` when safetensors exist, TF, Flax, ONNX, GGUF,
`original/`). A `.pt` beside the safetensors, such as Von's `option_marker.pt` head, is kept.

**Access token** is only needed for gated or private models. It is passed to that run as `HF_TOKEN` and not
saved. With it empty, a token from `HF_TOKEN` or from `huggingface-cli login` is used. A gated model whose terms
have not been accepted fails with a message linking to its page.

Under **Advanced options** are `--revision`, `--download-dir` (default:
`%LOCALAPPDATA%\AMQL\hf\<owner>--<name>\<revision>`, or `AMQL_HF_CACHE`) and `--all-files`. The separate
**download** command fetches a model without encoding it.

## Explorer

![Explorer](../screenshots/Model%20Explorer.png)

The container explorer shows index facts, components, per-layer attention policies and the object graph. Its
tensors open in a value grid. **Run and display inference…** runs a prompt and fills in the
[inference visualiser](inference-visualiser.md) live. **Open a saved trace…** opens a `--trace-json` capture.

## How it finds amql-cli and amql-server

There is no setting for where the executables are. AMQL Studio looks beside itself, in the folder above it, in
the repository's `bin`, `bin/Debug`, `bin/Release`, `bin-dev` and publish folders (found by walking up to
`AMQL.slnx`), and on `PATH`. It uses the most recently built copy. A copy whose libraries are missing from its
folder (a partial build output) is skipped. With no build at all but a repository present, it runs the
project from source with `dotnet run`. The Home page and the Server page show which build is in use. The Tools
page's ⋯ menu looks again after a rebuild, and starting the server always looks again.

## Where settings live

| What | Where |
|---|---|
| Model folders, server setup, playground choices, window position | `%LOCALAPPDATA%\AMQL Studio\settings.json` |
| The server's model list as last started | `%LOCALAPPDATA%\AMQL Studio\server-models.json` |
| Tool form values and run history | the open project (`%LOCALAPPDATA%\AMQL Studio\Default.amqlproj` unless you chose another) |
