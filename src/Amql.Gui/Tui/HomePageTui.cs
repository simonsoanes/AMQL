using System.IO;
using System.ComponentModel;
using Amql.Gui.Services;
using Terminal.Gui.ViewBase;
using Terminal.Gui.Views;

namespace Amql.Gui.Tui;

/// <summary>The start page: the three steps to a running model, shortcuts to common tools, and what this machine has.</summary>
internal sealed class HomePageTui : ITuiPage
{
    private static readonly (string Title, string Text, string Command)[] Jobs =
    {
        ("Import from Hugging Face", "Download a model by URL, or use a local checkpoint folder (encode)", "encode"),
        ("Import a GGUF file", "Make a container from a .gguf (from-gguf)", "from-gguf"),
        ("Benchmark a model", "Prefill and decode tokens per second (bench)", "bench"),
        ("Generate from a prompt", "One-off generation with logits and traces (generate)", "generate"),
        ("Make a 4-bit copy", "Quantise weights to NVFP4/MXFP4 with GPTQ (quantise)", "quantise"),
        ("Export", "Write an HF checkpoint, GGUF or ONNX file (export)", "export"),
    };

    private readonly Shell _shell;
    private readonly View _root = new() { CanFocus = true };
    private readonly Label _models, _serverText, _gpu, _ram, _cli, _serverTool;
    private readonly Button _start;

    public HomePageTui(Shell shell)
    {
        _shell = shell;
        int y = 0;
        _root.Add(new Label { Text = "Welcome", X = 1, Y = y++ });
        _root.Add(new Label { Text = "Run models locally behind an OpenAI-compatible server, chat with them, and use every AMQL tool from one place.", X = 1, Y = y++, Width = Dim.Fill(1) });
        y++;

        var steps = new FrameView { Title = "Get going", X = 0, Y = y, Width = Dim.Fill(), Height = 8 };
        steps.Add(new Label { Text = "1  Pick models", X = 0, Y = 0 });
        _models = new Label { Text = "Looking for models…", X = 3, Y = 1, Width = Dim.Fill(20), Height = 1 };
        steps.Add(_models, TuiKit.Button("Choose models", () => _shell.Navigate("server"), Pos.AnchorEnd(18), 0));
        steps.Add(new Label { Text = "2  Start the server", X = 0, Y = 2 });
        _serverText = new Label { X = 3, Y = 3, Width = Dim.Fill(20), Height = 1 };
        _start = TuiKit.Button("Start server", () => TuiKit.GuardAsync(_shell.ToggleServerAsync), Pos.AnchorEnd(18), 2);
        steps.Add(_serverText, _start);
        steps.Add(new Label { Text = "3  Use it", X = 0, Y = 4 });
        steps.Add(new Label { Text = "Chat in the playground, or point any OpenAI client at the server.", X = 3, Y = 5, Width = Dim.Fill(20) });
        steps.Add(TuiKit.Button("Playground", () => _shell.Navigate("chat"), Pos.AnchorEnd(30), 4));
        steps.Add(TuiKit.Button("Monitor", () => _shell.Navigate("monitor"), Pos.AnchorEnd(14), 4));
        _root.Add(steps);
        y += 8;

        var jobs = new FrameView { Title = "Common jobs", X = 0, Y = y, Width = Dim.Fill(), Height = Jobs.Length / 2 * 3 + 2 };
        for (int i = 0; i < Jobs.Length; i++)
        {
            var job = Jobs[i];
            int col = i % 2, row = i / 2 * 3;
            var b = TuiKit.Button(job.Title, () => _shell.OpenCommand(job.Command), col == 0 ? 0 : Pos.Percent(50), row);
            jobs.Add(b);
            jobs.Add(new Label { Text = job.Text, X = col == 0 ? 1 : Pos.Percent(50) + 1, Y = row + 1, Width = Dim.Percent(50) - 2 });
        }
        _root.Add(jobs);
        y += Jobs.Length / 2 * 3 + 2;

        var machine = new FrameView { Title = "This computer", X = 0, Y = y, Width = Dim.Fill(), Height = 7 };
        machine.Add(new Label { Text = "GPU", X = 0, Y = 0 });
        _gpu = new Label { Text = "Checking…", X = 14, Y = 0, Width = Dim.Fill() };
        machine.Add(new Label { Text = "Memory", X = 0, Y = 1 });
        _ram = new Label { Text = $"{Format.Bytes(SystemInfo.TotalRamBytes)} installed", X = 14, Y = 1, Width = Dim.Fill() };
        machine.Add(new Label { Text = "amql-cli", X = 0, Y = 2 });
        _cli = new Label { X = 14, Y = 2, Width = Dim.Fill() };
        machine.Add(new Label { Text = "amql-server", X = 0, Y = 3 });
        _serverTool = new Label { X = 14, Y = 3, Width = Dim.Fill() };
        machine.Add(new Label
        {
            Text = "AMQL Studio finds amql-cli and amql-server by itself: beside the app, in the repository's build folders, or on PATH — the newest build wins.",
            X = 0, Y = 4, Width = Dim.Fill(), Height = 2,
        });
        machine.Add(_gpu, _ram, _cli, _serverTool);
        _root.Add(machine);

        ServerController.Instance.PropertyChanged += OnServerChanged;
        RefreshServer();
    }

    public View View => _root;

    public void Shown() => TuiKit.GuardAsync(RefreshAsync);

    public void Focus() => _start.SetFocus();

    public void Hidden()
    {
    }

    private async Task RefreshAsync()
    {
        var cli = ToolLocator.Find(ToolLocator.Cli);
        var server = ToolLocator.Find(ToolLocator.Server);
        _cli.Text = cli?.Describe() ?? "Not found — build src/Amql.Cli or put amql-cli on PATH";
        _serverTool.Text = server?.Describe() ?? "Not found — build src/Amql.Server or put amql-server on PATH";
        var gpu = await SystemInfo.SampleGpuAsync();
        _gpu.Text = gpu is null
            ? "No NVIDIA GPU found — models run on the CPU"
            : gpu.Unified
                ? $"{gpu.Name} · shares the system's {Format.Bytes(gpu.TotalBytes)} ({Format.Bytes(gpu.TotalBytes - gpu.UsedBytes)} free)"
                : $"{gpu.Name} · {Format.Bytes(gpu.TotalBytes)} ({Format.Bytes(gpu.TotalBytes - gpu.UsedBytes)} free)";
        var models = await ModelLibrary.ScanAsync(AppSettings.Current);
        int chosen = AppSettings.Current.Server.Models.Count(m => m.Enabled);
        var byKind = models.Where(m => m.Kind != ModelKind.Unservable).GroupBy(m => m.KindLabel)
            .Select(g => $"{g.Count()} {g.Key.ToLowerInvariant()}");
        _models.Text = models.Count == 0
            ? "No models found yet. Add the folder that holds your containers, or import one below."
            : $"Found {string.Join(", ", byKind)}. {(chosen == 0 ? "None chosen to load yet." : $"{chosen} chosen to load.")}";
    }

    private void OnServerChanged(object? sender, PropertyChangedEventArgs e) => RefreshServer();

    private void RefreshServer()
    {
        var server = ServerController.Instance;
        _serverText.Text = server.State switch
        {
            ServerState.Running or ServerState.External => $"{server.Detail}. Clients connect to {server.BaseUrl}/v1.",
            ServerState.Starting or ServerState.Failed => server.Detail,
            _ => "Loads the chosen models and serves them at an OpenAI-compatible endpoint.",
        };
        _start.Text = server.State switch
        {
            ServerState.Running or ServerState.Starting => "Stop server",
            ServerState.Stopping => "Stopping…",
            ServerState.External => "Forget server",
            _ => "Start server",
        };
        _start.Enabled = server.State != ServerState.Stopping;
    }
}
