using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using Amql.Gui.Services;

namespace Amql.Gui.Pages;

/// <summary>The start page: the three steps to a running model, shortcuts to common tools, and what this machine has.</summary>
public partial class HomePage : UserControl
{
    private static readonly (string Glyph, string Title, string Text, string Command)[] Jobs =
    {
        ("", "Import from Hugging Face", "Download a model by URL, or use a local checkpoint folder (encode)", "encode"),
        ("", "Import a GGUF file", "Make a container from a .gguf (from-gguf)", "from-gguf"),
        ("", "Benchmark a model", "Prefill and decode tokens per second (bench)", "bench"),
        ("", "Generate from a prompt", "One-off generation with logits and traces (generate)", "generate"),
        ("", "Make a 4-bit copy", "Quantise weights to NVFP4/MXFP4 with GPTQ (quantise)", "quantise"),
        ("", "Export", "Write an HF checkpoint, GGUF or ONNX file (export)", "export"),
    };

    private readonly MainWindow _shell;

    public HomePage(MainWindow shell)
    {
        _shell = shell;
        InitializeComponent();
        WheelScrolling.Attach(PageScroll);
        foreach (var job in Jobs)
        {
            var button = new Button
            {
                Width = 320,
                Height = 84,
                VerticalContentAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 10, 10),
                Padding = new Thickness(14, 12, 14, 12),
                HorizontalContentAlignment = HorizontalAlignment.Left,
            };
            var row = new DockPanel();
            var icon = new TextBlock { Text = job.Glyph, Style = (Style)FindResource("Icon"), FontSize = 22, Margin = new Thickness(0, 0, 14, 0) };
            DockPanel.SetDock(icon, Dock.Left);
            row.Children.Add(icon);
            var text = new StackPanel();
            text.Children.Add(new TextBlock { Text = job.Title, FontWeight = FontWeights.SemiBold });
            text.Children.Add(new TextBlock { Text = job.Text, Style = (Style)FindResource("Caption") });
            row.Children.Add(text);
            button.Content = row;
            string command = job.Command;
            button.Click += (_, _) => _shell.OpenCommand(command);
            Shortcuts.Children.Add(button);
        }

        RamText.Text = $"{Format.Bytes(SystemInfo.TotalRamBytes)} installed";
        ServerController.Instance.PropertyChanged += OnServerChanged;
        RefreshServer();
        IsVisibleChanged += async (_, _) =>
        {
            if (IsVisible)
            {
                await RefreshAsync();
            }
        };
    }

    private async Task RefreshAsync()
    {
        var cli = ToolLocator.Find(ToolLocator.Cli);
        var server = ToolLocator.Find(ToolLocator.Server);
        CliText.Text = cli?.Describe() ?? "Not found — build src/Amql.Cli or put amql-cli on PATH";
        ServerToolText.Text = server?.Describe() ?? "Not found — build src/Amql.Server or put amql-server on PATH";

        var gpu = await SystemInfo.SampleGpuAsync();
        GpuText.Text = gpu is null
            ? "No NVIDIA GPU found — models run on the CPU"
            : $"{gpu.Name} · {Format.Bytes(gpu.TotalBytes)} ({Format.Bytes(gpu.TotalBytes - gpu.UsedBytes)} free)";

        var models = await ModelLibrary.ScanAsync(AppSettings.Current);
        int chosen = AppSettings.Current.Server.Models.Count(m => m.Enabled);
        var byKind = models.Where(m => m.Kind != ModelKind.Unservable).GroupBy(m => m.KindLabel)
            .Select(g => $"{g.Count()} {g.Key.ToLowerInvariant()}");
        ModelsText.Text = models.Count == 0
            ? "No models found yet. Add the folder that holds your containers, or import one below."
            : $"Found {string.Join(", ", byKind)}. {(chosen == 0 ? "None chosen to load yet." : $"{chosen} chosen to load.")}";
    }

    private void OnServerChanged(object? sender, PropertyChangedEventArgs e) => RefreshServer();

    private void RefreshServer()
    {
        var server = ServerController.Instance;
        ServerText.Text = server.State switch
        {
            ServerState.Running or ServerState.External => $"{server.Detail}. Clients connect to {server.BaseUrl}/v1.",
            ServerState.Starting => server.Detail,
            ServerState.Failed => server.Detail,
            _ => "Loads the chosen models and serves them at an OpenAI-compatible endpoint.",
        };
        StartButton.Content = server.State switch
        {
            ServerState.Running or ServerState.Starting => "Stop server",
            ServerState.Stopping => "Stopping…",
            ServerState.External => "Forget server",
            _ => "Start server",
        };
        StartButton.IsEnabled = server.State != ServerState.Stopping;
    }

    private void OnChooseModels(object sender, RoutedEventArgs e) => _shell.Navigate("server");

    private async void OnStart(object sender, RoutedEventArgs e) => await _shell.ToggleServerAsync();

    private void OnChat(object sender, RoutedEventArgs e) => _shell.Navigate("chat");

    private void OnMonitor(object sender, RoutedEventArgs e) => _shell.OpenMonitor();
}
