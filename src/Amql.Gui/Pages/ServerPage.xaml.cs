using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Amql.Gui.Services;
using Microsoft.Win32;

namespace Amql.Gui.Pages;

/// <summary>
/// Sets up and runs amql-server: which models load (each container's kind
/// read from its system graph), how much RAM and GPU memory they may take,
/// and — under Advanced — networking, request defaults and the server's
/// optional behaviours. Every change is saved straight away.
/// </summary>
public partial class ServerPage : UserControl
{
    private static readonly (string Label, string Help, Func<ServerProfile, bool> Get, Action<ServerProfile, bool> Set)[] Behaviours =
    {
        ("Don't warm up models", "--no-warm: skip the warm-up pass at start (faster start, slower first request)", p => p.NoWarm, (p, v) => p.NoWarm = v),
        ("One request at a time", "--no-batch: no continuous batching, even on the GPU", p => p.NoBatch, (p, v) => p.NoBatch = v),
        ("No prefix reuse", "--no-prefix-reuse: prefill every prompt whole", p => p.NoPrefixReuse, (p, v) => p.NoPrefixReuse = v),
        ("Page prompt state to disk", "--kv-disk: idle prompt state goes to kv-cache in the model's container (1 GiB, deleted after 30 min) and loads back instead of prefilling", p => p.KvDisk, (p, v) => p.KvDisk = v),
        ("BF16 batches (faster)", "--batch-bf16: batches of 4+ on tensor cores, ~1.6x throughput; replies depend slightly on their batch", p => p.BatchBf16, (p, v) => p.BatchBf16 = v),
        ("No loop guard", "--no-loop-guard: let a reply repeat itself to its token limit", p => p.NoLoopGuard, (p, v) => p.NoLoopGuard = v),
        ("Free tool calls", "--free-tool-calls: do not hold calls to the request's declared tools", p => p.FreeToolCalls, (p, v) => p.FreeToolCalls = v),
        ("No mention rule", "--no-mention-rule: do not open optional parameters the conversation names", p => p.NoMentionRule, (p, v) => p.NoMentionRule = v),
        ("No announce rule", "--no-announce-rule: allow a reply to end on a colon without a call", p => p.NoAnnounceRule, (p, v) => p.NoAnnounceRule = v),
        ("Allow repeated calls", "--allow-repeated-calls: allow a call that already returned the same result twice", p => p.AllowRepeatedCalls, (p, v) => p.AllowRepeatedCalls = v),
        ("Keep line marks", "--keep-line-marks: show tool results with their line decoration", p => p.KeepLineMarks, (p, v) => p.KeepLineMarks = v),
    };

    private readonly MainWindow _shell;
    private readonly List<CheckBox> _behaviourChecks = new();
    private List<ModelEntry> _found = new();
    private bool _loading = true;
    private bool _changedWhileRunning;
    private double _gpuTotalGb;
    private ServerState _lastState;

    private static AppSettings Settings => AppSettings.Current;
    private static ServerProfile Profile => Settings.Server;

    public ServerPage(MainWindow shell)
    {
        _shell = shell;
        InitializeComponent();
        WheelScrolling.Attach(PageScroll);
        foreach (var b in Behaviours)
        {
            var check = new CheckBox { Content = b.Label, ToolTip = b.Help, Margin = new Thickness(0, 2, 24, 2), MinWidth = 210 };
            check.Checked += OnSettingChanged;
            check.Unchecked += OnSettingChanged;
            _behaviourChecks.Add(check);
            BehaviourPanel.Children.Add(check);
        }
        RamSlider.Maximum = Math.Max(4, Math.Floor(SystemInfo.TotalRamBytes / (double)(1L << 30)));
        LoadSettings();
        _loading = false;

        ServerController.Instance.PropertyChanged += OnServerChanged;
        ServerController.Instance.ConsoleLine += OnConsoleLine;
        RefreshServer();
        RefreshTools();
        ConsoleBox.Text = string.Join(Environment.NewLine, ServerController.Instance.ConsoleSnapshot());
        Loaded += async (_, _) =>
        {
            if (_found.Count == 0)
            {
                await RescanAsync();
            }
        };
        _ = InitGpuAsync();
    }

    // ── settings ⇄ controls ─────────────────────────────────────────────────

    private void LoadSettings()
    {
        GpuRadio.IsChecked = Profile.Device != "cpu";
        CpuRadio.IsChecked = Profile.Device == "cpu";
        RamLimitCheck.IsChecked = Profile.MaxRamGb is not null;
        RamSlider.Value = Profile.MaxRamGb ?? Math.Min(10, RamSlider.Maximum);
        GpuLimitCheck.IsChecked = Profile.GpuMemoryGb is not null;
        GpuSlider.Value = Profile.GpuMemoryGb ?? Math.Min(8, GpuSlider.Maximum);
        PortBox.Text = Profile.Port.ToString(CultureInfo.InvariantCulture);
        NetworkCheck.IsChecked = Profile.ListenOnNetwork;
        ApiKeyBox.Password = Profile.ApiKey;
        StopOnExitCheck.IsChecked = Profile.StopOnExit;
        ThinkCheck.IsChecked = Profile.Think;
        MaxTokensBox.Text = Profile.MaxTokens?.ToString(CultureInfo.InvariantCulture) ?? "";
        TemperatureBox.Text = Num(Profile.Temperature);
        TopPBox.Text = Num(Profile.TopP);
        TopKBox.Text = Profile.TopK?.ToString(CultureInfo.InvariantCulture) ?? "";
        GpuFormatBox.SelectedItem = GpuFormatBox.Items.Cast<ComboBoxItem>().FirstOrDefault(i => (string)i.Tag == Profile.GpuFormat) ?? GpuFormatBox.Items[0];
        KvFormatBox.SelectedItem = KvFormatBox.Items.Cast<ComboBoxItem>().FirstOrDefault(i => (string)i.Tag == Profile.KvFormat) ?? KvFormatBox.Items[0];
        for (int i = 0; i < Behaviours.Length; i++)
        {
            _behaviourChecks[i].IsChecked = Behaviours[i].Get(Profile);
        }
        NGramBox.Text = string.Join(Environment.NewLine, Profile.NGramPacks);
        ExtraArgsBox.Text = Profile.ExtraArguments;
        UpdateMemoryText();
        UpdateCommandLine();
    }

    private void OnSettingChanged(object sender, RoutedEventArgs e)
    {
        if (_loading)
        {
            return;
        }
        Profile.Device = CpuRadio.IsChecked == true ? "cpu" : "gpu";
        Profile.MaxRamGb = RamLimitCheck.IsChecked == true ? RamSlider.Value : null;
        Profile.GpuMemoryGb = GpuLimitCheck.IsChecked == true ? GpuSlider.Value : null;
        if (int.TryParse(PortBox.Text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int port) && port is > 0 and < 65536)
        {
            Profile.Port = port;
        }
        Profile.ListenOnNetwork = NetworkCheck.IsChecked == true;
        Profile.ApiKey = ApiKeyBox.Password;
        Profile.StopOnExit = StopOnExitCheck.IsChecked == true;
        Profile.Think = ThinkCheck.IsChecked == true;
        Profile.MaxTokens = Int(MaxTokensBox.Text);
        Profile.Temperature = Double(TemperatureBox.Text);
        Profile.TopP = Double(TopPBox.Text);
        Profile.TopK = Int(TopKBox.Text);
        Profile.GpuFormat = (GpuFormatBox.SelectedItem as ComboBoxItem)?.Tag as string ?? "";
        Profile.KvFormat = (KvFormatBox.SelectedItem as ComboBoxItem)?.Tag as string ?? "";
        for (int i = 0; i < Behaviours.Length; i++)
        {
            Behaviours[i].Set(Profile, _behaviourChecks[i].IsChecked == true);
        }
        Profile.NGramPacks = NGramBox.Text.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
        Profile.ExtraArguments = ExtraArgsBox.Text;
        Changed();
    }

    private void OnSliderChanged(object sender, RoutedPropertyChangedEventArgs<double> e) => OnSettingChanged(sender, e);

    /// <summary>A change to what the server runs: saved, and flagged if it needs a restart.</summary>
    private void Changed()
    {
        Settings.Save();
        UpdateMemoryText();
        UpdateSelectionText();
        UpdateCommandLine();
        if (ServerController.Instance.State is ServerState.Running or ServerState.Starting)
        {
            _changedWhileRunning = true;
            RefreshServer();
        }
    }

    private static string Num(double? v) => v?.ToString("0.###", CultureInfo.InvariantCulture) ?? "";

    private static int? Int(string text) =>
        int.TryParse(text.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) ? v : null;

    private static double? Double(string text) =>
        double.TryParse(text.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? v : null;

    // ── memory ───────────────────────────────────────────────────────────────

    private async Task InitGpuAsync()
    {
        var gpu = await SystemInfo.SampleGpuAsync();
        if (gpu is null)
        {
            GpuSlider.IsEnabled = false;
            GpuLimitCheck.IsEnabled = false;
            GpuHelp.Text = "No NVIDIA GPU was found (nvidia-smi did not answer), so models run on the CPU.";
            return;
        }
        _gpuTotalGb = gpu.TotalBytes / (double)(1L << 30);
        _loading = true;
        GpuSlider.Maximum = Math.Max(1, Math.Floor(_gpuTotalGb));
        GpuSlider.Value = Profile.GpuMemoryGb ?? Math.Min(8, GpuSlider.Maximum);
        _loading = false;
        GpuRadio.Content = $"Use the GPU ({gpu.Name}, {_gpuTotalGb:0} GB)";
        UpdateMemoryText();
    }

    private void UpdateMemoryText()
    {
        double totalRam = SystemInfo.TotalRamBytes / (double)(1L << 30);
        RamValueText.Text = RamLimitCheck.IsChecked == true ? $"{RamSlider.Value:0} GB" : "No limit";
        RamHelp.Text = RamLimitCheck.IsChecked == true
            ? $"The server aims to stay under {RamSlider.Value:0} of {totalRam:0} GB: weights stream through a cache sized to fit. Slower, but runs models larger than memory."
            : $"Models are held in memory whole ({totalRam:0} GB installed). Limit it to run models bigger than that, or to leave room for other work.";
        GpuValueText.Text = GpuLimitCheck.IsChecked == true ? $"{GpuSlider.Value:0.#} GB" : "Use free memory";
        if (_gpuTotalGb > 0)
        {
            GpuHelp.Text = GpuLimitCheck.IsChecked == true
                ? (GpuSlider.Value == 0
                    ? "0 GB keeps every weight on the host; the GPU still computes."
                    : $"Weights may take up to {GpuSlider.Value:0.#} of {_gpuTotalGb:0} GB on the GPU; the rest stays on the host.")
                : "The server takes the GPU's free memory, less a margin.";
        }
    }

    // ── models ───────────────────────────────────────────────────────────────

    private async void OnRescan(object sender, RoutedEventArgs e) => await RescanAsync();

    private async Task RescanAsync()
    {
        FoldersText.Text = "Looking for models…";
        _found = await ModelLibrary.ScanAsync(Settings);
        // Models chosen earlier but no longer under a scanned folder stay listed.
        foreach (var served in Profile.Models.Where(m => _found.All(f => !SamePath(f.Path, m.Path))).ToList())
        {
            if (Directory.Exists(served.Path))
            {
                _found.Add(ModelLibrary.Describe(served.Path));
            }
        }
        BuildRows();
        FoldersText.Text = Settings.ModelFolders.Count == 0
            ? "No folders to search yet: add the folder your containers are in."
            : $"Searching {string.Join(", ", Settings.ModelFolders.Select(ShortPath))} · {_found.Count} model{(_found.Count == 1 ? "" : "s")} found";
    }

    private static bool SamePath(string a, string b) =>
        string.Equals(Path.GetFullPath(a).TrimEnd('\\', '/'), Path.GetFullPath(b).TrimEnd('\\', '/'), StringComparison.OrdinalIgnoreCase);

    private static string ShortPath(string p) => p.Length > 48 ? "…" + p[^46..] : p;

    private ServedModel? Served(ModelEntry m) => Profile.Models.FirstOrDefault(s => SamePath(s.Path, m.Path));

    private void BuildRows()
    {
        ModelRows.Children.Clear();
        if (_found.Count == 0)
        {
            ModelRows.Children.Add(new TextBlock
            {
                Text = "No containers found. Add a folder that holds them, or make one with Tools ▸ encode (Hugging Face) or from-gguf.",
                Style = (Style)FindResource("Caption"),
            });
        }
        foreach (var m in _found)
        {
            ModelRows.Children.Add(BuildRow(m));
        }
        UpdateSelectionText();
    }

    private FrameworkElement BuildRow(ModelEntry m)
    {
        bool servable = m.Kind is not (ModelKind.Unservable or ModelKind.Diffusion);
        var served = Served(m);
        var outer = new StackPanel { Margin = new Thickness(0, 0, 0, 2) };
        var row = new DockPanel { Margin = new Thickness(0, 3, 0, 3) };

        var options = new Button
        {
            Content = new TextBlock { Text = "", Style = (Style)FindResource("Icon") },
            ToolTip = "Options for this model",
            IsEnabled = servable,
            Margin = new Thickness(6, 0, 0, 0),
        };
        DockPanel.SetDock(options, Dock.Right);
        row.Children.Add(options);
        var explore = new Button
        {
            Content = new TextBlock { Text = "", Style = (Style)FindResource("Icon") },
            ToolTip = "Open in the Explorer",
            Margin = new Thickness(6, 0, 0, 0),
        };
        explore.Click += (_, _) => ((ExplorerPage)_shell.Navigate("explorer")).Open(m.Path);
        DockPanel.SetDock(explore, Dock.Right);
        row.Children.Add(explore);

        var check = new CheckBox { IsChecked = served?.Enabled == true, IsEnabled = servable, VerticalAlignment = VerticalAlignment.Center };
        var label = new StackPanel { Orientation = Orientation.Horizontal };
        label.Children.Add(new TextBlock { Text = served is { Id.Length: > 0 } ? served.Id : m.Name, FontWeight = FontWeights.SemiBold, VerticalAlignment = VerticalAlignment.Center });
        label.Children.Add(Badge(m.KindLabel, m.Kind));
        label.Children.Add(new TextBlock
        {
            Text = "  " + m.SizeLabel + (m.ContextLength is { } c ? $" · {c / 1024}K context" : "") + (m.StoredFormat is { } f ? $" · {f.ToUpperInvariant()}" : "")
                   + (m.Problem is { } p ? $" · {p}" : ""),
            Style = (Style)FindResource("Caption"),
            VerticalAlignment = VerticalAlignment.Center,
            TextTrimming = TextTrimming.CharacterEllipsis,
        });
        check.Content = label;
        check.ToolTip = m.Path;
        row.Children.Add(check);
        outer.Children.Add(row);

        var panel = BuildOptions(m);
        panel.Visibility = Visibility.Collapsed;
        outer.Children.Add(panel);
        options.Click += (_, _) => panel.Visibility = panel.Visibility == Visibility.Visible ? Visibility.Collapsed : Visibility.Visible;

        check.Checked += (_, _) => SetEnabled(m, true);
        check.Unchecked += (_, _) => SetEnabled(m, false);
        return outer;
    }

    private Border Badge(string text, ModelKind kind)
    {
        var color = kind switch
        {
            ModelKind.Chat => Color.FromRgb(0x3B, 0x82, 0xF6),
            ModelKind.Embedding => Color.FromRgb(0x8B, 0x5C, 0xF6),
            ModelKind.Classifier => Color.FromRgb(0xD9, 0x77, 0x06),
            ModelKind.Decision => Color.FromRgb(0x10, 0x9E, 0x6E),
            _ => Color.FromRgb(0x8A, 0x8F, 0x98),
        };
        return new Border
        {
            Style = (Style)FindResource("Badge"),
            Background = new SolidColorBrush(Color.FromArgb(0x30, color.R, color.G, color.B)),
            Child = new TextBlock { Text = text, FontSize = 11.5, Foreground = new SolidColorBrush(color), FontWeight = FontWeights.SemiBold },
        };
    }

    private ServedModel Ensure(ModelEntry m)
    {
        var served = Served(m);
        if (served is null)
        {
            served = new ServedModel { Path = m.Path, Enabled = false };
            Profile.Models.Add(served);
        }
        return served;
    }

    private void SetEnabled(ModelEntry m, bool enabled)
    {
        Ensure(m).Enabled = enabled;
        // Unticked models with nothing set are dropped, so the list does not fill with leftovers.
        Profile.Models.RemoveAll(s => !s.Enabled && s.Id.Length == 0 && s.Format.Length == 0 && s.MaxContext.Length == 0 && s.Think is null && s.MaxTokens is null
                                      && s.Temperature is null && s.TopP is null && s.TopK is null && s.Adapters.Count == 0);
        Changed();
    }

    /// <summary>The per-model panel: name, 4-bit format, defaults, adapters.</summary>
    private FrameworkElement BuildOptions(ModelEntry m)
    {
        var s = Served(m) ?? new ServedModel { Path = m.Path };
        var grid = new Grid { Margin = new Thickness(28, 4, 40, 10) };
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(160) });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        int row = 0;
        void Add(string label, FrameworkElement editor, string? help = null)
        {
            grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var l = new TextBlock { Text = label, Style = (Style)FindResource("FieldLabel"), ToolTip = help };
            Grid.SetRow(l, row);
            grid.Children.Add(l);
            editor.Margin = new Thickness(0, 2, 0, 2);
            editor.ToolTip ??= help;
            Grid.SetRow(editor, row);
            Grid.SetColumn(editor, 1);
            grid.Children.Add(editor);
            row++;
        }
        TextBox Box(string text, Action<string> set, double width = 120)
        {
            var b = new TextBox { Text = text, Width = width, HorizontalAlignment = HorizontalAlignment.Left };
            b.TextChanged += (_, _) => { set(b.Text); Changed(); };
            return b;
        }

        Add("Name", Box(s.Id, v => Ensure(m).Id = v.Trim(), 240), $"The id requests use (\"model\": …). Empty uses '{m.Name}'.");
        var format = new ComboBox { MinWidth = 200, HorizontalAlignment = HorizontalAlignment.Left };
        foreach (var (text, tag) in new[] { ("Server default", ""), ("As stored / FP16", "fp16"), ("NVFP4 (4-bit)", "nvfp4"), ("MXFP4 (4-bit)", "mxfp4") })
        {
            format.Items.Add(new ComboBoxItem { Content = text, Tag = tag });
        }
        format.SelectedItem = format.Items.Cast<ComboBoxItem>().First(i => (string)i.Tag == s.Format);
        format.SelectionChanged += (_, _) => { Ensure(m).Format = (string)((ComboBoxItem)format.SelectedItem).Tag; Changed(); };
        Add("GPU format", format, "A 4-bit format serves a GPTQ-rounded copy, built beside the container the first time");

        if (m.Kind == ModelKind.Chat)
        {
            var think = new ComboBox { MinWidth = 200, HorizontalAlignment = HorizontalAlignment.Left };
            foreach (var (text, value) in new (string, bool?)[] { ("Server default", null), ("Think", true), ("Don't think", false) })
            {
                think.Items.Add(new ComboBoxItem { Content = text, Tag = value });
            }
            think.SelectedItem = think.Items.Cast<ComboBoxItem>().First(i => (bool?)i.Tag == s.Think);
            think.SelectionChanged += (_, _) => { Ensure(m).Think = (bool?)((ComboBoxItem)think.SelectedItem).Tag; Changed(); };
            Add("Reasoning", think);
            var context = new StackPanel { Orientation = Orientation.Horizontal };
            var contextChoice = new ComboBox { MinWidth = 200 };
            foreach (var (text, tag) in new[] { ("The model's own (default)", ""), ("512K tokens", "512K"), ("1M tokens", "1M"), ("Other…", "other") })
            {
                contextChoice.Items.Add(new ComboBoxItem { Content = text, Tag = tag });
            }
            contextChoice.SelectedItem = contextChoice.Items.Cast<ComboBoxItem>()
                .FirstOrDefault(i => string.Equals((string)i.Tag, s.MaxContext.Trim(), StringComparison.OrdinalIgnoreCase)) ?? contextChoice.Items[^1];
            var contextBox = Box(s.MaxContext, v => Ensure(m).MaxContext = v.Trim());
            contextBox.Margin = new Thickness(6, 0, 0, 0);
            contextBox.ToolTip = "A token count, or K and M: 1048576, 768K, 1M";
            contextBox.Visibility = ReferenceEquals(contextChoice.SelectedItem, contextChoice.Items[^1]) ? Visibility.Visible : Visibility.Collapsed;
            contextChoice.SelectionChanged += (_, _) =>
            {
                string tag = (string)((ComboBoxItem)contextChoice.SelectedItem).Tag;
                contextBox.Visibility = tag == "other" ? Visibility.Visible : Visibility.Collapsed;
                if (tag != "other")
                {
                    contextBox.Text = tag;
                }
            };
            context.Children.Add(contextChoice);
            context.Children.Add(contextBox);
            Add("Context length", context,
                "How many tokens the model can hold. Longer than the model was trained to needs the GPU memory for it: see Prompt state (KV) format under Defaults.");
            Add("Max reply tokens", Box(s.MaxTokens?.ToString(CultureInfo.InvariantCulture) ?? "", v => Ensure(m).MaxTokens = Int(v)));
            Add("Temperature", Box(Num(s.Temperature), v => Ensure(m).Temperature = Double(v)));
            Add("Top-p", Box(Num(s.TopP), v => Ensure(m).TopP = Double(v)));
            Add("Top-k", Box(s.TopK?.ToString(CultureInfo.InvariantCulture) ?? "", v => Ensure(m).TopK = Int(v)));

            var adapters = new DockPanel();
            var adapterBox = new TextBox { Text = string.Join("; ", s.Adapters) };
            var browse = new Button { Content = new TextBlock { Text = "", Style = (Style)FindResource("Icon") }, Margin = new Thickness(6, 0, 0, 0), ToolTip = "Add an adapter folder" };
            browse.Click += (_, _) =>
            {
                var dlg = new OpenFolderDialog { Title = "Entail adapter folder" };
                if (dlg.ShowDialog(Window.GetWindow(this)) == true)
                {
                    adapterBox.Text = string.Join("; ", adapterBox.Text.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Append(dlg.FolderName));
                }
            };
            adapterBox.TextChanged += (_, _) =>
            {
                Ensure(m).Adapters = adapterBox.Text.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
                Changed();
            };
            DockPanel.SetDock(browse, Dock.Right);
            adapters.Children.Add(browse);
            adapters.Children.Add(adapterBox);
            Add("Entail adapters", adapters, "Adapter folders run beside the model, separated by ';'");
        }
        grid.Children.Add(new TextBlock()); // keeps the grid non-empty for non-chat models
        return grid;
    }

    private void UpdateSelectionText()
    {
        var chosen = _found.Where(f => Served(f)?.Enabled == true).ToList();
        if (chosen.Count == 0)
        {
            SelectionText.Text = "No models chosen — tick the ones to load.";
            return;
        }
        long bytes = chosen.Sum(c => c.SizeBytes);
        var kinds = chosen.GroupBy(c => c.KindLabel).Select(g => $"{g.Count()} {g.Key.ToLowerInvariant()}");
        string fit = "";
        if (RamLimitCheck.IsChecked != true && bytes > SystemInfo.TotalRamBytes * 0.7)
        {
            fit = " — more than fits comfortably in memory; consider a RAM limit";
        }
        SelectionText.Text = $"{chosen.Count} model{(chosen.Count == 1 ? "" : "s")} chosen ({string.Join(", ", kinds)}) · {Format.Bytes(bytes)} on disk{fit}";
    }

    private void OnAddFolder(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog { Title = "A folder that holds containers" };
        if (dlg.ShowDialog(Window.GetWindow(this)) == true && !Settings.ModelFolders.Contains(dlg.FolderName, StringComparer.OrdinalIgnoreCase))
        {
            Settings.ModelFolders.Add(dlg.FolderName);
            Settings.Save();
            _ = RescanAsync();
        }
    }

    private void OnAddContainer(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog { Title = "A container folder (it holds index.json)" };
        if (dlg.ShowDialog(Window.GetWindow(this)) != true) return;
        if (!File.Exists(Path.Combine(dlg.FolderName, "index.json")))
        {
            MessageBox.Show(Window.GetWindow(this), "That folder is not a container (it has no index.json). Make one with Tools ▸ encode or from-gguf.",
                "Add model", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        Settings.ExtraModels.Add(dlg.FolderName);
        Settings.Save();
        _ = RescanAsync();
    }

    // ── server state ────────────────────────────────────────────────────────

    private void OnServerChanged(object? sender, PropertyChangedEventArgs e) => RefreshServer();

    private void RefreshServer()
    {
        var server = ServerController.Instance;
        StatusText.Text = server.Detail;
        StatusDot.Fill = (Brush)FindResource(server.State switch
        {
            ServerState.Running or ServerState.External => "GoodBrush",
            ServerState.Starting or ServerState.Stopping => "WarnBrush",
            ServerState.Failed => "BadBrush",
            _ => "IdleBrush",
        });
        bool stop = server.State is ServerState.Running or ServerState.Starting;
        StartStopText.Text = server.State switch
        {
            ServerState.Stopping => "Stopping…",
            ServerState.External => "Forget server",
            _ when stop => "Stop server",
            _ => "Start server",
        };
        StartStopIcon.Text = stop ? "" : "";
        StartStopButton.IsEnabled = server.State != ServerState.Stopping;
        LoadProgress.Visibility = server.State == ServerState.Starting ? Visibility.Visible : Visibility.Collapsed;
        LoadProgress.IsIndeterminate = server.ModelsTotal == 0;
        LoadProgress.Maximum = Math.Max(1, server.ModelsTotal);
        LoadProgress.Value = server.ModelsLoaded;
        EndpointPanel.Visibility = server.IsRunning ? Visibility.Visible : Visibility.Collapsed;
        EndpointText.Text = server.BaseUrl + "/v1";
        if (!server.IsRunning && server.State != ServerState.Starting)
        {
            _changedWhileRunning = false;
        }
        if (server.State == ServerState.Failed && _lastState != ServerState.Failed)
        {
            // Show the server's output, which says why.
            AdvancedExpander.IsExpanded = true;
        }
        _lastState = server.State;
        RestartHint.Visibility = _changedWhileRunning ? Visibility.Visible : Visibility.Collapsed;
    }

    private void RefreshTools()
    {
        var found = ToolLocator.Find(ToolLocator.Server);
        ToolText.Text = found is null
            ? "amql-server was not found. Build src/Amql.Server (dotnet build) or put amql-server on PATH."
            : "Using " + found.Describe();
    }

    private void UpdateCommandLine()
    {
        var models = Profile.Models.Where(m => m.Enabled).Select(m => $"{m.EffectiveId}={m.Path}");
        CommandLineText.Text = "amql-server --config <" + string.Join(", ", models) + "> " +
                               string.Join(' ', ServerController.Arguments(Profile).Select(a => a.Contains(' ') ? $"\"{a}\"" : a));
        if (ServerController.Instance.LastCommandLine is { Length: > 0 } last)
        {
            CommandLineText.Text += Environment.NewLine + Environment.NewLine + "Last started as:" + Environment.NewLine + last;
        }
    }

    private void OnConsoleLine(string line)
    {
        if (ConsoleBox.LineCount > 2000)
        {
            ConsoleBox.Text = string.Join(Environment.NewLine, ServerController.Instance.ConsoleSnapshot());
        }
        else
        {
            ConsoleBox.AppendText((ConsoleBox.Text.Length > 0 ? Environment.NewLine : "") + line);
        }
        ConsoleBox.ScrollToEnd();
        if (line.StartsWith("> ", StringComparison.Ordinal))
        {
            UpdateCommandLine();
        }
    }

    private async void OnStartStop(object sender, RoutedEventArgs e)
    {
        if (ServerController.Instance.CanStart)
        {
            ToolLocator.Refresh();
            RefreshTools();
            ConsoleBox.Clear();
        }
        await _shell.ToggleServerAsync();
    }

    private void OnOpenMonitor(object sender, RoutedEventArgs e) => _shell.OpenMonitor();

    private void OnCopyEndpoint(object sender, RoutedEventArgs e) => Clipboard.SetText(EndpointText.Text);

    private void OnCopyConsole(object sender, RoutedEventArgs e) => Clipboard.SetText(ConsoleBox.Text);
}
