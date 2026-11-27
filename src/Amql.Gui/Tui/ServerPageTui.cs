// TextView is marked obsolete in favour of a separate editor package; it is still the text area Terminal.Gui ships.
#pragma warning disable CS0618
using System.IO;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Globalization;
using Amql.Gui.Services;
using Terminal.Gui.Input;
using Terminal.Gui.ViewBase;
using Terminal.Gui.Views;

namespace Amql.Gui.Tui;

/// <summary>
/// Sets up and runs amql-server: which models load (each container's kind
/// read from its system graph), how much RAM and GPU memory they may take,
/// and — under Advanced — networking, request defaults and the server's
/// optional behaviours. Every change is saved straight away, to the same
/// settings the desktop app uses.
/// </summary>
internal sealed class ServerPageTui : ITuiPage
{
    private readonly Shell _shell;
    private readonly View _root = new() { CanFocus = true };
    private readonly ColorLabel _dot = new(TuiKit.Idle) { Text = "●", X = 0, Y = 0, Width = 1 };
    private readonly Label _status = new() { X = 2, Y = 0, Width = Dim.Fill(46) };
    private readonly Label _endpoint = new() { X = 2, Y = 1, Width = Dim.Fill(46) };
    private readonly ProgressBar _progress = new() { X = 2, Y = 2, Width = Dim.Fill(46), Visible = false };
    private readonly Label _hint = new() { X = 2, Y = 2, Width = Dim.Fill(46), Visible = false };
    private readonly Label _tool = new() { X = 2, Y = 3, Width = Dim.Fill(2) };
    private readonly Button _startStop;
    private readonly Tabs _tabs;
    private readonly View _outputTab;

    private readonly ListView _modelList = new() { X = 0, Y = 1, Width = Dim.Fill(), Height = Dim.Fill(3) };
    private readonly ObservableCollection<string> _modelRows = new();
    private readonly Label _folders = new() { X = 0, Y = 0, Width = Dim.Fill() };
    private readonly Label _selection = new() { X = 0, Y = Pos.AnchorEnd(2), Width = Dim.Fill() };
    private readonly FieldList _memory = new(3);
    private readonly FieldList _advanced = new(3);
    private readonly TextView _commandLine = new() { X = 0, Y = 1, Width = Dim.Fill(), Height = 5, ReadOnly = true, WordWrap = true };
    private readonly TextView _console = new() { X = 0, Y = 8, Width = Dim.Fill(), Height = Dim.Fill(), ReadOnly = true };

    private List<ModelEntry> _found = new();
    private GpuSample? _gpu;
    private bool _changedWhileRunning;
    private ServerState _lastState;
    private bool _scanned;

    private static AppSettings Settings => AppSettings.Current;

    private static ServerProfile Profile => Settings.Server;

    public ServerPageTui(Shell shell)
    {
        _shell = shell;
        _root.Add(_dot, _status, _endpoint, _progress, _hint, _tool);
        _startStop = TuiKit.Button("Start server", () => TuiKit.GuardAsync(StartStopAsync), Pos.AnchorEnd(44), 0);
        var monitor = TuiKit.Button("Monitor", () => _shell.Navigate("monitor"), Pos.AnchorEnd(26), 0);
        var copy = TuiKit.Button("Copy URL", () => TuiKit.Copy(ServerController.Instance.BaseUrl + "/v1"), Pos.AnchorEnd(14), 0);
        _root.Add(_startStop, monitor, copy);

        _tabs = new Tabs { TabStop = TabBehavior.TabStop, X = 0, Y = 5, Width = Dim.Fill(), Height = Dim.Fill() };

        // Models
        var models = new View { Title = "_Models", Width = Dim.Fill(), Height = Dim.Fill(), CanFocus = true };
        _modelList.SetSource(_modelRows);
        _modelList.KeyDown += OnModelKey;
        TuiKit.OnLetter(_modelList, 'e', ExploreSelected);
        _modelList.Accepting += (_, e) =>
        {
            e.Handled = true;
            EditSelectedModel();
        };
        var buttons = new View { X = 0, Y = Pos.AnchorEnd(1), Width = Dim.Fill(), Height = 1 };
        var addFolder = TuiKit.Button("Add folder", AddFolder, 0, 0);
        var addModel = TuiKit.Button("Add model", AddContainer, Pos.Right(addFolder) + 1, 0);
        var rescan = TuiKit.Button("Rescan", () => TuiKit.GuardAsync(RescanAsync), Pos.Right(addModel) + 1, 0);
        var options = TuiKit.Button("Options", EditSelectedModel, Pos.Right(rescan) + 1, 0);
        var explore = TuiKit.Button("Explore", ExploreSelected, Pos.Right(options) + 1, 0);
        buttons.Add(addFolder, addModel, rescan, options, explore);
        buttons.Add(new Label { Text = "Space tick · Enter options · e explore", X = Pos.Right(explore) + 2, Y = 0, Width = Dim.Fill() });
        models.Add(_folders, _modelList, _selection, buttons);

        // Memory and device
        var memory = new View { Title = "Memory and _device", Width = Dim.Fill(), Height = Dim.Fill(), CanFocus = true };
        memory.Add(_memory);
        _memory.Changed += _ => Changed();

        // Advanced
        var advanced = new View { Title = "_Advanced", Width = Dim.Fill(), Height = Dim.Fill(), CanFocus = true };
        advanced.Add(_advanced);
        _advanced.Changed += _ => Changed();

        // Output
        _outputTab = new View { Title = "_Output", Width = Dim.Fill(), Height = Dim.Fill(), CanFocus = true };
        _outputTab.Add(new Label { Text = "Command line", X = 0, Y = 0 });
        _outputTab.Add(_commandLine);
        _outputTab.Add(new Label { Text = "Server output — the server's own messages, kept for diagnosing a start that fails", X = 0, Y = 7, Width = Dim.Fill(10) });
        _outputTab.Add(TuiKit.Button("Copy", () => TuiKit.Copy(_console.Text), Pos.AnchorEnd(8), 6));
        _outputTab.Add(_console);

        _tabs.Add(models, memory, advanced, _outputTab);
        _root.Add(_tabs);

        BuildMemoryFields();
        BuildAdvancedFields();
        ServerController.Instance.PropertyChanged += OnServerChanged;
        ServerController.Instance.ConsoleLine += OnConsoleLine;
        _console.Text = string.Join("\n", ServerController.Instance.ConsoleSnapshot());
        RefreshServer();
        RefreshTools();
        UpdateCommandLine();
        TuiKit.GuardAsync(InitGpuAsync);
    }

    public View View => _root;

    public void Shown()
    {
        if (!_scanned)
        {
            _scanned = true;
            TuiKit.GuardAsync(RescanAsync);
        }
    }

    public void Focus() => _modelList.SetFocus();

    public void Hidden()
    {
    }

    // ── memory and device ───────────────────────────────────────────────────

    private double MaxRamGb => Math.Max(4, Math.Floor(SystemInfo.TotalRamBytes / (double)(1L << 30)));

    private double MaxGpuGb => _gpu is null ? 24 : Math.Max(1, Math.Floor(_gpu.TotalBytes / (double)(1L << 30)));

    private static string Num(double? v) => v?.ToString("0.###", CultureInfo.InvariantCulture) ?? "";

    private static int? Int(string text) =>
        int.TryParse(text.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) ? v : null;

    private static double? Double(string text) =>
        double.TryParse(text.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? v : null;

    private Func<string, string?> Range(double min, Func<double> max, double step) => s =>
        double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) && v >= min && v <= max()
            ? null
            : $"Enter a number of GB from {min} to {max():0} (in steps of {step}).";

    // The sliders' last positions are kept while a limit is off, as the desktop page does.
    private double _ramValue = Profile.MaxRamGb ?? 10;
    private double _gpuValue = Profile.GpuMemoryGb ?? 8;

    private void BuildMemoryFields()
    {
        _ramValue = Math.Min(_ramValue, MaxRamGb);
        _memory.SetFields(new Field[]
        {
            new ChoiceField<string>("Device",
                new[] { (GpuLabel(), "gpu"), ("CPU only", "cpu") },
                () => Profile.Device == "cpu" ? "cpu" : "gpu", v => Profile.Device = v,
                "Use the GPU when there is one, or keep everything on the CPU."),
            new BoolField("Limit system memory (RAM)", () => Profile.MaxRamGb is not null,
                v => Profile.MaxRamGb = v ? _ramValue : null) { HelpText = () => ServerOptions.RamHelp(Profile.MaxRamGb) },
            new TextValueField("RAM limit (GB)", () => Num(Profile.MaxRamGb ?? _ramValue), v =>
            {
                _ramValue = Math.Round(Double(v) ?? _ramValue);
                Profile.MaxRamGb = _ramValue;
            })
            {
                HelpText = () => ServerOptions.RamHelp(Profile.MaxRamGb),
                IsEnabled = () => Profile.MaxRamGb is not null,
                Validate = Range(2, () => MaxRamGb, 1),
            },
            new BoolField("Limit GPU memory", () => Profile.GpuMemoryGb is not null,
                v => Profile.GpuMemoryGb = v ? _gpuValue : null)
            {
                HelpText = () => ServerOptions.GpuHelp(Profile.GpuMemoryGb, _gpu),
                IsEnabled = () => _gpu is not null,
            },
            new TextValueField("GPU limit (GB)", () => Num(Profile.GpuMemoryGb ?? _gpuValue), v =>
            {
                _gpuValue = Math.Round((Double(v) ?? _gpuValue) * 2) / 2;
                Profile.GpuMemoryGb = _gpuValue;
            })
            {
                HelpText = () => ServerOptions.GpuHelp(Profile.GpuMemoryGb, _gpu),
                IsEnabled = () => Profile.GpuMemoryGb is not null && _gpu is not null,
                Validate = Range(0, () => MaxGpuGb, 0.5),
            },
        });
    }

    private string GpuLabel() => _gpu is null
        ? "Use the GPU when there is one"
        : $"Use the GPU ({_gpu.Name}, {_gpu.TotalBytes / (double)(1L << 30):0} GB{(_gpu.Unified ? " shared" : "")})";

    private async Task InitGpuAsync()
    {
        _gpu = await SystemInfo.SampleGpuAsync();
        _gpuValue = Math.Min(_gpuValue, MaxGpuGb);
        BuildMemoryFields();
    }

    // ── advanced ────────────────────────────────────────────────────────────

    private void BuildAdvancedFields()
    {
        var fields = new List<Field>
        {
            new HeaderField("Network"),
            new TextValueField("Port", () => Profile.Port.ToString(CultureInfo.InvariantCulture), v => Profile.Port = int.Parse(v, CultureInfo.InvariantCulture),
                "The port the server listens on.")
            {
                Validate = s => int.TryParse(s, out var p) && p is > 0 and < 65536 ? null : "Enter a port from 1 to 65535.",
            },
            new BoolField("Accept connections from other computers", () => Profile.ListenOnNetwork, v => Profile.ListenOnNetwork = v,
                "Listen on 0.0.0.0 instead of only this machine (127.0.0.1)."),
            new TextValueField("API key", () => Profile.ApiKey, v => Profile.ApiKey = v,
                "Clients must send this as a Bearer token. Leave empty for no authentication.") { Secret = true, Placeholder = "none" },
            new BoolField("Stop the server when AMQL Studio closes", () => Profile.StopOnExit, v => Profile.StopOnExit = v),

            new HeaderField("Defaults for chat requests",
                "Used when a request leaves them out. A model's own options win over these. Empty means the server's default."),
            new BoolField("Think before answering", () => Profile.Think, v => Profile.Think = v, "Reasoning on by default."),
            new TextValueField("Max reply tokens", () => Profile.MaxTokens?.ToString(CultureInfo.InvariantCulture) ?? "", v => Profile.MaxTokens = Int(v))
                { Placeholder = "server default", Validate = TextValueField.OptionalInt },
            new TextValueField("Temperature", () => Num(Profile.Temperature), v => Profile.Temperature = Double(v))
                { Placeholder = "server default", Validate = TextValueField.OptionalNumber },
            new TextValueField("Top-p", () => Num(Profile.TopP), v => Profile.TopP = Double(v))
                { Placeholder = "server default", Validate = TextValueField.OptionalNumber },
            new TextValueField("Top-k", () => Profile.TopK?.ToString(CultureInfo.InvariantCulture) ?? "", v => Profile.TopK = Int(v))
                { Placeholder = "server default", Validate = TextValueField.OptionalInt },
            new ChoiceField<string>("GPU weight format", ServerOptions.GpuFormats, () => Profile.GpuFormat, v => Profile.GpuFormat = v,
                "NVFP4/MXFP4 serve each container's GPTQ-rounded 4-bit copy, built beside it the first time."),
            new ChoiceField<string>("Prompt state (KV) format", ServerOptions.KvFormats, () => Profile.KvFormat, v => Profile.KvFormat = v,
                "How a conversation is held in GPU memory. NVFP4 takes about a third of FP16's memory, for contexts past 256K."),

            new HeaderField("Serving behaviour", "Each of these turns off something the server does by default."),
        };
        foreach (var b in ServerOptions.Behaviours)
        {
            fields.Add(new BoolField(b.Label, () => b.Get(Profile), v => b.Set(Profile, v), b.Help));
        }
        fields.Add(new TextValueField("N-gram packs", () => string.Join("\n", Profile.NGramPacks),
            v => Profile.NGramPacks = v.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList(),
            "N-gram pack files loaded with --ngram-pack, one per line.") { Multiline = true, Placeholder = "none" });
        fields.Add(new TextValueField("Extra arguments", () => Profile.ExtraArguments, v => Profile.ExtraArguments = v,
            "Passed to amql-server as typed, after everything else (e.g. --ngram-lambda 0.3).") { Placeholder = "none" });
        _advanced.SetFields(fields);
    }

    /// <summary>A change to what the server runs: saved, and flagged if it needs a restart.</summary>
    private void Changed()
    {
        Settings.Save();
        UpdateSelection();
        UpdateCommandLine();
        if (ServerController.Instance.State is ServerState.Running or ServerState.Starting)
        {
            _changedWhileRunning = true;
            RefreshServer();
        }
    }

    // ── models ──────────────────────────────────────────────────────────────

    private async Task RescanAsync()
    {
        _folders.Text = "Looking for models…";
        _found = await ModelLibrary.ScanAsync(Settings);
        // Models chosen earlier but no longer under a scanned folder stay listed.
        foreach (var served in Profile.Models.Where(m => _found.All(f => !ServerOptions.SamePath(f.Path, m.Path))).ToList())
        {
            if (Directory.Exists(served.Path))
            {
                _found.Add(ModelLibrary.Describe(served.Path));
            }
        }
        BuildRows();
        _folders.Text = Settings.ModelFolders.Count == 0
            ? "No folders to search yet: add the folder your containers are in."
            : $"Searching {string.Join(", ", Settings.ModelFolders.Select(ServerOptions.ShortPath))} · {_found.Count} model{(_found.Count == 1 ? "" : "s")} found";
    }

    private static bool Servable(ModelEntry m) => m.Kind is not (ModelKind.Unservable or ModelKind.Diffusion);

    private void BuildRows()
    {
        int? keep = _modelList.SelectedItem;
        _modelRows.Clear();
        if (_found.Count == 0)
        {
            _modelRows.Add("  No containers found. Add a folder that holds them, or make one with Tools ▸ encode (Hugging Face) or from-gguf.");
        }
        int nameWidth = Math.Clamp(_found.Select(m => DisplayName(m).Length).DefaultIfEmpty(10).Max() + 1, 12, 44);
        foreach (var m in _found)
        {
            var served = ServerOptions.Served(Profile, m.Path);
            string mark = !Servable(m) ? "[-]" : served?.Enabled == true ? "[x]" : "[ ]";
            string details = m.SizeLabel + (m.ContextLength is { } c ? $" · {c / 1024}K context" : "") +
                             (m.StoredFormat is { } f ? $" · {f.ToUpperInvariant()}" : "") + (m.Problem is { } p ? $" · {p}" : "");
            string options = served is { } s && (s.Format.Length > 0 || s.MaxContext.Length > 0 || s.Think is not null || s.Adapters.Count > 0) ? " ⚙" : "";
            _modelRows.Add($"{mark} {TuiKit.Shorten(DisplayName(m), nameWidth).PadRight(nameWidth)} {m.KindLabel,-11} {details}{options}");
        }
        if (keep is { } k && _modelRows.Count > 0)
        {
            _modelList.SelectedItem = Math.Min(k, _modelRows.Count - 1);
        }
        UpdateSelection();
    }

    private static string DisplayName(ModelEntry m) =>
        ServerOptions.Served(Profile, m.Path) is { Id.Length: > 0 } s ? s.Id : m.Name;

    private ModelEntry? SelectedModel =>
        _modelList.SelectedItem is { } i && i >= 0 && i < _found.Count ? _found[i] : null;

    private void OnModelKey(object? sender, Key key)
    {
        if (key == Key.Space && SelectedModel is { } m)
        {
            key.Handled = true;
            if (!Servable(m))
            {
                TuiKit.Say($"{m.Name} cannot be served: {m.Problem ?? m.KindLabel}");
                return;
            }
            ServerOptions.SetEnabled(Profile, m.Path, ServerOptions.Served(Profile, m.Path)?.Enabled != true);
            BuildRows();
            Changed();
        }

    }

    private void UpdateSelection() => _selection.Text = ServerOptions.SelectionSummary(_found, Profile);

    private void ExploreSelected()
    {
        if (SelectedModel is { } m)
        {
            _shell.Navigate("explorer");
            _shell.Explorer.Open(m.Path);
        }
    }

    private void AddFolder()
    {
        if (TuiKit.PickPath("A folder that holds containers", directory: true, Settings.ModelFolders.FirstOrDefault() ?? "") is { } folder
            && !Settings.ModelFolders.Any(f => ServerOptions.SamePath(f, folder)))
        {
            Settings.ModelFolders.Add(folder);
            Settings.Save();
            TuiKit.GuardAsync(RescanAsync);
        }
    }

    private void AddContainer()
    {
        if (TuiKit.PickPath("A container folder (it holds index.json)", directory: true) is not { } folder)
        {
            return;
        }
        if (!File.Exists(Path.Combine(folder, "index.json")))
        {
            TuiKit.Info("Add model", "That folder is not a container (it has no index.json). Make one with Tools ▸ encode or from-gguf.");
            return;
        }
        Settings.ExtraModels.Add(folder);
        Settings.Save();
        TuiKit.GuardAsync(RescanAsync);
    }

    /// <summary>The per-model options: name, 4-bit format, defaults, adapters.</summary>
    private void EditSelectedModel()
    {
        if (SelectedModel is not { } m)
        {
            return;
        }
        if (!Servable(m))
        {
            TuiKit.Info(m.Name, $"This container cannot be served: {m.Problem ?? m.KindLabel}.");
            return;
        }
        ServedModel S() => ServerOptions.Ensure(Profile, m.Path);
        var current = ServerOptions.Served(Profile, m.Path) ?? new ServedModel { Path = m.Path };
        bool chat = m.Kind == ModelKind.Chat;
        string savedContext = current.MaxContext.Trim();
        string contextMode = ServerOptions.ContextChoices.Any(c => string.Equals(c.Value, savedContext, StringComparison.OrdinalIgnoreCase))
            ? ServerOptions.ContextChoices.First(c => string.Equals(c.Value, savedContext, StringComparison.OrdinalIgnoreCase)).Value
            : "other";

        var fields = new List<Field>
        {
            new InfoField("Container", () => m.Path),
            new InfoField("Kind", () => m.Summary),
            new TextValueField("Name", () => ServerOptions.Served(Profile, m.Path)?.Id ?? "", v => S().Id = v.Trim(),
                $"The id requests use (\"model\": …). Empty uses '{m.Name}'.") { Placeholder = m.Name },
            new ChoiceField<string>("GPU format", ServerOptions.ModelFormats, () => ServerOptions.Served(Profile, m.Path)?.Format ?? "", v => S().Format = v,
                "A 4-bit format serves a GPTQ-rounded copy, built beside the container the first time."),
            new ChoiceField<bool?>("Reasoning", ServerOptions.ThinkChoices, () => ServerOptions.Served(Profile, m.Path)?.Think, v => S().Think = v)
                { IsVisible = () => chat },
            new ChoiceField<string>("Context length",
                ServerOptions.ContextChoices.Append(("Other…", "other")).ToArray(),
                () => contextMode,
                v =>
                {
                    contextMode = v;
                    if (v != "other")
                    {
                        S().MaxContext = v;
                    }
                },
                "How many tokens the model can hold. Longer than the model was trained to needs the GPU memory for it: see Prompt state (KV) format under Advanced.")
                { IsVisible = () => chat },
            new TextValueField("Context (tokens)", () => ServerOptions.Served(Profile, m.Path)?.MaxContext ?? "", v => S().MaxContext = v.Trim(),
                "A token count, or K and M: 1048576, 768K, 1M.")
                { IsVisible = () => chat && contextMode == "other", Placeholder = "model's own" },
            new TextValueField("Max reply tokens", () => ServerOptions.Served(Profile, m.Path)?.MaxTokens?.ToString(CultureInfo.InvariantCulture) ?? "", v => S().MaxTokens = Int(v))
                { IsVisible = () => chat, Placeholder = "server default", Validate = TextValueField.OptionalInt },
            new TextValueField("Temperature", () => Num(ServerOptions.Served(Profile, m.Path)?.Temperature), v => S().Temperature = Double(v))
                { IsVisible = () => chat, Placeholder = "server default", Validate = TextValueField.OptionalNumber },
            new TextValueField("Top-p", () => Num(ServerOptions.Served(Profile, m.Path)?.TopP), v => S().TopP = Double(v))
                { IsVisible = () => chat, Placeholder = "server default", Validate = TextValueField.OptionalNumber },
            new TextValueField("Top-k", () => ServerOptions.Served(Profile, m.Path)?.TopK?.ToString(CultureInfo.InvariantCulture) ?? "", v => S().TopK = Int(v))
                { IsVisible = () => chat, Placeholder = "server default", Validate = TextValueField.OptionalInt },
            new TextValueField("Entail adapters", () => string.Join("\n", ServerOptions.Served(Profile, m.Path)?.Adapters ?? new()),
                v => S().Adapters = v.Split(new[] { '\n', ';' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList(),
                "Adapter folders run beside the model, one per line.")
                { IsVisible = () => chat, Multiline = true, Placeholder = "none" },
            new ActionField("Add an adapter folder…", () =>
            {
                if (TuiKit.PickPath("Entail adapter folder", directory: true) is { } dir)
                {
                    S().Adapters.Add(dir);
                }
            }) { IsVisible = () => chat },
        };

        var dialog = new Dialog { Title = $"Options — {m.Name}", Width = Dim.Percent(85), Height = Dim.Percent(80) };
        var list = new FieldList(3);
        list.SetFields(fields);
        list.Changed += _ => Changed();
        dialog.Add(list);
        dialog.AddButton(new Button { ShadowStyle = ShadowStyles.None, Text = "_Close" });
        list.FocusList();
        TuiKit.App.Run(dialog);
        dialog.Dispose();
        // Options set on an unticked model keep it in the profile; nothing set drops it again.
        ServerOptions.SetEnabled(Profile, m.Path, ServerOptions.Served(Profile, m.Path)?.Enabled == true);
        Changed();
        BuildRows();
    }

    // ── server state ────────────────────────────────────────────────────────

    private void OnServerChanged(object? sender, PropertyChangedEventArgs e) => RefreshServer();

    private void RefreshServer()
    {
        var server = ServerController.Instance;
        _status.Text = server.Detail;
        _dot.Color = TuiKit.StateColor(server.State);
        _dot.SetNeedsDraw();
        bool stop = server.State is ServerState.Running or ServerState.Starting;
        _startStop.Text = server.State switch
        {
            ServerState.Stopping => "Stopping…",
            ServerState.External => "Forget server",
            _ when stop => "Stop server",
            _ => "Start server",
        };
        _startStop.Enabled = server.State != ServerState.Stopping;
        _endpoint.Text = server.IsRunning ? $"Endpoint: {server.BaseUrl}/v1" : "";
        bool starting = server.State == ServerState.Starting;
        _progress.Visible = starting;
        _progress.Fraction = server.ModelsTotal == 0 ? 0 : server.ModelsLoaded / (float)server.ModelsTotal;
        if (!server.IsRunning && !starting)
        {
            _changedWhileRunning = false;
        }
        if (server.State == ServerState.Failed && _lastState != ServerState.Failed)
        {
            // Show the server's output, which says why.
            _tabs.Value = _outputTab;
        }
        _lastState = server.State;
        _hint.Visible = _changedWhileRunning && !starting;
        _hint.Text = "You changed the setup while the server runs. The changes take effect when it restarts.";
    }

    private void RefreshTools()
    {
        var found = ToolLocator.Find(ToolLocator.Server);
        _tool.Text = found is null
            ? "amql-server was not found. Build src/Amql.Server (dotnet build) or put amql-server on PATH."
            : "Using " + found.Describe();
    }

    private void UpdateCommandLine() => _commandLine.Text = ServerOptions.CommandLinePreview(Profile);

    private void OnConsoleLine(string line)
    {
        if (_console.Lines > 2000)
        {
            _console.Text = string.Join("\n", ServerController.Instance.ConsoleSnapshot());
        }
        else
        {
            _console.Text += (_console.Text.Length > 0 ? "\n" : "") + line;
        }
        _console.MoveEnd();
        if (line.StartsWith("> ", StringComparison.Ordinal))
        {
            UpdateCommandLine();
        }
    }

    private async Task StartStopAsync()
    {
        if (ServerController.Instance.CanStart)
        {
            ToolLocator.Refresh();
            RefreshTools();
            _console.Text = "";
        }
        await _shell.ToggleServerAsync();
    }
}
