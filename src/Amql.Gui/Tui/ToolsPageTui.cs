// TextView is marked obsolete in favour of a separate editor package; it is still the text area Terminal.Gui ships.
#pragma warning disable CS0618
using System.IO;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Text;
using Amql.Gui.Commands;
using Amql.Gui.Model;
using Amql.Gui.Services;
using Amql.Hf;
using Terminal.Gui.ViewBase;
using Terminal.Gui.Views;

namespace Amql.Gui.Tui;

/// <summary>
/// Every amql-cli command as a form: the command tree on the left (with a
/// search box), the selected command's parameters as a list of settings, its
/// command line, and Run. Output, run history and projects work as on the
/// desktop Tools page, on the same .amqlproj files.
/// </summary>
internal sealed class ToolsPageTui : ITuiPage
{
    private const int MaxUiOutputLines = 20000;

    private readonly Shell _shell;
    private readonly View _root = new() { CanFocus = true };
    private readonly TextField _search = new() { X = 6, Y = 0, Width = 24 };
    private readonly TreeView<object> _tree = new() { X = 0, Y = 1, Width = 30, Height = Dim.Fill() };
    private readonly Label _title = new() { X = 32, Y = 0, Width = Dim.Fill(36), Height = 1 };
    private readonly Label _project = new() { X = Pos.AnchorEnd(35), Y = 0, Width = 22, Height = 1 };
    private readonly Label _summary = new() { X = 32, Y = 1, Width = Dim.Fill(), Height = 2 };
    private readonly Button _run, _cancel;
    private readonly ProgressBar _progress = new() { X = 58, Y = 3, Width = 24 };
    private readonly Label _state = new() { X = 84, Y = 3, Width = Dim.Fill(), Height = 1 };
    private readonly Tabs _tabs;
    private readonly View _formTab, _outputTab, _historyTab;
    private readonly FieldList _form = new(3) { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill(4) };
    private readonly TextView _preview = new() { X = 0, Y = Pos.AnchorEnd(3), Width = Dim.Fill(), Height = 3, ReadOnly = true, WordWrap = true };
    private readonly TextView _output = new() { X = 0, Y = 1, Width = Dim.Fill(), Height = Dim.Fill(), ReadOnly = true };
    private readonly ListView _history = new() { X = 0, Y = 1, Width = Dim.Fill(), Height = Dim.Fill() };
    private readonly ObservableCollection<string> _historyRows = new();

    private readonly Dictionary<string, List<CommandDef>> _filtered = new();
    private readonly List<string> _outputLines = new();
    private CommandDef? _current;
    private Dictionary<string, string> _values = new(StringComparer.OrdinalIgnoreCase);
    private string? _token;
    private string? _suggestedOut;
    private CancellationTokenSource? _runCts;
    private RunRecord? _activeRun;
    private object? _outputTimer;
    private double _ramValue = 10, _gpuValue = 8;
    private double _gpuMax = 24;

    public ToolsPageTui(Shell shell)
    {
        _shell = shell;
        Session.OpenLastOrDefault();

        _search.TextChanged += (_, _) => BuildTree(_search.Text.Trim());
        _tree.AspectGetter = o => o switch
        {
            CommandDef c => c.Name,
            string cat => cat.IndexOf(". ", StringComparison.Ordinal) is var dot and > 0 and < 4 ? cat[(dot + 2)..] : cat,
            _ => o.ToString() ?? "",
        };
        _tree.TreeBuilder = new DelegateTreeBuilder<object>(
            o => o is string cat && _filtered.TryGetValue(cat, out var list) ? list : Enumerable.Empty<object>(),
            o => o is string);
        _tree.SelectionChanged += (_, e) =>
        {
            if (e.NewValue is CommandDef cmd && cmd != _current)
            {
                SelectCommand(cmd);
            }
        };
        _tree.Accepting += (_, e) =>
        {
            e.Handled = true;
            if (_tree.SelectedObject is CommandDef)
            {
                _form.FocusList();
            }
            else if (_tree.SelectedObject is { } category)
            {
                _tree.Toggle(category);
            }
        };
        _root.Add(new Label { Text = "Find:", X = 0, Y = 0 });
        _root.Add(_search, _tree);
        _root.Add(new Line { X = 31, Y = 0, Height = Dim.Fill(), Orientation = Terminal.Gui.ViewBase.Orientation.Vertical });

        var projectButton = TuiKit.Button("Project ▾", ProjectMenu, Pos.AnchorEnd(13), 0);
        _run = TuiKit.Button("Run (F5)", RunSelected, 32, 3);
        _cancel = TuiKit.Button("Cancel", CancelRun, Pos.Right(_run) + 1, 3);
        _cancel.Enabled = false;
        _root.Add(_title, _project, projectButton, _summary, _run, _cancel, _progress, _state);

        _tabs = new Tabs { TabStop = TabBehavior.TabStop, X = 32, Y = 4, Width = Dim.Fill(), Height = Dim.Fill() };
        _formTab = new View { Title = "_Form", Width = Dim.Fill(), Height = Dim.Fill(), CanFocus = true };
        _formTab.Add(_form);
        _formTab.Add(new Label { Text = "Command line", X = 0, Y = Pos.AnchorEnd(4) });
        _formTab.Add(_preview);
        _form.Changed += _ => FormChanged();
        _form.LabelWidth = 32;

        _outputTab = new View { Title = "_Output", Width = Dim.Fill(), Height = Dim.Fill(), CanFocus = true };
        _outputTab.Add(TuiKit.Button("Clear", ClearOutput, 0, 0), TuiKit.Button("Copy", () => TuiKit.Copy(string.Join("\n", _outputLines)), 10, 0));
        _outputTab.Add(_output);

        _historyTab = new View { Title = "_Recent runs", Width = Dim.Fill(), Height = Dim.Fill(), CanFocus = true };
        _history.SetSource(_historyRows);
        _history.Accepting += (_, e) =>
        {
            e.Handled = true;
            ShowHistoryRun();
        };
        _historyTab.Add(new Label { Text = "Enter shows a run's command line, status and output tail.", X = 0, Y = 0 });
        _historyTab.Add(TuiKit.Button("Clear history", ClearHistory, Pos.AnchorEnd(18), 0));
        _historyTab.Add(_history);

        _tabs.Add(_formTab, _outputTab, _historyTab);
        _root.Add(_tabs);

        TuiKit.GuardAsync(InitGpuAsync);
        ReloadProject();
    }

    public ProjectSession Session { get; } = new();

    public View View => _root;

    public void Shown()
    {
    }

    public void Focus() => _tree.SetFocus();

    public void Hidden() => SaveFormValues();

    // ── command tree ────────────────────────────────────────────────────────

    private void BuildTree(string filter = "")
    {
        _filtered.Clear();
        _tree.ClearObjects();
        foreach (var group in CommandCatalog.ByCategory())
        {
            var matches = group.Where(c => filter.Length == 0
                || c.Name.Contains(filter, StringComparison.OrdinalIgnoreCase)
                || c.Summary.Contains(filter, StringComparison.OrdinalIgnoreCase)).ToList();
            if (matches.Count == 0)
            {
                continue;
            }
            _filtered[group.Key] = matches;
            _tree.AddObject(group.Key);
            if (filter.Length > 0 || group.Key.StartsWith('3'))
            {
                _tree.Expand(group.Key);
            }
        }
    }

    public void SelectCommandByName(string name)
    {
        if (CommandCatalog.Find(name) is not { } cmd)
        {
            return;
        }
        if (_search.Text.Length > 0)
        {
            _search.Text = "";
        }
        var category = _filtered.FirstOrDefault(kv => kv.Value.Contains(cmd)).Key;
        if (category is not null)
        {
            _tree.Expand(category);
            _tree.SelectedObject = cmd;
            _tree.EnsureVisible(cmd);
        }
        SelectCommand(cmd);
    }

    private void SelectCommand(CommandDef cmd)
    {
        SaveFormValues();
        _current = cmd;
        Session.Project.LastCommand = cmd.Name;
        _title.Text = cmd.Name;
        _summary.Text = cmd.Summary;
        var state = Session.Project.Command(cmd.Name);
        _values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var p in cmd.Params)
        {
            _values[p.Key] = CommandForms.InitialValue(p, state);
            if (CommandForms.DefaultFor(Session.Project.Defaults, p.DefaultFrom) is { Length: > 0 } value
                && (!state.Values.TryGetValue(p.Key, out var saved) || string.IsNullOrEmpty(saved)))
            {
                _values[p.Key] = value;
                state.Values[p.Key] = value;
            }
        }
        _token = null;
        _suggestedOut = null;
        BuildForm(cmd);
        UpdatePreview();
    }

    // ── the form ────────────────────────────────────────────────────────────

    private string Get(string key) => _values.TryGetValue(key, out var v) ? v : "";

    private void Set(string key, string value) => _values[key] = value;

    private void BuildForm(CommandDef cmd)
    {
        var (basic, advanced) = CommandForms.Split(cmd);
        var fields = new List<Field>();
        if (cmd.Params.Count == 0)
        {
            fields.Add(new InfoField("This command takes no parameters.", () => ""));
        }
        foreach (var p in basic)
        {
            fields.AddRange(FieldsFor(p));
        }
        if (advanced.Count > 0)
        {
            fields.Add(new HeaderField($"Advanced options ({advanced.Count})"));
            foreach (var p in advanced)
            {
                fields.AddRange(FieldsFor(p));
            }
        }
        fields.AddRange(RunSettingsFields());
        _form.SetFields(fields);
    }

    private static string HelpOf(ParamDef p) =>
        (p.Kind == ParamKind.Positional ? p.Label : p.Flag ?? p.Label) + (p.Required ? " (required)" : "") + (p.Help is { Length: > 0 } h ? " — " + h : "");

    private IEnumerable<Field> FieldsFor(ParamDef p)
    {
        string label = CommandForms.Friendly(p) + (p.Required ? " *" : "");
        string help = HelpOf(p);
        if (p.Kind == ParamKind.Switch)
        {
            yield return new BoolField(label, () => CommandForms.IsOn(Get(p.Key)), v => Set(p.Key, v ? "true" : ""), help);
            yield break;
        }
        switch (p.Editor)
        {
            case EditorKind.ModelSource:
                foreach (var f in ModelSourceFields(p, label, help))
                {
                    yield return f;
                }
                yield break;
            case EditorKind.Choice:
                var choices = (p.Choices ?? Array.Empty<string>()).Select(c => (c.Length == 0 ? "(default)" : c, c)).ToArray();
                yield return new ChoiceField<string>(label, choices,
                    () => choices.Any(c => c.Item2 == Get(p.Key)) ? Get(p.Key) : choices.FirstOrDefault().Item2 ?? "",
                    v => Set(p.Key, v), help);
                yield break;
            case EditorKind.Directory:
                yield return new PathField(label, directory: true, () => Get(p.Key), v => Set(p.Key, v), help);
                yield break;
            case EditorKind.File:
                yield return new PathField(label, directory: false, () => Get(p.Key), v => Set(p.Key, v), help);
                yield break;
            case EditorKind.MultiLineText:
                yield return new TextValueField(label, () => Get(p.Key), v => Set(p.Key, v), help) { Multiline = true };
                yield break;
            case EditorKind.Int:
                yield return new TextValueField(label, () => Get(p.Key), v => Set(p.Key, v), help) { Validate = TextValueField.OptionalInt };
                yield break;
            case EditorKind.Float:
                yield return new TextValueField(label, () => Get(p.Key), v => Set(p.Key, v), help) { Validate = TextValueField.OptionalNumber };
                yield break;
            default:
                yield return new TextValueField(label, () => Get(p.Key), v => Set(p.Key, v), help);
                yield break;
        }
    }

    /// <summary>
    /// Where a checkpoint comes from: a folder on this computer, or a Hugging Face
    /// model (a URL or owner/name) that the CLI downloads before encoding. A token
    /// for gated or private models is passed to the CLI as HF_TOKEN and never saved.
    /// </summary>
    private IEnumerable<Field> ModelSourceFields(ParamDef p, string label, string help)
    {
        string initial = Get(p.Key);
        bool hub = initial.Length > 0 && !Directory.Exists(initial) && HfDownloader.TryParse(initial) is not null;
        string folder = hub ? "" : initial, url = hub ? initial : "";
        void Apply()
        {
            Set(p.Key, (hub ? url : folder).Trim());
            if (hub && HfDownloader.TryParse(url.Trim()) is { } model && (Get("out").Length == 0 || Get("out") == _suggestedOut)
                && _current?.Params.Any(x => x.Key == "out") == true)
            {
                _suggestedOut = Path.Combine(CommandForms.ContainersFolder(), model.Name);
                Set("out", _suggestedOut);
            }
        }
        yield return new ChoiceField<bool>(label, new[] { ("Folder on this computer", false), ("Hugging Face", true) },
            () => hub, v => { hub = v; Apply(); }, help);
        yield return new PathField("  Checkpoint folder", directory: true, () => folder, v => { folder = v; Apply(); },
            "A checkpoint folder: config.json, *.safetensors and the tokenizer files.") { IsVisible = () => !hub };
        yield return new TextValueField("  Hugging Face model", () => url, v => { url = v; Apply(); })
        {
            IsVisible = () => hub,
            HelpText = () => url.Trim().Length == 0
                ? "Paste the model's page URL (https://huggingface.co/Qwen/Qwen3.5-0.8B), or type owner/name; add @revision for a branch or commit."
                : HfDownloader.TryParse(url.Trim()) is { } model
                    ? $"Downloads {model} (safetensors, config and tokenizer files), then encodes it. It goes to {HfDownloader.DefaultDirectory(model)} unless a download folder is set."
                    : "That is not a Hugging Face model URL or owner/name.",
            Validate = s => s.Length == 0 || HfDownloader.TryParse(s) is not null ? null : "That is not a Hugging Face model URL or owner/name.",
        };
        yield return new TextValueField("  Access token", () => _token ?? "", v => _token = v.Trim().Length > 0 ? v.Trim() : null,
            HfDownloader.ResolveToken() is null
                ? "Only needed for gated or private models. It is passed to amql-cli for this run and not saved."
                : "Only needed for gated or private models. A saved Hugging Face login was found and is used when this is empty.")
        { IsVisible = () => hub, Secret = true, Placeholder = "none" };
    }

    private IEnumerable<Field> RunSettingsFields()
    {
        CliSettings C() => Session.Cli;
        string Num(double v) => v.ToString("0.#", CultureInfo.InvariantCulture);
        double MaxRam() => Math.Max(4, Math.Floor(SystemInfo.TotalRamBytes / (double)(1L << 30)));
        yield return new HeaderField("Run settings (device, memory, GPU format)", "The CLI's global options, saved in the project.");
        yield return new ChoiceField<string>("Device", new[] { ("Automatic", "auto"), ("GPU (fail if none)", "gpu"), ("CPU only", "cpu") },
            () => C().Device is { Length: > 0 } d ? d : "auto", v => C().Device = v, "--gpu / --cpu, or neither.");
        yield return new BoolField("Limit memory use", () => C().MaxRamGb is not null, v => C().MaxRamGb = v ? _ramValue : null,
            "--max-ram: the lean working set's process target.");
        yield return new TextValueField("  Memory limit (GB)", () => Num(C().MaxRamGb ?? _ramValue), v =>
        {
            _ramValue = Math.Round(double.Parse(v, CultureInfo.InvariantCulture));
            C().MaxRamGb = _ramValue;
        }, "--max-ram in GB.")
        {
            IsEnabled = () => C().MaxRamGb is not null,
            Validate = s => double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var g) && g >= 2 && g <= MaxRam() ? null : $"Enter 2 to {MaxRam():0} GB.",
        };
        yield return new BoolField("Limit GPU memory", () => C().GpuMemoryGb is not null, v => C().GpuMemoryGb = v ? _gpuValue : null,
            "--gpu-memory: how much of the GPU the weights may take.");
        yield return new TextValueField("  GPU memory limit (GB)", () => Num(C().GpuMemoryGb ?? _gpuValue), v =>
        {
            _gpuValue = Math.Round(double.Parse(v, CultureInfo.InvariantCulture) * 2) / 2;
            C().GpuMemoryGb = _gpuValue;
        }, "--gpu-memory in GB (0 keeps every weight on the host).")
        {
            IsEnabled = () => C().GpuMemoryGb is not null,
            Validate = s => double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var g) && g >= 0 && g <= _gpuMax ? null : $"Enter 0 to {_gpuMax:0} GB.",
        };
        yield return new ChoiceField<string>("GPU weight format", new[] { ("As stored", ""), ("FP16", "fp16"), ("NVFP4 (4-bit)", "nvfp4"), ("MXFP4 (4-bit)", "mxfp4") },
            () => C().GpuFormat, v => C().GpuFormat = v, "--gpu-format.");
        yield return new ChoiceField<string>("FP4 prefill", new[] { ("Default", ""), ("On", "on"), ("Off", "off") },
            () => C().Fp4Prefill, v => C().Fp4Prefill = v, "--fp4-prefill.");
        yield return new BoolField("Verbose output", () => C().Verbose, v => C().Verbose = v, "--verbose.");
        yield return new ChoiceField<string>("AMQL_WEIGHTS", new[] { ("(default)", ""), ("bf16", "bf16"), ("mxfp4", "mxfp4"), ("disk", "disk"), ("lean", "lean") },
            () => C().WeightsEnv, v => C().WeightsEnv = v, "The AMQL_WEIGHTS environment variable for the run.");
    }

    private async Task InitGpuAsync()
    {
        if (await SystemInfo.SampleGpuAsync() is { } gpu)
        {
            _gpuMax = Math.Max(1, Math.Floor(gpu.TotalBytes / (double)(1L << 30)));
        }
    }

    private void FormChanged()
    {
        SaveFormValues();
        UpdatePreview();
    }

    private void SaveFormValues()
    {
        if (_current is null)
        {
            return;
        }
        var state = Session.Project.Command(_current.Name);
        foreach (var (key, value) in _values)
        {
            state.Values[key] = value;
        }
    }

    private Dictionary<string, string> CurrentValues() =>
        _values.ToDictionary(kv => kv.Key, kv => kv.Value.Trim(), StringComparer.OrdinalIgnoreCase);

    private void UpdatePreview() =>
        _preview.Text = _current is null ? "" : CommandForms.CommandLine(Session.Cli, _current, CurrentValues());

    private void ApplyDefaults()
    {
        if (_current is null)
        {
            return;
        }
        foreach (var p in _current.Params)
        {
            if (CommandForms.DefaultFor(Session.Project.Defaults, p.DefaultFrom) is { Length: > 0 } value)
            {
                Set(p.Key, value);
            }
        }
        _form.Refresh();
        FormChanged();
    }

    // ── running ─────────────────────────────────────────────────────────────

    /// <summary>F5: runs the selected command (when this page is showing).</summary>
    public void RunSelected()
    {
        if (_current is null || _runCts is not null)
        {
            return;
        }
        var cmd = _current;
        var values = CurrentValues();
        if (CommandForms.MissingRequired(cmd, values) is { Count: > 0 } missing)
        {
            TuiKit.Info(cmd.Name, "Fill in: " + string.Join(", ", missing));
            return;
        }
        SaveFormValues();
        var environment = new Dictionary<string, string>();
        if (_token is { } token)
        {
            environment["HF_TOKEN"] = token;
        }
        ClearOutput();
        _tabs.Value = _outputTab;
        _run.Enabled = false;
        _cancel.Enabled = true;
        _progress.Fraction = 0;
        _state.Text = "running…";
        _runCts = new CancellationTokenSource();
        TuiKit.GuardAsync(async () =>
        {
            try
            {
                var record = await Session.RunAsync(cmd, values, environment,
                    onStarted: r =>
                    {
                        _activeRun = r;
                        RefreshHistory();
                    },
                    onLine: AppendOutput,
                    onProgress: pct =>
                    {
                        _progress.Fraction = (float)(pct / 100);
                        _state.Text = $"{pct:0.#}%";
                    },
                    _runCts.Token);
                if (record.Status is "Succeeded" or "Negative")
                {
                    _progress.Fraction = 1;
                }
                _state.Text = CommandForms.RunSummary(record);
                TuiKit.Say($"{cmd.Name}: {_state.Text}");
            }
            finally
            {
                _runCts?.Dispose();
                _runCts = null;
                _activeRun = null;
                _run.Enabled = true;
                _cancel.Enabled = false;
                FlushOutput();
                RefreshHistory();
                RefreshProjectText();
            }
        });
    }

    private void CancelRun()
    {
        if (_runCts is { } cts)
        {
            try
            {
                cts.Cancel();
            }
            catch (ObjectDisposedException)
            {
                // finished in the meantime
            }
            _state.Text = "cancelling…";
        }
    }

    private void AppendOutput(string line)
    {
        if (_outputLines.Count > MaxUiOutputLines)
        {
            _outputLines.Clear();
            _outputLines.Add("[studio] output truncated (too many lines)");
        }
        _outputLines.Add(line);
        _outputTimer ??= TuiKit.App.AddTimeout(TimeSpan.FromMilliseconds(150), () =>
        {
            _outputTimer = null;
            FlushOutput();
            return false;
        });
    }

    private void FlushOutput() => TuiKit.SetLog(_output, string.Join("\n", _outputLines));

    private void ClearOutput()
    {
        _outputLines.Clear();
        _output.Text = "";
    }

    // ── history ─────────────────────────────────────────────────────────────

    private void RefreshHistory()
    {
        int? keep = _history.SelectedItem;
        _historyRows.Clear();
        foreach (var r in Session.History)
        {
            _historyRows.Add(r.Display);
        }
        if (_historyRows.Count > 0)
        {
            _history.SelectedItem = Math.Min(keep ?? 0, _historyRows.Count - 1);
        }
    }

    private void ShowHistoryRun()
    {
        if (_history.SelectedItem is not { } i || i < 0 || i >= Session.History.Count)
        {
            return;
        }
        var record = Session.History[i];
        if (record == _activeRun)
        {
            _tabs.Value = _outputTab;
            return;
        }
        _outputLines.Clear();
        _outputLines.AddRange(CommandForms.DescribeRun(record).TrimEnd().Split('\n'));
        _output.Text = string.Join("\n", _outputLines);
        _tabs.Value = _outputTab;
    }

    private void ClearHistory()
    {
        if (_runCts is not null)
        {
            return;
        }
        Session.ClearHistory();
        RefreshHistory();
    }

    // ── projects ────────────────────────────────────────────────────────────

    private void ProjectMenu()
    {
        var items = new[]
        {
            "New project…", "Open project…", "Save project as…", "Project defaults…",
            "Fill the form from project defaults", "Find amql-cli again",
        };
        switch (TuiKit.Choose($"Project — {Session.Path}", items))
        {
            case 0: NewProject(); break;
            case 1: OpenProject(); break;
            case 2: SaveProjectAs(); break;
            case 3: EditDefaults(); break;
            case 4: ApplyDefaults(); break;
            case 5:
                ToolLocator.Refresh();
                var found = ToolLocator.Find(ToolLocator.Cli);
                TuiKit.Info("amql-cli", found is null ? "amql-cli was not found. Build src/Amql.Cli or put amql-cli on PATH." : "Using " + found.Describe());
                break;
        }
    }

    private static string ProjectsFolder() =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "AMQL Projects");

    private void NewProject()
    {
        if (!ConfirmClose())
        {
            return;
        }
        string name = "My AMQL Project";
        string path = Path.Combine(ProjectsFolder(), name + ProjectModel.FileExtension);
        string container = "", tokenizer = "";
        bool pathTouched = false;
        var fields = new Field[]
        {
            new TextValueField("Name", () => name, v =>
            {
                name = v;
                if (!pathTouched)
                {
                    path = Path.Combine(Path.GetDirectoryName(path) ?? ProjectsFolder(), CommandForms.SanitizeFileName(v) + ProjectModel.FileExtension);
                }
            }) { Validate = s => s.Length == 0 ? "Give the project a name." : null },
            new PathField("File", directory: false, () => path, v => { path = v; pathTouched = true; }, "Where to save the .amqlproj file."),
            new PathField("Container", directory: true, () => container, v => container = v, "Project default for commands that take a container (optional).") { MustExist = true },
            new PathField("Tokenizer / checkpoint", directory: true, () => tokenizer, v => tokenizer = v, "Project default for --tokenizer (optional).") { MustExist = true },
        };
        if (!RunFieldDialog("New AMQL Project", fields, "_Create"))
        {
            return;
        }
        if (Session.CreateNew(name.Trim(), path, container, tokenizer) is { } error)
        {
            TuiKit.Error("New AMQL Project", error);
            return;
        }
        ClearOutput();
        ReloadProject();
    }

    private void OpenProject()
    {
        if (_runCts is not null || !ConfirmClose())
        {
            return;
        }
        if (TuiKit.PickPath("Open AMQL Project", directory: false, ProjectsFolder()) is not { } path)
        {
            return;
        }
        if (!Session.TryOpen(path, out var error))
        {
            TuiKit.Error("AMQL Studio", $"Cannot open the project:\n{error}");
            return;
        }
        ReloadProject();
    }

    private void SaveProjectAs()
    {
        string initial = Path.Combine(ProjectsFolder(), CommandForms.SanitizeFileName(Session.Project.Name) + ProjectModel.FileExtension);
        if (TuiKit.PickSavePath("Save AMQL Project", initial) is not { } path)
        {
            return;
        }
        if (!path.EndsWith(ProjectModel.FileExtension, StringComparison.OrdinalIgnoreCase))
        {
            path += ProjectModel.FileExtension;
        }
        SaveFormValues();
        if (Session.SaveAs(path) is { } error)
        {
            TuiKit.Error("Save AMQL Project", error);
        }
        RefreshProjectText();
    }

    private void EditDefaults()
    {
        var d = Session.Project.Defaults;
        string container = d.ContainerDir, tokenizer = d.TokenizerDir, patch = d.PatchFile, component = d.Component;
        var fields = new Field[]
        {
            new PathField("Container", directory: true, () => container, v => container = v, "Stamped into commands that take a container."),
            new PathField("Tokenizer / checkpoint", directory: true, () => tokenizer, v => tokenizer = v, "Stamped into --tokenizer."),
            new PathField("Weight patch", directory: false, () => patch, v => patch = v, "A .safetensors patch, stamped into --patch."),
            new TextValueField("Component", () => component, v => component = v, "System-graph component id (target by default)."),
        };
        if (!RunFieldDialog("Project defaults", fields, "_Save"))
        {
            return;
        }
        d.ContainerDir = container.Trim();
        d.TokenizerDir = tokenizer.Trim();
        d.PatchFile = patch.Trim();
        d.Component = string.IsNullOrWhiteSpace(component) ? "target" : component.Trim();
        Session.Save();
    }

    /// <summary>A dialog of fields with Cancel and an accept button; true when accepted.</summary>
    private static bool RunFieldDialog(string title, IEnumerable<Field> fields, string accept)
    {
        var dialog = new Dialog { Title = title, Width = Dim.Percent(80), Height = 14 };
        var list = new FieldList(2);
        list.SetFields(fields);
        dialog.Add(list);
        dialog.AddButton(new Button { ShadowStyle = ShadowStyles.None, Text = "_Cancel" });
        dialog.AddButton(new Button { ShadowStyle = ShadowStyles.None, Text = accept });
        list.FocusList();
        TuiKit.App.Run(dialog);
        bool ok = dialog.Result == 1;
        dialog.Dispose();
        return ok;
    }

    private void ReloadProject()
    {
        RefreshHistory();
        BuildTree();
        var target = (Session.Project.LastCommand is { } name ? CommandCatalog.Find(name) : null)
                     ?? CommandCatalog.Find("generate") ?? CommandCatalog.All[0];
        _current = null;
        SelectCommandByName(target.Name);
        RefreshProjectText();
    }

    private void RefreshProjectText() => _project.Text = TuiKit.Shorten(Session.DisplayName, 22);

    /// <summary>Asks to stop a running command before closing or switching projects; saves.</summary>
    public bool ConfirmClose()
    {
        if (_runCts is not null)
        {
            if (!TuiKit.Confirm("AMQL Studio", "A command is still running. Stop it?"))
            {
                return false;
            }
            _runCts.Cancel();
        }
        SaveFormValues();
        Session.Save();
        return true;
    }
}
