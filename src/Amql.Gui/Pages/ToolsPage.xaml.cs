using System.Collections.ObjectModel;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Amql.Gui.Commands;
using Amql.Gui.Dialogs;
using Amql.Gui.Model;
using Amql.Gui.Run;
using Amql.Gui.Services;
using Microsoft.Win32;

namespace Amql.Gui.Pages;

/// <summary>
/// Every amql-cli command as a form. Required parameters come first; the
/// rest sit under "Advanced options". Form values and the run history live
/// in a project file, which is opened automatically (the last one used, or a
/// default project in the app's data folder) and saved after every run, so
/// there is nothing to set up before the first run.
/// </summary>
public partial class ToolsPage : UserControl
{
    private const int MaxTailLines = 400;
    private const int MaxHistoryRuns = 200;
    private const int MaxUiOutputLines = 20000;

    private ProjectModel _project = new();
    private string _projectPath = DefaultProjectPath;
    private bool _loadingGlobals;

    private CommandDef? _currentCommand;
    private readonly Dictionary<string, FrameworkElement> _editors = new(StringComparer.OrdinalIgnoreCase);

    private CancellationTokenSource? _runCts;
    private RunRecord? _activeRun;
    private int _nextRunId = 1;
    private readonly ObservableCollection<RunRecord> _history = new();

    private static string DefaultProjectPath => Path.Combine(AppSettings.Folder, "Default" + ProjectModel.FileExtension);

    public ToolsPage()
    {
        InitializeComponent();
        WheelScrolling.Attach(FormScroll);
        RamSlider.Maximum = Math.Max(4, Math.Floor(SystemInfo.TotalRamBytes / (double)(1L << 30)));
        HistoryList.ItemsSource = _history;
        _ = InitGpuSliderAsync();

        var last = AppSettings.Current.LastProject;
        if (!TryLoadProject(last is not null && File.Exists(last) ? last : DefaultProjectPath, quiet: true))
        {
            _project = new ProjectModel { Name = "Default" };
            _projectPath = DefaultProjectPath;
            ReloadProjectIntoUi();
        }
        PreviewKeyDown += (_, e) =>
        {
            if (e.Key == Key.F5 && _runCts is null)
            {
                e.Handled = true;
                OnRun(this, new RoutedEventArgs());
            }
        };
    }

    /// <summary>The project's CLI settings, for windows that run the CLI themselves.</summary>
    public CliSettings Cli => _project.Cli;

    private async Task InitGpuSliderAsync()
    {
        if (await SystemInfo.SampleGpuAsync() is { } gpu)
        {
            GpuSlider.Maximum = Math.Max(1, Math.Floor(gpu.TotalBytes / (double)(1L << 30)));
        }
    }

    // ── command tree ────────────────────────────────────────────────────────

    private void BuildCommandTree(string filter = "")
    {
        CommandTree.Items.Clear();
        foreach (var group in CommandCatalog.ByCategory())
        {
            var matches = group.Where(c => filter.Length == 0
                || c.Name.Contains(filter, StringComparison.OrdinalIgnoreCase)
                || c.Summary.Contains(filter, StringComparison.OrdinalIgnoreCase)).ToList();
            if (matches.Count == 0)
            {
                continue;
            }
            // "3a. Inference & Benchmarks" → "Inference & Benchmarks"
            string title = group.Key.IndexOf(". ", StringComparison.Ordinal) is var dot and > 0 and < 4 ? group.Key[(dot + 2)..] : group.Key;
            var category = new TreeViewItem { Header = title, IsExpanded = filter.Length > 0 || group.Key.StartsWith('3'), FontWeight = FontWeights.SemiBold };
            foreach (var cmd in matches)
            {
                category.Items.Add(new TreeViewItem { Header = cmd.Name, Tag = cmd, ToolTip = cmd.Summary, FontWeight = FontWeights.Normal });
            }
            CommandTree.Items.Add(category);
        }
    }

    private void OnSearchChanged(object sender, TextChangedEventArgs e) => BuildCommandTree(SearchBox.Text.Trim());

    private void OnCommandSelected(object sender, RoutedPropertyChangedEventArgs<object> e)
    {
        if (e.NewValue is TreeViewItem { Tag: CommandDef cmd })
        {
            SelectCommand(cmd);
        }
    }

    /// <summary>Shows one command's form (used by Home's shortcuts).</summary>
    public void SelectCommandByName(string name)
    {
        if (CommandCatalog.Find(name) is not { } cmd)
        {
            return;
        }
        SearchBox.Text = string.Empty;
        var item = FindTreeItem(CommandTree, name);
        if (item is not null)
        {
            if (item.Parent is TreeViewItem parent)
            {
                parent.IsExpanded = true;
            }
            item.IsSelected = true;
            item.BringIntoView();
        }
        else
        {
            SelectCommand(cmd);
        }
    }

    private void SelectCommand(CommandDef cmd)
    {
        SaveCurrentFormValues();
        _currentCommand = cmd;
        _project.LastCommand = cmd.Name;
        CommandTitle.Text = cmd.Name;
        CommandSummary.Text = cmd.Summary;
        BuildParamForm(cmd);
        AutoApplyDefaults(cmd);
        UpdateCommandLinePreview();
    }

    private string? DefaultFor(string? key)
    {
        var d = _project.Defaults;
        return key switch
        {
            "container" => d.ContainerDir,
            "tokenizer" => d.TokenizerDir,
            "patch" => d.PatchFile,
            "component" => d.Component,
            _ => null,
        };
    }

    /// <summary>Fills empty fields from the project defaults when a command is first shown.</summary>
    private void AutoApplyDefaults(CommandDef cmd)
    {
        var state = _project.Command(cmd.Name);
        foreach (var p in cmd.Params)
        {
            if (DefaultFor(p.DefaultFrom) is { Length: > 0 } value && _editors.TryGetValue(p.Key, out var editor)
                && (!state.Values.TryGetValue(p.Key, out var saved) || string.IsNullOrEmpty(saved)))
            {
                SetEditorValue(editor, value);
                state.Values[p.Key] = value;
            }
        }
    }

    // ── dynamic parameter form ──────────────────────────────────────────────

    private static bool IsBasic(ParamDef p) => p.Required || (p.Kind == ParamKind.Positional && p.DefaultFrom is not null);

    private void BuildParamForm(CommandDef cmd)
    {
        ParamPanel.Children.Clear();
        AdvancedPanel.Children.Clear();
        _editors.Clear();
        var state = _project.Command(cmd.Name);

        var basic = cmd.Params.Where(IsBasic).ToList();
        var advanced = cmd.Params.Where(p => !IsBasic(p)).ToList();
        if (basic.Count == 0)
        {
            // Nothing is required: show everything up front.
            basic = advanced;
            advanced = new List<ParamDef>();
        }
        foreach (var p in basic)
        {
            AddParam(ParamPanel, p, state);
        }
        foreach (var p in advanced)
        {
            AddParam(AdvancedPanel, p, state);
        }
        AdvancedExpander.Visibility = advanced.Count > 0 ? Visibility.Visible : Visibility.Collapsed;
        AdvancedExpander.Header = $"Advanced options ({advanced.Count})";
        // Open it when something in it has been set, so a set value is never hidden.
        AdvancedExpander.IsExpanded = advanced.Any(p => state.Values.TryGetValue(p.Key, out var v) && v.Length > 0 && v != p.DefaultValue);
        if (cmd.Params.Count == 0)
        {
            ParamPanel.Children.Add(new TextBlock { Text = "This command takes no parameters.", Style = (Style)FindResource("Caption") });
        }
    }

    private void AddParam(Panel panel, ParamDef p, CommandState state)
    {
        var grid = new Grid { Margin = new Thickness(0, 2, 0, 0) };
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(180) });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });

        var label = new TextBlock
        {
            Text = Friendly(p) + (p.Required ? " *" : ""),
            Style = (Style)FindResource("FieldLabel"),
            ToolTip = p.Kind == ParamKind.Positional ? p.Label : p.Flag,
            TextWrapping = TextWrapping.Wrap,
        };
        grid.Children.Add(label);

        FrameworkElement editor = p.Kind == ParamKind.Switch
            ? BuildSwitch(p, state)
            : p.Editor == EditorKind.ModelSource
            ? BuildModelSource(p, state)
            : p.Editor switch
            {
                EditorKind.Choice => BuildChoice(p, state),
                EditorKind.MultiLineText => BuildMultiLine(p, state),
                _ => BuildSingleLine(p, state),
            };
        Grid.SetColumn(editor, 1);
        grid.Children.Add(editor);

        if (BuildBrowseButton(p, editor) is { } browse)
        {
            Grid.SetColumn(browse, 2);
            grid.Children.Add(browse);
        }
        panel.Children.Add(grid);
        if (p.Help is { Length: > 0 })
        {
            panel.Children.Add(new TextBlock { Text = p.Help, Style = (Style)FindResource("HelpText"), Margin = new Thickness(180, 2, 4, 6) });
        }
        _editors[p.Key] = editor;
    }

    /// <summary>"--max-tokens" → "Max tokens"; positionals keep their label without the angle brackets.</summary>
    private static string Friendly(ParamDef p)
    {
        string text = p.Label;
        if (text.StartsWith("--", StringComparison.Ordinal))
        {
            text = text[2..];
        }
        text = text.Replace('-', ' ').Trim('<', '>');
        return text.Length == 0 ? p.Key : char.ToUpperInvariant(text[0]) + text[1..];
    }

    private static string InitialValue(ParamDef p, CommandState state) =>
        state.Values.TryGetValue(p.Key, out var saved) ? saved : p.DefaultValue;

    private FrameworkElement BuildSingleLine(ParamDef p, CommandState state)
    {
        var box = new TextBox { Text = InitialValue(p, state), Margin = new Thickness(0, 2, 0, 2) };
        box.TextChanged += (_, _) => UpdateCommandLinePreview();
        return box;
    }

    private FrameworkElement BuildMultiLine(ParamDef p, CommandState state)
    {
        var box = new TextBox
        {
            Text = InitialValue(p, state),
            AcceptsReturn = true,
            TextWrapping = TextWrapping.Wrap,
            MinHeight = 60,
            MaxHeight = 160,
            VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
            VerticalContentAlignment = VerticalAlignment.Top,
            Margin = new Thickness(0, 2, 0, 2),
        };
        box.TextChanged += (_, _) => UpdateCommandLinePreview();
        return box;
    }

    private FrameworkElement BuildChoice(ParamDef p, CommandState state)
    {
        var combo = new ComboBox { Margin = new Thickness(0, 2, 0, 2), MinWidth = 160, HorizontalAlignment = HorizontalAlignment.Left };
        foreach (var choice in p.Choices ?? Array.Empty<string>())
        {
            combo.Items.Add(choice);
        }
        var initial = InitialValue(p, state);
        combo.SelectedItem = combo.Items.Contains(initial) ? initial : (combo.Items.Count > 0 ? combo.Items[0] : null);
        combo.SelectionChanged += (_, _) => UpdateCommandLinePreview();
        return combo;
    }

    private string? _suggestedOut;

    /// <summary>A checkpoint from a local folder or the Hugging Face Hub. Naming a Hub model fills an
    /// empty output folder with one under the model folders, so the container shows on the Server page.</summary>
    private FrameworkElement BuildModelSource(ParamDef p, CommandState state)
    {
        var editor = new ModelSourceEditor(InitialValue(p, state), (Style)FindResource("Caption"), (Style)FindResource("Icon"))
        {
            Margin = new Thickness(0, 2, 0, 2),
        };
        editor.ValueChanged += () =>
        {
            if (editor.Model is { } model && _editors.TryGetValue("out", out var o) && o is TextBox outBox &&
                (outBox.Text.Length == 0 || outBox.Text == _suggestedOut))
            {
                _suggestedOut = Path.Combine(ContainersFolder(), model.Name);
                outBox.Text = _suggestedOut;
            }
            UpdateCommandLinePreview();
        };
        return editor;
    }

    /// <summary>Where new containers go by default: the repository's containers folder, else the first model folder.</summary>
    private static string ContainersFolder() =>
        ToolLocator.RepoRoot is { } repo ? Path.Combine(repo, "containers")
        : AppSettings.Current.ModelFolders.FirstOrDefault()
          ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "AMQL", "containers");

    private FrameworkElement BuildSwitch(ParamDef p, CommandState state)
    {
        var check = new CheckBox
        {
            IsChecked = InitialValue(p, state) is "true" or "1" or "on",
            Content = p.Flag,
            Margin = new Thickness(0, 2, 0, 2),
        };
        check.Checked += (_, _) => UpdateCommandLinePreview();
        check.Unchecked += (_, _) => UpdateCommandLinePreview();
        return check;
    }

    private Button? BuildBrowseButton(ParamDef p, FrameworkElement editor)
    {
        if (p.Editor is not (EditorKind.Directory or EditorKind.File) || editor is not TextBox box)
        {
            return null;
        }
        bool isDir = p.Editor == EditorKind.Directory;
        var button = new Button
        {
            Content = new TextBlock { Text = "", Style = (Style)FindResource("Icon") },
            Margin = new Thickness(6, 2, 0, 2),
            ToolTip = isDir ? "Choose a folder" : "Choose a file",
        };
        button.Click += (_, _) =>
        {
            var owner = Window.GetWindow(this);
            if (isDir)
            {
                var dialog = new OpenFolderDialog { Title = Friendly(p) };
                if (TryGetExistingDirectory(box.Text, out var start)) dialog.InitialDirectory = start;
                if (dialog.ShowDialog(owner) == true) box.Text = dialog.FolderName;
            }
            else
            {
                var dialog = new OpenFileDialog { Title = Friendly(p), CheckFileExists = false };
                if (TryGetExistingDirectory(Path.GetDirectoryName(box.Text), out var start)) dialog.InitialDirectory = start;
                if (dialog.ShowDialog(owner) == true) box.Text = dialog.FileName;
            }
        };
        return button;
    }

    private static bool TryGetExistingDirectory(string? path, out string dir)
    {
        dir = string.Empty;
        try
        {
            if (!string.IsNullOrWhiteSpace(path) && Directory.Exists(path))
            {
                dir = Path.GetFullPath(path);
                return true;
            }
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException or PathTooLongException)
        {
            // invalid path text — just don't preselect
        }
        return false;
    }

    // ── form ⇄ project model ───────────────────────────────────────────────

    private static string EditorValue(FrameworkElement editor) => editor switch
    {
        TextBox tb => tb.Text,
        ComboBox cb => cb.SelectedItem as string ?? string.Empty,
        CheckBox ch => ch.IsChecked == true ? "true" : string.Empty,
        ModelSourceEditor m => m.Value,
        _ => string.Empty,
    };

    private static void SetEditorValue(FrameworkElement editor, string value)
    {
        switch (editor)
        {
            case TextBox tb: tb.Text = value; break;
            case ComboBox cb when cb.Items.Contains(value): cb.SelectedItem = value; break;
            case CheckBox ch: ch.IsChecked = value is "true" or "1" or "on"; break;
            case ModelSourceEditor m: m.Value = value; break;
        }
    }

    private void SaveCurrentFormValues()
    {
        if (_currentCommand is null) return;
        var state = _project.Command(_currentCommand.Name);
        foreach (var (key, editor) in _editors)
        {
            state.Values[key] = EditorValue(editor);
        }
    }

    private Dictionary<string, string> CurrentFormValues()
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (key, editor) in _editors)
        {
            values[key] = EditorValue(editor).Trim();
        }
        return values;
    }

    private void OnApplyDefaults(object sender, RoutedEventArgs e)
    {
        if (_currentCommand is null) return;
        foreach (var p in _currentCommand.Params)
        {
            if (DefaultFor(p.DefaultFrom) is { Length: > 0 } value && _editors.TryGetValue(p.Key, out var editor))
            {
                SetEditorValue(editor, value);
            }
        }
        UpdateCommandLinePreview();
    }

    private void UpdateCommandLinePreview()
    {
        if (_currentCommand is null)
        {
            CommandLinePreview.Text = string.Empty;
            return;
        }
        var sb = new StringBuilder("amql-cli");
        foreach (var g in _project.Cli.GlobalArguments()) sb.Append(' ').Append(g);
        sb.Append(' ').Append(_currentCommand.Name);
        foreach (var a in _currentCommand.BuildArguments(CurrentFormValues()))
        {
            sb.Append(' ').Append(NeedsQuoting(a) ? Quote(a) : a);
        }
        CommandLinePreview.Text = sb.ToString();
    }

    private static bool NeedsQuoting(string arg) => arg.Length == 0 || arg.Any(char.IsWhiteSpace) || arg.Contains('"');

    private static string Quote(string arg) => "\"" + arg.Replace("\"", "\\\"") + "\"";

    /// <summary>Required fields left empty, by their friendly names.</summary>
    private List<string> MissingRequired(Dictionary<string, string> values) =>
        _currentCommand!.Params
            .Where(p => p.Required && p.Kind != ParamKind.Switch && (!values.TryGetValue(p.Key, out var v) || v.Length == 0))
            .Select(Friendly).ToList();

    // ── global run settings ────────────────────────────────────────────────

    private void LoadGlobals()
    {
        _loadingGlobals = true;
        var c = _project.Cli;
        SelectTag(DeviceBox, c.Device);
        RamLimitCheck.IsChecked = c.MaxRamGb is not null;
        RamSlider.Value = c.MaxRamGb ?? Math.Min(10, RamSlider.Maximum);
        GpuLimitCheck.IsChecked = c.GpuMemoryGb is not null;
        GpuSlider.Value = c.GpuMemoryGb ?? Math.Min(8, GpuSlider.Maximum);
        SelectTag(GpuFormatBox, c.GpuFormat);
        SelectTag(Fp4PrefillBox, c.Fp4Prefill);
        SelectTag(WeightsBox, c.WeightsEnv);
        VerboseCheck.IsChecked = c.Verbose;
        _loadingGlobals = false;
        UpdateLimitLabels();
    }

    private static void SelectTag(ComboBox box, string tag) =>
        box.SelectedItem = box.Items.Cast<ComboBoxItem>().FirstOrDefault(i => (string)i.Tag == tag) ?? box.Items[0];

    private static string TagOf(ComboBox box) => (box.SelectedItem as ComboBoxItem)?.Tag as string ?? string.Empty;

    private void OnGlobalChanged(object sender, RoutedEventArgs e)
    {
        if (_loadingGlobals || !IsInitialized || VerboseCheck is null)
        {
            return;
        }
        var c = _project.Cli;
        c.Device = TagOf(DeviceBox) is { Length: > 0 } d ? d : "auto";
        c.MaxRamGb = RamLimitCheck.IsChecked == true ? RamSlider.Value : null;
        c.GpuMemoryGb = GpuLimitCheck.IsChecked == true ? GpuSlider.Value : null;
        c.GpuFormat = TagOf(GpuFormatBox);
        c.Fp4Prefill = TagOf(Fp4PrefillBox);
        c.WeightsEnv = TagOf(WeightsBox);
        c.Verbose = VerboseCheck.IsChecked == true;
        UpdateLimitLabels();
        UpdateCommandLinePreview();
    }

    private void UpdateLimitLabels()
    {
        RamLimitText.Text = $"{RamSlider.Value:0} GB";
        GpuLimitText.Text = $"{GpuSlider.Value:0.#} GB";
    }

    // ── run / cancel ────────────────────────────────────────────────────────

    private async void OnRun(object sender, RoutedEventArgs e)
    {
        if (_currentCommand is null || _runCts is not null)
        {
            return;
        }
        var cmd = _currentCommand;
        var values = CurrentFormValues();
        if (MissingRequired(values) is { Count: > 0 } missing)
        {
            MessageBox.Show(Window.GetWindow(this), "Fill in: " + string.Join(", ", missing), cmd.Name,
                MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        SaveCurrentFormValues();
        var args = cmd.BuildArguments(values);
        var record = new RunRecord
        {
            Id = _nextRunId++,
            Command = cmd.Name,
            Arguments = args,
            StartedUtc = DateTime.UtcNow,
            Status = "Running",
        };
        _activeRun = record;
        _history.Insert(0, record);
        while (_history.Count > MaxHistoryRuns) _history.RemoveAt(_history.Count - 1);
        HistoryList.SelectedItem = record;

        OutputBox.Clear();
        RunButton.IsEnabled = false;
        CancelButton.IsEnabled = true;
        RunProgress.Value = 0;
        RunProgress.IsIndeterminate = true;
        RunStateText.Text = "running…";

        // A Hugging Face token typed into the form goes to this run only, as HF_TOKEN.
        var environment = new Dictionary<string, string>();
        if (_editors.Values.OfType<ModelSourceEditor>().FirstOrDefault(m => m.Token is not null) is { Token: { } token })
        {
            environment["HF_TOKEN"] = token;
        }

        _runCts = new CancellationTokenSource();
        var lineCount = 0;
        var tail = new Queue<string>();
        try
        {
            int exit = await CliRunner.RunAsync(
                _project.Cli,
                cmd.Name,
                args,
                line => Dispatcher.BeginInvoke(() =>
                {
                    lineCount++;
                    record.OutputLines = lineCount;
                    AppendOutputLine(line);
                    tail.Enqueue(line);
                    while (tail.Count > MaxTailLines) tail.Dequeue();
                }),
                pct => Dispatcher.BeginInvoke(() =>
                {
                    RunProgress.IsIndeterminate = false;
                    RunProgress.Value = pct;
                    record.ProgressPercent = pct;
                    RunStateText.Text = $"{pct:0.#}%";
                }),
                _runCts.Token,
                environment: environment);

            record.ExitCode = exit;
            if (exit == CliExitCodes.Ok && cmd.Name == "encode" && values.TryGetValue("out", out var made))
            {
                AnnounceContainer(made);
            }
            // 0 = success, 1 = a legitimate negative result, 2 = usage/runtime error ('amql-cli help').
            record.Status = exit switch
            {
                CliExitCodes.Ok => "Succeeded",
                CliExitCodes.NegativeResult => "Negative",
                _ => "Failed",
            };
        }
        catch (OperationCanceledException)
        {
            record.Status = "Cancelled";
            record.Error = "cancelled by user";
        }
        catch (Exception ex)
        {
            record.Status = "Failed";
            record.Error = ex.Message;
            AppendOutputLine("[studio] " + ex.Message);
        }
        finally
        {
            record.FinishedUtc = DateTime.UtcNow;
            record.OutputTail = tail.ToList();
            _runCts?.Dispose();
            _runCts = null;
            _activeRun = null;
            RunButton.IsEnabled = true;
            CancelButton.IsEnabled = false;
            RunProgress.IsIndeterminate = false;
            if (record.Status == "Running") record.Status = "Failed";
            if (record.Status is "Succeeded" or "Negative") RunProgress.Value = 100;
            var took = record.FinishedUtc.Value - record.StartedUtc;
            RunStateText.Text = record.Status.ToLowerInvariant() +
                                (record.ExitCode is { } code and not 0 ? $" (exit {code})" : string.Empty) +
                                $" in {Format.Duration(took)}";
            HistoryList.Items.Refresh();
            SaveProject();
        }
    }

    /// <summary>Makes a newly encoded container show on the Server page, adding it when no model folder covers it.</summary>
    private void AnnounceContainer(string folder)
    {
        try
        {
            string full = Path.GetFullPath(folder);
            if (!File.Exists(Path.Combine(full, "index.json")))
            {
                return;
            }
            // The library scans each model folder two levels deep.
            char sep = Path.DirectorySeparatorChar;
            bool covered = AppSettings.Current.ModelFolders.Any(root =>
            {
                string r = Path.GetFullPath(root).TrimEnd(sep);
                return full.StartsWith(r + sep, StringComparison.OrdinalIgnoreCase)
                       && full.Count(c => c == sep) - r.Count(c => c == sep) <= 2;
            });
            if (!covered && !AppSettings.Current.ExtraModels.Contains(full, StringComparer.OrdinalIgnoreCase))
            {
                AppSettings.Current.ExtraModels.Add(full);
                AppSettings.Current.Save();
            }
            AppendOutputLine($"[studio] {Path.GetFileName(full)} is ready: tick it on the Server page to serve it (rescan if the page is open).");
        }
        catch (Exception e) when (e is ArgumentException or IOException or NotSupportedException)
        {
            // not a usable path; nothing to announce
        }
    }

    private void OnCancelRun(object sender, RoutedEventArgs e)
    {
        if (_runCts is { } cts)
        {
            try { cts.Cancel(); } catch (ObjectDisposedException) { }
            RunStateText.Text = "cancelling…";
        }
    }

    private void AppendOutputLine(string line)
    {
        if (OutputBox.LineCount > MaxUiOutputLines)
        {
            OutputBox.Clear();
            OutputBox.AppendText("[studio] output truncated (too many lines)\n");
        }
        OutputBox.AppendText(line + Environment.NewLine);
        OutputBox.ScrollToEnd();
    }

    private void OnHistorySelected(object sender, SelectionChangedEventArgs e)
    {
        if (HistoryList.SelectedItem is not RunRecord record || record == _activeRun) return;
        var sb = new StringBuilder();
        sb.AppendLine($"# Run {record.Id}: amql-cli {record.Command} {string.Join(' ', record.Arguments.Select(a => NeedsQuoting(a) ? Quote(a) : a))}");
        sb.AppendLine($"# started {record.StartedUtc.ToLocalTime():yyyy-MM-dd HH:mm:ss}" +
                      (record.FinishedUtc is { } end ? $", finished {end.ToLocalTime():HH:mm:ss}" : string.Empty));
        sb.AppendLine($"# status: {record.Status}" +
                      (record.ExitCode is { } code ? $", exit {code}" : string.Empty) +
                      (record.ProgressPercent is { } pct ? $", progress {pct:0.#}%" : string.Empty));
        if (record.Error is { Length: > 0 }) sb.AppendLine($"# error: {record.Error}");
        sb.AppendLine($"# output tail ({record.OutputTail.Count} of {record.OutputLines} lines):");
        foreach (var line in record.OutputTail) sb.AppendLine(line);
        OutputBox.Text = sb.ToString();
        OutputBox.ScrollToHome();
    }

    private void OnClearOutput(object sender, RoutedEventArgs e) => OutputBox.Clear();

    private void OnClearHistory(object sender, RoutedEventArgs e)
    {
        if (_runCts is not null) return;
        _history.Clear();
        SaveProject();
    }

    // ── project lifecycle ───────────────────────────────────────────────────

    private void OnProjectMenu(object sender, RoutedEventArgs e)
    {
        ProjectButton.ContextMenu.PlacementTarget = ProjectButton;
        ProjectButton.ContextMenu.IsOpen = true;
    }

    private void OnRedetectCli(object sender, RoutedEventArgs e)
    {
        ToolLocator.Refresh();
        var found = ToolLocator.Find(ToolLocator.Cli);
        MessageBox.Show(Window.GetWindow(this),
            found is null ? "amql-cli was not found. Build src/Amql.Cli or put amql-cli on PATH." : "Using " + found.Describe(),
            "amql-cli", MessageBoxButton.OK, MessageBoxImage.Information);
    }

    private void OnNewProject(object sender, RoutedEventArgs e)
    {
        if (!ConfirmClose()) return;
        var dialog = new NewProjectDialog { Owner = Window.GetWindow(this) };
        if (dialog.ShowDialog() != true) return;
        _project = new ProjectModel { Name = dialog.ProjectName };
        _project.Defaults.ContainerDir = dialog.InitialContainer;
        _project.Defaults.TokenizerDir = dialog.InitialTokenizer;
        _projectPath = dialog.ProjectPath;
        _nextRunId = 1;
        OutputBox.Clear();
        ReloadProjectIntoUi();
        SaveProject();
    }

    private void OnOpenProject(object sender, RoutedEventArgs e)
    {
        if (_runCts is not null || !ConfirmClose()) return;
        var dialog = new OpenFileDialog { Filter = ProjectModel.FileFilter, Title = "Open AMQL Project" };
        if (dialog.ShowDialog(Window.GetWindow(this)) == true)
        {
            TryLoadProject(dialog.FileName, quiet: false);
        }
    }

    private void OnSaveProjectAs(object sender, RoutedEventArgs e)
    {
        var dialog = new SaveFileDialog
        {
            Filter = ProjectModel.FileFilter,
            Title = "Save AMQL Project",
            FileName = SanitizeFileName(_project.Name) + ProjectModel.FileExtension,
        };
        if (dialog.ShowDialog(Window.GetWindow(this)) != true) return;
        _projectPath = dialog.FileName;
        _project.Name = Path.GetFileNameWithoutExtension(_projectPath);
        SaveProject();
    }

    private bool TryLoadProject(string path, bool quiet)
    {
        try
        {
            if (!File.Exists(path))
            {
                return false;
            }
            _project = ProjectModel.Load(path);
            _projectPath = path;
            foreach (var run in _project.Runs.Where(r => r.Status == "Running"))
            {
                run.Status = "Interrupted";
                run.Error = "the app exited while this run was active";
                run.FinishedUtc ??= run.StartedUtc;
            }
            _nextRunId = _project.Runs.Count == 0 ? 1 : _project.Runs.Max(r => r.Id) + 1;
            ReloadProjectIntoUi();
            return true;
        }
        catch (Exception ex)
        {
            if (!quiet)
            {
                MessageBox.Show(Window.GetWindow(this), $"Cannot open the project:\n{ex.Message}", "AMQL Studio",
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
            return false;
        }
    }

    /// <summary>Saves silently: projects are working state, not documents to be asked about.</summary>
    public void SaveProject()
    {
        try
        {
            SaveCurrentFormValues();
            _project.Runs = _history.Reverse().ToList();
            Directory.CreateDirectory(Path.GetDirectoryName(_projectPath)!);
            _project.Save(_projectPath);
            AppSettings.Current.LastProject = _projectPath == DefaultProjectPath ? null : _projectPath;
            RefreshProjectText();
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            ProjectText.Text = $"Could not save: {ex.Message}";
        }
    }

    /// <summary>Called before the app closes or another project opens: saves, and refuses while a run is active.</summary>
    public bool ConfirmClose()
    {
        if (_runCts is not null)
        {
            var answer = MessageBox.Show(Window.GetWindow(this), "A command is still running. Stop it?", "AMQL Studio",
                MessageBoxButton.YesNo, MessageBoxImage.Question);
            if (answer != MessageBoxResult.Yes)
            {
                return false;
            }
            _runCts.Cancel();
        }
        SaveProject();
        return true;
    }

    private void ReloadProjectIntoUi()
    {
        _history.Clear();
        foreach (var run in Enumerable.Reverse(_project.Runs)) _history.Add(run);
        LoadGlobals();
        BuildCommandTree();
        var target = (_project.LastCommand is { } name ? CommandCatalog.Find(name) : null)
                     ?? CommandCatalog.Find("generate") ?? CommandCatalog.All[0];
        _currentCommand = null;
        SelectCommandByName(target.Name);
        RefreshProjectText();
    }

    private void RefreshProjectText()
    {
        ProjectText.Text = _projectPath == DefaultProjectPath ? "Default project" : _project.Name;
        ProjectButton.ToolTip = $"Project: {_projectPath}";
    }

    private static TreeViewItem? FindTreeItem(ItemsControl parent, string commandName)
    {
        foreach (var child in parent.Items)
        {
            if (child is TreeViewItem tvi)
            {
                if (tvi.Tag is CommandDef cmd && cmd.Name == commandName) return tvi;
                if (FindTreeItem(tvi, commandName) is { } nested) return nested;
            }
        }
        return null;
    }

    private static string SanitizeFileName(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var result = new string(name.Select(c => invalid.Contains(c) ? '_' : c).ToArray()).Trim();
        return result.Length == 0 ? "project" : result;
    }

    private void OnProjectDefaults(object sender, RoutedEventArgs e)
    {
        var dialog = new DefaultsDialog(_project.Defaults) { Owner = Window.GetWindow(this) };
        if (dialog.ShowDialog() == true)
        {
            dialog.ApplyTo(_project.Defaults);
            SaveProject();
        }
    }
}
