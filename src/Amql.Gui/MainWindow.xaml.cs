using System.Collections.ObjectModel;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using Amql.Gui.Commands;
using Amql.Gui.Dialogs;
using Amql.Gui.Model;
using Amql.Gui.Run;
using Microsoft.Win32;

namespace Amql.Gui;

public partial class MainWindow : Window
{
    private const int MaxTailLines = 400;      // output lines persisted per run
    private const int MaxHistoryRuns = 200;    // runs persisted in the project
    private const int MaxUiOutputLines = 20000;

    private ProjectModel _project = new();
    private string? _projectPath;
    private bool _dirty;

    private CommandDef? _currentCommand;
    private readonly Dictionary<string, FrameworkElement> _editors = new(StringComparer.OrdinalIgnoreCase);

    private CancellationTokenSource? _runCts;
    private RunRecord? _activeRun;
    private int _nextRunId = 1;
    private readonly ObservableCollection<RunRecord> _history = new();

    public MainWindow()
    {
        InitializeComponent();
        BuildCommandTree();
        HistoryList.ItemsSource = _history;
        Loaded += (_, _) => RefreshStatusBar();
    }

    protected override void OnPreviewKeyDown(KeyEventArgs e)
    {
        base.OnPreviewKeyDown(e);
        if (e.Key == Key.F5 && !_runInProgress)
        {
            e.Handled = true;
            OnRun(this, new RoutedEventArgs());
        }
    }

    private bool _runInProgress => _runCts is not null;

    // ── command tree ────────────────────────────────────────────────────────

    private void BuildCommandTree()
    {
        CommandTree.Items.Clear();
        foreach (var group in CommandCatalog.ByCategory())
        {
            var category = new TreeViewItem { Header = group.Key, IsExpanded = true };
            foreach (var cmd in group)
            {
                category.Items.Add(new TreeViewItem
                {
                    Header = cmd.Name,
                    Tag = cmd,
                    ToolTip = cmd.Summary,
                });
            }
            CommandTree.Items.Add(category);
        }
    }

    private void OnCommandSelected(object sender, RoutedPropertyChangedEventArgs<object> e)
    {
        if (e.NewValue is TreeViewItem { Tag: CommandDef cmd })
        {
            SelectCommand(cmd);
        }
    }

    private void SelectCommand(CommandDef cmd)
    {
        SaveCurrentFormValues();
        _currentCommand = cmd;
        _project.LastCommand = cmd.Name;
        CommandTitle.Text = $"amql-cli {cmd.Name}";
        CommandSummary.Text = cmd.Summary;
        BuildParamForm(cmd);
        UpdateCommandLinePreview();
        MarkDirty();
    }

    // ── dynamic parameter form ──────────────────────────────────────────────

    private void BuildParamForm(CommandDef cmd)
    {
        ParamPanel.Children.Clear();
        _editors.Clear();

        var state = _project.Command(cmd.Name);

        foreach (var p in cmd.Params)
        {
            var grid = new Grid { Margin = new Thickness(0, 0, 0, 2) };
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(180) });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });

            var label = new TextBlock
            {
                Text = p.Label + (p.Kind == ParamKind.Positional ? "" : ""),
                Style = (Style)FindResource("FieldLabel"),
                ToolTip = p.Help,
            };
            Grid.SetColumn(label, 0);
            grid.Children.Add(label);

            FrameworkElement editor = p.Editor switch
            {
                EditorKind.Choice => BuildChoice(p, state),
                EditorKind.MultiLineText => BuildMultiLine(p, state),
                _ => BuildSingleLine(p, state),
            };
            Grid.SetColumn(editor, 1);
            grid.Children.Add(editor);

            var browse = BuildBrowseButton(p, editor);
            if (browse is not null)
            {
                Grid.SetColumn(browse, 2);
                grid.Children.Add(browse);
            }

            ParamPanel.Children.Add(grid);

            if (p.Help is { Length: > 0 })
            {
                ParamPanel.Children.Add(new TextBlock
                {
                    Text = p.Help,
                    Style = (Style)FindResource("HelpText"),
                    Margin = new Thickness(184, 0, 4, 6),
                });
            }

            _editors[p.Key] = editor;
        }
    }

    private static string InitialValue(ParamDef p, CommandState state)
    {
        if (state.Values.TryGetValue(p.Key, out var saved))
        {
            return saved;
        }
        return p.DefaultValue;
    }

    private FrameworkElement BuildSingleLine(ParamDef p, CommandState state)
    {
        var box = new TextBox { Text = InitialValue(p, state), ToolTip = p.Help };
        box.TextChanged += (_, _) => { UpdateCommandLinePreview(); MarkDirty(); };
        return box;
    }

    private FrameworkElement BuildMultiLine(ParamDef p, CommandState state)
    {
        var box = new TextBox
        {
            Text = InitialValue(p, state),
            AcceptsReturn = true,
            TextWrapping = TextWrapping.Wrap,
            MinHeight = 52,
            MaxHeight = 140,
            VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
            VerticalContentAlignment = VerticalAlignment.Top,
            ToolTip = p.Help,
        };
        box.TextChanged += (_, _) => { UpdateCommandLinePreview(); MarkDirty(); };
        return box;
    }

    private FrameworkElement BuildChoice(ParamDef p, CommandState state)
    {
        var combo = new ComboBox { ToolTip = p.Help };
        foreach (var choice in p.Choices ?? Array.Empty<string>())
        {
            combo.Items.Add(choice);
        }
        var initial = InitialValue(p, state);
        combo.SelectedItem = combo.Items.Contains(initial) ? initial : (combo.Items.Count > 0 ? combo.Items[0] : null);
        combo.SelectionChanged += (_, _) => { UpdateCommandLinePreview(); MarkDirty(); };
        return combo;
    }

    private Button? BuildBrowseButton(ParamDef p, FrameworkElement editor)
    {
        if (p.Editor is not (EditorKind.Directory or EditorKind.File) || editor is not TextBox box)
        {
            return null;
        }
        bool isDir = p.Editor == EditorKind.Directory;
        var button = new Button { Content = "…", Padding = new Thickness(8, 3, 8, 3), Margin = new Thickness(4, 2, 2, 2) };
        button.Click += (_, _) =>
        {
            if (isDir)
            {
                var dialog = new OpenFolderDialog { Title = $"Select {p.Label}" };
                if (TryGetExistingDirectory(box.Text, out var start))
                {
                    dialog.InitialDirectory = start;
                }
                if (dialog.ShowDialog(this) == true)
                {
                    box.Text = dialog.FolderName;
                }
            }
            else
            {
                var dialog = new OpenFileDialog { Title = $"Select {p.Label}", CheckFileExists = false };
                if (TryGetExistingDirectory(Path.GetDirectoryName(box.Text), out var start))
                {
                    dialog.InitialDirectory = start;
                }
                if (dialog.ShowDialog(this) == true)
                {
                    box.Text = dialog.FileName;
                }
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
        catch (Exception)
        {
            // invalid path text — just don't preselect
        }
        return false;
    }

    // ── form ⇄ project model ───────────────────────────────────────────────

    private void SaveCurrentFormValues()
    {
        if (_currentCommand is null) return;
        var state = _project.Command(_currentCommand.Name);
        foreach (var (key, editor) in _editors)
        {
            state.Values[key] = editor switch
            {
                TextBox tb => tb.Text,
                ComboBox cb => cb.SelectedItem as string ?? string.Empty,
                _ => string.Empty,
            };
        }
    }

    private Dictionary<string, string> CurrentFormValues()
    {
        SaveCurrentFormValues();
        return _currentCommand is null
            ? new Dictionary<string, string>()
            : new Dictionary<string, string>(_project.Command(_currentCommand.Name).Values, StringComparer.OrdinalIgnoreCase);
    }

    private void OnApplyDefaults(object sender, RoutedEventArgs e)
    {
        if (_currentCommand is null) return;
        var d = _project.Defaults;
        var state = _project.Command(_currentCommand.Name);
        foreach (var p in _currentCommand.Params)
        {
            if (p.DefaultFrom is null) continue;
            var value = p.DefaultFrom switch
            {
                "container" => d.ContainerDir,
                "tokenizer" => d.TokenizerDir,
                "patch" => d.PatchFile,
                "component" => d.Component,
                _ => null,
            };
            if (!string.IsNullOrWhiteSpace(value) && _editors.TryGetValue(p.Key, out var editor))
            {
                SetEditorValue(editor, value!);
                state.Values[p.Key] = value!;
            }
        }
        UpdateCommandLinePreview();
        MarkDirty();
        StatusText.Text = "Project defaults applied to the form.";
    }

    private static void SetEditorValue(FrameworkElement editor, string value)
    {
        switch (editor)
        {
            case TextBox tb: tb.Text = value; break;
            case ComboBox cb when cb.Items.Contains(value): cb.SelectedItem = value; break;
        }
    }

    private void UpdateCommandLinePreview()
    {
        if (_currentCommand is null)
        {
            CommandLinePreview.Text = string.Empty;
            return;
        }
        var args = _currentCommand.BuildArguments(CurrentFormValuesQuiet());
        var sb = new StringBuilder("amql-cli");
        if (_project.Cli.Device is "cpu" or "gpu") sb.Append($" --{_project.Cli.Device}");
        sb.Append(' ').Append(_currentCommand.Name);
        foreach (var a in args)
        {
            sb.Append(' ');
            sb.Append(NeedsQuoting(a) ? Quote(a) : a);
        }
        CommandLinePreview.Text = sb.ToString();
    }

    /// <summary>Like <see cref="CurrentFormValues"/> but without re-saving (used from TextChanged).</summary>
    private Dictionary<string, string> CurrentFormValuesQuiet()
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (key, editor) in _editors)
        {
            values[key] = editor switch
            {
                TextBox tb => tb.Text.Trim(),
                ComboBox cb => cb.SelectedItem as string ?? string.Empty,
                _ => string.Empty,
            };
        }
        return values;
    }

    private static bool NeedsQuoting(string arg) =>
        arg.Length == 0 || arg.Any(char.IsWhiteSpace) || arg.Contains('"');

    private static string Quote(string arg) =>
        "\"" + arg.Replace("\"", "\\\"") + "\"";

    // ── run / cancel ────────────────────────────────────────────────────────

    private async void OnRun(object sender, RoutedEventArgs e)
    {
        if (_currentCommand is null)
        {
            MessageBox.Show(this, "Select a command first.", "AMQL Studio", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        if (_runInProgress) return;

        var cmd = _currentCommand;
        var args = cmd.BuildArguments(CurrentFormValues());

        var record = new RunRecord
        {
            Id = _nextRunId++,
            Command = cmd.Name,
            Arguments = args,
            StartedUtc = DateTime.UtcNow,
            Status = "Running",
        };
        _activeRun = record;
        _history.Add(record);
        TrimHistory();
        HistoryList.SelectedItem = record;
        HistoryList.ScrollIntoView(record);

        OutputBox.Clear();
        RunButton.IsEnabled = false;
        CancelButton.IsEnabled = true;
        RunProgress.Value = 0;
        RunProgress.IsIndeterminate = true;
        RunStateText.Text = $"running '{cmd.Name}'…";
        MarkDirty();

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
                _runCts.Token);

            record.ExitCode = exit;
            // Exit codes are documented by the CLI ('amql-cli help'):
            //   0 = success, 1 = a legitimate negative result (an answer,
            //   not a fault), 2 = usage/runtime error.
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
            AppendOutputLine("[gui] " + ex.Message);
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
            if (record.Status != "Cancelled")
            {
                RunProgress.Value = record.Status is "Succeeded" or "Negative" ? 100 : RunProgress.Value;
            }
            RunStateText.Text = $"{record.Status.ToLowerInvariant()}" +
                                (record.ExitCode is { } code ? $" (exit {code})" : string.Empty);
            RefreshHistoryDisplay();
            MarkDirty();
            AutoSaveIfPathSet();
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
            OutputBox.AppendText("[gui] output truncated (too many lines)\n");
        }
        OutputBox.AppendText(line + Environment.NewLine);
        OutputBox.ScrollToEnd();
    }

    private void TrimHistory()
    {
        while (_history.Count > MaxHistoryRuns)
        {
            _history.RemoveAt(0);
        }
    }

    private void RefreshHistoryDisplay()
    {
        // RunRecord has no INotifyPropertyChanged; refresh the items in place.
        var items = _history.ToList();
        _history.Clear();
        foreach (var r in items) _history.Add(r);
        if (_activeRun is not null) HistoryList.SelectedItem = _activeRun;
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
        if (_runInProgress)
        {
            MessageBox.Show(this, "A run is in progress — cancel it before clearing history.", "AMQL Studio",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        _history.Clear();
        MarkDirty();
    }

    // ── project lifecycle ───────────────────────────────────────────────────

    private void OnNewProject(object sender, RoutedEventArgs e)
    {
        if (!ConfirmDiscard()) return;

        var dialog = new NewProjectDialog { Owner = this };
        if (dialog.ShowDialog() != true) return;

        _project = new ProjectModel { Name = dialog.ProjectName };
        _projectPath = dialog.ProjectPath;
        _project.Cli.RepoPath = string.IsNullOrWhiteSpace(dialog.SuggestedRepoPath) ? null : dialog.SuggestedRepoPath;
        _project.Cli.CliExePath = dialog.CliExePath;
        _project.Cli.Device = dialog.Device;
        _nextRunId = 1;
        _history.Clear();
        OutputBox.Clear();

        // Seed defaults from the dialog's initial values.
        _project.Defaults.ContainerDir = dialog.InitialContainer;
        _project.Defaults.TokenizerDir = dialog.InitialTokenizer;

        ReloadProjectIntoUi();
        MarkDirty();
        AutoSaveIfPathSet();
        StatusText.Text = $"Created '{_projectPath}'";
    }

    private void OnOpenProject(object sender, RoutedEventArgs e)
    {
        if (!ConfirmDiscard()) return;

        var dialog = new OpenFileDialog { Filter = ProjectModel.FileFilter, Title = "Open AMQL Project" };
        if (dialog.ShowDialog(this) != true) return;

        try
        {
            _project = ProjectModel.Load(dialog.FileName);
            _projectPath = dialog.FileName;
            // Runs that were persisted mid-flight never finished — mark them.
            foreach (var run in _project.Runs.Where(r => r.Status == "Running"))
            {
                run.Status = "Interrupted";
                run.Error = "project was saved while this run was active (or the app exited)";
                run.FinishedUtc ??= run.StartedUtc;
            }
            _nextRunId = _project.Runs.Count == 0 ? 1 : _project.Runs.Max(r => r.Id) + 1;
            ReloadProjectIntoUi();
            _dirty = false;
            StatusText.Text = $"Opened '{_projectPath}'";
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, $"Cannot open the project:\n{ex.Message}", "AMQL Studio",
                MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void OnSaveProject(object sender, RoutedEventArgs e)
    {
        if (_projectPath is null)
        {
            OnSaveProjectAs(sender, e);
            return;
        }
        SaveProjectTo(_projectPath);
    }

    private void OnSaveProjectAs(object sender, RoutedEventArgs e)
    {
        var dialog = new SaveFileDialog
        {
            Filter = ProjectModel.FileFilter,
            Title = "Save AMQL Project",
            FileName = SanitizeFileName(_project.Name) + ProjectModel.FileExtension,
        };
        if (dialog.ShowDialog(this) != true) return;
        _projectPath = dialog.FileName;
        SaveProjectTo(_projectPath);
    }

    private void SaveProjectTo(string path)
    {
        try
        {
            PersistUiIntoProject();
            _project.Runs = _history.ToList();
            _project.Name = Path.GetFileNameWithoutExtension(path);
            _project.Save(path);
            _dirty = false;
            StatusText.Text = $"Saved '{path}' at {DateTime.Now:HH:mm:ss}";
            RefreshStatusBar();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, $"Cannot save the project:\n{ex.Message}", "AMQL Studio",
                MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    /// <summary>Autosave after each run when the project already has a path — keeps status fresh on disk.</summary>
    private void AutoSaveIfPathSet()
    {
        if (_projectPath is not null)
        {
            SaveProjectTo(_projectPath);
        }
    }

    private void PersistUiIntoProject()
    {
        SaveCurrentFormValues();
        if (_currentCommand is not null)
        {
            _project.LastCommand = _currentCommand.Name;
        }
    }

    private void ReloadProjectIntoUi()
    {
        _history.Clear();
        foreach (var run in _project.Runs) _history.Add(run);

        BuildCommandTree();

        var last = _project.LastCommand is { } name ? CommandCatalog.Find(name) : null;
        var target = last ?? CommandCatalog.All[0];
        var item = FindTreeItem(CommandTree, target.Name);
        if (item is not null)
        {
            item.IsSelected = true;
        }
        else
        {
            SelectCommand(target);
        }
        RefreshStatusBar();
        RefreshHistoryDisplay();
    }

    private static TreeViewItem? FindTreeItem(ItemsControl parent, string commandName)
    {
        foreach (var child in parent.Items)
        {
            if (child is TreeViewItem tvi)
            {
                if (tvi.Tag is CommandDef cmd && cmd.Name == commandName) return tvi;
                var nested = FindTreeItem(tvi, commandName);
                if (nested is not null) return nested;
            }
        }
        return null;
    }

    private bool ConfirmDiscard()
    {
        if (!_dirty) return true;
        var result = MessageBox.Show(this,
            "The current project has unsaved changes. Save before continuing?",
            "AMQL Studio", MessageBoxButton.YesNoCancel, MessageBoxImage.Question);
        switch (result)
        {
            case MessageBoxResult.Yes:
                if (_projectPath is null)
                {
                    // No path yet: ask for one. Cancelling the dialog aborts the operation.
                    var dialog = new SaveFileDialog
                    {
                        Filter = ProjectModel.FileFilter,
                        Title = "Save AMQL Project",
                        FileName = SanitizeFileName(_project.Name) + ProjectModel.FileExtension,
                    };
                    if (dialog.ShowDialog(this) != true) return false;
                    _projectPath = dialog.FileName;
                }
                SaveProjectTo(_projectPath);
                return !_dirty; // SaveProjectTo clears _dirty unless it failed
            case MessageBoxResult.No:
                return true;
            default:
                return false;
        }
    }

    private void MarkDirty()
    {
        _dirty = true;
        RefreshStatusBar();
    }

    private void RefreshStatusBar()
    {
        Title = $"AMQL Studio — {_project.Name}{(_dirty ? " •" : string.Empty)}{(_projectPath is null ? " (unsaved)" : string.Empty)}";
        DeviceText.Text = $"device: {_project.Cli.Device}";
        CliText.Text = "cli: " + (!string.IsNullOrWhiteSpace(_project.Cli.CliExePath)
            ? _project.Cli.CliExePath
            : !string.IsNullOrWhiteSpace(_project.Cli.RepoPath)
                ? $"dotnet run ({_project.Cli.RepoPath})"
                : "amql-cli on PATH");
    }

    private static string SanitizeFileName(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var sb = new StringBuilder(name.Length);
        foreach (var c in name) sb.Append(invalid.Contains(c) ? '_' : c);
        var result = sb.ToString().Trim();
        return result.Length == 0 ? "project" : result;
    }

    private void OnExit(object sender, RoutedEventArgs e)
    {
        if (_runCts is { } cts)
        {
            try { cts.Cancel(); } catch (ObjectDisposedException) { }
        }
        if (ConfirmDiscard())
        {
            Close();
        }
    }

    protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
    {
        base.OnClosing(e);
        if (!ConfirmDiscard())
        {
            e.Cancel = true;
            return;
        }
        if (_runCts is { } cts)
        {
            try { cts.Cancel(); } catch (ObjectDisposedException) { }
        }
    }

    // ── dialogs ─────────────────────────────────────────────────────────────

    private void OnProjectDefaults(object sender, RoutedEventArgs e)
    {
        var dialog = new DefaultsDialog(_project.Defaults) { Owner = this };
        if (dialog.ShowDialog() == true)
        {
            dialog.ApplyTo(_project.Defaults);
            MarkDirty();
        }
    }

    private void OnCliSettings(object sender, RoutedEventArgs e)
    {
        var dialog = new CliSettingsDialog(_project.Cli) { Owner = this };
        if (dialog.ShowDialog() == true)
        {
            dialog.ApplyTo(_project.Cli);
            UpdateCommandLinePreview();
            MarkDirty();
            RefreshStatusBar();
        }
    }

    private void OnAbout(object sender, RoutedEventArgs e)
    {
        MessageBox.Show(this,
            "AMQL Studio — a project-based WPF front-end for amql-cli.\n\n" +
            "Every command, option and global flag of the CLI is available; projects (.amqlproj) " +
            "store all parameters plus run status, progress and output tails.\n\n" +
            "The GUI drives amql-cli as a child process and references no AMQL library, " +
            "so it tracks the command line by construction.",
            "About AMQL Studio", MessageBoxButton.OK, MessageBoxImage.Information);
    }
}
