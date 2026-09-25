using System.Globalization;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;
using Amql.Gui.Model;
using Amql.Gui.Run;
using Amql.Inference.Tracing;
using Microsoft.Win32;

namespace Amql.Gui.Visualiser;

/// <summary>
/// Opens a captured operator trace and draws the forward pass as a zoomable
/// map, so the tensors that actually carried a given prompt can be found by
/// looking rather than by guessing — and then edited and re-run.
/// <para>
/// The window only renders and filters. Every number it shows comes from
/// <see cref="TraceMetrics"/>, which has no UI dependency and is covered by
/// tests, so what is on screen is what the tests assert on.
/// </para>
/// </summary>
public sealed partial class InferenceVisualiserWindow : Window
{
    private sealed record RankRow(int NodeId, int Layer, string Op, string Display);

    private RunTrace? _trace;
    private string? _tracePath;
    private IReadOnlyList<NodeDelta> _deltas = Array.Empty<NodeDelta>();

    // Live run state. Inference runs in a CLI child process and this window
    // tails its trace stream, so the model never loads here.
    private CliSettings _cli = new();
    private CancellationTokenSource? _liveCts;
    private DispatcherTimer? _liveTimer;
    private string? _livePath;
    private bool _liveFitted;
    private int _liveSteps;
    private bool _liveAutoFit = true;

    /// <summary>Supplies the CLI location and device/weights settings, normally
    /// from the open project. Without it the runner falls back to autodetect.</summary>
    public void SetCliSettings(CliSettings settings) => _cli = settings;

    /// <summary>Set once construction has finished. Several controls declare a
    /// default state in XAML (<c>IsChecked="True"</c>, a selected ComboBoxItem),
    /// and those fire their events from inside InitializeComponent — before the
    /// controls named later in the file exist. Without this the window throws a
    /// NullReferenceException while it is still being built.
    /// </summary>
    private bool _ready;

    public InferenceVisualiserWindow()
    {
        InitializeComponent();
        Map.NodeSelected += OnNodeSelected;
        Map.NodeHovered += OnNodeHovered;
        Map.NodeRightClicked += OnNodeRightClicked;
        _ready = true;
    }

    /// <summary>Opens directly on a trace file, so a generate run can hand its
    /// trace over without the user having to browse for it again.</summary>
    public InferenceVisualiserWindow(string tracePath) : this()
    {
        Loaded += (_, _) => LoadTrace(tracePath);
    }

    // ── loading ────────────────────────────────────────────────────────────

    private void OnOpenTrace(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Title = "Open an inference trace",
            Filter = "Inference trace (*.json)|*.json|All files (*.*)|*.*",
        };
        if (dialog.ShowDialog(this) == true)
        {
            LoadTrace(dialog.FileName);
        }
    }

    private void LoadTrace(string path, bool quiet = false)
    {
        try
        {
            // Re-reading is skipped when only the chrome needs restoring, e.g.
            // after leaving compare mode.
            if (!quiet || _trace is null)
            {
                _trace = TraceRecorder.ReadJson(path);
            }
            _tracePath = path;
            PresentTrace(_trace!, refit: true);
            Title = $"Inference Visualiser — {System.IO.Path.GetFileName(path)}";
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, $"Could not read '{path}':\n\n{ex.Message}",
                "Inference Visualiser", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    /// <summary>Pushes a trace into every part of the window. Shared by opening
    /// a saved file and by the live tail, so the two cannot present the same
    /// data differently.</summary>
    private void PresentTrace(RunTrace trace, bool refit)
    {
        _trace = trace;
        _deltas = Array.Empty<NodeDelta>();

        ModelText.Text = trace.Model;
        SummaryText.Text =
            $"{trace.Layers} layers · {trace.Nodes.Count} operators · {trace.Steps.Count} steps · {trace.HiddenSize} wide";

        StepSlider.Maximum = Math.Max(0, trace.Steps.Count - 1);
        if (StepSlider.Value > StepSlider.Maximum)
        {
            StepSlider.Value = StepSlider.Maximum;
        }
        StepSlider.IsEnabled = trace.Steps.Count > 0 && AggregateCheck.IsChecked != true;
        CausalMetricItem.IsEnabled = trace.Causal is not null;

        Map.SetTrace(trace, refit);
        Map.SetCausal(trace.Causal);
        if (trace.Causal is not null && MetricBox.SelectedIndex != 3)
        {
            MetricBox.SelectedIndex = 3;
        }
        else if (trace.Causal is null && MetricBox.SelectedIndex == 3)
        {
            MetricBox.SelectedIndex = 0;
            Map.SetMetric(MapMetric.MeanL2);
        }

        RefreshRanking();
        int step = AggregateCheck.IsChecked == true ? -1 : (int)StepSlider.Value;
        ShowStepInfo(step);
        ShowLens(step);
        ShowAttention(step);
        ShowMetricHint();
    }

    // ── live run ───────────────────────────────────────────────────────────

    /// <summary>Starts a generation in a CLI child process and tails its trace
    /// stream, updating the map as each step lands. Pressing the button again
    /// cancels: there is no CancellationToken inside Amql.Inference, so the
    /// child process is the unit that can actually be stopped.</summary>
    private async void OnRunLive(object sender, RoutedEventArgs e)
    {
        if (_liveCts is not null)
        {
            _liveCts.Cancel();
            return;
        }

        var dialog = new RunInferenceDialog(_trace?.ContainerPath ?? string.Empty, "The capital of France is")
        {
            Owner = this,
        };
        if (dialog.ShowDialog() != true)
        {
            return;
        }

        // Scratch owned by this window, so it goes in the temp directory rather
        // than beside the user's containers.
        _livePath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "amql-gui",
            $"live-{DateTime.Now:yyyyMMdd-HHmmss}-{Guid.NewGuid().ToString("N")[..8]}.jsonl");
        _liveFitted = false;
        _liveSteps = -1;
        _liveAutoFit = dialog.AutoFit;

        var args = new List<string>
        {
            dialog.ContainerPath,
            "--prompt", dialog.Prompt,
            "--steps", dialog.Steps.ToString(CultureInfo.InvariantCulture),
            "--temperature", dialog.Temperature.ToString(CultureInfo.InvariantCulture),
            "--trace-stream", _livePath,
        };
        if (dialog.LogitLens)
        {
            args.Add("--logit-lens");
        }
        if (dialog.TraceAttention)
        {
            args.Add("--trace-attention");
        }

        _liveCts = new CancellationTokenSource();
        _liveTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(250) };
        _liveTimer.Tick += (_, _) => RefreshLive();
        _liveTimer.Start();
        RunButton.Content = "■ Stop";
        AggregateCheck.IsChecked = true;
        HoverText.Text = $"running: amql-cli generate … --trace-stream ({dialog.Steps} steps)";

        // Initialised because a finally block is reachable from anywhere in the
        // try, including before the assignment.
        int exit = -1;
        try
        {
            exit = await CliRunner.RunAsync(_cli, "generate", args,
                line => Dispatcher.BeginInvoke(() => HoverText.Text = Truncate(line, 190)),
                onProgress: null, _liveCts.Token, machineProgress: false);
        }
        catch (Exception ex)
        {
            HoverText.Text = $"run failed: {ex.Message}";
            exit = -1;
        }
        finally
        {
            _liveTimer?.Stop();
            _liveTimer = null;
            bool cancelled = _liveCts?.IsCancellationRequested == true;
            _liveCts?.Dispose();
            _liveCts = null;
            RunButton.Content = "▶ Run…";
            // One last read: the final steps are written after the process has
            // already exited, and stopping the timer first would lose them.
            RefreshLive();
            HoverText.Text = cancelled
                ? $"cancelled after {_liveSteps} step(s) — the trace is partial"
                : exit == 0
                    ? $"run complete: {_liveSteps} step(s) → {_livePath}"
                    : $"run exited {exit}: {Truncate(HoverText.Text, 160)}";
        }
    }

    private void RefreshLive()
    {
        if (_livePath is null || !File.Exists(_livePath))
        {
            return;
        }
        try
        {
            var trace = TraceRecorder.ReadStream(_livePath);
            if (trace.Steps.Count == _liveSteps && _liveFitted)
            {
                return;     // nothing new since the last tick
            }
            _liveSteps = trace.Steps.Count;
            _tracePath = _livePath;
            // Refit while the map is still growing if the user asked for it, and
            // always on the first read so the run starts framed.
            bool refit = !_liveFitted || _liveAutoFit;
            _liveFitted = true;
            PresentTrace(trace, refit);
        }
        catch (IOException)
        {
            // Caught mid-append; the next tick picks it up.
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    private static string Truncate(string text, int max)
        => text.Length <= max ? text : text[..(max - 1)] + "…";

    // ── toolbar ────────────────────────────────────────────────────────────

    private void OnMetricChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_ready || Map is null)
        {
            return;
        }
        // "Whole run" owns the step selection; a metric change only matters for
        // the aggregate view, so leave a selected step alone.
        if (AggregateCheck.IsChecked == true)
        {
            Map.SetMetric(MetricBox.SelectedIndex switch
            {
                1 => MapMetric.MaxL2,
                2 => MapMetric.MeanMs,
                3 => MapMetric.Causal,
                _ => MapMetric.MeanL2,
            });
        }
        ShowMetricHint();
        RefreshRanking();
    }

    /// <summary>Causal attribution is per layer, so the map says so rather than
    /// letting a per-operator colour imply a resolution the measurement does
    /// not have.</summary>
    private void ShowMetricHint()
    {
        if (_trace?.Causal is { } c && MetricBox.SelectedIndex == 3)
        {
            HoverText.Text =
                $"causal share per LAYER (not per operator) · P(target) {c.CleanProbability:P2} clean → "
                + $"{c.CorruptProbability:P2} corrupted · total effect {c.TotalEffect:P2} · "
                + $"red = restoring that layer's residual recovered the target, blue = it made it worse";
        }
    }

    private void OnAggregateChanged(object sender, RoutedEventArgs e)
    {
        if (!_ready)
        {
            return;
        }
        bool aggregate = AggregateCheck.IsChecked == true;
        StepSlider.IsEnabled = !aggregate && (_trace?.Steps.Count ?? 0) > 0;
        MetricBox.IsEnabled = aggregate;
        ApplyStep();
    }

    private void OnStepChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (_ready)
        {
            ApplyStep();
        }
    }

    private void ApplyStep()
    {
        if (_trace is null)
        {
            return;
        }
        bool aggregate = AggregateCheck.IsChecked == true;
        int step = aggregate ? -1 : (int)StepSlider.Value;
        Map.SetStep(step);
        StepText.Text = aggregate ? "aggregate" : $"step {step + 1} / {_trace.Steps.Count}";
        ShowStepInfo(step);
        ShowLens(step);
        ShowAttention(step);
    }

    private void OnFit(object sender, RoutedEventArgs e) => Map.FitToView();

    private void OnClearSelection(object sender, RoutedEventArgs e) => Map.ClearSelection();

    // ── comparison ─────────────────────────────────────────────────────────

    /// <summary>Loads a second trace of the same model and recolours the map by
    /// how much each operator moved. Clicking again clears the comparison. This
    /// is the half of the loop that makes an edit judgeable: change a weight,
    /// re-run with the same prompt, and diff the two runs.</summary>
    private void OnCompare(object sender, RoutedEventArgs e)
    {
        if (Map.IsComparing)
        {
            ClearComparison();
            return;
        }
        if (_trace is null)
        {
            MessageBox.Show(this, "Open a trace first.", "Inference Visualiser",
                MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        var dialog = new OpenFileDialog
        {
            Title = "Open the baseline trace to compare against",
            Filter = "Inference trace (*.json)|*.json|All files (*.*)|*.*",
        };
        if (dialog.ShowDialog(this) != true)
        {
            return;
        }

        try
        {
            var baseline = TraceRecorder.ReadJson(dialog.FileName);
            if (!TraceMetrics.Comparable(baseline, _trace))
            {
                MessageBox.Show(this,
                    $"These traces do not describe the same model:\n\n"
                    + $"  loaded : {_trace.Model}, {_trace.Layers} layers, {_trace.Nodes.Count} operators\n"
                    + $"  baseline: {baseline.Model}, {baseline.Layers} layers, {baseline.Nodes.Count} operators\n\n"
                    + "Comparing them would line up unrelated operators.",
                    "Inference Visualiser", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            _deltas = TraceMetrics.Compare(baseline, _trace);
            Map.SetComparison(_deltas.ToDictionary(d => d.NodeId));

            CompareButton.Content = "⇄ Exit compare";
            AggregateCheck.IsChecked = true;
            MetricBox.IsEnabled = false;
            RankBox.SelectedIndex = 3;
            RefreshRanking();

            int moved = _deltas.Count(d => d.RelativeChange > 0.01f);
            SummaryText.Text =
                $"comparing against {System.IO.Path.GetFileName(dialog.FileName)} · "
                + $"{moved} of {_deltas.Count} operators moved >1%";
            Title = $"Inference Visualiser — {System.IO.Path.GetFileName(_tracePath ?? "trace")} vs "
                + $"{System.IO.Path.GetFileName(dialog.FileName)}";
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, $"Could not read the baseline trace:\n\n{ex.Message}",
                "Inference Visualiser", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void ClearComparison()
    {
        _deltas = Array.Empty<NodeDelta>();
        Map.SetComparison(null);
        CompareButton.Content = "⇄ Compare with…";
        MetricBox.IsEnabled = true;
        RankBox.SelectedIndex = 0;
        LoadTrace(_tracePath!, quiet: true);
    }

    // ── step detail ────────────────────────────────────────────────────────

    /// <summary>Renders the output distribution and MoE routing for one step,
    /// or a summary across the run when none is selected.</summary>
    private void ShowStepInfo(int step)
    {
        if (_trace is null || _trace.Steps.Count == 0)
        {
            StepInfoText.Text = string.Empty;
            return;
        }

        if (step < 0)
        {
            float meanEntropy = (float)_trace.Steps.Average(s => s.Entropy);
            var hottest = _trace.Steps
                .SelectMany(s => s.RoutedExperts)
                .GroupBy(e => e)
                .OrderByDescending(g => g.Count())
                .Take(6)
                .Select(g => $"e{g.Key}×{g.Count()}");
            string routing = _trace.Steps.Any(s => s.RoutedExperts.Count > 0)
                ? $"   most-routed experts: {string.Join(", ", hottest)}"
                : string.Empty;
            StepInfoText.Text =
                $"all {_trace.Steps.Count} steps · mean entropy {meanEntropy:F3} nats{routing}";
            return;
        }

        var s = _trace.Steps[Math.Min(step, _trace.Steps.Count - 1)];
        var text = new StringBuilder();
        text.Append($"fed {(s.TokenText is null ? s.TokenId.ToString() : Quote(s.TokenText))}"
            + $" · entropy {s.Entropy:F3} · top-1 margin {s.Top1Margin:P1}");
        if (s.TopK.Count > 0)
        {
            text.Append("   → ");
            text.Append(string.Join("  ", s.TopK.Take(5).Select(c =>
                $"{(c.Text is null ? c.Id.ToString() : Quote(c.Text))} {c.Probability:P1}")));
        }
        if (s.RoutedExperts.Count > 0)
        {
            text.Append($"   experts [{string.Join(", ", s.RoutedExperts)}]");
        }
        StepInfoText.Text = text.ToString();
    }

    /// <summary>Shows a token's text with whitespace visible, since leading
    /// spaces and newlines are exactly what makes a continuation readable.</summary>
    private static string Quote(string text)
        => "“" + text.Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t") + "”";

    // ── node detail ────────────────────────────────────────────────────────

    private void OnNodeSelected(NodeStats? stats)
    {
        if (stats is null)
        {
            SelectedOpText.Text = "(none)";
            SelectedWeightText.Text = string.Empty;
            SelectedStatsGrid.Visibility = Visibility.Collapsed;
            SelectedHintText.Visibility = Visibility.Collapsed;
            IntensityBar.Width = 0;
            return;
        }

        SelectedOpText.Text = $"{stats.Op}  ·  layer {stats.Layer}";
        SelectedWeightText.Text = stats.HasWeight
            ? $"{stats.WeightObject} / {stats.WeightTensor}"
            : "no single weight tensor (activation or fused operator)";
        SelectedWeightText.Foreground = stats.HasWeight
            ? new SolidColorBrush(Color.FromRgb(0x8E, 0x44, 0xAD))
            : Brushes.Gray;

        StatMeanL2.Text = stats.MeanL2.ToString("F2");
        StatMaxL2.Text = stats.MaxL2.ToString("F2");
        StatMeanAbs.Text = stats.MeanAbs.ToString("F4");
        StatMeanMs.Text = $"{stats.MeanMs:F3} ms";
        StatSteps.Text = stats.Steps.ToString();
        StatIntensity.Text = $"{stats.Intensity * 100:F1}%";
        SelectedStatsGrid.Visibility = Visibility.Visible;
        SelectedHintText.Visibility = stats.HasWeight ? Visibility.Visible : Visibility.Collapsed;
        IntensityBar.Width = Math.Clamp(stats.Intensity, 0f, 1f) * 320;
    }

    private void OnNodeHovered(NodeStats? stats)
    {
        if (stats is null)
        {
            HoverText.Text =
                "wheel = zoom at cursor · drag = pan · click = select · double-click = zoom to node / fit";
            return;
        }
        string weight = stats.HasWeight ? $"  ←  {stats.WeightTensor}" : string.Empty;
        HoverText.Text =
            $"layer {stats.Layer} · {stats.Op}{weight}   mean ‖·‖ {stats.MeanL2:F2}   "
            + $"max ‖·‖ {stats.MaxL2:F2}   mean |x| {stats.MeanAbs:F4}   {stats.MeanMs:F3} ms   "
            + $"intensity {stats.Intensity * 100:F1}%";
    }

    private sealed record LensRowVm(string LayerLabel, string ProbabilityLabel, string Token,
        Brush ProbabilityBrush, FontWeight Weight);

    /// <summary>Renders the logit lens for a step: what each layer would have
    /// emitted had the model stopped there. The row matching the token the step
    /// actually produced is bolded, so reading down the list shows the depth at
    /// which the decision formed and how early a wrong turn was already visible.
    /// Only present when the trace was captured with --logit-lens.</summary>
    private void ShowLens(int step)
    {
        if (_trace is null || LensPanel is null)
        {
            return;
        }
        int index = step < 0 ? _trace.Steps.Count - 1 : Math.Min(step, _trace.Steps.Count - 1);
        var current = index >= 0 && index < _trace.Steps.Count ? _trace.Steps[index] : null;
        if (current?.Lens is not { Count: > 0 } rows)
        {
            LensPanel.Visibility = Visibility.Collapsed;
            return;
        }

        LensPanel.Visibility = Visibility.Visible;
        // The lens at the final layer predicts what the step PRODUCED, which is
        // the next step's input — not the token this step consumed.
        string? produced = _trace.Steps.Count > index + 1 ? _trace.Steps[index + 1].TokenText : null;
        LensStepText.Text = step < 0
            ? $"last step ({index + 1} of {_trace.Steps.Count})"
            : $"step {index + 1} of {_trace.Steps.Count}";

        var confident = new SolidColorBrush(Color.FromRgb(0xC0, 0x39, 0x2B));
        var tentative = new SolidColorBrush(Color.FromRgb(0x6A, 0x77, 0x88));
        LensList.ItemsSource = rows.Select(r =>
        {
            string token = Escape(r.TopK.Count > 0 && r.TopK[0].Text is { } t ? t : r.TokenId.ToString());
            bool isProduced = produced is not null && token == Escape(produced);
            return new LensRowVm(
                $"L{r.Layer}",
                r.Probability.ToString("P1"),
                token,
                r.Probability >= 0.15f ? confident : tentative,
                isProduced ? FontWeights.Bold : FontWeights.Normal);
        }).ToList();
    }

    private static string Escape(string text)
        => text.Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");

    // ── attention ──────────────────────────────────────────────────────────

    private sealed record AttentionRowVm(string Label, string Spark, string Peak);

    private const string SparkGlyphs = "▁▂▃▄▅▆▇█";

    /// <summary>Renders each head's attention row as a sparkline of block
    /// characters, one glyph per key position. Text glyphs rather than a drawn
    /// heatmap: it needs no custom rendering, stays legible at any panel width,
    /// and a stack of heads reads as small multiples of the same thing.</summary>
    private void ShowAttention(int step)
    {
        if (_trace is null || AttentionPanel is null)
        {
            return;
        }
        int index = step < 0 ? _trace.Steps.Count - 1 : Math.Min(step, _trace.Steps.Count - 1);
        var current = index >= 0 && index < _trace.Steps.Count ? _trace.Steps[index] : null;
        if (current?.Attention is not { Count: > 0 } rows)
        {
            AttentionPanel.Visibility = Visibility.Collapsed;
            return;
        }

        AttentionPanel.Visibility = Visibility.Visible;
        int softmaxLayers = rows.Select(r => r.Layer).Distinct().Count();
        AttentionHintText.Text =
            $"last query row · {softmaxLayers} softmax-attention layer(s) of {_trace.Layers} — "
            + "recurrent layers have no attention matrix, so their absence is expected. "
            + "Each glyph is one key position.";
        AttentionStepText.Text = step < 0
            ? $"last step ({index + 1} of {_trace.Steps.Count})"
            : $"step {index + 1} of {_trace.Steps.Count}";

        AttentionList.ItemsSource = rows
            .OrderBy(r => r.Layer)
            .ThenBy(r => r.Head)
            .Select(r =>
            {
                int peak = 0;
                for (int i = 1; i < r.Weights.Count; i++)
                {
                    if (r.Weights[i] > r.Weights[peak])
                    {
                        peak = i;
                    }
                }
                float max = r.Weights.Count == 0 ? 0f : r.Weights[peak];
                return new AttentionRowVm(
                    $"L{r.Layer} h{r.Head}",
                    Sparkline(r.Weights, max),
                    r.Weights.Count == 0 ? string.Empty : $"p{peak} {r.Weights[peak]:P0}");
            })
            .ToList();
    }

    private static string Sparkline(IReadOnlyList<float> weights, float max)
    {
        if (weights.Count == 0)
        {
            return string.Empty;
        }
        var sb = new StringBuilder(weights.Count);
        foreach (float w in weights)
        {
            int level = max <= 0f
                ? 0
                : (int)Math.Clamp(w / max * (SparkGlyphs.Length - 1), 0, SparkGlyphs.Length - 1);
            sb.Append(SparkGlyphs[level]);
        }
        return sb.ToString();
    }

    // ── editing a tensor the map has identified ────────────────────────────

    private void OnNodeRightClicked(NodeStats? stats, Point at)
    {
        if (stats is null)
        {
            return;
        }
        var menu = new ContextMenu { PlacementTarget = Map };
        if (stats.HasWeight)
        {
            menu.Items.Add(MenuItem("Scale this tensor × 0.5", () => CopyEditCommand(stats, "--scale 0.5")));
            menu.Items.Add(MenuItem("Scale this tensor × 2", () => CopyEditCommand(stats, "--scale 2")));
            menu.Items.Add(MenuItem("Zero this tensor", () => CopyEditCommand(stats, "--zero")));
            menu.Items.Add(new Separator());
            menu.Items.Add(MenuItem("Copy re-run command (with this patch)", () => CopyRerunCommand(stats)));
            menu.Items.Add(new Separator());
        }
        menu.Items.Add(MenuItem("Copy tensor reference",
            () => CopyToClipboard($"{stats.WeightObject}/{stats.WeightTensor}")));
        menu.IsOpen = true;
    }

    private static MenuItem MenuItem(string header, Action onClick)
    {
        var item = new MenuItem { Header = header };
        item.Click += (_, _) => onClick();
        return item;
    }

    /// <summary>
    /// The patch file an edit to this node would write, kept deterministic so
    /// the edit command and the re-run command agree without any state between
    /// them.
    /// </summary>
    private string PatchPathFor(NodeStats stats)
    {
        string dir = string.IsNullOrEmpty(_tracePath)
            ? Environment.CurrentDirectory
            : System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(_tracePath))!;
        string safe = stats.Op.Replace('/', '_');
        return System.IO.Path.Combine(dir, $"patch-L{stats.Layer}-{safe}.safetensors");
    }

    private string TracePathFor(NodeStats stats)
    {
        string dir = string.IsNullOrEmpty(_tracePath)
            ? Environment.CurrentDirectory
            : System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(_tracePath))!;
        return System.IO.Path.Combine(dir, $"trace-L{stats.Layer}-{stats.Op}-edited.json");
    }

    /// <summary>Builds the edit-tensor command for this node's weight and puts
    /// it on the clipboard. The window does not run it: the main window already
    /// has a command runner with output capture, and a second place that knows
    /// how to invoke the CLI would be a second place to keep in step.</summary>
    private void CopyEditCommand(NodeStats stats, string operation)
    {
        if (string.IsNullOrEmpty(_trace?.ContainerPath))
        {
            MessageBox.Show(this,
                "This trace does not record the container it was generated from, so the command "
                + "cannot be completed.\n\nRe-capture it with a current build of "
                + "'generate --trace-json', or substitute the container path yourself.",
                "Inference Visualiser", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        CopyToClipboard(
            $"amql-cli edit-tensor \"{_trace!.ContainerPath}\" {stats.WeightObject} {stats.WeightTensor} "
            + $"{operation} --out \"{PatchPathFor(stats)}\"");
    }

    private void CopyRerunCommand(NodeStats stats)
    {
        if (string.IsNullOrEmpty(_trace?.ContainerPath))
        {
            CopyEditCommand(stats, "--scale 1");   // reports the same missing-container problem
            return;
        }
        CopyToClipboard(
            $"amql-cli generate \"{_trace!.ContainerPath}\" --prompt \"…\" "
            + $"--steps {_trace.Steps.Count} --temperature 0 "
            + $"--patch \"{PatchPathFor(stats)}\" --trace-json \"{TracePathFor(stats)}\"");
    }

    private void CopyToClipboard(string text)
    {
        try
        {
            Clipboard.SetText(text);
            HoverText.Text = $"copied to clipboard: {text}";
        }
        catch (Exception ex)
        {
            HoverText.Text = $"clipboard unavailable: {ex.Message}";
        }
    }

    // ── ranking ────────────────────────────────────────────────────────────

    private void OnRankChanged(object sender, RoutedEventArgs e)
    {
        if (_ready)
        {
            RefreshRanking();
        }
    }

    private void RefreshRanking()
    {
        if (_trace is null || RankGrid is null)
        {
            return;
        }

        // Comparing ranks a different type, so it builds its own rows.
        if (RankBox.SelectedIndex == 3)
        {
            IEnumerable<NodeDelta> deltas = _deltas
                .OrderByDescending(d => d.RelativeChange)
                .ThenBy(d => d.NodeId);
            if (WeightsOnlyCheck.IsChecked == true)
            {
                deltas = deltas.Where(d => d.HasWeight);
            }
            RankGrid.ItemsSource = deltas
                .Select(d => new RankRow(d.NodeId, d.Layer, d.Op,
                    $"{(d.Increased ? "+" : "−")}{d.RelativeChange * 100:F1}%   "
                    + $"{d.BaselineMeanL2:F1} → {d.ModifiedMeanL2:F1}"))
                .ToList();
            return;
        }

        // The branches return different concrete sequence types, so name the
        // common one explicitly or the switch has no best type.
        IEnumerable<NodeStats> ranked = RankBox.SelectedIndex switch
        {
            1 => TraceMetrics.Reduce(_trace).OrderByDescending(s => s.MaxL2).ThenBy(s => s.NodeId),
            2 => TraceMetrics.RankByMeanMs(_trace),
            _ => TraceMetrics.RankByMeanL2(_trace),
        };
        if (WeightsOnlyCheck.IsChecked == true)
        {
            ranked = ranked.Where(s => s.HasWeight);
        }

        RankGrid.ItemsSource = ranked
            .Select(s => new RankRow(
                s.NodeId,
                s.Layer,
                s.Op,
                RankBox.SelectedIndex switch
                {
                    1 => $"max {s.MaxL2:F1}",
                    2 => $"{s.MeanMs:F2} ms",
                    _ => $"‖·‖ {s.MeanL2:F1}",
                }))
            .ToList();
    }

    private void OnRankSelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (RankGrid.SelectedItem is RankRow row)
        {
            Map.ZoomToNode(row.NodeId);
        }
    }
}
