using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
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

            var trace = _trace!;
            ModelText.Text = trace.Model;
            SummaryText.Text =
                $"{trace.Layers} layers · {trace.Nodes.Count} operators · {trace.Steps.Count} steps · {trace.HiddenSize} wide";

            StepSlider.Maximum = Math.Max(0, trace.Steps.Count - 1);
            StepSlider.Value = 0;
            StepSlider.IsEnabled = trace.Steps.Count > 0;
            AggregateCheck.IsChecked = true;

            Map.SetTrace(trace);
            Map.SetMetric(MapMetric.MeanL2);
            RefreshRanking();
            ShowStepInfo(-1);
            Title = $"Inference Visualiser — {System.IO.Path.GetFileName(path)}";
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, $"Could not read '{path}':\n\n{ex.Message}",
                "Inference Visualiser", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

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
                _ => MapMetric.MeanL2,
            });
        }
        RefreshRanking();
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
