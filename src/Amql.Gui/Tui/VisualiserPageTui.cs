// TextView is marked obsolete in favour of a separate editor package; it is still the text area Terminal.Gui ships.
#pragma warning disable CS0618
using System.IO;
using System.Globalization;
using System.Text;
using Amql.Gui.Model;
using Amql.Gui.Run;
using Amql.Inference.Tracing;
using Terminal.Gui.Drawing;
using Terminal.Gui.Input;
using Terminal.Gui.ViewBase;
using Terminal.Gui.Views;
using Attribute = Terminal.Gui.Drawing.Attribute;

namespace Amql.Gui.Tui;

/// <summary>
/// The inference visualiser as a page. A captured operator trace (or a live
/// run's trace stream) is drawn as a map: one row per kind of operator, one
/// column per layer, each cell coloured by how much that operator carried the
/// prompt. Arrow keys move over the map; beside it are the ranking, the
/// selected operator, the logit lens and the attention rows. The desktop's
/// zoomable canvas becomes a fixed grid, and right-click edits become Enter.
/// Every number comes from <see cref="TraceMetrics"/>, as on the desktop.
/// </summary>
internal sealed class VisualiserPageTui : ITuiPage
{
    private sealed record RankRow(int NodeId, int Layer, string Op, string Value);

    /// <summary>What a live run is asked for (the desktop's Run Inference dialog).</summary>
    private sealed class RunRequest
    {
        public string ContainerPath = "";
        public string Prompt = "The capital of France is";
        public int Steps = 8;
        public float Temperature;
        public bool UseChat, LogitLens, TraceAttention;
    }

    private static readonly (string Label, MapMetric Value)[] Metrics =
    {
        ("Mean ‖output‖", MapMetric.MeanL2), ("Max ‖output‖", MapMetric.MaxL2), ("Mean time", MapMetric.MeanMs), ("Causal share", MapMetric.Causal),
    };

    private static readonly string[] Rankings = { "Mean activation", "Peak activation", "Mean time", "Change vs baseline" };

    private readonly Shell _shell;
    private readonly View _root = new() { CanFocus = true };
    private readonly Button _runButton, _compareButton, _metricButton, _rankButton;
    private readonly Label _modelText = new() { X = 48, Y = 0, Width = Dim.Fill(), Height = 1 };
    private readonly CheckBox _aggregate = new() { Text = "_Whole run", X = 30, Y = 1, Value = CheckState.Checked };
    private readonly Label _stepText = new() { X = 52, Y = 1, Width = 18, Height = 1 };
    private readonly Label _stepInfo = new() { X = 0, Y = 2, Width = Dim.Fill(), Height = 1 };
    private readonly MapView _map;
    private readonly Tabs _side;
    private readonly View _rankTab, _operatorTab;
    private readonly TableView _rank = new() { X = 0, Y = 1, Width = Dim.Fill(), Height = Dim.Fill(), FullRowSelect = true };
    private readonly CheckBox _weightsOnly = new() { Text = "weights only", X = Pos.AnchorEnd(16), Y = 0 };
    private readonly Label _operator = new() { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill() };
    private readonly TextView _lens = new() { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill(), ReadOnly = true };
    private readonly TextView _attention = new() { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill(), ReadOnly = true };
    private readonly Label _hover = new() { X = 0, Y = Pos.AnchorEnd(1), Width = Dim.Fill(), Height = 1 };

    private RunTrace? _trace;
    private string? _tracePath;
    private IReadOnlyList<NodeDelta> _deltas = Array.Empty<NodeDelta>();
    private int _metric;
    private int _ranking;
    private int _step;
    private List<RankRow> _rankRows = new();
    private CliSettings _cli = new();
    private CancellationTokenSource? _liveCts;
    private object? _liveTimer;
    private string? _livePath;
    private int _liveSteps = -1;

    public VisualiserPageTui(Shell shell)
    {
        _shell = shell;
        _runButton = TuiKit.Button("▶ Run…", () => TuiKit.GuardAsync(RunOrStopAsync), 0, 0);
        var open = TuiKit.Button("Open trace…", OpenTraceDialog, Pos.Right(_runButton) + 1, 0);
        _compareButton = TuiKit.Button("⇄ Compare with…", Compare, Pos.Right(open) + 1, 0);
        _modelText.X = Pos.Right(_compareButton) + 2;
        _root.Add(_runButton, open, _compareButton, _modelText);

        _root.Add(new Label { Text = "Colour by", X = 0, Y = 1 });
        _metricButton = TuiKit.Button(Metrics[0].Label + " ▾", ChooseMetric, 10, 1);
        _aggregate.ValueChanged += (_, _) => ApplyStep();
        var prev = TuiKit.Button("◀", () => MoveStep(-1), 45, 1);
        var next = TuiKit.Button("▶", () => MoveStep(1), 71, 1);
        _root.Add(_metricButton, _aggregate, prev, _stepText, next, _stepInfo);

        _map = new MapView { X = 0, Y = 3, Width = Dim.Percent(58), Height = Dim.Fill(1) };
        _map.CursorMoved += OnNodeSelected;
        _map.NodeActivated += NodeActions;
        _root.Add(_map);

        _side = new Tabs { TabStop = TabBehavior.TabStop, X = Pos.Right(_map) + 1, Y = 3, Width = Dim.Fill(), Height = Dim.Fill(1) };
        _rankTab = new View { Title = "_Ranked", Width = Dim.Fill(), Height = Dim.Fill(), CanFocus = true };
        _rankButton = TuiKit.Button(Rankings[0] + " ▾", ChooseRanking, 0, 0);
        _weightsOnly.ValueChanged += (_, _) => RefreshRanking();
        _rank.ValueChanged += (_, _) =>
        {
            if (_rank.Value is { } sel && sel.SelectedCell.Y >= 0 && sel.SelectedCell.Y < _rankRows.Count)
            {
                _map.MoveTo(_rankRows[sel.SelectedCell.Y].NodeId);
            }
        };
        _rank.Accepting += (_, e) =>
        {
            e.Handled = true;
            if (_map.Selected is { } s)
            {
                NodeActions(s);
            }
        };
        _rankTab.Add(_rankButton, _weightsOnly, _rank);
        _operatorTab = new View { Title = "_Operator", Width = Dim.Fill(), Height = Dim.Fill(), CanFocus = true };
        _operatorTab.Add(_operator);
        var lensTab = new View { Title = "_Lens", Width = Dim.Fill(), Height = Dim.Fill(), CanFocus = true };
        lensTab.Add(_lens);
        var attentionTab = new View { Title = "_Attention", Width = Dim.Fill(), Height = Dim.Fill(), CanFocus = true };
        attentionTab.Add(_attention);
        _side.Add(_rankTab, _operatorTab, lensTab, attentionTab);
        _root.Add(_side, _hover);

        foreach (var view in new View[] { _map, _rank })
        {
            TuiKit.OnLetter(view, '[', () => MoveStep(-1));
            TuiKit.OnLetter(view, ',', () => MoveStep(-1));
            TuiKit.OnLetter(view, ']', () => MoveStep(1));
            TuiKit.OnLetter(view, '.', () => MoveStep(1));
        }
        Present(null, refit: true);
    }

    public View View => _root;

    public void Shown()
    {
    }

    public void Focus() => _map.SetFocus();

    public void Hidden()
    {
    }

    // ── opening ─────────────────────────────────────────────────────────────

    private void OpenTraceDialog()
    {
        if (TuiKit.PickPath("Open an inference trace", directory: false, _tracePath ?? "") is { } path)
        {
            OpenTrace(path);
        }
    }

    /// <summary>Opens a saved trace (--trace-json, or a --trace-stream file).</summary>
    public void OpenTrace(string path)
    {
        try
        {
            var trace = path.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase) ? TraceRecorder.ReadStream(path) : TraceRecorder.ReadJson(path);
            _tracePath = path;
            ClearComparison(reload: false);
            Present(trace, refit: true);
            TuiKit.Say($"opened {Path.GetFileName(path)}");
        }
        catch (Exception ex)
        {
            TuiKit.Error("Inference Visualiser", $"Could not read '{path}':\n\n{ex.Message}");
        }
    }

    /// <summary>Pushes a trace into every part of the page (a saved file and the live tail share it).</summary>
    private void Present(RunTrace? trace, bool refit)
    {
        _trace = trace;
        if (trace is null)
        {
            _modelText.Text = "No trace: Run… captures one, Open trace… reads a saved one.";
            _map.SetTrace(null, null, refit);
            _stepText.Text = "";
            _stepInfo.Text = "";
            _hover.Text = "Run a prompt through a container (▶ Run…) or open a trace written by generate --trace-json.";
            return;
        }
        _modelText.Text = $"{trace.Model} · {trace.Layers} layers · {trace.Nodes.Count} operators · {trace.Steps.Count} steps · {trace.HiddenSize} wide"
                          + (_deltas.Count > 0 ? $" · comparing ({_deltas.Count(d => d.RelativeChange > 0.01f)} of {_deltas.Count} moved >1%)" : "");
        _step = Math.Clamp(_step, 0, Math.Max(0, trace.Steps.Count - 1));
        if (trace.Causal is not null && _metric != 3 && refit)
        {
            _metric = 3;
        }
        else if (trace.Causal is null && _metric == 3)
        {
            _metric = 0;
        }
        _metricButton.Text = Metrics[_metric].Label + " ▾";
        _map.SetTrace(trace, _deltas.Count > 0 ? _deltas.ToDictionary(d => d.NodeId) : null, refit);
        RefreshRanking();
        ApplyStep();
    }

    // ── metric, step, ranking ───────────────────────────────────────────────

    private bool Aggregate => _aggregate.Value == CheckState.Checked;

    private void ChooseMetric()
    {
        var labels = Metrics.Select(m => m.Value == MapMetric.Causal && _trace?.Causal is null ? m.Label + " (not in this trace)" : m.Label).ToList();
        if (TuiKit.Choose("Colour by", labels, _metric, "The metric colours the whole-run map; a single step is always coloured by its own output size.") is not { } i)
        {
            return;
        }
        if (Metrics[i].Value == MapMetric.Causal && _trace?.Causal is null)
        {
            TuiKit.Say("this trace has no causal attribution");
            return;
        }
        _metric = i;
        _metricButton.Text = Metrics[i].Label + " ▾";
        if (!Aggregate)
        {
            _aggregate.Value = CheckState.Checked;
        }
        ApplyStep();
        RefreshRanking();
    }

    private void MoveStep(int delta)
    {
        if (_trace is null || _trace.Steps.Count == 0)
        {
            return;
        }
        if (Aggregate)
        {
            _aggregate.Value = CheckState.UnChecked;
            _step = delta > 0 ? 0 : _trace.Steps.Count - 1;
        }
        else
        {
            _step = Math.Clamp(_step + delta, 0, _trace.Steps.Count - 1);
        }
        ApplyStep();
    }

    private void ApplyStep()
    {
        if (_trace is null)
        {
            return;
        }
        int step = Aggregate ? -1 : _step;
        _map.SetView(step, Metrics[_metric].Value);
        _stepText.Text = Aggregate ? "  aggregate" : $"  step {step + 1} / {_trace.Steps.Count}";
        ShowStepInfo(step);
        ShowLens(step);
        ShowAttention(step);
        ShowMetricHint();
        OnNodeSelected(_map.Selected);
    }

    private void ShowMetricHint()
    {
        if (_trace?.Causal is { } c && _metric == 3 && Aggregate)
        {
            _hover.Text = $"causal share per LAYER (not per operator) · P(target) {c.CleanProbability:P2} clean → {c.CorruptProbability:P2} corrupted · " +
                          $"total effect {c.TotalEffect:P2} · red: restoring that layer recovered the target, blue: it made it worse";
        }
    }

    private void ChooseRanking()
    {
        var labels = Rankings.Select((r, i) => i == 3 && _deltas.Count == 0 ? r + " (compare first)" : r).ToList();
        if (TuiKit.Choose("Rank by", labels, _ranking) is { } i && (i != 3 || _deltas.Count > 0))
        {
            _ranking = i;
            _rankButton.Text = Rankings[i] + " ▾";
            RefreshRanking();
        }
    }

    private void RefreshRanking()
    {
        if (_trace is null)
        {
            _rankRows = new();
        }
        else if (_ranking == 3)
        {
            IEnumerable<NodeDelta> deltas = _deltas.OrderByDescending(d => d.RelativeChange).ThenBy(d => d.NodeId);
            if (_weightsOnly.Value == CheckState.Checked)
            {
                deltas = deltas.Where(d => d.HasWeight);
            }
            _rankRows = deltas.Select(d => new RankRow(d.NodeId, d.Layer, d.Op,
                $"{(d.Increased ? "+" : "−")}{d.RelativeChange * 100:F1}%   {d.BaselineMeanL2:F1} → {d.ModifiedMeanL2:F1}")).ToList();
        }
        else
        {
            IEnumerable<NodeStats> ranked = _ranking switch
            {
                1 => TraceMetrics.Reduce(_trace).OrderByDescending(s => s.MaxL2).ThenBy(s => s.NodeId),
                2 => TraceMetrics.RankByMeanMs(_trace),
                _ => TraceMetrics.RankByMeanL2(_trace),
            };
            if (_weightsOnly.Value == CheckState.Checked)
            {
                ranked = ranked.Where(s => s.HasWeight);
            }
            _rankRows = ranked.Select(s => new RankRow(s.NodeId, s.Layer, s.Op, _ranking switch
            {
                1 => $"max {s.MaxL2:F1}",
                2 => $"{s.MeanMs:F2} ms",
                _ => $"‖·‖ {s.MeanL2:F1}",
            })).ToList();
        }
        _rank.Table = new EnumerableTableSource<RankRow>(_rankRows, new Dictionary<string, Func<RankRow, object>>
        {
            ["L"] = r => r.Layer,
            ["operator"] = r => r.Op,
            ["value"] = r => r.Value,
        });
    }

    // ── step detail, lens, attention ────────────────────────────────────────

    private static string Quote(string text) => "“" + text.Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t") + "”";

    private static string Escape(string text) => text.Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");

    private void ShowStepInfo(int step)
    {
        if (_trace is null || _trace.Steps.Count == 0)
        {
            _stepInfo.Text = "";
            return;
        }
        if (step < 0)
        {
            float meanEntropy = (float)_trace.Steps.Average(s => s.Entropy);
            var hottest = _trace.Steps.SelectMany(s => s.RoutedExperts).GroupBy(e => e).OrderByDescending(g => g.Count()).Take(6).Select(g => $"e{g.Key}×{g.Count()}");
            string routing = _trace.Steps.Any(s => s.RoutedExperts.Count > 0) ? $"   most-routed experts: {string.Join(", ", hottest)}" : "";
            _stepInfo.Text = $"all {_trace.Steps.Count} steps · mean entropy {meanEntropy:F3} nats{routing}";
            return;
        }
        var s = _trace.Steps[Math.Min(step, _trace.Steps.Count - 1)];
        var text = new StringBuilder($"fed {(s.TokenText is null ? s.TokenId.ToString() : Quote(s.TokenText))} · entropy {s.Entropy:F3} · top-1 margin {s.Top1Margin:P1}");
        if (s.TopK.Count > 0)
        {
            text.Append("   → ").Append(string.Join("  ", s.TopK.Take(5).Select(c => $"{(c.Text is null ? c.Id.ToString() : Quote(c.Text))} {c.Probability:P1}")));
        }
        if (s.RoutedExperts.Count > 0)
        {
            text.Append($"   experts [{string.Join(", ", s.RoutedExperts)}]");
        }
        _stepInfo.Text = text.ToString();
    }

    /// <summary>The logit lens: what each layer would have emitted had the model stopped there.
    /// The row matching the token the step produced is marked ◀.</summary>
    private void ShowLens(int step)
    {
        if (_trace is null)
        {
            _lens.Text = "";
            return;
        }
        int index = step < 0 ? _trace.Steps.Count - 1 : Math.Min(step, _trace.Steps.Count - 1);
        var current = index >= 0 && index < _trace.Steps.Count ? _trace.Steps[index] : null;
        if (current?.Lens is not { Count: > 0 } rows)
        {
            _lens.Text = "This trace has no logit lens. Capture one with Run… and Logit lens ticked (generate --logit-lens).";
            return;
        }
        string? produced = _trace.Steps.Count > index + 1 ? _trace.Steps[index + 1].TokenText : null;
        var sb = new StringBuilder(step < 0 ? $"last step ({index + 1} of {_trace.Steps.Count})\n" : $"step {index + 1} of {_trace.Steps.Count}\n");
        foreach (var r in rows)
        {
            string token = Escape(r.TopK.Count > 0 && r.TopK[0].Text is { } t ? t : r.TokenId.ToString());
            bool isProduced = produced is not null && token == Escape(produced);
            sb.AppendLine($"L{r.Layer,-4}{r.Probability,7:P1}  {TuiKit.Bar(r.Probability, 10)}  {token}{(isProduced ? "  ◀" : "")}");
        }
        _lens.Text = sb.ToString();
    }

    /// <summary>Each head's last attention row as a sparkline, one glyph per key position.</summary>
    private void ShowAttention(int step)
    {
        if (_trace is null)
        {
            _attention.Text = "";
            return;
        }
        int index = step < 0 ? _trace.Steps.Count - 1 : Math.Min(step, _trace.Steps.Count - 1);
        var current = index >= 0 && index < _trace.Steps.Count ? _trace.Steps[index] : null;
        if (current?.Attention is not { Count: > 0 } rows)
        {
            _attention.Text = "This trace has no attention rows. Capture them with Run… and Attention rows ticked (generate --trace-attention).";
            return;
        }
        int softmaxLayers = rows.Select(r => r.Layer).Distinct().Count();
        var sb = new StringBuilder($"last query row · {softmaxLayers} softmax-attention layer(s) of {_trace.Layers} — recurrent layers have no attention matrix.\n");
        sb.AppendLine(step < 0 ? $"last step ({index + 1} of {_trace.Steps.Count})" : $"step {index + 1} of {_trace.Steps.Count}");
        foreach (var r in rows.OrderBy(r => r.Layer).ThenBy(r => r.Head))
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
            sb.AppendLine($"L{r.Layer} h{r.Head,-3} {TuiKit.Sparkline(r.Weights.Select(w => (double)w), r.Weights.Count, max)}  {(r.Weights.Count == 0 ? "" : $"p{peak} {r.Weights[peak]:P0}")}");
        }
        _attention.Text = sb.ToString();
    }

    // ── the selected operator ───────────────────────────────────────────────

    private void OnNodeSelected(NodeStats? stats)
    {
        if (stats is null)
        {
            _operator.Text = "(none) — move over the map with the arrow keys, or pick a row under Ranked.";
            if (_trace is not null && !(_metric == 3 && Aggregate))
            {
                _hover.Text = "arrows move · [ ] or , . step · Enter: edit or copy commands for the operator · Tab to the panels";
            }
            return;
        }
        string weight = stats.HasWeight ? $"{stats.WeightObject} / {stats.WeightTensor}" : "no single weight tensor (activation or fused operator)";
        var sb = new StringBuilder();
        sb.AppendLine($"{stats.Op}  ·  layer {stats.Layer}");
        sb.AppendLine(weight);
        sb.AppendLine();
        sb.AppendLine($"mean ‖·‖   {stats.MeanL2,10:F2}     max ‖·‖  {stats.MaxL2,10:F2}");
        sb.AppendLine($"mean |x|   {stats.MeanAbs,10:F4}     mean ms  {stats.MeanMs,10:F3}");
        sb.AppendLine($"steps      {stats.Steps,10}     intensity {stats.Intensity * 100,9:F1}%");
        sb.AppendLine(TuiKit.Bar(stats.Intensity, 30));
        if (_deltas.FirstOrDefault(d => d.NodeId == stats.NodeId) is { } delta)
        {
            sb.AppendLine($"vs baseline: {(delta.Increased ? "+" : "−")}{delta.RelativeChange * 100:F1}%  ({delta.BaselineMeanL2:F1} → {delta.ModifiedMeanL2:F1})");
        }
        if (stats.HasWeight)
        {
            sb.AppendLine();
            sb.AppendLine("This operator reads exactly one weight tensor, so it is a candidate for change-tensor/edit-tensor: Enter on the map offers the commands.");
        }
        _operator.Text = sb.ToString();
        _hover.Text = $"layer {stats.Layer} · {stats.Op}{(stats.HasWeight ? $"  ←  {stats.WeightTensor}" : "")}   mean ‖·‖ {stats.MeanL2:F2}   max ‖·‖ {stats.MaxL2:F2}   " +
                      $"mean |x| {stats.MeanAbs:F4}   {stats.MeanMs:F3} ms   intensity {stats.Intensity * 100:F1}%";
    }

    /// <summary>The desktop's right-click menu: edits to the operator's weight, as commands to copy.</summary>
    private void NodeActions(NodeStats stats)
    {
        var items = new List<(string Label, Action Run)>();
        if (stats.HasWeight)
        {
            items.Add(("Scale this tensor × 0.5", () => EditCommand(stats, "--scale 0.5")));
            items.Add(("Scale this tensor × 2", () => EditCommand(stats, "--scale 2")));
            items.Add(("Zero this tensor", () => EditCommand(stats, "--zero")));
            items.Add(("Re-run command (with this patch)", () => RerunCommand(stats)));
        }
        items.Add(("Copy tensor reference", () => TuiKit.Copy($"{stats.WeightObject}/{stats.WeightTensor}")));
        if (TuiKit.Choose($"{stats.Op} · layer {stats.Layer}", items.Select(i => i.Label).ToList()) is { } chosen)
        {
            items[chosen].Run();
        }
    }

    private string TraceDir() => string.IsNullOrEmpty(_tracePath)
        ? Environment.CurrentDirectory
        : Path.GetDirectoryName(Path.GetFullPath(_tracePath))!;

    private string PatchPathFor(NodeStats stats) => Path.Combine(TraceDir(), $"patch-L{stats.Layer}-{stats.Op.Replace('/', '_')}.safetensors");

    private string TracePathFor(NodeStats stats) => Path.Combine(TraceDir(), $"trace-L{stats.Layer}-{stats.Op}-edited.json");

    /// <summary>The edit-tensor command for this node's weight, shown to copy (the page does not run it:
    /// the Tools page is the one place that runs CLI commands with output capture).</summary>
    private void EditCommand(NodeStats stats, string operation)
    {
        if (string.IsNullOrEmpty(_trace?.ContainerPath))
        {
            TuiKit.Info("Inference Visualiser",
                "This trace does not record the container it was generated from, so the command cannot be completed.\n\n" +
                "Re-capture it with a current build of 'generate --trace-json', or substitute the container path yourself.");
            return;
        }
        TuiKit.ShowText("edit-tensor", $"amql-cli edit-tensor \"{_trace!.ContainerPath}\" {stats.WeightObject} {stats.WeightTensor} {operation} --out \"{PatchPathFor(stats)}\"", wrap: true);
    }

    private void RerunCommand(NodeStats stats)
    {
        if (string.IsNullOrEmpty(_trace?.ContainerPath))
        {
            EditCommand(stats, "--scale 1");
            return;
        }
        TuiKit.ShowText("generate with the patch",
            $"amql-cli generate \"{_trace!.ContainerPath}\" --prompt \"…\" --steps {_trace.Steps.Count} --temperature 0 " +
            $"--patch \"{PatchPathFor(stats)}\" --trace-json \"{TracePathFor(stats)}\"", wrap: true);
    }

    // ── comparison ──────────────────────────────────────────────────────────

    /// <summary>Recolours the map by how much each operator moved against a baseline trace of the same model.</summary>
    private void Compare()
    {
        if (_deltas.Count > 0)
        {
            ClearComparison(reload: true);
            return;
        }
        if (_trace is null)
        {
            TuiKit.Info("Inference Visualiser", "Open a trace first.");
            return;
        }
        if (TuiKit.PickPath("Open the baseline trace to compare against", directory: false, _tracePath ?? "") is not { } path)
        {
            return;
        }
        try
        {
            var baseline = TraceRecorder.ReadJson(path);
            if (!TraceMetrics.Comparable(baseline, _trace))
            {
                TuiKit.Error("Inference Visualiser",
                    "These traces do not describe the same model:\n\n" +
                    $"  loaded : {_trace.Model}, {_trace.Layers} layers, {_trace.Nodes.Count} operators\n" +
                    $"  baseline: {baseline.Model}, {baseline.Layers} layers, {baseline.Nodes.Count} operators\n\n" +
                    "Comparing them would line up unrelated operators.");
                return;
            }
            _deltas = TraceMetrics.Compare(baseline, _trace);
            _compareButton.Text = "⇄ Exit compare";
            _aggregate.Value = CheckState.Checked;
            _ranking = 3;
            _rankButton.Text = Rankings[3] + " ▾";
            Present(_trace, refit: false);
        }
        catch (Exception ex)
        {
            TuiKit.Error("Inference Visualiser", $"Could not read the baseline trace:\n\n{ex.Message}");
        }
    }

    private void ClearComparison(bool reload)
    {
        if (_deltas.Count == 0)
        {
            return;
        }
        _deltas = Array.Empty<NodeDelta>();
        _compareButton.Text = "⇄ Compare with…";
        _ranking = 0;
        _rankButton.Text = Rankings[0] + " ▾";
        if (reload)
        {
            Present(_trace, refit: false);
        }
    }

    // ── live run ────────────────────────────────────────────────────────────

    /// <summary>From the Explorer: asks for the run with the container filled in, then shows it here as it runs.</summary>
    public void RunAndDisplay(string containerPath, CliSettings cli)
    {
        if (AskRun(containerPath) is not { } request)
        {
            return;
        }
        _cli = cli;
        _shell.Navigate("visualiser");
        TuiKit.GuardAsync(() => RunLiveAsync(request));
    }

    private async Task RunOrStopAsync()
    {
        if (_liveCts is not null)
        {
            _liveCts.Cancel();
            return;
        }
        if (AskRun(_trace?.ContainerPath ?? "") is { } request)
        {
            _cli = _shell.Tools.Session.Cli;
            await RunLiveAsync(request);
        }
    }

    private static RunRequest? AskRun(string containerPath)
    {
        var r = new RunRequest { ContainerPath = containerPath };
        var fields = new Field[]
        {
            new PathField("Container", directory: true, () => r.ContainerPath, v => r.ContainerPath = v, "The VINDEX3 container to run.") { MustExist = true },
            new TextValueField("Prompt", () => r.Prompt, v => r.Prompt = v, "There is nothing to trace without one.") { Multiline = true },
            new TextValueField("Steps", () => r.Steps.ToString(CultureInfo.InvariantCulture), v => r.Steps = int.Parse(v, CultureInfo.InvariantCulture),
                "Tokens to generate, or -1 to run until a stop token.")
            {
                Validate = s => int.TryParse(s, out var n) && (n > 0 || n == -1) ? null : "Steps must be a positive whole number, or -1 to run until a stop token.",
            },
            new TextValueField("Temperature", () => r.Temperature.ToString(CultureInfo.InvariantCulture), v => r.Temperature = float.Parse(v, CultureInfo.InvariantCulture),
                "0 for greedy decoding.")
            {
                Validate = s => float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var t) && t >= 0 ? null : "Temperature must be a number, 0 or greater.",
            },
            new BoolField("Chat template", () => r.UseChat, v => r.UseChat = v, "--chat: wrap the prompt in the model's chat template."),
            new BoolField("Logit lens", () => r.LogitLens, v => r.LogitLens = v, "--logit-lens: what each layer would have emitted."),
            new BoolField("Attention rows", () => r.TraceAttention, v => r.TraceAttention = v, "--trace-attention: each head's last attention row."),
        };
        var dialog = new Dialog { Title = "Run and display inference", Width = Dim.Percent(80), Height = 18 };
        var list = new FieldList(2);
        list.SetFields(fields);
        dialog.Add(list);
        dialog.Add(new Label
        {
            Text = "Runs amql-cli generate as a child process and tails its trace stream, so the model stays out of this process.",
            X = 0, Y = Pos.AnchorEnd(1), Width = Dim.Fill(), Height = 1,
        });
        list.Height = Dim.Fill(1);
        dialog.AddButton(new Button { ShadowStyle = ShadowStyles.None, Text = "_Cancel" });
        dialog.AddButton(new Button { ShadowStyle = ShadowStyles.None, Text = "_Run" });
        list.FocusList();
        while (true)
        {
            TuiKit.App.Run(dialog);
            if (dialog.Result != 1)
            {
                dialog.Dispose();
                return null;
            }
            if (r.ContainerPath.Length == 0)
            {
                TuiKit.Error("Run Inference", "A container directory is required.");
                continue;
            }
            if (r.Prompt.Length == 0)
            {
                TuiKit.Error("Run Inference", "A prompt is required — there is nothing to trace without one.");
                continue;
            }
            dialog.Dispose();
            return r;
        }
    }

    private async Task RunLiveAsync(RunRequest request)
    {
        if (_liveCts is not null)
        {
            return;     // one run at a time; the button is showing Stop
        }
        _livePath = Path.Combine(Path.GetTempPath(), "amql-gui",
            $"live-{DateTime.Now:yyyyMMdd-HHmmss}-{Guid.NewGuid().ToString("N")[..8]}.jsonl");
        Directory.CreateDirectory(Path.GetDirectoryName(_livePath)!);
        _liveSteps = -1;
        var args = new List<string>
        {
            request.ContainerPath,
            "--prompt", request.Prompt,
            "--steps", request.Steps.ToString(CultureInfo.InvariantCulture),
            "--temperature", request.Temperature.ToString(CultureInfo.InvariantCulture),
            "--trace-stream", _livePath,
        };
        if (request.LogitLens) args.Add("--logit-lens");
        if (request.TraceAttention) args.Add("--trace-attention");
        if (request.UseChat) args.Add("--chat");

        ClearComparison(reload: false);
        _liveCts = new CancellationTokenSource();
        _liveTimer = TuiKit.App.AddTimeout(TimeSpan.FromMilliseconds(250), () =>
        {
            RefreshLive();
            return true;
        });
        _runButton.Text = "■ Stop";
        _aggregate.Value = CheckState.Checked;
        _hover.Text = $"running: amql-cli generate … --trace-stream ({request.Steps} steps)";
        int exit = -1;
        try
        {
            exit = await CliRunner.RunAsync(_cli, "generate", args,
                line => Services.UiThread.Run(() => _hover.Text = TuiKit.Shorten(line, 190)),
                onProgress: null, _liveCts.Token, machineProgress: false);
        }
        catch (OperationCanceledException)
        {
            // stopped
        }
        catch (Exception ex)
        {
            _hover.Text = $"run failed: {ex.Message}";
        }
        finally
        {
            if (_liveTimer is not null)
            {
                TuiKit.App.RemoveTimeout(_liveTimer);
                _liveTimer = null;
            }
            bool cancelled = _liveCts?.IsCancellationRequested == true;
            _liveCts?.Dispose();
            _liveCts = null;
            _runButton.Text = "▶ Run…";
            // One last read: the final steps are written after the process exits.
            RefreshLive();
            _hover.Text = cancelled
                ? $"cancelled after {_liveSteps} step(s) — the trace is partial"
                : exit == 0
                    ? $"run complete: {_liveSteps} step(s) → {_livePath}"
                    : $"run exited {exit}: {TuiKit.Shorten(_hover.Text, 160)}";
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
            if (trace.Steps.Count == _liveSteps)
            {
                return;     // nothing new since the last tick
            }
            bool first = _liveSteps < 0;
            _liveSteps = trace.Steps.Count;
            _tracePath = _livePath;
            Present(trace, refit: first);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or System.Text.Json.JsonException)
        {
            // caught mid-append; the next tick picks it up
        }
    }

    // ── the map ─────────────────────────────────────────────────────────────

    /// <summary>The forward pass as a grid: operator kinds down, layers across.</summary>
    private sealed class MapView : View
    {
        private const int LabelWidth = 15;
        private RunTrace? _trace;
        private Dictionary<int, NodeStats> _stats = new();
        private Dictionary<(int Layer, int Slot), int> _grid = new();
        private List<string> _slotNames = new();
        private IReadOnlyDictionary<int, NodeDelta>? _deltas;
        private IReadOnlyDictionary<int, float>? _stepValues;
        private MapMetric _metric = MapMetric.MeanL2;
        private float _peakMaxL2, _peakMeanMs, _causalPeak = 1f;
        private int _layers;
        private int _cursorLayer, _cursorSlot;
        private int _scrollX, _scrollY;

        public MapView() => CanFocus = true;

        public event Action<NodeStats?>? CursorMoved;

        public event Action<NodeStats>? NodeActivated;

        public NodeStats? Selected =>
            _grid.TryGetValue((_cursorLayer, _cursorSlot), out var id) && _stats.TryGetValue(id, out var s) ? s : null;

        public void SetTrace(RunTrace? trace, IReadOnlyDictionary<int, NodeDelta>? deltas, bool refit)
        {
            _trace = trace;
            _deltas = deltas;
            _stats = trace is null ? new() : TraceMetrics.Reduce(trace).ToDictionary(s => s.NodeId);
            _grid = new();
            _slotNames = new();
            _layers = 0;
            if (trace is not null)
            {
                var slots = TraceMetrics.OpSlots(trace.Nodes);
                var used = trace.Nodes.Select(n => slots[n.Op]).Distinct().OrderBy(s => s).ToList();
                var rowOf = used.Select((slot, row) => (slot, row)).ToDictionary(x => x.slot, x => x.row);
                _slotNames = used.Select(slot => slots.First(kv => kv.Value == slot).Key).ToList();
                foreach (var node in trace.Nodes)
                {
                    _grid[(node.Layer, rowOf[slots[node.Op]])] = node.Id;
                }
                _layers = TraceMetrics.LayoutSize(trace).Layers;
                _peakMaxL2 = _stats.Count == 0 ? 0 : _stats.Values.Max(s => s.MaxL2);
                _peakMeanMs = _stats.Count == 0 ? 0 : (float)_stats.Values.Max(s => s.MeanMs);
                _causalPeak = trace.Causal is { LayerShare.Count: > 0 } c ? Math.Max(1e-6f, c.LayerShare.Max(v => Math.Abs(v))) : 1f;
            }
            if (refit)
            {
                _cursorLayer = 0;
                _cursorSlot = 0;
                _scrollX = _scrollY = 0;
            }
            SetNeedsDraw();
        }

        public void SetView(int step, MapMetric metric)
        {
            _stepValues = _trace is not null && step >= 0 ? TraceMetrics.StepValues(_trace, step) : null;
            _metric = metric;
            SetNeedsDraw();
        }

        public void MoveTo(int nodeId)
        {
            foreach (var ((layer, slot), id) in _grid)
            {
                if (id == nodeId)
                {
                    _cursorLayer = layer;
                    _cursorSlot = slot;
                    EnsureVisible();
                    CursorMoved?.Invoke(Selected);
                    SetNeedsDraw();
                    return;
                }
            }
        }

        private int CellWidth => _layers == 0 ? 2 : Math.Clamp((Viewport.Width - LabelWidth) / _layers, 1, 3);

        private int VisibleLayers => Math.Max(1, (Viewport.Width - LabelWidth) / CellWidth);

        private int VisibleRows => Math.Max(1, Viewport.Height - 2);

        private void EnsureVisible()
        {
            if (Viewport.Width <= LabelWidth || Viewport.Height < 3)
            {
                return;     // not laid out yet: the next draw scrolls to the cursor
            }
            if (_cursorLayer < _scrollX) _scrollX = _cursorLayer;
            if (_cursorLayer >= _scrollX + VisibleLayers) _scrollX = _cursorLayer - VisibleLayers + 1;
            if (_cursorSlot < _scrollY) _scrollY = _cursorSlot;
            if (_cursorSlot >= _scrollY + VisibleRows) _scrollY = _cursorSlot - VisibleRows + 1;
        }

        protected override bool OnKeyDown(Key key)
        {
            if (_trace is null)
            {
                return base.OnKeyDown(key);
            }
            int dl = 0, ds = 0;
            if (key == Key.CursorLeft) dl = -1;
            else if (key == Key.CursorRight) dl = 1;
            else if (key == Key.CursorUp) ds = -1;
            else if (key == Key.CursorDown) ds = 1;
            else if (key == Key.PageUp) dl = -10;
            else if (key == Key.PageDown) dl = 10;
            else if (key == Key.Home) dl = -_cursorLayer;
            else if (key == Key.End) dl = _layers;
            else if (key == Key.Enter)
            {
                if (Selected is { } s)
                {
                    NodeActivated?.Invoke(s);
                }
                return true;
            }
            else
            {
                return base.OnKeyDown(key);
            }
            _cursorLayer = Math.Clamp(_cursorLayer + dl, 0, Math.Max(0, _layers - 1));
            _cursorSlot = Math.Clamp(_cursorSlot + ds, 0, Math.Max(0, _slotNames.Count - 1));
            EnsureVisible();
            CursorMoved?.Invoke(Selected);
            SetNeedsDraw();
            return true;
        }

        private float IntensityOf(NodeStats s)
        {
            if (_metric == MapMetric.Causal && _trace?.Causal is { } c && _stepValues is null && _deltas is null)
            {
                return s.Layer < c.LayerShare.Count ? Math.Abs(c.LayerShare[s.Layer]) / _causalPeak : 0f;
            }
            if (_deltas is { } deltas)
            {
                return deltas.TryGetValue(s.NodeId, out var d) ? d.Intensity : 0f;
            }
            if (_stepValues is { } sv)
            {
                return sv.TryGetValue(s.NodeId, out float v) ? v : 0f;
            }
            return _metric switch
            {
                MapMetric.MaxL2 => _peakMaxL2 <= 0 ? 0 : s.MaxL2 / _peakMaxL2,
                MapMetric.MeanMs => _peakMeanMs <= 0 ? 0 : (float)s.MeanMs / _peakMeanMs,
                _ => s.Intensity,
            };
        }

        private static Color Lerp((int R, int G, int B) a, (int R, int G, int B) b, float t) =>
            new((int)(a.R + (b.R - a.R) * t), (int)(a.G + (b.G - a.G) * t), (int)(a.B + (b.B - a.B) * t));

        /// <summary>The desktop map's ramps: cool-to-hot for magnitudes, grey to red or blue for signed changes.</summary>
        private Color ColorFor(NodeStats s)
        {
            float t = Math.Clamp(IntensityOf(s), 0f, 1f);
            bool signed = _deltas is not null || (_metric == MapMetric.Causal && _trace?.Causal is not null && _stepValues is null);
            if (signed)
            {
                bool up = _deltas is { } deltas
                    ? deltas.TryGetValue(s.NodeId, out var d) && d.Increased
                    : _trace!.Causal!.LayerShare.ElementAtOrDefault(s.Layer) >= 0;
                return Lerp((0x6A, 0x6F, 0x78), up ? (0xC0, 0x39, 0x2B) : (0x1F, 0x6F, 0xEB), t);
            }
            return t < 0.5f ? Lerp((0x33, 0x5C, 0x8A), (0xE8, 0xB4, 0x3C), t * 2f) : Lerp((0xE8, 0xB4, 0x3C), (0xC0, 0x39, 0x2B), (t - 0.5f) * 2f);
        }

        protected override bool OnDrawingContent(DrawContext? context)
        {
            var normal = GetAttributeForRole(VisualRole.Normal);
            SetAttribute(normal);
            ClearViewport(context);
            if (_trace is null)
            {
                AddStr(1, 1, "No trace loaded.");
                return true;
            }
            _scrollX = Math.Clamp(_scrollX, 0, Math.Max(0, _layers - VisibleLayers));
            _scrollY = Math.Clamp(_scrollY, 0, Math.Max(0, _slotNames.Count - VisibleRows));
            EnsureVisible();
            int cw = CellWidth;
            // Layer numbers along the top, every fifth.
            for (int l = _scrollX; l < Math.Min(_layers, _scrollX + VisibleLayers); l++)
            {
                if (l % 5 == 0)
                {
                    AddStr(LabelWidth + (l - _scrollX) * cw, 0, l.ToString(CultureInfo.InvariantCulture));
                }
            }
            for (int row = _scrollY; row < Math.Min(_slotNames.Count, _scrollY + VisibleRows); row++)
            {
                int y = row - _scrollY + 1;
                SetAttribute(row == _cursorSlot ? GetAttributeForRole(VisualRole.HotNormal) : normal);
                AddStr(0, y, TuiKit.Shorten(_slotNames[row], LabelWidth - 1).PadRight(LabelWidth));
                for (int l = _scrollX; l < Math.Min(_layers, _scrollX + VisibleLayers); l++)
                {
                    int x = LabelWidth + (l - _scrollX) * cw;
                    bool cursor = l == _cursorLayer && row == _cursorSlot;
                    if (!_grid.TryGetValue((l, row), out var id) || !_stats.TryGetValue(id, out var s))
                    {
                        SetAttribute(cursor ? new Attribute(normal.Background, normal.Foreground) : normal);
                        AddStr(x, y, (_grid.ContainsKey((l, row)) ? "·" : " ").PadRight(cw));
                        continue;
                    }
                    var color = ColorFor(s);
                    if (cursor)
                    {
                        SetAttribute(new Attribute(normal.Background, normal.Foreground));
                        AddStr(x, y, new string('▒', cw));
                    }
                    else
                    {
                        SetAttribute(new Attribute(color, normal.Background));
                        AddStr(x, y, new string('█', cw));
                    }
                }
            }
            SetAttribute(normal);
            string legend = _layers > VisibleLayers
                ? $"layers {_scrollX}–{Math.Min(_layers, _scrollX + VisibleLayers) - 1} of {_layers} · cursor L{_cursorLayer} {(_cursorSlot < _slotNames.Count ? _slotNames[_cursorSlot] : "")}"
                : $"{_layers} layers · cursor L{_cursorLayer} {(_cursorSlot < _slotNames.Count ? _slotNames[_cursorSlot] : "")}";
            AddStr(0, Math.Min(Viewport.Height - 1, Math.Min(_slotNames.Count, VisibleRows) + 1), TuiKit.Shorten(legend, Viewport.Width));
            return true;
        }
    }
}

/// <summary>What colours the map (the desktop's MapMetric, for the text UI).</summary>
internal enum MapMetric
{
    MeanL2,
    MaxL2,
    MeanMs,
    Causal,
}
