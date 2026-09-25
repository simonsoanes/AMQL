using System.Globalization;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using Amql.Inference.Tracing;

namespace Amql.Gui.Visualiser;

/// <summary>Which reduced statistic drives node colour and edge thickness.</summary>
public enum MapMetric
{
    MeanL2,
    MaxL2,
    MeanMs,
    Step,
    /// <summary>Difference against a baseline run, on a diverging ramp.</summary>
    Compare,
}

/// <summary>
/// The zoomable model map: one column per layer, operators stacked in the
/// order the computation happens, edges thickened and coloured by the selected
/// metric.
/// <para>
/// Everything is drawn imperatively in <see cref="FrameworkElement.OnRender"/>
/// from a cached layout rather than being one visual per node. A 64-layer model
/// is well over a thousand nodes, and a WPF element for each would not survive
/// being redrawn per generated token — which is the whole point of the view.
/// </para>
/// </summary>
public sealed class ModelMapCanvas : FrameworkElement
{
    private const double NodeW = 150;
    private const double NodeH = 24;
    private const double ColGap = 66;
    private const double RowGap = 9;
    private const double Pad = 26;
    private const double HeaderH = 30;

    private static readonly Typeface NodeTypeface = new("Segoe UI");
    private static readonly Typeface MonoTypeface = new("Cascadia Mono, Consolas");
    private static readonly Pen EdgePenBase = new(Brushes.LightGray, 1);
    private static readonly Pen NodeBorder = new(new SolidColorBrush(Color.FromRgb(0xC8, 0xD0, 0xDC)), 1);
    private static readonly Pen SelectedBorder = new(new SolidColorBrush(Color.FromRgb(0x1F, 0x6F, 0xEB)), 2.4);
    private static readonly Pen HoverBorder = new(new SolidColorBrush(Color.FromRgb(0x6A, 0x9F, 0xE8)), 1.6);
    private static readonly Brush WeightTab = new SolidColorBrush(Color.FromRgb(0x8E, 0x44, 0xAD));

    private RunTrace? _trace;
    private IReadOnlyList<NodeStats> _stats = Array.Empty<NodeStats>();
    private IReadOnlyDictionary<int, NodeStats> _statsById = new Dictionary<int, NodeStats>();
    private IReadOnlyDictionary<int, float>? _stepValues;
    private IReadOnlyDictionary<int, NodeDelta>? _deltas;
    private readonly Dictionary<int, Rect> _nodeRects = new();
    private readonly List<(int From, int To)> _edges = new();
    private readonly List<(int Layer, Rect Header)> _headers = new();
    private Size _extent;
    private Matrix _view = Matrix.Identity;
    private Point _panStart;
    private bool _panning;
    private int _selectedNode = -1;
    private int _hoveredNode = -1;

    public ModelMapCanvas()
    {
        // The wheel only reaches a focused element, and panning needs the
        // cursor over us to hit-test, so both focus and a background are
        // required rather than optional.
        Focusable = true;
        ClipToBounds = true;
        SnapsToDevicePixels = true;
    }

    public MapMetric Metric { get; set; } = MapMetric.MeanL2;

    /// <summary>Raised when the user picks a node, or clears the selection.</summary>
    public event Action<NodeStats?>? NodeSelected;

    /// <summary>Raised as the cursor moves, for the status line.</summary>
    public event Action<NodeStats?>? NodeHovered;

    /// <summary>Raised on right-click with the node under the cursor, or null
    /// when the click landed on empty canvas. The window owns the menu itself —
    /// what to offer depends on the trace, not on the rendering.</summary>
    public event Action<NodeStats?, Point>? NodeRightClicked;

    public NodeStats? Selected => StatsOf(_selectedNode);

    public void SetMetric(MapMetric metric)
    {
        Metric = metric;
        InvalidateVisual();
    }

    /// <summary>Loads a trace and frames it. A null trace clears the map.</summary>
    public void SetTrace(RunTrace? trace)
    {
        _trace = trace;
        _selectedNode = -1;
        _hoveredNode = -1;
        Rebuild();
        FitToView();
        NodeSelected?.Invoke(null);
    }

    /// <summary>Switches between the whole-run aggregate and one step's values.
    /// A negative step means the aggregate.</summary>
    public void SetStep(int step)
    {
        _stepValues = _trace is not null && step >= 0
            ? TraceMetrics.StepValues(_trace, step)
            : null;
        Metric = step >= 0 ? MapMetric.Step : Metric;
        InvalidateVisual();
    }

    public void ClearSelection()
    {
        _selectedNode = -1;
        InvalidateVisual();
        NodeSelected?.Invoke(null);
    }

    /// <summary>Switches the map to difference-against-baseline colouring, or
    /// back to the run aggregate when given null. The deltas are keyed by the
    /// node ids of the trace currently on screen, which is the modified run.</summary>
    public void SetComparison(IReadOnlyDictionary<int, NodeDelta>? deltas)
    {
        _deltas = deltas;
        _stepValues = null;
        Metric = deltas is null ? MapMetric.MeanL2 : MapMetric.Compare;
        InvalidateVisual();
    }

    public bool IsComparing => _deltas is not null;

    /// <summary>Frames the whole map in the viewport.</summary>
    public void FitToView()
    {
        if (_extent.Width <= 0 || _extent.Height <= 0 || RenderSize.Width <= 0 || RenderSize.Height <= 0)
        {
            return;
        }
        double scale = Math.Min(RenderSize.Width / _extent.Width, RenderSize.Height / _extent.Height);
        scale = Math.Clamp(scale, 0.04, 3.0);
        var m = new Matrix();
        m.Scale(scale, scale);
        m.Translate((RenderSize.Width - _extent.Width * scale) / 2,
                    (RenderSize.Height - _extent.Height * scale) / 2);
        _view = m;
        InvalidateVisual();
    }

    /// <summary>Centres the viewport on one node and zooms in enough to read it.</summary>
    public void ZoomToNode(int nodeId)
    {
        if (!_nodeRects.TryGetValue(nodeId, out var rect))
        {
            return;
        }
        var m = new Matrix();
        m.Scale(1.6, 1.6);
        m.Translate(RenderSize.Width / 2 - (rect.X + rect.Width / 2) * 1.6,
                    RenderSize.Height / 2 - (rect.Y + rect.Height / 2) * 1.6);
        _view = m;
        SelectNode(nodeId);
    }

    // ── layout ─────────────────────────────────────────────────────────────

    private void Rebuild()
    {
        _nodeRects.Clear();
        _edges.Clear();
        _headers.Clear();

        if (_trace is null)
        {
            _stats = Array.Empty<NodeStats>();
            _statsById = new Dictionary<int, NodeStats>();
            _stepValues = null;
            _deltas = null;
            _peakMaxL2 = 0f;
            _peakMeanMs = 0f;
            _extent = default;
            InvalidateVisual();
            return;
        }

        _stats = TraceMetrics.Reduce(_trace);
        _statsById = _stats.ToDictionary(s => s.NodeId);
        _stepValues = null;
        _deltas = null;
        RecomputePeaks();
        var slots = TraceMetrics.OpSlots(_trace.Nodes);

        var columns = _stats
            .GroupBy(s => s.Layer)
            .OrderBy(g => g.Key)
            .Select(g => g.OrderBy(s => slots.TryGetValue(s.Op, out int slot) ? slot : int.MaxValue)
                          .ThenBy(s => s.NodeId)
                          .ToList())
            .ToList();

        int deepest = columns.Count == 0 ? 0 : columns.Max(c => c.Count);
        int previousLast = -1;
        for (int c = 0; c < columns.Count; c++)
        {
            double x = Pad + c * (NodeW + ColGap);
            var column = columns[c];
            int layer = column[0].Layer;
            _headers.Add((layer, new Rect(x, Pad, NodeW, HeaderH - 8)));

            int previous = -1;
            for (int r = 0; r < column.Count; r++)
            {
                var stats = column[r];
                var rect = new Rect(x, Pad + HeaderH + r * (NodeH + RowGap), NodeW, NodeH);
                _nodeRects[stats.NodeId] = rect;
                if (previous >= 0)
                {
                    _edges.Add((previous, stats.NodeId));
                }
                previous = stats.NodeId;
            }

            // The residual trunk: the last operator of one layer feeds the
            // first of the next, which is what makes the map read as a single
            // forward pass rather than a pile of columns.
            if (previousLast >= 0 && column.Count > 0)
            {
                _edges.Add((previousLast, column[0].NodeId));
            }
            previousLast = column.Count > 0 ? column[^1].NodeId : previousLast;
        }

        _extent = new Size(
            Pad * 2 + Math.Max(0, columns.Count) * NodeW + Math.Max(0, columns.Count - 1) * ColGap,
            Pad * 2 + HeaderH + Math.Max(0, deepest) * NodeH + Math.Max(0, deepest - 1) * RowGap);
        InvalidateVisual();
    }

    private NodeStats? StatsOf(int nodeId)
        => nodeId >= 0 && _statsById.TryGetValue(nodeId, out var s) ? s : null;

    /// <summary>Peaks for normalising each metric, computed once per trace.
    /// OnRender runs on every pan frame, so a Max() over all nodes inside the
    /// per-node colour lookup would make redraw quadratic.</summary>
    private float _peakMaxL2;
    private float _peakMeanMs;

    private void RecomputePeaks()
    {
        _peakMaxL2 = 0f;
        _peakMeanMs = 0f;
        foreach (var s in _stats)
        {
            if (s.MaxL2 > _peakMaxL2)
            {
                _peakMaxL2 = s.MaxL2;
            }
            if (s.MeanMs > _peakMeanMs)
            {
                _peakMeanMs = (float)s.MeanMs;
            }
        }
    }

    /// <summary>The 0..1 value driving a node's colour: one step's normalised
    /// magnitude when a step is selected, the movement against a baseline when
    /// comparing, otherwise the run aggregate for the chosen metric.</summary>
    private float IntensityOf(NodeStats s)
    {
        if (_deltas is { } deltas)
        {
            return deltas.TryGetValue(s.NodeId, out var delta) ? delta.Intensity : 0f;
        }
        if (_stepValues is { } sv)
        {
            return sv.TryGetValue(s.NodeId, out float v) ? v : 0f;
        }
        return Metric switch
        {
            MapMetric.MaxL2 => Normalised(s.MaxL2, _peakMaxL2),
            MapMetric.MeanMs => Normalised((float)s.MeanMs, _peakMeanMs),
            _ => s.Intensity,
        };
    }

    private Color ColorFor(NodeStats s)
    {
        float intensity = IntensityOf(s);
        if (_deltas is { } deltas)
        {
            return DeltaColor(intensity,
                deltas.TryGetValue(s.NodeId, out var delta) && delta.Increased);
        }
        return RampColor(intensity);
    }

    private static float Normalised(float value, float peak) => peak <= 0f ? 0f : value / peak;

    private static Color Lerp(Color a, Color b, float t)
        => Color.FromRgb(
            (byte)(a.R + (b.R - a.R) * t),
            (byte)(a.G + (b.G - a.G) * t),
            (byte)(a.B + (b.B - a.B) * t));

    /// <summary>Cool-to-hot ramp. Deliberately not a rainbow: a monotone ramp
    /// reads as "less to more" without implying categories that are not there.</summary>
    internal static Color RampColor(float t)
    {
        t = Math.Clamp(t, 0f, 1f);
        var cold = Color.FromRgb(0x33, 0x5C, 0x8A);
        var mid = Color.FromRgb(0xE8, 0xB4, 0x3C);
        var hot = Color.FromRgb(0xC0, 0x39, 0x2B);
        return t < 0.5f ? Lerp(cold, mid, t * 2f) : Lerp(mid, hot, (t - 0.5f) * 2f);
    }

    /// <summary>Diverging ramp for a comparison: blue for an operator that went
    /// quieter than the baseline, red for one that got louder, grey for
    /// unchanged. A single-hue ramp would hide the direction, and direction is
    /// the whole point of diffing two runs.</summary>
    internal static Color DeltaColor(float intensity, bool increased)
    {
        intensity = Math.Clamp(intensity, 0f, 1f);
        var neutral = Color.FromRgb(0xD8, 0xDD, 0xE4);
        var loud = Color.FromRgb(0xC0, 0x39, 0x2B);
        var quiet = Color.FromRgb(0x1F, 0x6F, 0xEB);
        return Lerp(neutral, increased ? loud : quiet, intensity);
    }

    // ── rendering ──────────────────────────────────────────────────────────

    protected override void OnRender(DrawingContext dc)
    {
        dc.DrawRectangle(new SolidColorBrush(Color.FromRgb(0xFB, 0xFC, 0xFD)), null, new Rect(RenderSize));

        if (_trace is null)
        {
            DrawHint(dc);
            return;
        }

        double zoom = _view.M11;
        bool drawText = zoom > 0.34;
        bool drawFineText = zoom > 0.72;
        double pixelsPerDip = VisualTreeHelper.GetDpi(this).PixelsPerDip;

        dc.PushTransform(new MatrixTransform(_view));

        // Layer headers first, so columns read as layers behind the nodes.
        foreach (var (layer, header) in _headers)
        {
            dc.DrawRectangle(new SolidColorBrush(Color.FromRgb(0xEC, 0xF0, 0xF6)), null,
                new Rect(header.X - 6, Pad - 6, NodeW + 12,
                    _extent.Height - Pad * 2 + 6));
            if (drawText)
            {
                dc.DrawText(
                    new FormattedText($"layer {layer}", CultureInfo.InvariantCulture, FlowDirection.LeftToRight,
                        NodeTypeface, 12, new SolidColorBrush(Color.FromRgb(0x55, 0x66, 0x7A)), pixelsPerDip)
                    { TextAlignment = TextAlignment.Center },
                    new Point(header.X + NodeW / 2, Pad - 4));
            }
        }

        // Edges under nodes.
        foreach (var (from, to) in _edges)
        {
            if (!_nodeRects.TryGetValue(from, out var a) || !_nodeRects.TryGetValue(to, out var b))
            {
                continue;
            }
            if (!_statsById.TryGetValue(from, out var src))
            {
                continue;
            }
            float intensity = IntensityOf(src);
            var pen = new Pen(new SolidColorBrush(ColorFor(src)), 1 + 5.5 * intensity)
            {
                LineJoin = PenLineJoin.Round,
                StartLineCap = PenLineCap.Round,
                EndLineCap = PenLineCap.Round,
            };
            bool sameColumn = Math.Abs(a.X - b.X) < 0.5;
            if (sameColumn)
            {
                dc.DrawLine(pen, new Point(a.X + NodeW / 2, a.Bottom), new Point(b.X + NodeW / 2, b.Top));
            }
            else
            {
                // Elbow out of the source column and into the top of the next.
                var mid = new Point(a.Right + ColGap / 2, a.Y + NodeH / 2);
                var figure = new StreamGeometry();
                using (var ctx = figure.Open())
                {
                    ctx.BeginFigure(new Point(a.Right, a.Y + NodeH / 2), false, false);
                    ctx.PolyLineTo(new[] { mid, new Point(mid.X, b.Y - RowGap / 2), new Point(b.X + NodeW / 2, b.Y - RowGap / 2), new Point(b.X + NodeW / 2, b.Top) }, true, false);
                }
                figure.Freeze();
                dc.DrawGeometry(null, pen, figure);
            }
        }

        foreach (var (id, rect) in _nodeRects)
        {
            var stats = _statsById[id];
            float intensity = IntensityOf(stats);
            var fill = new SolidColorBrush(ColorFor(stats));
            var border = id == _selectedNode ? SelectedBorder
                : id == _hoveredNode ? HoverBorder
                : NodeBorder;

            dc.DrawRoundedRectangle(fill, border, rect, 4, 4);

            // A violet tab marks an operator that reads exactly one weight
            // tensor — the things the user can select and then edit.
            if (stats.HasWeight)
            {
                dc.DrawRectangle(WeightTab, null, new Rect(rect.X, rect.Y + 4, 4, rect.Height - 8));
            }

            if (!drawText)
            {
                continue;
            }

            string label = stats.Op;
            var text = new FormattedText(label, CultureInfo.InvariantCulture, FlowDirection.LeftToRight,
                MonoTypeface, 10.5, Brushes.White, pixelsPerDip)
            {
                MaxTextWidth = Math.Max(20, NodeW - 14),
                Trimming = TextTrimming.CharacterEllipsis,
            };
            dc.DrawText(text, new Point(rect.X + 9, rect.Y + (NodeH - text.Height) / 2));

            if (drawFineText)
            {
                string value = Metric switch
                {
                    MapMetric.MeanMs => $"{stats.MeanMs:F2} ms",
                    MapMetric.MaxL2 => $"max {stats.MaxL2:F1}",
                    MapMetric.Step => $"{intensity * 100:F0}%",
                    MapMetric.Compare => _deltas is { } d && d.TryGetValue(id, out var delta)
                        ? $"{(delta.Increased ? "+" : "−")}{delta.RelativeChange * 100:F0}%"
                        : "—",
                    _ => $"‖·‖ {stats.MeanL2:F1}",
                };
                var sub = new FormattedText(value, CultureInfo.InvariantCulture, FlowDirection.LeftToRight,
                    MonoTypeface, 8.5, new SolidColorBrush(Color.FromArgb(0xDD, 0xFF, 0xFF, 0xFF)), pixelsPerDip);
                dc.DrawText(sub, new Point(rect.Right - sub.Width - 6, rect.Y + NodeH - sub.Height - 2));
            }
        }

        dc.Pop();
    }

    private void DrawHint(DrawingContext dc)
    {
        var text = new FormattedText(
            "No trace loaded.\n\nRun `amql-cli generate <container> --prompt \"…\" --trace-json trace.json`,\nthen open trace.json here.",
            CultureInfo.InvariantCulture, FlowDirection.LeftToRight, NodeTypeface, 14,
            new SolidColorBrush(Color.FromRgb(0x6A, 0x77, 0x88)), VisualTreeHelper.GetDpi(this).PixelsPerDip)
        {
            TextAlignment = TextAlignment.Center,
        };
        dc.DrawText(text, new Point(RenderSize.Width / 2 - text.Width / 2, RenderSize.Height / 2 - text.Height / 2));
    }

    // ── interaction ────────────────────────────────────────────────────────

    private Point ToModel(Point view)
    {
        // WPF's Matrix has no Inverse property — Invert() mutates, so work on a
        // copy rather than destroying the view transform.
        if (!_view.HasInverse)
        {
            return view;
        }
        var inverse = _view;
        inverse.Invert();
        return inverse.Transform(view);
    }

    private int HitTest(Point viewPoint)
    {
        if (_nodeRects.Count == 0)
        {
            return -1;
        }
        var model = ToModel(viewPoint);
        foreach (var (id, rect) in _nodeRects)
        {
            if (rect.Contains(model))
            {
                return id;
            }
        }
        return -1;
    }

    private void SelectNode(int nodeId)
    {
        _selectedNode = nodeId;
        InvalidateVisual();
        NodeSelected?.Invoke(StatsOf(nodeId));
    }

    protected override void OnMouseWheel(MouseWheelEventArgs e)
    {
        base.OnMouseWheel(e);
        if (_trace is null)
        {
            return;
        }
        double factor = e.Delta > 0 ? 1.18 : 1 / 1.18;
        var at = e.GetPosition(this);
        // ScaleAt keeps the point under the cursor fixed, which is what makes
        // wheel-zoom feel like it is going where you are pointing.
        _view.ScaleAt(factor, factor, at.X, at.Y);
        InvalidateVisual();
        e.Handled = true;
    }

    protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
    {
        base.OnMouseLeftButtonDown(e);
        Focus();
        var p = e.GetPosition(this);

        // A FrameworkElement has no OnMouseDoubleClick — that lives on Control
        // — so the double-click is recognised from the click count here.
        if (e.ClickCount == 2)
        {
            int dbl = HitTest(p);
            if (dbl >= 0)
            {
                ZoomToNode(dbl);
            }
            else
            {
                FitToView();
            }
            e.Handled = true;
            return;
        }

        int hit = HitTest(p);
        if (hit >= 0)
        {
            SelectNode(hit);
            e.Handled = true;
            return;
        }
        _panning = true;
        _panStart = p;
        CaptureMouse();
        Cursor = Cursors.SizeAll;
    }

    protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
    {
        base.OnMouseLeftButtonUp(e);
        if (_panning)
        {
            _panning = false;
            ReleaseMouseCapture();
            Cursor = Cursors.Arrow;
        }
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        var p = e.GetPosition(this);
        if (_panning)
        {
            _view.Translate(p.X - _panStart.X, p.Y - _panStart.Y);
            _panStart = p;
            InvalidateVisual();
            return;
        }
        int hit = HitTest(p);
        if (hit != _hoveredNode)
        {
            _hoveredNode = hit;
            Cursor = hit >= 0 ? Cursors.Hand : Cursors.Arrow;
            NodeHovered?.Invoke(StatsOf(hit));
            InvalidateVisual();
        }
    }

    protected override void OnMouseRightButtonUp(MouseButtonEventArgs e)
    {
        base.OnMouseRightButtonUp(e);
        int hit = HitTest(e.GetPosition(this));
        if (hit >= 0)
        {
            SelectNode(hit);
        }
        NodeRightClicked?.Invoke(StatsOf(hit), e.GetPosition(this));
        e.Handled = true;
    }

    protected override void OnRenderSizeChanged(SizeChangedInfo sizeInfo)
    {
        base.OnRenderSizeChanged(sizeInfo);
        FitToView();
    }
}
