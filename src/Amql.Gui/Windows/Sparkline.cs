using System.Globalization;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

namespace Amql.Gui.Windows;

/// <summary>
/// A single-series line over time: one measure, one axis starting at zero, a
/// 2px line with a faint fill, recessive gridlines, and a hover readout of the
/// value under the pointer. The title above it names the series, so it has no legend.
/// </summary>
public sealed class Sparkline : FrameworkElement
{
    private readonly List<(DateTime At, double Value)> _points = new();
    private Point? _hover;

    public TimeSpan Window { get; set; } = TimeSpan.FromMinutes(5);

    /// <summary>How a value is written in the hover readout and the top gridline label.</summary>
    public Func<double, string> FormatValue { get; set; } = v => v.ToString("0.#", CultureInfo.CurrentCulture);

    public Brush LineBrush { get; set; } = new SolidColorBrush(Color.FromRgb(0x3B, 0x82, 0xF6));

    public Sparkline()
    {
        MinHeight = 90;
        MouseMove += (_, e) => { _hover = e.GetPosition(this); InvalidateVisual(); };
        MouseLeave += (_, _) => { _hover = null; InvalidateVisual(); };
        Cursor = Cursors.Cross;
    }

    public void Add(double value)
    {
        var now = DateTime.Now;
        _points.Add((now, value));
        _points.RemoveAll(p => now - p.At > Window + TimeSpan.FromSeconds(5));
        InvalidateVisual();
    }

    public void Clear()
    {
        _points.Clear();
        InvalidateVisual();
    }

    protected override void OnRender(DrawingContext dc)
    {
        double w = ActualWidth, h = ActualHeight;
        if (w < 20 || h < 20)
        {
            return;
        }
        // Hit target: the whole plot, so hovering anywhere reads the nearest point.
        dc.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, w, h));

        var ink = TryFindResource("TextFillColorSecondaryBrush") as Brush ?? Brushes.Gray;
        var grid = new Pen(TryFindResource("CardStrokeColorDefaultBrush") as Brush ?? Brushes.LightGray, 1);
        double dpi = VisualTreeHelper.GetDpi(this).PixelsPerDip;
        const double top = 14, bottom = 4;
        double plotH = h - top - bottom;

        double max = _points.Count == 0 ? 1 : Math.Max(_points.Max(p => p.Value), 1e-9);
        max = NiceCeiling(max);
        // Recessive gridlines at 0, half and the top, labelled once at the top.
        foreach (double f in new[] { 0.0, 0.5, 1.0 })
        {
            double y = top + plotH * (1 - f);
            dc.DrawLine(grid, new Point(0, Math.Round(y) + 0.5), new Point(w, Math.Round(y) + 0.5));
        }
        dc.DrawText(Text(FormatValue(max), 10.5, ink, dpi), new Point(2, 0));

        if (_points.Count < 2)
        {
            dc.DrawText(Text("Waiting for data…", 11, ink, dpi), new Point(w / 2 - 45, top + plotH / 2 - 8));
            return;
        }
        var end = DateTime.Now;
        var start = end - Window;
        Point At((DateTime At, double Value) p) => new(
            Math.Clamp((p.At - start).TotalSeconds / Window.TotalSeconds, 0, 1) * w,
            top + plotH * (1 - p.Value / max));

        var visible = _points.Where(p => p.At >= start - TimeSpan.FromSeconds(5)).ToList();
        var line = new StreamGeometry();
        var area = new StreamGeometry();
        using (var g = line.Open())
        {
            g.BeginFigure(At(visible[0]), false, false);
            g.PolyLineTo(visible.Skip(1).Select(At).ToList(), true, true);
        }
        using (var g = area.Open())
        {
            g.BeginFigure(new Point(At(visible[0]).X, top + plotH), true, true);
            g.PolyLineTo(visible.Select(At).Append(new Point(At(visible[^1]).X, top + plotH)).ToList(), false, true);
        }
        var fill = LineBrush.Clone();
        fill.Opacity = 0.12;
        dc.DrawGeometry(fill, null, area);
        dc.DrawGeometry(null, new Pen(LineBrush, 2) { LineJoin = PenLineJoin.Round }, line);

        if (_hover is { } mouse)
        {
            var nearest = visible.MinBy(p => Math.Abs(At(p).X - mouse.X));
            var pt = At(nearest);
            dc.DrawLine(new Pen(ink, 1) { DashStyle = DashStyles.Dot }, new Point(pt.X, top), new Point(pt.X, top + plotH));
            dc.DrawEllipse(TryFindResource("CardBackgroundFillColorDefaultBrush") as Brush ?? Brushes.White, new Pen(LineBrush, 2), pt, 4, 4);
            var label = Text($"{FormatValue(nearest.Value)} · {nearest.At:HH:mm:ss}", 11,
                TryFindResource("TextFillColorPrimaryBrush") as Brush ?? Brushes.Black, dpi);
            double x = Math.Clamp(pt.X + 8, 0, w - label.Width - 4);
            var bg = TryFindResource("SolidBackgroundFillColorBaseBrush") as Brush ?? Brushes.White;
            dc.DrawRoundedRectangle(bg, grid, new Rect(x - 4, top, label.Width + 8, label.Height + 2), 3, 3);
            dc.DrawText(label, new Point(x, top + 1));
        }
    }

    private static double NiceCeiling(double v)
    {
        double mag = Math.Pow(10, Math.Floor(Math.Log10(v)));
        foreach (double step in new[] { 1, 2, 2.5, 5, 10 })
        {
            if (step * mag >= v)
            {
                return step * mag;
            }
        }
        return 10 * mag;
    }

    private static FormattedText Text(string s, double size, Brush brush, double dpi) =>
        new(s, CultureInfo.CurrentCulture, FlowDirection.LeftToRight, new Typeface("Segoe UI"), size, brush, dpi);
}
