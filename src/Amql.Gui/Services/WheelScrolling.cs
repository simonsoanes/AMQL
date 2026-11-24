using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace Amql.Gui.Services;

/// <summary>
/// Makes the mouse wheel scroll a page even when the pointer is over a text
/// box, combo box or slider inside it. WPF lets those swallow the wheel (a
/// single-line text box marks it handled although it cannot scroll), which
/// leaves a long form stuck. A box that can scroll itself — a multi-line
/// output with more lines than it shows — keeps the wheel.
/// </summary>
public static class WheelScrolling
{
    public static void Attach(ScrollViewer page)
    {
        page.PreviewMouseWheel += (_, e) =>
        {
            for (var element = e.OriginalSource as DependencyObject; element is not null && element != page; element = Parent(element))
            {
                if (element is ScrollViewer inner && inner.ScrollableHeight > 0)
                {
                    bool atEnd = e.Delta < 0 ? inner.VerticalOffset >= inner.ScrollableHeight : inner.VerticalOffset <= 0;
                    if (!atEnd)
                    {
                        return;     // the inner box scrolls first
                    }
                }
                if (element is ComboBox { IsDropDownOpen: true })
                {
                    return;
                }
            }
            page.ScrollToVerticalOffset(page.VerticalOffset - e.Delta);
            e.Handled = true;
        };
    }

    private static DependencyObject? Parent(DependencyObject d) =>
        d is Visual or System.Windows.Media.Media3D.Visual3D ? VisualTreeHelper.GetParent(d) : LogicalTreeHelper.GetParent(d);
}
