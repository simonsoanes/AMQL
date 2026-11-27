using System.IO;
using System.Collections.ObjectModel;
using System.Globalization;
using Terminal.Gui.Input;
using Terminal.Gui.ViewBase;
using Terminal.Gui.Views;

namespace Amql.Gui.Tui;

/// <summary>One row of a <see cref="FieldList"/>: a setting with a label, a value and a way to change it.</summary>
internal abstract class Field
{
    private readonly string? _help;

    protected Field(string label, string? help)
    {
        Label = label;
        _help = help;
    }

    public string Label { get; set; }

    /// <summary>Help that depends on the current values (it is read each time the row is selected).</summary>
    public Func<string?>? HelpText { get; init; }

    public string? Help => HelpText?.Invoke() ?? _help;

    /// <summary>When false the row shows its value but cannot be changed (a limit that is off, say).</summary>
    public Func<bool> IsEnabled { get; init; } = () => true;

    /// <summary>When false the row is left out (an option that only applies to chat models, say).</summary>
    public Func<bool> IsVisible { get; init; } = () => true;

    public virtual bool IsHeader => false;

    public abstract string Display();

    /// <summary>Enter: change the value. Returns true when it changed.</summary>
    public abstract bool Edit();

    /// <summary>Space: a quick change (flip a switch, step a choice). Defaults to <see cref="Edit"/>.</summary>
    public virtual bool Toggle() => Edit();
}

internal sealed class HeaderField : Field
{
    public HeaderField(string label, string? help = null) : base(label, help)
    {
    }

    public override bool IsHeader => true;

    public override string Display() => "";

    public override bool Edit() => false;
}

/// <summary>Read-only information (a computed summary).</summary>
internal sealed class InfoField : Field
{
    private readonly Func<string> _value;

    public InfoField(string label, Func<string> value, string? help = null) : base(label, help) => _value = value;

    public override string Display() => _value();

    public override bool Edit() => false;
}

internal sealed class TextValueField : Field
{
    private readonly Func<string> _get;
    private readonly Action<string> _set;

    public TextValueField(string label, Func<string> get, Action<string> set, string? help = null) : base(label, help)
    {
        _get = get;
        _set = set;
    }

    public bool Secret { get; init; }

    public bool Multiline { get; init; }

    /// <summary>What an empty value means, shown in its place ("server default").</summary>
    public string? Placeholder { get; init; }

    /// <summary>Checks a typed value: an error message, or null when it is fine.</summary>
    public Func<string, string?>? Validate { get; init; }

    public override string Display()
    {
        string v = _get();
        if (v.Length == 0)
        {
            return Placeholder is null ? "" : $"({Placeholder})";
        }
        if (Secret)
        {
            return new string('•', Math.Min(v.Length, 12));
        }
        return Multiline ? v.ReplaceLineEndings(" ⏎ ") : v;
    }

    public override bool Edit()
    {
        string current = _get();
        while (true)
        {
            var typed = TuiKit.Ask(Label, Multiline ? "One entry per line" : Label, current, Secret, Multiline, Help);
            if (typed is null)
            {
                return false;
            }
            if (!Multiline)
            {
                typed = typed.Trim();
            }
            if (Validate?.Invoke(typed) is { } problem)
            {
                TuiKit.Error(Label, problem);
                current = typed;
                continue;
            }
            if (typed == _get())
            {
                return false;
            }
            _set(typed);
            return true;
        }
    }

    // Validators for the numeric boxes the WPF forms parse leniently.
    public static string? OptionalInt(string s) =>
        s.Length == 0 || int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out _) ? null : "Enter a whole number, or leave it empty.";

    public static string? OptionalNumber(string s) =>
        s.Length == 0 || double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out _) ? null : "Enter a number (such as 0.7), or leave it empty.";
}

internal sealed class BoolField : Field
{
    private readonly Func<bool> _get;
    private readonly Action<bool> _set;

    public BoolField(string label, Func<bool> get, Action<bool> set, string? help = null) : base(label, help)
    {
        _get = get;
        _set = set;
    }

    public override string Display() => _get() ? "[x]" : "[ ]";

    public override bool Edit()
    {
        _set(!_get());
        return true;
    }
}

internal sealed class ChoiceField<T> : Field
{
    private readonly (string Label, T Value)[] _choices;
    private readonly Func<T> _get;
    private readonly Action<T> _set;

    public ChoiceField(string label, (string Label, T Value)[] choices, Func<T> get, Action<T> set, string? help = null) : base(label, help)
    {
        _choices = choices;
        _get = get;
        _set = set;
    }

    private int Current()
    {
        var value = _get();
        int i = Array.FindIndex(_choices, c => EqualityComparer<T>.Default.Equals(c.Value, value));
        return i;
    }

    public override string Display() => Current() is >= 0 and var i ? _choices[i].Label + "  ▾" : $"{_get()}  ▾";

    public override bool Edit()
    {
        int i = Current();
        if (TuiKit.Choose(Label, _choices.Select(c => c.Label).ToList(), Math.Max(0, i), Help) is not { } chosen || chosen == i)
        {
            return false;
        }
        _set(_choices[chosen].Value);
        return true;
    }

    public override bool Toggle()
    {
        if (_choices.Length == 0)
        {
            return false;
        }
        _set(_choices[(Current() + 1) % _choices.Length].Value);
        return true;
    }
}

internal sealed class PathField : Field
{
    private readonly Func<string> _get;
    private readonly Action<string> _set;
    private readonly bool _directory;

    public PathField(string label, bool directory, Func<string> get, Action<string> set, string? help = null) : base(label, help)
    {
        _directory = directory;
        _get = get;
        _set = set;
    }

    public bool MustExist { get; init; }

    public override string Display() => _get() is { Length: > 0 } v ? v : "(none)";

    public override bool Edit()
    {
        if (TuiKit.PickPath(Label, _directory, _get(), MustExist) is not { } path || path == _get())
        {
            return false;
        }
        _set(path);
        return true;
    }

    /// <summary>Space clears the path.</summary>
    public override bool Toggle()
    {
        if (_get().Length == 0)
        {
            return Edit();
        }
        _set("");
        return true;
    }
}

/// <summary>A row that does something (add a folder, open the options of a model).</summary>
internal sealed class ActionField : Field
{
    private readonly Action _action;
    private readonly Func<string>? _display;

    public ActionField(string label, Action action, string? help = null, Func<string>? display = null) : base(label, help)
    {
        _action = action;
        _display = display;
    }

    public override string Display() => _display?.Invoke() ?? "▸";

    public override bool Edit()
    {
        _action();
        return true;
    }
}

/// <summary>
/// A settings form as a list: one row per setting, the label on the left and
/// the value on the right. Enter changes the selected row (a prompt, a choice,
/// a path picker), Space flips a switch or steps a choice, and the selected
/// row's help is shown underneath. It stands in for the WPF pages' grids of
/// text boxes, check boxes and combo boxes: it scrolls to any length and works
/// with nothing but the keyboard.
/// </summary>
internal sealed class FieldList : View
{
    private readonly ListView _list;
    private readonly Label _help;
    private readonly ObservableCollection<string> _rows = new();
    private List<Field> _visible = new();
    private List<Field> _fields = new();
    private readonly int _helpLines;

    public FieldList(int helpLines = 2)
    {
        _helpLines = helpLines;
        CanFocus = true;
        _list = new ListView { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill(helpLines > 0 ? helpLines + 1 : 0) };
        _list.SetSource(_rows);
        _help = new Label { X = 0, Y = Pos.AnchorEnd(helpLines), Width = Dim.Fill(), Height = helpLines, Visible = helpLines > 0 };
        Add(_list, _help);
        if (helpLines > 0)
        {
            Add(new Line { X = 0, Y = Pos.AnchorEnd(helpLines + 1), Width = Dim.Fill() });
        }
        _list.Accepting += (_, e) =>
        {
            e.Handled = true;
            Act(f => f.Edit());
        };
        _list.KeyDown += (_, key) =>
        {
            if (key == Key.Space)
            {
                key.Handled = true;
                Act(f => f.Toggle());
            }
        };
        _list.ValueChanged += (_, _) => ShowHelp();
        Width = Dim.Fill();
        Height = Dim.Fill();
    }

    /// <summary>Raised after a row changed its value.</summary>
    public event Action<Field>? Changed;

    /// <summary>The width of the label column (wider labels are cut).</summary>
    public int LabelWidth { get; set; } = 30;

    public Field? Selected => _list.SelectedItem is { } i && i >= 0 && i < _visible.Count ? _visible[i] : null;

    public void SetFields(IEnumerable<Field> fields)
    {
        _fields = fields.ToList();
        Refresh();
        int first = _visible.FindIndex(f => !f.IsHeader);
        if (first >= 0)
        {
            _list.SelectedItem = first;
        }
        ShowHelp();
    }

    /// <summary>Redraws every row from the values it shows (after something else changed them).</summary>
    public void Refresh()
    {
        int? keep = _list.SelectedItem;
        var keepField = Selected;
        _visible = _fields.Where(f => f.IsVisible()).ToList();
        _rows.Clear();
        int width = Math.Min(LabelWidth, Math.Max(10, _visible.Where(f => !f.IsHeader).Select(f => f.Label.Length).DefaultIfEmpty(10).Max() + 2));
        foreach (var f in _visible)
        {
            if (f.IsHeader)
            {
                _rows.Add($"── {f.Label} " + new string('─', Math.Max(2, width + 20 - f.Label.Length)));
                continue;
            }
            string label = TuiKit.Shorten(f.Label, width - 1).PadRight(width);
            string value = f.Display();
            _rows.Add("  " + label + (f.IsEnabled() ? value : "· " + value));
        }
        int index = keepField is not null ? _visible.IndexOf(keepField) : -1;
        if (index < 0)
        {
            index = keep is { } k ? Math.Min(k, _rows.Count - 1) : 0;
        }
        if (_rows.Count > 0)
        {
            _list.SelectedItem = Math.Max(0, index);
        }
        ShowHelp();
        SetNeedsDraw();
    }

    private void Act(Func<Field, bool> change)
    {
        if (Selected is not { IsHeader: false } field)
        {
            return;
        }
        if (!field.IsEnabled())
        {
            TuiKit.Say($"{field.Label} cannot be changed now: {field.Help ?? "it depends on another setting"}");
            return;
        }
        bool changed = false;
        TuiKit.Guard(() => changed = change(field));
        if (changed)
        {
            Changed?.Invoke(field);
        }
        Refresh();
    }

    private void ShowHelp()
    {
        if (_helpLines > 0)
        {
            _help.Text = Selected?.Help ?? "";
        }
    }

    /// <summary>Focuses the list (rather than the help line).</summary>
    public void FocusList() => _list.SetFocus();
}
