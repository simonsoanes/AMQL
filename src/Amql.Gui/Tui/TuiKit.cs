// TextView is marked obsolete in favour of a separate editor package; it is still the text area Terminal.Gui ships.
#pragma warning disable CS0618
using System.IO;
using System.Collections.ObjectModel;
using System.Text;
using Terminal.Gui.App;
using Terminal.Gui.Drawing;
using Terminal.Gui.Input;
using Terminal.Gui.ViewBase;
using Terminal.Gui.Views;
using Attribute = Terminal.Gui.Drawing.Attribute;

namespace Amql.Gui.Tui;

/// <summary>
/// The small vocabulary the text UI's screens are written in: message boxes,
/// prompts, a path picker, a chooser, a text viewer, copying to the
/// terminal's clipboard, and text charts. Everything a WPF page does with a
/// window of its own (a file dialog, a context menu, a tooltip) has its
/// equivalent here.
/// </summary>
internal static class TuiKit
{
    private static IApplication? _app;

    public static IApplication App => _app ?? throw new InvalidOperationException("the text UI has not started");

    public static void Attach(IApplication app)
    {
        _app = app;
        _letters.Clear();
        // Printable keys reach a list or tree only through its own type-to-find, not its
        // KeyDown event, so single-letter shortcuts are caught at the application.
        app.Keyboard.KeyDown += (_, key) =>
        {
            if (key.Handled || key.IsCtrl || key.IsAlt || !key.TryGetPrintableRune(out var rune))
            {
                return;
            }
            var focused = app.Navigation?.GetFocused();
            foreach (var (view, letter, action) in _letters)
            {
                if (ReferenceEquals(view, focused) && char.ToLowerInvariant((char)rune.Value) == letter)
                {
                    key.Handled = true;
                    Guard(action);
                    return;
                }
            }
        };
    }

    private static readonly List<(View View, char Letter, Action Action)> _letters = new();

    /// <summary>A one-letter shortcut that works while <paramref name="view"/> has focus (e on the model list explores it).</summary>
    public static void OnLetter(View view, char letter, Action action) => _letters.Add((view, char.ToLowerInvariant(letter), action));

    // ── colours ─────────────────────────────────────────────────────────────

    public static readonly Color Good = new(0x22, 0xC5, 0x5E);
    public static readonly Color Warn = new(0xF5, 0x9E, 0x0B);
    public static readonly Color Bad = new(0xEF, 0x44, 0x44);
    public static readonly Color Idle = new(0x8A, 0x8F, 0x98);
    public static readonly Color Accent = new(0x3B, 0x82, 0xF6);
    public static readonly Color Violet = new(0x8B, 0x5C, 0xF6);

    public static Color StateColor(Services.ServerState state, bool answering = true) => state switch
    {
        Services.ServerState.Running or Services.ServerState.External when answering => Good,
        Services.ServerState.Running or Services.ServerState.External or Services.ServerState.Starting or Services.ServerState.Stopping => Warn,
        Services.ServerState.Failed => Bad,
        _ => Idle,
    };

    // ── dialogs ─────────────────────────────────────────────────────────────

    public static void Info(string title, string message) =>
        MessageBox.Query(App, title, message, "_OK");

    public static void Error(string title, string message) =>
        MessageBox.ErrorQuery(App, title, message, "_OK");

    /// <summary>A yes/no question; true for yes.</summary>
    public static bool Confirm(string title, string message, string yes = "_Yes", string no = "_No") =>
        MessageBox.Query(App, title, message, no, yes) == 1;

    /// <summary>Asks for one line (or, with <paramref name="multiline"/>, several) of text. Null when cancelled.</summary>
    public static string? Ask(string title, string label, string initial, bool secret = false, bool multiline = false, string? help = null)
    {
        var dialog = new Dialog { Title = title, Width = Dim.Percent(80), Height = multiline ? Dim.Percent(70) : Dim.Auto() };
        var caption = new Label { Text = label, X = 1, Y = 0 };
        dialog.Add(caption);
        View editor;
        Func<string> read;
        bool accepted = false;
        if (multiline)
        {
            caption.Text = label + "   (Enter adds a line · Ctrl+S or Tab then OK accepts)";
            var tv = new TextView { X = 1, Y = 1, Width = Dim.Fill(1), Height = Dim.Fill(help is null ? 1 : 3), Text = initial, WordWrap = false, TabKeyAddsTab = false };
            tv.KeyDown += (_, key) =>
            {
                if (key == Key.S.WithCtrl)
                {
                    key.Handled = true;
                    accepted = true;
                    dialog.RequestStop();
                }
            };
            editor = tv;
            read = () => tv.Text;
        }
        else
        {
            var tf = new TextField { X = 1, Y = 1, Width = Dim.Fill(1), Text = initial, Secret = secret };
            editor = tf;
            read = () => tf.Text;
        }
        dialog.Add(editor);
        if (help is not null)
        {
            dialog.Add(new Label { Text = help, X = 1, Y = multiline ? Pos.AnchorEnd(2) : 3, Width = Dim.Fill(1), Height = 2 });
        }
        dialog.AddButton(new Button { ShadowStyle = ShadowStyles.None, Text = "_Cancel" });
        dialog.AddButton(new Button { ShadowStyle = ShadowStyles.None, Text = "_OK" });
        editor.SetFocus();
        App.Run(dialog);
        string value = read();
        bool ok = accepted || dialog.Result == 1;
        dialog.Dispose();
        return ok ? value : null;
    }

    /// <summary>Chooses one of <paramref name="items"/>; the index, or null when cancelled.</summary>
    public static int? Choose(string title, IReadOnlyList<string> items, int selected = 0, string? help = null)
    {
        if (items.Count == 0)
        {
            return null;
        }
        int width = Math.Clamp(Math.Max(title.Length + 6, items.Max(i => i.Length) + 6), 30, 110);
        int height = Math.Clamp(items.Count + (help is null ? 4 : 6), 6, 30);
        var dialog = new Dialog { Title = title, Width = width, Height = height };
        var list = new ListView { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill(help is null ? 0 : 2) };
        list.SetSource(new ObservableCollection<string>(items));
        list.SelectedItem = Math.Clamp(selected, 0, items.Count - 1);
        int? chosen = null;
        list.Accepting += (_, e) =>
        {
            chosen = list.SelectedItem;
            e.Handled = true;
            dialog.RequestStop();
        };
        dialog.Add(list);
        if (help is not null)
        {
            dialog.Add(new Label { Text = help, X = 0, Y = Pos.AnchorEnd(2), Width = Dim.Fill(), Height = 2 });
        }
        list.SetFocus();
        App.Run(dialog);
        dialog.Dispose();
        return chosen;
    }

    /// <summary>A long text to read, scroll, copy or save.</summary>
    public static void ShowText(string title, string text, bool wrap = false)
    {
        var dialog = new Dialog { Title = title, Width = Dim.Percent(90), Height = Dim.Percent(85) };
        var view = new TextView { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill(), ReadOnly = true, Text = text, WordWrap = wrap };
        dialog.Add(view);
        var copy = new Button { ShadowStyle = ShadowStyles.None, Text = "Cop_y" };
        copy.Accepting += (_, e) =>
        {
            e.Handled = true;
            Copy(view.SelectedLength > 0 ? view.SelectedText : text);
        };
        var save = new Button { ShadowStyle = ShadowStyles.None, Text = "_Save…" };
        save.Accepting += (_, e) =>
        {
            e.Handled = true;
            if (PickSavePath("Save as", "") is { } path)
            {
                try
                {
                    File.WriteAllText(path, text);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    Error("Save", ex.Message);
                }
            }
        };
        dialog.AddButton(copy);
        dialog.AddButton(save);
        dialog.AddButton(new Button { ShadowStyle = ShadowStyles.None, Text = "_Close" });
        view.SetFocus();
        App.Run(dialog);
        dialog.Dispose();
    }

    /// <summary>
    /// Asks for a folder or file. The path can be typed or pasted (the usual way
    /// on a server reached over SSH), or browsed for with the file dialog.
    /// </summary>
    public static string? PickPath(string title, bool directory, string initial = "", bool mustExist = true)
    {
        var dialog = new Dialog { Title = title, Width = Dim.Percent(80), Height = Dim.Auto() };
        dialog.Add(new Label { Text = directory ? "Folder" : "File", X = 1, Y = 0 });
        var field = new TextField { X = 1, Y = 1, Width = Dim.Fill(14), Text = initial };
        var browse = new Button { ShadowStyle = ShadowStyles.None, Text = "_Browse…", X = Pos.Right(field) + 1, Y = 1 };
        var note = new Label { X = 1, Y = 3, Width = Dim.Fill(1), Text = "Type or paste a path, or browse. ~ is your home folder." };
        browse.Accepting += (_, e) =>
        {
            e.Handled = true;
            var open = new OpenDialog
            {
                Title = title,
                OpenMode = directory ? OpenMode.Directory : OpenMode.File,
                MustExist = mustExist,
            };
            string start = Expand(field.Text);
            if (Directory.Exists(start))
            {
                open.Path = start.EndsWith(Path.DirectorySeparatorChar) ? start : start + Path.DirectorySeparatorChar;
            }
            else if (Path.GetDirectoryName(start) is { Length: > 0 } parent && Directory.Exists(parent))
            {
                open.Path = parent + Path.DirectorySeparatorChar;
            }
            App.Run(open);
            if (!open.Canceled && open.FilePaths.Count > 0)
            {
                field.Text = open.FilePaths[0];
            }
            else if (!open.Canceled && !string.IsNullOrWhiteSpace(open.Path))
            {
                field.Text = open.Path;
            }
            open.Dispose();
            field.SetFocus();
        };
        dialog.Add(field, browse, note);
        dialog.AddButton(new Button { ShadowStyle = ShadowStyles.None, Text = "_Cancel" });
        dialog.AddButton(new Button { ShadowStyle = ShadowStyles.None, Text = "_OK" });
        field.SetFocus();
        while (true)
        {
            App.Run(dialog);
            if (dialog.Result != 1)
            {
                dialog.Dispose();
                return null;
            }
            string path = Expand(field.Text.Trim());
            if (path.Length == 0)
            {
                continue;
            }
            if (mustExist && (directory ? !Directory.Exists(path) : !File.Exists(path)))
            {
                Error(title, $"{(directory ? "No such folder" : "No such file")}:\n{path}");
                continue;
            }
            dialog.Dispose();
            return Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar) is { Length: > 0 } full ? full : path;
        }
    }

    public static string? PickSavePath(string title, string initial) => PickPath(title, directory: false, initial, mustExist: false);

    /// <summary>Expands a leading ~ to the home folder.</summary>
    public static string Expand(string path)
    {
        if (path == "~" || path.StartsWith("~/", StringComparison.Ordinal))
        {
            return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), path.Length > 2 ? path[2..] : "");
        }
        return path;
    }

    // ── clipboard ───────────────────────────────────────────────────────────

    public static event Action<string>? Notice;

    /// <summary>A short message for the status line.</summary>
    public static void Say(string message) => Notice?.Invoke(message);

    /// <summary>
    /// Copies text. On a desktop the system clipboard is used when there is one;
    /// it is also sent as an OSC 52 sequence, which most terminals (Windows
    /// Terminal, iTerm2, kitty, WezTerm, tmux with set-clipboard) put on the
    /// clipboard of the machine the terminal runs on — so copying works over SSH.
    /// </summary>
    public static void Copy(string text)
    {
        bool done = false;
        try
        {
            if (App.Clipboard is { IsSupported: true } clipboard)
            {
                done = clipboard.TrySetClipboardData(text);
            }
        }
        catch (Exception)
        {
            // no system clipboard here
        }
        try
        {
            string b64 = Convert.ToBase64String(Encoding.UTF8.GetBytes(text));
            App.Driver?.WriteRaw($"\u001b]52;c;{b64}\u0007");
            done = true;
        }
        catch (Exception)
        {
            // the driver cannot write raw sequences
        }
        Say(done ? $"copied: {Shorten(text.ReplaceLineEndings(" "), 90)}" : "no clipboard available");
    }

    // ── text charts ─────────────────────────────────────────────────────────

    private const string Spark = "▁▂▃▄▅▆▇█";

    /// <summary>One block glyph per value, scaled to <paramref name="max"/> (or the largest value).</summary>
    public static string Sparkline(IEnumerable<double> values, int width, double? max = null)
    {
        var list = values.ToList();
        if (list.Count > width)
        {
            list = list.Skip(list.Count - width).ToList();
        }
        double top = max ?? (list.Count == 0 ? 0 : list.Max());
        var sb = new StringBuilder(width);
        sb.Append(' ', Math.Max(0, width - list.Count));
        foreach (double v in list)
        {
            int level = top <= 0 ? 0 : (int)Math.Clamp(v / top * (Spark.Length - 1), 0, Spark.Length - 1);
            sb.Append(Spark[level]);
        }
        return sb.ToString();
    }

    /// <summary>A horizontal bar of <paramref name="width"/> cells filled to <paramref name="fraction"/>.</summary>
    public static string Bar(double fraction, int width)
    {
        fraction = Math.Clamp(double.IsFinite(fraction) ? fraction : 0, 0, 1);
        int eighths = (int)Math.Round(fraction * width * 8);
        var sb = new StringBuilder(width);
        sb.Append('█', eighths / 8);
        if (eighths % 8 > 0 && sb.Length < width)
        {
            sb.Append("▏▎▍▌▋▊▉"[eighths % 8 - 1]);
        }
        return sb.ToString().PadRight(width, '░');
    }

    public static string Shorten(string text, int max) => text.Length <= max ? text : text[..Math.Max(0, max - 1)] + "…";

    // ── building blocks ─────────────────────────────────────────────────────

    /// <summary>A button that runs <paramref name="action"/> (and does not let the press
    /// reach the dialog or window behind it).</summary>
    public static Button Button(string text, Action action, Pos? x = null, Pos? y = null)
    {
        var b = new Button { ShadowStyle = ShadowStyles.None, Text = text };
        if (x is not null) b.X = x;
        if (y is not null) b.Y = y;
        b.Accepting += (_, e) =>
        {
            e.Handled = true;
            Guard(action);
        };
        return b;
    }

    /// <summary>Runs a UI action, showing any exception instead of tearing the UI down.</summary>
    public static void Guard(Action action)
    {
        try
        {
            action();
        }
        catch (Exception ex)
        {
            Error("AMQL Studio", ex.Message);
        }
    }

    /// <summary>Runs an async UI action, showing any exception.</summary>
    public static async void GuardAsync(Func<Task> action)
    {
        try
        {
            await action();
        }
        catch (Exception ex)
        {
            Error("AMQL Studio", ex.Message);
        }
    }

    /// <summary>
    /// Ctrl+PgDn / Ctrl+PgUp: the next or previous tab of the tab strip that holds
    /// the focus (or the page's first one), with the keyboard moved into it.
    /// </summary>
    public static void CycleTabs(View page, int direction)
    {
        var all = new List<Tabs>();
        void Find(View v)
        {
            foreach (var sub in v.SubViews)
            {
                if (sub is Tabs t)
                {
                    all.Add(t);
                }
                Find(sub);
            }
        }
        Find(page);
        if (all.Count == 0)
        {
            return;
        }
        var focused = App.Navigation?.GetFocused();
        var tabs = all.FirstOrDefault(t => focused is not null && IsWithin(focused, t)) ?? all[0];
        var pages = tabs.TabCollection.ToList();
        if (pages.Count == 0)
        {
            return;
        }
        int index = tabs.Value is { } current ? pages.IndexOf(current) : 0;
        var next = pages[((index + direction) % pages.Count + pages.Count) % pages.Count];
        tabs.Value = next;
        next.SetFocus();
    }

    private static bool IsWithin(View view, View ancestor)
    {
        for (var v = view; v is not null; v = v.SuperView)
        {
            if (ReferenceEquals(v, ancestor))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>A label whose text is drawn in one colour.</summary>
    public static Label Colored(Color color, string text = "") => new ColorLabel(color) { Text = text };

    /// <summary>Sets a read-only text view's content and keeps the end in view.</summary>
    public static void SetLog(TextView view, string text)
    {
        view.Text = text;
        view.MoveEnd();
    }

    public static Attribute Attr(View view, Color fg) =>
        new(fg, view.GetAttributeForRole(VisualRole.Normal).Background);
}

/// <summary>A label drawn in a fixed colour (a status dot, a badge).</summary>
internal sealed class ColorLabel : Label
{
    public ColorLabel(Color color) => Color = color;

    public Color Color { get; set; }

    protected override bool OnDrawingText(DrawContext? context)
    {
        SetAttribute(new Attribute(Color, GetAttributeForRole(VisualRole.Normal).Background));
        int row = 0;
        foreach (var line in (Text ?? "").Split('\n'))
        {
            if (row >= Viewport.Height)
            {
                break;
            }
            AddStr(0, row++, TuiKit.Shorten(line, Math.Max(0, Viewport.Width)));
        }
        return true;
    }
}

/// <summary>
/// The text UI's label. Terminal.Gui's label treats the first underscore as a
/// hot-key marker and hides it, which mangles the data labels show (op names
/// like linear_qkv, paths, model ids); this one shows text as it is. Declared
/// in this namespace so every <c>new Label</c> here is this class.
/// </summary>
internal class Label : Terminal.Gui.Views.Label
{
    public Label() => HotKeySpecifier = new System.Text.Rune(0xFFFF);
}

/// <summary>The text UI's dialog: its title shows underscores (in tensor and object ids) rather than taking a hot key.</summary>
internal class Dialog : Terminal.Gui.Views.Dialog
{
    public Dialog() => HotKeySpecifier = new System.Text.Rune(0xFFFF);
}
