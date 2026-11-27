using System.IO;
using System.Collections.ObjectModel;
using System.ComponentModel;
using Amql.Gui.Services;
using Terminal.Gui.Drawing;
using Terminal.Gui.Drivers;
using Terminal.Gui.Input;
using Terminal.Gui.ViewBase;
using Terminal.Gui.Views;

namespace Amql.Gui.Tui;

/// <summary>A screen of the text UI.</summary>
internal interface ITuiPage
{
    View View { get; }

    /// <summary>The page became the one on screen.</summary>
    void Shown();

    /// <summary>Puts the keyboard on the page's main control (Alt+number, Enter in the page list, a jump from another page).</summary>
    void Focus() => View.SetFocus();

    /// <summary>Another page replaced it.</summary>
    void Hidden();
}

/// <summary>
/// The text UI's main window, laid out like the desktop one: a navigation
/// rail on the left with the server's state always in view, and the selected
/// page on the right. Pages are built on first visit and kept, so a half-typed
/// chat or command form survives switching.
/// </summary>
internal sealed class Shell : Window
{
    private sealed record NavEntry(string Key, string Title, Func<ITuiPage> Create);

    public static readonly string[] PageKeys = { "home", "server", "chat", "tools", "explorer", "monitor", "visualiser" };

    private readonly List<NavEntry> _entries;
    private readonly Dictionary<string, ITuiPage> _pages = new();
    private readonly ListView _nav;
    private readonly View _host;
    private readonly ColorLabel _dot;
    private readonly Label _serverText;
    private readonly Button _toggle;
    private readonly Label _notice;
    private readonly bool _snapshot;
    private ITuiPage? _current;
    private string _currentKey = "";
    private object? _noticeTimer;

    public Shell(string? startPage, string? tracePath, bool snapshot = false)
    {
        _snapshot = snapshot;
        // No automatic hot keys: they would make every letter press a button somewhere on the
        // page (e pressed "Explore", say) instead of reaching the list or tree that has focus.
        AssignHotKeys = false;
        Title = "AMQL Studio";
        BorderStyle = LineStyle.Rounded;
        _entries = new()
        {
            new("home", "Home", () => new HomePageTui(this)),
            new("server", "Server", () => new ServerPageTui(this)),
            new("chat", "Playground", () => new PlaygroundPageTui(this)),
            new("tools", "Tools", () => new ToolsPageTui(this)),
            new("explorer", "Explorer", () => new ExplorerPageTui(this)),
            new("monitor", "Monitor", () => new MonitorPageTui()),
            new("visualiser", "Visualiser", () => new VisualiserPageTui(this)),
        };

        // ── rail ────────────────────────────────────────────────────────────
        const int rail = 24;
        Add(new Label { Text = "Local inference and", X = 0, Y = 0, Width = rail });
        Add(new Label { Text = "model tools", X = 0, Y = 1, Width = rail });
        _nav = new ListView { X = 0, Y = 3, Width = rail, Height = _entries.Count };
        _nav.SetSource(new ObservableCollection<string>(_entries.Select((e, i) => $" {i + 1}  {e.Title}")));
        _nav.ValueChanged += (_, _) =>
        {
            if (_nav.SelectedItem is { } i && i >= 0 && i < _entries.Count)
            {
                ShowPage(_entries[i].Key);
            }
        };
        _nav.Accepting += (_, e) =>
        {
            e.Handled = true;
            _current?.Focus();
        };
        Add(_nav);

        var serverBox = new FrameView { Title = "Server", X = 0, Y = Pos.AnchorEnd(8), Width = rail, Height = 6 };
        _dot = new ColorLabel(TuiKit.Idle) { Text = "●", X = 0, Y = 0, Width = 1 };
        _serverText = new Label { X = 2, Y = 0, Width = Dim.Fill(), Height = 2 };
        _toggle = TuiKit.Button("Start", () => TuiKit.GuardAsync(ToggleServerAsync), 0, 2);
        var monitor = TuiKit.Button("Monitor", () => Navigate("monitor"), Pos.Right(_toggle) + 1, 2);
        serverBox.Add(_dot, _serverText, _toggle, monitor);
        Add(serverBox);

        Add(new Line { X = rail, Y = 0, Height = Dim.Fill(2), Orientation = Orientation.Vertical });

        // ── page host, notice line, status bar ──────────────────────────────
        _host = new View { X = rail + 1, Y = 0, Width = Dim.Fill(), Height = Dim.Fill(2), CanFocus = true };
        Add(_host);
        _notice = new Label { X = 0, Y = Pos.AnchorEnd(2), Width = Dim.Fill(), Height = 1 };
        Add(_notice);
        var status = new StatusBar(new[]
        {
            new Shortcut(Key.F1, "Help", ShowHelp, null),
            new Shortcut(Key.F2, "Start/stop server", () => TuiKit.GuardAsync(ToggleServerAsync), null),
            new Shortcut(Key.F5, "Run command", () => Tools.RunSelected(), null),
            new Shortcut(Key.F7, "Monitor", () => Navigate("monitor"), null),
            new Shortcut(Key.Q.WithCtrl, "Quit", Quit, null),
        })
        {
            Y = Pos.AnchorEnd(1),
        };
        Add(status);

        KeyDown += OnKeyDown;
        TuiKit.Notice += Say;
        ServerController.Instance.PropertyChanged += OnServerChanged;
        RefreshServer();

        string first = startPage is not null && PageKeys.Contains(startPage, StringComparer.OrdinalIgnoreCase)
            ? startPage.ToLowerInvariant()
            : tracePath is not null ? "visualiser"
            : AppSettings.Current.LastPage is { } last && PageKeys.Contains(last) ? last : "home";
        Navigate(first);
        if (tracePath is not null)
        {
            Visualiser.OpenTrace(tracePath);
        }
        if (!snapshot)
        {
            _ = ServerController.Instance.DetectExternalAsync(AppSettings.Current.Server);
        }
    }

    public bool IsSnapshot => _snapshot;

    public ToolsPageTui Tools => (ToolsPageTui)Page("tools");

    public ExplorerPageTui Explorer => (ExplorerPageTui)Page("explorer");

    public VisualiserPageTui Visualiser => (VisualiserPageTui)Page("visualiser");

    /// <summary>Shows a page by key (home, server, chat, tools, explorer, monitor, visualiser).</summary>
    public ITuiPage Navigate(string key)
    {
        int index = _entries.FindIndex(e => e.Key == key);
        if (index < 0)
        {
            index = 0;
        }
        if (_nav.SelectedItem != index)
        {
            _nav.SelectedItem = index;     // ValueChanged shows it
        }
        ShowPage(_entries[index].Key);
        if (!_snapshot)
        {
            Page(_entries[index].Key).Focus();
        }
        return Page(_entries[index].Key);
    }

    /// <summary>Opens the Tools page on one command.</summary>
    public void OpenCommand(string command)
    {
        Navigate("tools");
        Tools.SelectCommandByName(command);
    }

    private ITuiPage Page(string key)
    {
        if (!_pages.TryGetValue(key, out var page))
        {
            page = _entries.First(e => e.Key == key).Create();
            page.View.Width = Dim.Fill();
            page.View.Height = Dim.Fill();
            _pages[key] = page;
        }
        return page;
    }

    private void ShowPage(string key)
    {
        if (key == _currentKey && _current is not null)
        {
            return;
        }
        var page = Page(key);
        if (_current is not null)
        {
            _host.Remove(_current.View);
            _current.Hidden();
        }
        _current = page;
        _currentKey = key;
        _host.Add(page.View);
        page.Shown();
        AppSettings.Current.LastPage = key;
        SetNeedsLayout();
        SetNeedsDraw();
    }

    // ── keys ────────────────────────────────────────────────────────────────

    private void OnKeyDown(object? sender, Key key)
    {
        if (key.IsAlt && !key.IsCtrl)
        {
            var bare = key.NoAlt;
            for (int i = 0; i < _entries.Count; i++)
            {
                if (bare == new Key((KeyCode)('1' + i)))
                {
                    key.Handled = true;
                    Navigate(_entries[i].Key);
                    return;
                }
            }
        }
        if (key == Key.PageDown.WithCtrl || key == Key.PageUp.WithCtrl)
        {
            key.Handled = true;
            if (_current is not null)
            {
                TuiKit.CycleTabs(_current.View, key == Key.PageDown.WithCtrl ? 1 : -1);
            }
            return;
        }
        if (key == Key.Esc)
        {
            // Esc closes dialogs; on the main window it would quit, which is too easy to do by accident.
            key.Handled = true;
            _nav.SetFocus();
        }
    }

    private void ShowHelp()
    {
        TuiKit.ShowText("Help — AMQL Studio " + TuiApp.Version,
            """
            AMQL Studio — local inference and model tools: serve containers behind an
            OpenAI-compatible endpoint, watch what the server is doing, chat with models,
            and run every amql-cli command from a form.

            Pages
              Alt+1  Home        getting started, common jobs, this computer
              Alt+2  Server      models to load, memory and device, advanced, output
              Alt+3  Playground  chat with a model, or send any API request
              Alt+4  Tools       every amql-cli command as a form, runs, projects
              Alt+5  Explorer    look inside a container, view tensors
              Alt+6  Monitor     what the server is doing now (F7)
              Alt+7  Visualiser  inference traces: map, ranking, logit lens, attention

            Keys
              Tab / Shift+Tab     move between parts of a page
              Ctrl+PgUp/PgDn      change tab inside a page (or click the tab)
              Enter               change the selected setting, open, run
              Space               flip a switch, step through a choice, tick a model
              F2                  start or stop the server
              F5                  run the command on the Tools page
              Esc                 close a dialog; on a page, go back to the page list
              Ctrl+Q              quit

            Settings are shared with the desktop app: settings.json and projects live in
            """ + AppSettings.Folder + """


            Copying uses the terminal's clipboard (OSC 52), so it works over SSH in terminals
            that allow it (Windows Terminal, iTerm2, kitty, WezTerm, tmux with set-clipboard on).

            Copyright © 2026 Simon Soanes · MIT License · https://github.com/simonsoanes/AMQL
            """, wrap: true);
    }

    private void Quit()
    {
        if (_pages.TryGetValue("tools", out var tools) && !((ToolsPageTui)tools).ConfirmClose())
        {
            return;
        }
        App?.RequestStop();
    }

    /// <summary>After the main loop ends: save, and stop the server if asked to.</summary>
    public void Closing()
    {
        TuiKit.Notice -= Say;
        ServerController.Instance.PropertyChanged -= OnServerChanged;
        foreach (var page in _pages.Values.OfType<IDisposable>())
        {
            page.Dispose();
        }
        if (_snapshot)
        {
            return;
        }
        if (_pages.TryGetValue("tools", out var tools))
        {
            ((ToolsPageTui)tools).Session.Save();
        }
        AppSettings.Current.Save();
        if (AppSettings.Current.Server.StopOnExit)
        {
            ServerController.Instance.StopNow();
        }
    }

    // ── notices ─────────────────────────────────────────────────────────────

    public void Say(string message)
    {
        _notice.Text = " " + message;
        if (_noticeTimer is not null)
        {
            App?.RemoveTimeout(_noticeTimer);
        }
        _noticeTimer = App?.AddTimeout(TimeSpan.FromSeconds(8), () =>
        {
            _notice.Text = "";
            _noticeTimer = null;
            return false;
        });
    }

    // ── server ──────────────────────────────────────────────────────────────

    private void OnServerChanged(object? sender, PropertyChangedEventArgs e) => RefreshServer();

    private void RefreshServer()
    {
        var server = ServerController.Instance;
        _dot.Color = TuiKit.StateColor(server.State);
        _dot.SetNeedsDraw();
        _serverText.Text = server.Detail;
        _toggle.Text = server.State switch
        {
            ServerState.Running or ServerState.Starting => "Stop",
            ServerState.Stopping => "Stopping…",
            ServerState.External => "Forget",
            _ => "Start",
        };
        _toggle.Enabled = server.State != ServerState.Stopping;
    }

    /// <summary>Starts the server with the chosen models, or stops it.</summary>
    public async Task ToggleServerAsync()
    {
        var server = ServerController.Instance;
        if (server.State == ServerState.External)
        {
            server.Detach();
        }
        else if (server.CanStop)
        {
            await server.StopAsync();
        }
        else if (server.CanStart)
        {
            var profile = AppSettings.Current.Server;
            if (!profile.Models.Any(m => m.Enabled))
            {
                // Nothing chosen yet: the Server page is where models are picked.
                Navigate("server");
                TuiKit.Info("AMQL Studio", "Tick the models you want to load (Space), then press Start.");
                return;
            }
            AppSettings.Current.Save();
            ToolLocator.Refresh();      // pick up a build made since the app started
            await server.StartAsync(profile);
        }
    }
}
