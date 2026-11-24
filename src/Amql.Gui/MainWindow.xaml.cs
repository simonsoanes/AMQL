using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Amql.Gui.Pages;
using Amql.Gui.Services;
using Amql.Gui.Windows;

namespace Amql.Gui;

/// <summary>
/// The shell: a navigation rail on the left (with the server's state always
/// in view), and the selected page on the right. Pages are built on first
/// visit and kept, so a half-typed chat or command form survives switching.
/// </summary>
public partial class MainWindow : Window
{
    private sealed record NavEntry(string Key, string Glyph, string Title, Func<FrameworkElement> Create);

    private readonly List<NavEntry> _entries;
    private readonly Dictionary<string, FrameworkElement> _pages = new();
    private ServerMonitorWindow? _monitor;

    public MainWindow()
    {
        InitializeComponent();
        _entries = new()
        {
            new("home", "", "Home", () => new HomePage(this)),
            new("server", "", "Server", () => new ServerPage(this)),
            new("chat", "", "Playground", () => new PlaygroundPage(this)),
            new("tools", "", "Tools", () => new ToolsPage()),
            new("explorer", "", "Explorer", () => new ExplorerPage(this)),
        };
        foreach (var entry in _entries)
        {
            var row = new StackPanel { Orientation = Orientation.Horizontal };
            row.Children.Add(new TextBlock { Text = entry.Glyph, Style = (Style)FindResource("Icon"), FontSize = 16, Margin = new Thickness(0, 0, 12, 0) });
            row.Children.Add(new TextBlock { Text = entry.Title, FontSize = 14, VerticalAlignment = VerticalAlignment.Center });
            NavList.Items.Add(new ListBoxItem { Content = row, Tag = entry });
        }

        ApplySavedBounds();
        ServerController.Instance.PropertyChanged += OnServerChanged;
        RefreshServerState();
        Navigate(AppSettings.Current.LastPage ?? "home");
        Loaded += async (_, _) => await ServerController.Instance.DetectExternalAsync(AppSettings.Current.Server);
    }

    /// <summary>Puts the window where it was last time, if that is still on a screen.</summary>
    private void ApplySavedBounds()
    {
        if (AppSettings.Current.WindowBounds is [var left, var top, var width, var height]
            && width >= MinWidth && height >= MinHeight
            && left >= SystemParameters.VirtualScreenLeft && top >= SystemParameters.VirtualScreenTop
            && left + 100 <= SystemParameters.VirtualScreenLeft + SystemParameters.VirtualScreenWidth
            && top + 100 <= SystemParameters.VirtualScreenTop + SystemParameters.VirtualScreenHeight)
        {
            WindowStartupLocation = WindowStartupLocation.Manual;
            Left = left;
            Top = top;
            Width = width;
            Height = height;
        }
        if (AppSettings.Current.WindowMaximized)
        {
            WindowState = WindowState.Maximized;
        }
    }

    public ToolsPage Tools => (ToolsPage)Page("tools");

    /// <summary>Shows a page by key (home, server, chat, tools, explorer).</summary>
    public FrameworkElement Navigate(string key)
    {
        var item = NavList.Items.Cast<ListBoxItem>().FirstOrDefault(i => ((NavEntry)i.Tag).Key == key)
                   ?? (ListBoxItem)NavList.Items[0];
        NavList.SelectedItem = item;
        return Page(((NavEntry)item.Tag).Key);
    }

    /// <summary>Opens the Tools page on one command.</summary>
    public void OpenCommand(string command)
    {
        Navigate("tools");
        Tools.SelectCommandByName(command);
    }

    private FrameworkElement Page(string key)
    {
        if (!_pages.TryGetValue(key, out var page))
        {
            page = _entries.First(e => e.Key == key).Create();
            _pages[key] = page;
        }
        return page;
    }

    private void OnNavigate(object sender, SelectionChangedEventArgs e)
    {
        if (NavList.SelectedItem is ListBoxItem { Tag: NavEntry entry })
        {
            PageHost.Content = Page(entry.Key);
            AppSettings.Current.LastPage = entry.Key;
        }
    }

    // ── server ──────────────────────────────────────────────────────────────

    private void OnServerChanged(object? sender, PropertyChangedEventArgs e) => RefreshServerState();

    private void RefreshServerState()
    {
        var server = ServerController.Instance;
        ServerDetailText.Text = server.Detail;
        ServerDetailText.ToolTip = server.Detail;
        ServerDot.Fill = (Brush)FindResource(server.State switch
        {
            ServerState.Running or ServerState.External => "GoodBrush",
            ServerState.Starting or ServerState.Stopping => "WarnBrush",
            ServerState.Failed => "BadBrush",
            _ => "IdleBrush",
        });
        ServerToggleButton.Content = server.State switch
        {
            ServerState.Running or ServerState.Starting => "Stop",
            ServerState.Stopping => "Stopping…",
            ServerState.External => "Forget",
            _ => "Start",
        };
        ServerToggleButton.IsEnabled = server.State != ServerState.Stopping;
    }

    private async void OnToggleServer(object sender, RoutedEventArgs e) => await ToggleServerAsync();

    /// <summary>Starts the server with the chosen models, or stops it.</summary>
    public async Task ToggleServerAsync()
    {
        var server = ServerController.Instance;
        try
        {
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
                    MessageBox.Show(this, "Tick the models you want to load, then press Start.", "AMQL Studio",
                        MessageBoxButton.OK, MessageBoxImage.Information);
                    return;
                }
                AppSettings.Current.Save();
                ToolLocator.Refresh();      // pick up a build made since the app started
                await server.StartAsync(profile);
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "AMQL Studio", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void OnOpenMonitor(object sender, RoutedEventArgs e) => OpenMonitor();

    private void OnAbout(object sender, RoutedEventArgs e) => new Dialogs.AboutDialog { Owner = this }.ShowDialog();

    public void OpenMonitor()
    {
        if (_monitor is { IsLoaded: true })
        {
            _monitor.Activate();
            return;
        }
        // Not owned: the monitor is a window of its own that can sit on another screen or behind this one.
        _monitor = new ServerMonitorWindow();
        _monitor.Show();
    }

    protected override void OnClosing(CancelEventArgs e)
    {
        base.OnClosing(e);
        if (_pages.TryGetValue("tools", out var tools) && !((ToolsPage)tools).ConfirmClose())
        {
            e.Cancel = true;
            return;
        }
        var bounds = WindowState == WindowState.Normal ? new Rect(Left, Top, Width, Height) : RestoreBounds;
        AppSettings.Current.WindowBounds = new[] { bounds.Left, bounds.Top, bounds.Width, bounds.Height };
        AppSettings.Current.WindowMaximized = WindowState == WindowState.Maximized;
        AppSettings.Current.Save();
        _monitor?.Close();
        if (AppSettings.Current.Server.StopOnExit)
        {
            ServerController.Instance.StopNow();
        }
    }
}
