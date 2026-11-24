using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using Amql.Gui.Services;

namespace Amql.Gui.Pages;

/// <summary>
/// Chat with a model on the running server (streamed, with the reply's speed
/// shown under it), and a raw request tab for the other endpoints:
/// embeddings, classify, decisions and responses.
/// </summary>
public partial class PlaygroundPage : UserControl
{
    private readonly MainWindow _shell;
    private readonly List<JsonObject> _messages = new();
    private List<(string Id, string[] Capabilities)> _models = new();
    private CancellationTokenSource? _cts;
    private bool _loading = true;

    private static PlaygroundSettings Settings => AppSettings.Current.Playground;

    public PlaygroundPage(MainWindow shell)
    {
        _shell = shell;
        InitializeComponent();
        ThinkCheck.IsChecked = Settings.Think;
        SystemBox.Text = Settings.SystemPrompt;
        TemperatureBox.Text = Settings.Temperature?.ToString(CultureInfo.InvariantCulture) ?? "";
        MaxTokensBox.Text = Settings.MaxTokens?.ToString(CultureInfo.InvariantCulture) ?? "";
        _loading = false;
        ServerController.Instance.PropertyChanged += OnServerChanged;
        IsVisibleChanged += async (_, _) =>
        {
            if (IsVisible)
            {
                await RefreshModelsAsync();
            }
        };
        ShowEmptyHint();
    }

    private void OnServerChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(ServerController.State))
        {
            _ = RefreshModelsAsync();
        }
    }

    private async void OnRefreshModels(object sender, RoutedEventArgs e) => await RefreshModelsAsync();

    private async Task RefreshModelsAsync()
    {
        var server = ServerController.Instance;
        if (!server.IsRunning)
        {
            NoServerBanner.Visibility = Visibility.Visible;
            NoServerText.Text = server.State == ServerState.Starting
                ? "The server is starting — models are loading. This page will be ready when it is."
                : "The server isn't running. Choose models and start it on the Server page.";
            return;
        }
        try
        {
            _models = await ServerClient.ForRunningServer().GetModelsAsync();
            NoServerBanner.Visibility = Visibility.Collapsed;
        }
        catch (Exception e)
        {
            NoServerBanner.Visibility = Visibility.Visible;
            NoServerText.Text = "The server did not answer: " + e.Message;
            return;
        }
        string? keep = (ModelBox.SelectedItem as ComboBoxItem)?.Tag as string ?? Settings.Model;
        ModelBox.Items.Clear();
        foreach (var (id, caps) in _models)
        {
            ModelBox.Items.Add(new ComboBoxItem { Content = $"{id}  ({string.Join(", ", caps)})", Tag = id });
        }
        ModelBox.SelectedItem = ModelBox.Items.Cast<ComboBoxItem>().FirstOrDefault(i => (string)i.Tag == keep)
                                ?? ModelBox.Items.Cast<ComboBoxItem>().FirstOrDefault(i => IsChat((string)i.Tag))
                                ?? (ModelBox.Items.Count > 0 ? ModelBox.Items[0] : null);
    }

    private string? SelectedModel => (ModelBox.SelectedItem as ComboBoxItem)?.Tag as string;

    private bool IsChat(string id) => _models.Any(m => m.Id == id && m.Capabilities.Any(c => c.Contains("chat", StringComparison.OrdinalIgnoreCase)));

    private void OnModelChanged(object sender, SelectionChangedEventArgs e)
    {
        if (SelectedModel is { } id)
        {
            Settings.Model = id;
            AppSettings.Current.Save();
            // A model that does not chat is tried from the API tab.
            if (!IsChat(id) && Tabs.SelectedIndex == 0)
            {
                Tabs.SelectedIndex = 1;
            }
            FillExample();
        }
    }

    private void OnGoToServer(object sender, RoutedEventArgs e) => _shell.Navigate("server");

    private void OnOptionChanged(object sender, RoutedEventArgs e)
    {
        if (_loading) return;
        Settings.Think = ThinkCheck.IsChecked == true;
        Settings.SystemPrompt = SystemBox.Text;
        Settings.Temperature = double.TryParse(TemperatureBox.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out var t) ? t : null;
        Settings.MaxTokens = int.TryParse(MaxTokensBox.Text, out var m) ? m : null;
        AppSettings.Current.Save();
    }

    // ── chat ────────────────────────────────────────────────────────────────

    private void OnInputKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter && (Keyboard.Modifiers & ModifierKeys.Shift) == 0)
        {
            e.Handled = true;
            OnSend(sender, e);
        }
    }

    private void OnClear(object sender, RoutedEventArgs e)
    {
        _cts?.Cancel();
        _messages.Clear();
        Conversation.Children.Clear();
        LastStatsText.Text = "";
        ShowEmptyHint();
    }

    private void ShowEmptyHint() =>
        Conversation.Children.Add(new TextBlock
        {
            Text = "Type a message below and press Enter.",
            Style = (Style)FindResource("Caption"),
            HorizontalAlignment = HorizontalAlignment.Center,
            Margin = new Thickness(0, 40, 0, 0),
            Tag = "hint",
        });

    private void OnStop(object sender, RoutedEventArgs e) => _cts?.Cancel();

    private async void OnSend(object sender, RoutedEventArgs e)
    {
        string text = InputBox.Text.Trim();
        if (text.Length == 0 || _cts is not null)
        {
            return;
        }
        if (!ServerController.Instance.IsRunning || SelectedModel is not { } model)
        {
            await RefreshModelsAsync();
            if (SelectedModel is null)
            {
                return;
            }
            model = SelectedModel;
        }
        if (Conversation.Children.Count == 1 && Conversation.Children[0] is FrameworkElement { Tag: "hint" })
        {
            Conversation.Children.Clear();
        }

        InputBox.Clear();
        _messages.Add(new JsonObject { ["role"] = "user", ["content"] = text });
        AddBubble("You", text, user: true);

        var body = new JsonObject { ["model"] = model };
        var messages = new JsonArray();
        if (SystemBox.Text.Trim() is { Length: > 0 } system)
        {
            messages.Add(new JsonObject { ["role"] = "system", ["content"] = system });
        }
        foreach (var m in _messages)
        {
            messages.Add(m.DeepClone());
        }
        body["messages"] = messages;
        if (ThinkCheck.IsChecked == true)
        {
            body["chat_template_kwargs"] = new JsonObject { ["enable_thinking"] = true };
        }
        else
        {
            body["chat_template_kwargs"] = new JsonObject { ["enable_thinking"] = false };
        }
        if (Settings.Temperature is { } temperature) body["temperature"] = temperature;
        if (Settings.MaxTokens is { } maxTokens) body["max_tokens"] = maxTokens;
        // Thinking takes far more tokens than an answer: without a limit of its own a thinking
        // request would get the server's non-thinking default (1024) and stop mid-thought.
        else if (ThinkCheck.IsChecked == true) body["max_tokens"] = 4096;
        if (ShowRequestCheck.IsChecked == true)
        {
            AddBubble("Request", body.ToJsonString(new JsonSerializerOptions { WriteIndented = true }), user: true, mono: true);
        }

        var (bubble, thinking, reply, stats) = AddReplyBubble(model);
        SendButton.IsEnabled = false;
        StopButton.IsEnabled = true;
        _cts = new CancellationTokenSource();
        var clock = Stopwatch.StartNew();
        double? firstToken = null;
        int pieces = 0;
        var content = new System.Text.StringBuilder();
        var reasoning = new System.Text.StringBuilder();
        try
        {
            var usage = await ServerClient.ForRunningServer().StreamChatAsync(body, (delta, isReasoning) =>
            {
                Dispatcher.BeginInvoke(() =>
                {
                    firstToken ??= clock.Elapsed.TotalSeconds;
                    pieces++;
                    if (isReasoning)
                    {
                        reasoning.Append(delta);
                        thinking.Text = reasoning.ToString();
                        thinking.Visibility = ShowThinkingCheck.IsChecked == true ? Visibility.Visible : Visibility.Collapsed;
                    }
                    else
                    {
                        content.Append(delta);
                        reply.Text = content.ToString();
                    }
                    stats.Text = $"{pieces} tokens · {Rate(pieces, firstToken, clock.Elapsed.TotalSeconds)}";
                    ScrollToEnd();
                });
            }, _cts.Token);
            await Dispatcher.InvokeAsync(() => { });   // let queued deltas land
            double total = clock.Elapsed.TotalSeconds;
            int tokens = usage.CompletionTokens > 0 ? usage.CompletionTokens : pieces;
            string summary = $"{tokens} tokens · {Rate(tokens, firstToken, total)} · first token {firstToken ?? total:0.00} s · {total:0.0} s total" +
                             (usage.PromptTokens > 0 ? $" · prompt {usage.PromptTokens} tokens" : "");
            if (usage.FinishReason == "length")
            {
                summary += " · cut off at the token limit (raise Max tokens under Advanced)";
            }
            stats.Text = summary;
            LastStatsText.Text = "Last reply: " + summary;
            if (content.Length == 0)
            {
                reply.Text = reasoning.Length > 0 && usage.FinishReason == "length"
                    ? "(no answer: the token limit was reached while thinking)"
                    : reasoning.Length > 0 ? "(no answer after thinking)" : "(empty reply)";
            }
            _messages.Add(new JsonObject { ["role"] = "assistant", ["content"] = content.ToString() });
        }
        catch (OperationCanceledException)
        {
            stats.Text = "stopped";
            if (content.Length > 0)
            {
                _messages.Add(new JsonObject { ["role"] = "assistant", ["content"] = content.ToString() });
            }
        }
        catch (Exception ex)
        {
            reply.Text = ex.Message;
            reply.Foreground = (Brush)FindResource("BadBrush");
            stats.Text = "failed";
            _messages.RemoveAt(_messages.Count - 1);
        }
        finally
        {
            _cts.Dispose();
            _cts = null;
            SendButton.IsEnabled = true;
            StopButton.IsEnabled = false;
            InputBox.Focus();
            ScrollToEnd();
        }
    }

    private static string Rate(int tokens, double? first, double now) =>
        first is { } f && now - f > 0.05 && tokens > 1 ? $"{(tokens - 1) / (now - f):0.0} tok/s" : "…";

    private void ScrollToEnd() => ChatScroll.ScrollToEnd();

    private void AddBubble(string who, string text, bool user, bool mono = false)
    {
        var panel = new StackPanel();
        panel.Children.Add(new TextBlock { Text = who, FontWeight = FontWeights.SemiBold, Margin = new Thickness(0, 0, 0, 2) });
        panel.Children.Add(Selectable(text, mono));
        Conversation.Children.Add(new Border
        {
            Child = panel,
            CornerRadius = new CornerRadius(8),
            Padding = new Thickness(12, 8, 12, 10),
            Margin = new Thickness(user ? 80 : 0, 4, user ? 0 : 80, 4),
            Background = (Brush)FindResource(user ? "SubtleFillColorSecondaryBrush" : "CardBackgroundFillColorDefaultBrush"),
            HorizontalAlignment = user ? HorizontalAlignment.Right : HorizontalAlignment.Stretch,
        });
        ScrollToEnd();
    }

    private (Border Bubble, TextBox Thinking, TextBox Reply, TextBlock Stats) AddReplyBubble(string model)
    {
        var panel = new StackPanel();
        panel.Children.Add(new TextBlock { Text = model, FontWeight = FontWeights.SemiBold, Margin = new Thickness(0, 0, 0, 2) });
        var thinking = Selectable("", false);
        thinking.FontStyle = FontStyles.Italic;
        thinking.Foreground = (Brush)FindResource("TextFillColorSecondaryBrush");
        thinking.Visibility = Visibility.Collapsed;
        thinking.Margin = new Thickness(0, 0, 0, 6);
        panel.Children.Add(thinking);
        var reply = Selectable("…", false);
        panel.Children.Add(reply);
        var stats = new TextBlock { Style = (Style)FindResource("Caption"), Margin = new Thickness(0, 6, 0, 0) };
        panel.Children.Add(stats);
        var bubble = new Border
        {
            Child = panel,
            CornerRadius = new CornerRadius(8),
            Padding = new Thickness(12, 8, 12, 10),
            Margin = new Thickness(0, 4, 80, 4),
            BorderThickness = new Thickness(1),
            BorderBrush = (Brush)FindResource("CardStrokeColorDefaultBrush"),
        };
        Conversation.Children.Add(bubble);
        ScrollToEnd();
        return (bubble, thinking, reply, stats);
    }

    /// <summary>Read-only text that can be selected and copied, without a text box's chrome.</summary>
    private TextBox Selectable(string text, bool mono)
    {
        var box = new TextBox
        {
            Text = text,
            IsReadOnly = true,
            TextWrapping = TextWrapping.Wrap,
            BorderThickness = new Thickness(0),
            Background = Brushes.Transparent,
            Padding = new Thickness(0),
            MinHeight = 0,
        };
        if (mono)
        {
            box.FontFamily = (FontFamily)FindResource("MonoFont");
            box.FontSize = 11.5;
        }
        // The Fluent text box template draws an underline and focus fill; a plain template keeps it text.
        box.Template = PlainTextBoxTemplate;
        return box;
    }

    private static readonly ControlTemplate PlainTextBoxTemplate = CreatePlainTemplate();

    private static ControlTemplate CreatePlainTemplate()
    {
        var template = new ControlTemplate(typeof(TextBox));
        var host = new FrameworkElementFactory(typeof(ScrollViewer)) { Name = "PART_ContentHost" };
        host.SetValue(ScrollViewer.HorizontalScrollBarVisibilityProperty, ScrollBarVisibility.Disabled);
        host.SetValue(ScrollViewer.VerticalScrollBarVisibilityProperty, ScrollBarVisibility.Disabled);
        template.VisualTree = host;
        return template;
    }

    // ── raw API requests ────────────────────────────────────────────────────

    private void OnEndpointChanged(object sender, SelectionChangedEventArgs e) => FillExample();

    private void FillExample()
    {
        if (ApiBodyBox is null) return;
        string model = SelectedModel ?? "model-id";
        string path = (EndpointBox.SelectedItem as ComboBoxItem)?.Tag as string ?? "";
        if (EndpointBox.SelectedItem is null)
        {
            // Pick the endpoint the selected model serves.
            var caps = _models.FirstOrDefault(m => m.Id == model).Capabilities ?? Array.Empty<string>();
            path = caps.Any(c => c.Contains("embed")) ? "/v1/embeddings"
                : caps.Any(c => c.Contains("classif")) ? "/v1/classify"
                : caps.Any(c => c.Contains("decision")) ? "/v1/decisions"
                : "/v1/chat/completions";
            EndpointBox.SelectedItem = EndpointBox.Items.Cast<ComboBoxItem>().First(i => (string)i.Tag == path);
            return;     // SelectionChanged calls back in
        }
        JsonObject body = path switch
        {
            "/v1/embeddings" => new() { ["model"] = model, ["input"] = new JsonArray("The quick brown fox", "A fast auburn fox") },
            "/v1/classify" => new() { ["model"] = model, ["input"] = "The service was excellent." },
            "/v1/decisions" => new()
            {
                ["model"] = model,
                ["state"] = "The customer asked for a refund after 40 days.",
                ["questions"] = new JsonObject
                {
                    ["refund"] = new JsonObject
                    {
                        ["type"] = "choice",
                        ["instructions"] = "Decide whether to approve the refund.",
                        ["criteria"] = new JsonObject { ["approve"] = "Approve the refund", ["decline"] = "Decline the refund" },
                    },
                },
            },
            "/v1/responses" => new() { ["model"] = model, ["input"] = "Say hello in five words." },
            _ => new() { ["model"] = model, ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = "Say hello in five words." }) },
        };
        ApiBodyBox.Text = body.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
    }

    private async void OnSendApi(object sender, RoutedEventArgs e)
    {
        if (!ServerController.Instance.IsRunning)
        {
            ApiResponseBox.Text = "The server isn't running.";
            return;
        }
        string path = (EndpointBox.SelectedItem as ComboBoxItem)?.Tag as string ?? "/v1/chat/completions";
        ApiResponseBox.Text = "Waiting for the server…";
        var clock = Stopwatch.StartNew();
        try
        {
            string result = await ServerClient.ForRunningServer().PostJsonAsync(path, ApiBodyBox.Text, CancellationToken.None);
            ApiResponseBox.Text = $"{result}\n\n({clock.Elapsed.TotalSeconds:0.00} s)";
        }
        catch (Exception ex)
        {
            ApiResponseBox.Text = ex.Message;
        }
    }
}
