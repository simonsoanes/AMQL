// TextView is marked obsolete in favour of a separate editor package; it is still the text area Terminal.Gui ships.
#pragma warning disable CS0618
using System.IO;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Amql.Gui.Services;
using Terminal.Gui.Input;
using Terminal.Gui.ViewBase;
using Terminal.Gui.Views;

namespace Amql.Gui.Tui;

/// <summary>
/// Chat with a model on the running server (streamed, with its thinking), or
/// send any request to the server's API. The conversation is a scrolling
/// transcript rather than bubbles; everything else matches the desktop page.
/// </summary>
internal sealed class PlaygroundPageTui : ITuiPage
{
    private sealed class Turn
    {
        public required string Who { get; init; }
        public StringBuilder Text { get; } = new();
        public StringBuilder Thinking { get; } = new();
        public string Stats { get; set; } = "";
        public bool Mono { get; init; }
    }

    private static readonly (string Label, string Path)[] Endpoints =
    {
        ("POST /v1/chat/completions", "/v1/chat/completions"),
        ("POST /v1/responses", "/v1/responses"),
        ("POST /v1/embeddings", "/v1/embeddings"),
        ("POST /v1/classify", "/v1/classify"),
        ("POST /v1/decisions", "/v1/decisions"),
    };

    private readonly Shell _shell;
    private readonly View _root = new() { CanFocus = true };
    private readonly Button _modelButton;
    private readonly Label _banner = new() { X = 0, Y = 1, Width = Dim.Fill(18), Height = 1 };
    private readonly Button _goServer;
    private readonly Tabs _tabs;
    private readonly View _chatTab, _apiTab;
    private readonly TextView _conversation = new() { X = 0, Y = 0, Width = Dim.Fill(37), Height = Dim.Fill(3), ReadOnly = true, WordWrap = true };
    private readonly FieldList _settings = new(3) { X = Pos.AnchorEnd(36), Y = 0, Width = 36, Height = Dim.Fill(3) };
    private readonly TextField _input = new() { X = 9, Y = Pos.AnchorEnd(2), Width = Dim.Fill(20) };
    private readonly Button _send, _stop;
    private readonly Label _stats = new() { X = 0, Y = Pos.AnchorEnd(1), Width = Dim.Fill() };
    private readonly Button _endpointButton;
    private readonly TextView _apiBody = new() { X = 0, Y = 2, Width = Dim.Fill(), Height = Dim.Percent(45) };
    private readonly TextView _apiResponse = new() { X = 0, Y = 0, Width = Dim.Fill(), Height = Dim.Fill(), ReadOnly = true };

    private readonly List<JsonObject> _messages = new();
    private readonly List<Turn> _turns = new();
    private List<(string Id, string[] Capabilities)> _models = new();
    private string? _model;
    private string _endpoint = "";
    private CancellationTokenSource? _cts;
    private bool _showThinking = true, _showRequest;
    private string _lastStats = "";
    private object? _redraw;

    private static PlaygroundSettings Settings => AppSettings.Current.Playground;

    public PlaygroundPageTui(Shell shell)
    {
        _shell = shell;
        _model = Settings.Model;
        _root.Add(new Label { Text = "Model", X = 0, Y = 0 });
        _modelButton = TuiKit.Button("(no models)", ChooseModel, 6, 0);
        var refresh = TuiKit.Button("Refresh", () => TuiKit.GuardAsync(RefreshModelsAsync), Pos.Right(_modelButton) + 1, 0);
        _goServer = TuiKit.Button("Go to Server", () => _shell.Navigate("server"), Pos.AnchorEnd(16), 1);
        _root.Add(_modelButton, refresh, _banner, _goServer);

        _tabs = new Tabs { TabStop = TabBehavior.TabStop, X = 0, Y = 2, Width = Dim.Fill(), Height = Dim.Fill() };

        // Chat
        _chatTab = new View { Title = "_Chat", Width = Dim.Fill(), Height = Dim.Fill(), CanFocus = true };
        _send = TuiKit.Button("Send", () => TuiKit.GuardAsync(SendAsync), Pos.AnchorEnd(19), Pos.AnchorEnd(2));
        _stop = TuiKit.Button("Stop", () => _cts?.Cancel(), Pos.AnchorEnd(9), Pos.AnchorEnd(2));
        _stop.Enabled = false;
        _input.Accepting += (_, e) =>
        {
            e.Handled = true;
            TuiKit.GuardAsync(SendAsync);
        };
        _settings.SetFields(new Field[]
        {
            new HeaderField("Settings"),
            new BoolField("Think before answering", () => Settings.Think, v => Settings.Think = v),
            new BoolField("Show the thinking", () => _showThinking, v => _showThinking = v),
            new TextValueField("System prompt", () => Settings.SystemPrompt, v => Settings.SystemPrompt = v,
                "Sent first in every request of this conversation.") { Multiline = true, Placeholder = "none" },
            new HeaderField("Advanced"),
            new TextValueField("Temperature", () => Settings.Temperature?.ToString(CultureInfo.InvariantCulture) ?? "",
                v => Settings.Temperature = double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var t) ? t : null)
                { Placeholder = "server", Validate = TextValueField.OptionalNumber },
            new TextValueField("Max tokens", () => Settings.MaxTokens?.ToString(CultureInfo.InvariantCulture) ?? "",
                v => Settings.MaxTokens = int.TryParse(v, out var m) ? m : null,
                "Empty: the server's limit (4096 when thinking).") { Placeholder = "server", Validate = TextValueField.OptionalInt },
            new BoolField("Show each request's JSON", () => _showRequest, v => _showRequest = v),
            new ActionField("New conversation", NewConversation),
            new InfoField("Last reply", () => _lastStats),
        });
        _settings.LabelWidth = 24;
        _settings.Changed += _ =>
        {
            AppSettings.Current.Save();
            Render();
        };
        _chatTab.Add(new Label { Text = "Message›", X = 0, Y = Pos.AnchorEnd(2) });
        _chatTab.Add(_conversation, _settings, _input, _send, _stop, _stats);

        // API request
        _apiTab = new View { Title = "_API request", Width = Dim.Fill(), Height = Dim.Fill(), CanFocus = true };
        _endpointButton = TuiKit.Button(Endpoints[0].Label, ChooseEndpoint, 0, 0);
        var sendApi = TuiKit.Button("Send", () => TuiKit.GuardAsync(SendApiAsync), Pos.AnchorEnd(9), 0);
        _apiTab.Add(_endpointButton, sendApi);
        _apiTab.Add(new Label { Text = "Request body (JSON). Choosing an endpoint fills in an example for the selected model.", X = 0, Y = 1, Width = Dim.Fill() });
        _apiTab.Add(_apiBody);
        var responseLabel = new Label { Text = "Response", X = 0, Y = Pos.Bottom(_apiBody) };
        _apiResponse.Y = Pos.Bottom(responseLabel);
        _apiTab.Add(responseLabel, _apiResponse);

        _tabs.Add(_chatTab, _apiTab);
        _root.Add(_tabs);

        ServerController.Instance.PropertyChanged += OnServerChanged;
        Render();
    }

    public View View => _root;

    public void Shown() => TuiKit.GuardAsync(RefreshModelsAsync);

    public void Focus() => _input.SetFocus();

    public void Hidden()
    {
    }

    // ── models ──────────────────────────────────────────────────────────────

    private void OnServerChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(ServerController.State))
        {
            TuiKit.GuardAsync(RefreshModelsAsync);
        }
    }

    private void ShowBanner(string? text)
    {
        _banner.Text = text ?? "";
        _banner.Visible = text is not null;
        _goServer.Visible = text is not null;
    }

    private async Task RefreshModelsAsync()
    {
        var server = ServerController.Instance;
        if (!server.IsRunning)
        {
            ShowBanner(server.State == ServerState.Starting
                ? "The server is starting — models are loading. This page will be ready when it is."
                : "The server isn't running. Choose models and start it on the Server page.");
            return;
        }
        try
        {
            _models = await ServerClient.ForRunningServer().GetModelsAsync();
            ShowBanner(null);
        }
        catch (Exception e)
        {
            ShowBanner("The server did not answer: " + e.Message);
            return;
        }
        string? keep = _model ?? Settings.Model;
        _model = _models.Any(m => m.Id == keep) ? keep
            : _models.FirstOrDefault(m => IsChat(m.Id)).Id ?? _models.FirstOrDefault().Id;
        UpdateModelButton();
        if (_apiBody.Text.Length == 0)
        {
            FillExample(null);
        }
    }

    private bool IsChat(string id) =>
        _models.Any(m => m.Id == id && m.Capabilities.Any(c => c.Contains("chat", StringComparison.OrdinalIgnoreCase)));

    private void UpdateModelButton() =>
        _modelButton.Text = _model is null ? "(no models)"
            : $"{_model}  ({string.Join(", ", _models.FirstOrDefault(m => m.Id == _model).Capabilities ?? Array.Empty<string>())})  ▾";

    private void ChooseModel()
    {
        if (_models.Count == 0)
        {
            TuiKit.Info("Model", "No models: start the server on the Server page first.");
            return;
        }
        int current = Math.Max(0, _models.FindIndex(m => m.Id == _model));
        if (TuiKit.Choose("Model", _models.Select(m => $"{m.Id}  ({string.Join(", ", m.Capabilities)})").ToList(), current) is not { } i)
        {
            return;
        }
        _model = _models[i].Id;
        Settings.Model = _model;
        AppSettings.Current.Save();
        UpdateModelButton();
        if (!IsChat(_model))
        {
            _tabs.Value = _apiTab;
        }
        FillExample(null);
    }

    // ── chat ────────────────────────────────────────────────────────────────

    private void NewConversation()
    {
        _cts?.Cancel();
        _messages.Clear();
        _turns.Clear();
        _lastStats = "";
        _stats.Text = "";
        Render();
    }

    /// <summary>Rebuilds the transcript. Called at most ten times a second while a reply streams.</summary>
    private void Render()
    {
        if (_turns.Count == 0)
        {
            _conversation.Text = "\n\n          Type a message below and press Enter.";
            return;
        }
        var sb = new StringBuilder();
        foreach (var t in _turns)
        {
            sb.Append("── ").Append(t.Who).AppendLine(" " + new string('─', Math.Max(3, 40 - t.Who.Length)));
            if (t.Thinking.Length > 0 && _showThinking)
            {
                foreach (var line in t.Thinking.ToString().Split('\n'))
                {
                    sb.Append("  ┊ ").AppendLine(line);
                }
                sb.AppendLine();
            }
            sb.AppendLine(t.Text.ToString());
            if (t.Stats.Length > 0)
            {
                sb.Append("  · ").AppendLine(t.Stats);
            }
            sb.AppendLine();
        }
        _conversation.Text = sb.ToString();
        _conversation.MoveEnd();
    }

    private void ScheduleRender()
    {
        _redraw ??= TuiKit.App.AddTimeout(TimeSpan.FromMilliseconds(100), () =>
        {
            _redraw = null;
            Render();
            return false;
        });
    }

    private static string Rate(int tokens, double? first, double now) =>
        first is { } f && now - f > 0.05 && tokens > 1 ? $"{(tokens - 1) / (now - f):0.0} tok/s" : "…";

    private async Task SendAsync()
    {
        string text = _input.Text.Trim();
        if (text.Length == 0 || _cts is not null)
        {
            return;
        }
        if (!ServerController.Instance.IsRunning || _model is null)
        {
            await RefreshModelsAsync();
            if (_model is null)
            {
                return;
            }
        }
        string model = _model;
        _input.Text = "";
        _messages.Add(new JsonObject { ["role"] = "user", ["content"] = text });
        var you = new Turn { Who = "You" };
        you.Text.Append(text);
        _turns.Add(you);

        var body = new JsonObject { ["model"] = model };
        var messages = new JsonArray();
        if (Settings.SystemPrompt.Trim() is { Length: > 0 } system)
        {
            messages.Add(new JsonObject { ["role"] = "system", ["content"] = system });
        }
        foreach (var m in _messages)
        {
            messages.Add(m.DeepClone());
        }
        body["messages"] = messages;
        body["chat_template_kwargs"] = new JsonObject { ["enable_thinking"] = Settings.Think };
        if (Settings.Temperature is { } temperature) body["temperature"] = temperature;
        if (Settings.MaxTokens is { } maxTokens) body["max_tokens"] = maxTokens;
        else if (Settings.Think) body["max_tokens"] = 4096;
        if (_showRequest)
        {
            var req = new Turn { Who = "Request", Mono = true };
            req.Text.Append(body.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
            _turns.Add(req);
        }

        var reply = new Turn { Who = model };
        reply.Text.Append("…");
        _turns.Add(reply);
        Render();
        _send.Enabled = false;
        _stop.Enabled = true;
        _cts = new CancellationTokenSource();
        var clock = Stopwatch.StartNew();
        double? firstToken = null;
        int pieces = 0;
        bool any = false;
        try
        {
            var usage = await ServerClient.ForRunningServer().StreamChatAsync(body, (delta, isReasoning) => UiThread.Run(() =>
            {
                firstToken ??= clock.Elapsed.TotalSeconds;
                pieces++;
                if (isReasoning)
                {
                    reply.Thinking.Append(delta);
                }
                else
                {
                    if (!any)
                    {
                        reply.Text.Clear();
                        any = true;
                    }
                    reply.Text.Append(delta);
                }
                reply.Stats = $"{pieces} tokens · {Rate(pieces, firstToken, clock.Elapsed.TotalSeconds)}";
                ScheduleRender();
            }), _cts.Token);
            await Task.Delay(50);       // let queued deltas land
            double total = clock.Elapsed.TotalSeconds;
            int tokens = usage.CompletionTokens > 0 ? usage.CompletionTokens : pieces;
            string summary = $"{tokens} tokens · {Rate(tokens, firstToken, total)} · first token {firstToken ?? total:0.00} s · {total:0.0} s total" +
                             (usage.PromptTokens > 0 ? $" · prompt {usage.PromptTokens} tokens" : "");
            if (usage.FinishReason == "length")
            {
                summary += " · cut off at the token limit (raise Max tokens)";
            }
            reply.Stats = summary;
            _lastStats = summary;
            if (!any)
            {
                reply.Text.Clear().Append(reply.Thinking.Length > 0 && usage.FinishReason == "length"
                    ? "(no answer: the token limit was reached while thinking)"
                    : reply.Thinking.Length > 0 ? "(no answer after thinking)" : "(empty reply)");
            }
            _messages.Add(new JsonObject { ["role"] = "assistant", ["content"] = any ? reply.Text.ToString() : "" });
        }
        catch (OperationCanceledException)
        {
            reply.Stats = "stopped";
            if (any)
            {
                _messages.Add(new JsonObject { ["role"] = "assistant", ["content"] = reply.Text.ToString() });
            }
        }
        catch (Exception ex)
        {
            reply.Text.Clear().Append("✖ ").Append(ex.Message);
            reply.Stats = "failed";
            _messages.RemoveAt(_messages.Count - 1);
        }
        finally
        {
            _cts.Dispose();
            _cts = null;
            _send.Enabled = true;
            _stop.Enabled = false;
            _stats.Text = _lastStats.Length > 0 ? "Last reply: " + _lastStats : "";
            Render();
            _settings.Refresh();
            _input.SetFocus();
        }
    }

    // ── API request ─────────────────────────────────────────────────────────

    private void ChooseEndpoint()
    {
        int current = Math.Max(0, Array.FindIndex(Endpoints, e => e.Path == _endpoint));
        if (TuiKit.Choose("Endpoint", Endpoints.Select(e => e.Label).ToList(), current) is { } i)
        {
            FillExample(Endpoints[i].Path);
        }
    }

    /// <summary>Fills the body with an example for the endpoint (chosen from the model's capabilities when null).</summary>
    private void FillExample(string? path)
    {
        string model = _model ?? "model-id";
        if (path is null)
        {
            var caps = _models.FirstOrDefault(m => m.Id == model).Capabilities ?? Array.Empty<string>();
            path = caps.Any(c => c.Contains("embed")) ? "/v1/embeddings"
                : caps.Any(c => c.Contains("classif")) ? "/v1/classify"
                : caps.Any(c => c.Contains("decision")) ? "/v1/decisions"
                : "/v1/chat/completions";
        }
        _endpoint = path;
        _endpointButton.Text = Endpoints.First(e => e.Path == path).Label + "  ▾";
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
        _apiBody.Text = body.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
    }

    private async Task SendApiAsync()
    {
        if (!ServerController.Instance.IsRunning)
        {
            _apiResponse.Text = "The server isn't running.";
            return;
        }
        string path = _endpoint.Length > 0 ? _endpoint : "/v1/chat/completions";
        _apiResponse.Text = "Waiting for the server…";
        var clock = Stopwatch.StartNew();
        try
        {
            string result = await ServerClient.ForRunningServer().PostJsonAsync(path, _apiBody.Text, CancellationToken.None);
            _apiResponse.Text = $"{result}\n\n({clock.Elapsed.TotalSeconds:0.00} s)";
        }
        catch (Exception ex)
        {
            _apiResponse.Text = ex.Message;
        }
    }
}
