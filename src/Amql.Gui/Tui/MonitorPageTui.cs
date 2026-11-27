using System.IO;
using System.ComponentModel;
using Amql.Gui.Services;
using Terminal.Gui.ViewBase;
using Terminal.Gui.Views;

namespace Amql.Gui.Tui;

/// <summary>
/// A live view of the server: what is generating now, how fast, what each
/// model has done, and the last requests — read from /amql/status once a
/// second (and the GPU from nvidia-smi every two). The desktop draws the two
/// trends as charts; here they are sparklines of block characters, five
/// minutes wide when the terminal is.
/// </summary>
internal sealed class MonitorPageTui : ITuiPage, IDisposable
{
    private const int History = 300;    // samples: five minutes at one a second

    private readonly View _root = new() { CanFocus = true };
    private readonly ColorLabel _dot = new(TuiKit.Idle) { Text = "●", X = 0, Y = 0, Width = 1 };
    private readonly Label _state = new() { X = 2, Y = 0, Width = Dim.Fill(30), Height = 1 };
    private readonly Label _sub = new() { X = 2, Y = 1, Width = Dim.Fill(), Height = 1 };
    private readonly CheckBox _pause = new() { Text = "_Pause", X = Pos.AnchorEnd(10), Y = 0 };
    private readonly Label[] _tileValue = new Label[5];
    private readonly Label[] _tileSub = new Label[5];
    private readonly Label _speedTitle = new() { X = 0, Y = 7, Width = Dim.Fill(), Height = 1, Text = "Tokens per second, last 5 minutes" };
    private readonly ColorLabel _speedChart = new(TuiKit.Accent) { X = 0, Y = 8, Width = Dim.Fill(), Height = 1 };
    private readonly Label _loadTitle = new() { X = 0, Y = 9, Width = Dim.Fill(), Height = 1, Text = "GPU utilisation (%), last 5 minutes" };
    private readonly ColorLabel _loadChart = new(TuiKit.Violet) { X = 0, Y = 10, Width = Dim.Fill(), Height = 1 };
    private readonly TableView _models = new() { X = 0, Y = 12, Width = Dim.Fill(), Height = 6, FullRowSelect = true };
    private readonly Label _active = new() { X = 0, Y = 19, Width = Dim.Fill(), Height = 4 };
    private readonly TableView _recent = new() { X = 0, Y = 24, Width = Dim.Fill(), Height = Dim.Fill(), FullRowSelect = true };

    private readonly List<double> _speed = new(), _load = new();
    private object? _timer;
    private bool _polling;
    private int _tick;
    private long? _lastTokens;
    private DateTime _lastSampleAt;
    private double? _lastCpuSeconds;
    private bool _haveGpu = true;
    private GpuSample? _gpu;
    private long? _serverVram;

    public MonitorPageTui()
    {
        _root.Add(_dot, _state, _sub, _pause);
        string[] titles = { "Generating now", "Output speed", "Requests", "Server memory", "GPU" };
        for (int i = 0; i < titles.Length; i++)
        {
            var tile = new FrameView { Title = titles[i], X = Pos.Percent(i * 20), Y = 2, Width = Dim.Percent(20), Height = 4 };
            _tileValue[i] = new Label { X = 0, Y = 0, Width = Dim.Fill(), Height = 1, Text = "–" };
            _tileSub[i] = new Label { X = 0, Y = 1, Width = Dim.Fill(), Height = 1 };
            tile.Add(_tileValue[i], _tileSub[i]);
            _root.Add(tile);
        }
        _root.Add(_speedTitle, _speedChart, _loadTitle, _loadChart);
        _root.Add(new Label { Text = "Models", X = 0, Y = 11 }, _models);
        _root.Add(new Label { Text = "In progress", X = 0, Y = 18 }, _active);
        _root.Add(new Label { Text = "Recent requests", X = 0, Y = 23 }, _recent);
        ServerController.Instance.PropertyChanged += OnServerChanged;
        ShowModels(Array.Empty<ModelStatus>());
        ShowRecent(Array.Empty<RequestStatus>());
    }

    public View View => _root;

    public void Shown()
    {
        _timer ??= TuiKit.App.AddTimeout(TimeSpan.FromSeconds(1), () =>
        {
            TuiKit.GuardAsync(PollAsync);
            return true;
        });
        TuiKit.GuardAsync(PollAsync);
    }

    /// <summary>Polling stops while another page is showing (as closing the desktop monitor stops it); the trends are kept.</summary>
    public void Hidden()
    {
        if (_timer is not null)
        {
            TuiKit.App.RemoveTimeout(_timer);
            _timer = null;
        }
    }

    public void Dispose()
    {
        Hidden();
        ServerController.Instance.PropertyChanged -= OnServerChanged;
    }

    private void OnServerChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(ServerController.State) && !ServerController.Instance.IsRunning)
        {
            _lastTokens = null;
            _lastCpuSeconds = null;
        }
    }

    private async Task PollAsync()
    {
        if (_polling || _pause.Value == CheckState.Checked)
        {
            return;
        }
        _polling = true;
        try
        {
            var server = ServerController.Instance;
            if (_haveGpu && _tick++ % 2 == 0)
            {
                _gpu = await SystemInfo.SampleGpuAsync();
                _haveGpu = _gpu is not null;
                if (server.ProcessId is { } pid)
                {
                    _serverVram = await SystemInfo.GpuMemoryOfProcessAsync(pid);
                }
            }
            ShowGpu();
            ServerStatus? status = server.IsRunning ? await ServerClient.ForRunningServer().GetStatusAsync() : null;
            ShowState(server, status);
            if (status is not null)
            {
                Show(status);
            }
            DrawCharts();
        }
        finally
        {
            _polling = false;
        }
    }

    private static void Push(List<double> series, double value)
    {
        series.Add(value);
        if (series.Count > History)
        {
            series.RemoveAt(0);
        }
    }

    private void DrawCharts()
    {
        int width = Math.Max(10, _root.Viewport.Width - 16);
        double peak = _speed.Count == 0 ? 0 : _speed.Max();
        _speedChart.Text = TuiKit.Sparkline(_speed, width) + $"  peak {peak:0.#}";
        _loadChart.Text = TuiKit.Sparkline(_load, width, 100) + $"  {(_load.Count > 0 ? _load[^1] : 0):0}%";
        _speedChart.SetNeedsDraw();
        _loadChart.SetNeedsDraw();
    }

    private void ShowState(ServerController server, ServerStatus? status)
    {
        _dot.Color = TuiKit.StateColor(server.State, answering: status is not null);
        _dot.SetNeedsDraw();
        if (status is null)
        {
            _state.Text = server.IsRunning ? "Running — no status" : server.Detail;
            _sub.Text = server.IsRunning
                ? "The server does not answer /amql/status. It may be an older build: rebuild amql-server to see its activity here."
                : server.State == ServerState.Starting
                    ? $"{server.ModelsLoaded} of {server.ModelsTotal} models loaded"
                    : "Start the server from the Server page (or F2) to see what it is doing.";
            if (!server.IsRunning)
            {
                for (int i = 0; i < 4; i++)
                {
                    _tileValue[i].Text = "–";
                    _tileSub[i].Text = "";
                }
            }
            return;
        }
        _state.Text = $"Running · {status.Models.Count} model{(status.Models.Count == 1 ? "" : "s")}";
        _sub.Text = $"{server.BaseUrl}/v1 · up {Format.Duration(TimeSpan.FromSeconds(status.UptimeSeconds))} · " +
                    $"process {status.Process.Pid} · {(status.Process.Gpu ? "GPU" : "CPU only")}";
    }

    private void Show(ServerStatus s)
    {
        var now = DateTime.Now;
        // Throughput: tokens produced since the last sample, counting replies still streaming.
        long tokens = s.Models.Sum(m => m.CompletionTokens) + s.Active.Sum(a => (long)a.CompletionTokens);
        double rate = 0;
        double elapsed = (now - _lastSampleAt).TotalSeconds;
        if (_lastTokens is { } before && elapsed > 0.2)
        {
            rate = Math.Max(0, (tokens - before) / elapsed);
            Push(_speed, rate);
        }
        double cpuPercent = 0;
        if (_lastCpuSeconds is { } cpuBefore && elapsed > 0.2)
        {
            cpuPercent = Math.Clamp((s.Process.CpuSeconds - cpuBefore) / elapsed / Environment.ProcessorCount * 100, 0, 100);
            if (!_haveGpu)
            {
                Push(_load, cpuPercent);
            }
        }
        _lastTokens = tokens;
        _lastCpuSeconds = s.Process.CpuSeconds;
        _lastSampleAt = now;

        int generating = s.Active.Count;
        _tileValue[0].Text = generating.ToString();
        _tileSub[0].Text = generating == 0 ? "idle" : string.Join(", ", s.Active.GroupBy(a => a.Model ?? "?").Select(g => $"{g.Key} ×{g.Count()}"));
        _tileValue[1].Text = $"{rate:0} tok/s";
        _tileSub[1].Text = $"{s.Totals.CompletionTokensLastMinute / 60.0:0.#} tok/s last min";
        _tileValue[2].Text = s.Totals.Requests.ToString("N0");
        _tileSub[2].Text = $"{s.Totals.RequestsLastMinute} last min" + (s.Totals.Failures > 0 ? $" · {s.Totals.Failures} failed" : "");
        _tileValue[3].Text = Format.Bytes(s.Process.WorkingSetBytes);
        _tileSub[3].Text = $"{Format.Bytes(s.Process.PrivateBytes)} committed" + (_haveGpu ? "" : $" · CPU {cpuPercent:0}%");

        ShowModels(s.Models);
        _active.Text = generating == 0
            ? "Nothing is running."
            : string.Join("\n", s.Active.Select(a =>
                $"{a.Model ?? "(routing)"}  {a.Path}  {a.Seconds:0.0} s" +
                (a.CompletionTokens > 0 ? $" · {a.CompletionTokens} tokens" : " · reading the prompt") +
                (a.DecodeTps is { } d ? $" · {d:0.#} tok/s" : "") +
                (a.TtftSeconds is { } f ? $" · first token after {f:0.00} s" : "")));
        ShowRecent(s.Recent);
    }

    private void ShowModels(IReadOnlyList<ModelStatus> models)
    {
        _models.Table = new EnumerableTableSource<ModelStatus>(models, new Dictionary<string, Func<ModelStatus, object>>
        {
            ["Model"] = m => m.Id,
            ["Serves"] = m => string.Join(", ", m.Capabilities) + (m.Adapters.Count > 0 ? $" + {string.Join(", ", m.Adapters)}" : ""),
            ["State"] = m => m.Active > 0 ? $"● Busy ({m.Active})" : "○ Idle",
            ["Requests"] = m => m.Requests.ToString("N0") + (m.Errors > 0 ? $" ({m.Errors} failed)" : ""),
            ["Tokens out"] = m => m.CompletionTokens.ToString("N0"),
            ["Avg decode"] = m => m.MeanDecodeTps is { } t ? $"{t:0.#} tok/s" : "–",
            ["Last used"] = m => m.LastUsed is { } u ? Ago(u) : "never",
        });
    }

    private void ShowRecent(IReadOnlyList<RequestStatus> recent)
    {
        _recent.Table = new EnumerableTableSource<RequestStatus>(recent, new Dictionary<string, Func<RequestStatus, object>>
        {
            ["Time"] = r => r.Started.ToLocalTime().ToString("HH:mm:ss"),
            ["Model"] = r => r.Model ?? "–",
            ["Endpoint"] = r => $"{r.Method} {r.Path}",
            ["Status"] = r => r.Status is { } code ? (code < 400 ? $"{code} {r.FinishReason}".Trim() : $"{code} error") : "…",
            ["In"] = r => r.PromptTokens > 0 ? r.PromptTokens.ToString("N0") : "",
            ["Out"] = r => r.CompletionTokens > 0 ? r.CompletionTokens.ToString("N0") : "",
            ["First token"] = r => r.TtftSeconds is { } t ? $"{t:0.00} s" : "",
            ["Decode"] = r => r.DecodeTps is { } d ? $"{d:0.#} tok/s" : "",
            ["Total"] = r => Format.Duration(TimeSpan.FromSeconds(r.Seconds)),
        });
    }

    private void ShowGpu()
    {
        if (!_haveGpu || _gpu is null)
        {
            _tileValue[4].Text = "None";
            _tileSub[4].Text = "No NVIDIA GPU (or nvidia-smi) found";
            _loadTitle.Text = "Server CPU use (%), last 5 minutes";
            return;
        }
        _tileValue[4].Text = $"{_gpu.UtilisationPercent}%" + (_gpu.TemperatureC is { } c ? $" · {c}°C" : "");
        _tileSub[4].Text = $"{Format.Bytes(_gpu.UsedBytes)} of {Format.Bytes(_gpu.TotalBytes)}{(_gpu.Unified ? " shared" : "")}" +
                           (_serverVram is { } v ? $" ({Format.Bytes(v)} server)" : "");
        if (_tick % 2 == 1)
        {
            Push(_load, _gpu.UtilisationPercent);
        }
    }

    private static string Ago(DateTime utc)
    {
        var span = DateTime.UtcNow - utc.ToUniversalTime();
        return span.TotalSeconds < 5 ? "just now" : Format.Duration(span) + " ago";
    }
}
