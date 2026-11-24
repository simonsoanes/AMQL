using System.ComponentModel;
using System.Windows;
using System.Windows.Media;
using System.Windows.Threading;
using Amql.Gui.Services;

namespace Amql.Gui.Windows;

/// <summary>
/// A live view of the server: what is generating now, how fast, what each
/// model has done, and the last requests — read from the server's
/// /amql/status once a second (and the GPU from nvidia-smi every two), so it
/// shows the state rather than a log to scroll through.
/// </summary>
public partial class ServerMonitorWindow : Window
{
    private readonly DispatcherTimer _timer = new() { Interval = TimeSpan.FromSeconds(1) };
    private bool _polling;
    private int _tick;
    private long? _lastTokens;
    private DateTime _lastSampleAt;
    private double? _lastCpuSeconds;
    private bool _haveGpu = true;
    private GpuSample? _gpu;
    private long? _serverVram;

    public ServerMonitorWindow()
    {
        InitializeComponent();
        WheelScrolling.Attach(PageScroll);
        SpeedChart.FormatValue = v => $"{v:0.#} tok/s";
        LoadChart.FormatValue = v => $"{v:0}%";
        LoadChart.LineBrush = new SolidColorBrush(Color.FromRgb(0x8B, 0x5C, 0xF6));
        _timer.Tick += async (_, _) => await PollAsync();
        Loaded += async (_, _) =>
        {
            _timer.Start();
            await PollAsync();
        };
        Closed += (_, _) =>
        {
            _timer.Stop();
            ServerController.Instance.PropertyChanged -= OnServerChanged;
        };
        ServerController.Instance.PropertyChanged += OnServerChanged;
    }

    private void OnTopmost(object sender, RoutedEventArgs e) => Topmost = TopmostCheck.IsChecked == true;

    private void OnServerChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(ServerController.State) && !ServerController.Instance.IsRunning)
        {
            _lastTokens = null;
            _lastCpuSeconds = null;
        }
    }

    private Brush Res(string key) => (Brush)FindResource(key);

    private async Task PollAsync()
    {
        if (_polling || PauseCheck.IsChecked == true)
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
        }
        finally
        {
            _polling = false;
        }
    }

    private void ShowState(ServerController server, ServerStatus? status)
    {
        StateDot.Fill = Res(server.State switch
        {
            ServerState.Running or ServerState.External when status is not null => "GoodBrush",
            ServerState.Running or ServerState.External or ServerState.Starting or ServerState.Stopping => "WarnBrush",
            ServerState.Failed => "BadBrush",
            _ => "IdleBrush",
        });
        if (status is null)
        {
            StateText.Text = server.IsRunning ? "Running — no status" : server.Detail;
            SubText.Text = server.IsRunning
                ? "The server does not answer /amql/status. It may be an older build: rebuild amql-server to see its activity here."
                : server.State == ServerState.Starting
                    ? $"{server.ModelsLoaded} of {server.ModelsTotal} models loaded"
                    : "Start the server from the Server page to see what it is doing.";
            if (!server.IsRunning)
            {
                ActiveValue.Text = SpeedValue.Text = RequestsValue.Text = MemoryValue.Text = "–";
                ActiveSub.Text = SpeedSub.Text = RequestsSub.Text = MemorySub.Text = "";
            }
            return;
        }
        StateText.Text = $"Running · {status.Models.Count} model{(status.Models.Count == 1 ? "" : "s")}";
        SubText.Text = $"{server.BaseUrl}/v1 · up {Format.Duration(TimeSpan.FromSeconds(status.UptimeSeconds))} · " +
                       $"process {status.Process.Pid} · {(status.Process.Gpu ? "GPU" : "CPU only")}";
    }

    private void Show(ServerStatus s)
    {
        var now = DateTime.Now;

        // Throughput: tokens produced since the last sample, counting replies still streaming.
        long tokens = s.Models.Sum(m => m.CompletionTokens) + s.Active.Sum(a => (long)a.CompletionTokens);
        double rate = 0;
        if (_lastTokens is { } before && (now - _lastSampleAt).TotalSeconds > 0.2)
        {
            rate = Math.Max(0, (tokens - before) / (now - _lastSampleAt).TotalSeconds);
            SpeedChart.Add(rate);
        }
        double cpuPercent = 0;
        if (_lastCpuSeconds is { } cpuBefore && (now - _lastSampleAt).TotalSeconds > 0.2)
        {
            cpuPercent = Math.Clamp((s.Process.CpuSeconds - cpuBefore) / (now - _lastSampleAt).TotalSeconds / Environment.ProcessorCount * 100, 0, 100);
            if (!_haveGpu)
            {
                LoadChart.Add(cpuPercent);
            }
        }
        _lastTokens = tokens;
        _lastCpuSeconds = s.Process.CpuSeconds;
        _lastSampleAt = now;

        // Tiles
        int generating = s.Active.Count;
        ActiveValue.Text = generating.ToString();
        ActiveSub.Text = generating == 0 ? "idle" : string.Join(", ", s.Active.GroupBy(a => a.Model ?? "?").Select(g => $"{g.Key} ×{g.Count()}"));
        SpeedValue.Text = $"{rate:0} tok/s";
        SpeedSub.Text = $"{s.Totals.CompletionTokensLastMinute / 60.0:0.#} tok/s over the last minute";
        RequestsValue.Text = s.Totals.Requests.ToString("N0");
        RequestsSub.Text = $"{s.Totals.RequestsLastMinute} in the last minute" + (s.Totals.Failures > 0 ? $" · {s.Totals.Failures} failed" : "");
        MemoryValue.Text = Format.Bytes(s.Process.WorkingSetBytes);
        MemorySub.Text = $"{Format.Bytes(s.Process.PrivateBytes)} committed" + (_haveGpu ? "" : $" · CPU {cpuPercent:0}%");

        // Models
        ModelsGrid.ItemsSource = s.Models.Select(m => new
        {
            m.Id,
            Serves = string.Join(", ", m.Capabilities) + (m.Adapters.Count > 0 ? $" + {string.Join(", ", m.Adapters)}" : ""),
            State = m.Active > 0 ? $"Busy ({m.Active})" : "Idle",
            StateBrush = Res(m.Active > 0 ? "WarnBrush" : "GoodBrush"),
            Requests = m.Requests.ToString("N0") + (m.Errors > 0 ? $" ({m.Errors} failed)" : ""),
            TokensOut = m.CompletionTokens.ToString("N0"),
            Decode = m.MeanDecodeTps is { } t ? $"{t:0.#} tok/s" : "–",
            LastUsed = m.LastUsed is { } u ? Ago(u) : "never",
        }).ToList();

        // In flight
        NoActiveText.Visibility = generating == 0 ? Visibility.Visible : Visibility.Collapsed;
        ActiveList.ItemsSource = s.Active.Select(a => new
        {
            Model = a.Model ?? "(routing)",
            Endpoint = a.Path,
            Progress = $"{a.Seconds:0.0} s" +
                       (a.CompletionTokens > 0 ? $" · {a.CompletionTokens} tokens" : " · reading the prompt") +
                       (a.DecodeTps is { } d ? $" · {d:0.#} tok/s" : "") +
                       (a.TtftSeconds is { } f ? $" · first token after {f:0.00} s" : ""),
        }).ToList();

        // Recent
        RecentGrid.ItemsSource = s.Recent.Select(r => new
        {
            Time = r.Started.ToLocalTime().ToString("HH:mm:ss"),
            Model = r.Model ?? "–",
            Endpoint = $"{r.Method} {r.Path}",
            StatusText = r.Status is { } code ? (code < 400 ? $"{code} {r.FinishReason}".Trim() : $"{code} error") : "…",
            StatusBrush = Res(r.Status is >= 500 ? "BadBrush" : r.Status is >= 400 ? "WarnBrush" : "GoodBrush"),
            PromptTokens = r.PromptTokens > 0 ? r.PromptTokens.ToString("N0") : "",
            CompletionTokens = r.CompletionTokens > 0 ? r.CompletionTokens.ToString("N0") : "",
            Ttft = r.TtftSeconds is { } t ? $"{t:0.00} s" : "",
            Decode = r.DecodeTps is { } d ? $"{d:0.#} tok/s" : "",
            Duration = Format.Duration(TimeSpan.FromSeconds(r.Seconds)),
        }).ToList();
    }

    private void ShowGpu()
    {
        if (!_haveGpu || _gpu is null)
        {
            GpuValue.Text = "None";
            GpuSub.Text = "No NVIDIA GPU (or nvidia-smi) found";
            SecondChartTitle.Text = "Server CPU use (%), last 5 minutes";
            return;
        }
        GpuValue.Text = $"{_gpu.UtilisationPercent}%";
        GpuSub.Text = $"{Format.Bytes(_gpu.UsedBytes)} of {Format.Bytes(_gpu.TotalBytes)}" +
                      (_serverVram is { } v ? $" ({Format.Bytes(v)} the server's)" : "") +
                      (_gpu.TemperatureC is { } c ? $" · {c}°C" : "");
        if (_tick % 2 == 1)
        {
            LoadChart.Add(_gpu.UtilisationPercent);
        }
    }

    private static string Ago(DateTime utc)
    {
        var span = DateTime.UtcNow - utc.ToUniversalTime();
        return span.TotalSeconds < 5 ? "just now" : Format.Duration(span) + " ago";
    }
}
