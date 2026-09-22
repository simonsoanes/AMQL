using System.Text.Json;

namespace Amql.Cli;

/// <summary>
/// The machine-readable progress/result protocol (TODO.md item 1).
///
/// Off by default; enabled by the global <c>--progress</c> flag or
/// <c>AMQL_PROGRESS=1</c>. When enabled, one JSON object per line is written
/// to stdout with a fixed prefix, so front-ends can bind progress exactly
/// instead of scraping 'NN%' / 'N/M' out of human-readable text:
///
///   ##amql-progress {"command":"route","phase":"probe","done":3,"total":8,"percent":37.5}
///   ##amql-result   {"command":"path","found":false,"nodes":48}
///
/// Protocol lines are newline-terminated and flushed immediately. Fragment
/// progress (route's template probing, path's search ticks) is written through
/// <see cref="Fragment"/>: raw when attached to a console (keeping the inline
/// '···' display), newline-terminated when redirected so every update is
/// visible as it happens rather than at process exit (TODO.md item 5).
/// </summary>
internal static class CliProgress
{
    public const string ProgressPrefix = "##amql-progress ";
    public const string ResultPrefix = "##amql-result ";

    /// <summary>Protocol emission is on (--progress / AMQL_PROGRESS=1).</summary>
    public static bool Enabled { get; private set; }

    /// <summary>--verbose: print full stack traces for unexpected errors (TODO.md item 2).</summary>
    public static bool Verbose { get; private set; }

    private static readonly object Gate = new();
    private static string _command = string.Empty;
    private static string _phase = string.Empty;
    private static long _done;
    private static long _total = -1;
    private static bool _fragmentPending;

    private static readonly JsonSerializerOptions Json = new()
    {
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
        // camelCase so the wire format is stable regardless of the C#
        // property names, and matches what front-ends expect.
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };

    public static void Configure(string command, bool enabled, bool verbose)
    {
        lock (Gate)
        {
            _command = command;
            Enabled = enabled;
            Verbose = verbose;
        }
    }

    /// <summary>Announces a phase; <paramref name="total"/> &lt; 0 means indeterminate.</summary>
    public static void Phase(string phase, long total = -1)
    {
        lock (Gate)
        {
            _phase = phase;
            _done = 0;
            _total = total;
            EmitProgress(detail: null);
        }
    }

    /// <summary>Reports progress within the current phase.</summary>
    public static void Advance(long done, long total = -1, string? detail = null)
    {
        lock (Gate)
        {
            _done = done;
            if (total >= 0)
            {
                _total = total;
            }
            EmitProgress(detail);
        }
    }

    /// <summary>Terminal phase marker (also ends any pending fragment line).</summary>
    public static void Complete(string? detail = null)
    {
        lock (Gate)
        {
            if (_total > 0)
            {
                _done = _total;
            }
            _phase = "complete";
            EmitProgress(detail);
            EndFragmentLine();
        }
    }

    /// <summary>
    /// Machine-readable outcome of the command (TODO.md item 6): lets a
    /// front-end distinguish a legitimate negative result from an error
    /// without parsing prose, e.g. path's "no path found within the budget".
    /// </summary>
    public static void Result(object payload)
    {
        lock (Gate)
        {
            EndFragmentLine();
            Write(ResultPrefix, payload);
        }
    }

    /// <summary>
    /// Writes one progress fragment. Console-attached output stays inline;
    /// redirected output newline-terminates each fragment so it is visible
    /// immediately (a GUI/piped consumer never waits for process exit).
    /// </summary>
    public static void Fragment(string text)
    {
        lock (Gate)
        {
            EndFragmentLineIfProtocolPending(text);
            if (Console.IsOutputRedirected)
            {
                Console.Out.Write(text.EndsWith('\n') ? text : text + Environment.NewLine);
                _fragmentPending = false;
            }
            else
            {
                Console.Out.Write(text);
                _fragmentPending = !text.EndsWith('\n');
            }
            Console.Out.Flush();
        }
    }

    /// <summary>The fragment sink handed to RelationRouter / PathFinder in place of Console.Write.</summary>
    public static Action<string> FragmentWriter => Fragment;

    /// <summary>Ends a pending unterminated fragment line (called before protocol output and at exit).</summary>
    public static void EndFragmentLine()
    {
        lock (Gate)
        {
            if (_fragmentPending)
            {
                Console.Out.WriteLine();
                Console.Out.Flush();
                _fragmentPending = false;
            }
        }
    }

    private static void EndFragmentLineIfProtocolPending(string text)
    {
        // A fragment continuing an unterminated line keeps it open; anything
        // else closes it first so the streams never glue together.
        if (_fragmentPending && !text.StartsWith('·') && !text.StartsWith("\n"))
        {
            Console.Out.WriteLine();
            _fragmentPending = false;
        }
    }

    private static void EmitProgress(string? detail)
    {
        if (!Enabled)
        {
            return;
        }
        EndFragmentLine();
        double? percent = _total > 0 ? Math.Round(100.0 * Math.Min(_done, _total) / _total, 2) : null;
        Write(ProgressPrefix, new ProgressPayload
        {
            Command = _command,
            Phase = _phase,
            Done = _done,
            Total = _total >= 0 ? _total : null,
            Percent = percent,
            Detail = detail,
        });
    }

    private static void Write(string prefix, object payload)
    {
        if (_fragmentPending)
        {
            Console.Out.WriteLine();
            _fragmentPending = false;
        }
        try
        {
            Console.Out.Write(prefix);
            Console.Out.WriteLine(JsonSerializer.Serialize(payload, Json));
            Console.Out.Flush();
        }
        catch (Exception e) when (e is IOException or JsonException)
        {
            // The protocol is a convenience channel — never fail a run over it.
        }
    }

    private sealed record ProgressPayload
    {
        public string? Command { get; init; }
        public string? Phase { get; init; }
        public long? Done { get; init; }
        public long? Total { get; init; }
        public double? Percent { get; init; }
        public string? Detail { get; init; }
    }
}
