using System.Text.Json;

namespace Amql.Cli;

/// <summary>How a call's arguments are written.</summary>
public enum ArgumentStyle
{
    /// <summary>One JSON object.</summary>
    Json,

    /// <summary>One object with bare keys and strings between marks, nothing escaped (Gemma 4).</summary>
    Marked,

    /// <summary>Each parameter by itself, strings as they are and anything else as JSON (Qwen 3.5's XML).</summary>
    Pairs,

    /// <summary>As in a Python call: <c>key=value, key=value</c> (LFM 2.5).</summary>
    Python,
}

/// <summary>
/// How a chat template writes calls, learned from what it makes of probe calls. A reply's calls are
/// <c>SectionOpen (Lead NAME AfterName arguments End) Separator (…) SectionClose</c>; where a template
/// wraps each call by itself the section's marks are empty.
/// </summary>
public sealed record CallShape
{
    public required string SectionOpen { get; init; }
    public required string Lead { get; init; }
    public required string AfterName { get; init; }
    public required string End { get; init; }
    public required string Separator { get; init; }
    public required string SectionClose { get; init; }
    public required ArgumentStyle Style { get; init; }

    /// <summary>What strings stand between (<see cref="ArgumentStyle.Marked"/>).</summary>
    public string Quote { get; init; } = "";

    /// <summary>A parameter's parts (<see cref="ArgumentStyle.Pairs"/>): what comes before its name, between
    /// name and value, after the value, and between two of them.</summary>
    public string PairOpen { get; init; } = "";

    public string KeyEnd { get; init; } = "";

    /// <summary>A parameter's value that is true, false or nothing is written as Python names it, and one that is an
    /// object or array as Python prints it (<see cref="ArgumentStyle.Pairs"/>).</summary>
    public bool PythonNames { get; init; }

    public bool PythonObjects { get; init; }
    public string PairClose { get; init; } = "";
    public string PairSeparator { get; init; } = "";

    /// <summary>What a reply is watched for: the first mark of its calls.</summary>
    public required string Opener { get; init; }

    /// <summary>The opener is no mark of its own (a bare JSON object), so it counts only where the reply starts.</summary>
    public required bool AtStartOnly { get; init; }

    /// <summary>The last mark of a call or of the section, which ends what a call may write.</summary>
    public string Closer => (SectionOpen.Length > 0 ? End + SectionClose : End).Trim();

    /// <summary>What stands between the opener and the first call's name.</summary>
    public string AfterOpener => (SectionOpen + Lead).TrimStart()[Opener.Length..];

    /// <summary>The start of a call to <paramref name="tool"/> (null: to any tool).</summary>
    public string Start(string? tool) => SectionOpen + Lead + (tool is null ? "" : tool + AfterName + (Style is ArgumentStyle.Json or ArgumentStyle.Marked ? "{" : ""));

    /// <summary>Every fixed piece, for what special tokens are among them.</summary>
    public IEnumerable<string> Pieces => [SectionOpen, Lead, AfterName, End, Separator, SectionClose, Quote, PairOpen, KeyEnd, PairClose, PairSeparator];

    // ── learning it ───────────────────────────────────────────────────────

    private const string Alpha = "qzalphaq", Beta = "qzbetaq", KeyA = "qzkeya", KeyB = "qzkeyb", ValueA = "qzvala", ValueB = "qzvalb";

    /// <summary>The tools the probes call.</summary>
    internal static IReadOnlyList<JsonElement> ProbeTools { get; } = JsonDocument.Parse(("""
        [{"type":"function","function":{"name":"ALPHA","description":"qzd","parameters":{"type":"object","properties":{"KEYA":{"type":"string","description":"qz"},"KEYB":{"type":"string","description":"qz"}},"required":[]}}},
         {"type":"function","function":{"name":"BETA","description":"qzd","parameters":{"type":"object","properties":{},"required":[]}}}]
        """).Replace("ALPHA", Alpha).Replace("BETA", Beta).Replace("KEYA", KeyA).Replace("KEYB", KeyB)).RootElement.EnumerateArray().Select(e => e.Clone()).ToList();

    internal static string ProbeName => Alpha;

    internal static ChatToolCall Probe(string name, string arguments) =>
        new(name, JsonDocument.Parse(arguments).RootElement.Clone()) { Id = name == Alpha ? "qzid00001" : "qzid00002" };

    /// <summary>
    /// The shape of what <paramref name="write"/> makes of calls (an assistant turn's text that holds only
    /// those calls, from where its content would start to where its turn would end; null when the template
    /// refuses), or null with the reason when the calls cannot be told apart from it.
    /// </summary>
    internal static CallShape? Learn(Func<ChatToolCall[], string?> write, out string? reason)
    {
        reason = null;
        string? none = write([Probe(Alpha, "{}")]);
        string? one = write([Probe(Alpha, $"{{\"{KeyA}\":\"{ValueA}\"}}")]);
        string? two = write([Probe(Alpha, $"{{\"{KeyA}\":\"{ValueA}\",\"{KeyB}\":\"{ValueB}\"}}")]);
        string? number = write([Probe(Alpha, $"{{\"{KeyA}\":7321}}")]);
        if (none is null || one is null || two is null || number is null || !none.Contains(Alpha) || !one.Contains(ValueA))
        {
            reason = "its chat template does not write calls";
            return null;
        }
        (string lead, string after0) = Around(none, Alpha);
        string after1 = Around(one, Alpha).After, after2 = Around(two, Alpha).After, after3 = Around(number, Alpha).After;

        // What stands between two calls is the end of one, perhaps a separator, and the start of the next.
        string sectionOpen = "", separator = "", sectionClose = "", call0 = after0;
        if (write([Probe(Alpha, "{}"), Probe(Beta, "{}")]) is { } pair && pair.Contains(Beta))
        {
            string between = Around(pair, Alpha).After;
            (between, string final) = Around(between, Beta);
            string next = lead[^CommonEnd(between, lead)..];
            next = next.Length < lead.Length ? FromMark(next) : next;
            sectionOpen = lead[..(lead.Length - next.Length)];
            string ended = between[..(between.Length - next.Length)];
            string end = ended[..CommonStart(ended, final)];
            separator = ended[end.Length..];
            call0 = end;
            sectionClose = sectionOpen.Length > 0 ? final[end.Length..] : "";
            lead = next;
        }
        else
        {
            call0 = after0;
        }

        var shape = Json(after1) ?? Marked(after1);
        string quote = "", pairOpen = "", keyEnd = "", pairClose = "", pairSeparator = "", afterName, callEnd;
        ArgumentStyle style;
        if (shape is { } found)
        {
            (style, afterName, quote) = found;
            if (!call0.StartsWith(afterName + "{}", StringComparison.Ordinal))
            {
                reason = "a call without arguments is written another way than one with them";
                return null;
            }
            callEnd = call0[(afterName.Length + 2)..];
        }
        else
        {
            (string pre, string rest) = Around(after1, KeyA);
            (string mid, string close) = Around(rest, ValueA);
            string between = Around(Around(after2, ValueA).After, KeyB).Before;
            if (!Around(after2, ValueA).After.Contains(KeyB) || !rest.Contains(ValueA))
            {
                reason = "its calls' arguments are not written in a way that can be read back";
                return null;
            }
            // pre is what follows the name and then what opens a parameter; the second also ends `between`.
            int cut = -1;
            for (int k = 0; k <= CommonStart(call0, pre); k++)
            {
                if (between.EndsWith(pre[k..], StringComparison.Ordinal) && (cut < 0 || Boundary(pre, k) >= Boundary(pre, cut)))
                {
                    cut = k;
                }
            }
            if (cut < 0)
            {
                reason = "its calls' parameters do not open the same way each";
                return null;
            }
            afterName = pre[..cut];
            pairOpen = pre[cut..];
            callEnd = call0[afterName.Length..];
            string closing = between[..(between.Length - pairOpen.Length)];
            string midNumber = after3.Contains(KeyA) && after3.Contains("7321") ? Around(Around(after3, KeyA).After, "7321").Before : mid;
            if (mid != midNumber && mid.StartsWith(midNumber, StringComparison.Ordinal) && mid[midNumber.Length..] is "'" or "\"")
            {
                style = ArgumentStyle.Python;
                keyEnd = midNumber;
                quote = mid[midNumber.Length..];
                pairSeparator = closing.StartsWith(quote, StringComparison.Ordinal) ? closing[1..] : closing;
            }
            else
            {
                style = ArgumentStyle.Pairs;
                keyEnd = mid;
                // After the value comes its close, then the separator or the end of the call.
                pairClose = closing[..CommonStart(closing, close)];
                if (callEnd.Length > 0 && close.IndexOf(callEnd, StringComparison.Ordinal) is var at and >= 0 && at <= pairClose.Length)
                {
                    pairClose = close[..at];
                }
                pairSeparator = closing[pairClose.Length..];
            }
        }

        string whole = (sectionOpen + lead).TrimStart();
        if (whole.Length == 0 || whole[0] is not ('<' or '[' or '{'))
        {
            reason = "its calls do not start with a mark of their own";
            return null;
        }
        int marked = whole[0] == '<' ? whole.IndexOf('>') : whole[0] == '[' ? whole.IndexOf(']') : -1;
        bool bare = whole[0] != '<' && (marked < 0 || whole[..marked].IndexOfAny(['{', '"', '\'']) >= 0);
        if (bare && whole.IndexOf('"') < 0)
        {
            reason = "its calls do not start with a mark of their own";
            return null;
        }
        if (style != ArgumentStyle.Json && !System.Text.RegularExpressions.Regex.IsMatch(sectionOpen.Length > 0 ? callEnd + sectionClose : callEnd, @"<[^<>\s]+>|\[/?[A-Z][A-Z_]+\]"))
        {
            reason = "its calls do not end with a mark of their own";
            return null;
        }
        string Value(string json) => write([Probe(Alpha, $"{{\"{KeyA}\":{json}}}")]) is { } w && Around(w, KeyA).After is { Length: > 0 } v ? v : "";
        return new CallShape
        {
            PythonNames = style == ArgumentStyle.Pairs && Value("true").Contains("True"),
            PythonObjects = style == ArgumentStyle.Pairs && Value($"[\"{ValueA}\"]").Contains($"['{ValueA}']"),
            SectionOpen = sectionOpen, Lead = lead, AfterName = afterName, End = callEnd, Separator = separator, SectionClose = sectionClose,
            Style = style, Quote = quote, PairOpen = pairOpen, KeyEnd = keyEnd, PairClose = pairClose, PairSeparator = pairSeparator,
            Opener = marked >= 0 && !bare ? whole[..(marked + 1)] : whole.TrimEnd(), AtStartOnly = bare,
        };
    }

    private static (ArgumentStyle, string, string)? Json(string after)
    {
        for (int i = after.IndexOf('{'); i >= 0; i = after.IndexOf('{', i + 1))
        {
            int end = Balanced(after, i);
            if (end < 0)
            {
                return null;
            }
            try
            {
                using var doc = JsonDocument.Parse(after[i..end]);
                if (doc.RootElement.EnumerateObject().ToList() is [{ } only] && only.Name == KeyA && only.Value.GetString() == ValueA)
                {
                    return (ArgumentStyle.Json, after[..i], "");
                }
            }
            catch (Exception e) when (e is JsonException or InvalidOperationException)
            {
            }
        }
        return null;
    }

    private static (ArgumentStyle, string, string)? Marked(string after)
    {
        int key = after.IndexOf("{" + KeyA + ":", StringComparison.Ordinal);
        int value = key < 0 ? -1 : after.IndexOf(ValueA, key, StringComparison.Ordinal);
        if (value < 0)
        {
            return null;
        }
        string quote = after[(key + KeyA.Length + 2)..value];
        return quote.Length > 0 && after[(value + ValueA.Length)..].StartsWith(quote + "}", StringComparison.Ordinal)
            ? (ArgumentStyle.Marked, after[..key], quote) : null;
    }

    /// <summary>Where the JSON object or array at <paramref name="at"/> ends, or −1 while it has not.</summary>
    public static int Balanced(string s, int at)
    {
        int depth = 0;
        bool text = false;
        for (int i = at; i < s.Length; i++)
        {
            char c = s[i];
            if (text)
            {
                if (c == '\\')
                {
                    i++;
                }
                else if (c == '"')
                {
                    text = false;
                }
            }
            else if (c == '"')
            {
                text = true;
            }
            else if (c is '{' or '[')
            {
                depth++;
            }
            else if (c is '}' or ']' && --depth == 0)
            {
                return i + 1;
            }
        }
        return -1;
    }

    private static (string Before, string After) Around(string s, string what)
    {
        int i = s.IndexOf(what, StringComparison.Ordinal);
        return i < 0 ? (s, "") : (s[..i], s[(i + what.Length)..]);
    }

    internal static int CommonStart(string a, string b)
    {
        int n = 0;
        while (n < a.Length && n < b.Length && a[n] == b[n])
        {
            n++;
        }
        return n;
    }

    internal static int CommonEnd(string a, string b)
    {
        int n = 0;
        while (n < a.Length && n < b.Length && a[^(n + 1)] == b[^(n + 1)])
        {
            n++;
        }
        return n;
    }

    /// <summary>Text cut in the middle of a mark, from the next mark on.</summary>
    private static string FromMark(string s)
    {
        int i = s.IndexOfAny(['<', '[', '{']);
        return i < 0 ? s : s[i..];
    }

    /// <summary>How good a place to cut is: before a mark, then after blank space, then anywhere.</summary>
    private static int Boundary(string s, int at) => at < s.Length && s[at] is '<' or '[' ? 2 : at > 0 && char.IsWhiteSpace(s[at - 1]) ? 1 : 0;
}
