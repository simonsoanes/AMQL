using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Amql.Server;

/// <summary>
/// A per-line decoration that a harness's tool results carry and a file does not: line numbers and
/// hashes such as <c>L42:a3f9c1│</c>, or <c>  42→</c>. A small model that reads a file and then writes
/// it back copies the decoration with the code, and the build fails on every line. The shape is
/// learned from the request's own tool results (no harness is named here), and a call's value that
/// carries it on most of its lines has it removed, along with the result's undecorated header lines
/// copied above it.
/// </summary>
public sealed class LineDecoration
{
    private const int MinLines = 5;
    private const int MaxPrefix = 24;

    private readonly Regex[] _prefixes;
    private readonly HashSet<string> _headers;

    private LineDecoration(Regex[] prefixes, HashSet<string> headers)
    {
        _prefixes = prefixes;
        _headers = headers;
    }

    /// <summary>The longest learned prefix, as a pattern (for logs and tests).</summary>
    public string Pattern => _prefixes[0].ToString();

    /// <summary>
    /// The decoration the tool results share, or null when none does. Every string in a result
    /// (a JSON result's string values, else the text) with at least five lines is a candidate; a shape
    /// is a line start up to a separator, with digit and hex runs generalised. It is a decoration when
    /// at least 80% of a string's lines begin with it and it has a varying part (a line number), so a
    /// constant start such as <c>// </c> never is.
    /// </summary>
    public static LineDecoration? Learn(IEnumerable<string> toolResults)
    {
        var votes = new Dictionary<string, int>(StringComparer.Ordinal);
        var headers = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        foreach (string result in toolResults)
        {
            foreach (string text in Strings(result))
            {
                string[] lines = text.Split('\n');
                if (lines.Length < MinLines)
                {
                    continue;
                }
                var counts = new Dictionary<string, int>(StringComparer.Ordinal);
                foreach (string line in lines)
                {
                    foreach (string shape in Shapes(line))
                    {
                        counts[shape] = counts.GetValueOrDefault(shape) + 1;
                    }
                }
                foreach (var (shape, n) in counts)
                {
                    if (n >= MinLines && n >= 0.8 * lines.Count(l => l.Length > 0))
                    {
                        votes[shape] = votes.GetValueOrDefault(shape) + 1;
                        var re = new Regex("^" + shape);
                        var top = headers.TryGetValue(shape, out var h) ? h : headers[shape] = new HashSet<string>(StringComparer.Ordinal);
                        foreach (string line in lines)
                        {
                            if (re.IsMatch(line))
                            {
                                break;
                            }
                            if (line.Trim().Length > 0)
                            {
                                top.Add(line.TrimEnd('\r'));
                            }
                        }
                    }
                }
            }
        }
        if (votes.Count == 0)
        {
            return null;
        }
        // Every shape is kept, longest first: "L\d+:[0-9a-f]+│" is tried before its own beginning
        // "L\d+:", which a model that drops the hash but keeps the line number still writes.
        var shapes = votes.Keys.OrderByDescending(k => k.Length).ThenBy(k => k, StringComparer.Ordinal).ToArray();
        var all = new HashSet<string>(shapes.SelectMany(k => headers[k]), StringComparer.Ordinal);
        return new LineDecoration(shapes.Select(k => new Regex("^" + k, RegexOptions.Compiled)).ToArray(), all);
    }

    /// <summary>
    /// The value with the decoration removed, when at least 80% of its non-blank lines (three or more)
    /// carry it (the longest learned shape that does); header lines copied from a decorated result
    /// are dropped from its top. A copy of a copy, decorated twice, is stripped twice. Otherwise the
    /// value as it was.
    /// </summary>
    public string Strip(string value, out bool stripped)
    {
        stripped = false;
        for (int pass = 0; pass < 3; pass++)
        {
            string[] lines = value.Split('\n');
            int nonBlank = lines.Count(l => l.Trim().Length > 0);
            var prefix = _prefixes.FirstOrDefault(re =>
            {
                int decorated = lines.Count(l => re.IsMatch(l));
                return decorated >= 3 && decorated >= 0.8 * nonBlank;
            });
            if (prefix is null)
            {
                break;
            }
            stripped = true;
            value = StripWith(prefix, lines, value.Length);
        }
        return value;
    }

    private string StripWith(Regex prefix, string[] lines, int length)
    {
        var sb = new StringBuilder(length);
        bool top = true;
        bool first = true;
        foreach (string line in lines)
        {
            var m = prefix.Match(line);
            if (top && !m.Success && (line.Trim().Length == 0 || _headers.Contains(line.TrimEnd('\r'))))
            {
                continue;   // a copied header ("# 130 lines · LF"), or a blank line before the code
            }
            top = false;
            if (!first)
            {
                sb.Append('\n');
            }
            first = false;
            sb.Append(m.Success ? line[m.Length..] : line);
        }
        return sb.ToString();
    }

    // A line's candidate shapes: each start ending in a separator (punctuation other than a letter,
    // digit, space or the start of an identifier) within the first characters, with a varying part.
    private static IEnumerable<string> Shapes(string line)
    {
        int limit = Math.Min(line.Length, MaxPrefix);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 1; i < limit; i++)
        {
            char c = line[i];
            if (char.IsLetterOrDigit(c) || c is ' ' or '_' or '\t')
            {
                continue;
            }
            string head = line[..(i + 1)];
            if (!head.Any(char.IsDigit))
            {
                continue;
            }
            string shape = Generalise(head);
            if (seen.Add(shape))
            {
                yield return shape;
            }
        }
    }

    // Runs of hex with a digit become a hex class, runs of digits a digit class (a line number's width
    // varies), leading spaces any number of spaces; everything else is literal.
    private static string Generalise(string head)
    {
        var sb = new StringBuilder();
        int i = 0;
        while (i < head.Length)
        {
            char c = head[i];
            if (c == ' ' && i == 0)
            {
                while (i < head.Length && head[i] == ' ')
                {
                    i++;
                }
                sb.Append(" *");
                continue;
            }
            if (Uri.IsHexDigit(c))
            {
                int j = i;
                while (j < head.Length && Uri.IsHexDigit(head[j]))
                {
                    j++;
                }
                string run = head[i..j];
                // A run standing alone between punctuation is a number or a hash, the same class
                // whatever its digits happen to be ("e3b0c4", "109278", even "abcdef"); one touching
                // a letter is not ("L12" is "L" then a number).
                bool bounded = (i == 0 || !char.IsLetterOrDigit(head[i - 1])) && (j == head.Length || !char.IsLetterOrDigit(head[j]));
                if (bounded && (run.Any(char.IsDigit) || run.Length >= 4))
                {
                    sb.Append("[0-9a-fA-F]+");
                    i = j;
                    continue;
                }
            }
            if (char.IsDigit(c))
            {
                while (i < head.Length && char.IsDigit(head[i]))
                {
                    i++;
                }
                sb.Append(@"\d+");
                continue;
            }
            sb.Append(Regex.Escape(c.ToString()));
            i++;
        }
        return sb.ToString();
    }

    // The strings a result holds: a JSON result's string values (a file's content is one), else the text.
    private static IEnumerable<string> Strings(string result)
    {
        JsonDocument? doc = null;
        try
        {
            string t = result.TrimStart();
            if (t.StartsWith('{') || t.StartsWith('['))
            {
                doc = JsonDocument.Parse(result);
            }
        }
        catch (JsonException)
        {
        }
        if (doc is null)
        {
            yield return result;
            yield break;
        }
        using (doc)
        {
            var stack = new Stack<JsonElement>();
            stack.Push(doc.RootElement);
            var found = new List<string>();
            while (stack.Count > 0)
            {
                var e = stack.Pop();
                switch (e.ValueKind)
                {
                    case JsonValueKind.Object:
                        foreach (var p in e.EnumerateObject())
                        {
                            stack.Push(p.Value);
                        }
                        break;
                    case JsonValueKind.Array:
                        foreach (var v in e.EnumerateArray())
                        {
                            stack.Push(v);
                        }
                        break;
                    case JsonValueKind.String:
                        found.Add(e.GetString()!);
                        break;
                }
            }
            foreach (string s in found)
            {
                yield return s;
            }
        }
    }
}
