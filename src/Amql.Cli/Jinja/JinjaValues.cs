using System.Collections;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace Amql.Cli.Jinja;

// Values are .NET's own where one stands for the Python one: null for None, bool, long, double,
// string; and the classes below. Lists and dictionaries are never changed by a template.

/// <summary>A name or member that is not there.</summary>
internal sealed class Undefined(string what)
{
    public string What { get; } = what;
}

internal sealed class JList(List<object?> items, bool tuple = false)
{
    public List<object?> Items { get; } = items;

    public bool Tuple { get; } = tuple;
}

/// <summary>A dictionary in the order its keys were given.</summary>
internal sealed class JDict
{
    private readonly Dictionary<object, int> _at = new(KeyComparer.Instance);

    public List<(object? Key, object? Value)> Items { get; } = new();

    public void Set(object? key, object? value)
    {
        object k = key ?? DBNull.Value;
        if (_at.TryGetValue(k, out int i))
        {
            Items[i] = (Items[i].Key, value);
            return;
        }
        _at[k] = Items.Count;
        Items.Add((key, value));
    }

    public bool TryGet(object? key, out object? value)
    {
        if (_at.TryGetValue(key ?? DBNull.Value, out int i))
        {
            value = Items[i].Value;
            return true;
        }
        value = null;
        return false;
    }

    private sealed class KeyComparer : IEqualityComparer<object>
    {
        public static readonly KeyComparer Instance = new();

        public new bool Equals(object? a, object? b) => Py.AreEqual(a is DBNull ? null : a, b is DBNull ? null : b);

        public int GetHashCode(object o) => o switch
        {
            string s => s.GetHashCode(),
            bool b => b ? 1 : 0,
            long l => l.GetHashCode(),
            double d => d == Math.Floor(d) && Math.Abs(d) < 9e18 ? ((long)d).GetHashCode() : d.GetHashCode(),
            _ => 0,
        };
    }
}

/// <summary>A <c>namespace()</c>, the one thing a template can assign into.</summary>
internal sealed class JNamespace
{
    public Dictionary<string, object?> Members { get; } = new(StringComparer.Ordinal);
}

internal delegate object? Callable(List<object?> args, Dictionary<string, object?> named);

/// <summary>Python's behaviour, as far as templates reach it.</summary>
internal static class Py
{
    public static object? FromJson(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.String => e.GetString(),
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        JsonValueKind.Number => e.GetRawText().AsSpan().IndexOfAny('.', 'e', 'E') < 0 && e.TryGetInt64(out long l) ? (object)l : e.GetDouble(),
        JsonValueKind.Array => new JList(e.EnumerateArray().Select(FromJson).ToList()),
        JsonValueKind.Object => Dict(e),
        _ => null,
    };

    private static JDict Dict(JsonElement e)
    {
        var d = new JDict();
        foreach (var p in e.EnumerateObject())
        {
            d.Set(p.Name, FromJson(p.Value));
        }
        return d;
    }

    public static bool Truthy(object? v) => v switch
    {
        null or Undefined => false,
        bool b => b,
        long l => l != 0,
        double d => d != 0,
        string s => s.Length > 0,
        JList l => l.Items.Count > 0,
        JDict d => d.Items.Count > 0,
        _ => true,
    };

    public static bool IsNumber(object? v) => v is long or double or bool;

    public static double Num(object? v) => v switch
    {
        long l => l,
        double d => d,
        bool b => b ? 1 : 0,
        _ => throw new JinjaException($"{TypeName(v)} is not a number"),
    };

    public static long Int(object? v) => v switch
    {
        long l => l,
        bool b => b ? 1 : 0,
        _ => throw new JinjaException($"{TypeName(v)} is not an integer"),
    };

    public static string TypeName(object? v) => v switch
    {
        null => "None",
        Undefined u => $"'{u.What}', which is undefined,",
        bool => "a boolean",
        long => "an integer",
        double => "a number",
        string => "a string",
        JList { Tuple: true } => "a tuple",
        JList => "a list",
        JDict => "a dictionary",
        JNamespace => "a namespace",
        _ => "a function",
    };

    public static bool AreEqual(object? a, object? b)
    {
        if (a is Undefined || b is Undefined)
        {
            return a is Undefined && b is Undefined;
        }
        if (a is null || b is null)
        {
            return a is null && b is null;
        }
        if (IsNumber(a) && IsNumber(b))
        {
            return a is double || b is double ? Num(a) == Num(b) : Int(a) == Int(b);
        }
        return (a, b) switch
        {
            (string x, string y) => x == y,
            (JList x, JList y) => x.Tuple == y.Tuple && x.Items.Count == y.Items.Count && x.Items.Zip(y.Items).All(p => AreEqual(p.First, p.Second)),
            (JDict x, JDict y) => x.Items.Count == y.Items.Count && x.Items.All(p => y.TryGet(p.Key, out var v) && AreEqual(p.Value, v)),
            _ => ReferenceEquals(a, b),
        };
    }

    public static int CompareValues(object? a, object? b)
    {
        if (IsNumber(a) && IsNumber(b))
        {
            return Num(a).CompareTo(Num(b));
        }
        if (a is string x && b is string y)
        {
            return CompareText(x, y);
        }
        if (a is JList l && b is JList m)
        {
            for (int i = 0; i < Math.Min(l.Items.Count, m.Items.Count); i++)
            {
                int c = AreEqual(l.Items[i], m.Items[i]) ? 0 : CompareValues(l.Items[i], m.Items[i]);
                if (c != 0)
                {
                    return c;
                }
            }
            return l.Items.Count.CompareTo(m.Items.Count);
        }
        throw new JinjaException($"{TypeName(a)} and {TypeName(b)} have no order");
    }

    /// <summary>Strings by code point, as Python orders them.</summary>
    public static int CompareText(string a, string b)
    {
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++)
        {
            if (a[i] != b[i])
            {
                bool sa = char.IsSurrogate(a[i]), sb = char.IsSurrogate(b[i]);
                return sa == sb ? a[i].CompareTo(b[i]) : sa ? 1 : -1;
            }
        }
        return a.Length.CompareTo(b.Length);
    }

    public static string Str(object? v) => v switch
    {
        null => "None",
        Undefined => "",
        string s => s,
        _ => Repr(v),
    };

    public static string Repr(object? v)
    {
        switch (v)
        {
            case null: return "None";
            case Undefined: return "Undefined";
            case bool b: return b ? "True" : "False";
            case long l: return l.ToString(CultureInfo.InvariantCulture);
            case double d: return PyFormat.FloatRepr(d);
            case string s: return PyFormat.StringRepr(s);
            case JList l:
            {
                string inner = string.Join(", ", l.Items.Select(Repr));
                return l.Tuple ? l.Items.Count == 1 ? $"({inner},)" : $"({inner})" : $"[{inner}]";
            }
            case JDict d: return "{" + string.Join(", ", d.Items.Select(p => Repr(p.Key) + ": " + Repr(p.Value))) + "}";
            case JNamespace n: return "<Namespace {" + string.Join(", ", n.Members.Select(p => PyFormat.StringRepr(p.Key) + ": " + Repr(p.Value))) + "}>";
            default: return "<function>";
        }
    }

    public static IEnumerable<object?> Iterate(object? v) => v switch
    {
        JList l => l.Items,
        JDict d => d.Items.Select(p => p.Key),
        string s => Characters(s).Select(c => (object?)c),
        Undefined => [],
        _ => throw new JinjaException($"{TypeName(v)} cannot be gone through"),
    };

    public static IEnumerable<string> Characters(string s)
    {
        for (int i = 0; i < s.Length; i++)
        {
            if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
            {
                yield return s.Substring(i++, 2);
            }
            else
            {
                yield return s[i].ToString();
            }
        }
    }

    public static long Length(object? v) => v switch
    {
        string s => s.Length - Surrogates(s),
        JList l => l.Items.Count,
        JDict d => d.Items.Count,
        Undefined => 0,
        _ => throw new JinjaException($"{TypeName(v)} has no length"),
    };

    private static int Surrogates(string s)
    {
        int n = 0;
        for (int i = 0; i + 1 < s.Length; i++)
        {
            if (char.IsHighSurrogate(s[i]) && char.IsLowSurrogate(s[i + 1]))
            {
                n++;
                i++;
            }
        }
        return n;
    }

    public static bool Contains(object? container, object? item)
    {
        switch (container)
        {
            case string s: return item is string sub ? s.Contains(sub, StringComparison.Ordinal) : throw new JinjaException($"{TypeName(item)} cannot be in a string");
            case JDict d: return d.TryGet(item, out _);
            default: return Iterate(container).Any(x => AreEqual(x, item));
        }
    }

    public static object? Slice(object? v, object? from, object? to, object? step)
    {
        long by = step is null or Undefined ? 1 : Int(step);
        if (by == 0)
        {
            throw new JinjaException("a slice cannot step by zero");
        }
        // Text without surrogate pairs, the usual case, is cut as it is.
        if (v is string plain && by == 1 && Surrogates(plain) == 0)
        {
            (int a, int b) = Bounds(plain.Length, from, to, 1);
            return a < b ? plain[a..b] : "";
        }
        var items = v switch
        {
            string s => Characters(s).Cast<object?>().ToList(),
            JList l => l.Items,
            Undefined u => throw new JinjaException($"'{u.What}' is undefined"),
            _ => throw new JinjaException($"{TypeName(v)} cannot be sliced"),
        };
        (int start, int stop) = Bounds(items.Count, from, to, by);
        var picked = new List<object?>();
        for (long i = start; by > 0 ? i < stop : i > stop; i += by)
        {
            picked.Add(items[(int)i]);
        }
        return v is string ? string.Concat(picked) : new JList(picked, v is JList { Tuple: true });
    }

    private static (int, int) Bounds(int n, object? from, object? to, long by)
    {
        int At(object? v, int missing)
        {
            if (v is null or Undefined)
            {
                return missing;
            }
            long i = Int(v);
            i = i < 0 ? i + n : i;
            return (int)(by > 0 ? Math.Clamp(i, 0, n) : Math.Clamp(i, -1, n - 1));
        }
        return by > 0 ? (At(from, 0), At(to, n)) : (At(from, n - 1), At(to, -1));
    }

    // ── json.dumps ──

    public static string ToJson(object? v, bool ensureAscii = false, int? indent = null, string? itemSeparator = null, string? keySeparator = null, bool sortKeys = false)
    {
        var sb = new StringBuilder();
        itemSeparator ??= indent is null ? ", " : ",";
        Json(sb, v, ensureAscii, indent, itemSeparator, keySeparator ?? ": ", sortKeys, 0);
        return sb.ToString();
    }

    private static void Json(StringBuilder sb, object? v, bool ascii, int? indent, string comma, string colon, bool sort, int depth)
    {
        if (depth > 200)
        {
            throw new JinjaException("a value nests too deeply for JSON");
        }
        void Line(int d)
        {
            if (indent is { } n)
            {
                sb.Append('\n').Append(' ', n * d);
            }
        }
        switch (v)
        {
            case null: sb.Append("null"); break;
            case bool b: sb.Append(b ? "true" : "false"); break;
            case long l: sb.Append(l.ToString(CultureInfo.InvariantCulture)); break;
            case double d: sb.Append(double.IsNaN(d) ? "NaN" : double.IsInfinity(d) ? d > 0 ? "Infinity" : "-Infinity" : PyFormat.FloatRepr(d)); break;
            case string s: PyFormat.AppendJsonString(sb, s, ascii); break;
            case JList l:
                sb.Append('[');
                for (int i = 0; i < l.Items.Count; i++)
                {
                    sb.Append(i > 0 ? comma : "");
                    Line(depth + 1);
                    Json(sb, l.Items[i], ascii, indent, comma, colon, sort, depth + 1);
                }
                if (l.Items.Count > 0)
                {
                    Line(depth);
                }
                sb.Append(']');
                break;
            case JDict d:
            {
                var items = sort ? d.Items.OrderBy(p => p.Key, Comparer<object?>.Create(CompareValues)).ToList() : d.Items;
                sb.Append('{');
                for (int i = 0; i < items.Count; i++)
                {
                    sb.Append(i > 0 ? comma : "");
                    Line(depth + 1);
                    PyFormat.AppendJsonString(sb, items[i].Key switch
                    {
                        string s => s,
                        null => "null",
                        bool b => b ? "true" : "false",
                        var k => Str(k),
                    }, ascii);
                    sb.Append(colon);
                    Json(sb, items[i].Value, ascii, indent, comma, colon, sort, depth + 1);
                }
                if (items.Count > 0)
                {
                    Line(depth);
                }
                sb.Append('}');
                break;
            }
            default:
                throw new JinjaException($"{TypeName(v)} cannot be written as JSON");
        }
    }

    // ── str's methods ──

    public static string Strip(string s, string? chars, bool left, bool right)
    {
        bool Cut(char c) => chars is null ? IsSpace(c) : chars.Contains(c);
        int a = 0, b = s.Length;
        while (left && a < b && Cut(s[a]))
        {
            a++;
        }
        while (right && b > a && Cut(s[b - 1]))
        {
            b--;
        }
        return s[a..b];
    }

    public static bool IsSpace(char c) => char.IsWhiteSpace(c) || (int)c is >= 0x1c and <= 0x1f;

    public static List<object?> Split(string s, string? separator, long most)
    {
        var parts = new List<object?>();
        if (separator is null)
        {
            int i = 0;
            while (true)
            {
                while (i < s.Length && IsSpace(s[i]))
                {
                    i++;
                }
                if (i >= s.Length)
                {
                    return parts;
                }
                if (most >= 0 && parts.Count >= most)
                {
                    parts.Add(Strip(s[i..], null, false, true));
                    return parts;
                }
                int from = i;
                while (i < s.Length && !IsSpace(s[i]))
                {
                    i++;
                }
                parts.Add(s[from..i]);
            }
        }
        if (separator.Length == 0)
        {
            throw new JinjaException("a string cannot be split at nothing");
        }
        int at = 0;
        while (most < 0 || parts.Count < most)
        {
            int next = s.IndexOf(separator, at, StringComparison.Ordinal);
            if (next < 0)
            {
                break;
            }
            parts.Add(s[at..next]);
            at = next + separator.Length;
        }
        parts.Add(s[at..]);
        return parts;
    }

    public static List<object?> SplitRight(string s, string? separator, long most)
    {
        if (most < 0 || separator is null)
        {
            return Split(s, separator, most);
        }
        var parts = new List<object?>();
        int end = s.Length;
        while (parts.Count < most)
        {
            int at = end == 0 ? -1 : s.LastIndexOf(separator, end - 1, StringComparison.Ordinal);
            if (at < 0)
            {
                break;
            }
            parts.Insert(0, s[(at + separator.Length)..end]);
            end = at;
        }
        parts.Insert(0, s[..end]);
        return parts;
    }

    public static List<object?> Lines(string s, bool keepEnds)
    {
        var lines = new List<object?>();
        int from = 0;
        for (int i = 0; i < s.Length; i++)
        {
            if ((int)s[i] is 10 or 11 or 12 or 13 or 0x1c or 0x1d or 0x1e or 0x85 or 0x2028 or 0x2029)
            {
                int end = s[i] == '\r' && i + 1 < s.Length && s[i + 1] == '\n' ? i + 2 : i + 1;
                lines.Add(s[from..(keepEnds ? end : i)]);
                from = end;
                i = end - 1;
            }
        }
        if (from < s.Length)
        {
            lines.Add(s[from..]);
        }
        return lines;
    }

    public static string Capitalize(string s) => s.Length == 0 ? s : char.ToUpperInvariant(s[0]) + s[1..].ToLowerInvariant();

    public static string Title(string s)
    {
        var sb = new StringBuilder(s.Length);
        bool inside = false;
        foreach (char c in s)
        {
            sb.Append(inside ? char.ToLowerInvariant(c) : char.ToUpperInvariant(c));
            inside = char.IsLetter(c);
        }
        return sb.ToString();
    }

    /// <summary>Jinja's own <c>title</c>: every word's first letter, whatever is in it.</summary>
    public static string TitleWords(string s) => System.Text.RegularExpressions.Regex.Replace(s, @"[^-\s({\[<]+", m => Capitalize(m.Value));
}
