using System.Globalization;
using System.Text;

namespace Amql.Cli.Jinja;

/// <summary>The filters, tests, methods and functions a template may use.</summary>
internal static class Builtins
{
    private static object? Arg(List<object?> args, Dictionary<string, object?> named, int at, string name, object? missing = null) =>
        at < args.Count ? args[at] : named.TryGetValue(name, out var v) ? v : missing;

    private static string Text(object? v, string what) => v switch
    {
        string s => s,
        Undefined u => throw new JinjaException($"'{u.What}' is undefined"),
        _ => throw new JinjaException($"{what} takes a string, not {Py.TypeName(v)}"),
    };

    private static string? TextOrNone(object? v, string what) => v is null ? null : Text(v, what);

    public static object? Range(List<object?> args, Dictionary<string, object?> named)
    {
        long from = args.Count > 1 ? Py.Int(args[0]) : 0, to = Py.Int(args.Count > 1 ? args[1] : args[0]), step = args.Count > 2 ? Py.Int(args[2]) : 1;
        if (step == 0)
        {
            throw new JinjaException("range cannot step by zero");
        }
        long count = step > 0 ? (to - from + step - 1) / step : (from - to - step - 1) / -step;
        if (count > 100_000)
        {
            throw new JinjaException("range is too long");
        }
        var items = new List<object?>();
        for (long i = from; step > 0 ? i < to : i > to; i += step)
        {
            items.Add(i);
        }
        return new JList(items);
    }

    // ── methods ───────────────────────────────────────────────────────────

    /// <summary>A method of a value, or null when it has none of that name.</summary>
    public static Callable? Method(object? owner, string name) => owner switch
    {
        string s => TextMethod(s, name),
        JDict d => DictMethod(d, name),
        JList l => ListMethod(l, name),
        _ => null,
    };

    private static Callable? ListMethod(JList l, string name) => name switch
    {
        "index" => (a, _) => l.Items.FindIndex(x => Py.AreEqual(x, a[0])) is var i and >= 0 ? (long)i : throw new JinjaException("the list does not hold that"),
        "count" => (a, _) => (long)l.Items.Count(x => Py.AreEqual(x, a[0])),
        "copy" => (_, _) => l,
        "append" or "extend" or "insert" or "pop" or "remove" or "clear" or "sort" or "reverse" when !l.Tuple =>
            (_, _) => throw new JinjaException($"a template does not change a list (.{name})"),
        _ => null,
    };

    private static Callable? DictMethod(JDict d, string name) => name switch
    {
        "items" => (_, _) => new JList(d.Items.Select(p => (object?)new JList([p.Key, p.Value], true)).ToList()),
        "keys" => (_, _) => new JList(d.Items.Select(p => p.Key).ToList()),
        "values" => (_, _) => new JList(d.Items.Select(p => p.Value).ToList()),
        "get" => (a, n) => d.TryGet(a[0], out var v) ? v : Arg(a, n, 1, "default"),
        "copy" => (_, _) => d,
        "update" or "pop" or "popitem" or "setdefault" or "clear" => (_, _) => throw new JinjaException($"a template does not change a dictionary (.{name})"),
        _ => null,
    };

    private static Callable? TextMethod(string s, string name)
    {
        string? Chars(List<object?> a) => a.Count > 0 ? TextOrNone(a[0], name) : null;
        bool Any(object? v, Func<string, bool> holds) => v is JList { Tuple: true } many ? many.Items.Any(x => holds(Text(x, name))) : holds(Text(v, name));
        return name switch
        {
            "strip" => (a, _) => Py.Strip(s, Chars(a), true, true),
            "lstrip" => (a, _) => Py.Strip(s, Chars(a), true, false),
            "rstrip" => (a, _) => Py.Strip(s, Chars(a), false, true),
            "split" => (a, n) => new JList(Py.Split(s, TextOrNone(Arg(a, n, 0, "sep"), name), Py.Int(Arg(a, n, 1, "maxsplit", -1L)))),
            "rsplit" => (a, n) => new JList(Py.SplitRight(s, TextOrNone(Arg(a, n, 0, "sep"), name), Py.Int(Arg(a, n, 1, "maxsplit", -1L)))),
            "splitlines" => (a, n) => new JList(Py.Lines(s, Py.Truthy(Arg(a, n, 0, "keepends", false)))),
            "partition" => (a, _) => s.IndexOf(Text(a[0], name), StringComparison.Ordinal) is var i and >= 0
                ? new JList([s[..i], a[0], s[(i + ((string)a[0]!).Length)..]], true) : new JList([s, "", ""], true),
            "rpartition" => (a, _) => s.LastIndexOf(Text(a[0], name), StringComparison.Ordinal) is var i and >= 0
                ? new JList([s[..i], a[0], s[(i + ((string)a[0]!).Length)..]], true) : new JList(["", "", s], true),
            "startswith" => (a, _) => Any(a[0], x => s.StartsWith(x, StringComparison.Ordinal)),
            "endswith" => (a, _) => Any(a[0], x => s.EndsWith(x, StringComparison.Ordinal)),
            "removeprefix" => (a, _) => s.StartsWith(Text(a[0], name), StringComparison.Ordinal) ? s[((string)a[0]!).Length..] : s,
            "removesuffix" => (a, _) => s.EndsWith(Text(a[0], name), StringComparison.Ordinal) ? s[..^((string)a[0]!).Length] : s,
            "replace" => (a, n) => Replaced(s, Text(a[0], name), Text(a[1], name), Arg(a, n, 2, "count", -1L)),
            "upper" => (_, _) => s.ToUpperInvariant(),
            "lower" => (_, _) => s.ToLowerInvariant(),
            "title" => (_, _) => Py.Title(s),
            "capitalize" => (_, _) => Py.Capitalize(s),
            "join" => (a, _) => string.Join(s, Py.Iterate(a[0]).Select(x => Text(x, name))),
            "find" => (a, _) => (long)s.IndexOf(Text(a[0], name), StringComparison.Ordinal),
            "rfind" => (a, _) => (long)s.LastIndexOf(Text(a[0], name), StringComparison.Ordinal),
            "index" => (a, _) => s.IndexOf(Text(a[0], name), StringComparison.Ordinal) is var i and >= 0 ? (long)i : throw new JinjaException("substring not found"),
            "count" => (a, _) => Text(a[0], name) is { Length: > 0 } sub ? (long)(s.Split(sub).Length - 1) : Py.Length(s) + 1,
            "isdigit" or "isnumeric" or "isdecimal" => (_, _) => s.Length > 0 && s.All(char.IsDigit),
            "isalpha" => (_, _) => s.Length > 0 && s.All(char.IsLetter),
            "isalnum" => (_, _) => s.Length > 0 && s.All(char.IsLetterOrDigit),
            "isspace" => (_, _) => s.Length > 0 && s.All(Py.IsSpace),
            "isupper" => (_, _) => s.Any(char.IsUpper) && !s.Any(char.IsLower),
            "islower" => (_, _) => s.Any(char.IsLower) && !s.Any(char.IsUpper),
            "zfill" => (a, _) => s.Length > 0 && s[0] is '-' or '+' ? s[0] + s[1..].PadLeft((int)Py.Int(a[0]) - 1, '0') : s.PadLeft((int)Py.Int(a[0]), '0'),
            "ljust" => (a, _) => s.PadRight((int)Py.Int(a[0]), a.Count > 1 ? Text(a[1], name)[0] : ' '),
            "rjust" => (a, _) => s.PadLeft((int)Py.Int(a[0]), a.Count > 1 ? Text(a[1], name)[0] : ' '),
            "format" => (a, n) => Formatted(s, a, n),
            _ => null,
        };
    }

    private static string Replaced(string s, string from, string to, object? count)
    {
        long most = count is null or Undefined ? -1 : Py.Int(count);
        if (from.Length == 0)
        {
            var spread = new StringBuilder();
            long left = most < 0 ? long.MaxValue : most;
            foreach (string c in Py.Characters(s))
            {
                spread.Append(left-- > 0 ? to : "").Append(c);
            }
            return spread.Append(left > 0 ? to : "").ToString();
        }
        if (most < 0)
        {
            return s.Replace(from, to, StringComparison.Ordinal);
        }
        var sb = new StringBuilder();
        int at = 0;
        for (; most > 0 && s.IndexOf(from, at, StringComparison.Ordinal) is var i and >= 0; most--)
        {
            sb.Append(s, at, i - at).Append(to);
            at = i + from.Length;
        }
        return sb.Append(s, at, s.Length - at).ToString();
    }

    /// <summary><c>'{} {0} {name}'.format(…)</c>, without format specifications.</summary>
    private static string Formatted(string s, List<object?> args, Dictionary<string, object?> named)
    {
        var sb = new StringBuilder();
        int next = 0;
        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] is '{' or '}' && i + 1 < s.Length && s[i + 1] == s[i])
            {
                sb.Append(s[i++]);
            }
            else if (s[i] == '{')
            {
                int end = s.IndexOf('}', i);
                string field = end < 0 ? throw new JinjaException("a format string is cut short") : s[(i + 1)..end];
                object? value = field.Length == 0 ? args.ElementAtOrDefault(next++)
                    : int.TryParse(field, out int at) ? args.ElementAtOrDefault(at)
                    : named.TryGetValue(field, out var v) ? v : throw new JinjaException($"a format string names '{field}', which is not given");
                sb.Append(Py.Str(value));
                i = end;
            }
            else
            {
                sb.Append(s[i]);
            }
        }
        return sb.ToString();
    }

    /// <summary><c>'%s and %d' % (…)</c>.</summary>
    public static string Percent(string format, object? value)
    {
        var args = value is JList { Tuple: true } t ? t.Items : [value];
        int next = 0;
        return System.Text.RegularExpressions.Regex.Replace(format, @"%(?:\((\w+)\))?(0?)(\d*)(?:\.(\d+))?([sdrfix%])", m =>
        {
            if (m.Groups[5].Value == "%")
            {
                return "%";
            }
            object? v = m.Groups[1].Success
                ? value is JDict d && d.TryGet(m.Groups[1].Value, out var member) ? member : throw new JinjaException($"a format string names '{m.Groups[1].Value}', which is not given")
                : next < args.Count ? args[next++] : throw new JinjaException("a format string takes more values than it is given");
            int digits = m.Groups[4].Success ? int.Parse(m.Groups[4].Value, CultureInfo.InvariantCulture) : -1;
            string s = m.Groups[5].Value switch
            {
                "d" or "i" => ((long)Math.Truncate(Py.Num(v))).ToString(CultureInfo.InvariantCulture),
                "x" => Py.Int(v).ToString("x", CultureInfo.InvariantCulture),
                "f" => Py.Num(v).ToString("F" + (digits < 0 ? 6 : digits), CultureInfo.InvariantCulture),
                "r" => Py.Repr(v),
                _ => Py.Str(v),
            };
            return m.Groups[3].Value is { Length: > 0 } width ? s.PadLeft(int.Parse(width, CultureInfo.InvariantCulture), m.Groups[2].Value == "0" ? '0' : ' ') : s;
        });
    }

    public static string StrfTime(string format, DateTime t)
    {
        var sb = new StringBuilder();
        var c = CultureInfo.InvariantCulture;
        for (int i = 0; i < format.Length; i++)
        {
            if (format[i] != '%' || i + 1 >= format.Length)
            {
                sb.Append(format[i]);
                continue;
            }
            char f = format[++i];
            bool bare = f is '-' or '#' && i + 1 < format.Length;
            if (bare)
            {
                f = format[++i];
            }
            string s = f switch
            {
                'Y' => t.ToString("yyyy", c),
                'y' => t.ToString("yy", c),
                'm' => t.ToString("MM", c),
                'd' => t.ToString("dd", c),
                'e' => t.Day.ToString(c).PadLeft(2),
                'B' => t.ToString("MMMM", c),
                'b' or 'h' => t.ToString("MMM", c),
                'A' => t.ToString("dddd", c),
                'a' => t.ToString("ddd", c),
                'H' => t.ToString("HH", c),
                'I' => t.ToString("hh", c),
                'M' => t.ToString("mm", c),
                'S' => t.ToString("ss", c),
                'f' => t.ToString("ffffff", c),
                'p' => t.ToString("tt", c),
                'j' => t.DayOfYear.ToString("000", c),
                'w' => ((int)t.DayOfWeek).ToString(c),
                'u' => (t.DayOfWeek == 0 ? 7 : (int)t.DayOfWeek).ToString(c),
                'F' => t.ToString("yyyy-MM-dd", c),
                'D' => t.ToString("MM/dd/yy", c),
                'T' => t.ToString("HH:mm:ss", c),
                'R' => t.ToString("HH:mm", c),
                'z' or 'Z' => "",
                'n' => "\n",
                't' => "\t",
                '%' => "%",
                _ => "%" + f,
            };
            sb.Append(bare ? s.TrimStart('0', ' ') is { Length: > 0 } cut ? cut : "0" : s);
        }
        return sb.ToString();
    }

    // ── tests ─────────────────────────────────────────────────────────────

    public static bool Test(string name, object? v, List<object?> args)
    {
        switch (name)
        {
            case "defined": return v is not Undefined;
            case "undefined": return v is Undefined;
            case "none": return v is null;
            case "string": return v is string;
            case "mapping": return v is JDict;
            case "sequence": return v is string or JList or JDict;
            case "iterable": return v is string or JList or JDict or Undefined;
            case "number": return Py.IsNumber(v);
            case "integer": return v is long;
            case "float": return v is double;
            case "boolean": return v is bool;
            case "true": return v is true;
            case "false": return v is false;
            case "callable": return v is Callable;
            case "odd": return Math.Abs(Py.Int(v) % 2) == 1;
            case "even": return Py.Int(v) % 2 == 0;
            case "divisibleby": return Py.Int(v) % Py.Int(args[0]) == 0;
            case "lower": return Py.Str(v) is var l && l.Any(char.IsLower) && !l.Any(char.IsUpper);
            case "upper": return Py.Str(v) is var u && u.Any(char.IsUpper) && !u.Any(char.IsLower);
            case "eq" or "equalto" or "==": return Py.AreEqual(v, args[0]);
            case "ne" or "!=": return !Py.AreEqual(v, args[0]);
            case "lt" or "lessthan" or "<": return Py.CompareValues(v, args[0]) < 0;
            case "le" or "<=": return Py.CompareValues(v, args[0]) <= 0;
            case "gt" or "greaterthan" or ">": return Py.CompareValues(v, args[0]) > 0;
            case "ge" or ">=": return Py.CompareValues(v, args[0]) >= 0;
            case "in": return Py.Contains(args[0], v);
            case "sameas": return v is null ? args[0] is null : v is bool b ? args[0] is bool c && b == c : Py.IsNumber(v) ? Py.AreEqual(v, args[0]) : ReferenceEquals(v, args[0]);
            case "escaped": return false;
            case "filter": return v is string f && Filters.Contains(f);
            case "test": return true;
            default: throw new JinjaException($"there is no test '{name}'");
        }
    }

    // ── filters ───────────────────────────────────────────────────────────

    private static readonly HashSet<string> Filters =
    [
        "abs", "attr", "batch", "capitalize", "center", "count", "d", "default", "dictsort", "e", "escape", "first", "float", "format", "groupby", "indent",
        "int", "items", "join", "last", "length", "list", "lower", "map", "max", "min", "reject", "rejectattr", "replace", "reverse", "round", "safe",
        "select", "selectattr", "slice", "sort", "string", "striptags", "sum", "title", "tojson", "trim", "truncate", "unique", "upper", "urlencode", "wordcount", "fromjson", "from_json",
    ];

    public static object? Filter(string name, object? v, List<object?> args, Dictionary<string, object?> named)
    {
        object? A(int at, string key, object? missing = null) => Arg(args, named, at, key, missing);
        List<object?> Items() => Py.Iterate(v).ToList();
        switch (name)
        {
            case "default" or "d":
                return v is Undefined || (Py.Truthy(A(1, "boolean", false)) && !Py.Truthy(v)) ? A(0, "default_value", "") : v;
            case "trim": return Py.Strip(Py.Str(v), TextOrNone(A(0, "chars"), name), true, true);
            case "upper": return Py.Str(v).ToUpperInvariant();
            case "lower": return Py.Str(v).ToLowerInvariant();
            case "capitalize": return Py.Capitalize(Py.Str(v));
            case "title": return Py.TitleWords(Py.Str(v));
            case "string": return Py.Str(v);
            case "safe" or "forceescape": return v;
            case "e" or "escape":
                return Py.Str(v).Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("'", "&#39;").Replace("\"", "&#34;");
            case "length" or "count": return Py.Length(v);
            case "wordcount": return (long)System.Text.RegularExpressions.Regex.Count(Py.Str(v), @"\w+");
            case "list": return new JList(Items());
            case "first": return Items() is { Count: > 0 } f ? f[0] : new Undefined("the first of nothing");
            case "last": return Items() is { Count: > 0 } l ? l[^1] : new Undefined("the last of nothing");
            case "reverse": return v is string ? string.Concat(Py.Characters((string)v).Reverse()) : new JList(Enumerable.Reverse(Items()).ToList());
            case "join":
            {
                var items = Items();
                if (A(1, "attribute") is { } attribute and not Undefined)
                {
                    items = items.Select(x => Member(x, attribute)).ToList();
                }
                return string.Join(Py.Str(A(0, "d", "")), items.Select(Py.Str));
            }
            case "items":
                return v is JDict ? DictMethod((JDict)v, "items")!([], named) : v is Undefined ? new JList([]) : throw new JinjaException($"items of {Py.TypeName(v)}");
            case "dictsort":
            {
                var d = v as JDict ?? throw new JinjaException($"dictsort of {Py.TypeName(v)}");
                bool sensitive = Py.Truthy(A(0, "case_sensitive", false));
                bool byValue = Py.Str(A(1, "by", "key")) == "value";
                bool reverse = Py.Truthy(A(2, "reverse", false));
                return new JList(Ordered(d.Items, p => byValue ? p.Value : p.Key, sensitive, reverse).Select(p => (object?)new JList([p.Key, p.Value], true)).ToList());
            }
            case "sort":
            {
                object? attribute = A(2, "attribute");
                return new JList(Ordered(Items(), x => attribute is null ? x : Member(x, attribute), Py.Truthy(A(1, "case_sensitive", false)), Py.Truthy(A(0, "reverse", false))).ToList());
            }
            case "unique":
            {
                var seen = new JDict();
                bool sensitive = Py.Truthy(A(0, "case_sensitive", false));
                return new JList(Items().Where(x =>
                {
                    object? key = !sensitive && x is string s ? s.ToLowerInvariant() : x;
                    bool isNew = !seen.TryGet(key, out _);
                    seen.Set(key, true);
                    return isNew;
                }).ToList());
            }
            case "min" or "max":
            {
                var items = Items();
                if (items.Count == 0)
                {
                    return new Undefined("the least or most of nothing");
                }
                var ordered = Ordered(items, x => A(1, "attribute") is { } at ? Member(x, at) : x, Py.Truthy(A(0, "case_sensitive", false)), name == "max").ToList();
                return ordered[0];
            }
            case "sum":
            {
                object? attribute = A(0, "attribute");
                object? total = A(1, "start", 0L);
                foreach (var x in Items())
                {
                    object? n = attribute is null ? x : Member(x, attribute);
                    total = total is double || n is double ? (object)(Py.Num(total) + Py.Num(n)) : Py.Int(total) + Py.Int(n);
                }
                return total;
            }
            case "abs": return v is double n1 ? (object)Math.Abs(n1) : Math.Abs(Py.Int(v));
            case "round":
            {
                double scale = Math.Pow(10, Py.Int(A(0, "precision", 0L)));
                double x = Py.Num(v) * scale;
                return Py.Str(A(1, "method", "common")) switch { "ceil" => Math.Ceiling(x), "floor" => Math.Floor(x), _ => Math.Round(x, MidpointRounding.ToEven) } / scale;
            }
            case "int":
                return v switch
                {
                    long or bool => Py.Int(v),
                    double d => (long)Math.Truncate(d),
                    string s when long.TryParse(s.Trim(), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out long i) => i,
                    string s when double.TryParse(s.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double r) => (long)r,
                    _ => A(0, "default", 0L),
                };
            case "float":
                return v is string fs ? double.TryParse(fs.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double fd) ? fd : A(0, "default", 0.0)
                    : Py.IsNumber(v) ? Py.Num(v) : A(0, "default", 0.0);
            case "replace": return Replaced(Py.Str(v), Py.Str(A(0, "old")), Py.Str(A(1, "new")), A(2, "count"));
            case "tojson":
                return Py.ToJson(v, Py.Truthy(A(0, "ensure_ascii", false)), A(1, "indent") is { } indent and not Undefined ? (int)Py.Int(indent) : null,
                    A(2, "separators") is JList { Items.Count: 2 } sep ? Py.Str(sep.Items[0]) : null, A(2, "separators") is JList { Items.Count: 2 } s2 ? Py.Str(s2.Items[1]) : null,
                    Py.Truthy(A(3, "sort_keys", false)));
            case "fromjson" or "from_json":
                try
                {
                    using var doc = System.Text.Json.JsonDocument.Parse(Text(v, name));
                    return Py.FromJson(doc.RootElement);
                }
                catch (System.Text.Json.JsonException e)
                {
                    throw new JinjaException($"not JSON: {e.Message}");
                }
            case "indent":
            {
                string pad = A(0, "width", 4L) is string p ? p : new string(' ', (int)Py.Int(A(0, "width", 4L)));
                bool first = Py.Truthy(A(1, "first", false)), blank = Py.Truthy(A(2, "blank", false));
                var lines = (Py.Str(v) + "\n").Split('\n')[..^1];
                return string.Join("\n", lines.Select((line, i) => (i == 0 ? first : true) && (blank || line.Length > 0 || (i == 0 && first)) ? pad + line : line));
            }
            case "center":
            {
                string s = Py.Str(v);
                int width = (int)Py.Int(A(0, "width", 80L)), left = Math.Max(0, width - s.Length) / 2;
                return s.PadLeft(s.Length + left).PadRight(Math.Max(width, s.Length));
            }
            case "truncate":
            {
                string s = Py.Str(v), end = Py.Str(A(2, "end", "..."));
                int most = (int)Py.Int(A(0, "length", 255L)), leeway = (int)Py.Int(A(3, "leeway", 5L));
                if (s.Length <= most + leeway)
                {
                    return s;
                }
                string cut = s[..Math.Max(0, most - end.Length)];
                return (Py.Truthy(A(1, "killwords", false)) || cut.LastIndexOf(' ') < 0 ? cut : cut[..cut.LastIndexOf(' ')]) + end;
            }
            case "striptags": return System.Text.RegularExpressions.Regex.Replace(System.Text.RegularExpressions.Regex.Replace(Py.Str(v), "<[^>]*>", ""), @"\s+", " ").Trim();
            case "urlencode": return Uri.EscapeDataString(Py.Str(v));
            case "format": return Percent(Py.Str(v), new JList(args, true));
            case "attr": return Member(v, A(0, "name"));
            case "map":
            {
                if (args.Count == 0)
                {
                    object? attribute = named.GetValueOrDefault("attribute") ?? throw new JinjaException("map takes a filter or an attribute");
                    return new JList(Items().Select(x => Member(x, attribute) is Undefined u ? named.TryGetValue("default", out var d) ? d : u : Member(x, attribute)).ToList());
                }
                string each = Text(args[0], name);
                return new JList(Items().Select(x => Filter(each, x, args.Skip(1).ToList(), named)).ToList());
            }
            case "select" or "reject":
            {
                bool wanted = name == "select";
                return new JList(Items().Where(x => (args.Count == 0 ? Py.Truthy(x) : Test(Text(args[0], name), x, args.Skip(1).ToList())) == wanted).ToList());
            }
            case "selectattr" or "rejectattr":
            {
                bool wanted = name == "selectattr";
                return new JList(Items().Where(x =>
                {
                    object? member = Member(x, args[0]);
                    return (args.Count == 1 ? Py.Truthy(member) : Test(Text(args[1], name), member, args.Skip(2).ToList())) == wanted;
                }).ToList());
            }
            case "groupby":
            {
                object? attribute = A(0, "attribute");
                return new JList(Ordered(Items(), x => Member(x, attribute), Py.Truthy(A(2, "case_sensitive", false)), false)
                    .GroupBy(x => Member(x, attribute), EqualityComparer<object?>.Create(Py.AreEqual, _ => 0))
                    .Select(g => (object?)new JList([g.Key, new JList(g.ToList())], true)).ToList());
            }
            case "batch":
            {
                int size = (int)Py.Int(A(0, "linecount"));
                return new JList(Items().Chunk(size).Select(c => (object?)new JList(A(1, "fill_with") is { } fill
                    ? [.. c, .. Enumerable.Repeat(fill, size - c.Length)] : [.. c])).ToList());
            }
            case "slice":
            {
                var items = Items();
                int n = (int)Py.Int(A(0, "slices")), at = 0;
                var parts = new List<object?>();
                for (int i = 0; i < n; i++)
                {
                    int size = items.Count / n + (i < items.Count % n ? 1 : 0);
                    var part = items.GetRange(at, size);
                    if (A(1, "fill_with") is { } fill && i >= items.Count % n && items.Count % n != 0)
                    {
                        part.Add(fill);
                    }
                    parts.Add(new JList(part));
                    at += size;
                }
                return new JList(parts);
            }
            default: throw new JinjaException($"there is no filter '{name}'");
        }
    }

    private static IEnumerable<T> Ordered<T>(IEnumerable<T> items, Func<T, object?> key, bool sensitive, bool reverse)
    {
        object? Key(T x) => key(x) is string s && !sensitive ? s.ToLowerInvariant() : key(x);
        var order = Comparer<object?>.Create(Py.CompareValues);
        return reverse ? items.OrderByDescending(Key, order) : items.OrderBy(Key, order);
    }

    /// <summary>A member by name, dotted or by index, as filters take them.</summary>
    private static object? Member(object? v, object? path)
    {
        foreach (string part in (path is long i ? i.ToString(CultureInfo.InvariantCulture) : Py.Str(path)).Split('.'))
        {
            v = v switch
            {
                JDict d when d.TryGet(part, out var x) => x,
                JNamespace n when n.Members.TryGetValue(part, out var x) => x,
                JList l when int.TryParse(part, out int at) && at < l.Items.Count => l.Items[at],
                _ => new Undefined(part),
            };
        }
        return v;
    }
}
