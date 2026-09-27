using System.Globalization;
using System.Text;
using System.Text.Json;

namespace Amql.Cli;

/// <summary>
/// Python's text renderings of JSON values, reproduced exactly: Von's SDK turns
/// a request's structured <c>state</c> and <c>instructions</c> into model input
/// with <c>str()</c> and <c>json.dumps()</c>, so any difference in quoting,
/// escaping or float formatting would change the tokens the encoder sees — and
/// with them the answer.
/// </summary>
internal static class PyFormat
{
    /// <summary><c>str(value)</c> for a value <c>json.loads</c> produced.</summary>
    public static string Str(JsonElement value) =>
        value.ValueKind == JsonValueKind.String ? value.GetString()! : Repr(value);

    /// <summary><c>repr(value)</c> for a value <c>json.loads</c> produced.</summary>
    public static string Repr(JsonElement value)
    {
        var sb = new StringBuilder();
        AppendRepr(sb, value);
        return sb.ToString();
    }

    private static void AppendRepr(StringBuilder sb, JsonElement value)
    {
        switch (value.ValueKind)
        {
            case JsonValueKind.String:
                sb.Append(StringRepr(value.GetString()!));
                break;
            case JsonValueKind.Number:
                sb.Append(NumberRepr(value));
                break;
            case JsonValueKind.True:
                sb.Append("True");
                break;
            case JsonValueKind.False:
                sb.Append("False");
                break;
            case JsonValueKind.Null:
                sb.Append("None");
                break;
            case JsonValueKind.Array:
            {
                sb.Append('[');
                bool first = true;
                foreach (var item in value.EnumerateArray())
                {
                    if (!first)
                    {
                        sb.Append(", ");
                    }
                    first = false;
                    AppendRepr(sb, item);
                }
                sb.Append(']');
                break;
            }
            case JsonValueKind.Object:
            {
                sb.Append('{');
                bool first = true;
                foreach (var (key, item) in LastWins(value))
                {
                    if (!first)
                    {
                        sb.Append(", ");
                    }
                    first = false;
                    sb.Append(StringRepr(key)).Append(": ");
                    AppendRepr(sb, item);
                }
                sb.Append('}');
                break;
            }
            default:
                throw new ArgumentException($"no Python repr for JSON {value.ValueKind}");
        }
    }

    /// <summary>A JSON object as <c>json.loads</c> builds it: a duplicated key
    /// keeps its first position and its last value.</summary>
    public static List<(string Key, JsonElement Value)> LastWins(JsonElement obj)
    {
        var order = new List<string>();
        var values = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (var prop in obj.EnumerateObject())
        {
            if (!values.ContainsKey(prop.Name))
            {
                order.Add(prop.Name);
            }
            values[prop.Name] = prop.Value;
        }
        return order.Select(k => (k, values[k])).ToList();
    }

    /// <summary>Python's <c>str.__repr__</c>: single quotes unless the text
    /// holds a single quote and no double quote; <c>\\</c>, the quote,
    /// <c>\t \n \r</c> escaped; non-printable characters as <c>\xNN</c>,
    /// <c>\uNNNN</c> or <c>\UNNNNNNNN</c>.</summary>
    public static string StringRepr(string s)
    {
        char quote = s.Contains('\'') && !s.Contains('"') ? '"' : '\'';
        var sb = new StringBuilder(s.Length + 2).Append(quote);
        for (int i = 0; i < s.Length; i++)
        {
            int cp = CodePointAt(s, i, out int width);
            char c = s[i];
            if (c == quote || c == '\\')
            {
                sb.Append('\\').Append(c);
            }
            else if (c == '\t')
            {
                sb.Append("\\t");
            }
            else if (c == '\n')
            {
                sb.Append("\\n");
            }
            else if (c == '\r')
            {
                sb.Append("\\r");
            }
            else if (IsPrintable(cp))
            {
                sb.Append(s, i, width);
            }
            else if (cp < 0x100)
            {
                sb.Append("\\x").Append(cp.ToString("x2"));
            }
            else if (cp < 0x10000)
            {
                sb.Append("\\u").Append(cp.ToString("x4"));
            }
            else
            {
                sb.Append("\\U").Append(cp.ToString("x8"));
            }
            i += width - 1;
        }
        return sb.Append(quote).ToString();
    }

    private static int CodePointAt(string s, int i, out int width)
    {
        if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
        {
            width = 2;
            return char.ConvertToUtf32(s[i], s[i + 1]);
        }
        width = 1;
        return s[i];
    }

    /// <summary><c>str.isprintable()</c> for one code point: everything except
    /// the Other (Cc, Cf, Cs, Co, Cn) and Separator (Zl, Zp, Zs) categories —
    /// bar the ASCII space.</summary>
    private static bool IsPrintable(int cp)
    {
        if (cp == ' ')
        {
            return true;
        }
        var category = cp < 0x10000
            ? CharUnicodeInfo.GetUnicodeCategory((char)cp)
            : CharUnicodeInfo.GetUnicodeCategory(char.ConvertFromUtf32(cp), 0);
        return category switch
        {
            UnicodeCategory.Control or UnicodeCategory.Format or UnicodeCategory.Surrogate or
            UnicodeCategory.PrivateUse or UnicodeCategory.OtherNotAssigned or
            UnicodeCategory.LineSeparator or UnicodeCategory.ParagraphSeparator or
            UnicodeCategory.SpaceSeparator => false,
            _ => true,
        };
    }

    /// <summary>A JSON number the way Python holds it: an integer literal
    /// stays an arbitrary-precision int, anything with a fraction or exponent
    /// becomes a float and prints with <c>float.__repr__</c>.</summary>
    public static string NumberRepr(JsonElement number)
    {
        string raw = number.GetRawText();
        if (raw.IndexOfAny(['.', 'e', 'E']) < 0)
        {
            return System.Numerics.BigInteger.Parse(raw, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture);
        }
        return FloatRepr(double.Parse(raw, NumberStyles.Float, CultureInfo.InvariantCulture));
    }

    /// <summary><c>float.__repr__</c>: the shortest round-tripping digits,
    /// positional between 1e-4 and 1e16 (always with a fractional part),
    /// exponent form otherwise with at least two exponent digits.</summary>
    public static string FloatRepr(double x)
    {
        if (double.IsNaN(x))
        {
            return "nan";
        }
        if (double.IsInfinity(x))
        {
            return x > 0 ? "inf" : "-inf";
        }
        if (x == 0)
        {
            return double.IsNegative(x) ? "-0.0" : "0.0";
        }
        // "E16" is not shortest; "R" is. Normalise R's output into digits + exponent.
        string r = x.ToString("R", CultureInfo.InvariantCulture);
        bool negative = r.StartsWith('-');
        if (negative)
        {
            r = r[1..];
        }
        int exp10 = 0;
        int e = r.IndexOfAny(['E', 'e']);
        if (e >= 0)
        {
            exp10 = int.Parse(r[(e + 1)..], CultureInfo.InvariantCulture);
            r = r[..e];
        }
        int dot = r.IndexOf('.');
        string digits = dot < 0 ? r : r.Remove(dot, 1);
        int pointPos = (dot < 0 ? r.Length : dot) + exp10;   // decimal point position within digits
        int lead = digits.Length - digits.TrimStart('0').Length;
        digits = digits.TrimStart('0');
        pointPos -= lead;
        digits = digits.TrimEnd('0');
        if (digits.Length == 0)
        {
            digits = "0";
        }
        int sciExp = pointPos - 1;                            // value = d.ddd × 10^sciExp

        string body;
        if (sciExp >= -4 && sciExp < 16)
        {
            if (pointPos <= 0)
            {
                body = "0." + new string('0', -pointPos) + digits;
            }
            else if (pointPos >= digits.Length)
            {
                body = digits + new string('0', pointPos - digits.Length) + ".0";
            }
            else
            {
                body = digits[..pointPos] + "." + digits[pointPos..];
            }
        }
        else
        {
            string mantissa = digits.Length == 1 ? digits : digits[0] + "." + digits[1..];
            body = mantissa + "e" + (sciExp < 0 ? "-" : "+") + Math.Abs(sciExp).ToString("00", CultureInfo.InvariantCulture);
        }
        return negative ? "-" + body : body;
    }

    /// <summary><c>json.dumps(value, sort_keys=sortKeys)</c> with Python's
    /// defaults: <c>", "</c> / <c>": "</c> separators and
    /// <c>ensure_ascii=True</c>.</summary>
    public static string JsonDumps(JsonElement value, bool sortKeys)
    {
        var sb = new StringBuilder();
        AppendJson(sb, value, sortKeys);
        return sb.ToString();
    }

    private static void AppendJson(StringBuilder sb, JsonElement value, bool sortKeys)
    {
        switch (value.ValueKind)
        {
            case JsonValueKind.String:
                AppendJsonString(sb, value.GetString()!);
                break;
            case JsonValueKind.Number:
                sb.Append(NumberRepr(value));
                break;
            case JsonValueKind.True:
                sb.Append("true");
                break;
            case JsonValueKind.False:
                sb.Append("false");
                break;
            case JsonValueKind.Null:
                sb.Append("null");
                break;
            case JsonValueKind.Array:
            {
                sb.Append('[');
                bool first = true;
                foreach (var item in value.EnumerateArray())
                {
                    if (!first)
                    {
                        sb.Append(", ");
                    }
                    first = false;
                    AppendJson(sb, item, sortKeys);
                }
                sb.Append(']');
                break;
            }
            case JsonValueKind.Object:
            {
                sb.Append('{');
                var entries = LastWins(value);
                if (sortKeys)
                {
                    // Python sorts str keys by code point, i.e. ordinal on UTF-32.
                    entries.Sort((a, b) => CompareCodePoints(a.Key, b.Key));
                }
                bool first = true;
                foreach (var (key, item) in entries)
                {
                    if (!first)
                    {
                        sb.Append(", ");
                    }
                    first = false;
                    AppendJsonString(sb, key);
                    sb.Append(": ");
                    AppendJson(sb, item, sortKeys);
                }
                sb.Append('}');
                break;
            }
        }
    }

    private static int CompareCodePoints(string a, string b)
    {
        var ea = a.EnumerateRunes().GetEnumerator();
        var eb = b.EnumerateRunes().GetEnumerator();
        while (true)
        {
            bool ha = ea.MoveNext(), hb = eb.MoveNext();
            if (!ha || !hb)
            {
                return ha ? 1 : hb ? -1 : 0;
            }
            int c = ea.Current.Value.CompareTo(eb.Current.Value);
            if (c != 0)
            {
                return c;
            }
        }
    }

    private static void AppendJsonString(StringBuilder sb, string s)
    {
        sb.Append('"');
        foreach (char c in s)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                default:
                    if (c < 0x20 || c > 0x7e)
                    {
                        // ensure_ascii: UTF-16 code units, so astral characters
                        // come out as surrogate pairs, exactly as Python writes them.
                        sb.Append("\\u").Append(((int)c).ToString("x4"));
                    }
                    else
                    {
                        sb.Append(c);
                    }
                    break;
            }
        }
        sb.Append('"');
    }

    /// <summary>Python's <c>str.strip()</c>: removes leading and trailing
    /// whitespace as Python defines it (<c>str.isspace</c>).</summary>
    public static string Strip(string s)
    {
        int start = 0, end = s.Length;
        while (start < end && IsPySpace(s[start]))
        {
            start++;
        }
        while (end > start && IsPySpace(s[end - 1]))
        {
            end--;
        }
        return s[start..end];
    }

    /// <summary><c>str.isspace</c>: Unicode White_Space plus the ASCII
    /// separators \x1c–\x1f that Python also treats as whitespace.</summary>
    private static bool IsPySpace(char c) => char.IsWhiteSpace(c) || c is >= '\x1c' and <= '\x1f';
}
