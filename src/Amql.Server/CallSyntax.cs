using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Amql.Cli;

namespace Amql.Server;

/// <summary>
/// How a model writes its calls, from the shape learned of its chat template (<see cref="CallShape"/>):
/// what opens and closes them, how they are read (more loosely than they are written), and the grammar
/// that holds them to a request's tools.
/// </summary>
public sealed class CallSyntax
{
    private readonly CallShape _shape;
    private readonly (string Name, string Arguments)? _keys;   // in a JSON object that holds the whole call
    private readonly Regex? _head, _pair;

    public CallSyntax(CallShape shape)
    {
        _shape = shape;
        Close = Marks(shape.Closer).LastOrDefault();
        if (shape.Style == ArgumentStyle.Json)
        {
            string sample = shape.Lead + "qznameq" + shape.AfterName + "{}" + shape.End;
            int open = sample.IndexOf('{');
            int end = open >= 0 && open < shape.Lead.Length ? CallShape.Balanced(sample, open) : -1;
            _keys = end < 0 ? null : Keys(sample[open..end]);
        }
        string lead = shape.SectionOpen.Length > 0 ? shape.Lead : shape.AfterOpener;
        string after = Loose(shape.AfterName.Trim());
        if (after.Length == 0)
        {
            after = shape.PairOpen.Trim().Length > 0 ? $@"(?=\s*(?:{Loose(shape.PairOpen.Trim())}|$))" : @"(?=\s*\{)";
        }
        if (shape.Style is ArgumentStyle.Pairs or ArgumentStyle.Json)
        {
            _head = new Regex($@"{(shape.SectionOpen.Length > 0 ? "" : @"^\s*")}{Loose(lead.Trim())}\s*([^\n]+?)\s*{after}");
        }
        if (shape.Style == ArgumentStyle.Pairs)
        {
            _pair = new Regex($@"{Loose(shape.PairOpen.Trim())}([^\n]+?){Loose(shape.KeyEnd.Trim())}(?:\s*<!\[CDATA\[(.*?)\]\]>\s*|(.*?)){Loose(shape.PairClose.Trim())}", RegexOptions.Singleline);
        }
        Markup = new[] { shape.SectionOpen, shape.Lead, shape.End, shape.SectionClose, shape.PairOpen, shape.PairClose }
            .SelectMany(Marks).Append(shape.AtStartOnly ? "" : shape.Opener).Append(Marks(shape.PairOpen).Count == 0 && shape.PairOpen.Contains('<') ? shape.PairOpen.Trim() : "")
            .Where(m => m.Length > 1).Distinct().ToList();
    }

    /// <summary>What a reply's calls start with.</summary>
    public string Open => _shape.Opener;

    /// <summary>The opener counts only where the reply's text starts: it is no mark of its own.</summary>
    public bool AtStartOnly => _shape.AtStartOnly;

    /// <summary>The mark they end with, or null where only the arguments' end does.</summary>
    public string? Close { get; }

    /// <summary>The start of a call to <paramref name="tool"/> (null: to any tool), for <c>tool_choice</c>.</summary>
    public string Start(string? tool) => _shape.Start(tool);

    /// <summary>Text a reply that may not call never writes.</summary>
    public IReadOnlyList<string> Markup { get; }

    /// <summary>What is written around a call's names and values, in which tokens of their own may stand.</summary>
    public IEnumerable<string> Pieces => (_shape.Style == ArgumentStyle.Pairs ? _shape.Pieces.Append("<![CDATA[").Append("]]>") : _shape.Pieces)
        .Where(p => p.Length > 0);

    /// <summary>The grammar from just after <see cref="Open"/> to the calls' end.</summary>
    public ICallConstraint Constraint(ToolGrammar grammar) => new CallConstraint(_shape, grammar);

    /// <summary>The marks in a piece of text: <c>&lt;…&gt;</c>, <c>[NAME]</c>, and <c>&lt;name=</c>.</summary>
    internal static IReadOnlyList<string> Marks(string text) =>
        Regex.Matches(text, @"<[^<>\s=]+[>=]|<[^<>\s]+>|\[/?[A-Z][A-Z_]+\]").Select(m => m.Value).ToList();

    /// <summary>A literal as a pattern that takes any blank space where it has some.</summary>
    private static string Loose(string literal) => string.Join(@"\s*", Regex.Split(literal, @"\s+").Select(Regex.Escape));

    // ── reading ───────────────────────────────────────────────────────────

    /// <summary>The calls that open at <paramref name="at"/>: where what they say starts and ends, and where
    /// the text after them starts; null while they have not closed.</summary>
    public (int From, int To, int Next)? Block(string body, int at, bool final)
    {
        int from = _shape.AtStartOnly ? at : at + Open.Length;
        int whole = from;
        if (_shape.Style == ArgumentStyle.Json)
        {
            // A string may hold the closing mark, so it counts only after the arguments.
            int open = body.IndexOf(_shape.AtStartOnly || !_shape.AfterOpener.TrimStart().StartsWith('[') ? '{' : '[', from);
            int end = open < 0 ? -1 : CallShape.Balanced(body, open);
            if (Close is null)
            {
                return end < 0 ? null : (from, end, end);
            }
            if (end < 0 && !final)
            {
                return null;
            }
            whole = end < 0 ? from : end;
        }
        int close = Close is null ? -1 : body.IndexOf(Close, whole, StringComparison.Ordinal);
        return close < 0 ? null : (from, close, close + Close!.Length);
    }

    /// <summary>The calls in what stands between <see cref="Open"/> and <see cref="Close"/>.
    /// <paramref name="text"/> is given each string value over several lines, to change it.</summary>
    public IReadOnlyList<ParsedToolCall> Read(string inner, IReadOnlyDictionary<string, JsonElement> schemas, Func<string, string> text)
    {
        var given = new List<(string Name, JsonObject Arguments)>();
        switch (_shape.Style)
        {
            case ArgumentStyle.Json when _keys is null:
                ReadNamed(inner, given);
                break;
            case ArgumentStyle.Marked:
                ReadMarked(inner, given);
                break;
            case ArgumentStyle.Pairs:
                return ReadPairs(inner, schemas, text) is { Count: > 0 } calls ? calls : ReadObjects();
            case ArgumentStyle.Python:
                ReadPython(inner, schemas, given);
                break;
        }
        return given.Count == 0 ? ReadObjects() : given.Select(g => Finish(g.Name, g.Arguments, schemas, text)).ToList();

        // A call written as a JSON object, which models that know several forms fall back to.
        IReadOnlyList<ParsedToolCall> ReadObjects()
        {
            for (int at = inner.IndexOfAny(['{', '[']); at >= 0;)
            {
                int end = CallShape.Balanced(inner, at);
                if (end < 0)
                {
                    break;
                }
                try
                {
                    Collect(JsonNode.Parse(inner[at..end]), given);
                }
                catch (JsonException)
                {
                }
                at = end < inner.Length ? inner.IndexOfAny(['{', '['], end) : -1;
            }
            return given.Select(g => Finish(g.Name, g.Arguments, schemas, text)).ToList();
        }
    }

    private void Collect(JsonNode? node, List<(string, JsonObject)> given)
    {
        if (node is JsonArray array)
        {
            foreach (var item in array)
            {
                Collect(item, given);
            }
            return;
        }
        if (node is not JsonObject o)
        {
            return;
        }
        var name = (_keys is { } k ? o[k.Name] : null) ?? o["name"];
        if (name is JsonValue v && v.GetValueKind() == JsonValueKind.String && v.GetValue<string>().Trim() is { Length: > 0 } tool)
        {
            var args = (_keys is { } a ? o[a.Arguments] : null) ?? o["arguments"] ?? o["parameters"];
            if (args is JsonValue s && s.GetValueKind() == JsonValueKind.String)
            {
                try
                {
                    args = JsonNode.Parse(s.GetValue<string>());   // the object as a string
                }
                catch (JsonException)
                {
                }
            }
            given.Add((tool, args as JsonObject ?? new JsonObject()));
            return;
        }
        foreach (var (_, inner) in o)
        {
            if (inner is JsonObject)
            {
                Collect(inner, given);   // {"function": {"name": …}}
            }
        }
    }

    private static (string, string)? Keys(string json)
    {
        try
        {
            return Find(JsonNode.Parse(json));
        }
        catch (JsonException)
        {
            return null;
        }

        static (string, string)? Find(JsonNode? node)
        {
            if (node is not JsonObject o)
            {
                return null;
            }
            string? name = o.FirstOrDefault(p => p.Value is JsonValue v && v.GetValueKind() == JsonValueKind.String && v.GetValue<string>() == "qznameq").Key;
            string? args = o.FirstOrDefault(p => p.Value is JsonObject { Count: 0 }).Key;
            return name is not null && args is not null ? (name, args) : o.Select(p => Find(p.Value)).FirstOrDefault(x => x is not null);
        }
    }

    /// <summary>Calls that name the tool outside the JSON object of its arguments.</summary>
    private void ReadNamed(string inner, List<(string, JsonObject)> given)
    {
        for (int at = 0; at < inner.Length && _head!.Match(inner[at..]) is { Success: true } m;)
        {
            int open = inner.IndexOf('{', at + m.Index + m.Length);
            int end = open < 0 ? -1 : CallShape.Balanced(inner, open);
            if (end < 0)
            {
                return;
            }
            try
            {
                if (JsonNode.Parse(inner[open..end]) is JsonObject args)
                {
                    given.Add((m.Groups[1].Value.Trim(), args));
                }
            }
            catch (JsonException)
            {
            }
            at = end;
        }
    }

    private IReadOnlyList<ParsedToolCall> ReadPairs(string inner, IReadOnlyDictionary<string, JsonElement> schemas, Func<string, string> text)
    {
        var calls = new List<ParsedToolCall>();
        var heads = _head!.Matches(inner);
        // One call takes all the text (a value may quote another call's); a section holds several.
        int count = _shape.SectionOpen.Length > 0 ? heads.Count : Math.Min(1, heads.Count);
        for (int i = 0; i < count; i++)
        {
            string name = heads[i].Groups[1].Value.Trim();
            int from = heads[i].Index + heads[i].Length, to = i + 1 < count ? heads[i + 1].Index : inner.Length;
            var properties = Properties(schemas, name);
            var args = new JsonObject();
            foreach (Match p in _pair!.Matches(inner[from..to]))
            {
                string key = p.Groups[1].Value.Trim();
                bool literal = p.Groups[2].Success;
                string value = literal ? p.Groups[2].Value : p.Groups[3].Value;
                // Templates write a value on lines of its own.
                if (!literal && value.StartsWith('\n'))
                {
                    value = value[1..];
                }
                if (!literal && value.EndsWith('\n'))
                {
                    value = value[..^1];
                }
                string? type = TypeOf(properties, key);
                if (value.Contains('\n') && type is null or "string")
                {
                    value = text(value);
                }
                args[key] = Value(value, type);
            }
            calls.Add(new ParsedToolCall(name, args.ToJsonString(Relaxed)));
        }
        return calls;
    }

    private void ReadMarked(string inner, List<(string, JsonObject)> given)
    {
        string s = inner;
        string lead = (_shape.SectionOpen.Length > 0 ? _shape.Lead : _shape.AfterOpener).Trim();
        for (int at = 0; at < s.Length;)
        {
            int named = s.IndexOf(lead, at, StringComparison.Ordinal);
            int brace = named < 0 ? -1 : s.IndexOf('{', named + lead.Length);
            if (brace < 0 || (given.Count == 0 && s[..named].Trim().Length > 0))
            {
                return;
            }
            string name = s[(named + lead.Length)..brace].Trim();
            string after = _shape.AfterName.Trim();
            if (after.Length > 0 && name.EndsWith(after, StringComparison.Ordinal))
            {
                name = name[..^after.Length].Trim();
            }
            at = brace;
            if (name.Length == 0 || ReadValue(s, ref at) is not JsonObject args)
            {
                return;
            }
            given.Add((name, args));
        }
    }

    /// <summary>A value in an object's marked notation: strings between quote marks, keys bare. Text
    /// that is no value reads as a string, up to what ends it. Null is a JSON null or the end.</summary>
    private JsonNode? ReadValue(string s, ref int at)
    {
        string quote = _shape.Quote;
        Space(s, ref at);
        if (at >= s.Length)
        {
            return null;
        }
        if (Starts(s, at, quote))
        {
            return JsonValue.Create(ReadString(s, ref at));
        }
        if (s[at] == '{')
        {
            var o = new JsonObject();
            at++;
            while (true)
            {
                Space(s, ref at);
                if (at >= s.Length)
                {
                    return o;
                }
                if (s[at] is '}' or ',')
                {
                    if (s[at++] == '}')
                    {
                        return o;
                    }
                    continue;
                }
                int colon;
                string key;
                if (Starts(s, at, quote))
                {
                    key = ReadString(s, ref at);
                    Space(s, ref at);
                    colon = at;
                }
                else
                {
                    colon = s.IndexOf(':', at);
                    key = colon < 0 ? "" : s[at..colon].Trim().Trim('"');
                }
                if (colon < 0 || colon >= s.Length || s[colon] != ':')
                {
                    at = s.Length;
                    return o;
                }
                at = colon + 1;
                o[key] = ReadValue(s, ref at);
            }
        }
        if (s[at] == '[')
        {
            var a = new JsonArray();
            at++;
            while (true)
            {
                Space(s, ref at);
                if (at >= s.Length)
                {
                    return a;
                }
                if (s[at] is ']' or ',')
                {
                    if (s[at++] == ']')
                    {
                        return a;
                    }
                    continue;
                }
                int before = at;
                a.Add(ReadValue(s, ref at));
                if (at == before)
                {
                    at++;   // a stray closing mark
                }
            }
        }
        if (s[at] == '"')
        {
            // A string as JSON writes it, which a free call may hold.
            for (int end = at + 1; end < s.Length; end++)
            {
                if (s[end] == '\\')
                {
                    end++;
                }
                else if (s[end] == '"')
                {
                    try
                    {
                        var node = JsonNode.Parse(s[at..(end + 1)]);
                        at = end + 1;
                        return node;
                    }
                    catch (JsonException)
                    {
                        break;
                    }
                }
            }
        }
        int from = at;
        while (at < s.Length && s[at] is not (',' or '}' or ']'))
        {
            at++;
        }
        return Bare(s[from..at].Trim());
    }

    private string ReadString(string s, ref int at)
    {
        int from = at + _shape.Quote.Length;
        int end = s.IndexOf(_shape.Quote, from, StringComparison.Ordinal);
        string value = end < 0 ? s[from..] : s[from..end];
        at = end < 0 ? s.Length : end + _shape.Quote.Length;
        return value;
    }

    private void ReadPython(string s, IReadOnlyDictionary<string, JsonElement> schemas, List<(string, JsonObject)> given)
    {
        int at = 0;
        while (true)
        {
            while (at < s.Length && (char.IsWhiteSpace(s[at]) || s[at] is '[' or ']' or ','))
            {
                at++;
            }
            int open = at < s.Length ? s.IndexOf('(', at) : -1;
            string name = open < 0 ? "" : s[at..open].Trim();
            if (name.Length == 0 || name.Any(c => char.IsWhiteSpace(c) || c is '{' or '"' or '\''))
            {
                return;
            }
            at = open + 1;
            var names = Properties(schemas, name) is { ValueKind: JsonValueKind.Object } props ? props.EnumerateObject().Select(p => p.Name).ToList() : [];
            var args = new JsonObject();
            int position = 0;
            while (true)
            {
                Space(s, ref at);
                if (at >= s.Length)
                {
                    return;   // cut off
                }
                if (s[at] is ')' or ',')
                {
                    if (s[at++] == ')')
                    {
                        break;
                    }
                    continue;
                }
                int from = at;
                while (at < s.Length && (char.IsLetterOrDigit(s[at]) || s[at] is '_' or '-' or '.'))
                {
                    at++;
                }
                int equals = at;
                Space(s, ref equals);
                string? key = null;
                if (equals < s.Length && s[equals] == '=' && equals > from)
                {
                    key = s[from..at];
                    at = equals + 1;
                }
                else
                {
                    at = from;   // a positional argument is the parameter at its place
                    key = position < names.Count ? names[position] : null;
                }
                position++;
                int before = at;
                var value = PythonValue(s, ref at);
                if (at == before)
                {
                    return;
                }
                if (key is not null)
                {
                    args[key] = value;
                }
            }
            given.Add((name, args));
        }
    }

    /// <summary>A Python literal (or the JSON one it is close to).</summary>
    private static JsonNode? PythonValue(string s, ref int at)
    {
        Space(s, ref at);
        if (at >= s.Length)
        {
            return null;
        }
        char c = s[at];
        if (c is '\'' or '"')
        {
            var sb = new StringBuilder();
            bool triple = at + 2 < s.Length && s[at + 1] == c && s[at + 2] == c;
            at += triple ? 3 : 1;
            while (at < s.Length)
            {
                if (s[at] == c && (triple || Ends(s, at + 1)) && (!triple || (at + 2 < s.Length && s[at + 1] == c && s[at + 2] == c)))
                {
                    at += triple ? 3 : 1;
                    break;
                }
                if (s[at] != '\\' || at + 1 >= s.Length)
                {
                    sb.Append(s[at++]);
                    continue;
                }
                char e = s[at + 1];
                at += 2;
                int digits = e switch { 'x' => 2, 'u' => 4, 'U' => 8, _ => 0 };
                if (digits > 0 && at + digits <= s.Length &&
                    int.TryParse(s.AsSpan(at, digits), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out int code) && code is >= 0 and <= 0x10ffff)
                {
                    sb.Append(code is >= 0xd800 and < 0xe000 || code < 0x10000 ? ((char)code).ToString() : char.ConvertFromUtf32(code));
                    at += digits;
                    continue;
                }
                sb.Append(e switch { 'n' => "\n", 't' => "\t", 'r' => "\r", 'b' => "\b", 'f' => "\f", '\\' => "\\", '\'' => "'", '"' => "\"", '\n' => "", _ => "\\" + e });
            }
            return JsonValue.Create(sb.ToString());
        }
        if (c is '[' or '(')
        {
            char close = c == '[' ? ']' : ')';
            var a = new JsonArray();
            at++;
            while (at < s.Length)
            {
                Space(s, ref at);
                if (at < s.Length && (s[at] == close || s[at] == ','))
                {
                    if (s[at++] == close)
                    {
                        break;
                    }
                    continue;
                }
                int before = at;
                a.Add(PythonValue(s, ref at));
                if (at == before)
                {
                    at++;
                }
            }
            return a;
        }
        if (c == '{')
        {
            var o = new JsonObject();
            at++;
            while (at < s.Length)
            {
                Space(s, ref at);
                if (at < s.Length && s[at] is '}' or ',')
                {
                    if (s[at++] == '}')
                    {
                        break;
                    }
                    continue;
                }
                int before = at;
                string key = PythonValue(s, ref at) is JsonValue v ? v.GetValueKind() == JsonValueKind.String ? v.GetValue<string>() : v.ToJsonString() : "";
                Space(s, ref at);
                if (at >= s.Length || s[at] != ':')
                {
                    at = Math.Max(at, before + 1);
                    continue;
                }
                at++;
                o[key] = PythonValue(s, ref at);
            }
            return o;
        }
        int from = at;
        while (at < s.Length && s[at] is not (',' or ')' or ']' or '}' or ':'))
        {
            at++;
        }
        return Bare(s[from..at].Trim());
    }

    /// <summary>Whether a quote before here can end a string: text goes on after one that is written unescaped.</summary>
    private static bool Ends(string s, int at)
    {
        Space(s, ref at);
        return at >= s.Length || s[at] is ',' or ')' or ']' or '}' or ':';
    }

    private static JsonNode? Bare(string text) => text switch
    {
        "null" or "None" => null,
        "true" or "True" => JsonValue.Create(true),
        "false" or "False" => JsonValue.Create(false),
        _ => JsonNumber(text) ?? JsonValue.Create(text),
    };

    private static bool Starts(string s, int at, string what) => what.Length > 0 && string.CompareOrdinal(s, at, what, 0, what.Length) == 0;

    private static void Space(string s, ref int at)
    {
        while (at < s.Length && char.IsWhiteSpace(s[at]))
        {
            at++;
        }
    }

    // ── values ────────────────────────────────────────────────────────────

    // Arguments as a client would write them: "<", ">" and "&" stay as they are (still valid JSON).
    private static readonly JsonSerializerOptions Relaxed = new()
    {
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    /// <summary>Arguments whose values have types of their own, as their schemas ask for them.</summary>
    private static ParsedToolCall Finish(string name, JsonObject given, IReadOnlyDictionary<string, JsonElement> schemas, Func<string, string> text)
    {
        var properties = Properties(schemas, name);
        var args = new JsonObject();
        foreach (var (key, value) in given)
        {
            string? type = TypeOf(properties, key);
            args[key] = value is JsonValue v && v.GetValueKind() == JsonValueKind.String
                ? type is null or "string" ? JsonValue.Create(v.GetValue<string>() is var s && s.Contains('\n') ? text(s) : s) : Value(v.GetValue<string>(), type)
                : type == "string" && value is JsonValue bare ? JsonValue.Create(bare.ToJsonString()) : value?.DeepClone();
        }
        return new ParsedToolCall(name, args.ToJsonString(Relaxed));
    }

    private static JsonElement Properties(IReadOnlyDictionary<string, JsonElement> schemas, string tool) =>
        schemas.TryGetValue(tool, out var schema) && schema.ValueKind == JsonValueKind.Object
            && schema.TryGetProperty("properties", out var props) && props.ValueKind == JsonValueKind.Object ? props : default;

    private static string? TypeOf(JsonElement properties, string key)
    {
        if (properties.ValueKind != JsonValueKind.Object || !properties.TryGetProperty(key, out var schema) ||
            schema.ValueKind != JsonValueKind.Object || !schema.TryGetProperty("type", out var t))
        {
            return null;
        }
        if (t.ValueKind == JsonValueKind.String)
        {
            return t.GetString();
        }
        // ["integer", "null"] and the like: the first non-null type.
        return t.ValueKind == JsonValueKind.Array
            ? t.EnumerateArray().Select(x => x.GetString()).FirstOrDefault(x => x is not null and not "null")
            : null;
    }

    /// <summary>A parameter's text as the JSON value its schema asks for. A
    /// value that does not read as that type stays a string, and the client's
    /// own validation reports it.</summary>
    private static JsonNode? Value(string text, string? type)
    {
        string t = text.Trim();
        switch (type)
        {
            case "integer" when long.TryParse(t, NumberStyles.Integer, CultureInfo.InvariantCulture, out long l):
                return JsonValue.Create(l);
            case "integer" or "number" when JsonNumber(t) is { } big:
                return big;   // a valid JSON number too large for a long or double, kept as written
            case "number" when double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out double d) && double.IsFinite(d):
                return t.Contains('.') || t.Contains('e') || t.Contains('E') ? JsonValue.Create(d)
                    : long.TryParse(t, NumberStyles.Integer, CultureInfo.InvariantCulture, out long whole) ? JsonValue.Create(whole) : JsonValue.Create(d);
            case "boolean" when t.Equals("true", StringComparison.OrdinalIgnoreCase) || t.Equals("false", StringComparison.OrdinalIgnoreCase):
                return JsonValue.Create(t.Equals("true", StringComparison.OrdinalIgnoreCase));
            case "object" or "array":
                try
                {
                    var node = JsonNode.Parse(t);
                    if (node is JsonObject or JsonArray)
                    {
                        return node;
                    }
                }
                catch (JsonException)
                {
                    // Some templates write these as Python does.
                    int at = 0;
                    if (t.Length > 0 && t[0] is '{' or '[' && t[^1] is '}' or ']' && PythonValue(t, ref at) is { } python && at == t.Length)
                    {
                        return python;
                    }
                }
                return JsonValue.Create(text);
            case "null" when t is "null" or "None":
                return null;
            default:
                return JsonValue.Create(text);
        }
    }

    private static JsonNode? JsonNumber(string t)
    {
        try
        {
            return JsonNode.Parse(t) is JsonValue v && v.GetValueKind() == JsonValueKind.Number ? v : null;
        }
        catch (JsonException)
        {
            return null;
        }
    }
}
