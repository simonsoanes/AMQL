using System.Text.Json;

namespace Amql.Server;

/// <summary>
/// A JSON Schema, compiled to what constrained decoding needs: which kinds of
/// value may stand at a place, an object's properties in order, an array's
/// items, a value's allowed literals. The subset structured output asks for:
/// <c>type</c> (or a list of types), <c>properties</c>, <c>required</c>,
/// <c>items</c>, <c>enum</c>, <c>const</c>, <c>maxLength</c>, <c>anyOf</c> /
/// <c>oneOf</c> (merged), and <c>$ref</c> into the schema's own
/// <c>$defs</c>/<c>definitions</c>. Anything else is not enforced.
/// </summary>
public sealed class JsonSchemaNode
{
    public bool Obj, Arr, Str, Num, Int, Bool, Null;

    /// <summary>Declared properties in order (null: any keys, any values).</summary>
    public List<(string Name, JsonSchemaNode Node, bool Required)>? Properties;

    /// <summary>An array's items (null: any value).</summary>
    public JsonSchemaNode? Items;

    /// <summary>The allowed values as compact JSON text (null: no list).</summary>
    public List<string>? Enum;

    public int? MaxLength;

    /// <summary>The index in <see cref="Properties"/> of the required string enum whose value decides
    /// which other properties apply (an edit's <c>op</c>), or -1.</summary>
    public int Discriminator = -1;

    /// <summary>Per property: the discriminator values (compact JSON) it is only for, else null.</summary>
    public List<HashSet<string>?>? OnlyFor;

    /// <summary>Per property: a bit set of the properties it may not stand beside ("never both").</summary>
    public List<long>? Excludes;

    /// <summary>The properties may come in any order, rather than as declared.</summary>
    public bool AnyOrder;

    /// <summary>Lets this and everything in it take its properties in any order, and strings
    /// without a <c>maxLength</c> up to <paramref name="longest"/> characters.</summary>
    public void Relax(int longest)
    {
        AnyOrder = Properties is null or { Count: <= 64 };
        MaxLength ??= longest;
        foreach (var (_, node, _) in Properties ?? new())
        {
            node.Relax(longest);
        }
        Items?.Relax(longest);
    }

    /// <summary>Any JSON value at all.</summary>
    public static JsonSchemaNode Any() => new() { Obj = true, Arr = true, Str = true, Num = true, Bool = true, Null = true };

    /// <summary>Any JSON object (<c>json_object</c> mode).</summary>
    public static JsonSchemaNode AnyObject() => new() { Obj = true };

    public static JsonSchemaNode Compile(JsonElement schema) => Compile(schema, schema, 0);

    private static JsonSchemaNode Compile(JsonElement s, JsonElement root, int depth)
    {
        if (depth > 32)
        {
            throw new ArgumentException("the schema nests (or refers to itself) more than 32 levels deep");
        }
        if (s.ValueKind == JsonValueKind.True || s.ValueKind != JsonValueKind.Object)
        {
            return Any();
        }
        if (s.TryGetProperty("$ref", out var r) && r.GetString() is { } reference)
        {
            return Compile(Resolve(root, reference), root, depth + 1);
        }
        var node = new JsonSchemaNode();
        bool typed = false;
        if (s.TryGetProperty("type", out var type))
        {
            foreach (var t in type.ValueKind == JsonValueKind.Array ? type.EnumerateArray().ToList() : new List<JsonElement> { type })
            {
                typed = true;
                switch (t.GetString())
                {
                    case "object": node.Obj = true; break;
                    case "array": node.Arr = true; break;
                    case "string": node.Str = true; break;
                    case "number": node.Num = true; break;
                    case "integer": node.Int = true; break;
                    case "boolean": node.Bool = true; break;
                    case "null": node.Null = true; break;
                }
            }
        }
        foreach (string union in new[] { "anyOf", "oneOf" })
        {
            if (s.TryGetProperty(union, out var alts) && alts.ValueKind == JsonValueKind.Array)
            {
                foreach (var alt in alts.EnumerateArray())
                {
                    node.Merge(Compile(alt, root, depth + 1));
                    typed = true;
                }
            }
        }
        if (s.TryGetProperty("enum", out var en) && en.ValueKind == JsonValueKind.Array)
        {
            node.Enum = en.EnumerateArray().Select(Compact).ToList();
            typed = true;
        }
        if (s.TryGetProperty("const", out var cn))
        {
            node.Enum = new List<string> { Compact(cn) };
            typed = true;
        }
        if (s.TryGetProperty("properties", out var props) && props.ValueKind == JsonValueKind.Object)
        {
            var required = s.TryGetProperty("required", out var req) && req.ValueKind == JsonValueKind.Array
                ? req.EnumerateArray().Select(x => x.GetString()).ToHashSet()
                : new HashSet<string?>();
            node.Properties = props.EnumerateObject()
                .Select(p => (p.Name, Compile(p.Value, root, depth + 1), required.Contains(p.Name))).ToList();
            node.Obj = true;
            Relate(node, props);
            typed = true;
        }
        if (s.TryGetProperty("items", out var items))
        {
            node.Items = Compile(items, root, depth + 1);
            node.Arr = true;
            typed = true;
        }
        if (s.TryGetProperty("maxLength", out var ml) && ml.ValueKind == JsonValueKind.Number)
        {
            node.MaxLength = ml.GetInt32();
        }
        return typed ? node : Any();
    }

    // "(replace_text only)", "only for replace/delete", "insert_* only": a list of names before
    // "only", or after "only for/on/with".
    private static readonly System.Text.RegularExpressions.Regex OnlyBefore = new(
        @"([A-Za-z_*][\w*]*(?:\s*(?:/|,|\bor\b|\band\b)\s*[A-Za-z_*][\w*]*)*)\s+only\b");
    private static readonly System.Text.RegularExpressions.Regex OnlyAfter = new(
        @"\bonly\s+(?:for|on|with|when)\s+(?:op\s+|ops\s+)?([A-Za-z_*][\w*]*(?:\s*(?:/|,|\bor\b|\band\b)\s*[A-Za-z_*][\w*]*)*)");
    // "Set 'content_lines' or 'content', never both", "Set either 'find' or 'find_lines', never both".
    private static readonly System.Text.RegularExpressions.Regex NeverBoth = new(
        @"['`""]?(\w+)['`""]?\s+or\s+['`""]?(\w+)['`""]?,?\s+(?:but\s+)?never\s+both");

    /// <summary>
    /// What a schema says in words about how an object's properties go together, where its
    /// keywords cannot: a property whose description says it is only for some values of the
    /// required string enum (an edit's <c>op</c>) is offered only after the enum has one of them, and
    /// of two properties a description says to set one of, "never both", only one may be written.
    /// A small model otherwise fills an edit with fields from every op and the tool refuses it.
    /// </summary>
    private static void Relate(JsonSchemaNode node, JsonElement props)
    {
        var list = node.Properties!;
        var descriptions = props.EnumerateObject()
            .Select(p => p.Value.ValueKind == JsonValueKind.Object && p.Value.TryGetProperty("description", out var d)
                && d.ValueKind == JsonValueKind.String ? d.GetString()! : "").ToList();
        int disc = list.FindIndex(p => p.Required && p.Node.Enum is { Count: >= 2 } e && e.All(v => v.StartsWith('"')));
        if (disc >= 0)
        {
            var values = list[disc].Node.Enum!.Select(v => JsonSerializer.Deserialize<string>(v)!).ToList();
            var onlyFor = new List<HashSet<string>?>();
            bool any = false;
            for (int i = 0; i < list.Count; i++)
            {
                HashSet<string>? set = null;
                if (i != disc)
                {
                    foreach (var m in OnlyBefore.Matches(descriptions[i]).Concat(OnlyAfter.Matches(descriptions[i])))
                    {
                        foreach (string name in System.Text.RegularExpressions.Regex.Split(m.Groups[1].Value, @"\s*(?:/|,|\bor\b|\band\b)\s*"))
                        {
                            foreach (string v in values.Where(v => name.EndsWith('*') ? v.StartsWith(name[..^1], StringComparison.Ordinal) : v == name))
                            {
                                (set ??= new HashSet<string>(StringComparer.Ordinal)).Add(JsonSerializer.Serialize(v));
                            }
                        }
                    }
                }
                any |= set is not null;
                onlyFor.Add(set);
            }
            if (any)
            {
                node.Discriminator = disc;
                node.OnlyFor = onlyFor;
            }
        }
        if (list.Count <= 64)
        {
            var excludes = new long[list.Count];
            bool any = false;
            foreach (string d in descriptions)
            {
                foreach (System.Text.RegularExpressions.Match m in NeverBoth.Matches(d))
                {
                    int a = list.FindIndex(p => p.Name == m.Groups[1].Value), b = list.FindIndex(p => p.Name == m.Groups[2].Value);
                    if (a >= 0 && b >= 0 && a != b && !list[a].Required && !list[b].Required)
                    {
                        excludes[a] |= 1L << b;
                        excludes[b] |= 1L << a;
                        any = true;
                    }
                }
            }
            if (any)
            {
                node.Excludes = excludes.ToList();
            }
        }
    }

    private void Merge(JsonSchemaNode other)
    {
        Obj |= other.Obj; Arr |= other.Arr; Str |= other.Str; Num |= other.Num; Int |= other.Int; Bool |= other.Bool; Null |= other.Null;
        if (other.Properties is not null)
        {
            Properties ??= new();
            foreach (var p in other.Properties.Where(p => Properties.All(q => q.Name != p.Name)))
            {
                Properties.Add(p);
            }
        }
        Items ??= other.Items;
        if (other.Enum is not null)
        {
            Enum = (Enum ?? new()).Concat(other.Enum).Distinct().ToList();
        }
    }

    private static JsonElement Resolve(JsonElement root, string reference)
    {
        if (!reference.StartsWith("#", StringComparison.Ordinal))
        {
            throw new ArgumentException($"only local $ref is supported, not '{reference}'");
        }
        var at = root;
        foreach (var part in reference[1..].Split('/', StringSplitOptions.RemoveEmptyEntries))
        {
            string key = part.Replace("~1", "/").Replace("~0", "~");
            if (at.ValueKind != JsonValueKind.Object || !at.TryGetProperty(key, out at))
            {
                throw new ArgumentException($"$ref '{reference}' does not resolve");
            }
        }
        return at;
    }

    private static string Compact(JsonElement e) => JsonSerializer.Serialize(e);
}

/// <summary>
/// Which continuations of a reply keep it a JSON value its schema allows. The
/// state is immutable, so trying a token's text costs nothing to undo: every
/// character gives a new state, or none when it breaks the grammar. Objects
/// take their declared properties in order (an optional one may be skipped, a
/// required one may not); whitespace between tokens is allowed but bounded, and
/// a string without <c>maxLength</c> is capped, so a reply cannot spin forever.
/// The reply may end only where the value is complete.
/// </summary>
public sealed class JsonConstraint : ITokenConstraint
{
    private const int MaxWhitespace = 24;
    private const int DefaultMaxString = 8000;

    private Frame _state;

    public JsonConstraint(JsonSchemaNode schema) => _state = new RootFrame(schema, 0);

    /// <summary>A value about to start, as an opaque immutable state, for a grammar that embeds
    /// JSON (a tool call's typed parameter): <see cref="Step"/> it a character at a time.</summary>
    public static object Begin(JsonSchemaNode schema) => new RootFrame(schema, 0);

    /// <summary>The state after one more character, or null when it breaks the value.</summary>
    public static object? Step(object state, char c) => ((Frame)state).Feed(c);

    /// <summary>Whether the value is whole at this state.</summary>
    public static bool Complete(object state) => ((Frame)state).CanEnd;

    /// <summary>Whether the value is complete, so the reply may end here.</summary>
    public bool CanEnd => _state.CanEnd;

    /// <summary>Whether this text may come next.</summary>
    public bool Accepts(string text) => text.Length > 0 && Run(_state, text) is not null;

    public void Advance(string text) =>
        _state = Run(_state, text) ?? throw new InvalidOperationException($"'{text}' does not continue the JSON");

    private static Frame? Run(Frame? state, string text)
    {
        foreach (char c in text)
        {
            state = state!.Feed(c);
            if (state is null)
            {
                return null;
            }
        }
        return state;
    }

    private static bool IsWhitespace(char c) => c is ' ' or '\n' or '\r' or '\t';

    private abstract class Frame
    {
        public abstract Frame? Feed(char c);

        public virtual bool CanEnd => false;
    }

    /// <summary>A value is about to start at <paramref name="c"/>; <paramref name="after"/> continues once it ends.</summary>
    private static Frame? StartValue(JsonSchemaNode node, Frame after, char c)
    {
        if (node.Enum is { } literals)
        {
            return new LiteralFrame(literals, 0, after).Feed(c);
        }
        switch (c)
        {
            case '{' when node.Obj:
                return new ObjectFrame(node, ObjectPhase.Open, 0, after, 0);
            case '[' when node.Arr:
                return new ArrayFrame(node, ArrayPhase.Open, after, 0);
            case '"' when node.Str:
                return new StringFrame(node.MaxLength ?? DefaultMaxString, 0, 0, after);
            case 't' or 'f' when node.Bool:
                return new LiteralFrame(new List<string> { "true", "false" }, 0, after).Feed(c);
            case 'n' when node.Null:
                return new LiteralFrame(new List<string> { "null" }, 0, after).Feed(c);
            case '-' or (>= '0' and <= '9') when node.Num || node.Int:
                return new NumberFrame(!node.Num, NumberState.Start, after).Feed(c);
            default:
                return null;
        }
    }

    private sealed class RootFrame(JsonSchemaNode node, int ws) : Frame
    {
        public override Frame? Feed(char c) =>
            IsWhitespace(c) ? (ws < MaxWhitespace ? new RootFrame(node, ws + 1) : null) : StartValue(node, DoneFrame.Instance, c);
    }

    private sealed class DoneFrame : Frame
    {
        public static readonly DoneFrame Instance = new();

        public override Frame? Feed(char c) => null;

        public override bool CanEnd => true;
    }

    private enum ObjectPhase { Open, Comma, Colon, Value, After }

    /// <summary>An object: <paramref name="next"/> is the first property still to come, <paramref name="used"/>
    /// the properties written (a bit each), <paramref name="disc"/> the discriminator's value once written.</summary>
    private sealed class ObjectFrame(JsonSchemaNode node, ObjectPhase phase, int next, Frame after, int ws,
        JsonSchemaNode? valueNode = null, long used = 0, string? disc = null) : Frame
    {
        private bool Free => node.Properties is null;

        // A required property still to come.
        private bool Owed => node.Properties is { } props && (node.AnyOrder
            ? props.Where((p, i) => p.Required && (used >> i & 1) == 0 && Allowed(i)).Any()
            : props.Skip(next).Any(p => p.Required));

        private bool MoreToCome => Free || Candidates().Any();

        private Frame Same(ObjectPhase p, int ws) => new ObjectFrame(node, p, next, after, ws, valueNode, used, disc);

        public override Frame? Feed(char c)
        {
            if (IsWhitespace(c))
            {
                return ws < MaxWhitespace ? Same(phase, ws + 1) : null;
            }
            switch (phase)
            {
                case ObjectPhase.Open when c == '}':
                case ObjectPhase.After when c == '}':
                    return Owed ? null : after;
                case ObjectPhase.Open when c == '"' && MoreToCome:
                case ObjectPhase.Comma when c == '"':
                    return new KeyFrame(this, string.Empty, false);
                case ObjectPhase.After when c == ',':
                    return MoreToCome ? Same(ObjectPhase.Comma, 0) : null;
                case ObjectPhase.Colon when c == ':':
                    return Same(ObjectPhase.Value, 0);
                case ObjectPhase.Value when !Free && next - 1 == node.Discriminator && valueNode!.Enum is { } values:
                    // The discriminator's value is kept: it decides which properties follow.
                    return new LiteralFrame(values, 0, DoneFrame.Instance,
                        v => new ObjectFrame(node, ObjectPhase.After, next, after, 0, null, used, v)).Feed(c);
                case ObjectPhase.Value:
                    return StartValue(valueNode!, new ObjectFrame(node, ObjectPhase.After, next, after, 0, null, used, disc), c);
                default:
                    return null;
            }
        }

        // Whether property i may be written here: not one the schema's words rule out for the
        // discriminator's value, and not the other half of a "never both" pair already written.
        private bool Allowed(int i) =>
            (node.OnlyFor?[i] is not { } only || disc is null || only.Contains(disc)) &&
            (node.Excludes is not { } ex || (ex[i] & used) == 0);

        /// <summary>The properties a key typed so far may still name.</summary>
        public IEnumerable<(int Index, string Name)> Candidates()
        {
            if (Free)
            {
                yield break;
            }
            var props = node.Properties!;
            for (int i = node.AnyOrder ? 0 : next; i < props.Count; i++)
            {
                if (Allowed(i) && !(node.AnyOrder && (used >> i & 1) != 0))
                {
                    yield return (i, props[i].Name);
                }
                if (props[i].Required && !node.AnyOrder)
                {
                    yield break;   // a required property cannot be skipped
                }
            }
        }

        public bool KeyPrefix(string key) => Free || Candidates().Any(p => p.Name.StartsWith(key, StringComparison.Ordinal));

        public Frame? KeyDone(string key)
        {
            if (Free)
            {
                return new ObjectFrame(node, ObjectPhase.Colon, next, after, 0, Any);
            }
            foreach (var (index, name) in Candidates())
            {
                if (name == key)
                {
                    long bit = index < 64 ? 1L << index : 0;
                    return new ObjectFrame(node, ObjectPhase.Colon, index + 1, after, 0, node.Properties![index].Node, used | bit, disc);
                }
            }
            return null;
        }

        private static readonly JsonSchemaNode Any = JsonSchemaNode.Any();
    }

    /// <summary>A key being typed (escapes are not allowed in keys).</summary>
    private sealed class KeyFrame(ObjectFrame obj, string key, bool _) : Frame
    {
        public override Frame? Feed(char c)
        {
            if (c == '"')
            {
                return obj.KeyDone(key);
            }
            if (c < 0x20 || c == '\\')
            {
                return null;
            }
            string longer = key + c;
            return obj.KeyPrefix(longer) ? new KeyFrame(obj, longer, false) : null;
        }
    }

    private enum ArrayPhase { Open, Item, After }

    private sealed class ArrayFrame(JsonSchemaNode node, ArrayPhase phase, Frame after, int ws) : Frame
    {
        private static readonly JsonSchemaNode Any = JsonSchemaNode.Any();

        public override Frame? Feed(char c)
        {
            if (IsWhitespace(c))
            {
                return ws < MaxWhitespace ? new ArrayFrame(node, phase, after, ws + 1) : null;
            }
            switch (phase)
            {
                case ArrayPhase.Open when c == ']':
                case ArrayPhase.After when c == ']':
                    return after;
                case ArrayPhase.After when c == ',':
                    return new ArrayFrame(node, ArrayPhase.Item, after, 0);
                case ArrayPhase.Open:
                case ArrayPhase.Item:
                    return StartValue(node.Items ?? Any, new ArrayFrame(node, ArrayPhase.After, after, 0), c);
                default:
                    return null;
            }
        }
    }

    /// <summary>A string's body. <paramref name="escape"/>: 0 none, 1 after a backslash, 2-5 hex digits still to come.</summary>
    private sealed class StringFrame(int maxLength, int length, int escape, Frame after) : Frame
    {
        public override Frame? Feed(char c)
        {
            if (escape == 1)
            {
                return c switch
                {
                    '"' or '\\' or '/' or 'b' or 'f' or 'n' or 'r' or 't' => new StringFrame(maxLength, length + 1, 0, after),
                    'u' => new StringFrame(maxLength, length, 5, after),
                    _ => null,
                };
            }
            if (escape > 1)
            {
                return Uri.IsHexDigit(c) ? new StringFrame(maxLength, escape == 2 ? length + 1 : length, escape - 1 == 1 ? 0 : escape - 1, after) : null;
            }
            if (c == '"')
            {
                return after;
            }
            if (c < 0x20 || length >= maxLength)
            {
                return null;
            }
            return new StringFrame(maxLength, length + 1, c == '\\' ? 1 : 0, after);
        }
    }

    /// <summary>One of a few fixed texts: an enum's values, true/false, null. With <paramref name="then"/>,
    /// what follows depends on which text it was.</summary>
    private sealed class LiteralFrame(List<string> texts, int at, Frame after, Func<string, Frame>? then = null) : Frame
    {
        private Frame Next(string text) => then is null ? after : then(text);

        public override Frame? Feed(char c)
        {
            // A literal that is complete here ends, and c belongs to what follows it.
            if (texts.FirstOrDefault(t => t.Length == at) is { } done && !texts.Any(t => t.Length > at && t[at] == c))
            {
                return Next(done).Feed(c);
            }
            var still = texts.Where(t => t.Length > at && t[at] == c).ToList();
            if (still.Count == 0)
            {
                return null;
            }
            // A quoted literal (a string enum) ends at its closing quote; nothing can extend it.
            if (still.Count == 1 && still[0].Length == at + 1 && still[0][0] == '"')
            {
                return Next(still[0]);
            }
            return new LiteralFrame(still, at + 1, after, then);
        }

        public override bool CanEnd => texts.FirstOrDefault(t => t.Length == at) is { } done && Next(done).CanEnd;
    }

    private enum NumberState { Start, Minus, Zero, Int, Dot, Frac, Exp, ExpSign, ExpDigits }

    private sealed class NumberFrame(bool integer, NumberState state, Frame after, int length = 0) : Frame
    {
        private const int MaxLength = 32;   // a number has no business being longer

        private bool Complete => state is NumberState.Zero or NumberState.Int or NumberState.Frac or NumberState.ExpDigits;

        private Frame? To(NumberState s) => length < MaxLength ? new NumberFrame(integer, s, after, length + 1) : null;

        public override Frame? Feed(char c)
        {
            bool digit = c is >= '0' and <= '9';
            switch (state)
            {
                case NumberState.Start when c == '-':
                    return To(NumberState.Minus);
                case NumberState.Start or NumberState.Minus when c == '0':
                    return To(NumberState.Zero);
                case NumberState.Start or NumberState.Minus when digit:
                    return To(NumberState.Int);
                case NumberState.Int when digit:
                    return To(NumberState.Int);
                case NumberState.Zero or NumberState.Int when c == '.' && !integer:
                    return To(NumberState.Dot);
                case NumberState.Dot or NumberState.Frac when digit:
                    return To(NumberState.Frac);
                case NumberState.Zero or NumberState.Int or NumberState.Frac when (c is 'e' or 'E') && !integer:
                    return To(NumberState.Exp);
                case NumberState.Exp when c is '+' or '-':
                    return To(NumberState.ExpSign);
                case NumberState.Exp or NumberState.ExpSign or NumberState.ExpDigits when digit:
                    return To(NumberState.ExpDigits);
                default:
                    // The number ends here, and c belongs to what follows it.
                    return Complete ? after.Feed(c) : null;
            }
        }

        public override bool CanEnd => Complete && after.CanEnd;
    }
}
