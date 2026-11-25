using System.Text.Json;

namespace Amql.Server;

/// <summary>What constrained decoding asks of a grammar: whether a piece of text may come
/// next, taking it, and whether the reply may end where it stands.</summary>
public interface ITokenConstraint
{
    bool Accepts(string text);

    void Advance(string text);

    bool CanEnd { get; }
}

/// <summary>The tools a request offers, compiled for <see cref="CallConstraint"/>: each
/// tool's parameters, which are required, and the schema of those that are not plain text.</summary>
public sealed class ToolGrammar
{
    /// <param name="Choices">For a string enum: the values, written bare as a template that writes strings bare does.</param>
    public sealed record Parameter(string Name, bool Required, JsonSchemaNode? Json, IReadOnlyList<string>? Choices = null);

    /// <param name="Arguments">The arguments as one object, for a syntax that writes them as one.</param>
    public sealed record Tool(string Name, IReadOnlyList<Parameter> Parameters, JsonSchemaNode Arguments);

    // A value the model writes for a file is as long as the file.
    private const int LongestString = 1 << 22;

    public IReadOnlyList<Tool> Tools { get; }

    private ToolGrammar(IReadOnlyList<Tool> tools) => Tools = tools;

    /// <summary>Compiles chat-form tools (<c>{"type":"function","function":{name, parameters}}</c>).
    /// A parameter typed string (or not typed) is free text; any other is held to its schema
    /// as JSON. A schema this build cannot compile leaves that parameter free text.</summary>
    public static ToolGrammar Compile(IReadOnlyList<JsonElement> tools)
    {
        var list = new List<Tool>();
        foreach (var t in tools)
        {
            var f = t.TryGetProperty("function", out var fn) ? fn : t;
            if (!f.TryGetProperty("name", out var n) || n.GetString() is not { Length: > 0 } name)
            {
                continue;
            }
            var parameters = new List<Parameter>();
            if (f.TryGetProperty("parameters", out var ps) && ps.ValueKind == JsonValueKind.Object &&
                ps.TryGetProperty("properties", out var props) && props.ValueKind == JsonValueKind.Object)
            {
                var required = ps.TryGetProperty("required", out var req) && req.ValueKind == JsonValueKind.Array
                    ? req.EnumerateArray().Select(x => x.GetString()).ToHashSet()
                    : new HashSet<string?>();
                foreach (var p in props.EnumerateObject())
                {
                    JsonSchemaNode? json = null;
                    List<string>? choices = null;
                    try
                    {
                        var node = JsonSchemaNode.Compile(p.Value);
                        if (node.Enum is { } values && values.All(e => e.StartsWith('"')))
                        {
                            // A string enum: one of its values, bare.
                            choices = values.Select(e => JsonSerializer.Deserialize<string>(e)!).ToList();
                        }
                        else if (!node.Str)
                        {
                            json = node;   // not text: held to its schema as JSON
                        }
                    }
                    catch (ArgumentException)
                    {
                    }
                    parameters.Add(new Parameter(p.Name, required.Contains(p.Name), json, choices));
                }
            }
            list.Add(new Tool(name, parameters, ArgumentsOf(f.TryGetProperty("parameters", out var all) ? all : default)));
        }
        return new ToolGrammar(list);
    }

    /// <summary>The schema a tool's arguments are held to: its parameters, in any order.</summary>
    private static JsonSchemaNode ArgumentsOf(JsonElement parameters)
    {
        var node = new JsonSchemaNode { Obj = true, AnyOrder = true, Properties = new() };
        if (parameters.ValueKind != JsonValueKind.Object || !parameters.TryGetProperty("properties", out var props) || props.ValueKind != JsonValueKind.Object)
        {
            return node;
        }
        try
        {
            node = JsonSchemaNode.Compile(parameters);
        }
        catch (ArgumentException)
        {
            // Something in it does not compile: that parameter takes any value.
            var required = parameters.TryGetProperty("required", out var req) && req.ValueKind == JsonValueKind.Array
                ? req.EnumerateArray().Select(x => x.GetString()).ToHashSet()
                : new HashSet<string?>();
            foreach (var p in props.EnumerateObject())
            {
                JsonSchemaNode value;
                try
                {
                    value = JsonSchemaNode.Compile(p.Value);
                }
                catch (ArgumentException)
                {
                    value = JsonSchemaNode.Any();
                }
                node.Properties!.Add((p.Name, value, required.Contains(p.Name)));
            }
        }
        node.Obj = true;
        node.Arr = node.Str = node.Num = node.Int = node.Bool = node.Null = false;
        node.Relax(LongestString);
        return node;
    }
}
