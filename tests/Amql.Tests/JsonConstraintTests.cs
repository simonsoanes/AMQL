using System.Text.Json;
using Amql.Server;

namespace Amql.Tests;

/// <summary>
/// Structured output's grammar: what <see cref="JsonConstraint"/> lets a reply
/// continue with, for the schema an agent harness actually sends (Hydra's
/// action-and-knowledge assessment) and for the corners of JSON itself.
/// </summary>
public sealed class JsonConstraintTests
{
    private const string Assessment = """
        {"type":"object","properties":{
          "pendingAction":{"type":["string","null"]},
          "knowledgeSuggestions":{"type":"array","items":{"type":"object","properties":{
            "title":{"type":"string"},"content":{"type":"string"}},"required":["title","content"],"additionalProperties":false}},
          "skillSuggestions":{"type":"array","items":{"type":"object","properties":{
            "title":{"type":"string"},"mode":{"type":"string","enum":["new","revise"]}},"required":["title","mode"],"additionalProperties":false}},
          "score":{"type":"integer"},
          "ratio":{"type":"number"}},
         "required":["pendingAction","knowledgeSuggestions","skillSuggestions"],"additionalProperties":false}
        """;

    private static JsonConstraint For(string schema) =>
        new(JsonSchemaNode.Compile(JsonDocument.Parse(schema).RootElement));

    /// <summary>Feeds the text in pieces of <paramref name="piece"/> characters; true when every
    /// piece was accepted and the value is complete at the end.</summary>
    private static bool Valid(string schema, string text, int piece = 1)
    {
        var c = For(schema);
        for (int i = 0; i < text.Length; i += piece)
        {
            string part = text.Substring(i, Math.Min(piece, text.Length - i));
            if (!c.Accepts(part))
            {
                return false;
            }
            c.Advance(part);
        }
        return c.CanEnd;
    }

    // Hydra's edit_file 'edits' items, descriptions shortened to the words that carry the rules.
    private const string Edit = """
        {"type":"array","items":{"type":"object","required":["op"],"properties":{
          "op":{"type":"string","enum":["replace","delete","insert_after","insert_before","replace_text"],
                "description":"replace: swap the anchored line. replace_text: unverified literal find/replace (escape hatch)."},
          "anchor":{"type":"string","description":"The line to target, e.g. L42:a3f9c1. For replace/delete, the first line of the range."},
          "end":{"type":"string","description":"Last line of a multi-line replace/delete range. Omit for one line."},
          "content_lines":{"type":"array","items":{"type":"string"},"description":"New text for replace/insert_*. Set 'content_lines' or 'content', never both."},
          "content":{"type":"string","description":"Legacy single-string alternative to 'content_lines' for replace/insert_* ops."},
          "find":{"type":"string","description":"Text to find (replace_text only). Matched literally."},
          "find_lines":{"type":"array","items":{"type":"string"},"description":"Multi-line form of 'find' (replace_text only). Set either 'find' or 'find_lines', never both."},
          "replace":{"type":"string","description":"Replacement text (replace_text only)."},
          "replace_lines":{"type":"array","items":{"type":"string"},"description":"Multi-line form of 'replace' (replace_text only). Set either 'replace' or 'replace_lines', never both."}}}}
        """;

    [Theory]
    [InlineData("""[{"op":"replace_text","find":"a","replace":"b"}]""", true)]
    [InlineData("""[{"op":"replace","anchor":"L4:e3b0c4","content_lines":["x"]}]""", true)]
    [InlineData("""[{"op":"insert_after","anchor":"L4:e3b0c4","content":"x"}]""", true)]
    [InlineData("""[{"op":"replace","end":"L4","find":"a","replace":"b"}]""", false)]   // find is for replace_text
    [InlineData("""[{"op":"replace_text","find":"a","replace":"b","replace_lines":["b"]}]""", false)]   // never both
    [InlineData("""[{"op":"replace","anchor":"L1:aa","content_lines":["x"],"content":"x"}]""", false)]
    [InlineData("""[{"op":"replace_text","find":"a","replace":"b"},{"op":"delete","anchor":"L2:bb"}]""", true)]
    public void A_Description_That_Says_Where_A_Property_Belongs_Is_Held_To(string text, bool valid) =>
        Assert.Equal(valid, Valid(Edit, text));

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(17)]
    public void A_Reply_That_Follows_The_Schema_Is_Accepted_However_It_Is_Split(int piece)
    {
        const string reply = """
            {"pendingAction": "Run \"dotnet build\" é\n", "knowledgeSuggestions": [{"title": "Disk viewer", "content": "WPF 3D"}],
             "skillSuggestions": [], "score": -12, "ratio": 0.5e-3}
            """;
        Assert.True(Valid(Assessment, reply.Trim(), piece));
        Assert.True(Valid(Assessment, """{"pendingAction":null,"knowledgeSuggestions":[],"skillSuggestions":[{"title":"t","mode":"revise"}]}"""));
    }

    [Theory]
    [InlineData("""{"knowledgeSuggestions":[],"skillSuggestions":[]}""")]                                      // a required property skipped
    [InlineData("""{"pendingAction":null,"knowledgeSuggestions":[]}""")]                                        // closed before a required one
    [InlineData("""{"pendingAction":null,"knowledgeSuggestions":[],"skillSuggestions":[],"extra":1}""")]        // undeclared property
    [InlineData("""{"pendingAction":null,"knowledgeSuggestions":[],"skillSuggestions":[{"title":"t","mode":"old"}]}""")]   // not in the enum
    [InlineData("""{"pendingAction":1,"knowledgeSuggestions":[],"skillSuggestions":[]}""")]                     // number where string or null
    [InlineData("""{"pendingAction":null,"knowledgeSuggestions":[{"title":"t"}],"skillSuggestions":[]}""")]     // item missing content
    [InlineData("""{"pendingAction":null,"knowledgeSuggestions":[],"skillSuggestions":[],"score":1.5}""")]      // fraction in an integer
    [InlineData("""{"pendingAction":"a\qb","knowledgeSuggestions":[],"skillSuggestions":[]}""")]               // bad escape
    [InlineData("""{"pendingAction":null,"knowledgeSuggestions":[],"skillSuggestions":[]} trailing""")]         // text after the value
    [InlineData("""```json {"pendingAction":null}""")]                                                         // a fence
    public void A_Reply_That_Breaks_The_Schema_Is_Refused(string reply) => Assert.False(Valid(Assessment, reply));

    [Fact]
    public void The_Reply_May_End_Only_Where_The_Value_Is_Complete()
    {
        var c = For("""{"type":"number"}""");
        c.Advance("-1");
        Assert.True(c.CanEnd);                 // "-1" is a whole number
        c.Advance(".");
        Assert.False(c.CanEnd);                // "-1." is not
        c.Advance("25");
        Assert.True(c.CanEnd);
        Assert.False(c.Accepts("x"));

        var obj = For(Assessment);
        obj.Advance("""{"pendingAction":null""");
        Assert.False(obj.CanEnd);
        Assert.False(obj.Accepts("}"));        // required properties still to come
        Assert.True(obj.Accepts(", \"knowledge"));
        Assert.False(obj.Accepts(", \"score"));  // a required property cannot be skipped
    }

    [Fact]
    public void Json_Object_Mode_Takes_Any_Object_And_Nothing_Else()
    {
        bool Any(string text)
        {
            var c = new JsonConstraint(JsonSchemaNode.AnyObject());
            if (!c.Accepts(text))
            {
                return false;
            }
            c.Advance(text);
            return c.CanEnd;
        }
        Assert.True(Any("""{"a": [1, {"b": null}, "x", true], "c": {}}"""));
        Assert.True(Any("{}"));
        Assert.False(Any("[1]"));
        Assert.False(Any("\"just text\""));
    }

    [Fact]
    public void Whitespace_Is_Bounded_So_A_Reply_Cannot_Spin_On_It()
    {
        var c = For("""{"type":"object"}""");
        Assert.True(c.Accepts("{" + new string(' ', 20)));
        Assert.False(c.Accepts("{" + new string('\n', 200)));
    }

    [Fact]
    public void Refs_Anyof_And_Const_Compile()
    {
        const string schema = """
            {"$defs":{"item":{"type":"object","properties":{"k":{"const":"v"}},"required":["k"]}},
             "type":"object","properties":{"items":{"type":"array","items":{"$ref":"#/$defs/item"}},
               "maybe":{"anyOf":[{"type":"integer"},{"type":"null"}]}},"required":["items","maybe"]}
            """;
        Assert.True(Valid(schema, """{"items":[{"k":"v"},{"k":"v"}],"maybe":null}"""));
        Assert.True(Valid(schema, """{"items":[],"maybe":7}"""));
        Assert.False(Valid(schema, """{"items":[{"k":"w"}],"maybe":7}"""));
        Assert.Throws<ArgumentException>(() => JsonSchemaNode.Compile(JsonDocument.Parse("""{"$ref":"https://x/y"}""").RootElement));
    }
}
