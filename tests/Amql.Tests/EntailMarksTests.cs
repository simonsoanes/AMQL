using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Amql.Cli;
using Amql.Entail;
using Xunit;

namespace Amql.Tests;

/// <summary>One Entail program for any model: chat marks spelt from the partner's own template,
/// token texts bound in the partner's own spelling, and adapters a container carries.</summary>
public class EntailMarksTests
{
    /// <summary>Cuts text as these families' tokenizers do where it matters here: a chat marker is
    /// one token, a run of newlines is one, a word is one.</summary>
    private static string[] Cut(string text) =>
        Regex.Matches(text, @"<\|[a-z_]+\|>|<\|[a-z_""]+>|<[a-z_]+\|>|</?[a-z_]+>|\n+|[A-Za-z]+|.").Select(m => m.Value).ToArray();

    [Fact]
    public void Qwen_Marks_Are_Learned_From_Its_Template()
    {
        var marks = EntailMarks.Learn(Templates.Qwen, Cut);
        Assert.Equal("<think>", marks["think_open"]);
        Assert.Equal("</think>", marks["think_close"]);
        Assert.Equal("\n\n", marks["reply_open"]);   // the reply starts after the blank line
        Assert.Equal("<tool_call>", marks["call_open"]);
        Assert.Equal("<|im_start|>", marks["turn_open"]);
        Assert.Equal("<|im_end|>", marks["turn_close"]);
        Assert.Equal("system", marks["system"]);
        Assert.Equal("user", marks["user"]);
        Assert.Equal("assistant", marks["assistant"]);
    }

    [Fact]
    public void Gemma_Marks_Are_Learned_From_Its_Template()
    {
        var marks = EntailMarks.Learn(Templates.Gemma, Cut);
        Assert.Equal("<|channel>", marks["think_open"]);
        Assert.Equal("<channel|>", marks["think_close"]);
        Assert.Equal("<channel|>", marks["reply_open"]);   // the reply starts straight after the close
        Assert.Equal("<|tool_call>", marks["call_open"]);
        Assert.Equal("<|turn>", marks["turn_open"]);
        Assert.Equal("<turn|>", marks["turn_close"]);
        Assert.Equal("system", marks["system"]);
        Assert.Equal("user", marks["user"]);
        Assert.Equal("model", marks["assistant"]);
    }

    [Fact]
    public void A_Program_Naming_No_Mark_Is_Left_As_It_Is()
    {
        using var dir = new TempDir();
        string source = "program p\nstate a\ntoken \"\\{mark:user}\" means a = 1\nfor d in 0 to 9\n    token \"{d}\" means a = 1\nend\n";
        Assert.Same(source, EntailMarks.Fill(source, dir.Path));
    }

    [Fact]
    public void An_Unfilled_Mark_Says_What_Spells_It()
    {
        var e = Assert.Throws<EntailException>(() => EntailCompiler.Compile("program p\nstate a\ntoken \"{mark:think_close}\" means a = 1\n"));
        Assert.Contains("chat mark", e.Message);
    }

    /// <summary>A SentencePiece-style vocabulary may hold Ġ and Ċ as letters of their own: a space
    /// is still ▁ and a newline still itself there, never those.</summary>
    [Fact]
    public void Token_Texts_Bind_In_The_Partners_Own_Spelling()
    {
        using var spm = new TempDir();
        WriteTokenizer(spm.Path, byteLevel: false, ["<unk>", "\u2581", "\n", "\u0120", "\u010a", "\u2581was", "x"]);
        var program = EntailCompiler.Compile("program p\nstate a, b, c\ntoken \" \" means a = 1\ntoken \"\\n\" means b = 1\ntoken \" was\" means c = 1\n");
        var rows = EntailWeights.BindAdapter(program, spm.Path).Manifest.Rows;
        Assert.Equal(new[] { 1, 2, 5 }, rows.Keys.Order());   // ▁, \n and ▁was; not Ġ (3) or Ċ (4)

        using var bytes = new TempDir();
        WriteTokenizer(bytes.Path, byteLevel: true, ["<unk>", "\u2581", "\n", "\u0120", "\u010a", "\u0120was", "x"]);
        rows = EntailWeights.BindAdapter(program, bytes.Path).Manifest.Rows;
        Assert.Equal(new[] { 3, 4, 5 }, rows.Keys.Order());   // Ġ, Ċ and Ġwas
    }

    [Fact]
    public void A_Container_Carries_Adapters_In_Order_And_Refuses_Another_Tokenizers()
    {
        using var root = new TempDir();
        string container = Path.Combine(root.Path, "model"), other = Path.Combine(root.Path, "other");
        Directory.CreateDirectory(container);
        Directory.CreateDirectory(other);
        WriteTokenizer(container, byteLevel: false, ["<unk>", "\u2581", "\n", "a", "b"]);
        WriteTokenizer(other, byteLevel: false, ["<unk>", "\u2581", "\n", "b", "a"]);
        File.WriteAllText(Path.Combine(container, "index.json"), new JsonObject { ["format"] = "test" }.ToJsonString());
        string Adapter(string name, string partner)
        {
            string dir = Path.Combine(root.Path, name);
            EntailSparseAdapter.Write($"program {name.Replace('-', '_')}\nstate x\ntoken \"a\" means x = 1\nemit \"b\" when x\n", partner, dir);
            return dir;
        }
        string second = Adapter("second", container), first = Adapter("first", container);

        Assert.Empty(ShippedAdapters.List(container));
        Assert.Equal(["second", "first"], ShippedAdapters.Attach(container, [second, first]));
        Assert.Equal(["second", "first"], ShippedAdapters.Attach(container, [second]));   // replaced in its place
        var listed = ShippedAdapters.List(container);
        Assert.Equal(["second", "first"], listed.Select(Path.GetFileName));
        Assert.True(File.Exists(Path.Combine(listed[0], EntailSparseAdapter.ProgramFile)));
        Assert.Equal("test", (string?)JsonNode.Parse(File.ReadAllText(Path.Combine(container, "index.json")))!["format"]);

        Assert.Equal(listed[1], ShippedAdapters.Resolve("first", container));
        Assert.Equal(second, ShippedAdapters.Resolve(second, container));   // a folder that is there is itself
        Assert.Throws<EntailException>(() => ShippedAdapters.Resolve("third", container));
        Assert.Throws<EntailException>(() => ShippedAdapters.Attach(container, [Adapter("theirs", other)]));
    }

    private static void WriteTokenizer(string dir, bool byteLevel, string[] vocab)
    {
        var map = new JsonObject();
        for (int i = 0; i < vocab.Length; i++)
        {
            map[vocab[i]] = i;
        }
        var tokenizer = new JsonObject
        {
            ["model"] = new JsonObject { ["type"] = "BPE", ["vocab"] = map, ["merges"] = new JsonArray() },
            ["added_tokens"] = new JsonArray(),
            ["pre_tokenizer"] = byteLevel ? new JsonObject { ["type"] = "ByteLevel" } : new JsonObject { ["type"] = "Split" },
            ["decoder"] = byteLevel ? new JsonObject { ["type"] = "ByteLevel" }
                : new JsonObject { ["type"] = "Sequence", ["decoders"] = new JsonArray(new JsonObject { ["type"] = "Replace" }) },
        };
        File.WriteAllText(Path.Combine(dir, "tokenizer.json"), tokenizer.ToJsonString());
    }
}
