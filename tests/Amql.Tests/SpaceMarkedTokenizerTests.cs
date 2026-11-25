using System.Text.Json;
using Amql.Hf;
using Xunit;

namespace Amql.Tests;

/// <summary>SentencePiece-style BPE (Gemma's tokenizer.json): spaces marked, no
/// split, byte fallback. Held to what the tokenizers library gives
/// (<c>fixtures/make_spm_bpe.py</c>).</summary>
public class SpaceMarkedTokenizerTests
{
    private static string Fixture => Path.Combine(AppContext.BaseDirectory, "fixtures", "spm-bpe");

    [Fact]
    public void Encodes_And_Decodes_As_The_Tokenizers_Library_Does()
    {
        var tokenizer = HfTokenizer.FromModelDir(Fixture);
        using var golden = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Fixture, "golden.json")));
        foreach (var entry in golden.RootElement.EnumerateArray())
        {
            string text = entry.GetProperty("text").GetString()!;
            var ids = entry.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            Assert.Equal(ids, tokenizer.EncodeToIds(text));
            Assert.Equal(entry.GetProperty("decoded").GetString(), tokenizer.DecodeText(ids, skipSpecialTokens: false));
        }
    }

    [Fact]
    public void A_Pipeline_With_A_Stage_It_Does_Not_Serve_Is_Refused_By_Name()
    {
        using var dir = new TempDir();
        string json = File.ReadAllText(Path.Combine(Fixture, "tokenizer.json"));
        string path = Path.Combine(dir.Path, "tokenizer.json");
        File.WriteAllText(path, json.Replace("\"byte_fallback\": true", "\"byte_fallback\": false"));
        Assert.Contains("byte fallback", Assert.Throws<TokenizerException>(() => HfTokenizer.FromTokenizerFile(path)).Message);
        File.WriteAllText(path, json.Replace("{\"type\": \"Fuse\"}", "{\"type\": \"Strip\", \"content\": \" \", \"start\": 1, \"stop\": 0}"));
        Assert.Contains("decoder", Assert.Throws<TokenizerException>(() => HfTokenizer.FromTokenizerFile(path)).Message);
    }
}
