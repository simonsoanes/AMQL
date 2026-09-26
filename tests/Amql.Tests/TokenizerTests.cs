using System.Text.Json;
using Amql.Hf;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Tokenizer tests. The golden corpus was captured from the reference
/// implementation (HF <c>tokenizers</c> 0.22.1) against the real
/// Qwen3.5-0.8B tokenizer.json; the .NET engine must reproduce the ids
/// byte-for-byte and round-trip decode identically.
/// </summary>
public class TokenizerTests
{
    private static readonly string GoldenPath = Path.Combine(AppContext.BaseDirectory, "golden_tokenizer.json");
    private static readonly string RealTokenizerPath = @"D:\Models\Qwen3.5-0.8B\tokenizer.json";
    private static bool GoldenAvailable => File.Exists(GoldenPath) && File.Exists(RealTokenizerPath);

    // ── golden parity with the reference implementation ─────────────────────

    public static TheoryData<string, int[]> GoldenTheory =>
        Theory(() =>
            {
                var data = new TheoryData<string, int[]>();
                if (!GoldenAvailable)
                {
                    return data;
                }
                using var doc = JsonDocument.Parse(File.ReadAllBytes(GoldenPath));
                foreach (var entry in doc.RootElement.EnumerateArray())
                {
                    var ids = entry.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                    data.Add(entry.GetProperty("text").GetString()!, ids);
                }
                return data;
            });

    private static TheoryData<string, int[]> Theory(Func<TheoryData<string, int[]>> factory)
    {
        try
        {
            return factory();
        }
        catch
        {
            return new TheoryData<string, int[]>();
        }
    }

    [Theory]
    [MemberData(nameof(GoldenTheory))]
    public void Encode_Matches_Reference_Implementation(string text, int[] expectedIds)
    {
        if (!GoldenAvailable)
        {
            return;
        }
        var tokenizer = HfTokenizer.FromTokenizerFile(RealTokenizerPath);
        Assert.Equal(expectedIds, tokenizer.EncodeToIds(text));
    }

    [Theory]
    [MemberData(nameof(GoldenTheory))]
    public void Decode_RoundTrips_Encode(string text, int[] ids)
    {
        if (!GoldenAvailable)
        {
            return;
        }
        if (text.Length == 0)
        {
            return; // empty encodes to no ids — nothing to round-trip
        }
        var tokenizer = HfTokenizer.FromTokenizerFile(RealTokenizerPath);
        Assert.Equal(text, tokenizer.Decode(ids));
    }

    // ── special tokens and the chat encoding ───────────────────────────────

    [Fact]
    public void Special_Tokens_Encode_And_Decode_As_Content()
    {
        if (!GoldenAvailable)
        {
            return;
        }
        var tokenizer = HfTokenizer.FromTokenizerFile(RealTokenizerPath);

        const string chat = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
        var result = tokenizer.Encode(chat);
        Assert.Equal(new[] { 248045, 846, 198, 5834, 248046, 198, 248045, 74455, 198 }, result.Ids);

        // Specials decode to their content (unlike the reference's default
        // skip-special decode, which drops them).
        Assert.Equal(chat, tokenizer.Decode(result.Ids));

        // End-of-text round-trips too.
        const string eot = "<|endoftext|>";
        Assert.Equal(new[] { 248044 }, tokenizer.EncodeToIds(eot));
        Assert.Equal(eot, tokenizer.Decode(new[] { 248044 }));
    }

    [Fact]
    public void ByteLevel_Repris_Show_The_Space_Marker()
    {
        if (!GoldenAvailable)
        {
            return;
        }
        var tokenizer = HfTokenizer.FromTokenizerFile(RealTokenizerPath);
        // ' world' is token 1814, whose vocabulary form starts with the
        // byte-level space marker U+0120.
        var info = tokenizer.TokenInfo(1814);
        Assert.Equal("Ġworld", info.Representation);
        Assert.Equal(" world", info.DecodedText);
        Assert.False(info.IsSpecial);
    }

    // ── demo tokenizer (synth-model writes it) ─────────────────────────────

    [Fact]
    public void Demo_Tokenizer_Encodes_The_Demo_Alphabet()
    {
        using var dir = new TempDir();
        SyntheticCheckpoint.Write(dir.Path);
        var tokenizer = HfTokenizer.FromModelDir(dir.Path);

        Assert.Equal(12, tokenizer.VocabSize);

        // Space → Ġ (id 0), letters a..j → 1..10, '?' → 11.
        Assert.Equal(new[] { 1, 2, 3 }, tokenizer.EncodeToIds("abc"));
        Assert.Equal("abc", tokenizer.Decode(tokenizer.EncodeToIds("abc")));
        Assert.Equal(new[] { 1, 0, 2 }, tokenizer.EncodeToIds("a b"));
        Assert.Equal("a b", tokenizer.Decode(tokenizer.EncodeToIds("a b")));
        Assert.Equal(new[] { 10, 1, 7 }, tokenizer.EncodeToIds("jag"));

        // The decode side is lossless even for repeated spaces.
        Assert.Equal("  ", tokenizer.Decode(tokenizer.EncodeToIds("  ")));
    }

    [Fact]
    public void Demo_Tokenizer_Refuses_Unknown_Characters()
    {
        using var dir = new TempDir();
        SyntheticCheckpoint.Write(dir.Path);
        var tokenizer = HfTokenizer.FromModelDir(dir.Path);

        // 'z' is outside the demo alphabet — fail-closed, with the piece named.
        var ex = Assert.Throws<TokenizerException>(() => tokenizer.EncodeToIds("z"));
        Assert.Contains("'z'", ex.Message);
    }

    // ── tokenizer.json forms shipped by newer tokenizers (Granite) ───────

    [Fact]
    public void Array_Form_Merges_Rank_Like_String_Form()
    {
        // tokenizers ≥ 0.21 writes each merge as a two-element array; the
        // older "left right" string must rank identically.
        using var dir = new TempDir();
        var arrayPath = Path.Combine(dir.Path, "merges-array.json");
        var stringPath = Path.Combine(dir.Path, "merges-string.json");
        File.WriteAllText(arrayPath, MiniBpe("[ \"a\", \"b\" ]", Gpt2PreTokenizer));
        File.WriteAllText(stringPath, MiniBpe("\"a b\"", Gpt2PreTokenizer));

        var arrayForm = HfTokenizer.FromTokenizerFile(arrayPath);
        var stringForm = HfTokenizer.FromTokenizerFile(stringPath);

        // The merge fires under both spellings: "ab" is the single merged id.
        Assert.Equal(new[] { 3 }, arrayForm.EncodeToIds("ab"));
        Assert.Equal(arrayForm.EncodeToIds("ab"), stringForm.EncodeToIds("ab"));
        Assert.Equal("ab", arrayForm.Decode(arrayForm.EncodeToIds("ab")));
    }

    [Fact]
    public void ByteLevel_UseRegex_Serves_The_Gpt2_Split()
    {
        // A bare ByteLevel pre-tokenizer with use_regex=true IS the GPT-2
        // split: the leading space rides the word piece ("Ġab"), where a
        // whitespace-split fallback would emit the bare piece twice.
        using var dir = new TempDir();
        var path = Path.Combine(dir.Path, "gpt2-style.json");
        File.WriteAllText(path, MiniBpe("[ \"a\", \"b\" ], [ \"Ġ\", \"ab\" ]", Gpt2PreTokenizer));

        var tokenizer = HfTokenizer.FromTokenizerFile(path);
        Assert.Equal(new[] { 3, 5 }, tokenizer.EncodeToIds("ab ab"));
        Assert.Equal("ab ab", tokenizer.Decode(tokenizer.EncodeToIds("ab ab")));
    }

    private const string Gpt2PreTokenizer =
        "{ \"type\": \"ByteLevel\", \"add_prefix_space\": false, \"trim_offsets\": true, \"use_regex\": true }";

    private static string MiniBpe(string mergesJson, string preTokenizerJson) => $$"""
        {
          "version": "1.0.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [],
          "normalizer": { "type": "NFC" },
          "pre_tokenizer": {{preTokenizerJson}},
          "post_processor": null,
          "decoder": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false },
          "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": { "a": 1, "b": 2, "ab": 3, "Ġ": 4, "Ġab": 5, ",": 6 },
            "merges": [ {{mergesJson}} ]
          }
        }
        """;
}