using Amql.Inference.NGram;

namespace Amql.Tests;

/// <summary>
/// N-gram packs: counting a corpus in tokens, looking up the longest known context, saving and
/// loading, and the blend into a model's logits (kNN-LM: (1 − λ)·p_model + λ·p_pack over the
/// pack's tokens, other tokens' odds untouched), with its evidence weighting and its gates.
/// </summary>
public sealed class NGramPackTests
{
    // Two documents over a toy vocabulary of 10 tokens; "1 2 3" is followed by 4 three times, 5 once.
    private static readonly int[][] Docs =
    {
        new[] { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 5 },
        new[] { 7, 1, 2, 3, 4, 9 },
    };

    private static NGramPack Pack() => NGramPack.Build("toy", Docs, order: 4, tokenizer: "10:test", keywords: new[] { "toy" });

    [Fact]
    public void The_Longest_Known_Context_Answers_And_Shorter_Ones_Back_It_Up()
    {
        var pack = Pack();
        var hit = pack.Lookup(new[] { 9, 1, 2, 3 })!.Value;
        Assert.Equal(3, hit.Length);                    // "1 2 3"
        Assert.Equal(4u, hit.Entry.Total);
        Assert.Equal(new[] { 4, 5 }, hit.Entry.Tokens);  // most frequent first
        Assert.Equal(new[] { 3u, 1u }, hit.Entry.Counts);

        // "8 3" was never seen, "3" was: back off to one token of context.
        Assert.Equal(1, pack.Lookup(new[] { 8, 3 })!.Value.Length);
        Assert.Null(pack.Lookup(new[] { 8, 3 }, minLength: 2));
        // Seen once only: below the default minimum count.
        Assert.Null(pack.Lookup(new[] { 7 }));
        // N-grams do not cross documents: "5 7" never happened.
        Assert.Null(pack.Lookup(new[] { 5 }));
    }

    [Fact]
    public void A_Saved_Pack_Loads_As_It_Was()
    {
        using var dir = new TempDir();
        string path = Path.Combine(dir.Path, "toy.ngram");
        var pack = Pack();
        pack.Save(path);
        var loaded = NGramPack.Load(path);
        Assert.Equal(("toy", 4, "10:test", 18L), (loaded.Name, loaded.Order, loaded.Tokenizer, loaded.CorpusTokens));
        Assert.Equal(new[] { "toy" }, loaded.Keywords);
        Assert.Equal(pack.ContextCount, loaded.ContextCount);
        Assert.Equal(pack.Lookup(new[] { 1, 2, 3 })!.Value.Entry.Counts, loaded.Lookup(new[] { 1, 2, 3 })!.Value.Entry.Counts);
    }

    [Fact]
    public void The_Blend_Is_The_Mixture_And_Leaves_Other_Odds_Alone()
    {
        var prior = new NGramPrior(new[] { Pack() }) { Lambda = 0.5, HalfCount = 4 };
        var logits = new float[10];
        for (int i = 0; i < 10; i++)
        {
            logits[i] = 0.1f * i;   // the model barely prefers anything
        }
        var output = new float[10];
        Assert.True(prior.Apply(logits, new[] { 1, 2, 3 }, output));

        double Softmax(float[] l, int t) => Math.Exp(l[t]) / l.Sum(v => Math.Exp(v));
        double lambda = 0.5 * 4 / (4 + 4.0);            // evidence 4: half of λ_max
        for (int t = 0; t < 10; t++)
        {
            double pack = t == 4 ? 0.75 : t == 5 ? 0.25 : 0;
            double expected = (1 - lambda) * Softmax(logits, t) + lambda * pack;
            Assert.Equal(expected, Softmax(output, t), 6);
        }
        // Tokens the pack does not name keep their relative odds.
        Assert.Equal(logits[7] - logits[2], output[7] - output[2], 5);
    }

    [Fact]
    public void The_Gates_Keep_The_Pack_Silent_Where_It_Has_Nothing_To_Add()
    {
        var pack = Pack();
        var output = new float[10];
        var sure = new float[10];
        sure[6] = 20f;                                   // the model is certain of token 6
        Assert.False(new NGramPrior(new[] { pack }) { ModelConfident = 0.9 }.Apply(sure, new[] { 1, 2, 3 }, output));
        Assert.True(new NGramPrior(new[] { pack }).Apply(sure, new[] { 1, 2, 3 }, output));
        // The pack's top continuation has 3/4 of "1 2 3": silent if it must have 0.8.
        Assert.False(new NGramPrior(new[] { pack }) { PackConfident = 0.8 }.Apply(new float[10], new[] { 1, 2, 3 }, output));
        // A one-token context is below the default minimum context of 2.
        Assert.False(new NGramPrior(new[] { pack }).Apply(new float[10], new[] { 3 }, output));
    }

    [Fact]
    public void Code_Is_Inside_A_Fence_Past_Its_Opening_Line()
    {
        Assert.False(NGramPrior.InFencedCode("Here is the program:"));
        Assert.False(NGramPrior.InFencedCode("Here it is:\n```csharp"));           // still on the fence line
        Assert.True(NGramPrior.InFencedCode("Here it is:\n```csharp\nusing System;\nFile."));
        Assert.False(NGramPrior.InFencedCode("```cs\nint x;\n```\nThat builds."));  // the block closed
    }
}
