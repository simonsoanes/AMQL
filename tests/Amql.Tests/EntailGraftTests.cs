using System.Text.Json.Nodes;
using Amql.Entail;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Grafting judged from the host's own properties: a small program goes into
/// zoo models of many layer shapes (<c>fixtures/arch-zoo</c>). Where the
/// program is silent the host must keep transformers' logits; where it fires
/// its token must win; a host the block cannot be written into exactly must
/// be refused, naming why.
/// </summary>
public class EntailGraftTests
{
    /// <summary>After an "a", the next position says "b".</summary>
    private const string Program = """
        state x, y, z, w
        token "a" means x = 1
        attend back 1
            y <- x
        end
        entail z when y >= 1
        entail w when z >= 1
        emit "b" when w
        """;

    private const int A = 'a', B = 'b';

    /// <summary>A copy of a zoo model with a byte-level tokenizer over its 128
    /// ids, so the program's tokens bind: printable ASCII is itself.</summary>
    private static string HostCopy(string family, string dir)
    {
        string source = Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo", family);
        Directory.CreateDirectory(dir);
        foreach (var file in Directory.GetFiles(source))
        {
            File.Copy(file, Path.Combine(dir, Path.GetFileName(file)));
        }
        var vocab = new JsonObject();
        for (int i = 0; i < 128; i++)
        {
            vocab[i == ' ' ? "Ġ" : i > ' ' && i < 127 ? ((char)i).ToString() : $"<t{i}>"] = i;
        }
        var tokenizer = new JsonObject
        {
            ["version"] = "1.0",
            ["added_tokens"] = new JsonArray(),
            ["model"] = new JsonObject { ["type"] = "BPE", ["vocab"] = vocab, ["merges"] = new JsonArray() },
        };
        File.WriteAllText(Path.Combine(dir, "tokenizer.json"), tokenizer.ToJsonString());
        return dir;
    }

    private static string Graft(string family, string root)
    {
        string host = HostCopy(family, Path.Combine(root, "host"));
        var judged = EntailHost.Judge(host);
        var p = EntailCompiler.Compile(Program, Math.Min(3584, judged.Target.Intermediate), judged.Target.ProgramHeads);
        string grafted = Path.Combine(root, "graft");
        EntailWeights.WriteGraft(p, judged, grafted, context: 64);
        string container = Path.Combine(root, "c");
        ModelToContainer.Encode(grafted, container, family + "-graft",
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });
        return container;
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
        {
            if (row[i] > row[best])
            {
                best = i;
            }
        }
        return best;
    }

    [Theory]
    [InlineData("llama")]          // pre-norm, no q/k norm, square o_proj
    [InlineData("llama-bias")]     // zero biases in the block
    [InlineData("qwen3")]          // per-head q/k norm
    [InlineData("qwen2-yarn")]     // yarn's attention factor on cos and sin
    [InlineData("phi3")]           // fused qkv_proj and gate_up_proj
    [InlineData("phi3-longrope-short")] // a context within longrope's first regime
    [InlineData("glm4")]           // sandwich norms, interleaved partial rope
    [InlineData("gemma3_text")]    // sandwich, 1+w norms, sqrt(hidden) embedding, tied, sliding pattern
    [InlineData("olmo2")]          // post-norms only, q/k norm over the whole projection
    [InlineData("smollm3")]        // no_rope_layers
    [InlineData("olmo3")]          // layer_types with sliding layers
    [InlineData("ministral")]      // every layer windowed
    [InlineData("mixtral")]        // replicated experts, renormalised top-2
    [InlineData("olmoe")]          // replicated experts, top-2 of 4 not renormalised: down weights ÷ 1/2
    [InlineData("qwen2_moe")]      // shared expert behind its sigmoid gate
    [InlineData("qwen3_moe")]      // dense first layer declared by mlp_only_layers
    public void A_Graft_Keeps_The_Host_And_Answers(string family)
    {
        using var dir = new TempDir();
        string container = Graft(family, dir.Path);
        string original = Path.Combine(dir.Path, "original");
        ModelToContainer.Encode(Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo", family), original, family,
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });

        // Silent: the golden sequence has no "a", so the host's logits stand.
        var (ids, _) = ArchZooTests.Golden(family);
        // Widening leaves each host norm's epsilon where it was while the mean
        // it is added to shrinks, so the host moves by a little, in proportion.
        var reference = ArchZooTests.RunPrefill(original, ids);
        double tolerance = 1e-3 * Math.Max(1, reference.Max(r => r.Max(Math.Abs)));
        double worst = ArchZooTests.WorstDiff(ArchZooTests.RunPrefill(container, ids), reference);
        Assert.True(worst < tolerance, $"{family}: grafted host moved by {worst:G4}");

        // Fires: the token after an "a" is "b"; every other position is the host's.
        int[] firing = { 1, 7, A, 42, 3 };
        var grafted = ArchZooTests.RunPrefill(container, firing);
        var host = ArchZooTests.RunPrefill(original, firing);
        Assert.Equal(B, ArgMax(grafted[3]));
        Assert.True(grafted[3][B] - host[3].Max() > 100, $"{family}: 'b' wins by too little");
        foreach (int row in new[] { 0, 1, 2, 4 })
        {
            double moved = grafted[row].Zip(host[row], (x, y) => Math.Abs(x - y)).Max();
            Assert.True(moved < tolerance, $"{family}: silent position {row} moved by {moved:G4}");
        }
    }

    [Theory]
    [InlineData("stablelm", "LayerNorm")]
    [InlineData("cohere", "LayerNorm")]
    [InlineData("starcoder2", "LayerNorm")]
    [InlineData("granite", "embedding multiplier")]
    [InlineData("llama-dynamic", "dynamic rope")]
    [InlineData("gemma2", "makes every head a lookup")]           // its softcap flattens the peak
    [InlineData("phi3-longrope-long", "makes every head a lookup")] // its regimes share one pair
    public void A_Host_The_Block_Cannot_Be_Exact_In_Is_Refused(string family, string reason)
    {
        using var dir = new TempDir();
        string host = HostCopy(family, Path.Combine(dir.Path, "host"));
        var e = Assert.ThrowsAny<EntailException>(() =>
        {
            var judged = EntailHost.Judge(host);
            var p = EntailCompiler.Compile(Program, Math.Min(3584, judged.Target.Intermediate), judged.Target.ProgramHeads);
            EntailWeights.WriteGraft(p, judged, Path.Combine(dir.Path, "graft"), context: 64);
        });
        Assert.Contains(reason, e.Message);
    }
}
