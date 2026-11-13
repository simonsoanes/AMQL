using System.Text.Json;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Property-driven model support, measured: every model in
/// <c>fixtures/arch-zoo</c> (made by <c>make_arch_zoo.py</c>) is a family no
/// AMQL code names. Each one must encode from its config and tensors alone and
/// reproduce transformers' own logits.
/// </summary>
public class ArchZooTests
{
    private static string Zoo => Path.Combine(AppContext.BaseDirectory, "fixtures", "arch-zoo");

    public static TheoryData<string> Families()
    {
        var data = new TheoryData<string>();
        foreach (var dir in Directory.EnumerateDirectories(Zoo).Order(StringComparer.Ordinal))
        {
            data.Add(Path.GetFileName(dir));
        }
        return data;
    }

    internal static (int[] Ids, float[][] Logits) Golden(string family)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Zoo, family, "golden.json")));
        var ids = doc.RootElement.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var logits = doc.RootElement.GetProperty("logits").EnumerateArray()
            .Select(r => r.EnumerateArray().Select(v => v.GetSingle()).ToArray()).ToArray();
        return (ids, logits);
    }

    internal static float[][] RunPrefill(string containerDir, int[] ids)
    {
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var runtime = new GenericRuntime(plan, store);
        var hidden = runtime.Embed(ids);
        var positions = Enumerable.Range(0, ids.Length).ToArray();
        for (int l = 0; l < plan.Layers.Count; l++)
        {
            hidden = runtime.RunLayerInternal(hidden, l, positions, positions, appendKv: true);
        }
        var logits = runtime.FinalNormAndHead(hidden);
        return Enumerable.Range(0, ids.Length).Select(t => logits.Row(t).ToArray()).ToArray();
    }

    internal static double WorstDiff(float[][] mine, float[][] reference)
    {
        double worst = 0;
        for (int t = 0; t < reference.Length; t++)
        {
            for (int i = 0; i < reference[t].Length; i++)
            {
                worst = Math.Max(worst, Math.Abs(mine[t][i] - reference[t][i]));
            }
        }
        return worst;
    }

    [Theory]
    [MemberData(nameof(Families))]
    public void Encodes_And_Matches_Transformers_Logits(string family)
    {
        var (ids, golden) = Golden(family);
        using var dir = new TempDir();
        string containerDir = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(Path.Combine(Zoo, family), containerDir, family,
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });

        double worst = WorstDiff(RunPrefill(containerDir, ids), golden);
        Assert.True(worst < 2e-3, $"{family}: worst logit diff {worst:G4}");
    }

    /// <summary>GGUF export picks the llama.cpp architecture from the model's
    /// properties and names tensors by role; a model whose arithmetic no
    /// supported architecture reproduces is refused, naming the property.
    /// (Numerics: scripts/export/check_gguf.py reads the file back under
    /// llama.cpp's rules and compares with transformers.)</summary>
    [Theory]
    [InlineData("llama", "llama")]
    [InlineData("phi3", "llama")]
    [InlineData("qwen2", "qwen2")]
    [InlineData("qwen3", "qwen3")]
    [InlineData("granite", "granite")]
    [InlineData("olmo2", "olmo2")]
    [InlineData("llama-rope-linear", "llama")]
    [InlineData("llama3-scaled", "llama")]
    [InlineData("gemma2", null, "no llama.cpp architecture")]
    [InlineData("starcoder2", null, "no llama.cpp architecture")]
    [InlineData("qwen2-yarn", null, "rope scaling 'yarn'")]
    public void Gguf_Export_Chooses_Architecture_From_Properties(string family, string? arch, string? refusal = null)
    {
        using var dir = new TempDir();
        string containerDir = Path.Combine(dir.Path, "c");
        string gguf = Path.Combine(dir.Path, "m.gguf");
        ModelToContainer.Encode(Path.Combine(Zoo, family), containerDir, family,
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });
        using var container = Vindex3Container.Open(containerDir);
        if (arch is null)
        {
            var refused = Assert.ThrowsAny<Exception>(() => Amql.Cli.GgufExport.Export(container, gguf));
            Assert.Contains(refusal!, refused.Message);
            return;
        }
        var report = Amql.Cli.GgufExport.Export(container, gguf);
        Assert.Equal(arch, report.Architecture);
        using var reader = Amql.Gguf.GgufReader.Open(gguf);
        Assert.Equal(arch, reader.Arch);
        Assert.True(reader.Contains("blk.0.attn_q.weight"));
        Assert.True(reader.Contains("blk.3.ffn_down.weight"));
        Assert.True(reader.Contains("output_norm.weight"));
        // Phi-3's fused projections were split on import and stay split here.
        Assert.False(reader.Contains("blk.0.attn_qkv.weight"));
    }

    /// <summary>The HF export is the checkpoint again: its own config keys
    /// (architectures, model_type, family-only knobs) and its own tensor names —
    /// fused tensors fused back — and it re-imports to the same logits.</summary>
    [Theory]
    [MemberData(nameof(Families))]
    public void Hf_Export_Round_Trips(string family)
    {
        var (ids, golden) = Golden(family);
        using var dir = new TempDir();
        string source = Path.Combine(Zoo, family);
        string containerDir = Path.Combine(dir.Path, "c");
        string exportDir = Path.Combine(dir.Path, "hf");
        string again = Path.Combine(dir.Path, "c2");
        var options = new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false };
        ModelToContainer.Encode(source, containerDir, family, options);
        using (var container = Vindex3Container.Open(containerDir))
        {
            Amql.Cli.ModelExporter.Export(container, exportDir, patch: null);
        }

        using (var original = Amql.Safetensors.ModelDirectory.Open(source))
        using (var exported = Amql.Safetensors.ModelDirectory.Open(exportDir))
        {
            Assert.Equal(original.TensorNames.Order(StringComparer.Ordinal), exported.TensorNames.Order(StringComparer.Ordinal));
        }
        using (var a = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(source, "config.json"))))
        using (var b = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(exportDir, "config.json"))))
        {
            foreach (var key in a.RootElement.EnumerateObject())
            {
                Assert.True(b.RootElement.TryGetProperty(key.Name, out var exportedValue), $"{family}: export dropped config key '{key.Name}'");
                if (key.Value.ValueKind is not JsonValueKind.Number)
                {
                    Assert.Equal(key.Value.GetRawText(), exportedValue.GetRawText());
                }
            }
        }

        ModelToContainer.Encode(exportDir, again, family, options);
        double worst = WorstDiff(RunPrefill(again, ids), golden);
        Assert.True(worst < 2e-3, $"{family}: re-imported export worst logit diff {worst:G4}");
    }
}
