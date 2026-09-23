using Amql.Cli;
using Amql.Hf;
using Amql.Inference;
using Amql.Safetensors;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

public class LfmImportTests
{
    [Fact]
    public void LfmImport_Config_Facts()
    {
        using var dir = new TempDir();
        SyntheticLfmCheckpoint.Write(dir.Path);

        var facts = ModelConfig.ReadTextFacts(Path.Combine(dir.Path, "config.json"));
        Assert.Equal(SyntheticLfmCheckpoint.Family, facts.ModelType);
        Assert.Equal(SyntheticLfmCheckpoint.Hidden, facts.HiddenSize);
        Assert.Equal(SyntheticLfmCheckpoint.NumQHeads, facts.NumQueryHeads);
        Assert.Equal(SyntheticLfmCheckpoint.NumKvHeads, facts.NumKvHeads);
        Assert.Equal(SyntheticLfmCheckpoint.HeadDim, facts.HeadDim);
        Assert.Equal(2, facts.LayerTypes.Count);
        Assert.Equal("conv", facts.LayerTypes[0]);
        Assert.Equal("full_attention", facts.LayerTypes[1]);
        Assert.True(facts.TieWordEmbeddings);
    }

    [Fact]
    public void LfmImport_ConvLayer_Maps_Conv_Operator()
    {
        using var dir = new TempDir();
        SyntheticLfmCheckpoint.Write(dir.Path);

        var facts = ModelConfig.ReadTextFacts(Path.Combine(dir.Path, "config.json"));
        using var inventory = HfInventory.Open(dir.Path);
        var spec = ArchMapper.MapToContainerSpec("lfm-test", facts, inventory,
            new ArchMapper.EncodeOptions());

        var containerDir = Path.Combine(dir.Path, "container");
        ContainerEncoder.Encode(containerDir, spec);

        using var container = Vindex3Container.Open(containerDir);
        var attention = container.Graph!.Components[0].Attention!;
        Assert.Equal(LayerOperators.Conv, attention[0].Operator);
        Assert.Equal(LayerOperators.Softmax, attention[1].Operator);

        // Verify the container can export back
        var exportDir = Path.Combine(dir.Path, "exported");
        using (var c = Vindex3Container.Open(containerDir))
        {
            ModelExporter.Export(c, exportDir, patch: null);
        }
        using var file = SafetensorsFile.Open(Path.Combine(exportDir, "model.safetensors"));
        Assert.True(file.Contains("model.layers.0.operator_norm.weight"));
        Assert.True(file.Contains("model.layers.0.conv.in_proj.weight"));
        Assert.True(file.Contains("model.layers.1.self_attn.q_proj.weight"));
    }

    [Fact]
    public void LfmImport_AttentionLayer_Has_Qk_Norms()
    {
        using var dir = new TempDir();
        SyntheticLfmCheckpoint.Write(dir.Path);

        var facts = ModelConfig.ReadTextFacts(Path.Combine(dir.Path, "config.json"));
        using var inventory = HfInventory.Open(dir.Path);
        var spec = ArchMapper.MapToContainerSpec("lfm-test", facts, inventory,
            new ArchMapper.EncodeOptions());

        var containerDir = Path.Combine(dir.Path, "container");
        ContainerEncoder.Encode(containerDir, spec);

        using var container = Vindex3Container.Open(containerDir);
        var surface = container.Graph!.Components[0].Execution!;
        var attnSurface = surface.Attention!;
        Assert.True(attnSurface.QkNorm);
        Assert.Equal(32, attnSurface.QkNormDim);
    }
}