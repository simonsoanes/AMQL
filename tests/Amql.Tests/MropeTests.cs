using System.Text.Json;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;
using Xunit;

namespace Amql.Tests;

/// <summary>
/// Multimodal rotary, measured against transformers: the text decoders of a
/// tiny Qwen2-VL (contiguous sections) and Qwen3-VL (interleaved), run with
/// one position per token and with (t, h, w) streams laid out around an image
/// grid (<c>fixtures/make_mrope_zoo.py</c>).
/// </summary>
public class MropeTests
{
    private static string Zoo => Path.Combine(AppContext.BaseDirectory, "fixtures", "mrope-zoo");

    private static (int[] Ids, int[][]? Streams, float[][] Logits) Golden(string model, string file)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Zoo, model, file)));
        var root = doc.RootElement;
        var ids = root.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        int[][]? streams = root.TryGetProperty("streams", out var s)
            ? s.EnumerateArray().Select(r => r.EnumerateArray().Select(e => e.GetInt32()).ToArray()).ToArray()
            : null;
        var logits = root.GetProperty("logits").EnumerateArray()
            .Select(r => r.EnumerateArray().Select(v => v.GetSingle()).ToArray()).ToArray();
        return (ids, streams, logits);
    }

    private static float[][] Run(string containerDir, int[] ids, int[][]? streams)
    {
        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        var runtime = new GenericRuntime(plan, store);
        var hidden = runtime.Embed(ids);
        var positions = Enumerable.Range(0, ids.Length).ToArray();
        for (int l = 0; l < plan.Layers.Count; l++)
        {
            hidden = runtime.RunLayerInternal(hidden, l, positions, positions, appendKv: true, positionStreams: streams);
        }
        var logits = runtime.FinalNormAndHead(hidden);
        return Enumerable.Range(0, ids.Length).Select(t => logits.Row(t).ToArray()).ToArray();
    }

    [Theory]
    [InlineData("qwen2-vl", false)]
    [InlineData("qwen3-vl", true)]
    public void Text_And_Image_Grid_Positions_Match_Transformers(string model, bool interleaved)
    {
        using var dir = new TempDir();
        string containerDir = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(Path.Combine(Zoo, model), containerDir, model,
            new ArchMapper.EncodeOptions { IncludeVision = false, IncludeMtp = false });

        using (var container = Vindex3Container.Open(containerDir))
        {
            var policy = container.Graph!.Components.First(c => c.Attention is not null).Attention![0];
            var rope = Assert.IsType<PositionRope>(policy.Position);
            Assert.Equal(new[] { 2, 3, 3 }, rope.Mrope!.Section);
            Assert.Equal(interleaved, rope.Mrope.Interleaved);
        }

        var (ids, _, text) = Golden(model, "golden.json");
        double worstText = ArchZooTests.WorstDiff(Run(containerDir, ids, null), text);
        Assert.True(worstText < 2e-3, $"{model} text: worst logit diff {worstText:G4}");

        var (_, streams, grid) = Golden(model, "golden_streams.json");
        double worstGrid = ArchZooTests.WorstDiff(Run(containerDir, ids, streams), grid);
        Assert.True(worstGrid < 2e-3, $"{model} streams: worst logit diff {worstGrid:G4}");
    }

    [Fact]
    public void Interleaved_Streams_Cycle_Until_Each_Share_Is_Spent()
    {
        // Qwen3-VL's [24, 20, 20]: h on 1, 4, …, 58; w on 2, 5, …, 59; t elsewhere.
        var mrope = new RopeMultimodal { Section = new[] { 24, 20, 20 }, Interleaved = true };
        Assert.Equal(new[] { 0, 1, 2, 0, 1, 2 }, Enumerable.Range(0, 6).Select(mrope.StreamOf));
        Assert.Equal(1, mrope.StreamOf(58));
        Assert.Equal(2, mrope.StreamOf(59));
        Assert.Equal(0, mrope.StreamOf(60));
        Assert.Equal(0, mrope.StreamOf(61));
        var contiguous = new RopeMultimodal { Section = new[] { 16, 24, 24 } };
        Assert.Equal(0, contiguous.StreamOf(15));
        Assert.Equal(1, contiguous.StreamOf(16));
        Assert.Equal(2, contiguous.StreamOf(40));
    }
}
