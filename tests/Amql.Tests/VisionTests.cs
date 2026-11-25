using System.Text.Json;
using Amql.Hf;
using Amql.Inference;
using Amql.Vindex3;

namespace Amql.Tests;

/// <summary>
/// Images: files are read as Pillow reads them, cut into the patches transformers cuts, and a
/// tower on the GPU gives the model what transformers gives it (fixtures/gemma4-vision, made by
/// make_gemma4_vision.py), in a prompt whose windowed layers see each image whole.
/// </summary>
[Collection("gpu")]
public sealed class VisionTests
{
    private static string Dir => Path.Combine(AppContext.BaseDirectory, "fixtures", "gemma4-vision");

    private static readonly JsonElement Golden = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Dir, "golden.json"))).RootElement;

    private static float[] Floats(JsonElement e) => e.EnumerateArray().Select(v => v.GetSingle()).ToArray();

    public static IEnumerable<object[]> Files() => Golden.GetProperty("files").EnumerateObject().Select(f => new object[] { f.Name });

    [Theory]
    [MemberData(nameof(Files))]
    public void A_File_Reads_As_Pillow_Reads_It(string name)
    {
        var expected = Golden.GetProperty("files").GetProperty(name);
        var image = ImageFile.Read(File.ReadAllBytes(Path.Combine(Dir, name)));
        Assert.Equal((expected.GetProperty("width").GetInt32(), expected.GetProperty("height").GetInt32()), (image.Width, image.Height));
        var pixels = expected.GetProperty("pixels").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var apart = pixels.Zip(image.Pixels).Select(p => Math.Abs(p.First - p.Second)).ToArray();
        Assert.True(apart.Max() <= (name == "alpha.png" ? 1 : 0), $"{apart.Count(d => d > 0)} of {apart.Length} differ, by {apart.Max()} at most");
    }

    [Fact]
    public void A_File_That_Is_Damaged_Or_No_Image_Is_Refused()
    {
        Assert.Throws<ImageException>(() => ImageFile.Read([1, 2, 3]));
        Assert.Contains("WebP", Assert.Throws<ImageException>(() => ImageFile.Read([.. "RIFF0000WEBP"u8])).Message);
        foreach (string name in new[] { "rgb.png", "photo.jpg", "photo-progressive.jpg" })
        {
            byte[] whole = File.ReadAllBytes(Path.Combine(Dir, name));
            var random = new Random(5);
            for (int i = 0; i < 300; i++)
            {
                var bytes = (byte[])whole.Clone();
                for (int k = 0; k < 1 + i % 5; k++)
                {
                    bytes[random.Next(8, bytes.Length)] = (byte)random.Next(256);
                }
                try
                {
                    var image = ImageFile.Read(i % 3 == 0 ? bytes[..random.Next(9, bytes.Length)] : bytes);
                    Assert.Equal(3 * image.Width * image.Height, image.Pixels.Length);
                }
                catch (ImageException)
                {
                }
            }
        }
    }

    private static string Encode(TempDir dir)
    {
        string c = Path.Combine(dir.Path, "c");
        ModelToContainer.Encode(Dir, c, "gemma4-vision");
        return c;
    }

    [Fact]
    public void An_Image_Is_Cut_Into_The_Patches_Transformers_Cuts()
    {
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(Encode(dir));
        using var store = container.CreateOperandStore();
        var plan = VisionPlan.Build(container, store, out string? reason);
        Assert.Null(reason);
        byte[] picture = File.ReadAllBytes(Path.Combine(Dir, "picture.png"));
        var image = plan!.Prepare(picture, Golden.GetProperty("tokens_at_most").GetInt32());
        Assert.Equal(Golden.GetProperty("tokens").GetInt32(), image.Tokens);
        Assert.Equal(Golden.GetProperty("across").GetInt32(), image.Across);
        Assert.Equal(Golden.GetProperty("count").GetInt32(), image.Across * image.Down);
        float[] first = Floats(Golden.GetProperty("first")), last = Floats(Golden.GetProperty("last"));
        for (int i = 0; i < first.Length; i++)
        {
            Assert.Equal(2f * (first[i] - 0.5f), image.Patches[i], 1e-5f);
            Assert.Equal(2f * (last[i] - 0.5f), image.Patches[image.Patches.Length - last.Length + i], 1e-5f);
        }
        // Its own ids, the same each time, and another picture's are others.
        Assert.Equal(image.Ids(), plan.Prepare(picture, 12).Ids());
        Assert.All(image.Ids(), id => Assert.True(id < 0));
        Assert.NotEqual(image.Ids()[0], plan.Prepare(File.ReadAllBytes(Path.Combine(Dir, "rgb.png")), 12).Ids()[0]);
        // A strip keeps its length rather than its shape.
        Assert.Equal((12 * 12, 12), plan.SizeFor(4000, 2, 12));
    }

    [Theory]
    [InlineData("f32")]
    [InlineData("fp16")]
    [InlineData("nvfp4")]
    public void A_Prompt_With_An_Image_Gives_What_Transformers_Gives(string kv)
    {
        CudaShim.Reset();
        if (!CudaShim.Enabled)
        {
            return;
        }
        string before = GpuEngine.RequestedKvFormat;
        using var dir = new TempDir();
        using var container = Vindex3Container.Open(Encode(dir));
        using var store = container.CreateOperandStore();
        try
        {
            GpuEngine.RequestedKvFormat = kv;
            var plan = Planner.Plan(container, "target", store);
            var vision = VisionPlan.Build(container, store, out _)!;
            using var host = new DecodeSession(plan, store);
            host.Prefill(new[] { 1 });
            var engine = host.DeviceEngine()!;
            engine.UseVision(vision, store);
            Assert.True(vision.WholeImages);
            var image = vision.Prepare(File.ReadAllBytes(Path.Combine(Dir, "picture.png")), 12);
            var features = engine.EncodeImage(image);
            var expected = Golden.GetProperty("features").EnumerateArray().SelectMany(Floats).ToArray();
            Assert.Equal(expected.Length, features.Length);
            double error = Math.Sqrt(expected.Zip(features).Sum(p => Math.Pow(p.First - p.Second, 2)) / expected.Sum(v => v * (double)v));
            Assert.True(error < 0.02, $"features {error:P2} apart");

            engine.ExactPrefill = true;   // random routers make much of BF16's rounding
            var ids = Golden.GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            int start = Array.IndexOf(ids, -1);
            image.Ids().CopyTo(ids, start);
            var seq = engine.NewSequence();
            Assert.Throws<ArgumentException>(() => engine.Prefill(seq, ids));   // the image was not given
            engine.Reset(seq);
            seq.Images = [new ImageSpan(start, image.Tokens, expected)];
            var logits = Golden.GetProperty("logits").EnumerateArray().Select(Floats).ToArray();
            static double Apart(float[] a, float[] b) => a.Zip(b).Max(p => Math.Abs(p.First - p.Second)) / b.Max(Math.Abs);
            var row = engine.Prefill(seq, ids);
            if (kv == "f32")
            {
                Assert.True(Apart(row, logits[0]) < 1e-3, $"after the prompt: {Apart(row, logits[0]):P2}");
                foreach (int token in Golden.GetProperty("tokens_after").EnumerateArray().Select(v => v.GetInt32()).Take(3))
                {
                    row = engine.Step(new[] { seq }, new[] { token })[0];
                }
                Assert.True(Apart(row, logits[3]) < 1e-3, $"three tokens on: {Apart(row, logits[3]):P2}");
            }
            else
            {
                // The caches' format is back as it was after the tower ran: the prompt read again gives the same.
                engine.Reset(seq);
                engine.EncodeImage(image);
                Assert.Equal(row, engine.Prefill(seq, ids));
            }
        }
        finally
        {
            GpuEngine.RequestedKvFormat = before;
        }
    }
}
