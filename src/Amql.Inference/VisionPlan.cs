using System.Security.Cryptography;
using System.Text.Json;
using Amql.Vindex3;

namespace Amql.Inference;

/// <summary>An image made ready for a vision tower: its patches, and how many tokens it takes.</summary>
public sealed class PreparedImage
{
    /// <summary>Patches across and down.</summary>
    public required int Across { get; init; }
    public required int Down { get; init; }

    /// <summary>[Across × Down, patch² × 3], RGB in [−1, 1], row by row.</summary>
    public required float[] Patches { get; init; }

    /// <summary>The tokens it becomes.</summary>
    public required int Tokens { get; init; }

    /// <summary>What the tokens are told apart by: the same picture at the same size has the same key.</summary>
    public required long Key { get; init; }

    /// <summary>The ids that stand for the image in a prompt: below zero, where no token is, and its
    /// own, so a prompt with another picture in the same place is another prompt.</summary>
    public int[] Ids()
    {
        var ids = new int[Tokens];
        for (int i = 0; i < Tokens; i++)
        {
            ulong x = (ulong)Key + (ulong)i * 0x9E3779B97F4A7C15UL;
            x = (x ^ x >> 30) * 0xBF58476D1CE4E5B9UL;
            x = (x ^ x >> 27) * 0x94D049BB133111EBUL;
            ids[i] = -1 - (int)((x ^ x >> 31) & 0x3FFFFFFF);
        }
        return ids;
    }
}

/// <summary>An image's tokens where they stand in a prompt: what the tower made of it,
/// [<see cref="Tokens"/> × hidden].</summary>
public sealed record ImageSpan(int Start, int Tokens, float[] Features);

/// <summary>
/// Gemma 4's vision tower, read from the config a container carries for it: patches with a
/// position from two tables, encoder layers whose heads are rotated by a patch's two coordinates,
/// and soft tokens pooled from squares of patches and projected to the model's width.
/// </summary>
public sealed class VisionPlan
{
    public const string Tower = "vision.perception_tower", Projector = "vision.projector";

    public required int HiddenSize { get; init; }
    public required int Layers { get; init; }
    public required int Heads { get; init; }
    public required int HeadDim { get; init; }
    public required int Intermediate { get; init; }
    public required int PatchSize { get; init; }
    public required int Side { get; init; }
    public required int Positions { get; init; }
    public required double Theta { get; init; }
    public required float Eps { get; init; }
    public required bool Standardise { get; init; }

    /// <summary>The soft tokens an image takes at most, unless asked otherwise.</summary>
    public required int SoftTokens { get; init; }

    /// <summary>In the layers with a window, each token of an image sees all of that image.</summary>
    public required bool WholeImages { get; init; }

    /// <summary>The token that marks an image in a rendered prompt, and those put around its tokens.</summary>
    public required int ImageToken { get; init; }
    public required int BeginToken { get; init; }
    public required int EndToken { get; init; }

    /// <summary>The soft-token limits the tower was trained with.</summary>
    public static readonly int[] Budgets = [70, 140, 280, 560];

    /// <summary>The container's tower, or null with why it is not run.</summary>
    public static VisionPlan? Build(Vindex3Container container, OperandStore store, out string? reason)
    {
        reason = null;
        var graph = container.Graph;
        var objects = graph?.Objects;
        if (objects is null || !objects.Any(o => o.Id == Tower && o.Representations.Count > 0) || !objects.Any(o => o.Id == Projector && o.Representations.Count > 0))
        {
            reason = "the container has no vision tower";
            return null;
        }
        var facts = graph!.Components.FirstOrDefault(c => c.Role == ComponentRole.Perception)?.Perception;
        if (facts is not { ValueKind: JsonValueKind.Object } p || !p.TryGetProperty("hf_wrapper", out var wrapper) ||
            !wrapper.TryGetProperty("vision_config", out var v) || v.ValueKind != JsonValueKind.Object)
        {
            reason = "the container does not say what its vision tower is";
            return null;
        }
        string type = v.TryGetProperty("model_type", out var t) ? t.GetString() ?? "" : "";
        if (type != "gemma4_vision")
        {
            reason = $"a '{type}' vision tower is not one this build runs (gemma4_vision)";
            return null;
        }
        int Int(JsonElement e, string name) => e.TryGetProperty(name, out var x) && x.ValueKind == JsonValueKind.Number
            ? x.GetInt32() : throw new ContainerException($"the vision tower's config has no '{name}'");
        bool On(JsonElement e, string name) => e.TryGetProperty(name, out var x) && x.ValueKind == JsonValueKind.True;
        if (On(v, "use_clipped_linears") || On(v, "attention_bias") || Int(v, "num_attention_heads") != Int(v, "num_key_value_heads") ||
            !(v.TryGetProperty("hidden_activation", out var act) && act.GetString() == "gelu_pytorch_tanh"))
        {
            reason = "the vision tower has clipped projections, biases, shared heads or another activation, which this build does not run";
            return null;
        }
        return new VisionPlan
        {
            HiddenSize = Int(v, "hidden_size"), Layers = Int(v, "num_hidden_layers"), Heads = Int(v, "num_attention_heads"),
            HeadDim = Int(v, "head_dim"), Intermediate = Int(v, "intermediate_size"), PatchSize = Int(v, "patch_size"),
            Side = Int(v, "pooling_kernel_size"), Positions = Int(v, "position_embedding_size"),
            Theta = v.GetProperty("rope_parameters").GetProperty("rope_theta").GetDouble(), Eps = v.GetProperty("rms_norm_eps").GetSingle(),
            Standardise = On(v, "standardize"), SoftTokens = Int(v, "default_output_length"),
            WholeImages = SeesWhole(container, wrapper),
            ImageToken = Int(wrapper, "image_token_id"), BeginToken = Int(wrapper, "boi_token_id"), EndToken = Int(wrapper, "eoi_token_id"),
        };
    }

    /// <summary>Whether the text model sees each image whole: a fact of its own config, which the
    /// container keeps as the checkpoint had it.</summary>
    private static bool SeesWhole(Vindex3Container container, JsonElement wrapper)
    {
        static bool Says(JsonElement config) => config.TryGetProperty("text_config", out var text) &&
            text.TryGetProperty("use_bidirectional_attention", out var b) && b.ValueKind == JsonValueKind.String && b.GetString() == "vision";
        if (Says(wrapper))
        {
            return true;
        }
        string path = Path.Combine(container.Root, "source_config.json");
        if (!File.Exists(path))
        {
            return false;
        }
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        return Says(doc.RootElement);
    }

    /// <summary>The size an image is resized to: the largest with its shape whose patches pool to
    /// at most <paramref name="softTokens"/> whole tokens.</summary>
    public (int Width, int Height) SizeFor(int width, int height, int softTokens)
    {
        int unit = Side * PatchSize;
        double target = (double)softTokens * unit * unit;
        double factor = Math.Sqrt(target / ((double)width * height));
        int h = (int)Math.Floor(factor * height / unit) * unit, w = (int)Math.Floor(factor * width / unit) * unit;
        int longest = softTokens * unit;
        if (h == 0 && w == 0)
        {
            throw new ImageException("the image is too small to read");
        }
        if (h == 0)
        {
            h = unit;
            w = Math.Min(Math.Max(1, width / height) * unit, longest);
        }
        else if (w == 0)
        {
            w = unit;
            h = Math.Min(Math.Max(1, height / width) * unit, longest);
        }
        return (w, h);
    }

    /// <summary>An image file, resized and cut into patches.</summary>
    public PreparedImage Prepare(byte[] file, int? softTokens = null) => Prepare(ImageFile.Read(file), softTokens);

    public PreparedImage Prepare(RgbImage image, int? softTokens = null)
    {
        int budget = softTokens ?? SoftTokens;
        var (w, h) = SizeFor(image.Width, image.Height, budget);
        var sized = ImageResize.Bicubic(image, w, h);
        int across = w / PatchSize, down = h / PatchSize, values = 3 * PatchSize * PatchSize;
        var patches = new float[(long)across * down * values];
        Parallel.For(0, down, py =>
        {
            for (int px = 0; px < across; px++)
            {
                int at = (py * across + px) * values;
                for (int y = 0; y < PatchSize; y++)
                {
                    int from = 3 * ((py * PatchSize + y) * w + px * PatchSize);
                    for (int i = 0; i < 3 * PatchSize; i++)
                    {
                        patches[at++] = 2f * (sized.Pixels[from + i] / 255f - 0.5f);
                    }
                }
            }
        });
        var hash = SHA256.HashData(sized.Pixels);
        return new PreparedImage
        {
            Across = across, Down = down, Patches = patches, Tokens = across * down / (Side * Side),
            Key = BitConverter.ToInt64(hash) ^ ((long)w << 32 | (uint)h),
        };
    }
}

/// <summary>Bicubic resizing with antialiasing, as Pillow does it.</summary>
public static class ImageResize
{
    private const int Precision = 22;

    public static RgbImage Bicubic(RgbImage image, int width, int height)
    {
        if (image.Width == width && image.Height == height)
        {
            return image;
        }
        byte[] wide = image.Pixels;
        if (image.Width != width)
        {
            var (starts, weights, size) = Coefficients(image.Width, width);
            wide = new byte[checked(3 * width * image.Height)];
            Parallel.For(0, image.Height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    int from = 3 * (y * image.Width + starts[x].First), n = starts[x].Count;
                    for (int c = 0; c < 3; c++)
                    {
                        int sum = 1 << (Precision - 1);
                        for (int i = 0; i < n; i++)
                        {
                            sum += image.Pixels[from + 3 * i + c] * weights[x * size + i];
                        }
                        wide[3 * (y * width + x) + c] = Clip(sum);
                    }
                }
            });
        }
        if (image.Height == height)
        {
            return new RgbImage(width, height, wide);
        }
        var (rows, vertical, count) = Coefficients(image.Height, height);
        var pixels = new byte[checked(3 * width * height)];
        Parallel.For(0, height, y =>
        {
            for (int i = 0; i < 3 * width; i++)
            {
                int sum = 1 << (Precision - 1);
                for (int j = 0; j < rows[y].Count; j++)
                {
                    sum += wide[3 * (rows[y].First + j) * width + i] * vertical[y * count + j];
                }
                pixels[3 * y * width + i] = Clip(sum);
            }
        });
        return new RgbImage(width, height, pixels);
    }

    private static byte Clip(int v) => (byte)Math.Clamp(v >> Precision, 0, 255);

    private static double Cubic(double x)
    {
        const double a = -0.5;
        x = Math.Abs(x);
        return x < 1 ? ((a + 2) * x - (a + 3)) * x * x + 1 : x < 2 ? (((x - 5) * x + 8) * x - 4) * a : 0;
    }

    private static ((int First, int Count)[] Starts, int[] Weights, int Size) Coefficients(int from, int to)
    {
        double scale = (double)from / to, filter = Math.Max(1, scale), support = 2 * filter;
        int size = (int)Math.Ceiling(support) * 2 + 1;
        var starts = new (int, int)[to];
        var weights = new int[checked(to * size)];
        var k = new double[size];
        for (int x = 0; x < to; x++)
        {
            double centre = (x + 0.5) * scale, total = 0;
            int first = Math.Max(0, (int)(centre - support + 0.5)), last = Math.Min(from, (int)(centre + support + 0.5));
            for (int i = 0; i < last - first; i++)
            {
                k[i] = Cubic((i + first - centre + 0.5) / filter);
                total += k[i];
            }
            for (int i = 0; i < last - first; i++)
            {
                double w = total != 0 ? k[i] / total : k[i];
                weights[x * size + i] = (int)(w * (1 << Precision) + (w < 0 ? -0.5 : 0.5));
            }
            starts[x] = (first, last - first);
        }
        return (starts, weights, size);
    }
}
