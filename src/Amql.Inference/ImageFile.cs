using System.Buffers.Binary;
using System.IO.Compression;

namespace Amql.Inference;

/// <summary>An image that cannot be read.</summary>
public sealed class ImageException(string message) : Exception(message);

/// <summary>An image's pixels: 8-bit RGB, row by row.</summary>
public sealed record RgbImage(int Width, int Height, byte[] Pixels);

/// <summary>
/// Reads PNG and JPEG files (baseline and progressive, with their EXIF orientation applied)
/// into RGB. Transparency is laid over white. What a file claims is checked before memory is
/// taken for it, since files come from clients.
/// </summary>
public static class ImageFile
{
    /// <summary>The most pixels an image may have.</summary>
    public const long MaxPixels = 100_000_000;

    public static RgbImage Read(byte[] file)
    {
        try
        {
            if (file.Length >= 8 && file.AsSpan(0, 8).SequenceEqual((ReadOnlySpan<byte>)[0x89, (byte)'P', (byte)'N', (byte)'G', 13, 10, 26, 10]))
            {
                return Png(file);
            }
            if (file.Length >= 3 && file[0] == 0xFF && file[1] == 0xD8 && file[2] == 0xFF)
            {
                return new Jpeg(file).Read();
            }
        }
        catch (Exception e) when (e is IndexOutOfRangeException or ArgumentException or InvalidDataException or OverflowException or EndOfStreamException)
        {
            throw new ImageException("the image file is damaged or cut short");
        }
        string kind = file.Length >= 12 && file.AsSpan(8, 4).SequenceEqual("WEBP"u8) ? "WebP" : file.AsSpan(0, Math.Min(4, file.Length)).StartsWith("GIF8"u8) ? "GIF" : "this kind of file";
        throw new ImageException($"images are read as PNG or JPEG, not {kind}");
    }

    private static void CheckSize(long width, long height)
    {
        if (width < 1 || height < 1 || width * height > MaxPixels)
        {
            throw new ImageException($"an image of {width} x {height} pixels is not read (at most {MaxPixels:N0})");
        }
    }

    // ── PNG ───────────────────────────────────────────────────────────────

    private static RgbImage Png(byte[] f)
    {
        int width = 0, height = 0, depth = 0, color = 0, interlace = 0;
        byte[]? palette = null, transparent = null;
        using var data = new MemoryStream();
        for (int at = 8; at + 8 <= f.Length;)
        {
            int length = checked((int)BinaryPrimitives.ReadUInt32BigEndian(f.AsSpan(at)));
            var type = f.AsSpan(at + 4, 4);
            var body = f.AsSpan(at + 8, length);
            if (type.SequenceEqual("IHDR"u8))
            {
                width = checked((int)BinaryPrimitives.ReadUInt32BigEndian(body));
                height = checked((int)BinaryPrimitives.ReadUInt32BigEndian(body[4..]));
                (depth, color, interlace) = (body[8], body[9], body[12]);
                CheckSize(width, height);
            }
            else if (type.SequenceEqual("PLTE"u8))
            {
                palette = body.ToArray();
            }
            else if (type.SequenceEqual("tRNS"u8))
            {
                transparent = body.ToArray();
            }
            else if (type.SequenceEqual("IDAT"u8))
            {
                data.Write(body);
            }
            else if (type.SequenceEqual("IEND"u8))
            {
                break;
            }
            at += 12 + length;
        }
        int channels = color switch { 0 => 1, 2 => 3, 3 => 1, 4 => 2, 6 => 4, _ => 0 };
        if (width == 0 || channels == 0 || depth is not (1 or 2 or 4 or 8 or 16) || (depth < 8 && channels > 1) || (color == 3 && (palette is null || depth == 16)))
        {
            throw new ImageException("the PNG's header is not one this reads");
        }
        data.Position = 0;
        using var zlib = new ZLibStream(data, CompressionMode.Decompress);
        var pixels = new byte[checked(3 * width * height)];
        int bits = channels * depth, step = Math.Max(1, bits / 8);
        ReadOnlySpan<int> x0 = [0, 4, 0, 2, 0, 1, 0], y0 = [0, 0, 4, 0, 2, 0, 1], dx = [8, 8, 4, 4, 2, 2, 1], dy = [8, 8, 8, 4, 4, 2, 2];
        for (int pass = 0; pass < (interlace == 1 ? 7 : 1); pass++)
        {
            int px = interlace == 1 ? x0[pass] : 0, py = interlace == 1 ? y0[pass] : 0, sx = interlace == 1 ? dx[pass] : 1, sy = interlace == 1 ? dy[pass] : 1;
            int w = (width - px + sx - 1) / sx, h = (height - py + sy - 1) / sy;
            if (w <= 0 || h <= 0)
            {
                continue;
            }
            int stride = checked((w * bits + 7) / 8);
            byte[] row = new byte[stride], above = new byte[stride];
            for (int y = 0; y < h; y++)
            {
                int filter = zlib.ReadByte();
                zlib.ReadExactly(row);
                for (int i = 0; i < stride; i++)
                {
                    int a = i >= step ? row[i - step] : 0, b = above[i], c = i >= step ? above[i - step] : 0;
                    row[i] += filter switch
                    {
                        0 => (byte)0,
                        1 => (byte)a,
                        2 => (byte)b,
                        3 => (byte)((a + b) / 2),
                        4 => (byte)Paeth(a, b, c),
                        _ => throw new InvalidDataException(),
                    };
                }
                for (int x = 0; x < w; x++)
                {
                    int Sample(int channel) => depth switch
                    {
                        8 => row[x * channels + channel],
                        16 => row[2 * (x * channels + channel)],
                        _ => row[x * depth / 8] >> (8 - depth - x * depth % 8) & (1 << depth) - 1,
                    };
                    int r, g, bl, alpha = 255;
                    if (color == 3)
                    {
                        int i = Sample(0);
                        (r, g, bl) = (palette![3 * i], palette[3 * i + 1], palette[3 * i + 2]);
                        alpha = transparent is not null && i < transparent.Length ? transparent[i] : 255;
                    }
                    else if (channels < 3)
                    {
                        int v = Sample(0);
                        if (color == 0 && transparent is { Length: >= 2 } t && (depth == 16 ? t[0] : t[1]) == v)
                        {
                            alpha = 0;
                        }
                        r = g = bl = depth < 8 ? v * 255 / ((1 << depth) - 1) : v;
                        alpha = channels == 2 ? Sample(1) : alpha;
                    }
                    else
                    {
                        (r, g, bl) = (Sample(0), Sample(1), Sample(2));
                        if (color == 2 && transparent is { Length: >= 6 } t && depth == 8 && t[1] == r && t[3] == g && t[5] == bl)
                        {
                            alpha = 0;
                        }
                        alpha = channels == 4 ? Sample(3) : alpha;
                    }
                    int at = 3 * ((py + y * sy) * width + px + x * sx);
                    pixels[at] = Over(r, alpha);
                    pixels[at + 1] = Over(g, alpha);
                    pixels[at + 2] = Over(bl, alpha);
                }
                (row, above) = (above, row);
            }
        }
        return new RgbImage(width, height, pixels);
    }

    private static byte Over(int v, int alpha) => alpha == 255 ? (byte)v : (byte)((v * alpha + 255 * (255 - alpha) + 127) / 255);

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c, pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }

    // ── JPEG ──────────────────────────────────────────────────────────────

    private sealed class Jpeg(byte[] f)
    {
        private sealed class Component
        {
            public int Id, H, V, Table, Dc, Ac, BlocksWide, BlocksHigh, Wide, High, Predicted;
            public short[] Coefficients = [];   // [BlocksHigh x BlocksWide x 64], in natural order
            public byte[] Samples = [];
        }

        private sealed class Huffman
        {
            // By length: the largest code, and what to add to a code to find its symbol.
            public readonly int[] MaxCode = new int[18], Offset = new int[17];
            public byte[] Symbols = [];
        }

        private static readonly byte[] ZigZag =
        [
            0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
            35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
        ];

        private readonly int[][] _quantum = new int[4][];
        private readonly Huffman?[] _dc = new Huffman?[4], _ac = new Huffman?[4];
        private Component[] _components = [];
        private int _width, _height, _restart, _orientation = 1, _adobe = -1;
        private bool _progressive;

        // The scan's bits.
        private int _at, _bits, _count, _eobRun;

        public RgbImage Read()
        {
            int at = 2;
            while (at + 4 <= f.Length)
            {
                if (f[at] != 0xFF)
                {
                    at++;
                    continue;
                }
                int marker = f[at + 1];
                if (marker is 0xFF or 0x00 or (>= 0xD0 and <= 0xD8) or 0x01)
                {
                    at += marker == 0xFF ? 1 : 2;
                    continue;
                }
                if (marker == 0xD9)
                {
                    break;
                }
                int length = f[at + 2] << 8 | f[at + 3];
                var body = f.AsSpan(at + 4, length - 2);
                at += 2 + length;
                switch (marker)
                {
                    case 0xC0 or 0xC1 or 0xC2:
                        Frame(body, marker == 0xC2);
                        break;
                    case 0xC4:
                        Tables(body);
                        break;
                    case 0xDB:
                        Quantisers(body);
                        break;
                    case 0xDD:
                        _restart = body[0] << 8 | body[1];
                        break;
                    case 0xDA:
                        Scan(body, at);
                        at = _at;
                        break;
                    case 0xE1:
                        _orientation = Orientation(body);
                        break;
                    case 0xEE when body.Length >= 12 && body[..5].SequenceEqual("Adobe"u8):
                        _adobe = body[11];
                        break;
                    case 0xC3 or (>= 0xC5 and <= 0xCF and not 0xC8 and not 0xCC):
                        throw new ImageException("the JPEG is of a kind this does not read (lossless, hierarchical or arithmetic-coded)");
                }
            }
            if (_components.Length == 0)
            {
                throw new InvalidDataException();
            }
            return Oriented(Colours());
        }

        private void Frame(ReadOnlySpan<byte> b, bool progressive)
        {
            if (_components.Length > 0)
            {
                throw new InvalidDataException();
            }
            if (b[0] != 8)
            {
                throw new ImageException("the JPEG has more than 8 bits a sample");
            }
            _progressive = progressive;
            _height = b[1] << 8 | b[2];
            _width = b[3] << 8 | b[4];
            CheckSize(_width, _height);
            int n = b[5];
            if (n is not (1 or 3 or 4))
            {
                throw new InvalidDataException();
            }
            var components = new Component[n];
            for (int i = 0; i < n; i++)
            {
                components[i] = new Component { Id = b[6 + 3 * i], H = b[7 + 3 * i] >> 4, V = b[7 + 3 * i] & 15, Table = b[8 + 3 * i] & 3 };
                if (components[i].H is < 1 or > 4 || components[i].V is < 1 or > 4)
                {
                    throw new InvalidDataException();
                }
            }
            int hMax = components.Max(c => c.H), vMax = components.Max(c => c.V);
            int across = (_width + 8 * hMax - 1) / (8 * hMax), down = (_height + 8 * vMax - 1) / (8 * vMax);
            foreach (var c in components)
            {
                c.BlocksWide = across * c.H;
                c.BlocksHigh = down * c.V;
                c.Wide = (_width * c.H + hMax - 1) / hMax;
                c.High = (_height * c.V + vMax - 1) / vMax;
                c.Coefficients = new short[checked(64 * c.BlocksWide * c.BlocksHigh)];
            }
            _components = components;
        }

        private void Quantisers(ReadOnlySpan<byte> b)
        {
            for (int at = 0; at < b.Length;)
            {
                int wide = b[at] >> 4, id = b[at] & 3;
                at++;
                var q = new int[64];
                for (int i = 0; i < 64; i++)
                {
                    q[ZigZag[i]] = wide == 0 ? b[at++] : b[at++] << 8 | b[at++];
                }
                _quantum[id] = q;
            }
        }

        private void Tables(ReadOnlySpan<byte> b)
        {
            for (int at = 0; at < b.Length;)
            {
                int kind = b[at] >> 4, id = b[at] & 3;
                var counts = b.Slice(at + 1, 16);
                int total = 0;
                foreach (byte n in counts)
                {
                    total += n;
                }
                var table = new Huffman { Symbols = b.Slice(at + 17, total).ToArray() };
                for (int length = 1, code = 0, first = 0; length <= 16; length++)
                {
                    int n = counts[length - 1];
                    table.Offset[length] = first - code;
                    table.MaxCode[length] = n > 0 ? code + n - 1 : -1;
                    code = code + n << 1;
                    first += n;
                }
                (kind == 0 ? _dc : _ac)[id] = table;
                at += 17 + total;
            }
        }

        private static int Orientation(ReadOnlySpan<byte> b)
        {
            if (b.Length < 14 || !b[..6].SequenceEqual("Exif\0\0"u8))
            {
                return 1;
            }
            var t = b[6..];
            bool little = t[0] == 'I';
            int U16(ReadOnlySpan<byte> s) => little ? BinaryPrimitives.ReadUInt16LittleEndian(s) : BinaryPrimitives.ReadUInt16BigEndian(s);
            uint offset = little ? BinaryPrimitives.ReadUInt32LittleEndian(t[4..]) : BinaryPrimitives.ReadUInt32BigEndian(t[4..]);
            if (offset + 2 > t.Length)
            {
                return 1;
            }
            int entries = U16(t[(int)offset..]);
            for (int i = 0; i < entries && offset + 2 + 12 * (i + 1) <= t.Length; i++)
            {
                var entry = t.Slice((int)offset + 2 + 12 * i, 12);
                if (U16(entry) == 0x0112)
                {
                    int o = U16(entry[8..]);
                    return o is >= 1 and <= 8 ? o : 1;
                }
            }
            return 1;
        }

        // ── the scan ──

        private int Bit()
        {
            if (_count == 0)
            {
                int b = _at < f.Length ? f[_at] : 0;
                if (b == 0xFF)
                {
                    if (_at + 1 >= f.Length || f[_at + 1] != 0)
                    {
                        return 0;   // a marker: the scan is over, and what is missing reads as zeros
                    }
                    _at++;
                }
                _at++;
                _bits = b;
                _count = 8;
            }
            _count--;
            return _bits >> _count & 1;
        }

        private int Bits(int n)
        {
            int v = 0;
            while (n-- > 0)
            {
                v = v << 1 | Bit();
            }
            return v;
        }

        private int Extended(int n)
        {
            if (n == 0)
            {
                return 0;
            }
            int v = Bits(n);
            return v < 1 << (n - 1) ? v - (1 << n) + 1 : v;
        }

        private int Symbol(Huffman? table)
        {
            if (table is null)
            {
                throw new InvalidDataException();
            }
            int code = 0;
            for (int length = 1; length <= 16; length++)
            {
                code = code << 1 | Bit();
                if (table.MaxCode[length] >= 0 && code <= table.MaxCode[length] && code + table.Offset[length] is var i and >= 0 && i < table.Symbols.Length)
                {
                    return table.Symbols[i];
                }
            }
            return 0;   // no such code: the data is damaged, and the rest of it reads as best it can
        }

        private void Scan(ReadOnlySpan<byte> b, int data)
        {
            if (_components.Length == 0)
            {
                throw new InvalidDataException();
            }
            int n = b[0];
            var scan = new Component[n];
            for (int i = 0; i < n; i++)
            {
                int id = b[1 + 2 * i];
                scan[i] = _components.FirstOrDefault(c => c.Id == id) ?? throw new InvalidDataException();
                scan[i].Dc = b[2 + 2 * i] >> 4 & 3;
                scan[i].Ac = b[2 + 2 * i] & 3;
            }
            int first = b[1 + 2 * n], last = b[2 + 2 * n], high = b[3 + 2 * n] >> 4, low = b[3 + 2 * n] & 15;
            if (!_progressive)
            {
                (first, last, high, low) = (0, 63, 0, 0);
            }
            if (first > last || last > 63)
            {
                throw new InvalidDataException();
            }
            _at = data;
            StartInterval(scan);
            int unitsWide, unitsHigh;
            bool single = n == 1;
            if (single)
            {
                // A component alone is coded block by block, without the padding of whole units.
                unitsWide = (scan[0].Wide + 7) / 8;
                unitsHigh = (scan[0].High + 7) / 8;
            }
            else
            {
                unitsWide = scan[0].BlocksWide / scan[0].H;
                unitsHigh = scan[0].BlocksHigh / scan[0].V;
            }
            for (int unit = 0, total = unitsWide * unitsHigh; unit < total; unit++)
            {
                if (_restart > 0 && unit > 0 && unit % _restart == 0)
                {
                    SkipToMarker();
                    if (_at + 1 < f.Length && f[_at] == 0xFF && f[_at + 1] is >= 0xD0 and <= 0xD7)
                    {
                        _at += 2;
                    }
                    StartInterval(scan);
                }
                int ux = unit % unitsWide, uy = unit / unitsWide;
                foreach (var c in scan)
                {
                    for (int v = 0; v < (single ? 1 : c.V); v++)
                    {
                        for (int h = 0; h < (single ? 1 : c.H); h++)
                        {
                            int bx = single ? ux : ux * c.H + h, by = single ? uy : uy * c.V + v;
                            var block = c.Coefficients.AsSpan(64 * (by * c.BlocksWide + bx), 64);
                            if (!_progressive)
                            {
                                Sequential(c, block);
                            }
                            else if (first == 0)
                            {
                                if (high == 0)
                                {
                                    c.Predicted += Extended(Symbol(_dc[c.Dc]) & 15);
                                    block[0] = (short)(c.Predicted << low);
                                }
                                else if (Bit() == 1)
                                {
                                    block[0] |= (short)(1 << low);
                                }
                            }
                            else if (high == 0)
                            {
                                FirstAc(c, block, first, last, low);
                            }
                            else
                            {
                                RefineAc(c, block, first, last, 1 << low);
                            }
                        }
                    }
                }
            }
            _count = 0;
            SkipToMarker();
        }

        private void StartInterval(Component[] scan)
        {
            _count = 0;
            _eobRun = 0;
            foreach (var c in scan)
            {
                c.Predicted = 0;
            }
        }

        private void SkipToMarker()
        {
            while (_at + 1 < f.Length && !(f[_at] == 0xFF && f[_at + 1] is not (0 or 0xFF)))
            {
                _at++;
            }
        }

        private void Sequential(Component c, Span<short> block)
        {
            c.Predicted += Extended(Symbol(_dc[c.Dc]) & 15);
            block[0] = (short)c.Predicted;
            for (int k = 1; k < 64; k++)
            {
                int s = Symbol(_ac[c.Ac]), run = s >> 4, size = s & 15;
                if (size == 0)
                {
                    if (run < 15)
                    {
                        break;
                    }
                    k += 15;
                    continue;
                }
                k += run;
                if (k > 63)
                {
                    break;
                }
                block[ZigZag[k]] = (short)Extended(size);
            }
        }

        private void FirstAc(Component c, Span<short> block, int first, int last, int low)
        {
            if (_eobRun > 0)
            {
                _eobRun--;
                return;
            }
            for (int k = first; k <= last; k++)
            {
                int s = Symbol(_ac[c.Ac]), run = s >> 4, size = s & 15;
                if (size == 0)
                {
                    if (run < 15)
                    {
                        _eobRun = (1 << run) - 1 + Bits(run);
                        break;
                    }
                    k += 15;
                    continue;
                }
                k += run;
                if (k > 63)
                {
                    break;
                }
                block[ZigZag[k]] = (short)(Extended(size) << low);
            }
        }

        private void RefineAc(Component c, Span<short> block, int first, int last, int bit)
        {
            int k = first;
            if (_eobRun == 0)
            {
                for (; k <= last; k++)
                {
                    int s = Symbol(_ac[c.Ac]), run = s >> 4, value = 0;
                    if ((s & 15) == 0)
                    {
                        if (run < 15)
                        {
                            _eobRun = (1 << run) + Bits(run);
                            break;
                        }
                    }
                    else
                    {
                        value = Bit() == 1 ? bit : -bit;
                    }
                    // Past `run` zeros, correcting each coefficient that is not.
                    for (; k <= last; k++)
                    {
                        ref short v = ref block[ZigZag[k]];
                        if (v != 0)
                        {
                            Correct(ref v, bit);
                        }
                        else if (run-- == 0)
                        {
                            v = (short)value;
                            break;
                        }
                    }
                }
            }
            if (_eobRun > 0)
            {
                for (; k <= last; k++)
                {
                    ref short v = ref block[ZigZag[k]];
                    if (v != 0)
                    {
                        Correct(ref v, bit);
                    }
                }
                _eobRun--;
            }
        }

        private void Correct(ref short v, int bit)
        {
            if (Bit() == 1 && (v & bit) == 0)
            {
                v += (short)(v >= 0 ? bit : -bit);
            }
        }

        // ── samples to pixels ──

        private RgbImage Colours()
        {
            foreach (var c in _components)
            {
                Transform(c);
            }
            int hMax = _components.Max(c => c.H), vMax = _components.Max(c => c.V);
            var planes = _components.Select(c => Upsampled(c, hMax, vMax)).ToArray();
            var pixels = new byte[checked(3 * _width * _height)];
            bool four = planes.Length == 4, ycc = planes.Length == 3 ? _adobe != 0 && !(_components[0].Id == 'R' && _components[1].Id == 'G') : four && _adobe == 2;
            Parallel.For(0, _height, y =>
            {
                for (int i = y * _width, end = i + _width; i < end; i++)
                {
                    int r = planes[0][i], g = r, b = r;
                    if (planes.Length > 1)
                    {
                        g = planes[1][i];
                        b = planes[2][i];
                        if (ycc)
                        {
                            int cb = g - 128, cr = b - 128, half = 1 << 15;
                            g = Limit(r + ((-22554 * cb + half - 46802 * cr) >> 16));
                            b = Limit(r + ((116130 * cb + half) >> 16));
                            r = Limit(r + ((91881 * cr + half) >> 16));
                        }
                        if (four)
                        {
                            // Adobe stores CMYK inverted, and YCCK as the YCC of its complement.
                            int k = planes[3][i];
                            if (ycc)
                            {
                                (r, g, b) = (255 - r, 255 - g, 255 - b);
                            }
                            if (_adobe < 0)
                            {
                                (r, g, b, k) = (255 - r, 255 - g, 255 - b, 255 - k);
                            }
                            (r, g, b) = ((r * k + 127) / 255, (g * k + 127) / 255, (b * k + 127) / 255);
                        }
                    }
                    pixels[3 * i] = (byte)r;
                    pixels[3 * i + 1] = (byte)g;
                    pixels[3 * i + 2] = (byte)b;
                }
            });
            return new RgbImage(_width, _height, pixels);
        }

        private static int Limit(int v) => v < 0 ? 0 : v > 255 ? 255 : v;

        /// <summary>The component's blocks dequantised and transformed into its samples, in the integer
        /// arithmetic libjpeg uses, so that a file gives the pixels it gives everywhere.</summary>
        private void Transform(Component c)
        {
            var q = _quantum[c.Table] ?? throw new InvalidDataException();
            int stride = 8 * c.BlocksWide;
            var samples = new byte[checked(stride * 8 * c.BlocksHigh)];
            Parallel.For(0, c.BlocksHigh, by =>
            {
                Span<int> work = stackalloc int[64];
                for (int bx = 0; bx < c.BlocksWide; bx++)
                {
                    var block = c.Coefficients.AsSpan(64 * (by * c.BlocksWide + bx), 64);
                    for (int i = 0; i < 64; i++)
                    {
                        work[i] = block[i] * q[i];
                    }
                    for (int x = 0; x < 8; x++)
                    {
                        Idct(work, x, 8, 11);
                    }
                    for (int y = 0; y < 8; y++)
                    {
                        Idct(work, 8 * y, 1, 18);
                        for (int x = 0; x < 8; x++)
                        {
                            samples[(8 * by + y) * stride + 8 * bx + x] = (byte)Limit(work[8 * y + x] + 128);
                        }
                    }
                }
            });
            c.Samples = samples;
            c.Coefficients = [];
        }

        /// <summary>One 8-point inverse transform in place, at 13 bits of fraction, scaled down by <paramref name="shift"/> bits.</summary>
        private static void Idct(Span<int> w, int at, int step, int shift)
        {
            unchecked
            {
                int c0 = w[at], c1 = w[at + step], c2 = w[at + 2 * step], c3 = w[at + 3 * step];
                int c4 = w[at + 4 * step], c5 = w[at + 5 * step], c6 = w[at + 6 * step], c7 = w[at + 7 * step];
                int z1 = (c2 + c6) * 4433;
                int t2 = z1 - c6 * 15137, t3 = z1 + c2 * 6270;
                int t0 = (c0 + c4) << 13, t1 = (c0 - c4) << 13;
                int e0 = t0 + t3, e3 = t0 - t3, e1 = t1 + t2, e2 = t1 - t2;
                int a = c7 + c1, b = c5 + c3, d = c7 + c3, e = c5 + c1;
                int z5 = (d + e) * 9633;
                int o0 = c7 * 2446, o1 = c5 * 16819, o2 = c3 * 25172, o3 = c1 * 12299;
                a *= -7373;
                b *= -20995;
                d = d * -16069 + z5;
                e = e * -3196 + z5;
                o0 += a + d;
                o1 += b + e;
                o2 += b + d;
                o3 += a + e;
                int half = 1 << (shift - 1);
                w[at] = (e0 + o3 + half) >> shift;
                w[at + 7 * step] = (e0 - o3 + half) >> shift;
                w[at + step] = (e1 + o2 + half) >> shift;
                w[at + 6 * step] = (e1 - o2 + half) >> shift;
                w[at + 2 * step] = (e2 + o1 + half) >> shift;
                w[at + 5 * step] = (e2 - o1 + half) >> shift;
                w[at + 3 * step] = (e3 + o0 + half) >> shift;
                w[at + 4 * step] = (e3 - o0 + half) >> shift;
            }
        }

        /// <summary>The component at the image's size: samples as they are, halves interpolated as
        /// libjpeg does (three parts the nearer sample to one the farther), anything else repeated.</summary>
        private byte[] Upsampled(Component c, int hMax, int vMax)
        {
            int stride = 8 * c.BlocksWide;
            var plane = new byte[checked(_width * _height)];
            bool across = hMax == 2 * c.H && c.Wide > 2, down = vMax == 2 * c.V;
            bool smoothAcross = across && (vMax == c.V || down), smoothDown = down && (hMax == c.H || across);
            Parallel.For(0, _height, y =>
            {
                int sy = y * c.V / vMax;
                var row = plane.AsSpan(y * _width, _width);
                if (!smoothAcross && !smoothDown)
                {
                    for (int x = 0; x < _width; x++)
                    {
                        row[x] = c.Samples[sy * stride + x * c.H / hMax];
                    }
                    return;
                }
                int near = sy * stride, far = smoothDown ? Math.Clamp((y & 1) == 0 ? sy - 1 : sy + 1, 0, c.High - 1) * stride : near;
                if (!smoothAcross)
                {
                    for (int x = 0; x < _width; x++)
                    {
                        row[x] = (byte)((3 * c.Samples[near + x] + c.Samples[far + x] + ((y & 1) == 0 ? 1 : 2)) >> 2);
                    }
                    return;
                }
                for (int x = 0; x < _width; x++)
                {
                    int sx = x >> 1, other = (x & 1) == 0 ? sx - 1 : sx + 1;
                    bool edge = other < 0 || other >= c.Wide;
                    if (!smoothDown)
                    {
                        row[x] = edge ? c.Samples[near + sx] : (byte)((3 * c.Samples[near + sx] + c.Samples[near + other] + ((x & 1) == 0 ? 1 : 2)) >> 2);
                        continue;
                    }
                    int here = 3 * c.Samples[near + sx] + c.Samples[far + sx];
                    int beside = edge ? here : 3 * c.Samples[near + other] + c.Samples[far + other];
                    row[x] = (byte)((3 * here + beside + ((x & 1) == 0 ? 8 : 7)) >> 4);
                }
            });
            return plane;
        }

        /// <summary>The picture turned and mirrored as its EXIF orientation says.</summary>
        private RgbImage Oriented(RgbImage image)
        {
            if (_orientation == 1)
            {
                return image;
            }
            int w = image.Width, h = image.Height;
            bool swapped = _orientation >= 5;
            int ow = swapped ? h : w, oh = swapped ? w : h;
            var pixels = new byte[image.Pixels.Length];
            for (int y = 0; y < oh; y++)
            {
                for (int x = 0; x < ow; x++)
                {
                    (int sx, int sy) = _orientation switch
                    {
                        2 => (w - 1 - x, y),
                        3 => (w - 1 - x, h - 1 - y),
                        4 => (x, h - 1 - y),
                        5 => (y, x),
                        6 => (y, h - 1 - x),
                        7 => (w - 1 - y, h - 1 - x),
                        _ => (w - 1 - y, x),
                    };
                    image.Pixels.AsSpan(3 * (sy * w + sx), 3).CopyTo(pixels.AsSpan(3 * (y * ow + x)));
                }
            }
            return new RgbImage(ow, oh, pixels);
        }
    }
}
