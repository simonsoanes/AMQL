using System.Text;

namespace Amql.Hf.Tokenizers;

/// <summary>
/// The GPT-2 byte-level mapping used by ByteLevel pre-tokenizers and
/// decoders. Every byte of the UTF-8 encoding maps to a printable-ish
/// character (the "Ġ" space marker is the canonical example); BPE runs
/// over the mapped characters and the decoder maps them back to bytes.
/// The mapping is the reference's exact <c>bytes_to_unicode</c> table.
/// </summary>
public static class ByteLevel
{
    private static readonly char[] ByteToChar = BuildByteToChar();
    private static readonly Dictionary<char, byte> CharToByte = BuildCharToByte();

    /// <summary>Maps a raw word to its byte-level character form:
    /// <c>str.encode("utf-8")</c> then one character per byte.</summary>
    public static string EncodeWord(string word)
    {
        var bytes = Encoding.UTF8.GetBytes(word);
        var chars = new char[bytes.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            chars[i] = ByteToChar[bytes[i]];
        }
        return new string(chars);
    }

    /// <summary>Maps a byte-level character form back to the original word.</summary>
    public static string DecodeWord(string byteLevel)
    {
        var bytes = new byte[byteLevel.Length];
        for (int i = 0; i < byteLevel.Length; i++)
        {
            if (!CharToByte.TryGetValue(byteLevel[i], out var b))
            {
                throw new TokenizerException($"byte-level token '{byteLevel}' contains unmappable character U+{(int)byteLevel[i]:X4}");
            }
            bytes[i] = b;
        }
        return Encoding.UTF8.GetString(bytes);
    }

    public static bool TryDecodeWord(string byteLevel, out string word)
    {
        var bytes = new byte[byteLevel.Length];
        for (int i = 0; i < byteLevel.Length; i++)
        {
            if (!CharToByte.TryGetValue(byteLevel[i], out bytes[i]))
            {
                word = string.Empty;
                return false;
            }
        }
        try
        {
            word = Encoding.UTF8.GetString(bytes);
            return true;
        }
        catch (DecoderFallbackException)
        {
            word = string.Empty;
            return false;
        }
    }

    private static char[] BuildByteToChar()
    {
        var bytes = new List<byte>();
        var chars = new List<char>();
        for (char c = '!'; c <= '~'; c++)
        {
            bytes.Add((byte)c);
            chars.Add(c);
        }
        for (char c = '\u00A1'; c <= '\u00AC'; c++)
        {
            bytes.Add((byte)c);
            chars.Add(c);
        }
        for (char c = '\u00AE'; c <= '\u00FF'; c++)
        {
            bytes.Add((byte)c);
            chars.Add(c);
        }

        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bytes.Contains((byte)b))
            {
                bytes.Add((byte)b);
                chars.Add((char)(256 + n));
                n++;
            }
        }

        var table = new char[256];
        for (int i = 0; i < bytes.Count; i++)
        {
            table[bytes[i]] = chars[i];
        }
        return table;
    }

    private static Dictionary<char, byte> BuildCharToByte()
    {
        var map = new Dictionary<char, byte>();
        for (int b = 0; b < 256; b++)
        {
            map[ByteToChar[b]] = (byte)b;
        }
        return map;
    }
}