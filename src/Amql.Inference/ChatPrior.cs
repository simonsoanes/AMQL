namespace Amql.Inference;

/// <summary>
/// The blank of a chat prompt: the same conversation, system turn, tools and history, with
/// the latest message's content replaced by "N/A". What a model predicts for it is the
/// conversation's prior, so comparing a prediction with it shows what the latest message
/// itself contributed. Contextual calibration (Zhao et al., 2021), applied per prompt.
/// </summary>
/// <remarks>
/// It reads ChatML: the latest <c>&lt;|im_start|&gt;user</c> turn up to its
/// <c>&lt;|im_end|&gt;</c>. A turn that holds a tool's result keeps its
/// <c>&lt;tool_response&gt;</c> wrapper, so the blank is still a tool's turn.
/// </remarks>
public static class ChatPrior
{
    public const string Filler = "N/A";

    /// <summary>
    /// Returns a function from a prompt's token ids to its blank's, or null when the prompt
    /// has no user turn. <paramref name="encode"/> and <paramref name="decode"/> are the
    /// tokenizer's, with special tokens kept.
    /// </summary>
    public static Func<int[], int[]?> ForChatMl(Func<string, IReadOnlyList<int>> encode,
        Func<IReadOnlyList<int>, string> decode, int imStart, int imEnd)
    {
        return tokens =>
        {
            for (int start = tokens.Length - 1; start >= 0; start--)
            {
                if (tokens[start] != imStart)
                {
                    continue;
                }
                int end = Array.IndexOf(tokens, imEnd, start + 1);
                if (end < 0)
                {
                    continue;
                }
                string turn = decode(tokens[(start + 1)..end]);
                if (!turn.StartsWith("user\n", StringComparison.Ordinal))
                {
                    continue;
                }
                string content = turn["user\n".Length..];
                string blank = content.TrimStart().StartsWith("<tool_response>", StringComparison.Ordinal)
                    ? $"<tool_response>\n{Filler}\n</tool_response>"
                    : Filler;
                return tokens[..(start + 1)].Concat(encode("user\n" + blank)).Concat(tokens[end..]).ToArray();
            }
            return null;
        };
    }
}
