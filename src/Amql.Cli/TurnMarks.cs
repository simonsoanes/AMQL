using Amql.Hf;

namespace Amql.Cli;

/// <summary>The tokens a chat template marks its turns with, found in what it writes.</summary>
public static class TurnMarks
{
    /// <summary>The first token of its own (an added token) in a piece of template text, or null.</summary>
    public static int? First(HfTokenizer tokenizer, string text) => All(tokenizer, text) is [var first, ..] ? first : null;

    /// <summary>
    /// The token a reply ends on: the first of its own the template writes after a reply,
    /// passing over any it also writes to close reasoning. Where every message ends with the
    /// same token (<c>&lt;|end_message|&gt;</c>, then <c>&lt;|content_model_end_sampling|&gt;</c>
    /// after the last), that token closes the thinking too and must not end the reply.
    /// </summary>
    public static int? ReplyEnd(HfTokenizer tokenizer, ChatDialect dialect)
    {
        var closes = dialect.ThinkCloseMark.Length > 0 ? All(tokenizer, dialect.ThinkCloseMark) : [];
        var ends = All(tokenizer, dialect.AfterReply);
        foreach (int id in ends)
        {
            if (!closes.Contains(id))
            {
                return id;
            }
        }
        return ends.Count > 0 ? ends[0] : null;
    }

    /// <summary>The tokens that open a user's turn, name the user in it, open the assistant's turn (when that
    /// is another token), and open a call's result (when it is in the assistant's turn); each null when there is none.</summary>
    public static (int? Turn, int? User, int? Reply, int? Result) Openers(HfTokenizer tokenizer, ChatDialect dialect)
    {
        var ends = All(tokenizer, dialect.AfterReply);
        if (Past(All(tokenizer, dialect.BetweenTurns)) is not [var turn, ..])
        {
            return default;
        }
        var ids = tokenizer.EncodeToIds(dialect.BetweenTurns);
        int at = ids.ToList().IndexOf(turn);
        int? user = at + 1 < ids.Count ? ids[at + 1] : null;
        var reply = All(tokenizer, dialect.BeforeReply);
        var result = Past(All(tokenizer, dialect.BeforeResult));
        return (turn, user, reply.Count == 0 || reply.Contains(turn) ? null : reply[^1],
            result.Count == 0 || result.Contains(turn) ? null : result[0]);

        // What comes after the end of the turn before.
        IReadOnlyList<int> Past(IReadOnlyList<int> marks)
        {
            int same = 0;
            while (same < marks.Count && same < ends.Count && marks[same] == ends[same])
            {
                same++;
            }
            return marks.Skip(same).ToList();
        }
    }

    /// <summary>The tokens of their own in a piece of template text, in order.</summary>
    public static IReadOnlyList<int> All(HfTokenizer tokenizer, string text)
    {
        var ids = new List<int>();
        if (text.Length == 0)
        {
            return ids;
        }
        foreach (int id in tokenizer.EncodeToIds(text))
        {
            if (tokenizer.TokenInfo(id).Representation is { } content && tokenizer.AddedTokenId(content) == id)
            {
                ids.Add(id);
            }
        }
        return ids;
    }
}
