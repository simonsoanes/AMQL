using System.Text.RegularExpressions;

namespace Amql.Inference.Reasoning;

/// <summary>How a reply to a problem ended.</summary>
public enum ReplyOutcome
{
    /// <summary>It committed to an answer, and the answer is right.</summary>
    Correct,
    /// <summary>It committed to an answer, and the answer is wrong.</summary>
    Wrong,
    /// <summary>It finished, but its answer holds no number.</summary>
    NoNumber,
    /// <summary>It was still thinking when the budget ran out, repeating itself.</summary>
    Looped,
    /// <summary>It was still thinking when the budget ran out, without repeating itself.</summary>
    Unfinished,
}

/// <summary>
/// What a reply did, read from its tokens and text. Only a committed answer counts: the text
/// after the last <c>&lt;/think&gt;</c>, or the whole reply when thinking was off. A number
/// that appears only while thinking is not an answer, however right it is; that case is
/// reported separately (<see cref="FoundWhileThinking"/>), because it measures a failure to
/// commit rather than a failure to solve.
/// </summary>
public sealed record ReplyAnalysis(
    ReplyOutcome Outcome,
    string? Answer,
    bool Closed,
    int ThinkingTokens,
    int Tokens,
    int? LoopAt,
    int? FoundAt,
    bool HitBudget)
{
    /// <summary>The expected answer appeared as a number in the thinking.</summary>
    public bool FoundWhileThinking => FoundAt is not null;

    /// <summary>A run of this many tokens seen three times marks a loop.</summary>
    public const int LoopSpan = 24;

    private static readonly Regex Number = new(@"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", RegexOptions.Compiled);
    private static readonly Regex Boxed = new(@"\\boxed\{([^{}]*)\}", RegexOptions.Compiled);
    // A line that is only an answer: "Answer: 12", "**Final answer:** 12".
    private static readonly Regex AnswerLine = new(@"^[\s*#>_]*(?:final\s+)?answer[\s*_]*[:=][\s*_$]*(-?\d[\d,]*(?:\.\d+)?)[\s*_$.]*$",
        RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Multiline);

    /// <summary>
    /// Reads a reply. <paramref name="tokens"/> are the reply's ids, <paramref name="pieces"/>
    /// their decoded text one by one (so a position in tokens can be found from text), and
    /// <paramref name="thinking"/> says the reply began inside a reasoning block.
    /// </summary>
    public static ReplyAnalysis Read(IReadOnlyList<int> tokens, IReadOnlyList<string> pieces, string expected,
        bool thinking, bool hitBudget)
    {
        string text = string.Concat(pieces);
        int close = thinking ? IndexOfPiece(pieces, "</think>") : -1;
        bool closed = !thinking || close >= 0;
        int thinkingTokens = thinking ? (close >= 0 ? close : tokens.Count) : 0;
        string answerText = thinking ? (close >= 0 ? string.Concat(pieces.Skip(close + 1)) : string.Empty) : text;
        string? answer = closed ? Extract(answerText) : null;
        int? loopAt = FindLoop(tokens);
        int? foundAt = thinking ? FindNumber(pieces, expected, thinkingTokens) : null;

        var outcome = !closed
            ? (loopAt is not null ? ReplyOutcome.Looped : ReplyOutcome.Unfinished)
            : answer is null ? ReplyOutcome.NoNumber
            : answer == expected ? ReplyOutcome.Correct : ReplyOutcome.Wrong;
        return new(outcome, answer, closed, thinkingTokens, tokens.Count, loopAt, foundAt, hitBudget);
    }

    /// <summary>The answer in a final text: the last <c>\boxed{}</c> holding a number, or else
    /// the last line that is only an answer (<c>Answer: 12</c>), or else the last number. The
    /// last number is a guess: a reply that gives its answer and then explains it ends on a
    /// number of the explanation, so a suite should ask for an answer line
    /// (<see cref="ProblemSuite.AnswerLine"/>). Thousands separators are dropped and a whole
    /// decimal is read whole.</summary>
    public static string? Extract(string answerText)
    {
        var boxed = Boxed.Matches(answerText);
        for (int i = boxed.Count - 1; i >= 0; i--)
        {
            if (Number.Matches(boxed[i].Groups[1].Value) is { Count: > 0 } inBox)
            {
                return Normalise(inBox[^1].Value);
            }
        }
        if (AnswerLine.Matches(answerText) is { Count: > 0 } lines)
        {
            return Normalise(lines[^1].Groups[1].Value);
        }
        var numbers = Number.Matches(answerText);
        return numbers.Count == 0 ? null : Normalise(numbers[^1].Value);
    }

    private static string Normalise(string number)
    {
        string n = number.Replace(",", string.Empty, StringComparison.Ordinal);
        int dot = n.IndexOf('.');
        return dot >= 0 && n[(dot + 1)..].All(c => c == '0') ? n[..dot] : n;
    }

    private static int IndexOfPiece(IReadOnlyList<string> pieces, string piece)
    {
        for (int i = 0; i < pieces.Count; i++)
        {
            if (pieces[i] == piece)
            {
                return i;
            }
        }
        return -1;
    }

    /// <summary>The token at which some run of <see cref="LoopSpan"/> tokens is seen for the
    /// third time, or null when none is.</summary>
    public static int? FindLoop(IReadOnlyList<int> tokens)
    {
        var counts = new Dictionary<ulong, int>();
        for (int end = LoopSpan; end <= tokens.Count; end++)
        {
            ulong h = 1469598103934665603UL;
            for (int i = end - LoopSpan; i < end; i++)
            {
                h = (h ^ (uint)tokens[i]) * 1099511628211UL;
            }
            int seen = counts.GetValueOrDefault(h) + 1;
            counts[h] = seen;
            if (seen == 3)
            {
                return end - 1;
            }
        }
        return null;
    }

    /// <summary>The first token of the thinking at which the expected number is written.</summary>
    private static int? FindNumber(IReadOnlyList<string> pieces, string expected, int limit)
    {
        var at = new int[limit + 1];
        var text = new System.Text.StringBuilder();
        for (int i = 0; i < limit; i++)
        {
            at[i] = text.Length;
            text.Append(pieces[i]);
        }
        foreach (Match m in Number.Matches(text.ToString()))
        {
            if (Normalise(m.Value) == expected)
            {
                int token = Array.FindLastIndex(at, limit - 1, limit, p => p <= m.Index);
                return Math.Max(token, 0);
            }
        }
        return null;
    }
}
