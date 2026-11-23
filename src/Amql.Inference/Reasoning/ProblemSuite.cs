namespace Amql.Inference.Reasoning;

/// <summary>
/// One problem: its family, the question, and its whole-number answer. A case read from a file
/// may instead hold a prompt already rendered with its template and tools (<paramref name="Raw"/>),
/// the tool it should call first (<paramref name="Expect"/>, or "none"), its own token budget,
/// and whether it opens inside a reasoning block.
/// </summary>
public sealed record Problem(string Family, string Prompt, long Answer, bool Raw = false, string? Expect = null,
    int? MaxTokens = null, bool? Think = null, string? Id = null);

/// <summary>
/// Novel problems for measuring how a model solves problems it was not trained on. Each is
/// generated from a seed, so none can have been memorised, and each answer is a whole number,
/// so scoring needs no judge. The families test different traits:
/// <list type="bullet">
/// <item><c>operator</c>: apply a rule defined only in the prompt, nested.</item>
/// <item><c>state</c>: track registers through a list of updates and swaps.</item>
/// <item><c>plan</c>: find the fewest moves to a target (a breadth-first search).</item>
/// <item><c>sequence</c>: induce a two-parameter rule from examples.</item>
/// <item><c>count</c>: count under two constraints, which rewards a systematic method.</item>
/// <item><c>word</c>: a multi-step word problem with dependent quantities.</item>
/// <item><c>order</c>: deduce a rank from comparisons given out of order.</item>
/// <item><c>counterfactual</c>: follow a rule that contradicts world knowledge.</item>
/// </list>
/// A seed always gives the same problems: <see cref="Random"/> with a seed is a fixed algorithm.
/// </summary>
public static class ProblemSuite
{
    private static readonly string[] Names = ["Ava", "Ben", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun"];
    private static readonly string[] Items = ["apples", "marbles", "stickers", "coins", "pencils", "cards", "shells", "stamps"];

    public static readonly IReadOnlyDictionary<string, Func<Random, Problem>> Families =
        new Dictionary<string, Func<Random, Problem>>(StringComparer.Ordinal)
        {
            ["operator"] = MadeUpOperator,
            ["state"] = StateTracking,
            ["plan"] = ShortestMoves,
            ["sequence"] = SequenceRule,
            ["count"] = ConstrainedCount,
            ["word"] = WordProblem,
            ["order"] = Ordering,
            ["counterfactual"] = Counterfactual,
        };

    /// <summary><paramref name="per"/> problems of each family, in family order.</summary>
    public static List<Problem> Generate(int seed, int per, IEnumerable<string>? families = null)
    {
        var r = new Random(seed);
        var problems = new List<Problem>();
        foreach (string name in families ?? Families.Keys)
        {
            var make = Families.TryGetValue(name, out var f)
                ? f
                : throw new ArgumentException($"unknown problem family '{name}': use {string.Join(", ", Families.Keys)}");
            for (int i = 0; i < per; i++)
            {
                problems.Add(make(r));
            }
        }
        return problems;
    }

    private static Problem MadeUpOperator(Random r)
    {
        long a = r.Next(2, 10), b = r.Next(2, 10), c = r.Next(2, 10), k = r.Next(1, 5);
        (string Rule, Func<long, long, long> F)[] forms =
        [
            ("a # b = a * b + a - b", (x, y) => x * y + x - y),
            ("a # b = 2 * a + b * b", (x, y) => 2 * x + y * y),
            ($"a # b = (a + b) * {k} - a", (x, y) => (x + y) * k - x),
            ("a # b = a * a - b", (x, y) => x * x - y),
        ];
        var (rule, f) = forms[r.Next(forms.Length)];
        var (q, answer) = r.NextDouble() < 0.5
            ? ($"({a} # {b}) # {c}", f(f(a, b), c))
            : ($"{a} # ({b} # {c})", f(a, f(b, c)));
        return new("operator", $"A new operation # is defined by: {rule}. What is {q}? Give the final answer as a number.", answer);
    }

    private static Problem StateTracking(Random r)
    {
        string[] keys = ["x", "y", "z"];
        var v = keys.ToDictionary(k => k, _ => (long)r.Next(1, 10));
        var lines = new List<string> { $"x = {v["x"]}, y = {v["y"]}, z = {v["z"]}." };
        string[] ops = ["add", "swap", "double", "sub", "copy"];
        int steps = r.Next(4, 7);
        for (int i = 0; i < steps; i++)
        {
            string op = ops[r.Next(ops.Length)];
            int ai = r.Next(3), bi = (ai + 1 + r.Next(2)) % 3;
            string a = keys[ai], b = keys[bi];
            switch (op)
            {
                case "add": lines.Add($"{a} = {a} + {b}."); v[a] += v[b]; break;
                case "sub": lines.Add($"{a} = {a} - {b}."); v[a] -= v[b]; break;
                case "swap": lines.Add($"Swap {a} and {b}."); (v[a], v[b]) = (v[b], v[a]); break;
                case "double": lines.Add($"{a} = {a} * 2."); v[a] *= 2; break;
                default: lines.Add($"{a} = {b}."); v[a] = v[b]; break;
            }
        }
        string ask = keys[r.Next(3)];
        return new("state", $"Run these steps in order: {string.Join(" ", lines)} What is the final value of {ask}?", v[ask]);
    }

    private static Problem ShortestMoves(Random r)
    {
        while (true)
        {
            int up = r.Next(3, 8), down = r.Next(1, 5), target = r.Next(5, 26);
            if (up == down)
            {
                continue;
            }
            var seen = new Dictionary<int, int> { [0] = 0 };
            var frontier = new Queue<int>([0]);
            while (frontier.Count > 0)
            {
                int p = frontier.Dequeue();
                foreach (int q in new[] { p + up, p - down })
                {
                    if (q is >= -30 and <= 60 && seen.TryAdd(q, seen[p] + 1))
                    {
                        frontier.Enqueue(q);
                    }
                }
            }
            if (seen.TryGetValue(target, out int moves) && moves >= 3)
            {
                return new("plan", $"A robot starts at position 0 on a number line. Each move takes it either {up} steps " +
                    $"forward or {down} steps back. What is the smallest number of moves that puts it exactly at position {target}?", moves);
            }
        }
    }

    private static Problem SequenceRule(Random r)
    {
        long m = r.Next(2, 4), c = r.Next(-3, 5), x = r.Next(1, 6);
        var seq = new List<long> { x };
        for (int i = 0; i < 5; i++)
        {
            seq.Add(seq[^1] * m + c);
        }
        return new("sequence", $"What number comes next in the sequence {string.Join(", ", seq.Take(5))}, ...?", seq[5]);
    }

    private static Problem ConstrainedCount(Random r)
    {
        int n = r.Next(30, 81);
        int[] pool = [2, 3, 4, 5, 6, 7];
        int p = pool[r.Next(pool.Length)], q;
        do
        {
            q = pool[r.Next(pool.Length)];
        } while (q == p);
        long answer = Enumerable.Range(1, n).Count(i => i % p == 0 && i % q != 0);
        return new("count", $"How many whole numbers from 1 to {n} are divisible by {p} but not by {q}?", answer);
    }

    private static Problem WordProblem(Random r)
    {
        string a = Names[r.Next(Names.Length)], b;
        do
        {
            b = Names[r.Next(Names.Length)];
        } while (b == a);
        string item = Items[r.Next(Items.Length)];
        int x = r.Next(5, 31), give = r.Next(2, x), mult = r.Next(2, 5), extra = r.Next(1, 10);
        long bHas = (long)(x - give) * mult + extra;
        return r.NextDouble() < 0.5
            ? new("word", $"{a} has {x} {item} and gives {give} of them to a friend. {b} has {mult} times as many {item} as " +
                $"{a} has left, plus {extra} more. How many {item} do {a} and {b} have together?", bHas + x - give)
            : new("word", $"{a} has {x} {item} and gives {give} of them away. {b} then has {extra} more than {mult} times " +
                $"what {a} has left. How many {item} does {b} have?", bHas);
    }

    private static Problem Ordering(Random r)
    {
        var people = Names.OrderBy(_ => r.Next()).Take(r.Next(4, 6)).ToList();
        var facts = Enumerable.Range(0, people.Count - 1).Select(i => $"{people[i]} is taller than {people[i + 1]}.")
            .OrderBy(_ => r.Next()).ToList();
        int who = r.Next(people.Count);
        return new("order", $"{string.Join(" ", facts)} How many of these people are taller than {people[who]}?", who);
    }

    private static Problem Counterfactual(Random r)
    {
        int days = r.Next(5, 10), start = r.Next(1, days + 1), ahead = r.Next(days + 1, 4 * days + 1);
        return new("counterfactual", $"On the planet Vell a week has {days} days, numbered 1 to {days}, and after day {days} " +
            $"comes day 1 again. Today is day {start}. Which day number will it be {ahead} days from today?",
            (start - 1 + ahead) % days + 1);
    }
}
