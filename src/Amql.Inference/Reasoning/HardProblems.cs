namespace Amql.Inference.Reasoning;

/// <summary>
/// The same traits as <see cref="ProblemSuite"/>'s first suite at a size a model of a few
/// billion parameters does not saturate: more steps, more constraints, and searches that need
/// a method. Every answer is computed by brute force or direct evaluation, and is a whole number.
/// <list type="bullet">
/// <item><c>operators</c>: two made-up operators, nested three deep.</item>
/// <item><c>registers</c>: ten to fourteen updates on five registers, some conditional.</item>
/// <item><c>route</c>: the cheapest path through a small directed graph of road costs.</item>
/// <item><c>recurrence</c>: induce a rule that reads the two terms before.</item>
/// <item><c>sieve</c>: count under three constraints.</item>
/// <item><c>ledger</c>: a word problem of five dependent quantities.</item>
/// <item><c>seating</c>: the one arrangement that fits every clue.</item>
/// <item><c>truth</c>: how many of a group are truth-tellers, given what each says.</item>
/// </list>
/// </summary>
internal static class HardProblems
{
    private static readonly string[] Names = ["Ava", "Ben", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun"];

    public static readonly IReadOnlyDictionary<string, Func<Random, Problem>> Families =
        new Dictionary<string, Func<Random, Problem>>(StringComparer.Ordinal)
        {
            ["operators"] = Operators,
            ["registers"] = Registers,
            ["route"] = Route,
            ["recurrence"] = Recurrence,
            ["sieve"] = Sieve,
            ["ledger"] = Ledger,
            ["seating"] = Seating,
            ["truth"] = Truth,
        };

    private static Problem Operators(Random r)
    {
        int k = r.Next(2, 5), j = r.Next(1, 4);
        (string Rule, Func<long, long, long> F)[] hash =
        [
            ("a # b = a * b - a", (x, y) => x * y - x),
            ($"a # b = {k} * a - b", (x, y) => k * x - y),
            ("a # b = (a + b) * 2 - b", (x, y) => (x + y) * 2 - y),
        ];
        (string Rule, Func<long, long, long> F)[] at =
        [
            ($"a @ b = a + {k} * b", (x, y) => x + k * y),
            ($"a @ b = b * b - {j} * a", (x, y) => y * y - j * x),
            ("a @ b = a - b when a is greater than b, and a + b otherwise", (x, y) => x > y ? x - y : x + y),
        ];
        var (hr, h) = hash[r.Next(hash.Length)];
        var (ar, t) = at[r.Next(at.Length)];
        long a = r.Next(2, 8), b = r.Next(2, 8), c = r.Next(2, 8), d = r.Next(2, 8);
        var (q, answer) = r.Next(3) switch
        {
            0 => ($"(({a} # {b}) @ {c}) # {d}", h(t(h(a, b), c), d)),
            1 => ($"({a} @ {b}) # ({c} @ {d})", h(t(a, b), t(c, d))),
            _ => ($"{a} @ (({b} # {c}) @ {d})", t(a, t(h(b, c), d))),
        };
        return new("operators", $"Two new operations are defined by: {hr}, and {ar}. What is {q}? Give the final answer as a number.", answer);
    }

    private static Problem Registers(Random r)
    {
        string[] keys = ["p", "q", "r", "s", "t"];
        var v = keys.ToDictionary(k => k, _ => (long)r.Next(1, 10));
        var lines = new List<string> { string.Join(", ", keys.Select(k => $"{k} = {v[k]}")) + "." };
        int steps = r.Next(10, 15);
        for (int i = 0; i < steps; i++)
        {
            string a = keys[r.Next(5)], b;
            do
            {
                b = keys[r.Next(5)];
            } while (b == a);
            switch (r.Next(6))
            {
                case 0: lines.Add($"{a} = {a} + {b}."); v[a] += v[b]; break;
                case 1: lines.Add($"{a} = {a} - {b}."); v[a] -= v[b]; break;
                case 2: lines.Add($"Swap {a} and {b}."); (v[a], v[b]) = (v[b], v[a]); break;
                case 3: lines.Add($"If {a} is greater than {b}, then {a} = {a} - 3, otherwise {b} = {b} + 2."); if (v[a] > v[b]) v[a] -= 3; else v[b] += 2; break;
                case 4: lines.Add($"If {a} is even, then {a} = {a} + {b}, otherwise {a} = {a} + 1."); v[a] += v[a] % 2 == 0 ? v[b] : 1; break;
                default: lines.Add($"{a} = {b}."); v[a] = v[b]; break;
            }
        }
        string ask = keys[r.Next(5)];
        return new("registers", $"Run these steps in order. {string.Join(" ", lines)} What is the final value of {ask}?", v[ask]);
    }

    private static Problem Route(Random r)
    {
        while (true)
        {
            string[] towns = ["A", "B", "C", "D", "E", "F", "G"];
            int n = towns.Length;
            var cost = new int[n, n];
            var roads = new List<string>();
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i != j && r.NextDouble() < 0.4 && !(i == 0 && j == n - 1))
                    {
                        cost[i, j] = r.Next(2, 15);
                        roads.Add($"{towns[i]} to {towns[j]} costs {cost[i, j]}");
                    }
                }
            }
            var best = Enumerable.Repeat(int.MaxValue / 2, n).ToArray();
            var hops = new int[n];
            best[0] = 0;
            for (int pass = 0; pass < n; pass++)
            {
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        if (cost[i, j] > 0 && best[i] + cost[i, j] < best[j])
                        {
                            best[j] = best[i] + cost[i, j];
                            hops[j] = hops[i] + 1;
                        }
                    }
                }
            }
            if (best[n - 1] < int.MaxValue / 2 && hops[n - 1] >= 3)
            {
                var order = roads.OrderBy(_ => r.Next()).ToList();
                return new("route", $"One-way roads between towns have these costs: {string.Join("; ", order)}. " +
                    "What is the lowest total cost of travelling from A to G?", best[n - 1]);
            }
        }
    }

    private static Problem Recurrence(Random r)
    {
        long m = r.Next(1, 4), k = r.Next(1, 3), c = r.Next(-3, 4);
        var seq = new List<long> { r.Next(1, 5), r.Next(1, 6) };
        for (int i = 0; i < 5; i++)
        {
            seq.Add(m * seq[^1] + k * seq[^2] + c);
        }
        string told = $"each term from the third on is {m} times the term before it plus {k} times the term before that, plus a constant";
        return new("recurrence", $"In the sequence {string.Join(", ", seq.Take(4))}, ... {told} that is the same every time " +
            "(it may be negative or zero). What is the seventh term?", seq[6]);
    }

    private static Problem Sieve(Random r)
    {
        int n = r.Next(90, 161);
        int[] pool = [2, 3, 4, 5, 6, 7, 8, 9];
        var pick = pool.OrderBy(_ => r.Next()).Take(3).ToArray();
        long answer = Enumerable.Range(1, n).Count(i => (i % pick[0] == 0 || i % pick[1] == 0) && i % pick[2] != 0);
        return new("sieve", $"How many whole numbers from 1 to {n} are divisible by {pick[0]} or by {pick[1]} (or both), but not by {pick[2]}?", answer);
    }

    private static Problem Ledger(Random r)
    {
        while (true)
        {
            int boxes = r.Next(4, 10), per = r.Next(12, 31), broken = r.Next(3, 12), sold = r.Next(20, 61), price = r.Next(2, 8),
                restock = r.Next(2, 5), refund = r.Next(1, 7);
            int left = boxes * per - broken - sold;
            if (left <= 0)
            {
                continue;
            }
            long stock = left + (long)restock * per + refund;
            return r.NextDouble() < 0.5
                ? new("ledger", $"A shop receives {boxes} boxes of {per} mugs. {broken} mugs are broken and thrown away. It sells {sold} mugs at " +
                    $"{price} pounds each, then {refund} customers each return one mug for a full refund and those mugs go back on the shelf. It then receives " +
                    $"{restock} more boxes of the same size. How many mugs does the shop have now?", stock)
                : new("ledger", $"A shop receives {boxes} boxes of {per} mugs. {broken} mugs are broken and thrown away. It sells {sold} mugs at " +
                    $"{price} pounds each, then {refund} customers each return one mug for a full refund. Every mug still in the shop, returned ones included, is then sold " +
                    $"at 1 pound less than before. How many pounds has the shop taken in total, after refunds?",
                    (long)sold * price - (long)refund * price + (long)(left + refund) * (price - 1));
        }
    }

    private static Problem Seating(Random r)
    {
        while (true)
        {
            var people = Names.OrderBy(_ => r.Next()).Take(5).ToArray();
            var truth = people.OrderBy(_ => r.Next()).ToArray();   // seat i + 1 holds truth[i]
            int Seat(string p) => Array.IndexOf(truth, p) + 1;
            var clues = new List<(string Text, Func<string[], bool> Holds)>();
            int At(string[] o, string p) => Array.IndexOf(o, p) + 1;
            for (int tries = 0; tries < 12; tries++)
            {
                string a = people[r.Next(5)], b;
                do
                {
                    b = people[r.Next(5)];
                } while (b == a);
                int sa = Seat(a), sb = Seat(b);
                switch (r.Next(4))
                {
                    case 0 when sa < sb: clues.Add(($"{a} sits somewhere to the left of {b}.", o => At(o, a) < At(o, b))); break;
                    case 1 when Math.Abs(sa - sb) == 1: clues.Add(($"{a} sits next to {b}.", o => Math.Abs(At(o, a) - At(o, b)) == 1)); break;
                    case 2 when Math.Abs(sa - sb) != 1: clues.Add(($"{a} does not sit next to {b}.", o => Math.Abs(At(o, a) - At(o, b)) != 1)); break;
                    case 3 when sa is not (1 or 5): clues.Add(($"{a} does not sit in seat 1 or seat 5.", o => At(o, a) is not (1 or 5))); break;
                }
            }
            clues = clues.DistinctBy(c => c.Text).ToList();
            int Fits(List<(string Text, Func<string[], bool> Holds)> cs) => Permutations(people).Count(o => cs.All(c => c.Holds(o)));
            if (Fits(clues) != 1)
            {
                continue;
            }
            // Only the clues it takes.
            foreach (var c in clues.OrderBy(_ => r.Next()).ToList())
            {
                var fewer = clues.Where(x => x.Text != c.Text).ToList();
                if (Fits(fewer) == 1)
                {
                    clues = fewer;
                }
            }
            string ask = people[r.Next(5)];
            return new("seating", $"Five people sit in a row of seats numbered 1 to 5 from left to right: {string.Join(", ", people)}. " +
                $"{string.Join(" ", clues.Select(c => c.Text))} What is the number of {ask}'s seat?", Seat(ask));
        }
    }

    private static Problem Truth(Random r)
    {
        while (true)
        {
            var people = Names.OrderBy(_ => r.Next()).Take(4).ToArray();
            int mask = r.Next(16);   // bit i: person i tells the truth
            var said = new List<(string Text, Func<int, bool> Holds)>();
            for (int i = 0; i < 4; i++)
            {
                int j;
                do
                {
                    j = r.Next(4);
                } while (j == i);
                int other = j;
                int count = r.Next(1, 4);
                (string Text, Func<int, bool> Holds)[] options =
                [
                    ($"{people[j]} is a liar", m => (m >> other & 1) == 0),
                    ($"{people[j]} is a truth-teller", m => (m >> other & 1) == 1),
                    ($"exactly {count} of us {(count == 1 ? "is a truth-teller" : "are truth-tellers")}", m => System.Numerics.BitOperations.PopCount((uint)m) == count),
                    ($"at least {count} of us {(count == 1 ? "is a liar" : "are liars")}", m => 4 - System.Numerics.BitOperations.PopCount((uint)m) >= count),
                ];
                bool tells = (mask >> i & 1) == 1;
                var fitting = options.Where(o => o.Holds(mask) == tells).ToArray();
                if (fitting.Length == 0)
                {
                    break;
                }
                said.Add(fitting[r.Next(fitting.Length)]);
            }
            if (said.Count < 4)
            {
                continue;
            }
            var consistent = Enumerable.Range(0, 16).Where(m => Enumerable.Range(0, 4).All(i => said[i].Holds(m) == ((m >> i & 1) == 1))).ToList();
            if (consistent.Select(m => System.Numerics.BitOperations.PopCount((uint)m)).Distinct().Count() != 1)
            {
                continue;
            }
            return new("truth", $"Four people each always tell the truth or always lie. " +
                $"{string.Join(" ", Enumerable.Range(0, 4).Select(i => $"{people[i]} says: \"{char.ToUpperInvariant(said[i].Text[0]) + said[i].Text[1..]}.\""))} " +
                "How many of the four are truth-tellers?", System.Numerics.BitOperations.PopCount((uint)consistent[0]));
        }
    }

    private static IEnumerable<string[]> Permutations(string[] items)
    {
        if (items.Length <= 1)
        {
            yield return items;
            yield break;
        }
        for (int i = 0; i < items.Length; i++)
        {
            string head = items[i];
            foreach (var rest in Permutations(items.Where((_, j) => j != i).ToArray()))
            {
                yield return [head, .. rest];
            }
        }
    }
}
