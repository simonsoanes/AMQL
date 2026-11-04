using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Amql.Entail;

/// <summary>One relation of a subject: a template with slots, whose
/// continuation the model is asked for across a declared domain.</summary>
public sealed class SubjectRelation
{
    public required string Name { get; init; }
    public string Shots { get; init; } = string.Empty;
    public required string Template { get; init; }
    /// <summary>Slot name → its values, in template order.</summary>
    public required List<(string Name, IReadOnlyList<string> Values)> Slots { get; init; }
    public int Sample { get; init; }
    public string Until { get; init; } = "\n";
    public int MaxTokens { get; init; } = 8;

    /// <summary>Every instance of the template, or a seeded sample of them.</summary>
    public IEnumerable<IReadOnlyList<string>> Instances(int seed)
    {
        long total = Slots.Aggregate(1L, (n, s) => n * s.Values.Count);
        if (Sample <= 0 || Sample >= total)
        {
            for (long i = 0; i < total; i++)
            {
                yield return Instance(i);
            }
            yield break;
        }
        var random = new Random(seed);
        var seen = new HashSet<long>();
        while (seen.Count < Sample)
        {
            long i = random.NextInt64(total);
            if (seen.Add(i))
            {
                yield return Instance(i);
            }
        }
    }

    private List<string> Instance(long index)
    {
        var values = new List<string>();
        foreach (var (_, domain) in Slots)
        {
            values.Add(domain[(int)(index % domain.Count)]);
            index /= domain.Count;
        }
        return values;
    }

    public string Render(IReadOnlyList<string> values)
    {
        var text = Template;
        for (int i = 0; i < Slots.Count; i++)
        {
            text = text.Replace("{" + Slots[i].Name + "}", values[i], StringComparison.Ordinal);
        }
        return text;
    }
}

/// <summary>A subject: text that shows it, text that does not, and the
/// relations that define it.</summary>
public sealed class Subject
{
    public required string Name { get; init; }
    public required IReadOnlyList<string> Lines { get; init; }
    public required IReadOnlyList<string> Contrast { get; init; }
    public required IReadOnlyList<SubjectRelation> Relations { get; init; }

    public static Subject Load(string path)
    {
        var root = JsonNode.Parse(File.ReadAllText(path))!.AsObject();
        static List<string> Strings(JsonNode? node) =>
            node?.AsArray().Select(x => x!.GetValue<string>()).ToList() ?? new List<string>();
        var relations = new List<SubjectRelation>();
        foreach (var item in root["relations"]?.AsArray() ?? new JsonArray())
        {
            var r = item!.AsObject();
            var slots = new List<(string, IReadOnlyList<string>)>();
            foreach (var (name, domain) in r["slots"]!.AsObject())
            {
                slots.Add((name, Domain(domain!)));
            }
            var template = r["template"]!.GetValue<string>();
            slots.Sort((x, y) => template.IndexOf("{" + x.Item1 + "}", StringComparison.Ordinal)
                .CompareTo(template.IndexOf("{" + y.Item1 + "}", StringComparison.Ordinal)));
            relations.Add(new SubjectRelation
            {
                Name = r["name"]!.GetValue<string>(),
                Shots = r["shots"]?.GetValue<string>() ?? string.Empty,
                Template = template,
                Slots = slots,
                Sample = r["sample"]?.GetValue<int>() ?? 0,
                Until = r["until"]?.GetValue<string>() ?? "\n",
                MaxTokens = r["max_tokens"]?.GetValue<int>() ?? 8,
            });
        }
        return new Subject
        {
            Name = root["name"]!.GetValue<string>(),
            Lines = Strings(root["lines"]),
            Contrast = Strings(root["contrast"]),
            Relations = relations,
        };
    }

    /// <summary>A domain is a list of strings, or "lo-hi" for whole numbers.</summary>
    private static IReadOnlyList<string> Domain(JsonNode node)
    {
        if (node is JsonArray list)
        {
            return list.Select(x => x!.GetValue<string>()).ToList();
        }
        var parts = node.GetValue<string>().Split('-');
        int lo = int.Parse(parts[0], CultureInfo.InvariantCulture);
        int hi = int.Parse(parts[1], CultureInfo.InvariantCulture);
        return Enumerable.Range(lo, hi - lo + 1).Select(i => i.ToString(CultureInfo.InvariantCulture)).ToList();
    }
}

/// <summary>What a model did on one instance of a relation. Output tokens
/// exclude the terminator; <see cref="Margin"/> is the smallest gap between
/// the top two probabilities over the answer: how far the softmax was from a
/// decision.</summary>
public sealed record EvidenceRow(
    string Relation,
    IReadOnlyList<string> Values,
    IReadOnlyList<string> Context,
    IReadOnlyList<string> Output,
    double Margin,
    bool Holdout)
{
    public string Answer => string.Concat(Output);

    public JsonObject ToJson() => new()
    {
        ["relation"] = Relation,
        ["values"] = new JsonArray(Values.Select(v => (JsonNode?)v).ToArray()),
        ["context"] = new JsonArray(Context.Select(v => (JsonNode?)v).ToArray()),
        ["output"] = new JsonArray(Output.Select(v => (JsonNode?)v).ToArray()),
        ["margin"] = Math.Round(Margin, 5),
        ["holdout"] = Holdout,
    };

    public static EvidenceRow FromJson(JsonNode node)
    {
        var o = node.AsObject();
        static List<string> Strings(JsonNode? n) => n!.AsArray().Select(x => x!.GetValue<string>()).ToList();
        return new EvidenceRow(o["relation"]!.GetValue<string>(), Strings(o["values"]), Strings(o["context"]),
            Strings(o["output"]), o["margin"]!.GetValue<double>(), o["holdout"]!.GetValue<bool>());
    }

    public static List<EvidenceRow> Load(string path) =>
        File.ReadAllLines(path).Where(l => l.Trim().Length > 0).Select(l => FromJson(JsonNode.Parse(l)!)).ToList();
}

/// <summary>Ordinary text, and the token the model predicts after each of its
/// tokens: the negative evidence. Held-out rows show where a program should
/// speak; this shows where it should not.</summary>
public sealed record BackgroundLine(IReadOnlyList<string> Tokens, IReadOnlyList<string> Next)
{
    public JsonObject ToJson() => new()
    {
        ["tokens"] = new JsonArray(Tokens.Select(v => (JsonNode?)v).ToArray()),
        ["next"] = new JsonArray(Next.Select(v => (JsonNode?)v).ToArray()),
    };

    public static List<BackgroundLine> Load(string path)
    {
        if (!File.Exists(path))
        {
            return new List<BackgroundLine>();
        }
        static List<string> Strings(JsonNode? n) => n!.AsArray().Select(x => x!.GetValue<string>()).ToList();
        return File.ReadAllLines(path).Where(l => l.Trim().Length > 0)
            .Select(l => JsonNode.Parse(l)!.AsObject())
            .Select(o => new BackgroundLine(Strings(o["tokens"]), Strings(o["next"]))).ToList();
    }
}

/// <summary>A closed form over a relation's numeric slots:
/// value = c0 + c1·a + c2·b + c3·a·b, with whole coefficients.</summary>
public sealed record Law(long C0, long C1, long C2, long C3, int Inliers, int Rows)
{
    public long Eval(long a, long b) => C0 + C1 * a + C2 * b + C3 * a * b;

    public override string ToString()
    {
        var terms = new List<string>();
        void Term(long c, string name)
        {
            if (c != 0)
            {
                terms.Add(name.Length == 0 ? c.ToString(CultureInfo.InvariantCulture)
                    : c == 1 ? name : c == -1 ? "-" + name : $"{c}*{name}");
            }
        }
        Term(C0, string.Empty);
        Term(C1, "a");
        Term(C2, "b");
        Term(C3, "a*b");
        return terms.Count == 0 ? "0" : string.Join(" + ", terms).Replace("+ -", "- ", StringComparison.Ordinal);
    }
}

/// <summary>What the mechanical passes have decided about one relation.</summary>
public sealed class RelationPlan
{
    public required string Name { get; init; }
    public required List<EvidenceRow> Rows { get; set; }
    /// <summary>Context columns that hold slot tokens; the rest are literal.</summary>
    public required List<(int Start, int End)> SlotColumns { get; init; }
    public int Length { get; init; }
    /// <summary>R3: literal conditions hoisted into one shared fact.</summary>
    public bool Factored { get; set; }
    /// <summary>R4: slot groups that turned out not to matter.</summary>
    public HashSet<int> Dropped { get; } = new();
    /// <summary>R5: the closed form replacing the table.</summary>
    public Law? Law { get; set; }
}

/// <summary>The structured form of a subject's program, from which Entail
/// text is generated.</summary>
public sealed class SubjectPlan
{
    public required string Name { get; init; }
    public required List<RelationPlan> Relations { get; init; }
    public double Confidence { get; set; } = 0.5;

    public static SubjectPlan FromEvidence(string name, IReadOnlyList<EvidenceRow> rows)
    {
        var relations = new List<RelationPlan>();
        foreach (var group in rows.GroupBy(r => r.Relation))
        {
            int length = group.Max(r => r.Context.Count);
            var usable = group.Where(r => r.Context.Count == length).ToList();
            var varying = Enumerable.Range(0, length)
                .Select(i => usable.Select(r => r.Context[i]).Distinct().Count() > 1).ToArray();
            var columns = new List<(int, int)>();
            for (int i = 0; i < length; i++)
            {
                if (varying[i])
                {
                    int start = i;
                    while (i + 1 < length && varying[i + 1])
                    {
                        i++;
                    }
                    columns.Add((start, i));
                }
            }
            // Adjacent slots read as one run of varying tokens; declared slots
            // of a single token each are told apart by counting them.
            if (columns.Count < usable[0].Values.Count)
            {
                columns = Enumerable.Range(0, length).Where(i => varying[i]).Select(i => (i, i)).ToList();
            }
            relations.Add(new RelationPlan
            {
                Name = group.Key, Rows = usable, SlotColumns = columns, Length = length,
            });
        }
        return new SubjectPlan { Name = name, Relations = relations };
    }
}

/// <summary>A program's measured standing: what the gate compares.</summary>
public sealed record Standing(
    int Neurons, int Heads, int Layers, int Lanes,
    int FitAgree, int FitCount, int HoldAgree, int HoldCount, int Silent,
    IReadOnlyList<(EvidenceRow Row, string Said)> Overruled,
    int FalseFires = 0)
{
    public double HoldRate => HoldCount == 0 ? 0 : (double)HoldAgree / HoldCount;
    public double FitRate => FitCount == 0 ? 0 : (double)FitAgree / FitCount;
    public int Cost => Neurons + 8 * Heads;

    public override string ToString() =>
        $"held-out {HoldAgree}/{HoldCount} ({100 * HoldRate:F1}%), fitted {FitAgree}/{FitCount} ({100 * FitRate:F1}%), " +
        $"{FalseFires} false fires, {Neurons} neurons, {Heads} heads, {Layers} layers";
}

/// <summary>
/// The optimisation cycle's machinery. Each rule is documented in
/// docs/entail-optimisation.md; a change of any origin is kept only if
/// <see cref="Gate"/> accepts it.
/// </summary>
public static class EntailOptimiser
{
    // ── measuring ───────────────────────────────────────────────────────────

    /// <summary>Compiles a program and scores it against confident evidence:
    /// the rows whose softmax had actually decided.</summary>
    public static Standing Measure(string source, IReadOnlyList<EvidenceRow> rows, double confidence,
        IReadOnlyList<BackgroundLine>? background = null)
    {
        var p = EntailCompiler.Compile(source);
        int fitAgree = 0, fitCount = 0, holdAgree = 0, holdCount = 0, silent = 0;
        var overruled = new List<(EvidenceRow, string)>();
        var answers = new string[rows.Count];
        Parallel.For(0, rows.Count, i =>
            answers[i] = EntailInterpreter.Answer(p, rows[i].Context, rows[i].Output.Count + 2));
        for (int i = 0; i < rows.Count; i++)
        {
            var row = rows[i];
            string said = answers[i];
            bool agrees = said == row.Answer;
            if (said.Length == 0)
            {
                silent++;
            }
            else if (!agrees)
            {
                overruled.Add((row, said));
            }
            if (row.Margin < confidence)
            {
                continue;
            }
            if (row.Holdout)
            {
                holdCount++;
                holdAgree += agrees ? 1 : 0;
            }
            else
            {
                fitCount++;
                fitAgree += agrees ? 1 : 0;
            }
        }
        return new Standing(p.Ops.Count(o => o is Neuron), p.Ops.Count(o => o is Head), p.Layers, p.Lanes.Count,
            fitAgree, fitCount, holdAgree, holdCount, silent, overruled, FalseFires(p, background));
    }

    /// <summary>Positions of ordinary text where the program names a token
    /// other than the one the model predicts there.</summary>
    private static int FalseFires(EntailProgram p, IReadOnlyList<BackgroundLine>? background)
    {
        if (background is null || background.Count == 0)
        {
            return 0;
        }
        int total = 0;
        Parallel.ForEach(background, line =>
        {
            var lanes = EntailInterpreter.Run(p, line.Tokens);
            int fires = 0;
            for (int t = 0; t < lanes.Length; t++)
            {
                var said = EntailInterpreter.Output(p, lanes[t]);
                if (!string.IsNullOrEmpty(said) && said != line.Next[t])
                {
                    fires++;
                }
            }
            Interlocked.Add(ref total, fires);
        });
        return total;
    }

    /// <summary>The only way into the program. A change must not speak up
    /// more often where it should stay silent, must not lose held-out
    /// agreement, and must either gain some or cost less.</summary>
    public static (bool Accepted, string Why) Gate(Standing before, Standing after)
    {
        if (after.FalseFires > before.FalseFires)
        {
            return (false, $"false fires on ordinary text rose from {before.FalseFires} to {after.FalseFires}");
        }
        if (after.HoldRate < before.HoldRate - 1e-9)
        {
            return (false, $"held-out agreement fell from {100 * before.HoldRate:F1}% to {100 * after.HoldRate:F1}%");
        }
        if (after.HoldRate > before.HoldRate + 1e-9)
        {
            return (true, $"held-out agreement rose from {100 * before.HoldRate:F1}% to {100 * after.HoldRate:F1}%");
        }
        if (after.FitRate < before.FitRate - 1e-9)
        {
            return (false, $"fitted agreement fell from {100 * before.FitRate:F1}% to {100 * after.FitRate:F1}%");
        }
        if (after.Cost < before.Cost)
        {
            return (true, $"same agreement, cost fell from {before.Cost} to {after.Cost}");
        }
        return (false, "no gain in agreement or cost");
    }

    // ── R4: a slot that does not matter ─────────────────────────────────────

    /// <summary>Slot groups the answer does not depend on: rows that differ
    /// only there always give the same answer.</summary>
    public static List<int> IrrelevantSlots(RelationPlan relation, double confidence)
    {
        var result = new List<int>();
        var rows = relation.Rows.Where(r => r.Margin >= confidence && !r.Holdout).ToList();
        for (int slot = 0; slot < relation.SlotColumns.Count; slot++)
        {
            var (start, end) = relation.SlotColumns[slot];
            bool matters = rows
                .GroupBy(r => string.Join("\u0001", r.Context.Where((_, i) => i < start || i > end)))
                .Any(g => g.Select(r => r.Answer).Distinct().Count() > 1);
            // Needs evidence that the slot actually varied with the rest fixed.
            bool tested = rows
                .GroupBy(r => string.Join("\u0001", r.Context.Where((_, i) => i < start || i > end)))
                .Any(g => g.Select(r => string.Join(string.Empty, r.Context.Skip(start).Take(end - start + 1))).Distinct().Count() > 1);
            if (!matters && tested)
            {
                result.Add(slot);
            }
        }
        return result;
    }

    // ── R5: deriving a law ──────────────────────────────────────────────────

    /// <summary>
    /// Fits value = c0 + c1·a + c2·b + c3·a·b to the rows by trimmed least
    /// squares: fit, set aside the rows furthest from the fit, fit again. A
    /// law exists only if the coefficients come out whole and the rows agree
    /// with them exactly, which a noise-free relation does and a mere trend
    /// does not. Nothing is searched: the basis is fixed.
    /// </summary>
    public static Law? DeriveLaw(RelationPlan relation, double confidence)
    {
        if (relation.SlotColumns.Count is 0 or > 2)
        {
            return null;
        }
        var points = new List<(double A, double B, double V)>();
        foreach (var row in relation.Rows.Where(r => !r.Holdout && r.Margin >= confidence))
        {
            if (!Numeric(row, out var a, out var b, out var v))
            {
                return null;
            }
            points.Add((a, b, v));
        }
        bool two = relation.SlotColumns.Count == 2;
        int terms = two ? 4 : 2;
        if (points.Count < terms * 3)
        {
            return null;
        }

        double[] Basis((double A, double B, double V) p) =>
            two ? new[] { 1, p.A, p.B, p.A * p.B } : new[] { 1, p.A };

        var active = points;
        double[] c = Array.Empty<double>();
        for (int round = 0; round < 12; round++)
        {
            c = Solve(active.Select(Basis).ToList(), active.Select(p => p.V).ToList(), terms);
            if (c.Length == 0)
            {
                return null;
            }
            var coefficients = c;
            double Residual((double A, double B, double V) p) =>
                Math.Abs(Basis(p).Zip(coefficients, (x, y) => x * y).Sum() - p.V);
            var ordered = points.OrderBy(Residual).ToList();
            int keep = Math.Max(terms * 3, (int)(points.Count * (1 - 0.05 * (round + 1))));
            keep = Math.Max(keep, points.Count / 2);
            active = ordered.Take(keep).ToList();
        }
        if (c.Any(x => Math.Abs(x - Math.Round(x)) > 0.02))
        {
            return null;
        }
        var whole = c.Select(x => (long)Math.Round(x)).ToArray();
        var law = two
            ? new Law(whole[0], whole[1], whole[2], whole[3], 0, points.Count)
            : new Law(whole[0], whole[1], 0, 0, 0, points.Count);
        int inliers = points.Count(p => law.Eval((long)p.A, (long)p.B) == (long)p.V);
        if (inliers < points.Count * 0.6)
        {
            return null;
        }
        return law with { Inliers = inliers };
    }

    /// <summary>The slot values and answer of a row, if all are whole numbers.</summary>
    public static bool Numeric(EvidenceRow row, out double a, out double b, out double value)
    {
        a = b = value = 0;
        if (row.Values.Count is 0 or > 2 ||
            !double.TryParse(row.Values[0], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out a) ||
            (row.Values.Count == 2 && !double.TryParse(row.Values[1], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out b)))
        {
            return false;
        }
        return double.TryParse(row.Answer, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out value);
    }

    /// <summary>Normal equations, by Gaussian elimination with pivoting.</summary>
    private static double[] Solve(IReadOnlyList<double[]> x, IReadOnlyList<double> y, int n)
    {
        // Columns are rescaled first: a·b dwarfs 1 otherwise.
        var scale = new double[n];
        for (int j = 0; j < n; j++)
        {
            scale[j] = Math.Max(1e-12, x.Max(r => Math.Abs(r[j])));
        }
        var m = new double[n, n + 1];
        for (int r = 0; r < x.Count; r++)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    m[i, j] += x[r][i] / scale[i] * (x[r][j] / scale[j]);
                }
                m[i, n] += x[r][i] / scale[i] * y[r];
            }
        }
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < n; r++)
            {
                if (Math.Abs(m[r, col]) > Math.Abs(m[pivot, col]))
                {
                    pivot = r;
                }
            }
            if (Math.Abs(m[pivot, col]) < 1e-12)
            {
                return Array.Empty<double>();
            }
            for (int j = 0; j <= n; j++)
            {
                (m[col, j], m[pivot, j]) = (m[pivot, j], m[col, j]);
            }
            for (int r = 0; r < n; r++)
            {
                if (r != col)
                {
                    double f = m[r, col] / m[col, col];
                    for (int j = col; j <= n; j++)
                    {
                        m[r, j] -= f * m[col, j];
                    }
                }
            }
        }
        var result = new double[n];
        for (int i = 0; i < n; i++)
        {
            result[i] = m[i, n] / m[i, i] / scale[i];
        }
        return result;
    }

    // ── generating Entail ───────────────────────────────────────────────────

    private static string Quote(string token) => Quoted(token);

    private static bool Quotable(string token) => CanQuote(token);

    /// <summary>A token as an Entail string.</summary>
    public static string Quoted(string token) => "\"" + token
        .Replace("\n", "\\n", StringComparison.Ordinal)
        .Replace("\"", "\\q", StringComparison.Ordinal)
        .Replace("{", "\\{", StringComparison.Ordinal) + "\"";

    /// <summary>Whether a token can be written as an Entail string. A token
    /// holding a literal backslash escape cannot be told from the escape.</summary>
    public static bool CanQuote(string token) =>
        token.Length > 0 && !token.Contains('\r') && !token.Contains('\uFFFD') &&
        !token.Contains("\\n", StringComparison.Ordinal) && !token.Contains("\\q", StringComparison.Ordinal) &&
        !token.Contains("\\{", StringComparison.Ordinal);

    /// <summary>Writes a plan out as Entail: a readable program whose every
    /// line can be traced to rows of evidence.</summary>
    public static string Generate(SubjectPlan plan, string? provenance = null)
    {
        var text = new StringBuilder();
        text.AppendLine("# Generated by `amql-cli entail extract` and refined by `entail optimise`.");
        text.AppendLine("# The language is documented in docs/entail-language.md and the rewrite");
        text.AppendLine("# rules applied to this file in docs/entail-optimisation.md.");
        if (provenance is not null)
        {
            foreach (var line in provenance.Split('\n'))
            {
                text.Append("# ").AppendLine(line);
            }
        }
        text.AppendLine();
        text.AppendLine($"program {plan.Name}");
        text.AppendLine();

        // Tokens: a code each, and a value for the ones that are digits.
        var tokens = new SortedSet<string>(StringComparer.Ordinal);
        var outputs = new SortedSet<string>(StringComparer.Ordinal);
        int window = 1;
        bool anyLaw = plan.Relations.Any(r => r.Law is not null);
        foreach (var relation in plan.Relations)
        {
            int longest = relation.Rows.Max(r => r.Output.Count);
            if (relation.Law is not null)
            {
                longest = 1 + LawDigits(relation);
            }
            window = Math.Max(window, relation.Length + longest);
            foreach (var row in relation.Rows)
            {
                tokens.UnionWith(row.Context.Where(Quotable));
                tokens.UnionWith(row.Output.Where(Quotable));
                outputs.UnionWith(row.Output.Where(Quotable));
            }
        }
        if (anyLaw)
        {
            for (int d = 0; d <= 9; d++)
            {
                tokens.Add(d.ToString(CultureInfo.InvariantCulture));
                outputs.Add(d.ToString(CultureInfo.InvariantCulture));
            }
            tokens.Add("-");
            outputs.Add("-");
        }
        var code = tokens.Select((t, i) => (t, i + 1)).ToDictionary(x => x.t, x => x.Item2, StringComparer.Ordinal);
        var lane = outputs.Select((t, i) => (t, i)).ToDictionary(x => x.t, x => x.i, StringComparer.Ordinal);

        text.AppendLine("# What each token is.");
        text.AppendLine("state sym, num");
        foreach (var (token, id) in code)
        {
            bool digit = token.Length == 1 && char.IsAsciiDigit(token[0]);
            text.AppendLine($"token {Quote(token)} means sym = {id}" + (digit ? $", num = {token}" : string.Empty));
        }
        text.AppendLine();
        text.AppendLine("# The tokens behind this one.");
        text.AppendLine($"state w[{window + 1}], n[{window + 1}]");
        text.AppendLine($"for k in 1 to {window}");
        text.AppendLine("    attend back k");
        text.AppendLine("        w[k] <- sym");
        text.AppendLine("        n[k] <- num");
        text.AppendLine("    end");
        text.AppendLine("end");
        text.AppendLine("let c[0] = sym");
        text.AppendLine("let v[0] = num");
        text.AppendLine($"for k in 1 to {window}");
        text.AppendLine("    let c[k] = w[k]");
        text.AppendLine("    let v[k] = n[k]");
        text.AppendLine("end");
        text.AppendLine();
        text.AppendLine($"state o[{Math.Max(lane.Count, 1)}], stop");
        if (anyLaw)
        {
            int zero = code["0"], nine = code["9"], minus = code["-"];
            text.AppendLine();
            text.AppendLine("# Which tokens are digits, and which can stand in an answer.");
            text.AppendLine($"state dig[{window + 1}], ans[{window + 1}]");
            text.AppendLine($"for k in 0 to {window}");
            text.AppendLine($"    entail dig[k] when c[k] >= {zero}, c[k] <= {nine}");
            text.AppendLine($"    entail ans[k] when c[k] >= {zero}, c[k] <= {nine}");
            text.AppendLine($"    entail ans[k] when c[k] == {minus}");
            text.AppendLine("end");
        }

        foreach (var relation in plan.Relations)
        {
            text.AppendLine();
            if (relation.Law is not null)
            {
                WriteLaw(text, relation, code, lane);
            }
            else
            {
                WriteTable(text, relation, code, lane, plan.Confidence);
            }
        }

        text.AppendLine();
        text.AppendLine("# What must come next.");
        foreach (var (token, index) in lane)
        {
            text.AppendLine($"emit {Quote(token)} when o[{index}]");
        }
        text.AppendLine("emit end when stop");
        return text.ToString();
    }

    private static string Conditions(IEnumerable<int> columns, IReadOnlyList<string> context, int length, int shift,
        IReadOnlyDictionary<string, int> code)
    {
        var parts = new List<string>();
        foreach (int i in columns)
        {
            if (code.TryGetValue(context[i], out int id))
            {
                parts.Add($"c[{length - 1 - i + shift}] == {id}");
            }
        }
        return string.Join(", ", parts);
    }

    private static void WriteTable(StringBuilder text, RelationPlan r, IReadOnlyDictionary<string, int> code,
        IReadOnlyDictionary<string, int> lane, double confidence)
    {
        text.AppendLine($"# Relation {r.Name}: a table of what the model said, {r.Rows.Count(x => !x.Holdout)} instances.");
        var slotColumns = new HashSet<int>();
        for (int s = 0; s < r.SlotColumns.Count; s++)
        {
            if (r.Dropped.Contains(s))
            {
                continue;
            }
            for (int i = r.SlotColumns[s].Start; i <= r.SlotColumns[s].End; i++)
            {
                slotColumns.Add(i);
            }
        }
        var literal = Enumerable.Range(0, r.Length)
            .Where(i => !r.SlotColumns.Any(s => i >= s.Start && i <= s.End)).ToList();
        var first = r.Rows[0];
        var written = new HashSet<string>(StringComparer.Ordinal);
        int steps = r.Rows.Max(x => x.Output.Count) + 1;
        if (r.Factored)
        {
            text.AppendLine($"state {r.Name}_on[{steps}]");
            for (int step = 0; step < steps; step++)
            {
                text.AppendLine($"entail {r.Name}_on[{step}] when {Conditions(literal, first.Context, r.Length, step, code)}");
            }
        }
        foreach (var row in r.Rows.Where(x => !x.Holdout && x.Margin >= confidence))
        {
            var key = string.Join("\u0001", slotColumns.OrderBy(i => i).Select(i => row.Context[i]));
            for (int step = 0; step <= row.Output.Count; step++)
            {
                var conditions = new List<string>();
                conditions.Add(r.Factored
                    ? $"{r.Name}_on[{step}]"
                    : Conditions(literal, row.Context, r.Length, step, code));
                string slots = Conditions(slotColumns.OrderBy(i => i), row.Context, r.Length, step, code);
                if (slots.Length > 0)
                {
                    conditions.Add(slots);
                }
                for (int p = 0; p < step; p++)
                {
                    if (code.TryGetValue(row.Output[p], out int id))
                    {
                        conditions.Add($"c[{step - 1 - p}] == {id}");
                    }
                }
                string target = step < row.Output.Count
                    ? lane.TryGetValue(row.Output[step], out int index) ? $"o[{index}]" : string.Empty
                    : "stop";
                if (target.Length > 0 && written.Add(key + "\u0002" + step + "\u0002" + string.Join("\u0001", row.Output.Take(step)) + target))
                {
                    text.Append($"entail {target} when {string.Join(", ", conditions.Where(c => c.Length > 0))}");
                    if (step == 0)
                    {
                        text.Append($"   # {string.Join(" ", row.Values)} -> {row.Answer.Trim()} (margin {row.Margin:F2})");
                    }
                    text.AppendLine();
                }
            }
        }
    }

    private static int LawDigits(RelationPlan r)
    {
        long largest = 1;
        foreach (var row in r.Rows)
        {
            if (Numeric(row, out var a, out var b, out _))
            {
                largest = Math.Max(largest, Math.Abs(r.Law!.Eval((long)a, (long)b)));
            }
        }
        return largest.ToString(CultureInfo.InvariantCulture).Length + 1;
    }

    /// <summary>
    /// Replaces a table with arithmetic: read the slots as numbers, evaluate
    /// the law, and spell the result. Each answer position works it out
    /// afresh from the tokens it can see, so nothing is carried between them.
    /// </summary>
    private static void WriteLaw(StringBuilder text, RelationPlan r, IReadOnlyDictionary<string, int> code,
        IReadOnlyDictionary<string, int> lane)
    {
        var law = r.Law!;
        string p = r.Name;
        int digits = LawDigits(r);
        int steps = digits + 2;               // sign, digits, end of answer
        var first = r.Rows[0];
        var literal = Enumerable.Range(0, r.Length)
            .Where(i => !r.SlotColumns.Any(s => i >= s.Start && i <= s.End)).ToList();
        text.AppendLine($"# Relation {p}: value = {law}, derived from {law.Inliers} of {law.Rows} rows.");
        text.AppendLine($"# A slot wider than one digit reads as a number; the answer is spelled from it.");
        text.AppendLine($"state {p}_on[{steps}], {p}_prod[{steps}] scale 1/256, {p}_neg[{steps}], {p}_mag[{steps}] scale 1/256");
        text.AppendLine($"state {p}_d[{steps}][{digits}], {p}_big[{steps}][{digits + 1}], {p}_say[{steps}]");

        string Slot(int index, int step)
        {
            var (start, end) = r.SlotColumns[index];
            var parts = new List<string>();
            for (int i = start; i <= end; i++)
            {
                long place = (long)Math.Pow(10, end - i);
                parts.Add(place == 1 ? $"v[{r.Length - 1 - i + step}]" : $"{place}*v[{r.Length - 1 - i + step}]");
            }
            return string.Join(" + ", parts);
        }

        for (int step = 0; step < steps; step++)
        {
            var conditions = new List<string>();
            string literals = Conditions(literal, first.Context, r.Length, step, code);
            if (literals.Length > 0)
            {
                conditions.Add(literals);
            }
            // Slots must be digits, and whatever follows the template, answer.
            foreach (var (start, end) in r.SlotColumns)
            {
                for (int i = start; i <= end; i++)
                {
                    conditions.Add($"dig[{r.Length - 1 - i + step}]");
                }
            }
            for (int q = 0; q < step; q++)
            {
                conditions.Add($"ans[{q}]");
            }
            text.AppendLine();
            text.AppendLine($"# Answer position {step}.");
            text.AppendLine($"entail {p}_on[{step}] when {string.Join(", ", conditions)}");
            string a = Slot(0, step);
            string b = r.SlotColumns.Count == 2 ? Slot(1, step) : "0";
            var sum = new List<string>();
            if (law.C0 != 0)
            {
                sum.Add(law.C0.ToString(CultureInfo.InvariantCulture));
            }
            if (law.C1 != 0)
            {
                sum.Add($"{law.C1}*({a})");
            }
            if (law.C2 != 0)
            {
                sum.Add($"{law.C2}*({b})");
            }
            if (law.C3 != 0)
            {
                // a·b: every step of a's digits adds b, at that digit's place.
                var (start, end) = r.SlotColumns[0];
                for (int i = start; i <= end; i++)
                {
                    long place = (long)Math.Pow(10, end - i);
                    int offset = r.Length - 1 - i + step;
                    for (int d = 1; d <= 9; d++)
                    {
                        text.AppendLine($"entail {p}_prod[{step}] = {law.C3 * place}*({b}) when {p}_on[{step}], v[{offset}] >= {d}");
                    }
                }
                sum.Add($"{p}_prod[{step}]");
            }
            string value = sum.Count == 0 ? "0" : string.Join(" + ", sum);
            text.AppendLine($"let {p}_v[{step}] = {value}");
            text.AppendLine($"entail {p}_neg[{step}] when {p}_on[{step}], {p}_v[{step}] <= -1");
            text.AppendLine($"entail {p}_mag[{step}] = {p}_v[{step}] when {p}_on[{step}], not {p}_neg[{step}]");
            text.AppendLine($"entail {p}_mag[{step}] = -1*({p}_v[{step}]) when {p}_neg[{step}]");
            // Digits from the top: each is a staircase on what remains.
            string rest = $"{p}_mag[{step}]";
            for (int k = digits - 1; k >= 0; k--)
            {
                long place = (long)Math.Pow(10, k);
                for (int d = 1; d <= 9; d++)
                {
                    text.AppendLine($"entail {p}_d[{step}][{k}] when {rest} >= {d * place}");
                }
                rest += $" - {place}*{p}_d[{step}][{k}]";
            }
            for (int k = 1; k < digits; k++)
            {
                text.AppendLine($"entail {p}_big[{step}][{k}] when {p}_mag[{step}] >= {(long)Math.Pow(10, k)}");
            }

            // How long the answer is decides which digit this position shows.
            for (int length = 1; length <= digits; length++)
            {
                string exact = string.Join(", ", new[]
                {
                    length > 1 ? $"{p}_big[{step}][{length - 1}]" : null,
                    length < digits ? $"not {p}_big[{step}][{length}]" : null,
                }.Where(x => x is not null));
                string Sign(bool negative) => negative ? $"{p}_neg[{step}]" : $"not {p}_neg[{step}]";
                foreach (bool negative in new[] { false, true })
                {
                    string when = string.Join(", ", new[] { $"{p}_on[{step}]", Sign(negative), exact }.Where(x => x.Length > 0));
                    int position = step - (negative ? 1 : 0);
                    if (negative && step == 0)
                    {
                        text.AppendLine($"entail {p}_say[{step}] = 11 when {when}");
                    }
                    else if (position >= 0 && position < length)
                    {
                        text.AppendLine($"entail {p}_say[{step}] = {p}_d[{step}][{length - 1 - position}] + 1 when {when}");
                    }
                    else if (position == length)
                    {
                        text.AppendLine($"entail stop when {when}");
                    }
                }
            }
            for (int d = 0; d <= 9; d++)
            {
                text.AppendLine($"entail o[{lane[d.ToString(CultureInfo.InvariantCulture)]}] when {p}_say[{step}] == {d + 1}");
            }
            text.AppendLine($"entail o[{lane["-"]}] when {p}_say[{step}] == 11");
        }
    }
}
