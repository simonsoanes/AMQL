using System.Diagnostics;
using Amql.Inference;
using Amql.Inference.NGram;
using Amql.Vindex3;

namespace Amql.Cli;

// ── ngram: n-gram packs, a domain's next-token statistics blended into a model's logits ──

internal static partial class Program
{
    private static int NGramCommand(string[] args)
    {
        string sub = Arg(args, 0) ?? throw new CliException("ngram needs a subcommand: build, info or eval");
        return sub switch
        {
            "build" => NGramBuild(args[1..]),
            "info" => NGramInfo(args[1..]),
            "eval" => NGramEval(args[1..]),
            _ => throw new CliException($"unknown ngram subcommand '{sub}' (build, info, eval)"),
        };
    }

    /// <summary>Text files under the inputs (files, or folders searched for --ext), each a document.</summary>
    private static List<string> CorpusFiles(IEnumerable<string> inputs, string[] extensions)
    {
        var files = new List<string>();
        foreach (string input in inputs)
        {
            if (File.Exists(input))
            {
                files.Add(input);
            }
            else if (Directory.Exists(input))
            {
                files.AddRange(Directory.EnumerateFiles(input, "*", SearchOption.AllDirectories)
                    .Where(f => extensions.Contains(Path.GetExtension(f), StringComparer.OrdinalIgnoreCase))
                    .Where(f => !f.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar).Any(p => p is "bin" or "obj"))
                    .OrderBy(f => f, StringComparer.Ordinal));
            }
            else
            {
                throw new CliException($"'{input}' is neither a file nor a folder");
            }
        }
        return files;
    }

    private static string[] Extensions(string[] args) =>
        (OptionValue(args, "--ext") ?? ".cs,.txt,.md").Split(',', StringSplitOptions.RemoveEmptyEntries).Select(e => e.StartsWith('.') ? e : "." + e).ToArray();

    private static string TokenizerFingerprint(Amql.Hf.HfTokenizer tokenizer, int vocab) =>
        NGramPack.Fingerprint(vocab, id =>
        {
            try
            {
                return tokenizer.DecodeText(new[] { id }, skipSpecialTokens: false);
            }
            catch (Amql.Hf.TokenizerException)
            {
                return "";
            }
        });

    /// <summary><c>ngram build --tokenizer DIR --out PACK [--name N] [--order 4] [--keywords a,b] [--min-count 2] [--top-k 8] [--ext .cs,.txt] INPUTS…</c></summary>
    private static int NGramBuild(string[] args)
    {
        string tokenizerDir = OptionValue(args, "--tokenizer") ?? throw new CliException("ngram build needs --tokenizer <model or container dir>");
        string outPath = OptionValue(args, "--out") ?? throw new CliException("ngram build needs --out <pack file>");
        var tokenizer = Tokenizer(tokenizerDir);
        var options = new HashSet<string> { "--tokenizer", "--out", "--name", "--order", "--keywords", "--min-count", "--top-k", "--ext", "--vocab" };
        var inputs = new List<string>();
        for (int i = 0; i < args.Length; i++)
        {
            if (options.Contains(args[i]))
            {
                i++;
            }
            else
            {
                inputs.Add(args[i]);
            }
        }
        var files = CorpusFiles(inputs, Extensions(args));
        if (files.Count == 0)
        {
            throw new CliException("ngram build found no input files");
        }
        int vocab = IntOption(args, "--vocab", tokenizer.VocabSize);
        var clock = Stopwatch.StartNew();
        var docs = files.Select(f => tokenizer.EncodeToIds(File.ReadAllText(f)).ToArray()).ToList();
        string name = OptionValue(args, "--name") ?? Path.GetFileNameWithoutExtension(outPath);
        var keywords = (OptionValue(args, "--keywords") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var pack = NGramPack.Build(name, docs, IntOption(args, "--order", 4), TokenizerFingerprint(tokenizer, vocab), keywords,
            $"{files.Count} files", IntOption(args, "--min-count", 2), IntOption(args, "--top-k", 8));
        pack.Save(outPath);
        Console.WriteLine($"pack:      {pack.Name}, order {pack.Order}, {pack.ContextCount:N0} contexts from {pack.CorpusTokens:N0} tokens " +
                          $"in {files.Count} files ({clock.Elapsed.TotalSeconds:F1}s)");
        Console.WriteLine($"wrote:     {outPath} ({new FileInfo(outPath).Length / 1048576.0:F1} MB)");
        return 0;
    }

    private static int NGramInfo(string[] args)
    {
        var pack = NGramPack.Load(Arg(args, 0) ?? throw new CliException("ngram info needs a pack file"));
        Console.WriteLine($"pack:      {pack.Name}, order {pack.Order}, {pack.ContextCount:N0} contexts from {pack.CorpusTokens:N0} tokens ({pack.Source})");
        Console.WriteLine($"keywords:  {string.Join(", ", pack.Keywords)}");
        Console.WriteLine($"tokenizer: {pack.Tokenizer}");
        return 0;
    }

    /// <summary>
    /// <c>ngram eval CONTAINER --pack P[,P] --text FILES… [--lambda 0.1,0.3,0.5] [--max-tokens N] [--min-context 2]</c>:
    /// the model reads held-out text token by token (on the GPU engine when there is one), and each
    /// next token is scored under the model alone and blended with the packs at each λ: mean loss
    /// (nats), top-1 accuracy, and both again at the identifier after a '.', where a small model's
    /// spelling of a member name is the weak point.
    /// </summary>
    private static int NGramEval(string[] args)
    {
        string containerDir = Arg(args, 0) ?? throw new CliException("ngram eval needs a container");
        var packs = (OptionValue(args, "--pack") ?? throw new CliException("ngram eval needs --pack"))
            .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(NGramPack.Load).ToList();
        var lambdas = (OptionValue(args, "--lambda") ?? "0.1,0.2,0.3,0.5").Split(',')
            .Select(x => double.Parse(x, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
        int maxTokens = IntOption(args, "--max-tokens", 20000);
        int minContext = IntOption(args, "--min-context", 2);
        double halfCount = double.Parse(OptionValue(args, "--half-count") ?? "4", System.Globalization.CultureInfo.InvariantCulture);
        double modelConfident = double.Parse(OptionValue(args, "--model-confident") ?? "1", System.Globalization.CultureInfo.InvariantCulture);
        double packConfident = double.Parse(OptionValue(args, "--pack-confident") ?? "0", System.Globalization.CultureInfo.InvariantCulture);
        bool membersOnly = HasOption(args, "--members-only");
        string tokenizerDir = ResolveTokenizerDir(containerDir, args);
        var tokenizer = Tokenizer(tokenizerDir);
        var options = new HashSet<string> { "--pack", "--lambda", "--max-tokens", "--min-context", "--half-count", "--text", "--ext", "--tokenizer",
            "--model-confident", "--pack-confident" };
        var inputs = new List<string>();
        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--text")
            {
                continue;
            }
            if (options.Contains(args[i]))
            {
                i++;
            }
            else
            {
                inputs.Add(args[i]);
            }
        }
        var files = CorpusFiles(inputs, Extensions(args));

        using var container = Vindex3Container.Open(containerDir);
        using var store = container.CreateOperandStore();
        var plan = Planner.Plan(container, "target", store);
        using var session = new DecodeSession(plan, store);
        int vocab = plan.Output!.VocabSize;
        string fingerprint = TokenizerFingerprint(tokenizer, tokenizer.VocabSize);
        foreach (var p in packs.Where(p => p.Tokenizer != fingerprint))
        {
            Console.Error.WriteLine($"warning:   pack {p.Name} was built for tokenizer {p.Tokenizer}, this model's is {fingerprint}");
        }
        var pieces = new string?[vocab];
        string Piece(int id)
        {
            if (pieces[id] is null)
            {
                try
                {
                    pieces[id] = tokenizer.DecodeText(new[] { id });
                }
                catch (Amql.Hf.TokenizerException)
                {
                    pieces[id] = "";
                }
            }
            return pieces[id]!;
        }

        int k = lambdas.Length;
        double baseLoss = 0, baseMemberLoss = 0;
        long scored = 0, members = 0, spoke = 0, baseRight = 0, baseMemberRight = 0;
        var loss = new double[k];
        var memberLoss = new double[k];
        var right = new long[k];
        var memberRight = new long[k];
        var clock = Stopwatch.StartNew();
        foreach (string file in files)
        {
            if (scored >= maxTokens)
            {
                break;
            }
            var tokens = tokenizer.EncodeToIds(File.ReadAllText(file));
            if (tokens.Count < 2)
            {
                continue;
            }
            session.Reset();
            var row = session.Prefill(new[] { tokens[0] }).FirstRow().ToArray();
            var history = new List<int> { tokens[0] };
            for (int i = 1; i < tokens.Count && scored < maxTokens; i++)
            {
                int next = tokens[i];
                // The model's own distribution.
                float max = float.NegativeInfinity;
                int argmax = 0;
                for (int t = 0; t < row.Length; t++)
                {
                    if (row[t] > max)
                    {
                        max = row[t];
                        argmax = t;
                    }
                }
                double z = 0;
                foreach (float v in row)
                {
                    z += Math.Exp(v - max);
                }
                double pModel(int t) => Math.Exp(row[t] - max) / z;
                // A member name: after a '.' token, or a token that is the '.' and the name's start (".Length").
                string piece = Piece(next);
                bool member = piece.Length > 0 && char.IsLetter(piece[0]) && Piece(history[^1]).EndsWith('.')
                    || piece.Length > 1 && piece[0] == '.' && char.IsLetter(piece[1]);
                double pn = pModel(next);
                baseLoss -= Math.Log(Math.Max(pn, 1e-30));
                baseRight += argmax == next ? 1 : 0;
                if (member)
                {
                    members++;
                    baseMemberLoss -= Math.Log(Math.Max(pn, 1e-30));
                    baseMemberRight += argmax == next ? 1 : 0;
                }

                // Blended, exactly as NGramPrior does it, at each λ.
                (int Length, NGramPack.Entry Entry)? best = null;
                foreach (var pack in packs)
                {
                    if (pack.Lookup(history, minContext) is { } hit &&
                        (best is null || hit.Length > best.Value.Length || hit.Length == best.Value.Length && hit.Entry.Total > best.Value.Entry.Total))
                    {
                        best = hit;
                    }
                }
                // The same gates as NGramPrior: silent where the model is sure, or the pack is not.
                if (best is { } gate && (pModel(argmax) >= modelConfident || (double)gate.Entry.Counts[0] / gate.Entry.Total < packConfident
                    || membersOnly && !NGramPrior.AtMember(Piece, history, gate.Entry)))
                {
                    best = null;
                }
                spoke += best is null ? 0 : 1;
                for (int j = 0; j < k; j++)
                {
                    double p = pn;
                    int top = argmax;
                    if (best is { } found)
                    {
                        var e = found.Entry;
                        double lam = lambdas[j] * e.Total / (e.Total + halfCount);
                        double pack(int t)
                        {
                            int at = Array.IndexOf(e.Tokens, t);
                            return at < 0 ? 0 : (double)e.Counts[at] / e.Total;
                        }
                        p = (1 - lam) * pn + lam * pack(next);
                        // The blended argmax: the model's best, or a pack token that overtakes it.
                        double bestP = (1 - lam) * pModel(argmax) + lam * pack(argmax);
                        foreach (int t in e.Tokens)
                        {
                            double q = (1 - lam) * pModel(t) + lam * pack(t);
                            if (q > bestP)
                            {
                                bestP = q;
                                top = t;
                            }
                        }
                    }
                    loss[j] -= Math.Log(Math.Max(p, 1e-30));
                    right[j] += top == next ? 1 : 0;
                    if (member)
                    {
                        memberLoss[j] -= Math.Log(Math.Max(p, 1e-30));
                        memberRight[j] += top == next ? 1 : 0;
                    }
                }
                scored++;
                history.Add(next);
                row = session.Step(next).FirstRow().ToArray();
            }
        }

        Console.WriteLine($"eval:      {scored:N0} tokens from {files.Count} files, {members:N0} after a '.', pack spoke at {100.0 * spoke / Math.Max(1, scored):F1}% " +
                          $"({clock.Elapsed.TotalSeconds:F0}s, {session.Backend})");
        Console.WriteLine($"{"",-12} {"loss",8} {"top-1",8}   {"member loss",12} {"member top-1",13}");
        void Line(string label, double l, long r, double ml, long mr) =>
            Console.WriteLine($"{label,-12} {l / scored,8:F4} {100.0 * r / scored,7:F2}%   {ml / Math.Max(1, members),12:F4} {100.0 * mr / Math.Max(1, members),12:F2}%");
        Line("model", baseLoss, baseRight, baseMemberLoss, baseMemberRight);
        for (int j = 0; j < k; j++)
        {
            Line($"lambda {lambdas[j]:0.##}", loss[j], right[j], memberLoss[j], memberRight[j]);
        }
        return 0;
    }
}
