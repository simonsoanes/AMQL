# N-gram packs: a domain's next-token statistics as a logit prior

An n-gram pack is a corpus's next-token statistics, counted in the model's own tokens: for every context of 1 to 3
tokens seen at least twice, its 8 most frequent continuations and their counts. Blended into the model's logits at
the sampler, kNN-LM style, it is meant to supply what a small model cannot recall, such as the exact spelling of a
rare member name. This page is how packs are built and served, and what they did and did not do for Qwen 3.5 0.8B
writing C#.

**Result in one line:** a .NET pack lowers the model's invented-API errors, but it costs more builds than it saves,
at every setting tried. N-grams are lexical, and whether `x.Member` exists depends on the type of `x`, so the right
tool for API names is a type-aware symbol pack; an n-gram's frequencies can then rank the members that pack allows.

## Building and inspecting a pack

```bash
# the SDK's reference docs, written as calls: Directory.GetDirectories(path, searchPattern);
python scripts/ngram/make_dotnet_corpus.py "C:/Program Files/dotnet/packs/Microsoft.NETCore.App.Ref/11.0.0/ref/net11.0" corpus/docs
# source code with comments removed and generated or very large files left out
python scripts/ngram/make_code_corpus.py corpus/code src <other C# repos>
amql-cli ngram build --tokenizer <model> --out dotnet.ngram --name dotnet --order 4 \
    --keywords "C#,.NET,dotnet,csproj,.cs" --ext .txt corpus/docs corpus/code
amql-cli ngram info dotnet.ngram
amql-cli ngram eval <model> --pack dotnet.ngram --lambda 0.1,0.3 --ext .cs <held-out C#>
```

`ngram build` hashes every (context, next) pair, sorts and counts them, drops contexts seen fewer than
`--min-count` times and keeps each context's `--top-k` continuations: 5.5M tokens become 568k contexts and a 16 MB
file in 7 s. A pack records its tokenizer (vocabulary size and a hash of its pieces), and `ngram eval` warns when a
model's differs. `ngram eval` reads held-out text token by token on the GPU and reports the model's loss and top-1
accuracy alone and blended at each λ, overall and at member names (the token after a `.`, or a `.Name` token).

## The blend

`NGramPrior` takes, among the request's packs, the one that knows the longest context ending the text (at least
`MinContext` tokens; ties to more evidence), and mixes its distribution in:
p = (1 − λ)·p_model + λ·p_pack over the pack's tokens, with λ = λ_max · total / (total + 4), so a context seen
often speaks louder. It is applied in logit space: only the pack's tokens move, to
log(exp(logit) + λ/(1 − λ) · p_pack · Z), which leaves every other token's relative odds as they were. Gates keep it
silent where the model is already sure (`ModelConfident`), where the pack's own top continuation has little of the
context's count (`PackConfident`), outside member names (`MembersOnly`), and outside code (`Scope`, by default only
inside a fenced code block past its opening line or inside a tool call's value).

## Serving

```bash
amql-server ... --ngram-pack dotnet.ngram [--ngram-lambda 0.3] [--ngram-min-context 3] \
    [--ngram-model-confident 0.5] [--ngram-pack-confident 0.5] [--ngram-members-only]
```

**A container's own packs.** Each model's container may hold packs in an `ngram/` folder beside its weights. The
server loads them at startup and keeps the folder current while it runs: a pack is loaded when its file is added
or replaced and dropped when the file is deleted, with no restart (copy a pack in and the next request can use it).
A file still being written is retried until it reads whole; a pack counted in another tokenizer, or a file that is
not a pack, is refused with a log line; requests see a consistent set, never a pack half-swapped. The folder need not
exist at startup. Container packs sit beside `--ngram-pack` ones (a container pack of the same name wins), and
`/v1/models` lists each chat model's packs as `ngram_packs`.

A request chooses packs with `"ngram_packs": ["dotnet"]` (`[]` for none), else the model's own (`"ngram_packs"` in
`--config`), else every loaded pack one of whose keywords appears in the current task (the user's latest message and
the tool results since). `"ngram_lambda"` and `"ngram_scope"` (`code`, `members`, `all`) override per request. The
blend is applied before structured output and the tool-call grammar, so a constrained step chooses among blended
probabilities. Given the results below, serve packs with `--ngram-lambda 0` (off unless a request sets
`ngram_lambda`) until something better is found.

## What was measured

**Held-out loss.** Teacher-forced over 20,000 tokens of C# from three repositories the pack never saw
(LetsEncrypt-Win, Discord-Bot, MultiplayerExtensions), the pack (docs, AMQL's source, OpenTK, Accord.NET; 26M
tokens) raised the loss at every λ (1.2726 alone, 1.2800 at λ 0.05, 1.3686 at 0.3) and moved top-1 accuracy by at
most 0.3 points. With all three gates it was neutral (1.3003 against 1.3009). An earlier run on AMQL's own tests
showed a 5% gain, but that was the style of AMQL's source in the pack: held out properly, it disappeared.

**Compile benchmark.** `scripts/ngram/compile_bench.py`: 24 small console tasks that lean on the base class library
(file system, hashing, JSON, regex, HttpClient, processes, zip, drives), the model writes one Program.cs, and
`dotnet build` judges it. "Invented" counts the errors that mean an API does not exist (CS0117, CS0103, CS0246,
CS1061, CS0234). Qwen 3.5 0.8B, no thinking:

| Pack and blend | Greedy, 24 | Invented | Sampled 0.6/0.95/20, 72 | Invented |
|---|---|---|---|---|
| none | 9 built | 19 | **19 built** | 41 |
| first corpus (docs as declarations, comments in), λ 0.3 | 5 | 6 | | |
| first corpus, λ 0.6 | 0 | 0 | | |
| cleaned corpus, λ 0.2 | **10** | 14 | 12 | 40 |
| cleaned, λ 0.4 | 5 | 14 | | |
| cleaned, gated (model < 0.5, pack ≥ 0.5, context ≥ 3), λ 0.3 | 9 | 15 | | |
| cleaned, code only, λ 0.2 | | | 12 | 40 |
| cleaned, code only, λ 0.4 | | | 1 | 9 |
| cleaned, member names only, λ 0.5 | | | 16 | 51 |
| cleaned, member names only, λ 0.8 | | | 13 | 60 |

What broke, in order of discovery:

1. **The corpus's form becomes the output's form.** Docs written as declarations (`Directory.GetDirectories(string
   path)`) taught type names inside calls; code full of doc comments made the model write `/// <param name=…>` until
   the token limit, never closing its code block. Docs written as calls and code without comments fixed both.
2. **A code pack in prose.** Blended into the fence line, it turned ```` ```csharp ```` into ```` ```csharp.tm ````,
   so the code block was never found. The pack now speaks only in code.
3. **N-grams do not know types.** Restricted to member names, the pack made invented-API errors more frequent, not
   less: after `line.` it offers the members that follow `line.` in other codebases (`.Groups`, `.Key`), which a
   string does not have. Lower λ removes the effect along with the harm; higher λ breaks syntax.

## Where this leaves it

The mechanism (packs, blending, gates, scope, per-request choice) works and is cheap, and invented APIs do respond
to it. But on this model and this task it lowers build success at every setting, so it stays off by default. What
the results point to instead:

- **Type-aware member choice.** A symbol pack resolves the receiver's type (static access, `new T(...)`, explicitly
  typed locals, `using`s) and allows only members that exist on it. That is what the member-only runs lacked.
- **N-gram frequencies inside the symbol pack.** Among the members a type allows, how often real code uses each
  (`File.ReadAllText` far more than `File.ReadAllBytesAsync`) is a good ranking. The counting code here can produce
  it: count `Type.Member` pairs instead of raw token contexts.
- **Measure in generation, not teacher forcing.** Held-out loss said "neutral"; the compile benchmark said
  "harmful". The benchmark is the one that matches use.
