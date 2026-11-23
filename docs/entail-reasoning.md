# Reasoning: making a 0.8B model finish what it solves

This experiment asks which general traits stop Qwen 3.5 0.8B from solving problems it has
never seen, and whether any can be supplied generically, as an Entail program beside the
unchanged model. The answer so far: the model can usually work a problem out, but it can't
tell when it is done. It doubts its answer until it repeats itself and runs out of budget.
A 16-neuron program that notices the repetition and closes the reasoning block turns
reasoning from a loss into a gain. On problems with seeds never used in development it lifts
the model from 0% to 58%, against 46% without reasoning.

Everything here runs on Qwen 3.5 0.8B, greedy, on the GPU, with every program as a sparse
adapter (see [entail-adapter.md](entail-adapter.md)).

## Results in brief

| Question | Finding |
|---|---|
| Does reasoning help the 0.8B model? | Not as it stands. With thinking on and 2,048 tokens it answers 1 of 80 generated problems, against 37 of 80 with thinking off. Qwen's recommended sampling (temperature 0.6, top-p 0.95, top-k 20) gives 0 of 80. |
| Why? | It does not stop. 72 of 80 greedy replies end in a verbatim loop and 7 more run out of budget. In 63 of them the right number had already been written, at a median of 260 tokens. The loop starts at a median of 945. |
| Does the model know when it is done? | No, unlike the tool-call decision ([entail-strategies.md](entail-strategies.md)). In 74 of 80 replies p(`</think>`) never reaches 1% at any step, so no threshold on it can act. |
| Does breaking the loop help? | No. Ruling out the copied token (`unloop.entail.txt`) gives 2 of 80: the model paraphrases its doubts instead and runs out of budget in 75. |
| What does? | Stopping when it circles. `stop-circling.entail.txt` closes the block at the next line break once the last 16 tokens have been written three times before (or after 1,536 tokens). That gives **46/80 (58%)** on the development seed and **46/80 (58%)** on a held-out seed, winning 45 and 46 cases and losing none. |
| Does exact arithmetic add to it? | Not yet. With `mathematics.entail.txt` beside it, 44/80: in LaTeX such as `$3 \times 10 + 2 =$` the program reads only `10 + 2 =` and asserts 12. It needs to stay silent after an operator it doesn't know. |

## The benchmark: `entail bench`

Problems a model has not been trained on have to be new, so `entail bench` generates them
from a seed (`ProblemSuite` in `src/Amql.Inference/Reasoning`). Each answer is a whole
number, so no judge is needed. There are eight families, each testing a different trait:

| Family | Trait | Example |
|---|---|---|
| `operator` | apply a rule given only in the prompt | `a # b = a * a - b`. What is `(8 # 3) # 5`? |
| `state` | track values through steps | `x = 1, y = 3, z = 2. x = y. Swap y and z. ...` |
| `plan` | find a shortest route | moves of +5 or -4: fewest to reach 8? |
| `sequence` | induce a rule from examples | 4, 9, 19, 39, 79, ...? |
| `count` | count under constraints | 1 to 78, divisible by 3 but not by 5? |
| `word` | a multi-step word problem | gives some away, then 3 times what is left plus 2 |
| `order` | deduce from facts given out of order | how many are taller than Eli? |
| `counterfactual` | follow a rule that contradicts the world | a week of 9 days: which day in 15 days? |

```bash
amql-cli entail bench <container> --per 10 --seed 7 --report base.json
```

```bash
amql-cli entail bench <container> --per 10 --seed 7 --adapter adapters/stop-circling --report stop.json --compare base.json
```

Only a committed answer counts: the last number after `</think>`, or in `\boxed{}`. A
right number written only while thinking is reported as `uncommitted`, because it measures a
failure to commit rather than a failure to solve. Each reply is classed as `correct`,
`wrong`, `none` (no number), `looped` (out of budget, with a run of 24 tokens seen three
times) or `unfinished`. `--compare` matches cases by prompt and prints what was won and lost.
`--watch TOKEN` records a token's probability at every step, so a threshold rule can be
tuned without rerunning; `--temperature`, `--top-p`, `--top-k` and `--sample-seed` sample,
with each case drawing from its own seed; `--no-think` turns reasoning off; `--cases`
reads a file instead of generating.

## Where the model fails

On the development seed (7), 80 problems, 2,048 tokens:

| Run | Correct | Looped | Unfinished | Right number seen while thinking |
|---|---|---|---|---|
| thinking off | 37 | | | |
| thinking, greedy | 1 | 72 | 7 | 63 |
| thinking, sampled as Qwen recommends | 0 | 59 | 21 | 67 |

A typical reply works the problem, writes "The answer is 175", and goes on: "Wait, I should
check if the prompt implies a specific format... Is it possible the user meant $a \# b = a
\cdot b + a - b$? Yes. Is it possible the user meant..." The doubt is the loop. Sampling
does not remove it: it varies the wording of the doubt.

### The model gives no sign it is ready

The tool-call work found that the model knows when to call and is only miscalibrated, so a
threshold on its own probability fixed it. The same idea applied here would be `prefer
"</think>" at P while thinking`. The watch trace rules it out:

| Threshold P | Replies where p(`</think>`) ever reaches P |
|---|---|
| 0.01 | 6 of 80 |
| 0.05 | 2 of 80 |
| 0.3 | 1 of 80 |

So the evidence has to come from somewhere other than the model's view of the next token.

### Repetition is the sign

A run of text written for the third or fourth time means the reasoning has stopped
producing anything new. What it produced before is the best it has. To act on that, Entail
gained a primitive that matches the text against itself:

```
repeat DEST over N [times K]       # DEST = 1 when the last N tokens were written K times before
forbid repeat N when LANE          # rule out the token that continued them last time
```

This is what an induction head computes. It reads the whole text, so it runs in the
interpreter, the stream runner and the sparse adapter, and `compile` and `graft` refuse it,
as they refuse `carry`. See [entail-language.md](entail-language.md#matching-the-text-against-itself-repeat).

The run length and count were chosen on the baseline's own replies, for where they would
first fire against when the right number was first written:

| Span, times | Replies it fires in | After the answer was written | Before it | Median position |
|---|---|---|---|---|
| 12, 1 | 80 | 36 | 28 | 202 |
| 12, 2 | 80 | 47 | 17 | 483 |
| 16, 3 | 77 | 60 | 1 | 929 |
| 24, 2 | 72 | 56 | 2 | 954 |
| 32, 3 | 65 | 51 | 0 | 1320 |

Short spans fire on the legitimate repetition of structured reasoning ("Substitute $y$
(which is 2) into..."). 16 tokens seen three times fires late enough to leave the work
alone and early enough to save the answer.

## The program

[`stop-circling.entail.txt`](../scripts/entail/reasoning/stop-circling.entail.txt) is a
fold that knows the reasoning block's markers, line breaks and a count, and matches the
text against itself. When the last 16 tokens have been written three times before, or after
1,536 tokens of reasoning, it latches, and it says `</think>` after the next line break. It
has 12 lanes, 16 neurons and no word lists.

```bash
amql-cli entail adapter scripts/entail/reasoning/stop-circling.entail.txt --tokenizer <dir> --out adapters/stop-circling
```

The line break matters. A first version closed the block the moment the repetition was
seen, mid-sentence, with a span of 12 seen twice. It reached 24 of 80: the model carried the
broken sentence on after `</think>` as if still reasoning, and the short span fired before
the work was done.

| Program | Development seed 7 | Held-out seed 11 |
|---|---|---|
| none, thinking off | 37 (46%) | 37 (46%) |
| none, thinking on | 1 (1%) | 0 (0%) |
| `unloop` (rule the copy out) | 2 (3%) | |
| `stop-circling`, span 12 × 2, closing anywhere | 24 (30%) | |
| **`stop-circling`, span 16 × 3, closing at a line break** | **46 (58%)** | **46 (58%)** |
| `stop-circling` + `mathematics` | | 44 (55%) |

Per family on the held-out seed, against thinking off:

| Family | Thinking off | Thinking + stop-circling |
|---|---|---|
| operator | 10 | 9 |
| state | 7 | **10** |
| plan | 0 | 2 |
| sequence | 6 | 6 |
| count | 0 | 2 |
| word | 7 | **9** |
| order | 6 | 6 |
| counterfactual | 1 | 2 |

Reasoning now pays where tracking several steps matters (state, word problems) and costs
nothing elsewhere, except one operator case. Shortest moves, constrained counting and the
counterfactual week remain hard: in most of those failures the right method is never found,
which a stopping rule cannot supply.

## Acting with tools: does it talk instead of writing the file?

In real use (a desktop agent harness), the 0.8B model was asked to write a .NET disk-space
viewer with `create_file` on offer and answered with a code block in chat. Pressed, it
wrote the file, invented APIs (`System.Graphics`, `WindowsFormWindow`), and then ran the
same failing command, `dotnet /b ...`, a dozen times. That session holds the same two
failures as reasoning: not finishing, and copying itself.

`make_coding_cases.py` writes 20 coding requests, 4 that need another tool and 4 that need
none, rendered with the model's own template and 4, 12 or 42 tools on offer. The 12-tool set
is named and described as a desktop harness's tools. The 42-tool set adds 30 assistant
tools, so the request sits behind 2,836 tokens of definitions. `entail bench --score tools
--simulate 6` answers each call with a plausible, uneventful result (writes succeed, reads
find nothing, commands exit 0), lets the model carry on for up to 6 calls, and scores
whether it ever calls the expected tool, whether that call closes, and the code it wrote.
Nothing is executed.

| Setting (20 write tasks) | Calls the write tool | Talks instead | Call closes (file written) | Code with placeholders |
|---|---|---|---|---|
| 4 tools, agent prompt, one turn | 19 | 0 | | 1 |
| 4 tools, "You are a helpful assistant", one turn | 16 | 4 | | 0 |
| 12 harness tools, one turn | 9 (11 ask for the directory first) | 0 | | 1 |
| 12 harness tools, simulated | 19 | 0 | 13 | 1 |
| 12 harness tools, simulated, sampled ×3 seeds | 19, 19, 20 | 0 | | 1 to 3 |
| 12 harness tools, thinking, sampled ×3 seeds | 20, 20, 19 | 0 | | 0 |
| 42 tools, simulated | 20 | 0 | 14 | 1 |
| 42 tools, thinking, simulated | 19 | 0 | 14 | 0 |
| 12 harness tools, simulated, **+ `no-copy`** | 19 | 0 | **15** | 1 |

So in these prompts the model does not prefer prose to a call. With a harness's tools it
looks around first (`get_current_directory`, as it did in real use), then writes. The
failure that does reproduce is **a file that is never written**: in 6 or 7 of 20, the code
degenerates into a copy loop (the same helper defined again and again, or a name grown word
by word) and the budget runs out with the call still open. `no-copy.entail.txt` rules out the
continuation of a 16-token run written three times, outside reasoning. That rescued 2 of
those files and changed nothing else. Placeholders (`TODO`, `...`, `your code here`, stand-in
values such as `C:\Users\YourName`) appear in at most 3 of 20 replies, so on these tasks
"lazy" code is mostly code that loops or calls APIs that don't exist, not code left blank.

What the real session had that these prompts lack is not known yet: its actual system prompt
and tool schemas, the server's template and sampling, and the conversation before the
request. Reproducing the prose reply needs those.

## What this says about small models

- **The weakest trait is not solving but finishing.** The 0.8B model writes the right
  answer in its reasoning about four times as often as it commits to one. Its reasoning was
  trained on larger models' traces, where checking pays; at this size checking finds nothing
  new and does not stop.
- **Not every decision has evidence in the next token.** The tool-call decision did; this
  one doesn't. The evidence is in the shape of the text, which a fixed-offset head cannot
  see and a match against the whole text can.
- **Breaking a loop and leaving it are different things.** Forbidding the copied token
  changes the words and keeps the doubt.

## Limits

- 80 problems per seed, on one model, greedy. The two seeds agree to the case count, which
  is encouraging, but each family has only 10 problems.
- The span, the count and the budget were chosen on the development seed and only checked
  on one held-out seed.
- `repeat` keeps one id and one table entry per token for each span. That is not the fixed
  memory of a `carry` fold.
- `mathematics.entail.txt` misreads chained LaTeX expressions and should not run beside
  reasoning until it stays silent after an operator it doesn't know.

## Reproducing

```bash
amql-cli entail adapter scripts/entail/reasoning/stop-circling.entail.txt --tokenizer <dir> --out adapters/stop-circling
```

```bash
amql-cli entail bench <container> --per 10 --seed 11 --no-think --report nothink.json
```

```bash
amql-cli entail bench <container> --per 10 --seed 11 --report think.json --watch "</think>"
```

```bash
amql-cli entail bench <container> --per 10 --seed 11 --adapter adapters/stop-circling --report stop.json --compare think.json
```
