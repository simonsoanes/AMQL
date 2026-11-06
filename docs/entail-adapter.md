# Entail adapters: a program beside a model

A graft puts a program inside a model, which means rebuilding the model. An
adapter leaves the model alone. The program runs next to it as a blank model of
its own, reads the same tokens, and rules on the next token at the sampler.

The partner stays at whatever precision it was served at, and the program runs in
doubles whatever that is, so exactness no longer depends on the partner's weights.
The two share nothing but token ids, so what came from Entail and what came from
the model can always be told apart.

```bash
amql-cli entail adapter scripts/entail/dates.entail.txt --tokenizer export/qwen3.5-0.8b --out adapters/dates
```

```bash
amql-cli entail side <container> --tokenizer export/qwen3.5-0.8b --adapter adapters/dates,adapters/logic --prompt "The moon landing on 1969-07-20 was a"
```

`side` marks what a program said as `[[said]]`, and a token a rule pushed the
partner into as `{{steered}}`. `--chat` applies the chat template and `--think`
leaves the reasoning block open. `entail eval --adapter` runs the same pathway
over test cases and reports both per case, or over `--text` for prose loss.

## How it runs

Every token goes to every adapter, whoever chose it: the prompt, the chat
template, reasoning, and the partner's own choices. So an adapter always reads
exactly the text the partner reads, and it fires in reasoning as it does in an
answer. It only ever sees tokens, not what kind of text they are.

After each token an adapter returns a verdict on the next one:

| Kind | Effect |
| --- | --- |
| Say | This token comes next. It outranks every mask. |
| End | The answer is over: the program's output tokens are banned, except a space. |
| Only | Only these tokens may come next. |
| Forbid | These tokens may not come next. |

Within a program the first firing emit counts and every mask applies, as in the
interpreter. Adapters are joined in the order listed: the first `Say` wins, and
`Only` sets intersect. If two adapters demand different tokens, or the rules leave
nothing to choose, the verdict's `Conflict` flag is set; with nothing left the
partner goes free rather than the text stopping. Each token's provenance is
`Partner`, `Narrowed` (the partner's best permitted token, not its first choice)
or `Entail`.

`SideBySideSession` in `src/Amql.Inference/SideBySideSession.cs` is the whole
pathway: `Prefill`, `Step`, `Choose`. With no adapters it is the partner alone.

## Two forms of the same adapter

A compiled program is almost entirely zeros, because it is a block of exact
statements in a space sized for all of them at once. Across the four formal
programs:

| Program | Hidden | Layers | Dense size | Non-zero |
| --- | --- | --- | --- | --- |
| dates | 512 | 37 | 658 MB | 0.072% |
| bases | 1088 | 51 | 2.4 GB | 0.223% |
| JSON | 768 | 59 | 2.0 GB | 0.042% |
| logic | 2496 | 53 | 9.3 GB | 0.026% |
| all four | | | 14.4 GB | 0.063% (about 19 MB of index and value) |

So the default form doesn't hold weights at all. An adapter is a directory of
`adapter.json` and `program.entail.txt`, and `EntailSparseAdapter` compiles the
program at load and executes its statements directly, in doubles, one token at a
time. A head reads a fixed place in a ring of recent tokens instead of attending
to it, so memory is one row of lanes per token the program can see back, and
neither the length of the text nor attention margins come into it. This is valid
because a state is closed once read: a kept row holds the values a head would
have read at its layer.

`--dense` writes the same program as a blank model of f32 weights, for a runtime
that can only run a model. `--context` sets its context length, 8192 by default.
Both forms give the same verdicts, and `entail side` and `entail eval` pick the
form from what is in the directory.

Keep programs as separate adapters rather than merging them. Merged programs are
block-diagonal, so the dense form would hold about 25 GB and 6 billion weights a
token against 14.4 GB and 3.6 billion. Merging only shortens the serial chain,
and in the sparse form there is nothing to gain. Separate adapters can also be
switched on and off per request.

## The manifest

`adapter.json` joins a program to one partner:

```json
{
  "format": "amql-entail-adapter-v1",
  "program": "dates",
  "tokenizer": "<SHA-256 of tokenizer.json>",
  "rows": [[<partner token>, <adapter row>]],
  "outputs": [{"row": 0, "kind": "say", "tokens": [<partner token>]}]
}
```

A partner token the program doesn't know is row 0, which still takes up its place
in the text. Loading against a different `tokenizer.json` is refused, because ids
would silently mean other tokens. Nothing else about the partner matters: its
architecture, size and precision don't appear here.

## Results

Greedy decoding on the CPU beside Qwen 3.5 0.8B, whose weights are untouched, on
the cases in [entail-formal.md](entail-formal.md). All 1,176 replies from the sparse
form match the dense form exactly, and every one is the same with all four programs
running as with one alone.

| Right answers | Base | Graft | Dense | Sparse | Sparse, all four |
| --- | --- | --- | --- | --- | --- |
| dates | 54/250 | 250/250 | 250/250 | 250/250 | 250/250 |
| number bases | 41/315 | 315/315 | 315/315 | 315/315 | 315/315 |
| logic | 45/218 | 217/218 | 217/218 | 217/218 | 217/218 |
| JSON validity | 217/370 | 370/370 | 370/370 | 370/370 | 370/370 |
| JSON guard, texts still valid | 38/60 | 60/60 | 60/60 | 60/60 | |

| Beside the partner | |
| --- | --- |
| prose loss, adapters on and off | 3.55195 and 3.55195, identical |
| prose positions where a program spoke | 0 of 767 |
| cases no program touched, same as base | 19 of 19 |
| chat replies changed by all four | 1 of 40 |
| size of all four | 14.4 GB dense, about 1 MB sparse |
| time in all four adapters | 5.2 ms a token |

The graft moves prose loss slightly, to 3.55210, because widening a model changes
its rounding, and that can flip a near tie: asked to continue `The sky is`, the
graft says `blue` where the base and the adapter both say `green`. An adapter can't
do this, because the partner runs exactly as it did.

The one logic miss is the same in every form, and is covered in
[entail-formal.md](entail-formal.md).

The one chat reply that changed is `0xff in decimal is 255.`, where the model would
have written `**255**`. The number is the same but the program's token replaced the
bold, so a program that speaks also sets the formatting of what it says. No chat
answer improved: these are questions the model already gets right.

A program also speaks on text the partner wanders into. Past the end of a test
answer the base wrote `100 in base 2 is 4`; beside the adapter that line reads
`100 in base 2 is 1100100`.

The time was measured at below-normal priority, against a partner on the same CPU
at 1,382 ms a token, so read it as an order of magnitude. Each program compiles at load, in about 0.7 s for JSON.

Programs fire inside reasoning too. Asked `What day of the week was 15 March 2024?`
with `--think`, the model wrote `January 1, 2024 was a` twice while working, and the
dates program supplied ` Monday` both times, correctly. That is 2 tokens of 500, and
it shows the limit: the program fixed the fact the model stated, but the model then
counted 73 days to 15 March instead of 74 and ran out of tokens before answering. A
program only helps with a sentence it can read.

## What a production server must do

- **Same tokens.** Feed every token to every adapter: prompt, template, reasoning
  and generated tokens, in order, including tokens a program chose.
- **Same tokenizer.** Build the adapter against the served model's `tokenizer.json`.
- **One reader per request.** `Sibling()` gives a separate reader of a program.
  Its state is small and shares the compiled statements.
- **Apply the verdict before sampling.** `Choose` here is greedy. A sampling
  server should apply `Say`, `Allowed` and `Banned` to the distribution instead.
- **Roll back with speculative decoding.** Not built. A drafted token an adapter
  has read must be undone when rejected, and an adapter can only `Reset` today.
  The ring would need a saved position for it.

## Limits

- An adapter rules on tokens, not meaning, so it adds exact answers and valid
  form, not reasoning. A partner that sets up the wrong sum gets the wrong sum's
  exact answer.
- A program must read the text the way the partner's tokenizer cuts it. Each new
  tokenizer needs its own manifest, though the program itself is unchanged.
- Say outranks the partner absolutely. A rule that fires where it shouldn't is
  not softened by the partner disagreeing, and what it says replaces the partner's
  formatting as well as its answer.
