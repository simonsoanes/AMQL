# Strategies: tool calling and long context on Qwen 3.5 0.8B

Knowledge can be given to a small model on demand, in its context. This experiment asks
whether strategies can be handled the same way: written down as programs rather than
trained in. It covers two strategies:

- **Reading a context of any length**, with every part of it actually read.
- **Calling tools**: given request A, the model calls tool B with arguments C.

The method is the one Entail already uses for mathematics: train the behaviour, look at
what training changed, and write the result as a program. That needed three new pieces:
a LoRA trainer in AMQL, a translator from a LoRA to Entail, and a tracer for the tool
decisions. Everything below was run on Qwen 3.5 0.8B on the CPU, in float32.

## Results in brief

| Question | Finding |
|---|---|
| Where does the 0.8B model fail at tools? | The first token: calling now or talking first. Talking first sometimes ends in the call and sometimes in a refusal. Once it opens `<tool_call>` it picks the right tool and copies the right arguments, even for a renamed tool. It also lets the tool that best matches the request outvote a policy. |
| What drives that decision? | The template's instruction lines and the request's wording. Removing the relevant tool from the list barely changes it. The `<IMPORTANT>` block, which says to answer normally when no function applies, pushes against calling. |
| What does one training example teach? | A bias, not a rule. The LoRA learned to suppress the openers "To" and "I" on the reply's first token whatever the request, so the model then calls a tool for everything. |
| Is the tool the one nearest the request's intent? | Yes, but in layers 7 to 15, not in the embedding: there the request is nearest the chosen tool's description in 9 or 10 cases of 10, against 6 of 10 for static embeddings. Middle layers attend to that description, a layer-19 head copies its name, and argument values are copied from the request. |
| Can a program read an unbounded context? | Yes, as a fold. With the new `carry` statement, an Entail program read 2 MB and 8 MB texts in 2,640 bytes of state and found the one record that decided the answer. |
| A generic fix? | Split the work: the model supplies meaning (which tool is near, where a value lies), and a program supplies the strategy (when to call, what takes precedence, what to carry). A first call gate with a hand-written lexicon cost as much as it gained. |
| A generic fix for missed calls? | Yes. The model's first-token probability of `<tool_call>` already says when a call is due. It is only below prose's. A 10-neuron program, `decide "<tool_call>" at 2 times prior`, calls when the latest message at least doubles that probability over the same conversation with the message blanked. With two template lines removed, the clean set goes from 23 to 36 calls of 36 with no new false call. On 120 real policy conversations, calls rise from 64 to 94 and right tools from 46 to 70. No word lists are involved. |
| What fails next? | Choosing the tool: a tool listed late loses to one listed early. Restating the tools near the reply fixes that, but it blurs the call decision. |

## What was built

| Piece | Command | What it does |
|---|---|---|
| LoRA trainer | `amql-cli lora-train` | Trains LoRA factors on FFN down-projections without an autograd engine, and monitors itself as it goes. |
| LoRA to Entail | `amql-cli entail from-lora` | Writes each rank component of a LoRA as an Entail state, with where it fires, what it says and what in the context it depends on. |
| Tool tracer | `amql-cli tool-trace` | For the model's own reply, traces the call, tool and argument decisions: attention over the prompt's parts, and request-to-tool similarity. |
| Recurrent registers | `carry DEST from SOURCE` | A lane that starts each token holding another lane's value from the token before. A fold over the text. |
| Streaming | `entail run --file` | Runs a program over a file of any size in fixed memory. |
| Host-weighed outputs | `prefer`, `decide` | Say a token when the host's own model gives it enough probability, absolutely or relative to its prior for the same conversation with the latest message blanked. |
| The call program | `call-when-likely.entail.txt` | The tool-call decision as a 10-neuron fold plus one `decide`. |
| Decision probes | `entail eval --watch --prefix --logprobs` | First-token probabilities, forced openings and per-token probabilities. |
| Cases | `scripts/entail/toolcall/` | The synthetic and real cases, anchors, scorers, prompt variants, trace summary and the call programs. |

### Training without autograd

The forward pass is AMQL's own runtime. The backward pass runs exactly through the output
head and the final RMSNorm, which gives the loss gradient on the last residual. For a LoRA
on the **last** layer's down-projection that is the whole chain, so its gradient is exact.
A finite-difference check confirms it, and runs at the first step of every training run:

| Check, layer 23 | Predicted | Measured | Ratio |
|---|---|---|---|
| random direction | -0.008114 | -0.008113 | 1.0002 |
| along the gradient | 0.82811 | 0.82800 | 1.0001 |

For an earlier layer the gradient follows the direct residual path only, ignoring how the
layers above react to the change. The same check measures how far that is off, rather than
assuming. The loss the trainer reports and acts on always comes from the exact forward pass.

Other features of the trainer:

- **Cached prefix.** Layers below the first LoRA layer never change, so they run once per
  example, and each step runs only the layers above. A step on the last layer takes about
  1 s for one 680-token example.
- **Spike guard.** A step that raises the loss by more than half is undone and the learning
  rate halved.
- **Early stop.** Training stops once the loss is below target and every training example
  decodes exactly, which an evaluation confirms.
- **Anchors.** Unlabelled prompts that the base model answers itself at the start of
  training. Its own replies become extra targets, so the adapter learns the new behaviour
  without forgetting the old. They need no human labels.
- **Log.** `train.log.jsonl` has one line per step, check, anchor and evaluation.
- **On the GPU.** The LoRA runs as its own f32 path beside the frozen down-projection
  (`GenericRuntime.FfnAdapter`) and is never added into the weight. So with CUDA on (no
  `--cpu`), the managed path's GEMMs run on the device, and their f16 weight copies stay
  valid. A 4-step run gives the same losses on both devices (0.2362, 0.1369, 0.0263, 0.0044),
  with gradient checks within 0.14% on the GPU and 0.06% on the CPU. It is no faster yet: the
  prefix takes 55 s and a step 1 s either way, because attention, the Gated DeltaNet and the
  head backward still run on the CPU. Making it faster needs the device engine to take the
  same seam: hand out the FFN input at the LoRA layers and add the LoRA's output, keeping
  the final residual for the backward pass.

```bash
amql-cli --cpu lora-train <container> --tokenizer <dir> --data train.jsonl --eval eval.jsonl \
    --anchors anchors.jsonl --layers 23 --rank 8 --lr 2e-3 --steps 60 --out <lora-dir>
```

The output is an `amql-lora-v1` adapter, the format `save-lora` writes, plus a
`patch.safetensors` for `--patch`, the log and a `report.json` with every reply before
and after.

## The cases

`scripts/entail/toolcall/make_cases.py` renders every case with the model's own chat
template: tools in the system turn, thinking off. The completion is in Qwen's XML call
format. There are four sets:

| Set | Cases | Tests |
|---|---|---|
| Selection (`w:`) | 9 | The request names the need: weather, timer, currency, flights, email. Includes a reordered tool list and a renamed tool. |
| Need (`n:`) | 10 | The request implies a tool but doesn't name it ("flying to Oslo, do I need a coat?"), plus 3 that need none. |
| Policy (`p:`) | 7 | A system policy says to verify the customer before a cancellation, so the right first call is not the one the request names. |
| Anchors | 8 | Unlabelled prompts, half needing a tool. Not used for scoring. |

`score.py` scores by meaning: did the reply open with a call (or not, when none is
needed), did it name the right tool, and are the expected arguments there. Extra optional
arguments are allowed.

## Where the base model fails

| Set | Decision right | Tool right | Call fully right |
|---|---|---|---|
| Selection, 9 | 9 | 9 | 8 |
| Need, 10 | 8 | 8 | 8 |
| Policy, 7 | 5 | 1 | 1 |

Selection is nearly perfect. The one miss truncates an email body. The failures are of two
kinds:

- **No call when one is needed.** For "I'm flying to Oslo tomorrow. Do I need to pack a
  warm coat?" the model replied "To determine if you need a warm coat, I would need to know
  the current weather conditions in Oslo. I can check…", and stopped there. For "Let
  sarah@example.com know the meeting has moved to 3pm" it replied "I cannot send an email
  to sarah@example.com", with `send_email` on offer.
- **The surface match beats the situation.** In every policy case the model called the
  tool named by the request's verb (`cancel_order`, `update_address`, `delete_account`)
  rather than `send_verification_code`, which the policy requires first. The one case it
  got right in substance, it got right in words first: "I will verify the customer's
  email address first before proceeding with the cancellation", then the right call. The
  scorer counts that as a miss, since the reply didn't open with the call, but it shows
  the plan is there when the model says it.

This matches what was observed in use: the 0.8B model is good at calling once it calls,
but it often doesn't.

### What the call decision depends on

`entail from-lora --occlude` removes one part of the prompt at a time and measures the
probability of the reply's first token again. On the Oslo request, where the base model
gives `<tool_call>` a probability of 0.047:

| Removed | p(`<tool_call>`) |
|---|---|
| nothing | 0.047 |
| the line "If you choose to call a function ONLY reply in the following format…" | **0.000** |
| the `<IMPORTANT>` block | **0.202** |
| the word "flying" | 0.251 |
| the word "pack" | 0.170 |
| the `set_timer` tool | 0.160 |
| the `get_weather` tool, the one needed | **0.019** |
| the word "Oslo" | 0.023 |

The decision rests on the template's format line. The `<IMPORTANT>` block, which tells the
model to "answer the question like normal" when no function applies, suppresses calling by
a factor of four. Some words argue for prose ("flying", "pack" suggest a travel chat) and
some for the tool ("Oslo"). The needed tool itself is a weak voice: removing it lowers the
probability from 0.047 to 0.019. On the decision row, the model is hardly reading the
tools at all.

## One example, one layer: what training learns

Run 1 trains a rank-8 LoRA on layer 23 with the single Oslo example and nothing else.

| Step | Loss | Note |
|---|---|---|
| 1 | 0.2362 | gradient checks pass |
| 2 | 0.1369 | gradient clipped |
| 3 | 0.0263 | clipped |
| 4 | 0.0044 | stopped: target loss, and the example decodes exactly |

It fixes the example, and breaks most of what was right:

| Set | Before | After run 1 |
|---|---|---|
| Selection, fully right | 8/9 | 1/9 |
| Need, decision right | 8/10 | 7/10 |
| Need, fully right | 8/10 | 4/10 |
| Policy, fully right | 1/7 | 2/7 |

"What is the capital of France?" now gets `get_weather(city=Paris)`. Currency and email
requests get `get_weather` too, and every call stops after its first parameter.

### The LoRA, as Entail

`entail from-lora` writes each of the 8 rank components as a state: the token pairs it
fires on, and what its column says when read through the final norm and the output head.
For the last layer that reading is exact. From `run1-L23/lora.entail.txt`:

```
# L23 component 4: peak |u| 3.760, mean |u| 1.020 over 209 rows
# column promotes "<tool_call>" ".\"</" "')</" "'</" "\"</" ").</"
# column demotes  "To" "I" " To" " to" " I" "to"
...
# L23 component 2: peak |u| 3.572, mean |u| 1.282 over 209 rows
# column promotes "To" " To" "I" " to" "to" "Yes"
# column demotes  "<tool_call>" "()<" "<" " `<" ".\"<" "教于"
state lora_L23_2
entail lora_L23_2 = -7 when prev == 248069, sym == 271   # "</think>" "

" x7 mean u -3.078
```

Six of the eight components carry one axis: `<tool_call>` against "To", "I" and "Yes".
Those are the openers of the base model's wrong replies ("**To** determine if you need a
warm coat…", "**I** cannot send an email…"). Component 0 promotes `get` after
`<function`, which is where the `get_weather` bias comes from. Components 6 and 7 trade
`<` against `</`, opening another parameter against closing the function. The one
example had a single parameter, so after it the adapter prefers to close.

The activations on the reply's first row show that the decision is no longer a decision:

| Case | Base says | With LoRA | Components 0 to 7 |
|---|---|---|---|
| Oslo (trained) | To | `<tool_call>` | -1.46 -2.84 -3.57 -2.22 3.76 3.01 1.01 -2.70 |
| capital of France | The | `<tool_call>` | -1.17 -2.71 -2.84 -1.64 2.49 2.39 1.00 -2.89 |
| email sarah | I | `<tool_call>` | -0.90 -1.97 -3.26 -2.00 3.22 2.25 0.25 -3.01 |
| one line about the sea | The | `<tool_call>` | -1.32 -2.73 -2.62 -1.95 3.17 1.78 0.67 -2.27 |

The values barely move between a request that needs a tool and one that doesn't. The
occlusion study agrees: with the LoRA, p(`<tool_call>`) on the Oslo prompt is 1.000 with
any single part removed, including every tool. The layer-23 LoRA reads the local tokens
`</think>` `

`, which are the same for every prompt, and raises the call. That is the
bias the base model lacked, with none of the condition. The Entail shows it without
needing a held-out set: no rule mentions anything in the request.

This is the general lesson for training a tiny model on a few examples. The cheapest
change that fits an example is a prior on its first token. Everything else (choosing the
tool, copying the arguments, knowing when not to call) has to be supplied by more data,
by anchors, or by a program.

## How request A becomes tool B with arguments C

`amql-cli tool-trace` lets the base model write its own reply, then traces three decisions
in it. At each one it sums the softmax layers' attention over the prompt's parts, using the
6 full-attention layers (3, 7, 11, 15, 19, 23); the other 18 are linear attention and have
no weights to read. Separately, it pools the request and each tool's description at several
depths and asks which description is nearest the request. The pooled vectors are centred on
the prompt's mean row, because the residual's few huge dimensions are shared by every row.

Ten cases made a call: 5 selection, 1 need, 4 policy. The chosen tool comes first in each
measure in this many of them:

| Layer | Attention: description | Attention: name | Nearest description to the request |
|---|---|---|---|
| embedding | | | 6/10 |
| 3 | 4/10 | 2/10 | 7/10 |
| 7 | 6/10 | 6/10 | 9/10 |
| 11 | **10/10** | 5/10 | 9/10 |
| 15 | **10/10** | 6/10 | **10/10** |
| 19 | 8/10 | **9/10** | 9/10 |
| 23 | 9/10 | 9/10 | 7/10 |

So, to the question "is the tool used the closest match to the task's intent": **yes, but
not in the embedding.** Static token embeddings pick the right tool in 6 of 10 cases, and
names alone in 4. By layer 7 to 15 the request's representation is nearest to the chosen
tool's description in 9 or 10 of 10. In three of the four policy cases that nearest tool
is the wrong one, and the model calls it anyway.

The picture, layer by layer:

1. **Layers 7 to 15: match by meaning.** The row that is about to write the function name
   attends to the description of the tool that matches, and the request's representation
   is nearest to that description. A renamed tool (`fetch_conditions` for weather) is found
   the same way, and the top prediction is `fetch` at 0.9987.
2. **Layer 19: copy the name.** A head attends to the chosen tool's name, with up to 0.76
   of its weight, which is what writes it out.
3. **Arguments: copy from the request.** At the row that writes a value, layer 15 puts 0.90
   to 0.96 of the strongest head's weight on the request when the value is in it (Paris,
   London, 250, ORD-2290, 12 High Street). Values that must be inferred get less:
   `unit=celsius` on the Oslo request, which never mentions a unit, gets 0.13.

### Why policies lose

On "Please cancel my order ORD-4821" with a policy requiring verification first:

| | First token of the name |
|---|---|
| `cancel` (the request's verb, nearest in meaning) | 0.72 |
| `send` (what the policy requires) | 0.25 |

On the account-deletion policy the split is 0.53 against 0.47. The policy does get through,
as a second vote, but the nearest-meaning match outvotes it. The model has no step at which
a rule overrides the match: both are summed at the same position. A strategy here is a
precedence ("check the policy's preconditions first, then match"), and a small model only
gets a weighted average of the two.

### Thinking out loud first

The trace corrects one baseline figure. With 64 tokens to work with, the base model's reply
to the Oslo request is:

> To determine if you need a warm coat, I would need to know the current weather conditions
> in Oslo. I can check the weather for you.
>
> `<tool_call>` `get_weather(city=Oslo, unit=celsius)`

The template allows reasoning before a call, and the model uses it. The baseline above only
gave each reply 8 tokens past the expected answer's length, so it counted this as no call.
The email request is a genuine failure: "I cannot send an email to sarah@example.com. I can,
however, help you with other tasks". So the decision on the first token is between calling
now and talking first, and talking first sometimes leads to the call and sometimes talks the
model out of it. The fair comparison below gives every reply 72 tokens.

## A program for the call decision

The trace says where a program can help most cheaply. Choosing the tool and copying the
arguments already work. The weak step is the first token, and it is a single decision, at a
known position, that the model takes on weak evidence. So the program decides that one
token and nothing else.

`scripts/entail/toolcall/call-gate.entail.txt`, written by `make_callgate.py`, is a fold
over the prompt, 88 neurons and no heads:

- Turn markers (`<|im_start|>` and the role after it) say which turn each token is in.
- A word in the **system** turn marks the concept it belongs to as offered: weather, timer,
  money, email, travel, order or account.
- A word in the **latest user** turn marks its concept as asked.
- On the reply's first token (`\n\n` after `</think>`, in the assistant's turn), if an asked
  concept is offered, the program says `<tool_call>`. Otherwise it is silent.

It runs as an adapter beside the unchanged model, which then writes the call itself.

```bash
python scripts/entail/toolcall/make_callgate.py <tokenizer-dir> call-gate.entail.txt
```

```bash
amql-cli entail adapter call-gate.entail.txt --tokenizer <dir> --out adapters/call-gate
```

```bash
amql-cli --cpu entail eval <container> --tokenizer <dir> --cases gate-cases.jsonl \
    --adapter adapters/call-gate --report gate.json
```

The word lists were written from the tools' descriptions and common words for each concept,
before any gated reply was seen, and were not changed afterwards. Every reply has 72 tokens.

| Set | Base: opens with a call | Base: calls at all | Base: fully right | Gate: opens with a call | Gate: fully right |
|---|---|---|---|---|---|
| Need, 10 | 8 | 9 | **9** | 8 | **8** |
| Policy, 7 | 5 | 6 | 2 | 7 | 2 |
| Selection, 9 | 9 | 9 | 8 | 9 | 8 |

On these cases the gate does not win overall, and the reasons are instructive:

- **Where it fires correctly, the model does the rest.** On the Oslo request the reply now
  opens with `get_weather(city=Oslo)`, with no preamble. Without a policy, "Please cancel my
  order" now gets `cancel_order`, where the base model asked for a reason.
- **Its precision is its lexicon's.** "Is wool or down warmer for a winter coat?" mentions a
  coat, so the gate called, and the model wrote `get_weather(city=London)`. "Let
  sarah@example.com know the meeting has moved" contains no listed word, so the gate was
  silent and the model refused as before. A bag of words is the weakest part, as it was for
  the Python program in [entail-synthesis.md](entail-synthesis.md).
- **It cannot fix selection.** In the policy cases it opens the call, and the model still
  picks the tool nearest the request.
- **The base model is better than its first token.** Given room, it reasons, then calls.
  These single-turn cases do not reproduce how often the model fails to call in real use.
  The multi-turn conversations in the post-training set, with long policies, are where to
  measure that.

## Calling on the model's own evidence

The word-list gate showed where to act, the reply's first token, but not how to decide
there without a hand-written lexicon. This section finds a decision rule that needs no
words at all, and measures it on real conversations as well as on synthetic ones.

### Three sets

| Set | Cases | What it measures |
|---|---|---|
| Original | the 26 cases above | the need, policy and selection sets |
| Clean | 36 need a tool, 24 don't | the omission alone. One user turn, the same 8 tools in every case. The requests imply a tool without naming it ("give me a shout in 7 minutes", "is that true about NVDA?"). There are 12 decoys from a tool's domain that need none ("What does CC mean in an email?"). It was written before any strategy was run on it. |
| Real | 120 call turns, 60 text turns | one decision point from each of 180 recorded multi-turn conversations in `tool_calling.jsonl` (customer service with long policies), up to 6,000 tokens. The label is what the recording's much larger model did: called a tool, or wrote to the user. |

The real set's text labels are a larger model's judgement, often "ask the customer for
their email first", so a call there is not always wrong. Its call turns are the measure
that matters, and they reproduce the complaint. The base model opens a call in 36 of 120,
calls at all within 96 tokens in only 64, and names the recorded tool in 46. The typical
miss is "I understand you're concerned about… I can help you with that", and then nothing.

`make_need_cases.py` and `make_real_cases.py` write these sets, and `score_real.py` scores
them. `entail eval` gained `--watch TOKEN`, which records the probability of a token on
the reply's first row and the five most likely tokens there. It also gained `--prefix TEXT`,
which forces the reply's opening, and `--logprobs`.

### The model knows; it is miscalibrated

On the clean set, the probability the model gives `<tool_call>` on the first token
separates the requests that need a tool from those that don't almost perfectly (AUC 0.98).
In 36 of 36 need cases it is above 0.013. In 23 of 24 no-tool cases it is below 0.017. The
exception is "What is 17 times 23?", which wants a calculator. Yet the model only says
`<tool_call>` when it is the most likely token, which happened in 18 of 36. So the
evidence is there and the threshold is wrong: a 1-in-20 chance of a call is a strong
signal, and the model acts only on better than even.

Two instruments that ask the model more directly do no better on the real set. A hidden
question, "should your next reply be a call to one of the functions? Yes or No", scores AUC
0.64. Forcing a draft call and reading its confidence scores AUC 0.68, as does checking that
its argument values come from the context.

### Calibrating per conversation

On real conversations the raw first-token probability is much weaker (AUC 0.66), because
each has its own system prompt and policy, and so its own baseline inclination to call. The
fix is contextual calibration (Zhao et al., 2021), done per prompt. Run the same
conversation again with the latest message's content replaced by "N/A", and compare:

| Score | Clean AUC | Real AUC |
|---|---|---|
| p(`<tool_call>`) | 0.98 | 0.69 |
| p(`<tool_call>`) / p(`<tool_call>` with the latest message blanked) | 0.98 | **0.78** |

The ratio asks how much the latest message itself moved the model towards a call. That
question transfers from clean prompts to long policy conversations, which the raw
probability does not. A ratio of 2 works on both sets. Raw-threshold rules
(`call if p ≥ 0.03`) do just as well on the clean set but make 30 false calls of 60 on the
real one.

### The program

[`call-when-likely.entail.txt`](../scripts/entail/toolcall/call-when-likely.entail.txt)
is the whole strategy. It is a fold over the prompt that knows only the template's
structure: turn markers, whether the system turn showed the call format (the `<tool_call>`
token), and where the reply begins. At that point it applies one rule:

```
decide "<tool_call>" at 2 times prior when reply_start
```

`decide` is new in Entail (see [entail-language.md](entail-language.md#output)). The host
says `<tool_call>` when its own model gives it at least twice its prior, and rules it out
otherwise. The host measures the prior with one more prefill of the blanked prompt. The
program has 10 neurons and no word lists. The model supplies the evidence and the program
supplies the threshold and where to apply it.

```bash
amql-cli entail adapter scripts/entail/toolcall/call-when-likely.entail.txt --tokenizer <dir> --out adapters/call
```

```bash
amql-cli entail eval <container> --tokenizer <dir> --cases cases.jsonl --adapter adapters/call --watch "<tool_call>"
```

### The template

[`qwen3.5-calls.jinja`](../scripts/entail/toolcall/qwen3.5-calls.jinja) is Qwen 3.5's own
template with two lines of its `<IMPORTANT>` block removed:

- "You may provide optional reasoning for your function call in natural language BEFORE
  the function call", which invites talking first.
- "If there is no function call available, answer the question like normal…", which the
  occlusion study found suppresses calling by a factor of four.

Restating the tools near the reply was also tried: their names, their names and
descriptions, with or without an instruction, and before the reply or before the latest
turn. Each raises calling everywhere and blurs the decision (AUC 0.95 to 0.97 on the clean
set). That trades against the program, so the shipped template does not do it.

### Results

Every reply has 96 tokens. "Calls" counts a call anywhere in the reply, "tool" a call to
the expected tool, and "quiet" a no-tool case left without a call.

| Set | Template | Program | Opens a call | Calls | Right tool | Quiet |
|---|---|---|---|---|---|---|
| Clean, 36 + 24 | stock | none | 18 | 23 | 21 | 23 |
| | stock | `decide` | 34 | 34 | 24 | 23 |
| | edited | none | 28 | 31 | 26 | 23 |
| | edited | `decide` | **36** | **36** | 27 | **23** |
| Real, 120 + 60 | stock | none | 36 | 64 | 46 | 51 |
| | stock | `decide` | 77 | 89 | 67 | 48 |
| | edited | none | 47 | 75 | 56 | 48 |
| | edited | `decide` | **83** | **94** | **70** | 47 |
| Original need, 7 + 3 | stock | none | 8 | 9 | 9 | (counted with the rest) |
| | stock | `decide` | **10** | **10** | **10** | |

On the real conversations, the program and the template together turn 64 calls into 94
and 46 right tools into 70, for 4 more calls among the 60 turns the larger model answered
in words. With the stock template the program alone gets most of that: 89 calls and 67
right tools. The selection and policy sets of the original 26 are unchanged, except that
policy replies now open with their call (7 of 7, from 5), still to the tool nearest the
request.

The two examples:

| Request | Before | After |
|---|---|---|
| "I'm flying to Oslo tomorrow. Do I need to pack a warm coat?" | "To determine if you need a warm coat, I would need to know the current weather conditions in Oslo. I can check…" | `get_weather(city=Oslo, unit=celsius)` |
| "Let sarah@example.com know the meeting has moved to 3pm." | "I cannot send an email to sarah@example.com. I can, however, help you with other tasks…" | `send_email(to=sarah@example.com, subject=Meeting Moved to 3pm, …)` |
| "Is wool or down warmer for a winter coat?" (the decoy the word-list gate called on) | "For a winter coat, **wool** is generally warmer…" | unchanged |

On the original 5-tool prompts the program alone fixes both. On the clean set's 8 tools,
Oslo's ratio with the stock template is only 1.3. The edited template raises it to 3.0, and
the call follows.

### What it exposes: selection

With every needed call now made, the next failure is the choice of tool. On the clean set,
9 of the 36 calls name the wrong one. In 8 of them a tool listed early takes a request meant
for one listed late: `create_event`, `get_stock_price` and `send_sms` are last in the list,
and their requests go to `set_timer`, `get_weather` and `send_email`. Restating every tool
with its description just before the reply fixes it, with the right tool in 35 of 36 in
either list order. But that is the reminder that blurs the call decision. So decision and
selection want different contexts: decide on the plain prompt, then write the call with the
tools restated. That is a host strategy with two prefills, not yet built. On the real
conversations the restated list does not help selection (67 right tools against 70), so it
is not a general fix either.

## Reading a context of any length

### Why a model's context is limited, and a program's need not be

Qwen 3.5 0.8B has 24 layers. 18 are Gated DeltaNet, a recurrent layer: each keeps a fixed
state and updates it token by token, so it reads any length in the same memory. Only the 6
full-attention layers keep every past token, and only they tie the model to the lengths it
was trained on. The model is already mostly a fold. What it lacks is a guarantee that the
fold keeps what matters.

An Entail head reads a fixed place behind the current token, so before this work a program
could only see as far as its heads reached: one head per token of window, or log₂(n) heads
by doubling. A fixed-depth transformer cannot do better, because a value written at its top
layer for one token would have to be read at its bottom layer for the next.

### `carry`: a recurrent register

```
carry last_best from best
```

`last_best` starts each token holding `best`'s final value from the token before. That is
all a fold needs. The rule that a state is written and then read still holds within each
token, so a program remains a dataflow graph, now with one edge from each token to the next.
The interpreter and the sparse adapter already keep each past token's final row, so both run
a carry directly. `compile`, `graft` and `--dense` refuse one, naming the reason. In weights,
a carry is a recurrent layer: a DeltaNet head with a decay of 1 keeps a value, and its delta
rule replaces the value stored under a key. That is the route to running carry programs
inside a model rather than beside it.

### Demonstration: one answer from anywhere in 8 MB

`scripts/entail/scan.entail.txt` reads a question first, `?k`, then records such as `a7;`,
`k3;`, and on `=` answers the largest value filed under key `k`. The test texts put one
higher value under the asked key at a random place, so the answer depends on reading
everything. `scripts/entail/make_scan.py` writes the texts and their answers.

```bash
amql-cli entail run scripts/entail/scan.entail.txt --file first-2mb.txt
```

| Text | Bytes | Expected | Answer | State held | Time |
|---|---|---|---|---|---|
| first-64b | 66 | 9 | 9 | 2,640 bytes | 0.0 s |
| first-4kb | 4,098 | 8 | 8 | 2,640 bytes | 0.1 s |
| first-2mb | 2,097,153 | 8 | 8 | 2,640 bytes | 34 s |
| first-8mb | 8,388,609 | 8 | 8 | 2,640 bytes | 151 s |

The state is three rows of 110 lanes (the row being built, the one before it, and the first) whatever the length. The time is the interpreter's, in
doubles, with no attempt at speed.

### The order matters more than the length

`scan-index.entail.txt` answers the same question asked at the **end**. A fold that does not
yet know the question has to keep the answer to every question it might be asked: here one
register per key, 26 of them, so 5,640 bytes. It answers the 2 MB text correctly too, in 92 s.
With an open-ended question there is no such bound, and keeping everything is exactly what
attention does.

So the strategy for a tiny model with a huge context is:

1. **Put the question first.** A model, or a program, that reads with the question in hand
   can fold the context into a small state. Asked last, it must index it all.
2. **Fold in windows the model can handle.** The model only needs to be good at a short
   window: question, carried notes, next chunk. It rewrites the notes and moves on, and the
   loop is a program. Training could then start at a 64-byte window and widen it, because
   the length is no longer what the model learns.
3. **Keep the carried state small and typed.** The scan carries "the query" and "the best
   value so far". A tool-using model would carry "candidate tools", "facts found", "open
   questions". Writing those types down is the strategy.

What this shows is the program side: a strategy can read 8 MB exactly in fixed memory. It
does not yet show the model reading 2 MB, which needs the loop in step 2 run with the model
writing the notes. That is the next experiment.

## Toward programming an empty model

The question behind the tool experiment is what it would take to give an empty or very
small model this behaviour by writing it down. The trace splits tool use into four steps,
each a different kind of operation:

| Step | What the 0.8B model does | Kind of operation | As a program |
|---|---|---|---|
| 1. Call or talk | the right evidence on the first token, but below prose's | a threshold on relevance | a fold that finds the reply's start, and `decide "<tool_call>" at 2 times prior`: no lexicon needed |
| 2. Which tool | nearest description to the request, layers 7 to 15 | nearest neighbour in meaning | a score per offered tool from shared concepts; the highest wins |
| 3. Policy before match | a second, weaker vote at the same position | precedence | a rule that fires first: a precondition not yet met forces its own tool |
| 4. Which arguments | copy the span of the request the parameter names, layer 15 | pointer copy | `attend back by` a pointer to the span start, or the model itself |

Steps 1 and 3 are where a small model is weakest, and they are the easiest to state exactly.
Step 1 turned out to need no concepts at all, only a threshold on the model's own evidence,
measured against that conversation's prior.
Step 2 needs a notion of meaning, which is what a model is for, and step 4 is something the
model already does almost perfectly. So the division of labour that follows is:

- **The model supplies meaning:** how near a request is to a tool, and where an argument's
  value lies.
- **The program supplies the strategy:** when to act, what takes precedence, and what to
  carry across a long context.

Two things stand between that and a generic solution:

1. **A derived lexicon, for steps 2 and 3.** The first gate's concept words were written by hand. The trace gives a
   way to derive them: the request-to-description similarity at layer 15 picks the right tool
   in 10 of 10 cases, so concept membership can be read off the model's own layer-15
   representations, as `probe_words.py` did for Python from replies. Entail token facts are
   static, so tools that arrive in the context need either a fixed concept vocabulary (as
   here) or a model-supplied similarity fed to the program as a token-level signal.
2. **Precedence in the model's terms.** A policy is read from the context, so a program
   cannot hold it as static facts. The fold can, though: the policy's sentence marks the
   tool it names as a precondition of the actions it names, and the gate's step 3 then
   forces that tool. Building that needs the same lexicon.

## What one example can and cannot do

| Run | LoRA | Data | Need, fully right | Selection, fully right | Policy, fully right |
|---|---|---|---|---|---|
| base | none | | 8/10 | 8/9 | 1/7 |
| 1 | layer 23, rank 8 | the Oslo example | 4/10 | 1/9 | 2/7 |
| 2a | layer 23, rank 8 | the Oslo example and 8 anchors | 8/10 | 8/9 | 0/7 |
| 2b | layers 11, 15, 19 and 23, rank 8 | the same | stopped at step 21, loss oscillating | | |

Run 2b shows what the monitoring is for. Its first-step checks found the direct-path gradient
wrong below layer 19:

| Layer | Along its own gradient: predicted, measured | Random direction: ratio |
|---|---|---|
| 11 | 0.045, 0.0092 (4.9x too large) | -0.90 (wrong sign) |
| 15 | 0.096, 0.034 (2.8x) | -1.19 (wrong sign) |
| 19 | 0.113, 0.047 (2.4x) | 1.04 |
| 23 | 0.21880, 0.21879 | 1.00 |

The measured slope along the approximate gradient is still positive, so it is a descent
direction, but a poorly scaled one. Run 2b reached a loss of 0.021 at step 13, then
oscillated between 0.026 and 0.049; the spike guard undid step 16 and halved the rate, and
the loss kept oscillating, so the run was stopped at step 21. Run 2a, on the exact layer
alone, reached 0.0196 in 10 steps. Until the trainer has a backward pass through attention
and DeltaNet, train the last layer, or treat the lower layers' gradients as directions to be
line-searched rather than trusted.

These use the trainer's own decoding, the expected length plus 8 tokens, so they are
comparable with each other and not with the 72-token table above.

## Limits

- Most figures here are from a handful of synthetic, single-turn cases, on one model. They
  show mechanisms, not rates. The attention figures are from the 6 softmax layers only. The
  call decision is also measured on 180 real decision points. Those are labelled by a larger
  model's choices, so the text turns are a judgement, not ground truth.
- The calibrated decision costs a second prefill of the prompt, with the latest message
  blanked. It is measured for ChatML templates, at the first token of a fresh prefill only,
  and the factor of 2 was chosen on the clean set and checked on the other two.
- The trainer handles LoRA on dense FFN down-projections. Gradients are exact for the last
  layer and follow the direct residual path below it. The monitoring measures that
  approximation rather than correcting it.
- `entail from-lora` reads down-projection adapters. Its token rules see two tokens, the
  same ceiling as `decompile`. Its context notes come from occlusion, which costs two full
  forward passes per removed segment.
- `carry` runs in the interpreter, the stream runner and the sparse adapter, not in compiled
  weights.
- The word-list call gate's lexicon is hand-written, and it was only run on the cases it was
  evaluated on. `decide` replaces it.

## Reproducing

```bash
python scripts/entail/toolcall/make_cases.py <tokenizer-dir> data
```

```bash
amql-cli --cpu lora-train <container> --tokenizer <dir> --data data/need-train.jsonl \
    --anchors data/anchors.jsonl --eval data/eval-all.jsonl --out run2a --steps 60
```

```bash
python scripts/entail/toolcall/score.py run2a/report.json
```

```bash
amql-cli --cpu entail from-lora <container> --tokenizer <dir> --lora run2a \
    --cases data/lora-cases.jsonl --occlude 3 --out run2a/lora.entail.txt
```

```bash
amql-cli --cpu tool-trace <container> --tokenizer <dir> --cases data/trace-cases.jsonl --out trace.json
```

```bash
python scripts/entail/toolcall/analyse_trace.py trace.json
```

```bash
python scripts/entail/make_scan.py scan
```

```bash
amql-cli entail run scripts/entail/scan.entail.txt --file scan/first-2mb.txt
```

The call decision:

```bash
python scripts/entail/toolcall/make_need_cases.py <tokenizer-dir> need/cases.jsonl
```

```bash
python scripts/entail/toolcall/make_real_cases.py <tokenizer-dir> tool_calling.jsonl real/cases.jsonl
```

```bash
python scripts/entail/toolcall/variants.py need/cases.jsonl need/v base noimp
```

```bash
amql-cli entail adapter scripts/entail/toolcall/call-when-likely.entail.txt --tokenizer <dir> --out adapters/call
```

```bash
amql-cli entail eval <container> --tokenizer <dir> --cases need/v/noimp.jsonl --adapter adapters/call \
    --watch "<tool_call>" --report need/noimp-call.json
```

```bash
python scripts/entail/toolcall/score_real.py need/cases.jsonl call=need/noimp-call.json
```

`variants.py noimp` renders the same prompts as `qwen3.5-calls.jinja`. `watch_auc.py`
gives the threshold table for a report run with `--watch`.

`data/eval-all.jsonl` is the three held-out sets with prefixed ids (`n:`, `p:`, `w:`), and
`lora-cases.jsonl` and `trace-cases.jsonl` are subsets of it. `compare_eval.py` scores
`entail eval` reports the same way.
