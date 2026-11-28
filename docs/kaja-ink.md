# Kaja-Ink

Inkling 2 Mini as AMQL serves it, in one container: the checkpoint's own weights (NVFP4 and MXFP8
experts as published) and five Entail programs that run beside it. The model card that goes with it
is [model-cards/kaja-ink.md](model-cards/kaja-ink.md).

| | Inkling 2 Mini | Kaja-Ink |
|---|---|---|
| Dates, number bases, logic, JSON validity (1,164 cases with an answer) | 532 | **1,163** |
| Of 23 cases with no answer, kept silent | 23 | 23 |
| Hard reasoning suite, no thinking | 76/80 | **80/80** |
| Hard reasoning suite, thinking, 3,072 tokens | 80/80 | 80/80 |
| Decode, 80 replies in turn, no thinking (tok/s) | 129 | 131 |

| Adapter | Beside Inkling | Inkling alone |
|---|---|---|
| `dates` | 250/250 | 71/250 |
| `bases` | 315/315 | 106/315 |
| `logic` | 225/226 | 57/226 |
| `json` | 373/373 | 298/373 |

Logic's miss is an expression longer than its window, which Qwen and Gemma miss too.

**No thinking.** Alone, Inkling answers 76 of the 80 hard problems; beside the five programs, all
80: one each of `ledger`, `recurrence`, `registers` and `sieve` won, none lost.

**Thinking.** Unlike Qwen 3.5 and Gemma 4, Inkling closes its own thinking on every hard problem (595
tokens on average, the longest family `seating` at 1,077) and gets all 80, so `stop-circling` has
nothing to do there: no case won or lost.

## Making it

From a copy of the encoded container (`containers/inkling-2-mini`, encoded with `--no-vision`):

The programs are generated for the container's own tokenizer, which cuts digits in threes
(`1900` is `190|0`): see [entail-formal.md](entail-formal.md).

```bash
python scripts/entail/make_dates.py containers/kaja-ink/tokenizer.json dates.entail.txt
python scripts/entail/make_bases.py containers/kaja-ink/tokenizer.json bases.entail.txt
python scripts/entail/make_logic.py containers/kaja-ink/tokenizer.json logic.entail.txt
python scripts/entail/make_json.py containers/kaja-ink/tokenizer.json json.entail.txt
amql-cli entail adapter scripts/entail/reasoning/stop-circling.entail.txt --tokenizer containers/kaja-ink --out adapters/stop-circling
```

and likewise `entail adapter` for the other four. Then:

```bash
amql-cli attach containers/kaja-ink --adapter adapters/stop-circling,adapters/dates,adapters/bases,adapters/logic,adapters/json
```

The model card goes in as `README.md`.

## What was found on the way

Porting the programs to Inkling turned up faults that touched any model whose template or tokenizer
differs from Qwen's and Gemma's, all fixed generically:

- **Digits cut in threes.** Every number-reading program assumed a digit per token. `dates`, `bases`,
  `logic` and `json` are now generated from the tokenizer, reading and writing numbers in its own cuts;
  Qwen's and Gemma's programs are unchanged, reply for reply.
- **A close of several tokens.** Inkling closes its thinking with
  `<|end_message|><|message_model|><|content_text|>`. `{mark:think_end}` is what a program writes (the
  first token, and the model writes the rest), `{mark:think_close}` what it reads (the last); and
  `{mark:think_open}` is `<|content_thinking|>`, not the turn header both share.
- **Every message ends the same way.** `<|end_message|>` closes the thinking as well as the reply, so
  replies stopped at the thinking's close: the hard suite with thinking scored 0 of 80 until a reply's
  end became the first token after it that the close does not share (`<|content_model_end_sampling|>`).
- **An effort, not a switch.** The template takes `reasoning_effort`; not thinking passes `none`.
- **Entail runs it.** `entail eval`, `side` and `bench` open a container however it runs.

## Measuring it

```bash
amql-cli entail eval <container> --tokenizer <container> --cases <subject>.jsonl --adapter <adapter> --report side.json
python scripts/entail/formal_cases.py score <subject>.jsonl side.json --base base.json
amql-cli entail bench <container> --suite hard --seed 11 --no-think --max-tokens 2048 --adapter <five> --report five.json --compare nothink.json
amql-cli entail bench <container> --suite hard --seed 11 --max-tokens 3072 --adapter <stop-circling> --report stop.json --compare think.json
```

The cases are the ones Kaja-gem was measured on (`formal_cases.py`, seeds as there).

## Not done

- **Tool calls.** The template writes a call as `<|message_model|>NAME<|content_invoke_tool_json|>`
  `{"name":NAME,"args":{…}}<|end_message|>`, the name twice and the call's distinctive mark after the
  name. The template learner (`CallShape`) refuses it ("a call without arguments is written another way
  than one with them"), so the server offers no tools and `call-gate` has no `{mark:call_open}`.
- **The server with the adapters** was not measured; the numbers above are `entail eval` and `bench`.
- **The checkpoint's drafter** (`mtp.safetensors`), images, and long contexts with the adapters.
