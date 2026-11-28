---
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
base_model: thinkingmachines/Inkling-2-Mini
base_model_relation: adapter
pipeline_tag: text-generation
tags:
  - amql
  - inkling
  - nvfp4
  - mixture-of-experts
  - looped-depth
  - entail
---

# Kaja-Ink

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (26.4B parameters, 3.4B
active, its middle layers run three times) in one container for
[AMQL](https://github.com/simonsoanes/AMQL), with five Entail programs that run beside the model at
the sampler and rule on the next token: exact dates, number bases, propositional logic and JSON
validity, and `stop-circling`, which ends thinking that has started to repeat itself. Nothing was
trained, and the weights are the checkpoint's own (NVFP4 and MXFP8 experts as published).

| | Inkling 2 Mini | Kaja-Ink |
|---|---|---|
| Dates, number bases, logic, JSON validity (1,164 cases with an answer) | 532 | **1,163** |
| ... and of 23 cases with no answer, kept silent | 23 | 23 |
| Hard reasoning suite (80 generated problems), no thinking | 76/80 | **80/80** |
| Hard reasoning suite, thinking, 3,072 tokens | 80/80 | 80/80 |
| Decode, 80 replies in turn, no thinking (tok/s) | 129 | 131 |
| Size | 17.6 GB | 17.6 GB, the five programs 1.2 MB of it |

Measured with AMQL on one RTX 5090. The cases and the reasoning suite are generated, with answers
from Python's own libraries; details are in AMQL's `docs/kaja-ink.md`.

| Program | Does | Beside Inkling | Inkling alone |
|---|---|---|---|
| `dates` | weekdays, date sums and differences, leap years, month lengths | 250/250 | 71/250 |
| `bases` | whole numbers between bases 2 to 16 | 315/315 | 106/315 |
| `logic` | propositional constants, truth tables, tautologies | 225/226 | 57/226 |
| `json` | whether a JSON text is valid, and no ending inside one | 373/373 | 298/373 |
| `stop-circling` | closes the thinking at a line break once it repeats itself, or after 1,536 tokens | 80/80 | 80/80 |

The four formal programs answer exactly where the question is theirs and are silent otherwise, so
what none of them reads the model answers as before. Inkling finishes its own thinking on the hard
suite (595 tokens on average), so `stop-circling` changes nothing there; it is carried for the
replies that do circle.

## Running it

This is an AMQL container, not a transformers checkpoint.

```bash
amql-server --model kaja-ink=<this folder>
```

It serves OpenAI-compatible chat completions with streaming and thinking (`reasoning_content`;
`reasoning_effort` as the template takes it). The five programs run beside it unless a `--config`
entry says `"adapters": []`.

## Not in this release

- **Tool calls.** Inkling writes a call as `<|message_model|>NAME<|content_invoke_tool_json|>`
  `{"name":NAME,"args":{…}}<|end_message|>`: the tool's name in the turn header and again in the
  JSON. AMQL's template learner does not read that form yet, so tools are not offered.
- **Images.** The vision tower was left out when the container was encoded.
- **The checkpoint's drafter** (`mtp.safetensors`) is not attached.

## Licence

The weights are [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)'s,
under the [Apache 2.0 licence](https://www.apache.org/licenses/LICENSE-2.0) and its
[acceptable use policy](https://thinkingmachines.ai/model-acceptable-use-policy/). The Entail programs
are AMQL's.
