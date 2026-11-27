# Long contexts without training: finding what to edit

Gemma 4 26B A4B reads its whole 256K context and quotes from it, but asked for one fact among 126 of
the same form, it answered 14 of 48 at 256K (Qwen 3.5 4B: 17 of 24). The usual fix is
to train it further on long documents. Here, multiplying the query norms of its five global
attention layers by 1.5 brings it to 38 of 48, five vectors of 512 numbers changed, nothing trained.
This page is how that was found, so the same search can be run on the next model.

| Gemma 4 26B A4B, NVFP4 (GPTQ), NVFP4 KV | 28,672 | 126,976 | 258,048 |
|---|---|---|---|
| As shipped | 11/14 | 20/48 | 14/48 |
| Global layers' queries ×1.5 | **14/14** | **37/48** | **38/48** |
| Right code's log-loss, as shipped (nats) | 1.18 | 6.31 | 11.18 |
| Right code's log-loss, ×1.5 | **0.00** | **2.33** | **2.00** |
| Quoting, nats a token, as shipped | 0.276 | 0.529 | 0.489 |
| Quoting, ×1.5 | **0.220** | **0.260** | **0.220** |

Guessing five digits at random costs 11.5 nats: at 256K the model as shipped was nearly guessing.
The hard reasoning suite (80 questions, no thinking) is 80 of 80 with the edit, as it was without,
at 708 tokens a reply against 739. The server's tool-call checks pass 30 of 32. Both misses are the
check's own ambiguity: asked for the line `{gamma};`, the model writes it with the semicolon.

To make it:

```bash
amql-cli quantise containers/gemma-4-26b-a4b --out containers/gemma-4-26b-a4b-nvfp4-gptq-sharp --embedding \
  --reuse containers/gemma-4-26b-a4b-nvfp4-gptq --scale '^(5|11|17|23|29)\.self_attn\.q_norm\.weight$=1.5'
```

`--reuse` copies the 4-bit weights of a container already quantised, so this takes 40 seconds, not a
22-minute calibration. Attach the drafter and vision tower to the result as before
([gemma-4.md](gemma-4.md)).

## The search

Each step is one `amql-cli context bench` per candidate, about 4 minutes for Gemma at 256K:

```bash
amql-cli --kv-format nvfp4 context bench <container> --text <frozen text> --lengths 32K,128K,256K --questions 48 --out <file>
```

Freeze the text first (copy it somewhere nothing edits it). Every candidate reads the same document
with the same facts in it.

### 1. Measure finely

Right or wrong over 48 questions moves by 3 or 4 between two runs of the same thing. The bench also
reports each question's **right code's log-loss**: the reply is not generated but scored, the nats
the model puts on the true code. This tells "nearly had it" from "lost", and was what separated
the candidates below.

### 2. Read the misses

What a wrong answer is says where to look:

- **Another fact's code**: it found facts of the right form and could not tell them apart. This
  was most of Qwen 3.5 4B's misses ([context-extension.md](context-extension.md)).
- **Any number, or "not found"**: it did not reach the fact at all. Gemma's misses at 256K were
  `128`, `0.000`, `1e-3`, `10000`, numbers the source code is full of.

### 3. Found, or read?

`--focus 8` reads where the question's attention goes (summed over layers and heads, the
attention to the same document with nothing asked taken off) and picks the 8 passages that
drew most. For Gemma as shipped:

| | 126,976 | 258,048 |
|---|---|---|
| The fact's passage among the 8 | 38/48 | 39/48 |
| Answered right | 20/48 | 14/48 |

**The attention reached the fact and the answer did not.** That points away from position
encoding (which would lose the passage) and towards how sharp the attention is: a passage among
the 8 that drew most can still hold a small share of a softmax spread over 256K keys.

### 4. Rule out precision

Rounding to 4 bits does matter to Gemma's long contexts: rounding to nearest instead of GPTQ
recalled 2 of 24 at 258K against 5. So before editing anything, put the original weights back,
a group at a time, and see whether recall comes back. `--keep <regex>` leaves the projections whose
names match at full precision; with `--reuse` each try takes under a minute to make.

| Back at BF16 (the rest GPTQ) | +GB | 28,672 | 126,976 | 258,048 |
|---|---|---|---|---|
| Nothing (as shipped) | 0 | 11 (1.18) | 20 (6.31) | 14 (11.18) |
| All attention and the shared MLP | 2.4 | 12 (0.74) | 15 (7.81) | 10 (15.15) |
| Experts of layers 0–9 | 11 | 13 (0.11) | 16 (6.48) | 12 (11.31) |
| Experts of layers 10–19 | 11 | 13 (0.56) | 18 (6.75) | 13 (11.87) |
| Experts of layers 20–29 | 11 | 10 (1.51) | 23 (6.08) | 14 (11.57) |
| Everything | 36 | 13 (0.44) | | |

Right of 14 or 48, and the right code's log-loss. The early experts at full precision help at 32K
(0.11 nats against 1.18), but **no group helps at 128K or 256K**. The whole model at BF16 does not fit a
32 GB card; spilled into host memory it read at 56 tok/s and was stopped before 128K. Precision is
not what Gemma lacks at long range.

### 5. Sharpen the layers that reach far

Only Gemma's 5 global layers (5, 11, 17, 23, 29) attend to the whole context; the other 25 see a
1,024-token window, where nothing is diluted. Gemma 4 has no 1/√d in its attention: the query and key
norms set each head's sharpness, so multiplying a layer's query norm by *s* multiplies every one
of its attention logits by *s*. The values are left alone: in Gemma's global layers they come
from the keys before their norm.

| Queries × | 28,672 | 126,976 | 258,048 |
|---|---|---|---|
| 1 (as shipped) | 11 (1.18) | 20 (6.31) | 14 (11.18) |
| 1.15 | 14 (0.00) | 32 (2.59) | 26 (5.42) |
| 1.3 | 14 (0.00) | 37 (1.81) | 33 (3.19) |
| **1.5** | **14 (0.00)** | **37 (2.33)** | **38 (2.00)** |
| 1.75 | 13 (0.84) | 38 (1.26) | 40 (2.65) |
| 2.0 | 13 (0.24) | 35 (2.78) | 33 (4.05) |

1.5 and 1.75 are level within the noise at long range. 1.5 is the smaller change and does not
lose the short context. Past 2 it falls back.

### 6. Check what else it changes

An edit that helps one measure can cost others. Run what the model is used for: here the hard
reasoning suite and the server's live tool-call checks (both above), with the edited container
served as any other.

## Running it on another model

1. Bench it at three lengths, inside its trained length (`context bench`, a frozen text).
2. Read the misses: look-alikes, or anything?
3. `--focus 8`: is the fact's passage found more often than it is answered? Then the attention is
   too flat: step 5. Not found: the reach is the problem (position encoding, see
   [context-extension.md](context-extension.md)), or precision: step 4.
4. Restore groups with `--keep` and `--reuse`: attention, dense FFN, experts by thirds of the
   stack. Keep what moves the log-loss at long range, if anything does.
5. Sweep a factor on the query norms of the layers that attend to everything (global, full or
   NoPE layers in a hybrid; every layer in a model without windows) with `--scale`, in steps
   of 0.15 to 0.25, until the log-loss turns.
6. Check its ordinary work.

Two cautions for step 5:

- **The norm must multiply the query.** `--scale` multiplies what the norm applies. Where the norm
  is `x · w` (Gemma 4) that is the stored `w`. Where it is `x · (1 + w)` (Gemma 2 and 3, Qwen 3.5's
  offset norms) it is `1 + w`: the stored numbers become `s · (1 + w) − 1`, so the same factor
  means the same sharpening in either. The offset is read from the model's plan. With no query
  norm, scale `q_proj`'s rows instead. A model with QK-norm cancels any scaling of `q_proj` itself.
- **Leave windowed layers alone.** They never see more than their window.

## What is not known

- One kind of document (this repository's docs and C#), one form of fact, greedy replies, one seed:
  48 questions a length. MRCR-style multi-turn retrieval, summaries and images at long range were not
  measured.
- One factor at every length. A factor that grows with the length read (log *n*, as scalable softmax
  does) might do better at 32K and at 256K both, but needs the engine to apply it per sequence.
- Whether the BF16 model itself recalls better at long range: it could not be run there.
- Whether other models gain. Qwen 3.5 4B (offset query norms, which `--scale` now handles, misses
  among look-alikes) is the obvious next one to try.
