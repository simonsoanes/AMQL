# Entail: compiling statements into model weights

`bake-predicate` showed that a rule can be solved into weights, but what it
writes is fitted to examples of a model obeying the rule. Entail is the
deterministic counterpart. A predicate says what must be the case; an
entailment says what must *follow from what*, so it can be compiled. Every
statement becomes a specific group of weights, and nothing is learned, sampled
or fitted.

```bash
amql-cli entail check   scripts/entail/mathematics.entail.txt
amql-cli entail run     scripts/entail/mathematics.entail.txt --expr " 347*829="
amql-cli entail compile scripts/entail/mathematics.entail.txt --out <checkpoint>
amql-cli entail graft   scripts/entail/mathematics.entail.txt --host <checkpoint> --out <checkpoint>
amql-cli entail eval    <container> --cases <cases.jsonl> [--batch N] [--text <file>] [--report <json>]
```

`--batch N` answers N cases at once, on the CPU only, and gives the same replies as
one at a time. `run` executes the program's meaning with no weights at all, which is what the
compiled model has to reproduce. `scripts/entail/compile-and-evaluate.sh` does
the whole experiment end to end.

For physics and chemistry, see [entail-sciences.md](entail-sciences.md). For tool calling and reading a context of any length, see [entail-strategies.md](entail-strategies.md). For dates, number
bases, logic and JSON, see [entail-formal.md](entail-formal.md).

## The language

This is a summary; the full reference is [entail-language.md](entail-language.md).

| Statement | Meaning | Becomes |
|---|---|---|
| `token "7" means val = 7, kind = 1` | an input fact | embedding row |
| `state x[8] scale 1/16` | an interim fact | residual lanes |
| `let y = a + 2*b` | a name for a sum | nothing: inlined into its readers |
| `attend back k ... end` | what lies k tokens behind | an attention head |
| `attend back by ptr ... end` | the same, k computed | a head whose query is `ptr` |
| `entail y when E >= c` | a consequence | a pair of SwiGLU neurons |
| `entail y = V when f, not g` | a gated consequence | a pair of SwiGLU neurons |
| `emit "7" when o[7]` | an output fact | an output row |

`for`, `if`, `def`/`call` and `sum` are generators: they expand at compile time
and leave no trace in the weights.

Several `entail` statements on one state add up, so exclusive conditions read as
"or" and a value as a case analysis. A state is written, then read, and writing
to it after that is an error, so a program *is* a dataflow graph. Layers are
never written down: the scheduler places each statement at the earliest layer
its inputs allow, and dimensions are reused once nothing reads a lane any more.

## Why the logic is exact

Three properties of the emitted weights turn a stack of RMSNorms and SwiGLUs
into integer logic rather than an approximation of it.

- **Exact gain.** A constant ballast, spread over just enough dimensions that
  the mean of squares is a power of four, makes the norm's divisor a power of
  two. Everything else is too small to survive the rounding back to float, so a
  lane reads back as precisely the number written to it. Whole numbers stay
  whole from layer to layer; long division, which multiplies its remainder by
  ten at every step, depends on this.
- **Regeneration.** A threshold is `silu(g₁) − silu(g₂)`, exact at every integer
  with ±3/8 of noise margin. A gated value is `silu(g) − silu(g − c)`, flat above
  its gate, so a fact slightly off its height still gates a value exactly.
- **Attention as lookup.** Keys are constant and a query is a sum of RoPE phases,
  so a head's score peaks at exactly one relative offset. `entail check` proves
  the margin over the whole context (580 logits over 32k): the softmax output is
  one token's value, not a blend.

## Grafting into a host

Overwriting weights in place does not suit this. Measured on Qwen 3.5 0.8B, the
residual norm of one token type ranges from 1.7 to 12.6 and no quiet subspace
exists to carry a register, while a single head (layer 3, head 7) costs 5 nats
to remove. `graft` restructures the checkpoint instead, keeping every host
weight's value:

1. The hidden size widens, and the new dimensions are the program's lanes. Host
   matrices gain zeros there, so no host weight reads or writes a lane, and host
   norm weights are rescaled so the wider mean leaves them where they were.
2. The compiled layers go in front of the host's first layer. They read only
   lanes, so they are a pure function of the tokens that host noise cannot reach.
   They finish by clearing every lane and the ballast, exactly, so the host
   starts from its own embedding.
3. Each output lane is a column of the tied embedding, non-zero in its own
   token's row, and the only lanes the final norm passes. A set lane raises
   exactly one logit. The end-of-answer lane lowers the logits of the tokens an
   answer is made of, so the number stops and the host says what follows.

Wherever the program says nothing, the host's computation is unchanged. The same
compiled layers serve the empty model and the graft.

### Any host the importer can judge

`graft` judges the host with the importer's own rules (`EntailHost.Judge`,
through `ArchMapper`), then compiles the program for the host's heads and FFN
width, and writes the block in the host's own layer shape and tensor names.
Nothing in it names a family:

- **Attention.** The host's head count, KV groups and head dim. Lookups use up
  to 16 of the host's own rotary pairs, half-split, interleaved, partial or
  zero-tail, at its own frequencies, including linear, llama3, yarn and
  longrope scaling. The highest frequencies are tried first; pairs spread over
  the whole range come next, for a host whose first pairs repeat within the
  context. The score is set for the host's score scale (`query_pre_attn_scalar`,
  Granite's multiplier), rope attention factor and softcap. q and k carry their
  height through the host's q/k norm, per head or across the whole projection,
  or directly when there is none. An output gate is held open.
- **Norms.** `w` or `1 + w`. When the host also normalises each sub-block's
  output (Gemma's sandwich, OLMo 2's post-norms), the block keeps one head and
  one neuron back to put a ballast in that output too, so those norms divide by
  a power of two as well.
- **FFN.** The block's neurons go in the dense FFN. In an MoE layer they go in
  the shared expert, with the routed experts zero, or else in every expert
  alike, with the down weights divided by the routing weight an even split
  gives. Biases are zero. Fused `qkv_proj` / `gate_up_proj` are written fused.
- **Scales.** A residual or embedding multiplier must be a power of two. A
  host that scales its embedding by √hidden (Gemma) gets a width that is a
  power of four, so the scale stays a power of two; its own embedding rows move
  to keep their product, written in f32.
- **Outputs.** With a tied head, the output columns live in the embedding and
  the block clears them on the way in. Otherwise they go in `lm_head`, and the
  logits multiplier is divided out. `--output-norm` sets the residual norm the
  force is sized for (115, measured on Qwen 3.5 0.8B).
- **Widening.** Each host tensor gains zeros on its residual axis, found by its
  role. A square projection (Llama's `o_proj` when heads × head_dim = hidden)
  has two axes of that width, and only its role says which one is the stream.
- **Config.** Layer counts and every per-layer fact move with the inserted
  layers: `layer_types` (spelled out when the host gives only a period),
  `no_rope_layers`, `max_window_layers` and `mlp_only_layers`. `head_dim` is
  stated, because hidden ÷ heads no longer gives it. Any other per-layer table
  is refused.
- **Tokens.** Byte-level BPE, metaspace (`▁`) and byte-fallback pieces, in
  dictionary or Unigram vocabularies.

What a block cannot be exact in is refused, naming the property: LayerNorm
hosts (the mean subtraction moves every lane), parallel blocks, an ungated FFN,
multipliers that are not powers of two, uneven MoE splits, dynamic rope past
its maximum, and a host where no choice of pairs makes every head a lookup over
the context. The last includes Gemma 2: its softcap of 50 cannot separate
neighbouring offsets.

| Check | Result |
|---|---|
| Qwen 3.5 0.8B, `mathematics.entail.txt`, against the Qwen-only graft it replaces | all 705 tensors and the config identical |
| `EntailGraftTests`: 16 zoo hosts (Llama with and without biases, Qwen3, yarn Qwen2, Phi-3, longrope Phi-3, GLM-4, Gemma 3, OLMo 2, SmolLM3, OLMo 3, Ministral, Mixtral, OLMoE, Qwen2-MoE, Qwen3-MoE), run by AMQL | silent text within 1e-3 of the host's logit scale, every other position likewise; the program's token wins by over 100 logits |
| The same grafts loaded by transformers under each host's own class (`scripts/entail/check_graft.py`) | 14 of 16 behave as in AMQL. Two cannot be reloaded there, though AMQL runs both: Phi-3's longrope validation derives head_dim from hidden_size, and OLMoE sizes its q/k norms by hidden_size |
| Refused by name | StableLM, Cohere, Starcoder2 (LayerNorm), Granite (embedding × 3), dynamic rope past its maximum, Gemma 2's softcap, longrope past its original length (its two regimes share one pair) |

Silent text moves a little in every graft, by the same mechanism as on Qwen. A
host norm's epsilon stays the same while the mean it is added to shrinks with
the wider stream, so it acts like an epsilon h'/h times larger.

A program can also run beside a model instead of inside it, which leaves the model
untouched at its own precision: see [entail-adapter.md](entail-adapter.md).

## Results

Greedy decoding on the CPU, Qwen 3.5 0.8B before and after, on operands drawn
with seeds not used during development. Direct arithmetic gives both models a
short drill, so the baseline reflects ability rather than formatting.

| Exact answers | Base | Grafted |
|---|---|---|
| add, 2 / 3 / 5 / 8 digits | 15 / 15 / 11 / 4 of 15 | 15 / 15 / 15 / 15 |
| subtract, 2 / 3 / 5 / 8 digits | 14 / 6 / 5 / 3 of 15 | 15 / 15 / 15 / 15 |
| multiply, 2×2 / 3×3 / 4×4 / 6×6 | 13 / 3 / 0 / 0 of 15 | 15 / 15 / 15 / 15 |
| divide, exact / 5 by 2 digits | 2 / 1 of 15 | 15 / 15 |
| **direct, overall** | **92/210 (43.8%)** | **210/210 (100%)** |
| chat: add 5 digits | 13/15 | 15/15 |
| chat: multiply 3×3 | 1/15 | 15/15 |
| chat: subtract 4 digits | 7/15 | 7/15 |
| word problems, step by step | 10/15 | 15/15 |
| **chat, overall** | **31/60 (51.7%)** | **52/60 (86.7%)** |
| prose: top-1 agreement with base | | 767/767 |
| prose: loss, nats/token | 3.55195 | 3.55210 |

The empty model scores 300/300 on the final program, across every operation up
to the limits below. A build before grouped numbers were added scored 750/750,
including 400 random and 27 edge-case divisions.

The eight chat failures are all the same case: asked for a negative difference,
the model states a number without ever writing the expression, so the program
has nothing to read. The word problems are three generated templates with easy
reasoning and hard arithmetic, not a general reasoning benchmark.

## Limits

- This adds exact arithmetic, not reasoning. It acts only on an expression ending
  in `=`, so a model that sets up the wrong sum gets the wrong sum's exact answer.
- `mathematics.entail.txt` takes one binary operation on whole numbers: up to 8
  digits for `+` and `-`, 6 for `*`, 6 by 4 for `/`, quotients truncated to two
  places. Decimals, signed operands, chains and brackets entail nothing and are
  left to the host. An expression needs one token in front of it: a head cannot
  tell the start of the text from a run of identical tokens, so a number opening
  the text looks too long and the program stays silent.
- Exactness needs float32. The fp16 GPU weight path cannot hold the ballast and
  would round the block's weights, so run on the CPU, or run the program as an
  [adapter](entail-adapter.md), which doesn't depend on the host's precision. A grafted model is also
  larger and slower: 59 layers instead of 24, hidden 1664 instead of 1024, with
  the block stored in f32.
- Memory is not a limit, only speed. `graft` and `encode` stream to disk, and
  `entail eval` runs a graft whose f32 weights exceed the memory budget with
  `--weights disk`: exact f32 on the CPU, keeping what fits in memory and reading
  the rest from the container on every pass. It picks this mode by itself when
  the weights don't fit.
- A host must be one the importer can judge, with RMSNorm, gated FFNs and rotary
  attention; see [Any host the importer can judge](#any-host-the-importer-can-judge)
  for what is refused. Hosts other than Qwen 3.5 are verified on the tiny zoo
  models, not yet on real checkpoints.
