# AMQL - C# implementation of VIndex3 (Larql)

This is a port of the VIndex3 implementation, along with support for generating it from a model (Qwen 3.5 initially) and then allowing model independent inference, token relationship route following and exploration of the model internals in order to do some research into direct model manipulation and patching, with live LORA adapters in custom inferencing.

Credit for the design of the VIndex3 goes to Chris Hay.

## What is this for?

While building a continuous cognition platform, I ran into the problem that all current LLMs have flaws and no way to self-improve. This project is intended to eventually become a tool for AI self-improvement.

It lets you turn a model into a graph database, then query the relationships between tokens (or their text representations). Once the edges in the graph have been identified, it becomes possible to generate a specific LoRA adapter that adjusts that behaviour in the model (or rewrites the base model), effectively editing its input and output knowledge.

At the current level, it's possible to remove the concept of something being associated in a particular way, or to add a new association between two items in a familiar form (PlaceA is the capital of PlaceB) - useful for correcting flaws in the embedding layer. It's also possible to correct a relationship between two things where a relationship exists but is of the wrong type.

Looking at the next layer of abstraction, it's also possible to adjust relationships that humanity hasn't yet described linguistically but normally infers a connection between. This is especially useful when combined with positive and negative reinforcement derived from internal traces taken during a model's inference stage. The intent is to correct things ranging from hallucinations (where the relationship is a 'user satisfaction' signal added during post-training) to incorrect tool selection when operating agentically.

The hope is that it eventually becomes possible to calculate the representation of an approach to a problem space - and potentially to create new ones or transpose existing ones - for example, applying first-order logic to an understanding that isn't yet fully trained into the model, whether due to a lack of source material or insufficient model density.

Combined with an automatic self-learning process, this should extend a model's understanding and intelligence beyond currently trainable human textual representations, in cases where a known construct would be better applied. That's an outcome fine-tuning alone can't achieve, since it can only generate more intelligent outcomes by relying on scenarios that are inherently gated at human-level intellect.

Generating genuinely novel solution vectors for a problem space is a separate challenge requiring its own approach, but this project does allow alternate solution vectors to be applied once they've been identified.

## Usage

`amql-cli` is the loader/inference front-end: it turns a raw HF checkpoint into a canonical VINDEX3 container, then lets you run and inspect inference against it.

```
amql-cli encode <model-dir> --out <container-dir>   map + materialise
amql-cli verify <container-dir>                     integrity + readiness
amql-cli synth-model <dir>                          write an executable demo checkpoint
amql-cli tokens --tokenizer <checkpoint-dir> "text"
amql-cli decode --tokenizer <checkpoint-dir> <id,id,…>
amql-cli route <container-dir> <A> <B> --tokenizer <checkpoint-dir>
                [--top 5] [--templates 8] [--trace-layer-start 8]
                [--trace-layer-end 24] [--no-trace] [--corrupt the]
                [--patch <patch.safetensors>]
amql-cli path <container-dir> <A> <B>
                [--topk 6] [--max-nodes 48] [--max-depth 6]
                [--patch <patch.safetensors>]
amql-cli generate <container-dir>
                --prompt "text" --tokenizer <checkpoint-dir>
                [--steps 8] [--temperature 0] [--top-k 0] [--top-p 0]
                [--seed 42] [--logits K] [--component target]
                [--patch <patch.safetensors>]
amql-cli inspect-token <container-dir> <token>
                [--tokens ctx,ids] [--neighbors 5] [--logits K]
                [--tokenizer <checkpoint-dir>] [--component target]
                [--patch <patch.safetensors>]
amql-cli change-tensor <container-dir> <object> <tensor> <cell>
                (--set V | --add V | --scale F | --zero)
                --out <patch.safetensors>
amql-cli save-lora <patch.safetensors> --out <lora-dir>
                [--rank 8] [--alpha 16] [--container <container-dir>]
amql-cli help
```

Two kinds of directory are involved: the **container** (`<container-dir>`, encode output, weights only) and the **checkpoint** (`--tokenizer`, the original HF model directory whose `tokenizer.json` converts text to ids; `--model-dir` is an accepted alias). `--tokenizer` is optional once the container was encoded with a `tokenizer.json` beside it (encode copies it in).

### Quick start

```bash
amql-cli synth-model demo-model
amql-cli encode demo-model --out demo-container
amql-cli generate demo-container --prompt "hi" --tokenizer demo-model
amql-cli route demo-container France Paris --tokenizer demo-model --top 5
```

### Encoding and verifying a real checkpoint

```bash
amql-cli encode ./Qwen3.5-0.8B --out ./containers/Qwen3.5-0.8B
amql-cli verify ./containers/Qwen3.5-0.8B
```

`verify` re-derives every hash from disk alone, resolves real operand tensors through the container, prints the operator census (which layer operators are present), and reports whether the primary text component plans and executes — refusing by name for anything the runtime doesn't yet serve.

### Tokenizing text

```bash
amql-cli tokens --tokenizer ./Qwen3.5-0.8B "The capital of France is"
amql-cli decode --tokenizer ./Qwen3.5-0.8B 9419,11
```

### Generating text

```bash
amql-cli generate ./containers/Qwen3.5-0.8B --prompt "The capital of France is" \
  --tokenizer ./Qwen3.5-0.8B --steps 8 --logits 5
```

### Probing token relationships (`route`)

`route` names the relationship between two tokens using template probing, reports the (layer, head, position) attention coordinates carrying one token into the other's prediction, and — via causal tracing — the per-layer weights naming exactly which residual tensors to patch or LoRA to change that propensity:

```bash
amql-cli route ./containers/Qwen3.5-0.8B France Paris --tokenizer ./Qwen3.5-0.8B --top 5
```

Illustrative output (actual scores/coordinates depend on the checkpoint):

```
container: ./containers/Qwen3.5-0.8B (weights)   tokenizer: ./Qwen3.5-0.8B (checkpoint)

France -> capital-of (0.83 @ 14,3,5,2) -> Paris
     causal weights (patch targets), P(Paris) clean=0.831 corrupt=0.041:
       L14:  Δ 0.2140 (25.8% of effect)
       L11:  Δ 0.1385 (16.7% of effect)

scores = P(B) after template(A); coords = (layer, head, queryPos, keyPos) of the final-row attention onto A;
causal Δ = P(B) restored by reinstating that layer's clean residual (corrupt → clean) — the tensors to patch/LoRA.
```

### Finding the token-continuation path between two tokens (`path`)

Where `route` names the relationship, `path` shows the model's own route between two tokens without naming it — bidirectional best-first search over the next-token continuation graph:

```bash
amql-cli path ./containers/Qwen3.5-0.8B France Paris --tokenizer ./Qwen3.5-0.8B
```

Illustrative output (actual chain/costs depend on the checkpoint):

```
container: ./containers/Qwen3.5-0.8B (weights)   tokenizer: ./Qwen3.5-0.8B (checkpoint)
searching from 'France' (id 9419) toward 'Paris' (id 12958) — edges = top-6 continuations (cost −log P) …

    9419  France                   start
    ...
     603  is                       +1.42
   12958  Paris                    +0.61

meeting point: 'Paris' — fwd 3.11, bwd 0.61
total cost 3.72 · 9 model forwards · 22 nodes
path = token chain only (no relation names); costs are −log P of each continuation edge.
```

### Inspecting a token in vocabulary space

```bash
amql-cli inspect-token ./containers/Qwen3.5-0.8B 12958 --tokenizer ./Qwen3.5-0.8B \
  --tokens 9419,318 --logits 5
```

Reports the token's embedding profile (row, min/max/mean, L2 norm) and nearest neighbours by cosine similarity; with `--tokens` and an executable component, it also reports the model's logit and rank for that token at the end of the given context.

### Editing a weight by hand and creating a patch (`change-tensor`)

A patch is a set of **weight deltas** (patched value − original, stored as f32 in a safetensors file). `change-tensor` reads a weight from the container, applies a single-cell edit, and records the delta into a patch file — the container itself is never rewritten, so `verify` integrity and the original model stay untouched. The cell is `row,col` for a 2-D tensor (a projection matrix) or a flat index for a 1-D vector (a norm scale). `--add` and `--scale` apply against base + existing delta, so repeated calls compose into the same patch; an edit that lands back on the base value removes its entry.

```bash
# a tiny container is the friendliest scratchpad
amql-cli synth-model demo-model
amql-cli encode demo-model --out demo-container

# set the embedding cell (token 3, dim 1) to 2.5, then bump another cell
amql-cli change-tensor demo-container target.embedding weight 3,1 --set 2.5 --out patches/demo.safetensors
amql-cli change-tensor demo-container target.embedding weight 2,0 --add 0.5 --out patches/demo.safetensors
```

```
'target.embedding/weight' [12x4] F32 [3,1] 0.1 → 2.5 (Δ 2.4)
patch: patches/demo.safetensors (1 tensor)
```

On a real checkpoint, target the tensors `route` names as patch targets (the causal Δ layers), e.g. a projection in the decoder stack:

```bash
amql-cli change-tensor ./containers/Qwen3.5-0.8B target.decoder_stack 14.self_attn.q_proj.weight 100,50 \
  --add 0.01 --out patches/capital.safetensors
```

### Running a pathway with a patch

Every model pathway accepts `--patch <patch.safetensors>`. The deltas are merged into each weight as the runtime loads it, so `route`, `path`, `generate` and `inspect-token` all observe the patched model (the embedding table shows up in `inspect-token` too). `decode` and `tokens` accept and validate the file but cannot be affected — they never load weights.

```bash
amql-cli generate ./containers/Qwen3.5-0.8B --prompt "The capital of France is" \
  --tokenizer ./Qwen3.5-0.8B --patch patches/capital.safetensors --logits 5

amql-cli route ./containers/Qwen3.5-0.8B France Paris \
  --tokenizer ./Qwen3.5-0.8B --patch patches/capital.safetensors --top 5

amql-cli path ./containers/Qwen3.5-0.8B France Paris \
  --tokenizer ./Qwen3.5-0.8B --patch patches/capital.safetensors

amql-cli inspect-token ./containers/Qwen3.5-0.8B 9419 \
  --tokenizer ./Qwen3.5-0.8B --patch patches/capital.safetensors --neighbors 5
```

A patched `route` run prints the same template-scored links and causal weights as the clean run, but measured on the patched weights — the scores shift where the patch touches the mechanism.

### Saving a patch as a LoRA adapter (`save-lora`)

`save-lora` factors a patch's 2-D deltas into a LoRA adapter **for the original model**: each delta becomes a pair of low-rank matrices (`lora_A` r×in, `lora_B` out×r) via a truncated SVD, with the standard `alpha / r` scaling. Applying `scale · lora_B · lora_A` to the base weights reproduces the patch — exactly, when the delta's rank does not exceed `r`. 1-D deltas (norm scales) are not linear-layer LoRA targets and are skipped with a note. The adapter directory is ready for a future live-adapter runtime: `adapter_model.safetensors` plus `adapter_config.json`, whose `targets` array maps tensor names to the `lora_A.N`/`lora_B.N` pairs and records each target's rank and reconstruction error.

```bash
amql-cli save-lora patches/capital.safetensors --out lora/capital \
  --rank 8 --alpha 16 --container ./containers/Qwen3.5-0.8B
```

```
LoRA: lora/capital   rank 8 (scale alpha/r = 16/8 = 2)   model Qwen3.5-0.8B
  target.decoder_stack/14.self_attn.q_proj.weight [2048x2048] → r=8  lora_A.0 lora_B.0  (reconstruction error 1.2e-06)
apply to the base container: for each target, add scale · lora_B · lora_A to the tensor.
```

`--container` is optional; when given, the patch is shape-validated against the container's tensors before factoring.