# AMQL - C# implementation of VIndex3 (Larql)

This was a port of the VIndex3 implementation, along with support for generating it from a model (Qwen 3.5-3.8 primarily) and then allowing model independent inference, token relationship route following and exploration of the model internals in order to do some research into direct model manipulation and patching, with live LORA adapters in custom inferencing.

This implementation supports model merging (using reinforcement blending to avoid training time but get the same resultant effect as if the training sets of the two models had been combined and run) and custom tensor editing features for another project.

Credit for the design of the VIndex3 goes to Chris Hay.

## What is this for?

While building a continuous cognition platform, I ran into the problem that all current LLMs have flaws and no way to self-improve. This project is intended to support AI self-improvement and the libraries are used by my orchestrator platform.

It lets you turn a model into a graph database, then query the relationships between tokens (or their text representations). Once the edges in the graph have been identified, it becomes possible to generate a specific LoRA adapter that adjusts that behaviour in the model (or rewrites the base model), effectively editing its input and output knowledge.

At the current level, it's possible to remove the concept of something being associated in a particular way, or to add a new association between two items in a familiar form (PlaceA is the capital of PlaceB) - useful for correcting flaws in the embedding layer. It's also possible to correct a relationship between two things where a relationship exists but is of the wrong type.

At the next layer of abstraction, it's also possible to adjust relationships that humanity hasn't yet described linguistically but normally infers a connection between. This is especially useful when combined with positive and negative reinforcement derived from internal traces taken during a model's inference stage. The intent is to correct things ranging from hallucinations (where the relationship is a 'user satisfaction' signal added during post-training) to incorrect tool selection when operating agentically.

By blending models we're able to increase the token-space and apply the other models problem space vector into an existing model - this allows adding concepts and solution space vectors.

Combined with an automatic self-learning process, this should extend a model's understanding and intelligence beyond currently trainable human textual representations, in cases where a known construct would be better applied. That's an outcome fine-tuning alone can't achieve, since it can only generate more intelligent outcomes by relying on scenarios that are inherently gated at human-level intellect.

Generating genuinely novel solution vectors for a problem space is a separate challenge requiring its own approach, but this project does allow alternate solution vectors to be applied once they've been identified and the merging of solution vectors and knowledge between models.

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
amql-cli export <container-dir> --out <checkpoint-dir>
                [--patch <patch.safetensors>] [--quant mxfp4]
amql-cli layers <container-dir> [--component target]
amql-cli import <container-dir> <model> --out <merged-dir>
                [--container]
amql-cli moe-ify <container-dir> --out <moe-dir> --text <corpus.txt>
                [--experts 8] [--top-k 2] [--sample 4096] [--eval 1024]
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

### Exporting the container back to an original model (`export`)

`export` is the inverse of `encode`: it materialises a plain HF checkpoint directory (`config.json` + `model.safetensors` + `tokenizer.json`) from the container, so the model leaves the VINDEX3 world as an ordinary model again. HF tensor names are rebuilt from the graph's source bindings; `config.json` is regenerated from the judged graph facts (operator table, surface geometry, rope/position, vocabulary), and any `--patch` deltas are **baked into the stored tensors** — widened to f32, the delta added, then re-encoded to the tensor's own dtype (BF16/F32/…). Tensors a patch never touches are copied byte-identically, so an unpatched export is byte-exact; an operator without a judged `layer_types` spelling refuses the export by name rather than being approximated, and the tied output head is skipped with a note (it reuses the embedding table). Pass `--quant mxfp4` to export the
model as a 4-bit quantized checkpoint instead — see the "Exporting a quantized checkpoint"
section below.

```bash
# bake every delta in patches/capital.safetensors into the weights
amql-cli export ./containers/Qwen3.5-0.8B --out ./models/Qwen3.5-0.8B \
  --patch patches/capital.safetensors

# the exported directory is a normal checkpoint — re-encode to round trip
amql-cli encode ./models/Qwen3.5-0.8B --out ./containers/Qwen3.5-0.8B-rebuilt
```

```
exported:  models/Qwen3.5-0.8B
model:      Qwen3.5-0.8B
tensors:    685  (1.39 GiB)
note:       object 'target.output_head': carried only (no materialised tensors) — skipped
wrote:      model.safetensors, config.json, tokenizer.json
```

### Describing the layers (`layers`)

`layers` lists every component of the container, then — for the selected one (`--component`, default `target`) — the per-layer attention policy table (operator, span/window, position policy, head geometry) from the graph's authority table, the per-layer tensor inventory (name, stored dtype, shape) from the decoder-stack segment, and whether the planner serves the stack or refuses it by name:

```bash
amql-cli layers ./containers/Qwen3.5-0.8B
```

```
container: containers/Qwen3.5-0.8B   model 'Qwen3.5-0.8B' (qwen3_5_text)

component 'target' role=PrimaryText source=model layers=28 hidden=2048
  attention:  linear_attention × 4, softmax × 24
...
  layers:
    L 0: linear_attention  full      position none (NoPE)  heads 4×128
    L 1: softmax           full      position partial rope θ=10000 f=0.25  heads 16×128
    ...
  tensors of 'target.decoder_stack':
    L 0: linear_attn.in_proj_qkv.weight BF16 [1024x2048]; linear_attn.in_proj_z.weight BF16 [512x2048]; ...
    L 1: self_attn.q_proj.weight BF16 [2048x2048]; self_attn.k_proj.weight BF16 [512x2048]; ...
  runtime: [refused] layer 0: linear_attention has no judged runtime — the planner refuses it by name
```

### Exporting a quantized checkpoint (`--quant mxfp4`)

`export --quant mxfp4` produces an MXFP4 checkpoint (the OCP microscaling standard):
the stack's projection matrices (`q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/
`up_proj`/`down_proj`/linear-attention projections) are quantised to FP4 E2M1 elements
packed two per byte, with one FP8-E8M0 scale (a pure shared exponent) per 32-element
block — FP4 codes at a quarter of the BF16 size plus negligible E8M0 scales, so each
projection lands at roughly a quarter of its BF16 payload. Embeddings, norms, biases,
the log-space `A_log` tensors and the output head keep their full precision, per the
standard practice.

```bash
amql-cli export ./containers/merged --out ./models/merged-mxfp4 --quant mxfp4
```

```
tensors:    505  (1.36 GiB)
note:       186 stack projection tensors exported as MXFP4 (FP4 E2M1 grid elements, per-32-element F8_E8M0 scales) — embeddings, norms, biases and the output head keep their full precision
wrote:      model.safetensors, config.json, tokenizer.json  (quantization: mxfp4)
```

Each quantised weight becomes two safetensors tensors: `...weight` (dtype `FP4`, logical
shape, two elements per byte) and `...weight_scale` (dtype `F8_E8M0`, one per
32-element block, laid out per row). Dequantisation is `x ≈ DecodeFp4(q) × DecodeE8M0(scale)`,
and the element grid `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` is serialised into `config.json`
(`quantization_config.quant_method: "mxfp4"`) so the exact scheme is self-describing.
The quantised checkpoint is a terminal artifact for this build (the encoder reads
full-precision dtypes); the `Mxfp4` codec in `Amql.Safetensors` is the reference
implementation for any consumer runtime.

### Merging a second model into the container (`import`)

`import` merges another model into an existing container: both models end up in one VIndex3, and `export` materialises the result as a single checkpoint. It does this through three pieces:

- **The tokenization mapping layer.** The two tokenizers' vocabularies are related by token string (ids are opaque; identical strings are the only relationship this build judges). The merged vocabulary keeps the base model's ids stable and appends the imported-only tokens with fresh ids.
- **Anchored alignment of the token interface.** The embedding (and the output head, when untied) grow to the union vocabulary, and every row is built **in the scaffold model's space** (the model whose stack actually runs — the larger shape). A ridge least-squares map is fitted from the *other* model's space into the scaffold's on the shared-token anchors. Shared tokens then pass a **consensus gate** — the per-token cosine between the scaffold's row and the other model's aligned row: where the two models **agree**, the rows are blended and the result **restored to the scaffold's full energy** (consensus is reinforced, not averaged away); where they **disagree**, the scaffold's own row is kept verbatim instead of fabricating a direction neither model holds. Other-model-only tokens are mapped through the fit; scaffold-only tokens stay verbatim. Every merged row is re-encoded to the merged storage dtype (F32/BF16/F16).
- **Shape evolution of the stack.** The model with the larger shape (hidden size, then layers) scaffolds the result — its intermediate tensors are copied byte-identically, since differently-shaped stacks cannot be averaged. A tensor kind the scaffold lacks (e.g. `q_norm`) is grown in from the other model, zero-padded. **Per-layer provenance** records the source and operation of every tensor.

`token-map.json` in the merged container is the relationship tracker: every token's kind (`blended` / `aligned_new` / `base_only`) and both source ids, the anchor count, alignment residual and per-token consensus statistics (mean agreement, share above the gate), the per-layer provenance table, and the scaffold. The replaced (non-scaffold) stack is preserved under `segments/source/` so both models' data live in the same container. The union tokenizer replaces the base's, so every command that reads the vocabulary keeps working unchanged.

```bash
# import a bigger/hybrid model into the existing 0.8B container
amql-cli import ./containers/Qwen3.5-0.8B ./models/Qwen3.6-2B \
  --out ./containers/Qwen3.5-0.8B-x-Qwen3.6-2B

# ...or when the second model is already a container
amql-cli import ./containers/Qwen3.5-0.8B ./containers/other \
  --out ./containers/merged --container

# layers shows the merged provenance; export gives the plain checkpoint
amql-cli layers ./containers/merged
amql-cli export ./containers/merged --out ./models/merged
```

```
imported:   Qwen3.5-2B into Qwen3.5-0.8B
result:     Qwen3.5-0.8B+Qwen3.5-2B
scaffold:   Qwen3.5-2B  (hidden 2048, layers 24)
vocab:      248044 + 248044 → 248044 (248044 blended, 0 aligned-new, 0 base-only)
storage:    BF16 × 2048, head tied
preserved:  3 segments of the replaced stack under segments/source/
note:       base embedding 1024→2048: rows zero-extended to the merged width
note:       alignment: 248044 shared-token anchors, residual L² 402.1
wrote:      index.json, system_graph.json, token-map.json, tokenizer.json, segments/
the merged container tracks every token relationship in token-map.json and exports as one model:
  amql-cli export ./containers/merged --out <checkpoint-dir>
```

Both Qwen3.5 sizes ship the same 248,044-token vocabulary, so an 0.8B→2B merge is all-blended; merging a model family with a *different* tokenizer produces `aligned_new` rows for its extra tokens and `base_only` rows for the base's, with the new ids appended after the base vocabulary.

### Restructuring a dense model into a routed MoE (`moe-ify`)

`moe-ify` turns a dense transformer container into a mixture of experts in the MoEfication
lineage — no training, quality gated by perplexity:

```bash
amql-cli moe-ify ./containers/Qwen3.8-27B --out ./containers/Qwen3.8-27B-moe \
  --text ./corpus/wikipedia.txt --experts 8 --top-k 2 --sample 4096 --eval 1024
```

It samples FFN inputs through the model (a runtime activation seam), clusters each layer's
intermediate units by co-activation into `--experts` **balanced** groups (k-means, cosine),
slices the gate/up rows and down columns into per-expert tensors, and materialises a linear
per-layer router (each expert's row is its cluster's mean gate direction — the affinity
`x · routerᵀ` the routed kernel scores). The container is rebuilt with
`mlp.router.weight` + `mlp.experts.{e}.{gate,up,down}_proj.weight` per layer, so the
planner judges every FFN routed off-the-shelf and the existing top-k expert kernel runs it
— no new runtime. Each expert's tensors are its own segment, so experts are separable for
offload or distributed placement, and `export` (including `--quant mxfp4`) works unchanged.

The held-out perplexity gate prints `dense → moe`. Measured on the 2-layer demo container
(top-1, 2 experts): `dense 49.96 → moе 49.88 (−0.2%)`. The 27B is the real measurement —
expect a larger PPL cost at sparsity, and the per-expert slices are a natural starting point
for MoE-adaptation fine-tuning.

```
moе-ified:  demo-model-moe2x1
routing:    2 experts × top-1 (expert intermediate 4) over 2 layers
note:       sampled 6 tokens; 2 layers clustered into 2 balanced experts of 4 units
note:       perplexity over 4 held-out tokens: dense 49.96 → moе 49.88 (-0.2%)
wrote:      index.json, system_graph.json, segments/, tokenizer.json
```

### Pruning a container down to a byte budget (`prune`)

`prune` shrinks a container to a target total on-disk size by dropping **whole decoder
layers** — the only dimension a merge actually grows that can be removed without touching
the vocabulary. The vocabulary (embedding/head), the tokenizer, the token-map, and every
non-layer file are copied byte-identically; the stack segment is rebuilt without the
dropped layers, the kept layers are renumbered 0..K−1, and the graph, index and token-map
layer table are rewritten to match. The result is a **complete, mergeable container**: the
planner runs it, `export` materialises it, and `import` accepts it as either side of a
merge — which is what makes the prune→merge loop below work.

```bash
# drop as many layers as needed to get the whole container under 14.5 GiB
amql-cli prune ./containers/merged --out ./containers/merged-pruned --target-bytes 14.5GiB

# corpus-ranked: keep the layers whose removal changes the hidden state least
amql-cli prune ./containers/merged --out ./containers/merged-pruned \
  --target-bytes 14.5GiB --approach corpus --text ./corpus/wikipedia.txt

# seeded random baseline for ablations
amql-cli prune ./containers/merged --out ./containers/merged-pruned \
  --target-bytes 14.5GiB --approach random --seed 7
```

`--target-bytes` accepts bare bytes or `B`/`KB`/`MB`/`GB`/`TB`, `KiB`/`MiB`/`GiB`/`TiB`
suffixes (case-insensitive; `--target` is an accepted alias). The three approaches rank
which layers drop first, and the prune takes the **smallest prefix of that order** that
gets under the budget (never below `--min-layers`, default 1):

- **`provenance`** (default, deterministic, no corpus) — reads `token-map.json`'s
  per-layer provenance: layers whose tensors were *grown* (zero-padded from the other
  model during the merge) drop first as the least authentic, then by position — the top of
  the stack goes before the input-proximal layers. Without a token-map every layer ties
  and the ranking is purely positional.
- **`corpus`** — one forward pass over `--text` (needs a BPE tokenizer beside the
  container, which `encode` copies in) scores each layer's residual delta
  ‖h(l+1) − h(l)‖², mean over positions and tokens; the layers that change the hidden
  state least drop first. One-shot: scores come from the full stack, they are not
  re-measured after each drop.
- **`random`** — a seeded uniform layer order (`--seed`, default 42) as the ablation
  baseline.

Selection and the commit are exact: each candidate's rebuilt stack segment and JSON files
are sized before anything is written, the exact total is re-verified before the copies
land, and a target below the `--min-layers` floor refuses with the floor size rather than
shipping an over-budget container. The dropped layers are recorded in the index
(`prune.dropped_layers`), and an MTP drafter carried by the container is not a blocker —
its plain note records that its trunk (a copied layer) is now stale.

```
pruned:    ./containers/merged-pruned
model:     Qwen3.5-0.8B+Qwen3.5-2B
approach:  provenance     seed: 42
layers:    24 → 14  (dropped 10: 14,15,16,17,18,19,20,21,22,23)
bytes:     20.10 GiB → 13.20 GiB  (target 14.00 GiB, saved 6.90 GiB)
note:      an MTP drafter is present and its trunk is now stale …
the pruned container is a complete model — verify, export and import it as usual:
  amql-cli verify ./containers/merged-pruned
  amql-cli import ./containers/merged-pruned <smaller-model> --out <result>
```

The workflow this enables: merge two large models, prune the result to a target size, then
merge a smaller model into the pruned container — the prune output participates in `import`
as any complete container, so the chain closes on a fresh, verifiable merge.

### The MTP drafter and vision tower on export

The encoder **materialises the carried modules** whenever the source checkpoint holds
them: the `mtp.` tensors (fc projector, pre-fc norms, the single trunk layer, the pre-head
norm) land in a segment under `mtp.stack`, and the `model.visual.*` tower under
`vision.perception_tower`. The container becomes self-contained for the whole model;
without the source tensors the objects stay carried, per the "leave it" rule.

`export` then produces the complete artifact set:

- **`model.safetensors`** — the text model **and** the vision tower (it is part of the
  model), under `model.visual.*`, with `vision_config` carrying the tower's judged facts.
- **`mtp.safetensors` + `mtp.config.json`** — the MTP drafter, exported **automatically
  alongside** when materialised: the module under its original `mtp.` names composed with
  the shared embedding and head (`mtp_use_dedicated_embeddings: false`), a standalone
  checkpoint for speculative decoding. `export-mtp` emits it alone into its own directory.

```
tensors:    1184  (50.96 GiB)       note:       17 mtp drafter tensors exported alongside as mtp.safetensors (5.9 GiB)
wrote:      model.safetensors, config.json, mtp.safetensors, mtp.config.json, tokenizer.json
```

### Generating an MTP drafter for a model without one (`generate-mtp`)

Models that ship no MTP (e.g. Qwen3.5-0.8B) can get one **generated entirely from their
own weights** — the bootstrapping init the MTP line uses, no training:

```bash
amql-cli generate-mtp ./containers/Qwen3.5-0.8B --out ./containers/Qwen3.5-0.8B-mtp \
  --text ./corpus/wikipedia.txt --sample 1024
```

- the **trunk layer** is a verbatim copy of the model's **last full-attention layer**
  (the MTP trunk is always a softmax layer), self-attn + MLP + both norms;
- the **two `pre_fc_norm_*` norms and the drafter's `norm.weight`** copy the model's
  final norm;
- the **`fc` projector [2h → h]** boots deterministically as the mean-combination
  `0.5·(norm(h_t) + norm(e_{t+1}))`;
- the head and the conditioning embedding are the model's **shared tables**
  (`mtp_use_dedicated_embeddings: false`).

The module materialises as `mtp.stack` in a new container, so `export` emits
`mtp.safetensors` + `mtp.config.json` automatically. The **zero-shot draft-acceptance
gate** runs the drafter's real pipeline — pre-fc norms, projector, one trunk attention+FFN
pass, shared head — against the model's hidden states over `--sample` corpus tokens and
reports the share of positions where the drafter's top-1 second-next-token draft matches
the model, honestly labelled as a warm start: trained adapters (frozen model) are the
next step.

```
generated:  Qwen3.5-0.8B + MTP drafter
trunk:      a copy of full-attention layer 23 (15 module tensors)
note:       zero-shot draft acceptance over 794 positions: 0.3% — the bootstrapped
            drafter is a warm start; training (frozen model) is the next step
```

(Measured on the real Qwen3.5-0.8B. A 0.3% draft acceptance is above the random baseline
but far below a useful speculative drafter — the untrained boot is a
structure-and-warm-start deliverable, and the frozen-model adaptation pass is the
follow-up that turns it into a real drafter.)

### The calculated fit — "pre-trained" by calculation, no training

The follow-up replaces "training (frozen model)" with **calculation**: one forward pass
collects the model's own continuation map, and a **closed-form ridge least-squares solve**
fits the drafter's free block — no autograd, no SGD, no random seeds in the learning path
(same inputs → byte-identical weights; verified by a determinism test). Three commands
run it, or `generate-mtp --fit` runs the whole pipeline in one shot:

```bash
amql-cli generate-mtp ./containers/Qwen3.5-0.8B --out ./containers/Qwen3.5-0.8B-mtp \
  --text ./corpus/wikipedia.txt --sample 4096 --fit --eval 1024
```

1. **`collect-mtp` (phase 0)** — one dense forward over the corpus exports the pairs
   `x_t = concat(pre_fc_norm_hidden(final_norm(h_t)), pre_fc_norm_embedding(e_{t+1}))`
   (2h) and the regression target `y_t` = the model's **pre-final-norm residual at t+1**
   (h), with the **fit/gate split fixed in the sink** — the acceptance gate is never
   measured on fitted positions.
2. **`fit-mtp` (phase 1)** — the single ridge projector `W* = (XᵀX+λI)⁻¹XᵀY`
   (`λ` relative `1e-4`), written back into `mtp.stack` as `fc.weight`, re-encoded to the
   segment dtype.
3. **`fit-mtp --sweep 1,4,8` (phase 2)** — the **K-projector mixture**: the fit rows are
   clustered into K balanced groups (deterministic farthest-first k-means), one ridge
   projector is fit per cluster, and a linear router (L2-normalised cluster-mean rows, the
   moe-ify convention) picks the expert per row. The free block materialises as
   `fc.router.weight` + `fc.experts.{{e}}.weight` — the export and the acceptance gate
   resolve the routed structure by name, so it survives export → re-encode. The held-out
   acceptance gate decides the shipped K.

**Measured on the real Qwen3.5-0.8B** (512 fit + 256 held-out positions, split fixed in
the collector; the gate runs the drafter's real pipeline): boot 0.0% → fitted **14.8%**
(K=1) → **16.4%** (K=4) → **16.8%** (K=8, shipped — the gate's winner). R² over the fit
split is ≈1.0 (the system is under-determined at 512 rows < 2h dims — the held-out gate,
not R², is the evidence of generalisation; K=1's in-memory 14.8% reproduces the
materialised container's number exactly). The demo model tie-breaks to fewer clusters
(1), so ties at equal acceptance prefer the smaller K. `verify` passes on the fitted
container and `export` emits the drafter companion automatically.

```
fit:        K ∈ {1, 4, 8}: K=1 R² 1.000 acc 14.8%; K=4 R² 0.917 acc 16.4%; K=8 R² 0.941 acc 16.8%
note:       shipped K=8 (the acceptance gate's winner; ties prefer fewer clusters); weights 3e30e9c8…
note:       held-out draft acceptance over 256 positions: boot 0.0% → fitted 16.8%
```
