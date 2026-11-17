# Branch-adapt-route on Qwen 3.5 0.8B

This experiment trains topic experts separately, then routes between them as a mixture of experts, following Ai2's [BAR](https://allenai.org/blog/bar) (Morrison et al., arXiv 2604.18473):

- **Branch:** one expert per topic, each trained on its own data.
- **Adapt:** the base FFN is kept as a frozen *anchor* expert.
- **Route:** a linear router per layer is trained with everything else frozen.

BAR reports that this beats weight averaging and BTX. The run here reaches the same ordering on a 0.8B model with LoRA experts, entirely in AMQL.

## Setup

**Topics.** `scripts/moe/make_topic_sets.py` builds four topics from the post-training data:

| Topic | Source |
|---|---|
| code | `reasoning_off`, regex-filtered |
| math | `reasoning_off`, regex-filtered |
| structured | `structured_outputs`; the reply opens with JSON, XML or a fence; long prompts are cut in the middle |
| chat | `chat_if` |

Each topic has 256 train, 16 route and 24 eval examples. Prompts are capped at 320 tokens and completions at 96, rendered with Qwen's template with thinking off.

**Experts.** Each expert is a `lora-train` LoRA on `down_proj` of layers 18–23:

- rank 8, α 16, lr 5e-4
- 60 steps of batch 16, with `--keep-best`

Each expert shares the base's gate and up projections, so expert *e* computes `base + ΔW_e·act`.

| Expert | Held-out loss on its topic, base → expert |
|---|---|
| code | 1.0602 → 1.0225 (best at step 15, then overfit) |
| math | 0.7996 → 0.5675 |
| structured | 0.9261 → 0.7928 |
| chat | 2.0012 → 1.8291 |

**Router.** Each of layers 18–23 has a router: one linear map, hidden → 5 logits for anchor, code, math, structured and chat, followed by a softmax. There are two ways to train it:

- `--method loss`, BAR's: the LM loss on the routing split, with Adam, 30 steps, lr 3e-3. dL/dlogit is exact at layer 23. Lower layers use the direct residual path, which points downhill but overshoots by about 2×.
- `--method label`: logistic regression from each layer's FFN input to the example's topic.

Each method runs with one of two hold modes:

- `--hold token` routes every row.
- `--hold prompt` holds the prompt's mean logits for the whole reply.

## Results

The table shows mean completion loss (nats per token) on the 96 held-out examples, 24 per topic. "Gap closed" is how much of the distance from the base to the oracle a mix recovers; the oracle gives each example its own topic's expert.

| Mix | code | math | structured | chat | mean | gap closed |
|---|---:|---:|---:|---:|---:|---:|
| base | 1.0602 | 0.7996 | 0.9261 | 2.0012 | 1.1968 | 0% |
| only:math (best single) | 1.0614 | 0.5675 | 0.9541 | 1.9689 | 1.1380 | 41% |
| task arithmetic (sum) | 1.3573 | 0.7541 | 0.8645 | 2.1510 | 1.2817 | −59% |
| weight averaging | 1.0323 | 0.6743 | 0.8441 | 1.9057 | 1.1141 | 58% |
| loss, per token (BAR) | 1.0485 | 0.5705 | 0.7913 | 1.8670 | 1.0693 | 89% |
| loss, held from prompt | 1.0391 | 0.5748 | 0.7891 | 1.8532 | 1.0640 | 92% |
| label, per token | 1.0288 | 0.5732 | 0.7899 | 1.8622 | 1.0635 | 93% |
| **label, held from prompt** | **1.0225** | 0.5727 | 0.7927 | 1.8389 | **1.0567** | **97%** |
| oracle | 1.0225 | 0.5675 | 0.7928 | 1.8291 | 1.0530 | 100% |

Each router row is the dense mix of all five experts. Top-2 matches it to within 0.001, and top-1 costs 0.000–0.008.

What the results show:

- **Routing beats merging, as BAR found.** Weight averaging closes 58% of the gap; the BAR router closes 89%. Task arithmetic, the sum of the four deltas, does worse than the base: the structured delta hurts code by 0.10, and the sum stacks it at full strength.
- **The topic is easy to read.** By layer 20 a linear probe on the FFN input gets 100% of 7,831 routing rows right, and 98.7% at layer 18.
- **Labels beat the LM loss here.** That reverses BAR's choice, for a reason: the loss router barely uses the code expert, because code's gain is small (−0.04). Its usage goes mostly to math and structured, and code ends 0.026 worse than the oracle. On 64 routing examples, labels are the stronger signal. With more routing data, or experts with larger gains, the loss should catch up, and it needs no labels.
- **Holding the routing from the prompt helps every router.** A reply's topic is fixed by its prompt; per-token routing lets the reply's own tokens drift.
- **The merged container is the same model.** `moe-merge` with the loss/token router writes `Qwen3.5-0.8B-bar5x5` with 5 experts on layers 18–23. Scored as a plain container, its losses match the bench's to four places: 1.0485, 0.5705, 0.7913, 1.8670.

### Adding a topic

BAR's modular claim is that a new expert can be added without retraining the old ones. To test it:

1. Train a router over anchor + code, math and structured (`loss3`).
2. Add chat: `--init loss3` keeps the old rows by name and starts chat's row at zero.
3. Train for 10 more steps.

| Mix | code | math | structured | chat | mean |
|---|---:|---:|---:|---:|---:|
| loss3 (no chat expert) | 1.0476 | 0.5704 | 0.7890 | 1.9775 | 1.0961 |
| loss3 + chat, 10 steps | 1.0484 | 0.5723 | 0.7961 | 1.9089 | 1.0814 |
| loss, all four from scratch, 30 steps | 1.0485 | 0.5705 | 0.7913 | 1.8670 | 1.0693 |

The old topics hold, within 0.007. After 10 steps, chat has closed 46% of its gap; training from scratch closed 75% in 30 steps.

## Limits

- **Experts touch only `down_proj` on the top six layers.** The trainer's gradient is exact only at the last layer. Because the experts share gate and up, the mixture is linear in the routing weights: one FFN pass plus a rank-8 delta per expert. A full BTX-style expert, with its own gate and up, is a heavier but straightforward extension.
- **Routed layers run on the CPU.** The device engine has no routed-FFN kernel yet, so the merged container is correct but slow. `moe-merge` materialises a full f32 down-projection per expert (14.7 MB each, 2.3 GB for the merged container).
- **Prompt-held routing cannot be merged.** A container routes per row, so `moe-merge` refuses a prompt-held router; its gains are available only through the bench. Carrying it would need a per-sequence routing state, much like Entail's `carry`.
- **The numbers are small-sample.** There are 24 eval examples per topic and 64 routing examples. Differences under ~0.01 are within noise.

## Reproduce

```bash
python scripts/moe/make_topic_sets.py containers/qwen3.5-0.8b <post-training-dir> containers/moe/data2 --train 256 --scan 400000
for t in code math structured chat; do ./bin/amql-cli.exe lora-train containers/qwen3.5-0.8b --data containers/moe/data2/$t.train.jsonl --eval containers/moe/data2/$t.eval.jsonl --out containers/moe/experts2/$t --layers 18,19,20,21,22,23 --rank 8 --alpha 16 --lr 5e-4 --steps 60 --batch 16 --eval-every 15 --keep-best --target-loss 0 --grad-checks 0 --no-decode; done
sh scripts/moe/run_bar.sh
```

On the CPU path each expert takes 35–60 minutes and `run_bar.sh` about 4.5 hours, most of it the two loss-trained routers (70–80 minutes each).
