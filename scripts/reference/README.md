# Streamed transformers references

`stream_reference.py` runs a model through transformers, the model authors' own code, which makes it the authority AMQL's
runtimes are checked against. Loading the whole model is not needed, so a model far larger than memory still gets a
reference: Inkling-2-Mini's experts alone are 53 GB once dequantised to bf16, but its reference runs in a few GB.

transformers is used only to make references, during setup. Nothing in AMQL depends on it.

## How it stays small

- The model is built on PyTorch's meta device, which allocates no weights.
- Every checkpoint tensor is mapped once to the parameter it fills, by transformers' own conversion rules: the same
  renames, merges and dequantisation that `from_pretrained` applies.
- The decoder stack (the `ModuleList` holding most of the parameters) gets each layer's weights only while that
  layer runs, through transformers' own loader. A small cache (`--cache-gb`) keeps the most recently used layers,
  for a looped core or the next decode step, and frees the rest.
- Everything outside the stack (the embedding, the head, the final norm, a loop's mixing) is loaded once.
- Anything the checkpoint does not hold, such as buffers computed when the model is built, is filled by the model's
  own init, as `from_pretrained` does for its missing keys.
- What is gathered is reduced as it is gathered: per layer visit and position a few statistics and a 16-number
  fingerprint, running statistics across prompts, counts of the routers' choices, and logits as top-k (the full row
  only for each prompt's last position).

## Use

```bash
python -P scripts/reference/stream_reference.py <model dir> <out dir> \
    --prompt "The capital of France is" --chat "What is 17 times 23?" \
    --dtype bfloat16 --threads 8 --generate 8 --cache-gb 3
```

Run it at low priority, and alone: it is a heavy CPU job.
- Use `python -P`, not `-I`: `-I` hides the user site-packages where torch and transformers are installed.
- `--ids 1,2,3` feeds ids without a tokenizer.
- `--full-logits` writes every position's logits (for small vocabularies).
- `--dtype float32` computes in f32.

It writes:

| File | Holds |
|---|---|
| `summary.json` | Config facts, each prompt's text and ids, per position the top-k ids and logits, log-sum-exp and entropy, a greedy continuation, the run's time and peak working set |
| `visits.json` | Per prompt, per layer visit (each time a decoder layer runs, in order, so a looped core appears once per pass) and per position: RMS, mean, max \|x\|, the first 8 coordinates, and the fingerprint, a projection on 16 fixed random unit directions seeded by the width |
| `aggregates.json` | Per visit, running mean, deviation and maximum of the RMS and max \|x\| over every prompt and position; per router, how often each expert was chosen |
| `<prompt>.last.f32` | The last position's full logits, float32 little-endian |
| `<prompt>.all.f32` | Every position's logits, with `--full-logits` |

## Checked

`check_against_golden.py` compares a streamed run with `make_arch_zoo.py`'s goldens, which are transformers'
whole-model logits.

- **Memory:** Inkling-2-Mini (24 layers, 48 layer visits with its loop, NVFP4 and MXFP8 experts) ran two prompts and 8 greedy tokens each in 34 minutes, with a peak working set of 6.27 GiB and about 2 MB written.
- **Logits:** of the arch zoo's 40 models, 39 match to within 4e-5 in f32, with the cache set to nothing so that
  every layer is loaded each time it runs. The 40th, `qwen4_exp-nvfp4-tiny`, also fails in plain transformers 5.24.1.
- **Greedy decoding:** the streamed tokens are the same as whole-model `generate` on Inkling 2 looped and Gemma 4 MoE.
