# Decision Models — Von (ModernBERT option-marker)

**Status:** Import and export implemented and verified on the real checkpoint. Inference is not served — the runtime executes causal decoders only.
**Reference checkpoint:** [`wfzyx/von`](https://huggingface.co/wfzyx/von) (Von 1.2, revision `5df8185a`), code at [`wfzyx/von`](https://github.com/wfzyx/von).

## 1. What Von is — and why "BERT support" did not already cover it

Von is a non-autoregressive decision model: the premise and K candidate options are packed into
one sequence, each option opened by a `[MASK]` marker, and one encoder pass scores every option.
It is **not** covered by the existing encoder path:

| | nomic-bert (existing) | Von |
|---|---|---|
| backbone | `nomic_bert`, `encoder.layers.N.*` | **ModernBERT-large** (`modernbert`), bare `layers.N.*` |
| attention | every layer global | global every 3rd layer (RoPE θ 160 000), the rest **sliding ±64** (θ 10 000) |
| FFN | SwiGLU | **GeGLU**, fused `Wi` = `[input; gate]` |
| norms | LayerNorm with bias | LayerNorm **without bias**; an extra one after the embedding; layer 0 has **no** attention norm |
| head | none (pooling) | **option-marker scorer** in a separate PyTorch file |

Before this change, pointing `encode` at Von would have failed the text-prefix probe — and the config
reader would have silently defaulted ModernBERT's `hidden_activation: gelu` to SiLU.

## 2. The checkpoint, as shipped

| File | Contents |
|---|---|
| `model.safetensors` | the ModernBERT backbone, 170 F32 tensors, `ModernBertModel` names |
| `option_marker.pt` | a `torch.save` of the whole `OptionMarkerModel` state dict: the same 170 tensors under `encoder.`, plus 8 under `scorer.` |
| `marker_calibration.json` | fitted temperature map, `independent_options: true`, zero-shot noul prior |
| `calibration.json`, `tokenizer.json`, `tokenizer_config.json`, `config.json` | as usual |

The Von SDK builds `AutoModel` from `model.safetensors`, then `load_state_dict(strict=True)` from the
`.pt` — so **the `.pt` is what actually runs**. On the published weights its `encoder.` copy is
byte-identical to the safetensors backbone (checked for all 170 tensors).

The scorer is `LayerNorm(1024) → Linear(1024, 512) → GELU → LayerNorm(512) → Linear(512, 1)`, applied to
the final hidden state at each `[MASK]`; a softmax over the per-option logits is the answer.

## 3. Import (`amql-cli encode <von-dir> --out <container>`)

- The backbone becomes `target.embedding`, `target.encoder_stack` and `target.final_norm`.
- The scorer becomes `target.option_marker_head` (kind `classifier_head`), bound to artifact `option_marker.pt`, prefix `scorer`.
- The `.pt` is read by `TorchCheckpoint` (`Amql.Safetensors`), a Python-free reader for the torch zip format
  whose pickle machine accepts only the globals a state dict needs (the `weights_only=True` idea). Anything
  else — e.g. a pickle naming `os.system` — is refused by name, never executed.
- The `.pt`'s `encoder.` copy is compared byte-for-byte with `model.safetensors`. If they differ, the import
  **refuses**: the container would otherwise hold weights Von does not run.
- Per-layer table: `span: full` + θ 160 000, or `span: sliding`, `window: 128` + θ 10 000.
- New surface facts (`system_graph.json`): `execution.encoder` (bias switches, embedding norm, layer-0
  identity norm, fused-FFN layout, special token ids) and `execution.option_marker` (scorer width, marker
  token, `independent_options`, `digit_split` from `marker_calibration.json`).
- `marker_calibration.json` and `calibration.json` travel with the container as ancillary files.
- A plain ModernBERT backbone without `option_marker.pt` imports too, just without the head. Checkpoints that
  nest the backbone under `model.` (`ModernBertForMaskedLM`, …) or carry any tensor outside the backbone
  are refused by name rather than having weights dropped.

## 4. Export (`amql-cli export <container> --out <dir>`)

Writes `model.safetensors`, a regenerated `config.json`, `option_marker.pt`, the tokenizer and the
calibration files.

- `config.json` is rebuilt from the graph in `ModernBertModel` form, carrying both the transformers 5
  spellings (`layer_types`, per-type `rope_parameters`) and the 4.x ones (`global_attn_every_n_layers`,
  `global_rope_theta`, `local_rope_theta`, `local_attention`). The base model's leftover tasksource
  metadata (`tasks`, `classifiers_size`, `id2label`) is not carried: `ModernBertModel` does not read it.
- `option_marker.pt` is written by `TorchCheckpointWriter` in `torch.save`'s own layout: stored zip, protocol-2
  pickle, storages 64-byte aligned (so `mmap=True` works), keys in module-registration order. Its `encoder.`
  half is read back from the shard just written, so the two copies cannot disagree — a `--patch` lands in both.
- `--quantize` and `--arch` are refused: a quantised shard could not match the `.pt`'s copy.

## 5. Verification

On the real `wfzyx/von` weights (torch 2.x CPU, transformers 5):

- `encode → export`: all 170 `model.safetensors` tensors and all 178 `option_marker.pt` tensors are
  bit-identical to the originals, in the same key order; `torch.load(weights_only=True)` and
  `OptionMarkerModel.load_state_dict(strict=True)` accept the export.
- Von's own `OptionMarkerModel`, loaded from the original and from the export, gives **bit-identical
  logits** for the same packed inputs, with `independent_options` on and off.
- `encode → export → encode` reproduces identical segment payload hashes and an identical system graph.

`tests/Amql.Tests/ModernBertTests.cs` covers the same round trip on a synthetic Von-shaped checkpoint,
plus the torch reader against fixtures written by real `torch.save` (bf16/f16/int64, shared-storage views
with offsets, 0-d scalars, a non-contiguous tensor that must be refused).

## 6. Not done

- **Running Von.** `verify` reports the container as a bidirectional encoder the runtime refuses; serving it
  needs bidirectional attention with the order-invariant option mask, GeGLU and the scorer.
- **Big-endian or legacy (pre-1.6) `.pt` files** are refused.
- The `.pt` export holds the whole state dict in memory while writing (≈1.6 GB for Von).
