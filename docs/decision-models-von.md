# Decision Models — Von (ModernBERT option-marker)

**Status:** Import, export and inference implemented and verified on the real checkpoint. `amql-cli decide` answers TypeSafe `/v1/decisions` requests on an in-process ModernBERT encoder (no ONNX or other runtime) with output identical to the Von 1.2 SDK.
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

## 6. Running Von: `amql-cli decide`

```bash
amql-cli decide <von-container> --request @request.json      # or inline JSON, or '-' for stdin
amql-cli decide <von-container> --request - --envelope jevai < request.json
```

The request is the body of TypeSafe's native decisions endpoint — `POST /v1/decisions` on
[jevai.org](https://www.jevai.org/docs), `POST /v1/systemone` on [openjev.sh](https://openjev.sh/docs):

```json
{
  "model": "openjev",
  "state": { "customer_message": "I was charged twice for order ord_7429.", "duplicate_charge_usd": 680,
             "policy": "Refunds above USD 500 require human approval." },
  "questions": {
    "action": { "type": "choice", "instructions": "Choose the safest next action.",
                "criteria": { "allow": "Issue the refund immediately.",
                              "review": "Require human approval before issuing the refund.",
                              "deny": "Reject the refund request." } },
    "needs_human_review": { "type": "noul",
                            "instructions": "Does this refund require human review under the stated policy?" },
    "risk": { "type": "score", "instructions": "Score the financial and policy risk.",
              "criteria": ["Low", "Moderate", "High", "Critical"] }
  }
}
```

and stdout is the response body — `{ model, answers, usage }`, or with `--envelope jevai` the jevai.org
wrapping `{ "code": 0, "message": "ok", "data": { … } }`:

```json
{"model":"von-1.2.0","answers":{
  "action":{"type":"choice","choice":"review","probabilities":{"allow":0.1905,"review":0.4549,"deny":0.3546},"confidence":0.182},
  "needs_human_review":{"type":"noul","noul":0.6292},
  "risk":{"type":"score","score":1.63,"confidence":0.03,"legend":{"0":"Low","1":"Moderate","2":"High","3":"Critical"},
          "probabilities":{"0":0.1889,"1":0.266,"2":0.2723,"3":0.2728}}},
 "usage":{"input_tokens":74,"output_tokens":3}}
```

A request the API would reject with 422 exits 2 — with `--envelope jevai` as `{ "code": 422, "message", "data": null }`
on stdout, otherwise one line on stderr. Checked: `state` is a string, object or array; `questions` is a
non-empty object; each question's `type` is `choice` / `score` / `noul` and it has `instructions`
(string, object or array); choice `criteria` is an object of 1–255 options; score `criteria` an array of
1–10 levels; noul `criteria`, if given, has only `true` / `false`. `model` is accepted and not echoed:
like the Von SDK, the response names the model actually served (`model_id` in `marker_calibration.json`).
`--attention independent|full` overrides the checkpoint's attention mode (see below); the default is the
mode the weights were trained with.

### How a request becomes encoder passes

A line-for-line port of the Von 1.2 SDK's `OptionMarkerBackend`, because every detail changes the tokens:

- **state** — a string as-is; an object as `key: str(value)` lines; an array as Python's `str()`. Python's
  rendering is reproduced exactly (`True`/`None`, quote choice, escapes, `float.__repr__`); checked against
  CPython on 400 randomised values.
- **instructions** — objects and arrays become `json.dumps(…, sort_keys=<top level is an object>)`, with
  Python's separators and `ensure_ascii` escaping.
- **sequence** — `[CLS] {instructions} {state} [SEP] [MASK] {option} [MASK] {option} … [SEP]`; each question is
  one encoder pass and each `[MASK]`'s final hidden state is scored.
- **choice** — options are the criteria descriptions (the key when a description is null or empty); the answer
  is the arg-max logit.
- **noul** — two options, `criteria.true` / `criteria.false` or "Yes, condition holds true." / "No, condition is
  false.". Without explicit criteria, a second pass with an empty state measures the model's context-free "yes"
  lean and subtracts `a·bias + b` (the fitted `noul_zero_shot_prior`, else `0.7·bias`) from the yes logit.
- **score** — one option per level, in order; `score = Σ i·p_i`, and the legend echoes each level's text.
- **probabilities** — softmax at an input-conditioned temperature from `calibration_map` (normalised entropy,
  log state tokens, option count, clamped to `[lo, hi]`); temperature never changes the arg-max.
  `confidence = (n·p_max − 1)/(n − 1)`. Rounded as the SDK does (4 places, score 2, confidence 3).
- **attention** — Von 1.2 weights run *order-invariant*: each option attends only to the premise and itself,
  and its position ids restart after the premise, so reordering options cannot change any answer.

Differences from the SDK, each deliberate: structured (object/array) choice and noul descriptions, which the
API allows but the SDK rejects, are read as the same canonical JSON as structured instructions; a score level
that is an object *without* a `what` field is kept whole as JSON (the SDK reads only `what`/`examples` and
would leave such a level empty); and a request whose text contains `[MASK]` is refused instead of being scored
as an extra option.

### The encoder

`Amql.Inference.ModernBertEncoder` is written directly in C# from the container's judged facts:
embedding lookup + LayerNorm; per layer a fused `Wqkv`, rotate-half RoPE with the layer's base (160 000
global, 10 000 local), softmax attention — global layers over every allowed key, sliding layers also within
`|pos_q − pos_k| ≤ 64` measured on the position ids — then `Wo`, and a GeGLU MLP with the exact erf GELU;
final LayerNorm. `OptionMarkerScorer` is the head. Many-row GEMMs use a 4×2 register-blocked FMA kernel
(`EncoderGemm`); the decoder keeps its own kernel and numerics. The tokenizer gained what ModernBERT's needs:
`lstrip`/`rstrip` added tokens (`[MASK]` owns the whitespace before it, matched leftmost-longest as HF does)
and the `[CLS] $A [SEP]` template post-processor — checked id-for-id against HF `tokenizers`.

Run it from a **Release** build (`dotnet build -c Release`): the vector kernels are not optimised under Debug.
On a 4-core container, loading Von takes 6–15 s (1.5 GB of f32 weights, disk-bound) and each pass over a
~70-token request about 0.75 s.

### Verification

Against the Von 1.2 SDK and transformers on the real `wfzyx/von` weights:

- 18 answers over 12 requests (string/object/array state, nested values, structured instructions, null
  descriptions, explicit and zero-shot noul, score levels with examples, 1–6 options, Unicode): every field
  **identical** to the SDK's, `usage` included.
- 331–1 246-token requests, both attention modes (the modes give different answers there, so the sliding window
  and option isolation are genuinely exercised): identical.
- Raw marker logits on a 699-token sequence: max |Δ| 6e-6 (order-invariant) and 3e-5 (full) vs float32 torch.

`tests/Amql.Tests/VonDecisionTests.cs` repeats this on `fixtures/von-tiny`, a small random checkpoint in Von's
layout whose goldens were produced by transformers and the Von SDK (`fixtures/make_von_tiny.py` regenerates it):
hidden states and logits in both modes, `decide` answers, the 422 cases, Python formatting and the tokenizer.

## 7. Not done

- **Batching.** Each question is its own pass, as in the SDK; independent questions could share one batch.
- **Big-endian or legacy (pre-1.6) `.pt` files** are refused.
- The `.pt` export holds the whole state dict in memory while writing (≈1.6 GB for Von).
