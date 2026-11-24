# The KV cache: size, format and what to keep

How big the GPU engine's prompt state is, how far it compresses, the NVFP4 KV
format, and which rows a conversation's later questions actually read. The
measurements are on Qwen 3.5 4B on an RTX 5090. Serving-side behaviour (prefix
reuse, paging to disk) is in [server.md](server.md#concurrency).

## What the state is

Qwen 3.5 4B has 32 layers: 24 Gated DeltaNet layers and 8 softmax-attention
layers (every fourth). The two kinds keep very different state:

- **Gated DeltaNet layers** keep a fixed-size recurrent state: 32 heads of
  128 × 128, plus conv history, 50 MB for all 24 layers at any length. It is a
  learned, lossy summary of everything read so far.
- **Softmax layers** keep every position's key and value: 4 KV heads of 256, so
  8 KiB a token a layer in f32, 64 KiB a token for the 8 layers, plus 8 KiB for
  the MTP drafter's own layer.

A 14,889-token prompt's state is 1,097 MiB: 1,047 MiB of KV and 50 MiB of
recurrent state. The tokens themselves are 58 KiB.

## Lossless compression barely works

On a real 14,889-token state (`.kvpage` from the disk tier):

| Method | Ratio |
|---|---|
| zlib on the raw floats | 1.07x |
| Byte-plane shuffle, then zlib | 1.17x |
| Channel-major order + shuffle | 1.19x |
| XOR with the previous token's row | 1.16x |
| lzma | 1.17x |

No value is exactly representable in BF16 or FP16: the low mantissa bits come
out of f32 accumulation and are noise. The information is in the top 8 or so
bits, which is why 8-bit formats lose under 1%. There is no low-rank structure
either: keeping 99% of a head's energy needs about 230 of its 256 dimensions.

## Lossy formats, offline

Relative RMS error of K and V on the same state (layer 15, or the mean of all 8):

| Format | Bits a value | Smaller by | Error |
|---|---|---|---|
| FP16 | 16 | 2x | 2e-4 |
| BF16 | 16 | 2x | 1.7e-3 |
| int8, K per channel, V per token | 8.1 | 3.9x | 0.6–0.9% |
| FP8 E4M3, per-token scale | 8 | 4x | 2.6% |
| **NVFP4, blocks of 16 along a row** | 4.5 | 7.1x | 9.3% |
| NVFP4, blocks of 16 positions per channel | 4.5 | 7.1x | 9.9% |
| int4 KIVI-style (K per channel, V per token) | 4.25 | 7.5x | 7.3% K, 12.6% V |
| MXFP4 | 4.25 | 7.5x | 12.5% |
| Random Hadamard rotation + NVFP4 | 4.5 | 7.1x | 9.5% |
| int2 KIVI-style | 3 | 10.7x | 27–44% |

NVFP4's 16-value blocks with E4M3 scales already handle outliers: a rotation
adds nothing.

## NVFP4 KV in the engine

`--kv-format nvfp4` (amql-server and amql-cli, or `AMQL_KV_FORMAT=nvfp4`)
stores every softmax layer's cache, and the drafter's, as NVFP4 rows. The
default is FP16 (below); `--kv-format f32` keeps the values as computed.

- **A row** (one position, K or V) is the E2M1 codes two to a byte, an E4M3
  scale per 16 values, and an f32 scale per head (its largest magnitude over
  448 · 6, so every block scale fits E4M3), padded to 16 bytes: 592 bytes on the
  4B instead of 4,096.
- **The writer** (`attn_prep_kernel`) quantises each head once, after the norm
  and rope. Rounding is to nearest E2M1, as the weight encoder rounds.
- **The readers** dequantise as they load: the decode split kernel (one lane
  owns 8 contiguous dimensions, one 32-bit code word, one block scale and one head
  scale a key), the tiled prompt kernel, and the cuBLAS prompt path, which widens
  the key and value span to f32 first.
- **The guarantees among paths hold.** Every path reads the same rows, so a
  batched reply equals the same reply alone, and a rewound, restored or paged-in
  sequence carries on exactly. Replies differ from f32 KV's.
- Disk pages are 7x smaller too: a 14.9k-token state is about 160 MB instead of
  1.1 GB, so the 1 GiB disk budget holds ~100k tokens of the 4B.

**The intent stays exact.** A sequence can keep its first positions at FP16
or f32 inside an NVFP4 cache (`GpuEngine.SetExactRows`). amql-server does this
for every new conversation by default with NVFP4 KV
(`--kv-exact-prefix auto`): everything through the first user message, so the
system prompt, the tools and the first instruction, which every later token
reads, carry no 4-bit rounding. `--kv-exact-prefix N` keeps N tokens and `0`
none.

- **FP16 by default** (`--kv-exact-format f16`), each value rounded to nearest
  FP16: on the 4B it is indistinguishable from f32 (below), at half the memory.
  An 18k-token agent system prompt keeps 550 MB per conversation on the 4B;
  `--kv-exact-prefix N` caps it.
- **f32** (`--kv-exact-format f32`) keeps those rows exactly as computed:
  positions before the boundary then read only f32 rows, so their logits equal
  an f32 cache's bit for bit (`KvFormatTests`), at 1.1 GB for the same prompt.
- **Why not the weights' format:** K and V are computed in f32 from the f32
  hidden state, then normalised and rotated, so whatever the weights (BF16 or
  NVFP4), none of their values is a BF16 value and 0.01% are FP16 values.
  Storing them as the weights are stored would round them too.
- FP16 holds magnitudes up to 65,504. This model's K and V stay under 51; a
  model with much larger activations could overflow it, and would want f32.
- The cache holds those rows first and NVFP4 rows after them, in one buffer.
  The boundary and its format live in the sequence's position counter on the
  device, so every kernel reads them and a captured graph stays valid.
- They belong to the sequence: rewinds, batches, paging and restores keep them
  (a saved mark records them), and they change only while the sequence is
  empty. Every path reads the same rows, so the paths agree with each other
  in either format.

What an exact start protects, on the 4B over 8,192 tokens of docs and C#, scored
only on the second half (positions 4,096 to 8,191, `--score-from 4096`), against
f32 KV:

| NVFP4 KV, first positions exact | KL | Top-1 agrees | Perplexity |
|---|---|---|---|
| none | 0.0071 | 95.6% | +0.53% |
| 2,048 at f32 | 0.0057 (−20%) | 95.8% | +0.58% |
| 4,096 at f32 | 0.0047 (−34%) | 96.6% | +0.38% |
| 4,096 at FP16 | 0.0047 (−34%) | 96.4% | +0.34% |

FP16 protects as well as f32. On its own it costs nothing measurable: with
every position kept at FP16 (`@exact16=9000`), the 4B over all 8,192 tokens
gives perplexity 10.325 against f32's 10.326, KL to f32 under 0.00005 and the
same top token at every position.

Later tokens drift less when what they read first is exact. Served end to end
on the agent conversation below (six questions, 64 greedy tokens each), every
answer was right with f32, NVFP4 and NVFP4 with `auto`; with `auto` three of the
six were word for word f32's, against one without. That conversation's first
user message is short (96 tokens) and its bulk is tool output, which `auto`
leaves at NVFP4.

Quality on 8,192 tokens of this repo's docs and C#, teacher-forced
(`amql-cli quantise compare <dir> <dir>@kv=nvfp4 …`):

| Weights, KV | Perplexity | KL to BF16 + f32 KV | Top-1 agrees |
|---|---|---|---|
| BF16, f32 (reference) | 10.326 | — | — |
| BF16, **NVFP4 KV** | 10.371 (+0.4%) | 0.0076 | 95.8% |
| NVFP4 GPTQ, f32 | 10.470 (+1.4%) | 0.0196 | 92.4% |
| NVFP4 GPTQ, NVFP4 KV | 10.538 (+2.1%) | 0.0277 | 91.3% |

NVFP4 KV costs well under half what NVFP4 weights do, and the two add.

Speed at a 32,000-token context (`amql-cli --kv-format <f> bench <4b> --prompt-tokens 32000 --steps 128`),
three rounds run alternately on a shared desktop GPU:

| KV | KV a token | Prefill | Decode at 32k, range (middle) |
|---|---|---|---|
| f32 | 64 KiB | 9,135–9,357 tok/s | 102–114 tok/s (~109) |
| FP16 | 32 KiB | 9,029–9,154 tok/s | 110–120 tok/s (~117, +7%) |
| NVFP4 | ~9.3 KiB | 9,047–9,169 tok/s | 115–127 tok/s (~120, +10%) |

Decode attention reads half or a seventh of the bytes. The step is mostly the
weight GEMVs, so the gain at 32k is modest; it grows with context and batch,
and the memory saved is the larger effect.

**FP16 for the whole cache** (the default, `--kv-format fp16`) is the mixed layout with every
row exact at FP16: the same kernels, each value rounded to nearest FP16. On the
4B over 8,192 tokens it gives perplexity 10.325 against f32's 10.326, KL under
0.00005 and the same top token at every position, at half f32's memory. It
needs no exact prefix (every row is one already).
The first version dequantised one value at a time in the decode kernel and ran
at 60.6 tok/s: the scale lookups per value made it instruction-bound. The NVFP4
decode kernel now gives each lane whole code words, as the weight GEMVs do.
Heads of 64, 128 and 256 take that kernel; other widths take the per-value
path. On the 0.8B a 1,670-token conversation's sequence takes 105 MiB of GPU
memory instead of 152 (its fixed recurrent state dominates at that length), and
its page file 26 MiB instead of 65.

## What to keep

The question was whether a checkpoint could keep only part of the state and
still answer later questions, without prefilling again. The recurrent state
must be kept whole (it is small and it is the summary); the softmax layers'
rows are where a choice can be made.

`amql-cli kv-probe <container> --conversation <json> --out <dir>` records what
real questions attend to. It prefills a conversation once on the GPU engine,
then for each question rewinds to that prefix, reads the question and writes
its own greedy answer, recording every softmax layer's query rows as they run
(`GpuEngine.QueryProbe`) and the keys and values they read. Attention over the
prefix is then computed exactly offline for any choice of kept rows: how much of
the prefix's attention mass the kept rows hold, and how far the attention output
moves when attention is renormalised over them. The question's and answer's own
rows are always kept.

The conversation: an agent asked to read four files (the server docs, the config
reader, the engine's mark and rewind code and the page store, 17,875 tokens),
then six questions: three about details early, in the middle and late in the
context, one about the user's first message, and a summary. The 4B answered all
six correctly from the full context.

**Attention follows the content.** Each question's attention to the prefix goes
to the turn that holds the answer (53% to the docs for the docs question, 40%
to the config file for the config question, 48% to the page store for the file
extension, 31% to the user's first message for "what did I ask"), plus 8–15% to
the system prompt.

**But each question needs different rows.** The overlap of different questions'
top 10% rows is a Jaccard index of 0.14–0.42. No small fixed set serves them all.

Kept rows judged on the answer's own attention (mass kept / output error):

| Kept rows chosen by | 5% | 10% | 20% | 40% |
|---|---|---|---|---|
| The most recent rows | 6.6% / 0.55 | 8.4% / 0.52 | 14.7% / 0.46 | 38.9% / 0.30 |
| 64 sink rows + recent | 21.9% / 0.49 | 23.7% / 0.46 | 28.1% / 0.40 | 53.5% / 0.22 |
| Short turns whole + recent | 35.1% / 0.39 | 36.9% / 0.36 | 41.3% / 0.31 | 63.2% / 0.17 |
| SnapKV: the prefix's last 32 queries | 38.9% / 0.32 | 48.2% / 0.25 | 61.6% / 0.17 | 78.6% / 0.09 |
| **The question's own prompt rows** | **68.6% / 0.13** | **79.6% / 0.08** | **89.0% / 0.04** | **95.9% / 0.01** |
| The same, whole 64-token pages | 54.8% / 0.19 | 67.2% / 0.12 | 79.4% / 0.07 | 90.7% / 0.03 |
| Oracle: the answer's own attention | 79.2% / 0.07 | 87.4% / 0.04 | 93.6% / 0.02 | 97.8% / 0.01 |

- **Chosen before the question, a checkpoint forgets.** The best question-blind
  choice (SnapKV) holds half the attention at 10% and the questions about early
  content fare worst (47–52% at 20%).
- **Chosen by the question, 10–20% is enough.** The question is prefilled before
  any answer token, and its own attention over the prefix picks rows almost as
  well as the answer's: 80% of the mass at 10%, 89% at 20%, close to the oracle,
  and as well for early content as late (80–89% at 10% for every specific
  question; 62% for the summary, which reads broadly).
- Choosing per KV head barely beats one choice shared by the heads; whole pages
  of 64 rows cost about a third of what is missed.

### What that means for a design

A checkpoint that keeps every row in a cheap tier and brings only the
question's rows to the GPU would read 10–20% of the prefix per answer:

1. **Keep:** the recurrent state whole, and every softmax row as NVFP4 in host
   memory or on disk (170 MB for the 17.9k-token conversation above, against
   1.1 GB as f32).
2. **Select:** prefill the question against the full keys and sum its attention
   over the prefix per row. The cuBLAS prompt path already materialises those
   scores; the sum is nearly free.
3. **Serve the answer** from the selected rows only: the decode kernels read a
   list of rows instead of a span (positions stay as written, since keys carry
   their rope already).

Step 2 needs every key on the GPU for the question's prefill (half the KV, or a
quarter with NVFP4 values left behind); a page index (Quest's per-page bounds)
would avoid even that at some loss, as the 64-row pages above show. The open
measurement is end to end: answers and their quality with only the selected rows
in the decode kernels, which needs the engine to attend over a row list.
