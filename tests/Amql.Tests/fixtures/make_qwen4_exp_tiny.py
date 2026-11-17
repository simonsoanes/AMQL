"""Builds tests/Amql.Tests/fixtures/arch-zoo/qwen4_exp-tiny: a tiny, randomly
initialised Qwen4 (transformers `qwen4_exp`, the Qwen3.8-Flash-Next layout)
with the golden logits transformers itself produces.

It exercises everything Qwen4 adds over Qwen3.5, at sizes where each one
changes the answer:
  - the Gated Residual (4 streams) in place of every norm, and its final mixer;
  - QSA with a 4-token budget in 2-token blocks, so most of the 20 positions
    attend to a selected subset of keys rather than all of them;
  - the hashed n-gram embedding (PLE) on layer 2, with an end-of-sequence token
    inside the sequence (an n-gram must not reach back across it), its table
    saved as 4 row-shards the way the real checkpoint ships it;
  - Flash-Next's sigmoid-gated linear-attention norm;
  - routed experts plus a gated shared expert, stored packed.

Regenerate with (needs torch and a transformers that has qwen4_exp):
    python make_qwen4_exp_tiny.py
"""
import json, os, shutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "arch-zoo", "qwen4_exp-tiny")
IDS = [1, 7, 42, 3, 99, 17, 5, 64, 23, 11, 2, 88, 45, 6, 120, 33, 8, 77, 2, 19]

cfg = AutoConfig.for_model(
    "qwen4_exp_text",
    vocab_size=128, hidden_size=64, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    max_position_embeddings=256, tie_word_embeddings=False,
    bos_token_id=1, eos_token_id=2, pad_token_id=0,
    layer_types=["linear_attention", "linear_attention", "full_attention", "full_attention"],
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                     "mrope_section": [1, 1, 0], "mrope_interleaved": True},
    partial_rotary_factor=0.25,
    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
    linear_conv_kernel_dim=4, output_gate_type="sigmoid",
    num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32,
    norm_topk_prob=True,
    hc_count=4, hc_lowrank=16,
    indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8, indexer_budget=4, indexer_compress_ratio=2,
    ple_layer_ids=[2], ple_embed_dim=32, ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=2,
    ngram_vocab_size_base=50, make_ngram_vocab_size_divisible_by=16, split_ngram_parts=4, seed=1234,
    mtp_num_hidden_layers=0,
)

torch.manual_seed(1234)
model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32, attn_implementation="eager").eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        # Non-trivial weights everywhere, including the zero-initialised ones
        # (the PLE conv, the (1+w) norms), so a dropped term shows.
        p.copy_(0.2 * torch.randn_like(p))

shutil.rmtree(OUT, ignore_errors=True)
model.save_pretrained(OUT, safe_serialization=True)
with torch.no_grad():
    logits = model(torch.tensor([IDS])).logits[0].float()
json.dump({"ids": IDS, "logits": [[round(v, 6) for v in row] for row in logits.tolist()]},
          open(os.path.join(OUT, "golden.json"), "w"))
for junk in ("generation_config.json",):
    path = os.path.join(OUT, junk)
    if os.path.exists(path):
        os.remove(path)
print("wrote", OUT, "layer types", model.config.layer_types)
