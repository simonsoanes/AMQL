"""Builds tests/Amql.Tests/fixtures/qwen-lm-engine: qwen-lm-tiny's tokenizer, chat template and
generation config over a small random Qwen3.5 whose Gated DeltaNet heads are 64 wide, the
geometry the GPU engine runs (qwen-lm-tiny's 8-wide heads stay on the managed path). For
server tests of batching on the device engine; there are no goldens, the tests compare the
engine with itself.

    python tests/Amql.Tests/fixtures/make_qwen_lm_engine.py
"""
import json
import os
import shutil

import torch
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "qwen-lm-tiny")
OUT = os.path.join(HERE, "qwen-lm-engine")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(7)

for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    shutil.copy(os.path.join(SRC, f), os.path.join(OUT, f))
tiny = json.load(open(os.path.join(SRC, "config.json")))

cfg = Qwen3_5TextConfig(vocab_size=tiny["vocab_size"], hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                        linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=64, linear_value_head_dim=64,
                        linear_conv_kernel_dim=4, max_position_embeddings=4096,
                        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
                        tie_word_embeddings=True, eos_token_id=tiny["eos_token_id"], pad_token_id=tiny["pad_token_id"])
model = Qwen3_5ForCausalLM(cfg).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        p.copy_((0.5 if "A_log" in name or "dt_bias" in name else 0.3) * torch.randn_like(p))
model.save_pretrained(OUT, safe_serialization=True)
config = json.load(open(os.path.join(OUT, "config.json")))
config["attn_output_gate"] = True
json.dump(config, open(os.path.join(OUT, "config.json"), "w"), indent=1)
print(f"wrote {OUT}")
