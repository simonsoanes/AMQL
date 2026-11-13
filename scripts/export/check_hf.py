"""Loads an HF checkpoint written by `amql-cli export` in transformers and
compares its logits with a golden.json made by transformers from the original
(tests/Amql.Tests/fixtures/make_arch_zoo.py).

    python check_hf.py <exported-dir> golden.json
"""
import json, sys
import torch
from transformers import AutoModelForCausalLM

path, golden_path = sys.argv[1], sys.argv[2]
g = json.load(open(golden_path))
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager").eval()
with torch.no_grad():
    logits = model(torch.tensor([g["ids"]])).logits[0]
worst = float((logits - torch.tensor(g["logits"])).abs().max())
print(f"{model.__class__.__name__}: worst logit diff vs original {worst:.3g}")
sys.exit(0 if worst < 2e-3 else 1)
