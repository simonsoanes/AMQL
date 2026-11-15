"""Checks a graft written by `amql-cli entail graft` in transformers, under the
host's own model class: where the program is silent the grafted model must
give the host's logits, and where it fires, its token must win.

    python check_graft.py <graft-dir> <host-dir> <silent ids> <firing ids> <row> <token id>

ids are comma-separated; <row> is the position of <firing ids> where the
program says <token id>. Runs in float32 with eager attention.
"""
import sys
import torch
from transformers import AutoModelForCausalLM

graft_dir, host_dir, silent, firing, row, token = sys.argv[1:7]
silent = [int(x) for x in silent.split(",")]
firing = [int(x) for x in firing.split(",")]
row, token = int(row), int(token)


def load(path):
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").eval()


def logits(model, ids):
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0].float()


host, graft = load(host_dir), load(graft_dir)
reference = logits(host, silent)
tolerance = 1e-3 * max(1.0, float(reference.abs().max()))
moved = float((logits(graft, silent) - reference).abs().max())
fired = logits(graft, firing)
plain = logits(host, firing)
others = [r for r in range(len(firing)) if r != row]
quiet = float((fired[others] - plain[others]).abs().max())
winner = int(fired[row].argmax())
margin = float(fired[row, token] - plain[row].max())
ok = moved < tolerance and quiet < tolerance and winner == token and margin > 100
print(f"silent drift {moved:.3g}, other positions {quiet:.3g} (tolerance {tolerance:.3g}); "
      f"position {row} says {winner} (want {token}) by {margin:.1f}: {'ok' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
