"""Writes fixtures/gated_delta_golden.json: random q/k/v/g/beta and the output
of transformers' own torch_recurrent_gated_delta_rule (use_qk_l2norm_in_kernel,
as Qwen3.5 calls it), so GatedDeltaKernel.Recurrent is checked against the
reference implementation directly. Regenerate: python make_gated_delta_golden.py"""
import json, os, torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
torch.manual_seed(7)
T, H, K, V = 6, 3, 4, 5
q, k, v = torch.randn(1, T, H, K), torch.randn(1, T, H, K), torch.randn(1, T, H, V)
g = -torch.rand(1, T, H) * 1.5          # log-decay < 0, as -exp(A)·softplus(·) is
beta = torch.rand(1, T, H)
out, state = torch_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
flat = lambda x: x.reshape(-1).tolist()
json.dump({"T": T, "heads": H, "k_dim": K, "v_dim": V, "q": flat(q), "k": flat(k), "v": flat(v), "g": flat(g),
           "beta": flat(beta), "out": flat(out), "state": flat(state)},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gated_delta_golden.json"), "w"))
