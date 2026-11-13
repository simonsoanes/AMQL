"""Checks a GGUF written by `amql-cli export --format gguf` against transformers.

Reads the file with the `gguf` package and runs it with llama.cpp's semantics
for the architecture it declares: llama/granite rotate ADJACENT pairs (rope type
NORM, which is why the exporter permutes q/k), qwen2/qwen3/olmo2 rotate the two
HALVES of each head (NEOX). The logits are compared with a golden.json made by
transformers (tests/Amql.Tests/fixtures/make_arch_zoo.py).

This is not llama.cpp: it is an independent reading of the same file under the
same rules, to catch a wrong permutation, rope pairing, norm placement or
metadata key. Running the file in llama.cpp itself remains the final check.

    python check_gguf.py model.gguf golden.json
"""
import json, sys
import numpy as np
from gguf import GGUFReader

path, golden_path = sys.argv[1], sys.argv[2]
r = GGUFReader(path)
kv = {f.name: f for f in r.fields.values()}


def val(key, default=None):
    f = kv.get(key)
    if f is None:
        return default
    v = f.parts[f.data[0]]
    return v.tobytes().decode() if f.types[0].name == "STRING" else v[0].item()


arch = val("general.architecture")
T = {t.name: np.array(t.data, dtype=np.float32).reshape([int(d) for d in reversed(t.shape)]) for t in r.tensors}
L = val(f"{arch}.block_count"); H = val(f"{arch}.attention.head_count"); KV = val(f"{arch}.attention.head_count_kv")
D = val(f"{arch}.attention.key_length"); eps = val(f"{arch}.attention.layer_norm_rms_epsilon")
theta = val(f"{arch}.rope.freq_base"); neox = arch in ("qwen2", "qwen3", "olmo2")
scale = val(f"{arch}.attention.scale", 1 / np.sqrt(D))
emb_scale = val(f"{arch}.embedding_scale", 1.0); res_scale = val(f"{arch}.residual_scale", 1.0)
logit_scale = val(f"{arch}.logit_scale", 1.0)


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x, n):  # x [S, n*D]
    S = x.shape[0]; x = x.reshape(S, n, D).copy()
    inv = 1.0 / theta ** (np.arange(0, D, 2) / D)
    if val(f"{arch}.rope.scaling.type") == "linear":
        inv = inv / val(f"{arch}.rope.scaling.factor")
    if "rope_freqs.weight" in T:  # llama.cpp divides each pair's frequency by its factor
        inv = inv / T["rope_freqs.weight"]
    ang = np.arange(S)[:, None] * inv[None]  # [S, D/2]
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    if neox:
        a, b = x[..., : D // 2].copy(), x[..., D // 2:].copy()
        x[..., : D // 2], x[..., D // 2:] = a * c - b * s, a * s + b * c
    else:
        a, b = x[..., 0::2].copy(), x[..., 1::2].copy()
        x[..., 0::2], x[..., 1::2] = a * c - b * s, a * s + b * c
    return x.reshape(S, n * D)


def lin(x, name):
    y = x @ T[name + ".weight"].T
    return y + T[name + ".bias"] if name + ".bias" in T else y


g = json.load(open(golden_path))
ids = g["ids"]; S = len(ids)
x = T["token_embd.weight"][ids] * emb_scale
mask = np.triu(np.full((S, S), -np.inf), 1)
for l in range(L):
    b = f"blk.{l}."
    h = rms(x, T[b + "attn_norm.weight"]) if b + "attn_norm.weight" in T else x
    q, k, v = lin(h, b + "attn_q"), lin(h, b + "attn_k"), lin(h, b + "attn_v")
    if b + "attn_q_norm.weight" in T:
        wq, wk = T[b + "attn_q_norm.weight"], T[b + "attn_k_norm.weight"]
        if wq.shape[0] == D:
            q = rms(q.reshape(S, H, D), wq).reshape(S, H * D); k = rms(k.reshape(S, KV, D), wk).reshape(S, KV * D)
        else:
            q, k = rms(q, wq), rms(k, wk)
    q, k = rope(q, H).reshape(S, H, D), rope(k, KV).reshape(S, KV, D); v = v.reshape(S, KV, D)
    rep = H // KV; out = np.zeros((S, H, D), np.float32)
    for hh in range(H):
        sc = q[:, hh] @ k[:, hh // rep].T * scale + mask
        p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        out[:, hh] = p @ v[:, hh // rep]
    a = lin(out.reshape(S, H * D), b + "attn_output")
    if b + "post_attention_norm.weight" in T:
        a = rms(a, T[b + "post_attention_norm.weight"])
    x = x + a * res_scale
    h = rms(x, T[b + "ffn_norm.weight"]) if b + "ffn_norm.weight" in T else x
    gte = lin(h, b + "ffn_gate"); f = lin((gte / (1 + np.exp(-gte))) * lin(h, b + "ffn_up"), b + "ffn_down")
    if b + "post_ffw_norm.weight" in T:
        f = rms(f, T[b + "post_ffw_norm.weight"])
    x = x + f * res_scale
x = rms(x, T["output_norm.weight"])
logits = x @ T.get("output.weight", T["token_embd.weight"]).T / logit_scale
worst = float(np.abs(logits - np.array(g["logits"])).max())
print(f"{arch}: worst logit diff vs transformers {worst:.3g}")
sys.exit(0 if worst < 2e-3 else 1)
