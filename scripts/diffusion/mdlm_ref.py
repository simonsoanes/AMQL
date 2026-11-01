"""Reference for the A2D (autoregressive-to-diffusion) Qwen3 masked diffusion LM
(dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1), written from the published modeling
code without executing it: Qwen3 with the causal mask replaced by a padding-only
(here: none) mask, logits read at the same position they predict (no shift).

  python mdlm_ref.py <hf-dir> check             # causal mode vs transformers' Qwen3 (validates the port)
  python mdlm_ref.py <hf-dir> logits <ids.json> <out.json>   # bidirectional logits (full rows, or every 997th id)
  python mdlm_ref.py <hf-dir> sample <spec.json> <out.json>   # sampler from raw ids
  python mdlm_ref.py <hf-dir> generate "<prompt>" [--steps N --gen N --block N] [--out f.json]
"""
import json
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


class Ref:
    def __init__(self, d):
        self.cfg = json.load(open(f"{d}/config.json"))
        self.w = {k: v.float() for k, v in load_file(f"{d}/model.safetensors").items()}
        c = self.cfg
        self.L, self.H = c["num_hidden_layers"], c["hidden_size"]
        self.nq, self.nkv, self.hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.eps = c["rms_norm_eps"]
        inv = 1.0 / (c["rope_theta"] ** (torch.arange(0, self.hd, 2).float() / self.hd))
        self.inv_freq = inv
        self.head = self.w.get("lm_head.weight", self.w["model.embed_tokens.weight"])

    def rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def rope(self, x, pos):
        f = pos[:, None].float() * self.inv_freq[None]
        emb = torch.cat([f, f], -1)
        cos, sin = emb.cos()[None], emb.sin()[None]
        x1, x2 = x[..., : self.hd // 2], x[..., self.hd // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    @torch.no_grad()
    def forward(self, ids, causal=False):
        w, T = self.w, len(ids)
        x = w["model.embed_tokens.weight"][torch.tensor(ids)]
        pos = torch.arange(T)
        mask = torch.zeros(T, T)
        if causal:
            mask = torch.full((T, T), float("-inf")).triu(1)
        for l in range(self.L):
            p = f"model.layers.{l}."
            h = self.rms(x, w[p + "input_layernorm.weight"])
            q = (h @ w[p + "self_attn.q_proj.weight"].T).view(T, self.nq, self.hd)
            k = (h @ w[p + "self_attn.k_proj.weight"].T).view(T, self.nkv, self.hd)
            v = (h @ w[p + "self_attn.v_proj.weight"].T).view(T, self.nkv, self.hd)
            q = self.rms(q, w[p + "self_attn.q_norm.weight"]).transpose(0, 1)
            k = self.rms(k, w[p + "self_attn.k_norm.weight"]).transpose(0, 1)
            v = v.transpose(0, 1)
            q, k = self.rope(q, pos), self.rope(k, pos)
            rep = self.nq // self.nkv
            k, v = k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0)
            s = q @ k.transpose(1, 2) / self.hd ** 0.5 + mask
            o = (s.softmax(-1) @ v).transpose(0, 1).reshape(T, -1)
            x = x + o @ w[p + "self_attn.o_proj.weight"].T
            h = self.rms(x, w[p + "post_attention_layernorm.weight"])
            g = h @ w[p + "mlp.gate_proj.weight"].T
            u = h @ w[p + "mlp.up_proj.weight"].T
            x = x + (F.silu(g) * u) @ w[p + "mlp.down_proj.weight"].T
        x = self.rms(x, w["model.norm.weight"])
        return x @ self.head.T


def transfer_counts(n, steps):
    base, rem = divmod(n, steps)
    return [base + (1 if i < rem else 0) for i in range(steps)]


def generate(ref, prompt_ids, mask_id, pad_id, gen=64, steps=64, block=32, order="confidence"):
    """The model card's sampler at temperature 0 with low-confidence remasking:
    per block, each step commits the k masked positions whose argmax is most
    probable. The card ranks a float32 softmax with torch.topk, where padding
    positions tie at ~1.0 and the pick among them is arbitrary; here confidence
    is float64 and ties go to the lowest position, so two implementations can
    be compared step for step (AMQL's sampler does the same)."""
    x = list(prompt_ids) + [mask_id] * gen
    P = len(prompt_ids)
    assert gen % block == 0 and steps % (gen // block) == 0
    spb = steps // (gen // block)
    trace = []
    for b in range(gen // block):
        lo, hi = P + b * block, P + (b + 1) * block
        counts = transfer_counts(sum(1 for i in range(lo, hi) if x[i] == mask_id), spb)
        for s in range(spb):
            k = counts[s]
            if k == 0:
                continue
            logits = ref.forward(x)
            p = logits.double().softmax(-1)
            x0 = logits.argmax(-1)
            # order "ltr": every candidate ties, so the leftmost masked positions
            # commit first - what llama-diffusion-cli does at --top-k 1.
            cand = [(0.0 if order == "ltr" else -p[i, x0[i]].item(), i) for i in range(lo, hi) if x[i] == mask_id]
            sel = [i for _, i in sorted(cand)[:k]]
            for i in sel:
                x[i] = int(x0[i])
            trace.append(sorted(sel))
    return x[P:], trace


def main():
    d, mode = sys.argv[1], sys.argv[2]
    ref = Ref(d)
    if mode == "check":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = dict(ref.cfg)
        for k in ("auto_map", "architectures", "model_type"):
            cfg.pop(k, None)
        hf = Qwen3ForCausalLM(Qwen3Config(**cfg)).eval()
        hf.load_state_dict({k: v for k, v in ref.w.items()}, strict=False)
        hf.tie_weights()
        ids = [9707, 11, 1879, 374, 264, 1273, 13, 576, 6722, 315, 9625, 374]
        with torch.no_grad():
            a = hf(torch.tensor([ids])).logits[0]
        b = ref.forward(ids, causal=True)
        print("causal port max|diff|:", (a - b).abs().max().item())
        c = ref.forward(ids, causal=False)
        print("bidirectional vs causal max|diff| (must be large):", (c - b).abs().max().item())
    elif mode == "logits":
        ids = json.load(open(sys.argv[3]))
        lg = ref.forward(ids)
        stride = 1 if lg.shape[1] <= 4096 else 997
        out = {
            "ids": ids,
            "vocab_stride": stride,
            "argmax": lg.argmax(-1).tolist(),
            "logits": [[lg[i, j].item() for j in range(0, lg.shape[1], stride)] for i in range(len(ids))],
            "logsumexp": torch.logsumexp(lg, -1).tolist(),
        }
        json.dump(out, open(sys.argv[4], "w"))
        print("argmax", out["argmax"])
    elif mode == "sample":
        # sample <spec.json> <out.json>: spec {prompt_ids, mask_id, gen, steps, block}
        spec = json.load(open(sys.argv[3]))
        out, trace = generate(ref, spec["prompt_ids"], spec["mask_id"], 0, spec["gen"], spec["steps"], spec["block"])
        spec.update(output_ids=out, commit_trace=trace)
        json.dump(spec, open(sys.argv[4], "w"))
        print(out)
    elif mode == "generate":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(f"{d}/tokenizer.json")
        args = sys.argv[4:]
        opt = {args[i].lstrip("-"): args[i + 1] for i in range(0, len(args), 2)}
        prompt = sys.argv[3]
        text = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
        if opt.get("think", "0") != "1":
            text += "<think>\n\n</think>\n\n"  # enable_thinking=False, as the card recommends
        ids = tok.encode(text, add_special_tokens=False).ids
        mask_id = tok.token_to_id("<|mask|>")
        out, trace = generate(ref, ids, mask_id, 151643, int(opt.get("gen", 64)), int(opt.get("steps", 64)), int(opt.get("block", 32)),
                              opt.get("order", "confidence"))
        print(tok.decode(out, skip_special_tokens=False))
        if "out" in opt:
            json.dump({"prompt_ids": ids, "mask_id": mask_id, "gen": int(opt.get("gen", 64)),
                       "steps": int(opt.get("steps", 64)), "block": int(opt.get("block", 32)),
                       "output_ids": out, "commit_trace": trace}, open(opt["out"], "w"))


if __name__ == "__main__":
    main()
