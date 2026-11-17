"""Builds tests/Amql.Tests/fixtures/arch-zoo/qwen4_exp-nvfp4-tiny: the tiny
Qwen4 of make_qwen4_exp_tiny.py, quantised the way NVIDIA's
Qwen3.8-Flash-Next-NVFP4-QAD checkpoint is (ModelOpt, MIXED_PRECISION):

  - routed experts (one tensor set per expert, not packed) and the shared
    expert: NVFP4 - `weight` U8 [out, in/2] (two E2M1 codes a byte, the even
    column in the low nibble), `weight_scale` F8_E4M3 [out, in/16],
    `weight_scale_2` F32 [] and `input_scale` F32 [];
  - attention, linear-attention and indexer projections: MXFP8 - `weight`
    F8_E4M3 [out, in] and `weight_scale` U8 [out, in/32] (E8M0 exponents);
  - the n-gram table: FP8 shards with one shared BF16 `weight_scale` [1];
  - everything else as it was.

golden.json holds transformers' logits for the original model with every
quantised weight replaced by its dequantised value, so matching it checks
that each format is decoded exactly as written.

Regenerate (after make_qwen4_exp_tiny.py) with:
    python make_qwen4_exp_nvfp4_tiny.py
"""
import json, os, shutil
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "arch-zoo", "qwen4_exp-tiny")
OUT = os.path.join(HERE, "arch-zoo", "qwen4_exp-nvfp4-tiny")
GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def nvfp4(w):
    out, inp = w.shape
    amax = w.abs().max()
    g = (amax / (6.0 * 448.0)).float() if amax > 0 else torch.tensor(1.0)
    blocks = w.reshape(out, inp // 16, 16)
    scale = (blocks.abs().amax(-1) / 6.0 / g).to(torch.float8_e4m3fn)
    step = (scale.float() * g).unsqueeze(-1)
    q = torch.where(step > 0, blocks / torch.where(step > 0, step, torch.ones_like(step)), torch.zeros_like(blocks))
    idx = (q.abs().unsqueeze(-1) - GRID).abs().argmin(-1)
    codes = (idx | torch.where(q < 0, 8, 0)).to(torch.uint8).reshape(out, inp)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    value = GRID[idx] * torch.where(q < 0, -1.0, 1.0)
    deq = ((value * scale.float().unsqueeze(-1)) * g).reshape(out, inp)
    tensors = {"weight": packed, "weight_scale": scale.contiguous(),
               "weight_scale_2": g.reshape(()).float(), "input_scale": torch.tensor(1.0)}
    return tensors, deq


def mxfp8(w):
    out, inp = w.shape
    blocks = w.reshape(out, inp // 32, 32)
    amax = blocks.abs().amax(-1)
    e = torch.where(amax > 0, torch.ceil(torch.log2(amax / 448.0)), torch.full_like(amax, -127.0)).clamp(-127, 127)
    scale = torch.pow(2.0, e)
    q = (blocks / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    deq = (q.float() * scale.unsqueeze(-1)).reshape(out, inp)
    return {"weight": q.reshape(out, inp).contiguous(), "weight_scale": (e + 127).to(torch.uint8).contiguous()}, deq


def fp8_tensor(shards):
    amax = max(s.abs().max() for s in shards)
    scale = (amax / 448.0).to(torch.bfloat16)
    qs = [(s / scale.float()).to(torch.float8_e4m3fn) for s in shards]
    return qs, scale.reshape(1), [q.float() * scale.float() for q in qs]


src = load_file(os.path.join(SRC, "model.safetensors"))
config = json.load(open(os.path.join(SRC, "config.json")))
quant, deq = {}, dict(src)
quantized_layers = {}

for name, w in src.items():
    if ".mlp.experts." in name and name.endswith("_proj.weight"):
        # Experts are saved one tensor set each, as the NVIDIA checkpoint has them.
        t, d = nvfp4(w)
        for k, v in t.items():
            quant[name[: -len("weight")] + k] = v
        deq[name] = d
        quantized_layers[name.split(".mlp.experts.")[0] + ".mlp.experts"] = {"quant_algo": "NVFP4", "group_size": 16}
    elif ".mlp.shared_expert." in name and name.endswith(".weight"):
        t, d = nvfp4(w)
        for k, v in t.items():
            quant[name[: -len("weight")] + k] = v
        deq[name] = d
        quantized_layers[name[: -len(".weight")]] = {"quant_algo": "NVFP4", "group_size": 16}
    elif any(s in name for s in (".linear_attn.in_proj_", ".linear_attn.out_proj.", ".self_attn.q_proj.",
                                 ".self_attn.k_proj.", ".self_attn.v_proj.", ".self_attn.o_proj.",
                                 ".indexer.index_qk_proj.")) and name.endswith(".weight"):
        t, d = mxfp8(w)
        for k, v in t.items():
            quant[name[: -len("weight")] + k] = v
        deq[name] = d
        quantized_layers[name[: -len(".weight")]] = {"quant_algo": "MXFP8", "group_size": 32}
    elif ".ngram_embedding.shard_" in name:
        continue
    else:
        quant[name] = w

shards = sorted((n for n in src if ".ngram_embedding.shard_" in n), key=lambda n: int(n.split("shard_")[1].split(".")[0]))
qs, scale, deqs = fp8_tensor([src[n] for n in shards])
for n, q, d in zip(shards, qs, deqs):
    quant[n] = q.contiguous()
    deq[n] = d
table = shards[0].split(".shard_")[0]
quant[table + ".weight_scale"] = scale.contiguous()
quantized_layers[table] = {"quant_algo": "FP8"}

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)
save_file(quant, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})
config["quantization_config"] = {
    "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
    "producer": {"name": "modelopt", "version": "fixture"},
    "quantized_layers": quantized_layers, "ignore": ["lm_head"],
}
json.dump(config, open(os.path.join(OUT, "config.json"), "w"), indent=2)
json.dump({"producer": {"name": "modelopt", "version": "fixture"},
           "quantization": {"quant_algo": "MIXED_PRECISION", "group_size": 16,
                            "exclude_modules": ["lm_head"], "quantized_layers": quantized_layers}},
          open(os.path.join(OUT, "hf_quant_config.json"), "w"), indent=2)

# transformers converts the saved layout (one set per expert, row-sharded
# n-gram table) as it loads, so the dequantised weights go through a save.
scratch = os.path.join(OUT, "_dequantised")
os.makedirs(scratch)
save_file({k: v.contiguous() for k, v in deq.items()}, os.path.join(scratch, "model.safetensors"), metadata={"format": "pt"})
shutil.copy(os.path.join(SRC, "config.json"), os.path.join(scratch, "config.json"))
model = AutoModelForCausalLM.from_pretrained(scratch, dtype=torch.float32, attn_implementation="eager").eval()
shutil.rmtree(scratch)
golden = json.load(open(os.path.join(SRC, "golden.json")))
with torch.no_grad():
    logits = model(torch.tensor([golden["ids"]])).logits[0].float()
json.dump({"ids": golden["ids"], "logits": [[round(v, 6) for v in row] for row in logits.tolist()]},
          open(os.path.join(OUT, "golden.json"), "w"))
moved = max(abs(a - b) for ra, rb in zip(logits.tolist(), golden["logits"]) for a, b in zip(ra, rb))
print("wrote", OUT, "| quantisation moves the logits by up to", round(moved, 4))
