"""Builds tests/Amql.Tests/fixtures/arch-zoo: one tiny, randomly initialised
causal LM per transformers architecture, each with the golden logits
transformers itself produces for a fixed token sequence.

The zoo is the yardstick for property-driven model support: AMQL has no code
naming any of these families, so every one that encodes and matches its
golden logits is supported purely from its config and tensor properties.
An architecture that is not in the installed transformers is skipped.

Each model directory holds config.json, model.safetensors and golden.json:
  ids    — the input token ids
  logits — [len(ids), vocab] float logits from a single causal forward pass

Regenerate with (needs torch, transformers>=5):
    python make_arch_zoo.py
"""
import json, os, shutil, sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "arch-zoo")
VOCAB = 128
IDS = [1, 7, 42, 3, 99, 17, 5, 64, 23, 11, 2, 88]

# Shared tiny geometry; per-family overrides add the facts that family needs.
BASE = dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
            tie_word_embeddings=False, bos_token_id=1, eos_token_id=2, pad_token_id=0)

FAMILIES = {
    "llama": dict(rope_theta=10000.0),
    "llama-bias": dict(model_type="llama", attention_bias=True, mlp_bias=True),
    "mistral": dict(sliding_window=None),
    "qwen2": dict(),
    "qwen3": dict(),
    "phi3": dict(),
    "gemma": dict(hidden_act="gelu_pytorch_tanh", hidden_activation="gelu_pytorch_tanh", tie_word_embeddings=True),
    "gemma2": dict(hidden_activation="gelu_pytorch_tanh", sliding_window=4, query_pre_attn_scalar=16,
                   attn_logit_softcapping=50.0, final_logit_softcapping=30.0, tie_word_embeddings=True),
    "gemma3_text": dict(hidden_activation="gelu_pytorch_tanh", sliding_window=4, query_pre_attn_scalar=16,
                        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
                        tie_word_embeddings=True),
    "olmo2": dict(),
    "granite": dict(attention_multiplier=0.2, embedding_multiplier=3.0, residual_multiplier=0.5, logits_scaling=4.0),
    "smollm3": dict(no_rope_layers=[1, 1, 1, 0]),
    "stablelm": dict(partial_rotary_factor=0.25, use_qkv_bias=True),
    "starcoder2": dict(hidden_act="gelu_pytorch_tanh", use_bias=True, sliding_window=None),
    "cohere": dict(logit_scale=0.5, use_qk_norm=False),
    "helium": dict(),
    "glm4": dict(partial_rotary_factor=0.5),
    "exaone4": dict(),
    "ministral": dict(sliding_window=4),
    # Rope scaling (linear, llama3, yarn), including per-layer-type and partial rotary.
    "llama-rope-linear": dict(model_type="llama", rope_parameters={"rope_type": "linear", "factor": 4.0, "rope_theta": 10000.0}),
    "llama3-scaled": dict(model_type="llama", rope_parameters={
        "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 64, "rope_theta": 500000.0}),
    "qwen2-yarn": dict(model_type="qwen2", rope_parameters={
        "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64, "rope_theta": 10000.0}),
    "llama-yarn-tuned": dict(model_type="llama", rope_parameters={
        "rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 32, "rope_theta": 10000.0,
        "beta_fast": 16.0, "beta_slow": 2.0, "truncate": False, "attention_factor": 1.2}),
    "glm4-linear": dict(model_type="glm4", partial_rotary_factor=0.5, rope_parameters={
        "rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0, "partial_rotary_factor": 0.5}),
    "gemma3-scaled": dict(model_type="gemma3_text", hidden_activation="gelu_pytorch_tanh", sliding_window=4,
                          query_pre_attn_scalar=16, tie_word_embeddings=True,
                          layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
                          rope_parameters={"full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": 1000000.0},
                                           "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}}),
    "olmo3": dict(sliding_window=4,
                  layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]),
}


def build(name, overrides):
    overrides = dict(overrides)
    model_type = overrides.pop("model_type", name)
    try:
        cfg = AutoConfig.for_model(model_type, **{**BASE, **overrides})
    except Exception as e:  # family absent from this transformers
        print(f"skip {name}: {e}")
        return
    torch.manual_seed(1234)
    try:
        # Eager attention: SDPA silently drops Gemma 2's attention softcap,
        # so its logits are not the model's.
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32, attn_implementation="eager").eval()
    except Exception as e:
        print(f"skip {name}: {e}")
        return
    with torch.no_grad():
        for pname, p in model.named_parameters():
            # Non-trivial norms and biases, so a dropped offset or bias shows.
            p.copy_(0.2 * torch.randn_like(p) + (1.0 if p.dim() == 1 and "norm" in pname and model_type not in
                                                 ("gemma", "gemma2", "gemma3_text") else 0.0))
    out = os.path.join(OUT, name)
    shutil.rmtree(out, ignore_errors=True)
    model.save_pretrained(out, safe_serialization=True)
    try:
        with torch.no_grad():
            logits = model(torch.tensor([IDS])).logits[0].float()
    except Exception as e:
        print(f"skip {name}: forward failed: {e}")
        shutil.rmtree(out, ignore_errors=True)
        return
    json.dump({"ids": IDS, "logits": [[round(v, 6) for v in row] for row in logits.tolist()]},
              open(os.path.join(out, "golden.json"), "w"))
    for junk in ("generation_config.json",):
        p = os.path.join(out, junk)
        if os.path.exists(p):
            os.remove(p)
    print(f"built {name} ({model.__class__.__name__})")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    os.makedirs(OUT, exist_ok=True)
    for name, overrides in FAMILIES.items():
        if not only or name in only:
            build(name, overrides)
