"""Builds tests/Amql.Tests/fixtures/mrope-zoo: tiny vision-language models
whose text decoders use multimodal rotary (MRoPE), with transformers' own
logits for two position layouts:
  golden.json         — ids at positions 0..n-1 (text: every stream equal)
  golden_streams.json — the same ids with (t, h, w) streams laid out as text,
                        then a 2×3 image grid, then text again
Qwen2-VL splits the rotary pairs into contiguous sections; Qwen3-VL
interleaves them. The vision towers are present only because the
checkpoints carry them; AMQL imports the text decoder alone.

Regenerate with (needs torch, transformers>=5):
    python make_mrope_zoo.py
"""
import json, os, shutil
import torch
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration, Qwen3VLConfig, Qwen3VLForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mrope-zoo")
VOCAB = 128
IDS = [1, 7, 42, 3, 99, 17, 5, 64, 23, 11, 2, 88]
# Text 0..3, a 2×3 image grid at t = 4, then text from 7 (the grid's extent).
T = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 7, 8]
H = [0, 1, 2, 3, 4, 4, 4, 5, 5, 5, 7, 8]
W = [0, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8]

TEXT = dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
            rms_norm_eps=1e-6, tie_word_embeddings=False, bos_token_id=1, eos_token_id=2, pad_token_id=0)

MODELS = {
    "qwen2-vl": (Qwen2VLConfig, Qwen2VLForConditionalGeneration,
                 dict(text_config=dict(TEXT, rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                                              "mrope_section": [2, 3, 3]}),
                      vision_config=dict(depth=1, embed_dim=32, hidden_size=64, num_heads=2, in_channels=3,
                                         patch_size=14, spatial_merge_size=2, temporal_patch_size=2))),
    "qwen3-vl": (Qwen3VLConfig, Qwen3VLForConditionalGeneration,
                 dict(text_config=dict(TEXT, rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                                              "mrope_section": [2, 3, 3], "mrope_interleaved": True}),
                      vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                                         patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                                         num_position_embeddings=64, deepstack_visual_indexes=[]))),
}


def logits(model, position_ids):
    lm = model.model.language_model
    with torch.no_grad():
        hidden = lm(input_ids=torch.tensor([IDS]), position_ids=position_ids).last_hidden_state
        return model.lm_head(hidden)[0].float()


def save(path, rows, **extra):
    json.dump({"ids": IDS, **extra, "logits": [[round(v, 6) for v in row] for row in rows.tolist()]}, open(path, "w"))


for name, (config_class, model_class, kwargs) in MODELS.items():
    torch.manual_seed(1234)
    model = model_class(config_class(**kwargs)).eval()
    with torch.no_grad():
        for pname, p in model.named_parameters():
            p.copy_(0.2 * torch.randn_like(p) + (1.0 if p.dim() == 1 and "norm" in pname else 0.0))
    out = os.path.join(OUT, name)
    shutil.rmtree(out, ignore_errors=True)
    model.save_pretrained(out, safe_serialization=True)
    text = torch.arange(len(IDS))[None, None, :].expand(4, 1, -1)
    streams = torch.stack([torch.arange(len(IDS)), torch.tensor(T), torch.tensor(H), torch.tensor(W)])[:, None, :]
    plain = logits(model, text)
    grid = logits(model, streams)
    save(os.path.join(out, "golden.json"), plain)
    save(os.path.join(out, "golden_streams.json"), grid, streams=[T, H, W])
    for junk in ("generation_config.json",):
        if os.path.exists(os.path.join(out, junk)):
            os.remove(os.path.join(out, junk))
    print(f"built {name}: streams move the logits by {float((plain - grid).abs().max()):.3g}")
