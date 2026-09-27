"""Builds tests/Amql.Tests/fixtures/von-tiny: a tiny, randomly initialised
checkpoint laid out exactly like wfzyx/von (ModernBERT backbone in
model.safetensors, the full OptionMarkerModel state dict in option_marker.pt,
marker_calibration.json, a byte-level BPE tokenizer with an lstrip [MASK]),
plus golden.json: the Von SDK's own answers and raw encoder outputs for a set
of requests. AMQL's encoder and `decide` are tested against those goldens.

Regenerate with (needs torch, transformers>=5, tokenizers, pydantic and a
checkout of https://github.com/wfzyx/von):
    VON_SRC=/path/to/von/src python make_von_tiny.py
"""
import json, os, sys, warnings
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors, AddedToken
from transformers import ModernBertConfig, ModernBertModel

warnings.filterwarnings("ignore")
sys.path.insert(0, os.environ["VON_SRC"])
from von.models.option_marker import (OptionMarkerModel, build_independent_option_masks,
                                      build_option_invariant_position_ids)
from von.backends.option_marker_backend import OptionMarkerBackend

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "von-tiny")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(1234)

# ── tokenizer ────────────────────────────────────────────────────────────
corpus = ["the customer was charged twice for the order and wants a refund now",
          "refunds above five hundred dollars need human approval under the policy",
          "which team should handle billing technical or sales questions",
          "is the message urgent does it convey time pressure yes or no",
          "score the risk low moderate high critical", "0 1 2 3 4 5 6 7 8 9 { } [ ] : , ' \" ."]
specials = ["[UNK]", "[CLS]", "[SEP]", "[PAD]"]
tok = Tokenizer(models.BPE(unk_token=None))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
tok.decoder = decoders.ByteLevel()
tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=320, special_tokens=specials,
                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok.add_special_tokens([AddedToken("[MASK]", lstrip=True, special=True)])
cls, sep = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
                                                   special_tokens=[("[CLS]", cls), ("[SEP]", sep)])
tok.save(os.path.join(OUT, "tokenizer.json"))
json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "cls_token": "[CLS]", "sep_token": "[SEP]",
           "mask_token": "[MASK]", "pad_token": "[PAD]", "unk_token": "[UNK]", "model_max_length": 512},
          open(os.path.join(OUT, "tokenizer_config.json"), "w"), indent=1)

# ── backbone ─────────────────────────────────────────────────────────────
cfg = ModernBertConfig(vocab_size=tok.get_vocab_size(), hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                       num_attention_heads=4, global_attn_every_n_layers=3, local_attention=8,
                       max_position_embeddings=512, pad_token_id=tok.token_to_id("[PAD]"), bos_token_id=cls,
                       eos_token_id=sep, cls_token_id=cls, sep_token_id=sep, norm_bias=False, mlp_bias=False,
                       attention_bias=False, global_rope_theta=160000.0, local_rope_theta=10000.0)
backbone = ModernBertModel(cfg)
with torch.no_grad():
    for name, p in backbone.named_parameters():
        # Non-trivial norms, so a kernel that ignores a norm weight fails.
        p.copy_(1.0 + 0.2 * torch.randn_like(p) if name.endswith("norm.weight") else 0.08 * torch.randn_like(p))
backbone.save_pretrained(OUT, safe_serialization=True)

model = OptionMarkerModel(base_model_id=OUT, max_position_embeddings=512)
with torch.no_grad():
    for name, p in model.scorer.named_parameters():
        p.copy_(1.0 + 0.2 * torch.randn_like(p) if name.endswith("norm.weight") else 0.3 * torch.randn_like(p))
torch.save(model.state_dict(), os.path.join(OUT, "option_marker.pt"))
json.dump({"model_type": "option_marker", "model_id": "von-tiny-test", "independent_options": True,
           "temperature": 1.7,
           "calibration_map": {"bias": 1.2, "entropy": -0.8, "log_tokens": 2.5, "n_options": -0.6, "lo": 0.3, "hi": 12.0},
           "noul_zero_shot_prior": {"a": -0.5, "b": 0.3}},
          open(os.path.join(OUT, "marker_calibration.json"), "w"), indent=1)

# ── goldens ──────────────────────────────────────────────────────────────
golden = {"encoder": [], "decisions": []}
model = OptionMarkerModel(base_model_id=OUT, max_position_embeddings=512)
model.load_state_dict(torch.load(os.path.join(OUT, "option_marker.pt"), weights_only=True), strict=True)
model.eval()
text = model.pack_sequence("the customer was charged twice and the policy says refunds above five hundred "
                           "need human approval so which team should handle it now",
                           "which team?", ["billing refund", "technical", "sales team now", "no"])
enc = model.tokenizer(text, return_tensors="pt")
ids = enc["input_ids"]
pos = [(ids[0] == model.mask_token_id).nonzero(as_tuple=True)[0].tolist()]
with torch.no_grad():
    for independent in (False, True):
        if independent:
            position_ids = build_option_invariant_position_ids(ids, enc["attention_mask"], pos)
            masks = build_independent_option_masks(ids, enc["attention_mask"], pos, position_ids, cfg.sliding_window)
            hidden = model.encoder(input_ids=ids, attention_mask=masks, position_ids=position_ids).last_hidden_state
        else:
            hidden = model.encoder(input_ids=ids, attention_mask=enc["attention_mask"]).last_hidden_state
        logits = model(ids, enc["attention_mask"], pos, independent_options=independent)[0]
        golden["encoder"].append({"independent": independent, "ids": ids[0].tolist(), "markers": pos[0],
                                  "hidden": hidden[0].tolist(), "logits": logits.tolist()})

backend = OptionMarkerBackend(checkpoint_dir=OUT, device="cpu")
requests = [
    {"state": "the customer was charged twice for the order and wants a refund now",
     "questions": {
         "team": {"type": "choice", "instructions": "which team should handle this?",
                  "criteria": {"billing": "billing and refunds", "technical": None, "sales": "sales questions"}},
         "urgent": {"type": "noul", "instructions": "is the message urgent?"},
         "urgent2": {"type": "noul", "instructions": "is the message urgent?",
                     "criteria": {"true": "time pressure", "false": "no time pressure"}},
         "risk": {"type": "score", "instructions": "score the risk",
                  "criteria": ["low", {"what": "moderate", "examples": ["some", "risk"]}, "high", "critical"]}}},
    {"state": {"message": "refund of 680 dollars", "verified": True, "policy": {"limit": 500, "rate": 0.00005}},
     "questions": {"act": {"type": "choice", "instructions": {"question": "which action?", "scope": ["refund", "wait"]},
                           "criteria": {"allow": "refund now", "review": "human approval", "deny": "no"}}}},
    {"state": [{"speaker": "customer", "text": "is it urgent"}, {"speaker": "support", "text": "yes"}],
     "questions": {"q": {"type": "noul", "instructions": ["is the message urgent", "judge intent only"]}}},
]
for r in requests:
    golden["decisions"].append({"request": r, "response": backend.evaluate(state=r["state"], questions=r["questions"]).model_dump()})
json.dump(golden, open(os.path.join(OUT, "golden.json"), "w"))
print("wrote", OUT, sorted(os.listdir(OUT)))
