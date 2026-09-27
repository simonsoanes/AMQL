"""Builds tests/Amql.Tests/fixtures/qwen-cls-tiny: a tiny, randomly initialised
Qwen3.5 sequence classifier laid out like AlexWortega/openjev (GatedDeltaNet +
gated full-attention layers, a top-level `score` head, id2label, nli_template,
pad_token_id), with a `score.bias` added so the bias binding is exercised, and
golden.json: logits computed by transformers over a RIGHT-PADDED batch, pooled
at attention_mask.sum-1 exactly as the reference cross-encoder does.

Regenerate with (needs torch, transformers>=5, tokenizers, safetensors):
    python make_qwen_cls_tiny.py
"""
import json, os
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast, Qwen3_5TextConfig, Qwen3_5TextForSequenceClassification

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen-cls-tiny")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(2024)

corpus = ["Premise: a man is playing a guitar", "Hypothesis: someone is making music",
          "nobody is playing an instrument the man is a famous musician",
          "the order arrived three days late and the policy requires five days"]
tok = Tokenizer(models.BPE(unk_token=None))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
tok.decoder = decoders.ByteLevel()
tok.post_processor = processors.ByteLevel(trim_offsets=False)
tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=["<|pad|>"],
                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok.save(os.path.join(OUT, "tokenizer.json"))
pad = tok.token_to_id("<|pad|>")
json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "pad_token": "<|pad|>", "padding_side": "right"},
          open(os.path.join(OUT, "tokenizer_config.json"), "w"), indent=1)

labels = {0: "contradiction", 1: "entailment", 2: "neutral"}
cfg = Qwen3_5TextConfig(vocab_size=tok.get_vocab_size(), hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
                        linear_conv_kernel_dim=4, max_position_embeddings=512,
                        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                        tie_word_embeddings=True, pad_token_id=pad, num_labels=3, id2label=labels,
                        label2id={v: k for k, v in labels.items()}, problem_type="single_label_classification")
model = Qwen3_5TextForSequenceClassification(cfg).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name:
            p.copy_(0.3 * torch.randn_like(p))       # (1+w) norms and the gated norm: non-trivial weights
        elif "A_log" in name or "dt_bias" in name:
            p.copy_(0.5 * torch.randn_like(p))
        else:
            p.copy_(0.15 * torch.randn_like(p))
model.save_pretrained(OUT, safe_serialization=True)

# Real Qwen3.5 checkpoints declare these; add them the way the openjev config does.
config = json.load(open(os.path.join(OUT, "config.json")))
config["architectures"] = ["Qwen3_5ForSequenceClassification"]
config["attn_output_gate"] = True
config["nli_template"] = "Premise: {premise}\nHypothesis: {hypothesis}"
json.dump(config, open(os.path.join(OUT, "config.json"), "w"), indent=1)

bias = torch.tensor([0.25, -0.5, 0.125])
tensors = load_file(os.path.join(OUT, "model.safetensors"))
tensors["score.bias"] = bias
save_file(tensors, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})

pairs = [("a man is playing a guitar", "someone is making music"),
         ("a man is playing a guitar", "nobody is playing an instrument"),
         ("  the order arrived three days late and the policy requires five days  ", "the man is a famous musician"),
         ("x", "y")]
hf_tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(OUT, "tokenizer.json"), pad_token="<|pad|>")
hf_tok.padding_side = "right"
template = config["nli_template"]
texts = [template.format(premise=p.strip(), hypothesis=h.strip()) for p, h in pairs]
enc = hf_tok(texts, padding=True, return_tensors="pt")
assert enc["attention_mask"].sum(1).unique().numel() > 1, "the batch must actually be padded"
with torch.no_grad():
    hidden = model.model(**enc).last_hidden_state
    last = enc["attention_mask"].sum(1) - 1
    pooled = hidden[torch.arange(hidden.shape[0]), last]
    logits = pooled @ model.score.weight.T + bias
golden = [{"premise": p, "hypothesis": h, "ids": hf_tok(t)["input_ids"], "logits": l.tolist(),
           "probs": torch.softmax(l, -1).tolist()} for (p, h), t, l in zip(pairs, texts, logits)]
json.dump(golden, open(os.path.join(OUT, "golden.json"), "w"), indent=1)
print("wrote", OUT, sorted(os.listdir(OUT)))
