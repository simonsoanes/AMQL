"""Builds tests/Amql.Tests/fixtures/qwen-lm-tiny: a tiny, randomly initialised
Qwen3.5 causal LM (GatedDeltaNet + gated attention) with a byte-level tokenizer
that carries the ChatML/think special tokens, Qwen3.5's real chat template
(qwen35_chat_template.jinja, from Qwen3.5 checkpoints, Apache-2.0), and a
generation_config.json. golden.json holds what transformers produces:
  chat    — apply_chat_template(messages, add_generation_prompt=True) renderings
  greedy  — model.generate(do_sample=False) continuations of rendered prompts
  embed   — mean over tokens of the final (post-norm) hidden state, L2-normalised
Regenerate with (needs torch, transformers>=5, tokenizers, jinja2):
    python make_qwen_lm_tiny.py
"""
import json, os, shutil
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors, AddedToken
from transformers import PreTrainedTokenizerFast, Qwen3_5TextConfig, Qwen3_5ForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "qwen-lm-tiny")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(99)

corpus = ["hello how are you today", "the capital of france is paris", "I am a helpful assistant",
          "what is two plus two", "café 日本 😀 naïve", "system user assistant think"]
tok = Tokenizer(models.BPE(unk_token=None))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
tok.decoder = decoders.ByteLevel()
tok.post_processor = processors.ByteLevel(trim_offsets=False)
tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["<|endoftext|>"],
                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok.add_special_tokens([AddedToken("<|im_start|>", special=True), AddedToken("<|im_end|>", special=True)])
tok.add_tokens([AddedToken("<think>", special=False), AddedToken("</think>", special=False)])
tok.save(os.path.join(OUT, "tokenizer.json"))
im_end = tok.token_to_id("<|im_end|>")
template = open(os.path.join(HERE, "qwen35_chat_template.jinja")).read()
shutil.copy(os.path.join(HERE, "qwen35_chat_template.jinja"), os.path.join(OUT, "chat_template.jinja"))
json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"},
          open(os.path.join(OUT, "tokenizer_config.json"), "w"), indent=1)

cfg = Qwen3_5TextConfig(vocab_size=tok.get_vocab_size(), hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
                        linear_conv_kernel_dim=4, max_position_embeddings=512,
                        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
                        tie_word_embeddings=True, eos_token_id=im_end, pad_token_id=tok.token_to_id("<|endoftext|>"))
model = Qwen3_5ForCausalLM(cfg).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name:
            p.copy_(0.3 * torch.randn_like(p))
        elif "A_log" in name or "dt_bias" in name:
            p.copy_(0.5 * torch.randn_like(p))
        else:
            p.copy_(0.3 * torch.randn_like(p))
model.save_pretrained(OUT, safe_serialization=True)
config = json.load(open(os.path.join(OUT, "config.json")))
config["attn_output_gate"] = True
json.dump(config, open(os.path.join(OUT, "config.json"), "w"), indent=1)
json.dump({"eos_token_id": [im_end], "pad_token_id": cfg.pad_token_id},
          open(os.path.join(OUT, "generation_config.json"), "w"), indent=1)

hf = PreTrainedTokenizerFast(tokenizer_file=os.path.join(OUT, "tokenizer.json"), eos_token="<|im_end|>")
hf.chat_template = template
conversations = [
    [{"role": "user", "content": "hello how are you today"}],
    [{"role": "system", "content": "  I am a helpful assistant \n"}, {"role": "user", "content": "what is two plus two"}],
    [{"role": "user", "content": "the capital of france"}, {"role": "assistant", "content": "<think>\nhmm\n</think>\n\nparis"},
     {"role": "user", "content": "café 日本 😀"}],
]
golden = {"chat": [], "greedy": [], "embed": []}
for msgs in conversations:
    for thinking in (False, True):
        golden["chat"].append({"messages": msgs, "enable_thinking": thinking,
                               "text": hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                              enable_thinking=thinking)})
for msgs in conversations:
    text = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = hf(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=8, do_sample=False, eos_token_id=[im_end], pad_token_id=cfg.pad_token_id)
    golden["greedy"].append({"messages": msgs, "prompt_ids": ids[0].tolist(), "generated": out[0, ids.shape[1]:].tolist()})
for text in ["hello how are you today", "the capital of france is paris", "😀"]:
    ids = hf(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        hidden = model.model(input_ids=ids).last_hidden_state[0]
    v = hidden.mean(0)
    golden["embed"].append({"input": text, "ids": ids[0].tolist(), "embedding": (v / v.norm()).tolist()})
json.dump(golden, open(os.path.join(OUT, "golden.json"), "w"), indent=1)
print("wrote", OUT, sorted(os.listdir(OUT)))
