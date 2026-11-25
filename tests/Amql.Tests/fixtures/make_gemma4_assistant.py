"""A tiny random Gemma 4 assistant (drafter) for the arch-zoo's gemma4-moe, with the drafts
transformers gives after that model has read the zoo's tokens.

    python tests/Amql.Tests/fixtures/make_gemma4_assistant.py

golden.json: ids (what the model read), token (chosen after it), drafts, and logits
([len(drafts), vocab]).
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, Gemma4AssistantForCausalLM
from transformers.models.gemma4_assistant.configuration_gemma4_assistant import Gemma4AssistantConfig

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "arch-zoo", "gemma4-moe")
OUT = os.path.join(HERE, "gemma4-assistant")
COUNT = 5


def main():
    target = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.float32, attn_implementation="eager").eval()
    text = json.load(open(os.path.join(TARGET, "config.json")))
    base = dict(vocab_size=text["vocab_size"], hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_kv_shared_layers=4,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32, num_global_key_value_heads=1,
                attention_k_eq_v=True, hidden_size_per_layer_input=0, vocab_size_per_layer_input=0, sliding_window=text["sliding_window"],
                hidden_activation="gelu_pytorch_tanh", final_logit_softcapping=None, max_position_embeddings=256,
                layer_types=["sliding_attention"] * 3 + ["full_attention"], rope_parameters=text["rope_parameters"],
                tie_word_embeddings=True, rms_norm_eps=text["rms_norm_eps"], pad_token_id=0, bos_token_id=1, eos_token_id=2)
    config = Gemma4AssistantConfig(text_config=base, backbone_hidden_size=text["hidden_size"], use_ordered_embeddings=False, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    model = Gemma4AssistantForCausalLM(config).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("layer_scalar"):
                p.fill_(0.75)
            elif p.dim() == 2:
                p.normal_(0, 0.08)
    for name, buffer in model.named_buffers():
        if name.endswith("layer_scalar"):
            buffer.fill_(0.75)
    model.save_pretrained(OUT)

    ids = json.load(open(os.path.join(TARGET, "golden.json")))["ids"]
    with torch.no_grad():
        out = target(torch.tensor([ids]), output_hidden_states=True, return_shared_kv_states=True, use_cache=True)
        token = torch.tensor([[int(out.logits[0, -1].argmax())]])
        first = int(token)
        hidden = out.hidden_states[-1][:, -1:]
        position = torch.tensor([[len(ids)]])
        drafts, logits = [], []
        for _ in range(COUNT):
            embeds = torch.cat([target.get_input_embeddings()(token), hidden], dim=-1)
            r = model(inputs_embeds=embeds, position_ids=position, shared_kv_states=out.shared_kv_states)
            logits.append(r.logits[0, 0].tolist())
            token = r.logits.argmax(-1)
            drafts.append(int(token))
            hidden = r.last_hidden_state
    json.dump({"ids": ids, "token": first, "drafts": drafts, "logits": logits}, open(os.path.join(OUT, "golden.json"), "w"))
    print({k: v.shape for k, v in out.shared_kv_states.items()} if hasattr(next(iter(out.shared_kv_states.values())), "shape")
          else {k: v[0].shape for k, v in out.shared_kv_states.items()}, drafts, os.listdir(OUT))


if __name__ == "__main__":
    main()
