"""Builds tests/Amql.Tests/fixtures/spm-bpe: a small SentencePiece-style tokenizer.json
(spaces as U+2581, no split, BPE over characters with byte fallback, as Gemma's) and
golden.json, what the tokenizers library makes of some texts.

Regenerate with (needs tokenizers):
    python make_spm_bpe.py
"""
import json, os
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "spm-bpe")
text = open(os.path.join(HERE, "..", "..", "..", "docs", "entail.md"), encoding="utf-8").read()

t = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True, fuse_unk=True))
t.normalizer = normalizers.Replace(" ", "▁")
t.pre_tokenizer = pre_tokenizers.Split(" ", "merged_with_previous")
t.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()])
special = ["<pad>", "<eos>", "<bos>", "<unk>", "<|turn>", "<turn|>"]
# Trained on words, so merges stay short, then used without a split.
t.pre_tokenizer = pre_tokenizers.Split(Regex("▁?[^▁]+|▁"), "isolated")
t.train_from_iterator([text], trainers.BpeTrainer(vocab_size=900, special_tokens=special + [f"<0x{b:02X}>" for b in range(256)],
                                                  show_progress=False))
t.pre_tokenizer = pre_tokenizers.Split(" ", "merged_with_previous")
t.save(os.path.join(OUT, "tokenizer.json"))
# The byte tokens are vocabulary, not added tokens.
j = json.load(open(os.path.join(OUT, "tokenizer.json"), encoding="utf-8"))
j["added_tokens"] = [a for a in j["added_tokens"] if a["content"] in special]
json.dump(j, open(os.path.join(OUT, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)
t = Tokenizer.from_file(os.path.join(OUT, "tokenizer.json"))
texts = ["", "Hello, world!", "  two  spaces\n\nand newlines\t tab", "def f(x):\n    return x**2  # note",
         "Ünïcödé 日本語 \U0001F389", "<|turn>user\nhi there<turn|>\n", "aaaaaaaaaaaa the the the", " lead", "trail ",
         "0x41 <0x41> is not a byte token here", text[1000:4000]]
json.dump([{"text": x, "ids": t.encode(x, add_special_tokens=False).ids,
            "decoded": t.decode(t.encode(x, add_special_tokens=False).ids, skip_special_tokens=False)} for x in texts],
          open(os.path.join(OUT, "golden.json"), "w"), indent=0)
print("built", len(texts), "texts, vocab", t.get_vocab_size())
