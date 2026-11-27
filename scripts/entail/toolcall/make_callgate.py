"""Writes call-gate.entail.txt: a program that decides whether a reply opens
with a tool call, and nothing else.

The 0.8B model writes good calls once it starts one, but often starts in prose
instead. The gate is a fold over the whole prompt (carry, so any length):

  * words in the system turn mark which concepts the offered tools cover
  * words in the latest user turn mark which concepts the request touches
  * on the reply's first token, if a requested concept is offered, open a call

The turn, reasoning and call markers are chat marks ({mark:turn_open} and so on),
spelt in the partner's own tokens by `amql-cli entail adapter`, so one program
serves Qwen's <|im_start|>/<tool_call> and Gemma 4's <|turn>/<|tool_call> alike.

Otherwise it is silent and the model decides. The lexicon below is written by
hand from the tools' descriptions plus common words for each concept, BEFORE
looking at any gated reply; deriving it is the open part (see the write-up).

    python scripts/entail/toolcall/make_callgate.py <tokenizer-dir> <out.entail.txt>
"""
import sys

from transformers import AutoTokenizer

CONCEPTS = {
    "weather": "weather forecast temperature rain raining snow sunny cold hot warm coat umbrella jacket picnic",
    "timer": "timer countdown remind reminder alarm minutes minute seconds",
    "money": "currency convert money euros euro dollars dollar pounds yen francs owe exchange",
    "email": "email mail recipient send message",
    "travel": "flights flight fly flying travel trip airport",
    "order": "order orders cancel cancelling address delivery status refund",
    "account": "account delete verification verify code",
}


def single_tokens(tok, word):
    out = set()
    for variant in (word, " " + word, word.capitalize(), " " + word.capitalize()):
        ids = tok(variant, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            out.add(tok.decode(ids))
    return sorted(out)


def esc(text):
    return text.replace("\\", "\\e").replace('"', "\\q").replace("\n", "\\n").replace("{", "\\{")


def main():
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    names = list(CONCEPTS)
    lines = ["program call_gate",
             "# Says <tool_call> on the reply's first token when the request touches a concept an",
             "# offered tool covers. A fold over the prompt: memory is one row, whatever its length.",
             "",
             "state im_start, is_sys, is_user, is_asst, think_end, blank",
             'token "{mark:turn_open}" means im_start = 1',
             'token "{mark:system}" means is_sys = 1',
             'token "{mark:user}" means is_user = 1',
             'token "{mark:assistant}" means is_asst = 1',
             'token "{mark:think_close}" means think_end = 1',
             'token "{mark:reply_open}" means blank = 1',
             "",
             f"state cw[{len(names)}]"]
    dropped = []
    for c, name in enumerate(names):
        lines.append(f"# {name}")
        for word in CONCEPTS[name].split():
            pieces = single_tokens(tok, word)
            if not pieces:
                dropped.append(word)
            for p in pieces:
                lines.append(f'token "{esc(p)}" means cw[{c}] = 1')
    n = len(names)
    lines += f"""
# Which turn this token is in, held until the turn's next opening mark.
state after_start, prev_think
carry after_start from im_start
carry prev_think from think_end
state role_sys, last_sys, role_user, last_user, role_asst, last_asst
carry last_sys from role_sys
carry last_user from role_user
carry last_asst from role_asst
entail role_sys = last_sys when not im_start
entail role_sys when after_start, is_sys
entail role_user = last_user when not im_start
entail role_user when after_start, is_user
entail role_asst = last_asst when not im_start
entail role_asst when after_start, is_asst

# Concepts offered (system turn) and asked (the latest user turn).
state user_begins
entail user_begins when after_start, is_user
state offered[{n}], last_offered[{n}], asked[{n}], last_asked[{n}]
for c in 0 to {n - 1}
    carry last_offered[c] from offered[c]
    carry last_asked[c] from asked[c]
    entail offered[c] = last_offered[c]
    entail offered[c] when role_sys, cw[c], not last_offered[c]
    entail asked[c] = last_asked[c] when not user_begins
    entail asked[c] when role_user, cw[c], not last_asked[c]
end

# The reply's first token comes after its lead: the close mark itself (Gemma 4), or
# what the template writes straight after it ("\\n\\n" after Qwen's </think>).
state hit, call
for c in 0 to {n - 1}
    entail hit when offered[c], asked[c]
end
state closed
entail closed when think_end
entail closed when prev_think
entail call when blank, closed, role_asst, hit >= 1
emit "{{mark:call_open}}" when call
""".splitlines()
    open(sys.argv[2], "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print(f"wrote {sys.argv[2]}; words with no single-token spelling: {' '.join(dropped) or 'none'}")


if __name__ == "__main__":
    main()
