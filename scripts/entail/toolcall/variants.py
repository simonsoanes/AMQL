"""Rewrites rendered tool cases into prompt variants, to find what moves the call decision.

Every transform is generic: it uses only the template's own structure and the names of
the offered tools, never the request's words.

  base      unchanged
  noimp     the <IMPORTANT> bullets that invite prose are removed ("answer the question like
            normal", "optional reasoning ... BEFORE the function call")
  remind    a system line just before the reply names the functions on offer
  both      noimp and remind
  desc      noimp, and a system line just before the reply listing every function with its
            description
  list      noimp, and the functions listed again just before the reply, with no instruction
  listpre   noimp, and that list just before the latest turn
  null      noimp, with the latest message's content replaced by "N/A" (the prior)
  ask       a hidden question replaces the reply: "should your reply be a function call?"
            (score it with --watch Yes --max-tokens 1)

    python scripts/entail/toolcall/variants.py <cases.jsonl> <out-dir> [variant ...]
"""
import json
import sys
from pathlib import Path

GEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
INVITE_PROSE = [
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n",
    "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n",
]


def noimp(prompt, tools):
    for line in INVITE_PROSE:
        prompt = prompt.replace(line, "", 1)
    return prompt


def remind(prompt, tools):
    assert prompt.endswith(GEN)
    line = ("Functions you can call now: " + ", ".join(tools) +
            ". If one of them does what the user needs, reply with the call.")
    return prompt[: -len(GEN)] + f"<|im_start|>system\n{line}<|im_end|>\n" + GEN


def offered(prompt):
    """The offered functions, read back from the template's <tools> block: (name, description)."""
    block = prompt.split("<tools>\n", 1)[1].split("</tools>", 1)[0]
    out = []
    for line in block.splitlines():
        if line.strip():
            f = json.loads(line)
            f = f.get("function", f)
            out.append((f["name"], f.get("description", "").strip()))
    return out


def remind_desc(prompt, tools):
    """Every function's name and description again, just before the reply, so a tool listed
    first and one listed last are equally near the decision."""
    assert prompt.endswith(GEN)
    lines = "\n".join(f"- {n}: {d}" for n, d in offered(prompt))
    text = f"Functions you can call now:\n{lines}\nIf one of them does what the user needs, reply with the call."
    return prompt[: -len(GEN)] + f"<|im_start|>system\n{text}<|im_end|>\n" + GEN


def listing(prompt):
    return "Functions on offer:\n" + "\n".join(f"- {n}: {d}" for n, d in offered(prompt))


def list_last(prompt, tools):
    """The functions listed again just before the reply, with no instruction."""
    assert prompt.endswith(GEN)
    return prompt[: -len(GEN)] + f"<|im_start|>system\n{listing(prompt)}<|im_end|>\n" + GEN


def list_before_turn(prompt, tools):
    """The functions listed again just before the latest turn, so the request stays last."""
    at = prompt.rindex("<|im_start|>user\n")
    return prompt[:at] + f"<|im_start|>system\n{listing(prompt)}<|im_end|>\n" + prompt[at:]


def null(prompt, tools):
    """The latest message's content replaced by "N/A": the same system prompt, tools and
    history, with nothing to act on. Its p(<tool_call>) is the conversation's prior."""
    at = prompt.rindex("<|im_start|>user\n") + len("<|im_start|>user\n")
    end = prompt.rindex("<|im_end|>\n" + GEN)
    body = prompt[at:end]
    blank = "<tool_response>\nN/A\n</tool_response>" if body.startswith("<tool_response>") else "N/A"
    return prompt[:at] + blank + prompt[end:]


def ask(prompt, tools):
    assert prompt.endswith(GEN)
    question = ("Before you reply: should your next reply be a call to one of the functions ("
                + ", ".join(tools) + ")? Answer Yes or No.")
    return prompt[: -len(GEN)] + f"<|im_start|>user\n{question}<|im_end|>\n" + GEN


VARIANTS = {
    "base": lambda p, t: p,
    "noimp": noimp,
    "remind": remind,
    "both": lambda p, t: remind(noimp(p, t), t),
    "ask": ask,
    "desc": lambda p, t: remind_desc(noimp(p, t), t),
    "list": lambda p, t: list_last(noimp(p, t), t),
    "listpre": lambda p, t: list_before_turn(noimp(p, t), t),
    "null": lambda p, t: null(noimp(p, t), t),
}


def main():
    src, out = sys.argv[1], Path(sys.argv[2])
    names = sys.argv[3:] or list(VARIANTS)
    out.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(l) for l in open(src, encoding="utf-8")]
    for name in names:
        with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for c in cases:
                f.write(json.dumps({**c, "prompt": VARIANTS[name](c["prompt"], c["tools"])}, ensure_ascii=False) + "\n")
    print(f"wrote {len(names)} variants of {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
