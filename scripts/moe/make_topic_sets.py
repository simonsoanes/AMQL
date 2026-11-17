"""Topic data sets for the branch-adapt-route experiment, as `lora-train` JSONL.

Each line is {"id", "topic", "prompt", "completion"}: the prompt is one user turn
(with its system turn, if any) rendered with the model's own template, thinking off,
and the completion is the recorded assistant reply, cut to a token budget.

Topics, from the post-training sets on the NAS:

    code        reasoning_off.jsonl, replies with a fenced code block
    math        reasoning_off.jsonl, prompts that ask for a calculation or proof
    structured  structured_outputs.jsonl, schema-conforming JSON/XML/YAML replies
    chat        chat_if.jsonl, open conversation

Each topic is split into train / route / eval, disjoint, in file order after a seeded
shuffle of the first records that qualify:

    python scripts/moe/make_topic_sets.py <tokenizer-dir> <data-dir> <out-dir>
        [--topics code,structured,chat] [--train 32] [--route 16] [--eval 24]
        [--max-prompt 320] [--max-completion 96] [--scan 40000] [--seed 7]
"""
import argparse
import json
import os
import random
import re

from transformers import AutoTokenizer

CODE_FENCE = re.compile(r"```[a-zA-Z+#]*\n")
MATH_ASK = re.compile(
    r"\b(solve|calculate|compute|evaluate|prove|integral|derivative|equation|probability|"
    r"how many|what is the (sum|product|value)|find the (value|sum|number))\b", re.I)
MATH_MARK = re.compile(r"[0-9]\s*[-+*/^=]\s*[0-9(x]|\\frac|\\sqrt|\$")
CODE_ASK = re.compile(r"\b(code|function|python|javascript|java|c\+\+|sql|script|bug|error|program)\b", re.I)


def records(path, scan):
    """The file's first `scan` well-formed records. The NAS files hold bare CRs and
    NULs inside records, so lines are split on LF only and bad records skipped."""
    n = 0
    with open(path, encoding="utf-8", errors="replace", newline="\n") as f:
        for line in f:
            if n >= scan:
                return
            line = line.strip()
            if not line or len(line) > 200_000:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            yield r


def first_turn(r):
    """(system, user, assistant) for the record's first exchange, or None."""
    msgs = r.get("messages") or []
    system = ""
    i = 0
    if msgs and msgs[0].get("role") == "system":
        system = msgs[0].get("content") or ""
        i = 1
    if len(msgs) < i + 2 or msgs[i].get("role") != "user" or msgs[i + 1].get("role") != "assistant":
        return None
    user = msgs[i].get("content") or ""
    reply = msgs[i + 1].get("content") or ""
    if not isinstance(user, str) or not isinstance(reply, str) or not user.strip() or not reply.strip():
        return None
    if "<think>" in reply or "<tool_call>" in reply:
        return None
    return system, user, reply


def is_topic(topic, system, user, reply):
    if topic == "code":
        return bool(CODE_FENCE.search(reply)) and bool(CODE_ASK.search(user)) and not MATH_ASK.search(user)
    if topic == "math":
        return bool(MATH_ASK.search(user)) and bool(MATH_MARK.search(user + reply)) and not CODE_FENCE.search(reply)
    if topic == "structured":
        s = reply.lstrip()
        return s.startswith(("{", "[", "<", "```json", "```xml", "```yaml"))
    if topic == "chat":
        return not CODE_FENCE.search(reply) and not MATH_ASK.search(user)
    raise ValueError(topic)


# Topics whose prompts are long documents: the middle of the user turn is cut to fit,
# and the system turn dropped. The reply, which is what the expert learns, is whole.
TRIMMED = {"structured"}


def render(tok, system, user):
    msgs = ([{"role": "system", "content": system}] if system.strip() else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def fit_prompt(tok, system, user, budget, trim):
    prompt = render(tok, system, user)
    n = len(tok(prompt, add_special_tokens=False)["input_ids"])
    if n <= budget:
        return prompt
    if not trim:
        return None
    ids = tok(user, add_special_tokens=False)["input_ids"]
    keep = budget - 40
    if keep < 64:
        return None
    head, tail = ids[: keep * 2 // 3], ids[-(keep - keep * 2 // 3):]
    user = tok.decode(head) + "\n[...]\n" + tok.decode(tail)
    prompt = render(tok, "", user)
    return prompt if len(tok(prompt, add_special_tokens=False)["input_ids"]) <= budget + 8 else None


SOURCES = {
    "code": "reasoning_off.jsonl",
    "math": "reasoning_off.jsonl",
    "structured": "structured_outputs.jsonl",
    "chat": "chat_if.jsonl",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--topics", default="code,structured,chat")
    ap.add_argument("--train", type=int, default=32)
    ap.add_argument("--route", type=int, default=16)
    ap.add_argument("--eval", type=int, default=24)
    ap.add_argument("--max-prompt", type=int, default=320)
    ap.add_argument("--max-completion", type=int, default=96)
    ap.add_argument("--min-completion", type=int, default=48)
    ap.add_argument("--scan", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out_dir, exist_ok=True)
    need = args.train + args.route + args.eval
    seen_users = set()
    summary = {}
    for topic in args.topics.split(","):
        found = []
        for r in records(os.path.join(args.data_dir, SOURCES[topic]), args.scan):
            t = first_turn(r)
            if t is None:
                continue
            system, user, reply = t
            if user in seen_users or not is_topic(topic, system, user, reply):
                continue
            prompt = fit_prompt(tok, system, user, args.max_prompt, trim=topic in TRIMMED)
            if prompt is None:
                continue
            ids = tok(reply, add_special_tokens=False)["input_ids"]
            if len(ids) < args.min_completion:
                continue
            completion = tok.decode(ids[: args.max_completion]) if len(ids) > args.max_completion else reply + "<|im_end|>"
            seen_users.add(user)
            found.append({"topic": topic, "prompt": prompt, "completion": completion,
                          "uuid": r.get("uuid", "")})
            if len(found) >= need * 3:
                break
        if len(found) < need:
            raise SystemExit(f"{topic}: only {len(found)} records qualify, need {need}; raise --scan")
        random.Random(args.seed).shuffle(found)
        found = found[:need]
        splits = {"train": found[: args.train],
                  "route": found[args.train: args.train + args.route],
                  "eval": found[args.train + args.route:]}
        for split, rows in splits.items():
            with open(os.path.join(args.out_dir, f"{topic}.{split}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
                for i, row in enumerate(rows):
                    row = {"id": f"{topic}:{split}:{i}", **row}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[topic] = {k: len(v) for k, v in splits.items()}
        print(topic, summary[topic], flush=True)
    for split in ("train", "route", "eval"):
        with open(os.path.join(args.out_dir, f"all.{split}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for topic in args.topics.split(","):
                with open(os.path.join(args.out_dir, f"{topic}.{split}.jsonl"), encoding="utf-8") as g:
                    f.write(g.read())


if __name__ == "__main__":
    main()
