"""Decision points from real multi-turn tool conversations, as `entail eval` cases.

Each case is one assistant turn of a recorded conversation: the prompt is the whole
conversation before it, rendered with the model's own template (tools in the system
turn, thinking off, earlier reasoning dropped), and the label is what the recorded
assistant did there: called a tool (which one), or wrote to the user.

The recorded assistant is a much larger model, so a text turn is not proof that no
call was possible. The measure that matters is on the call turns: does the small
model call at all, and does it name the same tool.

    python scripts/entail/toolcall/make_real_cases.py <tokenizer-dir> <tool_calling.jsonl> <out.jsonl>
        [--calls 120] [--texts 60] [--max-prompt 6000] [--seed 7] [--template <file.jinja>]
"""
import argparse
import ast
import json
import random

from transformers import AutoTokenizer


def tool_calls(message):
    raw = message.get("tool_calls")
    if not raw:
        return []
    calls = ast.literal_eval(raw) if isinstance(raw, str) else raw
    out = []
    for c in calls:
        f = c.get("function", c)
        args = f.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        out.append({"type": "function", "function": {"name": f["name"], "arguments": args}})
    return out


def clean(messages):
    """Messages as the template wants them: arguments as mappings, no recorded reasoning."""
    out = []
    for m in messages:
        role = m["role"]
        if role == "assistant":
            out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tool_calls(m)})
        elif role == "tool":
            out.append({"role": "tool", "content": m.get("content") or ""})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    return out


def xml_call(call):
    f = call["function"]
    body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                   for k, v in f["arguments"].items())
    return f"<tool_call>\n<function={f['name']}>\n{body}</function>\n</tool_call><|im_end|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--calls", type=int, default=120)
    ap.add_argument("--texts", type=int, default=60)
    ap.add_argument("--max-prompt", type=int, default=6000)
    ap.add_argument("--scan", type=int, default=4000, help="conversations to read")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--template", help="a chat template to render with instead of the tokenizer's")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    template = open(args.template, encoding="utf-8").read() if args.template else None
    rng = random.Random(args.seed)
    calls, texts = [], []
    # newline="\n": the records hold bare \r characters, which must not split a line.
    with open(args.data, encoding="utf-8", errors="replace", newline="\n") as f:
        for n, line in enumerate(f):
            if n >= args.scan:
                break
            try:
                d = json.loads(line, strict=False)
            except json.JSONDecodeError:
                continue  # the file has runs of NUL bytes in places
            tools = d.get("tools") or []
            msgs = d["messages"]
            if not tools:
                continue
            # One decision point per conversation, so no conversation dominates.
            points = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0]
            if not points:
                continue
            i = rng.choice(points)
            label = tool_calls(msgs[i])
            (calls if label else texts).append((n, i, tools, clean(msgs[:i]), label, msgs[i].get("content") or ""))
    rng.shuffle(calls)
    rng.shuffle(texts)

    written = {"call": 0, "text": 0}
    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        for kind, pool, want in (("call", calls, args.calls), ("text", texts, args.texts)):
            for n, i, tools, history, label, content in pool:
                if written[kind] >= want:
                    break
                try:
                    prompt = tok.apply_chat_template(history, tools=tools, tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=False,
                                                     chat_template=template)
                except Exception:
                    continue
                if len(tok(prompt)["input_ids"]) > args.max_prompt:
                    continue
                completion = xml_call(label[0]) if label else content + "<|im_end|>"
                last = next(m for m in reversed(history) if m["role"] in ("user", "tool"))["role"]
                out.write(json.dumps({"category": f"{kind}:{n}:{i}", "prompt": prompt, "answer": "0",
                                      "max_tokens": 96, "completion": completion,
                                      "tools": [t["function"]["name"] for t in tools], "after": last},
                                     ensure_ascii=False) + "\n")
                written[kind] += 1
    print(f"wrote {args.out}: {written['call']} call turns, {written['text']} text turns "
          f"(from {len(calls)} and {len(texts)} candidates)")


if __name__ == "__main__":
    main()
