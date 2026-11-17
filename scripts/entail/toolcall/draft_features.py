"""What a forced draft call says about whether a call was due.

Reads an `entail eval` report run with --prefix "<tool_call>\\n<function=" --logprobs,
so every reply is a call the model was made to start. For each draft it measures:

  name_p    probability of the tool name's tokens (product)
  name_p0   probability of the name's first token
  value_p   lowest, over the arguments, of the probability of a value's first token
  grounded  share of argument values (3+ characters) found verbatim in the prompt
  offered   the name is one of the offered tools

and how well each separates call turns from text turns (AUC), plus how often the draft
names the recorded tool on the call turns.

    python scripts/entail/toolcall/draft_features.py <cases.jsonl> <report.json>
"""
import json
import math
import re
import sys

from score import parse


def features(case, row):
    toks, ps = row["logprobs"]["tokens"], row["logprobs"]["p"]
    text = ""
    name_p, name_p0, in_name = 1.0, None, True
    value_ps, at_value_start = [], False
    for t, p in zip(toks, ps):
        if in_name:
            if ">" in t:
                in_name = False
            else:
                name_p *= p
                name_p0 = p if name_p0 is None else name_p0
        if at_value_start and t.strip():
            value_ps.append(p)
            at_value_start = False
        text += t
        if re.search(r"<parameter=[^>\n]+>\n?$", text):
            at_value_start = True
    name, args = parse("<tool_call>\n<function=" + text)
    prompt = case["prompt"].lower()
    values = [v for v in args.values() if len(v) >= 3]
    grounded = sum(v.lower() in prompt for v in values) / len(values) if values else 1.0
    return {"name_p": name_p, "name_p0": name_p0 or 0.0, "value_p": min(value_ps) if value_ps else 1.0,
            "grounded": grounded, "offered": float(name in case["tools"]),
            "name": name, "values": values}


def auc(pos, neg):
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / max(1, len(pos) * len(neg))


def main():
    cases = {json.loads(l)["category"]: json.loads(l) for l in open(sys.argv[1], encoding="utf-8")}
    rows = {r["category"]: r for r in json.load(open(sys.argv[2], encoding="utf-8"))["cases"]}
    feats = {cid: features(cases[cid], rows[cid]) for cid in cases if cid in rows}
    call = [cid for cid in feats if cid.startswith("call")]
    text = [cid for cid in feats if cid.startswith("text")]
    right = sum(feats[c]["name"] == parse(cases[c]["completion"])[0] for c in call)
    print(f"draft names the recorded tool on {right}/{len(call)} call turns")
    for k in ("name_p", "name_p0", "value_p", "grounded", "offered"):
        print(f"  {k:9} AUC {auc([feats[c][k] for c in call], [feats[t][k] for t in text]):.3f}   "
              f"median call {sorted(feats[c][k] for c in call)[len(call) // 2]:.3f}  "
              f"text {sorted(feats[t][k] for t in text)[len(text) // 2]:.3f}")
    combo = lambda f: math.log(max(f["name_p"], 1e-9)) + math.log(max(f["value_p"], 1e-9)) + 3 * f["grounded"]
    print(f"  combined  AUC {auc([combo(feats[c]) for c in call], [combo(feats[t]) for t in text]):.3f}")
    json.dump(feats, open(sys.argv[2].replace(".json", ".features.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
