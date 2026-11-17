"""How well the partner's own first-token p(<tool_call>) separates call turns from text turns.

    python scripts/entail/toolcall/watch_auc.py <report.json> [<report.json> ...]
"""
import json
import math
import sys


def main():
    for path in sys.argv[1:]:
        rows = json.load(open(path, encoding="utf-8"))["cases"]
        call = sorted(r["watch"] for r in rows if r["category"].startswith("call"))
        text = sorted(r["watch"] for r in rows if r["category"].startswith("text"))
        auc = sum((c > t) + 0.5 * (c == t) for c in call for t in text) / (len(call) * len(text))
        print(f"== {path}: AUC {auc:.3f}, {len(call)} call turns, {len(text)} text turns")
        for tau in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005):
            tp = sum(c >= tau for c in call)
            fp = sum(t >= tau for t in text)
            print(f"  call when p >= {tau:<6} calls {tp:3}/{len(call)}  wrongly on text turns {fp:2}/{len(text)}")
        q = lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))]
        for name, xs in (("call", call), ("text", text)):
            print(f"  {name}: log10 p quartiles " + " ".join(f"{math.log10(max(q(xs, f), 1e-12)):.2f}" for f in (0.1, 0.25, 0.5, 0.75, 0.9)))


if __name__ == "__main__":
    main()
