"""Scores `entail eval` reports on the real decision points (make_real_cases.py).

For call turns (the recorded assistant called a tool):
  opens  the reply opens with <tool_call>
  calls  a call appears anywhere in the reply
  tool   it names the recorded tool
  full   and every recorded argument has the recorded value
For text turns (the recorded assistant wrote to the user):
  quiet  the reply makes no call

    python scripts/entail/toolcall/score_real.py <cases.jsonl> <name>=<report.json> ... [--show name]
"""
import json
import sys

from score import parse


def judge(case, reply):
    want, want_args = parse(case["completion"])
    got, got_args = parse(reply)
    if want is None:
        return {"quiet": got is None}
    norm = lambda v: " ".join(v.lower().split())
    full = got == want and all(norm(got_args.get(k, "")) == norm(v) for k, v in want_args.items())
    return {"opens": reply.lstrip().startswith("<tool_call>"), "calls": got is not None, "tool": got == want,
            "full": full}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = sys.argv[sys.argv.index("--show") + 1] if "--show" in sys.argv else None
    cases = {json.loads(l)["category"]: json.loads(l) for l in open(args[0], encoding="utf-8")}
    for arg in args[1:]:
        if arg == show:
            continue
        name, path = arg.split("=", 1)
        rows = {c["category"]: c for c in json.load(open(path, encoding="utf-8"))["cases"]}
        totals = {}
        for cid, case in cases.items():
            if cid not in rows:
                continue
            j = judge(case, rows[cid]["reply"])
            groups = [cid.split(":")[0]]
            if groups[0] == "call":
                groups.append(f"call after {case['after']}")
            for g in groups:
                t = totals.setdefault(g, {"n": 0})
                t["n"] += 1
                for k, v in j.items():
                    t[k] = t.get(k, 0) + v
            if name == show:
                mark = "".join(k[0].upper() if v else "." for k, v in j.items())
                print(f"{cid:18} {mark:4} want {parse(case['completion'])[0]!s:28} {rows[cid]['reply'][:90]!r}")
        line = "  ".join(f"{g}: " + " ".join(f"{k} {v}/{t['n']}" for k, v in t.items() if k != "n")
                         for g, t in sorted(totals.items()))
        print(f"{name:12} {line}")


if __name__ == "__main__":
    main()
