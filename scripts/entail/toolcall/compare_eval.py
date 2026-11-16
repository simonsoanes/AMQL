"""Compares `entail eval` reports on the tool cases, by meaning.

opens  the reply opens with <tool_call> (for a no-tool case: it doesn't call at all)
calls  a call appears anywhere in the reply (reasoning first is allowed by the template)
tool   that call names the expected function
full   and every expected argument has the expected value

    python scripts/entail/toolcall/compare_eval.py <cases.jsonl> <name>=<report.json> ...
"""
import json
import sys

from score import parse


def judge(expected, reply):
    want_tool, want_args = parse(expected)
    got_tool, got_args = parse(reply)
    opens = reply.lstrip().startswith("<tool_call>")
    if want_tool is None:
        quiet = got_tool is None
        return {"opens": quiet, "calls": quiet, "tool": quiet, "full": quiet}
    full = got_tool == want_tool and all(got_args.get(k, "").lower() == v.lower() for k, v in want_args.items())
    return {"opens": opens, "calls": got_tool is not None, "tool": got_tool == want_tool, "full": full}


def main():
    cases = {json.loads(l)["category"]: json.loads(l) for l in open(sys.argv[1], encoding="utf-8")}
    runs = []
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1)
        report = json.load(open(path, encoding="utf-8"))
        runs.append((name, {c["category"]: c for c in report["cases"]}))
    keys = ("opens", "calls", "tool", "full")
    totals = {}
    for cid, case in cases.items():
        suite = cid.split(":")[0]
        marks = []
        for name, rows in runs:
            row = rows.get(cid)
            if row is None:
                marks.append("  ?  ")
                continue
            j = judge(case["completion"], row["reply"])
            t = totals.setdefault((suite, name), {k: 0 for k in keys} | {"n": 0})
            for k in keys:
                t[k] += j[k]
            t["n"] += 1
            marks.append("".join("OCTF"[i] if j[k] else "." for i, k in enumerate(keys)))
        print(f"{cid:22} " + "  ".join(f"{n}:{m}" for (n, _), m in zip(runs, marks)) +
              f"   {runs[-1][1].get(cid, {}).get('reply', '')[:70]!r}")
    print()
    for (suite, name), t in sorted(totals.items()):
        print(f"{suite:3} {name:10} " + "  ".join(f"{k} {t[k]}/{t['n']}" for k in keys))


if __name__ == "__main__":
    main()
