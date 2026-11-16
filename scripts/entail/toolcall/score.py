"""Scores lora-train report.json replies by meaning, not by exact tokens.

called  the reply opens with <tool_call> (or, for no-tool cases, does not)
tool    the first call names the expected function
args    every expected argument is present with the expected value (extra optional ones allowed)

    python scripts/entail/toolcall/score.py <report.json> [<report.json> ...]
"""
import json
import re
import sys

CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def parse(text):
    m = CALL.search(text)
    if not m:
        return None, {}
    return m.group(1).strip(), {k.strip(): v.strip() for k, v in PARAM.findall(m.group(2))}


def score(case, field):
    reply, expected = case[field], case["expected"]
    want_tool, want_args = parse(expected)
    opens = reply.lstrip().startswith("<tool_call>")
    got_tool, got_args = parse(reply)
    if want_tool is None:
        return {"called": not opens and got_tool is None, "tool": True, "args": True}
    return {"called": opens, "tool": got_tool == want_tool,
            "args": got_tool == want_tool and all(got_args.get(k, "").lower() == v.lower() for k, v in want_args.items())}


def main():
    for path in sys.argv[1:]:
        r = json.load(open(path, encoding="utf-8"))
        cases = [{**c, "id": c["id"] if c["id"].count(":") else "TRAIN"} for c in r["train"]] + r["eval"]
        print(f"== {path}")
        totals = {}
        for c in cases:
            b, a = score(c, "before"), score(c, "after")
            suite = c["id"].split(":")[0] if ":" in c["id"] else "train"
            for when, s in (("before", b), ("after", a)):
                t = totals.setdefault((suite, when), [0, 0, 0, 0])
                t[0] += s["called"]; t[1] += s["called"] and s["tool"]; t[2] += s["called"] and s["tool"] and s["args"]; t[3] += 1
            mark = lambda s: ("C" if s["called"] else "-") + ("T" if s["tool"] else "-") + ("A" if s["args"] else "-")
            print(f"  {c['id']:22} {mark(b)} -> {mark(a)}   {c['after'][:90]!r}")
        for (suite, when), (c_, t_, a_, n) in sorted(totals.items()):
            print(f"  {suite:6} {when:6}: decision {c_}/{n}  tool {t_}/{n}  full {a_}/{n}")


if __name__ == "__main__":
    main()
