"""Summarises a tool-trace report: is the chosen tool the one attended to, and is
it the one nearest the request in meaning?

    python scripts/entail/toolcall/analyse_trace.py <trace.json>
"""
import json
import sys


def rank(scores, key):
    order = sorted(scores, key=lambda k: -scores[k])
    return order.index(key) + 1 if key in order else None


def main():
    report = json.load(open(sys.argv[1], encoding="utf-8"))
    att_hits, sim_hits = {}, {}
    for case in report:
        tool_decision = next((d for d in case["decisions"] if d["what"].startswith("tool ")), None)
        print(f"\n{case['id']}: {case['reply'][:80]!r}")
        if tool_decision is None:
            print("  (no call)")
            continue
        chosen = tool_decision["what"][5:]
        tools = sorted({s["label"].split(":", 1)[1] for s in case["spans"] if s["label"].startswith("name:")})
        line = []
        for layer, spans in tool_decision["attention"].items():
            for kind in ("name", "desc", "schema"):
                scores = {t: spans.get(f"{kind}:{t}", {}).get("max", 0) for t in tools}
                r = rank(scores, chosen)
                att_hits.setdefault((layer, kind), []).append(r == 1)
            names = {t: spans[f"name:{t}"]["max"] for t in tools}
            line.append(f"{layer}:{max(names, key=names.get)}({names[chosen]:.2f})")
        print(f"  tool row, most-attended name per layer: {'  '.join(line)}")
        print(f"  top: {[(t['token'], t['p']) for t in tool_decision['top'][:3]]}")
        sims = []
        for layer, s in case["similarity"].items():
            d = {k[5:]: v for k, v in s.items() if k.startswith("desc:")}
            n = {k[5:]: v for k, v in s.items() if k.startswith("name:")}
            sim_hits.setdefault((layer, "desc"), []).append(rank(d, chosen) == 1)
            sim_hits.setdefault((layer, "name"), []).append(rank(n, chosen) == 1)
            sims.append(f"{layer}:{max(d, key=d.get)}")
        print(f"  request nearest description: {'  '.join(sims)}")
        for d in case["decisions"]:
            if d["what"].startswith("arg "):
                l23 = d["attention"].get("L23", {})
                req = l23.get("request", {}).get("max", 0)
                print(f"  {d['what'][:40]:40} said {d['said']!r:12} request attention L23 max {req:.2f}"
                      f"  L15 {d['attention'].get('L15', {}).get('request', {}).get('max', 0):.2f}")
    print("\nchosen tool ranked first, share of cases:")
    for (layer, kind), hits in sorted(att_hits.items()):
        print(f"  attention {layer:4} {kind:6}: {sum(hits)}/{len(hits)}")
    for (layer, kind), hits in sorted(sim_hits.items()):
        print(f"  similarity {layer:9} request~{kind}: {sum(hits)}/{len(hits)}")


if __name__ == "__main__":
    main()
