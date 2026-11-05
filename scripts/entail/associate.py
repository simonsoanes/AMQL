"""Which words of a request go with which Python operation, from two sources.

  python associate.py <raw.entail.txt> <anchors.json> <out.json> [--top 12]

anchors.json maps a concept to the Python names that carry it:
  {"reverse": ["reversed"], "sum": ["sum"], ...}

*The extract:* two tokens are related if the same neurons fire on them. Each
token becomes a vector over neurons, weighted by how rare the neuron is, and a
word's score against a concept is its cosine with the anchors'.

*The documentation:* the words of the first lines of each anchor's docstring.

A word is proposed only for the concept it scores highest on, so it can never
mean two things. Nothing here sees the words a test will use.
"""
import builtins
import inspect
import json
import math
import re
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from fold_raw import unquote  # noqa: E402

STOP = set("the a an of to in is it and or if for with on by be as at that this from return "
           "not are all each any new its which into than when will can may none true false "
           "object objects value values given default".split())


def read_vectors(path):
    token = re.compile(r'^token "(.*)" means sym = (\d+)$')
    rule = re.compile(r'^entail (\S+) = (\d+) when sym == (\d+)')
    name, vectors = {}, defaultdict(dict)
    with open(path, encoding="utf8") as lines:
        for line in lines:
            if line.startswith("token "):
                m = token.match(line.rstrip("\n"))
                name[int(m[2])] = unquote(m[1])
            elif line.startswith("entail "):
                m = rule.match(line)
                if m:
                    vectors[name[int(m[3])]][m[1]] = float(m[2])
    return vectors


def doc_words(anchor):
    item = getattr(builtins, anchor, None)
    for kind in (str, list, dict, set):
        item = item or getattr(kind, anchor, None)
    text = (inspect.getdoc(item) or "") if item else ""
    return {w.lower() for w in re.findall(r"[A-Za-z]{3,}", text[:240])} - STOP


def main():
    program, anchors_path, out = sys.argv[1:4]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 12
    anchors = json.load(open(anchors_path, encoding="utf8"))
    vectors = read_vectors(program)

    frequency = defaultdict(int)
    for vector in vectors.values():
        for unit in vector:
            frequency[unit] += 1
    count = len(vectors)

    def weighted(vector):
        return {u: v * math.log(count / frequency[u]) for u, v in vector.items()}

    def cosine(a, b):
        dot = sum(v * b.get(u, 0) for u, v in a.items())
        return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())) or 1)

    words = {t: weighted(v) for t, v in vectors.items()
             if re.fullmatch(r" [a-z]{3,}", t) and t.strip() not in STOP}
    centres = {}
    for concept, names in anchors.items():
        centre = defaultdict(float)
        for n in names:
            for form in (" " + n, "." + n, n):
                for u, v in weighted(vectors.get(form, {})).items():
                    centre[u] += v
        centres[concept] = centre

    best = {}
    for word, vector in words.items():
        scores = {c: cosine(vector, centre) for c, centre in centres.items() if centre}
        if scores:
            concept = max(scores, key=scores.get)
            best[word] = (concept, scores[concept])

    proposals = {}
    for concept, names in anchors.items():
        documented = set().union(*[doc_words(n) for n in names]) if names else set()
        near = sorted(((s, w.strip()) for w, (c, s) in best.items() if c == concept), reverse=True)[:top]
        proposals[concept] = {"extract": [[w, round(s, 3)] for s, w in near], "docs": sorted(documented)}
        print(f"{concept:10} extract: {' '.join(w for _, w in near)}")
        print(f"{'':10} docs:    {' '.join(sorted(documented))[:150]}")
    json.dump(proposals, open(out, "w", encoding="utf8"), indent=1)


if __name__ == "__main__":
    main()
