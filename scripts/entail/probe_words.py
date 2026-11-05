"""Asks the model what a word means, by what it writes when asked to do it.

  python probe_words.py cases <raw.entail.txt> <associations.json> <cases.jsonl> <template>
  python probe_words.py read <report.json> [<report.json> ...] <out.json>

The neurons say which tokens look alike. They do not say which operation a word
asks for: `highest` sits by `shortest`. The model's own reply does. Each word goes
into a request, and the Python operation in the reply is what the word asks for.

Replies are not equal evidence. The prompt's example indexes a list, so for a word it
does not understand the model copies that shape: `numbers[0] + 1` came back for 600
words. A call such as `max(numbers)` is a decision, and one template is enough if no
other names a different call. A bare index is weak: it must be the whole reply, and
every template must give it.
"""
import json
import re
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from python_requests import PREFIX  # noqa: E402

SIGNS = [
    ("max", r"\bmax\("), ("min", r"\bmin\("), ("sum", r"\bsum\("), ("count", r"\blen\("),
    ("sort", r"\bsorted\(|\.sort\("), ("unique", r"\bset\(|fromkeys"), ("reverse", r"reversed\(|\[::-1\]|\.reverse\("),
    ("upper", r"\.upper\("), ("lower", r"\.lower\("),
]
WEAK = [("first", r"^numbers\[0\]$"), ("last", r"^numbers\[-1\]$")]


def meaning(reply):
    line = reply.strip().split("\n")[0]
    found = [concept for concept, sign in SIGNS if re.search(sign, line)]
    if found:
        return (found[0], True) if len(found) == 1 else None
    return next(((concept, False) for concept, sign in WEAK if re.search(sign, line.strip())), None)


def settle(replies):
    """The concept these replies agree on, or None."""
    calls = {r[0] for r in replies if r and r[1]}
    if calls:
        return calls.pop() if len(calls) == 1 else None
    return replies[0][0] if all(replies) and len({r[0] for r in replies}) == 1 else None


def cases(program, associations, out, template):
    words = set()
    with open(program, encoding="utf8") as lines:
        for line in lines:
            if line.startswith('token " '[:7]):
                m = re.match(r'^token " ([a-z]{3,})" means', line)
                if m:
                    words.add(m[1])
    with open(out, "w", encoding="utf8", newline="\n") as target:
        for word in sorted(words):
            request = template.replace("WORD", word)
            target.write(json.dumps({"category": word, "prompt": f"{PREFIX}# {request}\nresult =",
                                     "answer": "", "max_tokens": 8}) + "\n")
    print(len(words), "words")


def read(paths, out):
    votes = {}
    for path in paths:
        for case in json.load(open(path, encoding="utf8"))["cases"]:
            votes.setdefault(case["category"], []).append(meaning(case["reply"]))
    settled = {w: settle(v) for w, v in votes.items() if len(v) == len(paths)}
    agreed = {w: c for w, c in settled.items() if c}
    by_concept = {}
    for word, concept in sorted(agreed.items()):
        by_concept.setdefault(concept, []).append(word)
    for concept, words in by_concept.items():
        print(f"{concept:9}{len(words):4}  {' '.join(words)[:170]}")
    print(f"{len(agreed)} of {len(votes)} words agreed")
    json.dump(by_concept, open(out, "w", encoding="utf8"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "cases":
        cases(*sys.argv[2:6])
    else:
        read(sys.argv[2:-1], sys.argv[-1])
