"""Folds a raw extract into the one function it computes.

  python fold_raw.py <raw.entail.txt> <holdout.jsonl> <folded.json>

Every lane of a raw extract is set by the current token and the one before it,
and `entail` adds, so the whole program is a function of those two tokens. This
evaluates it once for each token, and for each pair that has a rule of its own,
and keeps the winner with the runner-up. The score on `holdout.jsonl` must equal
what `entail fidelity` says of the program: that is the proof the fold is exact.
"""
import json
import re
import sys
from collections import defaultdict

UNQUOTE = (("\\n", "\n"), ("\\q", '"'), ("\\{", "{"))


def unquote(text):
    for a, b in UNQUOTE:
        text = text.replace(a, b)
    return text


def read(path):
    code = {}
    by_token = defaultdict(list)    # sym -> [(lane, level)]
    by_pair = defaultdict(list)     # (prev, sym) -> [(lane, level)]
    votes = defaultdict(list)       # lane -> [(token, weight)]
    token = re.compile(r'^token "(.*)" means sym = (\d+)$')
    rule = re.compile(r'^entail (\S+) = (\d+) when (?:prev == (\d+), )?sym == (\d+)')
    vote = re.compile(r'^vote "(.*)" by ([\d.]+)\*(\S+)')
    with open(path, encoding="utf8") as lines:
        for line in lines:
            if line.startswith("token "):
                m = token.match(line.rstrip("\n"))
                code[unquote(m[1])] = int(m[2])
            elif line.startswith("entail "):
                m = rule.match(line)
                target = by_pair[(int(m[3]), int(m[4]))] if m[3] else by_token[int(m[4])]
                target.append((m[1], int(m[2])))
            elif line.startswith("vote "):
                m = vote.match(line)
                votes[m[3]].append((unquote(m[1]), float(m[2])))
    return code, by_token, by_pair, votes


def main():
    program, holdout, out = sys.argv[1:4]
    code, by_token, by_pair, votes = read(program)
    name = {v: k for k, v in code.items()}

    def scores(rules):
        total = defaultdict(float)
        lanes = defaultdict(int)
        for lane, level in rules:
            lanes[lane] += level
        for lane, value in lanes.items():
            for token, weight in votes.get(lane, ()):
                total[token] += weight * value
        return total

    def ranked(total):
        top = sorted(total.items(), key=lambda x: -x[1])[:2]
        if not top or top[0][1] <= 0:
            return None
        return top[0][0], top[0][1], top[1][1] if len(top) > 1 else 0.0

    single = {s: scores(r) for s, r in by_token.items()}
    table = {}
    for s, total in single.items():
        if (best := ranked(total)) is not None:
            table[name[s]] = best
    pairs = {}
    for (p, s), rules in by_pair.items():
        total = defaultdict(float, single.get(s, {}))
        for token, value in scores(rules).items():
            total[token] += value
        # A lane fed by both rules is one sum, and votes are linear, so these add.
        if (best := ranked(total)) is not None:
            pairs[f"{name[p]}\u0000{name[s]}"] = best

    right = total = spoke = 0
    with open(holdout, encoding="utf8") as lines:
        for line in lines:
            row = json.loads(line)
            for i, token in enumerate(row["tokens"]):
                total += 1
                before = row["tokens"][i - 1] if i else None
                said = pairs.get(f"{before}\u0000{token}") or table.get(token)
                if said:
                    spoke += 1
                    right += said[0] == row["next"][i]
    print(f"folded: {len(table)} tokens, {len(pairs)} pairs")
    print(f"agreement {right} of {total} ({100 * right / total:.1f}%), spoke at {spoke}")
    json.dump({"tokens": table, "pairs": pairs}, open(out, "w", encoding="utf8"))


if __name__ == "__main__":
    main()
