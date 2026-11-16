"""Writes the scan programs and test texts for unbounded-context Entail.

  scan.entail.txt        query first: "?k" then records "a7;k3;..." then "="
                         state: the query and the best value so far. O(1).
  scan-index.entail.txt  query last: records first, "?k=" at the end
                         state: the best value under EVERY key, an index. O(keys).

Both read one character per token, so `entail run --file` streams a text of any
size through them in fixed memory. The texts' answers are computed here.

    python scripts/entail/make_scan.py <out-dir>
"""
import random
import sys
from pathlib import Path

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def tokens():
    lines = ["state is_q, is_end, is_digit, val, is_key, key[26]",
             'token "?" means is_q = 1',
             'token "=" means is_end = 1']
    lines += [f'token "{d}" means is_digit = 1, val = {d}' for d in range(10)]
    lines += [f'token "{c}" means is_key = 1, key[{i}] = 1' for i, c in enumerate(LETTERS)]
    return "\n".join(lines)


OUTPUT = """
# The answer: the best value, then the end.
state o[10], e_prev
carry e_prev from is_end
for d in 0 to 9
    entail o[d] when is_end, answer == d
    emit "{d}" when o[d]
end
emit end when e_prev
"""

SCAN = f"""program scan
# Query first, then any number of records, then "=":   ?k a7;k3;b9;k5; ... =
# Answers the largest value filed under the query's key.
#
# Nothing here looks back more than one token. Every fact that must survive the
# text is a carried register, so the state is a few dozen lanes whether the text
# is 64 bytes or 2 MB. The question comes first, so only one key's best needs
# keeping: a fold, not an index.

{tokens()}

# The query: set by the letter after "?", then held.
state q_prev
carry q_prev from is_q
state query[26], last_query[26]
for c in 0 to 25
    carry last_query[c] from query[c]
    entail query[c] = last_query[c]
    entail query[c] when q_prev, key[c]
end

# A record's key matches the query; its value is the next token.
state match, m_prev
for c in 0 to 25
    entail match when key[c], last_query[c]
end
carry m_prev from match

# The best value so far: kept, or raised by a matching record.
state best, last_best
carry last_best from best
entail best = last_best
entail best = val - last_best when m_prev, is_digit, val - last_best >= 1
let answer = best
{OUTPUT}"""

INDEX = f"""program scan_index
# Records first, query last:   a7;k3;b9;k5; ... ?k=
# The same answer, but the question arrives after the data, so a fold must keep
# the best value under every key it might be asked about: an index with one
# register per key. Still fixed memory here, because there are only 26 keys;
# with open-ended questions it is the whole text, which is what attention keeps.

{tokens()}

state k_prev[26], q_prev
carry q_prev from is_q
for c in 0 to 25
    carry k_prev[c] from key[c]
end

state best[26], last_best[26]
for c in 0 to 25
    carry last_best[c] from best[c]
    entail best[c] = last_best[c]
    entail best[c] = val - last_best[c] when k_prev[c], is_digit, val - last_best[c] >= 1
end

# "?k": the asked key, held until "=".
state asked[26], last_asked[26]
for c in 0 to 25
    carry last_asked[c] from asked[c]
    entail asked[c] = last_asked[c]
    entail asked[c] when q_prev, key[c]
end

state pick[26]
for c in 0 to 25
    entail pick[c] = last_best[c] when last_asked[c]
end
let answer = (sum c in 0 to 25 : pick[c])
{OUTPUT}"""


def records(rng, n_bytes):
    out = []
    size = 0
    while size < n_bytes:
        r = f"{rng.choice(LETTERS)}{rng.randrange(10)};"
        out.append(r)
        size += len(r)
    return out


def best(recs, key):
    vals = [int(r[1]) for r in recs if r[0] == key]
    return max(vals) if vals else 0


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "scan.entail.txt").write_text(SCAN, encoding="utf-8", newline="\n")
    (out / "scan-index.entail.txt").write_text(INDEX, encoding="utf-8", newline="\n")
    rng = random.Random(7)
    for name, n in (("64b", 64), ("4kb", 4096), ("2mb", 2 * 1024 * 1024), ("8mb", 8 * 1024 * 1024)):
        # Values under the asked key stay low except one, placed at random, so
        # the answer depends on reading the whole text, not its start or end.
        recs = records(rng, n)
        key = rng.choice(LETTERS)
        recs = [f"{key}{min(int(r[1]), 6)};" if r[0] == key else r for r in recs]
        special = rng.randrange(7, 10)
        recs[rng.randrange(len(recs))] = f"{key}{special};"
        body = "".join(recs)
        (out / f"first-{name}.txt").write_text(f" ?{key}{body}=", encoding="utf-8", newline="\n")
        (out / f"last-{name}.txt").write_text(f" {body}?{key}=", encoding="utf-8", newline="\n")
        print(f"{name}: {len(body):,} bytes, key {key}, expected {best(recs, key)}")


if __name__ == "__main__":
    main()
