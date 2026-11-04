"""Measures an extracted Entail program against Python's own documentation.

  python python_coverage.py <program.entail.txt> <tokenizer-dir> [--missing out.txt]

The checklist comes from what CPython documents about itself, on this machine:
the keyword list, the builtin namespace, the methods of the core types, the
standard library's module names, the operators of the tokenizer, and the topics
of the language reference that `pydoc` ships. Nothing is downloaded.

An item is *recognised* if every token of it can fire a rule, *produced* if every
token of it can be voted for, and *missing* if neither. This says the name is
somewhere in the program, not that the program knows what it means: it is a
floor on coverage, not a measure of understanding.
"""
import builtins
import inspect
import json
import keyword
import re
import sys
import token as tokens
import types

from pydoc_data import topics


def checklist():
    groups = {
        "keywords": list(keyword.kwlist),
        "soft keywords": list(keyword.softkwlist),
        "operators": sorted(tokens.EXACT_TOKEN_TYPES),
    }
    names = [n for n in dir(builtins) if not n.startswith("__")]
    groups["exceptions"] = [n for n in names if inspect.isclass(getattr(builtins, n))
                            and issubclass(getattr(builtins, n), BaseException)]
    groups["builtin functions"] = [n for n in names if n not in groups["exceptions"]
                                   and n not in keyword.kwlist and n != "_"]
    for kind in (str, list, dict, set, tuple, bytes, int, float):
        groups[f"{kind.__name__} methods"] = [n for n in dir(kind) if not n.startswith("_")]
    groups["dunder protocol"] = sorted(n for n in dir(type("C", (), {})) if n.startswith("__")) + [
        "__init__", "__call__", "__iter__", "__next__", "__enter__", "__exit__", "__getitem__",
        "__setitem__", "__len__", "__contains__", "__add__", "__await__", "__aiter__"]
    groups["standard library"] = sorted(m for m in sys.stdlib_module_names if not m.startswith("_"))
    return groups


def topic_words():
    """Identifiers the language reference discusses, weighted by how often."""
    counts = {}
    for text in topics.topics.values():
        for word in re.findall(r"[A-Za-z_][A-Za-z_0-9]{2,}", text):
            counts[word] = counts.get(word, 0) + 1
    return [w for w, n in sorted(counts.items(), key=lambda x: -x[1]) if n >= 5]


def read_program(path):
    text = open(path, encoding="utf8").read()
    code = {}
    for match in re.finditer(r'^token "(.*)" means sym = (\d+)$', text, re.M):
        value = match.group(1).replace("\\n", "\n").replace("\\q", '"').replace("\\{", "{")
        code[int(match.group(2))] = value
    fired = {code[int(c)] for c in re.findall(r"sym == (\d+)", text) if int(c) in code}
    fired |= {code[int(c)] for c in re.findall(r"prev == (\d+)", text) if int(c) in code}
    voted = {m.replace("\\n", "\n").replace("\\q", '"').replace("\\{", "{")
             for m in re.findall(r'^vote "(.*)" by ', text, re.M)}
    return fired, voted


def main():
    from tokenizers import Tokenizer
    path, tokenizer_dir = sys.argv[1], sys.argv[2]
    tokenizer = Tokenizer.from_file(tokenizer_dir + "/tokenizer.json")
    fired, voted = read_program(path)

    def pieces(item):
        # The spellings an item takes in code: bare, after a space, after a dot.
        for spelling in (item, " " + item, "." + item):
            ids = tokenizer.encode(spelling, add_special_tokens=False).ids
            yield [tokenizer.decode([i]) for i in ids]

    def status(item):
        recognised = any(all(p in fired for p in parts) for parts in pieces(item))
        produced = any(all(p in voted for p in parts) for parts in pieces(item))
        return recognised, produced

    groups = checklist()
    groups["language reference words"] = topic_words()
    report, missing = {}, {}
    print(f"{'documented group':<28}{'items':>7}{'recognised':>12}{'produced':>10}{'either':>8}")
    total = [0, 0, 0, 0]
    for group, items in groups.items():
        items = sorted(set(items))
        results = {i: status(i) for i in items}
        r = sum(1 for v in results.values() if v[0])
        p = sum(1 for v in results.values() if v[1])
        e = sum(1 for v in results.values() if v[0] or v[1])
        print(f"{group:<28}{len(items):>7}{r:>12}{p:>10}{100 * e / len(items):>7.0f}%")
        for i, v in enumerate((len(items), r, p, e)):
            total[i] += v
        report[group] = {"items": len(items), "recognised": r, "produced": p, "either": e}
        missing[group] = [i for i, v in results.items() if not (v[0] or v[1])]
    print(f"{'all':<28}{total[0]:>7}{total[1]:>12}{total[2]:>10}{100 * total[3] / total[0]:>7.0f}%")
    if "--missing" in sys.argv:
        out = sys.argv[sys.argv.index("--missing") + 1]
        json.dump({"report": report, "missing": missing}, open(out, "w", encoding="utf8"), indent=1)


if __name__ == "__main__":
    main()
