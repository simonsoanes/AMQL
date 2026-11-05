"""Test cases for dates, bases, logic and json, and their scoring.

  python formal_cases.py dates <out.jsonl> [seed]
  python formal_cases.py bases <out.jsonl> [seed]
  python formal_cases.py logic <out.jsonl> [seed]
  python formal_cases.py json  <out.jsonl> [seed]
  python formal_cases.py score <cases.jsonl> <report.json> [--base <report.json>]

Answers come from Python's own libraries or a separate evaluator, not from the programs.
Each prompt opens with a short drill, so an unmodified model answers rather than chats.
A reply is right if it opens with the answer and no letter or digit follows it.
"""
import datetime
import itertools
import json
import random
import re
import sys

DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
MONTHS = "January February March April May June July August September October November December".split()


def write(path, cases):
    with open(path, "w", encoding="utf8", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(len(cases), "cases")


# ─── dates ──────────────────────────────────────────────────────────────────

def spell(d, style):
    if style == 0:
        return d.isoformat()
    if style == 1:
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def any_date(rng):
    """Mostly recent dates, with the edges of months and centuries well represented."""
    year = rng.choice([rng.randint(1900, 2100), rng.randint(1583, 3000), rng.choice([1600, 1700, 1800, 1900, 2000, 2100, 2400])])
    month = rng.randint(1, 12)
    last = (datetime.date(year + month // 12, month % 12 + 1, 1) - datetime.timedelta(days=1)).day
    return datetime.date(year, month, rng.choice([rng.randint(1, last), 1, last]))


def dates(out, seed):
    rng = random.Random(seed)
    cases = []

    def add(category, drill, text, answer, tokens=14):
        cases.append(dict(category=category, prompt=f"{drill}\n{text}", answer=answer, max_tokens=tokens))

    d_week = "2000-01-01 was a Saturday\nJuly 4, 1776 was a Thursday"
    d_gap = "Days from 2000-01-01 to 2000-01-31 = 30 days"
    d_add = "2000-01-01 + 10 days = 2000-01-11"
    d_leap = "Is 2001 a leap year? No\nIs 2004 a leap year? Yes"
    d_has = "January 2001 has 31 days\nApril 2001 has 30 days"
    for i in range(90):
        d = any_date(rng)
        verb = [" was a", " is a", " falls on a", " fell on a", " was on a"][i % 5]
        add("weekday", d_week, spell(d, i % 3) + verb, DAYS[d.weekday()], 4)
    for i in range(60):
        a = any_date(rng)
        b = a + datetime.timedelta(days=rng.choice([rng.randint(0, 400), rng.randint(0, 40000), rng.randint(0, 100000)]))
        if b.year > 9999 or (b.year - a.year) > 299:
            continue
        if i % 4 == 3:
            a, b = b, a
        word = ["Days from {} to {} =", "Days between {} and {} =", "How many days from {} to {} is"][i % 3]
        n = abs((b - a).days)
        add("difference", d_gap, word.format(spell(a, i % 3), spell(b, (i // 3) % 3)), f"{n} day" + ("" if n == 1 else "s"))
    for i in range(60):
        d = any_date(rng)
        n = rng.choice([rng.randint(1, 60), rng.randint(1, 999), rng.randint(1, 9999)])
        sign = rng.choice([1, -1])
        r = d + datetime.timedelta(days=sign * n)
        if r.year < 1583:
            continue
        word = "s" if n != 1 else ""
        if i % 2:
            text = f"{n} day{word} {'after' if sign > 0 else 'before'} {spell(d, i % 3)} is"
        else:
            text = f"{spell(d, 0)} {'+' if sign > 0 else '-'} {n} day{word} ="
        add("add-days", d_add, text, r.isoformat())
    for y in [1600, 1700, 1800, 1900, 2000, 2100, 2400, 1996, 2023, 2024, 2025, 2100, 1752, 1984, 2028, 3000]:
        leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
        add("leap", d_leap, f"Is {y} a leap year?", "Yes" if leap else "No", 3)
    for i in range(24):
        y, m = rng.choice([1900, 2000, 2023, 2024, rng.randint(1600, 2400)]), i % 12 + 1
        last = (datetime.date(y + m // 12, m % 12 + 1, 1) - datetime.timedelta(days=1)).day
        add("month-length", d_has, f"{MONTHS[m - 1]} {y} has", f"{last} days", 6)
    for text in ["2023-02-29 was a", "2024-13-01 was a", "2024-04-31 was a", "1500-01-01 was a", "31 June 2024 was a", "The cat was a"]:
        add("silent", d_week, text, "", 4)
    write(out, cases)


# ─── bases ──────────────────────────────────────────────────────────────────

DIGITS = "0123456789abcdef"


def in_base(n, b):
    text = ""
    while n:
        text = DIGITS[n % b] + text
        n //= b
    return text or "0"


def any_value(rng):
    bits = rng.choice([4, 8, 8, 12, 16, 16, 20, 24, 32, 32])
    return rng.choice([rng.getrandbits(bits), 2 ** bits - 1, rng.getrandbits(bits)])


def bases(out, seed):
    rng = random.Random(seed)
    drill = "12 in binary is 1100\n200 in hex is c8\n0x1f in decimal is 31\n0b101 = 5\n10 in base 3 is 101"
    names = {2: "binary", 8: "octal", 10: "decimal", 16: "hex"}
    cases = []

    def add(category, text, answer):
        cases.append(dict(category=category, prompt=f"{drill}\n{text}", answer=answer, max_tokens=38))

    for _ in range(40):
        v = any_value(rng)
        add("to-binary", f"{v} in binary is", in_base(v, 2))
    for _ in range(30):
        v = any_value(rng)
        add("to-hex", f"{v} in {rng.choice(['hex', 'hexadecimal'])} is", in_base(v, 16))
    for _ in range(20):
        v = any_value(rng)
        add("to-octal", f"{v} in octal is", in_base(v, 8))
    for _ in range(40):
        v = any_value(rng)
        text = in_base(v, 16)
        text = rng.choice([text, text.upper()])
        add("from-hex", f"0x{text} {rng.choice(['in decimal is', '='])}", str(v))
    for _ in range(40):
        v = any_value(rng)
        add("from-binary", f"0b{in_base(v, 2)} {rng.choice(['in decimal is', '='])}", str(v))
    for _ in range(20):
        v = any_value(rng)
        add("from-octal", f"0o{in_base(v, 8)} in decimal is", str(v))
    for _ in range(20):
        v = rng.getrandbits(rng.choice([8, 16, 24]))
        add("binary-word", f"{in_base(v, 2)} binary in decimal is", str(v))
    for _ in range(30):
        a, b = rng.choice([2, 8, 16]), rng.choice([2, 8, 16])
        v = any_value(rng)
        prefix = {2: "0b", 8: "0o", 16: "0x"}[a]
        add("between", f"{prefix}{in_base(v, a)} in {names[b]} is", in_base(v, b))
    for _ in range(45):
        v, b = any_value(rng), rng.choice([3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15])
        add("to-base-n", f"{v} in base {b} is", in_base(v, b))
    for _ in range(30):
        v, b = any_value(rng), rng.choice([3, 4, 5, 6, 7, 9])
        add("from-base-n", f"{in_base(v, b)} base {b} in decimal is", str(v))
    for text in ["4294967296 in binary is", "0b102 =", "0o9 in decimal is", "12 base 2 in decimal is", "0xfffffffff =", "5.5 in binary is", "hello in binary is"]:
        add("silent", text, "")
    write(out, cases)


# ─── logic ──────────────────────────────────────────────────────────────────

STYLES = [
    dict(no="not ", an=" and ", orr=" or ", xor=" xor ", imp=" -> ", iff=" <-> ", t="true", f="false"),
    dict(no="NOT ", an=" AND ", orr=" OR ", xor=" XOR ", imp=" implies ", iff=" iff ", t="True", f="False"),
    dict(no="!", an=" && ", orr=" || ", xor=" ^ ", imp=" => ", iff=" <=> ", t="T", f="F"),
    dict(no="~", an=" & ", orr=" | ", xor=" ^ ", imp=" -> ", iff=" <-> ", t="T", f="F"),
]
LEVEL = dict(atom=4, no=4, an=3, orr=2, xor=2, imp=1, iff=1)


def tree(rng, depth, names, brackets):
    """A random expression; brackets counts how many may still open."""
    if depth == 0 or rng.random() < 0.2:
        leaf = ("atom", rng.choice(names))
        return ("no", leaf) if rng.random() < 0.3 else leaf
    op = rng.choice(["an", "an", "orr", "orr", "xor", "imp", "iff"])
    if rng.random() < 0.12:
        return ("no", tree(rng, depth, names, brackets))
    return (op, tree(rng, depth - 1, names, brackets), tree(rng, depth - 1, names, brackets))


def run(e, env):
    k = e[0]
    if k == "atom":
        return env[e[1]]
    if k == "no":
        return not run(e[1], env)
    a, b = run(e[1], env), run(e[2], env)
    return dict(an=a and b, orr=a or b, xor=a != b, imp=(not a) or b, iff=a == b)[k]


def show(e, s, parent=0, right=False):
    """Text and bracket depth. Brackets go wherever precedence or a rule asks for them."""
    k = e[0]
    if k == "atom":
        return {"T": s["t"], "F": s["f"]}.get(e[1], e[1]), 0
    if k == "no":
        text, d = show(e[1], s, 4)
        return s["no"] + text, d
    a, da = show(e[1], s, LEVEL[k], False)
    b, db = show(e[2], s, LEVEL[k], True)
    text, depth = a + s[k] + b, max(da, db)
    same = LEVEL[k] == parent and (right or k in ("imp", "iff", "xor") or parent == 2)
    if LEVEL[k] < parent or same:
        return "(" + text + ")", depth + 1
    return text, depth


def atoms(e):
    return {e[1]} if e[0] == "atom" else set().union(*[atoms(x) for x in e[1:]])


def verdict(e):
    values = [run(e, dict(p=p, q=q, r=r, T=True, F=False)) for p, q, r in itertools.product([False, True], repeat=3)]
    return values


def logic(out, seed):
    rng = random.Random(seed)
    drill = "true and false is false@not false is true@p or not p is a tautology@p and not p is a contradiction@p and q is contingent@Truth table of p and q: 0001".replace("@", "\n")
    cases = []

    def add(category, text, answer):
        cases.append(dict(category=category, prompt=f"{drill}\n{text}", answer=answer, max_tokens=8))

    def expression(names, depth):
        while True:
            e = tree(rng, depth, names, 2)
            s = rng.choice(STYLES)
            text, d = show(e, s)
            if d <= 2 and e[0] != "atom" and 6 <= len(text) <= 60:
                return e, text

    for i in range(80):
        e, text = expression(["T", "F"], rng.choice([2, 3, 3]))
        add("constants", text + rng.choice([" is", " ="]), "true" if verdict(e)[0] else "false")
    for i in range(90):
        e, text = expression(["p", "q", "r", "p", "q"], rng.choice([2, 3]))
        v = verdict(e)
        answer = "a tautology" if all(v) else "a contradiction" if not any(v) else "contingent"
        add("verdict", text + " is", answer)
    laws = ["p or not p", "not (p and not p)", "(p -> q) <-> (not q -> not p)", "not (p and q) <-> (not p or not q)",
            "not (p or q) <-> (not p and not q)", "((p -> q) and p) -> q", "((p -> q) and (q -> r)) -> (p -> r)",
            "(p and (q or r)) <-> ((p and q) or (p and r))", "(p or (q and r)) <-> ((p or q) and (p or r))", "(p -> q) <-> (not p or q)",
            "(p xor q) <-> ((p or q) and not (p and q))", "p -> (q -> p)"]
    fallacies = ["((p -> q) and q) -> p", "((p -> q) and not p) -> not q", "(p -> q) <-> (q -> p)", "(p or q) -> (p and q)", "not (p and q) <-> (not p and not q)", "p -> (p and q)"]
    for text in laws:
        add("laws", text + " is", "a tautology")
    for text in fallacies:
        add("fallacies", text + " is", "contingent")
    for i in range(40):
        e, text = expression(rng.choice([["p", "q"], ["p", "q", "r"]]), rng.choice([1, 2, 3]))
        if "p" not in atoms(e):
            continue
        used = atoms(e)
        if "r" in used:
            rows = [run(e, dict(p=p, q=q, r=r, T=True, F=False)) for p, q, r in itertools.product([False, True], repeat=3)]
        elif "q" in used:
            rows = [run(e, dict(p=p, q=q, r=False, T=True, F=False)) for p, q in itertools.product([False, True], repeat=2)]
        else:
            rows = [run(e, dict(p=p, q=False, r=False, T=True, F=False)) for p in [False, True]]
        add("table", f"Truth table of {text}:", "".join("1" if x else "0" for x in rows))
    for text in ["and or true is", "true and is", "(true or false is", "true or false) is", "p xor q or r is", "p -> q -> r is", "((((p)))) and q is", "p q is", "The sky is"]:
        add("silent", text, "")
    write(out, cases)


# ─── json ───────────────────────────────────────────────────────────────────

WORDS = "name city days user id tags items value active price note query limit path mode kind".split()


def refuse(text):
    raise ValueError(text)


def valid_json(text):
    """Python's own parser is the judge, without its NaN extension."""
    try:
        json.loads(text, parse_constant=refuse)
        return True
    except ValueError:
        return False


def any_value_json(rng, depth):
    pick = rng.random()
    if depth > 0 and pick < 0.3:
        return {rng.choice(WORDS) + str(i): any_value_json(rng, depth - 1) for i in range(rng.randint(0, 4))}
    if depth > 0 and pick < 0.55:
        return [any_value_json(rng, depth - 1) for _ in range(rng.randint(0, 4))]
    return rng.choice([
        rng.randint(-1000, 100000), round(rng.uniform(-100, 100), 3), rng.choice([True, False, None]), 0,
        rng.choice(WORDS), "Paris, France", 'say "hi"', "a{b}c", "[x]", "line\nbreak", "tab\there", "back\\slash", "",
        float(f"{rng.randint(1, 9)}e{rng.randint(2, 9)}"),
    ])


def render(rng, value):
    style = rng.choice([dict(), dict(indent=2), dict(separators=(",", ":"))])
    return json.dumps(value, **style)


def spoil_json(rng, text):
    """One typical slip. Whether it broke the text is for the parser to say."""
    marks = [i for i, c in enumerate(text) if c in '{}[],:"']
    i = rng.choice(marks) if marks else 0
    kind = rng.randrange(10)
    if kind == 0:      # a mark goes missing
        return "drop", text[:i] + text[i + 1:]
    if kind == 1:      # a closer is the wrong one
        closers = [j for j, c in enumerate(text) if c in "}]"]
        if closers:
            j = rng.choice(closers)
            return "swap", text[:j] + ("]" if text[j] == "}" else "}") + text[j + 1:]
    if kind == 2:      # trailing comma
        closers = [j for j, c in enumerate(text) if c in "}]"]
        if closers:
            j = rng.choice(closers)
            return "comma", text[:j] + "," + text[j:]
    if kind == 3:      # cut short
        return "cut", text[:rng.randint(1, max(1, len(text) - 1))]
    if kind == 4:      # one closer too many
        return "extra", text + rng.choice("}]")
    if kind == 5:      # single quotes
        return "quotes", text.replace('"', "'", 2)
    if kind == 6:      # a bad word or number
        return "word", text[:i] + rng.choice([" True", " None", " 01", " 1.", " .5", " -", " 1e", " 1.2.3", " nul"]) + text[i:]
    if kind == 7:      # an extra comma or colon
        return "double", text[:i] + rng.choice([",", ":", ",,"]) + text[i:]
    if kind == 8:      # a real line end in a string
        strings = [j for j, c in enumerate(text) if c == '"']
        if strings:
            j = rng.choice(strings)
            return "newline", text[:j + 1] + "\n" + text[j + 1:]
    return "escape", text[:i] + '"bad\\q"' + text[i:]


def json_cases(out, seed):
    rng = random.Random(seed)
    drill = 'JSON: {"a": 1}\nValid: yes\nJSON: {"a": 1,}\nValid: no\nJSON: [1, 2}\nValid: no\nJSON: ["x", {"b": null}]\nValid: yes'
    cases = []

    def add(category, text):
        cases.append(dict(category=category, prompt=f"{drill}\nJSON: {text}\nValid:", answer="yes" if valid_json(text) else "no", max_tokens=3))

    for _ in range(150):
        value = any_value_json(rng, rng.choice([1, 2, 3, 4]))
        add("valid", render(rng, value if isinstance(value, (dict, list)) else [value]))
    for _ in range(250):
        value = any_value_json(rng, rng.choice([2, 3, 4]))
        text = render(rng, value if isinstance(value, (dict, list)) else [value])
        kind, bad = spoil_json(rng, text)
        if len(bad) > 3:
            add("spoilt-" + kind if not valid_json(bad) else "spoilt-still-valid", bad)
    cases = [c for c in cases if len(c["prompt"]) < 900]
    cases.append(dict(category="silent", prompt="There is no text here.\nValid:", answer="", max_tokens=3))
    write(out, cases)


def guard_cases(out, seed):
    """Prefixes for the host to finish. Only the guard can see every bracket still open."""
    rng = random.Random(seed)
    cases = []
    for i in range(60):
        depth = rng.randint(2, 7)
        text, stack = "", []
        for _ in range(depth):
            if rng.random() < 0.5:
                text += '{"' + rng.choice(WORDS) + '": '
                stack.append("}")
            else:
                text += "[" + rng.choice(['1, ', '"x", ', 'true, ', '{"id": 2}, ', ""])
                stack.append("]")
        tail = rng.choice(['"a', '1', 'true', '"say \\"hi', '[1, 2', '{"k": "v"'])
        cases.append(dict(category="finish", prompt=f"Here is the data.\nJSON: {text}{tail}", prefix=text + tail, answer="", max_tokens=48))
    write(out, cases)


def string_end(text, at):
    """Where the string opening at `at` ends: None if it breaks, -1 if the text stops first."""
    i = at + 1
    while i < len(text):
        c = text[i]
        if c == '"':
            return i + 1
        if c == "\\":
            if i + 1 >= len(text):
                return -1
            e = text[i + 1]
            if e == "u":
                digits = text[i + 2:i + 6]
                if any(x not in "0123456789abcdefABCDEF" for x in digits):
                    return None
                if len(digits) < 4:
                    return -1
                i += 6
                continue
            if e not in '"\\/bfnrt':
                return None
            i += 2
            continue
        if ord(c) < 32:
            return None
        i += 1
    return -1


def json_state(text):
    """'bad' if text cannot start a JSON value, 'closed' if it holds a whole one, or 'open'.
    What follows a whole value is not its business."""
    stack, want, i, n = [], "value", 0, len(text)
    while True:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if want == "after" and not stack:
            return "closed"
        if i >= n:
            return "open"
        c = text[i]
        if want in ("value", "first"):
            if want == "first" and c == "]":
                stack.pop()
                i, want = i + 1, "after"
            elif c == "{":
                stack.append("}")
                i, want = i + 1, "firstkey"
            elif c == "[":
                stack.append("]")
                i, want = i + 1, "first"
            elif c == '"':
                end = string_end(text, i)
                if end is None:
                    return "bad"
                if end < 0:
                    return "open"
                i, want = end, "after"
            elif c in "-0123456789":
                j = i
                while j < n and text[j] in "+-0123456789.eE":
                    j += 1
                if j == n:
                    return "open"
                try:
                    json.loads(text[i:j])
                except ValueError:
                    return "bad"
                i, want = j, "after"
            else:
                for word in ("true", "false", "null"):
                    if text.startswith(word, i):
                        i, want = i + len(word), "after"
                        break
                    if word.startswith(text[i:]):
                        return "open"
                else:
                    return "bad"
        elif want in ("firstkey", "key"):
            if c == "}" and want == "firstkey":
                stack.pop()
                i, want = i + 1, "after"
            elif c == '"':
                end = string_end(text, i)
                if end is None:
                    return "bad"
                if end < 0:
                    return "open"
                i, want = end, "colon"
            else:
                return "bad"
        elif want == "colon":
            if c != ":":
                return "bad"
            i, want = i + 1, "value"
        elif c == ",":
            i, want = i + 1, "key" if stack[-1] == "}" else "value"
        elif c == stack[-1]:
            stack.pop()
            i += 1
        else:
            return "bad"


def guard_score(cases_path, report_path, base_path):
    """A mask keeps a text valid; it cannot make the host stop. So the score is how many
    replies never broke the JSON, and how many also reached its end."""
    cases = [json.loads(l) for l in open(cases_path, encoding="utf8") if l.strip()]
    for name, path in (("graft", report_path), ("base", base_path)):
        if not path:
            continue
        replies = [c["reply"] for c in json.load(open(path, encoding="utf8"))["cases"]]
        states = [json_state(c.get("prefix", "") + r) for c, r in zip(cases, replies)]
        blank = sum("\n\n\n" in r for r in replies)
        print(f"{name:8} still valid: {len(states) - states.count('bad')} of {len(cases)}, "
              f"of which closed: {states.count('closed')}, stuck on blank lines: {blank}")


# ─── scoring ────────────────────────────────────────────────────────────────

def right(answer, reply):
    text = reply.strip()
    if not answer:
        return False
    if not text.lower().startswith(answer.lower()):
        return False
    rest = text[len(answer):]
    return not rest or not rest[0].isalnum()


def line(reply):
    return reply.strip().split("\n")[0].strip()


def score(cases_path, report_path, base_path):
    cases = [json.loads(l) for l in open(cases_path, encoding="utf8") if l.strip()]
    replies = [c["reply"] for c in json.load(open(report_path, encoding="utf8"))["cases"]]
    base = [c["reply"] for c in json.load(open(base_path, encoding="utf8"))["cases"]] if base_path else None
    rows, wrong = {}, []
    for i, c in enumerate(cases):
        r = rows.setdefault(c["category"], dict(n=0, ok=0, base=0, broke=0, silent=0))
        r["n"] += 1
        good = right(c["answer"], replies[i])
        r["ok"] += good
        if c["category"] == "silent":
            r["silent"] += line(replies[i]) == "" or (base is not None and line(replies[i]) == line(base[i]))
        elif not good:
            wrong.append((c["prompt"].split("\n")[-1][:70], c["answer"], line(replies[i])[:40]))
        if base:
            was = right(c["answer"], base[i])
            r["base"] += was
            r["broke"] += was and not good
    head = f"{'category':22}{'n':>5}{'right':>7}" + (f"{'base':>7}{'broke':>7}" if base else "") + f"{'silent':>8}"
    print(head)
    for name, r in sorted(rows.items()):
        print(f"{name:22}{r['n']:5}{r['ok']:7}" + (f"{r['base']:7}{r['broke']:7}" if base else "") + f"{r['silent']:8}")
    for w in wrong[:25]:
        print("  ", w)


def main():
    a = sys.argv
    if a[1] in ("score", "guard_score"):
        globals()[a[1]](a[2], a[3], a[a.index("--base") + 1] if "--base" in a else None)
    else:
        globals()[a[1] if a[1] != "json" else "json_cases"](a[2], int(a[3]) if len(a) > 3 else 1)


if __name__ == "__main__":
    main()
