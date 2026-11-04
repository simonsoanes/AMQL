"""Evaluation cases for `amql-cli entail eval`, one JSON object per line.

  python make_cases.py empty  [per-category] [seed]   # the compiled empty model
  python make_cases.py direct [per-category] [seed]   # "Calculate: a op b=" on a host
  python make_cases.py chat   [per-category] [seed]   # questions and word problems, in chat

Operands are drawn at random, so a different seed is a fresh test. A case
passes when the first number of the reply (or, for chat, the last) is the
answer. Quotients are truncated to two places, as mathematics.entail.txt says.
"""
import json
import random
import sys


def number(digits):
    return random.randint(10 ** (digits - 1) if digits > 1 else 0, 10 ** digits - 1)


def quotient(a, b):
    whole, places = a // b, (a % b) * 100 // b
    text = str(whole)
    if places:
        text += "." + f"{places:02d}".rstrip("0")
    return text


def emit(**case):
    print(json.dumps(case))


def empty(n):
    def case(category, a, op, b, answer):
        emit(category=category, prompt=f" {a}{op}{b}=", answer=str(answer), first=True, max_tokens=16)

    for _ in range(n):
        for d in (2, 4, 8):
            a, b = number(d), number(d)
            case(f"add-{d}", a, "+", b, a + b)
        for d in (3, 6, 8):
            a, b = number(d), number(d)
            case(f"sub-{d}", a, "-", b, a - b)
        for d in (2, 3, 4, 6):
            a, b = number(d), number(d)
            case(f"mul-{d}x{d}", a, "*", b, a * b)
        b, q = number(2), number(3)
        case("div-exact", b * q, "/", b, q)
        a, b = random.randint(1, 10 ** random.randint(1, 6) - 1), random.randint(1, 10 ** random.randint(1, 4) - 1)
        case("div-any", a, "/", b, quotient(a, b))


def direct(n):
    def case(category, a, op, b, answer):
        # A short drill, so the unmodified model answers rather than chats:
        # the baseline has to be its real ability, not a formatting accident.
        spaced = random.random() < 0.5
        drill = ("3 + 4 = 7\n12 * 5 = 60\n90 - 8 = 82\n84 / 4 = 21\n" if spaced
                 else "3+4=7\n12*5=60\n90-8=82\n84/4=21\n")
        prompt = drill + (f"{a} {op} {b} =" if spaced else f"{a}{op}{b}=")
        emit(category=category, prompt=prompt, answer=str(answer), first=True, max_tokens=16)

    for _ in range(n):
        for d in (2, 3, 5, 8):
            a, b = number(d), number(d)
            case(f"add-{d}", a, "+", b, a + b)
        for d in (2, 3, 5, 8):
            a, b = number(d), number(d)
            case(f"sub-{d}", a, "-", b, a - b)
        for d in (2, 3, 4, 6):
            a, b = number(d), number(d)
            case(f"mul-{d}x{d}", a, "*", b, a * b)
        b, q = number(2), number(3)
        case("div-exact", b * q, "/", b, q)
        a, b = number(5), number(2)
        case("div-5by2", a, "/", b, quotient(a, b))


def chat(n):
    ask = (" Work it out step by step, writing each calculation with an equals sign,"
           " and finish with the final number.")
    line = " Answer in one line, as an equation."

    def case(category, prompt, answer, tokens):
        emit(category=category, prompt=prompt, answer=str(answer), chat=True, max_tokens=tokens)

    r = random.randint
    for i in range(n):
        a, b = r(100, 999), r(100, 999)
        case("chat-mul-3x3", f"What is {a} * {b}?" + line, a * b, 48)
        a, b = r(10000, 99999), r(10000, 99999)
        case("chat-add-5", f"What is {a} + {b}?" + line, a + b, 48)
        a, b = r(1000, 9999), r(1000, 9999)
        case("chat-sub-4", f"What is {a} - {b}?" + line, a - b, 48)
        # Word problems whose reasoning is easy and whose arithmetic is not.
        kind = i % 3
        if kind == 0:
            a, b, c = r(120, 890), r(13, 97), r(1000, 9000)
            case("words", f"A warehouse receives {a} crates holding {b} items each, and then ships out {c} "
                          f"items. How many items are left?" + ask, a * b - c, 320)
        elif kind == 1:
            a, b, c = r(23, 97), r(110, 480), r(3, 9)
            case("words", f"A school buys {a} boxes of pens at {b} dollars per box, and {c} printers at "
                          f"{b * 7} dollars each. How many dollars does it spend in total?" + ask,
                 a * b + c * b * 7, 320)
        else:
            a, b, c = r(1200, 8800), r(1100, 8900), r(14, 88)
            case("words", f"A library has {a} novels and {b} textbooks. It adds {c} more books to each of "
                          f"its 12 rooms. How many books does it have now?" + ask, a + b + c * 12, 320)


if __name__ == "__main__":
    suite = sys.argv[1] if len(sys.argv) > 1 else "direct"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 11)
    {"empty": empty, "direct": direct, "chat": chat}[suite](count)
