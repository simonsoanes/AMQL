"""Requests for the synthetic Python program, and how to score a reply by running it.

  python python_requests.py write <cases.jsonl> [--chat] [--fresh]
  python python_requests.py score <cases.jsonl> <report.json> [...] [--base <report.json>]

A reply is useful if the line it wrote computes the right thing, so each request
carries a reference expression and both run on the same inputs. `plain` requests
use Python's own words, `paraphrase` ones say the same thing in other words, and
`impossible` ones ask for something with no meaning, where the right reply is
not to claim an answer. This file was written before any lexicon existed.

`fresh` requests were written after the program was finished and changed nothing in
it. The filler words were chosen after seeing `paraphrase` fail, so `fresh` is the
honest measure of how far the vocabulary reaches.
"""
import json
import subprocess
import sys

# The model is told what exists and shown the form. Without this it cannot know
# `text` or `numbers` are there, and the comparison measures nothing.
PREFIX = ('text = "the cat sat"\nitems = ["a", "b"]\nnumbers = [1, 2, 3]\n\n'
          "# double the second of numbers\nresult = numbers[1] * 2\n\n")
CHAT = ("The Python variables text (a string), items (a list of strings) and numbers (a list of "
        "integers) exist. Reply with one Python expression and nothing else, or the single word "
        "NONE if the request makes no sense. Request: ")

TEXT = "the quick brown fox\njumps over the lazy dog"
INPUTS = {
    "text": TEXT,
    "items": ["pear", "fig", "banana", "fig", "kiwi"],
    "numbers": [3, -1, 4, 1, -5, 9, 2, 6, 4],
}

PLAIN = [
    ("reverse text", "text[::-1]"),
    ("reverse items", "items[::-1]"),
    ("reverse the words in text", "' '.join(text.split()[::-1])"),
    ("sort numbers", "sorted(numbers)"),
    ("sort items", "sorted(items)"),
    ("sort the words in text", "sorted(text.split())"),
    ("sort numbers in descending order", "sorted(numbers, reverse=True)"),
    ("count the words in text", "len(text.split())"),
    ("count the lines in text", "len(text.splitlines())"),
    ("count items", "len(items)"),
    ("sum numbers", "sum(numbers)"),
    ("sum the even numbers", "sum(n for n in numbers if n % 2 == 0)"),
    ("sum the odd numbers", "sum(n for n in numbers if n % 2 == 1)"),
    ("max of numbers", "max(numbers)"),
    ("min of numbers", "min(numbers)"),
    ("longest word in text", "max(text.split(), key=len)"),
    ("shortest word in text", "min(text.split(), key=len)"),
    ("unique items", "list(dict.fromkeys(items))"),
    ("first word in text", "text.split()[0]"),
    ("last line in text", "text.splitlines()[-1]"),
    ("first of numbers", "numbers[0]"),
    ("upper text", "text.upper()"),
    ("lower text", "text.lower()"),
    ("sort the unique numbers", "sorted(set(numbers))"),
    ("count the unique items", "len(set(items))"),
    ("count the even numbers", "len([n for n in numbers if n % 2 == 0])"),
    ("sort the positive numbers", "sorted(n for n in numbers if n > 0)"),
    ("max of the negative numbers", "max(n for n in numbers if n < 0)"),
]

PARAPHRASE = [
    ("flip text backwards", "text[::-1]"),
    ("put the words in text backwards", "' '.join(text.split()[::-1])"),
    ("arrange numbers in order", "sorted(numbers)"),
    ("order the items", "sorted(items)"),
    ("arrange numbers from highest down, descending", "sorted(numbers, reverse=True)"),
    ("how many words are in text", "len(text.split())"),
    ("number of lines in text", "len(text.splitlines())"),
    ("size of items", "len(items)"),
    ("length of items", "len(items)"),
    ("total of numbers", "sum(numbers)"),
    ("add up the even numbers", "sum(n for n in numbers if n % 2 == 0)"),
    ("the biggest of numbers", "max(numbers)"),
    ("the largest of numbers", "max(numbers)"),
    ("highest of numbers", "max(numbers)"),
    ("the smallest of numbers", "min(numbers)"),
    ("lowest of numbers", "min(numbers)"),
    ("the biggest word in text", "max(text.split(), key=len)"),
    ("drop duplicates from items", "list(dict.fromkeys(items))"),
    ("distinct items", "list(dict.fromkeys(items))"),
    ("make text uppercase", "text.upper()"),
    ("convert text to lowercase", "text.lower()"),
    ("final line of text", "text.splitlines()[-1]"),
    ("how many unique items are there", "len(set(items))"),
    ("total of the positive numbers", "sum(n for n in numbers if n > 0)"),
]

FRESH = [
    ("give the sum of the numbers", "sum(numbers)"),
    ("find the maximum of numbers", "max(numbers)"),
    ("find the minimum of numbers", "min(numbers)"),
    ("the numbers in ascending order", "sorted(numbers)"),
    ("the numbers sorted from largest to smallest", "sorted(numbers, reverse=True)"),
    ("the length of text in words", "len(text.split())"),
    ("return the last word of text", "text.split()[-1]"),
    ("the first line of text", "text.splitlines()[0]"),
    ("the last of numbers", "numbers[-1]"),
    ("the final item of items", "items[-1]"),
    ("text in capitals", "text.upper()"),
    ("text in upper case", "text.upper()"),
    ("text in lower case", "text.lower()"),
    ("remove repeated items", "list(dict.fromkeys(items))"),
    ("the items without duplicates", "list(dict.fromkeys(items))"),
    ("count how many numbers there are", "len(numbers)"),
    ("length of numbers", "len(numbers)"),
    ("the sum of the odd numbers", "sum(n for n in numbers if n % 2 == 1)"),
    ("the smallest positive number", "min(n for n in numbers if n > 0)"),
    ("the largest negative number", "max(n for n in numbers if n < 0)"),
    ("the shortest word in text", "min(text.split(), key=len)"),
    ("the longest line in text", "max(text.splitlines(), key=len)"),
    ("text reversed", "text[::-1]"),
    ("items in reverse order", "items[::-1]"),
    ("the words in text, sorted", "sorted(text.split())"),
    ("number of even numbers", "len([n for n in numbers if n % 2 == 0])"),
]

IMPOSSIBLE = [
    "sum the words in text",
    "sum text",
    "uppercase numbers",
    "sum the even words in text",
    "lowercase numbers",
    "sum items",
    "reverse",
    "the weather in paris",
]


def write(path, chat=False, fresh=False):
    sets = (("fresh", FRESH),) if fresh else (("plain", PLAIN), ("paraphrase", PARAPHRASE),
                                              ("impossible", [(r, None) for r in IMPOSSIBLE]))
    with open(path, "w", encoding="utf8", newline="\n") as out:
        for category, rows in sets:
            for request, reference in rows:
                prompt = CHAT + request if chat else f"{PREFIX}# {request}\nresult ="
                out.write(json.dumps({
                    "category": category, "prompt": prompt, "chat": chat, "request": request,
                    "answer": reference or "", "max_tokens": 40}) + "\n")


RUNNER = r"""
import json, sys
SAFE = {n: __builtins__.__dict__[n] for n in
        'len sum max min sorted reversed set list dict tuple str int float abs range enumerate zip map filter any all print'.split()}
row = json.load(sys.stdin)
env = {'__builtins__': SAFE, **row['inputs']}
try:
    value = eval(row['code'], env)
    if hasattr(value, '__next__'):
        value = list(value)
    print(json.dumps({'value': repr(value)}))
except BaseException as e:
    print(json.dumps({'error': type(e).__name__}))
"""


def run(code):
    """Runs an expression a small model wrote: its own process, no imports, a time limit."""
    try:
        done = subprocess.run([sys.executable, "-I", "-c", RUNNER], capture_output=True, text=True, timeout=5,
                              input=json.dumps({"code": code, "inputs": INPUTS}))
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (subprocess.TimeoutExpired, IndexError, ValueError):
        return {"error": "timeout"}


def first_line(reply):
    lines = [l.strip() for l in reply.split("\n") if l.strip() and not l.strip().startswith("```")]
    line = lines[0].strip("`") if lines else ""
    return line.split("=", 1)[1].strip() if line.startswith("result =") else line


def score(path, base=None):
    """Right, wrong or silent. With `base`, a first line equal to the base model's is
    the program having stayed silent: the host answered, and it is counted that way.
    Only the first line counts, because replies run on past it and the host can drift
    onto a later line that the program then answers."""
    report = json.load(open(path, encoding="utf8"))
    requests = [json.loads(l) for l in open(CASES, encoding="utf8")]
    before = [c["reply"] for c in json.load(open(base, encoding="utf8"))["cases"]] if base else None
    totals, misses = {}, []
    for index, case in enumerate(report["cases"]):
        code = first_line(case["reply"])
        got = run(code) if code and code != "NONE" else {"error": "silent"}
        category = case["category"]
        t = totals.setdefault(category, {"right": 0, "wrong": 0, "failed": 0, "count": 0, "spoke": 0})
        t["count"] += 1
        spoke = before is None or code != first_line(before[index])
        t["spoke"] += spoke
        if category == "impossible":
            ok = "value" not in got
            t["wrong" if not ok else "right"] += 1
        else:
            ok = "value" in got and got["value"] == run(case["answer"]).get("value")
            t["right" if ok else "wrong" if "value" in got else "failed"] += 1
        if not ok:
            mark = "program" if spoke and before else "host" if before else ""
            misses.append(f"  {category:11}{requests[index]['request']:46}{mark:8}{code[:56]}")
    print(path.rsplit("/", 1)[-1])
    for category, t in totals.items():
        spoke = f", program spoke {t['spoke']}" if before else ""
        print(f"  {category:11}{t['right']:3}/{t['count']:<3} right, {t['wrong']} wrong, {t['failed']} did not run{spoke}")
    return misses


def main():
    global CASES
    if sys.argv[1] == "write":
        write(sys.argv[2], "--chat" in sys.argv, "--fresh" in sys.argv)
    else:
        CASES = sys.argv[2]
        base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else None
        for path in [a for a in sys.argv[3:] if a != "--base" and a != base]:
            print("\n".join(score(path, base)))


if __name__ == "__main__":
    main()
