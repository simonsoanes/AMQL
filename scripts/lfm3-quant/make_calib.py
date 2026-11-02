"""Self-generated calibration corpus: LFM3 (BF16, via llama-server) answers a
diverse prompt set under its own chat template — thinking on, tools offered
on a share of prompts — and every rendered conversation is written out as
raw text for llama-imatrix / KLD evaluation.

usage: make_calib.py <server-url> <out.txt> [n_prompts]"""
import json, random, sys, urllib.request, urllib.error

url, out = sys.argv[1], sys.argv[2]
limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else 10**9
SEED = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 1
WIDE = '--wide' in sys.argv
rng = random.Random(SEED)

topics = ["photosynthesis", "the French Revolution", "black holes", "TCP vs UDP", "compound interest", "the immune system",
          "plate tectonics", "how vaccines work", "Rust ownership", "the Fourier transform", "supply and demand",
          "CRISPR", "the Roman Empire's fall", "neural networks", "climate feedback loops", "public-key cryptography",
          "the water cycle", "inflation", "quantum entanglement", "the printing press", "sorting algorithms",
          "garbage collection", "the Krebs cycle", "volcanoes", "music theory intervals", "the stock market"]
templates = [
    "Explain {t} to a curious 12-year-old.", "Give me three surprising facts about {t}.",
    "Write a short paragraph summarising {t}.", "What are common misconceptions about {t}?",
    "Compare {t} with something from everyday life.", "Write five quiz questions (with answers) about {t}.",
]
coding = [
    "Write a Python function that checks whether a string is a palindrome, ignoring punctuation.",
    "Write a C# method that returns the n-th Fibonacci number iteratively.",
    "Explain what this SQL does: SELECT dept, AVG(salary) FROM staff GROUP BY dept HAVING COUNT(*) > 5;",
    "Write a bash one-liner that counts lines in all .py files under the current directory.",
    "Find the bug: def mean(xs): return sum(xs) / len(xs) - what happens for an empty list and how do I fix it?",
    "Write a JavaScript debounce function and explain how it works.",
    "Convert this loop to a list comprehension: out = []\nfor x in data:\n    if x % 2: out.append(x * x)",
    "Write a regular expression that matches ISO dates like 2026-11-01 and explain each part.",
    "What's the time complexity of binary search and why?",
    "Write a Rust function that reverses the words in a sentence.",
]
math = [
    "A train travels 180 km in 2.5 hours. What's its average speed in m/s?",
    "Solve for x: 3x + 7 = 2x - 5.", "What is 17% of 2,340?", "If I invest 1000 at 5% compounded yearly, what do I have after 10 years?",
    "How many ways can 5 people sit in a row?", "Is 221 prime? Show your reasoning.",
    "A rectangle has perimeter 36 and area 80. What are its sides?", "Convert 98.6 Fahrenheit to Celsius.",
]
writing = [
    "Write a polite email declining a meeting invitation.", "Write a haiku about the sea at night.",
    "Draft a two-sentence product description for a reusable water bottle.", "Rewrite this more formally: hey, can u send me the report asap?",
    "Write a limerick about a cat who learned to code.", "Give me a short motivational message for someone starting a new job.",
    "Translate into Spanish: The museum opens at nine and closes at five.", "Summarise the plot of Romeo and Juliet in three sentences.",
]
tools = json.load(open(r'D:\Dev\AMQL\tests\Amql.Tests\fixtures\tool-eval-cases.jsonl'.replace('.jsonl', '.jsonl'), encoding='utf-8')) if False else None
tool_cases = [json.loads(l) for l in open(r'D:\Dev\AMQL\tests\Amql.Tests\fixtures\tool-eval-cases.jsonl', encoding='utf-8') if l.strip() and not l.startswith('//')]
tool_prompts = [
    ("get_weather", "Is it going to be cold in {c} today?"), ("get_weather", "Do I need an umbrella in {c}?"),
    ("get_time", "What's the local time in {c}?"), ("search_web", "Find recent news about {t}."),
    ("convert_units", "How many pounds is {n} kilograms?"), ("calculate", "Compute {n} * {m} for me."),
    ("translate", "Say 'thank you very much' in {l}."), ("get_stock_price", "How is {k} stock doing?"),
]
cities = ["Berlin", "Lagos", "Lima", "Osaka", "Toronto", "Mumbai", "Sydney", "Cairo", "Oslo", "Denver"]
langs = ["Italian", "Japanese", "Portuguese", "Korean", "Dutch", "Swahili"]
tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOG"]
tool_defs = {}
for c in tool_cases:
    for t in c["tools"]:
        tool_defs[t["function"]["name"]] = t

prompts = []
for t in topics:
    for tpl in (templates if WIDE else rng.sample(templates, 2)):
        prompts.append({"messages": [{"role": "user", "content": tpl.format(t=t)}]})
prompts += [{"messages": [{"role": "user", "content": p}]} for p in coding + math + writing] * (3 if WIDE else 1)
for name, tpl in tool_prompts:
    for _ in range(12 if WIDE else 4):
        q = tpl.format(c=rng.choice(cities), t=rng.choice(topics), n=rng.randint(2, 900), m=rng.randint(2, 900),
                       l=rng.choice(langs), k=rng.choice(tickers))
        offered = [tool_defs[name]] + rng.sample([d for n2, d in tool_defs.items() if n2 != name], 2)
        prompts.append({"messages": [{"role": "user", "content": q}], "tools": offered})
for c in tool_cases:
    prompts.append({"messages": c["messages"], "tools": c["tools"]})
rng.shuffle(prompts)
prompts = prompts[:limit]
results = [rng.choice(["42", "sunny, 21C", "shipped", "done"]) for _ in prompts]

def post(path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=600))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{path}: {e.code} {e.read()[:300]!r}")

def render(i):
    p = prompts[i]
    body = {"messages": p["messages"], "temperature": 0.8 if WIDE else 0.7, "top_p": 0.95, "max_tokens": 1200,
            "seed": i + 1000 * SEED}
    if "tools" in p:
        body["tools"] = p["tools"]
    msg = post("/v1/chat/completions", body)["choices"][0]["message"]
    convo = list(p["messages"])
    reply = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("reasoning_content"):
        reply["reasoning_content"] = msg["reasoning_content"]
    if msg.get("tool_calls"):
        reply["tool_calls"] = [{"type": "function", "id": tc.get("id"), "function": {"name": tc["function"]["name"],
                                "arguments": json.loads(tc["function"]["arguments"] or "{}")}} for tc in msg["tool_calls"]]
    convo.append(reply)
    if msg.get("tool_calls"):
        # A tool result closes the turn (apply-template refuses to end on a call).
        convo.append({"role": "tool", "tool_call_id": msg["tool_calls"][0].get("id") or "call_0",
                      "content": json.dumps({"status": "ok", "result": results[i]})})
    tpl_body = {"messages": convo}
    if "tools" in p:
        tpl_body["tools"] = p["tools"]
    return post("/apply-template", tpl_body)["prompt"]

from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(4) as pool:
    texts = list(pool.map(render, range(len(prompts))))
with open(out, "w", encoding="utf-8") as f:
    for text in texts:
        f.write(text + chr(10) + chr(10))
print("wrote", out, len(texts))
