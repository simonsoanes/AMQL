"""A compile benchmark for code generation: small C# console tasks that lean on the base class
library, answered by a model behind an OpenAI-compatible server (no tools, a single file), and
judged by `dotnet build`. Counts the programs that build, and separately the errors that mean the
model invented an API (CS0117 no such member, CS0103 no such name, CS0246 no such type, CS1061 no
such instance member). Each configuration (--config name=json-fields, repeatable) is merged into
the request, so the same tasks run with and without an n-gram pack:

    python scripts/ngram/compile_bench.py --url http://127.0.0.1:18080 --model qwen --work <dir> \
        --config 'base={"ngram_packs":[]}' --config 'pack={"ngram_packs":["dotnet"]}'
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import urllib.request

TASKS = [
    "Print the names and sizes in bytes of the files in the current directory, largest first.",
    "List every immediate subfolder of the folder given as the first argument, with the total size of all files inside it (recursively), largest first.",
    "Read a text file given as the first argument and print how many lines, words and characters it has.",
    "Count how often each word occurs in a text file given as the first argument and print the ten most frequent words.",
    "Print the current date and time, the day of the week, and the number of days until the end of the year.",
    "Read all lines of input.txt, remove duplicate lines while keeping their order, and write the result to output.txt.",
    "Compute and print the SHA-256 hash of a file given as the first argument, as lowercase hex.",
    "Serialize a list of three Person objects (Name, Age) to indented JSON with System.Text.Json, print it, then deserialize it back and print the names.",
    "Find all files with the extension .log under a folder given as the first argument (recursively) and delete those older than 7 days, printing each one.",
    "Use a regular expression to find all email addresses in a text file given as the first argument and print them sorted and without duplicates.",
    "Download the page https://example.com with HttpClient and print its length and the text of its <title> element.",
    "Start the process 'dotnet --version', capture its standard output and print it.",
    "Print every environment variable whose name starts with 'DOTNET', sorted by name.",
    "Read a CSV file given as the first argument (with a header row, comma separated, no quotes) and print the sum of its second column.",
    "Watch the current directory for created files with FileSystemWatcher and print the name of each new file until a key is pressed.",
    "Generate 10 random passwords of 16 characters using letters, digits and symbols with RandomNumberGenerator, one per line.",
    "Print the first 20 Fibonacci numbers using a long and an iterator method with yield return.",
    "Copy every .txt file from a folder given as the first argument into a subfolder called backup, creating it if needed.",
    "Print the total and free space of every ready drive on the machine using DriveInfo.",
    "Compress a folder given as the first argument into a .zip file with ZipFile and print the size of the zip.",
    "Parse a date given as the first argument in the format yyyy-MM-dd and print how many days ago it was.",
    "Read numbers from the console until an empty line, then print their count, minimum, maximum and average.",
    "Split a large text file given as the first argument into files of 1000 lines each, named part1.txt, part2.txt and so on.",
    "Print a tree of the folders under the current directory, indented by depth, using recursion.",
]

SYSTEM = ("You are a C# programmer. Answer with one complete Program.cs file for a .NET 11 console app (top-level "
          "statements or a Main method) in a single ```csharp code block, and nothing else. Use only the base class library.")

CSPROJ = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net11.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>disable</Nullable>
    <TreatWarningsAsErrors>false</TreatWarningsAsErrors>
  </PropertyGroup>
</Project>
"""

INVENTED = ("CS0117", "CS0103", "CS0246", "CS1061", "CS0234")


def ask(url, model, task, extra, max_tokens, temperature=0.0, seed=0):
    body = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "seed": seed, "top_p": 0.95, "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]}
    body.update(extra)
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def code_of(reply):
    m = re.search(r"```(?:csharp|cs|c#)?\s*\n(.*?)```", reply, re.S)
    return m.group(1) if m else reply


def build(work, name, code):
    d = os.path.join(work, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    open(os.path.join(d, "bench.csproj"), "w").write(CSPROJ)
    open(os.path.join(d, "Program.cs"), "w", encoding="utf8").write(code)
    r = subprocess.run(["dotnet", "build", "-nologo", "-v", "q", "-clp:NoSummary"], cwd=d, capture_output=True, text=True, timeout=300)
    errors = sorted({l.split(": error ")[1].split(":")[0] + ": " + l.split(": error ")[1].split(":", 1)[1].strip()[:110]
                     for l in (r.stdout + r.stderr).splitlines() if ": error " in l})
    return r.returncode == 0, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--work", required=True)
    ap.add_argument("--config", action="append", required=True, help="name=json merged into each request")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=len(TASKS))
    ap.add_argument("--report")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=1, help="draws per task (with --temperature)")
    ap.add_argument("--jobs", type=int, default=6, help="requests at once (the server batches them)")
    args = ap.parse_args()
    configs = [(c.split("=", 1)[0], json.loads(c.split("=", 1)[1])) for c in args.config]
    report = {}
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(i, t, seed) for i, t in enumerate(TASKS[:args.limit]) for seed in range(args.seeds)]
    for name, extra in configs:
        def one(job):
            i, task, seed = job
            reply = ask(args.url, args.model, task, extra, args.max_tokens, args.temperature, seed)
            ok, errors = build(args.work, f"{name}-{i:02d}-{seed}", code_of(reply))
            return {"task": task, "seed": seed, "built": ok, "errors": errors}
        with ThreadPoolExecutor(args.jobs) as pool:
            rows = list(pool.map(one, jobs))
        built = sum(r["built"] for r in rows)
        invented = sum(1 for r in rows for e in r["errors"] if e.split(":")[0] in INVENTED)
        report[name] = {"built": built, "tasks": len(rows), "invented_errors": invented, "cases": rows}
        print(f"== {name}: {built}/{len(rows)} built, {invented} invented-API errors", flush=True)
    if args.report:
        json.dump(report, open(args.report, "w"), indent=1)


if __name__ == "__main__":
    main()
