"""Coding requests with file and terminal tools on offer, to measure whether a small model
acts (writes the file) or talks (answers with a code block), and whether what it writes is
complete or full of placeholders.

The tools are definitions only: nothing a reply asks for is ever executed. `entail bench
--score tools` reads the reply's call, if any, and scores it.

Rendered with the model's own chat template (tools in the system turn), so the prompts are
exactly what the model is served. Each line has "raw": true, so the bench uses the prompt
as it is, "expect" (the tool that should be called first, or "none"), and "category".

    python scripts/entail/coding/make_coding_cases.py <tokenizer-dir> <out.jsonl> [--system agent|plain|none] [--think]
"""
import argparse
import json

from transformers import AutoTokenizer

WRITE = {"name": "write_file", "description": "Create or overwrite a file in the workspace with the given content",
         "parameters": {"type": "object", "properties": {
             "path": {"type": "string", "description": "Path of the file, relative to the workspace"},
             "content": {"type": "string", "description": "The complete content of the file"}},
             "required": ["path", "content"]}}
READ = {"name": "read_file", "description": "Read a file from the workspace",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
RUN = {"name": "run_terminal", "description": "Run a shell command in the workspace and return its output",
       "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}
LIST = {"name": "list_directory", "description": "List the files in a directory of the workspace",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
TOOLS = [WRITE, READ, RUN, LIST]


def tool(tool_name, description, **params):
    return {"name": tool_name, "description": description, "parameters": {"type": "object", "properties": {
        k: {"type": "string", "description": v} for k, v in params.items()}, "required": list(params)}}


# A larger set, named as a desktop agent harness names them: what the model meets in real use.
HARNESS = [
    tool("get_current_directory", "Return the working directory and the directories the agent may use"),
    tool("list_directory", "List the files and folders in a directory", path="Directory to list"),
    tool("glob", "Find files whose names match a pattern", pattern="Glob pattern, e.g. **/*.cs", path="Where to search"),
    tool("search_file_content", "Search files for text or a regular expression", search_text="Text to find", path="Where to search"),
    tool("read_file", "Read a file", path="File to read"),
    tool("create_file", "Create a new file with the given content", path="Path of the new file", content="The complete content of the file"),
    tool("edit_file", "Replace text in an existing file", path="File to edit", old_text="Exact text to replace", new_text="Replacement text"),
    tool("create_directory", "Create a directory", path="Directory to create"),
    tool("run_terminal", "Run a shell command (PowerShell on Windows) and return its output", command="The command line"),
    tool("web_search", "Search the web", query="What to search for"),
    tool("fetch_url", "Fetch a web page as text", url="The address"),
    tool("ask_user", "Ask the user a question and wait for the answer", question="The question"),
]
HARNESS_SYSTEM = """You are Hydra, an autonomous assistant running on the user's Windows desktop. You can read and \
write files in the permitted directories, run terminal commands, search the web and ask the user questions, using \
the tools provided.

Guidelines:
- Work out what the user wants, then do it with the tools rather than describing what you would do.
- When asked to build or write software, create the files in the working directory and build or run them to check \
they work.
- Prefer small, verifiable steps. Read a file before editing it.
- If a command fails, read the error and change your approach rather than repeating it.
- Keep replies to the user short: say what you did and what remains.
- Ask the user only when you cannot proceed without an answer."""
WRITE_NAMES = {"write_file": "create_file"}

# What else a desktop assistant offers beside its coding tools, so the request sits behind
# several thousand tokens of tool definitions, as it does in real use.
EXTRA = [
    tool(f"{verb}_{noun}", f"{verb.capitalize()} {noun.replace('_', ' ')} for the user", **{arg: f"The {arg}"})
    for verb, noun, arg in [
        ("send", "email", "to"), ("read", "inbox", "folder"), ("search", "email", "query"),
        ("create", "calendar_event", "title"), ("list", "calendar_events", "date"), ("delete", "calendar_event", "event_id"),
        ("create", "note", "text"), ("search", "notes", "query"), ("update", "note", "note_id"),
        ("remember", "fact", "fact"), ("recall", "memory", "query"), ("forget", "memory", "memory_id"),
        ("set", "reminder", "when"), ("list", "reminders", "date"), ("send", "chat_message", "channel"),
        ("get", "weather", "city"), ("convert", "currency", "amount"), ("translate", "text", "text"),
        ("summarise", "document", "path"), ("take", "screenshot", "window"), ("open", "application", "name"),
        ("play", "music", "track"), ("create", "spreadsheet", "title"), ("create", "presentation", "title"),
        ("start", "timer", "minutes"), ("search", "contacts", "name"), ("call", "phone_number", "number"),
        ("order", "taxi", "destination"), ("book", "meeting_room", "time"), ("schedule", "task", "cron"),
    ]
]

SYSTEMS = {
    "agent": "You are a coding agent working in the user's workspace. Use the tools to do what the user asks: "
             "write code into files rather than showing it in chat.",
    "plain": "You are a helpful assistant.",
    "none": None,
}

# (id, request, first tool expected)
WRITE_TASKS = [
    ("fizzbuzz", "Write a Python script that prints FizzBuzz from 1 to 100.", "write_file"),
    ("wordcount", "Write a Python program that counts how often each word appears in a text file given on the command line.", "write_file"),
    ("csvsum", "Create a script that reads sales.csv and prints the total of the 'amount' column.", "write_file"),
    ("todo-cli", "Make me a small command-line todo list app in Python that stores items in a JSON file, with add, list and done commands.", "write_file"),
    ("primes", "Write a C# console program that prints all primes below 10000 using a sieve.", "write_file"),
    ("stack", "Implement a generic Stack<T> class in C# with Push, Pop, Peek and Count, backed by an array that grows.", "write_file"),
    ("bank", "Write a Python class BankAccount with deposit, withdraw (refusing overdrafts) and a transaction history.", "write_file"),
    ("json-pretty", "Write a Node.js script that pretty-prints a JSON file passed as an argument.", "write_file"),
    ("dedupe", "Create a Python tool that finds duplicate files in a folder by hashing their contents.", "write_file"),
    ("temp", "Write a JavaScript function that converts between Celsius and Fahrenheit, with a small test at the bottom.", "write_file"),
    ("matrix", "Write a Python module with functions to add, multiply and transpose matrices stored as lists of lists.", "write_file"),
    ("diskusage", "Write a .NET console utility that prints the size of each folder under a given path, largest first.", "write_file"),
    ("snake", "Write a complete snake game in Python using pygame.", "write_file"),
    ("http", "Write a minimal HTTP server in Python using only the standard library that serves files from the current folder.", "write_file"),
    ("calc", "Build a calculator in Python that evaluates expressions with + - * / and brackets, without using eval.", "write_file"),
    ("roman", "Write a C# method that converts integers to Roman numerals and back, in a file RomanNumerals.cs.", "write_file"),
    ("linkedlist", "Implement a doubly linked list in TypeScript with insert, remove and iteration.", "write_file"),
    ("password", "Write a Python script that generates a random password with options for length and symbols.", "write_file"),
    ("markdown", "Create a script that converts a Markdown file's headings into an HTML table of contents.", "write_file"),
    ("timer-web", "Make a single HTML page with a countdown timer the user can start, pause and reset.", "write_file"),
]
OTHER_TASKS = [
    ("run-tests", "Run the test suite with pytest and tell me what fails.", "run_terminal"),
    ("what-files", "What files are in the src folder?", "list_directory"),
    ("explain-readme", "Read README.md and summarise what this project does.", "read_file"),
    ("git-status", "Show me the git status of this repository.", "run_terminal"),
]
NO_TOOL = [
    ("explain-gc", "Explain how garbage collection works in .NET, briefly.", "none"),
    ("compare", "What's the difference between a list and a tuple in Python?", "none"),
    ("bigo", "What is the time complexity of binary search, and why?", "none"),
    ("naming", "Is camelCase or snake_case more common in Python?", "none"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("out")
    ap.add_argument("--system", default="agent", choices=list(SYSTEMS) + ["harness", "harness-big"])
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=1536)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    harness = args.system.startswith("harness")
    tools = (HARNESS + EXTRA if args.system == "harness-big" else HARNESS) if harness else TOOLS
    system = HARNESS_SYSTEM if harness else SYSTEMS[args.system]
    with open(args.out, "w", encoding="utf-8") as f:
        for group, tasks in (("write", WRITE_TASKS), ("other", OTHER_TASKS), ("chat", NO_TOOL)):
            for cid, request, expect in tasks:
                if harness:
                    expect = WRITE_NAMES.get(expect, expect)
                messages = ([{"role": "system", "content": system}] if system else [])
                messages.append({"role": "user", "content": request})
                prompt = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=args.think)
                f.write(json.dumps({"id": cid, "category": group, "prompt": prompt, "raw": True,
                                    "think": args.think, "expect": expect, "answer": "0",
                                    "max_tokens": args.max_tokens}) + "\n")


if __name__ == "__main__":
    main()
