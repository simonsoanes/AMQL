"""Goldens for tool-calling chat prompts: what transformers' apply_chat_template renders with
Qwen3.5's real template (fixtures/qwen-lm-tiny) for conversations with tools, calls, tool
results and reasoning. ServerTests renders the same conversations and compares exactly.

    python tests/Amql.Tests/fixtures/make_tool_chat_golden.py
"""
import json
import os

from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))

WRITE = {"type": "function", "function": {
    "name": "create_file", "description": "Create a file — the whole content, no placeholders",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Where"},
        "content": {"type": "string"}},
        "required": ["path", "content"]}}}
RUN = {"type": "function", "function": {
    "name": "run_terminal", "description": "Run a command",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}, "timeout": {"type": "integer"}, "env": {"type": "object"},
        "args": {"type": "array"}, "elevated": {"type": "boolean"}, "ratio": {"type": "number"}},
        "required": ["command"]}}}

CASES = [
    {"name": "tools-system", "thinking": False, "tools": [WRITE, RUN], "messages": [
        {"role": "system", "content": "You are Hydra.\n"},
        {"role": "user", "content": "Write fizzbuzz.py"}]},
    {"name": "tools-no-system-thinking", "thinking": True, "tools": [WRITE], "messages": [
        {"role": "user", "content": "Écris un fichier, s'il te plaît"}]},
    {"name": "tools-empty-system", "thinking": False, "tools": [RUN], "messages": [
        {"role": "system", "content": "  "},
        {"role": "user", "content": "ls"}]},
    {"name": "agent-loop", "thinking": True, "tools": [WRITE, RUN], "messages": [
        {"role": "system", "content": "You are Hydra."},
        {"role": "user", "content": "Make a script and run it."},
        {"role": "assistant", "content": "", "reasoning_content": "I should write it first.",
         "tool_calls": [{"type": "function", "function": {"name": "create_file", "arguments": {
             "path": "a.py", "content": "print('hi')\nprint(2)"}}}]},
        {"role": "tool", "content": "created a.py (2 lines)"},
        {"role": "assistant", "content": "Now I run it.",
         "tool_calls": [
             {"type": "function", "function": {"name": "run_terminal", "arguments": {
                 "command": "python a.py", "timeout": 30, "env": {"A": "1", "B": [1, 2]},
                 "args": ["-u", 3], "elevated": False, "ratio": 0.5}}},
             {"type": "function", "function": {"name": "run_terminal", "arguments": {"command": "echo done"}}}]},
        {"role": "tool", "content": "hi\n2"},
        {"role": "tool", "content": "done"}]},
    {"name": "history-before-query", "thinking": False, "tools": [RUN], "messages": [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "<think>\nold thoughts\n</think>\n\nold answer",
         "tool_calls": [{"type": "function", "function": {"name": "run_terminal", "arguments": {"command": "dir"}}}]},
        {"role": "tool", "content": "a.txt"},
        {"role": "assistant", "content": "There is a.txt."},
        {"role": "user", "content": "second"}]},
    {"name": "tool-turns-without-tools", "thinking": False, "tools": None, "messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "calling",
         "tool_calls": [{"type": "function", "function": {"name": "x", "arguments": {}}}]},
        {"role": "tool", "content": "r"}]},
]


def main():
    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "qwen-lm-tiny"))
    out = []
    for case in CASES:
        rendered = tok.apply_chat_template(case["messages"], tools=case["tools"], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=case["thinking"])
        out.append({**case, "rendered": rendered})
    path = os.path.join(HERE, "qwen35_tool_chat_golden.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"cases": out}, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} cases to {path}")


if __name__ == "__main__":
    main()
