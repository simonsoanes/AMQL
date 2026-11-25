"""Goldens for Gemma 4's tool-calling prompts: what transformers renders from the model's own
template (fixtures/gemma4_chat_template.jinja) for conversations with tools, calls, results and
reasoning. TurnTemplateTests renders the same conversations and compares exactly.

    python tests/Amql.Tests/fixtures/make_gemma4_tool_golden.py
"""
import json
import os

from transformers.utils.chat_template_utils import _compile_jinja_template

HERE = os.path.dirname(os.path.abspath(__file__))

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "The weather in a place — now",
    "parameters": {"type": "object", "properties": {
        "place": {"type": "string", "description": "Where"},
        "unit": {"type": "string", "enum": ["C", "F"], "description": "Which"},
        "kind": {"type": "string", "enum": ["now", "later"]},
        "Days": {"type": "integer", "nullable": True}},
        "required": ["place", "unit"]}}}
EDIT = {"type": "function", "function": {
    "name": "edit_file", "description": "Edit a file",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"},
        "edits": {"type": "array", "description": "What to do", "items": {"type": "object", "properties": {
            "op": {"type": "string", "enum": ["replace", "delete"]},
            "lines": {"type": "array", "items": {"type": "string"}}}, "required": ["op"]}},
        "tags": {"type": "array", "items": {"type": ["string", "null"], "description": "a tag", "enum": ["x", 2]}},
        "any": {"type": "array"},
        "opts": {"type": "object", "description": "More", "properties": {
            "deep": {"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]},
            "force": {"type": "boolean", "description": "Even so", "nullable": True}}, "required": ["force"]},
        "env": {"type": "object", "additionalProperties": {"type": "string"}},
        "free": {"type": "object"},
        "either": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Either"},
        "maybe": {"type": ["string", "null"], "enum": ["q"]}},
        "required": ["path"]}}}
BARE = {"type": "function", "function": {"name": "now", "description": "The time"}}
EMPTY = {"type": "function", "function": {"name": "ping", "description": "", "parameters": {"type": "object", "properties": {}}}}


def call(name, args, id):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": args}}


AGENT = [
    {"role": "system", "content": " You are Hydra.\n"},
    {"role": "user", "content": "Make it warm. "},
    {"role": "assistant", "content": "", "reasoning_content": "I should look first.",
     "tool_calls": [call("get_weather", {"unit": "C", "place": "Paris\n{\"x\"}", "Days": 3}, "a")]},
    {"role": "tool", "content": "{\"temp\": 12}\n", "tool_call_id": "a"},
]
CASES = [
    {"name": "tools", "tools": [WEATHER, EDIT, BARE, EMPTY], "messages": [{"role": "user", "content": "hi"}]},
    {"name": "tools-system", "tools": [WEATHER], "messages": [{"role": "system", "content": " Be brief. "}, {"role": "user", "content": "hi"}]},
    {"name": "one-result", "tools": [WEATHER], "messages": AGENT},
    {"name": "second-call", "tools": [WEATHER, EDIT], "messages": AGENT + [
        {"role": "assistant", "content": "", "reasoning_content": "Now edit.", "tool_calls": [
            call("edit_file", {"path": "a", "edits": [{"op": "replace", "lines": ["b", ""]}], "opts": {"force": True, "deep": {"a": 0.5}},
                               "free": {"z": None, "B": [1, 2.0, False]}}, "b"),
            call("now", {}, "c")]},
        {"role": "tool", "content": "14:00", "tool_call_id": "c"},
        {"role": "tool", "content": "done", "tool_call_id": "b"}]},
    {"name": "answered", "tools": [WEATHER], "messages": AGENT + [
        {"role": "assistant", "content": " It is 12 degrees. ", "reasoning_content": "Simple."},
        {"role": "user", "content": "And now?"}]},
    {"name": "answer-continues", "tools": [WEATHER], "messages": AGENT + [
        {"role": "assistant", "content": "It is 12 degrees."}]},
    {"name": "unanswered", "tools": [BARE], "messages": [
        {"role": "user", "content": "time?"}, {"role": "assistant", "content": "", "tool_calls": [call("now", {}, "a")]}]},
    {"name": "named-result", "tools": None, "messages": [
        {"role": "user", "content": "time?"}, {"role": "assistant", "content": "", "tool_calls": [call("now", {}, "a")]},
        {"role": "tool", "content": "5", "tool_call_id": "zz", "name": "clock"}]},
]


def main():
    template = _compile_jinja_template(open(os.path.join(HERE, "gemma4_chat_template.jinja"), encoding="utf-8").read())
    out = []
    for case in CASES:
        for thinking in (False, True):
            out.append({**case, "thinking": thinking, "rendered": template.render(
                messages=case["messages"], tools=case["tools"], add_generation_prompt=True, enable_thinking=thinking, bos_token="<bos>")})
    path = os.path.join(HERE, "gemma4_tool_golden.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} cases to {path}")


if __name__ == "__main__":
    main()
