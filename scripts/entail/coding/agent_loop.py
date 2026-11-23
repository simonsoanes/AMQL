"""Drives amql-server (or any OpenAI-compatible server) through an agent loop with a desktop
harness's tools, and answers every call with a simulated result: nothing is ever executed or
written. Prints each turn: reasoning length, content, calls. For checking that a model served
over the API reasons, calls tools and finishes the job.

    python scripts/entail/coding/agent_loop.py --url http://127.0.0.1:18080 "Write a Python script that prints FizzBuzz"
"""
import argparse
import json
import time
import urllib.request

from make_coding_cases import HARNESS, HARNESS_SYSTEM

TOOLS = [{"type": "function", "function": t} for t in HARNESS]


def simulate(name, args):
    """What a harmless harness would answer: a workspace, success for writes, empty listings."""
    if "current" in name or "cwd" in name:
        return json.dumps({"working_directory": "D:\\work", "permitted": ["D:\\work"]})
    if any(w in name for w in ("create", "write", "edit")):
        text = args.get("content") or args.get("new_text") or ""
        return json.dumps({"ok": True, "path": args.get("path"), "lines": text.count("\n") + 1 if text else 0})
    if "read" in name:
        return json.dumps({"error": f"file not found: {args.get('path')}"})
    if any(w in name for w in ("list", "glob", "search")):
        return json.dumps({"entries": []})
    if "run" in name or "terminal" in name:
        return json.dumps({"exitCode": 0, "stdout": "", "stderr": ""})
    if "ask" in name:
        return "Please just go ahead with your best judgement."
    return json.dumps({"ok": True})


def post(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request")
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--temperature", type=float, default=0)
    args = ap.parse_args()
    messages = [{"role": "system", "content": HARNESS_SYSTEM}, {"role": "user", "content": args.request}]
    for turn in range(args.turns):
        t = time.time()
        body = {"messages": messages, "tools": TOOLS, "max_tokens": args.max_tokens, "temperature": args.temperature,
                "chat_template_kwargs": {"enable_thinking": not args.no_think}}
        reply = post(args.url, body)
        choice = reply["choices"][0]
        m = choice["message"]
        calls = m.get("tool_calls") or []
        print(f"--- turn {turn + 1}: {choice['finish_reason']}, {reply['usage']['completion_tokens']} tokens, "
              f"{time.time() - t:.1f}s, reasoning {len(m.get('reasoning_content') or '')} chars")
        if m.get("content"):
            print(m["content"][:1500])
        for c in calls:
            a = json.loads(c["function"]["arguments"])
            shown = {k: (v[:600] + "…" if isinstance(v, str) and len(v) > 600 else v) for k, v in a.items()}
            print(f"CALL {c['function']['name']} {json.dumps(shown, ensure_ascii=False)}")
        messages.append({"role": "assistant", "content": m.get("content"), "tool_calls": calls or None,
                         "reasoning_content": m.get("reasoning_content")})
        if not calls:
            break
        for c in calls:
            result = simulate(c["function"]["name"], json.loads(c["function"]["arguments"]))
            print(f"RESULT {result}")
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})


if __name__ == "__main__":
    main()
