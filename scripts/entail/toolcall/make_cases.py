"""Synthetic tool-calling cases for the LoRA -> Entail experiment.

One training entry, and held-out probes that separate the hypotheses about how a
model maps a request (A) to a tool (B) with arguments (C):

  * memorised:  the training request's exact tool and values come back whatever is asked
  * by name:    the tool is chosen by its name's tokens, so renaming it breaks the choice
  * by meaning: the tool is chosen by its description, so renaming it does not
  * copy:       argument values are copied from the request, so a new city is copied too

Rendered with the model's own chat template (tools in the system turn, thinking
off), so the prompts are exactly what the model is served.

    python scripts/entail/toolcall/make_cases.py <tokenizer-dir> <out-dir>
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

WEATHER = {"name": "get_weather", "description": "Get the current weather for a city",
           "parameters": {"type": "object", "properties": {
               "city": {"type": "string", "description": "The city name"},
               "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
               "required": ["city"]}}
EMAIL = {"name": "send_email", "description": "Send an email to a recipient",
         "parameters": {"type": "object", "properties": {
             "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
             "required": ["to", "subject", "body"]}}
TIMER = {"name": "set_timer", "description": "Start a countdown timer",
         "parameters": {"type": "object", "properties": {
             "minutes": {"type": "integer", "description": "Length of the timer in minutes"}},
             "required": ["minutes"]}}
FLIGHTS = {"name": "search_flights", "description": "Search for flights between two cities on a date",
           "parameters": {"type": "object", "properties": {
               "origin": {"type": "string"}, "destination": {"type": "string"},
               "date": {"type": "string", "description": "YYYY-MM-DD"}},
               "required": ["origin", "destination", "date"]}}
CURRENCY = {"name": "convert_currency", "description": "Convert an amount of money from one currency to another",
            "parameters": {"type": "object", "properties": {
                "amount": {"type": "number"}, "from": {"type": "string", "description": "ISO code"},
                "to": {"type": "string", "description": "ISO code"}},
                "required": ["amount", "from", "to"]}}

TOOLS = [EMAIL, WEATHER, TIMER, FLIGHTS, CURRENCY]


def renamed(tool, name):
    return {**tool, "name": name}


def call(name, **args):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call><|im_end|>"


TRAIN = [
    ("train", TOOLS, "What's the weather like in Paris right now? Please give it in celsius.",
     call("get_weather", city="Paris", unit="celsius")),
]

HOLDOUT = [
    # same tool, new values: is the city copied or remembered?
    ("new-city", TOOLS, "What's the weather like in Tokyo right now? Please give it in fahrenheit.",
     call("get_weather", city="Tokyo", unit="fahrenheit")),
    ("paraphrase", TOOLS, "Is it going to be cold in Berlin today? Answer in celsius.",
     call("get_weather", city="Berlin", unit="celsius")),
    # same request, tool list reordered: is the choice positional?
    ("reordered", [WEATHER, CURRENCY, TIMER, EMAIL, FLIGHTS],
     "What's the weather like in Paris right now? Please give it in celsius.",
     call("get_weather", city="Paris", unit="celsius")),
    # renamed tool: is the choice by name or by description?
    ("renamed", [EMAIL, renamed(WEATHER, "fetch_conditions"), TIMER, FLIGHTS, CURRENCY],
     "What's the weather like in Paris right now? Please give it in celsius.",
     call("fetch_conditions", city="Paris", unit="celsius")),
    # other tools: did training on one call damage the rest?
    ("timer", TOOLS, "Set a timer for 15 minutes please.", call("set_timer", minutes=15)),
    ("currency", TOOLS, "How much is 250 US dollars in euros?",
     call("convert_currency", amount=250, **{"from": "USD", "to": "EUR"})),
    ("flights", TOOLS, "Find me flights from London to New York on 2026-12-03.",
     call("search_flights", origin="London", destination="New York", date="2026-12-03")),
    ("email", TOOLS, "Email bob@example.com with the subject Lunch and say I'll be 10 minutes late.",
     call("send_email", to="bob@example.com", subject="Lunch", body="I'll be 10 minutes late.")),
]


# ── situational planning: the policy, not the request's surface, picks the tool ──

ORDER_STATUS = {"name": "get_order_status", "description": "Look up the delivery status of an order",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                               "required": ["order_id"]}}
CANCEL = {"name": "cancel_order", "description": "Cancel an order",
          "parameters": {"type": "object", "properties": {
              "order_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["order_id"]}}
ADDRESS = {"name": "update_address", "description": "Change the delivery address of an order",
           "parameters": {"type": "object", "properties": {
               "order_id": {"type": "string"}, "address": {"type": "string"}}, "required": ["order_id", "address"]}}
SEND_CODE = {"name": "send_verification_code", "description": "Email a one-time code to the customer",
             "parameters": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]}}
CHECK_CODE = {"name": "verify_code", "description": "Check the code the customer read back",
              "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}

SHOP_TOOLS = [ORDER_STATUS, CANCEL, ADDRESS, SEND_CODE, CHECK_CODE]
POLICY = ("You are the support agent for an online shop.\n"
          "Policy: before cancelling an order or changing its address you MUST verify the customer: "
          "first call send_verification_code with their email. Only after the code is verified may you "
          "call cancel_order or update_address. Looking up an order's status needs no verification.")

POLICY_TRAIN = [
    ("train", SHOP_TOOLS, POLICY, "Please cancel my order ORD-4821, my email is anna@example.com.",
     call("send_verification_code", email="anna@example.com")),
]

POLICY_HOLDOUT = [
    # same situation, new values: is the email copied?
    ("new-values", SHOP_TOOLS, POLICY, "Cancel order ORD-1177 please, I'm tom.h@mailbox.net.",
     call("send_verification_code", email="tom.h@mailbox.net")),
    # a different gated action: did it learn "gated actions", or "cancel"?
    ("address", SHOP_TOOLS, POLICY,
     "Change the delivery address of ORD-2290 to 12 High Street, Leeds. My email is li@site.org.",
     call("send_verification_code", email="li@site.org")),
    # an ungated action: does it now verify everything?
    ("status", SHOP_TOOLS, POLICY, "What's the status of ORD-4821? My email is anna@example.com.",
     call("get_order_status", order_id="ORD-4821")),
    # no policy: the right call is the direct one. Strategy read from context, or baked in?
    ("no-policy", SHOP_TOOLS, "You are the support agent for an online shop.",
     "Please cancel my order ORD-4821, my email is anna@example.com.",
     call("cancel_order", order_id="ORD-4821")),
    # tool list reordered
    ("reordered", [SEND_CODE, CANCEL, CHECK_CODE, ORDER_STATUS, ADDRESS], POLICY,
     "Please cancel my order ORD-4821, my email is anna@example.com.",
     call("send_verification_code", email="anna@example.com")),
    # a policy of the same shape over different tools: transfer of the strategy itself
    ("other-domain", [WEATHER, TIMER, {**CANCEL, "name": "delete_account", "description": "Delete the user's account",
                                       "parameters": {"type": "object", "properties": {"user": {"type": "string"}},
                                                      "required": ["user"]}}, SEND_CODE],
     "You are an assistant.\nPolicy: before deleting an account you MUST first call "
     "send_verification_code with the user's email.",
     "Delete my account please, user jkim, email jkim@corp.io.",
     call("send_verification_code", email="jkim@corp.io")),
]


# ── the call decision: a tool is needed, but the request never names it ──
# The 0.8B model's typical failure is answering in prose when it should call.
# The decision is the reply's first token: "<tool_call>" or text.

NEED_TRAIN = [
    ("train", TOOLS, None, "I'm flying to Oslo tomorrow. Do I need to pack a warm coat?",
     call("get_weather", city="Oslo")),
]

NEED_HOLDOUT = [
    ("implicit-weather", TOOLS, None, "Thinking of a picnic in Madrid this afternoon, good idea?",
     call("get_weather", city="Madrid")),
    ("implicit-timer", TOOLS, None, "Remind me in 20 minutes to take the pizza out of the oven.",
     call("set_timer", minutes=20)),
    ("implicit-currency", TOOLS, None, "I owe my friend 40 euros. How many pounds is that?",
     call("convert_currency", amount=40, **{"from": "EUR", "to": "GBP"})),
    ("implicit-email", TOOLS, None, "Let sarah@example.com know the meeting has moved to 3pm.",
     call("send_email", to="sarah@example.com", subject="Meeting moved", body="The meeting has moved to 3pm.")),
    ("implicit-flights", TOOLS, None, "I need to get from Rome to Vienna on 2026-12-20, what are my options?",
     call("search_flights", origin="Rome", destination="Vienna", date="2026-12-20")),
    ("explicit-weather", TOOLS, None, "What's the weather in Oslo?", call("get_weather", city="Oslo")),
    # needs no tool: the right first token is prose. Completions are only indicative.
    ("none-capital", TOOLS, None, "What is the capital of France?", "The capital of France is Paris.<|im_end|>"),
    ("none-coat", TOOLS, None, "Is wool or down warmer for a winter coat?", "Down is generally warmer<|im_end|>"),
    ("none-poem", TOOLS, None, "Write one line about the sea.", "The sea<|im_end|>"),
]


# Anchors: prompts with no label. The trainer has the base model answer them
# itself and keeps those answers, so a single example cannot drag everything with
# it. Half need a tool and half do not; none overlaps a held-out case.
ANCHORS = [
    ("fact", TOOLS, None, "Tell me a fun fact about octopuses.", ""),
    ("japanese", TOOLS, None, "How do I say thank you in Japanese?", ""),
    ("percent", TOOLS, None, "What's 15% of 80?", ""),
    ("lima", TOOLS, None, "What's the weather in Lima? Use fahrenheit.", ""),
    ("timer5", TOOLS, None, "Start a 5 minute timer.", ""),
    ("francs", TOOLS, None, "Convert 100 Swiss francs to US dollars.", ""),
    ("where", SHOP_TOOLS, POLICY, "Where is my order ORD-5512?", ""),
    ("thanks", SHOP_TOOLS, POLICY, "Thanks for your help earlier!", ""),
]


def render(tok, tools, request, system=None):
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": request}]
    return tok.apply_chat_template(
        messages,
        tools=[{"type": "function", "function": t} for t in tools],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main():
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    suites = (("train", [(i, t, None, r, c) for i, t, r, c in TRAIN]),
              ("holdout", [(i, t, None, r, c) for i, t, r, c in HOLDOUT]),
              ("policy-train", POLICY_TRAIN),
              ("policy-holdout", POLICY_HOLDOUT),
              ("need-train", NEED_TRAIN),
              ("need-holdout", NEED_HOLDOUT),
              ("anchors", ANCHORS))
    for name, rows in suites:
        with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for case_id, tools, system, request, completion in rows:
                prompt = render(tok, tools, request, system)
                f.write(json.dumps({"id": case_id, "prompt": prompt, "completion": completion,
                                    "request": request, "tools": [t["name"] for t in tools]},
                                   ensure_ascii=False) + "\n")
                print(f"{name:8} {case_id:12} prompt {len(tok(prompt)['input_ids'])} tokens, "
                      f"completion {len(tok(completion)['input_ids'])} tokens")


if __name__ == "__main__":
    main()
