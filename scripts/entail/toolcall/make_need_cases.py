"""A fresh set for the call decision: requests that need a tool without naming it, and
requests that need none, with the same eight tools on offer in every case.

Written before any strategy was tried on it, so it is a held-out set for the omission
problem: "I'm flying to Oslo, do I need a coat?" gets prose instead of get_weather, and
"Let sarah@example.com know..." gets "I cannot send an email".

    python scripts/entail/toolcall/make_need_cases.py <tokenizer-dir> <out.jsonl> [--template <file.jinja>]
"""
import argparse
import json

from transformers import AutoTokenizer

from make_cases import CURRENCY, EMAIL, FLIGHTS, TIMER, WEATHER, call

CALENDAR = {"name": "create_event", "description": "Add an event to the user's calendar",
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
                "time": {"type": "string", "description": "HH:MM"}}, "required": ["title", "date"]}}
STOCK = {"name": "get_stock_price", "description": "Get the latest share price for a ticker symbol",
         "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}
SMS = {"name": "send_sms", "description": "Send a text message to a phone number",
       "parameters": {"type": "object", "properties": {
           "number": {"type": "string"}, "text": {"type": "string"}}, "required": ["number", "text"]}}

TOOLS = [EMAIL, WEATHER, TIMER, FLIGHTS, CURRENCY, CALENDAR, STOCK, SMS]

# (id, request, expected tool, the argument values that must be copied)
NEED = [
    # the two failures this set is about
    ("oslo-coat", "I'm flying to Oslo tomorrow. Do I need to pack a warm coat?", "get_weather", {"city": "Oslo"}),
    ("tell-sarah", "Let sarah@example.com know the meeting has moved to 3pm.", "send_email", {"to": "sarah@example.com"}),
    # weather, implied
    ("umbrella", "Should I take an umbrella when I go out in Dublin today?", "get_weather", {"city": "Dublin"}),
    ("shorts", "Is it shorts weather in Barcelona right now?", "get_weather", {"city": "Barcelona"}),
    ("run", "I want to go for a run in Chicago this morning, is it too icy?", "get_weather", {"city": "Chicago"}),
    ("wash", "Good day to dry the washing outside in Leeds?", "get_weather", {"city": "Leeds"}),
    # email, implied
    ("tell-boss", "Tell my boss at k.patel@work.com that I'm off sick today.", "send_email", {"to": "k.patel@work.com"}),
    ("thank-tom", "Can you thank tom@acme.org for the flowers? Send it by email.", "send_email", {"to": "tom@acme.org"}),
    ("invite-li", "Drop li.wei@uni.edu a note inviting her to the seminar on Friday.", "send_email", {"to": "li.wei@uni.edu"}),
    ("remind-team", "Remind team@startup.io by email that the demo is at noon.", "send_email", {"to": "team@startup.io"}),
    # timer, implied
    ("eggs", "I'm boiling eggs, give me a shout in 7 minutes.", "set_timer", {"minutes": "7"}),
    ("nap", "Wake me up from my nap in 25 minutes.", "set_timer", {"minutes": "25"}),
    ("tea", "My tea needs to steep for 4 minutes, let me know when.", "set_timer", {"minutes": "4"}),
    ("parking", "My parking runs out in 45 minutes, keep an eye on that for me.", "set_timer", {"minutes": "45"}),
    # flights, implied
    ("wedding", "My sister's wedding is in Lisbon on 2026-12-12 and I live in Paris. How do I get there?", "search_flights",
     {"origin": "Paris", "destination": "Lisbon", "date": "2026-12-12"}),
    ("conference", "I've a conference in Toronto on 2026-12-01, flying out of Boston. What's available?", "search_flights",
     {"origin": "Boston", "destination": "Toronto", "date": "2026-12-01"}),
    ("home", "I need to be home in Manchester from Madrid on 2026-12-23.", "search_flights",
     {"origin": "Madrid", "destination": "Manchester", "date": "2026-12-23"}),
    ("cheapest", "What's the cheapest way to fly Delhi to Mumbai on 2026-12-05?", "search_flights",
     {"origin": "Delhi", "destination": "Mumbai", "date": "2026-12-05"}),
    # currency, implied
    ("hotel", "The hotel is 300 euros a night. What's that in dollars?", "convert_currency", {"amount": "300"}),
    ("yen", "I've got 5000 yen left over from my trip, roughly how many pounds is it?", "convert_currency", {"amount": "5000"}),
    ("split", "We split a bill of 90 Swiss francs. How much is that in euros?", "convert_currency", {"amount": "90"}),
    ("rent", "My rent is 1200 pounds; my parents want to know what that is in Canadian dollars.", "convert_currency", {"amount": "1200"}),
    # calendar, implied
    ("dentist", "I've got the dentist on 2026-12-09 at 10:30, don't let me forget.", "create_event", {"date": "2026-12-09"}),
    ("party", "Put Mia's birthday party on 2026-12-14 in my diary.", "create_event", {"date": "2026-12-14"}),
    ("call-mum", "Pencil in a call with Mum on 2026-12-20 at 18:00.", "create_event", {"date": "2026-12-20"}),
    ("review", "Block out 2026-12-03 at 14:00 for the quarterly review.", "create_event", {"date": "2026-12-03"}),
    # stock, implied
    ("apple", "How are my Apple shares (AAPL) doing today?", "get_stock_price", {"ticker": "AAPL"}),
    ("sell", "Is now a good moment to sell my TSLA? What's it trading at?", "get_stock_price", {"ticker": "TSLA"}),
    ("msft", "Has MSFT gone up since this morning?", "get_stock_price", {"ticker": "MSFT"}),
    ("nvda", "My friend says NVDA crashed. Is that true?", "get_stock_price", {"ticker": "NVDA"}),
    # sms, implied
    ("text-dad", "Text my dad on +44 7700 900123 that I've landed safely.", "send_sms", {"number": "+44 7700 900123"}),
    ("late", "Let +1 555 0134 know I'm running ten minutes late.", "send_sms", {"number": "+1 555 0134"}),
    ("door", "Message +61 400 123 456 to say the door code is 4821.", "send_sms", {"number": "+61 400 123 456"}),
    ("pickup", "Tell the driver on +33 6 12 34 56 78 I'm at gate B.", "send_sms", {"number": "+33 6 12 34 56 78"}),
    # explicit, as a control
    ("explicit-weather", "What's the weather in Prague?", "get_weather", {"city": "Prague"}),
    ("explicit-timer", "Set a timer for 10 minutes.", "set_timer", {"minutes": "10"}),
]

NONE = [
    ("capital", "What is the capital of Australia?"),
    ("poem", "Write a two-line poem about autumn."),
    ("maths", "What is 17 times 23?"),
    ("define", "What does 'ephemeral' mean?"),
    ("joke", "Tell me a joke about cats."),
    ("history", "Who was the first person to walk on the moon?"),
    ("recipe", "How do I make a simple omelette?"),
    ("advice", "How can I sleep better at night?"),
    ("translate", "How do you say 'good morning' in Spanish?"),
    ("code", "How do I reverse a list in Python?"),
    ("hello", "Hi there! How are you?"),
    ("thanks", "Thanks, that's all I needed."),
    # decoys: words from a tool's domain, but nothing to look up or do
    ("coat-fabric", "Is wool or down warmer for a winter coat?"),
    ("rain-why", "Why is the sky grey when it rains?"),
    ("email-tips", "How should I sign off a formal email?"),
    ("email-what", "What does CC mean in an email?"),
    ("timer-history", "Who invented the egg timer?"),
    ("flight-fear", "How can I stop being nervous about flying?"),
    ("jetlag", "Any tips for beating jet lag?"),
    ("euro-history", "When was the euro introduced?"),
    ("stock-what", "What is a stock split?"),
    ("calendar-why", "Why does February have 28 days?"),
    ("sms-length", "How many characters can a text message have?"),
    ("currency-why", "Why do exchange rates change?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("out")
    ap.add_argument("--template")
    ap.add_argument("--reverse", action="store_true", help="list the tools in reverse order")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    template = open(args.template, encoding="utf-8").read() if args.template else None
    tools = [{"type": "function", "function": t} for t in (TOOLS[::-1] if args.reverse else TOOLS)]

    def render(request):
        return tok.apply_chat_template([{"role": "user", "content": request}], tools=tools, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False, chat_template=template)

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for cid, request, tool, values in NEED:
            f.write(json.dumps({"category": f"call:{cid}", "prompt": render(request), "answer": "0", "max_tokens": 96,
                                "completion": call(tool, **values), "tools": [t["name"] for t in TOOLS],
                                "after": "user", "request": request}, ensure_ascii=False) + "\n")
        for cid, request in NONE:
            f.write(json.dumps({"category": f"text:{cid}", "prompt": render(request), "answer": "0", "max_tokens": 96,
                                "completion": "<|im_end|>", "tools": [t["name"] for t in TOOLS],
                                "after": "user", "request": request}, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {len(NEED)} need a tool, {len(NONE)} need none")


if __name__ == "__main__":
    main()
