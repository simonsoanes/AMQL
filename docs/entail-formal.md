# Dates, number bases, logic and JSON in Entail

These four subjects are rule systems with nothing to learn: the calendar, positional
notation, truth tables and a grammar. Each rule fixes the answer once a request is written
in the subject's notation. This page records what the programs do, how they scored, and
where they stop.

| Program | Reads | Writes |
|---|---|---|
| [`dates.entail.txt`](../scripts/entail/dates.entail.txt) | `2024-03-15 was a` | `Friday` |
| | `Days from 2024-01-01 to 2024-03-15 =` | `74 days` |
| | `30 days after 15 March 2024 is` | `2024-04-14` |
| [`bases.entail.txt`](../scripts/entail/bases.entail.txt) | `255 in binary is` | `11111111` |
| | `0xff in decimal is` | `255` |
| | `16777215 in base 14 is` | `232a207` |
| [`logic.entail.txt`](../scripts/entail/logic.entail.txt) | `p or not p is` | `a tautology` |
| | `((p -> q) and q) -> p is` | `contingent` |
| | `Truth table of p or r:` | `01011111` |
| [`json.entail.txt`](../scripts/entail/json.entail.txt) | `JSON: {"a": [1, 2]}` then `Valid:` | ` yes` |
| | a JSON text being written | only tokens that keep it valid |

## The idea in each

**Dates are one law.** A year has 365 days, plus one every 4 years, less one every 100,
plus one every 400. Weekdays, differences and date sums all follow from thresholds on those
counts, and nothing is listed year by year. A date that doesn't exist entails nothing.

**A number is the same in any base.** Each digit of the source adds a fixed amount to each
column of the target, so a conversion is a linear sum followed by carrying. Every number
passes through 8 hexadecimal columns, and one carry chain serves all the target bases, with
the base as an input. No large number is ever held, so 32 bits fit easily.

**Logic tries every case.** There are three letters, so eight assignments. Tokens become
events, and a small evaluator runs once per assignment. All eight true is a tautology, none
is a contradiction, and anything else is contingent. Precedence is not, then and, then or
and xor, then `->` and `<->`.

**JSON is a grammar, used in two ways.** A verdict reads a whole text and says whether it is
valid. A guard reads a text as it is written, and rules out every token that would break it.
That is constrained decoding held in the weights, with no sampler outside the model. It
needed the new `only` and `forbid` statements in
[the language](entail-language.md#output), which narrow what the host may say without
choosing for it.

Three of the programs are generated, because a tokenizer doesn't respect the notation. Qwen
cuts `0xDEADBEEF` as `0|xDE|ADB|EE|F`, merges `((p`, and writes `":[` as one token, which
means different things inside and outside a string. So `make_bases.py`, `make_logic.py` and
`make_json.py` list each token with what it contributes. `make_json.py` runs one small
scanner over the whole vocabulary, from inside and outside a string, and that scanner
decides both the verdict and the guard's token sets, so the two can't disagree.

## Results

Qwen 3.5 0.8B, on the CPU. Each prompt opens with a short drill so that the base model
answers. Answers were worked out separately in Python.

| Test | Cases | Base | Graft |
|---|---|---|---|
| Weekdays | 90 | 12 | 90 |
| Days between dates | 60 | 1 | 60 |
| Adding and taking days | 60 | 5 | 60 |
| Leap years | 16 | 13 | 16 |
| Month lengths | 24 | 23 | 24 |
| To binary, hex and octal | 90 | 7 | 90 |
| From hex, binary and octal | 120 | 28 | 120 |
| Hex, octal and binary to each other | 30 | 4 | 30 |
| To and from bases 2 to 16 | 75 | 2 | 75 |
| Expressions with true and false | 80 | 20 | 80 |
| Tautology, contradiction or contingent | 90 | 11 | 89 |
| Named laws | 12 | 12 | 12 |
| Named fallacies | 6 | 0 | 6 |
| Truth tables | 30 | 2 | 30 |
| JSON verdicts, valid texts | 150 | 63 | 150 |
| JSON verdicts, altered texts | 220 | 154 | 220 |

The graft broke no answer the base model had right. The one logic miss is an expression of
29 events, over the limit of 28: the program stayed silent and the host answered wrongly.
On 23 requests outside the programs' range, all stayed silent.

The base model gets all 12 laws and none of the 6 fallacies, so its 12 says little. The
altered JSON texts are mostly invalid, which flatters a model that tends to say `no`.

**The guard.** Each of 60 unfinished texts, 2 to 8 brackets deep, was given to the host to
continue for 48 tokens.

| | Base | Graft |
|---|---|---|
| Still valid | 38 | 60 |
| Reached the end of the document | 1 | 4 |

The guard keeps a text valid, but it can't make the host finish: most replies were still
going, validly, when the 48 tokens ran out.

| Program | Layers (host 24) | Loss on prose (base 3.55195) |
|---|---|---|
| Dates | 63 | 3.55207 |
| Bases | 77 | 3.55222 |
| Logic | 79 | 3.55256 |
| JSON | 85 | 3.55213 |

Further sets with new seeds were scored through `entail run` with Qwen's tokenizer, without
weights: dates 747 of 747, bases 945 of 945, JSON 1,126 of 1,126, and logic 649 of 658, all
9 misses being silence on an expression over the length limit. The JSON set predates the
blank-space rule, and the verdicts were rechecked after it, through the weights.

## What this doesn't do

**No chat answer got better.** Over 10 plain-English questions per program, the graft and
base scored the same: dates 3, bases 2, logic 7, and JSON 8 replies that parse. Dates and
JSON gave identical replies. Five replies changed for bases and logic, with no change in
score. In chat Qwen works through a method, such as Zeller's congruence or repeated
division, and never writes the phrase a program waits for. As with the sciences, reaching
chat needs a host that writes the notation, or a program that reads the question itself.

**JSON helps structured output, not Qwen's tool calls.** Qwen's native tool format is XML
(`<tool_call><function=…>`), which the guard doesn't know. It fits a reply or arguments asked
for as JSON. In chat the model already wrote valid JSON: the 2 failures of 10 were cut off by
the token limit, and the guard had nothing to correct. Its value showed on hard unfinished
texts, and it switches on only after `JSON:` or `json` and a new line.

The rest are limits of scope:

- **Dates** run from 1583 to 9999. Two dates may be 300 years apart, and at most 9999 days
  may be added or taken away. There are no times, time zones or Julian dates.
- **Bases** take whole numbers up to 32 bits. There are no negatives, fractions, bitwise
  operations or arithmetic between bases, and no `hex()` style of request. ` octal` isn't
  one token, and the program reads the two pieces.
- **Logic** has three letters, p, q and r, up to 30 tokens or 28 events, and two levels of
  bracket. Mixing `or` and `xor` at one level, or chaining `->` without brackets, is silent.
  `∨` isn't a single token, so it isn't supported. An expression needs another token in
  front of it.
- **JSON** sees 512 tokens and 8 levels of nesting. Deeper or longer texts are outside what
  was tested.
- **The guard says what is legal, not what is true.** A valid reply can still hold the wrong
  value, and no schema is checked.

## What was learned

**A mask is a different kind of statement.** Until now a program either chose the next token
or kept quiet. Narrowing the choice lets a program enforce a rule on a host that still does
the writing.

**Legal is not the same as useful.** Blank space is always valid, so the first guard let the
host write blank lines for ever, in 2 replies of 60. A rule against more than two blanks in
a row ended that, without touching the verdicts. A constraint moves a model's failures, so
read what it writes as well as whether it passes.

**Doubling sees a long way cheaply.** Nine heads carry a fact over 512 tokens, by letting
each token combine with what lies 1, 2, 4, 8 and more back. See
[Seeing further than a window](entail-language.md#seeing-further-than-a-window).

**A graft spends neurons the scheduler can't see.** It clears dead lanes as well, so logic,
packed to the host's 3,584 neurons, overflowed. It compiles with `--neurons-per-layer 3200`.

**Check the measure before the model.** Two alarming results were the test's fault. The guard
looked like 4 of 60, because the test asked for a finished text, and a mask can't finish one.
Truth tables looked like 14 of 30, because the host writes the leading space as its own token,
which left room for 7 digits of 8. Both came right when the test did.

## Reproducing it

`dates.entail.txt` is written by hand. The others are generated:

```bash
python scripts/entail/make_bases.py <tokenizer.json> scripts/entail/bases.entail.txt
```

```bash
python scripts/entail/make_logic.py <tokenizer.json> scripts/entail/logic.entail.txt
```

```bash
python scripts/entail/make_json.py <tokenizer.json> scripts/entail/json.entail.txt
```

Make cases with `dates`, `bases`, `logic` or `json`, then run and score them:

```bash
python scripts/entail/formal_cases.py dates cases.jsonl 7
```

```bash
amql-cli entail run scripts/entail/dates.entail.txt --tokenizer <dir> --cases cases.jsonl --report replies.json
```

```bash
python scripts/entail/formal_cases.py score cases.jsonl replies.json
```

Graft and evaluate as in [entail-synthesis.md](entail-synthesis.md), then pass the base
model's report to `score` with `--base`. For logic, add `--neurons-per-layer 3200` to the
graft. The guard can only be tested in a graft, because the interpreter ignores masks: its
cases come from `guard_cases()` in `formal_cases.py`, and `guard_score` scores them.
