# The Entail language

Reference for the language compiled by `amql-cli entail`. For why it exists, how
the weights stay exact and what it achieved on a real model, see
[entail.md](entail.md). The full worked program is
[`scripts/entail/mathematics.entail.txt`](../scripts/entail/mathematics.entail.txt).

An Entail program says what each token means, what follows from that, and what
token must come next. It has no runtime of its own: `compile` turns it into a
model, and from then on it is just weights.

## A first program

This program answers `7?` with `8`: the digit after the one before the `?`,
wrapping 9 round to 0.

```
program next_digit

# Input facts: what each token is.
state val, known, asked
for d in 0 to 9
    token "{d}" means val = d, known = 1
end
token "?" means asked = 1

# Sight: the token one place back.
state before, there
attend back 1
    before <- val
    there  <- known
end

# Consequences.
state ask, wrap, answer
entail ask when asked, there
entail wrap when before >= 9, ask
entail answer = before + 1 - 10*wrap when ask

# Output facts: the token that must come next.
state o[10]
for d in 0 to 9
    entail o[d] when answer == d, ask
    emit "{d}" when o[d]
end
```

Save it as `next-digit.entail.txt`, then:

```bash
amql-cli entail check next-digit.entail.txt
```

```bash
amql-cli entail run next-digit.entail.txt --expr " 9?"
```

`check` reports the size of the compiled model and `run` prints ` 9?0`. Note the
leading space: see [The start of the text](#the-start-of-the-text).

## Lexical rules

- A statement is one line. A line may break after a comma in a `state`, `token`
  or `when` list.
- `#` starts a comment that runs to the end of the line.
- Indentation means nothing; blocks close with `end`.
- Names are letters, digits and `_`, starting with a letter or `_`.
- Numbers are decimal, with an optional fraction: `9`, `0.5`.
- Strings are in double quotes. `\n` is a newline, `\q` is a double quote and `\{` is
  a literal brace. There are no other escape sequences: `"\"` is one backslash.
- `{x}` inside a name or string is replaced by the value of the loop variable or
  `def` parameter `x`: `token "{d}"`, `state {p}_carry[n]`. Constants cannot be
  used this way.

Reserved words: `program const state scale let token means for in to if else end
def call attend back by entail when not emit vote sum min floor`.

## Expressions

An expression is a *linear form*: a constant plus lanes multiplied by constants.

| Form | Meaning |
|---|---|
| `a + b`, `a - b`, `-a` | sum, difference, negation |
| `3 * a`, `a * 3` | scaling; one side must be constant |
| `a / 2` | scaling; the divisor must be constant |
| `2 ^ k` | power, of constants only |
| `(sum i in 0 to n-1 : 10^i * d[i])` | sum over a range; the brackets are required |
| `min(a, b)`, `floor(a)` | of constants only |
| `x`, `x[i]`, `x[i][j]` | a constant, loop variable, `let` or state |

A product of two states is refused, because it is not linear. Gate one by the
other instead, with `when`. Index expressions must be constant.

## Declarations

```
program NAME
```
Names the program. Optional.

```
const NAME = EXPR
```
A compile-time number. Constants are global, and assigning one again replaces it.

```
state NAME, NAME[DIM], NAME[DIM][DIM] scale EXPR, ...
```
Declares interim facts. A state with dimensions is an array, indexed from 0. Each
element is one residual lane, allocated the first time it is used. Dimensions are
reused once a lane is dead, so a large program can compile to a narrow model.

`scale` is the raw value stored per unit of meaning. Use a power of two, and use
one for any state that holds large numbers: `state rem[9] scale 1/8192` holds a
remainder near 100,000 as a raw value near 12, keeping it clear of the norm's
exactness budget (see [Limits](#limits-a-program-must-respect)).

```
let NAME = EXPR
let NAME[I] = EXPR
```
A name for a linear form. It costs no weights: its value is substituted wherever
it is used. Its right-hand side is evaluated at this point, and `let`s are global.

```
token "TEXT", "TEXT" means LANE = VALUE, LANE = VALUE
```
What an input token is: the lanes its embedding row sets. Values are constants.
Tokens missing from a model's vocabulary are skipped and listed in the build
report, so one program can list spellings for several tokenizers.

## Generators

These run at compile time and leave nothing in the weights.

```
for VAR in FROM to TO
    ...
end
```
Both bounds are inclusive. If `FROM` is above `TO` the body does not run.

```
if LEFT OP RIGHT
    ...
else
    ...
end
```
A compile-time choice between constants. `OP` is one of `< > <= >= == !=`, and
`else` is optional. This is not a runtime branch: a condition on a state belongs
in `when`.

```
def NAME(P, Q)
    ...
end

call NAME(ARG, ARG)
```
A macro. A bare name that is not a number binds as a name, so a `def` can be told
which states to use; anything else binds as a number. Use `{p}` to build names from
a parameter:

```
def chain(p, n)
    state {p}_c[n + 1]
    ...
end
call chain(add, ADD)        # declares and defines add_c
```

## Attention

```
attend back K
    DEST <- EXPR
    DEST <- EXPR
end
```
One attention head. Each destination receives the value its expression has at the
token `K` places back, where `K` is a constant. The body may use `for` and `if`.

```
attend back by POINTER
    DEST <- EXPR
end
```
The same, with a computed distance. `POINTER` is a one-index state: if `POINTER[c]`
is set, the head looks `c` tokens back. At most one element may be set. If none is,
the head reads a blend of the context, so gate whatever consumes its result.

## Consequences

Each `entail` adds to its target. Several statements on one state sum, so exclusive
conditions read as "or", and a value under exclusive conditions as a case analysis.

```
entail Y when E >= C              # Y += 1 if the whole number E reaches C
entail Y when F, not G, E == C    # Y += 1 if every condition holds
entail Y = V when F, G            # Y += V if every condition holds
entail Y = V                      # Y += V, unconditionally
```

`V` is any linear form. A `when` list is a conjunction, and each item is one of:

| Item | Requires |
|---|---|
| `F`, `not F` | a form that is exactly 0 or 1 |
| `E >= C`, `E <= C`, `E > C`, `E < C` | a whole-number form `E` |
| `E == C` | a whole-number form `E` |

An `entail` with a value and no `when` adds the value unconditionally. It is rarely
needed, since `let` does the same thing for free.

A threshold inside a longer list costs one extra layer, because it needs a lane of
its own to be gated on. The same threshold asked twice shares that lane. A lone
threshold compiles directly, so to save depth in a chain, prefer
`entail fit when rem - k*b >= 0` to adding a redundant condition.

Thresholds regenerate their input: their output is exactly 0 or 1 even if the input
is slightly off. A chain of them therefore cannot accumulate noise.

## Output

```
emit "TEXT" when LANE
emit end when LANE
```
When the lane is set, that token must come next. `emit end` says the answer is over.
In an empty model it is an end-of-sequence token. In a host, where text simply goes
on, it forbids the program's other output tokens at that position (except a blank)
and the host says what comes next.

Where no output lane is set, the program says nothing. In a host, this is what
leaves the model's own prediction untouched.

```
vote "TEXT" by EXPRESSION
```
A soft output, for programs that describe a model before they replace it. Votes for
one token add up, and the token with the highest positive score is said, as a softmax
would choose. An `emit` that fires comes first. `entail decompile` writes votes, the
interpreter runs them, and `compile` and `graft` refuse them: a vote is what is left to
be refined into `emit`.

## Rules the compiler enforces

- **A state is written, then read.** Once anything has read a state it is closed,
  and writing to it again is an error. So a program is a dataflow graph, written
  in order, and it cannot contain a cycle.
- **States are declared.** Using an undeclared state, or an index outside its
  dimensions, is an error, reported with its line.
- **Forms stay linear.** See [Expressions](#expressions).

There are no layers in the language. The compiler places each statement at the
earliest layer its inputs allow, and depth follows from the longest chain of
dependencies. Attention at a layer runs before that layer's neurons.

## Compilation at a glance

| You write | You get |
|---|---|
| `token` | values in an embedding row |
| `state` | a residual dimension, for as long as the lane is live |
| `let`, `const`, `for`, `if`, `def`, `sum` | nothing |
| `attend back` | one query head, sharing a constant key and a value table |
| `E >= C`, `E <= C` and so on | two neurons |
| `E == C` | four neurons |
| a `when` list, or a value | two neurons, plus any thresholds it needs |
| `emit`, tied head (host) | one column of the embedding |
| `emit`, untied head (empty model) | one entry of an output row |

## Limits a program must respect

The compiler checks the structural limits. The numeric ones are the author's to
keep, and `entail run` will not reveal a breach, because the interpreter uses
doubles: test the compiled model too.

| Limit | Value |
|---|---|
| Heads per layer | 8 (4 per KV group); extra heads move to a later layer |
| Distinct values one KV group reads in a layer | 256 |
| Neurons per layer | 3584 by default (`--neurons-per-layer`); extra ones move later |
| Largest whole number in a threshold | below 131072, for 24-bit exactness at gain 128 |
| Sum of squares of all raw lanes at one position | about 65536 × ballast dimensions |
| `scale` | a power of two |
| A `token` value in a graft | exact in bf16, so whole numbers up to 256 |

A host's embedding is bf16, so a larger value has to arrive in parts and be joined by a
`let`: see `lfh` and `lfl` in `chemistry.entail.txt`. Keep coefficients to values a float holds exactly. The two ramps of a threshold only
cancel exactly if they do, which is why the carry chain in the mathematics program
compares eight places at a time.

## The start of the text

A head can only look back as far as the text goes. Past the start it finds the first
token, and softmax with RoPE cannot tell where the text began from inside a run of
identical tokens. So a program should treat what it sees past the start as harmless.
In `mathematics.entail.txt`, a first token that is not a digit simply ends the window,
while a number that opens the text looks too long and nothing is entailed. That is why
the examples here start with a space.

## Running a program

```bash
amql-cli entail check program.entail.txt --context 32768
```
Compiles, reports the size, and proves every head's attention margin over the context.

```bash
amql-cli entail run program.entail.txt --expr " 12+34="
```
Runs the meaning with no weights, one character per token, feeding each output back.
`--tokenizer <dir>` splits the text as that model would, and `--cases <file>` answers
each case, with `--report <json>` for the replies. This is the quick way to test a
program before grafting it.

```bash
amql-cli entail compile program.entail.txt --out model
```
Writes a checkpoint for an empty model. Use `amql-cli encode` to make it a container.

```bash
amql-cli entail graft program.entail.txt --host qwen --out grafted
```
Writes a restructured copy of a Qwen3.5-family checkpoint.

```bash
amql-cli entail eval container --tokenizer dir --cases cases.jsonl --report out.json
```
Greedy-decodes JSONL cases of `{"category", "prompt", "answer"}`, with optional
`chat`, `first` and `max_tokens` fields. A case passes when the last number of the
reply (or the first, with `first`) equals `answer`. `--text <file>` also measures
next-token loss and records top-1 predictions, for comparing a model with its graft.
