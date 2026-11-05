# Extracting and optimising a subject

This guide does two jobs. It tells a person how to take a subject out of a trained
model and refine it, and it is the brief given to whichever model reviews a program.
A reviewer needs nothing else: this guide, [the language reference](entail-language.md)
and the program are enough. A rule belongs here before anyone relies on it, so
improving the optimiser usually means improving this document.

## What this does and does not recover

Trained weights cannot be decompiled directly: what a model knows about a subject is
spread across many units and superposed with everything else it knows. So the cycle
does not translate weights into Entail. It works from two kinds of evidence instead:

- **Where** the subject lives, measured from activations and tested by ablation.
- **What** the model does, measured by probing it across a declared domain.

The result is a program that reproduces the model's behaviour on that domain and,
where a law exists, replaces it with something exact. It is a statement of what the
model does, not a reading of how the weights do it.

## The cycle

```bash
amql-cli entail find <container> --subject <file> --out <dir> --top 128 --verify
```
Locates the units that carry the subject, and checks the claim causally.

```bash
amql-cli entail extract <container> --subject <file> --out <dir>
```
Probes each relation and writes `evidence.jsonl` and `v0.entail.txt`, a faithful table.

```bash
amql-cli entail optimise <dir>
```
Applies rules R1 to R4 below, each through the gate, and writes `current.entail.txt`,
`journal.jsonl` and `overruled.jsonl`.

```bash
amql-cli entail review <dir> --endpoint <url> --model <name>
```
Offers the program to a reviewer through an OpenAI-compatible server. Without
`--endpoint`, it writes `review.prompt.md`, which any reviewer can answer, and the
answer goes through `entail accept`:

```bash
amql-cli entail accept <dir> --candidate <file> --rule R5 --why "<reason>"
```

## Subject files

A subject is a JSON file; see `scripts/entail/subjects/`.

| Field | Meaning |
|---|---|
| `name` | The program's name |
| `lines` | Text that shows the subject, for `find` |
| `contrast` | Text that does not, in a similar register, for `find` |
| `relations` | What the subject consists of, for `extract` |

A relation is a question the model can be asked over a domain:

| Field | Meaning |
|---|---|
| `name` | A name, used as a prefix in the program |
| `template` | Text with `{slot}` placeholders |
| `slots` | Each slot's domain: a list of strings, or `"lo-hi"` |
| `shots` | Worked examples in front of the template, ending in a newline |
| `sample` | How many instances to draw; all of them if absent |
| `until` | Text that ends the answer; a newline by default |
| `max_tokens` | How long an answer may be |

Every instance of a relation must tokenise to the same length, so keep numeric slots
to one width (`10-99`, not `1-99`). Write shots so the unmodified model simply answers:
the evidence is only as good as the question.

## The evidence

Each row records the context tokens, the model's answer, and its **margin**: the
smallest gap between the top two probabilities anywhere in the answer. The margin is
what the softmax leaves behind. A row near 1 was decided; a row near 0 was a coin toss
that greedy decoding happened to resolve.

About a third of the rows are held out. No rule may be fitted to them. Only the gate
reads them.

Rows only show where a program should speak, so they cannot catch a rule that has
become too broad. `background.jsonl` is the negative evidence: every position of the
subject's `lines` and `contrast`, with the token the model predicts there. A **false
fire** is a position where the program names a different token.

## The gate

Every change, from any author, passes through the same mechanical gate. A change is
kept only if it compiles, adds no false fires, and either:

- raises agreement with decided held-out rows, or
- keeps agreement on decided rows, held-out and fitted, and lowers cost.

Cost is neurons plus eight per attention head. Agreement counts rows with a margin of
at least the confidence threshold, 0.5 by default. A reviewer's argument counts for
nothing here: an elegant change that loses a row is refused.

The gate measures agreement with the model, which is not the same as being right. A
law can be correct and still disagree with confident rows, because the model can be
confidently wrong. Those rows are listed in `overruled.jsonl`, most decided first, and
they are the most interesting output of the cycle: either the model is wrong there, or
the law is. The gate cannot tell which. A person or an independent check must.

## The rules

Cite a rule by number. Each states when it applies and what it must preserve.

### R1 Keep only what was decided
*Applies to:* table rows with a margin below the confidence threshold.
*Change:* delete them. A probability is not a fact, and a coin toss is not evidence.
*Preserves:* every decided row. *Mechanical.*

### R2 Hoist shared conditions
*Applies to:* rows of a table that repeat the same literal conditions.
*Change:* state them once as a fact, `name_on[step]`, and use that fact in each row.
*Preserves:* meaning exactly. *Mechanical.*

### R3 Drop what does not matter
*Applies to:* a slot where rows differing only in that slot always give the same answer,
and at least one such pair of rows exists.
*Change:* remove the slot's conditions, which merges those rows. Unseen values of the
slot are then answered too, which is a generalisation: the held-out rows test it.
*Mechanical.*

### R4 Replace a table with a law
*Applies to:* a relation whose slots and answers are whole numbers.
*Change:* derive `value = c0 + c1·a + c2·b + c3·a·b` by trimmed least squares on the
decided, fitted rows. A law exists only if the coefficients come out whole and at least
60% of the rows satisfy it exactly. Nothing is searched: the basis is fixed, and a mere
trend produces fractions and is rejected.
*Preserves:* the rows that obey it. Rows that don't are overruled, and listed.
*Mechanical.*

### R5 Reuse a proven definition
*Applies to:* a law or table that covers part of something already defined exactly, such
as `scripts/entail/mathematics.entail.txt`.
*Change:* use the general definition instead of the narrow one, keeping every relation's
tokens and answers. The result usually costs more and covers far more, so justify it by
held-out agreement, or by coverage the evidence does not reach. *Reviewer.*

### R6 Share work between answer positions
*Applies to:* the same computation repeated for each answer position.
*Change:* compute it once, at the position that ends the question, and have later
positions read the result with `attend back by`, as `mathematics.entail.txt` does with
`from` and `seen`. *Preserves:* every answer. *Reviewer.*

### R7 Collapse thresholds
*Applies to:* several thresholds on one form that feed one sum, or a `when` list whose
thresholds could be one comparison.
*Change:* use the fewest thresholds that decide the same whole numbers. Remember that a
lone threshold costs no extra layer, while one inside a list does. *Reviewer.*

### R8 Name the structure
*Applies to:* a table that resists R3 and R4, but whose rows fall into classes.
*Change:* give the class a fact, by giving its tokens a shared value in `token ... means`,
and write one rule per class instead of one row per member. *Reviewer.*

### R9 Add a rule to this guide
*Applies to:* an improvement no rule covers.
*Change:* make it, cite R9, and say what the new rule should be. If it survives the gate
and recurs, it belongs in this document, where every later reviewer inherits it.

## Reviewing a program

1. Read the journal and `Where the program and the model differ`. Rules already applied
   need no repeating, and a refused rule was refused for a reason.
2. Choose one rule, and make one change. Small changes are easier to gate and to read.
3. Keep every `token`, `emit` and answer the evidence depends on. Add nothing the evidence
   and the rules do not justify.
4. Return the whole program in one fenced code block, naming the rule and the reason.

Do not edit a program to agree with overruled rows without a reason to think the model
was right. Removing the model's mistakes is the point of the exercise.

## The raw extract

`extract` asks narrow questions of a declared subject. `decompile` is the opposite: it reads
what the model's units do across a corpus and writes all of it down, trained-in material
unrelated to the subject included. This is the state before optimisation, and it is
meant to be large and wrong.

```bash
amql-cli entail decompile <container> --corpus <dir> --out <dir> --contrast <prose.txt> \
    --extra <targeted.txt> --cover --units 1000000
amql-cli entail fidelity <dir>/raw.entail.txt --holdout <dir>/holdout.jsonl
python scripts/entail/python_coverage.py <dir>/raw.entail.txt <tokenizer-dir> --missing missing.json
python scripts/entail/python_doc_corpus.py missing.json targeted.txt
```

Each chunk of the corpus goes through the model once. At each position, the 24 neurons per
layer that stand furthest above their own running mean are recorded, with the tokens that
fired them and what the model said next. One neuron becomes:

```
# L23 neuron 67: fired 20823x, subject-specific (5.6x prose)
state L23_n67
entail L23_n67 = 4 when sym == 12   # " the" x948
entail L23_n67 = 6 when sym == 124   # " ->" x349
entail L23_n67 = 4 when prev == 49, sym == 44   # "    " " #" x136
# column promotes " SB" " PB" " CB" " FS" " MF" " GB"
vote " the" by 0.0064*L23_n67   # x1384
vote " " by 0.0053*L23_n67   # x1007
```

- **Rules** say which tokens fired the neuron, in levels of 1 to 8. Only the current and
  previous token are visible, so this is the ceiling on what the form can express.
- **Votes** are what the model was seen to say when the neuron fired, weighted by
  `--vote`. The default, `lift`, is P(token | fired) · log(P(token | fired) / P(token)).
  They are measured correlations, not read from the weights.
- **The column comment** is what the weights themselves promote: the neuron's output column
  read through the final norm and output head. It is a comment because, read alone, it
  is noise before the last few layers. Used as votes, it put Chinese, Cyrillic and Arabic
  tokens on most Python neurons, and agreement fell below always saying a newline.

`--cover` keeps the strongest neuron for every token seen, so rare names survive. `--extra`
adds targeted text on top of `--corpus-tokens`. `study.bin` holds everything the study
recorded, so `--from <study.bin>` rewrites the program in minutes with other settings.
`--no-columns` skips the column comments, which are most of that time.

Not represented: the 18 linear-attention layers, how neurons read each other, and
attention beyond the previous token. Head offsets are sampled from each chunk's last row
and written as comments.

`fidelity` runs the program over held-out text. Read its score against two references:
always saying the commonest token, and the `reference:` line from the study, which is a plain
table of what the model said after each token or pair. Coverage says a name is somewhere in
the program. It does not say the program knows what the name means.

Text in the held-out set is not studied, so items that appear only there read as missing.
`--only-holdout` records what the model says and studies nothing, which makes a test set
from any corpus.

To go from an extract to a program that answers requests, see
[entail-synthesis.md](entail-synthesis.md).

## Results

Qwen 3.5 0.8B, using `scripts/entail/subjects/`. The refined programs are kept there.

**Find.** Removing the 128 highest-scoring of 86,016 neurons, against 128 at random from
the same layers, changes loss in nats per token by:

| Subject | Top units: subject | Top units: contrast | Random: subject | Random: contrast |
|---|---|---|---|---|
| Python | +0.087 | +0.011 | -0.001 | +0.009 |
| Mathematics | +0.028 | -0.003 | +0.005 | +0.012 |

The units matter to their subject and the effect is modest, as expected from knowledge
this widely spread. On a routed model `find` ranks experts instead. That has only been
run on a `moe-ify`'d Qwen, whose perplexity the conversion had ruined (35 to 4,927), so it
shows the path works and says nothing about a well-trained MoE.

**Mathematics.** Two-digit `+`, `-` and `*`, 300 instances, 91 of them held out.

| Stage | Held-out agreement | False fires | Cost |
|---|---|---|---|
| v0, extracted table | 0/70 | 0 | 2236 |
| R1, undecided rows removed | 0/70 | 0 | 1882 |
| R4, laws for add, sub and mul | 69/70 | 0 | 3408 |
| R6, reviewer: compute once, read back | 69/70 | 0 | 498 |

R4 derived `a + b`, `a - b` and `a*b` from the model's answers, with no arithmetic given
to it. The refined program disagrees with the model on 47 of the 300 rows. Checked against
true arithmetic afterwards, the program is right on all 47, and the model wrong. Mean
softmax margin is 0.79 on the model's right answers and 0.14 on its wrong ones, so the
margin does mark most of the loss, and 46 of the 47 fall below the threshold. The one
that does not is `51*47 = 2407`, a confident mistake.

**Reloading.** The refined program, grafted back into Qwen, on 150 operand pairs that were
not in the evidence, scored against true arithmetic:

| | Base | Reloaded |
|---|---|---|
| add | 50/50 | 50/50 |
| sub | 31/50 | 50/50 |
| mul | 45/50 | 50/50 |
| **overall** | **126/150 (84.0%)** | **150/150 (100%)** |
| prose: top-1 agreement with base | | 767/767 |
| prose: loss, nats/token | 3.55195 | 3.55198 |

The reloaded model has hidden size 1152 and 41 layers, against 1664 and 59 for the
hand-written `mathematics.entail.txt`, which in exchange covers far more.

**Python.** 38 instances, 15 held out. R3 was offered both slots of `if ready` → `:`. The
gate refused to drop the keyword, which would have fired wrongly on ordinary text, and
kept dropping the name, which took held-out agreement from 0/15 to 15/15 and neurons from
296 to 128. The subject is deliberately small. A rule like this speaks after any word that
follows a keyword on a new line, so it is a statement about these relations, not about
Python in general, and it has not been reloaded into a model.

The R6 rewrite above was written by Claude, reviewing from the generated prompt alone.
Other reviewers are untested. `entail review --endpoint` has been exercised against a stub
server, not a real model, and whether a small model can do this job is unknown.

**Raw extract of Python.** Stdlib source and targeted documentation, 70,469 tokens studied,
3,686 held out. 82,751 of 85,067 neurons seen, 1,804,954 rule lines, 103 MB.

| Iteration | Rule lines | Documented items present | Agreement | Past the 20 commonest |
|---|---:|---:|---:|---:|
| Column votes, 8,000 neurons | 187,003 | 73% (inflated) | 5.5% | |
| Observed share, 8,000 neurons | 187,275 | 38% | 9.9% | |
| Lift, all neurons, documentation added | 1,804,954 | 93% | 20.5% | 9.8% |

For the last row, always saying a newline scores 8.0%, and the table of what the model said
scores 22.2% by token and 26.5% by pair. On the same study, `share` scores 11.3% and
pure log lift (`bayes`) 2.5%. The second iteration's documentation was chosen from what the
first lacked, so its 93% shows names reached the program, not that it can use them.
Agreement is against the model's top prediction, not the correct text.

## Scope

What exists today, and what does not:

- Slots must have a fixed token width, and an instance must be one continuation of one
  template. Free-form code or prose is out of reach: a subject has to be expressible as
  relations over a declared domain.
- R4 covers a bilinear basis over at most two numeric slots. Division, powers and anything
  else that needs a wider basis stay tables, unless a reviewer applies R5 or R8.
- `find` tests whether units matter, not what they compute. Its results point to where
  to look. They do not feed the generated program.
- A subject's program is a program in its own right, not a part cut out of the model. The
  source model is unchanged, and removing the units does not remove the subject cleanly.
