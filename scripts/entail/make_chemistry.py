"""Writes chemistry.entail.txt: the rules below, after a table of elements and tokens.

  python make_chemistry.py <periodic-table.csv> <tokenizer.json> <out.entail.txt>

The rules are the same for any model. The table is not: what a formula looks like
depends on how a tokenizer cuts it up. Qwen reads NaCl as Na|Cl from one place and
(N|a|Cl from another, so every token that spells chemistry has to be listed with
what it contributes. The table is too long to write by hand and easy to derive.

The CSV is PubChem's: https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV
Elements with no standard atomic weight are left out, so formulas using them
entail nothing. Weights are rounded to three decimal places.
"""
import csv
import json
import re
import sys

NO_WEIGHT = {"Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac"}


def elements(path):
    table = {}
    with open(path, encoding="utf8") as f:
        for row in csv.DictReader(f):
            z = int(row["AtomicNumber"])
            if z <= 92 and row["Symbol"] not in NO_WEIGHT:
                table[row["Symbol"]] = round(float(row["AtomicMass"]) * 1000)
    return table


def split(milli):
    return milli // 1000, milli % 1000


def code(letter):
    return ord(letter.upper()) - 64


def parse(text, table):
    """The atoms a token spells, and a capital left waiting for a lowercase letter."""
    atoms, i = [], 0
    while i < len(text):
        if i + 1 < len(text) and text[i + 1].islower():
            if text[i:i + 2] not in table:
                return None
            atoms.append(text[i:i + 2])
            i += 2
        elif text[i].isupper() and text[i].isascii():
            atoms.append(text[i])
            i += 1
        else:
            return None
    for atom in atoms[:-1]:
        if atom not in table:
            return None
    if atoms and atoms[-1] not in table and len(atoms[-1]) == 2:
        return None
    return atoms


def starters(table):
    return {s[0] for s in table}


def token_lines(vocabulary, table):
    lines = []
    openers = {"(": 1, "((": 2}
    caps = starters(table)
    for text in sorted(vocabulary):
        m = re.fullmatch(r"( ?)(\(*)([A-Za-z]+)", text)
        if not m:
            continue
        space, brackets, body = m.groups()
        if len(body) > 6 or (brackets and brackets not in openers) or not body[0].isupper():
            continue
        atoms = parse(body, table)
        if not atoms:
            continue
        last = atoms[-1]
        pending = last not in table
        if pending and last not in caps:
            continue
        pre = sum(table[a] for a in atoms[:-1])
        mass = 0 if pending else table[last]
        if pre >= 256000:  # BUFF and friends: a word, and too heavy for bf16 besides
            continue
        facts = [f"ok = {0 if pending else 1}"]
        if pre:
            facts += ["pw = %d" % split(pre)[0], "pfh = %d" % (split(pre)[1] // 32), "pfl = %d" % (split(pre)[1] % 32)]
        if mass:
            facts += ["lw = %d" % split(mass)[0], "lfh = %d" % (split(mass)[1] // 32), "lfl = %d" % (split(mass)[1] % 32), "atom = 1"]
        if len(last) == 1:
            facts.append(f"cap = {code(last)}")
        if pending:
            facts.append("pend = 1")
        if brackets:
            facts.append(f"opn = {openers[brackets]}")
        lines.append(f'token "{text}" means ' + ", ".join(facts))
    for c in sorted({s[1] for s in table if len(s) == 2}):
        lines.append(f'token "{c}" means low = {code(c)}')
    return lines


def pair_lines(table):
    lines, i = [], 0
    for symbol, mass in sorted(table.items()):
        if len(symbol) != 2:
            continue
        delta = mass - table.get(symbol[0], 0)
        whole, part = (delta // 1000, delta % 1000)
        lines.append(f"entail pr[{i}] when 32*capB + low == {32 * code(symbol[0]) + code(symbol[1])}   # {symbol}")
        lines.append(f"entail dw = {whole} when pr[{i}]")
        lines.append(f"entail df = {part} when pr[{i}]")
        lines.append(f"entail aw = {mass // 1000} when pr[{i}]")
        lines.append(f"entail af = {mass % 1000} when pr[{i}]")
        i += 1
    return i, lines


RULES = r'''
program chemistry

const W     = 16    # tokens in a species
const G     = 10    # tokens in a bracket
const WB    = 34    # tokens on one side of an equation
const SLOTS = 11    # answer tokens
const CAR   = 130   # thousandths carried into the whole number

# ─── 1. Tokens ──────────────────────────────────────────────────────────────
#
# Every token says what it adds to a mass, in two parts: whole grams and
# thousandths, because a threshold can only compare numbers below 131072.
#
#   pw pf   atoms that are complete: the S of SO
#   lw lf   the last atom, the one a subscript acts on: the O of SO
#   cap     the last atom's capital letter, if a lowercase one could follow
#   pend    that capital is no element by itself, as in the M of Mg
#   opn     brackets it opens     ok   it can stand in a formula as it is
#
# A host's embedding is bf16, which holds whole numbers exactly only up to 256.
# So thousandths arrive in two parts, and a let puts them together.

state pw, pfh, pfl, lw, lfh, lfl
state ok, atom, cap, pend, opn, low
state val, dig, gap, cl, tight, spaced, ans
state mk, ofk, massw, plus, arrow, lead, isw
state isBal, isUn

let pf = 32*pfh + pfl
let lf = 32*lfh + lfl

@TOKENS@

for d in 0 to 9
    token "{d}" means val = d, dig = 1, ans = 1
end
token "₁" means val = 1, dig = 1
token "₂" means val = 2, dig = 1
token "₃" means val = 3, dig = 1
token "₄" means val = 4, dig = 1
token " "    means gap = 1, ok = 1
token "(", " ("    means opn = 1, ok = 1
token "(("   means opn = 2, ok = 1
token ")"    means cl = 1, ok = 1
token ")="   means cl = 1, tight = 1
token "="    means tight = 1
token " ="   means spaced = 1
token " is"  means spaced = 1, isw = 1
token "."    means ans = 1
token " g"   means ans = 1
token "/mol" means ans = 1
token "M"    means pend = 1, mk = 1, cap = 13
token " M"   means mk = 1, cap = 13, pend = 1
token " of"  means ofk = 1
token " mass", "mass", " weight", " Mass", "Mass" means massw = 1
token "+", " +" means plus = 1
token "->", " ->", "→", " →" means arrow = 1
token "\n", "\n\n", ":", ": ", " equation", " reaction" means lead = 1
token " balanced", "balanced" means isBal = 1
token " un"  means isUn = 1

# ─── 2. Elements a tokenizer has cut in two ─────────────────────────────────
#
# A lowercase letter finishes the element its neighbour began: the a of (N|a
# turns nitrogen into sodium, so it adds the difference.

state capB, pendB, b1mk, b1of, b1plus, b1arrow, b1lead, b1mass
attend back 1
    capB <- cap
    pendB <- pend
    b1mk <- mk
    b1plus <- plus
    b1arrow <- arrow
    b1lead <- lead
    b1mass <- massw
    b1of <- ofk
end

const PAIRS = @PAIRS@
state pr[PAIRS], dw scale 1/8, df scale 1/64, aw scale 1/8, af scale 1/64

@PAIRLINES@

state fixed
for i in 0 to PAIRS-1
    entail fixed when pr[i]
end

# atomW, atomF: the atom that ends at this token, if one does.
let atomW = lw + aw
let atomF = lf + af
let hasAtom = atom + fixed

# ─── 3. Subscripts ──────────────────────────────────────────────────────────
#
# Each token adds only what it changes. In H12, the H says 1, the 1 says nothing
# more, and the 2 says the other 11: 9 times the digit before it, plus itself.

state a1w scale 1/8, a1f scale 1/64, a2w scale 1/8, a2f scale 1/64, has1, has2
state v1, d1, dg2
attend back 1
    a1w <- atomW
    a1f <- atomF
    has1 <- hasAtom
    v1 <- val
    d1 <- dig
end
attend back 2
    a2w <- atomW
    a2f <- atomF
    has2 <- hasAtom
    dg2 <- dig
end

state sub1, sub2
entail sub1 when dig, has1, val >= 1
state sub1B
attend back 1
    sub1B <- sub1
end
entail sub2 when dig, sub1B

state s1[10], t1[10], t2[10], sw scale 1/8, sf scale 1/64
for v in 2 to 9
    entail s1[v] when sub1, val >= v
    entail sw = a1w when s1[v]
    entail sf = a1f when s1[v]
end
for v in 1 to 9
    entail t1[v] when sub2, v1 >= v
    entail sw = 9*a2w when t1[v]
    entail sf = 9*a2f when t1[v]
    entail t2[v] when sub2, val >= v
    entail sw = a2w when t2[v]
    entail sf = a2f when t2[v]
end

# ─── 4. Where a species starts ──────────────────────────────────────────────
#
# After M, at its bracket. After "mass of", after + or ->, or after a new line.

state mof, mofB
entail mof when ofk, b1mass
attend back 1
    mofB <- mof
end

state start, molar, startB, bnd
entail bnd when b1plus
entail bnd when b1arrow
entail bnd when b1lead
entail bnd when mofB
entail molar when b1mk, opn >= 1
entail molar when mofB
entail start when molar
entail start when bnd, not molar
attend back 1
    startB <- start
end

# A digit at the front of a species is its coefficient, with a blank before it or not.
state gapStart, coef
entail gapStart when gap, start
state gapStartB
attend back 1
    gapStartB <- gapStart
end
entail coef when dig, start
entail coef when dig, gapStartB

# ─── 5. Brackets ────────────────────────────────────────────────────────────
#
# A digit straight after ")" repeats what is inside, which it finds by looking
# for the nearest opening bracket. Nested brackets are refused.

let c1w = pw + lw + dw + sw
let c1f = pf + lf + df + sf

state gw[G + 1] scale 1/8, gf[G + 1] scale 1/64, go[G + 1], gc[G + 1], clB
for k in 1 to G
    attend back k
        gw[k] <- c1w
        gf[k] <- c1f
        go[k] <- opn
        gc[k] <- cl
    end
end
attend back 1
    clB <- cl
end

state gd, gsel[G + 1], inW scale 1/8, inF scale 1/64, gok
entail gd when dig, clB, val >= 1
for o in 2 to G
    if o == 2
        entail gsel[o] when gd, go[o] >= 1
    else
        entail gsel[o] when gd, go[o] >= 1, (sum k in 2 to o-1 : go[k] + gc[k]) <= 0
    end
    entail inW = (sum k in 2 to o : gw[k]) when gsel[o]
    entail inF = (sum k in 2 to o : gf[k]) when gsel[o]
    entail gok when gsel[o]
end

state gm[10], c2w scale 1/8, c2f scale 1/64
for v in 2 to 9
    entail gm[v] when gok, val >= v
    entail c2w = inW when gm[v]
    entail c2f = inF when gm[v]
end

# ─── 6. Summing a species, from the token after it ──────────────────────────

let ctw = c1w + c2w
let ctf = c1f + c2f
let okv = ok + fixed + sub1 + sub2 + gok + coef
state pendFix
entail pendFix when fixed, pendB
let okAll = okv + pendFix
let cv = coef

state xw[W + 1] scale 1/8, xf[W + 1] scale 1/64, xo[W + 1], xs[W + 1], xd[W + 1], xc[W + 1], xm[W + 1]
for k in 1 to W
    attend back k
        xw[k] <- ctw
        xf[k] <- ctf
        xo[k] <- okAll
        xs[k] <- start
        xd[k] <- opn - cl
        xc[k] <- val
        xm[k] <- molar
    end
end

state xq[W + 1]
for k in 1 to W
    attend back k
        xq[k] <- coef
    end
end

# An end: the "=" of a formula, the "is" of a sentence, or a + or -> in an equation.
state isEnd
entail isEnd when tight
entail isEnd when spaced
entail isEnd when plus
entail isEnd when arrow

# span[s]: the species began s tokens back, and every token in it can stand there.
state span[W + 1], sv, sm, spW scale 1/8, spF scale 1/64, cfv
for s in 1 to W
    if s == 1
        entail span[s] when isEnd, xs[s], xo[1] >= 1, xd[1] - cl == 0
    else
        entail span[s] when isEnd, xs[s], (sum k in 1 to s-1 : xs[k]) <= 0, (sum k in 1 to s : xo[k]) >= s, (sum k in 1 to s : xd[k]) - cl == 0
    end
    entail sv when span[s]
    entail sm = xm[s] when span[s]
    entail spW = (sum k in 1 to s : xw[k]) when span[s]
    entail spF = (sum k in 1 to s : xf[k]) when span[s]
end

# The coefficient: the one value that was marked as one.
state cq[W + 1]
for s in 1 to W
    entail cfv = xc[s] when span[s], xq[s]
    if s >= 2
        entail cfv = xc[s-1] when span[s], xq[s-1]
    end
end

state cm[10], yw scale 1/8, yf scale 1/64
for v in 2 to 9
    entail cm[v] when cfv >= v
    entail yw = spW when cm[v]
    entail yf = spF when cm[v]
end
let totW = spW + yw
let totF = spF + yf

# ─── 7. A molar mass, as digits ─────────────────────────────────────────────

state doM, carry
entail doM when sv, sm, tight
entail doM when sv, sm, spaced
for k in 1 to CAR
    entail carry when totF >= 1000*k
end
let fracT = totF - 1000*carry
let wholeT = totW + carry

state e3, e2, e1, f2, f1
for k in 1 to 9
    entail e3 when wholeT >= 1000*k
end
let r3 = wholeT - 1000*e3
for k in 1 to 9
    entail e2 when r3 >= 100*k
end
let r2 = r3 - 100*e2
for k in 1 to 9
    entail e1 when r2 >= 10*k
end
let r1 = r2 - 10*e1
for k in 1 to 9
    entail f2 when fracT >= 100*k
end
let q2 = fracT - 100*f2
for k in 1 to 9
    entail f1 when q2 >= 10*k
end
let q1 = q2 - 10*f1

const C_POINT = 11
const C_G = 12
const C_MOL = 13
const C_END = 14

let dgt[3] = e3
let dgt[2] = e2
let dgt[1] = e1
let dgt[0] = r1

state hi3, hi2, hi1, hi0
entail hi3 when doM, wholeT >= 1000
entail hi2 when doM, wholeT >= 100, wholeT <= 999
entail hi1 when doM, wholeT >= 10, wholeT <= 99
entail hi0 when doM, wholeT <= 9

state code[SLOTS]
def spell(sel, t)
    for s in 0 to t
        entail code[s] = dgt[t-s] + 1 when sel
    end
    entail code[t+1] = C_POINT when sel
    entail code[t+2] = f2 + 1 when sel
    entail code[t+3] = f1 + 1 when sel
    entail code[t+4] = q1 + 1 when sel
    entail code[t+5] = C_G when sel
    entail code[t+6] = C_MOL when sel
    entail code[t+7] = C_END when sel
end
call spell(hi3, 3)
call spell(hi2, 2)
call spell(hi1, 1)
call spell(hi0, 0)

# ─── 8. Conservation of mass ────────────────────────────────────────────────
#
# Each end carries the total of its side so far, by reading the end before it.
# Four species a side. An end that follows anything but + starts a new total.

state boundAt[W + 2], plusAt[W + 2], bs[W + 2]
state yP[W + 2], yA[W + 2]
for k in 1 to W + 1
    attend back k
        yP[k] <- plus
        yA[k] <- arrow
    end
end
state pAt[W + 2], here
for s in 1 to W
    entail pAt[s+1] when span[s], yP[s+1] >= 1
    entail here = s + 1 when span[s]
end

def side(from, to)
    state r{to}w scale 1/8, r{to}f scale 1/64, r{to}n, r{to}b
    attend back by pAt
        r{to}w <- t{from}w
        r{to}f <- t{from}f
        r{to}n <- t{from}n
        r{to}b <- t{from}b
    end
    state c{to}w scale 1/8, c{to}f scale 1/64, c{to}n, c{to}b, pp{to}
    entail pp{to} when plusBefore
    entail c{to}w = r{to}w when pp{to}
    entail c{to}f = r{to}f when pp{to}
    entail c{to}n = r{to}n when pp{to}
    entail c{to}b = r{to}b when pp{to}
    let t{to}w = totW + c{to}w
    let t{to}f = totF + c{to}f
    let t{to}n = here + c{to}n
    let t{to}b = 1 - sv + c{to}b
end

state plusBefore
for s in 2 to W + 1
    entail plusBefore when pAt[s]
end

let t0w = totW
let t0f = totF
let t0n = here
let t0b = 1 - sv
call side(0, 1)
call side(1, 2)
call side(2, 3)

# At the "is": the arrow is t3n tokens back, and it holds the left-hand side.
state fin, arrowAt[WB + 1]
entail fin when isw, sv, not sm
for c in 2 to WB
    entail arrowAt[c] when fin, t3n == c
end

state lW scale 1/8, lF scale 1/64, lB, lA
attend back by arrowAt
    lW <- t3w
    lF <- t3f
    lB <- t3b
    lA <- arrow
end

state judged, same
entail judged when fin, lA, t3b + lB <= 0
entail same when judged, 1000*(t3w - lW) + t3f - lF == 0

# ─── 9. Writing it out ──────────────────────────────────────────────────────

state et[SLOTS + 2], es[SLOTS + 2], wa[SLOTS + 2], gp[SLOTS + 2]
for k in 1 to SLOTS + 1
    attend back k
        et[k] <- tight
        es[k] <- spaced
        wa[k] <- ans
        gp[k] <- gap
    end
end
let ansF[0] = ans
let gpF[0] = gap
for k in 1 to SLOTS + 1
    let ansF[k] = wa[k]
    let gpF[k] = gp[k]
end

state from[SLOTS + 2], slot[SLOTS]
entail from[0] when tight
entail from[0] when spaced
entail slot[0] when tight
entail from[1] when et[1], ans
entail from[1] when et[1], gap
entail from[1] when es[1], gap
entail slot[1] when et[1], ans
entail slot[0] when et[1], gap
entail slot[0] when es[1], gap
for c in 2 to SLOTS
    entail from[c] when et[c], (sum i in 0 to c-1 : ansF[i]) >= c
    entail from[c] when et[c], (sum i in 0 to c-2 : ansF[i]) >= c-1, gpF[c-1]
    entail from[c] when es[c], (sum i in 0 to c-2 : ansF[i]) >= c-1, gpF[c-1]
    if c < SLOTS
        entail slot[c] when et[c], (sum i in 0 to c-1 : ansF[i]) >= c
    end
    entail slot[c-1] when et[c], (sum i in 0 to c-2 : ansF[i]) >= c-1, gpF[c-1]
    entail slot[c-1] when es[c], (sum i in 0 to c-2 : ansF[i]) >= c-1, gpF[c-1]
end

state seen[SLOTS]
attend back by from
    for s in 0 to SLOTS-1
        seen[s] <- code[s]
    end
end

state say
for s in 0 to SLOTS-1
    entail say = seen[s] when slot[s]
end

# ─── 10. Output ─────────────────────────────────────────────────────────────

state o[C_END + 1], blank, sayBal, sayUn, unDone, balDone
state judgedB, sameB
attend back 1
    judgedB <- judged
    sameB <- same
end
for d in 1 to C_END
    entail o[d] when say == d
end
entail blank when spaced, seen[0] >= 1
entail sayBal when same
entail sayUn when judged, not same
entail unDone when isUn, judgedB
entail balDone when isBal, sameB
state balAfterUn, unB
attend back 1
    unB <- isUn
end
entail balAfterUn when judgedB, isBal
entail balAfterUn when unB, isBal

for d in 0 to 9
    emit "{d}" when o[d+1]
end
emit "." when o[C_POINT]
emit " g" when o[C_G]
emit "/mol" when o[C_MOL]
emit " " when blank
emit " balanced" when sayBal
emit " un" when sayUn
emit "balanced" when unDone
emit end when o[C_END]
emit end when balAfterUn
'''

HEADER = """# ═══════════════════════════════════════════════════════════════════════════
#  chemistry.entail — molar mass and the conservation of mass
# ═══════════════════════════════════════════════════════════════════════════
#
#  Generated by make_chemistry.py. Edit the rules there: the token table below
#  belongs to one tokenizer.
#
#  A chemical formula is a sum. Every atom has a mass, a subscript repeats the atom
#  before it, and a bracket repeats everything inside. So a molar mass is exact:
#
#     M(C6H12O6)=                    180.156 g/mol
#     The molar mass of Ca(OH)2 is   74.094 g/mol
#
#  And a reaction cannot create mass, so an equation can be checked:
#
#     2H2 + O2 -> 2H2O is            balanced
#     H2 + O2 -> H2O is              unbalanced
#
#  What is defined: formulas of known elements with subscripts up to 99 and one
#  level of brackets. Equations have one-digit coefficients and up to four species
#  a side, and the first species needs a new line or ":" in front of it. The check
#  compares total mass, not each element, so it would pass an equation that swapped
#  one element for another of exactly equal weight. Anything else entails nothing.
"""


def main():
    table = elements(sys.argv[1])
    with open(sys.argv[2], encoding="utf8") as f:
        tokenizer = json.load(f)
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(sys.argv[2])
    vocabulary = {t.decode([i]) for i in tokenizer["model"]["vocab"].values()}
    skip = {" M", "M", " H2O"}
    lines = [x for x in token_lines(vocabulary - skip, table)]
    pairs, pair_text = pair_lines(table)
    text = HEADER + RULES.replace("@TOKENS@", "\n".join(lines)).replace("@PAIRS@", str(pairs)).replace("@PAIRLINES@", "\n".join(pair_text))
    with open(sys.argv[3], "w", encoding="utf8", newline="\n") as f:
        f.write(text)
    print(f"{len(table)} elements, {len(lines)} tokens, {pairs} two-letter elements")


if __name__ == "__main__":
    main()
