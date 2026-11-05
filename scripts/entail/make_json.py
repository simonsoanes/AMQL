"""Writes json.entail.txt: the rules below, after a table of tokens and token sets.

  python make_json.py <tokenizer.json> <out.entail.txt>

A tokenizer merges JSON's punctuation: ":[ is the end of a string, a colon and an opening
bracket, and what it means depends on whether a string was open. So each token is read
here by a small scanner, once from outside a string and once from inside, and its effect
is listed. The same scanner decides which tokens may follow in each state, so the verdict
and the guard cannot disagree.
"""
import json
import sys

# Events, in the order the text gives them.
LB, LK, CM, CL, SO, SC, CB, LIT, NUM = 1, 2, 3, 4, 5, 6, 7, 8, 9
# Number characters.
N_MINUS, N_DIGIT, N_ZERO, N_DOT, N_EXP, N_SIGN = 1, 2, 3, 4, 5, 6

FOLLOWS = {
    LB: {SO, CB}, LK: {LB, LK, SO, LIT, NUM, CB}, CM: {LB, LK, SO, LIT, NUM}, CL: {LB, LK, SO, LIT, NUM},
    SO: {SC}, SC: {CM, CL, CB}, CB: {CM, CB}, LIT: {CM, CB}, NUM: {CM, CB},
}
NEXT_CHAR = {
    N_MINUS: {N_DIGIT, N_ZERO}, N_DIGIT: {N_DIGIT, N_ZERO, N_DOT, N_EXP}, N_ZERO: {N_DIGIT, N_ZERO, N_DOT, N_EXP},
    N_DOT: {N_DIGIT, N_ZERO}, N_EXP: {N_DIGIT, N_ZERO, N_SIGN}, N_SIGN: {N_DIGIT, N_ZERO},
}
VALUE = {LB, LK, SO, LIT, NUM}

# The states a guard knows, and the first events each allows.
#   start  a value, with nothing around it      valo vala  a value, in an object or array
#   kc     a key or }                           vc         a value or ]
#   key    a key                                colon
#   after  a comma or the closer                num        more of a number, or what follows
#   numi   a number that needs another digit    ink inv    inside a key or a value
#   done   the text is complete
ALLOWED = dict(
    start=VALUE, valo=VALUE, vala=VALUE, kc={SO, CB}, vc=VALUE | {CB}, key={SO}, colon={CL},
    aftero={CM, CB}, aftera={CM, CB}, numo={CM, CB, NUM}, numa={CM, CB, NUM}, num0={NUM}, numi={NUM},
    ink={SC}, invo={SC}, inva={SC}, inv0={SC}, done=set(),
)
INSIDE = ("ink", "invo", "inva", "inv0")


class Read:
    """What a token does from one starting mode."""

    def __init__(self, text, inside):
        self.events, self.closes, self.opens = [], [], []
        self.first_num = self.last_num = 0
        self.quotes = 0
        self.bad = False           # no JSON text could hold this
        self.unsure = False        # it ends part-way through an escape
        self.comma_value = False   # a comma, then something that cannot be a key
        self.scan(text, inside)

    def event(self, e):
        if self.events and e not in FOLLOWS[self.events[-1]]:
            self.bad = True
        if self.events and self.events[-1] == CM and e != SO:
            self.comma_value = True
        self.events.append(e)

    def number(self, k):
        if self.events and self.events[-1] == NUM:
            if k not in NEXT_CHAR[self.last_num]:
                self.bad = True
        else:
            self.event(NUM)
            self.first_num = k
        self.last_num = k

    def scan(self, text, inside):
        i, n = 0, len(text)
        while i < n:
            c = text[i]
            if inside:
                if c == "\\":
                    if i + 1 >= n:
                        self.unsure = True
                    elif text[i + 1] not in '"\\/bfnrtu':
                        self.bad = True
                    i += 2
                    continue
                if c == '"':
                    inside = False
                    self.quotes += 1
                    self.event(SC)
                elif ord(c) < 32:
                    self.bad = True
                i += 1
            elif c in " \n\t\r":
                i += 1
            elif c in "{[":
                self.event(LB if c == "{" else LK)
                self.opens.append(1 if c == "{" else 2)
                i += 1
            elif c in "}]":
                kind = 1 if c == "}" else 2
                self.event(CB)
                if self.opens:
                    if self.opens.pop() != kind:
                        self.bad = True
                else:
                    self.closes.append(kind)
                i += 1
            elif c == ",":
                self.event(CM)
                i += 1
            elif c == ":":
                self.event(CL)
                i += 1
            elif c == '"':
                self.event(SO)
                self.quotes += 1
                inside = True
                i += 1
            elif text.startswith(("true", "null"), i):
                self.event(LIT)
                i += 4
            elif text.startswith("false", i):
                self.event(LIT)
                i += 5
            elif c in "-0123456789.eE+":
                if c == "-":
                    self.number(N_SIGN if self.events[-1:] == [NUM] and self.last_num == N_EXP else N_MINUS)
                elif c == "+":
                    self.number(N_SIGN)
                elif c == ".":
                    self.number(N_DOT)
                elif c in "eE":
                    self.number(N_EXP)
                else:
                    self.number(N_ZERO if c == "0" else N_DIGIT)
                i += 1
            else:
                self.bad = True
                i += 1
        if len(self.closes) > 3 or len(self.opens) > 3:
            self.bad = True

    def lanes(self, m):
        """The token's lanes in this mode, leaving out zeros."""
        e = self.events
        pairs = dict(f=e[0] if e else 0, s=e[1] if len(e) > 1 else 0,
                     a=e[-1] if e else 0, b=e[-2] if len(e) > 1 else 0, c=e[-3] if len(e) > 2 else 0,
                     n=min(len(e), 3), d=len(self.opens) - len(self.closes), v=int(self.comma_value),
                     h=self.first_num, k=self.last_num)
        for i, x in enumerate(self.closes):
            pairs[f"x{i + 1}"] = x
        for i, x in enumerate(self.opens):
            pairs[f"y{i + 1}"] = x
        return {m + name: value for name, value in pairs.items() if value}


def fits(token, state):
    """Whether a token can follow in this state, knowing only the innermost bracket."""
    r = Read(token, state in INSIDE)
    if r.bad or r.unsure:
        return False
    top = 1 if state in ("valo", "aftero", "numo", "invo", "kc", "key", "ink", "colon") else \
        2 if state in ("vala", "aftera", "numa", "inva", "vc") else 0
    if len(r.closes) > (1 if top else 0) or (r.closes and r.closes[0] != top):
        return False
    e = r.events
    if not e:
        return state != "numi"
    if e[0] not in ALLOWED[state]:
        return False
    if r.comma_value and (r.closes or top != 2):
        return False
    if e[0] == NUM:
        if state.startswith("num"):
            legal = (N_DIGIT, N_ZERO, N_SIGN) if state == "numi" else (N_DIGIT, N_ZERO, N_DOT, N_EXP)
        else:
            legal = (N_MINUS, N_DIGIT, N_ZERO)
        if r.first_num not in legal:
            return False
    if state == "ink" and len(e) > 1 and e[1] != CL:
        return False
    if state in ("invo", "inva", "inv0") and len(e) > 1 and e[1] == CL:
        return False
    # a key opened and closed within the token must meet its colon there too
    if state in ("kc", "key") and len(e) >= 3 and e[0] == SO and e[2] != CL:
        return False
    return True


def quoted(token):
    """A token as an Entail string, or None if the language can't write it."""
    if "\ufffd" in token or any(ord(c) < 32 and c not in "\n\t\r" for c in token):
        return None
    out = token.replace("\\", "\\e").replace('"', "\\q").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r").replace("{", "\\{")
    return f'"{out}"'


RULES = r"""# ═══════════════════════════════════════════════════════════════════════════
#  json.entail — the shape of JSON
# ═══════════════════════════════════════════════════════════════════════════
#
#  Generated by make_json.py. Edit the rules there.
#
#  After a marker, "JSON:" or "json" and a new line, every token is read against
#  the grammar of JSON. The program then does two things.
#
#    A verdict.   JSON: {"a": [1, 2}
#                 Valid:                      no
#
#    A guard.     While a text is being written, tokens that would break the
#                 grammar are ruled out, so the host can only close what it
#                 opened, in order, and can't leave a string or a comma wrong.
#
#  A bracket may lie far behind the token it matches, so no fixed window can see
#  it. Instead each token combines with what lies 1 back, then 2, 4, 8 and so on:
#  nine steps reach 512 tokens. Strings come first, because a brace in a string
#  means nothing; then depth; then what stands open at each depth.
#
#  Brackets go 8 deep, and a text may run to 512 tokens. Past that the program
#  says nothing. It checks the shape of a text, not what it says.

program json

const D  = 8     # depth of brackets
const NS = 9     # steps of doubling: 2^9 = 512 tokens
const QM = 640   # the most quotes in a text

const E_CM  = 3
const E_CL  = 4
const E_SO  = 5
const E_SC  = 6
const E_NUM = 9

const N_DIGIT = 2
const N_ZERO  = 3
const N_DOT   = 4
const N_EXP   = 5
const N_SIGN  = 6

const T_WORD  = 1    # names JSON, so a marker may follow
const T_END   = 2    # a line end, which completes a marker
const T_VALID = 3
const T_COLON = 4    # completes a marker or "Valid:"

# ─── 1. Tokens ──────────────────────────────────────────────────────────────
#
# Lanes starting u are the token read from outside a string, and w from inside.
#   f s    first and second events        a b c   last, before it, before that
#   n      events, up to 3                d       change in depth
#   x1..   closers    y1..  openers       v       a comma, then a non-key
#   h k    first and last number characters
# okO: legal outside a string. noI: illegal inside one. un: stops inside an escape.
# q: quotes that count. lw: white space comes first.
# ec: after a lone backslash, 1 if this token starts a legal escape, 2 if it starts
# with a quote or backslash, which would change how it reads.

state uf, us, ua, ub, uc, un_, ud, uv, uh, uk, ux1, ux2, ux3, uy1, uy2, uy3
state wf, ws, wa, wb, wc, wn, wd, wv, wh, wk, wx1, wx2, wx3, wy1, wy2, wy3
state q, okO, noI, un, tag, ec, lw, blank

@TOKENS@

# ─── 2. Where a text starts ─────────────────────────────────────────────────

state pTag
attend back 1
    pTag <- tag
end
state mark, atValid
entail mark when pTag == T_WORD, tag == T_END
entail mark when pTag == T_WORD, tag == T_COLON
entail atValid when pTag == T_VALID, tag == T_COLON

# ─── 3. Strings ─────────────────────────────────────────────────────────────
#
# cq[j]: quotes in the last 2^j tokens, back to the marker.
# rs[j]: the marker is among them.

state cq[NS + 1] scale 1/4, rs[NS + 1], pq[NS] scale 1/4, pr[NS]
entail cq[0] = q
entail rs[0] when mark
for j in 0 to NS - 1
    attend back 2^j
        pq[j] <- cq[j]
        pr[j] <- rs[j]
    end
    entail cq[j+1] = cq[j] when rs[j]
    entail cq[j+1] = cq[j] + pq[j] when not rs[j]
    entail rs[j+1] when rs[j]
    entail rs[j+1] when not rs[j], pr[j]
end
let active = rs[NS]

let before = cq[NS] - q
state qt[QM + 1]
for t in 1 to QM
    entail qt[t] when before >= t
end
state ins
entail ins = (sum t in 1 to QM : (2*(t - 2*floor(t/2)) - 1) * qt[t])

# The token, in the mode it was entered.
state tF, tS, tA, tB, tC, tN, tD, tV, tH, tK, tX1, tX2, tX3, tY1, tY2, tY3
def pick(out, u, w)
    entail out = u when not ins
    entail out = w when ins
end
call pick(tF, uf, wf)
call pick(tS, us, ws)
call pick(tA, ua, wa)
call pick(tB, ub, wb)
call pick(tC, uc, wc)
call pick(tN, un_, wn)
call pick(tD, ud, wd)
call pick(tV, uv, wv)
call pick(tH, uh, wh)
call pick(tK, uk, wk)
call pick(tX1, ux1, wx1)
call pick(tX2, ux2, wx2)
call pick(tX3, ux3, wx3)
call pick(tY1, uy1, wy1)
call pick(tY2, uy2, wy2)
call pick(tY3, uy3, wy3)

# ─── 4. Depth ───────────────────────────────────────────────────────────────

state cd[NS + 1] scale 1/4, pd[NS] scale 1/4
entail cd[0] = tD when not mark
for j in 0 to NS - 1
    attend back 2^j
        pd[j] <- cd[j]
    end
    entail cd[j+1] = cd[j] when rs[j]
    entail cd[j+1] = cd[j] + pd[j] when not rs[j]
end
let dep = cd[NS]

state c1f, c2f, c3f
entail c1f when tX1 >= 1
entail c2f when tX2 >= 1
entail c3f when tX3 >= 1
let depOpen = dep - tD - c1f - c2f - c3f    # where this token's first opener lands

state lv[D + 3]
for l in 0 to D - 1
    entail lv[l] when depOpen == l
end

# ─── 5. What stands open, and the last events ───────────────────────────────
#
# Each lane keeps the latest value written. n1, n2, n3: at least that many
# events are known here, so anything older than the third can be dropped.
# A marker counts as three empty events and clears every level.

state sa[NS + 1], sb[NS + 1], sc[NS + 1], n1[NS + 1], n2[NS + 1], n3[NS + 1]
state sq[NS + 1], hq[NS + 1], ty[NS + 1][D], hy[NS + 1][D]
state pa[NS], pb[NS], pc[NS], p1[NS], p2[NS], p3[NS], pst[NS], phs[NS], pt[NS][D], ph[NS][D]

entail sa[0] = tA when not mark
entail sb[0] = tB when not mark
entail sc[0] = tC when not mark
entail n1[0] when tN >= 1
entail n1[0] when mark, tN <= 0
entail n2[0] when tN >= 2
entail n2[0] when mark, tN <= 1
entail n3[0] when tN >= 3
entail n3[0] when mark, tN <= 2

# The stage of a number: 0 whole, 1 after a point, 2 after an exponent.
state numLast
entail numLast when tA == E_NUM
entail sq[0] = 1 when numLast, tK == N_DOT
entail sq[0] = 2 when numLast, tK == N_EXP
entail sq[0] = 2 when numLast, tK == N_SIGN
entail hq[0] when numLast, tK >= N_DOT
entail hq[0] when numLast, tK <= 1
entail hq[0] when not numLast, tN >= 1
entail hq[0] when not numLast, tN <= 0, mark

for l in 0 to D - 1
    entail ty[0][l] = tY1 when lv[l], not mark
    entail hy[0][l] when lv[l], tY1 >= 1
    if l >= 1
        entail ty[0][l] = tY2 when lv[l-1], not mark
        entail hy[0][l] when lv[l-1], tY2 >= 1
    end
    if l >= 2
        entail ty[0][l] = tY3 when lv[l-2], not mark
        entail hy[0][l] when lv[l-2], tY3 >= 1
    end
    entail hy[0][l] when mark, (tY1 + tY2 + tY3) <= 0
end

for j in 0 to NS - 1
    attend back 2^j
        pa[j] <- sa[j]
        pb[j] <- sb[j]
        pc[j] <- sc[j]
        p1[j] <- n1[j]
        p2[j] <- n2[j]
        p3[j] <- n3[j]
        pst[j] <- sq[j]
        phs[j] <- hq[j]
        for l in 0 to D - 1
            pt[j][l] <- ty[j][l]
            ph[j][l] <- hy[j][l]
        end
    end
    entail sa[j+1] = sa[j] when n1[j]
    entail sa[j+1] = pa[j] when not n1[j]
    entail sb[j+1] = sb[j] when n2[j]
    entail sb[j+1] = pa[j] when n1[j], not n2[j]
    entail sb[j+1] = pb[j] when not n1[j]
    entail sc[j+1] = sc[j] when n3[j]
    entail sc[j+1] = pa[j] when n2[j], not n3[j]
    entail sc[j+1] = pb[j] when n1[j], not n2[j]
    entail sc[j+1] = pc[j] when not n1[j]
    entail n1[j+1] when n1[j]
    entail n1[j+1] when not n1[j], p1[j]
    entail n2[j+1] when n2[j]
    entail n2[j+1] when n1[j], not n2[j], p1[j]
    entail n2[j+1] when not n1[j], p2[j]
    entail n3[j+1] when n3[j]
    entail n3[j+1] when n2[j], not n3[j], p1[j]
    entail n3[j+1] when n1[j], not n2[j], p2[j]
    entail n3[j+1] when not n1[j], p3[j]
    entail sq[j+1] = sq[j] when hq[j]
    entail sq[j+1] = pst[j] when not hq[j]
    entail hq[j+1] when hq[j]
    entail hq[j+1] when not hq[j], phs[j]
    for l in 0 to D - 1
        entail ty[j+1][l] = ty[j][l] when hy[j][l]
        entail ty[j+1][l] = pt[j][l] when not hy[j][l]
        entail hy[j+1][l] when hy[j][l]
        entail hy[j+1][l] when not hy[j][l], ph[j][l]
    end
end

let A = sa[NS]
let B = sb[NS]
let C = sc[NS]
let stage = sq[NS]

# ─── 6. The state after this token ──────────────────────────────────────────

state dl[D + 2], topv, tO, tR
for d in 0 to D
    entail dl[d] when dep == d
end
for d in 1 to D
    entail topv = ty[NS][d-1] when dl[d]
end
entail tO when topv == 1
entail tR when topv == 2

state keyOpen, keyDone, numNow, numFull
entail keyOpen when A == E_SO, B == 1
entail keyOpen when A == E_SO, B == E_CM, tO
entail keyDone when A == E_SC, B == E_SO, C == 1
entail keyDone when A == E_SC, B == E_SO, C == E_CM, tO
entail numNow when tA == E_NUM, tN >= 1
entail numFull when numNow, tK >= N_DIGIT, tK <= N_ZERO

state vEnd
entail vEnd when A == 7
entail vEnd when A == 8
entail vEnd when A == E_SC, not keyDone
entail vEnd when A == E_NUM, not numNow

state g_start, g_valo, g_vala, g_kc, g_vc, g_key, g_colon, g_aftero, g_aftera
state g_numo, g_numa, g_num0, g_numi, g_ink, g_invo, g_inva, g_inv0, g_done
entail g_start when A == 0
entail g_kc when A == 1
entail g_vc when A == 2
entail g_key when A == E_CM, tO
entail g_vala when A == E_CM, tR
entail g_valo when A == E_CL
entail g_colon when keyDone
entail g_aftero when vEnd, tO
entail g_aftera when vEnd, tR
entail g_done when vEnd, dl[0]
entail g_numo when numFull, tO
entail g_numa when numFull, tR
entail g_num0 when numFull, dl[0]
entail g_numi when numNow, not numFull
entail g_ink when keyOpen
entail g_invo when A == E_SO, not keyOpen, tO
entail g_inva when A == E_SO, not keyOpen, tR
entail g_inv0 when A == E_SO, dl[0]

# ─── 7. Whether this token was allowed ──────────────────────────────────────

state p_start, p_valo, p_vala, p_kc, p_vc, p_key, p_colon, p_aftero, p_aftera
state p_numo, p_numa, p_num0, p_numi, p_ink, p_invo, p_inva, p_inv0, p_done
state pDl[D + 2], pTy[D], pK, pSt
attend back 1
@PREV@
    for d in 0 to D
        pDl[d] <- dl[d]
    end
    for l in 0 to D - 1
        pTy[l] <- ty[NS][l]
    end
    pK <- tK
    pSt <- stage
end
state ppK
attend back 2
    ppK <- tK
end

state m1, m2, m3
for d in 1 to D
    entail m1 = pTy[d-1] when pDl[d]
end
for d in 2 to D
    entail m2 = pTy[d-2] when pDl[d]
end
for d in 3 to D
    entail m3 = pTy[d-3] when pDl[d]
end

state fault, pUn
attend back 1
    pUn <- un
end
entail fault when not ins, okO <= 0, not mark
entail fault when ins, pUn >= 1, ec <= 0
entail fault when ins, noI >= 1

@LEGAL@

# A key is followed by its colon, and a value is not.
entail fault when p_ink, tS >= 1, tS <= 3
entail fault when p_ink, tS >= 5
entail fault when p_invo, tS == E_CL
entail fault when p_inva, tS == E_CL
entail fault when p_inv0, tS == E_CL

# A closer meets the bracket it closes.
entail fault when c1f, tX1 - m1 >= 1
entail fault when c1f, tX1 - m1 <= -1
entail fault when c2f, tX2 - m2 >= 1
entail fault when c2f, tX2 - m2 <= -1
entail fault when c3f, tX3 - m3 >= 1
entail fault when c3f, tX3 - m3 <= -1
entail fault when dep >= D + 1
entail fault when dep <= -1

# A comma followed by anything but a key belongs to an array.
entail fault when tV >= 1, not c1f, m1 <= 1
entail fault when tV >= 1, c1f, not c2f, m2 <= 1
entail fault when tV >= 1, c2f, m3 <= 1

# A number, one character at a time.
let pNum = p_numo + p_numa + p_num0 + p_numi
state curNum
entail curNum when tF == E_NUM
@NUMBER@
entail fault when curNum, pNum >= 1, tH == N_DOT, pSt >= 1
entail fault when curNum, pNum >= 1, tH == N_EXP, pSt == 2
entail fault when curNum, pNum >= 1, pK == N_ZERO, tH >= N_DIGIT, tH <= N_ZERO, pSt <= 0, ppK <= 1
entail fault when p_numi, not curNum
entail fault when curNum, pNum >= 1, lw

state cf[NS + 1] scale 1/4, cu[NS + 1] scale 1/4, pf[NS] scale 1/4, pu[NS] scale 1/4
entail cf[0] = fault when not mark
entail cu[0] = 1 when ins, pUn >= 1, ec >= 2
for j in 0 to NS - 1
    attend back 2^j
        pf[j] <- cf[j]
        pu[j] <- cu[j]
    end
    entail cf[j+1] = cf[j] when rs[j]
    entail cf[j+1] = cf[j] + pf[j] when not rs[j]
    entail cu[j+1] = cu[j] when rs[j]
    entail cu[j+1] = cu[j] + pu[j] when not rs[j]
end

# ─── 8. The verdict ─────────────────────────────────────────────────────────
#
# "Valid:" is not part of the text, so the verdict reads two tokens back.

state clean, sure
entail clean when active, cf[NS] <= 0, g_done + g_num0 >= 1
entail sure when active, cu[NS] <= 0

state vClean, vSure
attend back 2
    vClean <- clean
    vSure <- sure
end

state oYes, oNo, oEnd, said
entail oYes when atValid, vSure, vClean
entail oNo when atValid, vSure, not vClean
entail said when oYes
entail said when oNo
state wasSaid
attend back 1
    wasSaid <- said
end
entail oEnd when wasSaid
emit " yes" when oYes
emit " no" when oNo
emit end when oEnd

# ─── 9. The guard ───────────────────────────────────────────────────────────
#
# A clean text that is still open can only go on in ways that keep it clean.

state guard
entail guard when active, cf[NS] <= 0, cu[NS] <= 0, not atValid

@GUARD@

# Blank space is always legal between tokens, so a host with nothing better to say can
# say it for ever. Two blanks in a row are enough for any layout.
state pBlank, idle
attend back 1
    pBlank <- blank
end
entail idle when guard, blank, pBlank, not ins
forbid @BLANKS@ when idle
"""

PREV = "\n".join(f"    p_{s} <- g_{s}" for s in ALLOWED)


def legal_lines():
    lines = []
    for state, allowed in ALLOWED.items():
        for e in range(1, 10):
            if e not in allowed:
                lines.append(f"entail fault when p_{state}, tF == {e}")
    return lines


def number_lines():
    lines = [f"entail fault when curNum, p_{s}, tH >= N_DOT" for s in ("start", "valo", "vala", "vc")]
    for previous, following in NEXT_CHAR.items():
        for c in range(1, 7):
            if c not in following:
                lines.append(f"entail fault when curNum, pNum >= 1, pK == {previous}, tH == {c}")
    return lines


def main():
    from tokenizers import Tokenizer
    with open(sys.argv[1], encoding="utf8") as f:
        vocabulary = json.load(f)["model"]["vocab"]
    t = Tokenizer.from_file(sys.argv[1])
    words = sorted({t.decode([i]) for i in vocabulary.values()})
    tags = {"json": 1, " json": 1, "JSON": 1, " JSON": 1, "\n": 2, " Valid": 3, "Valid": 3, ":": 4}
    table, blanks, count = [], [], 0
    for w in words:
        text = quoted(w)
        if text is None or not w:
            continue
        out, within = Read(w, False), Read(w, True)
        lanes = {}
        if not out.bad:
            lanes["okO"] = 1
            lanes.update({("un_" if k == "un" else k): v for k, v in out.lanes("u").items()})
        if not within.bad:
            lanes.update(within.lanes("w"))
        else:
            lanes["noI"] = 1
        if within.unsure:
            lanes["un"] = 1
        if within.quotes:
            lanes["q"] = within.quotes
        if w in tags:
            lanes["tag"] = tags[w]
        if not out.bad and w[0] in " \n\t\r":
            lanes["lw"] = 1
        if not out.bad and not w.strip(" \n\t\r"):
            lanes["blank"] = 1
            blanks.append(text)
        if w[0] in "bfnrtu/":
            lanes["ec"] = 1
        elif w[0] in '"\\':
            lanes["ec"] = 2
        if lanes:
            table.append(f"token {text} means " + ", ".join(f"{k} = {v}" for k, v in lanes.items()))
            count += 1
    guard, sizes = [], {}
    for state in ALLOWED:
        if state == "done":
            continue
        guard.append(f"state s_{state}")
        guard.append(f"entail s_{state} when g_{state}, guard")
        if state in INSIDE:
            banned = [q for q in (quoted(w) for w in words if not fits(w, state)) if q]
            banned += ['"<|im_end|>"', '"<|endoftext|>"']
            guard.append("forbid " + ",\n    ".join(banned) + f" when s_{state}")
            sizes[state] = -len(banned)
        else:
            allowed = [q for q in (quoted(w) for w in words if w and fits(w, state)) if q]
            guard.append("only " + ",\n    ".join(allowed) + f" when s_{state}")
            sizes[state] = len(allowed)
    text = (RULES.replace("@TOKENS@", "\n".join(table)).replace("@PREV@", PREV)
            .replace("@LEGAL@", "\n".join(legal_lines())).replace("@NUMBER@", "\n".join(number_lines()))
            .replace("@GUARD@", "\n".join(guard)).replace("@BLANKS@", ",\n    ".join(blanks)))
    with open(sys.argv[2], "w", encoding="utf8", newline="\n") as f:
        f.write(text)
    print(count, "tokens listed;", sizes)


if __name__ == "__main__":
    main()
