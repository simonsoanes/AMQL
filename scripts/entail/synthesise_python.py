"""Writes the synthetic Python program: a request, what follows from it, and the code.

  python synthesise_python.py <associations.json> <probed.json> <tokenizer-dir> <out.entail.txt>

The vocabulary of the request comes from the model's replies (`probe_words.py`), the
raw extract and the documentation (`associate.py`). The derivation below is written by
hand: it is what Python's documentation says the operations are for, and it is the part
an extract cannot supply. See docs/entail-synthesis.md.
"""
import json
import sys

WINDOW = 14        # tokens of request that can be read
SLOTS = 40         # tokens of answer
NEAR = 0.1         # how close to a concept the extract must put a word

# Words that carry no operation. Any other word the program cannot place stops it:
# acting on the part of a request it understood is how it would be confidently wrong.
FILLER = ("the a an of in on to from for all every each are is there that this and by with "
          "make convert put find get give compute calculate return turn").split()

SOURCES = ["text", "items", "numbers"]
OBJECTS = ["words", "lines"]
ACTIONS = ["reverse", "sort", "desc", "count", "sum", "max", "min", "longest", "shortest",
           "unique", "upper", "lower", "first", "last"]
FILTERS = ["even", "odd", "positive", "negative"]
CONCEPTS = SOURCES + OBJECTS + ACTIONS + FILTERS

# Stages of the answer, outermost first. Each piece is a flag and what it writes.
# Open pieces and close pieces mirror one another round the core.
STAGES = [
    ("join open", [("j_sp", '" ".join('), ("j_em", '"".join(')]),
    ("aggregate open", [("a_len", "len("), ("a_sum", "sum("), ("a_max", "max("), ("a_maxk", "max("),
                        ("a_min", "min("), ("a_mink", "min(")]),
    ("order open", [("o_sort", "sorted("), ("o_desc", "sorted(")]),
    ("unique open", [("u_on", "list(dict.fromkeys(")]),
    ("filter open", [("f_on", "[n for n in ")]),
    ("core", [("k_text", "text"), ("k_words", "text.split()"), ("k_lines", "text.splitlines()"),
              ("k_items", "items"), ("k_numbers", "numbers")]),
    ("filter close", [("c_even", " if n % 2 == 0]"), ("c_odd", " if n % 2 == 1]"),
                      ("c_positive", " if n > 0]"), ("c_negative", " if n < 0]")]),
    ("unique close", [("u_on", "))")]),
    ("order close", [("o_sort", ")"), ("o_desc", ", reverse=True)"), ("o_rev", "[::-1]")]),
    ("aggregate close", [("a_len", ")"), ("a_sum", ")"), ("a_max", ")"), ("a_min", ")"),
                         ("a_maxk", ", key=len)"), ("a_mink", ", key=len)"), ("a_first", "[0]"),
                         ("a_last", "[-1]"), ("a_up", ".upper()"), ("a_lo", ".lower()")]),
    ("join close", [("j_sp", ")"), ("j_em", ")")]),
]

DERIVATION = """
# ─── 3. What follows from the request ─────────────
#
# Each rule is something Python's documentation says. Anything that does not
# derive leaves `ok` unset, and the program says nothing: the host answers.

let nsrc = c_text + c_items + c_numbers
let nfilter = c_even + c_odd + c_positive + c_negative
let nagg = c_count + c_sum + c_max + c_min + c_longest + c_shortest + c_first + c_last + c_upper + c_lower

# What is being worked on. A string has words and lines; a list is already a sequence.
state one_src, obj_w, obj_l, s_str, noagg
entail one_src when nsrc == 1
entail obj_w when c_words, c_text
entail obj_l when c_lines, c_text, not c_words
entail s_str when c_text, not c_words, not c_lines
entail noagg when nagg <= 0
let s_strs = obj_w + obj_l + c_items
let s_num = c_numbers

# Order. Sorting in reverse is sorting descending, not sorting then reversing.
state o_sort, o_desc, o_rev
entail o_sort when c_sort, not c_reverse, not c_desc
entail o_desc when c_desc
entail o_desc when c_sort, c_reverse, not c_desc
entail o_rev when c_reverse, not c_sort, not c_desc

# Aggregates. max of numbers compares them; max of strings means the longest.
state a_len, a_sum, a_max, a_min, a_maxk, a_mink, a_first, a_last, a_up, a_lo
entail a_len when c_count
entail a_sum when c_sum
entail a_max when c_max, s_num
entail a_min when c_min, s_num
entail a_maxk when c_max, s_strs
entail a_mink when c_min, s_strs
entail a_maxk when c_longest, s_strs
entail a_mink when c_shortest, s_strs
entail a_first when c_first
entail a_last when c_last
entail a_up when c_upper
entail a_lo when c_lower

# Results. Reversed words are still a sentence; a sorted string is still a string.
state j_sp, j_em, u_on, f_on
entail j_sp when obj_w, o_rev, noagg
entail j_em when s_str, noagg, o_sort
entail j_em when s_str, noagg, o_desc
entail j_em when s_str, noagg, c_unique, not o_sort, not o_desc
entail u_on when c_unique
entail f_on when nfilter >= 1

state k_text, k_words, k_lines, k_items, k_numbers
entail k_text when s_str
entail k_words when obj_w
entail k_lines when obj_l
entail k_items when c_items
entail k_numbers when c_numbers

# What has no meaning. Only numbers add up or have parity; only strings have case;
# a string alone has no longest element to speak of.
state bad
entail bad when nagg >= 2
entail bad when nfilter >= 2
entail bad when f_on, not s_num
entail bad when c_sum, not s_num
entail bad when c_max, s_str
entail bad when c_min, s_str
entail bad when c_longest, s_num
entail bad when c_shortest, s_num
entail bad when c_longest, s_str
entail bad when c_shortest, s_str
entail bad when c_upper, not s_str
entail bad when c_lower, not s_str
entail bad when lost >= 1

let acted = nagg + nfilter + c_sort + c_reverse + c_desc + c_unique
state did, sound, ok
entail did when acted >= 1
entail sound when bad <= 0
entail ok when ask, comment, one_src, did, sound
"""


def main():
    associations_path, probed_path, tokenizer_dir, out = sys.argv[1:5]
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_dir + "/tokenizer.json")
    associations = json.load(open(associations_path, encoding="utf8"))
    probed = json.load(open(probed_path, encoding="utf8"))

    def pieces(text):
        return [tokenizer.decode([i]) for i in tokenizer.encode(text, add_special_tokens=False).ids]

    # Vocabulary: a concept's own name first, then what the model turned the word
    # into, then what the extract puts near it, then what only its documentation
    # mentions. The first claim on a word stands.
    claims, skipped = {}, []
    code = {c: i + 1 for i, c in enumerate(CONCEPTS)}
    own = {"text": "text", "items": "items", "numbers": "numbers", "words": "word words",
           "lines": "line lines", "desc": "descending", "upper": "upper uppercase",
           "lower": "lower lowercase", "unique": "unique", "sum": "sum", "max": "max",
           "min": "min", "count": "count"}
    for concept in CONCEPTS:
        for word in own.get(concept, concept).split():
            claims.setdefault(word, (concept, "name"))
    for concept in CONCEPTS:
        for word in probed.get(concept, []):
            if word not in FILLER:
                claims.setdefault(word, (concept, "model"))
    for concept in CONCEPTS:
        for word, score in associations.get(concept, {}).get("extract", []):
            if score >= NEAR:
                claims.setdefault(word, (concept, f"extract {score}"))
    seen_in = {}
    for concept, entry in associations.items():
        for word in entry["docs"]:
            seen_in.setdefault(word, set()).add(concept)
    for concept in CONCEPTS:
        for word in associations.get(concept, {}).get("docs", []):
            if len(seen_in[word]) == 1:
                claims.setdefault(word, (concept, "documentation"))

    facts = {}          # token -> {fact: value}
    notes = {}
    for word, (concept, why) in sorted(claims.items()):
        forms = [" " + word, " " + word.capitalize()]
        for form in forms:
            parts = pieces(form)
            if len(parts) == 1:
                facts.setdefault(form, {}).update({"w": code[concept], "kn": 1})
                notes[form] = f"{concept}, from {why}"
            else:
                skipped.append(form)
    for word in FILLER:
        for form in (" " + word, " " + word.capitalize()):
            if len(pieces(form)) == 1 and form not in notes:
                facts.setdefault(form, {})["kn"] = 1
                notes[form] = "filler"
    facts.setdefault(",", {})["kn"] = 1
    facts.setdefault("\n", {})["nl"] = 1
    facts.setdefault("\n\n", {})["nl"] = 1      # a blank line is one token, and ends a line too
    facts.setdefault("result", {})["res"] = 1
    facts.setdefault(" =", {})["eq"] = 1
    facts.setdefault("#", {}).update({"hash": 1, "kn": 1})

    outputs = {}

    def output(token):
        if token not in outputs:
            outputs[token] = len(outputs) + 1
        facts.setdefault(token, {})["ans"] = 1
        return outputs[token]

    output(" ")
    spelled = {text: pieces(text) for _, alternatives in STAGES for _, text in alternatives}
    for parts in spelled.values():
        for part in parts:
            output(part)
    newline = len(outputs) + 1
    outputs["\n"] = newline         # ends the answer, so it is not marked as answer

    def q(token):
        return '"' + token.replace("\n", "\\n").replace('"', "\\q") + '"'

    w = []
    w.append("# Synthetic Python: from a request to the code that answers it.")
    w.append("# Generated by scripts/entail/synthesise_python.py. See docs/entail-synthesis.md.")
    w.append("#")
    w.append("#   # sum the even numbers")
    w.append("#   result = sum([n for n in numbers if n % 2 == 0])")
    w.append("#")
    w.append("# 1 reads the request line into concepts, 2 gathers them at the \"=\", 3 derives")
    w.append("# what they mean together, 4 spells the result, and 5 writes it out. It does")
    w.append("# not know Python: it knows these operations, and is silent about the rest.")
    w.append("")
    w.append("program python_synthetic")
    w.append("")
    w.append(f"const WINDOW = {WINDOW}")
    w.append(f"const SLOTS = {SLOTS}")
    for concept in CONCEPTS:
        w.append(f"const W_{concept} = {code[concept]}")
    w.append("")
    w.append("# ─── 1. Reading: what each token of a request is ───")
    w.append("")
    w.append("state w, kn, nl, res, eq, hash, ans")
    for token in sorted(facts):
        means = ", ".join(f"{k} = {v}" for k, v in facts[token].items())
        note = f"   # {notes[token]}" if token in notes else ""
        w.append(f"token {q(token)} means {means}{note}")
    w.append("")
    w.append("# ─── 2. Gathering: the request line, seen from the \"=\" ───")
    w.append("#")
    w.append("# The prompt ends \"# request\\nresult =\": the request starts three places back.")
    w.append("# Nothing past a newline counts, so an earlier line cannot leak in.")
    w.append("")
    w.append("state isres, isnl, ask")
    w.append("attend back 1\n    isres <- res\nend")
    w.append("attend back 2\n    isnl <- nl\nend")
    w.append("entail ask when eq, isres, isnl")
    w.append("")
    w.append("state cw[WINDOW], nw[WINDOW], hw[WINDOW], kw[WINDOW], open[WINDOW]")
    w.append("for c in 0 to WINDOW-1")
    w.append("    attend back c+3\n        cw[c] <- w\n        nw[c] <- nl\n        hw[c] <- hash\n        kw[c] <- kn\n    end")
    w.append("    entail open[c] when (sum i in 0 to c : nw[i]) <= 0")
    w.append("end")
    w.append("")
    w.append("state hits[" + str(len(CONCEPTS) + 1) + "], hashes, comment, lost")
    w.append("for c in 0 to WINDOW-1")
    w.append(f"    for k in 1 to {len(CONCEPTS)}")
    w.append("        entail hits[k] when cw[c] == k, open[c], ask")
    w.append("    end")
    w.append("    entail hashes when hw[c] >= 1, open[c], ask")
    w.append("    entail lost when open[c], not kw[c], ask")
    w.append("end")
    w.append("entail comment when hashes >= 1")
    w.append("")
    w.append("state " + ", ".join(f"c_{c}" for c in CONCEPTS))
    for concept in CONCEPTS:
        w.append(f"entail c_{concept} when hits[W_{concept}] >= 1")
    w.append(DERIVATION)
    w.append("# ─── 4. Spelling: each stage writes on from where the last stopped ───")
    w.append("")
    # The filter's parity flags share names with concepts' c_ lanes, by design.
    for token, value in outputs.items():
        w.append(f"const T{value} = {value}   # {q(token)}")
    w.append("")
    w.append("state code[SLOTS]")
    w.append(f"state at0[SLOTS]")
    w.append(f"entail code[0] = T{outputs[' ']} when ok")
    w.append("entail at0[1] when ok")
    for number, (title, alternatives) in enumerate(STAGES):
        here, there = f"at{number}", f"at{number + 1}"
        w.append("")
        w.append(f"# {title}")
        w.append(f"state {there}[SLOTS]")
        flags = sorted({flag for flag, _ in alternatives})
        w.append(f"let any{number} = " + " + ".join(flags))
        longest = max(len(spelled[text]) for _, text in alternatives)
        w.append(f"for i in 0 to SLOTS-{longest + 2}")
        for flag, text in alternatives:
            parts = spelled[text]
            for offset, part in enumerate(parts):
                w.append(f"    entail code[i+{offset}] = T{outputs[part]} when {here}[i], {flag}   # {q(part)}")
            w.append(f"    entail {there}[i+{len(parts)}] when {here}[i], {flag}")
        w.append(f"    entail {there}[i] when {here}[i], not any{number}")
        w.append("end")
    last = f"at{len(STAGES)}"
    w.append("")
    w.append("# The end of the line.")
    w.append("for i in 0 to SLOTS-1")
    w.append(f"    entail code[i] = T{newline} when {last}[i]")
    w.append("end")
    w.append("")
    w.append("# ─── 5. Writing: each answer token looks back at the \"=\" for its own slot ───")
    w.append("")
    w.append("state aw[SLOTS], ww[SLOTS], from[SLOTS]")
    w.append("for c in 1 to SLOTS-1")
    w.append("    attend back c\n        aw[c] <- ask\n        ww[c] <- ans\n    end")
    w.append("end")
    w.append("entail from[0] when ask")
    w.append("entail from[1] when aw[1], ans")
    w.append("for c in 2 to SLOTS-1")
    w.append("    entail from[c] when aw[c], ans + (sum i in 1 to c-1 : ww[i]) >= c")
    w.append("end")
    w.append("")
    w.append("state seen[SLOTS]")
    w.append("attend back by from\n    for s in 0 to SLOTS-1\n        seen[s] <- code[s]\n    end\nend")
    w.append("")
    w.append("state say")
    w.append("for s in 0 to SLOTS-1\n    entail say = seen[s] when from[s]\nend")
    w.append("")
    w.append(f"state o[{len(outputs) + 1}]")
    for token, value in outputs.items():
        w.append(f"entail o[{value}] when say == {value}")
        w.append(f"emit {q(token)} when o[{value}]")
    open(out, "w", encoding="utf8", newline="\n").write("\n".join(w) + "\n")
    print(f"{len(claims)} request words, {len(outputs)} output tokens, longest answer ~{sum(max(len(spelled[t]) for _, t in a) for _, a in STAGES) + 2}")
    print("not single tokens:", " ".join(skipped)[:300])
    by_source = {}
    for word, (concept, why) in claims.items():
        by_source.setdefault(why.split()[0], []).append(word)
    print({k: len(v) for k, v in by_source.items()})


if __name__ == "__main__":
    main()
