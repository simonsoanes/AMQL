"""Test cases for chemistry.entail.txt and physics.entail.txt, and their scoring.

  python science_cases.py chemistry <periodic-table.csv> <out.jsonl> [seed]
  python science_cases.py physics <out.jsonl> [seed]
  python science_cases.py score <cases.jsonl> <report.json> [--base <report.json>]

Answers come from an ordinary calculation here, not from the programs. Each prompt
opens with a short drill, so an unmodified model answers rather than chats and the
baseline is its real ability. The last line of the prompt is the test.

A reply is right if its first line opens with the right number and unit. Molar masses
get 0.01 of tolerance, so a model isn't marked down for using a slightly different
table of weights. A "refuse" case wants nothing wrong said: its right answer is
"incompatible units" from physics, and silence is scored as what the host then says.
"""
import csv
import json
import random
import re
import sys
from decimal import Decimal, ROUND_DOWN, getcontext
from fractions import Fraction

getcontext().prec = 60

COMPOUNDS = """H2O CO2 NaCl NH3 CH4 HCl H2SO4 HNO3 NaOH KOH CaCO3 C6H12O6 C2H5OH CH3COOH Fe2O3 Al2O3
MgO CaO SO2 SO3 NO2 N2O H2O2 KCl KBr NaBr LiF AgNO3 AgCl CuSO4 ZnO ZnCl2 FeCl3 BaSO4 PbO2 KMnO4
K2Cr2O7 NaHCO3 Na2CO3 C12H22O11 C8H10N4O2 C9H8O4 C2H4 C2H6 C3H8 C4H10 C6H6 CH3OH H3PO4 Na3PO4
Ca(OH)2 Mg(OH)2 Al(OH)3 Fe(OH)3 Ca(NO3)2 Mg(NO3)2 Cu(NO3)2 Pb(NO3)2 (NH4)2SO4 (NH4)3PO4 Al2(SO4)3
Ca3(PO4)2 Fe2(SO4)3 Ba(OH)2 NH4Cl NH4NO3 TiO2 SiO2 U3O8 Au2O3 HgCl2""".split()

EQUATIONS = """2H2 + O2 -> 2H2O
N2 + 3H2 -> 2NH3
CH4 + 2O2 -> CO2 + 2H2O
C3H8 + 5O2 -> 3CO2 + 4H2O
2Na + Cl2 -> 2NaCl
2Mg + O2 -> 2MgO
4Fe + 3O2 -> 2Fe2O3
CaCO3 -> CaO + CO2
2H2O2 -> 2H2O + O2
Zn + 2HCl -> ZnCl2 + H2
2Al + 3Cl2 -> 2AlCl3
C2H5OH + 3O2 -> 2CO2 + 3H2O
Fe2O3 + 3CO -> 2Fe + 3CO2
2KClO3 -> 2KCl + 3O2
HCl + NaOH -> NaCl + H2O
H2SO4 + 2NaOH -> Na2SO4 + 2H2O
2C2H6 + 7O2 -> 4CO2 + 6H2O
Ca(OH)2 + 2HCl -> CaCl2 + 2H2O
3Mg + N2 -> Mg3N2
2Al + 3CuSO4 -> Al2(SO4)3 + 3Cu""".splitlines()


def formula_mass(text, table):
    """Molar mass in thousandths, with an ordinary parser."""
    def run(s, i):
        total = 0
        while i < len(s) and s[i] != ")":
            if s[i] == "(":
                inner, i = run(s, i + 1)
                i += 1
            else:
                m = re.match(r"[A-Z][a-z]?", s[i:])
                inner, i = table[m.group()], i + len(m.group())
            m = re.match(r"\d*", s[i:])
            total += inner * int(m.group() or 1)
            i += len(m.group())
        return total, i
    return run(text, 0)[0]


def weights(path):
    with open(path, encoding="utf8") as f:
        return {r["Symbol"]: round(float(r["AtomicMass"]) * 1000) for r in csv.DictReader(f)}


def show_mass(milli):
    return f"{milli // 1000}.{milli % 1000:03d} g/mol"


def species(side):
    for term in side.split(" + "):
        m = re.match(r"(\d*)(.*)", term)
        yield int(m.group(1) or 1), m.group(2)


def balanced(equation, table):
    left, right = equation.split(" -> ")
    mass = lambda side: sum(c * formula_mass(f, table) for c, f in species(side))
    return mass(left) == mass(right)


def spoil(equation, rng):
    """The same equation with one coefficient changed, so it no longer balances."""
    terms = list(re.finditer(r"(^| )(\d*)([A-Z(])", equation))
    m = rng.choice(terms)
    old = int(m.group(2) or 1)
    new = old + rng.choice([1, 2]) if old < 7 else old - 1
    return equation[:m.start(2)] + str(new) + equation[m.end(2):]


def chemistry(path, out, seed):
    table = weights(path)
    rng = random.Random(seed)
    drill = f"M(O2)={show_mass(formula_mass('O2', table))}\nM(HCl)={show_mass(formula_mass('HCl', table))}\n"
    said = (f"The molar mass of O2 is {show_mass(formula_mass('O2', table))}.\n"
            f"The molar mass of HCl is {show_mass(formula_mass('HCl', table))}.\n")
    check = "\nH2 + Cl2 -> 2HCl is balanced\nH2 + N2 -> NH3 is unbalanced\n"
    cases = []
    for c in COMPOUNDS:
        answer = show_mass(formula_mass(c, table))
        cases.append(dict(category="mass-formula", prompt=f"{drill}M({c})=", answer=answer, max_tokens=14))
        cases.append(dict(category="mass-sentence", prompt=f"{said}The molar mass of {c} is", answer=answer, max_tokens=14))
    for e in EQUATIONS:
        assert balanced(e, table), e
        cases.append(dict(category="equation-balanced", prompt=f"{check}{e} is", answer="balanced", max_tokens=4))
        bad = spoil(e, rng)
        assert not balanced(bad, table), bad
        cases.append(dict(category="equation-unbalanced", prompt=f"{check}{bad} is", answer="unbalanced", max_tokens=4))
    for c in ["Xx2", "H2Q", "Ca(OH", "K4[Fe(CN)6]", "water"]:
        cases.append(dict(category="silent", prompt=f"{drill}M({c})=", answer="", max_tokens=8))
    write(out, cases)


# name, numerator unit, its exponents in kg m s A, denominator, result
LAWS = [
    ("F=ma", "kg", "m/s^2", "*", "N"),
    ("W=Fd", "N", "m", "*", "J"),
    ("P=W/t", "J", "s", "/", "W"),
    ("P=VI", "V", "A", "*", "W"),
    ("V=IR", "A", "Ω", "*", "V"),
    ("I=V/R", "V", "Ω", "/", "A"),
    ("R=V/I", "V", "A", "/", "Ω"),
    ("p=F/A", "N", "m^2", "/", "Pa"),
    ("rho=m/V", "kg", "m^3", "/", "kg/m^3"),
    ("v=d/t", "m", "s", "/", "m/s"),
    ("a=v/t", "m/s", "s", "/", "m/s^2"),
    ("Q=It", "A", "s", "*", "C"),
    ("d=vt", "m/s", "s", "*", "m"),
    ("A=lw", "m", "m", "*", "m^2"),
    ("p=mv", "kg", "m/s", "*", "kg m/s"),
    ("E=Pt", "W", "s", "*", "J"),
]

PREFIXED = [
    ("kN", 3, "m", 0, "*", "J"), ("kW", 3, "s", 0, "*", "J"), ("km", 3, "s", 0, "/", "m/s"),
    ("mA", -3, "V", 0, "*", "W"), ("g", -3, "m/s^2", 0, "*", "N"), ("cm", -2, "m", 0, "*", "m^2"),
    ("kPa", 3, "m^2", 0, "*", "N"), ("mV", -3, "A", 0, "/", "Ω"), ("kJ", 3, "s", 0, "/", "W"),
]

SUMS = [("km", 3, "m", 0, "m"), ("cm", -2, "m", 0, "m"), ("g", -3, "kg", 0, "kg"), ("ms", -3, "s", 0, "s"),
        ("kN", 3, "N", 0, "N"), ("mm", -3, "cm", -2, "m"), ("mA", -3, "A", 0, "A"), ("kJ", 3, "J", 0, "J")]

CLASHES = [("m", "s"), ("kg", "m"), ("N", "J"), ("V", "A"), ("W", "J"), ("m/s", "m"), ("m", "m^2"), ("Pa", "N")]


def quantity(rng, whole_digits=3):
    places = rng.choice([0, 0, 1, 2])
    digits = rng.randint(1, whole_digits)
    text = str(rng.randint(10 ** (digits - 1), 10 ** digits - 1))
    if places:
        text += "." + "".join(rng.choice("0123456789") for _ in range(places - 1)) + rng.choice("123456789")
    return text


def plain(value):
    text = format(value.normalize(), "f")
    return text


def quotient(a, b, ea, eb):
    """Division as the program does it: FRAC digits past the last digit of a."""
    fa = -a.as_tuple().exponent if a.as_tuple().exponent < 0 else 0
    fb = -b.as_tuple().exponent if b.as_tuple().exponent < 0 else 0
    ai, bi = int(a * 10 ** fa), int(b * 10 ** fb)
    whole = ai * 10 ** 4 // bi
    return Decimal(whole).scaleb(ea - fa - eb + fb - 4)


def physics(out, seed):
    rng = random.Random(seed)
    drill = "2 kg * 3 m/s^2 = 6 N\n10 m / 2 s = 5 m/s\n1 km + 5 m = 1005 m\n"
    cases = []

    def add(category, a, ua, op, b, ub, answer):
        cases.append(dict(category=category, prompt=f"{drill}{a} {ua} {op} {b} {ub} =", answer=answer, max_tokens=20))

    for name, ua, ub, op, result in LAWS:
        for _ in range(6):
            a, b = quantity(rng), quantity(rng, 2 if op == "/" else 3)
            value = Decimal(a) * Decimal(b) if op == "*" else quotient(Decimal(a), Decimal(b), 0, 0)
            add("law-" + ("multiply" if op == "*" else "divide"), a, ua, op, b, ub, f"{plain(value)} {result}")
    for ua, ea, ub, eb, op, result in PREFIXED:
        for _ in range(4):
            a, b = quantity(rng), quantity(rng, 2)
            if op == "*":
                value = (Decimal(a) * Decimal(b)).scaleb(ea + eb)
            else:
                value = quotient(Decimal(a), Decimal(b), ea, eb)
            add("prefix", a, ua, op, b, ub, f"{plain(value)} {result}")
    for ua, ea, ub, eb, result in SUMS:
        for op in "+-":
            for _ in range(3):
                a, b = quantity(rng), quantity(rng)
                x, y = Decimal(a).scaleb(ea), Decimal(b).scaleb(eb)
                value = x + y if op == "+" else x - y
                add("sum", a, ua, op, b, ub, f"{plain(value)} {result}")
    for ua, ub in CLASHES:
        for op in "+-":
            add("refuse", rng.randint(2, 99), ua, op, rng.randint(2, 99), ub, "incompatible units")
    for text in ["3 h * 2 km", "20 min + 1 h", "5 kg * 2 m * 3 s", "4 °C + 2 K", "3 lb * 2 ft", "7 m^2 * 2 N"]:
        cases.append(dict(category="silent", prompt=f"{drill}{text} =", answer="", max_tokens=8))
    write(out, cases)


def write(path, cases):
    with open(path, "w", encoding="utf8", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(len(cases), "cases")


def first_line(reply):
    return reply.strip().split("\n")[0].strip()


def number_and_unit(text):
    m = re.match(r"(-?\d+(?:\.\d+)?)\s*(.*)$", text)
    if not m:
        return None, text
    unit = m.group(2).rstrip(" .").replace("²", "^2").replace("³", "^3").replace("ohm", "Ω").replace("·", " ")
    return Decimal(m.group(1)), unit


def right(answer, reply):
    line = first_line(reply)
    if not answer:
        return False
    x, unit = number_and_unit(answer)
    if x is None:
        return re.sub(r"[.\s]+$", "", line.lower()) == answer
    y, said = number_and_unit(line)
    # the host may go on after the answer: only the unit has to be there first
    if y is None or not re.match(re.escape(unit) + r"(?![\w^/])", said):
        return False
    return abs(x - y) <= (Decimal("0.01") if "g/mol" in unit else Decimal(0))


def score(cases_path, report_path, base_path):
    cases = [json.loads(l) for l in open(cases_path, encoding="utf8") if l.strip()]
    replies = [c["reply"] for c in json.load(open(report_path, encoding="utf8"))["cases"]]
    base = [c["reply"] for c in json.load(open(base_path, encoding="utf8"))["cases"]] if base_path else None
    rows = {}
    wrong = []
    for i, c in enumerate(cases):
        r = rows.setdefault(c["category"], dict(n=0, ok=0, base=0, broke=0, fixed=0, silent=0))
        r["n"] += 1
        good = right(c["answer"], replies[i])
        r["ok"] += good
        if c["category"] == "silent":
            r["silent"] += first_line(replies[i]) == "" or (base is not None and first_line(replies[i]) == first_line(base[i]))
        elif not good:
            wrong.append((c["prompt"].split("\n")[-1], c["answer"], first_line(replies[i])))
        if base:
            was = right(c["answer"], base[i])
            r["base"] += was
            r["fixed"] += good and not was
            r["broke"] += was and not good
    print(f"{'category':22}{'n':>5}{'right':>7}" + (f"{'base':>7}{'fixed':>7}{'broke':>7}" if base else "") + f"{'silent':>8}")
    for name, r in sorted(rows.items()):
        print(f"{name:22}{r['n']:5}{r['ok']:7}" + (f"{r['base']:7}{r['fixed']:7}{r['broke']:7}" if base else "") + f"{r['silent']:8}")
    for w in wrong[:25]:
        print("  ", w)


def main():
    a = sys.argv
    if a[1] == "chemistry":
        chemistry(a[2], a[3], int(a[4]) if len(a) > 4 else 1)
    elif a[1] == "physics":
        physics(a[2], int(a[3]) if len(a) > 3 else 1)
    else:
        score(a[2], a[3], a[a.index("--base") + 1] if "--base" in a else None)


if __name__ == "__main__":
    main()
