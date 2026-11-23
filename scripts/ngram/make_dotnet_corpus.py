"""Turns the .NET SDK's reference XML documentation into text an n-gram pack can learn from: for
every public member, how C# calls it (`Directory.GetDirectories(path, searchPattern);`), grouped by
type (with --summaries, what it does as a comment above it). This is
where the exact spellings of member names after `Type.` come from; real code supplies the idioms.

    python scripts/ngram/make_dotnet_corpus.py <ref-pack-dir> <out-dir>
    e.g. "C:/Program Files/dotnet/packs/Microsoft.NETCore.App.Ref/11.0.0/ref/net11.0"
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

ALIASES = {
    "System.String": "string", "System.Int32": "int", "System.Int64": "long", "System.Boolean": "bool",
    "System.Double": "double", "System.Single": "float", "System.Object": "object", "System.Byte": "byte",
    "System.Char": "char", "System.Decimal": "decimal", "System.Int16": "short", "System.UInt32": "uint",
    "System.UInt64": "ulong", "System.UInt16": "ushort", "System.SByte": "sbyte", "System.Void": "void",
}


def short_type(t):
    t = t.replace("{", "<").replace("}", ">").replace("@", "")
    t = re.sub(r"`+\d+", "", t)
    for full, alias in ALIASES.items():
        t = re.sub(rf"\b{re.escape(full)}\b", alias, t)
    # System.IO.SearchOption -> SearchOption (the namespace is what `using` supplies)
    return re.sub(r"\b(?:[A-Z][A-Za-z0-9_]*\.)+([A-Z][A-Za-z0-9_]*)", r"\1", t)


def split_params(s):
    out, depth, cur = [], 0, ""
    for c in s:
        if c in "{(<[":
            depth += 1
        elif c in "})>]":
            depth -= 1
        if c == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
    if cur:
        out.append(cur)
    return out


def text_of(el):
    if el is None:
        return ""
    s = "".join(el.itertext())
    for ref in el.iter():
        if ref.tag in ("see", "seealso", "paramref", "typeparamref") and not (ref.text or "").strip():
            cref = ref.get("cref") or ref.get("name") or ref.get("langword") or ""
            s += ""  # the text form keeps its words; crefs are added below
    return re.sub(r"\s+", " ", s).strip()


def main(ref_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for xml in sorted(f for f in os.listdir(ref_dir) if f.endswith(".xml")):
        try:
            root = ET.parse(os.path.join(ref_dir, xml)).getroot()
        except ET.ParseError:
            continue
        by_type = {}
        for m in root.iter("member"):
            name = m.get("name", "")
            kind, _, ident = name.partition(":")
            if kind not in ("M", "P", "F", "E", "T"):
                continue
            summary = text_of(m.find("summary"))
            if kind == "T":
                ns, _, typ = ident.rpartition(".")
                by_type.setdefault(ident, []).insert(0, f"// {summary}\nnamespace {ns} {{ class {short_type(typ)} }}" if summary else f"namespace {ns} {{ class {short_type(typ)} }}")
                continue
            sig, _, params = ident.partition("(")
            params = params.rstrip(")")
            owner, _, member = sig.rpartition(".")
            member = re.sub(r"``\d+", "", member)
            tshort = short_type(owner)
            if member == "#ctor":
                member, call = None, f"new {tshort}"
            else:
                call = f"{tshort}.{member}"
            names = [p.get("name") for p in m.findall("param")]
            types = [short_type(p) for p in split_params(params)] if params else []
            # Written as a call, the way code uses it: argument names, no types (a pack learns what
            # follows "Directory.", not declarations to paste into a call).
            args = ", ".join(names[i] if i < len(names) and names[i] else "x" for i in range(len(types)))
            line = f"{call}({args});" if kind == "M" else f"var value = {call};"
            if SUMMARIES and summary:
                line = f"// {summary}\n" + line
            by_type.setdefault(owner, []).append(line)
        if not by_type:
            continue
        with open(os.path.join(out_dir, xml[:-4] + ".txt"), "w", encoding="utf-8") as f:
            for owner in sorted(by_type):
                f.write("\n".join(by_type[owner]) + "\n\n")
    print(f"wrote {len(os.listdir(out_dir))} files to {out_dir}")


SUMMARIES = False   # --summaries: each member's summary as a comment above it

if __name__ == "__main__":
    SUMMARIES = "--summaries" in sys.argv
    main(sys.argv[1], sys.argv[2])
