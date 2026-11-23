"""Copies C# source into a corpus an n-gram pack can learn idioms from: comments removed (doc
comments are the most frequent text in most codebases, and a pack built on them makes a model
write `/// <param …>` blocks), blank lines dropped, generated files (*.g.cs, *.Designer.cs,
"Generated" in the name) and very large files (generated bindings, tables) left out.

    python scripts/ngram/make_code_corpus.py <out-dir> <source-dir> [<source-dir> …] [--max-kb 120]
"""
import os
import re
import sys

COMMENT = re.compile(r'//[^\n]*|/\*.*?\*/|("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')', re.S)


def strip(code):
    # Keep string and char literals, drop comments.
    code = COMMENT.sub(lambda m: m.group(1) or "", code)
    return "\n".join(l.rstrip() for l in code.splitlines() if l.strip())


def main():
    args = sys.argv[1:]
    max_kb = 120
    if "--max-kb" in args:
        i = args.index("--max-kb")
        max_kb = int(args[i + 1])
        del args[i:i + 2]
    out, sources = args[0], args[1:]
    os.makedirs(out, exist_ok=True)
    kept = skipped = 0
    for src in sources:
        for dp, dn, fn in os.walk(src):
            dn[:] = [d for d in dn if d not in ("bin", "obj", ".git", "node_modules")]
            for f in fn:
                if not f.endswith(".cs") or f.endswith((".g.cs", ".Designer.cs")) or "Generated" in f:
                    continue
                path = os.path.join(dp, f)
                if os.path.getsize(path) > max_kb * 1024:
                    skipped += 1
                    continue
                try:
                    text = open(path, encoding="utf-8-sig", errors="replace").read()
                except OSError:
                    continue
                name = re.sub(r"[^\w.-]", "_", os.path.relpath(path, src))
                open(os.path.join(out, f"{os.path.basename(src.rstrip('/\\'))}-{name}.txt"), "w", encoding="utf-8").write(strip(text))
                kept += 1
    print(f"kept {kept} files, skipped {skipped} larger than {max_kb} KB, in {out}")


if __name__ == "__main__":
    main()
