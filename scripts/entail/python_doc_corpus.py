"""Text for the next decompile, aimed at what the last one missed.

  python python_doc_corpus.py <coverage.json> <out.txt>

Takes the `--missing` report of python_coverage.py and writes Python's own
documentation for those items: docstrings and signatures of the builtins, methods
and modules, and the language reference topics. The model reads this text, so the
units that respond to those items get a chance to be seen. It tells the model
nothing: extraction only ever records what the model already does.
"""
import ast
import builtins
import importlib.machinery
import inspect
import json
import sys

from pydoc_data import topics


def describe(owner, name, prefix=""):
    item = getattr(owner, name, None)
    if item is None:
        return ""
    try:
        signature = str(inspect.signature(item))
    except (TypeError, ValueError):
        signature = "()"
    doc = inspect.getdoc(item) or ""
    return f"{prefix}{name}{signature}\n{doc[:600]}\n\n"


def module_doc(name):
    """A module's docstring, read from its source. Importing would run it, and
    some of the standard library does things when imported."""
    spec = importlib.machinery.PathFinder.find_spec(name)
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return ""
    try:
        with open(spec.origin, encoding="utf8") as source:
            return ast.get_docstring(ast.parse(source.read())) or ""
    except (OSError, SyntaxError, UnicodeDecodeError):
        return ""


def main():
    report = json.load(open(sys.argv[1], encoding="utf8"))["missing"]
    parts = []
    for name in report.get("builtin functions", []) + report.get("exceptions", []):
        parts.append(describe(builtins, name))
    for kind in (str, list, dict, set, tuple, bytes, int, float):
        for name in report.get(f"{kind.__name__} methods", []):
            parts.append(describe(kind, name, f"value = {kind.__name__}()\nvalue."))
    for name in report.get("standard library", []):
        parts.append(f"import {name}\n" + module_doc(name)[:400] + "\n\n")
    for text in topics.topics.values():
        parts.append(text[:1500] + "\n\n")
    open(sys.argv[2], "w", encoding="utf8", newline="\n").write("".join(parts))
    print(f"{sum(len(p) for p in parts)} characters")


if __name__ == "__main__":
    main()
