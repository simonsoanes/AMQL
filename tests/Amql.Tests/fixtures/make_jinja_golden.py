"""Goldens for the Jinja interpreter: what Python's Jinja renders, in the environment transformers
compiles chat templates with. JinjaTests renders the same and compares exactly.

    python tests/Amql.Tests/fixtures/make_jinja_golden.py

Give it files of {"name", "template", "bos", "eos"} (chat templates with their tokens) and it writes
what transformers renders from them for a set of conversations beside them, as <file>.golden.
"""
import json
import os
import sys

import jinja2
from transformers.utils.chat_template_utils import _compile_jinja_template

HERE = os.path.dirname(os.path.abspath(__file__))

VARS = {
    "name": "World", "n": 7, "x": 2.5, "yes": True, "no": False, "nothing": None, "empty": "", "words": ["b", "A", "c", "a"],
    "nums": [3, 1, 2], "nested": [[1, 2], [3, [4, 5]]], "text": "  some text\nwith lines \n", "emoji": "a\U0001F600bé",
    "quotes": "it's \"so\" \\ \t", "d": {"b": 1, "a": "x", "C": [True, None], "items": "shadow"},
    "people": [{"name": "Zed", "age": 30, "tags": ["p"]}, {"name": "amy", "age": 25}, {"name": "Bob", "age": 30, "tags": []}],
    "big": 12345678901234, "tiny": 1e-7, "huge": 1e22, "whole": 3.0, "tool": {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"q": {"type": "string", "description": "ü <b>"}}, "required": ["q"]}}},
}

SNIPPETS = [
    # output and whitespace
    "Hello {{ name }}!", "{{ n }} {{ x }} {{ yes }} {{ nothing }} {{ missing }}|", "{{ words }} {{ d }} {{ (1, 'a') }} {{ (1,) }} {{ () }}",
    "{{ big }} {{ tiny }} {{ huge }} {{ whole }} {{ 1/3 }} {{ 10/5 }}", "{{ quotes }} {{ [quotes] }} {{ [emoji] }}",
    "a\n{% if yes %}\nb\n{% endif %}\nc\n", "a\n  {% if yes %}\n  b\n  {% endif %}\nc", "a  {% if yes %}  b  {% endif %}  c",
    "a  {%- if yes -%}  b  {%- endif -%}  c", "a\n\n{%- if yes %}\n\nb{% endif -%}\n\n c", "x {{- name -}} y", "x {{- ' ' + name }} y\n\n",
    "a {# comment #} b", "a\n{# comment #}\nb", "a\n  {#- comment -#}\n  b", "  {%+ if yes %}keep{% endif %}", "a\n{% if yes +%}\nb{% endif %}",
    "{% raw %}{{ not }} {% this %}{% endraw %}!", "a\n{%- raw -%}\n {{x}} \n{%- endraw -%}\n b", "line one\r\nline two\r\n{{ n }}\r\n", "{{ '\\n' }}|{{ '\\t\\x41\\u00e9\\\\' }}|{{ 'a\\'b' }}|{{ \"q\\\"q\" }}|{{ '\\d' }}",
    "{{ 'a' 'b' \"c\" }}", "{{ '{{ x }}' }} {{ '%}' }}", "\n\n{% set a = 1 %}\n\n{{ a }}\n\n", "\t{% set a = 1 %}\t{{ a }}",
    "{% for i in nums %}\n  {{ i }}\n{% endfor %}\nend", "{%- for i in nums -%}\n  {{ i }},\n{%- endfor -%}\n end", "{{\n name\n}}",
    # expressions
    "{{ 1 + 2 * 3 - 4 // 3 % 2 ** 2 }} {{ 7 // 2 }} {{ -7 // 2 }} {{ -7 % 3 }} {{ 7.5 // 2 }} {{ 2 ** 10 }} {{ 2 ** -1 }}", "{{ 'a' ~ 1 ~ none ~ true }}",
    "{{ 'ab' * 3 }} {{ [1] * 2 }} {{ [1] + [2, 3] }}", "{{ 1 < 2 < 3 }} {{ 1 < 3 < 2 }} {{ 'a' < 'b' }} {{ 1 == 1.0 }} {{ true == 1 }} {{ 'a' != 'b' }}",
    "{{ 'b' in words }} {{ 'z' not in words }} {{ 'or' in 'World' }} {{ 'a' in d }} {{ 2 in nums }}", "{{ not yes }} {{ not no and yes }} {{ no or 'other' }} {{ empty or none or 0 or 'last' }} {{ yes and 'then' }}",
    "{{ 'yes' if n > 5 else 'no' }} {{ 'x' if no }}|{{ 'a' if no else 'b' if no else 'c' }}", "{{ -n }} {{ +n }} {{ -n|abs }} {{ (-n)|abs }} {{ -2.5 }}",
    "{{ words[0] }} {{ words[-1] }} {{ words[1:3] }} {{ words[::-1] }} {{ words[:-1] }} {{ words[10] }}| {{ words[5:] }}", "{{ name[0] }} {{ name[1:3] }} {{ name[::-1] }} {{ name[-2:] }} {{ emoji[1] }} {{ emoji[1:] }} {{ emoji|length }}",
    "{{ d.b }} {{ d['a'] }} {{ d.missing }}| {{ d['missing'] }}| {{ d.C[0] }} {{ d['items'] }} {{ d.get('b') }} {{ d.get('z', 'dflt') }} {{ d.get('z') }}",
    "{{ {'a': 1, 'b': [1, 2]} }} {{ {'a': 1}.a }} {{ {1: 'one'}[1] }} {{ {} }}", "{{ 1_000 }} {{ 1.5e3 }} {{ 0x1F }} {{ 5 .__class__ if no else 'ok' }}", "{{ (1, 2, 3)[1] }} {{ [1, 2, 3][-1] }}",
    "{{ '%s is %d, %.2f%%' % ('a', 3, 2.5) }} {{ '%s' % 'x' }}", "{{ '{} and {}'.format(1, 'b') }} {{ '{0}{1}{0}'.format('x', 'y') }} {{ '{k}'.format(k=3) }}", "{{ none is none }} {{ 1 is number and yes }}",
    # tests
    "{{ name is defined }} {{ missing is defined }} {{ missing is undefined }} {{ missing is not defined }} {{ nothing is none }} {{ nothing is not none }}", "{{ name is string }} {{ n is string }} {{ d is mapping }} {{ words is sequence }} {{ d is sequence }} {{ name is iterable }} {{ n is iterable }} {{ missing is iterable }}",
    "{{ n is number }} {{ x is float }} {{ n is integer }} {{ yes is boolean }} {{ yes is integer }} {{ yes is true }} {{ n is true }} {{ no is false }} {{ yes is number }}", "{{ n is odd }} {{ n is even }} {{ n is divisibleby 7 }} {{ n is divisibleby(2) }} {{ 3 is in nums }} {{ n is eq 7 }} {{ n is gt(5) }} {{ n is lt 5 }} {{ 'abc' is lower }} {{ 'ABC' is upper }} {{ none is sameas none }}",
    "{% if words is iterable and words is not mapping and words is not string %}list{% endif %}", "{% if missing is defined and missing.x %}a{% else %}b{% endif %}",
    # filters
    "{{ text|trim }}| {{ '--a--'|trim('-') }} {{ name|upper }} {{ name|lower }} {{ 'hello wOrld'|capitalize }} {{ 'hello wOrld-foo (bar)'|title }}", "{{ words|length }} {{ words|count }} {{ name|length }} {{ d|length }} {{ missing|length }}", "{{ words|join }} {{ words|join(', ') }} {{ people|join('/', attribute='name') }} {{ nums|join('-') }}",
    "{{ words|first }} {{ words|last }} {{ []|first }}| {{ 'abc'|first }} {{ 'abc'|last }} {{ words|reverse|list }} {{ 'abc'|reverse }} {{ 'abc'|list }}", "{{ missing|default('d') }} {{ empty|default('d') }}| {{ empty|default('d', true) }} {{ nothing|default('d') }} {{ nothing|d('d', boolean=true) }}", "{{ words|sort }} {{ words|sort(case_sensitive=true) }} {{ nums|sort(reverse=true) }} {{ people|sort(attribute='name')|map(attribute='name')|list }}",
    "{{ d|dictsort }} {{ {'b': 1, 'A': 3, 'c': 2}|dictsort }} {{ {'b': 1, 'A': 3, 'c': 2}|dictsort(true) }} {{ {'b': 1, 'A': 3, 'c': 2}|dictsort(by='value') }} {{ {'b': 1, 'a': 3}|dictsort(reverse=true) }}", "{{ words|map('upper')|list }} {{ people|map(attribute='age')|list }} {{ people|map(attribute='tags', default='-')|list }} {{ ['1', 'x']|map('int', 5)|list }}", "{{ nums|select('gt', 1)|list }} {{ nums|reject('odd')|list }} {{ [0, 1, '', 'a']|select|list }} {{ people|selectattr('age', 'equalto', 30)|map(attribute='name')|list }} {{ people|selectattr('tags')|list|length }} {{ people|rejectattr('tags', 'defined')|map(attribute='name')|join }}",
    "{{ [1, 2, 1, 'a', 'A']|unique|list }} {{ nums|max }} {{ nums|min }} {{ nums|sum }} {{ people|sum(attribute='age') }} {{ words|max }} {{ [1, 2.5]|sum }}", "{{ '3'|int }} {{ 'x'|int }} {{ 'x'|int(9) }} {{ 3.9|int }} {{ '2.5'|float }} {{ 2.567|round(2) }} {{ 2.5|round }} {{ 2.4|round(0, 'ceil') }} {{ n|string + '!' }} {{ 3|float }}", "{{ 'aXbXc'|replace('X', '-') }} {{ 'aXbXc'|replace('X', '-', 1) }} {{ 'a b'|replace(' ', '') }}",
    "{{ tool|tojson }}", "{{ tool|tojson(indent=2) }}", "{{ d|tojson }} {{ [1, 2.0, 'é', none, true]|tojson }} {{ []|tojson }} {{ {}|tojson(indent=4) }} {{ 'a\"b\\\\c\\n'|tojson }}", "{{ d|tojson(sort_keys=true) }} {{ {'a': 1, 'b': [1]}|tojson(separators=(',', ':')) }} {{ 'é😀'|tojson(ensure_ascii=true) }} {{ [[1], []]|tojson(indent=1) }}", "{{ text|indent }}|{{ 'a\\nb\\n\\nc'|indent(2, true) }}|{{ 'a\\nb'|indent('>>') }}",
    "{{ d|items|list }} {{ '<a & \"b\">'|e }} {{ 'x'|safe }} {{ 'one two three'|wordcount }} {{ 'abc'|center(9) }}| {{ 'a long sentence here'|truncate(10) }}", "{% for a, g in people|groupby('age') %}{{ a }}:{{ g|map(attribute='name')|join(',') }};{% endfor %} {{ nums|batch(2)|list }} {{ [1,2,3,4,5]|slice(2)|list }} {{ [1,2,3]|batch(2, 'x')|list }}",
    # methods
    "{{ '  a '.strip() }}|{{ 'xxaxx'.strip('x') }}|{{ '\\n\\na\\n'.lstrip('\\n') }}|{{ 'a\\n\\n'.rstrip('\\n') }}|", "{{ 'a,b,c'.split(',') }} {{ 'a b  c'.split() }} {{ 'a,b,c'.split(',', 1) }} {{ 'a,b,c'.rsplit(',', 1) }} {{ 'x</think>y</think>z'.split('</think>')[-1] }} {{ ''.split(',') }} {{ ''.split() }} {{ 'a--b'.split('--')[0] }}", "{{ 'abc'.startswith('ab') }} {{ 'abc'.endswith('bc') }} {{ 'abc'.startswith(('x', 'a')) }} {{ 'abc'.upper() }} {{ 'a-b'.replace('-', '+') }} {{ 'hello'.find('l') }} {{ 'hello'.count('l') }} {{ 'ab'.join(['1', '2']) }} {{ 'a\\nb\\r\\nc'.splitlines() }} {{ 'they\\'re here'.title() }} {{ 'aBc'.capitalize() }} {{ '42'.isdigit() }} {{ 'prefix-x'.removeprefix('prefix-') }} {{ 'a=b=c'.partition('=') }} {{ '7'.zfill(3) }}",
    "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %} {{ d.keys()|list }} {{ d.values()|list }} {{ words.index('c') }} {{ words.count('a') }}",
    # statements
    "{% for w in words %}{{ loop.index }}{{ loop.index0 }}{{ loop.revindex }}{{ loop.first }}{{ loop.last }}{{ loop.length }}{{ loop.previtem }}{{ loop.nextitem }} {% endfor %}", "{% for w in words if w != 'A' %}{{ loop.index }}{{ w }}{% else %}none{% endfor %} {% for w in [] %}x{% else %}none{% endfor %} {% for w in missing %}x{% else %}undefined{% endfor %}", "{% for p in people recursive %}{{ p.name }}{% if loop.first and loop.depth == 1 %}[{{ loop([{'name': 'inner'}]) }}]{% endif %}{% endfor %}",
    "{% for i in range(5) %}{% if i == 1 %}{% continue %}{% endif %}{% if i == 4 %}{% break %}{% endif %}{{ i }}{% endfor %} {{ range(3)|list }} {{ range(1, 10, 3)|list }} {{ range(3, 0, -1)|list }}", "{% for a, b in [(1, 2), (3, 4)] %}{{ a }}{{ b }}{% endfor %} {% for a, (b, c) in [(1, (2, 3))] %}{{ a }}{{ b }}{{ c }}{% endfor %}", "{% for i in 'abc' %}{{ loop.cycle('x', 'y') }}{{ i }}{% endfor %} {% for k in d %}{{ k }}{% endfor %} {% for i in nums %}{{ loop.changed(i > 1) }}{% endfor %}",
    "{% set a = 1 %}{% for i in nums %}{% set a = a + i %}{{ a }}{% endfor %}{{ a }}", "{% set ns = namespace(a=1, b='x') %}{% for i in nums %}{% set ns.a = ns.a + i %}{% endfor %}{{ ns.a }}{{ ns.b }}{{ ns.zz }}", "{% if yes %}{% set inside = 5 %}{% endif %}{{ inside }}", "{% set a, b = 1, 2 %}{{ a }}{{ b }} {% set (c, d2) = (3, 4) %}{{ c }}{{ d2 }} {% set l = 1, 2 %}{{ l }}", "{% set block %}some {{ name }}{% endset %}[{{ block }}] {% set up | upper %}loud {{ name }}{% endset %}{{ up }}",
    "{% macro m(a, b=2, c='z') %}{{ a }}-{{ b }}-{{ c }}{% endmacro %}{{ m(1) }} {{ m(1, 3) }} {{ m(c=9, a=0) }} {{ m() }}", "{% macro rec(n) %}{{ n }}{% if n > 0 %},{{ rec(n - 1) }}{% endif %}{% endmacro %}{{ rec(3) }}", "{% macro wrap(tag) %}<{{ tag }}>{{ caller() }}</{{ tag }}>{% endmacro %}{% call wrap('b') %}bold {{ name }}{% endcall %} {% macro each(l) %}{% for x in l %}{{ caller(x) }}{% endfor %}{% endmacro %}{% call(v) each(nums) %}[{{ v }}]{% endcall %}",
    "{% macro outer() %}{{ later }}{% endmacro %}{% set later = 'seen' %}{{ outer() }}", "{% macro va() %}{{ varargs }} {{ kwargs }}{% endmacro %}{{ va(1, 2, k=3) }}", "{% filter upper %}shout {{ name }}{% endfilter %} {% filter trim|lower %}  ABC {% endfilter %}", "{% with a = 1, b = 2 %}{{ a }}{{ b }}{% endwith %}{{ a }}", "{% if n > 10 %}a{% elif n > 5 %}b{% elif yes %}c{% else %}d{% endif %}", "{% generation %}gen {{ name }}{% endgeneration %}!", "{% block thing %}in {{ name }}{% endblock thing %}",
    "{% set messages = [{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'}] %}{% for m in messages[::-1] %}{% set index = (messages|length - 1) - loop.index0 %}{{ index }}{{ m.role }}{% endfor %}", "{{ dict(a=1, b=2) }} {{ namespace(a=1).a }} {% set c = cycler('a', 'b') %}{{ c.next() }}{{ c.next() }}{{ c.next() }} {% set j = joiner('|') %}{{ j() }}a{{ j() }}b{{ j() }}c", "{{ strftime_now('%Y') is string }}",
]
# what each of these does is fail, and the reader must too
FAILURES = [
    "{{ missing.x }}", "{{ missing['x'] }}", "{{ raise_exception('no ' ~ name) }}", "{{ 'a' + 1 }}", "{{ 1 / 0 }}", "{% for i in n %}{% endfor %}", "{{ name() }}", "{{ words.append('x') }}", "{{ missing + 'a' }}", "{{ 1 < 'a' }}",
    "{% if %}", "{% for x in %}", "{{ 'open }}", "{% endif %}", "{% if yes %}", "{{ x | nofilter }}", "{{ x is notest }}", "{% unknown %}", "{{ a b }}", "{% set name.x = 1 %}",
]


def call(name, args, id):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": args}}


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "The weather — now", "parameters": {"type": "object", "properties": {
    "place": {"type": "string", "description": "Where"}, "days": {"type": "integer"}, "unit": {"type": "string", "enum": ["C", "F"]},
    "opts": {"type": "object", "properties": {"a": {"type": "array", "items": {"type": "string"}}}}}, "required": ["place"]}}},
    {"type": "function", "function": {"name": "now", "description": "The time", "parameters": {"type": "object", "properties": {}}}}]
TALKS = [
    [{"role": "user", "content": "hi "}],
    [{"role": "system", "content": " Be brief.\n"}, {"role": "user", "content": "hi"}],
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}],
    [{"role": "system", "content": "s"}, {"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "reasoning_content": "why"}, {"role": "user", "content": "c"}],
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "", "reasoning_content": "look", "tool_calls": [call("get_weather", {"place": "Paris <b>", "days": 2, "opts": {"a": ["x"]}}, "call12345")]},
     {"role": "tool", "content": "{\"t\": 3}", "tool_call_id": "call12345", "name": "get_weather"}],
    [{"role": "system", "content": "s"}, {"role": "user", "content": "a"}, {"role": "assistant", "content": "I will.", "tool_calls": [call("get_weather", {"place": "Rome"}, "call12345"), call("now", {}, "call12346")]},
     {"role": "tool", "content": "3", "tool_call_id": "call12345", "name": "get_weather"}, {"role": "tool", "content": "14:00", "tool_call_id": "call12346", "name": "now"},
     {"role": "assistant", "content": "It is 3."}, {"role": "user", "content": "thanks"}],
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "<think>\nold\n</think>\n\nb"}, {"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "two"}]}],
]


def templates(paths):
    for path in paths:
        t = json.load(open(path, encoding="utf-8"))
        out = []
        for talk in TALKS:
            for tools in (None, TOOLS):
                for extra in ({}, {"enable_thinking": False}, {"enable_thinking": True}):
                    vars = {"messages": talk, "tools": tools, "add_generation_prompt": True, "bos_token": t["bos"], "eos_token": t["eos"], **extra}
                    if tools is None:
                        del vars["tools"]
                    try:
                        text = _compile_jinja_template(t["template"]).render(**vars)
                    except Exception as e:
                        text = None
                    out.append({"vars": vars, "text": text})
        json.dump(out, open(path + ".golden", "w", encoding="utf-8"), ensure_ascii=False)
        print(t["name"], sum(o["text"] is not None for o in out), "of", len(out), "rendered")


def main():
    if len(sys.argv) > 1:
        return templates(sys.argv[1:])
    out = []
    for s in SNIPPETS:
        out.append({"template": s, "text": _compile_jinja_template(s).render(**VARS)})
    for s in FAILURES:
        try:
            text = _compile_jinja_template(s).render(**VARS)
            raise SystemExit(f"{s!r} renders {text!r}")
        except (jinja2.TemplateError, TypeError, ZeroDivisionError, AttributeError):
            out.append({"template": s, "text": None})
    with open(os.path.join(HERE, "jinja_golden.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"vars": VARS, "cases": out}, f, ensure_ascii=False, indent=1)
    print(len(out), "cases")


if __name__ == "__main__":
    main()
