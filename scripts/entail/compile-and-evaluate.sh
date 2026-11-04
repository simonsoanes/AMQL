#!/usr/bin/env bash
# Compiles mathematics.entail.txt twice — into a new, empty model, and in situ
# into a Qwen3.5 checkpoint — and evaluates both, with the unmodified host as
# the baseline. Nothing here trains anything.
#
#   compile-and-evaluate.sh <amql-cli> <host-checkpoint-dir> <work-dir> [prose.txt]
#
# Inference runs on the CPU: the compiled block depends on float32 being exact
# for whole numbers, which the fp16 GPU weight path is not.
set -euo pipefail

cli=$1
host=$2
work=$3
prose=${4:-}
here=$(cd "$(dirname "$0")" && pwd)
program="$here/mathematics.entail.txt"
mkdir -p "$work"

"$cli" entail check "$program" --context 32768

# 1. A new, empty model: statements in, weights out, then test those weights.
"$cli" entail compile "$program" --out "$work/empty" --context 32768
"$cli" encode "$work/empty" --out "$work/empty.container"
python "$here/make_cases.py" empty 25 1 > "$work/empty.jsonl"
"$cli" entail eval "$work/empty.container" --tokenizer "$work/empty" \
    --cases "$work/empty.jsonl" --report "$work/empty.report.json"

# 2. The same program, grafted into the host.
"$cli" entail graft "$program" --host "$host" --out "$work/graft" --context 32768
"$cli" encode "$work/graft" --out "$work/graft.container"
"$cli" encode "$host" --out "$work/base.container"

# 3. Before and after, on the same cases.
python "$here/make_cases.py" direct 20 2 > "$work/direct.jsonl"
python "$here/make_cases.py" chat 20 3 > "$work/chat.jsonl"
text=()
if [[ -n "$prose" ]]; then
    text=(--text "$prose" --text-tokens 768)
fi
for model in base graft; do
    for suite in direct chat; do
        echo "=== $model: $suite ==="
        "$cli" entail eval "$work/$model.container" --tokenizer "$host" \
            --cases "$work/$suite.jsonl" --report "$work/$model.$suite.json"
    done
    if [[ -n "$prose" ]]; then
        "$cli" entail eval "$work/$model.container" --tokenizer "$host" \
            "${text[@]}" --report "$work/$model.text.json"
    fi
done
