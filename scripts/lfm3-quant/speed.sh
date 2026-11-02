#!/usr/bin/env bash
# usage: speed.sh <label>=<gguf> ...   GPU (all layers) and CPU-only (16 threads) pp512/tg128, 3 reps.
BIN=/d/Dev/Tools/llama.cpp/build/bin
OUT=D:/Models/AMQL/lfm3-work/speed.md
[ -f $OUT ] || { echo "| model | size MB | GPU pp512 | GPU tg128 | CPU pp512 | CPU tg128 |" > $OUT; echo "|---|---|---|---|---|---|" >> $OUT; }
for spec in "$@"; do
  label="${spec%%=*}"; f="${spec#*=}"
  size=$(du -m "$f" | cut -f1)
  g=$($BIN/llama-bench.exe -m "$f" -ngl 99 -p 512 -n 128 -r 3 -o csv 2>/dev/null | tail -2 | awk -F, '{print $(NF-1)}' | tr -d '"' | xargs printf '%.0f ')
  c=$($BIN/llama-bench.exe -m "$f" -ngl 0 -t 16 -p 512 -n 128 -r 2 -o csv 2>/dev/null | tail -2 | awk -F, '{print $(NF-1)}' | tr -d '"' | xargs printf '%.1f ')
  set -- $g; gp=$1; gt=$2
  set -- $c; cp=$1; ct=$2
  echo "| $label | $size | $gp | $gt | $cp | $ct |" | tee -a $OUT
done
