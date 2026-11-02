#!/usr/bin/env bash
# usage: eval_model.sh <model.gguf> <label> [--no-tools] [--no-bench]
# Appends a row to D:/Models/AMQL/lfm3-work/results.md:
#   label | size | wiki PPL | OOD PPL | OOD mean KLD vs BF16 | same top-1 | tool-eval | pp512/tg128
set -u
GGUF="$1"; LABEL="$2"; shift 2
TOOLS=1; BENCH=1
for a in "$@"; do [ "$a" = "--no-tools" ] && TOOLS=0; [ "$a" = "--no-bench" ] && BENCH=0; done
BIN=/d/Dev/Tools/llama.cpp/build/bin
W=D:/Models/AMQL/lfm3-work
LOG=$W/logs; mkdir -p $LOG
SIZE=$(du -m "$GGUF" | cut -f1)
PPL=$($BIN/llama-perplexity.exe -m "$GGUF" -f D:/Models/AMQL/eval/wikitext/wikitext-2-raw/wiki.test.raw -ngl 99 -c 2048 --chunks 40 2>&1 | tee $LOG/$LABEL-ppl.log | grep -oE 'PPL = [0-9.]+' | sed 's/PPL = //')
$BIN/llama-perplexity.exe -m "$GGUF" -ngl 99 -c 1024 --kl-divergence --kl-divergence-base D:/Models/AMQL/eval/kld/lfm3-bf16.kld > $LOG/$LABEL-kld.log 2>&1
OODPPL=$(grep -oE 'Mean PPL\(Q\) +: +[0-9.]+' $LOG/$LABEL-kld.log | grep -oE '[0-9.]+$')
KLD=$(grep -oE 'Mean +KLD: +[0-9.]+' $LOG/$LABEL-kld.log | grep -oE '[0-9.]+$')
TOP=$(grep -oE 'Same top p: +[0-9.]+' $LOG/$LABEL-kld.log | grep -oE '[0-9.]+$')
$BIN/llama-perplexity.exe -m "$GGUF" -ngl 99 -c 1024 --kl-divergence --kl-divergence-base D:/Models/AMQL/eval/kld/chat-bf16.kld > $LOG/$LABEL-kldchat.log 2>&1
CKLD=$(grep -oE 'Mean +KLD: +[0-9.]+' $LOG/$LABEL-kldchat.log | grep -oE '[0-9.]+$')
CTOP=$(grep -oE 'Same top p: +[0-9.]+' $LOG/$LABEL-kldchat.log | grep -oE '[0-9.]+$')
TG="-"
if [ $BENCH = 1 ]; then
  TG=$($BIN/llama-bench.exe -m "$GGUF" -ngl 99 -p 512 -n 128 -o csv 2>/dev/null | tail -2 | awk -F, '{print $(NF-1)}' | tr -d '"' | xargs printf '%.0f\n' | paste -sd/)
fi
TOOL="-"
if [ $TOOLS = 1 ]; then
  $BIN/llama-server.exe -m "$GGUF" -ngl 99 -c 16384 --jinja --port 18091 --host 127.0.0.1 -np 1 > $LOG/$LABEL-server.log 2>&1 &
  for i in $(seq 1 60); do curl -s http://127.0.0.1:18091/health 2>/dev/null | grep -q ok && break; sleep 1; done
  TOOL=$(/d/Dev/AMQL/bin/Release/amql-cli.exe tool-eval --endpoint http://127.0.0.1:18091/v1 --cases /d/Dev/AMQL/tests/Amql.Tests/fixtures/tool-eval-cases.jsonl --report $LOG/$LABEL-tools.json 2>&1 | tee $LOG/$LABEL-tools.log | grep -oE '^tool-eval: [0-9]+/[0-9]+' | sed 's/tool-eval: //')
  powershell -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force" >/dev/null 2>&1
  sleep 1
fi
LINE="| $LABEL | ${SIZE} MB | $PPL | $OODPPL | $KLD | ${TOP}% | $CKLD | ${CTOP}% | $TOOL | $TG |"
echo "$LINE" | tee -a $W/results2.md
