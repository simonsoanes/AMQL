#!/usr/bin/env bash
# usage: sensitivity.sh <base-type> <label=regex:type> ...
# Each spec alone at its type, everything else at <base-type>; mixed imatrix.
# Appends: base,label,sizeMB,oodKLD,oodTop,chatKLD,chatTop to sens2.csv
set -u
BASE="$1"; shift
W=D:/Models/AMQL/lfm3-work
BIN=/d/Dev/Tools/llama.cpp/build/bin
OUT=$W/sens2.csv
SRC=${SRC:-$W/ablate-a0.05}
IMAT=${IMAT:-$W/merged-a005.mix.imatrix.gguf}
for spec in "$@"; do
  label="${spec%%=*}"; rest="${spec#*=}"; regex="${rest%:*}"; low="${rest##*:}"
  R=$W/recipes/sens-$BASE-$label.txt
  printf '%s %s\n* %s\n' "$regex" "$low" "$BASE" > $R
  G=$W/sens-tmp.gguf
  AMQL_GGML=$BIN D:/Models/AMQL/amql-bin/amql-cli.exe to-gguf $SRC --out $G --recipe $R --imatrix $IMAT --force > $W/logs/sens-quant.log 2>&1 || { echo "$label,FAILED"; tail -2 $W/logs/sens-quant.log; continue; }
  SIZE=$(du -m $G | cut -f1)
  $BIN/llama-perplexity.exe -m $G -ngl 99 -c 1024 --kl-divergence --kl-divergence-base D:/Models/AMQL/eval/kld/lfm3-bf16.kld > $W/logs/sens-kld.log 2>&1
  K1=$(grep -oE 'Mean +KLD: +[0-9.]+' $W/logs/sens-kld.log | grep -oE '[0-9.]+$'); T1=$(grep -oE 'Same top p: +[0-9.]+' $W/logs/sens-kld.log | grep -oE '[0-9.]+$')
  $BIN/llama-perplexity.exe -m $G -ngl 99 -c 1024 --kl-divergence --kl-divergence-base D:/Models/AMQL/eval/kld/chat-bf16.kld > $W/logs/sens-kld.log 2>&1
  K2=$(grep -oE 'Mean +KLD: +[0-9.]+' $W/logs/sens-kld.log | grep -oE '[0-9.]+$'); T2=$(grep -oE 'Same top p: +[0-9.]+' $W/logs/sens-kld.log | grep -oE '[0-9.]+$')
  echo "$BASE,$label,$SIZE,$K1,$T1,$K2,$T2" | tee -a $OUT
done
