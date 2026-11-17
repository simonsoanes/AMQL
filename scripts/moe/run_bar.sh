#!/bin/sh
# Branch-adapt-route on Qwen 3.5 0.8B: four topic experts beside the base FFN, routers
# trained two ways, the merged MoE, and an add-a-topic upgrade. Expects the experts
# trained by lora-train into $W/experts2/{code,math,structured,chat}, and the topic
# sets from make_topic_sets.py in $W/data.
#
#   sh scripts/moe/run_bar.sh [cli] [work-dir] [container]
set -e
CLI=${1:-./bin/amql-cli.exe}
W=${2:-containers/moe}
BASE=${3:-containers/qwen3.5-0.8b}
E="code=$W/experts2/code,math=$W/experts2/math,structured=$W/experts2/structured,chat=$W/experts2/chat"
E3="code=$W/experts2/code,math=$W/experts2/math,structured=$W/experts2/structured"
mkdir -p $W/routers $W/eval

# 1. Fixed mixtures: the base, each expert alone, weight averaging, task arithmetic, oracle.
$CLI moe-eval $BASE --experts $E --data $W/data2/all.eval.jsonl --mix base,experts,average,sum,oracle \
  --report $W/eval/fixed.json > $W/eval/fixed.txt

# 2. Routers on the routing split: BAR's LM loss and the topic labels, per token or held from the prompt.
$CLI moe-router $BASE --experts $E --data $W/data2/all.route.jsonl --method label --hold token --steps 300 --lr 3e-3 \
  --out $W/routers/label-token.safetensors --log $W/routers/label-token.log.jsonl > $W/routers/label-token.txt
$CLI moe-router $BASE --experts $E --data $W/data2/all.route.jsonl --method label --hold prompt --steps 300 --lr 3e-3 \
  --out $W/routers/label-prompt.safetensors --log $W/routers/label-prompt.log.jsonl > $W/routers/label-prompt.txt
$CLI moe-router $BASE --experts $E --data $W/data2/all.route.jsonl --method loss --hold token --steps 30 --lr 3e-3 \
  --out $W/routers/loss-token.safetensors --log $W/routers/loss-token.log.jsonl > $W/routers/loss-token.txt
$CLI moe-router $BASE --experts $E --data $W/data2/all.route.jsonl --method loss --hold prompt --steps 30 --lr 3e-3 \
  --out $W/routers/loss-prompt.safetensors --log $W/routers/loss-prompt.log.jsonl > $W/routers/loss-prompt.txt

R=$W/routers
$CLI moe-eval $BASE --experts $E --data $W/data2/all.eval.jsonl --mix "" --top-k 0,1,2 \
  --router "label-token=$R/label-token.safetensors,label-prompt=$R/label-prompt.safetensors,loss-token=$R/loss-token.safetensors,loss-prompt=$R/loss-prompt.safetensors" \
  --report $W/eval/routed.json > $W/eval/routed.txt

# 3. The merged MoE, scored as a plain container: it must match loss-token@all.
rm -rf $W/bar-moe
$CLI moe-merge $BASE --experts $E --router $R/loss-token.safetensors --out $W/bar-moe > $W/eval/merge.txt
$CLI moe-eval $W/bar-moe --data $W/data2/all.eval.jsonl --mix base --report $W/eval/merged.json > $W/eval/merged.txt

# 4. Upgrade: a router over three experts, then chat added with the router warm-started.
$CLI moe-router $BASE --experts $E3 --data $W/data2/all.route.jsonl --method loss --hold token --steps 30 --lr 3e-3 \
  --out $R/loss3.safetensors --log $R/loss3.log.jsonl > $R/loss3.txt
$CLI moe-router $BASE --experts $E --data $W/data2/all.route.jsonl --method loss --hold token --steps 10 --lr 3e-3 \
  --init $R/loss3.safetensors --out $R/loss3plus.safetensors --log $R/loss3plus.log.jsonl > $R/loss3plus.txt
$CLI moe-eval $BASE --experts $E3 --data $W/data2/all.eval.jsonl --mix average,oracle --top-k 0 \
  --router "loss3=$R/loss3.safetensors" --report $W/eval/three.json > $W/eval/three.txt
$CLI moe-eval $BASE --experts $E --data $W/data2/all.eval.jsonl --mix "" --top-k 0 \
  --router "loss3plus=$R/loss3plus.safetensors" --report $W/eval/upgrade.json > $W/eval/upgrade.txt
echo bar-done
