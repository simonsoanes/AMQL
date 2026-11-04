#!/usr/bin/env bash
# Takes a subject out of a model and refines it, as far as the mechanical rules
# go. What comes next is a reviewer's: see docs/entail-optimisation.md.
#
#   extract-subject.sh <amql-cli> <container> <tokenizer-dir> <subject.json> <work-dir>
#
# To finish the round trip, once review is done:
#
#   amql-cli entail graft <work-dir>/current.entail.txt --host <checkpoint> --out <dir>
#   amql-cli encode <dir> --out <container>
set -euo pipefail

cli=$1
container=$2
tokenizer=$3
subject=$4
work=$5

"$cli" entail find "$container" --tokenizer "$tokenizer" --subject "$subject" --out "$work" --top 128 --verify
"$cli" entail extract "$container" --tokenizer "$tokenizer" --subject "$subject" --out "$work"
"$cli" entail optimise "$work"
"$cli" entail review "$work"
