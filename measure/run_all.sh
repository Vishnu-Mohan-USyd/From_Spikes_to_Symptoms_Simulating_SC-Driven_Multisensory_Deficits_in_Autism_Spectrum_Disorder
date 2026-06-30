#!/usr/bin/env bash
# ============================================================================================
# Reproduce ALL 7 validations on the route-C tau40 10-model ensemble (seeds 42-51), end to end.
#
#   gates 1-6 (RATE / TBW / E-I / MEI / SBW-CRE / latency inverse-effectiveness): parallel_measure.py
#             runs them across BOTH local GPUs (5090=cuda:0 + A6000=cuda:1), aggregates mean/SD/SEM,
#             and renders the 6 house-style figures.
#   gate 7    (cue-reliability / MLE inverse-variance cue weighting): measured for the paper-faithful
#             gain_exp=1 HEADLINE and the gain_exp=2 transparency variant, then both figures rendered.
#
# Outputs:  JSONs -> ../results,  figures (png+svg) -> ../figures.
# Every gate asserts the frozen TBW/SBW md5 (80d33465 / 73b7d136) BEFORE==AFTER and weight bit-identity.
# Local GPUs only; no network, no sudo.
#
# Usage:  bash run_all.sh [SEED ...]        # default: 42 43 44 45 46 47 48 49 50 51
# ============================================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # <root>/measure
ROOT="$(dirname "$HERE")"
PLOTS="$ROOT/plots"
export MPLBACKEND=Agg
SEEDS="${*:-42 43 44 45 46 47 48 49 50 51}"
echo "=================================================================="
echo "[bundle] reproduce 7 validations   seeds: $SEEDS   $(date '+%F %H:%M:%S')"
echo "=================================================================="

echo; echo "### [1/3] gates 1-6  (parallel across both GPUs) ###"
python "$HERE/parallel_measure.py" $SEEDS

echo; echo "### [2/3] gate 7 cue-reliability  (gain_exp=1 headline + gain_exp=2 transparency) ###"
gpu=0
for s in $SEEDS; do
  for ge in 1 2; do
    CUDA_VISIBLE_DEVICES=$gpu python "$HERE/run_gate7_cuerel.py" --seed "$s" --gain_exp "$ge" --device cuda:0 \
      2>&1 | grep -iE "R.|wrote|SCORECARD|empty_trials" | sed "s/^/  s$s g$ge: /"
  done
  gpu=$((1 - gpu))                                        # alternate 5090/A6000 across seeds
done
python "$HERE/run_gate7_cuerel.py" --aggregate --gain_exp 1
python "$HERE/run_gate7_cuerel.py" --aggregate --gain_exp 2

echo; echo "### [3/3] gate 7 figures (house style) ###"
python "$PLOTS/run_gate7_ens.py" gate7_cuerel_g1
python "$PLOTS/run_gate7_ens.py" gate7_cuerel_g2

echo; echo "=================================================================="
echo "[bundle] DONE  $(date '+%F %H:%M:%S')   figures -> $ROOT/figures"
ls -1 "$ROOT/figures"/*.png 2>/dev/null
echo "=================================================================="
