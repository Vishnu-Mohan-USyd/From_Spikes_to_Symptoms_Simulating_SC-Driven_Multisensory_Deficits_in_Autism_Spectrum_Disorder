#!/usr/bin/env bash
# TASK #70 core runner — measure baseline (bs256) + test (bs250) through ONE apparatus, then A/B.
# Locked protocol (PREREG_70.md): delayfix forward, SAME seed for both (common-mode noise),
# ep79, metrics tbw,sbw,ei, registered tolerances. SERIAL on cuda:0 (one ckpt at a time).
# Usage: run_70.sh <BASELINE_bs256_ckpt> [SEED=42] [EPOCH=79] [TEST_bs250_ckpt]
# Run ONLY after the lead confirms the baseline ckpt is on disk and cuda:0 is free.
set -euo pipefail   # fail fast if a measurement crashes (don't waste run 2 / compare on stale JSON)
HERE=/home/vishnu/coding_proj/fsts_5/fsts_perilog_20260607/val36_traj_20260614
MEAS=/home/vishnu/coding_proj/fsts_5/fsts_perilog_20260607/delayfix_20260614/measopt_20260614
PY=/home/vishnu/miniconda3/bin/python3
BASE_CK="${1:?baseline bs256 ckpt path}"
SEED="${2:-42}"
EPOCH="${3:-79}"
TEST_CK="${4:-$MEAS/ckpt_ep79_seed42_bs250.pt}"
cd "$HERE"; mkdir -p out logs
[ -f "$BASE_CK" ] || { echo "MISSING baseline ckpt: $BASE_CK"; exit 1; }
[ -f "$TEST_CK" ] || { echo "MISSING test ckpt: $TEST_CK"; exit 1; }
echo "RUN70 $(date +%H:%M:%S)  baseline=$BASE_CK  test=$TEST_CK  seed=$SEED  ep=$EPOCH"

run_one () {  # $1=ckpt  $2=tag
  echo "--- measuring $2 ($(date +%H:%M:%S)) ---"
  CUDA_VISIBLE_DEVICES=0 VAL36_BUILD=delayfix "$PY" val36_traj.py \
    --ckpt "$1" --tag "$2" --seed "$SEED" --epoch "$EPOCH" \
    --metrics tbw,sbw,ei --out_json "out/$2.json" 2>&1 | tee "logs/$2.log"
}

run_one "$BASE_CK" "bs256_ep${EPOCH}"     # baseline FIRST
run_one "$TEST_CK" "bs250_ep${EPOCH}"     # then test (serial — no GPU contention)

echo "=== A/B verdict ($(date +%H:%M:%S)) ==="
CUDA_VISIBLE_DEVICES="" "$PY" compare_ab.py \
  --baseline "out/bs256_ep${EPOCH}.json" --test "out/bs250_ep${EPOCH}.json" \
  --out_json "out/ab_verdict_ep${EPOCH}.json" || true   # exit 2=NO-GO must not abort the script
echo "RUN70 DONE $(date +%H:%M:%S) -> out/ab_verdict_ep${EPOCH}.json"
