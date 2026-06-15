#!/usr/bin/env bash
# TASK #70 EMPIRICAL ANCHOR (borderline branch of PREREG_70.md).
# The core verdict was borderline (SBW curve max|Δ|=0.12 vs 0.10; TBW curve at the 0.10 edge).
# Ground the run-to-run noise floor: re-measure the SAME bs256 BASELINE at a SECOND seed, take the
# same-model two-seed curve max|Δ| as the floor, then RE-JUDGE bs256-vs-bs250 with
# tol = max(registered, 3×observed_floor). Same-model/different-seed isolates PURE measurement noise.
set -euo pipefail
HERE=/home/vishnu/coding_proj/fsts_5/fsts_perilog_20260607/val36_traj_20260614
MEAS=/home/vishnu/coding_proj/fsts_5/fsts_perilog_20260607/delayfix_20260614/measopt_20260614
PY=/home/vishnu/miniconda3/bin/python3
BASE="$MEAS/ckpt_ep79_seed42_bs256.pt"
cd "$HERE"; mkdir -p out logs
echo "ANCHOR70 $(date +%H:%M:%S)  second-seed baseline re-measure (seed 43, tbw+sbw)"

# 1) second-seed baseline measurement — the borderline metrics are the two curves (tbw,sbw)
CUDA_VISIBLE_DEVICES=0 VAL36_BUILD=delayfix "$PY" val36_traj.py \
  --ckpt "$BASE" --tag bs256_ep79_s43 --seed 43 --epoch 79 \
  --metrics tbw,sbw --out_json out/bs256_ep79_s43.json 2>&1 | tee logs/bs256_ep79_s43.log

# 2) floor = same-model two-seed curve max|Δ| (pure measurement noise)
CUDA_VISIBLE_DEVICES="" "$PY" compare_ab.py \
  --baseline out/bs256_ep79.json --test out/bs256_ep79_s43.json \
  --out_json out/floor_anchor.json || true

# 3) anchored tols = max(registered 0.10, 3× observed floor); RE-JUDGE the real bs256-vs-bs250
read TBW_TOL SBW_TOL < <("$PY" -c "import json;fl=json.load(open('out/floor_anchor.json'));ft=fl.get('tbw_curve_maxabs') or 0.0;fs=fl.get('sbw_curve_maxabs') or 0.0;print(round(max(0.10,3*ft),4),round(max(0.10,3*fs),4))")
echo "ANCHOR70 floor->anchored tols: tbw_curve_tol=$TBW_TOL  sbw_curve_tol=$SBW_TOL"
echo "=== RE-JUDGED verdict (anchored curve tols; all other tols unchanged) ==="
CUDA_VISIBLE_DEVICES="" "$PY" compare_ab.py \
  --baseline out/bs256_ep79.json --test out/bs250_ep79.json \
  --tbw_curve_tol "$TBW_TOL" --sbw_curve_tol "$SBW_TOL" \
  --out_json out/ab_verdict_ep79_anchored.json || true
echo "ANCHOR70 DONE $(date +%H:%M:%S) -> out/ab_verdict_ep79_anchored.json"
