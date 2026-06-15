#!/usr/bin/env bash
# TASK #76 PATH B equivalence re-val — ship the <=10-min L8-full build or fall back to the
# certified 12.23-min L6 build. SAME #70 protocol, pre-registered LOCKED tolerances.
#  STEP 1  PRIMARY functional A/B: measure TBW/SBW/E-I on the L8-full candidate via the EXACT
#          apparatus that produced the GOLD baseline (delayfix forward, tau_nmda_inh=21.6,
#          g_rec(ep79)=0.1, plasticity OFF, batch 256, SAME seed read from bs256_ep79.json),
#          then compare vs the FROZEN gold baseline (out/bs256_ep79.json — do NOT re-measure)
#          at ANCHORED tols (tbw_curve 0.12, sbw_curve 0.36) AND at REGISTERED tols (0.10/0.10).
#  STEP 2  SUPPORTING MSI-proxy flag: _latest_sMSI.mean() (alias of _g_out_sMSI) on L8-full vs
#          certified L6 under one identical sync-bimodal drive @ same seed. Benign if ratio>=0.5.
# SERIAL on cuda:0 only (A6000/cuda:1 + H200 OFF-LIMITS). Self-gates on 0 compute procs first.
set -euo pipefail
HERE=/home/vishnu/coding_proj/fsts_5/fsts_perilog_20260607/val36_traj_20260614
MEAS=/home/vishnu/coding_proj/fsts_5/fsts_perilog_20260607/delayfix_20260614/measopt_20260614
PY=/home/vishnu/miniconda3/bin/python3
SEED=42          # READ from out/bs256_ep79.json (seed=42) — common-mode cancellation, do NOT pick a new seed
EPOCH=79
L8="$MEAS/ckpt_ep79_seed42_bs250_L8full.pt"       # CANDIDATE (test)
L6="$MEAS/ckpt_ep79_seed42_bs250.pt"              # certified L6 (MSI apples-to-apples)
BASE="out/bs256_ep79.json"                        # GOLD baseline — READ, do NOT re-measure
cd "$HERE"; mkdir -p out logs
[ -f "$L8" ]  || { echo "MISSING candidate L8-full: $L8"; exit 1; }
[ -f "$L6" ]  || { echo "MISSING certified L6: $L6"; exit 1; }
[ -f "$BASE" ]|| { echo "MISSING gold baseline json: $BASE"; exit 1; }

# ── self-gate: cuda:0 must have 0 compute procs (serial discipline) ──
NPROC=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 2>/dev/null | grep -c . || true)
[ "${NPROC:-0}" = "0" ] || { echo "ABORT: $NPROC compute proc(s) on cuda:0 — not launching"; nvidia-smi -i 0; exit 3; }
echo "RUN76 $(date +%H:%M:%S)  cuda:0 free (0 compute procs)  candidate=$L8  seed=$SEED ep=$EPOCH"

# ── STEP 1: measure the L8-full candidate EXACTLY as the gold baseline was produced ──
echo "--- STEP1 measure l8full_ep79 ($(date +%H:%M:%S)) ---"
CUDA_VISIBLE_DEVICES=0 VAL36_BUILD=delayfix "$PY" val36_traj.py \
  --ckpt "$L8" --tag l8full_ep79 --seed "$SEED" --epoch "$EPOCH" \
  --metrics tbw,sbw,ei --out_json out/l8full_ep79.json 2>&1 | tee logs/l8full_ep79.log

# ── STEP 1 compare A: ANCHORED tols (locked: tbw_curve 0.12, sbw_curve 0.36; others registered) ──
echo "=== STEP1 A/B  ANCHORED tols (tbw_curve=0.12 sbw_curve=0.36; box50=40 fwhm=40 peak=0.07 p0=0.07 hw=5.0 ei_rel=0.20) ==="
CUDA_VISIBLE_DEVICES="" "$PY" compare_ab.py \
  --baseline "$BASE" --test out/l8full_ep79.json \
  --tbw_curve_tol 0.12 --sbw_curve_tol 0.36 \
  --out_json out/ab_verdict_l8full_anchored.json 2>&1 | tee logs/cmp_l8full_anchored.log || true

# ── STEP 1 compare B: REGISTERED tols (all defaults: tbw_curve 0.10, sbw_curve 0.10) — full delta table vs registered ──
echo "=== STEP1 A/B  REGISTERED tols (tbw_curve=0.10 sbw_curve=0.10; others identical) ==="
CUDA_VISIBLE_DEVICES="" "$PY" compare_ab.py \
  --baseline "$BASE" --test out/l8full_ep79.json \
  --out_json out/ab_verdict_l8full_registered.json 2>&1 | tee logs/cmp_l8full_registered.log || true

# ── STEP 2: SUPPORTING MSI-proxy flag (L8full vs certified L6) ──
echo "=== STEP2 MSI proxy  L8full vs certified L6 (seed $SEED, benign if ratio>=0.5) ==="
CUDA_VISIBLE_DEVICES=0 VAL36_BUILD=delayfix "$PY" msi_proxy.py \
  --l8full "$L8" --l6 "$L6" --seed "$SEED" --epoch "$EPOCH" --batch 256 \
  --out_json out/msi_proxy_l8full_vs_l6.json 2>&1 | tee logs/msi_proxy.log || true

echo "RUN76 DONE $(date +%H:%M:%S)  -> out/{l8full_ep79,ab_verdict_l8full_anchored,ab_verdict_l8full_registered,msi_proxy_l8full_vs_l6}.json"
