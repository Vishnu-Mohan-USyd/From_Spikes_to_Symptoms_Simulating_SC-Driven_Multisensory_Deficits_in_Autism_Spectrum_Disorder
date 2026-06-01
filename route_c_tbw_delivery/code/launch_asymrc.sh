#!/usr/bin/env bash
# task#51 route-c retrain — 5 ckpts, FRESH ep0->80, asym-delay config + tau_nmda_inh=25.
# STAGED: run ONLY on lead GO.  Usage: bash launch_asymrc.sh [prefix]   (default prefix=asymrc)
# New prefix protects the existing asymd_* ep80 ckpts (#52 inference-precheck baseline).
set -u
CODE=/scratch/fsts_retrain_asym_20260530/code
LOGS=/scratch/fsts_retrain_asym_20260530/logs
PY=/scratch/calibenv/bin/python
PREFIX="${1:-asymrc}"
EXPECT=6b649d153850cfc3c872f4dcafd71b95   # route-c Training.py md5 on dev (verified)

cd "$CODE" || { echo "ABORT: no $CODE"; exit 1; }
GOT=$(md5sum Training.py | cut -d' ' -f1)
if [ "$GOT" != "$EXPECT" ]; then
  echo "ABORT: Training.py md5 $GOT != route-c $EXPECT (refusing to launch)"; exit 1
fi
echo "preflight OK: Training.py md5=$GOT  prefix=$PREFIX  n_epochs=80  seeds=42..46  PY=$PY"

mkdir -p "$LOGS"
: > "$LOGS/${PREFIX}_pids.txt"
for M in 0 1 2 3 4; do
  nohup "$PY" run_task192_disynaptic_ep40_retrain.py \
      --model_idx "$M" --prefix "$PREFIX" --n_epochs 80 \
      > "$LOGS/${PREFIX}_m${M}.log" 2>&1 &
  echo "m${M} pid=$!" >> "$LOGS/${PREFIX}_pids.txt"
done
cat "$LOGS/${PREFIX}_pids.txt"
echo "LAUNCHED_${PREFIX}_5"
