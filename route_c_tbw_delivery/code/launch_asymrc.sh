#!/usr/bin/env bash
# task#51 route-c retrain — 5 ckpts, FRESH ep0->80, asym-delay config + task#14 TBW fix
# (tau_nmda_inh=45 GluN2A-fast + FF->IN NMDA short-term depression scale_F=0.8, baked into Training.py)
# + task#200 Phase-6 Turrigiano homeostatic exc synaptic-scaling scaffold.
# STAGED: run ONLY on lead GO.  Usage: bash launch_asymrc.sh [prefix]   (default prefix=rcfix45std_hss)
# Default prefix rcfix45std_hss PRESERVES the rcfix45std box-evidence ckpts AND the
# asymd_* / #52 baselines. NEVER use prefix 'asymrc' (clobbers #52 baseline) — guarded below.
set -u
CODE=/scratch/fsts_retrain_asym_20260530/code
LOGS=/scratch/fsts_retrain_asym_20260530/logs
PY=/scratch/calibenv/bin/python
PREFIX="${1:-rcfix45std_hss}"   # task#200 Phase-6; explicit arg overrides. NEVER 'asymrc'.
[ "$PREFIX" = "asymrc" ] && { echo "ABORT: prefix 'asymrc' is forbidden (clobbers #52 baseline)"; exit 1; }
EXPECT=8dfabd96a1cfa5caea9f6fc665f06a10   # task#200 Phase-6 PER-BATCH (Option A) + purple silent-neuron rule (deficit clamp[-1,1], <5%r* freeze scale-up, 1e-9 floor, r* tracker-units); prior per-batch-no-silent ad4de5a3, per-frame 90b9e335, plain route-c 3a29fa88, pristine 6b649d15

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
      --hss_sensor net_rate --hss_r_target 30.77 --hss_alpha 2e-3 --hss_step_clip 0.10 \
      > "$LOGS/${PREFIX}_m${M}.log" 2>&1 &
  echo "m${M} pid=$!" >> "$LOGS/${PREFIX}_pids.txt"
done
cat "$LOGS/${PREFIX}_pids.txt"
echo "LAUNCHED_${PREFIX}_5"
