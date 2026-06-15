#!/usr/bin/env bash
# Dose-response BATCHED across (ckpt x dose) on the H200 (Rule 6 / lead spec:
# NOT a sequential dose loop). For each metric we launch ALL unique-dose x 5-seed
# jobs at once, wait, then the next metric. 7 unique doses x 5 seeds = 35 procs/metric.
# Unique = the 4 dose configs byte-identical to jointA are skipped (already measured).
set -uo pipefail
PHASE2=${PHASE2:-/scratch/phase2}; CODE=$PHASE2/code; SBWD=$PHASE2/sbw
APP=${APP:-/scratch/eval_apparatus}; CK=${CK:-/scratch/fsts_retrain_asym_20260530/checkpoints_asymdelay}
CFG=$PHASE2/cfg; OUT=$PHASE2/out; LOG=$PHASE2/logs; PY=${PY:-/scratch/calibenv/bin/python}
PREFIX=asymrc; EPOCH=80; NTRIALS=${NTRIALS:-50}; OMIN=-30; OMAX=30; OSTEP=2; SEEDS="0 1 2 3 4"
mkdir -p "$OUT/tbw" "$OUT/sbw" "$OUT/ei" "$OUT/agg" "$LOG"

echo "##### DOSE BATCH @ $(date) PY=$PY #####"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | head -1
"$PY" "$CODE/make_configs.py" --out_dir "$CFG" --doses > "$LOG/make_configs_dose.log" 2>&1

DOSES=()
for cf in "$CFG"/dose_*.json; do
  cmp -s "$cf" "$CFG/jointA.json" && { echo "SKIP $(basename "$cf") == jointA"; continue; }
  DOSES+=("$cf")
done
echo "UNIQUE_DOSES=${#DOSES[@]}"; for cf in "${DOSES[@]}"; do echo "  $(basename "$cf")"; done

launch_batch () {   # $1=metric label, rest handled per-metric below
  :
}

# ---------- TBW batch ----------
pids=()
for cf in "${DOSES[@]}"; do tag=$(basename "$cf" .json)
  for k in $SEEDS; do
    "$PY" "$CODE/tbw_routec_curve.py" --epoch $EPOCH --prefix $PREFIX --ntrials $NTRIALS \
      --ckpt_dir "$CK" --out_dir "$OUT/tbw" --omin $OMIN --omax $OMAX --ostep $OSTEP \
      --only_model "$k" --config_tag "$tag" --override_json "$cf" \
      > "$LOG/tbw_${tag}_m${k}.log" 2>&1 &
    pids+=($!)
  done
done
echo "TBW batch: ${#pids[@]} jobs @ $(date +%H:%M:%S)"
for p in "${pids[@]}"; do wait "$p"; done
echo "TBW batch DONE @ $(date +%H:%M:%S)"

# ---------- SBW batch ----------
pids=()
for cf in "${DOSES[@]}"; do tag=$(basename "$cf" .json)
  for k in $SEEDS; do
    "$PY" "$SBWD/sbw_routec_run.py" --eval_app "$APP" --code_dir "$CODE" \
      --ckpt "$CK/${PREFIX}_m${k}_ep${EPOCH}.pt" --tag "${PREFIX}_m${k}" \
      --out_json "$OUT/sbw/sbw_${tag}_m${k}.json" --override_json "$cf" \
      > "$LOG/sbw_${tag}_m${k}.log" 2>&1 &
    pids+=($!)
  done
done
echo "SBW batch: ${#pids[@]} jobs @ $(date +%H:%M:%S)"
for p in "${pids[@]}"; do wait "$p"; done
echo "SBW batch DONE @ $(date +%H:%M:%S)"

# ---------- EI batch ----------
pids=()
for cf in "${DOSES[@]}"; do tag=$(basename "$cf" .json)
  for k in $SEEDS; do
    "$PY" "$SBWD/ei_routec_run.py" --eval_app "$APP" --code_dir "$CODE" \
      --ckpt "$CK/${PREFIX}_m${k}_ep${EPOCH}.pt" --tag "${PREFIX}_m${k}" \
      --out_json "$OUT/ei/ei_${tag}_m${k}.json" --override_json "$cf" \
      > "$LOG/ei_${tag}_m${k}.log" 2>&1 &
    pids+=($!)
  done
done
echo "EI batch: ${#pids[@]} jobs @ $(date +%H:%M:%S)"
for p in "${pids[@]}"; do wait "$p"; done
echo "EI batch DONE @ $(date +%H:%M:%S)"

# ---------- aggregate each dose ----------
for cf in "${DOSES[@]}"; do tag=$(basename "$cf" .json)
  "$PY" "$CODE/tbw_aggregate.py" --code_dir "$CODE" \
     --glob "$OUT/tbw/${PREFIX}_tbw_ep${EPOCH}_${tag}_m*.npz" --prefix "$tag" \
     --out_json "$OUT/agg/tbw_${tag}.json" > "$LOG/agg_tbw_${tag}.log" 2>&1
  "$PY" "$SBWD/aggregate_sbw.py" --eval_app "$APP" --code_dir "$CODE" \
     --glob "$OUT/sbw/sbw_${tag}_m*.json" --prefix "$tag" \
     --out_json "$OUT/agg/sbw_${tag}.json" > "$LOG/agg_sbw_${tag}.log" 2>&1
  "$PY" "$SBWD/ei_aggregate.py" \
     --glob "$OUT/ei/ei_${tag}_m*.json" --prefix "$tag" \
     --out_json "$OUT/agg/ei_${tag}.json" > "$LOG/agg_ei_${tag}.log" 2>&1
  echo "aggregated $tag"
done
echo "DOSE_BATCH_DONE @ $(date +%H:%M:%S)"
