#!/usr/bin/env bash
# ============================================================================
# Phase-2 inference screen orchestrator  (runs ON the dev H200 pod)
# Applies Phase-1 biology-corrected hparams to the 5 asymrc ep80 ckpts AT
# INFERENCE (post-load override via routec_overrides; NO retrain, canonical
# md5-gated Training.py untouched). Measures TBW(+per-SOA #peaks)/SBW/E-I.
#
# Per condition: 5 seeds parallel within each metric (Rule 6), metrics run
# sequentially (TBW->SBW->EI) to avoid CPU thrash on the pod's small core req.
# Conditions: baseline (no-op) -> jointA (gNMDA 0.7) -> jointB (gNMDA rescaled
# from baseline weighted v_rep) -> [doses].
#
# Usage:  run_phase2_screen.sh <mode>
#   smoke          3-seed fast plumbing check (baseline + jointA), coarse grid
#   baseline       baseline only
#   joint          jointA + jointB (requires baseline EI agg already present)
#   baseline_joint baseline -> jointA -> jointB   (DEFAULT)
#   dose           all dose-response configs (requires jointA cfg)
#   all            baseline -> joint -> dose
# ============================================================================
set -uo pipefail

PHASE2=${PHASE2:-/scratch/phase2}
CODE=$PHASE2/code
SBWD=$PHASE2/sbw
APP=${APP:-/scratch/eval_apparatus}
CK=${CK:-/scratch/fsts_retrain_asym_20260530/checkpoints_asymdelay}
CFG=$PHASE2/cfg
OUT=$PHASE2/out
LOG=$PHASE2/logs
PY=${PY:-/scratch/calibenv/bin/python}
PREFIX=asymrc
EPOCH=80

NTRIALS=${NTRIALS:-50}
OMIN=${OMIN:--30}; OMAX=${OMAX:-30}; OSTEP=${OSTEP:-2}   # SOA -300..+300ms step 20ms
SEEDS=${SEEDS:-"0 1 2 3 4"}

mkdir -p "$OUT/tbw" "$OUT/sbw" "$OUT/ei" "$OUT/agg" "$LOG" "$CFG"

# ---- wait on a set of PIDs, count failures -------------------------------
_waitall () { local f=0 p; for p in "$@"; do wait "$p" || f=$((f+1)); done; return $f; }

# ---- run one condition: tag + override-json-path ("" = baseline) ---------
run_condition () {
  local tag="$1" ovr="$2"
  local ovrarg=(); [ -n "$ovr" ] && ovrarg=(--override_json "$ovr")
  echo "=== CONDITION $tag  (override='${ovr:-BASELINE}')  ntrials=$NTRIALS seeds='$SEEDS'  @ $(date +%H:%M:%S) ==="
  local k pids fail=0

  # -- TBW: 5 seeds parallel (one --only_model per proc) --
  pids=()
  for k in $SEEDS; do
    "$PY" "$CODE/tbw_routec_curve.py" --epoch $EPOCH --prefix $PREFIX --ntrials $NTRIALS \
        --ckpt_dir "$CK" --out_dir "$OUT/tbw" --omin $OMIN --omax $OMAX --ostep $OSTEP \
        --only_model "$k" --config_tag "$tag" "${ovrarg[@]}" \
        > "$LOG/tbw_${tag}_m${k}.log" 2>&1 &
    pids+=($!)
  done
  _waitall "${pids[@]}" || { fail=$((fail+$?)); echo "  [WARN] $tag TBW had failures"; }

  # -- SBW: 5 seeds parallel (one ckpt per proc) --
  pids=()
  for k in $SEEDS; do
    "$PY" "$SBWD/sbw_routec_run.py" --eval_app "$APP" --code_dir "$CODE" \
        --ckpt "$CK/${PREFIX}_m${k}_ep${EPOCH}.pt" --tag "${PREFIX}_m${k}" \
        --out_json "$OUT/sbw/sbw_${tag}_m${k}.json" "${ovrarg[@]}" \
        > "$LOG/sbw_${tag}_m${k}.log" 2>&1 &
    pids+=($!)
  done
  _waitall "${pids[@]}" || { fail=$((fail+$?)); echo "  [WARN] $tag SBW had failures"; }

  # -- EI: 5 seeds parallel (one ckpt per proc) --
  pids=()
  for k in $SEEDS; do
    "$PY" "$SBWD/ei_routec_run.py" --eval_app "$APP" --code_dir "$CODE" \
        --ckpt "$CK/${PREFIX}_m${k}_ep${EPOCH}.pt" --tag "${PREFIX}_m${k}" \
        --out_json "$OUT/ei/ei_${tag}_m${k}.json" "${ovrarg[@]}" \
        > "$LOG/ei_${tag}_m${k}.log" 2>&1 &
    pids+=($!)
  done
  _waitall "${pids[@]}" || { fail=$((fail+$?)); echo "  [WARN] $tag EI had failures"; }

  echo "  [$tag] measurement done (job failures=$fail) @ $(date +%H:%M:%S); aggregating..."
  "$PY" "$CODE/tbw_aggregate.py" --code_dir "$CODE" \
      --glob "$OUT/tbw/${PREFIX}_tbw_ep${EPOCH}_${tag}_m*.npz" \
      --prefix "$tag" --out_json "$OUT/agg/tbw_${tag}.json" 2>&1 | tee "$LOG/agg_tbw_${tag}.log"
  "$PY" "$SBWD/aggregate_sbw.py" --eval_app "$APP" --code_dir "$CODE" \
      --glob "$OUT/sbw/sbw_${tag}_m*.json" \
      --prefix "$tag" --out_json "$OUT/agg/sbw_${tag}.json" 2>&1 | tee "$LOG/agg_sbw_${tag}.log"
  "$PY" "$SBWD/ei_aggregate.py" \
      --glob "$OUT/ei/ei_${tag}_m*.json" \
      --prefix "$tag" --out_json "$OUT/agg/ei_${tag}.json" 2>&1 | tee "$LOG/agg_ei_${tag}.log"
  return $fail
}

emit_jointB () {  # compute gNMDA rescale from baseline weighted v_rep, write jointB.json
  local vrep
  vrep=$("$PY" -c "import json;print(json.load(open('$OUT/agg/ei_baseline.json'))['v_rep_weighted_sync_mean'])") || {
    echo "[ERR] cannot read baseline weighted v_rep (need baseline EI agg first)"; return 2; }
  echo "=== baseline weighted v_rep = ${vrep} mV  -> emit jointB (gNMDA rescaled) ==="
  "$PY" "$CODE/make_configs.py" --out_dir "$CFG" --v_rep "$vrep" 2>&1 | tee -a "$LOG/make_configs.log"
}

MODE="${1:-baseline_joint}"
echo "##### PHASE2 SCREEN mode=$MODE  PY=$PY  CK=$CK  @ $(date) #####"
"$PY" --version 2>&1
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>&1 | head -1

# jointA + dose configs (jointB emitted after baseline)
"$PY" "$CODE/make_configs.py" --out_dir "$CFG" --doses 2>&1 | tee "$LOG/make_configs.log"

case "$MODE" in
  smoke)
    SEEDS="0 1 2"; NTRIALS=6; OMIN=-2; OMAX=2; OSTEP=2   # 3 seeds, 3 SOAs, fast
    run_condition baseline ""
    run_condition jointA "$CFG/jointA.json"
    ;;
  baseline)
    run_condition baseline ""
    ;;
  joint)
    emit_jointB
    run_condition jointA "$CFG/jointA.json"
    run_condition jointB "$CFG/jointB.json"
    ;;
  baseline_joint)
    run_condition baseline ""
    emit_jointB
    run_condition jointA "$CFG/jointA.json"
    run_condition jointB "$CFG/jointB.json"
    ;;
  dose)
    for cf in "$CFG"/dose_*.json; do
      if cmp -s "$cf" "$CFG/jointA.json"; then echo "SKIP $(basename "$cf") == jointA (already measured)"; continue; fi
      run_condition "$(basename "$cf" .json)" "$cf"
    done
    ;;
  all)
    run_condition baseline ""
    emit_jointB
    run_condition jointA "$CFG/jointA.json"
    run_condition jointB "$CFG/jointB.json"
    for cf in "$CFG"/dose_*.json; do
      if cmp -s "$cf" "$CFG/jointA.json"; then echo "SKIP $(basename "$cf") == jointA (already measured)"; continue; fi
      run_condition "$(basename "$cf" .json)" "$cf"
    done
    ;;
  *) echo "unknown mode '$MODE'"; exit 2;;
esac

echo "PHASE2_SCREEN_DONE mode=$MODE @ $(date +%H:%M:%S)"
