#!/usr/bin/env bash
# ============================================================================================
# Reproduce ALL 7 validations on the route-C *dm10 (tau18)* 10-model ensemble (seeds 42-51),
# end to end, via the dm10 runner shim (val394_dm10_stage2.py).
#
# WHY the shim: the stock measure/ drivers hard-set TAU_GABA=10 / GNMDA=0.50 and resolve the
# tau10 checkpoints, so they FATAL on the dm10 operating point (aM=0.02 dM=10 tau_gaba=18
# gNMDA=0.51). The shim runs the OFFICIAL gate modules VERBATIM (their measurement code + the
# frozen TBW/SBW md5 firewall run byte-identical) after a CPU pre-flight that reads every seed's
# mutable_hparams, asserts the dm10 point, and derives the TAU_GABA/GNMDA overrides FROM the ckpts.
#
#   [0] preflight (CPU)          config-fidelity over all seeds (aborts by name on any drift)
#   [1] main  -> gates 1-4        RATE / TBW / E-I / MEI            -> results/gates_main.json
#       cre   -> gate 5           SBW cross-modal enhancement (CRE) -> results/gate5_cre.json
#       lat   -> gate 6           latency vs intensity              -> results/gate6_latency_sweep.json
#       g7    -> gate 7           cue-reliability (gain_exp=1)      -> results/gate7_cuerel_g1_seed*.json
#       g7agg -> gate 7 pool                                        -> results/gate7_cuerel_g1_aggregate.json
#   [2] grade (pure JSON -> per-seed + ensemble GO/NO-GO)
#   [3] figures (house style, no GPU): gates 1-6 + gate 7 (gain_exp=1)  -> figures/*.{png,svg}
#
# Every gate asserts the frozen TBW/SBW md5 (80d33465 / 73b7d136) BEFORE==AFTER and weight
# bit-identity; this wrapper additionally asserts the two locked md5s at start AND end.
# Local GPU only (set CUDA_VISIBLE_DEVICES; default 0); no network, no sudo.
#
# Usage:  bash run_all.sh [SEED ...]        # default: 42 43 44 45 46 47 48 49 50 51
# ============================================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # <root>/measure
ROOT="$(dirname "$HERE")"
cd "$ROOT"
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VAL394_OUT="$ROOT/results"                        # shim routes every gate's JSON here
SEEDS="${*:-42 43 44 45 46 47 48 49 50 51}"
SHIM="measure/val394_dm10_stage2.py"
GRADE="measure/val394_dm10_grade.py"
TBW_MD5=80d33465c4bf55d6e85b5990acb92da7
SBW_MD5=73b7d13626964d851cc090818b728311

check_frozen () {   # assert the frozen readouts are byte-identical to the locked md5s
  printf '%s  %s\n' "$TBW_MD5" "TBW_test.py" "$SBW_MD5" "SBW_test.py" | md5sum -c - \
    || { echo "[run_all] FATAL: frozen-readout md5 mismatch ($1) — refusing to proceed"; exit 1; }
}

echo "=================================================================="
echo "[dm10] reproduce 7 validations   seeds: $SEEDS   gpu: $CUDA_VISIBLE_DEVICES   $(date '+%F %H:%M:%S')"
echo "=================================================================="
echo "### frozen-readout firewall (BEFORE) ###"
check_frozen BEFORE

echo; echo "### [0/3] config-fidelity pre-flight (CPU) ###"
python "$SHIM" --gate preflight --seeds $SEEDS

echo; echo "### [1/3] gates 1-7 via the shim (one gate per process) ###"
python "$SHIM" --gate main  --seeds $SEEDS
python "$SHIM" --gate cre   --seeds $SEEDS
python "$SHIM" --gate lat   --seeds $SEEDS
python "$SHIM" --gate g7    --seeds $SEEDS --gain_exp 1
python "$SHIM" --gate g7agg --gain_exp 1

echo; echo "### [2/3] grade (per-seed + ensemble GO/NO-GO) ###"
python "$GRADE" --dir "$ROOT/results" --gain_exp 1

echo; echo "### [3/3] figures (house style, no GPU) ###"
for g in 1 2 3 4 5 6; do python "plots/run_gate${g}_ens.py"; done
python "plots/run_gate7_ens.py" gate7_cuerel_g1

echo; echo "### frozen-readout firewall (AFTER) ###"
check_frozen AFTER

echo; echo "=================================================================="
echo "[dm10] DONE  $(date '+%F %H:%M:%S')   figures -> $ROOT/figures"
ls -1 "$ROOT/figures"/*.png 2>/dev/null
echo "=================================================================="
