#!/usr/bin/env bash
# Launch all 10 single-ckpt SBW measurements in PARALLEL (5 asymrc + 5 asymd),
# wait, then aggregate each lineage. Self-guards against double-launch.
set -u
W=/home/vishnu/coding_proj/routec_sbw_work
PY=/home/vishnu/miniconda3/bin/python3
APP=$W/apparatus
RC_CODE=/home/vishnu/coding_proj/fsts_routec_delivery/code
RC_CK=/home/vishnu/coding_proj/fsts_routec_delivery/checkpoints
AD_CODE=/home/vishnu/coding_proj/fsts_asym_delay_ep80_20260530/code
AD_CK=/home/vishnu/coding_proj/fsts_asym_delay_ep80_20260530/checkpoints

if pgrep -f 'sbw_routec_run.py' >/dev/null 2>&1; then
  echo "ABORT: sbw_routec_run.py already running"; exit 1
fi
mkdir -p "$W/logs" "$W/out"
rm -f "$W/out"/sbw_asymrc_m*.json "$W/out"/sbw_asymd_m*.json
echo "LAUNCH_10 $(date '+%H:%M:%S')"
for m in 0 1 2 3 4; do
  "$PY" "$W/sbw_routec_run.py" --eval_app "$APP" --code_dir "$RC_CODE" \
     --ckpt "$RC_CK/asymrc_m${m}_ep80.pt" --tag "asymrc_m${m}" \
     --out_json "$W/out/sbw_asymrc_m${m}.json" > "$W/logs/asymrc_m${m}.log" 2>&1 &
  "$PY" "$W/sbw_routec_run.py" --eval_app "$APP" --code_dir "$AD_CODE" \
     --ckpt "$AD_CK/asymd_m${m}_ep80.pt" --tag "asymd_m${m}" \
     --out_json "$W/out/sbw_asymd_m${m}.json" > "$W/logs/asymd_m${m}.log" 2>&1 &
done
wait
echo "ALL_10_DONE $(date '+%H:%M:%S')"
"$PY" "$W/aggregate_sbw.py" --eval_app "$APP" --code_dir "$RC_CODE" \
   --glob "$W/out/sbw_asymrc_m*.json" --prefix asymrc --out_json "$W/sbw_asymrc_AGG.json" 2>&1 | tee "$W/logs/agg_asymrc.log"
"$PY" "$W/aggregate_sbw.py" --eval_app "$APP" --code_dir "$AD_CODE" \
   --glob "$W/out/sbw_asymd_m*.json" --prefix asymd --out_json "$W/sbw_asymd_AGG.json" 2>&1 | tee "$W/logs/agg_asymd.log"
echo "AGG_DONE $(date '+%H:%M:%S')"
