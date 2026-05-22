"""Task #143-A — Regenerate SBW cache for all 5 conds × 10 ckpts under
POST-fix Training.py (AGC reset fix at L1283-4 / L1929-30 + unified NMDA
* dt_linear_scale at L2185 / L2284).

Pre-fix cache files (sbw_*_t10.npz) were moved to /tmp/sbw_t10_prefix/ —
they were generated 03:16-03:32 (BEFORE the 04:24 AGC fix) and were
tainted by the AGC time-tracker persistence bug.

This wrapper:
  1. Monkey-patches generate_all_fresh.CONDITIONS["ff_inhibition"] to the
     pristine pv_nmda=0.8 + targ_ratio=0.8 condition (paper-fair) BEFORE
     calling run_sbw_fresh.
  2. Calls generate_all_fresh.run_sbw_fresh() which iterates all 5 conds ×
     10 ckpts, writing fresh sbw_{cond}_t10.npz files.
  3. Reports per-cond HW.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Step 1: monkey-patch CONDITIONS BEFORE run_sbw_fresh is invoked
import generate_all_fresh as gaf

PRISTINE_FF_INH = lambda n: (
    setattr(n, "pv_nmda", 0.8) or setattr(n, "targ_ratio", 0.8)
)
print(f"[task_A] CONDITIONS['ff_inhibition'] originally: {gaf.CONDITIONS['ff_inhibition']}")
gaf.CONDITIONS["ff_inhibition"] = PRISTINE_FF_INH
print(f"[task_A] CONDITIONS['ff_inhibition'] now:       {gaf.CONDITIONS['ff_inhibition']}")
print(f"[task_A] Paper-fair ff_inh:  pv_nmda=0.8, targ_ratio=0.8")
print()

# Step 2: run the SBW fresh sim across all 5 conds × 10 ckpts
t0 = time.time()
sbw_summary, floor = gaf.run_sbw_fresh()
dt_total = time.time() - t0

print()
print("=" * 70)
print(f"Task #143-A SBW regen — total time {dt_total:.0f}s = {dt_total/60:.1f} min")
print(f"Control-floor subtracted: {floor:.4f}")
print()
print(f"{'condition':>20}  {'HW (deg)':>10}")
print("  " + "-" * 36)
for cname, hw in sbw_summary.items():
    print(f"{cname:>20}  {hw:10.2f}")
print()
print("Cache files written to: cache/sbw_{cond}_t10.npz")
print("Next: python task136_logs/recover_sbw_hw.py")
