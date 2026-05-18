"""Phase 1 (v2) — replicate canonical SBW pipeline on 10 ckpts.

If the v1 single-ckpt reproducer fails (negative enhancement, can't fit
pedestal), try the canonical 10-ckpt pipeline with cross-condition floor
subtraction. This should reproduce the validator's task #29 result:
  SBW control HW: ~24° @ dt=0.1, ~20° @ dt=0.05.

Once this works, we know our baseline.  We can then drop to single-ckpt
for fast hypothesis testing (with paired-seed cross-checks).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from TBW_test import _find_crossings
from SBW_test import run_spatial_binding_across_models
from replot_all_cosmetic import subtract_control_floor, fit_sbw_pedestal


BASE = ROOT / "checkpoint"
MODELS = [BASE / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
SBW_SEPARATIONS = tuple(range(-80, 85, 5))
ENH_THRESHOLD = 10.0
N_TRIALS = 50


def make_modify_net(dt_val: float, n_sub: int, *, gNMDA: float = 1.30,
                    freeze_agc: bool = False):
    def _mod(n):
        n.dt = float(dt_val)
        n.n_substeps = int(n_sub)
        n.dt_correct_nmda = True
        n.gNMDA = float(gNMDA)
        n.plasticity_enabled = False
        n.freeze_g_FFinh = bool(freeze_agc)
        n._reset_delay_buffers()
    return _mod


def fit_hw_from_pooled(pooled):
    # Debug: print mean_prob curve to understand if fit is degenerate
    sep = pooled["separations_deg"]
    y = pooled["mean_prob"]
    print(f"       sep range {sep.min():.0f}..{sep.max():.0f}, "
          f"y range {y.min():.4f}..{y.max():.4f}, mid={y[len(y)//2]:.4f}")
    try:
        xs_fit, ys_fit, popt, r2 = fit_sbw_pedestal(pooled)
        cross = _find_crossings(xs_fit, ys_fit, 0.5)
        if len(cross) >= 2:
            return (cross[-1] - cross[0]) / 2.0, popt, r2
        return float("nan"), popt, r2
    except Exception as e:
        print(f"       fit failed: {e}")
        return float("nan"), None, 0.0


def measure_one(*, dt: float, n_sub: int, models=MODELS,
                n_trials: int = N_TRIALS, freeze_agc: bool = False):
    t0 = time.time()
    pooled = run_spatial_binding_across_models(
        models, separations_deg=SBW_SEPARATIONS, device="cuda",
        method="enhancement", n_trials=n_trials,
        enhancement_threshold=ENH_THRESHOLD,
        modify_net=make_modify_net(dt, n_sub, freeze_agc=freeze_agc),
    )
    el = time.time() - t0
    return pooled, el


def main():
    print(f"Canonical SBW dt-drift reproducer — {len(MODELS)} ckpts, "
          f"n_trials={N_TRIALS}, intensity=1, duration=20, enh_thresh={ENH_THRESHOLD}")
    print()

    # IMPORTANT: do NOT freeze AGC for the canonical baseline; the validator
    # in dt_variance_sweep does not freeze it either.  We freeze later for
    # isolation tests.
    print("--- pass 1: dt=0.1, n_substeps=100 (AGC unfrozen) ---")
    pooled01, el01 = measure_one(dt=0.1, n_sub=100, freeze_agc=False)
    print(f"   done in {el01:.0f}s")

    print("--- pass 2: dt=0.05, n_substeps=200 (AGC unfrozen) ---")
    pooled005, el005 = measure_one(dt=0.05, n_sub=200, freeze_agc=False)
    print(f"   done in {el005:.0f}s")

    print()
    print("--- floor subtraction (per-dt, using each pass's control as floor) ---")
    sbw_01,  floor_01  = subtract_control_floor({"control": pooled01})
    sbw_005, floor_005 = subtract_control_floor({"control": pooled005})
    print(f"   floor @ dt=0.10: {floor_01:.4f}")
    print(f"   floor @ dt=0.05: {floor_005:.4f}")

    hw01,  popt01,  r2_01  = fit_hw_from_pooled(sbw_01["control"])
    hw005, popt005, r2_005 = fit_hw_from_pooled(sbw_005["control"])
    print()
    print(f"SBW control HW @ dt=0.1  = {hw01:.2f}°   popt=(base={popt01[0]:.3f}, "
          f"top={popt01[1]:.3f}, half_width={popt01[2]:.2f})  r2={r2_01:.4f}")
    print(f"SBW control HW @ dt=0.05 = {hw005:.2f}°   popt=(base={popt005[0]:.3f}, "
          f"top={popt005[1]:.3f}, half_width={popt005[2]:.2f})  r2={r2_005:.4f}")
    print(f"  ΔHW = {hw005 - hw01:+.2f}°   ({(hw005-hw01)/hw01*100:+.1f}%)")


if __name__ == "__main__":
    main()
