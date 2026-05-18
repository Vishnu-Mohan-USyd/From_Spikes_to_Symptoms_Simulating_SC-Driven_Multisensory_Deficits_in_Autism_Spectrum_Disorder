"""H_agc — does the SBW dt-drift persist when AGC is frozen?

Phase 1 reproducer (AGC UNFROZEN) gave SBW control HW:
   dt=0.1:  24.13°
   dt=0.05: 20.37°  (-15.6%)

The AGC update (Training.py:2454-2469) runs ONCE PER EXTERNAL FRAME (not
per substep), with alpha_fast=1e-3 NOT scaled by dt.  The update uses
inh_mean = mean(clamp(-I_M, min=0)).  If I_M's dt-dependence is non-zero,
AGC will track different equilibria at different dt.

Test:
  Cond A: AGC unfrozen, dt=0.1  → expected ~24°
  Cond B: AGC unfrozen, dt=0.05 → expected ~20° (drift confirmed in Phase 1)
  Cond C: AGC frozen,   dt=0.1
  Cond D: AGC frozen,   dt=0.05

Verdict logic:
  - If HW(D) - HW(C) ≈ -3.76°: drift PERSISTS without AGC → network mechanism
  - If HW(D) - HW(C) ≈ 0°:    drift VANISHES without AGC → AGC artifact
  - If HW(D) - HW(C) is intermediate: AGC is partial contributor

10-ckpt canonical pipeline.  Same protocol as repro_sbw_dt_v2.py.
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
    sep = pooled["separations_deg"]
    y = pooled["mean_prob"]
    try:
        xs_fit, ys_fit, popt, r2 = fit_sbw_pedestal(pooled)
        cross = _find_crossings(xs_fit, ys_fit, 0.5)
        hw = (cross[-1] - cross[0]) / 2.0 if len(cross) >= 2 else float("nan")
        return hw, popt, r2, y.max()
    except Exception as e:
        print(f"       fit failed: {e}")
        return float("nan"), None, 0.0, y.max()


def measure_one(*, dt: float, n_sub: int, freeze_agc: bool):
    t0 = time.time()
    pooled = run_spatial_binding_across_models(
        MODELS, separations_deg=SBW_SEPARATIONS, device="cuda",
        method="enhancement", n_trials=N_TRIALS,
        enhancement_threshold=ENH_THRESHOLD,
        modify_net=make_modify_net(dt, n_sub, freeze_agc=freeze_agc),
    )
    el = time.time() - t0
    return pooled, el


def main():
    print(f"H_agc test — does SBW dt-drift persist when AGC frozen?")
    print(f"  10 ckpts, n_trials={N_TRIALS}, intensity=1, dur=20, "
          f"enh_thresh={ENH_THRESHOLD}, gNMDA=1.30, Form-2 ON, delays-ms ON")
    print()

    results = {}
    conds = [
        ("A: AGC unfrozen, dt=0.1",  0.1,  100, False),
        ("B: AGC unfrozen, dt=0.05", 0.05, 200, False),
        ("C: AGC FROZEN,   dt=0.1",  0.1,  100, True),
        ("D: AGC FROZEN,   dt=0.05", 0.05, 200, True),
    ]
    for label, dt, n, freeze in conds:
        print(f"--- {label} ---")
        pooled, el = measure_one(dt=dt, n_sub=n, freeze_agc=freeze)
        # control-floor self-subtract for HW measurement
        sbw_sub, floor = subtract_control_floor({"control": pooled})
        hw, popt, r2, ymax = fit_hw_from_pooled(sbw_sub["control"])
        results[label] = (hw, floor, r2, ymax)
        if popt is not None:
            print(f"   HW={hw:.2f}°  floor={floor:.4f}  r2={r2:.4f}  "
                  f"ymax(pre-floor-sub)={ymax + floor:.3f}  el={el:.0f}s")
        else:
            print(f"   FIT FAILED   floor={floor:.4f}  el={el:.0f}s")
        torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    hwA = results["A: AGC unfrozen, dt=0.1"][0]
    hwB = results["B: AGC unfrozen, dt=0.05"][0]
    hwC = results["C: AGC FROZEN,   dt=0.1"][0]
    hwD = results["D: AGC FROZEN,   dt=0.05"][0]

    drift_unfrozen = hwB - hwA
    drift_frozen   = hwD - hwC
    print(f"  AGC unfrozen drift: ΔHW = {drift_unfrozen:+.2f}° "
          f"({drift_unfrozen/hwA*100:+.1f}%)")
    print(f"  AGC FROZEN  drift: ΔHW = {drift_frozen:+.2f}° "
          f"({drift_frozen/hwC*100:+.1f}%)")
    print(f"  Fraction of drift attributable to AGC: "
          f"{(drift_unfrozen - drift_frozen) / drift_unfrozen * 100:+.1f}%"
          if abs(drift_unfrozen) > 1e-3 else "  drift too small to attribute")


if __name__ == "__main__":
    main()
