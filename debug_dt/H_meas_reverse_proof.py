"""H_meas REVERSE proof — vary threshold to simulate the halving artifact.

Mechanism claim: BUGGY measurement at dt=0.05 produces HALVED spike counts
relative to dt=0.1.  With fixed threshold=10, halved counts shift the
P(fusion) curve so HW shrinks 24°→20°.

REVERSE proof: at dt=0.1 with BUGGY measurement, increase threshold to
~2× (=20).  If HW shrinks to ~20°, that simulates the halved-count effect
and confirms the threshold/halving interaction is the exact mechanism.

Equivalently: at dt=0.05 BUGGY with threshold ≈ 5 (=half), HW should
return to ~24°.

This is the REVERSE direction of the forward proof in H_meas_bug_proof.py.
Together they isolate the threshold/halving interaction as causal.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import SBW_test as SBW
from TBW_test import _find_crossings
from replot_all_cosmetic import subtract_control_floor, fit_sbw_pedestal

BASE = ROOT / "checkpoint"
MODELS = [BASE / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
SBW_SEPARATIONS = tuple(range(-80, 85, 5))
N_TRIALS = 50


def make_modify_net(dt_val: float, n_sub: int, *, gNMDA: float = 1.30):
    def _mod(n):
        n.dt = float(dt_val)
        n.n_substeps = int(n_sub)
        n.dt_correct_nmda = True
        n.gNMDA = float(gNMDA)
        n.plasticity_enabled = False
        n.freeze_g_FFinh = False
        n._reset_delay_buffers()
    return _mod


def measure(*, dt: float, n_sub: int, threshold: float):
    t0 = time.time()
    pooled = SBW.run_spatial_binding_across_models(
        MODELS, separations_deg=SBW_SEPARATIONS, device="cuda",
        method="enhancement", n_trials=N_TRIALS,
        enhancement_threshold=threshold,
        modify_net=make_modify_net(dt, n_sub),
    )
    el = time.time() - t0
    sbw_sub, floor = subtract_control_floor({"control": pooled})
    try:
        xs_fit, ys_fit, popt, r2 = fit_sbw_pedestal(sbw_sub["control"])
        cross = _find_crossings(xs_fit, ys_fit, 0.5)
        hw = (cross[-1] - cross[0]) / 2 if len(cross) >= 2 else float("nan")
        return hw, floor, r2, el
    except Exception as e:
        return float("nan"), floor, 0.0, el


def main():
    print("H_meas REVERSE proof — vary threshold to simulate halving artifact")
    print(f"  10 ckpts, n_trials={N_TRIALS}, gNMDA=1.30")
    print()

    conds = [
        ("A: BUGGY, dt=0.1,  thr=10   (baseline)", 0.1,  100, 10.0),
        ("B: BUGGY, dt=0.05, thr=10   (drift)",    0.05, 200, 10.0),
        ("C: BUGGY, dt=0.1,  thr=20   (rev: doubled threshold)", 0.1, 100, 20.0),
        ("D: BUGGY, dt=0.05, thr=5    (rev: halved threshold)",  0.05, 200, 5.0),
    ]
    results = {}
    for label, dt, n_sub, thr in conds:
        print(f"--- {label} ---")
        hw, floor, r2, el = measure(dt=dt, n_sub=n_sub, threshold=thr)
        results[label] = hw
        print(f"   HW={hw:.2f}°  floor={floor:.4f}  r2={r2:.4f}  el={el:.0f}s")

    print()
    print("=" * 70)
    print("REVERSE-DIRECTION VERDICT")
    print("=" * 70)
    hwA = results["A: BUGGY, dt=0.1,  thr=10   (baseline)"]
    hwB = results["B: BUGGY, dt=0.05, thr=10   (drift)"]
    hwC = results["C: BUGGY, dt=0.1,  thr=20   (rev: doubled threshold)"]
    hwD = results["D: BUGGY, dt=0.05, thr=5    (rev: halved threshold)"]

    print(f"  HW(dt=0.1,  thr=10): {hwA:.2f}°   — baseline")
    print(f"  HW(dt=0.05, thr=10): {hwB:.2f}°   — drift (-{hwA-hwB:.2f}° = {(hwB-hwA)/hwA*100:+.1f}%)")
    print(f"  HW(dt=0.1,  thr=20): {hwC:.2f}°   — reverse: if ≈ {hwB:.1f}, threshold/halving confirmed")
    print(f"  HW(dt=0.05, thr= 5): {hwD:.2f}°   — reverse: if ≈ {hwA:.1f}, threshold/halving confirmed")
    print()
    print(f"  ΔHW(C → B): {abs(hwC - hwB):.2f}° — should be small (≪ |drift|=4.24°)")
    print(f"  ΔHW(D → A): {abs(hwD - hwA):.2f}° — should be small (≪ |drift|=4.24°)")


if __name__ == "__main__":
    main()
