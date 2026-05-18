"""Stage C: calibrate gNMDA at dt=0.1 with flag ON.

Sweep gNMDA on M00 fixed-seed canonical TBW to find the value that produces
HW closest to the dt=0.1 reference (~149 ms).

Reference: HW = 149.06 ms with flag OFF (legacy) at gNMDA = 0.05 baseline.

Theory: applying the fix removes implicit amplification of NMDA by
tau_syn/dt = 2.5/0.1 = 25 at dt=0.1, so initial guess for new gNMDA is
~25× the current 0.05 = 1.25. Voltage-Mg feedback is nonlinear, so we
calibrate empirically.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from TBW_test import (
    load_msi_model,
    compute_tbw_temporal_fusion_persep,
    fit_psychometric_curve_improved,
)

OFFSETS = list(range(-50, 51, 2))
OFFSETS_MS = [o * 10 for o in OFFSETS]
TRIALS = 50
SEED = 12345
TARGET_HW = 149.06  # Stage B value


def measure_hw(ckpt: Path, dt: float, nsub: int, flag: bool, gNMDA: float):
    rng = np.random.default_rng(SEED)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        net = load_msi_model(ckpt, device="cuda")
        net.dt = dt
        net.n_substeps = nsub
        net.dt_correct_nmda = flag
        net.gNMDA = gNMDA
        net.plasticity_enabled = False
        net.freeze_g_FFinh = True
        t0 = time.time()
        p, _ = compute_tbw_temporal_fusion_persep(
            net, OFFSETS, n_trials=TRIALS, T=60, D=5, stim_in=1.0,
            sigma=2.0, valley_threshold=0.4, min_peak_height=0.2,
            min_peak_separation=3, min_total=10.0,
        )
        elapsed = time.time() - t0
        del net
        torch.cuda.empty_cache()
    finally:
        np.random.default_rng = orig
    try:
        fit = fit_psychometric_curve_improved(np.asarray(OFFSETS_MS), p)
        hw = float(fit.get("tbw", float("nan")))
    except Exception:
        hw = float("nan")
    return hw, p, elapsed


def main():
    ckpt = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
    print(f"Stage C gNMDA sweep — flag ON, dt=0.1, M00 fixed-seed.")
    print(f"  Target HW = {TARGET_HW:.2f} ms (Stage B baseline with flag OFF).")
    print()

    sweep = [0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5]
    results = {}
    print(f"  {'gNMDA':>7s}  {'HW (ms)':>10s}  {'Δ vs target':>12s}  {'elapsed':>8s}")
    for g in sweep:
        hw, _, t = measure_hw(ckpt, dt=0.1, nsub=100, flag=True, gNMDA=g)
        results[g] = hw
        print(f"  {g:>7.3f}  {hw:>10.2f}  {hw - TARGET_HW:>+12.2f}  {t:>7.1f}s")

    # Closest
    closest = min(results.keys(), key=lambda g: abs(results[g] - TARGET_HW))
    print()
    print(f"  Closest to target: gNMDA = {closest}, HW = {results[closest]:.2f} ms "
          f"(Δ = {results[closest] - TARGET_HW:+.2f} ms)")
    print()
    print(f"  Full results: {results}")


if __name__ == "__main__":
    main()
