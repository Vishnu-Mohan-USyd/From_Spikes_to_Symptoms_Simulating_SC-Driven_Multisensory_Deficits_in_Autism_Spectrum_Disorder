"""Test hypotheses for the residual 86-ms HW gap (post Form 2 + gNMDA=1.30).

Baseline (verified): HW(dt=0.1, Form2 ON, gNMDA=1.30) = 148.91 ms
                     HW(dt=0.05, Form2 ON, gNMDA=1.30) = 62.46 ms

Each hypothesis is tested by installing a monkey-patch that modifies ONE
mechanism, with Form 2 still active. If the patched HW(dt=0.1) collapses
toward 62 ms, that mechanism is what makes dt=0.1 WIDER (i.e., it
contributes spurious widening due to coarse-dt integration).
Conversely if HW(dt=0.05) rises toward 149 ms when the mechanism is fixed
at dt=0.05, that mechanism contributes the narrowing.
"""
from __future__ import annotations
import inspect
import sys
import textwrap
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
import debug_dt.patch_variants as PV
import Training as TRN

OFFSETS = list(range(-50, 51, 2))
OFFSETS_MS = [o * 10 for o in OFFSETS]
TRIALS = 50
SEED = 12345
CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
GNMDA = 1.30


def measure_hw(dt: float, nsub: int):
    rng = np.random.default_rng(SEED)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        net = load_msi_model(CKPT, device="cuda")
        net.dt = dt
        net.n_substeps = nsub
        net.dt_correct_nmda = True   # Form 2 ON
        net.gNMDA = GNMDA
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
    return hw, elapsed


def banner(s):
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def main():
    banner("Baselines (Form 2 ON, gNMDA=1.30)")
    PV.restore_original()
    h01, t01 = measure_hw(0.1, 100)
    h05, t05 = measure_hw(0.05, 200)
    print(f"  dt=0.1, raw  : HW = {h01:.2f} ms  ({t01:.1f}s)")
    print(f"  dt=0.05, raw : HW = {h05:.2f} ms  ({t05:.1f}s)")
    print(f"  Δ = {h05 - h01:+.2f} ms")

    banner("H_b: Izhikevich half-step at dt=0.1")
    print("Prediction: if Izhi forward-Euler overshoot produces spurious spikes")
    print("at dt=0.1, halving the Izhi step should drop HW(dt=0.1) toward ~62 ms.")
    PV.restore_original()
    PV.apply_izhi_half()
    h01_ih, t = measure_hw(0.1, 100)
    PV.restore_original()
    print(f"  HW(dt=0.1, izhi-half) = {h01_ih:.2f} ms  ({t:.1f}s)")
    print(f"  Δ vs raw dt=0.1 (149) = {h01_ih - h01:+.2f} ms")
    print(f"  Δ vs raw dt=0.05 (62) = {h01_ih - h05:+.2f} ms")

    banner("H_b: Izhikevich half-step at dt=0.05 (sanity)")
    print("Prediction: if Izhi already converged at dt=0.05, no change.")
    PV.restore_original()
    PV.apply_izhi_half()
    h05_ih, t = measure_hw(0.05, 200)
    PV.restore_original()
    print(f"  HW(dt=0.05, izhi-half) = {h05_ih:.2f} ms  ({t:.1f}s)")
    print(f"  Δ vs raw dt=0.05 (62) = {h05_ih - h05:+.2f} ms")


if __name__ == "__main__":
    main()
