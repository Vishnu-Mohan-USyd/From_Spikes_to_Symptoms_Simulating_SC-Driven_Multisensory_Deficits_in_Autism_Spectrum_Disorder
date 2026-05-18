"""Stage D' for task #27: verify dt-independence after delays-in-ms patch.

Configuration: flag ON, gNMDA = 1.30, delays converted to physical-ms at runtime.

Expected (per debugger's task #21+#26 findings):
  dt=0.1  HW ≈ 149 ms (unchanged from prior Stage D — same substep counts at dt=0.1)
  dt=0.05 HW ≈ 128 ms (much improved from 62; residual ~21 ms / 14% nmda_m transient
                       accepted as qualitative match per team-lead's task #27 brief)
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
GNMDA = 1.30


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
        # Re-allocate ring buffers at new dt/flag (task #27).
        net._reset_delay_buffers()
        net.plasticity_enabled = False
        net.freeze_g_FFinh = True
        t0 = time.time()
        p, _ = compute_tbw_temporal_fusion_persep(
            net, OFFSETS, n_trials=TRIALS, T=60, D=5, stim_in=1.0,
            sigma=2.0, valley_threshold=0.4, min_peak_height=0.2,
            min_peak_separation=3, min_total=10.0,
        )
        elapsed = time.time() - t0
        # Also report the actual substep counts being used (for the record)
        delay_a2msi = net._delay_substeps_from_ms(net.conduction_delay_a2msi_ms) if flag else net.conduction_delay_a2msi
        delay_v2msi = net._delay_substeps_from_ms(net.conduction_delay_v2msi_ms) if flag else net.conduction_delay_v2msi
        del net
        torch.cuda.empty_cache()
    finally:
        np.random.default_rng = orig
    try:
        fit = fit_psychometric_curve_improved(np.asarray(OFFSETS_MS), p)
        hw = float(fit.get("tbw", float("nan")))
    except Exception:
        hw = float("nan")
    return hw, p, elapsed, delay_a2msi, delay_v2msi


def main():
    ckpt = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
    print(f"Stage D' (task #27) — delays-in-ms + flag ON + gNMDA={GNMDA}, M00 fixed-seed.")
    print()
    hw_01,  _, t_01,  da_01, dv_01 = measure_hw(ckpt, dt=0.1,  nsub=100, flag=True, gNMDA=GNMDA)
    hw_005, _, t_005, da_005, dv_005 = measure_hw(ckpt, dt=0.05, nsub=200, flag=True, gNMDA=GNMDA)
    print(f"  dt=0.1   (nsub=100): HW = {hw_01:.2f} ms   substeps (a2msi/v2msi) = {da_01}/{dv_01}   ({t_01:.1f}s)")
    print(f"  dt=0.05  (nsub=200): HW = {hw_005:.2f} ms   substeps (a2msi/v2msi) = {da_005}/{dv_005}   ({t_005:.1f}s)")
    print(f"  ΔHW = {hw_005 - hw_01:+.2f} ms")
    print()
    print(f"  Expected (team-lead task #27 brief): dt=0.1 ≈ 149, dt=0.05 ≈ 128, residual ~21 ms.")
    if abs(hw_005 - hw_01) <= 25:
        print(f"  → PASS (qualitative, within 25 ms band per brief's 14% residual).")
    else:
        print(f"  → FAIL: gap {abs(hw_005 - hw_01):.1f} ms > 25 ms.")


if __name__ == "__main__":
    main()
