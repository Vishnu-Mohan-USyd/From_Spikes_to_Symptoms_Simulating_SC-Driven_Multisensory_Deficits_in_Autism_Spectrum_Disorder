"""Paired-seed comparison: dt=0.1 raw vs dt=0.05+delays×2+gNMDA=1.45.

If H_d + small gNMDA recalibration fully closes the dt-gap, the paired
ΔHW should be ≈ 0 across seeds.
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
import debug_dt.patch_variants as PV

OFFSETS = list(range(-50, 51, 2))
OFFSETS_MS = [o * 10 for o in OFFSETS]
TRIALS = 50
CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"


def measure_hw(dt: float, nsub: int, *, delay_scale: float = 1.0,
                gNMDA: float = 1.30, seed: int = 12345):
    rng = np.random.default_rng(seed)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        net = load_msi_model(CKPT, device="cuda")
        net.dt = dt
        net.n_substeps = nsub
        net.dt_correct_nmda = True
        net.gNMDA = gNMDA
        net.plasticity_enabled = False
        net.freeze_g_FFinh = True
        if delay_scale != 1.0:
            for attr in ["conduction_delay_a2msi", "conduction_delay_v2msi",
                         "conduction_delay_inA_inh", "conduction_delay_inV_inh",
                         "conduction_delay_a2msi_inh", "conduction_delay_v2msi_inh",
                         "conduction_delay_msi_inh2exc", "conduction_delay_msi2out"]:
                setattr(net, attr, int(round(getattr(net, attr) * delay_scale)))
            import torch as _t
            B = TRIALS
            for buf_name, delay_attr, shape in [
                ("buffer_a2msi", "conduction_delay_a2msi", (B, net.n)),
                ("buffer_v2msi", "conduction_delay_v2msi", (B, net.n)),
                ("buffer_inA_inh", "conduction_delay_inA_inh", (B, net.n)),
                ("buffer_inV_inh", "conduction_delay_inV_inh", (B, net.n)),
                ("buffer_a2msi_inh", "conduction_delay_a2msi_inh", (B, net.n)),
                ("buffer_v2msi_inh", "conduction_delay_v2msi_inh", (B, net.n)),
                ("buffer_msi_inh2exc", "conduction_delay_msi_inh2exc", (B, net.n_inh)),
                ("buffer_msi2out", "conduction_delay_msi2out", (B, net.n)),
            ]:
                d = getattr(net, delay_attr)
                if d > 0:
                    setattr(net, buf_name, _t.zeros((d,) + shape, device=net.device))
                    net._delay_positions[buf_name] = 0
        p, _ = compute_tbw_temporal_fusion_persep(
            net, OFFSETS, n_trials=TRIALS, T=60, D=5, stim_in=1.0,
            sigma=2.0, valley_threshold=0.4, min_peak_height=0.2,
            min_peak_separation=3, min_total=10.0,
        )
        del net
        torch.cuda.empty_cache()
    finally:
        np.random.default_rng = orig
    try:
        fit = fit_psychometric_curve_improved(np.asarray(OFFSETS_MS), p)
        hw = float(fit.get("tbw", float("nan")))
    except Exception:
        hw = float("nan")
    return hw


def main():
    seeds = [42, 12345, 9999, 1, 7777]

    print("Paired-seed test:")
    print(f"  Cond A: dt=0.1, n=100, gNMDA=1.30, no delay scale")
    print(f"  Cond B: dt=0.05, n=200, gNMDA=1.30, delays×2")
    print(f"  Cond C: dt=0.05, n=200, gNMDA=1.45, delays×2 (proposed fix)")
    print()
    print(f"{'seed':>8s}  {'A:dt0.1':>10s}  {'B:dt0.05+d×2 g1.3':>20s}  "
          f"{'C:dt0.05+d×2 g1.45':>22s}  {'Δ B-A':>8s}  {'Δ C-A':>8s}")
    hwsA, hwsB, hwsC = [], [], []
    for s in seeds:
        PV.restore_original()
        hA = measure_hw(0.1, 100, gNMDA=1.30, seed=s)
        PV.restore_original()
        hB = measure_hw(0.05, 200, delay_scale=2.0, gNMDA=1.30, seed=s)
        PV.restore_original()
        hC = measure_hw(0.05, 200, delay_scale=2.0, gNMDA=1.45, seed=s)
        hwsA.append(hA); hwsB.append(hB); hwsC.append(hC)
        print(f"  {s:>8d}  {hA:>10.2f}  {hB:>20.2f}  {hC:>22.2f}  "
              f"{hB-hA:>+8.2f}  {hC-hA:>+8.2f}")

    A = np.asarray(hwsA); B = np.asarray(hwsB); C = np.asarray(hwsC)
    print()
    print(f"  A:dt0.1            mean={A.mean():.2f} std={A.std():.2f}")
    print(f"  B:dt0.05+d×2 g1.3  mean={B.mean():.2f} std={B.std():.2f}")
    print(f"  C:dt0.05+d×2 g1.45 mean={C.mean():.2f} std={C.std():.2f}")
    print(f"  Δ B-A (mean) = {(B-A).mean():+.2f}  std = {(B-A).std():.2f}")
    print(f"  Δ C-A (mean) = {(C-A).mean():+.2f}  std = {(C-A).std():.2f}")


if __name__ == "__main__":
    main()
