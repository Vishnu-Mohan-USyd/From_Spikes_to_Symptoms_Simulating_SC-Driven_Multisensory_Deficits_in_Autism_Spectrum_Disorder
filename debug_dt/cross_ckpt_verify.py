"""Cross-checkpoint verification of the (delays×2 + gNMDA recalibration) fix.

For each of M00, M01, M02:
  Cond A: dt=0.1, n_substeps=100, gNMDA=1.30, no delay scale (target).
  Cond C: dt=0.05, n_substeps=200, gNMDA=1.45, delays×2 (proposed fix).

If both root-cause fixes are correct, A and C should match within seed noise.
Also test secondary I_M_inh path: turn off Form 2 for the I_M_inh.add_(I_nmda_inh)
line only — to confirm whether that contributes residual.
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
SEED = 12345


def measure_hw(ckpt: Path, dt: float, nsub: int, *, delay_scale: float = 1.0,
                gNMDA: float = 1.30, seed: int = 12345):
    rng = np.random.default_rng(seed)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        net = load_msi_model(ckpt, device="cuda")
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
    print("Cross-ckpt verification: A vs C (proposed fix)")
    print(f"  Cond A: dt=0.1, gNMDA=1.30, delays raw")
    print(f"  Cond C: dt=0.05, gNMDA=1.45, delays×2")
    print()
    print(f"{'ckpt':>6s}  {'A (dt0.1)':>12s}  {'C (dt0.05+fixes)':>18s}  {'Δ C-A':>8s}")
    for i in [0, 1, 2]:
        ckpt = ROOT / "checkpoint" / f"msi_model_surr_10_{i:02d}.pt"
        PV.restore_original()
        hA = measure_hw(ckpt, 0.1, 100, gNMDA=1.30)
        PV.restore_original()
        hC = measure_hw(ckpt, 0.05, 200, delay_scale=2.0, gNMDA=1.45)
        print(f"  M{i:02d}  {hA:>12.2f}  {hC:>18.2f}  {hC-hA:>+8.2f}")


if __name__ == "__main__":
    main()
