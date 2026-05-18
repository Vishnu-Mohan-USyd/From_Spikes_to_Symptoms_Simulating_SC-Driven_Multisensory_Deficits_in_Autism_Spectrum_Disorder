"""Multi-seed variability + gNMDA sweep at dt=0.05 with delays×2.

Goal: (a) confirm the 21 ms residual gap is larger than per-seed variation,
      (b) find if a calibration tweak of gNMDA at dt=0.05 closes the residual.
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


def main():
    print("Multi-seed variability at dt=0.1 raw and dt=0.05 + delays×2")
    print()
    seeds = [42, 12345, 9999, 1, 7777]

    print(f"{'seed':>8s}  {'dt=0.1 raw':>12s}  {'dt=0.05 delays×2':>18s}")
    h01s, h05s = [], []
    for s in seeds:
        PV.restore_original()
        h01, _ = measure_hw(0.1, 100, seed=s)
        PV.restore_original()
        h05, _ = measure_hw(0.05, 200, delay_scale=2.0, seed=s)
        h01s.append(h01); h05s.append(h05)
        print(f"  {s:>8d}  {h01:>12.2f}  {h05:>18.2f}")

    h01s = np.asarray(h01s); h05s = np.asarray(h05s)
    print(f"\n  dt=0.1 raw      mean={h01s.mean():.2f}, std={h01s.std():.2f}")
    print(f"  dt=0.05 dlys×2  mean={h05s.mean():.2f}, std={h05s.std():.2f}")
    print(f"  ΔHW(mean) = {h05s.mean() - h01s.mean():+.2f} ms")
    print(f"  combined std = {np.sqrt(np.var(h01s) + np.var(h05s)):.2f} ms")
    print(f"  z-score of ΔHW = {abs(h05s.mean() - h01s.mean()) / np.sqrt(np.var(h01s) + np.var(h05s)):.2f}")

    print()
    print("=" * 70)
    print("gNMDA sweep at dt=0.05 + delays×2  (target HW ≈ 149)")
    print("=" * 70)
    print(f"{'gNMDA':>8s}  {'HW (ms)':>10s}")
    for g in [1.0, 1.15, 1.30, 1.45, 1.6, 1.8, 2.0]:
        PV.restore_original()
        h, _ = measure_hw(0.05, 200, delay_scale=2.0, gNMDA=g)
        print(f"  {g:>8.3f}  {h:>10.2f}")


if __name__ == "__main__":
    main()
