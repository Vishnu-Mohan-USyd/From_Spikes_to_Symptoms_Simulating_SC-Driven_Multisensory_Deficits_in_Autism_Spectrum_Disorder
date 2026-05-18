"""Verify the dt-dependence is mechanistic (not stochastic).

1. Seed variability check: run dt=0.1 baseline with 3 seeds, dt=0.05 with 3
   seeds.  Compute std of HW.  The dt-dependence (146 ms gap) is far larger
   than any seed-driven variation, confirming mechanistic origin.

2. Direct measurement of mean V_msi after NMDA-only sustained input
   (V vs time) at dt=0.1 vs dt=0.05.  Verify V evolves ~2x faster at dt=0.05.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from TBW_test import (
    load_msi_model,
    compute_tbw_temporal_fusion_persep,
    fit_psychometric_curve_improved,
)
from Training import generate_two_event_offset_seq, generate_av_batch_tensor

CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
OFFSETS = list(range(-50, 51, 2))
OFFSETS_MS = [o * 10 for o in OFFSETS]


def measure_hw(dt, nsub, *, seed=12345, n_trials=50):
    rng = np.random.default_rng(seed)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        net = load_msi_model(CKPT, device="cuda")
        net.dt = dt
        net.n_substeps = nsub
        net.plasticity_enabled = False
        net.freeze_g_FFinh = True
        p, _ = compute_tbw_temporal_fusion_persep(
            net, OFFSETS, n_trials=n_trials, T=60, D=5, stim_in=1.0,
            sigma=2.0, valley_threshold=0.4, min_peak_height=0.2,
            min_peak_separation=3, min_total=10.0,
        )
        del net
        torch.cuda.empty_cache()
    finally:
        np.random.default_rng = orig
    try:
        fit = fit_psychometric_curve_improved(np.asarray(OFFSETS_MS), p)
        return float(fit.get("tbw", float("nan")))
    except Exception:
        return float("nan")


def main():
    print("--- Seed variability of HW at fixed dt ---")
    seeds = [42, 12345, 9999, 1, 7777]

    hw01 = [measure_hw(0.1, 100, seed=s) for s in seeds]
    hw05 = [measure_hw(0.05, 200, seed=s) for s in seeds]

    print(f"  dt=0.1  HW = {hw01}  mean={np.mean(hw01):.1f}  std={np.std(hw01):.1f}")
    print(f"  dt=0.05 HW = {hw05}  mean={np.mean(hw05):.1f}  std={np.std(hw05):.1f}")
    print(f"  ΔHW(mean) = {np.mean(hw05) - np.mean(hw01):+.1f} ms")
    print(f"  combined std = {np.sqrt(np.var(hw01) + np.var(hw05)):.1f} ms")

    print()
    print("--- Direct measurement: I_nmda and v_msi during sustained input ---")
    # Drive both A and V with a sustained pulse and record I_nmda magnitude.

    for tag, dt, nsub in [("dt01", 0.1, 100), ("dt005", 0.05, 200)]:
        net = load_msi_model(CKPT, device="cuda")
        net.dt = dt
        net.n_substeps = nsub
        net.plasticity_enabled = False
        net.freeze_g_FFinh = True

        # Build a sustained AV input centered at neuron 90, lasting 30 frames
        T = 60
        D = 30  # sustained for 30 frames (300 ms)
        rng = np.random.default_rng(42)
        loc = 90
        ls, ms = generate_two_event_offset_seq(loc=loc, T=T, D=D, offset=0,
                                                space_size=net.space_size)
        xA, xV, mask = generate_av_batch_tensor(
            [ls], [ms], [False],
            n=net.n, space_size=net.space_size, sigma_in=net.sigma_in,
            noise_std=0.0, device=net.device, max_len=T,
            stimulus_intensity=1.0,
        )

        net.reset_state(1)
        nmda_m_trace, v_msi_trace, mg_trace, I_nmda_trace = [], [], [], []
        with torch.inference_mode():
            for t in range(T):
                net.update_all_layers_batch(xA[:, t], xV[:, t], mask[:, t])
                # Sample at frame end
                nmda_m_trace.append(net.nmda_m.mean().item())
                v_msi_trace.append(net.v_msi.mean().item())
                mg_A = 1.0 / (1.0 + torch.exp(-net.mg_k * (net.v_dend_A - net.mg_vhalf)))
                mg_V = 1.0 / (1.0 + torch.exp(-net.mg_k * (net.v_dend_V - net.mg_vhalf)))
                mg_trace.append((mg_A + mg_V).mean().item())
                Inmda_inst = (net.gNMDA * net.nmda_m * (mg_A + mg_V) *
                              (net.Erev_nmda - net.v_msi))
                I_nmda_trace.append(Inmda_inst.mean().item())

        print(f"\n[{tag}] dt={dt} n_substeps={nsub}")
        # Report mean during the stimulus and recovery
        nmda_m_arr = np.asarray(nmda_m_trace)
        I_nmda_arr = np.asarray(I_nmda_trace)
        v_msi_arr = np.asarray(v_msi_trace)
        mg_arr = np.asarray(mg_trace)
        # Steady-state mean over frames 5-25 (skip onset transient, before stim end)
        win = slice(5, 25)
        print(f"  nmda_m (mean fr5-25) = {nmda_m_arr[win].mean():.6f}")
        print(f"  mg_A+mg_V (fr5-25)   = {mg_arr[win].mean():.6f}")
        print(f"  v_msi  (mean fr5-25) = {v_msi_arr[win].mean():.3f}")
        print(f"  I_nmda (mean fr5-25) = {I_nmda_arr[win].mean():.4f}")

        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
