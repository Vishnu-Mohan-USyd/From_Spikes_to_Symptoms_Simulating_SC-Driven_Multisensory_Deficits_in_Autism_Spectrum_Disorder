"""H_agc trace — instrument g_FFinh, exc_fast and inh_mean over time at both dts.

Goal: see whether g_FFinh drifts to different equilibria at dt=0.1 vs dt=0.05
during the SBW pass.  If yes, identify which component (exc vs inh) of the
AGC objective drives the drift.

Runs a single SBW-style AV pass on M00 only (1 ckpt, 1 separation=0, 50 trials).
Each external frame, log: g_FFinh, exc_fast = mean(I_ampa_filtered), inh_mean = mean(clamp(-I_M, min=0)).

This is INSTRUMENTATION ONLY — does not modify the AGC behaviour.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from TBW_test import load_msi_model
import Training as TRN

CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"


def run_and_trace(dt: float, n_sub: int, *, n_frames: int = 60, n_trials: int = 50,
                  seed: int = 12345, gNMDA: float = 1.30):
    """Run a synchronous AV pulse for n_frames external frames, recording AGC values."""
    rng_fixed = np.random.default_rng(seed)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng_fixed
    try:
        net = load_msi_model(CKPT, device="cuda")
        net.dt = float(dt)
        net.n_substeps = int(n_sub)
        net.dt_correct_nmda = True
        net.gNMDA = float(gNMDA)
        net.plasticity_enabled = False
        net.freeze_g_FFinh = False  # AGC ON — that's the point
        net._reset_delay_buffers()
        net.reset_state(batch_size=n_trials)

        # synchronous AV at center
        N = net.n
        idx_c = N // 2
        xs = torch.arange(N, dtype=torch.float32, device=net.device)
        gauss = torch.exp(-0.5 * ((xs - idx_c) / net.sigma_in) ** 2) * 1.0
        stim = gauss.expand(n_trials, N).contiguous()
        zero = torch.zeros_like(stim)

        records = []
        for t in range(n_frames):
            # First 20 frames are stimulus (matching SBW duration=20)
            sA = stim if t < 20 else zero
            sV = stim if t < 20 else zero
            net.update_all_layers_batch(sA, sV)
            # Record at END of frame (after AGC update at line 2454-2469)
            exc_fast = net.I_ampa_filtered.clamp(min=0).mean().item()
            inh_mean = (-net.I_M).clamp(min=0).mean().item()
            exc_long = net.I_M.clamp(min=0).mean().item()
            records.append({
                "t": t,
                "g_FFinh": float(net.g_FFinh),
                "exc_fast": exc_fast,
                "inh_mean": inh_mean,
                "exc_long": exc_long,
                "I_M_mean": float(net.I_M.mean().item()),
                "I_M_std": float(net.I_M.std().item()),
            })
        del net
        torch.cuda.empty_cache()
    finally:
        np.random.default_rng = orig
    return records


def main():
    print("H_agc trace — per-frame AGC values at dt=0.1 vs dt=0.05")
    print("Sync AV pulse, 20 frames on, 40 frames off, n_trials=50, M00 only")
    print()

    rec_01 = run_and_trace(0.1, 100)
    rec_05 = run_and_trace(0.05, 200)

    print(f"{'frame':>5}  {'g(0.1)':>8}  {'g(0.05)':>8}  "
          f"{'exc_f(0.1)':>10}  {'exc_f(0.05)':>10}  "
          f"{'inh_m(0.1)':>10}  {'inh_m(0.05)':>10}  "
          f"{'I_M_m(0.1)':>10}  {'I_M_m(0.05)':>10}")
    for r1, r2 in zip(rec_01, rec_05):
        print(f"  {r1['t']:>3d}   {r1['g_FFinh']:>7.4f}  {r2['g_FFinh']:>7.4f}  "
              f"{r1['exc_fast']:>10.4f}  {r2['exc_fast']:>10.4f}  "
              f"{r1['inh_mean']:>10.4f}  {r2['inh_mean']:>10.4f}  "
              f"{r1['I_M_mean']:>10.4f}  {r2['I_M_mean']:>10.4f}")

    print()
    print("Summary over stim window (frames 5-19):")
    g1 = np.array([r['g_FFinh'] for r in rec_01[5:20]]).mean()
    g2 = np.array([r['g_FFinh'] for r in rec_05[5:20]]).mean()
    e1 = np.array([r['exc_fast'] for r in rec_01[5:20]]).mean()
    e2 = np.array([r['exc_fast'] for r in rec_05[5:20]]).mean()
    i1 = np.array([r['inh_mean'] for r in rec_01[5:20]]).mean()
    i2 = np.array([r['inh_mean'] for r in rec_05[5:20]]).mean()
    print(f"  g_FFinh    : dt=0.1 = {g1:.4f}    dt=0.05 = {g2:.4f}    Δ = {g2-g1:+.4f}")
    print(f"  exc_fast   : dt=0.1 = {e1:.4f}    dt=0.05 = {e2:.4f}    Δ = {e2-e1:+.4f}  (ratio={e2/e1:.3f})")
    print(f"  inh_mean   : dt=0.1 = {i1:.4f}    dt=0.05 = {i2:.4f}    Δ = {i2-i1:+.4f}  (ratio={i2/i1:.3f})")
    print(f"  exc/inh    : dt=0.1 = {e1/i1:.4f}  dt=0.05 = {e2/i2:.4f}")


if __name__ == "__main__":
    main()
