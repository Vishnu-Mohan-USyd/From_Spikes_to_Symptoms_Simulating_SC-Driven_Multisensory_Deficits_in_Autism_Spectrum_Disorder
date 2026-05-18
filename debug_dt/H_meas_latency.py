"""H_meas propagation — latency drift explained by same bug?

response_latency_test.py:649: `if net._latest_sMSI.sum().item() > 0` checks
if ANY neuron fired in the LAST SUBSTEP of the current frame.  At dt=0.05,
the last substep covers half the wall time → P(spike in last substep) is
halved → first detection frame is LATER → larger latency.

Test:
  Cond A: BUGGY (_latest_sMSI),  dt=0.1   → reference latency
  Cond B: BUGGY,                 dt=0.05  → larger latency (+5-11%)
  Cond C: FIXED (return_spike_sum),  dt=0.1
  Cond D: FIXED,                 dt=0.05

If C ≈ D ≈ A: same root cause as SBW. Otherwise: different mechanism.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from response_latency_test import load_msi_model

BASE = ROOT / "checkpoint"
MODELS = [BASE / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]


def measure_latency_buggy(net, *, modality="B", centre_deg=90.0, sigma_in=5.0,
                          intensity=1.0, pulse_frames=10, n_frames=40):
    """ORIGINAL: uses _latest_sMSI.sum() check (buggy)."""
    N = net.n
    dt_ms = net.dt
    substeps = net.n_substeps
    frame_ms = dt_ms * substeps
    net.reset_state(batch_size=1)
    xA = torch.zeros(n_frames, N, device=net.device)
    xV = torch.zeros_like(xA)
    idx_c = int(round(centre_deg * (N - 1) / (net.space_size - 1)))
    xs = torch.arange(N, dtype=torch.float32, device=net.device)
    gauss_vec = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * intensity
    if modality in ("A", "B"):
        xA[:pulse_frames] = gauss_vec
    if modality in ("V", "B"):
        xV[:pulse_frames] = gauss_vec
    for t in range(n_frames):
        net.update_all_layers_batch(xA[t].unsqueeze(0), xV[t].unsqueeze(0))
        if net._latest_sMSI.sum().item() > 0:
            return (t + 1) * frame_ms
    return float('nan')


def measure_latency_fixed(net, *, modality="B", centre_deg=90.0, sigma_in=5.0,
                          intensity=1.0, pulse_frames=10, n_frames=40):
    """FIXED: uses return_spike_sum=True for TRUE per-frame spike count."""
    N = net.n
    dt_ms = net.dt
    substeps = net.n_substeps
    frame_ms = dt_ms * substeps
    net.reset_state(batch_size=1)
    xA = torch.zeros(n_frames, N, device=net.device)
    xV = torch.zeros_like(xA)
    idx_c = int(round(centre_deg * (N - 1) / (net.space_size - 1)))
    xs = torch.arange(N, dtype=torch.float32, device=net.device)
    gauss_vec = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * intensity
    if modality in ("A", "B"):
        xA[:pulse_frames] = gauss_vec
    if modality in ("V", "B"):
        xV[:pulse_frames] = gauss_vec
    for t in range(n_frames):
        ret = net.update_all_layers_batch(xA[t].unsqueeze(0), xV[t].unsqueeze(0),
                                          return_spike_sum=True)
        sum_sM = ret[-1]
        if sum_sM.sum().item() > 0:
            return (t + 1) * frame_ms
    return float('nan')


def make_modify_net(dt_val: float, n_sub: int, *, gNMDA: float = 1.30):
    def _mod(n):
        n.dt = float(dt_val)
        n.n_substeps = int(n_sub)
        n.dt_correct_nmda = True
        n.gNMDA = float(gNMDA)
        n.plasticity_enabled = False
        n._reset_delay_buffers()
        # Match canonical response_latency_test override (line 686)
        n.aM, n.bM, n.cM, n.dM = 0.001, 0.2, -60.0, 0.1
    return _mod


def run_lat_one(dt, n_sub, *, fixed_meas: bool):
    mod_fn = make_modify_net(dt, n_sub)
    fn = measure_latency_fixed if fixed_meas else measure_latency_buggy
    rows = []
    for p in MODELS:
        net = load_msi_model(p, device="cuda")
        mod_fn(net)
        latA = fn(net, modality="A")
        latV = fn(net, modality="V")
        latB = fn(net, modality="B")
        rows.append({"A": latA, "V": latV, "B": latB})
        del net
        torch.cuda.empty_cache()
    A = np.array([r["A"] for r in rows])
    V = np.array([r["V"] for r in rows])
    B = np.array([r["B"] for r in rows])
    return {
        "A_mean": float(np.nanmean(A)), "A_std": float(np.nanstd(A)),
        "V_mean": float(np.nanmean(V)), "V_std": float(np.nanstd(V)),
        "B_mean": float(np.nanmean(B)), "B_std": float(np.nanstd(B)),
    }


def main():
    print("H_meas propagation — latency with BUGGY vs FIXED measurement")
    print(f"  10 ckpts, gNMDA=1.30, adaptation override (aM/bM/cM/dM)")
    print()

    results = {}
    conds = [
        ("A: BUGGY, dt=0.1 ", 0.1,  100, False),
        ("B: BUGGY, dt=0.05", 0.05, 200, False),
        ("C: FIXED, dt=0.1 ", 0.1,  100, True),
        ("D: FIXED, dt=0.05", 0.05, 200, True),
    ]
    for label, dt, n_sub, fixed in conds:
        print(f"--- {label} ---")
        t0 = time.time()
        r = run_lat_one(dt, n_sub, fixed_meas=fixed)
        el = time.time() - t0
        results[label] = r
        print(f"   A = {r['A_mean']:.2f}±{r['A_std']:.2f}   "
              f"V = {r['V_mean']:.2f}±{r['V_std']:.2f}   "
              f"B = {r['B_mean']:.2f}±{r['B_std']:.2f}   el={el:.0f}s")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    A = results["A: BUGGY, dt=0.1 "]
    B = results["B: BUGGY, dt=0.05"]
    C = results["C: FIXED, dt=0.1 "]
    D = results["D: FIXED, dt=0.05"]

    print(f"  BUGGY drift A: {A['A_mean']:.2f} → {B['A_mean']:.2f}  "
          f"({(B['A_mean']-A['A_mean'])/A['A_mean']*100:+.1f}%)")
    print(f"  FIXED drift A: {C['A_mean']:.2f} → {D['A_mean']:.2f}  "
          f"({(D['A_mean']-C['A_mean'])/C['A_mean']*100:+.1f}%)")
    print(f"  BUGGY drift V: {A['V_mean']:.2f} → {B['V_mean']:.2f}  "
          f"({(B['V_mean']-A['V_mean'])/A['V_mean']*100:+.1f}%)")
    print(f"  FIXED drift V: {C['V_mean']:.2f} → {D['V_mean']:.2f}  "
          f"({(D['V_mean']-C['V_mean'])/C['V_mean']*100:+.1f}%)")
    print(f"  BUGGY drift B: {A['B_mean']:.2f} → {B['B_mean']:.2f}  "
          f"({(B['B_mean']-A['B_mean'])/A['B_mean']*100:+.1f}%)")
    print(f"  FIXED drift B: {C['B_mean']:.2f} → {D['B_mean']:.2f}  "
          f"({(D['B_mean']-C['B_mean'])/C['B_mean']*100:+.1f}%)")


if __name__ == "__main__":
    main()
