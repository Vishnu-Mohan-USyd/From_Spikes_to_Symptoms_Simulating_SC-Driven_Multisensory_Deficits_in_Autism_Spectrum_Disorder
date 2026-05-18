"""H_meas propagation — cue reliability collapse explained by same bug?

Hypothesis: cue_reliability_test.py:79 has the same _latest_sMSI bug.
At dt=0.05 the per-frame spike_sum is HALVED (sampled only at last substep).
Halved counts = noisier decoded location (fewer samples per neuron) =
empirical w_V noisier = R² collapses.

Test:
  Cond A: BUGGY (_latest_sMSI), dt=0.1   → expected R² ≈ 0.78
  Cond B: BUGGY (_latest_sMSI), dt=0.05  → expected R² ≈ 0.18 (collapse)
  Cond C: FIXED (return_spike_sum=True), dt=0.1
  Cond D: FIXED, dt=0.05

If C ≈ D ≈ A: same root cause as SBW (propagation confirmed).
If C ≠ D: different mechanism, more investigation needed.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cue_reliability_test import (
    make_dataset, pool_to_mean_sem, load_msi_model as cue_load,
)
import Training as TRN

BASE = ROOT / "checkpoint"
MODELS = [BASE / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]


def reliability_buggy(net, xA, xV, meta, stim_frames=10, method="com"):
    """ORIGINAL implementation: uses _latest_sMSI (buggy)."""
    from Training import decode_msi_location
    B, T, n = xA.shape
    device = net.device
    net.reset_state(batch_size=B)
    spike_sum = torch.zeros(B, n, device=device)
    for t in range(T):
        net.update_all_layers_batch(xA[:, t], xV[:, t])
        spike_sum += net._latest_sMSI            # BUG: last substep only
    est_deg = decode_msi_location(spike_sum, space_size=net.space_size, method=method)
    SA, SV = 80.0, 100.0
    wV = (est_deg.cpu().numpy() - SA) / (SV - SA)
    out = {}
    for i, (sigA, sigV) in enumerate(meta):
        out.setdefault((sigA, sigV), []).append(wV[i])
    return out


def reliability_fixed(net, xA, xV, meta, stim_frames=10, method="com"):
    """FIXED: uses return_spike_sum=True to get TRUE per-frame total."""
    from Training import decode_msi_location
    B, T, n = xA.shape
    device = net.device
    net.reset_state(batch_size=B)
    spike_sum = torch.zeros(B, n, device=device)
    for t in range(T):
        ret = net.update_all_layers_batch(xA[:, t], xV[:, t], return_spike_sum=True)
        sum_sM = ret[-1]
        spike_sum += sum_sM
    est_deg = decode_msi_location(spike_sum, space_size=net.space_size, method=method)
    SA, SV = 80.0, 100.0
    wV = (est_deg.cpu().numpy() - SA) / (SV - SA)
    out = {}
    for i, (sigA, sigV) in enumerate(meta):
        out.setdefault((sigA, sigV), []).append(wV[i])
    return out


def make_modify_net(dt_val: float, n_sub: int, *, gNMDA: float = 1.30):
    def _mod(n):
        n.dt = float(dt_val)
        n.n_substeps = int(n_sub)
        n.dt_correct_nmda = True
        n.gNMDA = float(gNMDA)
        n.plasticity_enabled = False
        n._reset_delay_buffers()
    return _mod


def compute_metrics(pooled):
    w_pred_all, w_emp_all, abs_errs, sq_errs = [], [], [], []
    for d in pooled:
        var_a, var_v = d["sigma_a"] ** 2, d["sigma_v"] ** 2
        w_pred = 1.0 / var_v / (1.0 / var_a + 1.0 / var_v)
        w_emp = d["mean_w"]
        w_pred_all.append(w_pred); w_emp_all.append(w_emp)
        abs_errs.append(abs(w_emp - w_pred)); sq_errs.append((w_emp - w_pred) ** 2)
    mae = float(np.mean(abs_errs))
    rmse = float(np.sqrt(np.mean(sq_errs)))
    r2 = float(np.corrcoef(w_pred_all, w_emp_all)[0, 1] ** 2)
    return r2, mae, rmse


def run_one(dt: float, n_sub: int, *, fixed_meas: bool, n_trials: int = 20):
    xA, xV, meta = make_dataset(n_trials=n_trials, device="cuda")
    mod_fn = make_modify_net(dt, n_sub)
    fn = reliability_fixed if fixed_meas else reliability_buggy
    all_results = []
    for p in MODELS:
        net = cue_load(p, device="cuda")
        mod_fn(net)
        net._probe = None
        net.allow_inhib_plasticity = False
        res = fn(net, xA, xV, meta)
        all_results.append(res)
        del net
        torch.cuda.empty_cache()
    pooled = pool_to_mean_sem(all_results)
    return compute_metrics(pooled)


def main():
    print("H_meas propagation — cue reliability with BUGGY vs FIXED measurement")
    print(f"  10 ckpts, n_trials=20, gNMDA=1.30")
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
        r2, mae, rmse = run_one(dt=dt, n_sub=n_sub, fixed_meas=fixed)
        el = time.time() - t0
        results[label] = (r2, mae, rmse)
        print(f"   R² = {r2:.3f}   MAE = {mae:.3f}   RMSE = {rmse:.3f}   el={el:.0f}s")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    A = results["A: BUGGY, dt=0.1 "]
    B = results["B: BUGGY, dt=0.05"]
    C = results["C: FIXED, dt=0.1 "]
    D = results["D: FIXED, dt=0.05"]
    print(f"  BUGGY R² drop: {A[0]:.3f} → {B[0]:.3f}  Δ = {B[0]-A[0]:+.3f}")
    print(f"  FIXED R² drop: {C[0]:.3f} → {D[0]:.3f}  Δ = {D[0]-C[0]:+.3f}")
    print(f"  BUGGY MAE rise: {A[1]:.3f} → {B[1]:.3f}  Δ = {B[1]-A[1]:+.3f}")
    print(f"  FIXED MAE rise: {C[1]:.3f} → {D[1]:.3f}  Δ = {D[1]-C[1]:+.3f}")


if __name__ == "__main__":
    main()
