"""ISSUE B causal isolation — one-variable-at-a-time.

Production (R²=0.287) and Wrapper (R²=0.971) differ in three settable
parameters:
  1. gNMDA           : 0.05 (ckpt)  vs  1.30 (forced)
  2. plasticity_enabled : True (default) vs False (forced)
  3. _reset_delay_buffers() : not called vs called

For each of 8 conditions (2x2x2 across these three flags) we measure R²
on the same checkpoints, same dataset, same return_spike_sum=True
measurement (both scripts already use it).  If a single parameter
explains the entire 0.685 R² gap, the table makes it unambiguous.

We then run REVERSE confirmation in two directions:
  - production baseline + ONLY that one knob flipped to wrapper value
  - wrapper baseline    + ONLY that one knob flipped to production value
"""
from __future__ import annotations
import sys, time, itertools
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cue_reliability_test import (
    make_dataset, reliability_sweep_batched, pool_to_mean_sem,
    load_msi_model,
)

BASE = ROOT / "checkpoint"
MODELS = [BASE / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]


def compute_metrics(pooled):
    w_pred_all, w_emp_all, abs_errs, sq = [], [], [], []
    for d in pooled:
        var_a, var_v = d["sigma_a"] ** 2, d["sigma_v"] ** 2
        w_pred = 1.0 / var_v / (1.0 / var_a + 1.0 / var_v)
        w_emp = d["mean_w"]
        w_pred_all.append(w_pred); w_emp_all.append(w_emp)
        abs_errs.append(abs(w_emp - w_pred))
        sq.append((w_emp - w_pred) ** 2)
    r2 = float(np.corrcoef(w_pred_all, w_emp_all)[0, 1] ** 2)
    mae = float(np.mean(abs_errs))
    rmse = float(np.sqrt(np.mean(sq)))
    return r2, mae, rmse


def run_one(*, gNMDA, plasticity, reset_delays, n_trials=20, label=""):
    xA, xV, meta = make_dataset(n_trials=n_trials, device="cuda")
    all_res = []
    t0 = time.time()
    for p in MODELS:
        net = load_msi_model(p, device="cuda")
        net._probe = None
        net.allow_inhib_plasticity = False
        if gNMDA is not None:
            net.gNMDA = float(gNMDA)
        if plasticity is not None:
            net.plasticity_enabled = bool(plasticity)
        if reset_delays:
            net._reset_delay_buffers()
        res = reliability_sweep_batched(net, xA, xV, meta)
        all_res.append(res)
        del net; torch.cuda.empty_cache()
    pooled = pool_to_mean_sem(all_res)
    r2, mae, rmse = compute_metrics(pooled)
    el = time.time() - t0
    print(f"{label:<55s}  R²={r2:.4f}  MAE={mae:.4f}  el={el:.0f}s", flush=True)
    return r2, mae, rmse


def main():
    print("ISSUE B isolation — one variable at a time")
    print("=" * 75)
    print()
    print("Baselines:")
    print("-" * 75)
    # Production baseline: gNMDA from ckpt (0.05), plasticity True, no reset
    r2_prod, _, _ = run_one(
        gNMDA=None, plasticity=None, reset_delays=False,
        label="PROD baseline (gNMDA=ckpt, plast=T, reset=F)"
    )
    # Wrapper baseline: gNMDA=1.30, plasticity False, reset True
    r2_wrap, _, _ = run_one(
        gNMDA=1.30, plasticity=False, reset_delays=True,
        label="WRAP baseline (gNMDA=1.30, plast=F, reset=T)"
    )

    print()
    print("FORWARD: production baseline + ONE wrapper knob:")
    print("-" * 75)
    r2_p_gnmda, _, _ = run_one(
        gNMDA=1.30, plasticity=None, reset_delays=False,
        label="PROD + gNMDA=1.30   (only this)"
    )
    r2_p_plast, _, _ = run_one(
        gNMDA=None, plasticity=False, reset_delays=False,
        label="PROD + plast=False  (only this)"
    )
    r2_p_reset, _, _ = run_one(
        gNMDA=None, plasticity=None, reset_delays=True,
        label="PROD + reset_delays (only this)"
    )

    print()
    print("REVERSE: wrapper baseline - ONE wrapper knob:")
    print("-" * 75)
    r2_w_gnmda, _, _ = run_one(
        gNMDA=None, plasticity=False, reset_delays=True,
        label="WRAP - gNMDA  (gNMDA=ckpt, plast=F, reset=T)"
    )
    r2_w_plast, _, _ = run_one(
        gNMDA=1.30, plasticity=True, reset_delays=True,
        label="WRAP - plast  (gNMDA=1.30, plast=T, reset=T)"
    )
    r2_w_reset, _, _ = run_one(
        gNMDA=1.30, plasticity=False, reset_delays=False,
        label="WRAP - reset  (gNMDA=1.30, plast=F, reset=F)"
    )

    print()
    print("=" * 75)
    print("VERDICT")
    print("=" * 75)
    print(f"  Baseline gap:                  ΔR² = {r2_wrap - r2_prod:+.4f}")
    print()
    print(f"  PROD + gNMDA=1.30:             ΔR² = {r2_p_gnmda - r2_prod:+.4f}")
    print(f"  PROD + plast=False:            ΔR² = {r2_p_plast - r2_prod:+.4f}")
    print(f"  PROD + reset_delays:           ΔR² = {r2_p_reset - r2_prod:+.4f}")
    print()
    print(f"  WRAP - gNMDA (use ckpt):       ΔR² = {r2_w_gnmda - r2_wrap:+.4f}")
    print(f"  WRAP - plast (re-enable):      ΔR² = {r2_w_plast - r2_wrap:+.4f}")
    print(f"  WRAP - reset (skip reset):     ΔR² = {r2_w_reset - r2_wrap:+.4f}")


if __name__ == "__main__":
    main()
