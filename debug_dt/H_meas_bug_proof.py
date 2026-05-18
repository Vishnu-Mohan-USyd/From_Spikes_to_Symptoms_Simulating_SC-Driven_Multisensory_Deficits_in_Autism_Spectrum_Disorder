"""H_meas — CAUSAL PROOF: SBW dt-drift is a MEASUREMENT bug.

SBW_test.py line 663:  msi_sum += net._latest_sMSI

_latest_sMSI is the LAST SUBSTEP's spike mask (binary 0/1), not the per-frame
spike total.  Per-substep mean = rate * dt → HALVED at dt=0.05 vs dt=0.1.

With enhancement threshold=10 fixed, halved spike counts cause:
  at sep≠0: enhancement falls below 10 → "not fused"  → P(fusion) drops
  at sep=0: enhancement still > 10 → "fused"          → P(fusion) holds
  net effect: NARROWER P(fusion) curve → SMALLER HW.

PROOF:
  We monkey-patch compute_sbw_enhancement_persep to use return_spike_sum=True
  (which returns the TRUE per-frame spike sum, summing across all substeps).
  If the dt-drift VANISHES, the bug is confirmed.

Forward direction:
  dt=0.1   with TRUE measurement: should equal current ~24°
  dt=0.05  with TRUE measurement: should equal ~24° (drift gone)
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import inspect

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import SBW_test as SBW
import Training as TRN
from TBW_test import _find_crossings
from replot_all_cosmetic import subtract_control_floor, fit_sbw_pedestal

BASE = ROOT / "checkpoint"
MODELS = [BASE / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
SBW_SEPARATIONS = tuple(range(-80, 85, 5))
ENH_THRESHOLD = 10.0
N_TRIALS = 50


def make_modify_net(dt_val: float, n_sub: int, *, gNMDA: float = 1.30):
    def _mod(n):
        n.dt = float(dt_val)
        n.n_substeps = int(n_sub)
        n.dt_correct_nmda = True
        n.gNMDA = float(gNMDA)
        n.plasticity_enabled = False
        n.freeze_g_FFinh = False
        n._reset_delay_buffers()
    return _mod


# ────────────────────────────────────────────────────────────────────
# Patched compute_sbw_enhancement_persep that uses return_spike_sum=True
# instead of _latest_sMSI (last substep only).
# ────────────────────────────────────────────────────────────────────
ORIG_COMPUTE_SBW = SBW.compute_sbw_enhancement_persep


def compute_sbw_fixed(net, *, separations_deg, n_trials=50, intensity=0.5,
                     duration=20, block_size=32, roi_half=20):
    """Replica of compute_sbw_enhancement_persep but sums TRUE per-frame spikes."""
    import torch
    N = net.n
    S = net.space_size
    rng = np.random.default_rng()

    initial_g_FFinh = net.g_FFinh
    initial_step_counter = net.step_counter

    xs = torch.arange(N, device=net.device, dtype=torch.float32)

    n_sep = len(separations_deg)
    total_batch = n_sep * n_trials

    def to_idx(deg):
        return torch.round(
            torch.as_tensor(deg, device=net.device, dtype=torch.float32)
            * (N - 1) / (S - 1)).long()

    def make_gauss(idx_centres):
        return torch.exp(
            -0.5 * ((xs - idx_centres[:, None]) / net.sigma_in) ** 2
        ) * intensity

    def roi_spikes(msi_sum, center_idx):
        offsets = torch.arange(-roi_half, roi_half + 1, device=msi_sum.device)
        indices = (center_idx[:, None] + offsets[None, :]) % N
        return msi_sum.gather(1, indices).sum(dim=1)

    all_locA = np.zeros(total_batch, dtype=int)
    all_locV = np.zeros(total_batch, dtype=int)
    for k, sep in enumerate(separations_deg):
        s, e = k * n_trials, (k + 1) * n_trials
        base = rng.integers(0, S, size=n_trials)
        all_locA[s:e] = base
        all_locV[s:e] = (base + sep) % S

    idxA = to_idx(all_locA)
    idxV = to_idx(all_locV)
    gA = make_gauss(idxA)
    gV = make_gauss(idxV)
    zeros = torch.zeros_like(gA)

    def run_pass(stim_A, stim_V):
        net.g_FFinh = initial_g_FFinh
        net.step_counter = initial_step_counter
        net.reset_state(batch_size=total_batch)
        msi_sum = torch.zeros(total_batch, N, device=net.device)
        for _ in range(duration):
            # KEY CHANGE: use return_spike_sum=True to get the TRUE per-frame sum
            ret = net.update_all_layers_batch(stim_A, stim_V, return_spike_sum=True)
            # vanilla branch returns (sA, sV, sM, sO, sum_sM)
            sum_sM = ret[-1]
            msi_sum += sum_sM
        return msi_sum

    av_sum = run_pass(gA, gV)
    a_sum = run_pass(gA, zeros)
    v_sum = run_pass(zeros, gV)

    av_roi = roi_spikes(av_sum, idxA)
    a_roi = roi_spikes(a_sum, idxA)
    v_roi = roi_spikes(v_sum, idxA)
    enh_all = (av_roi - torch.max(a_roi, v_roi)).cpu().numpy()

    del gA, gV, zeros, av_sum, a_sum, v_sum
    if net.device.type == "cuda":
        torch.cuda.empty_cache()

    enh_reshaped = enh_all.reshape(n_sep, n_trials)
    mean_enhancement = enh_reshaped.mean(axis=1)
    all_trial_enh = [enh_reshaped[k] for k in range(n_sep)]

    net.g_FFinh = initial_g_FFinh
    net.step_counter = initial_step_counter
    return mean_enhancement, all_trial_enh


def fit_hw(pooled):
    try:
        xs_fit, ys_fit, popt, r2 = fit_sbw_pedestal(pooled)
        cross = _find_crossings(xs_fit, ys_fit, 0.5)
        if len(cross) >= 2:
            return (cross[-1] - cross[0]) / 2.0, popt, r2
        return float("nan"), popt, r2
    except Exception as e:
        print(f"   fit failed: {e}")
        return float("nan"), None, 0.0


def measure(*, dt: float, n_sub: int, use_fixed: bool):
    """Use canonical pipeline; optionally patch compute_sbw_enhancement_persep."""
    SBW.compute_sbw_enhancement_persep = compute_sbw_fixed if use_fixed else ORIG_COMPUTE_SBW
    try:
        t0 = time.time()
        pooled = SBW.run_spatial_binding_across_models(
            MODELS, separations_deg=SBW_SEPARATIONS, device="cuda",
            method="enhancement", n_trials=N_TRIALS,
            enhancement_threshold=ENH_THRESHOLD,
            modify_net=make_modify_net(dt, n_sub),
        )
        el = time.time() - t0
    finally:
        SBW.compute_sbw_enhancement_persep = ORIG_COMPUTE_SBW

    sbw_sub, floor = subtract_control_floor({"control": pooled})
    hw, popt, r2 = fit_hw(sbw_sub["control"])
    return hw, floor, r2, el


def main():
    print("H_meas — CAUSAL PROOF: SBW dt-drift caused by _latest_sMSI measurement bug")
    print(f"  10 ckpts, n_trials={N_TRIALS}, gNMDA=1.30, Form-2 ON, delays-ms ON")
    print()

    conds = [
        ("A: BUGGY measurement, dt=0.1 ", 0.1,  100, False),
        ("B: BUGGY measurement, dt=0.05", 0.05, 200, False),
        ("C: FIXED measurement, dt=0.1 ", 0.1,  100, True),
        ("D: FIXED measurement, dt=0.05", 0.05, 200, True),
    ]
    results = {}
    for label, dt, n_sub, use_fixed in conds:
        print(f"--- {label} ---")
        hw, floor, r2, el = measure(dt=dt, n_sub=n_sub, use_fixed=use_fixed)
        results[label] = (hw, floor, r2)
        print(f"   HW={hw:.2f}°  floor={floor:.4f}  r2={r2:.4f}  el={el:.0f}s")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    hwA, fA, _ = results["A: BUGGY measurement, dt=0.1 "]
    hwB, fB, _ = results["B: BUGGY measurement, dt=0.05"]
    hwC, fC, _ = results["C: FIXED measurement, dt=0.1 "]
    hwD, fD, _ = results["D: FIXED measurement, dt=0.05"]

    print(f"  BUGGY meas drift:  ΔHW = {hwB-hwA:+.2f}° ({(hwB-hwA)/hwA*100:+.1f}%)")
    print(f"  FIXED meas drift:  ΔHW = {hwD-hwC:+.2f}° ({(hwD-hwC)/hwC*100:+.1f}%)")
    print()
    print(f"  HW(BUGGY  0.1)={hwA:.2f}   HW(FIXED  0.1)={hwC:.2f}   "
          f"Δ_within_0.1={hwC-hwA:+.2f}")
    print(f"  HW(BUGGY 0.05)={hwB:.2f}   HW(FIXED 0.05)={hwD:.2f}   "
          f"Δ_within_0.05={hwD-hwB:+.2f}")
    print()
    print(f"  Floor(BUGGY  0.1)={fA:.4f}   Floor(FIXED  0.1)={fC:.4f}")
    print(f"  Floor(BUGGY 0.05)={fB:.4f}   Floor(FIXED 0.05)={fD:.4f}")


if __name__ == "__main__":
    main()
