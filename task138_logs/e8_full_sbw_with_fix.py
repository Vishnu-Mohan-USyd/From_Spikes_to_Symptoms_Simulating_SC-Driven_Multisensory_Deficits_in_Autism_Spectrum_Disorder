"""Task #138 E8 — FULL SBW curve on current Training.py with H6 fix applied.

This wraps compute_sbw_enhancement_persep to reset _last_agc_*_t alongside
step_counter. Expected: HW ≈ 25° (matches pristine paper SBW). If HW does
recover, this confirms the fix direction is correct.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import SBW_test as sbw
from SBW_test import load_msi_model

CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
DEVICE = "cuda"

# Monkey-patch run_pass inside compute_sbw_enhancement_persep
# Strategy: re-implement the function with the time-tracker reset
import torch.nn.functional as F


@torch.inference_mode()
def compute_sbw_enhancement_persep_FIXED(net, *, separations_deg, n_trials=50,
                                          intensity=1.0, duration=20, roi_half=20):
    N = net.n
    S = net.space_size
    rng = np.random.default_rng()

    initial_g_FFinh = float(net.g_FFinh)
    initial_step_counter = int(net.step_counter)
    initial_t_fast = float(getattr(net, "_last_agc_fast_t", 0.0))
    initial_t_slow = float(getattr(net, "_last_agc_slow_t", 0.0))

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
        # H6 fix: also reset AGC physical-time trackers to match the reset step
        if hasattr(net, "_last_agc_fast_t"):
            net._last_agc_fast_t = initial_step_counter * net.dt
            net._last_agc_slow_t = initial_step_counter * net.dt
        net.reset_state(batch_size=total_batch)
        msi_sum = torch.zeros(total_batch, N, device=net.device)
        for _ in range(duration):
            ret = net.update_all_layers_batch(stim_A, stim_V, return_spike_sum=True)
            sum_sM = ret[-1]
            msi_sum += sum_sM
        return msi_sum

    av_sum = run_pass(gA, gV)
    a_sum = run_pass(gA, zeros)
    v_sum = run_pass(zeros, gV)

    av_roi = roi_spikes(av_sum, idxA)
    a_roi = roi_spikes(a_sum, idxA)
    v_roi = roi_spikes(v_sum, idxA)
    enh = (av_roi - torch.max(a_roi, v_roi)).cpu().numpy()
    return enh.reshape(n_sep, n_trials)


def find_hw_one_sided(seps, p_fusion, thresh=0.5):
    """Symmetrize and find half-width crossing."""
    sym = {}
    for s, p in zip(seps, p_fusion):
        sym.setdefault(abs(s), []).append(p)
    mags = sorted(sym.keys())
    p_sym = [np.mean(sym[m]) for m in mags]
    for i in range(len(mags) - 1):
        if (p_sym[i] - thresh) * (p_sym[i + 1] - thresh) < 0:
            t = (thresh - p_sym[i]) / (p_sym[i + 1] - p_sym[i])
            return mags[i] + t * (mags[i + 1] - mags[i])
    return None


def main():
    print("=" * 92)
    print("E8: Full SBW curve on CURRENT Training.py — with H6 fix applied")
    print("=" * 92)

    seps = list(range(-80, 85, 5))
    net = load_msi_model(CKPT, device=DEVICE)
    print(f"  ckpt: g_FFinh={float(net.g_FFinh):.4f}")
    enh_per_sep = compute_sbw_enhancement_persep_FIXED(
        net, separations_deg=seps, n_trials=50, intensity=1.0, duration=20)
    p_fusion = (enh_per_sep > 10.0).mean(axis=1)

    print(f"\n{'sep':>5} {'mean_enh':>10} {'p(enh>10)':>11}")
    for s, e_row, p in zip(seps, enh_per_sep, p_fusion):
        print(f"{s:>5} {e_row.mean():10.2f} {p:11.3f}")

    hw = find_hw_one_sided(seps, p_fusion)
    print(f"\n→ HW (one-sided 0.5 crossing): {hw}°")
    print(f"→ Paper SBW HW target: 24.3°")
    print(f"→ Pristine ctrl HW (single ckpt, E2): ~25°")


if __name__ == "__main__":
    main()
