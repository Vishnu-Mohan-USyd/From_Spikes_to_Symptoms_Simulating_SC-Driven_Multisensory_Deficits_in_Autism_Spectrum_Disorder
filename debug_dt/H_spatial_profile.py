"""H_spatial — measure MSI spike spatial profile at dt=0.1 vs dt=0.05.

SBW HW shrinks 24°→20° (-15.6%). Total MSI spikes are +4% only. So spatial
DISTRIBUTION of MSI spikes must SHARPEN at dt=0.05.

Test: run identical AV stim at three separations (0°, 25°, 80°) and measure
the MSI spike count profile as a function of neuron position relative to the
auditory stim location.  Use n_trials=50 with deterministic RNG.

If profile FWHM decreases at dt=0.05, that's the network sharpening.  Then we
need to identify which mechanism causes it.
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


def run_one_sep(dt: float, n_sub: int, *, sep_deg: int, n_trials: int = 50,
                seed: int = 12345, gNMDA: float = 1.30,
                duration: int = 20, intensity: float = 1.0,
                cond: str = "AV"):
    """Run one separation, return MSI spike count profile (averaged over trials).

    cond: 'AV' = both A and V stim; 'A' = A only; 'V' = V only.
    Returns (profile of length N centered on idx_A, total_spikes_in_ROI).
    """
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
        net.freeze_g_FFinh = False
        net._reset_delay_buffers()

        N = net.n
        S = net.space_size
        roi_half = 20

        def to_idx(deg):
            return int(round(deg * (N - 1) / (S - 1)))

        # Generate a single base location for all trials (deterministic)
        base = rng_fixed.integers(0, S, size=n_trials)
        idxA_np = np.array([to_idx(int(b)) for b in base])
        idxV_np = np.array([to_idx(int((b + sep_deg) % S)) for b in base])
        idxA = torch.as_tensor(idxA_np, device=net.device, dtype=torch.long)
        idxV = torch.as_tensor(idxV_np, device=net.device, dtype=torch.long)

        xs = torch.arange(N, device=net.device, dtype=torch.float32)
        def make_gauss(centres):
            return torch.exp(-0.5 * ((xs - centres[:, None]) / net.sigma_in) ** 2) * intensity

        gA = make_gauss(idxA)
        gV = make_gauss(idxV)
        zero = torch.zeros_like(gA)

        if cond == "AV":
            stim_A, stim_V = gA, gV
        elif cond == "A":
            stim_A, stim_V = gA, zero
        elif cond == "V":
            stim_A, stim_V = zero, gV
        else:
            raise ValueError(cond)

        net.reset_state(batch_size=n_trials)
        msi_sum = torch.zeros(n_trials, N, device=net.device)
        for _ in range(duration):
            net.update_all_layers_batch(stim_A, stim_V)
            msi_sum += net._latest_sMSI

        # Center the per-trial profiles on idxA, then average across trials.
        # For each trial, roll the profile so idxA is at index 0 (centered later).
        msi_sum_np = msi_sum.cpu().numpy()  # (B, N)
        centered = np.zeros((n_trials, N))
        for i in range(n_trials):
            ia = int(idxA_np[i])
            centered[i] = np.roll(msi_sum_np[i], -ia)
        profile = centered.mean(axis=0)  # length N, centered at 0 (== idxA)
        total_roi = float(centered[:, list(range(0, roi_half+1)) +
                                  list(range(N-roi_half, N))].sum() / n_trials)
        del net
        torch.cuda.empty_cache()
    finally:
        np.random.default_rng = orig
    return profile, total_roi


def fwhm(profile):
    """Compute FWHM of a 'centered' profile (centered at index 0, wraps around).
    Roll so center is at len/2 for easier processing."""
    N = len(profile)
    rolled = np.roll(profile, N // 2)  # center at N//2
    peak = rolled.max()
    if peak <= 0:
        return float("nan")
    half = peak / 2.0
    above = rolled >= half
    if not above.any():
        return float("nan")
    idxs = np.where(above)[0]
    return float(idxs[-1] - idxs[0] + 1)  # in neuron index units


def main():
    print("H_spatial_profile — MSI spike spatial profile at dt=0.1 vs dt=0.05")
    print("3 separations × 3 conditions (AV/A/V) × n_trials=50, deterministic seed")
    print()

    seps = [0, 25, 80]
    conds = ["AV", "A", "V"]

    results = {}
    for cond in conds:
        for sep in seps:
            p1, roi1 = run_one_sep(0.1, 100, sep_deg=sep, cond=cond)
            p2, roi2 = run_one_sep(0.05, 200, sep_deg=sep, cond=cond)
            results[(cond, sep)] = (p1, roi1, p2, roi2)
            fwhm1 = fwhm(p1)
            fwhm2 = fwhm(p2)
            print(f"  {cond:>3s}  sep={sep:>3d}°:  "
                  f"FWHM(0.1)={fwhm1:>5.1f} pix  FWHM(0.05)={fwhm2:>5.1f} pix  "
                  f"Δ={fwhm2-fwhm1:+.1f}    "
                  f"ROI_spikes(0.1)={roi1:>5.1f}  ROI(0.05)={roi2:>5.1f}  ratio={roi2/roi1 if roi1>0 else 0:.3f}")

    print()
    print("Enhancement at each sep (AV_ROI - max(A_ROI, V_ROI)):")
    for sep in seps:
        _, av1, _, av2 = results[("AV", sep)]
        _, a1, _, a2 = results[("A", sep)]
        _, v1, _, v2 = results[("V", sep)]
        e1 = av1 - max(a1, v1)
        e2 = av2 - max(a2, v2)
        print(f"  sep={sep:>3d}°:  enh(0.1)={e1:>+6.2f}  enh(0.05)={e2:>+6.2f}  Δ={e2-e1:+.2f}")

    print()
    print("Profile center spike count (peak):")
    N = len(results[("AV", 0)][0])
    for cond in conds:
        for sep in seps:
            p1, _, p2, _ = results[(cond, sep)]
            print(f"  {cond:>3s}  sep={sep:>3d}°:  peak(0.1)={p1[0]:>5.2f}  "
                  f"peak(0.05)={p2[0]:>5.2f}  ratio={p2[0]/p1[0] if p1[0]>0 else 0:.3f}")


if __name__ == "__main__":
    main()
