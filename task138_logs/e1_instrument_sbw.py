"""Task #138 E1 — instrument single-ckpt control SBW; dump raw enhancement
distributions and compare `sum_sM` (current SBW_test path) vs `_latest_sMSI`
(pristine fb6d3f6 path) on the SAME forward pass.

Hypothesis (H5 PRIMARY): SBW_test.py changed from
  `msi_sum += net._latest_sMSI`  (pristine, accumulates LAST substep only)
to
  `msi_sum += sum_sM`  (current, accumulates ALL n_substeps=100)
which inflates enhancement values ~100× and trivially crosses threshold=10
at all separations, even ±80°.

Test: run SBW at sep ∈ {0, 40, 80}, capture BOTH sum_sM (current) and
_latest_sMSI (pristine) accumulations on the same forward pass. Print
enhancement distributions for each path.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from SBW_test import load_msi_model

CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
DEVICE = "cuda"
N_TRIALS = 50
DURATION = 20
INTENSITY = 1.0
ROI_HALF = 20


def measure_enh_dual_path(net, sep_deg, intensity=INTENSITY, duration=DURATION,
                          roi_half=ROI_HALF, n_trials=N_TRIALS, seed=42):
    """Run AV / A / V passes for one separation; return enhancement for BOTH paths."""
    N = net.n
    S = net.space_size

    rng = np.random.default_rng(seed)
    xs = torch.arange(N, device=net.device, dtype=torch.float32)

    initial_g_FFinh = net.g_FFinh
    initial_step_counter = net.step_counter

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

    # Generate stimuli
    base = rng.integers(0, S, size=n_trials)
    locA = base
    locV = (base + sep_deg) % S
    idxA = to_idx(locA)
    idxV = to_idx(locV)
    gA = make_gauss(idxA)
    gV = make_gauss(idxV)
    zeros = torch.zeros_like(gA)

    def run_pass(stim_A, stim_V):
        """Run forward pass; capture BOTH sum_sM (current) and last-sub _latest_sMSI accumulation."""
        net.g_FFinh = initial_g_FFinh
        net.step_counter = initial_step_counter
        net.reset_state(batch_size=n_trials)

        # Current SBW path: accumulate sum_sM (sum over substeps per frame)
        msi_sum_current = torch.zeros(n_trials, net.n, device=net.device)
        # Pristine SBW path: accumulate _latest_sMSI (last substep per frame)
        msi_sum_pristine = torch.zeros(n_trials, net.n, device=net.device)

        for _ in range(duration):
            ret = net.update_all_layers_batch(stim_A, stim_V, return_spike_sum=True)
            sum_sM = ret[-1]
            msi_sum_current += sum_sM
            msi_sum_pristine += net._latest_sMSI  # last substep only

        return msi_sum_current, msi_sum_pristine

    av_cur, av_pri = run_pass(gA, gV)
    a_cur, a_pri = run_pass(gA, zeros)
    v_cur, v_pri = run_pass(zeros, gV)

    av_roi_cur = roi_spikes(av_cur, idxA)
    a_roi_cur = roi_spikes(a_cur, idxA)
    v_roi_cur = roi_spikes(v_cur, idxA)
    enh_cur = (av_roi_cur - torch.max(a_roi_cur, v_roi_cur)).cpu().numpy()

    av_roi_pri = roi_spikes(av_pri, idxA)
    a_roi_pri = roi_spikes(a_pri, idxA)
    v_roi_pri = roi_spikes(v_pri, idxA)
    enh_pri = (av_roi_pri - torch.max(a_roi_pri, v_roi_pri)).cpu().numpy()

    return {
        "enh_current_sum_sM": enh_cur,
        "enh_pristine_latest_sMSI": enh_pri,
        "av_roi_current_mean": float(av_roi_cur.mean().item()),
        "a_roi_current_mean": float(a_roi_cur.mean().item()),
        "v_roi_current_mean": float(v_roi_cur.mean().item()),
        "av_roi_pristine_mean": float(av_roi_pri.mean().item()),
        "a_roi_pristine_mean": float(a_roi_pri.mean().item()),
        "v_roi_pristine_mean": float(v_roi_pri.mean().item()),
    }


def main():
    print("=" * 84)
    print("E1: SBW enhancement distribution — current (sum_sM) vs pristine (_latest_sMSI)")
    print("  ckpt: msi_model_surr_10_00.pt  |  dt:", end=" ")
    net = load_msi_model(CKPT, device=DEVICE)
    print(f"{net.dt}  n_substeps: {net.n_substeps}")
    print(f"  n_trials={N_TRIALS}, duration={DURATION}, intensity={INTENSITY}, roi_half={ROI_HALF}")
    print("=" * 84)

    print(f"\n{'sep':>5} {'path':>10} {'AV_roi':>9} {'A_roi':>9} {'V_roi':>9} "
          f"{'enh_mean':>10} {'enh_min':>9} {'enh_max':>9} {'enh>10':>8} {'enh>0':>7}")
    print("-" * 84)

    # MATCH cache-generation conditions exactly:
    # - load_msi_model sets plasticity_enabled=False (matches cache)
    # - freeze_g_FFinh: cache DID NOT set this → leave AGC drift active
    for sep in [0, 40, 80]:
        result = measure_enh_dual_path(net, sep_deg=sep)

        for path_name in ["pristine_latest_sMSI", "current_sum_sM"]:
            key = f"enh_{path_name}"
            enh = result[key]
            tag = "pristine" if "pri" in path_name else "current"
            av = result[f"av_roi_{tag}_mean"]
            a = result[f"a_roi_{tag}_mean"]
            v = result[f"v_roi_{tag}_mean"]
            pct_above_10 = (enh > 10).mean()
            pct_above_0 = (enh > 0).mean()
            print(f"{sep:>5} {path_name:>10} {av:9.2f} {a:9.2f} {v:9.2f} "
                  f"{enh.mean():10.2f} {enh.min():9.1f} {enh.max():9.1f} "
                  f"{pct_above_10:8.2f} {pct_above_0:7.2f}")

    print("\n" + "=" * 84)
    print("Diagnostic verdict")
    print("-" * 84)
    print("If current/sum_sM enh is ~n_substeps× pristine/_latest_sMSI enh, and pristine")
    print("path gives enh ~< 10 at sep=80 while current gives enh >> 10 at sep=80, then")
    print("H5 is CONFIRMED: the cache-time SBW_test code path inflates enhancement values")
    print("via per-substep accumulation, while pristine threshold=10 was calibrated for")
    print("last-substep-only accumulation.")
    print("=" * 84)


if __name__ == "__main__":
    main()
