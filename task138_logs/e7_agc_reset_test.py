"""Task #138 E7: test H6 — current Training.py's AGC physical-time gates
(_last_agc_fast_t, _last_agc_slow_t) persist across SBW passes while
step_counter is reset. This causes AGC to STOP FIRING in subsequent passes
because (current_t_ms - _last_agc_*_t) goes negative.

Test plan:
  Variant A: load CURRENT Training.py, run AV → A → V passes WITHOUT resetting
             _last_agc fields between passes (= existing buggy behavior).
  Variant B: load CURRENT Training.py, run AV → A → V passes WITH resetting
             _last_agc fields to step_counter*dt at start of each pass.

  In each variant, dump A_roi and V_roi means. If H6 is correct:
    Variant A: A_roi ≈ 0.66 (matches buggy E1)
    Variant B: A_roi ≈ 64 (matches PRISTINE E3)
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


def measure_3passes(net, sep_deg, reset_agc_t_each_pass=False, seed=42):
    N = net.n
    S = net.space_size
    rng = np.random.default_rng(seed)
    xs = torch.arange(N, device=net.device, dtype=torch.float32)
    initial_g_FFinh = float(net.g_FFinh)
    initial_step_counter = int(net.step_counter)

    def to_idx(deg):
        return torch.round(
            torch.as_tensor(deg, device=net.device, dtype=torch.float32)
            * (N - 1) / (S - 1)).long()

    def make_gauss(idx_centres):
        return torch.exp(
            -0.5 * ((xs - idx_centres[:, None]) / net.sigma_in) ** 2
        ) * INTENSITY

    def roi_spikes(msi_sum, center_idx):
        offsets = torch.arange(-ROI_HALF, ROI_HALF + 1, device=msi_sum.device)
        indices = (center_idx[:, None] + offsets[None, :]) % N
        return msi_sum.gather(1, indices).sum(dim=1)

    base = rng.integers(0, S, size=N_TRIALS)
    locA = base
    locV = (base + sep_deg) % S
    idxA = to_idx(locA)
    idxV = to_idx(locV)
    gA = make_gauss(idxA)
    gV = make_gauss(idxV)
    zeros = torch.zeros_like(gA)

    def run_pass(stim_A, stim_V):
        net.g_FFinh = initial_g_FFinh
        net.step_counter = initial_step_counter
        if reset_agc_t_each_pass:
            # Reset _last_agc time-tracker to step_counter*dt so AGC fires normally
            current_t_ms = net.step_counter * net.dt
            if hasattr(net, "_last_agc_fast_t"):
                net._last_agc_fast_t = current_t_ms
                net._last_agc_slow_t = current_t_ms
        net.reset_state(batch_size=N_TRIALS)
        msi_sum_pri = torch.zeros(N_TRIALS, net.n, device=net.device)
        msi_sum_cur = torch.zeros(N_TRIALS, net.n, device=net.device)
        for _ in range(DURATION):
            ret = net.update_all_layers_batch(stim_A, stim_V, return_spike_sum=True)
            sum_sM = ret[-1]
            msi_sum_pri += net._latest_sMSI
            msi_sum_cur += sum_sM
        g_end = float(net.g_FFinh)
        agc_fast_t = float(getattr(net, '_last_agc_fast_t', float('nan')))
        agc_slow_t = float(getattr(net, '_last_agc_slow_t', float('nan')))
        return msi_sum_pri, msi_sum_cur, g_end, agc_fast_t, agc_slow_t

    av_pri, av_cur, g_av, t_fast_av, t_slow_av = run_pass(gA, gV)
    a_pri, a_cur, g_a, t_fast_a, t_slow_a = run_pass(gA, zeros)
    v_pri, v_cur, g_v, t_fast_v, t_slow_v = run_pass(zeros, gV)

    av_roi_pri = roi_spikes(av_pri, idxA).mean().item()
    a_roi_pri = roi_spikes(a_pri, idxA).mean().item()
    v_roi_pri = roi_spikes(v_pri, idxA).mean().item()
    av_roi_cur = roi_spikes(av_cur, idxA).mean().item()
    a_roi_cur = roi_spikes(a_cur, idxA).mean().item()
    v_roi_cur = roi_spikes(v_cur, idxA).mean().item()

    return {
        "av_roi_pri": av_roi_pri, "a_roi_pri": a_roi_pri, "v_roi_pri": v_roi_pri,
        "av_roi_cur": av_roi_cur, "a_roi_cur": a_roi_cur, "v_roi_cur": v_roi_cur,
        "g_end": [g_av, g_a, g_v],
        "agc_fast_t_end": [t_fast_av, t_fast_a, t_fast_v],
        "agc_slow_t_end": [t_slow_av, t_slow_a, t_slow_v],
    }


def main():
    print("=" * 100)
    print("E7: TEST H6 — CURRENT Training.py with vs without AGC time-tracker reset between passes")
    print("=" * 100)

    for sep in [0, 80]:
        print(f"\n--- sep_deg={sep} ---")
        for label, reset_flag in [("VARIANT A (buggy: NO reset)", False),
                                   ("VARIANT B (fix:  YES reset)", True)]:
            net = load_msi_model(CKPT, device=DEVICE)
            res = measure_3passes(net, sep_deg=sep, reset_agc_t_each_pass=reset_flag)
            print(f"  {label}")
            print(f"    pristine path (_latest_sMSI): AV={res['av_roi_pri']:7.2f} "
                  f"A={res['a_roi_pri']:7.2f} V={res['v_roi_pri']:7.2f}")
            print(f"    current  path (sum_sM):       AV={res['av_roi_cur']:9.2f} "
                  f"A={res['a_roi_cur']:9.2f} V={res['v_roi_cur']:9.2f}")
            print(f"    g_FFinh at end of [AV, A, V] passes: {[f'{g:.4f}' for g in res['g_end']]}")
            print(f"    _last_agc_slow_t at end:             {res['agc_slow_t_end']}")
            del net
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
