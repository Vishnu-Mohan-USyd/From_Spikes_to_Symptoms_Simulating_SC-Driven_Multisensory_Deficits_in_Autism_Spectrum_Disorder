"""H_spikes_components — measure per-component spikes and currents at dt=0.1 vs 0.05.

Phase 1 reproduced SBW HW drop 24.13 → 20.37° (-15.6%).
H_agc trace showed g_FFinh trajectories nearly identical at both dts but
inh_mean differs by 5% during stim, suggesting AGC is NOT the primary driver.

Now instrument the underlying mechanism: per-frame spike counts (A, V, MSI, MSI_inh, Out)
and the I_E (AMPA/NMDA) / I_I (FFInh/RecurInh/LatInh) components.

If at dt=0.05 the inh population fires MORE per wall-time, that's the mechanism.
If at dt=0.05 the exc population fires LESS per wall-time, that's another mechanism.

Test: M00 only, sync AV pulse intensity=1, batch=50 trials. Captures total spikes
per frame over 20-frame stim + 20-frame off.

This uses net._ei_record to capture per-substep AMPA/NMDA/FF/Recur/Lat means,
then aggregates per frame.
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


def run_and_count(dt: float, n_sub: int, *, n_frames: int = 40, n_trials: int = 50,
                  seed: int = 12345, gNMDA: float = 1.30):
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
        net.reset_state(batch_size=n_trials)

        N = net.n
        idx_c = N // 2
        xs = torch.arange(N, dtype=torch.float32, device=net.device)
        gauss = torch.exp(-0.5 * ((xs - idx_c) / net.sigma_in) ** 2) * 1.0
        stim = gauss.expand(n_trials, N).contiguous()
        zero = torch.zeros_like(stim)

        # Initialise EI recording (per-substep array buffers)
        net.start_ei_recording()

        per_frame = []
        for t in range(n_frames):
            sA_in = stim if t < 20 else zero
            sV_in = stim if t < 20 else zero
            # capture pre-frame ei_record length to slice per-frame block later
            pre_len = len(net._ei_record["I_E_mean"])
            # capture pre/post spike sums
            net.update_all_layers_batch(sA_in, sV_in)
            # net._latest_sA/V/MSI/MSI_inh hold LAST substep's spike mask only
            # We want total spikes for this frame; can sum the new spikes added
            # to the dbg counters internally.  But _dbg_spk_* accumulate across
            # frames so use pre/post diff.
            # Instead, use the ei_record which stores per-substep means
            post_len = len(net._ei_record["I_E_mean"])
            n_sub_seen = post_len - pre_len
            slice_E_ampa = np.array(net._ei_record["AMPA"][pre_len:post_len])
            slice_E_nmda = np.array(net._ei_record["NMDA"][pre_len:post_len])
            slice_I_ff   = np.array(net._ei_record["FFInh"][pre_len:post_len])
            slice_I_re   = np.array(net._ei_record["RecurInh"][pre_len:post_len])
            slice_I_la   = np.array(net._ei_record["LatInh"][pre_len:post_len])
            per_frame.append({
                "t": t,
                "n_sub": n_sub_seen,
                "I_E_ampa_mean": float(slice_E_ampa.mean()),
                "I_E_nmda_mean": float(slice_E_nmda.mean()),
                "I_I_ff_mean":   float(slice_I_ff.mean()),
                "I_I_re_mean":   float(slice_I_re.mean()),
                "I_I_la_mean":   float(slice_I_la.mean()),
                # Integral across the frame (per-substep mean * n_sub)
                "I_E_ampa_int":  float(slice_E_ampa.sum() * dt),
                "I_E_nmda_int":  float(slice_E_nmda.sum() * dt),
                "I_I_ff_int":    float(slice_I_ff.sum() * dt),
                "I_I_re_int":    float(slice_I_re.sum() * dt),
                "I_I_la_int":    float(slice_I_la.sum() * dt),
            })

        # cumulative spike counts (across frames + neurons + trials)
        sA_total = float(net._dbg_spk_A)
        sV_total = float(net._dbg_spk_V)
        sM_total = float(net._dbg_spk_MSI)
        net.stop_ei_recording()
        del net
        torch.cuda.empty_cache()
    finally:
        np.random.default_rng = orig

    return {
        "per_frame": per_frame,
        "total_spk_A": sA_total,
        "total_spk_V": sV_total,
        "total_spk_M": sM_total,
    }


def main():
    print("H_spikes_components — total spikes and per-component current INTEGRALS over frames")
    print("Sync AV pulse, intensity=1, 20-frame on + 20-frame off, n_trials=50, M00")
    print()

    print("--- dt=0.1, n=100 ---")
    r1 = run_and_count(0.1, 100)
    print(f"  Total A spikes (cumulative GPU): {r1['total_spk_A']:.0f}")
    print(f"  Total V spikes:                   {r1['total_spk_V']:.0f}")
    print(f"  Total MSI spikes:                 {r1['total_spk_M']:.0f}")

    print("\n--- dt=0.05, n=200 ---")
    r2 = run_and_count(0.05, 200)
    print(f"  Total A spikes (cumulative GPU): {r2['total_spk_A']:.0f}")
    print(f"  Total V spikes:                   {r2['total_spk_V']:.0f}")
    print(f"  Total MSI spikes:                 {r2['total_spk_M']:.0f}")

    print()
    print("Per-frame current INTEGRALS (current·dt, charge-like, units arbitrary)")
    print(f"{'frame':>5}  "
          f"{'A_amp(.1)':>9}  {'A_amp(.05)':>10}  {'r':>5}  "
          f"{'A_nmd(.1)':>9}  {'A_nmd(.05)':>10}  {'r':>5}  "
          f"{'I_ff(.1)':>9}  {'I_ff(.05)':>10}  {'r':>5}  "
          f"{'I_la(.1)':>9}  {'I_la(.05)':>10}  {'r':>5}")
    for f1, f2 in zip(r1["per_frame"], r2["per_frame"]):
        def ratio(a, b):
            return f"{(b/a) if abs(a)>1e-9 else 0.0:.2f}"
        print(f"  {f1['t']:>3d}   "
              f"{f1['I_E_ampa_int']:>9.3f}  {f2['I_E_ampa_int']:>10.3f}  {ratio(f1['I_E_ampa_int'], f2['I_E_ampa_int']):>5}  "
              f"{f1['I_E_nmda_int']:>9.3f}  {f2['I_E_nmda_int']:>10.3f}  {ratio(f1['I_E_nmda_int'], f2['I_E_nmda_int']):>5}  "
              f"{f1['I_I_ff_int']:>9.3f}  {f2['I_I_ff_int']:>10.3f}  {ratio(f1['I_I_ff_int'], f2['I_I_ff_int']):>5}  "
              f"{f1['I_I_la_int']:>9.3f}  {f2['I_I_la_int']:>10.3f}  {ratio(f1['I_I_la_int'], f2['I_I_la_int']):>5}")

    print()
    print("Per-component frame-sums over STIM window (frames 0-19):")
    pf1 = r1["per_frame"][:20]
    pf2 = r2["per_frame"][:20]
    for key in ["I_E_ampa_int", "I_E_nmda_int", "I_I_ff_int", "I_I_re_int", "I_I_la_int"]:
        s1 = sum(f[key] for f in pf1)
        s2 = sum(f[key] for f in pf2)
        print(f"  {key:>16s}: dt=0.1 = {s1:>10.3f}    dt=0.05 = {s2:>10.3f}    "
              f"ratio = {(s2/s1) if s1!=0 else 0.0:.3f}    "
              f"abs Δ = {s2-s1:+.3f}")


if __name__ == "__main__":
    main()
