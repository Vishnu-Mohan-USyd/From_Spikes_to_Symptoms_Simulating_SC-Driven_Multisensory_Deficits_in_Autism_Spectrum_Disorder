"""Decompose per-substep currents with Form 2 fix ON, gNMDA=1.30.

Inventory all dt-dependent paths. For each, compute mean per substep AND
mean per ms (= per_substep / dt) AND mean per frame (= per_substep * nsub).

For continuous-source-in-leaky-decay paths: per_frame should be dt-invariant
if correctly scaled, or should scale as 1/dt if bare-add (the unfixed bug).

For spike-driven paths: per_frame should be dt-invariant naturally.

Outputs comparison table dt=0.1 vs dt=0.05 with ratios.
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from TBW_test import load_msi_model
from Training import generate_two_event_offset_seq, generate_av_batch_tensor

CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"


def run_with_recording(dt, nsub, *, T=60, D=5, n_trials=8, seed=12345,
                       enable_form2=True, gNMDA=1.30):
    rng = np.random.default_rng(seed)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        net = load_msi_model(CKPT, device="cuda")
        net.dt = dt
        net.n_substeps = nsub
        net.dt_correct_nmda = enable_form2
        net.gNMDA = gNMDA
        net.plasticity_enabled = False
        net.freeze_g_FFinh = True

        # Print bias snapshot
        b_msi = net.b_msi.abs().mean().item()
        b_msi_inh = net.b_msi_inh.abs().mean().item()
        b_out = net.b_out.abs().mean().item()
        b_uniA = net.b_uniA.abs().mean().item()
        b_uniV = net.b_uniV.abs().mean().item()

        rng2 = np.random.default_rng(42)
        loc_seqs, mod_seqs = [], []
        for _ in range(n_trials):
            loc = int(rng2.integers(0, net.space_size))
            ls, ms = generate_two_event_offset_seq(
                loc=loc, T=T, D=D, offset=0, space_size=net.space_size)
            loc_seqs.append(ls)
            mod_seqs.append(ms)
        off_flags = [False] * n_trials
        xA, xV, mask = generate_av_batch_tensor(
            loc_seqs, mod_seqs, off_flags,
            n=net.n, space_size=net.space_size, sigma_in=net.sigma_in,
            noise_std=0.0, device=net.device, max_len=T,
            stimulus_intensity=1.0,
        )

        net.reset_state(n_trials)
        net._ei_record = defaultdict(list)
        rast = torch.zeros((T, n_trials), device=net.device)
        v_trace = torch.zeros((T, n_trials), device=net.device)
        with torch.inference_mode():
            for t in range(T):
                net.update_all_layers_batch(xA[:, t], xV[:, t], mask[:, t])
                rast[t] = net._latest_sMSI.sum(dim=1)
                v_trace[t] = net.v_msi.mean(dim=1)
        rec = {k: np.asarray(v) for k, v in net._ei_record.items()}
        net._ei_record = None
        return {
            "biases": dict(b_msi=b_msi, b_msi_inh=b_msi_inh, b_out=b_out,
                           b_uniA=b_uniA, b_uniV=b_uniV),
            "ei_rec": rec,
            "rast_per_frame": rast.cpu().numpy(),
            "v_per_frame": v_trace.cpu().numpy(),
            "dt": dt, "nsub": nsub,
        }
    finally:
        np.random.default_rng = orig
        del net
        torch.cuda.empty_cache()


def main():
    print(f"Decomposition: Form 2 ON, gNMDA=1.30, offset=0, T=60, D=5")
    print()

    out01 = run_with_recording(0.1, 100)
    out05 = run_with_recording(0.05, 200)

    print("--- Biases (M00 checkpoint snapshot) ---")
    for k, v in out01["biases"].items():
        print(f"  abs-mean {k} = {v:.6e}")
    print()

    print("--- Per-substep means (averaged over all recorded substeps in 60 frames) ---")
    print(f"{'key':<20s}  {'dt=0.1':>12s}  {'dt=0.05':>12s}  "
          f"{'ratio(05/01)':>12s}  {'frame_total_ratio':>18s}")
    keys = ["I_E_mean", "I_I_mean", "AMPA", "NMDA",
            "FFInh", "RecurInh", "LatInh"]
    for k in keys:
        a = out01["ei_rec"][k].mean()
        b = out05["ei_rec"][k].mean()
        ratio = b / a if abs(a) > 1e-12 else float("nan")
        # frame-total = per-substep * n_substeps (since each substep contributes once)
        ft_a = a * out01["nsub"]
        ft_b = b * out05["nsub"]
        ft_ratio = ft_b / ft_a if abs(ft_a) > 1e-12 else float("nan")
        print(f"{k:<20s}  {a:>12.4f}  {b:>12.4f}  {ratio:>12.4f}  {ft_ratio:>18.4f}")

    print()
    print("--- Charges (Q = I*dt summed over trial) ---")
    Q_E_01 = out01["ei_rec"]["Q_E"].sum()
    Q_E_05 = out05["ei_rec"]["Q_E"].sum()
    Q_I_01 = out01["ei_rec"]["Q_I"].sum()
    Q_I_05 = out05["ei_rec"]["Q_I"].sum()
    print(f"  Q_E: dt=0.1 = {Q_E_01:.4f}, dt=0.05 = {Q_E_05:.4f}, "
          f"ratio = {Q_E_05/Q_E_01:.4f}")
    print(f"  Q_I: dt=0.1 = {Q_I_01:.4f}, dt=0.05 = {Q_I_05:.4f}, "
          f"ratio = {Q_I_05/Q_I_01:.4f}")
    print(f"  E/I: dt=0.1 = {Q_E_01/Q_I_01:.4f}, dt=0.05 = {Q_E_05/Q_I_05:.4f}")

    print()
    print("--- Spike & voltage stats ---")
    s01 = out01["rast_per_frame"].sum()
    s05 = out05["rast_per_frame"].sum()
    v01 = out01["v_per_frame"].mean()
    v05 = out05["v_per_frame"].mean()
    print(f"  Total MSI spikes (all 8 trials, 60 frames): dt=0.1 = {s01:.0f}, "
          f"dt=0.05 = {s05:.0f}")
    print(f"  Mean v_msi (over all frames): dt=0.1 = {v01:.3f}, dt=0.05 = {v05:.3f}")


if __name__ == "__main__":
    main()
