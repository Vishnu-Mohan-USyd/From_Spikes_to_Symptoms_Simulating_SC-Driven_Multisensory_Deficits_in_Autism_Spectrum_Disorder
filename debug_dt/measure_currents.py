"""Directly measure per-substep currents at both dt values, no fitting.

Goal: confirm or refute the claim that `self.I_M.add_(I_nmda)` produces
I_M steady-state that scales as 1/dt for continuous NMDA injection.

We capture the substep-by-substep I_M trace via the existing `_ei_record`
hook, which stores per-substep means of I_E (AMPA+NMDA), I_I (sum of
inhibitions), Q_E, Q_I, AMPA, NMDA, FFInh, RecurInh, LatInh.

Then we report:
  - mean I_E, mean I_I per ms (i.e. per_substep / dt)
  - mean V_msi per substep
  - integrated drive Q_E - Q_I per frame in nA*ms
  - mean spike count per substep
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


def run_one_offset(net, offset, *, T=60, D=5, n_trials=8, stim_in=1.0,
                   seed=12345):
    rng = np.random.default_rng(seed)

    loc_seqs, mod_seqs = [], []
    for _ in range(n_trials):
        loc = int(rng.integers(0, net.space_size))
        ls, ms = generate_two_event_offset_seq(
            loc=loc, T=T, D=D, offset=offset, space_size=net.space_size,
        )
        loc_seqs.append(ls)
        mod_seqs.append(ms)
    off_flags = [False] * n_trials

    xA, xV, mask = generate_av_batch_tensor(
        loc_seqs, mod_seqs, off_flags,
        n=net.n, space_size=net.space_size, sigma_in=net.sigma_in,
        noise_std=0.0, device=net.device, max_len=T,
        stimulus_intensity=stim_in,
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
        "rast_per_frame": rast.cpu().numpy(),
        "v_per_frame": v_trace.cpu().numpy(),
        "ei": rec,
    }


def main():
    print(f"Loading {CKPT}")

    summary = {}

    for tag, dt, nsub in [("dt01", 0.1, 100), ("dt005", 0.05, 200)]:
        net = load_msi_model(CKPT, device="cuda")
        net.dt = dt
        net.n_substeps = nsub
        net.plasticity_enabled = False
        net.freeze_g_FFinh = True

        out = run_one_offset(net, offset=0, T=60, D=5, n_trials=8,
                              stim_in=1.0, seed=12345)
        rec = out["ei"]
        # Each list element = mean across batch+neurons for one SUBSTEP
        # Total substeps recorded ≈ T * nsub
        n_recorded = len(rec["I_E_mean"])
        T_frames = 60
        per_frame = n_recorded // T_frames
        if per_frame != nsub:
            print(f"  warning: recorded {per_frame}/frame, expected {nsub}")
        I_E = rec["I_E_mean"].mean()
        I_I = rec["I_I_mean"].mean()
        Q_E = rec["Q_E"].sum()         # already integrated over dt
        Q_I = rec["Q_I"].sum()
        AMPA = rec["AMPA"].mean()
        NMDA = rec["NMDA"].mean()
        FFInh = rec["FFInh"].mean()
        RecurInh = rec["RecurInh"].mean()
        LatInh = rec["LatInh"].mean()
        rast = out["rast_per_frame"]
        spikes_per_frame = rast.mean(axis=1)   # mean across trials
        total_spikes = rast.sum(axis=1).sum()  # total across all trials and frames
        v_mean = out["v_per_frame"].mean()

        summary[tag] = dict(
            dt=dt, nsub=nsub,
            I_E_mean=I_E, I_I_mean=I_I,
            Q_E_total=Q_E, Q_I_total=Q_I,
            AMPA_mean=AMPA, NMDA_mean=NMDA,
            FFInh_mean=FFInh, RecurInh_mean=RecurInh, LatInh_mean=LatInh,
            total_MSI_spikes=float(total_spikes),
            v_msi_mean=v_mean,
        )

        print(f"\n[{tag}] dt={dt} n_substeps={nsub}")
        print(f"  I_E mean per substep  : {I_E:.6f}")
        print(f"  I_I mean per substep  : {I_I:.6f}")
        print(f"  Q_E total over trial  : {Q_E:.4f}  (charge = I × dt summed)")
        print(f"  Q_I total over trial  : {Q_I:.4f}")
        print(f"  AMPA mean per substep : {AMPA:.6f}")
        print(f"  NMDA mean per substep : {NMDA:.6f}")
        print(f"  FFInh mean per substep: {FFInh:.6f}")
        print(f"  Recur Inh per substep : {RecurInh:.6f}")
        print(f"  Lat Inh per substep   : {LatInh:.6f}")
        print(f"  MSI spikes (all)      : {total_spikes:.1f}")
        print(f"  v_msi mean            : {v_mean:.3f}")

        del net
        torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print("Ratios (dt=0.05 / dt=0.1)")
    for key in ["I_E_mean", "I_I_mean", "AMPA_mean", "NMDA_mean",
                "FFInh_mean", "RecurInh_mean", "LatInh_mean",
                "total_MSI_spikes", "Q_E_total", "Q_I_total"]:
        a = summary["dt01"][key]
        b = summary["dt005"][key]
        ratio = b / a if abs(a) > 1e-9 else float("inf")
        print(f"  {key:25s} dt01={a:12.4f}  dt05={b:12.4f}  ratio={ratio:6.3f}")


if __name__ == "__main__":
    main()
