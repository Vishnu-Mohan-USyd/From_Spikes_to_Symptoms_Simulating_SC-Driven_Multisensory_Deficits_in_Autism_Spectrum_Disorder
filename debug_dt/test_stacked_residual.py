"""Stacked tests on top of (dt=0.05, Form 2, gNMDA=1.30, delays×2) baseline = 128.

Probe each remaining mechanism to see what CLOSES the residual 21 ms gap toward 149.
"""
from __future__ import annotations
import inspect
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from TBW_test import (
    load_msi_model,
    compute_tbw_temporal_fusion_persep,
    fit_psychometric_curve_improved,
)
import debug_dt.patch_variants as PV
import Training as TRN

OFFSETS = list(range(-50, 51, 2))
OFFSETS_MS = [o * 10 for o in OFFSETS]
TRIALS = 50
SEED = 12345
CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
GNMDA = 1.30


def measure_hw(dt: float, nsub: int, delay_scale: float = 1.0):
    rng = np.random.default_rng(SEED)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        net = load_msi_model(CKPT, device="cuda")
        net.dt = dt
        net.n_substeps = nsub
        net.dt_correct_nmda = True
        net.gNMDA = GNMDA
        net.plasticity_enabled = False
        net.freeze_g_FFinh = True
        if delay_scale != 1.0:
            for attr in ["conduction_delay_a2msi", "conduction_delay_v2msi",
                         "conduction_delay_inA_inh", "conduction_delay_inV_inh",
                         "conduction_delay_a2msi_inh", "conduction_delay_v2msi_inh",
                         "conduction_delay_msi_inh2exc", "conduction_delay_msi2out"]:
                setattr(net, attr, int(round(getattr(net, attr) * delay_scale)))
            import torch as _t
            B = TRIALS
            for buf_name, delay_attr, shape in [
                ("buffer_a2msi", "conduction_delay_a2msi", (B, net.n)),
                ("buffer_v2msi", "conduction_delay_v2msi", (B, net.n)),
                ("buffer_inA_inh", "conduction_delay_inA_inh", (B, net.n)),
                ("buffer_inV_inh", "conduction_delay_inV_inh", (B, net.n)),
                ("buffer_a2msi_inh", "conduction_delay_a2msi_inh", (B, net.n)),
                ("buffer_v2msi_inh", "conduction_delay_v2msi_inh", (B, net.n)),
                ("buffer_msi_inh2exc", "conduction_delay_msi_inh2exc", (B, net.n_inh)),
                ("buffer_msi2out", "conduction_delay_msi2out", (B, net.n)),
            ]:
                d = getattr(net, delay_attr)
                if d > 0:
                    setattr(net, buf_name, _t.zeros((d,) + shape, device=net.device))
                    net._delay_positions[buf_name] = 0
        t0 = time.time()
        p, _ = compute_tbw_temporal_fusion_persep(
            net, OFFSETS, n_trials=TRIALS, T=60, D=5, stim_in=1.0,
            sigma=2.0, valley_threshold=0.4, min_peak_height=0.2,
            min_peak_separation=3, min_total=10.0,
        )
        elapsed = time.time() - t0
        del net
        torch.cuda.empty_cache()
    finally:
        np.random.default_rng = orig
    try:
        fit = fit_psychometric_curve_improved(np.asarray(OFFSETS_MS), p)
        hw = float(fit.get("tbw", float("nan")))
    except Exception:
        hw = float("nan")
    return hw, elapsed


def banner(s):
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def main():
    banner("Reference: dt=0.1 raw and dt=0.05 raw (no delay fix)")
    PV.restore_original()
    h01, t = measure_hw(0.1, 100); print(f"  dt=0.1 raw      : {h01:.2f} ms")
    h05r, t = measure_hw(0.05, 200); print(f"  dt=0.05 raw     : {h05r:.2f} ms")
    h05d, t = measure_hw(0.05, 200, delay_scale=2.0); print(f"  dt=0.05 delays×2: {h05d:.2f} ms")

    banner("Stack 1: dt=0.05 + delays×2 + izhi-half")
    PV.restore_original(); PV.apply_izhi_half()
    h, t = measure_hw(0.05, 200, delay_scale=2.0); PV.restore_original()
    print(f"  HW = {h:.2f} ms  (Δ vs 128 = {h-h05d:+.2f}, vs 149 = {h-h01:+.2f})")

    banner("Stack 2: dt=0.1 + izhi-half (already tested)")
    PV.restore_original(); PV.apply_izhi_half()
    h, t = measure_hw(0.1, 100); PV.restore_original()
    print(f"  HW = {h:.2f} ms  (Δ vs 149 = {h-h01:+.2f})")

    banner("Stack 3: dt=0.05 + delays×2 + scale-all-IM (legacy NMDA bug back)")
    # Per legacy logic; this tests if turning OFF Form 2 still helps after delay fix.
    PV.restore_original()
    h_off, t = measure_hw_flagged(0.05, 200, delay_scale=2.0, form2=False, gNMDA=GNMDA)
    print(f"  HW(form2 OFF, gNMDA=1.30, dt=0.05, delays×2) = {h_off:.2f}")
    h_off_def, t = measure_hw_flagged(0.05, 200, delay_scale=2.0, form2=False, gNMDA=0.05)
    print(f"  HW(form2 OFF, gNMDA=0.05, dt=0.05, delays×2) = {h_off_def:.2f}")


def measure_hw_flagged(dt: float, nsub: int, delay_scale: float, form2: bool,
                       gNMDA: float):
    rng = np.random.default_rng(SEED)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        net = load_msi_model(CKPT, device="cuda")
        net.dt = dt
        net.n_substeps = nsub
        net.dt_correct_nmda = form2
        net.gNMDA = gNMDA
        net.plasticity_enabled = False
        net.freeze_g_FFinh = True
        if delay_scale != 1.0:
            for attr in ["conduction_delay_a2msi", "conduction_delay_v2msi",
                         "conduction_delay_inA_inh", "conduction_delay_inV_inh",
                         "conduction_delay_a2msi_inh", "conduction_delay_v2msi_inh",
                         "conduction_delay_msi_inh2exc", "conduction_delay_msi2out"]:
                setattr(net, attr, int(round(getattr(net, attr) * delay_scale)))
            import torch as _t
            B = TRIALS
            for buf_name, delay_attr, shape in [
                ("buffer_a2msi", "conduction_delay_a2msi", (B, net.n)),
                ("buffer_v2msi", "conduction_delay_v2msi", (B, net.n)),
                ("buffer_inA_inh", "conduction_delay_inA_inh", (B, net.n)),
                ("buffer_inV_inh", "conduction_delay_inV_inh", (B, net.n)),
                ("buffer_a2msi_inh", "conduction_delay_a2msi_inh", (B, net.n)),
                ("buffer_v2msi_inh", "conduction_delay_v2msi_inh", (B, net.n)),
                ("buffer_msi_inh2exc", "conduction_delay_msi_inh2exc", (B, net.n_inh)),
                ("buffer_msi2out", "conduction_delay_msi2out", (B, net.n)),
            ]:
                d = getattr(net, delay_attr)
                if d > 0:
                    setattr(net, buf_name, _t.zeros((d,) + shape, device=net.device))
                    net._delay_positions[buf_name] = 0
        t0 = time.time()
        p, _ = compute_tbw_temporal_fusion_persep(
            net, OFFSETS, n_trials=TRIALS, T=60, D=5, stim_in=1.0,
            sigma=2.0, valley_threshold=0.4, min_peak_height=0.2,
            min_peak_separation=3, min_total=10.0,
        )
        elapsed = time.time() - t0
        del net
        torch.cuda.empty_cache()
    finally:
        np.random.default_rng = orig
    try:
        fit = fit_psychometric_curve_improved(np.asarray(OFFSETS_MS), p)
        hw = float(fit.get("tbw", float("nan")))
    except Exception:
        hw = float("nan")
    return hw, elapsed


if __name__ == "__main__":
    main()
