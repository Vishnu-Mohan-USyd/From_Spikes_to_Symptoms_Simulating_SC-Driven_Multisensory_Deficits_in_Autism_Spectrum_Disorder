"""Test H_d: conduction delays in substep units (post Form 2 fix).

At dt=0.05 with n_substeps=200, the raw delays (in substeps) translate to
HALF the physical ms of their dt=0.1 values. Rescale them to preserve
physical ms (multiply by 2) and check if HW(dt=0.05) moves toward
HW(dt=0.1)=149 ms.

Delays in M00 (read from Training.py constructor):
  delay_a2msi      = 250 substeps  -> 25 ms (dt=0.1) / 12.5 ms (dt=0.05)
  delay_v2msi      = 400 substeps  -> 40 ms (dt=0.1) / 20 ms (dt=0.05)
  delay_inA_inh    = same as a2msi
  delay_inV_inh    = same as v2msi
  delay_a2msi_inh  = same
  delay_v2msi_inh  = same
  delay_msi_inh2exc = ?
  delay_msi2out    = ?
"""
from __future__ import annotations
import sys
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
        # ===== Rescale all conduction delays =====
        if delay_scale != 1.0:
            print(f"  [delays before scale] a2msi={net.conduction_delay_a2msi}, "
                  f"v2msi={net.conduction_delay_v2msi}, "
                  f"inA_inh={net.conduction_delay_inA_inh}, "
                  f"a2msi_inh={net.conduction_delay_a2msi_inh}, "
                  f"v2msi_inh={net.conduction_delay_v2msi_inh}, "
                  f"msi_inh2exc={net.conduction_delay_msi_inh2exc}, "
                  f"msi2out={net.conduction_delay_msi2out}")
            for attr in ["conduction_delay_a2msi",
                         "conduction_delay_v2msi",
                         "conduction_delay_inA_inh",
                         "conduction_delay_inV_inh",
                         "conduction_delay_a2msi_inh",
                         "conduction_delay_v2msi_inh",
                         "conduction_delay_msi_inh2exc",
                         "conduction_delay_msi2out"]:
                old_d = getattr(net, attr)
                new_d = int(round(old_d * delay_scale))
                setattr(net, attr, new_d)
            # ALSO need to rebuild buffers because they're sized by delays
            # Easier: reset_state with the new delays
            print(f"  [delays after scale]  a2msi={net.conduction_delay_a2msi}, "
                  f"v2msi={net.conduction_delay_v2msi}, "
                  f"inA_inh={net.conduction_delay_inA_inh}, "
                  f"a2msi_inh={net.conduction_delay_a2msi_inh}, "
                  f"v2msi_inh={net.conduction_delay_v2msi_inh}, "
                  f"msi_inh2exc={net.conduction_delay_msi_inh2exc}, "
                  f"msi2out={net.conduction_delay_msi2out}")
            # Rebuild ring buffers
            B = TRIALS
            zero_exc_shape = (1, B, net.n)
            zero_inh_shape = (1, B, net.n_inh)
            import torch as _t
            for buf_name, delay_attr, shape in [
                ("buffer_a2msi", "conduction_delay_a2msi", (B, net.n)),
                ("buffer_v2msi", "conduction_delay_v2msi", (B, net.n)),
                ("buffer_inA_inh", "conduction_delay_inA_inh", (B, net.n)),
                ("buffer_inV_inh", "conduction_delay_inV_inh", (B, net.n)),
                ("buffer_a2msi_inh", "conduction_delay_a2msi_inh", (B, net.n)),
                ("buffer_v2msi_inh", "conduction_delay_v2msi_inh", (B, net.n)),
                ("buffer_msi_inh2exc", "conduction_delay_msi_inh2exc",
                 (B, net.n_inh)),
                ("buffer_msi2out", "conduction_delay_msi2out", (B, net.n)),
            ]:
                d = getattr(net, delay_attr)
                if d > 0:
                    setattr(net, buf_name,
                            _t.zeros((d,) + shape, device=net.device))
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
    banner("H_d Baseline (Form 2 ON, gNMDA=1.30, no delay rescaling)")
    h01, t = measure_hw(0.1, 100); print(f"  dt=0.1 raw : HW={h01:.2f} ms ({t:.1f}s)")
    h05, t = measure_hw(0.05, 200); print(f"  dt=0.05 raw: HW={h05:.2f} ms ({t:.1f}s)")
    print(f"  ΔHW = {h05-h01:+.2f} ms")

    banner("H_d Test 1: dt=0.05 with delays × 2 (preserve physical ms)")
    print("Prediction: if shorter physical delays at dt=0.05 narrow the TBW,")
    print("doubling the delays should widen it toward 149 ms.")
    h05_2x, t = measure_hw(0.05, 200, delay_scale=2.0)
    print(f"  HW(dt=0.05, delays×2): {h05_2x:.2f} ms  ({t:.1f}s)")
    print(f"  Δ vs raw dt=0.05 (62) = {h05_2x - h05:+.2f} ms")
    print(f"  Δ vs raw dt=0.1 (149) = {h05_2x - h01:+.2f} ms")

    banner("H_d Test 2 (control): dt=0.1 with delays × 0.5 (mimic dt=0.05 delays)")
    print("Reverse test: if shorter delays narrow the TBW, halving delays at")
    print("dt=0.1 should narrow it toward 62 ms.")
    h01_half, t = measure_hw(0.1, 100, delay_scale=0.5)
    print(f"  HW(dt=0.1, delays×0.5): {h01_half:.2f} ms  ({t:.1f}s)")
    print(f"  Δ vs raw dt=0.1 (149) = {h01_half - h01:+.2f} ms")
    print(f"  Δ vs raw dt=0.05 (62) = {h01_half - h05:+.2f} ms")


if __name__ == "__main__":
    main()
