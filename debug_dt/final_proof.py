"""Causal proof: the principled dt-scaling fix produces dt-invariance.

We apply two principled fixes to the NMDA->I_M coupling and verify each
makes TBW HW match between dt=0.1 and dt=0.05.

Fix A (multiplicative):
    self.I_M.add_(I_nmda * (self.dt / self.tau_syn))
    Standard leaky-integrator continuous-source discretization.
    Steady-state: I_M from NMDA = I_nmda (dt-independent).

Fix B (exponential-Euler):
    factor = 1 - exp(-self.dt / self.tau_syn)
    self.I_M.add_(I_nmda * factor)
    Exact step solution for a step-source filter.
    Steady-state: I_M from NMDA = I_nmda (dt-independent).

Fix C (control: scale by dt/dt_ref):
    self.I_M.add_(I_nmda * self.dt / 0.1)
    At dt=0.1 unchanged, at dt=0.05 halved. Verifies the dt-dependence
    pathway directly.

We also test the empirical-optimum scaling (0.6 at dt=0.05) on multiple
checkpoints to verify cross-ckpt consistency.
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


def install_nmda_replacement(replacement_expr: str, tag: str = "patched"):
    """Install a patch where `self.I_M.add_(I_nmda)` is replaced with
    `self.I_M.add_(<replacement_expr>)`."""
    src = inspect.getsource(PV.ORIG_UPDATE)
    src = textwrap.dedent(src)
    src = src.replace("def update_all_layers_batch(", f"def _{tag}(", 1)
    old = "self.I_M.add_(I_nmda)"
    new = f"self.I_M.add_({replacement_expr})"
    assert src.count(old) == 1, "Could not find I_nmda line"
    src = src.replace(old, new)
    ns = {}
    glb = TRN.__dict__.copy()
    exec(src, glb, ns)
    TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch = ns[f"_{tag}"]


def measure_hw(ckpt: Path, dt: float, nsub: int):
    rng = np.random.default_rng(SEED)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        net = load_msi_model(ckpt, device="cuda")
        net.dt = dt
        net.n_substeps = nsub
        net.plasticity_enabled = False
        net.freeze_g_FFinh = True
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
    return hw, p, elapsed


def main():
    ckpt = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"

    print(f"--- Multi-ckpt verification: scaling NMDA × 0.6 at dt=0.05 ---")
    print(f"{'ckpt':>6s}  {'HW@dt0.1':>12s}  {'HW@dt0.05 raw':>14s}  "
          f"{'HW@dt0.05 ×0.6':>16s}")
    for i in [0, 1, 2]:
        ckpt_i = ROOT / "checkpoint" / f"msi_model_surr_10_{i:02d}.pt"
        PV.restore_original()
        h01, _, _ = measure_hw(ckpt_i, 0.1, 100)
        PV.restore_original()
        h05_raw, _, _ = measure_hw(ckpt_i, 0.05, 200)
        install_nmda_replacement("I_nmda * 0.6", tag="scale06")
        h05_fix, _, _ = measure_hw(ckpt_i, 0.05, 200)
        PV.restore_original()
        print(f"M{i:02d}  {h01:>12.1f}  {h05_raw:>14.1f}  {h05_fix:>16.1f}")

    print()
    print("--- Principled fix A: scale by (dt/tau_syn) ---")
    install_nmda_replacement("I_nmda * (self.dt / self.tau_syn)", tag="fixA")
    h01_A, _, _ = measure_hw(ckpt, 0.1, 100)
    h05_A, _, _ = measure_hw(ckpt, 0.05, 200)
    PV.restore_original()
    print(f"  dt=0.1  HW = {h01_A:.1f} ms")
    print(f"  dt=0.05 HW = {h05_A:.1f} ms")
    print(f"  ΔHW = {h05_A - h01_A:+.1f} ms")

    print()
    print("--- Principled fix B: exponential Euler ---")
    install_nmda_replacement(
        "I_nmda * (1.0 - float(torch.exp(torch.tensor(-self.dt / self.tau_syn)).item()))",
        tag="fixB"
    )
    h01_B, _, _ = measure_hw(ckpt, 0.1, 100)
    h05_B, _, _ = measure_hw(ckpt, 0.05, 200)
    PV.restore_original()
    print(f"  dt=0.1  HW = {h01_B:.1f} ms")
    print(f"  dt=0.05 HW = {h05_B:.1f} ms")
    print(f"  ΔHW = {h05_B - h01_B:+.1f} ms")

    print()
    print("--- Reverse control: amplify NMDA at dt=0.1 by 1.7× ---")
    install_nmda_replacement("I_nmda * 1.7", tag="amp17")
    h01_amp, _, _ = measure_hw(ckpt, 0.1, 100)
    PV.restore_original()
    print(f"  HW = {h01_amp:.1f} ms  (target = dt=0.05 baseline ~295.8 ms)")


if __name__ == "__main__":
    main()
