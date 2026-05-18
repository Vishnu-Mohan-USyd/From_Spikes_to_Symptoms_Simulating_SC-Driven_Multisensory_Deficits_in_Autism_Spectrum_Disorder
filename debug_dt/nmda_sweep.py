"""Sweep NMDA per-substep scaling factor at both dt to find the dt-equivalence point.

Each row is (scale_factor, dt, HW).  We want to find scale where dt=0.05 HW
matches dt=0.1 baseline HW.
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
import debug_dt.patch_variants as PV
import Training as TRN

CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
OFFSETS = list(range(-50, 51, 2))
OFFSETS_MS = [o * 10 for o in OFFSETS]
TRIALS = 50
SEED = 12345


def install_nmda_scale_patch(scale_factor: float):
    """Install patch that scales I_M.add_(I_nmda) by a fixed scalar."""
    import inspect, textwrap
    src = inspect.getsource(PV.ORIG_UPDATE)
    src = textwrap.dedent(src)
    src = src.replace("def update_all_layers_batch(",
                      "def _scaled_nmda(", 1)
    old = "self.I_M.add_(I_nmda)"
    new = f"self.I_M.add_(I_nmda * {scale_factor!r})"
    assert src.count(old) == 1
    src = src.replace(old, new)
    ns = {}
    glb = TRN.__dict__.copy()
    exec(src, glb, ns)
    TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch = ns["_scaled_nmda"]


def measure_hw(dt, nsub):
    rng = np.random.default_rng(SEED)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        net = load_msi_model(CKPT, device="cuda")
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
    except Exception as e:
        hw = float("nan")
    return hw, p, elapsed


def main():
    print("dt-scaling sweep: vary NMDA per-substep scaling factor")
    print(f"{'NMDA scale':>12s}  {'dt':>5s}  {'nsub':>5s}  {'HW (ms)':>10s}  {'t(s)':>6s}")

    # Baseline at dt=0.1 (reference)
    PV.restore_original()
    hw_ref, _, t = measure_hw(0.1, 100)
    print(f"{'(none)':>12s}  {0.1:>5.2f}  {100:>5d}  {hw_ref:>10.1f}  {t:>6.1f}")

    # Sweep at dt=0.05
    print()
    print(f"--- dt=0.05 scanning ---")
    results = []
    for scale in [1.0, 0.85, 0.75, 0.7, 0.6, 0.5]:
        PV.restore_original()
        install_nmda_scale_patch(scale)
        hw, _, t = measure_hw(0.05, 200)
        results.append((scale, hw))
        marker = " <-- ref" if abs(hw - hw_ref) < 5 else ""
        print(f"{scale:>12.2f}  {0.05:>5.2f}  {200:>5d}  {hw:>10.1f}  {t:>6.1f}{marker}")
    PV.restore_original()

    # Reverse test: amplify NMDA at dt=0.1 to see if it widens to dt=0.05 baseline
    print()
    print(f"--- dt=0.1 scanning (amplify NMDA) ---")
    for scale in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        PV.restore_original()
        install_nmda_scale_patch(scale)
        hw, _, t = measure_hw(0.1, 100)
        print(f"{scale:>12.2f}  {0.1:>5.2f}  {100:>5d}  {hw:>10.1f}  {t:>6.1f}")
    PV.restore_original()


if __name__ == "__main__":
    main()
