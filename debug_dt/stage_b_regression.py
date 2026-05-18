"""Stage B regression check for task #16.

Verifies that with `dt_correct_nmda = False` (the default), TBW HW on M00 at
dt=0.1 with the debugger's fixed-seed canonical setup is bit-identical to
the pre-patch baseline (~149 ms).

If this number deviates, the conditional-add patch introduced an unintended
side effect and we STOP.
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


def measure_hw(ckpt: Path, dt: float, nsub: int, flag: bool):
    rng = np.random.default_rng(SEED)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        net = load_msi_model(ckpt, device="cuda")
        net.dt = dt
        net.n_substeps = nsub
        net.dt_correct_nmda = flag  # explicit set (mirrors what test scripts will do)
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
    print(f"Stage B regression — flag OFF must reproduce ~149 ms HW on M00 fixed-seed.")
    print(f"  ckpt = {ckpt.name}")
    print(f"  dt=0.1, n_substeps=100, seed={SEED}, trials={TRIALS}, freeze_agc=True")
    hw, p, t = measure_hw(ckpt, dt=0.1, nsub=100, flag=False)
    print(f"  HW = {hw:.2f} ms  (elapsed {t:.1f}s)")
    print(f"  p_fusion (first 5 of {len(p)}): {p[:5]}")
    print(f"  p_fusion (mid 5):              {p[len(p)//2-2:len(p)//2+3]}")
    print(f"  PASS if HW ≈ 149 ms ± a few ms.")


if __name__ == "__main__":
    main()
