"""Phase 1 — deterministic SBW reproducer for dt-drift.

Replicates the SBW-control HW measurement at dt=0.1 (n_substeps=100) and
dt=0.05 (n_substeps=200) on a single fixed-seed M00 checkpoint with the
Form-2 NMDA fix + delays-in-ms + gNMDA=1.30 already in place.

Goal: confirm the SBW HW drop reported by task #29 (control ~24° → ~20°)
exists on M00 alone with a deterministic seed, so we have a fast (single
ckpt) reproducer to drive hypothesis testing.

If the drop is robust on one checkpoint, we can iterate quickly.

Usage:
    python -m debug_dt.repro_sbw_dt
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from TBW_test import load_msi_model, _find_crossings
from SBW_test import (
    compute_sbw_enhancement_persep,
    fit_pedestal_curve,
)

CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
SBW_SEPARATIONS = tuple(range(-80, 85, 5))   # 33 values, canonical
ENH_THRESHOLD = 10.0
N_TRIALS = 50
INTENSITY = 1.0
DURATION = 20
SEED = 12345


def make_one_sided_pooled(p_fusion_curve, separations):
    """Mirror-symmetrize like run_spatial_binding_across_models (but no
    floor subtraction, no model-pool averaging).  Single-ckpt, no SEM.
    """
    seps = np.asarray(separations, float)
    mags = np.unique(np.abs(seps))
    one_sided = np.zeros(len(mags))
    for j, mag in enumerate(mags):
        cols = np.where(np.abs(seps) == mag)[0]
        one_sided[j] = p_fusion_curve[cols].mean()
    full_sep = np.concatenate([-mags[::-1], mags[1:]])
    full_mean = np.concatenate([one_sided[::-1], one_sided[1:]])
    return {"separations_deg": full_sep, "mean_prob": full_mean,
            "sem_prob": np.zeros_like(full_mean)}


def fit_hw(pooled):
    xs, ys, popt = fit_pedestal_curve(pooled)
    cross = _find_crossings(xs, ys, 0.5)
    if len(cross) >= 2:
        return (cross[-1] - cross[0]) / 2.0
    return float("nan")


def measure_sbw_hw(*, dt: float, n_sub: int, ckpt: Path = CKPT,
                   seed: int = SEED, gNMDA: float = 1.30,
                   subtract_floor: bool = True):
    """Measure SBW HW on a single checkpoint with deterministic RNG."""
    # Pin numpy default_rng with a fixed seed so that compute_sbw_..._persep's
    # internal np.random.default_rng() returns deterministic positions.
    rng_fixed = np.random.default_rng(seed)
    orig_default_rng = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng_fixed
    try:
        net = load_msi_model(ckpt, device="cuda")
        net.dt = float(dt)
        net.n_substeps = int(n_sub)
        net.dt_correct_nmda = True
        net.gNMDA = float(gNMDA)
        net.plasticity_enabled = False
        net.freeze_g_FFinh = True
        net._reset_delay_buffers()
        # Sanity print
        print(f"      net.dt={net.dt}, n_substeps={net.n_substeps}, "
              f"gNMDA={net.gNMDA}, dt_correct_nmda={net.dt_correct_nmda}, "
              f"delay_a2msi={net.conduction_delay_a2msi}, "
              f"a2msi_ms={getattr(net,'conduction_delay_a2msi_ms','--')}")

        t0 = time.time()
        mean_enh, all_trial_enh = compute_sbw_enhancement_persep(
            net,
            separations_deg=SBW_SEPARATIONS,
            n_trials=N_TRIALS,
            intensity=INTENSITY,
            duration=DURATION,
            block_size=32,
        )
        elapsed = time.time() - t0

        p_fusion = np.array([
            (trial_enh > ENH_THRESHOLD).mean() for trial_enh in all_trial_enh
        ])
        # Diagnostic: print raw enhancement stats at sep=0 and sep=80
        print(f"      mean_enh: at sep=0 = {mean_enh[len(SBW_SEPARATIONS)//2]:.2f}, "
              f"at sep=±80 = {mean_enh[0]:.2f}/{mean_enh[-1]:.2f}, "
              f"max = {mean_enh.max():.2f}")

        del net
        torch.cuda.empty_cache()
    finally:
        np.random.default_rng = orig_default_rng

    pooled = make_one_sided_pooled(p_fusion, SBW_SEPARATIONS)

    if subtract_floor:
        sep = pooled["separations_deg"]
        far_mask = np.abs(sep) >= 50
        floor = pooled["mean_prob"][far_mask].mean()
        pooled["mean_prob"] = np.clip(pooled["mean_prob"] - floor, 0.0, None)
    else:
        floor = 0.0

    hw = fit_hw(pooled)
    return {
        "hw": hw,
        "floor": float(floor),
        "elapsed_s": elapsed,
        "p_fusion_raw": p_fusion,
        "pooled": pooled,
    }


def main():
    print(f"SBW dt-drift reproducer — M00, seed={SEED}, gNMDA=1.30, "
          f"Form-2 ON, delays-in-ms ON, floor-subtract ON")
    print(f"  separations: {SBW_SEPARATIONS[0]}..{SBW_SEPARATIONS[-1]} step "
          f"{SBW_SEPARATIONS[1]-SBW_SEPARATIONS[0]}, n_trials={N_TRIALS}, "
          f"intensity={INTENSITY}, duration={DURATION} frames, "
          f"enh_thresh={ENH_THRESHOLD}")
    print()

    for dt, n_sub in [(0.1, 100), (0.05, 200)]:
        r = measure_sbw_hw(dt=dt, n_sub=n_sub)
        print(f"  dt={dt:>5.2f}, n_sub={n_sub:>3d}: HW = {r['hw']:>6.2f}°  "
              f"floor = {r['floor']:.4f}  elapsed = {r['elapsed_s']:>5.1f}s")
        print(f"    pooled mean_prob (one-sided after symmetrize+floor): "
              f"min={r['pooled']['mean_prob'].min():.3f}, "
              f"max={r['pooled']['mean_prob'].max():.3f}, "
              f"at 0deg = {r['pooled']['mean_prob'][len(r['pooled']['mean_prob'])//2]:.3f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
