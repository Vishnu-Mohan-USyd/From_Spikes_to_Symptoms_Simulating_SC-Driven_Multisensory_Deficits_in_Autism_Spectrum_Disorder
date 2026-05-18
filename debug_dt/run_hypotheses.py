"""Run the candidate dt-hypothesis tests on M00 and report a single HW pair
per hypothesis.

Designed to be fast: M00 only, 50 trials, full offset density.  ~30 s per
condition × ~10 conditions ≈ 5-7 minutes.

Each hypothesis is tested by:
  1. Restoring the original update method.
  2. Optionally installing a patch (variant from patch_variants).
  3. Measuring TBW HW at dt=0.1 and dt=0.05 with identical seeds.
  4. Reporting ΔHW and percentage drift.
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


CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
OFFSETS = list(range(-50, 51, 2))
OFFSETS_MS = [o * 10 for o in OFFSETS]
TRIALS = 50
SEED = 12345


def measure_hw(dt, nsub, *, gnmda=None, preserve_delays=False,
               extra_setup=None):
    """Returns (hw_ms, p_fusion array)."""
    rng = np.random.default_rng(SEED)
    orig_rng = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        net = load_msi_model(CKPT, device="cuda")
        net.dt = dt
        net.n_substeps = nsub
        if preserve_delays:
            from debug_dt.repro_dt_tbw import preserve_delays_ms
            preserve_delays_ms(net, ref_dt=0.1)
        net.plasticity_enabled = False
        net.freeze_g_FFinh = True
        if gnmda is not None:
            net.gNMDA = gnmda
        if callable(extra_setup):
            extra_setup(net)
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
        np.random.default_rng = orig_rng
    try:
        fit = fit_psychometric_curve_improved(np.asarray(OFFSETS_MS), p)
        hw = float(fit.get("tbw", float("nan")))
    except Exception as e:
        hw = float("nan")
        print(f"        fit error: {e}")
    return hw, p, elapsed


def run_condition(name, *, patch_fn=None, **kwargs):
    PV.restore_original()
    if callable(patch_fn):
        patch_fn()
    hw01, p01, t01 = measure_hw(0.1, 100, **kwargs)
    hw05, p05, t05 = measure_hw(0.05, 200, **kwargs)
    delta = hw05 - hw01
    pct = 100 * delta / hw01 if hw01 > 0 else float("nan")
    print(f"[{name:32s}] HW@dt0.1={hw01:6.1f}ms  HW@dt0.05={hw05:6.1f}ms  "
          f"Δ={delta:+6.1f} ({pct:+5.1f}%)  (t={t01+t05:.1f}s)")
    return dict(name=name, hw01=hw01, hw05=hw05, delta=delta, pct=pct,
                p01=p01, p05=p05)


def main():
    out_npz = ROOT / "debug_dt" / "hypothesis_results.npz"
    results = []

    # Baseline (no patches)
    results.append(run_condition("baseline"))

    # H1: preserve delays in ms
    results.append(run_condition("H1: preserve_delays_ms", preserve_delays=True))

    # H2 family: scale per-substep additions by dt/0.1
    results.append(run_condition("H2a: scale_nmda_to_IM",
                                  patch_fn=PV.apply_scale_nmda_to_IM))
    results.append(run_condition("H2b: scale_ampa_to_IM",
                                  patch_fn=PV.apply_scale_ampa_to_IM))
    results.append(run_condition("H2c: scale_inh_only",
                                  patch_fn=PV.apply_scale_inh_only))
    results.append(run_condition("H2d: scale_all_IM",
                                  patch_fn=PV.apply_scale_all_IM))
    results.append(run_condition("H2e: scale_all_leaky",
                                  patch_fn=PV.apply_scale_all_leaky))

    # H3: exponential decay (numerical Euler error in I_M decay)
    results.append(run_condition("H3: exp_decay", patch_fn=PV.apply_exp_decay))

    # H4: Izhikevich half-step
    results.append(run_condition("H4: izhi_half", patch_fn=PV.apply_izhi_half))

    PV.restore_original()

    print()
    print("=" * 70)
    print(f"{'Hypothesis':32s}  HW01      HW05      ΔHW       %drift")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:32s}  {r['hw01']:8.1f}  {r['hw05']:8.1f}  "
              f"{r['delta']:+8.1f}  {r['pct']:+6.1f}%")

    # Save
    payload = {}
    for r in results:
        nm = r["name"].split(":")[0].strip()
        payload[f"{nm}_hw01"] = np.asarray(r["hw01"])
        payload[f"{nm}_hw05"] = np.asarray(r["hw05"])
        payload[f"{nm}_p01"] = r["p01"]
        payload[f"{nm}_p05"] = r["p05"]
    np.savez(out_npz, **payload)
    print(f"\nSaved to {out_npz}")


if __name__ == "__main__":
    main()
