"""Task #143-B — V2 dt-sensitivity sweep (Supp Fig 1).

TBW HW comparison at dt ∈ {0.1, 0.05} across all 10 legacy ckpts under
POST-fix Training.py (unified NMDA * dt_linear_scale at L2185/L2284 +
AGC reset fix).

Paper Supp Fig 1 numbers:
  dt=0.1   →  215.8 ms
  dt=0.05  →  256.5 ms
  drift   +40.7 ms

Coder #135 verified +1.0 ms drift on surr_10_00 (single ckpt) post-NMDA
unification. This task confirms across all 10 ckpts.

Protocol matches task135_logs/verify.py:measure_hw:
  - offsets = list(range(-50, 51, 2)) (51 vals × 10 ms = -500..+500 ms)
  - n_trials = 50
  - T=60, D=5, stim_in=1.0, sigma=2.0
  - At dt=0.1 → n_substeps=100  (total = 10ms window per step)
  - At dt=0.05 → n_substeps=200 (total = 10ms window per step)
  - rng seeded per call (12345) → identical stimulus sequences across dts
"""
from __future__ import annotations
import sys, time, json
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

CKPTS = [ROOT / "checkpoint" / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
OUT_JSON = ROOT / "task143_logs" / "task_B_dt_sweep.json"


def measure_hw(ckpt, dt, nsub, *, trials=50, seed=12345):
    """TBW HW measurement matching task #135 verify protocol."""
    offsets = list(range(-50, 51, 2))
    offsets_ms = [o * 10 for o in offsets]
    net = load_msi_model(str(ckpt), device="cuda")
    net.dt = dt
    net.n_substeps = nsub
    net.plasticity_enabled = False
    rng = np.random.default_rng(seed)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        p_fusion, _ = compute_tbw_temporal_fusion_persep(
            net, offsets, n_trials=trials, T=60, D=5, stim_in=1.0,
            sigma=2.0, valley_threshold=0.4, min_peak_height=0.2,
            min_peak_separation=3, min_total=10.0,
        )
    finally:
        np.random.default_rng = orig
    fit = fit_psychometric_curve_improved(np.asarray(offsets_ms),
                                          np.asarray(p_fusion))
    hw = float(fit.get("tbw", float("nan")))
    del net
    torch.cuda.empty_cache()
    return hw


def main():
    print("=" * 80)
    print("TASK #143-B  V2 dt-sensitivity sweep — TBW HW @ dt ∈ {0.1, 0.05}")
    print("=" * 80)
    print(f"  Paper Supp Fig 1: dt=0.1 → 215.8 ms, dt=0.05 → 256.5 ms (drift +40.7 ms)")
    print(f"  Post-fix Training.py target: drift ≈ +1 ms (NMDA unification + AGC fix)")
    print()

    results = {"hw_dt010": [], "hw_dt005": [], "drift": [], "ckpt": []}

    t_total = time.time()
    for i, ckpt in enumerate(CKPTS):
        print(f"--- ckpt {i:02d}: {ckpt.name} ---", flush=True)
        t0 = time.time()
        hw01 = measure_hw(ckpt, 0.1,  100)
        t1 = time.time()
        hw05 = measure_hw(ckpt, 0.05, 200)
        t2 = time.time()
        drift = hw05 - hw01
        print(f"  dt=0.1   HW={hw01:7.2f} ms   ({t1 - t0:5.1f}s)")
        print(f"  dt=0.05  HW={hw05:7.2f} ms   ({t2 - t1:5.1f}s)")
        print(f"  drift =  {drift:+7.2f} ms")
        results["ckpt"].append(ckpt.name)
        results["hw_dt010"].append(hw01)
        results["hw_dt005"].append(hw05)
        results["drift"].append(drift)
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  (interim json saved to {OUT_JSON.name})", flush=True)

    arr_01 = np.array(results["hw_dt010"], dtype=float)
    arr_05 = np.array(results["hw_dt005"], dtype=float)
    arr_dr = np.array(results["drift"], dtype=float)

    def sem(x):
        x = x[~np.isnan(x)]
        return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")

    print()
    print("=" * 80)
    print("SUMMARY (n=10 ckpts)")
    print("=" * 80)
    print(f"{'ckpt':>22}  {'HW @ 0.1':>10}  {'HW @ 0.05':>11}  {'drift':>9}")
    print("  " + "-" * 56)
    for c, a, b, d in zip(results["ckpt"], arr_01, arr_05, arr_dr):
        print(f"{c:>22}  {a:10.2f}  {b:11.2f}  {d:+9.2f}")
    print("  " + "-" * 56)
    print(f"{'mean ± SEM':>22}  {np.nanmean(arr_01):7.2f} ± {sem(arr_01):4.2f}   "
          f"{np.nanmean(arr_05):7.2f} ± {sem(arr_05):4.2f}   "
          f"{np.nanmean(arr_dr):+5.2f} ± {sem(arr_dr):4.2f}")
    print()
    print(f"Paper Supp Fig 1   : 215.80      256.50      +40.70")
    print()
    print(f"Total time: {(time.time() - t_total) / 60:.1f} min")
    print(f"Wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
