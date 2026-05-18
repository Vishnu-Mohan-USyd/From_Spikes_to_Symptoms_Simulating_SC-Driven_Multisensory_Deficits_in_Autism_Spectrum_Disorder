"""Task #28 — Supp Fig verification: post-fix TBW at dt=0.1 vs dt=0.05.

Same methodology as Supp Fig 1, but with all Stage F changes locked in:
  - dt_correct_nmda=True (now default in Training.py)
  - gNMDA = 1.30 (set by generate_all_fresh.py CONDITIONS["control"])
  - delays-in-ms via _reset_delay_buffers (no-op at dt=0.1, active at dt=0.05)
  - perturbations rescaled ×25 — not applied here, control-only

Reuses generate_all_fresh.py's control pipeline (run_fusion_across_models with
fusion_method='temporal_fusion', n_trials=50, full TBW_OFFSETS).
Only dt and n_substeps are varied via modify_net.

Reference values (for context, NOT comparison targets):
  - Pre-fix supp-fig (paper):  dt=0.1 → 215.8 ms,  dt=0.05 → 256.5 ms  (Δ ≈ +40 ms)
  - Pre-fix debugger (M00 fixed-seed):  dt=0.1 → 149,  dt=0.05 → 296   (Δ ≈ +147 ms)
  - Stage E sanity (dt=0.1, post-fix):  ~107 ms

Output:
  - cache files in debug_dt/suppfig_cache/
  - prints HW(dt=0.1), HW(dt=0.05), Δ ms, Δ%
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from TBW_test import run_fusion_across_models, _find_crossings
from replot_all_cosmetic import plot_tbw, _load_pooled

BASE = ROOT / "checkpoint"
MODELS = [BASE / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
CACHE = Path(__file__).parent / "suppfig_cache"
CACHE.mkdir(exist_ok=True)
TMP_PLOT_DIR = CACHE / "_plots"
TMP_PLOT_DIR.mkdir(exist_ok=True)

# === Match generate_all_fresh.py exactly ===
TBW_OFFSETS = list(range(-50, 51, 2))   # 51 values
N_TRIALS = 50
GNMDA_CONTROL = 1.30                    # CONDITIONS["control"] from generate_all_fresh.py


def make_modify_net(dt_val: float, n_sub: int):
    """Return modify_net callable that sets dt, n_substeps, gNMDA, then resets buffers.

    Mirrors generate_all_fresh.py CONDITIONS["control"] (gNMDA=1.30) plus the
    dt-sweep parameters. _reset_delay_buffers() must be called *after* dt change
    because buffer sizes are derived from conduction_delay_*_ms / dt.
    """
    def _mod(n):
        n.dt = float(dt_val)
        n.n_substeps = int(n_sub)
        n.gNMDA = float(GNMDA_CONTROL)
        n._reset_delay_buffers()
    return _mod


def _save_pooled(pooled, path):
    saveable = {k: np.asarray(v) for k, v in pooled.items() if v is not None}
    np.savez(path, **saveable)


def run_one(dt_val: float, n_sub: int, label: str):
    print("=" * 60)
    print(f"TBW CONTROL @ dt={dt_val}, n_substeps={n_sub}  ({label})")
    print(f"  10 models x {N_TRIALS} trials x {len(TBW_OFFSETS)} offsets, gNMDA={GNMDA_CONTROL}")
    print("=" * 60)
    cache_path = CACHE / f"tbw_control_{label}.npz"
    t0 = time.time()
    pooled = run_fusion_across_models(
        MODELS, TBW_OFFSETS, device="cuda",
        fusion_method='temporal_fusion',
        n_trials=N_TRIALS,
        modify_net=make_modify_net(dt_val, n_sub),
    )
    _save_pooled(pooled, cache_path)
    el = time.time() - t0
    print(f"  done in {el:.0f}s -> {cache_path.name}")

    # Match generate_all_fresh.py fitting path: _load_pooled -> plot_tbw -> _find_crossings
    pooled = _load_pooled(cache_path)
    fig, ax, fit_info = plot_tbw(
        pooled, reference_fit=None,
        out_path=str(TMP_PLOT_DIR / f"TBW_control_{label}.svg"))
    plt.close(fig)
    xs_f, ys_f = fit_info["xs"], fit_info["ys"]
    cross = _find_crossings(xs_f, ys_f, 0.5)
    hw = (cross[-1] - cross[0]) / 2 if len(cross) >= 2 else float("nan")
    print(f"  HW @ dt={dt_val}: {hw:.1f} ms")
    torch.cuda.empty_cache()
    return hw, el


if __name__ == "__main__":
    t_start = time.time()
    print(f"start = {time.strftime('%Y-%m-%dT%H:%M:%S')}")

    hw_01, t_01 = run_one(0.1,  100, "dt010")
    hw_005, t_005 = run_one(0.05, 200, "dt005")

    delta = hw_005 - hw_01
    pct = (delta / hw_01 * 100.0) if hw_01 not in (0.0, float("nan")) else float("nan")

    print("\n" + "=" * 60)
    print("TASK #28  SUPP-FIG POST-FIX RESULT")
    print("=" * 60)
    print(f"  HW @ dt=0.10 (n_sub=100):  {hw_01:.1f} ms   [{t_01:.0f}s run]")
    print(f"  HW @ dt=0.05 (n_sub=200):  {hw_005:.1f} ms  [{t_005:.0f}s run]")
    print(f"  Delta = HW(0.05) - HW(0.10) = {delta:+.1f} ms ({pct:+.1f}%)")
    print()
    print("Reference (NOT a comparison target):")
    print("  Pre-fix supp-fig (paper):      dt=0.1 -> 215.8 ms,  dt=0.05 -> 256.5 ms  (Δ ≈ +40 ms)")
    print("  Pre-fix debugger (M00 1-ckpt): dt=0.1 -> 149   ms,  dt=0.05 -> 296   ms  (Δ ≈ +147 ms)")
    print("  Stage E sanity (dt=0.1):       ~107 ms")

    np.savez(CACHE / "suppfig_post_fix_summary.npz",
             hw_dt010=hw_01, hw_dt005=hw_005,
             delta_ms=delta, delta_pct=pct,
             n_trials=N_TRIALS, n_offsets=len(TBW_OFFSETS),
             n_models=len(MODELS), gNMDA=GNMDA_CONTROL)
    print(f"\nSaved summary -> {CACHE / 'suppfig_post_fix_summary.npz'}")
    print(f"\nTotal elapsed: {time.time() - t_start:.0f}s ({(time.time() - t_start) / 60:.1f} min)")
