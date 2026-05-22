"""Task #136-V1c — paper-fair ff_inhibition re-run on surr_10 ckpts.

Monkeypatches generate_all_fresh.CONDITIONS["ff_inhibition"] to the pristine
fb6d3f6 value `pv_nmda=0.8 + targ_ratio=0.8`, then runs ONLY this condition's
full 8-metric battery on the 10 legacy surr_10 ckpts under current
v2 Training.py state. No codebase edit.

Outputs:
  - task136_logs/paper_fair_ff_inh_grid.json  (full battery)
  - task136_logs/paper_fair_ff_inh_sbw_overlay.png
  - cache/tbw_ff_inh_pristine.npz
  - cache/sbw_ff_inh_pristine_t10.npz

Compared in print-out to the current g_GABA=300 numbers (same metric set) and
to paper-target ranges.
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Monkeypatch CONDITIONS BEFORE importing run_perturbations
import generate_all_fresh as gaf

PRISTINE_FF_INH = lambda n: (
    setattr(n, "pv_nmda", 0.8) or setattr(n, "targ_ratio", 0.8)
)
gaf.CONDITIONS["ff_inhibition"] = PRISTINE_FF_INH
# Keep COND_ORDER unchanged so other CONDITIONS keys remain importable, but
# we will iterate ONLY on ff_inhibition below.

# 2. Now import run_perturbations — it will see the patched CONDITIONS
import task114_logs.run_perturbations as rp  # noqa: E402
from TBW_test import run_fusion_across_models, _find_crossings  # noqa: E402
from SBW_test import run_spatial_binding_across_models  # noqa: E402

# Ensure MODELS resolved relative to HERE (rp.MODELS uses Path("checkpoint"))
print(f"[paper_fair_ff_inh] HERE = {HERE}")
print(f"[paper_fair_ff_inh] MODELS = {[p.name for p in rp.MODELS]}")
print(f"[paper_fair_ff_inh] CONDITIONS['ff_inhibition'] -> "
      f"{gaf.CONDITIONS['ff_inhibition']}")

COND = "ff_inhibition"
OUT_JSON = Path("task136_logs/paper_fair_ff_inh_grid.json")
OUT_JSON.parent.mkdir(exist_ok=True)


def run_all_six_metrics():
    """Execute all 6 run_perturbations metrics on the single ff_inhibition cond."""
    grid = {COND: {}}
    for mname, mfn in rp.METRICS:
        tm = time.time()
        try:
            grid[COND][mname] = mfn(COND)
            dt = time.time() - tm
            summary = grid[COND][mname].get("summary", grid[COND][mname])
            print(f"  [{mname:9s}] done in {dt:5.1f}s  | {summary}", flush=True)
        except Exception as e:
            print(f"  [{mname:9s}] FAILED: {type(e).__name__}: {e}", flush=True)
            grid[COND][mname] = {"error": f"{type(e).__name__}: {e}"}
        with open(OUT_JSON, "w") as f:
            json.dump(grid, f, indent=2, default=float)
    return grid


def run_tbw_ff_inh():
    """TBW HW for ff_inhibition only (one-shot run_fusion_across_models)."""
    print(f"\n--- TBW (paper-fair ff_inh) ---", flush=True)
    t0 = time.time()
    pooled = run_fusion_across_models(
        rp.MODELS, gaf.TBW_OFFSETS, device="cuda",
        fusion_method="temporal_fusion",
        n_trials=gaf.N_TRIALS,
        modify_net=PRISTINE_FF_INH,
    )
    cache_path = Path("cache/tbw_ff_inh_pristine.npz")
    saveable = {k: np.asarray(v) for k, v in pooled.items() if v is not None}
    np.savez(cache_path, **saveable)
    dt = time.time() - t0
    print(f"  Simulation done in {dt:.0f}s, cached → {cache_path}")

    # Compute HW from pooled curve directly (avoid plot_tbw which calls a fitter
    # that may rely on a control reference). Use the raw pooled mean_prob with
    # threshold 0.5 (paper convention).
    seps = np.asarray(pooled["offsets_ms"]) if "offsets_ms" in pooled \
        else np.asarray(gaf.TBW_OFFSETS) * 10
    if "mean_prob" in pooled:
        ys = np.asarray(pooled["mean_prob"])
    elif "p_fusion" in pooled:
        ys = np.asarray(pooled["p_fusion"])
    else:
        # last-resort: average all_prob over models
        ys = np.asarray(pooled.get("all_prob", np.zeros_like(seps))).mean(0)

    cross = _find_crossings(seps, ys, 0.5)
    hw = (cross[-1] - cross[0]) / 2 if len(cross) >= 2 else float("nan")
    print(f"  TBW HW (paper-fair ff_inh) = {hw:.1f} ms "
          f"[pooled p_fusion: min={ys.min():.3f}, max={ys.max():.3f}]")
    return {"hw_ms": float(hw), "pooled_min": float(ys.min()),
            "pooled_max": float(ys.max())}


def run_sbw_ff_inh():
    """SBW HW for ff_inhibition only — direct half-max-width on pooled curve."""
    print(f"\n--- SBW (paper-fair ff_inh) ---", flush=True)
    t0 = time.time()
    pooled = run_spatial_binding_across_models(
        rp.MODELS, separations_deg=gaf.SBW_SEPARATIONS, device="cuda",
        method="enhancement", n_trials=gaf.N_TRIALS,
        enhancement_threshold=gaf.ENH_THRESHOLD,
        modify_net=PRISTINE_FF_INH,
    )
    cache_path = Path("cache/sbw_ff_inh_pristine_t10.npz")
    saveable = {k: np.asarray(v) for k, v in pooled.items() if v is not None}
    np.savez(cache_path, **saveable)
    dt = time.time() - t0
    print(f"  Simulation done in {dt:.0f}s, cached → {cache_path}")

    seps = np.asarray(pooled["separations_deg"])
    p = np.asarray(pooled["mean_prob"])
    sem = np.asarray(pooled["sem_prob"])
    print(f"  Pooled: min={p.min():.3f} max={p.max():.3f} "
          f"p(0)={p[len(p)//2]:.3f} p(±80)={p[0]:.3f}/{p[-1]:.3f}")

    cross = _find_crossings(seps, p, 0.5)
    if len(cross) >= 2:
        hw = (cross[-1] - cross[0]) / 2
        hw_str = f"{hw:.1f}"
    elif p.min() >= 0.5:
        hw = float("inf")
        hw_str = ">80 (saturated_above)"
    else:
        hw = 0.0
        hw_str = "0 (saturated_below)"
    print(f"  SBW HW (paper-fair ff_inh, thr=10) = {hw_str} deg")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(seps, p, yerr=sem, marker="o", color="#b469a3", ms=4, lw=1.5,
                label=f"ff_inh pristine (HW={hw_str})")
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Spatial separation (deg)")
    ax.set_ylabel("P(fusion) — enhancement thr=10")
    ax.set_title("SBW — paper-fair ff_inh (pv_nmda=0.8, targ_ratio=0.8) "
                 "on surr_10 v2 Training.py")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    plot_path = Path("task136_logs/paper_fair_ff_inh_sbw_overlay.png")
    fig.savefig(plot_path, dpi=150)
    print(f"  Plot → {plot_path}")
    return {"hw_deg": float(hw) if np.isfinite(hw) else None,
            "pooled_min": float(p.min()), "pooled_max": float(p.max())}


def main():
    print("=" * 78)
    print("Task #136-V1c  paper-fair ff_inh on surr_10  (v2 Training.py)")
    print("  CONDITIONS['ff_inhibition'] = pv_nmda=0.8 + targ_ratio=0.8  (pristine fb6d3f6)")
    print("=" * 78)

    t0 = time.time()

    # 1. Full 6-metric battery
    print("\n[1/3] Six-metric battery (latency, E/I, precision, Fano, MEI, cuerel)")
    grid = run_all_six_metrics()

    # 2. TBW
    tbw = run_tbw_ff_inh()
    grid[COND]["tbw"] = tbw

    # 3. SBW
    sbw = run_sbw_ff_inh()
    grid[COND]["sbw"] = sbw

    with open(OUT_JSON, "w") as f:
        json.dump(grid, f, indent=2, default=float)

    dt = time.time() - t0
    print(f"\nTotal elapsed: {dt:.0f}s ({dt/60:.1f}min)")
    print(f"Output: {OUT_JSON}")


if __name__ == "__main__":
    main()
