"""Generate all 10 TBW + SBW figures FRESH (full simulations, not from cache).

Conditions (5 per axis, 10 total figures)
-----------------------------------------
Each TBW and SBW figure is produced for the following 5 perturbation
conditions, applied via per-condition ``modify_net`` callables before
each forward pass:

  1. ``control``        — baseline (no modification).
  2. ``ff_inhibition``  — feed-forward inhibition perturbation.
  3. ``adaptation``     — short-term-depression / adaptation perturbation.
  4. ``nmda``           — NMDA conductance reduction.
  5. ``nmda_increase``  — NMDA conductance increase.

Methods
-------
TBW: ``temporal_fusion`` P(fusion) via ``is_temporally_fused()`` classifier
     (peak/valley separation on the smoothed MSI population time-series).
     10 models x 50 trials x 51 offsets (-50..+50 in steps of 2).
SBW: P(fusion) via *enhancement thresholding* with threshold = 10 spikes.
     Per-trial: enhancement = AV_roi - max(A_roi, V_roi).
     10 models x 50 trials x 33 separations (-80..+80 in steps of 5).

Plotting
--------
Plots use the template cosmetic style:
  - Data: #939598 grey with black edge
  - Perturbation fit: #b469a3 purple/mauve, solid 3pt
  - Control overlay: #39b54a green, dashed (7.4, 3.2)
  - SBW uses symmetric pedestal fit + control-floor subtraction.

Inputs
------
Reads checkpoints from ``checkpoint/msi_model_surr_10_{00..09}.pt``.

Outputs / side effects
----------------------
Writes figures to ``Saved_Images/{TBW,SBW}_{condition}.svg`` (and .png)
and the underlying pooled metrics to ``cache/{tbw,sbw}_{condition}.npz``
for later cosmetic-only replots.

Randomness
----------
Per-trial stimulus locations sampled from ``np.random.default_rng()``.
No seed is pinned at this entry-point — check ``run_fusion_across_models``
and ``run_spatial_binding_across_models`` for any internal seeding.
"""
import sys, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from TBW_test import (
    run_fusion_across_models,
    _find_crossings,
)
from SBW_test import run_spatial_binding_across_models

from replot_all_cosmetic import (
    plot_tbw, plot_sbw, fit_sbw_pedestal,
    subtract_control_floor, _load_pooled,
)

BASE = Path("checkpoint")
MODELS = [BASE / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
CACHE = Path("cache")
SAVE = Path("Saved_Images")
CACHE.mkdir(exist_ok=True)
SAVE.mkdir(exist_ok=True)

TBW_OFFSETS = list(range(-50, 51, 2))        # 51 values
SBW_SEPARATIONS = tuple(range(-80, 85, 5))   # 33 values
ENH_THRESHOLD = 10.0
N_TRIALS = 50

CONDITIONS = {
    "control": None,
    "ff_inhibition": lambda n: (setattr(n, "pv_nmda", 0.8) or setattr(n, "targ_ratio", 0.8)),
    "adaptation": lambda n: (setattr(n, "aM", 0.001) or setattr(n, "bM", 0.2) or setattr(n, "cM", -60.0) or setattr(n, "dM", 0.01)),
    "nmda": lambda n: setattr(n, "gNMDA", 0.02),
    "nmda_increase": lambda n: setattr(n, "gNMDA", 0.2),
}

COND_ORDER = ["control", "ff_inhibition", "adaptation", "nmda", "nmda_increase"]


def _save_pooled(pooled, path):
    saveable = {k: np.asarray(v) for k, v in pooled.items() if v is not None}
    np.savez(path, **saveable)


# ====================================================================
# TBW — fresh simulations + cosmetic plots
# ====================================================================
def run_tbw_fresh():
    print("=" * 60)
    print("TBW — FRESH SIMULATIONS (temporal fusion)")
    print(f"  {len(MODELS)} models x {N_TRIALS} trials x {len(TBW_OFFSETS)} offsets")
    print("=" * 60)

    tbw_ref_fit = None
    summary = {}

    for cname in COND_ORDER:
        mod_fn = CONDITIONS[cname]
        cache_path = CACHE / f"tbw_{cname}.npz"
        print(f"\n  --- TBW {cname.upper()} ---")
        t0 = time.time()

        pooled = run_fusion_across_models(
            MODELS, TBW_OFFSETS, device="cuda",
            fusion_method='temporal_fusion',
            n_trials=N_TRIALS,
            modify_net=mod_fn,
        )
        _save_pooled(pooled, cache_path)
        dt = time.time() - t0
        print(f"  Simulation done in {dt:.0f}s, saved to {cache_path}")

        # Reload from cache to match replot_all_cosmetic format
        pooled = _load_pooled(cache_path)

        if cname == "control":
            fig, ax, fit_info = plot_tbw(
                pooled, reference_fit=None,
                out_path=str(SAVE / f"TBW_{cname}.svg"))
            tbw_ref_fit = fit_info
        else:
            fig, ax, fit_info = plot_tbw(
                pooled, reference_fit=tbw_ref_fit,
                out_path=str(SAVE / f"TBW_{cname}.svg"))

        fig.savefig(str(SAVE / f"TBW_{cname}.png"), dpi=300)
        plt.close(fig)

        xs_f, ys_f = fit_info["xs"], fit_info["ys"]
        cross = _find_crossings(xs_f, ys_f, 0.5)
        hw = (cross[-1] - cross[0]) / 2 if len(cross) >= 2 else 0
        summary[cname] = hw
        print(f"  TBW {cname:20s}: HW={hw:.0f}ms")

        torch.cuda.empty_cache()

    return summary


# ====================================================================
# SBW — fresh simulations + pedestal fit + floor subtraction + cosmetic plots
# ====================================================================
def run_sbw_fresh():
    print(f"\n{'=' * 60}")
    print("SBW — FRESH SIMULATIONS (enhancement, threshold=10)")
    print(f"  {len(MODELS)} models x {N_TRIALS} trials x {len(SBW_SEPARATIONS)} separations")
    print("=" * 60)

    # Step 1: Run all 5 conditions
    all_sbw_raw = {}
    for cname in COND_ORDER:
        mod_fn = CONDITIONS[cname]
        cache_path = CACHE / f"sbw_{cname}_t{ENH_THRESHOLD:.0f}.npz"
        print(f"\n  --- SBW {cname.upper()} ---")
        t0 = time.time()

        pooled = run_spatial_binding_across_models(
            MODELS, separations_deg=SBW_SEPARATIONS, device="cuda",
            method="enhancement", n_trials=N_TRIALS,
            enhancement_threshold=ENH_THRESHOLD,
            modify_net=mod_fn,
        )
        _save_pooled(pooled, cache_path)
        dt = time.time() - t0
        print(f"  Simulation done in {dt:.0f}s, saved to {cache_path}")

        all_sbw_raw[cname] = _load_pooled(cache_path)
        torch.cuda.empty_cache()

    # Step 2: Control-floor subtraction
    all_sbw, floor = subtract_control_floor(all_sbw_raw)
    print(f"\n  Control floor subtracted: {floor:.4f}")

    # Step 3: Pedestal fit + cosmetic plots
    sbw_ref_fit = None
    summary = {}

    for cname in COND_ORDER:
        pooled = all_sbw[cname]

        if cname == "control":
            fig, ax, fit_info = plot_sbw(
                pooled, reference_fit=None,
                out_path=str(SAVE / f"SBW_{cname}.svg"))
            sbw_ref_fit = (fit_info[0], fit_info[1])
        else:
            fig, ax, fit_info = plot_sbw(
                pooled, reference_fit=sbw_ref_fit,
                out_path=str(SAVE / f"SBW_{cname}.svg"))

        fig.savefig(str(SAVE / f"SBW_{cname}.png"), dpi=300)
        plt.close(fig)

        xs_f, ys_f = fit_info[0], fit_info[1]
        cross = _find_crossings(xs_f, ys_f, 0.5)
        hw = (cross[-1] - cross[0]) / 2 if len(cross) >= 2 else 0
        summary[cname] = hw
        print(f"  SBW {cname:20s}: HW={hw:.1f}°")

    return summary, floor


# ====================================================================
# Main
# ====================================================================
if __name__ == "__main__":
    t_total = time.time()
    print("FRESH GENERATION — all 10 TBW + SBW figures\n")

    tbw_summary = run_tbw_fresh()

    sbw_summary, floor = run_sbw_fresh()

    # Final report
    dt_total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"ALL DONE — 10 figures in {dt_total:.0f}s ({dt_total/60:.1f}min)")
    print("=" * 60)

    print("\nTBW results:")
    hw_c = tbw_summary["control"]
    for c in COND_ORDER:
        hw = tbw_summary[c]
        d = hw - hw_c
        ds = f" (d={d:+.0f}ms)" if c != "control" else ""
        print(f"  {c:20s}: HW={hw:.0f}ms{ds}")

    print(f"\nSBW results (floor={floor:.4f} subtracted):")
    hw_c = sbw_summary["control"]
    for c in COND_ORDER:
        hw = sbw_summary[c]
        d = hw - hw_c
        ds = f" (d={d:+.1f}°)" if c != "control" else ""
        print(f"  {c:20s}: HW={hw:.1f}°{ds}")

    print(f"\nAll figures saved to {SAVE}/")
