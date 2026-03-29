"""Run TBW + SBW for NMDA INCREASE condition (gNMDA=0.2, 4x baseline).

TBW: temporal fusion (is_temporally_fused) — 10 models, 50 trials, offsets -50..+50 step 2
SBW: enhancement-based P(fusion) — 10 models, 50 trials, threshold=10

Both produce SVG+PNG with control overlay.
"""
import sys, time, numpy as np, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import curve_fit

from Training import *
from TBW_test import (
    run_fusion_across_models,
    _find_crossings,
    fit_psychometric_curve_improved,
    plot_psychometric_tbw_ax,
)
from SBW_test import (
    run_spatial_binding_across_models,
)

BASE_DIR = Path("checkpoint")
MODEL_PATHS = [BASE_DIR / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
SAVE_DIR = Path("Saved_Images")
SAVE_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

NMDA_INCREASE_FN = lambda net: setattr(net, "gNMDA", 0.2)

CONTROL_SBW_FLOOR = 0.0679  # from previous analysis


# ====================================================================
# Helpers
# ====================================================================
def _save_pooled(pooled, path):
    saveable = {k: np.asarray(v) for k, v in pooled.items() if v is not None}
    np.savez(path, **saveable)


def _load_pooled(path):
    d = np.load(path)
    return {k: (float(d[k]) if d[k].ndim == 0 else d[k]) for k in d.files}


def _gaussian_centered(x, amp, sigma):
    return amp * np.exp(-0.5 * (x / sigma) ** 2)


def reconstruct_tbw_control_pooled():
    """Reconstruct TBW control pooled result from per-model cache files."""
    offsets = list(range(-50, 51, 2))
    all_pfusion = []
    for i in range(10):
        p = np.load(CACHE_DIR / f"tbw_tfusion_control_model{i:02d}.npy")
        all_pfusion.append(p)
    all_pfusion = np.vstack(all_pfusion)
    n_models = all_pfusion.shape[0]
    return {
        "offsets_ms": [o * 10 for o in offsets],
        "mean_fusion": all_pfusion.mean(0),
        "sem_fusion": all_pfusion.std(0, ddof=1) / np.sqrt(n_models),
        "all_fusion": all_pfusion,
    }


# ====================================================================
# TBW — NMDA increase
# ====================================================================
def run_tbw_nmda_increase():
    print("=" * 60)
    print("TBW — NMDA INCREASE (gNMDA=0.2, temporal fusion)")
    print("=" * 60)

    offsets = list(range(-50, 51, 2))

    # Get control reference fit
    print("\nReconstructing control pooled from per-model cache...")
    pooled_ctrl = reconstruct_tbw_control_pooled()
    fit_ctrl = fit_psychometric_curve_improved(
        np.asarray(pooled_ctrl["offsets_ms"]),
        pooled_ctrl["mean_fusion"]
    )
    ctrl_crossings = _find_crossings(fit_ctrl["xs"], fit_ctrl["ys"], 0.5)
    if len(ctrl_crossings) >= 2:
        ctrl_hw = (ctrl_crossings[-1] - ctrl_crossings[0]) / 2
        print(f"  Control TBW half-width: {ctrl_hw:.1f} ms")

    # Run NMDA increase
    cache_path = CACHE_DIR / "tbw_nmda_increase_tfusion.npz"
    if cache_path.exists():
        print("\n  Loading from cache...")
        pooled = _load_pooled(cache_path)
    else:
        print("\n  Running temporal fusion (10 models x 50 trials)...")
        t0 = time.time()
        pooled = run_fusion_across_models(
            MODEL_PATHS, offsets, device="cuda",
            fusion_method='temporal_fusion',
            n_trials=50,
            modify_net=NMDA_INCREASE_FN,
        )
        _save_pooled(pooled, cache_path)
        print(f"  Computed in {time.time()-t0:.1f}s")

    # Plot
    fig, ax, fit_M = plot_psychometric_tbw_ax(
        pooled,
        reference_fit=fit_ctrl,
        title_suffix="  (NMDA increase)",
        cont=False,
        out_path=str(SAVE_DIR / "TBW_nmda_increase.svg"),
    )
    # Also save PNG
    fig.savefig(str(SAVE_DIR / "TBW_nmda_increase.png"), format='png', dpi=150)
    plt.close(fig)

    # Report
    mf = pooled["mean_fusion"]
    offs_ms = np.asarray(pooled["offsets_ms"])
    crossings = _find_crossings(fit_M["xs"], fit_M["ys"], 0.5)
    if len(crossings) >= 2:
        hw = (crossings[-1] - crossings[0]) / 2
        print(f"\n  NMDA increase TBW half-width: {hw:.1f} ms")
        print(f"  Delta vs control: {hw - ctrl_hw:+.1f} ms")
    else:
        hw = 0
        print("\n  No 50% crossings found")

    print(f"  Peak P(fusion): {mf.max():.3f} at {offs_ms[mf.argmax()]:.0f} ms")
    print(f"  Saved: TBW_nmda_increase.svg + .png")
    return hw


# ====================================================================
# SBW — NMDA increase
# ====================================================================
def run_sbw_nmda_increase():
    print("\n" + "=" * 60)
    print("SBW — NMDA INCREASE (gNMDA=0.2, enhancement threshold=10)")
    print("=" * 60)

    separations = tuple(range(-80, 85, 5))
    cache_path = CACHE_DIR / "sbw_nmda_increase_t10.npz"

    if cache_path.exists():
        print("  Loading from cache...")
        pooled = _load_pooled(cache_path)
    else:
        print("  Running spatial binding (10 models x 50 trials)...")
        t0 = time.time()
        pooled = run_spatial_binding_across_models(
            MODEL_PATHS,
            separations_deg=separations,
            device="cuda:0",
            method="enhancement",
            n_trials=50,
            enhancement_threshold=10.0,
            modify_net=NMDA_INCREASE_FN,
        )
        _save_pooled(pooled, cache_path)
        print(f"  Computed in {time.time()-t0:.1f}s")

    # Floor subtraction using CONTROL's floor
    sep = pooled["separations_deg"]
    mean_p = pooled["mean_prob"]
    sem_p = pooled["sem_prob"]

    own_floor = mean_p[np.abs(sep) >= 50].mean()
    print(f"  Own floor: {own_floor:.4f}, using control floor: {CONTROL_SBW_FLOOR:.4f}")

    mean_sub = np.clip(mean_p - CONTROL_SBW_FLOOR, 0, 1.0)

    # Fit Gaussian (amp <= 1.0)
    p0 = [mean_sub.max(), 20.0]
    popt, _ = curve_fit(_gaussian_centered, sep, mean_sub, p0=p0,
                        bounds=([0.01, 5], [1.0, 80]))
    amp, sigma = popt
    xs_fit = np.linspace(sep.min(), sep.max(), 600)
    ys_fit = np.clip(_gaussian_centered(xs_fit, *popt), 0, 1.0)
    print(f"  Gaussian fit: amp={amp:.3f}, sigma={sigma:.1f} deg")

    # Load control reference fit
    ctrl_cache = np.load(CACHE_DIR / "sbw_control_t10.npz")
    ctrl_sep = ctrl_cache["separations_deg"]
    ctrl_mean = np.clip(ctrl_cache["mean_prob"] - CONTROL_SBW_FLOOR, 0, 1.0)
    ctrl_p0 = [ctrl_mean.max(), 20.0]
    ctrl_popt, _ = curve_fit(_gaussian_centered, ctrl_sep, ctrl_mean, p0=ctrl_p0,
                             bounds=([0.01, 5], [1.0, 80]))
    xs_ctrl = np.linspace(ctrl_sep.min(), ctrl_sep.max(), 600)
    ys_ctrl = np.clip(_gaussian_centered(xs_ctrl, *ctrl_popt), 0, 1.0)
    ref_fit = (xs_ctrl, ys_ctrl)

    # Half-widths
    cross = _find_crossings(xs_fit, ys_fit, 0.5)
    cross_ctrl = _find_crossings(xs_ctrl, ys_ctrl, 0.5)
    hw = (cross[-1] - cross[0]) / 2 if len(cross) >= 2 else 0
    hw_ctrl = (cross_ctrl[-1] - cross_ctrl[0]) / 2 if len(cross_ctrl) >= 2 else 0
    print(f"  SBW half-width: {hw:.1f} deg (control: {hw_ctrl:.1f} deg, delta: {hw - hw_ctrl:+.1f} deg)")

    # ── Plot ──
    try:
        font_path = './fonts/Roboto-Regular.ttf'
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Roboto'
    except Exception:
        pass
    plt.rcParams['font.size'] = 60
    plt.rcParams['xtick.labelsize'] = 60
    plt.rcParams['ytick.labelsize'] = 60
    plt.rcParams['axes.titlesize'] = 50
    plt.rcParams['axes.labelsize'] = 60
    plt.rcParams['legend.fontsize'] = 60
    plt.rcParams['svg.fonttype'] = 'none'

    fig, ax = plt.subplots(figsize=(25, 14))
    y_bottom = -0.05
    level = 0.5

    ax.errorbar(sep, mean_sub, yerr=sem_p,
                fmt='o', ms=5, capsize=3, color='C0', zorder=3, elinewidth=0.5)
    ax.plot(xs_fit, ys_fit, color="C1", lw=2.5, zorder=2)

    # Shaded 50% region
    if len(cross) >= 2:
        xL, xR = cross[0], cross[-1]
        mask = (xs_fit >= xL) & (xs_fit <= xR)
        ax.fill_between(xs_fit, y_bottom, level, where=mask,
                        color="C1", alpha=0.18, zorder=0.5)
        ax.vlines([xL, xR], y_bottom, level, ls=":", color="C1", lw=1.3)

    # Control overlay
    xs_ref, ys_ref = ref_fit
    ax.plot(xs_ref, ys_ref, ls="--", color=".5", lw=2, zorder=1.5)
    if len(cross_ctrl) >= 2:
        xL_C, xR_C = cross_ctrl[0], cross_ctrl[-1]
        mask_r = (xs_ref >= xL_C) & (xs_ref <= xR_C)
        ax.fill_between(xs_ref, y_bottom, level, where=mask_r,
                        color=".5", alpha=0.12, zorder=0.4)
        ax.vlines([xL_C, xR_C], y_bottom, level, ls=":", color=".5", lw=1.3)

    ax.axhline(level, ls=":", color=".4", lw=1)
    xticks = np.arange(-80, 85, 20)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(t):d}" for t in xticks])
    y_max = max(1.05, ys_fit.max() * 1.05, mean_sub.max() * 1.05)
    ax.set(xlabel="Spatial disparity (deg)",
           ylabel="Fusion probability",
           title="Spatial Binding Window",
           ylim=(y_bottom, y_max),
           xlim=(xs_fit.min(), xs_fit.max()))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.grid(False)
    ax.tick_params(axis='both', which='major', length=20, width=1)
    fig.tight_layout()

    fig.savefig(str(SAVE_DIR / "SBW_nmda_increase.svg"), format='svg')
    fig.savefig(str(SAVE_DIR / "SBW_nmda_increase.png"), format='png', dpi=150)
    plt.close(fig)
    print(f"  Saved: SBW_nmda_increase.svg + .png")

    return hw


# ====================================================================
# Main
# ====================================================================
if __name__ == "__main__":
    t_total = time.time()

    tbw_hw = run_tbw_nmda_increase()
    sbw_hw = run_sbw_nmda_increase()

    print("\n" + "=" * 60)
    print("NMDA INCREASE SUMMARY (gNMDA=0.2)")
    print("=" * 60)
    print(f"  TBW half-width: {tbw_hw:.1f} ms")
    print(f"  SBW half-width: {sbw_hw:.1f} deg")
    print(f"  Total time: {time.time()-t_total:.1f}s")
