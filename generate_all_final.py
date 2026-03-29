"""Generate all 8 final figures (4 TBW + 4 SBW enhancement).

TBW: temporal fusion P(fusion) via is_temporally_fused() classifier
SBW: P(fusion) via enhancement thresholding (AV - max(A,V) > threshold)

Saves 8 SVGs to ./Saved_Images/
"""
import sys, time, numpy as np, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from Training import *
from TBW_test import (
    run_fusion_across_models,
    _find_crossings,
    fit_psychometric_curve_improved,
    plot_psychometric_tbw_ax,
)
from SBW_test import (
    run_spatial_binding_across_models,
    fit_gaussian_curve,
    _gaussian_centered,
)

BASE_DIR = Path("checkpoint")
MODEL_PATHS = [BASE_DIR / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
SAVE_DIR = Path("Saved_Images")
SAVE_DIR.mkdir(exist_ok=True)

CONDITIONS = {
    "control": None,
    "ff_inhibition": lambda net: (
        setattr(net, "pv_nmda", 0.8) or setattr(net, "targ_ratio", 0.8)
    ),
    "adaptation": lambda net: (
        setattr(net, "aM", 0.001) or
        setattr(net, "bM", 0.2) or
        setattr(net, "cM", -60.0) or
        setattr(net, "dM", 0.01)
    ),
    "nmda": lambda net: setattr(net, "gNMDA", 0.02),
}


# ====================================================================
# TBW figures — P(fusion) via temporal fusion classifier
# ====================================================================
def run_tbw_all(*, use_cache=False):
    """Run TBW P(fusion) for all conditions using temporal fusion classifier.

    Uses is_temporally_fused() with calibrated params:
    sigma=2, valley_threshold=0.4, min_peak_height=0.2,
    min_peak_separation=3, min_total=10.
    10 models, 50 trials per offset.
    """
    offsets = list(range(-50, 51, 2))
    saved = {}

    cache_ctrl = CACHE_DIR / "tbw_control_tfusion.npz"

    # Control first
    print("=" * 60)
    print("TBW — CONTROL (temporal fusion P(fusion))")
    print("=" * 60)
    t0 = time.time()
    if use_cache and cache_ctrl.exists():
        pooled_ctrl = _load_pooled(cache_ctrl)
        print("  Loaded from cache")
    else:
        pooled_ctrl = run_fusion_across_models(
            MODEL_PATHS, offsets, device="cuda",
            fusion_method='temporal_fusion',
            n_trials=50,
        )
        _save_pooled(pooled_ctrl, cache_ctrl)

    fig_ctrl, ax_ctrl, fit_ctrl = plot_psychometric_tbw_ax(
        pooled_ctrl, cont=True,
        out_path=str(SAVE_DIR / "TBW_control.svg"),
    )
    plt.close(fig_ctrl)
    saved["TBW_control"] = SAVE_DIR / "TBW_control.svg"

    mf = pooled_ctrl["mean_fusion"]
    print(f"  Peak: {mf.max():.3f} at {np.asarray(pooled_ctrl['offsets_ms'])[mf.argmax()]:.0f}ms")
    print(f"  Extremes: {mf[:3].mean():.4f} / {mf[-3:].mean():.4f}")
    print(f"  Saved: {saved['TBW_control']}  ({time.time()-t0:.1f}s)")

    # Perturbation conditions
    for cond_name, modify_fn in CONDITIONS.items():
        if cond_name == "control":
            continue
        cache_path = CACHE_DIR / f"tbw_{cond_name}_tfusion.npz"
        print(f"\nTBW — {cond_name.upper()} (temporal fusion P(fusion))")
        t0 = time.time()
        if use_cache and cache_path.exists():
            pooled = _load_pooled(cache_path)
            print("  Loaded from cache")
        else:
            pooled = run_fusion_across_models(
                MODEL_PATHS, offsets, device="cuda",
                fusion_method='temporal_fusion',
                n_trials=50,
                modify_net=modify_fn,
            )
            _save_pooled(pooled, cache_path)

        fig, ax, _ = plot_psychometric_tbw_ax(
            pooled,
            reference_fit=fit_ctrl,
            title_suffix=f"  ({cond_name})",
            cont=False,
            out_path=str(SAVE_DIR / f"TBW_{cond_name}.svg"),
        )
        plt.close(fig)
        saved[f"TBW_{cond_name}"] = SAVE_DIR / f"TBW_{cond_name}.svg"

        mf = pooled["mean_fusion"]
        print(f"  Peak: {mf.max():.3f} at {np.asarray(pooled['offsets_ms'])[mf.argmax()]:.0f}ms")
        print(f"  Extremes: {mf[:3].mean():.4f} / {mf[-3:].mean():.4f}")
        print(f"  Saved: {saved[f'TBW_{cond_name}']}  ({time.time()-t0:.1f}s)")

    return saved


# ====================================================================
# SBW figures — P(fusion) via enhancement thresholding
# ====================================================================
def plot_sbw_gaussian(pooled, *, reference_fit=None, cont=False, out_path=None):
    """Plot SBW P(fusion) with zero-centered Gaussian fit."""
    x, y, err = (pooled[k] for k in ("separations_deg", "mean_prob", "sem_prob"))
    level = 0.5

    xs_fit, ys_fit, popt = fit_gaussian_curve(pooled)
    amp, sigma = popt
    print(f"  Gaussian fit: amp={amp:.3f}, sigma={sigma:.1f}")

    cross_m = _find_crossings(xs_fit, ys_fit, level)
    if len(cross_m) >= 2:
        xL_M, xR_M = cross_m[0], cross_m[-1]
        print(f"  Fit 50% crossings: [{xL_M:.1f}, {xR_M:.1f}] -> width={xR_M-xL_M:.1f} deg")
    else:
        xL_M = xR_M = 0
        print("  No 50% crossings found")

    cross_raw = _find_crossings(x, y, level)
    if len(cross_raw) >= 2:
        print(f"  Raw 50% crossings: [{cross_raw[0]:.1f}, {cross_raw[-1]:.1f}] -> width={cross_raw[-1]-cross_raw[0]:.1f} deg")

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

    fig, ax = plt.subplots(figsize=(25, 14))
    y_bottom = -0.05

    ax.errorbar(x, y, yerr=err,
                fmt='o', ms=5, capsize=3, color='C0', zorder=3, elinewidth=0.5)
    ax.plot(xs_fit, ys_fit, color="C1", lw=2.5, zorder=2)

    if len(cross_m) >= 2:
        mask_m = (xs_fit >= xL_M) & (xs_fit <= xR_M)
        ax.fill_between(xs_fit, y_bottom, level, where=mask_m,
                        color="C1", alpha=0.18, zorder=0.5)
        ax.vlines([xL_M, xR_M], y_bottom, level, ls=":", color="C1", lw=1.3)

    if reference_fit is not None:
        xs_ref, ys_ref = reference_fit
        ax.plot(xs_ref, ys_ref, ls="--", color=".5", lw=2, zorder=1.5)
        cross_r = _find_crossings(xs_ref, ys_ref, level)
        if len(cross_r) >= 2:
            xL_C, xR_C = cross_r[0], cross_r[-1]
            mask_r = (xs_ref >= xL_C) & (xs_ref <= xR_C)
            ax.fill_between(xs_ref, y_bottom, level, where=mask_r,
                            color=".5", alpha=0.12, zorder=0.4)
            ax.vlines([xL_C, xR_C], y_bottom, level, ls=":", color=".5", lw=1.3)

    ax.axhline(level, ls=":", color=".4", lw=1)
    xticks = np.arange(-80, 85, 20)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(t):d}" for t in xticks])
    ax.set(xlabel="Spatial disparity (deg)",
           ylabel="Fusion probability",
           title="Spatial Binding Window",
           ylim=(y_bottom, max(1.05, ys_fit.max() * 1.05, y.max() * 1.05)),
           xlim=(xs_fit.min(), xs_fit.max()))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.grid(False)
    ax.tick_params(axis='both', which='major', length=20, width=1)
    fig.tight_layout()

    save_path = out_path or str(SAVE_DIR / "SBW_curve.svg")
    plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(save_path, format='svg')
    return fig, ax


CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def _save_pooled(pooled, path):
    """Save pooled dict to .npz for fast re-plotting."""
    # Filter out None values (can't save to npz)
    saveable = {k: np.asarray(v) for k, v in pooled.items() if v is not None}
    np.savez(path, **saveable)


def _load_pooled(path):
    """Load pooled dict from .npz cache."""
    d = np.load(path)
    return {k: (float(d[k]) if d[k].ndim == 0 else d[k]) for k in d.files}


def run_sbw_all(*, use_cache=False, enhancement_threshold=0.0):
    """Run SBW P(fusion) for all conditions.

    Parameters
    ----------
    use_cache : bool
        If True, load cached pooled data instead of recomputing.
    enhancement_threshold : float
        Spike-count threshold for P(fusion) classification.
        A trial is "fused" if enhancement > threshold.
    """
    separations = tuple(range(-80, 85, 5))
    saved = {}

    cache_ctrl = CACHE_DIR / f"sbw_control_t{enhancement_threshold:.0f}.npz"

    # Control first
    print("=" * 60)
    print(f"SBW — CONTROL (P(fusion), threshold={enhancement_threshold})")
    print("=" * 60)
    t0 = time.time()
    if use_cache and cache_ctrl.exists():
        pooled_ctrl = _load_pooled(cache_ctrl)
        print("  Loaded from cache")
    else:
        pooled_ctrl = run_spatial_binding_across_models(
            MODEL_PATHS, separations_deg=separations, device="cuda:0",
            method="enhancement", n_trials=50,
            enhancement_threshold=enhancement_threshold,
        )
        _save_pooled(pooled_ctrl, cache_ctrl)

    xs_ref, ys_ref, _ = fit_gaussian_curve(pooled_ctrl)
    fig_ctrl, _ = plot_sbw_gaussian(
        pooled_ctrl, reference_fit=None, cont=True,
        out_path=str(SAVE_DIR / "SBW_control.svg"),
    )
    plt.close(fig_ctrl)
    saved["SBW_control"] = SAVE_DIR / "SBW_control.svg"
    dt = time.time() - t0

    mp = pooled_ctrl["mean_prob"]
    sep = pooled_ctrl["separations_deg"]
    cross = _find_crossings(sep, mp, 0.5)
    width = cross[-1] - cross[0] if len(cross) >= 2 else 0
    print(f"  Peak: {mp.max():.3f} at {sep[mp.argmax()]:.0f} deg")
    print(f"  Raw 50% SBW: {width:.1f} deg (half-width: {width/2:.1f} deg)")
    print(f"  Extremes: {mp[:3].mean():.4f} / {mp[-3:].mean():.4f}")
    print(f"  Saved: {saved['SBW_control']}  ({dt:.1f}s)")

    # Perturbation conditions — same threshold, no normalization needed
    for cond_name, modify_fn in CONDITIONS.items():
        if cond_name == "control":
            continue
        cache_path = CACHE_DIR / f"sbw_{cond_name}_t{enhancement_threshold:.0f}.npz"
        print(f"\nSBW — {cond_name.upper()} (P(fusion), threshold={enhancement_threshold})")
        t0 = time.time()
        if use_cache and cache_path.exists():
            pooled = _load_pooled(cache_path)
            print("  Loaded from cache")
        else:
            pooled = run_spatial_binding_across_models(
                MODEL_PATHS, separations_deg=separations,
                modify_net=modify_fn, device="cuda:0",
                method="enhancement", n_trials=50,
                enhancement_threshold=enhancement_threshold,
            )
            _save_pooled(pooled, cache_path)

        fig, _ = plot_sbw_gaussian(
            pooled,
            reference_fit=(xs_ref, ys_ref),
            cont=False,
            out_path=str(SAVE_DIR / f"SBW_{cond_name}.svg"),
        )
        plt.close(fig)
        saved[f"SBW_{cond_name}"] = SAVE_DIR / f"SBW_{cond_name}.svg"
        dt = time.time() - t0

        mp = pooled["mean_prob"]
        sep = pooled["separations_deg"]
        cross = _find_crossings(sep, mp, 0.5)
        width = cross[-1] - cross[0] if len(cross) >= 2 else 0
        print(f"  Peak: {mp.max():.3f} at {sep[mp.argmax()]:.0f} deg")
        print(f"  Raw 50% SBW: {width:.1f} deg (half-width: {width/2:.1f} deg)")
        print(f"  Extremes: {mp[:3].mean():.4f} / {mp[-3:].mean():.4f}")
        print(f"  Saved: {saved[f'SBW_{cond_name}']}  ({dt:.1f}s)")

    return saved


# ====================================================================
# Main
# ====================================================================
if __name__ == "__main__":
    replot = "--replot" in sys.argv
    sbw_only = "--sbw-only" in sys.argv
    tbw_only = "--tbw-only" in sys.argv

    # Parse --threshold=N (default 0)
    enh_threshold = 0.0
    for arg in sys.argv[1:]:
        if arg.startswith("--threshold="):
            enh_threshold = float(arg.split("=", 1)[1])

    if replot:
        print("REPLOT MODE — using cached data (no recomputation)\n")
    if sbw_only:
        print("SBW-ONLY MODE — skipping TBW\n")
    if tbw_only:
        print("TBW-ONLY MODE — skipping SBW\n")
    print(f"Enhancement threshold: {enh_threshold}\n")

    t_total = time.time()
    all_paths = {}

    if not sbw_only:
        print("Generating TBW figures (temporal fusion P(fusion))\n")
        tbw_paths = run_tbw_all(use_cache=replot)
        all_paths.update(tbw_paths)
        print()

    if not tbw_only:
        sbw_paths = run_sbw_all(use_cache=replot, enhancement_threshold=enh_threshold)
        all_paths.update(sbw_paths)

    print(f"\n{'='*60}")
    print(f"ALL DONE — {len(all_paths)} figures in {time.time()-t_total:.1f}s")
    print("=" * 60)
    for name, p in all_paths.items():
        print(f"  {name}: {p}")
