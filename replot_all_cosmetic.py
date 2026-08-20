"""Replot all 10 TBW + SBW figures matching template style from cached data.

Cosmetic-only re-plot: reads the pooled per-condition metrics produced
by ``generate_all_fresh.py`` and saved under the canonical cache names
and re-renders the figures without re-running any simulations.  This path is
for cosmetic changes such as colors, fonts, axis labels, fit ranges, and
control overlays.

Inputs
------
Reads:  ``cache/tbw_{cond}.npz``, ``cache/sbw_{cond}_t1110.npz``
        for cond in
        {control, ff_inhibition, adaptation, nmda, nmda_increase}.

Each ``.npz`` contains pooled offset/separation, mean P(fusion), SEM,
and the per-model curves.

Cosmetic conventions
--------------------
Color scheme (kept consistent across all figures):
  - Data points       : ``#939598`` grey, black edge
  - Control fit/line  : ``#39b54a`` green, dashed (7.4, 3.2)
  - Perturbation fit  : ``#b469a3`` purple/mauve, solid 3pt

SBW figures additionally apply:
  - **Control-floor subtraction**: each perturbation curve has the
    fitted control "base" P(fusion) at extreme separations subtracted,
    so figures focus on relative narrowing/broadening rather than
    absolute floor differences.
  - **Symmetric pedestal fit**: a difference-of-sigmoids with a single
    half-width (hw_left == hw_right) and free centre — sharper than the
    asymmetric variant used for TBW.

TBW figures use the asymmetric pedestal fit from
``TBW_test.fit_psychometric_curve_improved`` so audio-leading vs.
visual-leading widths can differ.

Outputs / side effects
----------------------
Writes:  ``Saved_Images/{TBW,SBW}_{cond}.svg`` (+ matching .png).
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import curve_fit

from TBW_test import (
    _find_crossings, _asymmetric_pedestal, fit_psychometric_curve_improved,
)

CACHE = Path("cache")
SAVE = Path("Saved_Images")

# ── Color scheme ──
CLR_DATA   = "#939598"   # grey — data points
CLR_CTRL   = "#39b54a"   # green — control fit
CLR_PERT   = "#b469a3"   # purple/mauve — perturbation fit
CTRL_DASH  = (7.4, 3.2)  # dash pattern for control overlay

CONDITIONS = ["control", "ff_inhibition", "adaptation", "nmda", "nmda_increase"]


def _load_pooled(path):
    d = np.load(path)
    return {k: (float(d[k]) if d[k].ndim == 0 else d[k]) for k in d.files}


def _setup_font():
    try:
        font_manager.fontManager.addfont('./fonts/Roboto-Regular.ttf')
        plt.rcParams['font.family'] = 'Roboto'
    except Exception:
        pass
    plt.rcParams['font.size'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


def _style_ax(ax, fig):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis='both', which='major', length=5, width=0.8)
    fig.tight_layout()


# ====================================================================
# TBW plotting — template style
# ====================================================================
def plot_tbw(pooled, *, reference_fit=None, out_path=None):
    """Plot TBW P(fusion) matching template style."""
    offs = np.asarray(pooled["offsets_ms"])
    mean_fus = pooled["mean_fusion"]
    sem_fus = pooled["sem_fusion"]
    criterion = 0.5

    fit_M = fit_psychometric_curve_improved(offs, mean_fus)
    xs_M, ys_M = fit_M["xs"], fit_M["ys"]
    crossings_M = _find_crossings(xs_M, ys_M, criterion)
    if len(crossings_M) >= 2:
        xL_M, xR_M = crossings_M[0], crossings_M[-1]
    else:
        xL_M = xR_M = fit_M["mu"]

    have_ctrl = isinstance(reference_fit, dict)
    if have_ctrl:
        xs_C, ys_C = reference_fit["xs"], reference_fit["ys"]
        crossings_C = _find_crossings(xs_C, ys_C, criterion)
        if len(crossings_C) >= 2:
            xL_C, xR_C = crossings_C[0], crossings_C[-1]
        else:
            xL_C = xR_C = reference_fit["mu"]

    main_color = CLR_PERT

    _setup_font()
    fig, ax = plt.subplots(figsize=(10, 5.6))

    # Data points — grey with black edge, ~4pt
    ax.errorbar(offs, mean_fus, yerr=sem_fus,
                fmt='o', ms=4, capsize=2, color=CLR_DATA,
                markeredgecolor='black', markeredgewidth=0.3,
                zorder=3, elinewidth=0.5)

    # Main fit curve — solid ~3pt
    ax.plot(xs_M, ys_M, lw=3, color=main_color, solid_capstyle='round', zorder=2)

    # Perturbation 50% shading — light purple, y=0 to y=0.5
    mask_M = (xs_M >= xL_M) & (xs_M <= xR_M)
    ax.fill_between(xs_M, 0, criterion, where=mask_M,
                    color=main_color, alpha=0.18, zorder=1)
    # Perturbation crossing verticals — dashed, drop to x-axis (y=0)
    ax.vlines([xL_M, xR_M], 0, criterion, ls='--', lw=1, color=main_color, zorder=1.5)

    # Control overlay on perturbation plots
    if have_ctrl:
        ax.plot(xs_C, ys_C, lw=2.5, color=CLR_CTRL, dashes=CTRL_DASH,
                solid_capstyle='round', zorder=1.5)
        # Control crossing verticals — green dashed
        if len(crossings_C) >= 2:
            ax.vlines([xL_C, xR_C], 0, criterion, ls='--', lw=1, color=CLR_CTRL, zorder=1.2)

    # 50% horizontal line — thick black dashed, full width
    ax.axhline(criterion, ls='--', lw=1.5, color='black', zorder=0.9)
    # Center vertical — thick black dashed, full height
    ax.axvline(0, ls='--', lw=1.5, color='black', zorder=0.9)

    # Axes
    ax.set_yticks([0, 0.5, 1])
    ax.set(xlabel='Audio – Visual onset (ms)',
           ylabel='fusion probability',
           ylim=(-0.05, 1.05),
           xlim=(offs.min() - 20, offs.max() + 20))

    _style_ax(ax, fig)
    save_path = out_path or str(SAVE / 'TBW_curve.svg')
    fig.savefig(save_path, format='svg')
    return fig, ax, fit_M


# ====================================================================
# SBW plotting (symmetric pedestal + control-floor subtraction)
# ====================================================================
def _symmetric_pedestal(x, base, top, hw, k):
    left = 1.0 / (1.0 + np.exp(-(x + hw) / k))
    right = 1.0 / (1.0 + np.exp((x - hw) / k))
    return base + (top - base) * left * right


def fit_sbw_pedestal(pooled):
    x = pooled["separations_deg"]
    y = pooled["mean_prob"]
    base0 = float(np.mean(np.concatenate([y[:3], y[-3:]])))
    top0 = float(y.max())
    p0 = [base0, top0, 25.0, 5.0]
    bounds = ([-0.05, 0.2, 5, 1], [0.5, 1.2, 80, 30])
    popt, _ = curve_fit(_symmetric_pedestal, x, y, p0=p0,
                        bounds=bounds, maxfev=10000)
    xs = np.linspace(x.min(), x.max(), 600)
    ys = _symmetric_pedestal(xs, *popt)
    y_pred = _symmetric_pedestal(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return xs, ys, popt, r2


def subtract_control_floor(all_pooled):
    ctrl = all_pooled["control"]
    sep = ctrl["separations_deg"]
    far_mask = np.abs(sep) >= 50
    floor = ctrl["mean_prob"][far_mask].mean()
    corrected = {}
    for cname, pooled in all_pooled.items():
        p = dict(pooled)
        p["mean_prob"] = np.clip(p["mean_prob"] - floor, 0, None)
        if "all_prob" in p:
            all_corr = np.clip(p["all_prob"] - floor, 0, None)
            p["all_prob"] = all_corr
            n = all_corr.shape[0]
            p["sem_prob"] = all_corr.std(0, ddof=1) / np.sqrt(n)
        else:
            p["sem_prob"] = np.clip(p["sem_prob"], 0, None)
        corrected[cname] = p
    return corrected, floor


def plot_sbw(pooled, *, reference_fit=None, out_path=None):
    """Plot SBW P(fusion) matching template style."""
    x, y, err = (pooled[k] for k in ("separations_deg", "mean_prob", "sem_prob"))
    level = 0.5

    xs_fit, ys_fit, popt, r2 = fit_sbw_pedestal(pooled)

    cross_m = _find_crossings(xs_fit, ys_fit, level)
    if len(cross_m) >= 2:
        xL_M, xR_M = cross_m[0], cross_m[-1]
    else:
        xL_M = xR_M = 0

    main_color = CLR_PERT

    _setup_font()
    fig, ax = plt.subplots(figsize=(10, 5.6))

    # Data points — grey with black edge, ~4pt
    ax.errorbar(x, y, yerr=err,
                fmt='o', ms=4, capsize=2, color=CLR_DATA,
                markeredgecolor='black', markeredgewidth=0.3,
                zorder=3, elinewidth=0.5)

    # Main fit curve — solid ~3pt
    ax.plot(xs_fit, ys_fit, color=main_color, lw=3,
            solid_capstyle='round', zorder=2)

    # Perturbation 50% shading — light purple, y=0 to y=0.5
    if len(cross_m) >= 2:
        mask_m = (xs_fit >= xL_M) & (xs_fit <= xR_M)
        ax.fill_between(xs_fit, 0, level, where=mask_m,
                        color=main_color, alpha=0.18, zorder=0.5)
        # Perturbation crossing verticals — dashed, drop to x-axis (y=0)
        ax.vlines([xL_M, xR_M], 0, level, ls='--', lw=1, color=main_color, zorder=1.5)

    # Control overlay on perturbation plots
    if reference_fit is not None:
        xs_ref, ys_ref = reference_fit
        ax.plot(xs_ref, ys_ref, lw=2.5, color=CLR_CTRL, dashes=CTRL_DASH,
                solid_capstyle='round', zorder=1.5)
        cross_r = _find_crossings(xs_ref, ys_ref, level)
        if len(cross_r) >= 2:
            xL_C, xR_C = cross_r[0], cross_r[-1]
            # Control crossing verticals — green dashed
            ax.vlines([xL_C, xR_C], 0, level, ls='--', lw=1, color=CLR_CTRL, zorder=1.2)

    # 50% horizontal line — thick black dashed, full width
    ax.axhline(level, ls='--', lw=1.5, color='black', zorder=0.9)
    # Center vertical — thick black dashed, full height
    ax.axvline(0, ls='--', lw=1.5, color='black', zorder=0.9)

    # Axes
    ax.set_yticks([0, 0.5, 1])
    xticks = np.arange(-80, 85, 20)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(t):d}" for t in xticks])
    ax.set(xlabel=u"spatial disparity (\u00b0)",
           ylabel="fusion probability",
           ylim=(-0.05, max(1.05, ys_fit.max() * 1.05, y.max() * 1.05)),
           xlim=(xs_fit.min(), xs_fit.max()))

    _style_ax(ax, fig)
    save_path = out_path or str(SAVE / "SBW_curve.svg")
    fig.savefig(save_path, format='svg')
    return fig, ax, (xs_fit, ys_fit, popt, r2)


# ====================================================================
# Main
# ====================================================================
if __name__ == "__main__":
    print("COSMETIC REPLOT — all 10 TBW + SBW figures (template style)\n")

    # ── TBW ──
    print("=" * 60)
    print("TBW FIGURES")
    print("=" * 60)
    tbw_ref_fit = None
    for cname in CONDITIONS:
        cache_path = CACHE / f"tbw_{cname}.npz"
        if not cache_path.exists():
            print(f"  SKIP TBW {cname}: not found")
            continue
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
        print(f"  TBW {cname:20s}: HW={hw:.0f}ms  saved")

    # ── SBW ──
    print(f"\n{'='*60}")
    print("SBW FIGURES")
    print("=" * 60)

    all_sbw = {}
    for cname in CONDITIONS:
        cache_path = CACHE / f"sbw_{cname}_t1110.npz"
        if not cache_path.exists():
            print(f"  SKIP SBW {cname}: not found")
            continue
        all_sbw[cname] = _load_pooled(cache_path)

    all_sbw, floor = subtract_control_floor(all_sbw)
    print(f"  Control floor subtracted: {floor:.4f}")

    sbw_ref_fit = None
    for cname in CONDITIONS:
        if cname not in all_sbw:
            continue
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
        print(f"  SBW {cname:20s}: HW={hw:.1f}°  saved")

    print(f"\nDone — 10 figures saved to {SAVE}/")
