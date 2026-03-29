"""Regenerate 4 TBW SVGs + PNGs from per-model cache .npy files.

NO simulations. Reads only:
  cache/tbw_tfusion_{cond}_model{00-09}.npy

Produces:
  Saved_Images/TBW_{cond}.svg  and  .png
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from TBW_test import (
    fit_psychometric_curve_improved,
    _find_crossings,
    plot_psychometric_tbw_ax,
)

CACHE = Path("cache")
SAVE = Path("Saved_Images")
SAVE.mkdir(exist_ok=True)

CONDITIONS = ["control", "ff_inhibition", "adaptation", "nmda"]
OFFSETS = list(range(-50, 51, 2))  # 51 offsets
OFFSETS_MS = [o * 10 for o in OFFSETS]
N_MODELS = 10


def reconstruct_pooled(cond):
    """Reconstruct pooled TBW result from per-model .npy cache files."""
    all_pfusion = []
    for i in range(N_MODELS):
        path = CACHE / f"tbw_tfusion_{cond}_model{i:02d}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        p = np.load(path)
        all_pfusion.append(p)
    all_pfusion = np.vstack(all_pfusion)  # (10, 51)
    n = all_pfusion.shape[0]
    return {
        "offsets_ms": np.array(OFFSETS_MS, dtype=float),
        "mean_fusion": all_pfusion.mean(0),
        "sem_fusion": all_pfusion.std(0, ddof=1) / np.sqrt(n),
        "all_fusion": all_pfusion,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("TBW Replot from per-model cache (NO simulations)")
    print("=" * 60)

    # ── Control first (needed as reference for perturbations) ──
    print("\n--- CONTROL ---")
    pooled_ctrl = reconstruct_pooled("control")
    fit_ctrl = fit_psychometric_curve_improved(
        pooled_ctrl["offsets_ms"], pooled_ctrl["mean_fusion"]
    )
    cross_ctrl = _find_crossings(fit_ctrl["xs"], fit_ctrl["ys"], 0.5)
    if len(cross_ctrl) >= 2:
        hw_ctrl = (cross_ctrl[-1] - cross_ctrl[0]) / 2
        print(f"  TBW half-width: {hw_ctrl:.1f} ms")
    else:
        hw_ctrl = 0
        print("  No 50% crossings")

    # Plot control (no reference overlay)
    fig, ax, fit_M = plot_psychometric_tbw_ax(
        pooled_ctrl,
        reference_fit=None,
        title_suffix="",
        cont=False,
        out_path=str(SAVE / "TBW_control.svg"),
    )
    fig.savefig(str(SAVE / "TBW_control.png"), format="png", dpi=150)
    plt.close(fig)
    print(f"  Saved: TBW_control.svg + .png")

    summary = {"control": hw_ctrl}

    # ── Perturbation conditions ──
    for cond in ["ff_inhibition", "adaptation", "nmda"]:
        print(f"\n--- {cond.upper()} ---")
        pooled = reconstruct_pooled(cond)

        fig, ax, fit_M = plot_psychometric_tbw_ax(
            pooled,
            reference_fit=fit_ctrl,
            title_suffix=f"  ({cond.replace('_', ' ')})",
            cont=False,
            out_path=str(SAVE / f"TBW_{cond}.svg"),
        )
        fig.savefig(str(SAVE / f"TBW_{cond}.png"), format="png", dpi=150)
        plt.close(fig)

        cross = _find_crossings(fit_M["xs"], fit_M["ys"], 0.5)
        if len(cross) >= 2:
            hw = (cross[-1] - cross[0]) / 2
            print(f"  TBW half-width: {hw:.1f} ms")
        else:
            hw = 0
            print("  No 50% crossings")

        summary[cond] = hw
        print(f"  Saved: TBW_{cond}.svg + .png")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("TBW HALF-WIDTHS (50% criterion)")
    print("=" * 60)
    for cond, hw in summary.items():
        delta = hw - summary["control"]
        delta_str = f"  (delta = {delta:+.1f} ms)" if cond != "control" else ""
        print(f"  {cond:20s}: {hw:6.1f} ms{delta_str}")
