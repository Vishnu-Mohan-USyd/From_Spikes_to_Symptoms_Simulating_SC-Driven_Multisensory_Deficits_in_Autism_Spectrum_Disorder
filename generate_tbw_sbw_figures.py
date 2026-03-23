"""
Generate TBW and SBW figures for control + 3 perturbation conditions.

Conditions:
    1. Control            — no perturbation
    2. FF inhibition ↓    — pv_nmda=0.8, targ_ratio=0.8
    3. Adaptation ↓       — aM=0.001, bM=0.2, cM=-60.0, dM=0.01
    4. NMDA ↓             — gNMDA=0.02

Saves 8 figures (4 conditions × {TBW, SBW}) to ./Saved_Images/.
"""

from pathlib import Path
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt

# ── import everything from Training (the model + helpers) ──────────
from Training import *

# ── import TBW helpers ──────────────────────────────────────────────
from TBW_test import (
    load_msi_model as load_msi_model_tbw,
    run_fusion_across_models,
    plot_psychometric_tbw_ax,
    fit_psychometric_curve_improved,
    _find_crossings,
)

# ── import SBW helpers ──────────────────────────────────────────────
from SBW_test import (
    run_spatial_binding_across_models,
    fit_pedestal_curve,
    plot_spatial_binding_pedestal_ax,
)

# ── paths ───────────────────────────────────────────────────────────
BASE_DIR = Path("checkpoint")
MODEL_PATHS = [BASE_DIR / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
SAVE_DIR = Path("Saved_Images")
SAVE_DIR.mkdir(exist_ok=True)

# ── conditions ──────────────────────────────────────────────────────
CONDITIONS = {
    "control": None,  # no modification
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
#  SBW generation
# ====================================================================
def run_sbw_all_conditions():
    """Run SBW for all 4 conditions; save figures."""
    separations = tuple(range(-80, 85, 5))
    saved_paths = {}

    # ── 1) Control ──────────────────────────────────────────────────
    print("=" * 60)
    print("SBW — CONTROL")
    print("=" * 60)
    t0 = time.time()

    pooled_ctrl = run_spatial_binding_across_models(
        MODEL_PATHS,
        separations_deg=separations,
        device="cuda:0",
    )
    xs_ref, ys_ref, _ = fit_pedestal_curve(pooled_ctrl)

    fig_ctrl, _ = plot_spatial_binding_pedestal_ax(
        pooled_ctrl, reference_fit=None, cont=True
    )
    path_ctrl = SAVE_DIR / "SBW_control.svg"
    fig_ctrl.savefig(str(path_ctrl), format="svg")
    plt.close(fig_ctrl)
    saved_paths["SBW_control"] = path_ctrl
    print(f"  Saved: {path_ctrl}  ({time.time() - t0:.1f}s)")

    # ── 2-4) Perturbation conditions ────────────────────────────────
    for cond_name, modify_fn in CONDITIONS.items():
        if cond_name == "control":
            continue

        print(f"\nSBW — {cond_name.upper()}")
        t0 = time.time()

        pooled = run_spatial_binding_across_models(
            MODEL_PATHS,
            separations_deg=separations,
            modify_net=modify_fn,
            device="cuda:0",
        )

        fig, _ = plot_spatial_binding_pedestal_ax(
            pooled,
            reference_fit=(xs_ref, ys_ref),
            cont=False,
        )
        fname = f"SBW_{cond_name}.svg"
        path = SAVE_DIR / fname
        fig.savefig(str(path), format="svg")
        plt.close(fig)
        saved_paths[f"SBW_{cond_name}"] = path
        print(f"  Saved: {path}  ({time.time() - t0:.1f}s)")

    return saved_paths


# ====================================================================
#  TBW generation
# ====================================================================
def run_tbw_all_conditions():
    """Run TBW for all 4 conditions; save figures."""
    offsets = list(range(-50, 51, 2))
    saved_paths = {}

    # ── 1) Control ──────────────────────────────────────────────────
    print("=" * 60)
    print("TBW — CONTROL")
    print("=" * 60)
    t0 = time.time()

    pooled_ctrl = run_fusion_across_models(
        MODEL_PATHS, offsets, device="cuda"
    )

    fig_ctrl, ax_ctrl, fit_ctrl = plot_psychometric_tbw_ax(
        pooled_ctrl, cont=True
    )
    path_ctrl = SAVE_DIR / "TBW_control.svg"
    fig_ctrl.savefig(str(path_ctrl), format="svg")
    plt.close(fig_ctrl)
    saved_paths["TBW_control"] = path_ctrl
    print(f"  Saved: {path_ctrl}  ({time.time() - t0:.1f}s)")

    # ── 2-4) Perturbation conditions ────────────────────────────────
    for cond_name, modify_fn in CONDITIONS.items():
        if cond_name == "control":
            continue

        print(f"\nTBW — {cond_name.upper()}")
        t0 = time.time()

        pooled = run_fusion_across_models(
            MODEL_PATHS, offsets, device="cuda", modify_net=modify_fn
        )

        fig, ax, _ = plot_psychometric_tbw_ax(
            pooled,
            reference_fit=fit_ctrl,
            title_suffix=f"  ({cond_name})",
            cont=False,
        )
        fname = f"TBW_{cond_name}.svg"
        path = SAVE_DIR / fname
        fig.savefig(str(path), format="svg")
        plt.close(fig)
        saved_paths[f"TBW_{cond_name}"] = path
        print(f"  Saved: {path}  ({time.time() - t0:.1f}s)")

    return saved_paths


# ====================================================================
#  Main
# ====================================================================
if __name__ == "__main__":
    print("Generating TBW + SBW figures for 4 conditions (8 total)\n")
    t_total = time.time()

    sbw_paths = run_sbw_all_conditions()
    print()
    tbw_paths = run_tbw_all_conditions()

    all_paths = {**sbw_paths, **tbw_paths}
    print("\n" + "=" * 60)
    print(f"ALL DONE — {len(all_paths)} figures in {time.time() - t_total:.1f}s")
    print("=" * 60)
    for name, p in all_paths.items():
        print(f"  {name}: {p}")
