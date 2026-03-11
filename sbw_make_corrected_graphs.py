"""
Generate corrected SBW (spatial binding window) graphs on GPU.

This uses the corrected SBW analysis in `SBW_test.py` that integrates MSI spikes
over all sub-steps in each 10 ms frame (via `return_spike_sum=True`), rather
than sampling only the last sub-step.

Outputs are written to `Saved_Images/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from SBW_test import run_spatial_binding_across_models


def _abs_symmetrise(curve: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Convert signed disparity curves into |disparity| by averaging symmetric pairs.

    Expects:
      curve["separations_deg"]: (K,)
      curve["all_prob"]: (M, K)
    Returns:
      {"abs_sep": (K_abs,), "mean": (K_abs,), "sem": (K_abs,), "all": (M, K_abs)}
    """
    seps = np.asarray(curve["separations_deg"], dtype=float)
    all_prob = np.asarray(curve["all_prob"], dtype=float)

    abs_vals = np.unique(np.abs(seps))
    abs_vals.sort()

    all_abs = np.zeros((all_prob.shape[0], abs_vals.size), dtype=float)
    for i, a in enumerate(abs_vals):
        if a == 0:
            cols = np.where(seps == 0)[0]
        else:
            cols = np.where((seps == a) | (seps == -a))[0]
        all_abs[:, i] = all_prob[:, cols].mean(axis=1)

    mean = all_abs.mean(axis=0)
    sem = all_abs.std(axis=0, ddof=1) / np.sqrt(all_abs.shape[0])
    return {"abs_sep": abs_vals, "mean": mean, "sem": sem, "all": all_abs}


def _plot_single(abs_curve: Dict[str, np.ndarray], *, title: str, out_path: Path) -> None:
    x, y, e = abs_curve["abs_sep"], abs_curve["mean"], abs_curve["sem"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(x, y, yerr=e, fmt="o-", lw=1.5, ms=4, capsize=2)
    ax.set(xlabel="Spatial disparity |Δazimuth| (°)", ylabel="P(fusion)", ylim=(-0.05, 1.05))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_overlay(
    abs_curves: Dict[str, Dict[str, np.ndarray]],
    *,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, c in abs_curves.items():
        x, y, e = c["abs_sep"], c["mean"], c["sem"]
        ax.plot(x, y, marker="o", ms=3.5, lw=1.6, label=name)
        ax.fill_between(x, y - e, y + e, alpha=0.15)
    ax.set(xlabel="Spatial disparity |Δazimuth| (°)", ylabel="P(fusion)", ylim=(-0.05, 1.05))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--n-models", type=int, default=10)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--duration", type=int, default=20)
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--sep-max", type=int, default=80)
    ap.add_argument("--sep-step", type=int, default=5)
    ap.add_argument("--outdir", type=Path, default=Path("Saved_Images"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this script is GPU-only.")
    if not str(args.device).startswith("cuda"):
        raise SystemExit("Refusing to run on CPU; pass --device cuda:0 (or similar).")

    base_dir = Path("checkpoint")
    model_paths = [base_dir / f"msi_model_surr_10_{i:02d}.pt" for i in range(int(args.n_models))]
    separations = tuple(range(-int(args.sep_max), int(args.sep_max) + 1, int(args.sep_step)))

    def nmda_hypo(net):
        net.gNMDA = 0.02

    def ffi_reduced(net):
        net.g_FFinh *= 0.3

    def adapt_reduced(net):
        net.aM = 0.0001
        net.bM = 0.2
        net.cM = -60.0
        net.dM = 0.01

    conditions: Dict[str, Callable] = {
        "control": None,
        "NMDA_low": nmda_hypo,
        "FFI_reduced": ffi_reduced,
        "Adapt_reduced": adapt_reduced,
    }

    abs_curves: Dict[str, Dict[str, np.ndarray]] = {}
    for name, tweak_fn in conditions.items():
        pooled = run_spatial_binding_across_models(
            model_paths,
            separations_deg=separations,
            n_trials=int(args.n_trials),
            intensity=float(args.intensity),
            duration=int(args.duration),
            device=str(args.device),
            modify_net=tweak_fn,
        )
        abs_curve = _abs_symmetrise(pooled)
        abs_curves[name] = abs_curve

        np.savez_compressed(
            args.outdir / f"SBW_corrected_{name}.npz",
            abs_sep=abs_curve["abs_sep"],
            mean=abs_curve["mean"],
            sem=abs_curve["sem"],
            raw_separations_deg=np.asarray(pooled["separations_deg"], dtype=float),
            raw_all_prob=np.asarray(pooled["all_prob"], dtype=float),
        )
        _plot_single(
            abs_curve,
            title=f"SBW (corrected) — {name}",
            out_path=args.outdir / f"SBW_corrected_{name}.png",
        )

    _plot_overlay(
        abs_curves,
        title="SBW (corrected) — control + perturbations",
        out_path=args.outdir / "SBW_corrected_overlay.png",
    )

    print(f"[done] Wrote per-condition plots + overlay to {args.outdir}")


if __name__ == "__main__":
    main()

