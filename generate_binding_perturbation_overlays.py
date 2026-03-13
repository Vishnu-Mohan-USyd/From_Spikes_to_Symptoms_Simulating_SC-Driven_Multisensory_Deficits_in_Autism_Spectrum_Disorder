from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from SBW_test import (
    fit_pedestal_curve,
    plot_spatial_binding_pedestal_ax,
    run_spatial_binding_across_models,
)
from TBW_test import plot_psychometric_tbw_ax, run_fusion_across_models


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    label: str
    apply: Callable[[object], None]


def _apply_ff_inh_reduction(net) -> None:
    net.pv_nmda = 1
    net.targ_ratio = 1
    net.g_FFinh *= 0.3


def _apply_adaptation_reduction(net) -> None:
    net.aM = 0.001
    net.bM = 0.2
    net.cM = -60.0
    net.dM = 0.1


def _apply_nmda_reduction(net) -> None:
    net.gNMDA = 0.02


PERTURBATIONS: tuple[PerturbationSpec, ...] = (
    PerturbationSpec(
        name="ff_inh_reduction",
        label="Reduced FF inhibition",
        apply=_apply_ff_inh_reduction,
    ),
    PerturbationSpec(
        name="adaptation_reduction",
        label="Reduced adaptation",
        apply=_apply_adaptation_reduction,
    ),
    PerturbationSpec(
        name="nmda_reduction",
        label="Reduced NMDA",
        apply=_apply_nmda_reduction,
    ),
)


def _save_png(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")


def _build_contact_sheet(image_paths: list[Path], titles: list[str], out_path: Path, *, title: str) -> None:
    fig, axes = plt.subplots(1, len(image_paths), figsize=(24, 8))
    if len(image_paths) == 1:
        axes = [axes]
    for ax, image_path, panel_title in zip(axes, image_paths, titles, strict=True):
        ax.imshow(mpimg.imread(image_path))
        ax.set_title(panel_title, fontsize=18)
        ax.axis("off")
    fig.suptitle(title, fontsize=22)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def generate_overlays(*, bg_lambda: float, device: str) -> None:
    """
    Generate SBW/TBW control-vs-perturbation overlays for the three requested
    manipulations and save both individual figures and contact sheets.
    """
    base_dir = Path("checkpoint")
    model_paths = [base_dir / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
    saved_dir = Path("Saved_Images")
    saved_dir.mkdir(exist_ok=True)

    separations = tuple(range(-80, 85, 5))
    offsets = list(range(-50, 51, 2))

    print(f"[setup] device={device} bg_lambda={bg_lambda}")
    print("[setup] computing shared control curves")

    sbw_ctrl = run_spatial_binding_across_models(
        model_paths,
        separations_deg=separations,
        bg_lambda=bg_lambda,
        device=device,
    )
    sbw_ref_xs, sbw_ref_ys, _ = fit_pedestal_curve(sbw_ctrl)

    tbw_ctrl = run_fusion_across_models(
        model_paths,
        offsets,
        device=device,
        bg_lambda=bg_lambda,
    )
    _, _, tbw_fit_ctrl = plot_psychometric_tbw_ax(
        tbw_ctrl,
        cont=True,
        out_path=str(saved_dir / "TBW_curve_bg_control_reference.svg"),
    )
    plt.close("all")

    sbw_pngs: list[Path] = []
    tbw_pngs: list[Path] = []
    titles: list[str] = []

    for perturbation in PERTURBATIONS:
        print(f"[sbw] {perturbation.name}")
        sbw_pooled = run_spatial_binding_across_models(
            model_paths,
            separations_deg=separations,
            bg_lambda=bg_lambda,
            modify_net=perturbation.apply,
            device=device,
        )
        sbw_svg = saved_dir / f"SBW_overlay_{perturbation.name}_bg.svg"
        sbw_png = saved_dir / f"SBW_overlay_{perturbation.name}_bg.png"
        fig_sbw, _ = plot_spatial_binding_pedestal_ax(
            sbw_pooled,
            reference_fit=(sbw_ref_xs, sbw_ref_ys),
            cont=False,
            out_path=str(sbw_svg),
        )
        _save_png(fig_sbw, sbw_png)
        plt.close(fig_sbw)

        print(f"[tbw] {perturbation.name}")
        tbw_pooled = run_fusion_across_models(
            model_paths,
            offsets,
            device=device,
            bg_lambda=bg_lambda,
            modify_net=perturbation.apply,
        )
        tbw_svg = saved_dir / f"TBW_overlay_{perturbation.name}_bg.svg"
        tbw_png = saved_dir / f"TBW_overlay_{perturbation.name}_bg.png"
        fig_tbw, _, _ = plot_psychometric_tbw_ax(
            tbw_pooled,
            reference_fit=tbw_fit_ctrl,
            title_suffix="  (comparison)",
            cont=False,
            out_path=str(tbw_svg),
        )
        _save_png(fig_tbw, tbw_png)
        plt.close(fig_tbw)

        sbw_pngs.append(sbw_png)
        tbw_pngs.append(tbw_png)
        titles.append(perturbation.label)

    _build_contact_sheet(
        sbw_pngs,
        titles,
        saved_dir / "SBW_overlay_perturbations_bg_summary.png",
        title="SBW Control vs Perturbation",
    )
    _build_contact_sheet(
        tbw_pngs,
        titles,
        saved_dir / "TBW_overlay_perturbations_bg_summary.png",
        title="TBW Control vs Perturbation",
    )

    print("[done] saved figures:")
    for image_path in sbw_pngs + tbw_pngs:
        print(image_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bg-lambda", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    generate_overlays(bg_lambda=args.bg_lambda, device=args.device)


if __name__ == "__main__":
    main()
