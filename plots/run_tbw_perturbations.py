#!/usr/bin/env python3
"""Plot the exploratory corrected-v2 P(fusion) perturbation curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "results" / "tbw_conductance_seed42"
DEFAULT_OUT = ROOT / "figures" / "tbw_perturbations"

PANELS = (
    {
        "title": "Adaptation",
        "condition": "adaptation_0p6",
        "perturbation_label": r"0.6× parameters ($a_M=.012$, $d_M=6$)",
        "color": "#D55E00",
    },
    {
        "title": "Local GABA timing",
        "condition": "tau_gaba_local_40",
        "perturbation_label": r"Slower local GABA ($\tau=40$ ms)",
        "color": "#009E73",
    },
    {
        "title": "NMDA",
        "condition": "gnmda_0p383",
        "perturbation_label": r"Reduced NMDA ($g_{NMDA}=.383$)",
        "color": "#CC79A7",
    },
)
CONTROL_COLOR = "#3E5C76"
CONTROL_LABEL = "conductance-retrain control (seed 42)"


def load_record(path: Path, expected_condition: str) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("protocol") == "dm10-tbw-perturbation-curves-v1":
        raise ValueError(
            f"{path} contains the legacy max-normalised observer; "
            "a corrected v2 capture is required for scientific plotting"
        )
    if record.get("protocol") != "dm10-corrected-temporal-fusion-curves-v2":
        raise ValueError(f"Unexpected protocol in {path}")
    if record.get("condition") != expected_condition:
        raise ValueError(f"Expected {expected_condition!r} in {path}, got {record.get('condition')!r}")
    if record["summary"]["legacy_regression_widths_match_exactly"] is False:
        raise ValueError(f"Legacy regression check failed in {path}")
    if record["hardware"]["torch_device_name"].find("RTX 5090") < 0:
        raise ValueError(f"Non-canonical hardware in {path}")
    return record


def curve_arrays(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.asarray(record["runs"][0]["offsets_ms"], dtype=float)
    curves = np.asarray([run["p_fusion"] for run in record["runs"]], dtype=float)
    for run in record["runs"][1:]:
        if not np.array_equal(offsets, np.asarray(run["offsets_ms"], dtype=float)):
            raise ValueError(f"SOA grid differs within {record['condition']}")
    return offsets, curves.mean(axis=0), curves.std(axis=0, ddof=1)


def median_width_span(record: dict[str, Any]) -> tuple[float, float, float]:
    median = float(record["summary"]["median_raw_half_max_width_ms"])
    candidates = [
        run for run in record["runs"] if float(run["raw_half_max_width_ms"]) == median
    ]
    run = candidates[0]
    return (
        median,
        float(run["raw_half_max_left_ms"]),
        float(run["raw_half_max_right_ms"]),
    )


def add_curve(ax: Any, record: dict[str, Any], color: str, label: str, marker: str) -> None:
    offsets, mean, sd = curve_arrays(record)
    ax.fill_between(offsets, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                    color=color, alpha=0.12, linewidth=0)
    ax.plot(offsets, mean, color=color, lw=2.4, marker=marker, ms=4.3,
            markerfacecolor="white", markeredgewidth=1.1, label=label, zorder=3)


def add_width_bracket(ax: Any, record: dict[str, Any], y: float, color: str) -> None:
    width, left, right = median_width_span(record)
    ax.annotate("", xy=(right, y), xytext=(left, y),
                arrowprops={"arrowstyle": "|-|", "color": color, "lw": 2.0})
    ax.text((left + right) / 2, y + 0.035, f"{width:.0f} ms", color=color,
            ha="center", va="bottom", fontsize=10.5, fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = ROOT / "fonts" / "Roboto-Regular.ttf"
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = "Roboto"
    plt.rcParams.update({
        "font.size": 11.5,
        "axes.titlesize": 14,
        "axes.labelsize": 12.5,
        "legend.fontsize": 9.2,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    control = load_record(args.data_dir / "shipped_control.json", "shipped")
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 4.65), sharex=True, sharey=True)
    for index, (ax, spec) in enumerate(zip(axes, PANELS)):
        perturbed = load_record(args.data_dir / f"{spec['condition']}.json", spec["condition"])
        add_curve(ax, control, CONTROL_COLOR, CONTROL_LABEL, "o")
        add_curve(ax, perturbed, spec["color"], spec["perturbation_label"], "s")
        ax.axhline(0.5, color="#6B7280", ls=(0, (3, 3)), lw=1.1, alpha=0.75, zorder=0)
        ax.axvline(0, color="#9CA3AF", ls="--", lw=1.0, alpha=0.65, zorder=0)
        add_width_bracket(ax, control, 0.08, CONTROL_COLOR)
        add_width_bracket(ax, perturbed, 0.17, spec["color"])

        control_width = control["summary"]["median_raw_half_max_width_ms"]
        perturbed_width = perturbed["summary"]["median_raw_half_max_width_ms"]
        delta = perturbed_width - control_width
        ax.text(0.98, 0.96, rf"$\Delta$TBW = {delta:+.0f} ms", transform=ax.transAxes,
                ha="right", va="top", fontsize=11.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5})
        ax.set_title(f"{chr(65 + index)}   {spec['title']}", loc="left", fontweight="bold")
        ax.set_xlim(-320, 320)
        ax.set_ylim(-0.025, 1.055)
        ax.set_xticks([-300, -150, 0, 150, 300])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Audio − Visual onset (ms)")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.89), handlelength=2.0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", length=4.5, width=1)
        ax.grid(False)
    axes[0].set_ylabel("Fusion probability")
    figure.suptitle("Corrected P(fusion) TBW: exploratory frozen perturbations",
                    fontsize=16, fontweight="bold", y=1.02)
    figure.text(0.995, 0.01,
                "Single experimental checkpoint; points/bands are mean ± SD across three repeat streams",
                ha="right", va="bottom", fontsize=9, color="#4B5563")
    figure.tight_layout(w_pad=1.8)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / "tbw_perturbations_conductance_seed42_v2"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {stem}.png/.svg/.pdf")


if __name__ == "__main__":
    main()
