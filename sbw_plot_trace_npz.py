"""
Plot reviewer-ready SBW trace files saved by `sbw_gpu_trace_disparity.py`.

The default mode produces an annotated explanatory inset for the *current* SBW
metric, showing the raster, the raw + smoothed MSI profile, the detected peaks,
and the valley criterion that determines whether the trial is counted as fused.

Example
-------
python sbw_gpu_trace_disparity.py \
    --disparity-deg 80 \
    --bg-lambda 1e-5 \
    --target two_peak_shallow_last \
    --out /tmp/sbw_trace_80deg_fused.npz

python sbw_plot_trace_npz.py \
    --npz /tmp/sbw_trace_80deg_fused.npz \
    --mode explain_last \
    --out Saved_Images/sbw_trace_80deg_fused_explain.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _set_style() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = Path("fonts/Roboto-Regular.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Roboto"

    plt.rcParams["font.size"] = 18
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["legend.fontsize"] = 14


def _load_scalar_str(npz: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in npz:
        return default
    value = npz[key]
    if np.isscalar(value):
        return str(value)
    return str(np.asarray(value).item())


def _label_from_npz(npz: np.lib.npyio.NpzFile, prefix: str) -> dict[str, object]:
    return {
        "label": _load_scalar_str(npz, f"{prefix}_label"),
        "fused": bool(np.asarray(npz[f"{prefix}_fused"]).item()),
        "smoothed": np.asarray(npz[f"{prefix}_smoothed"], dtype=float),
        "peaks": np.asarray(npz[f"{prefix}_peaks"], dtype=int),
        "peak_heights": np.asarray(npz[f"{prefix}_peak_heights"], dtype=float),
        "top_peaks": np.asarray(npz[f"{prefix}_top_peaks"], dtype=int),
        "top_peak_heights": np.asarray(npz[f"{prefix}_top_peak_heights"], dtype=float),
        "valley_idx": int(np.asarray(npz[f"{prefix}_valley_idx"]).item()),
        "valley_value": float(np.asarray(npz[f"{prefix}_valley_value"]).item()),
        "valley_ratio": float(np.asarray(npz[f"{prefix}_valley_ratio"]).item()),
        "valley_path_indices": np.asarray(npz[f"{prefix}_valley_path_indices"], dtype=int),
    }


def _input_pair_from_npz(npz: np.lib.npyio.NpzFile, prefix: str) -> dict[str, object]:
    if f"{prefix}_input_peaks" not in npz:
        return {
            "input_peaks": np.array([-1, -1], dtype=int),
            "input_peak_heights": np.array([np.nan, np.nan], dtype=float),
            "input_pair_distinct": False,
            "input_valley_idx": -1,
            "input_valley_value": np.nan,
            "input_valley_ratio": np.nan,
            "input_valley_path_indices": np.empty(0, dtype=int),
        }
    return {
        "input_peaks": np.asarray(npz[f"{prefix}_input_peaks"], dtype=int),
        "input_peak_heights": np.asarray(npz[f"{prefix}_input_peak_heights"], dtype=float),
        "input_pair_distinct": bool(np.asarray(npz[f"{prefix}_input_pair_distinct"]).item()),
        "input_valley_idx": int(np.asarray(npz[f"{prefix}_input_valley_idx"]).item()),
        "input_valley_value": float(np.asarray(npz[f"{prefix}_input_valley_value"]).item()),
        "input_valley_ratio": float(np.asarray(npz[f"{prefix}_input_valley_ratio"]).item()),
        "input_valley_path_indices": np.asarray(npz[f"{prefix}_input_valley_path_indices"], dtype=int),
    }


def _decision_text(label: dict[str, object], *, metric_name: str) -> str:
    if str(label["label"]) == "silent":
        return f"{metric_name}: fused\nreason: silent profile"
    if str(label["label"]) == "single_peak":
        return f"{metric_name}: fused\nreason: ≤1 detected peak"
    if str(label["label"]) == "two_peak_shallow":
        return (
            f"{metric_name}: fused\n"
            f"reason: shallow valley between top-two peaks\n"
            f"valley / min(peak) = {float(label['valley_ratio']):.2f} > 0.60"
        )
    return (
        f"{metric_name}: separated\n"
        f"reason: deep valley between top-two peaks\n"
        f"valley / min(peak) = {float(label['valley_ratio']):.2f} ≤ 0.60"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--mode",
        choices=("last", "full", "both", "explain_last", "explain_full"),
        default="explain_last",
        help="Which raster/profile view to render.",
    )
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_style()

    data = np.load(args.npz, allow_pickle=True)
    audio_idx = int(data["audio_idx"])
    visual_idx = int(data["visual_idx"])
    audio_deg = float(data["audio_deg"])
    visual_deg = float(data["visual_deg"])
    disparity_deg = float(data["disparity_deg"])
    neuron_count = int(data["msi_profile_last"].size)
    xs_deg = np.linspace(0, 180, neuron_count, endpoint=False)
    bg_lambda = float(np.asarray(data["bg_lambda"]).item()) if "bg_lambda" in data else 0.0
    last_label = _label_from_npz(data, "last")
    full_label = _label_from_npz(data, "full")
    last_input_pair = _input_pair_from_npz(data, "last")
    full_input_pair = _input_pair_from_npz(data, "full")

    def plot_basic(ax_raster, ax_profile, raster, profile, title: str) -> None:
        ax_raster.imshow(raster.T, aspect="auto", origin="lower", cmap="hot")
        ax_raster.set_title(title)
        ax_raster.set_xlabel("Time bin (10 ms)")
        ax_raster.set_ylabel("MSI neuron index")
        ax_raster.tick_params(axis="both", which="major", length=10, width=1)

        y = np.asarray(profile, dtype=float)
        if y.max() > 0:
            y = y / y.max()
        ax_profile.plot(xs_deg, y, lw=1.5)
        ax_profile.axvline(xs_deg[audio_idx], ls="--", c="tab:red", lw=1.6, label=f"A {audio_deg:.1f}°")
        ax_profile.axvline(xs_deg[visual_idx], ls="--", c="tab:green", lw=1.6, label=f"V {visual_deg:.1f}°")
        ax_profile.set_xlabel("Azimuth (°)")
        ax_profile.set_ylabel("Normalised spikes")
        ax_profile.set_ylim(0, 1.05)
        ax_profile.set_xlim(0, 180)
        ax_profile.set_xticks([0, 45, 90, 135, 180])
        ax_profile.legend(frameon=False, loc="upper right")
        for side in ("top", "right"):
            ax_profile.spines[side].set_visible(False)
        ax_profile.tick_params(axis="both", which="major", length=10, width=1)

    def plot_explain(
        ax_raster,
        ax_profile,
        raster,
        profile,
        label: dict[str, object],
        input_pair: dict[str, object],
        metric_name: str,
    ) -> None:
        ax_raster.imshow(raster.T, aspect="auto", origin="lower", cmap="magma")
        ax_raster.set_title(f"{metric_name} raster")
        ax_raster.set_xlabel("Time bin (10 ms)")
        ax_raster.set_ylabel("MSI neuron index")
        ax_raster.tick_params(axis="both", which="major", length=10, width=1)

        raw = np.asarray(profile, dtype=float)
        if raw.max() > 0:
            raw = raw / raw.max()
        smoothed = np.asarray(label["smoothed"], dtype=float)

        ax_profile.plot(xs_deg, raw, color="0.70", lw=2.0, label="Raw profile")
        ax_profile.plot(xs_deg, smoothed, color="C0", lw=2.5, label="Smoothed profile")
        ax_profile.axvline(xs_deg[audio_idx], ls="--", c="tab:red", lw=1.6, label=f"A {audio_deg:.1f}°")
        ax_profile.axvline(xs_deg[visual_idx], ls="--", c="tab:green", lw=1.6, label=f"V {visual_deg:.1f}°")

        all_peaks = np.asarray(label["peaks"], dtype=int)
        if all_peaks.size:
            ax_profile.scatter(
                xs_deg[all_peaks],
                smoothed[all_peaks],
                facecolors="none",
                edgecolors="0.35",
                s=50,
                linewidths=1.2,
                zorder=4,
                label="All detected peaks",
            )
        top_peaks = np.asarray(label["top_peaks"], dtype=int)
        valid_top_peaks = top_peaks[top_peaks >= 0]
        if valid_top_peaks.size:
            ax_profile.scatter(
                xs_deg[valid_top_peaks],
                smoothed[valid_top_peaks],
                color="C1",
                s=70,
                zorder=5,
                label="Top detected peaks",
            )

        valley_path = np.asarray(label["valley_path_indices"], dtype=int)
        if valley_path.size:
            ax_profile.plot(xs_deg[valley_path], smoothed[valley_path], color="C0", lw=4, alpha=0.18)

        valley_idx = int(label["valley_idx"])
        if valley_idx >= 0:
            ax_profile.scatter(
                xs_deg[valley_idx],
                smoothed[valley_idx],
                color="black",
                s=70,
                zorder=6,
                label="Valley",
            )

        input_peaks = np.asarray(input_pair["input_peaks"], dtype=int)
        valid_input_peaks = input_peaks[input_peaks >= 0]
        if valid_input_peaks.size:
            ax_profile.scatter(
                xs_deg[valid_input_peaks],
                smoothed[valid_input_peaks],
                marker="s",
                facecolors="none",
                edgecolors="tab:purple",
                s=90,
                linewidths=1.4,
                zorder=5,
                label="Peaks nearest A/V",
            )
        input_valley_idx = int(input_pair["input_valley_idx"])
        if input_valley_idx >= 0:
            ax_profile.scatter(
                xs_deg[input_valley_idx],
                smoothed[input_valley_idx],
                marker="x",
                color="tab:purple",
                s=90,
                zorder=6,
                label="A/V valley",
            )

        top_peak_heights = np.asarray(label["top_peak_heights"], dtype=float)
        valid_heights = top_peak_heights[np.isfinite(top_peak_heights)]
        if str(label["label"]).startswith("two_peak_") and valid_heights.size >= 2:
            smaller_peak = float(np.min(valid_heights[:2]))
            ax_profile.axhline(
                0.60 * smaller_peak,
                color="0.45",
                ls=":",
                lw=1.4,
                label="0.6 × smaller peak",
            )

        ax_profile.set_xlabel("Azimuth (°)")
        ax_profile.set_ylabel("Normalised spikes")
        ax_profile.set_xlim(0, 180)
        ax_profile.set_ylim(0, 1.05)
        ax_profile.set_xticks([0, 45, 90, 135, 180])
        for side in ("top", "right"):
            ax_profile.spines[side].set_visible(False)
        ax_profile.tick_params(axis="both", which="major", length=10, width=1)
        ax_profile.legend(frameon=False, loc="lower right")
        top_text = ", ".join(str(int(v)) for v in valid_top_peaks.tolist()) if valid_top_peaks.size else "none"
        extra = ""
        if bool(input_pair["input_pair_distinct"]):
            extra = (
                f"\nnearest A/V peaks: {int(input_peaks[0])}, {int(input_peaks[1])}"
                f"\nA/V valley ratio = {float(input_pair['input_valley_ratio']):.2f}"
            )
        ax_profile.text(
            0.02,
            0.98,
            _decision_text(label, metric_name=metric_name) + f"\nused top peaks: {top_text}" + extra,
            transform=ax_profile.transAxes,
            va="top",
            ha="left",
            fontsize=13,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.8",
                "alpha": 0.95,
            },
        )

    if args.mode == "both":
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        plot_basic(
            axes[0, 0],
            axes[0, 1],
            data["msi_raster_last"],
            data["msi_profile_last"],
            "MSI activity (last sub-step)",
        )
        plot_basic(
            axes[1, 0],
            axes[1, 1],
            data["msi_raster_full"],
            data["msi_profile_full"],
            "MSI activity (full-frame sum)",
        )
    elif args.mode == "last":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
        plot_basic(
            axes[0],
            axes[1],
            data["msi_raster_last"],
            data["msi_profile_last"],
            "MSI activity (last sub-step)",
        )
    elif args.mode == "full":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
        plot_basic(
            axes[0],
            axes[1],
            data["msi_raster_full"],
            data["msi_profile_full"],
            "MSI activity (full-frame sum)",
        )
    elif args.mode == "explain_full":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
        plot_explain(
            axes[0],
            axes[1],
            data["msi_raster_full"],
            data["msi_profile_full"],
            full_label,
            full_input_pair,
            "Full-frame metric",
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
        plot_explain(
            axes[0],
            axes[1],
            data["msi_raster_last"],
            data["msi_profile_last"],
            last_label,
            last_input_pair,
            "Current SBW metric",
        )

    title = (
        f"SBW example at {disparity_deg:.0f}° disparity "
        f"(A={audio_deg:.1f}°, V={visual_deg:.1f}°, bg={bg_lambda:g})"
    )
    target = _load_scalar_str(data, "target")
    if target:
        title += f" | target={target}"
    fig.suptitle(title)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format=args.out.suffix.lstrip("."))
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
