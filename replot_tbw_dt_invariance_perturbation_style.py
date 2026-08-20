#!/usr/bin/env python3
"""Render the validated TBW ``dt`` comparison in perturbation-figure style.

This is a plotting-only companion to :mod:`replot_tbw_dt_invariance`.  It
reads the frozen validated bundle and its provenance record, repeats the full
data/metric validation, and renders the same result using the visual language
of the manuscript's TBW perturbation panels.  It never loads a checkpoint,
runs a simulation, or invokes a CUDA kernel.

The ``dt=0.10 ms`` curve is the green dashed control reference.  The paired
``dt=0.05 ms`` result is shown as the mauve comparison curve with checkpoint
mean +/- SEM observations.  Reported widths are full 50%-crossing spans, not
half-widths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from replot_tbw_dt_invariance import (
    CONDITIONS,
    EXPECTED_METRICS,
    EXPECTED_OFFSETS,
    EXPECTED_OFFSETS_MS,
    N_MODELS,
    N_SOAS,
    N_TRIALS,
    _load_and_validate_bundle,
    _normalise_svg_whitespace,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / "Saved_Data" / "TBW_dt_invariance_validated.npz"
DEFAULT_SOURCE_PROVENANCE = (
    ROOT / "Saved_Data" / "TBW_dt_invariance_validated.provenance.json"
)
DEFAULT_SIDECAR = (
    ROOT / "Saved_Data" / "TBW_dt_invariance_perturbation_style.provenance.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "Saved_Images"
OUTPUT_STEM = "TBW_dt_invariance_perturbation_style"
FONT_PATH = ROOT / "fonts" / "Roboto-Regular.ttf"

BUNDLE_SHA256 = "2d8da8b04bf02055ae8e2a9f2a1c87d63237e703764613c4a604fc2c0d78e11b"
SOURCE_PROVENANCE_SHA256 = (
    "4828747678945040e0553e8626b99c2a4caa094ea2a3e6ecd568e426259391f5"
)

FIGURE_SIZE_INCHES = (10.0, 5.6)
FIGURE_SIZE_POINTS = (720.0, 403.2)
PNG_DPI = 300
PNG_DIMENSIONS = (3000, 1680)
CRITERION = 0.5

COLOR_DATA = "#939598"
COLOR_CONTROL = "#39b54a"
COLOR_FINE = "#b469a3"
CONTROL_DASHES = (7.4, 3.2)

EXPECTED_FULL_SPANS_MS = np.asarray(
    [219.76208778128776, 221.97261161209735], dtype=np.float64
)
EXPECTED_HALF_WIDTHS_MS = np.asarray(
    [109.88104389064388, 110.98630580604868], dtype=np.float64
)
EXPECTED_POOLED_FULL_DRIFT_MS = 2.210523830809592
EXPECTED_PAIRED_MEAN_FULL_DRIFT_MS = 2.2845003679824516
EXPECTED_PAIRED_CI90_MS = np.asarray(
    [1.4823309194598147, 3.0866698165050885], dtype=np.float64
)
EXPECTED_CURVE_R = 0.99971242611273
EXPECTED_CURVE_NRMSE = 0.010151789178731595
EXPECTED_CURVE_MAX_DELTA = 0.06400000000000006

LEGEND_FINE = "dt=0.05 ms (200 substeps)"
LEGEND_CONTROL = "dt=0.10 ms control (100 substeps)"
X_LABEL = "Audio – Visual onset (ms)"
Y_LABEL = "fusion probability"


def _require(condition: bool, message: str) -> None:
    """Raise a stable error when a provenance or rendering invariant fails."""

    if not condition:
        raise AssertionError(message)


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_close(name: str, actual: Any, expected: Any, atol: float = 5e-9) -> None:
    """Require zero-relative-tolerance numerical agreement."""

    try:
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol)
    except AssertionError as exc:
        raise AssertionError(f"{name} differs from validated expectation: {exc}") from exc


def _validate_environment() -> None:
    """Require an explicitly GPU-hidden, headless rendering process."""

    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
        "CUDA_VISIBLE_DEVICES must be exactly '-1' for this CPU-only renderer",
    )
    _require(
        os.environ.get("MPLBACKEND", "").lower() == "agg",
        "MPLBACKEND must be exactly Agg",
    )
    _require(matplotlib.get_backend().lower() == "agg", "Matplotlib backend is not Agg")


def _validate_sources(
    bundle_path: Path,
    source_provenance_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Validate source hashes, schema, pairing, aggregation, and all metrics."""

    _require(bundle_path.is_file(), f"Missing validated bundle: {bundle_path}")
    _require(
        source_provenance_path.is_file(),
        f"Missing validated provenance: {source_provenance_path}",
    )
    _require(
        _sha256_file(bundle_path) == BUNDLE_SHA256,
        f"Validated bundle hash mismatch: {bundle_path}",
    )
    _require(
        _sha256_file(source_provenance_path) == SOURCE_PROVENANCE_SHA256,
        f"Validated provenance hash mismatch: {source_provenance_path}",
    )

    source_provenance = json.loads(source_provenance_path.read_text(encoding="utf-8"))
    _require(
        source_provenance["artifacts"]["bundle"]["sha256"] == BUNDLE_SHA256,
        "Source provenance does not identify the validated bundle hash",
    )

    data, metrics = _load_and_validate_bundle(bundle_path)

    # Exact sample, ordering, physical-grid, and pairing contract.
    _require(np.array_equal(data["condition_keys"], ["dt0p1", "dt0p05"]), "Condition order mismatch")
    _require(np.array_equal(data["dt_ms"], [0.10, 0.05]), "dt order mismatch")
    _require(np.array_equal(data["n_substeps"], [100, 200]), "Substep order mismatch")
    _require(np.array_equal(data["external_frame_ms"], [10.0]), "External-frame mismatch")
    _require(np.array_equal(data["model_indices"], np.arange(N_MODELS)), "Checkpoint order mismatch")
    _require(np.array_equal(data["seeds"], np.arange(12345, 12345 + N_MODELS)), "Seed order mismatch")
    _require(np.array_equal(data["n_trials_per_soa"], [N_TRIALS]), "Trial count mismatch")
    _require(np.array_equal(data["offsets_frames"], EXPECTED_OFFSETS), "Frame-grid mismatch")
    _require(np.array_equal(data["offsets_ms"], EXPECTED_OFFSETS_MS), "SOA-grid mismatch")
    _require(data["locations"].shape == (N_MODELS, N_SOAS, N_TRIALS), "Paired location shape mismatch")
    _require(data["p_fusion"].shape == (2, N_MODELS, N_SOAS), "p_fusion shape mismatch")
    _require(
        data["fused_flags"].shape == (2, N_MODELS, N_SOAS, N_TRIALS),
        "fused_flags shape mismatch",
    )
    _require(
        source_provenance["sample"]["paired_inputs"]
        == "identical 51×50 location matrix for both dt values per checkpoint",
        "Source provenance pairing contract mismatch",
    )

    recomputed_means = data["p_fusion"].mean(axis=1)
    recomputed_sems = data["p_fusion"].std(axis=1, ddof=1) / np.sqrt(N_MODELS)
    _require(
        np.array_equal(recomputed_means, data["pooled_mean_fusion"]),
        "Stored means are not the exact recomputed checkpoint means",
    )
    _require(
        np.array_equal(recomputed_sems, data["pooled_sem_fusion"]),
        "Stored SEMs are not the exact ddof=1 recomputation",
    )
    _require(np.array_equal(metrics["means"], recomputed_means), "Analysed means differ")
    _require(np.array_equal(metrics["sems"], recomputed_sems), "Analysed SEMs differ")

    # Explicitly repeat the scientific assertions used by this figure.
    _assert_close("pooled full spans", metrics["pooled_full_width_ms"], EXPECTED_FULL_SPANS_MS)
    _assert_close("pooled half-widths", metrics["pooled_half_width_ms"], EXPECTED_HALF_WIDTHS_MS)
    _assert_close(
        "pooled full-span drift",
        metrics["pooled_full_width_drift_ms"],
        EXPECTED_POOLED_FULL_DRIFT_MS,
    )
    _assert_close(
        "paired mean full-span drift",
        metrics["paired_mean_full_width_drift_ms"],
        EXPECTED_PAIRED_MEAN_FULL_DRIFT_MS,
    )
    _assert_close(
        "paired full-span 90% CI",
        metrics["paired_full_width_drift_ci90_ms"],
        EXPECTED_PAIRED_CI90_MS,
    )
    _assert_close("curve Pearson r", metrics["pooled_curve_pearson_r"], EXPECTED_CURVE_R)
    _assert_close("curve NRMSE", metrics["pooled_curve_nrmse"], EXPECTED_CURVE_NRMSE)
    _assert_close(
        "curve maximum absolute difference",
        metrics["pooled_curve_max_abs_delta"],
        EXPECTED_CURVE_MAX_DELTA,
    )
    for index, fit in enumerate(metrics["pooled_fits"]):
        crossings = np.asarray(
            [
                x
                for x in _crossings(np.asarray(fit["xs"]), np.asarray(fit["ys"]), CRITERION)
            ],
            dtype=np.float64,
        )
        _require(crossings.shape == (2,), f"Condition {index} lacks two valid crossings")
        _require(crossings[0] < 0.0 < crossings[1], f"Condition {index} crossings do not bracket zero")
        _assert_close(
            f"condition {index} crossing span",
            crossings[1] - crossings[0],
            EXPECTED_FULL_SPANS_MS[index],
        )

    # Reuse the validated metric contract, including exact per-model/R2 checks.
    for key, expected in EXPECTED_METRICS.items():
        _assert_close(f"validated metric {key}", metrics[key], expected)
    return data, metrics, source_provenance


def _crossings(xs: np.ndarray, ys: np.ndarray, criterion: float) -> list[float]:
    """Return linearly interpolated curve crossings without simulation code."""

    crossings: list[float] = []
    shifted = ys - criterion
    for index in np.flatnonzero(np.diff(np.signbit(shifted))):
        x0, x1 = float(xs[index]), float(xs[index + 1])
        y0, y1 = float(shifted[index]), float(shifted[index + 1])
        _require(y1 != y0, "Degenerate fitted crossing segment")
        crossings.append(x0 - y0 * (x1 - x0) / (y1 - y0))
    return crossings


def _configure_style() -> None:
    """Apply the manuscript perturbation-panel style deterministically."""

    _require(FONT_PATH.is_file(), f"Missing required Roboto font: {FONT_PATH}")
    font_manager.fontManager.addfont(str(FONT_PATH))
    _require(
        font_manager.FontProperties(fname=str(FONT_PATH)).get_name() == "Roboto",
        "Bundled font does not identify as Roboto",
    )
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "Roboto",
            "font.size": 14.0,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 10.5,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "fsts10-tbw-dt-invariance-perturbation-style",
            "pdf.fonttype": 42,
            "pdf.compression": 9,
        }
    )


def _artist_clear_of_science(
    artist_bbox: Bbox,
    axis: plt.Axes,
    metrics: Mapping[str, Any],
    label: str,
) -> None:
    """Prove a legend/annotation does not cover points, fits, or guides."""

    padded = Bbox.from_extents(
        artist_bbox.x0 - 3.0,
        artist_bbox.y0 - 3.0,
        artist_bbox.x1 + 3.0,
        artist_bbox.y1 + 3.0,
    )
    science_points: list[np.ndarray] = []
    for condition_index in range(2):
        mean = metrics["means"][condition_index]
        sem = metrics["sems"][condition_index]
        for x, value, error in zip(EXPECTED_OFFSETS_MS, mean, sem, strict=True):
            for y in (value, np.clip(value - error, 0.0, 1.0), np.clip(value + error, 0.0, 1.0)):
                science_points.append(axis.transData.transform((x, y)))
        fit = metrics["pooled_fits"][condition_index]
        for x, y in zip(fit["xs"], fit["ys"], strict=True):
            science_points.append(axis.transData.transform((x, y)))
    _require(
        not any(padded.contains(float(point[0]), float(point[1])) for point in science_points),
        f"{label} overlaps a plotted observation, SEM endpoint, or fitted curve",
    )

    x_zero = float(axis.transData.transform((0.0, 0.0))[0])
    y_half = float(axis.transData.transform((0.0, CRITERION))[1])
    _require(not (padded.x0 <= x_zero <= padded.x1), f"{label} overlaps the x=0 guide")
    _require(not (padded.y0 <= y_half <= padded.y1), f"{label} overlaps the y=.5 guide")


def _render_one(output_dir: Path, metrics: Mapping[str, Any]) -> dict[str, Path]:
    """Render one deterministic SVG/PDF/PNG triplet."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        suffix: output_dir / f"{OUTPUT_STEM}.{suffix}"
        for suffix in ("svg", "pdf", "png")
    }
    _configure_style()
    figure, axis = plt.subplots(figsize=FIGURE_SIZE_INCHES)
    figure.subplots_adjust(left=0.105, right=0.975, bottom=0.16, top=0.965)

    control_fit = metrics["pooled_fits"][0]
    fine_fit = metrics["pooled_fits"][1]
    control_crossings = _crossings(control_fit["xs"], control_fit["ys"], CRITERION)
    fine_crossings = _crossings(fine_fit["xs"], fine_fit["ys"], CRITERION)
    _require(len(control_crossings) == 2, "Control fit lacks exactly two crossings")
    _require(len(fine_crossings) == 2, "Fine-dt fit lacks exactly two crossings")

    # Perturbation-family reference and comparison conventions.
    axis.plot(
        control_fit["xs"],
        control_fit["ys"],
        color=COLOR_CONTROL,
        linewidth=2.5,
        dashes=CONTROL_DASHES,
        solid_capstyle="round",
        zorder=2.0,
    )
    axis.vlines(
        control_crossings,
        0.0,
        CRITERION,
        colors=COLOR_CONTROL,
        linestyles="--",
        linewidth=1.0,
        zorder=1.4,
    )

    fine_mean = metrics["means"][1]
    fine_sem = metrics["sems"][1]
    lower_error = fine_mean - np.clip(fine_mean - fine_sem, 0.0, 1.0)
    upper_error = np.clip(fine_mean + fine_sem, 0.0, 1.0) - fine_mean
    axis.errorbar(
        EXPECTED_OFFSETS_MS,
        fine_mean,
        yerr=np.vstack([lower_error, upper_error]),
        fmt="o",
        linestyle="none",
        markersize=4.0,
        color=COLOR_DATA,
        markerfacecolor=COLOR_DATA,
        markeredgecolor="black",
        markeredgewidth=0.3,
        ecolor=COLOR_DATA,
        elinewidth=0.5,
        capsize=2.0,
        zorder=4.0,
    )
    axis.plot(
        fine_fit["xs"],
        fine_fit["ys"],
        color=COLOR_FINE,
        linewidth=3.0,
        solid_capstyle="round",
        zorder=3.0,
    )
    fine_mask = (fine_fit["xs"] >= fine_crossings[0]) & (
        fine_fit["xs"] <= fine_crossings[1]
    )
    axis.fill_between(
        fine_fit["xs"],
        0.0,
        CRITERION,
        where=fine_mask,
        color=COLOR_FINE,
        alpha=0.18,
        zorder=1.0,
    )
    axis.vlines(
        fine_crossings,
        0.0,
        CRITERION,
        colors=COLOR_FINE,
        linestyles="--",
        linewidth=1.0,
        zorder=1.6,
    )

    axis.axhline(CRITERION, color="black", linestyle="--", linewidth=1.5, zorder=0.9)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.5, zorder=0.9)
    axis.set_xlim(-520.0, 520.0)
    axis.set_ylim(-0.05, 1.05)
    axis.set_xticks([-400.0, -200.0, 0.0, 200.0, 400.0])
    axis.set_yticks([0.0, 0.5, 1.0])
    axis.set_xlabel(X_LABEL)
    axis.set_ylabel(Y_LABEL)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", which="major", length=5.0, width=0.8)

    legend = axis.legend(
        handles=[
            Line2D([0], [0], color=COLOR_FINE, linewidth=3.0, label=LEGEND_FINE),
            Line2D(
                [0],
                [0],
                color=COLOR_CONTROL,
                linewidth=2.5,
                dashes=CONTROL_DASHES,
                label=LEGEND_CONTROL,
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        frameon=False,
        borderaxespad=0.0,
        handlelength=3.0,
        handletextpad=0.7,
        labelspacing=0.45,
    )
    metric_text = (
        "Full span (dt 0.10 / 0.05 ms)\n"
        f"{metrics['pooled_full_width_ms'][0]:.3f} / "
        f"{metrics['pooled_full_width_ms'][1]:.3f} ms\n"
        f"Δfull = {metrics['pooled_full_width_drift_ms']:+.3f} ms"
    )
    annotation = axis.text(
        0.985,
        0.965,
        metric_text,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        linespacing=1.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        zorder=8.0,
    )

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_bbox = axis.get_window_extent(renderer)
    legend_bbox = legend.get_window_extent(renderer)
    annotation_bbox = annotation.get_window_extent(renderer)
    for bbox, label in ((legend_bbox, "Legend"), (annotation_bbox, "Metric annotation")):
        _require(
            axes_bbox.contains(float(bbox.x0), float(bbox.y0))
            and axes_bbox.contains(float(bbox.x1), float(bbox.y1)),
            f"{label} is clipped by or leaves the axes canvas",
        )
        _artist_clear_of_science(bbox, axis, metrics, label)
    _require(not legend_bbox.overlaps(annotation_bbox), "Legend overlaps metric annotation")

    title = "Validated TBW integration-step invariance in perturbation-panel style"
    description = (
        "Paired control TBW curves at dt=0.10/100 and dt=0.05/200; "
        "10 checkpoints, 51 SOAs, and 50 trials per SOA."
    )
    figure.savefig(
        outputs["svg"],
        format="svg",
        metadata={"Title": title, "Description": description, "Creator": "Matplotlib", "Date": None},
    )
    _normalise_svg_whitespace(outputs["svg"])
    figure.savefig(
        outputs["pdf"],
        format="pdf",
        metadata={
            "Title": title,
            "Subject": description,
            "Creator": "Matplotlib",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        outputs["png"],
        format="png",
        dpi=PNG_DPI,
        metadata={"Title": title, "Description": description, "Software": "Matplotlib"},
    )
    plt.close(figure)
    return outputs


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Read PNG pixel dimensions from the IHDR header."""

    header = path.read_bytes()[:24]
    _require(header[:8] == b"\x89PNG\r\n\x1a\n", f"Invalid PNG signature: {path}")
    return tuple(int(value) for value in struct.unpack(">II", header[16:24]))


def _qa_outputs(outputs: Mapping[str, Path]) -> dict[str, Any]:
    """Verify dimensions, formats, vector text, fonts, and SVG cleanliness."""

    _require(_png_dimensions(outputs["png"]) == PNG_DIMENSIONS, "PNG dimensions mismatch")
    _require(outputs["pdf"].read_bytes()[:5] == b"%PDF-", "Invalid PDF signature")

    svg_text = outputs["svg"].read_text(encoding="utf-8")
    svg_root = ET.fromstring(svg_text)
    width_match = re.fullmatch(r"([0-9.]+)pt", svg_root.attrib.get("width", ""))
    height_match = re.fullmatch(r"([0-9.]+)pt", svg_root.attrib.get("height", ""))
    _require(width_match is not None and height_match is not None, "SVG dimensions lack pt units")
    _assert_close("SVG width", float(width_match.group(1)), FIGURE_SIZE_POINTS[0], atol=1e-9)
    _assert_close("SVG height", float(height_match.group(1)), FIGURE_SIZE_POINTS[1], atol=1e-9)
    _require("<image" not in svg_text.lower(), "SVG contains a raster image element")
    _require("data:image" not in svg_text.lower(), "SVG contains embedded raster data")
    _require("Roboto" in svg_text, "SVG does not retain the required Roboto font family")
    _require(svg_text.endswith("\n"), "SVG lacks a final newline")
    _require(
        all(not line.endswith((" ", "\t")) for line in svg_text.splitlines()),
        "SVG contains trailing horizontal whitespace",
    )
    for text in (X_LABEL, Y_LABEL, LEGEND_FINE, LEGEND_CONTROL, "Δfull = +2.211 ms"):
        _require(text in svg_text, f"Searchable SVG text is missing: {text}")

    pdfinfo = subprocess.check_output(["pdfinfo", str(outputs["pdf"])], text=True)
    page_match = re.search(r"^Pages:\s+(\d+)\s*$", pdfinfo, flags=re.MULTILINE)
    size_match = re.search(
        r"^Page size:\s+([0-9.]+) x ([0-9.]+) pts",
        pdfinfo,
        flags=re.MULTILINE,
    )
    _require(page_match is not None and int(page_match.group(1)) == 1, "PDF is not one page")
    _require(size_match is not None, "PDF page size is unavailable")
    _assert_close("PDF width", float(size_match.group(1)), FIGURE_SIZE_POINTS[0], atol=1e-6)
    _assert_close("PDF height", float(size_match.group(2)), FIGURE_SIZE_POINTS[1], atol=1e-6)
    pdf_text = subprocess.check_output(["pdftotext", str(outputs["pdf"]), "-"], text=True)
    for text in (X_LABEL, Y_LABEL, LEGEND_FINE, LEGEND_CONTROL, "Δfull = +2.211 ms"):
        _require(text in pdf_text, f"Searchable PDF text is missing: {text}")
    pdf_fonts = subprocess.check_output(["pdffonts", str(outputs["pdf"])], text=True)
    _require("Roboto" in pdf_fonts, "PDF does not embed Roboto")

    file_types = {
        suffix: subprocess.check_output(["file", "-b", str(path)], text=True).strip()
        for suffix, path in outputs.items()
    }
    _require("SVG" in file_types["svg"], "file(1) does not identify the SVG")
    _require("PDF" in file_types["pdf"], "file(1) does not identify the PDF")
    _require("PNG" in file_types["png"], "file(1) does not identify the PNG")
    return {
        "png_dimensions": list(PNG_DIMENSIONS),
        "svg_size_points": list(FIGURE_SIZE_POINTS),
        "pdf_size_points": list(FIGURE_SIZE_POINTS),
        "pdf_pages": 1,
        "file_types": file_types,
        "searchable_svg_text": True,
        "searchable_pdf_text": True,
        "pdf_embeds_roboto": True,
        "svg_contains_raster_image": False,
        "svg_trailing_whitespace": False,
        "geometry_nonoverlap_and_canvas_clipping_assertions": "passed during both renders",
    }


def _protected_hashes() -> dict[str, str]:
    """Hash all pre-existing TBW figure exports, excluding this new stem."""

    return {
        str(path.relative_to(ROOT)): _sha256_file(path)
        for path in sorted((ROOT / "Saved_Images").glob("TBW_*"))
        if path.is_file() and path.stem != OUTPUT_STEM
    }


def _canonical_command() -> str:
    """Return the exact environment-prefixed invocation."""

    return shlex.join(
        [
            "env",
            f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}",
            f"MPLBACKEND={os.environ['MPLBACKEND']}",
            "python",
            *sys.argv,
        ]
    )


def _git_record(scoped_paths: Sequence[Path]) -> dict[str, Any]:
    """Record Git identity and status only for this renderer's intended files."""

    relative = [str(path.resolve().relative_to(ROOT)) for path in scoped_paths]
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *relative],
        cwd=ROOT,
        text=True,
    ).splitlines()
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "status_scope": relative,
        "status_porcelain": status,
        "scope_note": (
            "Status is intentionally limited to the perturbation-style renderer, "
            "sidecar, and exports; files outside that scope are excluded."
        ),
    }


def _metric_record(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact validated metrics relevant to this rendering."""

    return {
        "pooled_full_width_ms": metrics["pooled_full_width_ms"].tolist(),
        "pooled_half_width_ms": metrics["pooled_half_width_ms"].tolist(),
        "pooled_full_width_drift_ms": metrics["pooled_full_width_drift_ms"],
        "pooled_half_width_drift_ms": metrics["pooled_half_width_drift_ms"],
        "paired_mean_full_width_drift_ms": metrics["paired_mean_full_width_drift_ms"],
        "paired_mean_half_width_drift_ms": metrics["paired_mean_half_width_drift_ms"],
        "paired_full_width_drift_ci90_ms": metrics["paired_full_width_drift_ci90_ms"].tolist(),
        "pooled_curve_pearson_r": metrics["pooled_curve_pearson_r"],
        "pooled_curve_nrmse": metrics["pooled_curve_nrmse"],
        "pooled_curve_max_abs_delta": metrics["pooled_curve_max_abs_delta"],
    }


def _write_sidecar(
    path: Path,
    bundle_path: Path,
    source_provenance_path: Path,
    outputs: Mapping[str, Path],
    metrics: Mapping[str, Any],
    qa: Mapping[str, Any],
    protected_hashes: Mapping[str, str],
    deterministic_hashes: Mapping[str, str],
) -> None:
    """Write timestamp-free provenance for this alternate visual encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    intended_paths = [Path(__file__).resolve(), path, *outputs.values()]
    exports = {
        suffix: {
            "path": str(output.relative_to(ROOT)),
            "size_bytes": output.stat().st_size,
            "sha256": _sha256_file(output),
        }
        for suffix, output in outputs.items()
    }
    exports["png"]["pixel_dimensions"] = list(_png_dimensions(outputs["png"]))

    sidecar = {
        "schema_version": 1,
        "purpose": (
            "Alternate perturbation-panel-style rendering of the unchanged validated "
            "TBW dt-invariance dataset; no simulation or scientific value was changed."
        ),
        "source": {
            "bundle": {
                "path": str(bundle_path.relative_to(ROOT)),
                "sha256": _sha256_file(bundle_path),
                "size_bytes": bundle_path.stat().st_size,
            },
            "validated_provenance": {
                "path": str(source_provenance_path.relative_to(ROOT)),
                "sha256": _sha256_file(source_provenance_path),
            },
            "source_data_policy": "Only the validated bundle and its provenance supply plotted data.",
        },
        "sample": {
            "conditions_in_bundle_order": [
                {"dt_ms": 0.10, "n_substeps": 100, "role": "control reference"},
                {"dt_ms": 0.05, "n_substeps": 200, "role": "fine-dt comparison"},
            ],
            "checkpoints": N_MODELS,
            "soas": N_SOAS,
            "offsets_ms": EXPECTED_OFFSETS_MS.tolist(),
            "trials_per_soa": N_TRIALS,
            "paired_inputs": "one shared 51×50 location matrix per checkpoint across dt values",
        },
        "aggregation": {
            "mean": "arithmetic mean across 10 checkpoint-level p_fusion curves at each SOA",
            "sem": "sample SD across checkpoints (ddof=1) divided by sqrt(10)",
            "fine_dt_observations": "dt=0.05 ms pooled mean +/- SEM",
            "fits": "canonical validated pooled fits; fits are not averaged",
            "width": "full separation of the two fitted P(fusion)=0.5 crossings",
        },
        "metrics": _metric_record(metrics),
        "style_contract": {
            "family": "legacy TBW perturbation panel",
            "figure_size_inches": list(FIGURE_SIZE_INCHES),
            "vector_size_points": list(FIGURE_SIZE_POINTS),
            "png_dpi": PNG_DPI,
            "font": {"family": "Roboto", "path": str(FONT_PATH.relative_to(ROOT))},
            "labels_pt": 14,
            "ticks_pt": 12,
            "xlim_ms": [-520, 520],
            "ylim": [-0.05, 1.05],
            "xticks_ms": [-400, -200, 0, 200, 400],
            "yticks": [0, 0.5, 1],
            "axis_labels": {"x": X_LABEL, "y": Y_LABEL},
            "control": {
                "color": COLOR_CONTROL,
                "line": "dashed",
                "linewidth_pt": 2.5,
                "markers": False,
                "crossing_verticals": True,
            },
            "fine_dt": {
                "fit_color": COLOR_FINE,
                "fit_line": "solid",
                "fit_linewidth_pt": 3,
                "data_color": COLOR_DATA,
                "mean_sem_points": True,
                "span_shading_alpha": 0.18,
                "crossing_verticals": True,
            },
            "guides": "black dashed y=0.5 and x=0, linewidth 1.5 pt",
            "title": None,
        },
        "artifacts": {
            "renderer": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "exports": exports,
        },
        "determinism": {
            "method": "two independent temporary-directory renders compared byte for byte",
            "passed": True,
            "matching_export_sha256": dict(deterministic_hashes),
            "timestamp_free_svg_pdf_metadata": True,
        },
        "qa": {
            **dict(qa),
            "protected_existing_tbw_exports_unchanged": True,
            "protected_existing_tbw_export_sha256": dict(protected_hashes),
            "visual_inspection": "performed separately on the full-resolution PNG and thumbnail",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
            "renderer": "CPU-only Matplotlib Agg; GPUs hidden and no simulation invoked",
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "repository": _git_record(intended_paths),
        "command": _canonical_command(),
    }
    path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the validated paired TBW dt comparison in the manuscript's "
            "legacy perturbation-panel style; plotting only, with no simulation."
        )
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--source-provenance", type=Path, default=DEFAULT_SOURCE_PROVENANCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _validate_environment()
    bundle_path = args.bundle.resolve()
    source_provenance_path = args.source_provenance.resolve()
    output_dir = args.output_dir.resolve()
    sidecar_path = args.sidecar.resolve()
    data, metrics, _ = _validate_sources(bundle_path, source_provenance_path)
    _require(data["p_fusion"].shape == (2, 10, 51), "Validated source shape changed")

    protected_before = _protected_hashes()
    with tempfile.TemporaryDirectory(prefix="tbw-dt-style-a-") as first_temp, tempfile.TemporaryDirectory(
        prefix="tbw-dt-style-b-"
    ) as second_temp:
        first = _render_one(Path(first_temp), metrics)
        second = _render_one(Path(second_temp), metrics)
        first_hashes = {suffix: _sha256_file(path) for suffix, path in first.items()}
        second_hashes = {suffix: _sha256_file(path) for suffix, path in second.items()}
        _require(first_hashes == second_hashes, "Independent renders are not byte-identical")

        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            suffix: output_dir / f"{OUTPUT_STEM}.{suffix}"
            for suffix in ("svg", "pdf", "png")
        }
        for suffix, destination in outputs.items():
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(first[suffix], temporary)
            os.replace(temporary, destination)

    _require(
        {suffix: _sha256_file(path) for suffix, path in outputs.items()} == first_hashes,
        "Installed exports differ from the independently checked render",
    )
    qa = _qa_outputs(outputs)
    protected_after = _protected_hashes()
    _require(
        protected_before == protected_after,
        "A pre-existing historical/current TBW export changed during rendering",
    )
    _write_sidecar(
        sidecar_path,
        bundle_path,
        source_provenance_path,
        outputs,
        metrics,
        qa,
        protected_after,
        first_hashes,
    )

    print("Validated perturbation-style TBW dt-invariance figure")
    print(f"  full spans (dt .10/.05): {EXPECTED_FULL_SPANS_MS[0]:.3f} / {EXPECTED_FULL_SPANS_MS[1]:.3f} ms")
    print(f"  pooled full-span drift: {EXPECTED_POOLED_FULL_DRIFT_MS:+.3f} ms")
    print(f"  curve r / NRMSE / max|delta|: {EXPECTED_CURVE_R:.6f} / {EXPECTED_CURVE_NRMSE:.6f} / {EXPECTED_CURVE_MAX_DELTA:.3f}")
    for suffix, output in outputs.items():
        print(f"  {output.relative_to(ROOT)}  sha256={_sha256_file(output)}")
    print(f"  {sidecar_path.relative_to(ROOT)}  sha256={_sha256_file(sidecar_path)}")


if __name__ == "__main__":
    main()
