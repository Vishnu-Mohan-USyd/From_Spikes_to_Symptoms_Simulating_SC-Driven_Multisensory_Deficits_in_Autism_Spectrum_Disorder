#!/usr/bin/env python3
"""Verify the corrected Gate-2 observer against immutable stored rasters.

This command is intentionally offline: it never loads a checkpoint or runs a
GPU simulation.  It independently recomputes P(fusion), half-peak TBW and
dropout auditing from the debugger-captured raw MSI temporal rasters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mechanism_influence.tbw_temporal_fusion_observer import (  # noqa: E402
    DEFAULT_CALIBRATION,
    classify_temporal_rasters,
)


TRACE_FILES = {
    "shipped": "trace_shipped.npz",
    "PV 0.5x": "trace_pv_gaba_0p5.npz",
    "PV 4x": "trace_pv_gaba_4.npz",
    "shared tauGABA 10": "trace_tau_gaba_10.npz",
    "shared tauGABA 40": "trace_tau_gaba_40.npz",
    "gNMDA .383": "trace_gnmda_0p383.npz",
    "gNMDA .765": "trace_gnmda_0p765.npz",
    "adapt .012/6": "trace_adaptation_0p6.npz",
    "adapt dM-only (.02/6)": "trace_adaptation_d_only.npz",
    "adapt aM-only (.012/10)": "trace_adaptation_a_only.npz",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference_summary.read_text(encoding="utf-8"))
    expected_calibration = reference["calibration"]
    calibration_checks = {
        "H_COMMON": DEFAULT_CALIBRATION.one_peak_amplitude_floor_spikes,
        "L_A": DEFAULT_CALIBRATION.auditory_latency_frames,
        "L_V": DEFAULT_CALIBRATION.visual_latency_frames,
        "W_MERGE": DEFAULT_CALIBRATION.maximum_merged_latency_separation_frames,
        "JITTER": DEFAULT_CALIBRATION.peak_interval_margin_frames,
        "MIN_TOTAL": DEFAULT_CALIBRATION.minimum_total_spikes,
    }
    for name, actual in calibration_checks.items():
        if not np.isclose(float(actual), float(expected_calibration[name]), rtol=0.0, atol=0.0):
            raise AssertionError(
                f"observer calibration {name}={actual!r}, expected {expected_calibration[name]!r}"
            )
    expected_results = reference["results"]
    rows: list[tuple[str, float, float, float]] = []
    for label, filename in TRACE_FILES.items():
        trace_path = args.trace_dir / filename
        with np.load(trace_path) as trace:
            result = classify_temporal_rasters(
                trace["raster_msi"],
                trace["offsets_ms"],
                trace["effective_offsets_frames"],
            )
        expected = expected_results[label]
        actual_curve = np.asarray(result["p_fusion"], dtype=float)
        expected_curve = np.asarray(expected["pf"], dtype=float)
        if not np.array_equal(actual_curve, expected_curve):
            raise AssertionError(f"{label}: corrected P(fusion) curve differs from reference")
        checks = {
            "width": (result["half_peak_width_ms"], expected["width"]),
            "left": (result["half_peak_left_ms"], expected["left"]),
            "right": (result["half_peak_right_ms"], expected["right"]),
            "peak": (result["peak_p_fusion"], expected["peak"]),
            "dropout": (result["overall_dropout_fraction"], expected["dropout"]),
            "separate": (result["overall_separate_fraction"], expected["separate"]),
        }
        for metric, (actual, wanted) in checks.items():
            if not np.isclose(float(actual), float(wanted), rtol=0.0, atol=1e-15):
                raise AssertionError(
                    f"{label}: {metric}={actual!r}, expected {wanted!r}"
                )
        rows.append(
            (
                label,
                float(result["half_peak_width_ms"]),
                float(result["peak_p_fusion"]),
                100.0 * float(result["overall_dropout_fraction"]),
            )
        )

    print("condition\tcorrected_width_ms\tpeak_p_fusion\tdropout_percent")
    for label, width, peak, dropout_percent in rows:
        print(f"{label}\t{width:.0f}\t{peak:.2f}\t{dropout_percent:.1f}")
    print(f"verified {len(rows)}/{len(TRACE_FILES)} immutable raster conditions")


if __name__ == "__main__":
    main()
