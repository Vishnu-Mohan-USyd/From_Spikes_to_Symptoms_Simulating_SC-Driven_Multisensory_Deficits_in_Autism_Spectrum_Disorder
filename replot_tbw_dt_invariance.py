#!/usr/bin/env python3
"""Bundle and replot the validated TBW integration-step comparison.

The one-time ``--source-dir`` path verifies the frozen validator manifest,
summary hashes, all 20 source NPZ hashes, exact NPZ schemas, paired seeded
locations, trial-level fusion labels, and canonical fitted metrics before
writing a compact deterministic bundle. Subsequent runs need only the bundle
under ``Saved_Data``; the session-local validator directory is provenance, not
a replotting dependency.

All widths are in milliseconds. ``full width`` means the separation between
the two fitted P(fusion)=0.5 crossings; ``half-width`` is half that separation.
Random locations were frozen upstream with seed ``12345 + model_index``. This
script performs no simulation and invokes no CUDA kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shlex
import struct
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from scipy import stats

from TBW_test import _find_crossings, fit_psychometric_curve_improved


ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / "Saved_Data" / "TBW_dt_invariance_validated.npz"
DEFAULT_PROVENANCE = (
    ROOT / "Saved_Data" / "TBW_dt_invariance_validated.provenance.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "Saved_Images"
OUTPUT_STEM = "TBW_dt_invariance_validated"

EXPECTED_ANALYSIS_SHA256 = (
    "1719f80e3a3f5ad5e3461fe7875954880aa45295965992dd1e922d0d241f2a81"
)
EXPECTED_PAIRED_SUMMARY_SHA256 = (
    "0c96c10237075030bed0155d61444c9ea4ce123f8c4b7e6b6887d3e9ade318bb"
)
EXPECTED_MANIFEST_SHA256 = (
    "349386c66d871977eff6abc8920ab96df8dc8723e3b35abded7f5f072dabab59"
)
EXPECTED_OFFSETS = np.arange(-50, 51, 2, dtype=np.int64)
EXPECTED_OFFSETS_MS = np.arange(-500.0, 501.0, 20.0, dtype=np.float64)
N_MODELS = 10
N_SOAS = 51
N_TRIALS = 50
SEED_BASE = 12345
EXTERNAL_FRAME_MS = 10.0


@dataclass(frozen=True)
class Condition:
    """Validated integration condition and its visual encoding."""

    key: str
    dt_ms: float
    n_substeps: int
    color: str
    linestyle: str
    marker: str
    filled: bool


CONDITIONS = (
    Condition("dt0p1", 0.10, 100, "#009E73", "-", "o", True),
    Condition("dt0p05", 0.05, 200, "#CC79A7", "--", "D", False),
)

SOURCE_SCHEMA: dict[str, tuple[tuple[int, ...], str]] = {
    "offsets": ((51,), "int64"),
    "offsets_ms": ((51,), "float64"),
    "locations": ((51, 50), "int64"),
    "p_fusion": ((51,), "float64"),
    "fused_flags": ((51, 50), "uint8"),
    "fit_xs": ((600,), "float64"),
    "fit_ys": ((600,), "float64"),
    "fit_params": ((6,), "float64"),
    "fit_cov": ((6, 6), "float64"),
    "temporal_profiles_sum": ((51, 50, 60), "float32"),
    "temporal_profiles_last_substep": ((51, 50, 60), "float32"),
    "p_fusion_last_substep": ((51,), "float64"),
    "fused_flags_last_substep": ((51, 50), "uint8"),
    "frame_state_trace": ((60, 9), "float64"),
}

EXPECTED_PER_MODEL_FULL_WIDTH_MS = np.asarray(
    [
        [
            222.41002132052122,
            222.01999021885513,
            219.18879477880590,
            222.32948707647222,
            218.73255308372342,
            222.66834412652070,
            217.40228839857934,
            214.12456967335310,
            219.89805648470570,
            215.26672156278676,
        ],
        [
            223.13303000546006,
            222.66877427246575,
            221.36932330589390,
            224.63492475658683,
            221.44450972663208,
            223.83513933809440,
            219.75381599459962,
            219.61196736591570,
            222.82345003812026,
            217.61089560037948,
        ],
    ],
    dtype=np.float64,
)

EXPECTED_METRICS: dict[str, Any] = {
    "pooled_full_width_ms": np.asarray(
        [219.76208778128776, 221.97261161209735], dtype=np.float64
    ),
    "pooled_half_width_ms": np.asarray(
        [109.88104389064388, 110.98630580604868], dtype=np.float64
    ),
    "pooled_full_width_drift_ms": 2.210523830809592,
    "pooled_half_width_drift_ms": 1.105261915404796,
    "paired_mean_full_width_drift_ms": 2.2845003679824516,
    "paired_mean_half_width_drift_ms": 1.1422501839912258,
    "paired_full_width_drift_ci90_ms": np.asarray(
        [1.4823309194598147, 3.0866698165050885], dtype=np.float64
    ),
    "pooled_curve_pearson_r": 0.99971242611273,
    "pooled_curve_rmse": 0.010151789178731595,
    "pooled_curve_nrmse": 0.010151789178731595,
    "pooled_curve_max_abs_delta": 0.06400000000000006,
    "pooled_curve_mean_abs_delta": 0.0021960784313725503,
    "pooled_fit_r_squared": np.asarray(
        [0.9983860472263326, 0.9985019179377927], dtype=np.float64
    ),
    "minimum_individual_fit_r_squared": 0.995994918007222,
}


def _require(condition: bool, message: str) -> None:
    """Raise an explicit assertion with a stable diagnostic."""

    if not condition:
        raise AssertionError(message)


def _assert_close(
    name: str,
    actual: Any,
    expected: Any,
    *,
    atol: float = 5e-9,
) -> None:
    """Require numerical agreement with zero relative tolerance."""

    try:
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol)
    except AssertionError as exc:
        raise AssertionError(f"{name} differs from frozen expectation: {exc}") from exc


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    """Match the frozen validator's dtype/shape/value array digest."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _source_filename(model_index: int, condition: Condition) -> str:
    return (
        f"model{model_index:02d}_dt{condition.dt_ms:.3f}"
        f"_nsub{condition.n_substeps}.npz"
    )


def _verify_environment() -> None:
    """Enforce the requested device isolation and headless renderer."""

    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "1",
        "CUDA_VISIBLE_DEVICES must be exactly '1' (physical GPU1 only)",
    )
    _require(
        os.environ.get("MPLBACKEND", "").lower() == "agg",
        "MPLBACKEND must be exactly Agg",
    )
    _require(matplotlib.get_backend().lower() == "agg", "Matplotlib is not using Agg")


def _parse_verified_manifest(source_dir: Path) -> tuple[dict[Path, str], dict[str, str]]:
    """Verify the frozen validator manifest and its two summary artifacts."""

    validator_root = source_dir.parent
    manifest_path = validator_root / "artifacts.sha256"
    analysis_path = validator_root / "analysis" / "summary.json"
    paired_summary_path = source_dir / "summary.json"

    for path in (manifest_path, analysis_path, paired_summary_path):
        _require(path.is_file(), f"Missing frozen validator artifact: {path}")

    _require(
        _sha256_file(manifest_path) == EXPECTED_MANIFEST_SHA256,
        f"Frozen manifest hash mismatch: {manifest_path}",
    )
    _require(
        _sha256_file(analysis_path) == EXPECTED_ANALYSIS_SHA256,
        f"Analysis summary hash mismatch: {analysis_path}",
    )
    _require(
        _sha256_file(paired_summary_path) == EXPECTED_PAIRED_SUMMARY_SHA256,
        f"Paired summary hash mismatch: {paired_summary_path}",
    )

    entries: dict[Path, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, raw_path = line.split(maxsplit=1)
        entries[Path(raw_path).resolve()] = digest

    for path, expected in (
        (analysis_path, EXPECTED_ANALYSIS_SHA256),
        (paired_summary_path, EXPECTED_PAIRED_SUMMARY_SHA256),
    ):
        _require(entries.get(path.resolve()) == expected, f"Manifest entry mismatch: {path}")

    summary_hashes = {
        "validator_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "analysis_summary_sha256": EXPECTED_ANALYSIS_SHA256,
        "paired_summary_sha256": EXPECTED_PAIRED_SUMMARY_SHA256,
    }
    return entries, summary_hashes


def _verify_source_npz(path: Path) -> dict[str, np.ndarray]:
    """Validate one frozen condition file and return the arrays used downstream."""

    with np.load(path, allow_pickle=False) as source:
        _require(set(source.files) == set(SOURCE_SCHEMA), f"Unexpected NPZ keys: {path}")
        for key, (shape, dtype) in SOURCE_SCHEMA.items():
            value = source[key]
            _require(value.shape == shape, f"{path.name}:{key} shape {value.shape} != {shape}")
            _require(str(value.dtype) == dtype, f"{path.name}:{key} dtype {value.dtype} != {dtype}")
            _require(np.isfinite(value).all(), f"{path.name}:{key} contains non-finite values")

        offsets = np.asarray(source["offsets"])
        offsets_ms = np.asarray(source["offsets_ms"])
        p_fusion = np.asarray(source["p_fusion"])
        fused_flags = np.asarray(source["fused_flags"])
        locations = np.asarray(source["locations"])

        _require(np.array_equal(offsets, EXPECTED_OFFSETS), f"Offset grid mismatch: {path}")
        _require(
            np.array_equal(offsets_ms, EXPECTED_OFFSETS_MS),
            f"Physical SOA grid mismatch: {path}",
        )
        _require(
            np.logical_or(fused_flags == 0, fused_flags == 1).all(),
            f"Non-binary fused_flags: {path}",
        )
        _require(
            np.logical_or(
                source["fused_flags_last_substep"] == 0,
                source["fused_flags_last_substep"] == 1,
            ).all(),
            f"Non-binary last-substep flags: {path}",
        )
        _require(
            np.logical_and(p_fusion >= 0.0, p_fusion <= 1.0).all(),
            f"p_fusion outside [0,1]: {path}",
        )
        _require(
            np.logical_and(
                source["p_fusion_last_substep"] >= 0.0,
                source["p_fusion_last_substep"] <= 1.0,
            ).all(),
            f"last-substep p_fusion outside [0,1]: {path}",
        )
        _require(
            np.array_equal(p_fusion, fused_flags.mean(axis=1)),
            f"p_fusion != fused_flags.mean(1): {path}",
        )
        _require(
            np.logical_and(locations >= 0, locations < 180).all(),
            f"Locations outside [0,180): {path}",
        )

        stored_crossings = _find_crossings(source["fit_xs"], source["fit_ys"], 0.5)
        _require(len(stored_crossings) == 2, f"Stored fit lacks two crossings: {path}")
        _require(np.any(source["fit_cov"] != 0.0), f"Stored fit used fallback: {path}")

        return {
            "p_fusion": p_fusion.copy(),
            "fused_flags": fused_flags.copy(),
            "locations": locations.copy(),
        }


def _fit_valid_curve(
    offsets_ms: np.ndarray,
    probabilities: np.ndarray,
    label: str,
) -> tuple[dict[str, Any], float]:
    """Fit one curve and require a finite, non-fallback, two-crossing result."""

    _require(probabilities.shape == (N_SOAS,), f"{label}: invalid curve shape")
    _require(np.isfinite(probabilities).all(), f"{label}: non-finite probabilities")
    _require(
        np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all(),
        f"{label}: probabilities outside [0,1]",
    )
    above = np.flatnonzero(probabilities >= 0.5)
    _require(above.size > 0, f"{label}: observed curve never reaches 0.5")
    _require(
        np.any(probabilities[: above[0]] < 0.5)
        and np.any(probabilities[above[-1] + 1 :] < 0.5),
        f"{label}: observed curve does not bracket both 0.5 crossings",
    )

    fit = fit_psychometric_curve_improved(offsets_ms, probabilities)
    _require(np.isfinite(fit["params"]).all(), f"{label}: non-finite fit parameters")
    _require(np.isfinite(fit["cov"]).all(), f"{label}: non-finite fit covariance")
    _require(np.any(fit["cov"] != 0.0), f"{label}: canonical fit fallback used")
    crossings = _find_crossings(fit["xs"], fit["ys"], 0.5)
    _require(len(crossings) == 2, f"{label}: fitted curve lacks exactly two crossings")
    _require(np.isfinite(fit["r_squared"]), f"{label}: non-finite fitted R²")
    return fit, float(crossings[1] - crossings[0])


def _analyse_arrays(data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Recompute pooled/individual fits and every metric used in the figure."""

    p_fusion = np.asarray(data["p_fusion"], dtype=np.float64)
    _require(p_fusion.shape == (2, N_MODELS, N_SOAS), "Invalid p_fusion bundle shape")

    means = p_fusion.mean(axis=1)
    sems = p_fusion.std(axis=1, ddof=1) / np.sqrt(N_MODELS)
    pooled_fits: list[dict[str, Any]] = []
    pooled_full = np.empty(2, dtype=np.float64)
    per_model_full = np.empty((2, N_MODELS), dtype=np.float64)
    per_model_r2 = np.empty((2, N_MODELS), dtype=np.float64)

    for condition_index, condition in enumerate(CONDITIONS):
        pooled_fit, pooled_full[condition_index] = _fit_valid_curve(
            EXPECTED_OFFSETS_MS,
            means[condition_index],
            f"pooled {condition.key}",
        )
        pooled_fits.append(pooled_fit)
        for model_index in range(N_MODELS):
            fit, width = _fit_valid_curve(
                EXPECTED_OFFSETS_MS,
                p_fusion[condition_index, model_index],
                f"model{model_index:02d} {condition.key}",
            )
            per_model_full[condition_index, model_index] = width
            per_model_r2[condition_index, model_index] = float(fit["r_squared"])

    paired_full = per_model_full[1] - per_model_full[0]
    paired_half = paired_full / 2.0
    paired_sem = paired_full.std(ddof=1) / np.sqrt(N_MODELS)
    t90 = float(stats.t.ppf(0.95, df=N_MODELS - 1))
    paired_mean = float(paired_full.mean())
    paired_ci90 = np.asarray(
        [paired_mean - t90 * paired_sem, paired_mean + t90 * paired_sem]
    )
    difference = means[1] - means[0]
    rmse = float(np.sqrt(np.mean(difference**2)))
    reference_range = float(np.ptp(means[0]))
    _require(reference_range > 0.0, "Reference pooled curve has zero range")

    return {
        "means": means,
        "sems": sems,
        "pooled_fits": pooled_fits,
        "pooled_full_width_ms": pooled_full,
        "pooled_half_width_ms": pooled_full / 2.0,
        "pooled_full_width_drift_ms": float(pooled_full[1] - pooled_full[0]),
        "pooled_half_width_drift_ms": float((pooled_full[1] - pooled_full[0]) / 2.0),
        "per_model_full_width_ms": per_model_full,
        "per_model_half_width_ms": per_model_full / 2.0,
        "per_model_fit_r_squared": per_model_r2,
        "paired_full_width_drift_ms": paired_full,
        "paired_half_width_drift_ms": paired_half,
        "paired_mean_full_width_drift_ms": paired_mean,
        "paired_mean_half_width_drift_ms": float(paired_half.mean()),
        "paired_full_width_drift_ci90_ms": paired_ci90,
        "pooled_curve_pearson_r": float(np.corrcoef(means[0], means[1])[0, 1]),
        "pooled_curve_rmse": rmse,
        "pooled_curve_nrmse": rmse / reference_range,
        "pooled_curve_max_abs_delta": float(np.max(np.abs(difference))),
        "pooled_curve_mean_abs_delta": float(np.mean(np.abs(difference))),
        "pooled_fit_r_squared": np.asarray(
            [float(fit["r_squared"]) for fit in pooled_fits]
        ),
        "minimum_individual_fit_r_squared": float(per_model_r2.min()),
        "minimum_individual_fit_index": tuple(
            int(index) for index in np.unravel_index(per_model_r2.argmin(), per_model_r2.shape)
        ),
    }


def _validate_metrics(metrics: Mapping[str, Any]) -> None:
    """Assert agreement with the frozen independent validation."""

    _assert_close(
        "per-model full widths",
        metrics["per_model_full_width_ms"],
        EXPECTED_PER_MODEL_FULL_WIDTH_MS,
    )
    for key, expected in EXPECTED_METRICS.items():
        _assert_close(key, metrics[key], expected)
    _require(
        metrics["minimum_individual_fit_index"] == (1, 2),
        "Minimum individual R² is not M02 dt=.05/nsub200",
    )


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a compressed NPZ with fixed member order and ZIP metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.asanyarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=9)
    os.replace(temporary, path)


def _build_bundle(source_dir: Path, bundle_path: Path) -> None:
    """Verify the frozen 20-file source set and create the durable bundle."""

    source_dir = source_dir.resolve()
    _require(source_dir.is_dir(), f"Missing source directory: {source_dir}")
    manifest_entries, summary_hashes = _parse_verified_manifest(source_dir)

    expected_names = {
        _source_filename(model_index, condition)
        for condition in CONDITIONS
        for model_index in range(N_MODELS)
    }
    actual_paths = sorted(source_dir.glob("model??_dt*_nsub*.npz"))
    _require(len(actual_paths) == 20, f"Expected 20 source NPZ files, found {len(actual_paths)}")
    _require(
        {path.name for path in actual_paths} == expected_names,
        "Frozen source NPZ filename set differs from the expected 20 files",
    )

    p_fusion = np.empty((2, N_MODELS, N_SOAS), dtype=np.float64)
    fused_flags = np.empty((2, N_MODELS, N_SOAS, N_TRIALS), dtype=np.uint8)
    locations = np.empty((N_MODELS, N_SOAS, N_TRIALS), dtype=np.int64)
    location_hashes: list[str] = []
    source_names: list[str] = []
    source_hashes: list[str] = []

    for condition_index, condition in enumerate(CONDITIONS):
        for model_index in range(N_MODELS):
            path = source_dir / _source_filename(model_index, condition)
            expected_hash = manifest_entries.get(path.resolve())
            _require(expected_hash is not None, f"Source is absent from manifest: {path}")
            actual_hash = _sha256_file(path)
            _require(actual_hash == expected_hash, f"Source NPZ hash mismatch: {path}")

            verified = _verify_source_npz(path)
            p_fusion[condition_index, model_index] = verified["p_fusion"]
            fused_flags[condition_index, model_index] = verified["fused_flags"]
            if condition_index == 0:
                locations[model_index] = verified["locations"]
                expected_locations = np.random.default_rng(SEED_BASE + model_index).integers(
                    0,
                    180,
                    size=(N_SOAS, N_TRIALS),
                    dtype=np.int64,
                )
                _require(
                    np.array_equal(verified["locations"], expected_locations),
                    f"Seeded locations mismatch for model {model_index:02d}",
                )
                location_hashes.append(_sha256_array(verified["locations"]))
            else:
                _require(
                    np.array_equal(verified["locations"], locations[model_index]),
                    f"Paired location mismatch for model {model_index:02d}",
                )

            source_names.append(path.name)
            source_hashes.append(actual_hash)

    data_for_analysis = {"p_fusion": p_fusion}
    metrics = _analyse_arrays(data_for_analysis)
    _validate_metrics(metrics)

    arrays: dict[str, np.ndarray] = {
        "condition_keys": np.asarray([condition.key for condition in CONDITIONS]),
        "dt_ms": np.asarray([condition.dt_ms for condition in CONDITIONS]),
        "n_substeps": np.asarray([condition.n_substeps for condition in CONDITIONS]),
        "external_frame_ms": np.asarray([EXTERNAL_FRAME_MS]),
        "offsets_frames": EXPECTED_OFFSETS,
        "offsets_ms": EXPECTED_OFFSETS_MS,
        "model_indices": np.arange(N_MODELS, dtype=np.int64),
        "seeds": np.arange(SEED_BASE, SEED_BASE + N_MODELS, dtype=np.int64),
        "n_trials_per_soa": np.asarray([N_TRIALS], dtype=np.int64),
        "p_fusion": p_fusion,
        "fused_flags": fused_flags,
        "locations": locations,
        "locations_sha256": np.asarray(location_hashes),
        "pooled_mean_fusion": metrics["means"],
        "pooled_sem_fusion": metrics["sems"],
        "per_model_full_width_ms": metrics["per_model_full_width_ms"],
        "per_model_half_width_ms": metrics["per_model_half_width_ms"],
        "per_model_fit_r_squared": metrics["per_model_fit_r_squared"],
        "paired_full_width_drift_ms": metrics["paired_full_width_drift_ms"],
        "paired_half_width_drift_ms": metrics["paired_half_width_drift_ms"],
        "source_npz_names": np.asarray(source_names),
        "source_npz_sha256": np.asarray(source_hashes),
        "source_analysis_summary_sha256": np.asarray(
            [summary_hashes["analysis_summary_sha256"]]
        ),
        "source_paired_summary_sha256": np.asarray(
            [summary_hashes["paired_summary_sha256"]]
        ),
        "source_validator_manifest_sha256": np.asarray(
            [summary_hashes["validator_manifest_sha256"]]
        ),
        "source_recorded_dir": np.asarray([str(source_dir)]),
        "bundle_build_command": np.asarray([_canonical_command()]),
    }
    _write_deterministic_npz(bundle_path, arrays)


def _load_and_validate_bundle(bundle_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load the durable bundle and repeat all data/metric assertions."""

    _require(bundle_path.is_file(), f"Missing durable bundle: {bundle_path}")
    with np.load(bundle_path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}

    required = {
        "condition_keys",
        "dt_ms",
        "n_substeps",
        "external_frame_ms",
        "offsets_frames",
        "offsets_ms",
        "model_indices",
        "seeds",
        "n_trials_per_soa",
        "p_fusion",
        "fused_flags",
        "locations",
        "locations_sha256",
        "pooled_mean_fusion",
        "pooled_sem_fusion",
        "per_model_full_width_ms",
        "per_model_half_width_ms",
        "per_model_fit_r_squared",
        "paired_full_width_drift_ms",
        "paired_half_width_drift_ms",
        "source_npz_names",
        "source_npz_sha256",
        "source_analysis_summary_sha256",
        "source_paired_summary_sha256",
        "source_validator_manifest_sha256",
        "source_recorded_dir",
        "bundle_build_command",
    }
    _require(set(data) == required, "Durable bundle key set mismatch")
    _require(
        np.array_equal(data["condition_keys"], [condition.key for condition in CONDITIONS]),
        "Bundle condition order mismatch",
    )
    _assert_close("bundle dt_ms", data["dt_ms"], [0.10, 0.05], atol=0.0)
    _require(np.array_equal(data["n_substeps"], [100, 200]), "Bundle substep mismatch")
    _assert_close("external frame", data["external_frame_ms"], [10.0], atol=0.0)
    _require(np.array_equal(data["offsets_frames"], EXPECTED_OFFSETS), "Bundle offsets mismatch")
    _require(np.array_equal(data["offsets_ms"], EXPECTED_OFFSETS_MS), "Bundle SOAs mismatch")
    _require(
        data["p_fusion"].shape == (2, N_MODELS, N_SOAS),
        "Bundle p_fusion shape mismatch",
    )
    _require(
        data["fused_flags"].shape == (2, N_MODELS, N_SOAS, N_TRIALS),
        "Bundle fused_flags shape mismatch",
    )
    _require(
        np.array_equal(data["p_fusion"], data["fused_flags"].mean(axis=3)),
        "Bundled p_fusion != fused_flags.mean(1 trial axis)",
    )
    _require(
        np.logical_and(data["p_fusion"] >= 0.0, data["p_fusion"] <= 1.0).all(),
        "Bundled p_fusion outside [0,1]",
    )
    _require(np.isfinite(data["p_fusion"]).all(), "Bundled p_fusion is non-finite")

    for model_index in range(N_MODELS):
        expected_locations = np.random.default_rng(SEED_BASE + model_index).integers(
            0,
            180,
            size=(N_SOAS, N_TRIALS),
            dtype=np.int64,
        )
        _require(
            np.array_equal(data["locations"][model_index], expected_locations),
            f"Bundled seeded locations mismatch for model {model_index:02d}",
        )
        _require(
            data["locations_sha256"][model_index]
            == _sha256_array(data["locations"][model_index]),
            f"Bundled location hash mismatch for model {model_index:02d}",
        )

    expected_names = [
        _source_filename(model_index, condition)
        for condition in CONDITIONS
        for model_index in range(N_MODELS)
    ]
    _require(
        np.array_equal(data["source_npz_names"], expected_names),
        "Bundled source filename order mismatch",
    )
    for digest in data["source_npz_sha256"]:
        _require(
            len(str(digest)) == 64 and all(char in "0123456789abcdef" for char in str(digest)),
            "Malformed bundled source SHA-256",
        )
    _require(
        data["source_analysis_summary_sha256"].item() == EXPECTED_ANALYSIS_SHA256,
        "Bundled analysis summary hash mismatch",
    )
    _require(
        data["source_paired_summary_sha256"].item() == EXPECTED_PAIRED_SUMMARY_SHA256,
        "Bundled paired summary hash mismatch",
    )
    _require(
        data["source_validator_manifest_sha256"].item() == EXPECTED_MANIFEST_SHA256,
        "Bundled validator manifest hash mismatch",
    )

    metrics = _analyse_arrays(data)
    _validate_metrics(metrics)
    for key in (
        "pooled_mean_fusion",
        "pooled_sem_fusion",
        "per_model_full_width_ms",
        "per_model_half_width_ms",
        "per_model_fit_r_squared",
        "paired_full_width_drift_ms",
        "paired_half_width_drift_ms",
    ):
        metric_key = {
            "pooled_mean_fusion": "means",
            "pooled_sem_fusion": "sems",
        }.get(key, key)
        _assert_close(f"bundled {key}", data[key], metrics[metric_key])
    return data, metrics


def _configure_figure_style() -> None:
    """Set deterministic, searchable vector-text defaults."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.7,
            "svg.fonttype": "none",
            "svg.hashsalt": "fsts10-tbw-dt-invariance-validated",
            "pdf.fonttype": 42,
            "pdf.compression": 9,
        }
    )


def _render_figure(
    output_dir: Path,
    metrics: Mapping[str, Any],
) -> dict[str, Path]:
    """Render the validated 8×5-inch SVG/PDF/300-dpi PNG figure."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        extension: output_dir / f"{OUTPUT_STEM}.{extension}"
        for extension in ("svg", "pdf", "png")
    }
    historical = {
        output_dir / "TBW_stepsize_sensitivity.svg",
        output_dir / "TBW_stepsize_sensitivity.png",
    }
    _require(not historical.intersection(outputs.values()), "Historical figure collision")

    _configure_figure_style()
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    figure.subplots_adjust(left=0.105, right=0.975, bottom=0.175, top=0.80)

    for condition_index, condition in enumerate(CONDITIONS):
        mean = metrics["means"][condition_index]
        sem = metrics["sems"][condition_index]
        lower_error = mean - np.clip(mean - sem, 0.0, 1.0)
        upper_error = np.clip(mean + sem, 0.0, 1.0) - mean
        marker_face = condition.color if condition.filled else "white"
        axis.errorbar(
            EXPECTED_OFFSETS_MS,
            mean,
            yerr=np.vstack([lower_error, upper_error]),
            fmt=condition.marker,
            linestyle="none",
            markersize=3.5 if condition.filled else 3.8,
            markerfacecolor=marker_face,
            markeredgecolor=condition.color,
            markeredgewidth=0.85,
            ecolor=condition.color,
            elinewidth=0.65,
            capsize=1.5,
            alpha=0.92,
            zorder=4 if condition.filled else 5,
        )
        fit = metrics["pooled_fits"][condition_index]
        axis.plot(
            fit["xs"],
            fit["ys"],
            color=condition.color,
            linestyle=condition.linestyle,
            linewidth=2.0,
            zorder=3,
        )

    axis.axhline(0.5, color="#666666", linestyle=":", linewidth=1.0, zorder=1)
    axis.set_xlim(-510.0, 510.0)
    axis.set_ylim(-0.025, 1.04)
    axis.set_xticks([-500, -250, 0, 250, 500])
    axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.set_xlabel("Audiovisual SOA (ms)", labelpad=8)
    axis.set_ylabel("P(fusion)")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=4.0, width=0.8)

    legend_handles = []
    for condition_index, condition in enumerate(CONDITIONS):
        full_width = metrics["pooled_full_width_ms"][condition_index]
        half_width = metrics["pooled_half_width_ms"][condition_index]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=condition.color,
                linestyle=condition.linestyle,
                linewidth=2.0,
                marker=condition.marker,
                markersize=4.2,
                markerfacecolor=condition.color if condition.filled else "white",
                markeredgecolor=condition.color,
                markeredgewidth=0.9,
                label=(
                    f"dt={condition.dt_ms:.2f} ms / {condition.n_substeps} substeps\n"
                    f"Full span {full_width:.3f} ms; half-width {half_width:.3f} ms"
                ),
            )
        )
    legend = axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        columnspacing=1.6,
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        framealpha=0.96,
        handlelength=3.2,
        borderpad=0.65,
        labelspacing=0.75,
    )

    r_text = f"{metrics['pooled_curve_pearson_r']:.4f}".removeprefix("0")
    nrmse_text = f"{metrics['pooled_curve_nrmse']:.4f}".removeprefix("0")
    max_delta_text = f"{metrics['pooled_curve_max_abs_delta']:.3f}".removeprefix("0")
    callout = (
        "Full span = separation of 50% crossings\n"
        f"Pooled Δfull {metrics['pooled_full_width_drift_ms']:+.3f} ms; "
        f"Δhalf {metrics['pooled_half_width_drift_ms']:+.3f} ms\n"
        f"Paired mean Δfull {metrics['paired_mean_full_width_drift_ms']:+.2f} ms\n"
        f"90% CI [{metrics['paired_full_width_drift_ci90_ms'][0]:.2f}, "
        f"{metrics['paired_full_width_drift_ci90_ms'][1]:.2f}] ms\n"
        f"r={r_text}; NRMSE={nrmse_text}\n"
        f"max|ΔP|={max_delta_text}"
    )
    callout_artist = axis.text(
        0.985,
        0.965,
        callout,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#D0D0D0",
            "linewidth": 0.7,
            "alpha": 0.96,
        },
        zorder=10,
    )

    figure.suptitle(
        "Temporal binding window across integration step sizes",
        x=0.5,
        y=0.965,
        fontsize=13.0,
        fontweight="semibold",
    )
    subtitle_artist = figure.text(
        0.5,
        0.905,
        "10 checkpoints • 51 SOAs • 50 trials/SOA • paired inputs • 10-ms external frame",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#333333",
    )
    footer_artist = figure.text(
        0.5,
        0.055,
        (
            "Negative SOA: visual leads auditory; positive SOA: auditory leads visual.  "
            "Observed points ±1 SEM across checkpoints; error bars display-clipped to [0,1]."
        ),
        ha="center",
        va="center",
        fontsize=7.4,
        color="#444444",
    )

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_bbox = axis.get_window_extent(renderer)
    legend_bbox = legend.get_frame().get_window_extent(renderer)
    subtitle_bbox = subtitle_artist.get_window_extent(renderer)
    footer_bbox = footer_artist.get_window_extent(renderer)
    callout_bbox = callout_artist.get_window_extent(renderer)

    transition_indices = np.flatnonzero(
        (np.max(metrics["sems"], axis=0) > 0.0)
        | (np.abs(metrics["means"][1] - metrics["means"][0]) > 0.0)
    )
    _require(
        np.array_equal(EXPECTED_OFFSETS_MS[transition_indices], [-140.0, -120.0, 100.0]),
        "Unexpected marker/SEM transition geometry",
    )
    transition_points: list[np.ndarray] = []
    for condition_index in range(len(CONDITIONS)):
        mean = metrics["means"][condition_index]
        sem = metrics["sems"][condition_index]
        for index in transition_indices:
            low = float(np.clip(mean[index] - sem[index], 0.0, 1.0))
            high = float(np.clip(mean[index] + sem[index], 0.0, 1.0))
            transition_points.extend(
                [
                    axis.transData.transform((EXPECTED_OFFSETS_MS[index], low)),
                    axis.transData.transform((EXPECTED_OFFSETS_MS[index], high)),
                ]
            )
    transition_array = np.asarray(transition_points)
    transition_bbox = Bbox.from_extents(
        float(transition_array[:, 0].min() - 6.0),
        float(transition_array[:, 1].min() - 6.0),
        float(transition_array[:, 0].max() + 6.0),
        float(transition_array[:, 1].max() + 6.0),
    )

    _require(
        legend_bbox.y0 >= axes_bbox.y1,
        "Legend frame intrudes into the plotting axes",
    )
    _require(
        not legend_bbox.overlaps(transition_bbox),
        "Legend frame overlaps marker/SEM transition geometry",
    )
    _require(
        subtitle_bbox.y0 >= legend_bbox.y1,
        "Subtitle does not clear the external legend",
    )
    _require(
        not callout_bbox.overlaps(transition_bbox),
        "Metric callout overlaps marker/SEM transition geometry",
    )

    pixels_to_points = 72.0 / figure.dpi
    footer_points = np.asarray(
        [footer_bbox.x0, footer_bbox.y0, footer_bbox.x1, footer_bbox.y1]
    ) * pixels_to_points
    _require(
        footer_points[0] >= 0.0
        and footer_points[1] >= 0.0
        and footer_points[2] <= 576.0
        and footer_points[3] <= 360.0,
        f"Footer leaves the 576×360 pt canvas: {footer_points.tolist()}",
    )

    title = "Temporal binding window across integration step sizes"
    description = (
        "Paired control TBW curves for dt=.10/100 and dt=.05/200; "
        "10 checkpoints, 51 SOAs, 50 trials per SOA, 10-ms external frames."
    )
    figure.savefig(
        outputs["svg"],
        format="svg",
        metadata={"Title": title, "Description": description, "Creator": "Matplotlib", "Date": None},
    )
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
        dpi=300,
        metadata={"Title": title, "Description": description, "Software": "Matplotlib"},
    )
    plt.close(figure)
    return outputs


def _png_dimensions(path: Path) -> list[int]:
    """Read PNG dimensions from its fixed-format IHDR header."""

    header = path.read_bytes()[:24]
    _require(header[:8] == b"\x89PNG\r\n\x1a\n", f"Invalid PNG signature: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return [int(width), int(height)]


def _git_record() -> dict[str, Any]:
    """Capture repository identity and exact working-tree status."""

    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()

    return {
        "head": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "status_porcelain": run("git", "status", "--porcelain", "--untracked-files=all").splitlines(),
    }


def _metric_record(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Select exact scalar/list metrics for JSON provenance."""

    return {
        "pooled_full_width_ms": metrics["pooled_full_width_ms"].tolist(),
        "pooled_half_width_ms": metrics["pooled_half_width_ms"].tolist(),
        "pooled_full_width_drift_ms": metrics["pooled_full_width_drift_ms"],
        "pooled_half_width_drift_ms": metrics["pooled_half_width_drift_ms"],
        "per_model_full_width_ms": metrics["per_model_full_width_ms"].tolist(),
        "paired_full_width_drift_ms": metrics["paired_full_width_drift_ms"].tolist(),
        "paired_mean_full_width_drift_ms": metrics["paired_mean_full_width_drift_ms"],
        "paired_mean_half_width_drift_ms": metrics["paired_mean_half_width_drift_ms"],
        "paired_full_width_drift_ci90_ms": metrics[
            "paired_full_width_drift_ci90_ms"
        ].tolist(),
        "pooled_curve_pearson_r": metrics["pooled_curve_pearson_r"],
        "pooled_curve_rmse": metrics["pooled_curve_rmse"],
        "pooled_curve_nrmse": metrics["pooled_curve_nrmse"],
        "pooled_curve_max_abs_delta": metrics["pooled_curve_max_abs_delta"],
        "pooled_curve_mean_abs_delta": metrics["pooled_curve_mean_abs_delta"],
        "pooled_fit_r_squared": metrics["pooled_fit_r_squared"].tolist(),
        "minimum_individual_fit_r_squared": metrics["minimum_individual_fit_r_squared"],
        "minimum_individual_fit": "M02 dt=.05 ms / 200 substeps",
    }


def _canonical_command() -> str:
    """Record the environment-prefixed invocation used for this render."""

    return shlex.join(
        [
            "env",
            f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}",
            f"MPLBACKEND={os.environ['MPLBACKEND']}",
            "python",
            *sys.argv,
        ]
    )


def _write_provenance(
    path: Path,
    bundle_path: Path,
    outputs: Mapping[str, Path],
    data: Mapping[str, np.ndarray],
    metrics: Mapping[str, Any],
    source_dir: Path | None,
) -> None:
    """Write a timestamp-free JSON record after every output is complete."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")  # Make git-status capture stable.

    exports = {
        extension: {
            "path": str(output.relative_to(ROOT)),
            "size_bytes": output.stat().st_size,
            "sha256": _sha256_file(output),
        }
        for extension, output in outputs.items()
    }
    exports["png"]["pixel_dimensions"] = _png_dimensions(outputs["png"])

    bundle_schema = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(data.items())
    }
    source_hashes = {
        str(name): str(digest)
        for name, digest in zip(
            data["source_npz_names"],
            data["source_npz_sha256"],
            strict=True,
        )
    }
    provenance = {
        "schema_version": 1,
        "figure": {
            "title": "Temporal binding window across integration step sizes",
            "size_inches": [8.0, 5.0],
            "png_dpi": 300,
            "full_width_definition": "separation between fitted P(fusion)=0.5 crossings",
            "half_width_definition": "full width divided by two",
            "sem_display": "mean ±1 SEM, clipped to [0,1] for error-bar display only",
            "fit_aggregation": (
                "fit_psychometric_curve_improved applied to each pooled mean curve; "
                "individual fits are never averaged"
            ),
        },
        "sample": {
            "condition": "control, gNMDA=1.30, plasticity disabled",
            "checkpoints": N_MODELS,
            "soas": N_SOAS,
            "offsets_ms": EXPECTED_OFFSETS_MS.tolist(),
            "trials_per_soa": N_TRIALS,
            "total_condition_trials": N_MODELS * N_SOAS * N_TRIALS * 2,
            "seed_rule": "12345 + model_index, reset per paired condition",
            "paired_inputs": "identical 51×50 location matrix for both dt values per checkpoint",
            "conditions": [
                {
                    "dt_ms": condition.dt_ms,
                    "n_substeps": condition.n_substeps,
                    "external_frame_ms": condition.dt_ms * condition.n_substeps,
                }
                for condition in CONDITIONS
            ],
            "frames_per_trial": 60,
            "stimulus_duration_frames": 5,
            "stimulus_intensity": 1.0,
            "noise_std": 0.0,
            "fusion_classifier": {
                "sigma_frames": 2.0,
                "valley_threshold": 0.4,
                "min_peak_height": 0.2,
                "min_peak_separation_frames": 3,
                "min_total_spikes": 10.0,
            },
        },
        "aggregation": {
            "mean": "arithmetic mean of 10 checkpoint-level p_fusion values at each SOA",
            "sem": "sample SD across checkpoints (ddof=1) divided by sqrt(10)",
            "sem_inputs": "p_fusion[condition, checkpoint, SOA] stored in the bundle",
        },
        "metrics": _metric_record(metrics),
        "source": {
            "recorded_source_dir": data["source_recorded_dir"].item(),
            "source_dir_is_required_for_replot": False,
            "bundle_build_command": data["bundle_build_command"].item(),
            "validator_manifest_sha256": data[
                "source_validator_manifest_sha256"
            ].item(),
            "analysis_summary_sha256": data["source_analysis_summary_sha256"].item(),
            "paired_summary_sha256": data["source_paired_summary_sha256"].item(),
            "npz_sha256": source_hashes,
        },
        "artifacts": {
            "bundle": {
                "path": str(bundle_path.relative_to(ROOT)),
                "size_bytes": bundle_path.stat().st_size,
                "sha256": _sha256_file(bundle_path),
                "schema": bundle_schema,
            },
            "replot_script_sha256": _sha256_file(Path(__file__).resolve()),
            "fit_source_sha256": _sha256_file(ROOT / "TBW_test.py"),
            "exports": exports,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
            "renderer": "CPU-only Matplotlib Agg; no CUDA kernels invoked",
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "repository": _git_record(),
        "command": _canonical_command(),
    }
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="One-time frozen paired_sweep directory used to verify/build the bundle.",
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _verify_environment()
    bundle_path = args.bundle.resolve()
    if args.source_dir is not None:
        _build_bundle(args.source_dir, bundle_path)

    data, metrics = _load_and_validate_bundle(bundle_path)
    outputs = _render_figure(args.output_dir.resolve(), metrics)
    _write_provenance(
        args.provenance.resolve(),
        bundle_path,
        outputs,
        data,
        metrics,
        args.source_dir,
    )

    print(f"Bundle: {bundle_path}")
    for extension, output in outputs.items():
        print(f"{extension.upper()}: {output} ({output.stat().st_size} bytes)")
    print(f"Provenance: {args.provenance.resolve()}")
    print(
        "Metrics: "
        f"full={metrics['pooled_full_width_ms'][0]:.6f}/"
        f"{metrics['pooled_full_width_ms'][1]:.6f} ms, "
        f"Δfull={metrics['pooled_full_width_drift_ms']:+.6f} ms, "
        f"paired mean Δfull={metrics['paired_mean_full_width_drift_ms']:+.6f} ms, "
        f"r={metrics['pooled_curve_pearson_r']:.6f}, "
        f"NRMSE={metrics['pooled_curve_nrmse']:.6f}"
    )


if __name__ == "__main__":
    main()
