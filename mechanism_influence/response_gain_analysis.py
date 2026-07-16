#!/usr/bin/env python3
"""Deterministic checkpoint-level analysis of Gate 4/5 perturbations.

The checkpoint is the independent unit.  Every effect is the paired difference
``perturbation - shipped`` from the same checkpoint and common random stream.
Exact row-wise sign flips use the same sign for an entire endpoint family, with
max-|T| adjusted p values and centered-residual simultaneous confidence bands.

Normalized MEI/CRE effects remain descriptive when a perturbation collapses the
absolute unisensory or AV response below ten percent of its paired shipped
response.  Raw response ratios and every exclusion reason are retained.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mechanism_influence import response_gain_perturbation as measure


ANALYSIS_SCHEMA = "fsts-response-gain-analysis-v1"
ALPHA = 0.05
MIN_PAIRED_RESPONSE_FRACTION = 0.10
TIE_EPS_MULTIPLIER = 64.0


def _finite_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)


def _studentized(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column means, standard errors, and signed t statistics."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("studentized matrix must be finite, 2-D, and have at least two rows")
    means = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
    t_values = np.zeros_like(means)
    nonzero_se = se > 0
    t_values[nonzero_se] = means[nonzero_se] / se[nonzero_se]
    zero_se_effect = (~nonzero_se) & (means != 0)
    t_values[zero_se_effect] = np.copysign(np.inf, means[zero_se_effect])
    return means, se, t_values


def _conservative_ge(null_value: float, observed: float) -> bool:
    if math.isinf(observed):
        return math.isinf(null_value) and null_value >= observed
    tolerance = TIE_EPS_MULTIPLIER * np.finfo(float).eps * max(1.0, abs(observed))
    return null_value >= observed - tolerance


def exact_row_sign_flip_max_t(matrix: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    """Exact common-row two-sided max-|T| inference.

    Args:
        matrix: Paired checkpoint differences with shape ``(n_checkpoints,
            n_endpoints)``.
        labels: Stable endpoint labels in matrix column order.

    Returns:
        Exact adjusted-p numerators/denominator and approximate simultaneous
        centered-residual bands.  No Monte Carlo randomness is used.
    """

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(labels):
        raise ValueError("matrix/label shape mismatch")
    if len(set(labels)) != len(labels):
        raise ValueError("endpoint labels must be unique")
    if not np.all(np.isfinite(values)):
        raise ValueError("exact family contains nonfinite values")
    n_rows = values.shape[0]
    if n_rows < 2 or n_rows > 20:
        raise ValueError("exact sign flips require 2..20 checkpoint rows")
    means, se, observed_t = _studentized(values)
    patterns = np.asarray(list(itertools.product((-1.0, 1.0), repeat=n_rows)), dtype=float)
    null_max = np.empty(patterns.shape[0], dtype=float)
    centered_max = np.empty(patterns.shape[0], dtype=float)
    residuals = values - means[None, :]
    for index, signs in enumerate(patterns):
        _, _, null_t = _studentized(values * signs[:, None])
        null_max[index] = np.max(np.abs(null_t))
        _, _, centered_t = _studentized(residuals * signs[:, None])
        centered_max[index] = np.max(np.abs(centered_t))
    denominator = int(patterns.shape[0])
    critical_index = int(math.ceil((1.0 - ALPHA) * denominator) - 1)
    critical = float(np.sort(centered_max)[critical_index])
    achieved_coverage = float(np.mean(centered_max <= critical))
    endpoints = []
    for column, label in enumerate(labels):
        observed_abs = abs(float(observed_t[column]))
        numerator = int(sum(_conservative_ge(float(value), observed_abs) for value in null_max))
        lower = float(means[column] - critical * se[column])
        upper = float(means[column] + critical * se[column])
        endpoints.append({
            "label": str(label),
            "mean_delta": float(means[column]),
            "se": float(se[column]),
            "t": float(observed_t[column]),
            "adjusted_p_numerator": numerator,
            "adjusted_p_denominator": denominator,
            "adjusted_p": float(numerator / denominator),
            "simultaneous_lower": lower,
            "simultaneous_upper": upper,
            "familywise_significant": bool(numerator / denominator < ALPHA),
        })
    return {
        "method": "exact_common_checkpoint_row_sign_flip_max_abs_t",
        "n_checkpoints": int(n_rows),
        "n_endpoints": int(values.shape[1]),
        "n_sign_patterns": denominator,
        "tail": "two_sided",
        "alpha": ALPHA,
        "plus_one_correction": False,
        "simultaneous_band_method": "centered_residual_row_sign_flip_max_abs_t",
        "simultaneous_critical": critical,
        "simultaneous_achieved_coverage": achieved_coverage,
        "endpoints": endpoints,
    }


def _validate_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != measure.SCHEMA_VERSION:
        raise ValueError(f"schema mismatch: {path}")
    if record.get("protocol_id") != measure.PROTOCOL_ID:
        raise ValueError(f"protocol mismatch: {path}")
    payload = copy.deepcopy(record)
    expected_hash = payload["integrity"].pop("scientific_payload_sha256")
    if measure.canonical_sha256(payload) != expected_hash:
        raise ValueError(f"payload SHA-256 mismatch: {path}")
    if record["device"]["physical_uuid"] != measure.RTX5090_UUID:
        raise ValueError(f"non-RTX5090 record: {path}")
    if measure.RTX5090_NAME_TOKEN not in record["device"]["physical_name"]:
        raise ValueError(f"GPU name mismatch: {path}")
    if record["integrity"]["readouts_unchanged"] is not True:
        raise ValueError(f"readout firewall failed: {path}")
    if record["integrity"]["plasticity_enabled"] is not False:
        raise ValueError(f"plasticity enabled: {path}")
    for gate in ("gate4", "gate5"):
        if record[gate]["state_bit_identical"] is not True:
            raise ValueError(f"{gate} state firewall failed: {path}")
        if record[gate]["effective"] != record["setting"]["requested"]:
            raise ValueError(f"{gate} override mismatch: {path}")
        if record[gate]["effective_after"] != record[gate]["effective"]:
            raise ValueError(f"{gate} override drift: {path}")
    expected_path = measure.checkpoint_path(int(record["checkpoint"]["seed"]))
    if measure.sha256_file(expected_path) != record["checkpoint"]["sha256"]:
        raise ValueError(f"checkpoint SHA-256 mismatch: {path}")
    return record


def load_run(run_dir: Path, require_primary_complete: bool = True) -> dict[str, dict[int, dict[str, Any]]]:
    """Load, validate, and index atomic cell records."""

    indexed: dict[str, dict[int, dict[str, Any]]] = {}
    for path in sorted((run_dir / "raw").glob("seed*/*.json")):
        record = _validate_record(path)
        setting = str(record["setting"]["id"])
        seed = int(record["checkpoint"]["seed"])
        if setting not in measure.ALL_REGISTRY or seed not in measure.CHECKPOINT_SEEDS:
            raise ValueError(f"unregistered cell in {path}")
        if seed in indexed.setdefault(setting, {}):
            raise ValueError(f"duplicate cell {setting}/seed{seed}")
        indexed[setting][seed] = record
    required_settings = measure.PRIMARY_REGISTRY if require_primary_complete else {"shipped": None}
    for setting in required_settings:
        got = tuple(sorted(indexed.get(setting, {})))
        if got != measure.CHECKPOINT_SEEDS:
            raise ValueError(f"incomplete {setting}: got checkpoints {got}")
    return indexed


def _paired_qc(perturbation: Mapping[str, Any], shipped: Mapping[str, Any], gate: str) -> dict[str, Any]:
    response_keys = ("R_A", "R_V", "R_AV")
    pert_response = np.concatenate([_finite_array(perturbation[gate][key]) for key in response_keys])
    base_response = np.concatenate([_finite_array(shipped[gate][key]) for key in response_keys])
    with np.errstate(divide="ignore", invalid="ignore"):
        fractions = pert_response / base_response
    finite = bool(np.all(np.isfinite(pert_response)) and np.all(np.isfinite(base_response)))
    base_positive = bool(np.all(base_response > 0))
    pert_positive = bool(np.all(pert_response > 0))
    min_fraction = float(np.nanmin(fractions)) if np.any(np.isfinite(fractions)) else float("nan")
    collapse_free = bool(
        finite and base_positive and pert_positive
        and np.all(fractions >= MIN_PAIRED_RESPONSE_FRACTION)
    )
    endpoint_key = "mei" if gate == "gate4" else "cre_percent_ratio_of_means"
    endpoint = _finite_array(perturbation[gate][endpoint_key])
    base_endpoint = _finite_array(shipped[gate][endpoint_key])
    normalized_finite = bool(np.all(np.isfinite(endpoint)) and np.all(np.isfinite(base_endpoint)))
    eligible = bool(collapse_free and normalized_finite)
    reasons = []
    if not finite:
        reasons.append("nonfinite_absolute_response")
    if not base_positive:
        reasons.append("nonpositive_shipped_response")
    if not pert_positive:
        reasons.append("nonpositive_perturbation_response")
    if finite and base_positive and pert_positive and not collapse_free:
        reasons.append("response_below_10_percent_of_paired_shipped")
    if not normalized_finite:
        reasons.append("nonfinite_normalized_endpoint")
    return {
        "eligible_for_normalized_inference": eligible,
        "collapse_free": collapse_free,
        "minimum_absolute_response_fraction_vs_shipped": min_fraction,
        "minimum_required_fraction": MIN_PAIRED_RESPONSE_FRACTION,
        "normalized_endpoint_finite": normalized_finite,
        "exclusion_reasons": reasons,
    }


def _descriptive_setting(
    setting: str, records: Mapping[int, Mapping[str, Any]], shipped: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    seeds = sorted(records)
    gate4_delta = []
    gate5_delta = []
    qc4 = []
    qc5 = []
    gate4_summary_names = ("mei_vs_log_intensity_slope", "low_minus_high_mei")
    gate5_summary_names = (
        "peak_cre", "zero_cross_deg", "center_hwhm_deg", "surround_min_cre",
        "surround_min_disparity_deg", "far_floor_cre",
    )
    summary4 = {name: [] for name in gate4_summary_names}
    summary5 = {name: [] for name in gate5_summary_names}
    for seed in seeds:
        record = records[seed]
        anchor = shipped[seed]
        gate4_delta.append(_finite_array(record["gate4"]["mei"]) - _finite_array(anchor["gate4"]["mei"]))
        gate5_delta.append(
            _finite_array(record["gate5"]["cre_percent_ratio_of_means"])
            - _finite_array(anchor["gate5"]["cre_percent_ratio_of_means"])
        )
        qc4.append(_paired_qc(record, anchor, "gate4"))
        qc5.append(_paired_qc(record, anchor, "gate5"))
        for name in gate4_summary_names:
            left = record["gate4"]["summary"][name]
            right = anchor["gate4"]["summary"][name]
            summary4[name].append(np.nan if left is None or right is None else float(left) - float(right))
        for name in gate5_summary_names:
            left = record["gate5"]["summary"][name]
            right = anchor["gate5"]["summary"][name]
            summary5[name].append(np.nan if left is None or right is None else float(left) - float(right))
    delta4 = np.asarray(gate4_delta, dtype=float)
    delta5 = np.asarray(gate5_delta, dtype=float)
    return {
        "setting": setting,
        "factor": records[seeds[0]]["setting"]["factor"],
        "checkpoint_seeds": seeds,
        "gate4": {
            "intensities": records[seeds[0]]["gate4"]["intensities"],
            "delta_mei_by_checkpoint": delta4.tolist(),
            "mean_delta_mei": np.nanmean(delta4, axis=0).tolist(),
            "paired_qc": qc4,
            "all_checkpoints_eligible": bool(all(row["eligible_for_normalized_inference"] for row in qc4)),
            "summary_deltas": {name: list(values) for name, values in summary4.items()},
        },
        "gate5": {
            "disparities_deg": records[seeds[0]]["gate5"]["disparities_deg"],
            "delta_cre_by_checkpoint": delta5.tolist(),
            "mean_delta_cre": np.nanmean(delta5, axis=0).tolist(),
            "paired_qc": qc5,
            "all_checkpoints_eligible": bool(all(row["eligible_for_normalized_inference"] for row in qc5)),
            "summary_deltas": {name: list(values) for name, values in summary5.items()},
        },
    }


def _curve_family(descriptions: Mapping[str, Mapping[str, Any]], gate: str) -> dict[str, Any] | None:
    values_key = "delta_mei_by_checkpoint" if gate == "gate4" else "delta_cre_by_checkpoint"
    coordinates_key = "intensities" if gate == "gate4" else "disparities_deg"
    coordinate_label = "intensity" if gate == "gate4" else "disparity_deg"
    columns: list[np.ndarray] = []
    labels: list[str] = []
    included = []
    excluded: dict[str, str] = {}
    for setting in sorted(descriptions):
        description = descriptions[setting]
        if not description[gate]["all_checkpoints_eligible"]:
            excluded[setting] = "absolute-response collapse/denominator QC"
            continue
        matrix = np.asarray(description[gate][values_key], dtype=float)
        if not np.all(np.isfinite(matrix)):
            excluded[setting] = "nonfinite paired normalized endpoint"
            continue
        coordinates = description[gate][coordinates_key]
        for index, coordinate in enumerate(coordinates):
            columns.append(matrix[:, index])
            labels.append(f"{setting}|{coordinate_label}={coordinate:g}")
        included.append(setting)
    if not columns:
        return None
    matrix = np.column_stack(columns)
    family = exact_row_sign_flip_max_t(matrix, labels)
    family["included_settings"] = included
    family["excluded_settings"] = excluded
    return family


def _summary_family(descriptions: Mapping[str, Mapping[str, Any]], gate: str) -> dict[str, Any] | None:
    columns: list[np.ndarray] = []
    labels: list[str] = []
    excluded: dict[str, list[str]] = {}
    for setting in sorted(descriptions):
        description = descriptions[setting]
        if not description[gate]["all_checkpoints_eligible"]:
            excluded.setdefault(setting, []).append("absolute-response collapse/denominator QC")
            continue
        for metric, values in description[gate]["summary_deltas"].items():
            vector = np.asarray(values, dtype=float)
            if not np.all(np.isfinite(vector)):
                excluded.setdefault(setting, []).append(f"{metric}: nonfinite fit/summary")
                continue
            columns.append(vector)
            labels.append(f"{setting}|{metric}")
    if not columns:
        return None
    family = exact_row_sign_flip_max_t(np.column_stack(columns), labels)
    family["excluded_endpoints"] = excluded
    return family


def analyze(run_dir: Path, require_primary_complete: bool = True) -> dict[str, Any]:
    indexed = load_run(run_dir, require_primary_complete=require_primary_complete)
    shipped = indexed["shipped"]
    descriptions = {
        setting: _descriptive_setting(setting, records, shipped)
        for setting, records in sorted(indexed.items())
        if setting != "shipped" and tuple(sorted(records)) == measure.CHECKPOINT_SEEDS
    }
    families = {
        "gate4_delta_mei_curve": _curve_family(descriptions, "gate4"),
        "gate4_summary": _summary_family(descriptions, "gate4"),
        "gate5_delta_cre_curve": _curve_family(descriptions, "gate5"),
        "gate5_summary": _summary_family(descriptions, "gate5"),
    }
    significant_adaptation = False
    for family in families.values():
        if family is None:
            continue
        significant_adaptation |= any(
            endpoint["label"].startswith("adaptation_prior|")
            and endpoint["familywise_significant"]
            for endpoint in family["endpoints"]
        )
    adaptation_valid = bool(
        "adaptation_prior" in descriptions
        and descriptions["adaptation_prior"]["gate4"]["all_checkpoints_eligible"]
        and descriptions["adaptation_prior"]["gate5"]["all_checkpoints_eligible"]
    )
    raw_files = sorted((run_dir / "raw").glob("seed*/*.json"))
    result: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA,
        "protocol_id": measure.PROTOCOL_ID,
        "delta_definition": "perturbation_minus_shipped_paired_by_checkpoint",
        "independent_unit": "trained_checkpoint",
        "checkpoint_seeds": list(measure.CHECKPOINT_SEEDS),
        "collapse_qc": {
            "minimum_response_fraction_vs_paired_shipped": MIN_PAIRED_RESPONSE_FRACTION,
            "meaning": "normalized MEI/CRE inference excluded if any absolute A/V/AV response falls below this fraction",
        },
        "settings": descriptions,
        "families": families,
        "triggered_scalar_attribution": {
            "compound_adaptation_valid": adaptation_valid,
            "compound_adaptation_familywise_effect": significant_adaptation,
            "required": bool(adaptation_valid and significant_adaptation),
            "cells": list(measure.TRIGGERED_REGISTRY),
            "already_complete": bool(
                all(
                    tuple(sorted(indexed.get(setting, {}))) == measure.CHECKPOINT_SEEDS
                    for setting in measure.TRIGGERED_REGISTRY
                )
            ),
        },
        "provenance": {
            "analyzer_path": str(Path(__file__).resolve().relative_to(measure.ROOT)),
            "analyzer_sha256": measure.sha256_file(Path(__file__).resolve()),
            "raw_file_count": len(raw_files),
            "raw_files": [
                {"path": str(path.relative_to(run_dir)), "sha256": measure.sha256_file(path)}
                for path in raw_files
            ],
        },
    }
    result = measure._strict_json(result)
    result["payload_sha256"] = measure.canonical_sha256(result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-incomplete-primary", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or args.run_dir / "analysis" / "response_gain_analysis.json"
    result = analyze(args.run_dir, require_primary_complete=not args.allow_incomplete_primary)
    measure.atomic_write_json(output, result)
    print(output, flush=True)


if __name__ == "__main__":
    main()
