#!/usr/bin/env python3
"""Portable CPU analysis for the two completed Phase-2 race-model protocols.

The acquisition workflows used CUDA, but analysis is deliberately standard-
library-only.  Trials remain nested within their checkpoint: seeds 43--51 are
the nine independent rows, and every endpoint in a family shares the same
exact row sign in each of the 2**9 permutations.  Latencies and endpoint
quantities are measured in milliseconds.

Supported protocol versions are the held-out four-candidate confirmation and
the triggered scalar-adaptation attribution frozen in the ``dbcf447`` lineage.
The obsolete 120-cell preregistration is intentionally not accepted.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import csv
import hashlib
import itertools
import json
import math
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


JsonObject = dict[str, Any]

CHECKPOINT_SEEDS: tuple[int, ...] = tuple(range(43, 52))
Q_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
INTENSITY = 0.05
CONFIDENCE = 0.95
N_PATTERNS = 2 ** len(CHECKPOINT_SEEDS)
REJECTING_NUMERATOR_MAX = 25

BASELINE_FAMILY_FILE_SHA256 = (
    "aaa253392285ca0725d9353d2eed918e24cfbc84b41d9ab413b0277c13b9c60b"
)
BASELINE_FAMILY_PAYLOAD_SHA256 = (
    "43905ff28511722569014e5138a82560f648bd7ad3ead93edabfec96a6f9b3a6"
)
BASELINE_ACQUISITION_HEAD = "4da0644b03a39f094fc2e9f80a7ec682b8a49df3"
BASELINE_DRIVER_SHA256 = (
    "b047c142c3627857bf28b6a6e8bf246b16980d92d89535d3c43bd0abea913388"
)
NETWORK_IO_SHA256 = "afb91212c40de1bc45be5bb7a8408a218dca03194d38f1e43520525bf60a29aa"
LATENCY_MODULE_SHA256 = (
    "171efa9cbdbc78c215ec58e56b490ae66f13bd52d3944a86e387846edc8808bd"
)

HELDOUT_REFERENCE_FILE_SHA256 = (
    "5a3311604507a698add9494d274171d8d08269a2532b8c36d6010d06be13cac7"
)
HELDOUT_REFERENCE_PAYLOAD_SHA256 = (
    "c058ee7655b2c65c960ae379ab200d872956341befed51d09b6b5ea307ae5a6f"
)
ADAPTATION_REFERENCE_FILE_SHA256 = (
    "704c84628b6d9501e23fd0d73327d1967085cc960832aa2fdb263bcafa57f126"
)
ADAPTATION_REFERENCE_PAYLOAD_SHA256 = (
    "f61d2e136b757ae8a77c31a20322978644635c96229e0d1391fe0e9d055db39f"
)


class AnalysisValidationError(AssertionError):
    """Raised when integrity, provenance, design, or QC cannot be proven."""


@dataclass(frozen=True)
class WorkflowSpec:
    """Immutable pins and dimensions for one completed acquisition protocol."""

    workflow: str
    protocol_id: str
    ledger_schema: str
    ledger_file_sha256: str
    ledger_payload_sha256: str
    cells: tuple[str, ...]
    baseline_seeds: tuple[int, ...]
    acquisition_head: str
    manifest_file_sha256: str
    manifest_semantic_sha256: str
    driver_sha256: str
    reference_file_sha256: str
    reference_payload_sha256: str


HELDOUT_SPEC = WorkflowSpec(
    workflow="heldout_confirmation",
    protocol_id="dm10-phase2-i005-heldout-exploratory-confirmation-v1",
    ledger_schema="fsts-phase2-i005-confirmation-acquisition-ledger-v1",
    ledger_file_sha256=(
        "dfe040ad8f853f27254302d231786cee299b15ffa30f65410f8b17c12d268bd0"
    ),
    ledger_payload_sha256=(
        "95c1679b483ed2183e1a198582bd0bd8d807358ba5e556c367a950e5263aceff"
    ),
    cells=("adaptation_prior", "pv_gaba_scale=0", "tau_gaba=10", "gNMDA=.765"),
    baseline_seeds=CHECKPOINT_SEEDS,
    acquisition_head="19133d6f969b0feedd2847d25410a1af4c4b14cc",
    manifest_file_sha256=(
        "a5d8e71baa883651bedb41a14050d260e0b1cce31ea05b9d9e19a910d85f7f02"
    ),
    manifest_semantic_sha256=(
        "4b4c274cae5a420086863a34d063df9d3a1f2dac027cea3379de1e7ab56b0452"
    ),
    driver_sha256="5c776cbe24b3e577724f68c0b848abc844c782262c2bee73f325afa9f15893a0",
    reference_file_sha256=HELDOUT_REFERENCE_FILE_SHA256,
    reference_payload_sha256=HELDOUT_REFERENCE_PAYLOAD_SHA256,
)

ADAPTATION_SPEC = WorkflowSpec(
    workflow="adaptation_attribution",
    protocol_id="dm10-phase2-i005-triggered-adaptation-attribution-v1",
    ledger_schema="fsts-adaptation-attribution-ledger-v1",
    ledger_file_sha256=(
        "c4c1b91186c41c456e53d5ea06ea1616cc0b5fab89ba2a16d1e882321f36ba38"
    ),
    ledger_payload_sha256=(
        "cb1c62c396828e4b7d9a3980d47c57ebe9f5d0284b8b2be4851c1da27506adc1"
    ),
    cells=("aM=.008", "dM=8"),
    baseline_seeds=(42, *CHECKPOINT_SEEDS),
    acquisition_head="dbcf447346bbbd948aea6c95b4324b14340a22d8",
    manifest_file_sha256=(
        "dee0433d79cfe6dd69e85e5bfa6ad1b28c64251af9d58732e2d85f1816dc0ea9"
    ),
    manifest_semantic_sha256=(
        "dc928c70bc2e6c1a4da500d3d72f347d1c009a57412584e020102839aea7895e"
    ),
    driver_sha256="6f565d285ad6758987a9b5ec2863515da090fe7053598f096b2ae9f127b35a82",
    reference_file_sha256=ADAPTATION_REFERENCE_FILE_SHA256,
    reference_payload_sha256=ADAPTATION_REFERENCE_PAYLOAD_SHA256,
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash JSON with the canonical encoding used by the frozen artifacts."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> JsonObject:
    """Load a strict JSON object, rejecting duplicate keys and NaN/Infinity."""

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    if not isinstance(value, dict):
        raise ValueError(f"top-level JSON is not an object: {path}")
    return value


def verify_self_checksum(record: JsonObject, container: str, field: str) -> str:
    """Verify and return an embedded canonical JSON payload SHA-256."""

    clone = copy.deepcopy(record)
    try:
        embedded = clone[container].pop(field) if container else clone.pop(field)
    except (KeyError, TypeError) as exc:
        raise AnalysisValidationError(f"missing payload checksum {container}.{field}") from exc
    if not isinstance(embedded, str):
        raise AnalysisValidationError("payload checksum is not a string")
    observed = canonical_sha256(clone)
    if observed != embedded:
        raise AnalysisValidationError(
            f"payload checksum mismatch: observed {observed}, recorded {embedded}"
        )
    return embedded


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisValidationError(message)


def _resolve_recorded_file(root: Path, recorded_path: Any) -> Path:
    """Resolve a ledger entry by basename under a caller-supplied artifact root.

    Frozen ledgers retain their original absolute acquisition paths.  Those
    strings are provenance, not live filesystem requirements.  Relocation is
    safe because each entry's bytes and embedded payload are independently
    pinned before use.
    """

    _require(isinstance(recorded_path, str), "ledger path is not a string")
    filename = Path(recorded_path).name
    _require(filename not in ("", ".", ".."), "ledger path has no filename")
    candidate = (root / filename).resolve()
    resolved_root = root.resolve()
    _require(candidate.parent == resolved_root, "resolved artifact escaped its root")
    return candidate


def _validate_embedded_configuration(record: Mapping[str, Any]) -> str:
    configuration = record.get("configuration")
    recorded = record.get("configuration_sha256")
    _require(isinstance(configuration, dict), "configuration is not an object")
    _require(isinstance(recorded, str), "configuration_sha256 is missing")
    observed = canonical_sha256(configuration)
    _require(observed == recorded, "configuration payload hash mismatch")
    return recorded


def _validate_state_integrity(record: Mapping[str, Any]) -> None:
    integrity = record.get("integrity")
    _require(isinstance(integrity, dict), "integrity block is missing")
    states = [
        integrity.get("state_sha256_loaded"),
        integrity.get("state_sha256_before"),
        integrity.get("state_sha256_after"),
    ]
    _require(all(isinstance(value, str) for value in states), "state hashes are missing")
    _require(len(set(states)) == 1, "model state changed during acquisition")
    _require(integrity.get("state_bit_identical") is True, "state identity flag is false")
    _require(
        integrity.get("readout_md5_before") == integrity.get("readout_md5_after"),
        "readout weights changed during acquisition",
    )
    _require(integrity.get("readouts_unchanged") is True, "readout identity flag is false")
    _require(integrity.get("plasticity_enabled") is False, "plasticity was enabled")


def _validate_measurement(record: Mapping[str, Any], *, candidate: bool) -> None:
    measurement = record.get("measurement")
    _require(isinstance(measurement, dict), "measurement block is missing")
    expected = {
        "n_trials_per_condition": 1000,
        "run_seed": 20260712,
        "substream_count": 10,
        "trials_per_substream": 100,
        "centre_deg": 90.0,
        "pulse_frames": 10,
        "n_frames": 40,
        "frame_ms": 10.0,
        "dt_ms": 0.1,
        "n_substeps": 100,
        "horizon_ms": 400.0,
        "soa_ms": 0.0,
    }
    for key, value in expected.items():
        _require(measurement.get(key) == value, f"measurement {key} changed")
    _require(
        measurement.get("condition_order") == ["A", "V", "AV", "catch"],
        "condition order changed",
    )
    intensities = measurement.get("intensities")
    _require(isinstance(intensities, list) and INTENSITY in intensities, "0.05 intensity missing")
    if candidate:
        _require(intensities == [INTENSITY], "candidate acquired unexpected intensities")
    endpoint = measurement.get("endpoint")
    _require(isinstance(endpoint, dict), "endpoint contract is missing")
    _require(
        endpoint.get("response_rule_id") == "first_msi_population_spike_v1",
        "endpoint response rule changed",
    )


def _validate_candidate_code(record: Mapping[str, Any], spec: WorkflowSpec) -> None:
    manifest = record.get("manifest")
    code = record.get("code")
    _require(isinstance(manifest, dict) and isinstance(code, dict), "code provenance missing")
    _require(manifest.get("literal_sha256") == spec.manifest_file_sha256, "manifest file pin changed")
    _require(
        manifest.get("semantic_sha256") == spec.manifest_semantic_sha256,
        "manifest semantic pin changed",
    )
    git = code.get("git")
    _require(isinstance(git, dict) and git.get("head") == spec.acquisition_head, "acquisition HEAD changed")
    _require(git.get("clean") is True, "acquisition checkout was not clean")
    sources = code.get("source_sha256")
    _require(isinstance(sources, dict), "source hash ledger missing")
    _require(sources.get("driver") == spec.driver_sha256, "driver source hash changed")
    _require(
        sources.get("shared_acquisition_kernel") == BASELINE_DRIVER_SHA256,
        "shared acquisition kernel hash changed",
    )
    _require(sources.get("network_io") == NETWORK_IO_SHA256, "network I/O hash changed")
    _require(sources.get("latency_module") == LATENCY_MODULE_SHA256, "latency source hash changed")
    reconciliations = code.get("reconciliations")
    _require(
        isinstance(reconciliations, dict) and all(reconciliations.values()),
        "source reconciliation did not pass",
    )


def _validate_baseline_code(record: Mapping[str, Any]) -> None:
    code = record.get("code")
    _require(isinstance(code, dict), "baseline code provenance missing")
    git = code.get("git")
    _require(
        isinstance(git, dict) and git.get("head") == BASELINE_ACQUISITION_HEAD,
        "baseline acquisition HEAD changed",
    )
    _require(git.get("clean") is True, "baseline acquisition checkout was not clean")
    sources = code.get("source_sha256")
    _require(isinstance(sources, dict), "baseline source hash ledger missing")
    _require(sources.get("driver") == BASELINE_DRIVER_SHA256, "baseline driver hash changed")
    _require(sources.get("network_io") == NETWORK_IO_SHA256, "baseline network I/O hash changed")
    _require(sources.get("latency_module") == LATENCY_MODULE_SHA256, "baseline latency hash changed")


def _validate_analysis_contract(record: Mapping[str, Any], expected_cells: int) -> None:
    confirmation = record.get("confirmation_protocol")
    _require(isinstance(confirmation, dict), "confirmation protocol is missing")
    _require(
        confirmation.get("confirmation_checkpoint_seeds") == list(CHECKPOINT_SEEDS),
        "confirmation checkpoint seeds changed",
    )
    contract = confirmation.get("analysis_contract")
    _require(isinstance(contract, dict), "analysis contract is missing")
    _require(contract.get("tails") == "all_two_sided", "tail contract changed")
    _require(
        contract.get("exact_common_checkpoint_row_sign_patterns") == N_PATTERNS,
        "sign-pattern count changed",
    )
    _require(contract.get("plus_one_correction") is False, "plus-one correction changed")
    _require(contract.get("seed42_in_analysis") is False, "seed 42 leaked into inference")
    primary_text = contract.get("primary_family")
    _require(
        isinstance(primary_text, str) and primary_text.startswith(f"{expected_cells * len(Q_LEVELS)} "),
        "primary family size changed",
    )


def _load_candidate_record(
    entry: Mapping[str, Any],
    root: Path,
    spec: WorkflowSpec,
) -> tuple[JsonObject, JsonObject]:
    path = _resolve_recorded_file(root, entry.get("path"))
    _require(path.is_file(), f"candidate raw is missing: {path.name}")
    observed_file = sha256_file(path)
    _require(observed_file == entry.get("file_sha256"), f"candidate file hash mismatch: {path.name}")
    record = load_json(path)
    payload = verify_self_checksum(record, "integrity", "raw_payload_sha256")
    _require(payload == entry.get("payload_sha256"), f"candidate payload hash mismatch: {path.name}")
    configuration = _validate_embedded_configuration(record)
    _require(configuration == entry.get("configuration_sha256"), "ledger configuration hash mismatch")
    _require(record.get("schema_version") == "fsts-race-perturbation-trials-v1", "wrong raw schema")
    _require(record.get("protocol_id") == spec.protocol_id, "wrong raw protocol")
    condition = record.get("condition")
    checkpoint = record.get("checkpoint")
    _require(isinstance(condition, dict) and isinstance(checkpoint, dict), "raw identity blocks missing")
    _require(condition.get("cell_id") == entry.get("cell_id"), "candidate cell identity mismatch")
    _require(checkpoint.get("seed") == entry.get("seed"), "candidate seed identity mismatch")
    _validate_measurement(record, candidate=True)
    _validate_state_integrity(record)
    _validate_candidate_code(record, spec)
    _validate_analysis_contract(record, len(spec.cells))
    provenance = {
        "cell_id": entry.get("cell_id"),
        "seed": entry.get("seed"),
        "filename": path.name,
        "file_sha256": observed_file,
        "payload_sha256": payload,
        "configuration_sha256": configuration,
        "checkpoint_sha256": checkpoint.get("sha256"),
    }
    return record, provenance


def _load_baseline_record(
    entry: Mapping[str, Any], root: Path
) -> tuple[JsonObject, JsonObject]:
    path = _resolve_recorded_file(root, entry.get("path"))
    _require(path.is_file(), f"baseline raw is missing: {path.name}")
    observed_file = sha256_file(path)
    _require(observed_file == entry.get("file_sha256"), f"baseline file hash mismatch: {path.name}")
    record = load_json(path)
    payload = verify_self_checksum(record, "integrity", "raw_payload_sha256")
    _require(payload == entry.get("payload_sha256"), f"baseline payload hash mismatch: {path.name}")
    configuration = _validate_embedded_configuration(record)
    _require(record.get("schema_version") == "fsts-race-trials-v1", "wrong baseline raw schema")
    condition = record.get("condition")
    checkpoint = record.get("checkpoint")
    _require(isinstance(condition, dict) and condition.get("cell_id") == "baseline", "wrong baseline cell")
    _require(isinstance(checkpoint, dict) and checkpoint.get("seed") == entry.get("seed"), "wrong baseline seed")
    _validate_measurement(record, candidate=False)
    _validate_state_integrity(record)
    _validate_baseline_code(record)
    provenance = {
        "seed": entry.get("seed"),
        "filename": path.name,
        "file_sha256": observed_file,
        "payload_sha256": payload,
        "configuration_sha256": configuration,
        "checkpoint_sha256": checkpoint.get("sha256"),
    }
    return record, provenance


def _finite_hits(
    block: Mapping[str, Any],
    *,
    condition: str,
    intensity: float,
    horizon_ms: float,
) -> list[float]:
    """Validate one condition block and return finite hit latencies in ms."""

    n = block.get("n_trials")
    latencies = block.get("latency_ms")
    hits = block.get("hit")
    trial_ids = block.get("trial_id")
    _require(isinstance(n, int) and n > 0, "invalid n_trials")
    _require(
        all(isinstance(values, list) and len(values) == n for values in (latencies, hits, trial_ids)),
        "trial arrays do not match n_trials",
    )
    _require(block.get("condition") == condition, "condition label mismatch")
    _require(block.get("stimulus_intensity") == intensity, "condition intensity mismatch")
    _require(block.get("censor_ms") == horizon_ms, "condition censor horizon mismatch")
    _require(block.get("silence_encoding") == "+Inf", "silence encoding changed")
    _require(
        block.get("trial_index") == {"start": 0, "stop_exclusive": n},
        "trial index contract changed",
    )
    _require(len(set(trial_ids)) == n, "trial IDs are not unique")
    rng = block.get("rng")
    _require(isinstance(rng, dict), "RNG provenance is missing")
    substreams = rng.get("substreams")
    _require(isinstance(substreams, list) and len(substreams) == 10, "RNG substream ledger changed")

    restored: list[float] = []
    for index, (latency, hit) in enumerate(zip(latencies, hits)):
        _require(isinstance(hit, bool), f"hit[{index}] is not boolean")
        if hit:
            _require(
                not isinstance(latency, bool) and isinstance(latency, (int, float)),
                f"hit latency[{index}] is not numeric",
            )
            value = float(latency)
            _require(
                math.isfinite(value) and 0.0 < value <= horizon_ms,
                f"hit latency[{index}] is outside the horizon",
            )
            restored.append(value)
        else:
            _require(latency == "+Inf", f"miss latency[{index}] is not '+Inf'")
    _require(sum(hits) == block.get("n_hits"), "n_hits disagrees with hit flags")
    _require(float(block.get("hit_rate")) == sum(hits) / n, "hit_rate disagrees with flags")
    return restored


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _step_area_float(
    a: Sequence[float],
    v: Sequence[float],
    av: Sequence[float],
    n_trials: int,
    horizon_ms: float,
) -> float:
    """Integrate the positive Miller-bound violation area in milliseconds."""

    support = sorted(set([0.0, horizon_ms, *a, *v, *av]))
    a_sorted, v_sorted, av_sorted = sorted(a), sorted(v), sorted(av)
    total = 0.0
    for left, right in zip(support[:-1], support[1:]):
        f_a = bisect.bisect_right(a_sorted, left) / float(n_trials)
        f_v = bisect.bisect_right(v_sorted, left) / float(n_trials)
        f_av = bisect.bisect_right(av_sorted, left) / float(n_trials)
        total += max(f_av - min(f_a + f_v, 1.0), 0.0) * (right - left)
    return float(total)


def _step_area_exact(
    a: Sequence[float],
    v: Sequence[float],
    av: Sequence[float],
    n_trials: int,
    horizon_ms: float,
) -> Fraction:
    a_f = sorted(Fraction.from_float(value) for value in a)
    v_f = sorted(Fraction.from_float(value) for value in v)
    av_f = sorted(Fraction.from_float(value) for value in av)
    support = sorted(
        set([Fraction(0), Fraction.from_float(horizon_ms), *a_f, *v_f, *av_f])
    )
    total = Fraction(0)
    for left, right in zip(support[:-1], support[1:]):
        f_a = Fraction(bisect.bisect_right(a_f, left), n_trials)
        f_v = Fraction(bisect.bisect_right(v_f, left), n_trials)
        f_av = Fraction(bisect.bisect_right(av_f, left), n_trials)
        difference = f_av - min(f_a + f_v, Fraction(1))
        if difference > 0:
            total += difference * (right - left)
    return total


def reconstruct_endpoint(
    record: Mapping[str, Any], intensity: float = INTENSITY
) -> JsonObject:
    """Reconstruct QC, G(q), and positive violation area from trial arrays.

    ``G(q) = Q_bound(q) - Q_AV(q)`` is in milliseconds.  Misses remain
    censored at the recorded 400-ms horizon and are never silently dropped
    from the empirical-CDF denominator.
    """

    rows = record.get("rows")
    _require(isinstance(rows, list), "rows is not a list")
    selected = [row for row in rows if float(row.get("intensity")) == intensity]
    _require(len(selected) == 1, f"expected exactly one intensity={intensity} row")
    row = selected[0]
    blocks = row.get("conditions")
    _require(
        isinstance(blocks, dict) and set(blocks) == {"A", "V", "AV", "catch"},
        "condition blocks are malformed",
    )
    horizon = float(record["measurement"]["horizon_ms"])
    samples: dict[str, list[float]] = {}
    n_values: set[int] = set()
    for condition in ("A", "V", "AV", "catch"):
        block = blocks[condition]
        samples[condition] = _finite_hits(
            block,
            condition=condition,
            intensity=0.0 if condition == "catch" else intensity,
            horizon_ms=horizon,
        )
        n_values.add(int(block["n_trials"]))
    _require(len(n_values) == 1, "condition trial counts are unbalanced")
    n_trials = n_values.pop()
    rates = {name: len(samples[name]) / n_trials for name in ("A", "V", "AV")}
    imbalance = max(rates.values()) - min(rates.values())
    catch_rate = len(samples["catch"]) / n_trials

    bound_events = sorted([*samples["A"], *samples["V"]])
    av_events = sorted(samples["AV"])
    q_bound_values: list[float | None] = []
    q_av_values: list[float | None] = []
    g_values: list[float | None] = []
    for q in Q_LEVELS:
        target = _ceil_fraction(Fraction(str(q)) * n_trials)
        q_bound = bound_events[target - 1] if len(bound_events) >= target else None
        q_av = av_events[target - 1] if len(av_events) >= target else None
        q_bound_values.append(q_bound)
        q_av_values.append(q_av)
        g_values.append(None if q_bound is None or q_av is None else q_bound - q_av)

    all_reached = all(value is not None for value in g_values)
    classification = "pure_latency"
    reasons: list[str] = []
    if all(not samples[name] for name in ("A", "V", "AV")):
        classification = "insufficient_hits"
        reasons.append("all A/V/AV trials censored")
    elif catch_rate > 0.05:
        classification = "uninterpretable"
        reasons.append("catch false-positive rate exceeds 0.05")
    elif not all_reached:
        classification = "insufficient_hits"
        reasons.append("one or more q levels unreachable")
    elif min(rates.values()) < 0.95 or imbalance > 0.05:
        classification = "joint_detection_latency"
        if min(rates.values()) < 0.95:
            reasons.append("A/V/AV hit rate below 0.95")
        if imbalance > 0.05:
            reasons.append("A/V/AV hit-rate imbalance exceeds 0.05")

    area_float = _step_area_float(
        samples["A"], samples["V"], samples["AV"], n_trials, horizon
    )
    area_exact = _step_area_exact(
        samples["A"], samples["V"], samples["AV"], n_trials, horizon
    )
    _require(
        abs(area_float - float(area_exact)) <= 2e-12,
        "floating and rational positive-area reconstructions disagree",
    )
    return {
        "n_trials": n_trials,
        "hit_rates": rates,
        "max_hit_rate_imbalance": imbalance,
        "catch_false_positive_rate": catch_rate,
        "q_reached": all_reached,
        "qc_classification": classification,
        "qc_pass": classification == "pure_latency",
        "qc_reasons": reasons,
        "Q_bound_ms": q_bound_values,
        "Q_AV_ms": q_av_values,
        "G_ms": g_values,
        "Aplus_full_ms": area_float,
        "Aplus_full_exact": f"{area_exact.numerator}/{area_exact.denominator}",
    }


def _t_float(values: Sequence[float]) -> tuple[float, float, float]:
    n = len(values)
    _require(n >= 2, "at least two checkpoints are required")
    mean = sum(values) / n
    sum_squares = sum((value - mean) ** 2 for value in values)
    standard_deviation = math.sqrt(sum_squares / (n - 1))
    standard_error = standard_deviation / math.sqrt(n)
    if standard_error == 0:
        statistic = math.copysign(math.inf, mean) if mean != 0 else 0.0
    else:
        statistic = mean / standard_error
    return mean, standard_error, statistic


def _t_squared_exact(values: Sequence[Fraction], signs: Sequence[int]) -> Fraction | None:
    n = len(values)
    signed = [sign * value for sign, value in zip(signs, values)]
    total = sum(signed, Fraction(0))
    square_total = sum((value * value for value in signed), Fraction(0))
    denominator = n * square_total - total * total
    if denominator == 0:
        return None if total != 0 else Fraction(0)
    return Fraction(n - 1) * total * total / denominator


def _t_squared_ge(left: Fraction | None, right: Fraction | None) -> bool:
    if left is None:
        return True
    if right is None:
        return False
    return left >= right


def _max_t_squared(values: Iterable[Fraction | None]) -> Fraction | None:
    sentinel = object()
    result: Fraction | None | object = sentinel
    for value in values:
        if value is None:
            return None
        if result is sentinel or value > result:
            result = value
    _require(result is not sentinel, "empty max-T family")
    return result  # type: ignore[return-value]


def _residual_band(matrix: Sequence[Sequence[float]]) -> JsonObject:
    n = len(matrix)
    n_columns = len(matrix[0])
    columns = [[float(matrix[row][column]) for row in range(n)] for column in range(n_columns)]
    means: list[float] = []
    standard_errors: list[float] = []
    residual_columns: list[list[float]] = []
    for column in columns:
        mean, standard_error, _ = _t_float(column)
        means.append(mean)
        standard_errors.append(standard_error)
        residual_columns.append([value - mean for value in column])
    patterns = list(itertools.product((-1, 1), repeat=n))
    order_index = math.ceil(CONFIDENCE * len(patterns)) - 1
    residual_maxima: list[float] = []
    for signs in patterns:
        row_maximum = 0.0
        for residuals in residual_columns:
            signed = [sign * value for sign, value in zip(signs, residuals)]
            _, _, statistic = _t_float(signed)
            if math.isfinite(statistic):
                row_maximum = max(row_maximum, abs(statistic))
        residual_maxima.append(row_maximum)
    critical = sorted(residual_maxima)[order_index]
    achieved = sum(value <= critical for value in residual_maxima) / len(residual_maxima)

    exact_columns = [[Fraction.from_float(value) for value in column] for column in columns]
    exact_residuals: list[list[Fraction]] = []
    for column in exact_columns:
        mean = sum(column, Fraction(0)) / n
        exact_residuals.append([value - mean for value in column])
    exact_maxima = [
        _max_t_squared(_t_squared_exact(column, signs) for column in exact_residuals)
        for signs in patterns
    ]
    _require(not any(value is None for value in exact_maxima), "infinite residual statistic")
    exact_sorted = sorted(value for value in exact_maxima if value is not None)
    exact_critical = math.sqrt(float(exact_sorted[order_index]))
    _require(
        math.isclose(critical, exact_critical, rel_tol=2e-12, abs_tol=2e-12),
        "floating and rational residual critical values disagree",
    )
    exact_achieved = sum(value <= exact_sorted[order_index] for value in exact_sorted) / len(exact_sorted)
    _require(achieved == exact_achieved, "band coverage audit disagrees")
    return {
        "method": "centered-residual common-checkpoint-row sign-flip max-|t|",
        "approximate": True,
        "symmetry_assumption": "checkpoint residual vectors are rowwise sign-symmetric",
        "critical_value": critical,
        "critical_value_rational_audit": exact_critical,
        "nominal_coverage": CONFIDENCE,
        "achieved_discrete_coverage": achieved,
        "order_index_zero_based": order_index,
        "n_sign_patterns": len(patterns),
        "lower": [mean - critical * se for mean, se in zip(means, standard_errors)],
        "upper": [mean + critical * se for mean, se in zip(means, standard_errors)],
    }


def exact_family_inference(matrix: Sequence[Sequence[float]]) -> JsonObject:
    """Run exact common-checkpoint sign flips and max-|T| family adjustment.

    Rows are independent checkpoints; columns are endpoints in one frozen
    family.  Comparisons use exact rational squared |t| values so ties are
    conservative, including zero-variance/infinite-statistic cases.
    """

    n = len(matrix)
    _require(n == len(CHECKPOINT_SEEDS), "inference requires exactly nine checkpoints")
    _require(bool(matrix) and bool(matrix[0]), "endpoint matrix is empty")
    n_columns = len(matrix[0])
    _require(all(len(row) == n_columns for row in matrix), "endpoint matrix is ragged")
    _require(
        all(math.isfinite(float(value)) for row in matrix for value in row),
        "endpoint matrix contains a non-finite value",
    )
    columns = [[float(matrix[row][column]) for row in range(n)] for column in range(n_columns)]
    exact_columns = [[Fraction.from_float(value) for value in column] for column in columns]
    patterns = list(itertools.product((-1, 1), repeat=n))
    observed_signs = (1,) * n
    observed_squared = [_t_squared_exact(column, observed_signs) for column in exact_columns]
    null_squared: list[list[Fraction | None]] = []
    null_maxima: list[Fraction | None] = []
    for signs in patterns:
        row = [_t_squared_exact(column, signs) for column in exact_columns]
        null_squared.append(row)
        null_maxima.append(_max_t_squared(row))
    observed_maximum = _max_t_squared(observed_squared)
    global_numerator = sum(
        _t_squared_ge(value, observed_maximum) for value in null_maxima
    )

    means: list[float] = []
    standard_errors: list[float] = []
    statistics: list[float] = []
    unadjusted: list[int] = []
    adjusted: list[int] = []
    for column_index, column in enumerate(columns):
        mean, standard_error, statistic = _t_float(column)
        means.append(mean)
        standard_errors.append(standard_error)
        statistics.append(statistic)
        unadjusted.append(
            sum(
                _t_squared_ge(null_squared[row][column_index], observed_squared[column_index])
                for row in range(len(patterns))
            )
        )
        adjusted.append(
            sum(_t_squared_ge(value, observed_squared[column_index]) for value in null_maxima)
        )
    return {
        "method": "exact common-checkpoint-row sign-flip max-|t|",
        "tail": "two-sided",
        "n_checkpoints": n,
        "n_family_cells": n_columns,
        "n_sign_patterns": len(patterns),
        "plus_one_correction": False,
        "global_p_numerator": global_numerator,
        "global_p_denominator": len(patterns),
        "global_p": global_numerator / len(patterns),
        "means": means,
        "ses": standard_errors,
        "t": statistics,
        "unadjusted_p_numerators": unadjusted,
        "adjusted_p_numerators": adjusted,
        "p_denominator": len(patterns),
        "unadjusted_p": [value / len(patterns) for value in unadjusted],
        "adjusted_p": [value / len(patterns) for value in adjusted],
        "simultaneous_band": _residual_band(matrix),
    }


def _validate_ledger(path: Path, spec: WorkflowSpec) -> JsonObject:
    _require(path.is_file(), f"ledger is missing: {path}")
    observed_file = sha256_file(path)
    _require(observed_file == spec.ledger_file_sha256, "ledger file hash changed")
    ledger = load_json(path)
    payload = verify_self_checksum(ledger, "", "ledger_payload_sha256")
    _require(payload == spec.ledger_payload_sha256, "ledger payload hash changed")
    _require(ledger.get("schema_version") == spec.ledger_schema, "ledger schema changed")
    _require(ledger.get("protocol_id") == spec.protocol_id, "ledger protocol changed")

    raw_entries = ledger.get("confirmation_raws")
    baseline_entries = ledger.get("immutable_baselines")
    _require(isinstance(raw_entries, list), "candidate ledger is not a list")
    _require(isinstance(baseline_entries, list), "baseline ledger is not a list")
    expected_raw_keys = _validate_candidate_design(raw_entries, spec.cells)
    expected_baselines = set(spec.baseline_seeds)
    observed_baselines = {entry.get("seed") for entry in baseline_entries}
    _require(observed_baselines == expected_baselines, "baseline ledger design changed")
    _require(len(baseline_entries) == len(expected_baselines), "baseline ledger contains duplicates")
    _require(all(entry.get("seed") != 42 for entry in raw_entries), "seed 42 leaked into inference")
    filenames = [Path(str(entry.get("path"))).name for entry in raw_entries]
    _require(len(filenames) == len(set(filenames)), "candidate filenames are not unique")

    validation = ledger.get("validation")
    sentinel = ledger.get("sentinel")
    _require(isinstance(validation, dict), "ledger validation block is missing")
    _require(
        isinstance(sentinel, dict)
        and sentinel.get("status") == "PASS"
        and sentinel.get("zero_tolerance_deep_equality") is True,
        "determinism sentinel did not pass",
    )
    if spec == HELDOUT_SPEC:
        selection = ledger.get("selection_frozen")
        execution = ledger.get("execution")
        _require(isinstance(selection, dict), "held-out selection block is missing")
        _require(selection.get("selected_cell_ids") == list(spec.cells), "held-out cells changed")
        _require(
            selection.get("confirmation_checkpoint_seeds") == list(CHECKPOINT_SEEDS),
            "held-out seeds changed",
        )
        _require(selection.get("prohibited_screen_seed") == 42, "screen exclusion changed")
        _require(selection.get("outcomes_may_change_selection") is False, "selection was not frozen")
        contract = selection.get("analysis_contract")
        _require(
            isinstance(contract, dict)
            and contract.get("tails") == "all_two_sided"
            and contract.get("exact_common_checkpoint_row_sign_patterns") == N_PATTERNS
            and contract.get("plus_one_correction") is False
            and contract.get("seed42_in_analysis") is False,
            "held-out inference contract changed",
        )
        _require(isinstance(execution, dict), "held-out execution ledger is missing")
        _require(execution.get("git_head") == spec.acquisition_head, "held-out acquisition HEAD changed")
        _require(
            execution.get("manifest_file_sha256") == spec.manifest_file_sha256
            and execution.get("manifest_semantic_sha256") == spec.manifest_semantic_sha256
            and execution.get("wrapper_file_sha256") == spec.driver_sha256,
            "held-out acquisition source pins changed",
        )
        _require(
            validation.get("raw_count") == len(expected_raw_keys)
            and validation.get("all_completed_payload_checks_pass") is True
            and validation.get("all_protocol_source_checkpoint_device_configuration_checks_pass") is True
            and validation.get("all_rng_trial_ledger_checks_pass") is True,
            "held-out acquisition validation did not pass",
        )
    else:
        protocol = ledger.get("protocol")
        gate = ledger.get("seed42_gate")
        _require(isinstance(protocol, dict), "adaptation protocol ledger is missing")
        _require(protocol.get("git_head") == spec.acquisition_head, "adaptation acquisition HEAD changed")
        _require(
            protocol.get("manifest_file_sha256") == spec.manifest_file_sha256
            and protocol.get("manifest_semantic_sha256") == spec.manifest_semantic_sha256
            and protocol.get("wrapper_file_sha256") == spec.driver_sha256,
            "adaptation acquisition source pins changed",
        )
        _require(
            isinstance(gate, dict)
            and gate.get("status") == "PASS"
            and gate.get("excluded_from_confirmation") is True,
            "seed-42 continuation/exclusion gate did not pass",
        )
        _require(
            validation.get("raw_count") == len(expected_raw_keys)
            and validation.get("all_integrity_protocol_source_checkpoint_rng_log_checks_pass") is True
            and validation.get("dM_first_spike_exact_null_all_9") is True,
            "adaptation acquisition validation did not pass",
        )
    return ledger


def _validate_candidate_design(
    raw_entries: Sequence[Mapping[str, Any]], cells: Sequence[str]
) -> set[tuple[str, int]]:
    """Require the exact cells × seeds-43--51 design and exclude seed 42."""

    expected = {(cell, seed) for cell in cells for seed in CHECKPOINT_SEEDS}
    observed = {(entry.get("cell_id"), entry.get("seed")) for entry in raw_entries}
    _require(observed == expected, "candidate ledger design changed")
    _require(len(raw_entries) == len(expected), "candidate ledger contains duplicates")
    _require(all(entry.get("seed") != 42 for entry in raw_entries), "seed 42 leaked into inference")
    return expected


def _validate_baseline_family(path: Path) -> tuple[JsonObject, dict[float, bool], JsonObject]:
    _require(path.is_file(), f"baseline family artifact is missing: {path}")
    file_sha = sha256_file(path)
    _require(file_sha == BASELINE_FAMILY_FILE_SHA256, "baseline family file hash changed")
    artifact = load_json(path)
    payload = verify_self_checksum(artifact, "", "derived_payload_sha256")
    _require(payload == BASELINE_FAMILY_PAYLOAD_SHA256, "baseline family payload hash changed")
    formal_by_q = {
        float(cell["q"]): bool(cell["formal_rmi_violation"])
        for cell in artifact["condition_violation_inference"]["cells"]
        if float(cell["intensity"]) == INTENSITY and cell["condition"] == "baseline"
    }
    _require(set(formal_by_q) == set(Q_LEVELS), "baseline same-q RMI family is incomplete")
    _require(all(formal_by_q.values()), "baseline same-q formal RMI gate did not pass")
    provenance = {
        "filename": path.name,
        "file_sha256": file_sha,
        "payload_sha256": payload,
    }
    return artifact, formal_by_q, provenance


def _load_workflow_records(
    ledger: Mapping[str, Any],
    *,
    raw_dir: Path,
    baseline_dir: Path,
    spec: WorkflowSpec,
) -> tuple[
    dict[tuple[str, int], JsonObject],
    dict[int, JsonObject],
    list[JsonObject],
    list[JsonObject],
]:
    raw_records: dict[tuple[str, int], JsonObject] = {}
    raw_provenance: list[JsonObject] = []
    for entry in ledger["confirmation_raws"]:
        record, provenance = _load_candidate_record(entry, raw_dir, spec)
        key = (str(entry["cell_id"]), int(entry["seed"]))
        _require(key not in raw_records, "duplicate candidate identity")
        raw_records[key] = record
        raw_provenance.append(provenance)

    baseline_records: dict[int, JsonObject] = {}
    baseline_provenance: list[JsonObject] = []
    for entry in ledger["immutable_baselines"]:
        record, provenance = _load_baseline_record(entry, baseline_dir)
        seed = int(entry["seed"])
        _require(seed not in baseline_records, "duplicate baseline seed")
        baseline_records[seed] = record
        baseline_provenance.append(provenance)

    raw_provenance.sort(key=lambda value: (value["cell_id"], value["seed"]))
    baseline_provenance.sort(key=lambda value: value["seed"])
    return raw_records, baseline_records, raw_provenance, baseline_provenance


def _condition_blocks(record: Mapping[str, Any], intensity: float = INTENSITY) -> Mapping[str, Any]:
    rows = record.get("rows")
    _require(isinstance(rows, list), "rows are missing")
    matches = [row for row in rows if float(row.get("intensity")) == intensity]
    _require(len(matches) == 1, "paired record does not contain exactly one target row")
    conditions = matches[0].get("conditions")
    _require(isinstance(conditions, dict), "paired condition blocks are missing")
    return conditions


def _assert_paired_checkpoint(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> None:
    """Prove checkpoint identity and common trial/RNG streams for a paired row."""

    _require(
        candidate["checkpoint"]["sha256"] == baseline["checkpoint"]["sha256"],
        "paired checkpoint SHA-256 differs",
    )
    candidate_blocks = _condition_blocks(candidate)
    baseline_blocks = _condition_blocks(baseline)
    for condition in ("A", "V", "AV", "catch"):
        for field in ("trial_id", "trial_index", "rng"):
            _require(
                candidate_blocks[condition][field] == baseline_blocks[condition][field],
                f"paired {condition} {field} differs",
            )


def _trial_outcomes_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_blocks = _condition_blocks(left)
    right_blocks = _condition_blocks(right)
    return all(
        left_blocks[condition][field] == right_blocks[condition][field]
        for condition in ("A", "V", "AV", "catch")
        for field in ("latency_ms", "hit", "trial_id", "trial_index", "rng")
    )


def _inference_without_vectors(inference: Mapping[str, Any]) -> JsonObject:
    omitted = {
        "means",
        "ses",
        "t",
        "unadjusted_p_numerators",
        "adjusted_p_numerators",
        "unadjusted_p",
        "adjusted_p",
    }
    return {key: copy.deepcopy(value) for key, value in inference.items() if key not in omitted}


def _compute_summary(
    *,
    ledger_path: Path,
    ledger: Mapping[str, Any],
    raw_records: Mapping[tuple[str, int], JsonObject],
    baseline_records: Mapping[int, JsonObject],
    raw_provenance: list[JsonObject],
    baseline_provenance: list[JsonObject],
    baseline_family_provenance: JsonObject,
    formal_by_q: Mapping[float, bool],
    spec: WorkflowSpec,
) -> tuple[JsonObject, JsonObject]:
    baseline_endpoints = {
        seed: reconstruct_endpoint(baseline_records[seed]) for seed in CHECKPOINT_SEEDS
    }
    candidate_endpoints = {
        (cell, seed): reconstruct_endpoint(raw_records[(cell, seed)])
        for cell in spec.cells
        for seed in CHECKPOINT_SEEDS
    }
    for seed in CHECKPOINT_SEEDS:
        for cell in spec.cells:
            _assert_paired_checkpoint(raw_records[(cell, seed)], baseline_records[seed])

    all_baseline_qc = all(endpoint["qc_pass"] for endpoint in baseline_endpoints.values())
    all_candidate_qc = {
        cell: all(candidate_endpoints[(cell, seed)]["qc_pass"] for seed in CHECKPOINT_SEEDS)
        for cell in spec.cells
    }
    _require(all_baseline_qc, "baseline QC failed; confirmatory inference is closed")
    _require(all(all_candidate_qc.values()), "candidate QC failed; confirmatory inference is closed")

    g_matrix: list[list[float]] = []
    area_matrix: list[list[float]] = []
    paired_deltas: dict[tuple[str, int], JsonObject] = {}
    for seed in CHECKPOINT_SEEDS:
        g_row: list[float] = []
        area_row: list[float] = []
        baseline = baseline_endpoints[seed]
        for cell in spec.cells:
            candidate = candidate_endpoints[(cell, seed)]
            delta_g: list[float] = []
            for q_index in range(len(Q_LEVELS)):
                candidate_g = candidate["G_ms"][q_index]
                baseline_g = baseline["G_ms"][q_index]
                _require(
                    candidate_g is not None and baseline_g is not None,
                    "G(q) is unreachable despite QC",
                )
                delta = float(candidate_g - baseline_g)
                delta_g.append(delta)
                g_row.append(delta)
            delta_area = float(candidate["Aplus_full_ms"] - baseline["Aplus_full_ms"])
            area_row.append(delta_area)
            paired_deltas[(cell, seed)] = {
                "cell_id": cell,
                "seed": seed,
                "delta_G_ms": delta_g,
                "delta_Aplus_full_ms": delta_area,
            }
        g_matrix.append(g_row)
        area_matrix.append(area_row)

    primary = exact_family_inference(g_matrix)
    secondary = exact_family_inference(area_matrix)
    primary_cells: list[JsonObject] = []
    for index, (cell, q) in enumerate(
        (cell, q) for cell in spec.cells for q in Q_LEVELS
    ):
        adjusted_numerator = primary["adjusted_p_numerators"][index]
        lower = primary["simultaneous_band"]["lower"][index]
        upper = primary["simultaneous_band"]["upper"][index]
        mean = primary["means"][index]
        heldout_gate = bool(
            spec == HELDOUT_SPEC
            and mean < 0
            and adjusted_numerator <= REJECTING_NUMERATOR_MAX
            and upper < 0
            and formal_by_q[q]
        )
        attribution_gate = bool(
            spec == ADAPTATION_SPEC
            and cell == "aM=.008"
            and mean > 0
            and adjusted_numerator <= REJECTING_NUMERATOR_MAX
            and lower > 0
        )
        primary_cells.append(
            {
                "cell_id": cell,
                "q": q,
                "n_checkpoints": len(CHECKPOINT_SEEDS),
                "mean_delta_G_ms": mean,
                "se_delta_G_ms": primary["ses"][index],
                "t": primary["t"][index],
                "unadjusted_p_numerator": primary["unadjusted_p_numerators"][index],
                "adjusted_p_numerator": adjusted_numerator,
                "p_denominator": primary["p_denominator"],
                "unadjusted_p": primary["unadjusted_p"][index],
                "adjusted_p": primary["adjusted_p"][index],
                "adjusted_significant": adjusted_numerator <= REJECTING_NUMERATOR_MAX,
                "simultaneous_lower_delta_G_ms": lower,
                "simultaneous_upper_delta_G_ms": upper,
                "qc_pass_all_checkpoints": all_candidate_qc[cell],
                "shipped_same_q_formal_RMI": formal_by_q[q],
                "ASD_like_reduction_gate": heldout_gate,
                "aM_attribution_gate": attribution_gate,
            }
        )

    secondary_cells: list[JsonObject] = []
    for index, cell in enumerate(spec.cells):
        adjusted_numerator = secondary["adjusted_p_numerators"][index]
        secondary_cells.append(
            {
                "cell_id": cell,
                "n_checkpoints": len(CHECKPOINT_SEEDS),
                "mean_delta_Aplus_full_ms": secondary["means"][index],
                "se_delta_Aplus_full_ms": secondary["ses"][index],
                "t": secondary["t"][index],
                "unadjusted_p_numerator": secondary["unadjusted_p_numerators"][index],
                "adjusted_p_numerator": adjusted_numerator,
                "p_denominator": secondary["p_denominator"],
                "unadjusted_p": secondary["unadjusted_p"][index],
                "adjusted_p": secondary["adjusted_p"][index],
                "adjusted_significant": adjusted_numerator <= REJECTING_NUMERATOR_MAX,
                "simultaneous_lower_delta_Aplus_full_ms": secondary["simultaneous_band"]["lower"][index],
                "simultaneous_upper_delta_Aplus_full_ms": secondary["simultaneous_band"]["upper"][index],
                "qc_pass_all_checkpoints": all_candidate_qc[cell],
                "role": "secondary_convergent_endpoint_not_a_primary_substitute",
            }
        )

    ledger_file_sha = sha256_file(ledger_path)
    sources = {
        "ledger": {
            "filename": ledger_path.name,
            "file_sha256": ledger_file_sha,
            "payload_sha256": spec.ledger_payload_sha256,
        },
        "baseline_family": baseline_family_provenance,
        "candidate_raws": raw_provenance,
        "immutable_baselines": baseline_provenance,
    }
    sources["artifact_set_payload_sha256"] = canonical_sha256(sources)
    summary: JsonObject = {
        "schema_version": "fsts-race-completed-workflow-analysis-v1",
        "protocol_id": spec.protocol_id,
        "workflow": spec.workflow,
        "status": "complete_qc_and_provenance_validated",
        "analysis_runtime": "CPU standard library only",
        "checkpoint_seeds": list(CHECKPOINT_SEEDS),
        "seed42_excluded_from_inference": True,
        "intensity": INTENSITY,
        "q_levels": list(Q_LEVELS),
        "delta_definition": "paired checkpoint perturbation_minus_shipped",
        "inference_contract": {
            "independent_unit": "checkpoint",
            "trials_and_rng_substreams_nested_within_checkpoint": True,
            "common_row_sign_across_family": True,
            "tails": "all_two_sided",
            "n_sign_patterns": N_PATTERNS,
            "plus_one_correction": False,
            "conservative_exact_rational_ties": True,
            "adjusted_rejection_rule": "numerator <= 25 out of 512",
            "simultaneous_band": "centered-residual row-sign-flip max-|t|",
        },
        "quality_control": {
            "baseline_pass_all_checkpoints": all_baseline_qc,
            "candidate_pass_all_checkpoints": all_candidate_qc,
            "minimum_A_V_AV_hit_rate": 0.95,
            "maximum_A_V_AV_hit_rate_imbalance": 0.05,
            "maximum_catch_false_positive_rate": 0.05,
        },
        "primary_family": {
            **_inference_without_vectors(primary),
            "cells": primary_cells,
        },
        "secondary_family": {
            **_inference_without_vectors(secondary),
            "cells": secondary_cells,
        },
        "source_artifacts": sources,
        "interpretation_scope": (
            "Model first-spike endpoint only; not clinical ASD, a unique mechanism, "
            "an SC locus, or human reaction time."
        ),
    }
    context = {
        "baseline_endpoints": baseline_endpoints,
        "candidate_endpoints": candidate_endpoints,
        "paired_deltas": paired_deltas,
        "raw_records": dict(raw_records),
        "baseline_records": dict(baseline_records),
    }
    return summary, context


def _assert_close(observed: Any, expected: Any, label: str) -> None:
    if isinstance(expected, bool) or isinstance(observed, bool):
        _require(observed is expected, f"saved reference mismatch: {label}")
    elif isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        _require(
            math.isclose(float(observed), float(expected), rel_tol=2e-12, abs_tol=2e-12),
            f"saved reference mismatch: {label}",
        )
    else:
        _require(observed == expected, f"saved reference mismatch: {label}")


def _validate_saved_reference(
    summary: Mapping[str, Any], path: Path, spec: WorkflowSpec
) -> JsonObject:
    """Verify that recomputed numerical families reproduce the saved result."""

    _require(path.is_file(), f"saved reference analysis is missing: {path}")
    file_sha = sha256_file(path)
    _require(file_sha == spec.reference_file_sha256, "saved reference file hash changed")
    reference = load_json(path)
    payload = verify_self_checksum(reference, "", "analysis_payload_sha256")
    _require(payload == spec.reference_payload_sha256, "saved reference payload hash changed")
    _require(reference.get("protocol_id") == spec.protocol_id, "saved reference protocol changed")

    primary = summary["primary_family"]
    secondary = summary["secondary_family"]
    reference_primary = reference["primary_family"]
    reference_secondary = reference["secondary_family"]
    if spec == HELDOUT_SPEC:
        _assert_close(
            primary["simultaneous_band"]["critical_value"],
            reference_primary["critical_value"],
            "primary critical value",
        )
        _assert_close(
            primary["simultaneous_band"]["achieved_discrete_coverage"],
            reference_primary["achieved_discrete_coverage"],
            "primary coverage",
        )
        _assert_close(
            secondary["simultaneous_band"]["critical_value"],
            reference_secondary["critical_value"],
            "secondary critical value",
        )
        reference_primary_map = {
            (cell["cell_id"], float(cell["q"])): cell
            for cell in reference_primary["cells"]
        }
        reference_secondary_map = {
            cell["cell_id"]: cell for cell in reference_secondary["cells"]
        }
        for cell in primary["cells"]:
            key = (cell["cell_id"], float(cell["q"]))
            expected = reference_primary_map[key]
            comparisons = {
                "mean_delta_G_ms": "mean_delta_G_ms",
                "se_delta_G_ms": "se_delta_G_ms",
                "t": "t",
                "unadjusted_p_numerator": "unadjusted_p_numerator",
                "adjusted_p_numerator": "adjusted_p_numerator",
                "p_denominator": "p_denominator",
                "simultaneous_lower_delta_G_ms": "approx_simultaneous_lower_ms",
                "simultaneous_upper_delta_G_ms": "approx_simultaneous_upper_ms",
                "ASD_like_reduction_gate": "ASD_like_reduction_gate_pass",
            }
            for observed_key, expected_key in comparisons.items():
                _assert_close(cell[observed_key], expected[expected_key], f"primary {key} {observed_key}")
        for cell in secondary["cells"]:
            expected = reference_secondary_map[cell["cell_id"]]
            comparisons = {
                "mean_delta_Aplus_full_ms": "mean_delta_Aplus_full_ms",
                "se_delta_Aplus_full_ms": "se_delta_Aplus_full_ms",
                "t": "t",
                "unadjusted_p_numerator": "unadjusted_p_numerator",
                "adjusted_p_numerator": "adjusted_p_numerator",
                "p_denominator": "p_denominator",
                "simultaneous_lower_delta_Aplus_full_ms": "approx_simultaneous_lower_ms",
                "simultaneous_upper_delta_Aplus_full_ms": "approx_simultaneous_upper_ms",
            }
            for observed_key, expected_key in comparisons.items():
                _assert_close(
                    cell[observed_key], expected[expected_key], f"secondary {cell['cell_id']} {observed_key}"
                )
    else:
        _assert_close(
            primary["simultaneous_band"]["critical_value"],
            reference_primary["critical"],
            "adaptation primary critical value",
        )
        _assert_close(
            primary["simultaneous_band"]["achieved_discrete_coverage"],
            reference_primary["achieved_coverage"],
            "adaptation primary coverage",
        )
        _assert_close(
            secondary["simultaneous_band"]["critical_value"],
            reference_secondary["critical"],
            "adaptation secondary critical value",
        )
        reference_primary_map = {
            (cell["cell_id"], float(cell["q"])): cell
            for cell in reference_primary["cells"]
        }
        reference_secondary_map = {
            cell["cell_id"]: cell for cell in reference_secondary["cells"]
        }
        for cell in primary["cells"]:
            key = (cell["cell_id"], float(cell["q"]))
            expected = reference_primary_map[key]
            comparisons = {
                "mean_delta_G_ms": "mean_delta_G_ms",
                "se_delta_G_ms": "se_ms",
                "t": "t",
                "unadjusted_p_numerator": "unadjusted_numerator",
                "adjusted_p_numerator": "adjusted_numerator",
                "p_denominator": "denominator",
                "simultaneous_lower_delta_G_ms": "approx_simultaneous_lower_ms",
                "simultaneous_upper_delta_G_ms": "approx_simultaneous_upper_ms",
            }
            for observed_key, expected_key in comparisons.items():
                _assert_close(cell[observed_key], expected[expected_key], f"adaptation primary {key} {observed_key}")
        for cell in secondary["cells"]:
            expected = reference_secondary_map[cell["cell_id"]]
            comparisons = {
                "mean_delta_Aplus_full_ms": "mean_delta_Aplus_full_ms",
                "se_delta_Aplus_full_ms": "se_ms",
                "t": "t",
                "unadjusted_p_numerator": "unadjusted_numerator",
                "adjusted_p_numerator": "adjusted_numerator",
                "p_denominator": "denominator",
                "simultaneous_lower_delta_Aplus_full_ms": "approx_simultaneous_lower_ms",
                "simultaneous_upper_delta_Aplus_full_ms": "approx_simultaneous_upper_ms",
            }
            for observed_key, expected_key in comparisons.items():
                _assert_close(
                    cell[observed_key], expected[expected_key], f"adaptation secondary {cell['cell_id']} {observed_key}"
                )
    return {
        "filename": path.name,
        "file_sha256": file_sha,
        "payload_sha256": payload,
        "numerical_reproduction": "PASS",
    }


def _validate_auxiliary_payload(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str,
    payload_field: str,
) -> JsonObject:
    _require(path.is_file(), f"required protocol artifact is missing: {path}")
    file_sha = sha256_file(path)
    _require(file_sha == expected_file_sha256, f"artifact file hash changed: {path.name}")
    value = load_json(path)
    payload = verify_self_checksum(value, "", payload_field)
    _require(payload == expected_payload_sha256, f"artifact payload hash changed: {path.name}")
    return {
        "filename": path.name,
        "file_sha256": file_sha,
        "payload_sha256": payload,
    }


def _attach_summary_checksum(summary: JsonObject) -> JsonObject:
    result = copy.deepcopy(summary)
    result["summary_payload_sha256"] = canonical_sha256(result)
    return result


def analyze_heldout(
    *,
    ledger_path: Path,
    raw_dir: Path,
    baseline_dir: Path,
    baseline_family_path: Path,
    reference_analysis_path: Path,
) -> JsonObject:
    """Analyze the frozen four-candidate held-out confirmation on CPU.

    All paths are explicit and may point to a relocated artifact tree.  The
    immutable ledger and every raw/payload hash remain mandatory.
    """

    spec = HELDOUT_SPEC
    ledger = _validate_ledger(ledger_path, spec)
    _, formal_by_q, baseline_family_provenance = _validate_baseline_family(
        baseline_family_path
    )
    raw_records, baselines, raw_provenance, baseline_provenance = _load_workflow_records(
        ledger,
        raw_dir=raw_dir,
        baseline_dir=baseline_dir,
        spec=spec,
    )
    summary, _ = _compute_summary(
        ledger_path=ledger_path,
        ledger=ledger,
        raw_records=raw_records,
        baseline_records=baselines,
        raw_provenance=raw_provenance,
        baseline_provenance=baseline_provenance,
        baseline_family_provenance=baseline_family_provenance,
        formal_by_q=formal_by_q,
        spec=spec,
    )
    summary["selected_cells"] = list(spec.cells)
    summary["screen_seed_role"] = "selection_only_excluded_from_heldout_inference"
    summary["source_artifacts"]["saved_reference_analysis"] = _validate_saved_reference(
        summary, reference_analysis_path, spec
    )
    summary["source_artifacts"]["artifact_set_payload_sha256"] = canonical_sha256(
        {key: value for key, value in summary["source_artifacts"].items() if key != "artifact_set_payload_sha256"}
    )
    return _attach_summary_checksum(summary)


def analyze_adaptation(
    *,
    ledger_path: Path,
    raw_dir: Path,
    baseline_dir: Path,
    baseline_family_path: Path,
    seed42_gate_path: Path,
    trigger_analysis_path: Path,
    bridge_ledger_path: Path,
    bridge_raw_dir: Path,
    reference_analysis_path: Path,
) -> JsonObject:
    """Analyze triggered scalar adaptation attribution on seeds 43--51.

    Seed 42 is required only through its pinned continuation-gate artifact and
    is excluded from every inferential matrix.  ``dM=8`` is recomputed as an
    exact first-spike negative control; ``aM=.008`` is also compared with the
    independently acquired historical compound-adaptation raw trials.
    """

    spec = ADAPTATION_SPEC
    ledger = _validate_ledger(ledger_path, spec)
    _, formal_by_q, baseline_family_provenance = _validate_baseline_family(
        baseline_family_path
    )
    raw_records, baselines, raw_provenance, baseline_provenance = _load_workflow_records(
        ledger,
        raw_dir=raw_dir,
        baseline_dir=baseline_dir,
        spec=spec,
    )
    summary, context = _compute_summary(
        ledger_path=ledger_path,
        ledger=ledger,
        raw_records=raw_records,
        baseline_records=baselines,
        raw_provenance=raw_provenance,
        baseline_provenance=baseline_provenance,
        baseline_family_provenance=baseline_family_provenance,
        formal_by_q=formal_by_q,
        spec=spec,
    )

    gate_pin = ledger["seed42_gate"]
    gate_provenance = _validate_auxiliary_payload(
        seed42_gate_path,
        expected_file_sha256=gate_pin["file_sha256"],
        expected_payload_sha256=gate_pin["payload_sha256"],
        payload_field="payload_sha256",
    )
    selection_source = raw_records[(spec.cells[0], CHECKPOINT_SEEDS[0])][
        "confirmation_protocol"
    ]["selection_source"]
    trigger_provenance = _validate_auxiliary_payload(
        trigger_analysis_path,
        expected_file_sha256=selection_source["analysis_file_sha256"],
        expected_payload_sha256=selection_source["analysis_payload_sha256"],
        payload_field="analysis_payload_sha256",
    )
    _require(
        selection_source["acquisition_ledger_file_sha256"] == HELDOUT_SPEC.ledger_file_sha256,
        "trigger acquisition-ledger pin changed",
    )

    bridge_ledger = _validate_ledger(bridge_ledger_path, HELDOUT_SPEC)
    bridge_records: dict[tuple[str, int], JsonObject] = {}
    bridge_provenance: list[JsonObject] = []
    for entry in bridge_ledger["confirmation_raws"]:
        if entry["cell_id"] != "adaptation_prior":
            continue
        record, provenance = _load_candidate_record(entry, bridge_raw_dir, HELDOUT_SPEC)
        bridge_records[(entry["cell_id"], entry["seed"])] = record
        bridge_provenance.append(provenance)
    _require(len(bridge_records) == len(CHECKPOINT_SEEDS), "compound bridge is incomplete")

    dm_equal: dict[str, bool] = {}
    compound_equal: dict[str, bool] = {}
    for seed in CHECKPOINT_SEEDS:
        dm_equal[str(seed)] = _trial_outcomes_equal(
            raw_records[("dM=8", seed)], baselines[seed]
        )
        compound_equal[str(seed)] = _trial_outcomes_equal(
            raw_records[("aM=.008", seed)], bridge_records[("adaptation_prior", seed)]
        )
    _require(all(dm_equal.values()), "dM=8 is not an exact first-spike negative control")
    _require(all(compound_equal.values()), "aM=.008 does not reproduce compound first-spike trials")
    dm_primary_zero = all(
        value == 0.0
        for seed in CHECKPOINT_SEEDS
        for value in context["paired_deltas"][("dM=8", seed)]["delta_G_ms"]
    )
    dm_secondary_zero = all(
        context["paired_deltas"][("dM=8", seed)]["delta_Aplus_full_ms"] == 0.0
        for seed in CHECKPOINT_SEEDS
    )
    _require(dm_primary_zero and dm_secondary_zero, "dM endpoint deltas are not exact zero")

    summary["selected_cells"] = list(spec.cells)
    summary["seed42_role"] = "exploratory_continuation_gate_only_excluded_from_inference"
    summary["dM_negative_control"] = {
        "raw_first_spike_exact_by_seed": dm_equal,
        "raw_first_spike_exact_all_9": True,
        "all_primary_deltas_exact_zero": dm_primary_zero,
        "secondary_delta_exact_zero": dm_secondary_zero,
    }
    summary["compound_bridge"] = {
        "comparison": "isolated aM=.008,dM=10 versus prior compound aM=.008,dM=8",
        "raw_first_spike_exact_by_seed": compound_equal,
        "all_9_trial_outcomes_exact": True,
    }
    summary["source_artifacts"].update(
        {
            "seed42_continuation_gate": gate_provenance,
            "heldout_trigger_analysis": trigger_provenance,
            "compound_bridge_ledger": {
                "filename": bridge_ledger_path.name,
                "file_sha256": HELDOUT_SPEC.ledger_file_sha256,
                "payload_sha256": HELDOUT_SPEC.ledger_payload_sha256,
            },
            "compound_bridge_raws": sorted(
                bridge_provenance, key=lambda value: value["seed"]
            ),
        }
    )
    summary["source_artifacts"]["saved_reference_analysis"] = _validate_saved_reference(
        summary, reference_analysis_path, spec
    )
    summary["source_artifacts"]["artifact_set_payload_sha256"] = canonical_sha256(
        {key: value for key, value in summary["source_artifacts"].items() if key != "artifact_set_payload_sha256"}
    )
    summary["interpretation"] = (
        "At the frozen first-spike endpoint, the confirmed compound-adaptation effect is "
        "attributable to lowering aM from .02 to .008; lowering post-spike dM from 10 "
        "to 8 is an exact negative control."
    )
    return _attach_summary_checksum(summary)


def endpoint_table_rows(summary: Mapping[str, Any]) -> list[JsonObject]:
    """Return deterministic, flat aggregate endpoint rows for CSV export."""

    rows: list[JsonObject] = []
    workflow = summary["workflow"]
    protocol_id = summary["protocol_id"]
    for cell in summary["primary_family"]["cells"]:
        rows.append(
            {
                "workflow": workflow,
                "protocol_id": protocol_id,
                "endpoint": "delta_G_q",
                "units": "ms",
                "cell_id": cell["cell_id"],
                "q": cell["q"],
                "n_checkpoints": cell["n_checkpoints"],
                "mean_delta": cell["mean_delta_G_ms"],
                "standard_error": cell["se_delta_G_ms"],
                "t": cell["t"],
                "unadjusted_p_numerator": cell["unadjusted_p_numerator"],
                "adjusted_p_numerator": cell["adjusted_p_numerator"],
                "p_denominator": cell["p_denominator"],
                "adjusted_p": cell["adjusted_p"],
                "simultaneous_lower": cell["simultaneous_lower_delta_G_ms"],
                "simultaneous_upper": cell["simultaneous_upper_delta_G_ms"],
                "qc_pass_all_checkpoints": cell["qc_pass_all_checkpoints"],
                "confirmatory_gate": bool(
                    cell["ASD_like_reduction_gate"] or cell["aM_attribution_gate"]
                ),
            }
        )
    for cell in summary["secondary_family"]["cells"]:
        rows.append(
            {
                "workflow": workflow,
                "protocol_id": protocol_id,
                "endpoint": "delta_Aplus_full",
                "units": "ms",
                "cell_id": cell["cell_id"],
                "q": "",
                "n_checkpoints": cell["n_checkpoints"],
                "mean_delta": cell["mean_delta_Aplus_full_ms"],
                "standard_error": cell["se_delta_Aplus_full_ms"],
                "t": cell["t"],
                "unadjusted_p_numerator": cell["unadjusted_p_numerator"],
                "adjusted_p_numerator": cell["adjusted_p_numerator"],
                "p_denominator": cell["p_denominator"],
                "adjusted_p": cell["adjusted_p"],
                "simultaneous_lower": cell["simultaneous_lower_delta_Aplus_full_ms"],
                "simultaneous_upper": cell["simultaneous_upper_delta_Aplus_full_ms"],
                "qc_pass_all_checkpoints": cell["qc_pass_all_checkpoints"],
                "confirmatory_gate": False,
            }
        )
    return rows


def _temporary_output_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def write_outputs(
    summary: Mapping[str, Any],
    *,
    summary_path: Path,
    endpoint_table_path: Path,
    overwrite: bool = False,
) -> JsonObject:
    """Atomically write canonical JSON and CSV without partial final files."""

    for path in (summary_path, endpoint_table_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    summary_temp = _temporary_output_path(summary_path)
    table_temp = _temporary_output_path(endpoint_table_path)
    fieldnames = [
        "workflow",
        "protocol_id",
        "endpoint",
        "units",
        "cell_id",
        "q",
        "n_checkpoints",
        "mean_delta",
        "standard_error",
        "t",
        "unadjusted_p_numerator",
        "adjusted_p_numerator",
        "p_denominator",
        "adjusted_p",
        "simultaneous_lower",
        "simultaneous_upper",
        "qc_pass_all_checkpoints",
        "confirmatory_gate",
    ]
    try:
        with summary_temp.open("x", encoding="utf-8") as handle:
            json.dump(
                summary,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        with table_temp.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(endpoint_table_rows(summary))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(summary_temp, summary_path)
        os.replace(table_temp, endpoint_table_path)
    finally:
        for temp in (summary_temp, table_temp):
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
    return {
        "summary_filename": summary_path.name,
        "summary_file_sha256": sha256_file(summary_path),
        "summary_payload_sha256": summary["summary_payload_sha256"],
        "endpoint_table_filename": endpoint_table_path.name,
        "endpoint_table_file_sha256": sha256_file(endpoint_table_path),
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--baseline-family", type=Path, required=True)
    parser.add_argument("--reference-analysis", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--endpoint-table-out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one completed protocol analysis from explicitly located artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="workflow", required=True)
    heldout_parser = subparsers.add_parser(
        "heldout", help="analyze the four-candidate seeds-43--51 confirmation"
    )
    _add_common_arguments(heldout_parser)
    adaptation_parser = subparsers.add_parser(
        "adaptation", help="analyze triggered scalar adaptation attribution"
    )
    _add_common_arguments(adaptation_parser)
    adaptation_parser.add_argument("--seed42-gate", type=Path, required=True)
    adaptation_parser.add_argument("--trigger-analysis", type=Path, required=True)
    adaptation_parser.add_argument("--bridge-ledger", type=Path, required=True)
    adaptation_parser.add_argument("--bridge-raw-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.workflow == "heldout":
        summary = analyze_heldout(
            ledger_path=args.ledger,
            raw_dir=args.raw_dir,
            baseline_dir=args.baseline_dir,
            baseline_family_path=args.baseline_family,
            reference_analysis_path=args.reference_analysis,
        )
    else:
        summary = analyze_adaptation(
            ledger_path=args.ledger,
            raw_dir=args.raw_dir,
            baseline_dir=args.baseline_dir,
            baseline_family_path=args.baseline_family,
            seed42_gate_path=args.seed42_gate,
            trigger_analysis_path=args.trigger_analysis,
            bridge_ledger_path=args.bridge_ledger,
            bridge_raw_dir=args.bridge_raw_dir,
            reference_analysis_path=args.reference_analysis,
        )
    written = write_outputs(
        summary,
        summary_path=args.summary_out,
        endpoint_table_path=args.endpoint_table_out,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "workflow": summary["workflow"],
                "protocol_id": summary["protocol_id"],
                **written,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
