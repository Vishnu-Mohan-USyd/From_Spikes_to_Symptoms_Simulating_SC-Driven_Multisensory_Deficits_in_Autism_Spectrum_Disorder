#!/usr/bin/env python3
"""Pure-CPU Miller race-model analysis for frozen neural first-spike trials.

The formal checkpoint statistic is quantile gain
``G(q) = Q_bound(q) - Q_AV(q)``, where
``F_bound(t) = min(F_A(t) + F_V(t), 1)`` and quantiles are exact empirical
step quantiles without interpolation.  Positive ``G`` has the Miller race
inequality implication.  Physical-time ``D(t)``, positive/signed areas, and the
independent-race comparator remain descriptive.

A positive result rejects the race inequality under context invariance.  It
does not prove a unique coactivation mechanism, a superior-colliculus locus,
or human reaction-time behavior; the endpoint is a model neural-latency
surrogate.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACQUISITION_SCHEMA = "fsts-race-trials-v1"
ANALYSIS_SCHEMA = "fsts-race-analysis-v2"
FAMILY_SCHEMA = "fsts-race-family-v2"
Q_LEVELS = tuple(value / 100.0 for value in range(5, 36, 5))
PRIMARY_SEEDS = tuple(range(42, 52))
PRIMARY_INTENSITIES = (0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.6)
PRIMARY_HORIZON_MS = 400.0
PRIMARY_N_TRIALS = 1000
PRIMARY_SUBSTREAMS = 10
TRIALS_PER_SUBSTREAM = 100
MIN_FORMAL_HIT_RATE = 0.95
MAX_FORMAL_HIT_IMBALANCE = 0.05
MAX_CATCH_FALSE_POSITIVE_RATE = 0.05
PERMUTATION_TIE_EPS_MULTIPLIER = 64.0
ANALYZER_RELATIVE_PATH = "mechanism_influence/race_model_analysis.py"
ANALYSIS_HOTFIX_ALLOWED_PATHS = (
    ANALYZER_RELATIVE_PATH,
    "tests/test_race_model.py",
)
ACQUISITION_CRITICAL_CODE_PATH_KEYS = (
    "driver",
    "network_io",
    "latency_module",
)
BASELINE_MECHANISMS = {
    "aM": 0.02,
    "dM": 10.0,
    "pv_gaba_scale": 1.0,
    "tau_gaba": 18.0,
    "gNMDA": 0.51,
}
SCALAR_RMI_FAMILY = "scalar_rmi_v1"
HISTORICAL_ADAPTATION_FAMILY = "historical_adaptation_v1"
RNG_ALGORITHM = "torch global RNG; ten preregistered vectorized substreams"
STREAM_SEED_DERIVATION = (
    "sha256(run_seed:checkpoint_seed:float17_intensity:"
    "sensory_condition_label:substream_index)"
)
CONDITION_ORDER = ("A", "V", "AV", "catch")
ENDPOINT_RESPONSE_RULE = "first substep with MSI population spike count > 0"
ENDPOINT_THRESHOLD = "MSI population spike count > 0"
ENDPOINT_FORMULA = "(_first_spike_substep + 1) * dt_ms"
ENDPOINT_INTERPRETATION = (
    "model neural first-spike latency surrogate; not human reaction time"
)
ENDPOINT_KEYS = {
    "response_rule_id",
    "response_rule",
    "threshold",
    "latency_formula",
    "latency_domain_ms",
    "resolution_ms",
    "interpretation",
    "sensitivity_horizons_ms",
}


def _mechanism_vector(**updates: float) -> dict[str, float]:
    vector = dict(BASELINE_MECHANISMS)
    vector.update({name: float(value) for name, value in updates.items()})
    return vector


MECHANISM_CELL_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {
    SCALAR_RMI_FAMILY: {
        "baseline": {"factor_id": "baseline", "requested": _mechanism_vector()},
        "aM=0": {"factor_id": "aM", "requested": _mechanism_vector(aM=0.0)},
        "aM=.008": {"factor_id": "aM", "requested": _mechanism_vector(aM=0.008)},
        "aM=.04": {"factor_id": "aM", "requested": _mechanism_vector(aM=0.04)},
        "dM=0": {"factor_id": "dM_negative_control", "requested": _mechanism_vector(dM=0.0)},
        "dM=8": {"factor_id": "dM_negative_control", "requested": _mechanism_vector(dM=8.0)},
        "dM=16": {"factor_id": "dM_negative_control", "requested": _mechanism_vector(dM=16.0)},
        "pv_gaba_scale=0": {
            "factor_id": "MSI-inhibitory_GABA_scale",
            "requested": _mechanism_vector(pv_gaba_scale=0.0),
        },
        "pv_gaba_scale=.5": {
            "factor_id": "MSI-inhibitory_GABA_scale",
            "requested": _mechanism_vector(pv_gaba_scale=0.5),
        },
        "pv_gaba_scale=1.5": {
            "factor_id": "MSI-inhibitory_GABA_scale",
            "requested": _mechanism_vector(pv_gaba_scale=1.5),
        },
        "pv_gaba_scale=4": {
            "factor_id": "MSI-inhibitory_GABA_scale",
            "requested": _mechanism_vector(pv_gaba_scale=4.0),
        },
        "tau_gaba=10": {"factor_id": "tau_gaba", "requested": _mechanism_vector(tau_gaba=10.0)},
        "tau_gaba=40": {"factor_id": "tau_gaba", "requested": _mechanism_vector(tau_gaba=40.0)},
        "tau_gaba=60": {"factor_id": "tau_gaba", "requested": _mechanism_vector(tau_gaba=60.0)},
        "gNMDA=0": {"factor_id": "gNMDA", "requested": _mechanism_vector(gNMDA=0.0)},
        "gNMDA=.255": {"factor_id": "gNMDA", "requested": _mechanism_vector(gNMDA=0.255)},
        "gNMDA=.765": {"factor_id": "gNMDA", "requested": _mechanism_vector(gNMDA=0.765)},
    },
    HISTORICAL_ADAPTATION_FAMILY: {
        "adaptation_off": {
            "factor_id": "compound_adaptation_replication",
            "requested": _mechanism_vector(aM=0.0, dM=0.0),
        },
        "adaptation_prior": {
            "factor_id": "compound_adaptation_replication",
            "requested": _mechanism_vector(aM=0.008, dM=8.0),
        },
        "adaptation_shipped": {
            "factor_id": "compound_adaptation_replication",
            "requested": _mechanism_vector(aM=0.02, dM=10.0),
        },
    },
}
DISCLAIMER = (
    "A formal RMI violation rejects race plus context-invariance assumptions; "
    "it does not prove unique neural coactivation, an SC locus, or human RT."
)


@dataclass(frozen=True)
class TrialSample:
    """Validated latencies, with censored silence retained as logical ``+Inf``."""

    latency_ms: np.ndarray
    hit: np.ndarray
    censor_ms: float

    @property
    def n_trials(self) -> int:
        return int(self.latency_ms.size)


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return value


def _require_sequence(value: Any, location: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{location} must be an array")
    return value


def _is_hex_digest(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_output_path(path: Path) -> Path:
    """Require derived artifacts to be outside the repo or git-ignored."""

    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError:
        return resolved
    ignored = subprocess.run(
        ("git", "check-ignore", "-q", "--", str(relative)),
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ignored.returncode != 0:
        raise ValueError(
            f"output inside repository must be git-ignored to preserve release integrity: {resolved}"
        )
    return resolved


@lru_cache(maxsize=1)
def _analyzer_provenance() -> dict[str, Any]:
    """Fingerprint this analyzer and its current git revision for derived output."""

    path = Path(__file__).resolve()

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=path.parents[1],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    root = path.parents[1]
    relative_path = str(path.relative_to(root))
    tracked_probe = subprocess.run(
        ("git", "ls-files", "--error-unmatch", "--", relative_path),
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tracked = tracked_probe.returncode == 0
    head_blob_sha256 = None
    local_matches_head = False
    if tracked:
        head_blob_probe = subprocess.run(
            ("git", "show", f"HEAD:{relative_path}"),
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if head_blob_probe.returncode == 0:
            head_blob_sha256 = hashlib.sha256(head_blob_probe.stdout).hexdigest()
            local_matches_head = head_blob_sha256 == _sha256_file(path)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "tracked": tracked,
        "head_blob_sha256": head_blob_sha256,
        "local_matches_head": local_matches_head,
        "git": {
            "head": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "clean": git("status", "--porcelain") == "",
        },
    }


def analyzer_provenance() -> dict[str, Any]:
    """Return a copy of cached analyzer release provenance."""

    return copy.deepcopy(_analyzer_provenance())


def _run_git_bytes(repository_root: Path, *arguments: str) -> tuple[int, bytes]:
    """Run one read-only Git probe, returning a fail-closed status and bytes."""

    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return 127, b""
    return completed.returncode, completed.stdout


def _git_blob_sha256(
    repository_root: Path, commit: str, relative_path: str
) -> str | None:
    return_code, payload = _run_git_bytes(
        repository_root, "show", f"{commit}:{relative_path}"
    )
    return hashlib.sha256(payload).hexdigest() if return_code == 0 else None


def _validated_raw_hash_manifest(
    records: Sequence[Mapping[str, Any]],
    analyses: Sequence[Mapping[str, Any]],
    *,
    verify_checkpoint_files: bool,
) -> tuple[bool, dict[str, Any]]:
    """Reconcile raw/config/state/readout/checkpoint hashes for release gating."""

    manifest: dict[str, Any] = {}
    all_exact = len(records) == len(analyses)
    for record, derived in zip(records, analyses):
        try:
            condition = _require_mapping(record.get("condition"), "condition")
            checkpoint = _require_mapping(record.get("checkpoint"), "checkpoint")
            integrity = _require_mapping(record.get("integrity"), "integrity")
            key = f"{condition['label']}|seed{int(checkpoint['seed'])}"
            raw_hash = str(integrity["raw_payload_sha256"])
            configuration_hash = str(record["configuration_sha256"])
            checkpoint_hash = str(checkpoint["sha256"])
            state_hashes = [
                str(integrity[name])
                for name in (
                    "state_sha256_loaded",
                    "state_sha256_before",
                    "state_sha256_after",
                )
            ]
            readout_before = [str(value) for value in integrity["readout_md5_before"]]
            readout_after = [str(value) for value in integrity["readout_md5_after"]]
            raw_exact = raw_hash == raw_payload_sha256(record)
            configuration_exact = configuration_hash == _canonical_sha256(
                _require_mapping(record.get("configuration"), "configuration")
            )
            state_exact = len(set(state_hashes)) == 1 and all(
                _is_hex_digest(value, 64) for value in state_hashes
            )
            readout_exact = readout_before == readout_after and bool(readout_before) and all(
                _is_hex_digest(value, 32)
                for value in readout_before + readout_after
            )
            derived_integrity_exact = (
                _require_mapping(derived.get("integrity"), "derived integrity").get("pass")
                is True
            )
            checkpoint_actual: str | None = None
            checkpoint_exact = _is_hex_digest(checkpoint_hash, 64)
            if verify_checkpoint_files:
                checkpoint_path = Path(str(checkpoint.get("path", "")))
                checkpoint_exact = checkpoint_path.is_file()
                if checkpoint_exact:
                    checkpoint_actual = _sha256_file(checkpoint_path)
                    checkpoint_exact = checkpoint_actual == checkpoint_hash
            entry_exact = all(
                (
                    raw_exact,
                    configuration_exact,
                    state_exact,
                    readout_exact,
                    checkpoint_exact,
                    derived_integrity_exact,
                )
            )
            all_exact = all_exact and entry_exact
            manifest[key] = {
                "raw_payload_sha256": raw_hash,
                "configuration_sha256": configuration_hash,
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_file_sha256": checkpoint_actual,
                "state_sha256": state_hashes[0] if state_exact else state_hashes,
                "readout_md5": readout_before,
                "exact": entry_exact,
            }
        except (KeyError, TypeError, ValueError):
            all_exact = False
    return all_exact and len(manifest) == len(records), manifest


def _acquisition_critical_source_manifest(
    records: Sequence[Mapping[str, Any]],
) -> tuple[bool, dict[str, dict[str, str]]]:
    """Resolve recorded acquisition source paths to safe Git-relative paths."""

    common: dict[str, dict[str, str]] | None = None
    complete = bool(records)
    for record in records:
        current: dict[str, dict[str, str]] = {}
        try:
            code = _require_mapping(record.get("code"), "code")
            root = Path(str(code["root"]))
            sources = _require_mapping(code.get("source_sha256"), "code.source_sha256")
            if not root.is_absolute():
                raise ValueError("recorded code root must be absolute")
            for name in ACQUISITION_CRITICAL_CODE_PATH_KEYS:
                source_path = Path(str(code[name]))
                if not source_path.is_absolute():
                    raise ValueError("recorded source path must be absolute")
                relative = source_path.relative_to(root)
                relative_path = relative.as_posix()
                if (
                    not relative.parts
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or ":" in relative_path
                    or "\x00" in relative_path
                ):
                    raise ValueError("recorded source path is unsafe")
                expected_hash = str(sources[name])
                if not _is_hex_digest(expected_hash, 64):
                    raise ValueError("recorded source hash is malformed")
                current[name] = {
                    "relative_path": relative_path,
                    "recorded_sha256": expected_hash,
                }
            if str(code.get("driver_sha256")) != current["driver"]["recorded_sha256"]:
                raise ValueError("driver hash disagrees with source manifest")
        except (KeyError, TypeError, ValueError):
            complete = False
            continue
        if common is None:
            common = current
        elif current != common:
            complete = False
    return complete and common is not None, {} if common is None else common


def _analysis_release_policy(
    records: Sequence[Mapping[str, Any]],
    analyses: Sequence[Mapping[str, Any]],
    release: Mapping[str, Any],
    *,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    """Authorize an exact analyzer revision or a narrow analysis-only hotfix.

    There is no runtime override.  A differing analyzer revision is eligible
    only when Git proves ancestry, the complete diff is allowlisted, all
    acquisition source blobs are byte-identical at base and head, and every
    raw/checkpoint/state/readout/configuration hash remains reconciled.
    """

    git = _require_mapping(release.get("git"), "analyzer git")
    acquisition_heads = {
        str(_require_mapping(record.get("code"), "code")["git"]["head"])
        for record in records
    }
    base_commit = next(iter(acquisition_heads)) if len(acquisition_heads) == 1 else None
    head_commit = str(git.get("head", ""))
    common_checks = {
        "analyzer_sha256_valid": _is_hex_digest(release.get("sha256"), 64),
        "analyzer_tracked": release.get("tracked") is True,
        "analyzer_local_matches_head": release.get("local_matches_head") is True
        and release.get("head_blob_sha256") == release.get("sha256"),
        "analyzer_git_clean": git.get("clean") is True,
        "analyzer_path_is_canonical": Path(str(release.get("path", ""))).resolve()
        == (repository_root / ANALYZER_RELATIVE_PATH).resolve(),
        "single_acquisition_commit": base_commit is not None
        and _is_hex_digest(base_commit, 40),
        "analyzer_head_valid": _is_hex_digest(head_commit, 40),
    }
    exact_head = base_commit is not None and base_commit == head_commit
    raw_exact, raw_manifest = _validated_raw_hash_manifest(
        records,
        analyses,
        verify_checkpoint_files=not exact_head,
    )
    preserved_hashes: dict[str, Any] = {
        "acquisition_critical_files": {},
        "validated_raw_artifacts": raw_manifest,
    }
    if exact_head:
        checks = {
            **common_checks,
            "exact_head_match": True,
            "raw_checkpoint_state_readout_config_hashes_exact": raw_exact,
        }
        passed = all(checks.values())
        return {
            "mode": "exact_head",
            "status": "not_required" if passed else "rejected",
            "pass": passed,
            "base_commit": base_commit,
            "head_commit": head_commit,
            "allowed_changed_paths": list(ANALYSIS_HOTFIX_ALLOWED_PATHS),
            "changed_paths": [],
            "checks": checks,
            "failed_checks": [name for name, value in checks.items() if not value],
            "preserved_hashes": preserved_hashes,
        }

    base_valid = bool(common_checks["single_acquisition_commit"])
    head_valid = bool(common_checks["analyzer_head_valid"])
    base_exists = False
    head_exists = False
    ancestor = False
    diff_readable = False
    changed_paths: list[str] = []
    if base_valid and head_valid and base_commit is not None:
        base_exists = (
            _run_git_bytes(repository_root, "cat-file", "-e", f"{base_commit}^{{commit}}")[0]
            == 0
        )
        head_exists = (
            _run_git_bytes(repository_root, "cat-file", "-e", f"{head_commit}^{{commit}}")[0]
            == 0
        )
        if base_exists and head_exists:
            ancestor = (
                _run_git_bytes(
                    repository_root,
                    "merge-base",
                    "--is-ancestor",
                    base_commit,
                    head_commit,
                )[0]
                == 0
            )
            diff_code, diff_payload = _run_git_bytes(
                repository_root,
                "diff",
                "--no-renames",
                "--name-only",
                "-z",
                base_commit,
                head_commit,
                "--",
            )
            if diff_code == 0:
                try:
                    changed_paths = sorted(
                        path.decode("utf-8")
                        for path in diff_payload.split(b"\x00")
                        if path
                    )
                    diff_readable = True
                except UnicodeDecodeError:
                    diff_readable = False

    sources_complete, source_specs = _acquisition_critical_source_manifest(records)
    base_sources_match = sources_complete and base_exists and base_commit is not None
    head_sources_match = sources_complete and head_exists
    critical_files: dict[str, Any] = {}
    for name, specification in source_specs.items():
        relative_path = specification["relative_path"]
        expected_hash = specification["recorded_sha256"]
        base_hash = (
            _git_blob_sha256(repository_root, base_commit, relative_path)
            if base_commit is not None and base_exists
            else None
        )
        head_hash = (
            _git_blob_sha256(repository_root, head_commit, relative_path)
            if head_exists
            else None
        )
        base_matches = base_hash == expected_hash
        head_matches = head_hash == expected_hash
        base_sources_match = base_sources_match and base_matches
        head_sources_match = head_sources_match and head_matches
        critical_files[name] = {
            **specification,
            "base_commit_sha256": base_hash,
            "head_commit_sha256": head_hash,
            "preserved": base_matches and head_matches,
        }
    preserved_hashes["acquisition_critical_files"] = critical_files
    allowed_paths = set(ANALYSIS_HOTFIX_ALLOWED_PATHS)
    checks = {
        **common_checks,
        "exact_head_match": False,
        "base_commit_exists": base_exists,
        "head_commit_exists": head_exists,
        "acquisition_commit_is_ancestor": ancestor,
        "changed_paths_readable": diff_readable,
        "changed_paths_allowlisted": diff_readable
        and bool(changed_paths)
        and set(changed_paths).issubset(allowed_paths),
        "analyzer_changed_in_hotfix": ANALYZER_RELATIVE_PATH in changed_paths,
        "acquisition_source_manifest_complete": sources_complete,
        "acquisition_sources_match_base": base_sources_match,
        "acquisition_sources_match_head": head_sources_match,
        "raw_checkpoint_state_readout_config_hashes_exact": raw_exact,
    }
    gating_checks = {
        name: value for name, value in checks.items() if name != "exact_head_match"
    }
    passed = all(gating_checks.values())
    return {
        "mode": "analysis_only_hotfix",
        "status": "verified" if passed else "rejected",
        "pass": passed,
        "base_commit": base_commit,
        "head_commit": head_commit,
        "allowed_changed_paths": list(ANALYSIS_HOTFIX_ALLOWED_PATHS),
        "changed_paths": changed_paths,
        "checks": checks,
        "failed_checks": [name for name, value in gating_checks.items() if not value],
        "preserved_hashes": preserved_hashes,
    }


def registered_mechanism_cell(
    settings: Mapping[str, float], design_family: str
) -> dict[str, Any]:
    """Resolve an exact scalar or historical cell; arbitrary mixtures fail."""

    if set(settings) != set(BASELINE_MECHANISMS):
        raise ValueError("requested mechanism keyset is not exact")
    for name, value in settings.items():
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"requested mechanism must be non-negative: {name}")
        if name == "tau_gaba" and numeric <= 0:
            raise ValueError("tau_gaba must be strictly positive")
    if design_family not in MECHANISM_CELL_REGISTRY:
        raise ValueError(f"unknown mechanism design family: {design_family}")
    matches = []
    for cell_id, specification in MECHANISM_CELL_REGISTRY[design_family].items():
        if all(
            math.isclose(
                float(settings[name]),
                float(specification["requested"][name]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name in BASELINE_MECHANISMS
        ):
            matches.append((cell_id, specification))
    if len(matches) != 1:
        raise ValueError(f"unregistered mechanism vector for {design_family}: {dict(settings)}")
    cell_id, specification = matches[0]
    changed = [
        name
        for name, baseline in BASELINE_MECHANISMS.items()
        if not math.isclose(float(settings[name]), baseline, rel_tol=0.0, abs_tol=1e-12)
    ]
    return {
        "design_family": design_family,
        "cell_id": cell_id,
        "factor_id": specification["factor_id"],
        "changed_scalars": changed,
        "requested": {name: float(settings[name]) for name in BASELINE_MECHANISMS},
        "scalar_one_factor": design_family == SCALAR_RMI_FAMILY and len(changed) <= 1,
    }


def _validate_endpoint(measurement: Mapping[str, Any]) -> None:
    endpoint = _require_mapping(measurement.get("endpoint"), "measurement.endpoint")
    if set(endpoint) != ENDPOINT_KEYS:
        raise ValueError("endpoint metadata keyset is not exact")
    horizon = float(measurement.get("horizon_ms"))
    dt_ms = float(measurement.get("dt_ms"))
    expected = {
        "response_rule_id": "first_msi_population_spike_v1",
        "response_rule": ENDPOINT_RESPONSE_RULE,
        "threshold": ENDPOINT_THRESHOLD,
        "latency_formula": ENDPOINT_FORMULA,
        "latency_domain_ms": f"0 < latency <= {format(horizon, '.17g')}",
        "resolution_ms": 0.1,
        "interpretation": ENDPOINT_INTERPRETATION,
        "sensitivity_horizons_ms": [300.0, 350.0, 400.0],
    }
    if dict(endpoint) != expected or not math.isclose(dt_ms, 0.1, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("endpoint semantics do not match the frozen first-spike contract")


def raw_payload_sha256(record: Mapping[str, Any]) -> str:
    """Hash a raw record after removing only its self-referential checksum."""

    payload = copy.deepcopy(dict(record))
    integrity = payload.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("raw_payload_sha256", None)
    return _canonical_sha256(payload)


def _attach_derived_checksum(result: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(result)
    payload.pop("derived_payload_sha256", None)
    result["derived_payload_sha256"] = _canonical_sha256(payload)
    return result


def restore_trial_sample(block: Mapping[str, Any], expected_horizon_ms: float) -> TrialSample:
    """Restore strict ``+Inf`` serialization and enforce ``0 < latency <= H``."""

    block = _require_mapping(block, "condition")
    raw_latency = _require_sequence(block.get("latency_ms"), "condition.latency_ms")
    raw_hit = _require_sequence(block.get("hit"), "condition.hit")
    if not raw_latency or len(raw_latency) != len(raw_hit):
        raise ValueError("latency_ms and hit must have the same non-zero length")
    n_trials = block.get("n_trials")
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials != len(raw_latency):
        raise ValueError("condition.n_trials must be an integer matching serialized trials")
    censor_ms = float(block.get("censor_ms"))
    if not math.isfinite(censor_ms) or censor_ms <= 0:
        raise ValueError("condition.censor_ms must be finite and positive")
    if not math.isclose(censor_ms, expected_horizon_ms, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("condition censor horizon disagrees with measurement horizon")

    latency = np.empty(n_trials, dtype=np.float64)
    hit = np.empty(n_trials, dtype=np.bool_)
    for index, (value, flag) in enumerate(zip(raw_latency, raw_hit)):
        if not isinstance(flag, bool):
            raise ValueError(f"hit[{index}] must be a JSON boolean")
        hit[index] = flag
        if flag:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"hit latency_ms[{index}] must be numeric")
            restored = float(value)
            if not math.isfinite(restored) or restored <= 0 or restored > censor_ms:
                raise ValueError(f"hit latency_ms[{index}] must satisfy 0 < latency <= censor_ms")
            latency[index] = restored
        else:
            if value != "+Inf":
                raise ValueError(f"censored latency_ms[{index}] must be the string '+Inf'")
            latency[index] = np.inf
    n_hits = int(hit.sum())
    if not isinstance(block.get("n_hits"), int) or int(block["n_hits"]) != n_hits:
        raise ValueError("condition.n_hits disagrees with hit flags")
    return TrialSample(latency_ms=latency, hit=hit, censor_ms=censor_ms)


def empirical_cdf(sample: TrialSample, support_ms: np.ndarray) -> np.ndarray:
    """Evaluate an ECDF whose denominator includes every censored trial."""

    support = np.asarray(support_ms, dtype=np.float64)
    if support.ndim != 1 or support.size == 0 or np.any(np.diff(support) < 0):
        raise ValueError("support_ms must be a non-empty sorted vector")
    finite = np.sort(sample.latency_ms[np.isfinite(sample.latency_ms)])
    return np.searchsorted(finite, support, side="right") / float(sample.n_trials)


def exact_support_ms(samples: Iterable[TrialSample], horizon_ms: float) -> np.ndarray:
    """Return the exact union of zero, the horizon, and finite event times."""

    arrays = [np.asarray([0.0, float(horizon_ms)], dtype=np.float64)]
    arrays.extend(sample.latency_ms[np.isfinite(sample.latency_ms)] for sample in samples)
    support = np.unique(np.concatenate(arrays))
    if support[0] < 0 or support[-1] > horizon_ms:
        raise ValueError("finite latency lies outside the analysis horizon")
    return support


def exact_step_quantile(support_ms: np.ndarray, cdf: np.ndarray, probability: float) -> float:
    """Return ``inf{t:F(t)>=p}`` on exact step support without interpolation."""

    if not 0 < probability <= 1:
        raise ValueError("probability must lie in (0, 1]")
    indices = np.flatnonzero(np.asarray(cdf) >= probability)
    return float(support_ms[indices[0]]) if indices.size else math.inf


def step_value(support_ms: np.ndarray, values: np.ndarray, time_ms: float) -> float:
    """Evaluate a right-continuous step function at one finite time."""

    if not math.isfinite(time_ms):
        raise ValueError("step evaluation time must be finite")
    index = int(np.searchsorted(support_ms, time_ms, side="right") - 1)
    return 0.0 if index < 0 else float(values[index])


def integrate_step(
    support_ms: np.ndarray,
    values: np.ndarray,
    lower_ms: float,
    upper_ms: float,
    *,
    positive_part: bool,
) -> float:
    """Integrate an exact right-continuous step function over a finite window."""

    if not (math.isfinite(lower_ms) and math.isfinite(upper_ms)):
        raise ValueError("integration bounds must be finite")
    if upper_ms < lower_ms:
        raise ValueError("integration upper bound precedes lower bound")
    total = 0.0
    for left, right, value in zip(support_ms[:-1], support_ms[1:], values[:-1]):
        width = min(float(right), upper_ms) - max(float(left), lower_ms)
        if width > 0:
            integrand = max(float(value), 0.0) if positive_part else float(value)
            total += integrand * width
    return float(total)


def _encoded_time(value: float) -> float | str:
    return float(value) if math.isfinite(value) else "+Inf"


def _finite_latency_summary(sample: TrialSample) -> dict[str, float | int | None]:
    finite = sample.latency_ms[sample.hit]
    return {
        "n_trials": sample.n_trials,
        "n_hits": int(sample.hit.sum()),
        "hit_rate": float(sample.hit.mean()),
        "censor_rate": float(1.0 - sample.hit.mean()),
        "mean_hit_latency_ms": float(np.mean(finite)) if finite.size else None,
        "median_hit_latency_ms": float(np.median(finite)) if finite.size else None,
    }


def _rse_mean_benefit(samples: Mapping[str, TrialSample]) -> dict[str, Any]:
    means: dict[str, float | None] = {}
    for condition in ("A", "V", "AV"):
        finite = samples[condition].latency_ms[samples[condition].hit]
        means[condition] = float(np.mean(finite)) if finite.size else None
    benefit = None
    if all(means[name] is not None for name in ("A", "V", "AV")):
        benefit = min(float(means["A"]), float(means["V"])) - float(means["AV"])
    return {
        "name": "rse_mean_benefit",
        "formal_rmi": False,
        "means_ms": means,
        "benefit_ms": benefit,
        "warning": "descriptive hit-only mean; it is not a Miller inequality test",
    }


def _qc_status(samples: Mapping[str, TrialSample], quantiles_reached: bool) -> dict[str, Any]:
    rates = {name: float(samples[name].hit.mean()) for name in ("A", "V", "AV")}
    imbalance = max(rates.values()) - min(rates.values())
    catch_rate = float(samples["catch"].hit.mean())
    reasons: list[str] = []
    if all(int(samples[name].hit.sum()) == 0 for name in ("A", "V", "AV")):
        classification = "insufficient_hits"
        reasons.append("all A/V/AV trials were censored")
    elif catch_rate > MAX_CATCH_FALSE_POSITIVE_RATE:
        classification = "uninterpretable"
        reasons.append("catch false-positive rate exceeds 0.05")
    elif not quantiles_reached:
        classification = "insufficient_hits"
        reasons.append("one or more preregistered quantiles is unreachable")
    elif min(rates.values()) < MIN_FORMAL_HIT_RATE or imbalance > MAX_FORMAL_HIT_IMBALANCE:
        classification = "joint_detection_latency"
        if min(rates.values()) < MIN_FORMAL_HIT_RATE:
            reasons.append("A/V/AV hit rate below 0.95")
        if imbalance > MAX_FORMAL_HIT_IMBALANCE:
            reasons.append("A/V/AV hit-rate imbalance exceeds 0.05")
    else:
        classification = "pure_latency"
    passed = classification == "pure_latency"
    return {
        "classification": classification,
        "formal_latency_qc_pass": passed,
        "A_V_AV_hit_rates": rates,
        "max_hit_rate_imbalance": float(imbalance),
        "catch_false_positive_rate": catch_rate,
        "thresholds": {
            "minimum_A_V_AV_hit_rate": MIN_FORMAL_HIT_RATE,
            "maximum_hit_rate_imbalance": MAX_FORMAL_HIT_IMBALANCE,
            "maximum_catch_false_positive_rate": MAX_CATCH_FALSE_POSITIVE_RATE,
        },
        "reasons": reasons,
    }


def analyze_intensity_row(row: Mapping[str, Any], horizon_ms: float) -> dict[str, Any]:
    """Compute exact ECDF, quantile-gain, descriptive area, and QC summaries."""

    row = _require_mapping(row, "row")
    intensity = float(row.get("intensity"))
    if not math.isfinite(intensity) or intensity < 0:
        raise ValueError("row.intensity must be finite and non-negative")
    blocks = _require_mapping(row.get("conditions"), "row.conditions")
    required = {"A", "V", "AV", "catch"}
    if set(blocks) != required:
        raise ValueError(f"row.conditions must contain exactly {sorted(required)}")
    samples = {
        name: restore_trial_sample(_require_mapping(blocks[name], name), horizon_ms)
        for name in sorted(required)
    }
    if len({sample.n_trials for sample in samples.values()}) != 1:
        raise ValueError("A, V, AV, and catch must have balanced trial counts")

    support = exact_support_ms((samples["A"], samples["V"], samples["AV"]), horizon_ms)
    f_a = empirical_cdf(samples["A"], support)
    f_v = empirical_cdf(samples["V"], support)
    f_av = empirical_cdf(samples["AV"], support)
    bound = np.minimum(f_a + f_v, 1.0)
    independent = f_a + f_v - f_a * f_v
    difference = f_av - bound

    quantiles: list[dict[str, Any]] = []
    for q in Q_LEVELS:
        q_bound = exact_step_quantile(support, bound, q)
        q_av = exact_step_quantile(support, f_av, q)
        reached = math.isfinite(q_bound) and math.isfinite(q_av)
        quantiles.append(
            {
                "q": q,
                "Q_bound_ms": _encoded_time(q_bound),
                "Q_AV_ms": _encoded_time(q_av),
                "G_ms": float(q_bound - q_av) if reached else None,
                "D_at_Q_bound": step_value(support, difference, q_bound)
                if math.isfinite(q_bound)
                else None,
            }
        )
    quantiles_reached = all(entry["G_ms"] is not None for entry in quantiles)
    q_low = exact_step_quantile(support, bound, Q_LEVELS[0])
    q_high = exact_step_quantile(support, bound, Q_LEVELS[-1])

    full_width = float(horizon_ms)
    full_positive = integrate_step(support, difference, 0.0, horizon_ms, positive_part=True)
    full_signed = integrate_step(support, difference, 0.0, horizon_ms, positive_part=False)
    if math.isfinite(q_low) and math.isfinite(q_high):
        early_width: float | None = float(q_high - q_low)
        early_positive: float | None = integrate_step(
            support, difference, q_low, q_high, positive_part=True
        )
        early_signed: float | None = integrate_step(
            support, difference, q_low, q_high, positive_part=False
        )
        early_normalized = early_positive / early_width if early_width > 0 else None
    else:
        early_width = early_positive = early_signed = early_normalized = None

    max_value = float(np.max(difference))
    max_time = float(support[np.flatnonzero(difference == max_value)[0]])
    positive_intervals = [
        (float(left), float(right))
        for left, right, value in zip(support[:-1], support[1:], difference[:-1])
        if value > 0 and right > left
    ]
    positive_q = [entry["q"] for entry in quantiles if entry["G_ms"] is not None and entry["G_ms"] > 0]
    qc = _qc_status(samples, quantiles_reached)
    return {
        "intensity": intensity,
        "soa_ms": 0.0,
        "formal_rmi": {
            "primary_statistic": "G(q)=Q_bound(q)-Q_AV(q)",
            "miller_bound": "min(F_A(t)+F_V(t),1)",
            "q_levels": list(Q_LEVELS),
            "eligible_for_checkpoint_inference": bool(qc["formal_latency_qc_pass"] and quantiles_reached),
        },
        "independent_race": {"definition": "F_A+F_V-F_A*F_V", "role": "comparator only"},
        "qc": {
            "conditions": {
                name: _finite_latency_summary(samples[name]) for name in ("A", "V", "AV", "catch")
            },
            **qc,
        },
        "rse_mean_benefit": _rse_mean_benefit(samples),
        "exact_support": {
            "time_ms": support.tolist(),
            "F_A": f_a.tolist(),
            "F_V": f_v.tolist(),
            "F_AV": f_av.tolist(),
            "miller_bound": bound.tolist(),
            "independent_race_cdf": independent.tolist(),
            "D": difference.tolist(),
        },
        "quantile_summaries": quantiles,
        "descriptive_windows": {
            "early": {
                "definition": "[Q_bound(.05),Q_bound(.35)]",
                "lower_ms": _encoded_time(q_low),
                "upper_ms": _encoded_time(q_high),
                "window_width_ms": early_width,
                "positive_area_ms": early_positive,
                "positive_area_normalized": early_normalized,
                "signed_area_ms": early_signed,
                "inferential_role": False,
            },
            "full": {
                "definition": "[0,H]",
                "lower_ms": 0.0,
                "upper_ms": float(horizon_ms),
                "window_width_ms": full_width,
                "positive_area_ms": full_positive,
                "positive_area_normalized": full_positive / full_width,
                "signed_area_ms": full_signed,
                "inferential_role": False,
            },
        },
        "descriptive_shape": {
            "max_D": max_value,
            "max_D_time_ms": max_time,
            "positive_width_ms": float(sum(right - left for left, right in positive_intervals)),
            "first_positive_time_ms": positive_intervals[0][0] if positive_intervals else None,
            "last_positive_time_ms": positive_intervals[-1][1] if positive_intervals else None,
            "first_positive_q": positive_q[0] if positive_q else None,
            "last_positive_q": positive_q[-1] if positive_q else None,
        },
    }


def validate_raw_integrity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute all frozen/one-factor attestations; never trust summary booleans."""

    measurement = _require_mapping(record.get("measurement"), "measurement")
    _validate_endpoint(measurement)
    condition = _require_mapping(record.get("condition"), "condition")
    baseline = _require_mapping(condition.get("baseline"), "condition.baseline")
    requested = _require_mapping(condition.get("requested"), "condition.requested")
    if set(baseline) != set(BASELINE_MECHANISMS) or set(requested) != set(BASELINE_MECHANISMS):
        raise ValueError("baseline/requested mechanism keysets are not exact")
    for name, expected in BASELINE_MECHANISMS.items():
        if not math.isfinite(float(baseline[name])) or not math.isclose(
            float(baseline[name]), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"baseline mechanism mismatch: {name}")
    design_family = str(condition.get("design_family"))
    cell = registered_mechanism_cell(requested, design_family)
    changed = list(cell["changed_scalars"])
    target = changed[0] if len(changed) == 1 else None
    if (
        condition.get("changed_scalar") != target
        or list(condition.get("changed_scalars", [])) != changed
        or condition.get("label") != cell["cell_id"]
        or condition.get("cell_id") != cell["cell_id"]
        or condition.get("factor_id") != cell["factor_id"]
        or condition.get("one_factor") is not cell["scalar_one_factor"]
    ):
        raise ValueError("condition declarations disagree with the registered mechanism cell")
    manifest = _require_mapping(condition.get("mechanism_manifest"), "mechanism_manifest")
    manifest_core = {key: manifest.get(key) for key in cell}
    if manifest.get("version") != "dm10-registered-mechanisms-v2" or manifest_core != cell:
        raise ValueError("mechanism manifest does not exactly match the registered cell")

    loaded = _require_mapping(record.get("loaded_effective"), "loaded_effective")
    effective = _require_mapping(record.get("effective"), "effective")
    effective_after = _require_mapping(record.get("effective_after"), "effective_after")
    for name in BASELINE_MECHANISMS:
        if not math.isclose(float(loaded[name]), float(baseline[name]), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"loaded effective scalar is not baseline: {name}")
        if not math.isclose(float(effective[name]), float(requested[name]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"effective scalar is not requested value: {name}")
        if not math.isclose(
            float(effective_after[name]), float(effective[name]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"effective scalar changed during acquisition: {name}")
        if name not in changed and not math.isclose(
            float(effective[name]), float(baseline[name]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"non-target scalar changed: {name}")

    integrity = _require_mapping(record.get("integrity"), "integrity")
    state_values = [
        integrity.get("state_sha256_loaded"),
        integrity.get("state_sha256_before"),
        integrity.get("state_sha256_after"),
    ]
    state_equal = all(_is_hex_digest(value, 64) for value in state_values) and len(set(state_values)) == 1
    readout_before = list(_require_sequence(integrity.get("readout_md5_before"), "readout before"))
    readout_after = list(_require_sequence(integrity.get("readout_md5_after"), "readout after"))
    readouts_equal = (
        len(readout_before) > 0
        and all(_is_hex_digest(value, 32) for value in readout_before + readout_after)
        and readout_before == readout_after
    )
    if integrity.get("state_bit_identical") is not state_equal or not state_equal:
        raise ValueError("state hashes/summary do not prove bit-identical frozen inference")
    if integrity.get("readouts_unchanged") is not readouts_equal or not readouts_equal:
        raise ValueError("readout hashes/summary do not prove frozen readouts")
    if integrity.get("plasticity_enabled") is not False:
        raise ValueError("plasticity must be disabled")
    stored_raw_hash = integrity.get("raw_payload_sha256")
    if not _is_hex_digest(stored_raw_hash, 64) or stored_raw_hash != raw_payload_sha256(record):
        raise ValueError("raw payload checksum mismatch")
    configuration = _require_mapping(record.get("configuration"), "configuration")
    if not _is_hex_digest(record.get("configuration_sha256"), 64) or record.get(
        "configuration_sha256"
    ) != _canonical_sha256(configuration):
        raise ValueError("configuration checksum mismatch")
    if set(configuration) != {
        "baseline",
        "requested",
        "effective",
        "environment",
        "intensities",
        "n_trials",
        "soa_ms",
        "horizon_ms",
    }:
        raise ValueError("configuration manifest keyset is not exact")
    if (
        dict(configuration["baseline"]) != dict(baseline)
        or dict(configuration["requested"]) != dict(requested)
        or any(
            not math.isclose(
                float(configuration["effective"][name]),
                float(effective[name]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name in BASELINE_MECHANISMS
        )
        or list(configuration["intensities"]) != list(measurement["intensities"])
        or int(configuration["n_trials"]) != int(measurement["n_trials_per_condition"])
        or float(configuration["soa_ms"]) != float(measurement["soa_ms"])
        or float(configuration["horizon_ms"]) != float(measurement["horizon_ms"])
    ):
        raise ValueError("configuration manifest disagrees with effective measurement metadata")

    checkpoint = _require_mapping(record.get("checkpoint"), "checkpoint")
    if not _is_hex_digest(checkpoint.get("sha256"), 64):
        raise ValueError("checkpoint SHA-256 is missing or malformed")
    code = _require_mapping(record.get("code"), "code")
    sources = _require_mapping(code.get("source_sha256"), "code.source_sha256")
    if set(sources) != {"driver", "network_io", "latency_module"} or not all(
        _is_hex_digest(value, 64) for value in sources.values()
    ):
        raise ValueError("source hashes are incomplete")
    if not _is_hex_digest(code.get("driver_sha256"), 64) or code.get(
        "driver_sha256"
    ) != sources["driver"]:
        raise ValueError("driver SHA-256 disagrees with acquisition source manifest")
    git = _require_mapping(code.get("git"), "code.git")
    if not _is_hex_digest(git.get("head"), 40) or not isinstance(git.get("branch"), str):
        raise ValueError("git provenance is incomplete")
    device = _require_mapping(record.get("device"), "device")
    return {
        "pass": True,
        "design_family": design_family,
        "cell_id": cell["cell_id"],
        "factor_id": cell["factor_id"],
        "derived_changed_scalars": changed,
        "state_hash_reconciled": True,
        "readout_hashes_reconciled": True,
        "one_factor_reconciled": True,
        "raw_payload_checksum_reconciled": True,
        "raw_provenance": {
            "raw_payload_sha256": stored_raw_hash,
            "configuration_sha256": record["configuration_sha256"],
            "configuration": copy.deepcopy(dict(configuration)),
            "checkpoint_sha256": checkpoint["sha256"],
            "acquisition_source_sha256": copy.deepcopy(dict(sources)),
            "acquisition_driver_sha256": code["driver_sha256"],
            "acquisition_git": copy.deepcopy(dict(git)),
            "device": copy.deepcopy(dict(device)),
        },
    }


def _evaluate_acquisition_design(
    record: Mapping[str, Any], analyses: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    measurement = _require_mapping(record.get("measurement"), "measurement")
    checkpoint = _require_mapping(record.get("checkpoint"), "checkpoint")
    condition_manifest = _require_mapping(record.get("condition"), "condition")
    code = _require_mapping(record.get("code"), "code")
    endpoint = _require_mapping(measurement.get("endpoint"), "measurement.endpoint")
    try:
        rng_manifest_pass = (
            list(measurement["condition_order"]) == list(CONDITION_ORDER)
            and measurement["stream_seed_derivation"] == STREAM_SEED_DERIVATION
        )
        expected_trial_ids = [
            f"s{substream:02d}:t{trial:03d}"
            for substream in range(PRIMARY_SUBSTREAMS)
            for trial in range(TRIALS_PER_SUBSTREAM)
        ]
        for row in _require_sequence(record.get("rows"), "rows"):
            intensity = float(row["intensity"])
            for condition in ("A", "V", "AV", "catch"):
                block = row["conditions"][condition]
                rng = block["rng"]
                rng_manifest_pass = rng_manifest_pass and set(rng) == {
                    "algorithm",
                    "stable_cell_key",
                    "substreams",
                }
                rng_manifest_pass = rng_manifest_pass and rng["algorithm"] == RNG_ALGORITHM
                stable_key = rng["stable_cell_key"]
                substreams = rng["substreams"]
                rng_manifest_pass = rng_manifest_pass and block["trial_id"] == expected_trial_ids
                rng_manifest_pass = rng_manifest_pass and stable_key == {
                    "checkpoint_seed": int(checkpoint["seed"]),
                    "intensity": intensity,
                    "sensory_condition": condition,
                }
                rng_manifest_pass = rng_manifest_pass and len(substreams) == PRIMARY_SUBSTREAMS
                for substream_index, substream in enumerate(substreams):
                    rng_manifest_pass = rng_manifest_pass and set(substream) == {
                        "index",
                        "stream_seed",
                        "trial_start",
                        "trial_stop_exclusive",
                        "state_before_sha256",
                        "state_after_sha256",
                    }
                    payload = (
                        f"{int(measurement['run_seed'])}:{int(checkpoint['seed'])}:"
                        f"{format(intensity, '.17g')}:{condition}:{substream_index}"
                    ).encode("ascii")
                    expected_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
                        2**31 - 1
                    )
                    rng_manifest_pass = rng_manifest_pass and substream["index"] == substream_index
                    rng_manifest_pass = rng_manifest_pass and substream["stream_seed"] == expected_seed
                    rng_manifest_pass = rng_manifest_pass and substream["trial_start"] == (
                        substream_index * TRIALS_PER_SUBSTREAM
                    )
                    rng_manifest_pass = rng_manifest_pass and substream[
                        "trial_stop_exclusive"
                    ] == ((substream_index + 1) * TRIALS_PER_SUBSTREAM)
                    rng_manifest_pass = rng_manifest_pass and _is_hex_digest(
                        substream["state_before_sha256"], 64
                    )
                    rng_manifest_pass = rng_manifest_pass and _is_hex_digest(
                        substream["state_after_sha256"], 64
                    )
    except (KeyError, TypeError, ValueError):
        rng_manifest_pass = False
    checks = {
        "checkpoint_seed_42_to_51": int(checkpoint.get("seed")) in PRIMARY_SEEDS,
        "exact_registered_cell": condition_manifest.get("design_family")
        in MECHANISM_CELL_REGISTRY
        and condition_manifest.get("cell_id")
        in MECHANISM_CELL_REGISTRY[condition_manifest.get("design_family")],
        "seven_intensity_grid": tuple(float(row["intensity"]) for row in analyses)
        == PRIMARY_INTENSITIES,
        "balanced_1000_trials": all(
            summary["n_trials"] == PRIMARY_N_TRIALS
            for row in analyses
            for summary in row["qc"]["conditions"].values()
        ),
        "soa_zero_ms": float(measurement.get("soa_ms")) == 0.0,
        "horizon_400_ms": float(measurement.get("horizon_ms")) == PRIMARY_HORIZON_MS,
        "ten_substreams_of_100": measurement.get("substream_count") == PRIMARY_SUBSTREAMS
        and measurement.get("trials_per_substream") == TRIALS_PER_SUBSTREAM,
        "stable_label_rng_and_trial_ids": rng_manifest_pass,
        "fixed_q_grid": Q_LEVELS == tuple(value / 100.0 for value in range(5, 36, 5)),
        "endpoint_first_spike": dict(endpoint)
        == {
            "response_rule_id": "first_msi_population_spike_v1",
            "response_rule": ENDPOINT_RESPONSE_RULE,
            "threshold": ENDPOINT_THRESHOLD,
            "latency_formula": ENDPOINT_FORMULA,
            "latency_domain_ms": "0 < latency <= 400",
            "resolution_ms": 0.1,
            "interpretation": ENDPOINT_INTERPRETATION,
            "sensitivity_horizons_ms": [300.0, 350.0, 400.0],
        },
        "clean_git_checkout": _require_mapping(code.get("git"), "code.git").get("clean") is True,
        "checkpoint_load_exact": list(checkpoint.get("load_missing_keys", [])) == []
        and list(checkpoint.get("load_unexpected_keys", [])) == [],
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {"pass": not failed, "checks": checks, "failed_checks": failed}


def analyze_acquisition(record: Mapping[str, Any]) -> dict[str, Any]:
    """Analyze one raw acquisition after strict integrity reconciliation."""

    record = _require_mapping(record, "record")
    if record.get("schema_version") != ACQUISITION_SCHEMA:
        raise ValueError(f"unsupported acquisition schema: {record.get('schema_version')!r}")
    integrity = validate_raw_integrity(record)
    measurement = _require_mapping(record.get("measurement"), "measurement")
    horizon_ms = float(measurement.get("horizon_ms"))
    if not math.isfinite(horizon_ms) or horizon_ms <= 0:
        raise ValueError("measurement.horizon_ms must be finite and positive")
    if float(measurement.get("soa_ms")) != 0.0:
        raise ValueError("canonical primary race analysis requires SOA=0 ms")
    rows = _require_sequence(record.get("rows"), "rows")
    analyses = [analyze_intensity_row(_require_mapping(row, "row"), horizon_ms) for row in rows]
    intensities = [row["intensity"] for row in analyses]
    if len(set(intensities)) != len(intensities):
        raise ValueError("acquisition contains duplicate intensity rows")
    declared = [
        float(value)
        for value in _require_sequence(measurement.get("intensities"), "measurement.intensities")
    ]
    if intensities != declared:
        raise ValueError("row intensities/order disagree with measurement.intensities")
    checkpoint = _require_mapping(record.get("checkpoint"), "checkpoint")
    condition = _require_mapping(record.get("condition"), "condition")
    result = {
        "schema_version": ANALYSIS_SCHEMA,
        "source_schema": ACQUISITION_SCHEMA,
        "condition_label": str(condition.get("label")),
        "design_family": condition.get("design_family"),
        "cell_id": condition.get("cell_id"),
        "factor_id": condition.get("factor_id"),
        "changed_scalar": condition.get("changed_scalar"),
        "changed_scalars": list(condition.get("changed_scalars", [])),
        "checkpoint_seed": int(checkpoint.get("seed")),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "horizon_ms": horizon_ms,
        "q_levels": list(Q_LEVELS),
        "integrity": integrity,
        "input_provenance": copy.deepcopy(integrity["raw_provenance"]),
        "analyzer_provenance": analyzer_provenance(),
        "design": _evaluate_acquisition_design(record, analyses),
        "inference_unit": "checkpoint; trials and substreams are nested and never pooled",
        "endpoint": measurement.get("endpoint"),
        "execution_limitations": {
            "nested_trial_checkpoint_bootstrap": "not_implemented",
            "alternate_endpoint_execution": "not_implemented",
            "common_horizon_sensitivity_execution": "not_implemented",
            "directional_mechanism_matrix_runner": "not_implemented",
        },
        "disclaimer": DISCLAIMER,
        "intensities": analyses,
    }
    return _attach_derived_checksum(result)


def _one_sample_t(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=0)
    sd = np.std(values, axis=0, ddof=1)
    se = sd / math.sqrt(values.shape[0])
    statistic = np.divide(mean, se, out=np.zeros_like(mean), where=se > 0)
    statistic[(se == 0) & (mean > 0)] = np.inf
    statistic[(se == 0) & (mean < 0)] = -np.inf
    return statistic


def _residual_studentized(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=0)
    se = np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])
    return np.divide(mean, se, out=np.zeros_like(mean), where=se > 0)


def _permutation_exceeds_or_ties(
    null_statistic: np.ndarray | float, observed_statistic: np.ndarray | float
) -> np.ndarray:
    """Conservatively compare permutation statistics with stable tie handling.

    Finite statistics within 64 scaled binary64 epsilons are treated as ties.
    This tolerance is deliberately narrow: it covers last-bit differences from
    equivalent NumPy memory layouts without changing genuinely ordered values.
    Infinities retain exact IEEE ordering, and NaNs never count as exceedances.
    """

    null = np.asarray(null_statistic, dtype=np.float64)
    observed = np.asarray(observed_statistic, dtype=np.float64)
    null, observed = np.broadcast_arrays(null, observed)
    result = np.greater_equal(null, observed)
    finite = np.isfinite(null) & np.isfinite(observed)
    scale = np.maximum(1.0, np.maximum(np.abs(null), np.abs(observed)))
    tolerance = PERMUTATION_TIE_EPS_MULTIPLIER * np.finfo(np.float64).eps * scale
    difference = np.zeros_like(scale)
    np.subtract(null, observed, out=difference, where=finite)
    result |= finite & (np.abs(difference) <= tolerance)
    result &= ~np.isnan(null) & ~np.isnan(observed)
    return result


def _sign_patterns(n_checkpoints: int, random_seed: int, maximum: int) -> tuple[np.ndarray, bool]:
    if n_checkpoints <= 16:
        return np.asarray(list(itertools.product((-1.0, 1.0), repeat=n_checkpoints))), True
    rng = np.random.default_rng(random_seed)
    signs = rng.choice((-1.0, 1.0), size=(maximum, n_checkpoints))
    signs[0, :] = 1.0
    return signs, False


def checkpoint_sign_flip_max_stat(
    checkpoint_values: np.ndarray,
    cell_labels: Sequence[Mapping[str, Any]],
    *,
    random_seed: int = 20260712,
    max_random_flips: int = 100_000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Exact raw global sign-flip p values plus an approximate max-T lower band."""

    values = np.asarray(checkpoint_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("checkpoint_values must have shape [at least 2, at least 1]")
    if not np.all(np.isfinite(values)) or len(cell_labels) != values.shape[1]:
        raise ValueError("checkpoint matrix/labels are malformed")
    signs, exact = _sign_patterns(values.shape[0], random_seed, max_random_flips)
    raw_null_t = np.empty((len(signs), values.shape[1]), dtype=np.float64)
    for index, sign in enumerate(signs):
        raw_null_t[index] = _one_sample_t(values * sign[:, None])
    observed_t = _one_sample_t(values)
    raw_null_max = np.max(raw_null_t, axis=1)
    add_one = 0 if exact else 1
    denominator = len(signs) if exact else len(signs) + 1

    means = np.mean(values, axis=0)
    se = np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])
    residuals = values - means
    residual_max = np.empty(len(signs), dtype=np.float64)
    for index, sign in enumerate(signs):
        residual_max[index] = float(
            np.max(_residual_studentized(residuals * sign[:, None]))
        )
    order = max(0, int(math.ceil(confidence * len(residual_max))) - 1)
    critical = float(np.sort(residual_max)[order])
    achieved = float(np.mean(residual_max <= critical))
    lower = means - critical * se
    observed_global = float(np.max(observed_t))
    global_p = (
        int(np.count_nonzero(_permutation_exceeds_or_ties(raw_null_max, observed_global)))
        + add_one
    ) / denominator

    cells = []
    for column, label in enumerate(cell_labels):
        uncorrected = (
            int(
                np.count_nonzero(
                    _permutation_exceeds_or_ties(
                        raw_null_t[:, column], observed_t[column]
                    )
                )
            )
            + add_one
        ) / denominator
        corrected = (
            int(
                np.count_nonzero(
                    _permutation_exceeds_or_ties(raw_null_max, observed_t[column])
                )
            )
            + add_one
        ) / denominator
        cells.append(
            {
                **dict(label),
                "checkpoint_mean_G_ms": float(means[column]),
                "checkpoint_se_G_ms": float(se[column]),
                "checkpoint_t": _json_safe_float(float(observed_t[column])),
                "p_one_sided_uncorrected": float(uncorrected),
                "p_one_sided_global_max_stat": float(corrected),
                "approx_simultaneous_lower_G_ms": float(lower[column]),
            }
        )
    return {
        "status": "complete",
        "method": "formal_rmi quantile-gain checkpoint sign-flip global max-T",
        "primary_statistic": "G(q)=Q_bound(q)-Q_AV(q), milliseconds",
        "unit": "checkpoint; no trial pooling",
        "tail": "positive",
        "n_checkpoints": int(values.shape[0]),
        "n_family_cells": int(values.shape[1]),
        "n_sign_patterns": int(len(signs)),
        "exact_raw_sign_flip": exact,
        "global_p_one_sided_max_stat": float(global_p),
        "simultaneous_lower_band": {
            "method": "centered-residual rowwise sign-flip studentized max-T",
            "approximate": True,
            "symmetry_assumption": "checkpoint residual vectors are rowwise sign-symmetric",
            "zero_se_residual_statistic": 0.0,
            "critical_value": critical,
            "nominal_coverage": confidence,
            "achieved_discrete_coverage": achieved,
            "n_sign_patterns": int(len(signs)),
        },
        "cells": cells,
    }


def _json_safe_float(value: float) -> float | str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if math.isnan(value):
        raise ValueError("analysis produced NaN")
    return float(value)


def _g_lookup(intensity_analysis: Mapping[str, Any]) -> dict[float, float | None]:
    return {
        float(entry["q"]): None if entry["G_ms"] is None else float(entry["G_ms"])
        for entry in intensity_analysis["quantile_summaries"]
    }


def _family_match_checks(
    records: Sequence[Mapping[str, Any]], labels: Sequence[str], seeds: Sequence[int]
) -> dict[str, Any]:
    by_raw = {
        (str(record["condition"]["label"]), int(record["checkpoint"]["seed"])): record
        for record in records
    }
    checkpoint_matched = all(
        len({by_raw[(label, seed)]["checkpoint"]["sha256"] for label in labels}) == 1
        for seed in seeds
    )
    horizon_matched = len({float(record["measurement"]["horizon_ms"]) for record in records}) == 1
    endpoint_matched = len(
        {_canonical_sha256(record["measurement"]["endpoint"]) for record in records}
    ) == 1
    source_matched = len({_canonical_sha256(record["code"]["source_sha256"]) for record in records}) == 1
    git_head_matched = len({record["code"]["git"]["head"] for record in records}) == 1
    measurement_matched = len({_canonical_sha256(record["measurement"]) for record in records}) == 1
    environment_matched = len(
        {_canonical_sha256(record["configuration"]["environment"]) for record in records}
    ) == 1
    loaded_config_matched = len({_canonical_sha256(record["loaded_effective"]) for record in records}) == 1
    return {
        "matched_checkpoint_sha_by_seed": checkpoint_matched,
        "common_horizon": horizon_matched,
        "common_endpoint": endpoint_matched,
        "common_source_hashes": source_matched,
        "common_git_head": git_head_matched,
        "common_measurement_manifest": measurement_matched,
        "common_environment": environment_matched,
        "common_loaded_configuration": loaded_config_matched,
    }


def _not_run_inference(reason: str) -> dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        "formal_rmi_violation": False,
        "cells": [],
    }


def formal_claim_gate(
    corrected_p: float,
    simultaneous_lower_g_ms: float,
    *,
    qc_pass: bool,
    integrity_pass: bool,
    design_pass: bool,
) -> bool:
    """Return the preregistered conjunction required for a formal RMI claim."""

    return bool(
        math.isfinite(corrected_p)
        and corrected_p < 0.05
        and math.isfinite(simultaneous_lower_g_ms)
        and simultaneous_lower_g_ms > 0
        and qc_pass
        and integrity_pass
        and design_pass
    )


def analyze_family(
    records: Sequence[Mapping[str, Any]], *, baseline_label: str = "baseline"
) -> dict[str, Any]:
    """Perform checkpoint-level global inference and conjunctive formal gating."""

    if not records:
        raise ValueError("at least one acquisition record is required")
    analyses = [analyze_acquisition(record) for record in records]
    design_families = {str(item["design_family"]) for item in analyses}
    if len(design_families) != 1:
        raise ValueError("mechanism design families cannot be mixed in one inference")
    design_family = next(iter(design_families))
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for item in analyses:
        key = (str(item["condition_label"]), int(item["checkpoint_seed"]))
        if key in by_key:
            raise ValueError(f"duplicate condition/checkpoint acquisition: {key}")
        by_key[key] = item
    labels = sorted({key[0] for key in by_key})
    if baseline_label not in labels and design_family != HISTORICAL_ADAPTATION_FAMILY:
        raise ValueError(f"baseline label {baseline_label!r} is absent")
    checkpoint_sets = {label: {seed for cell, seed in by_key if cell == label} for label in labels}
    shared_seeds = set.intersection(*checkpoint_sets.values())
    if len(shared_seeds) < 2:
        raise ValueError("family diagnostics require at least two shared checkpoint seeds")
    seeds = sorted(shared_seeds)
    analysis_anchor = baseline_label if baseline_label in labels else labels[0]
    match_checks = _family_match_checks(records, labels, seeds)
    if not all(match_checks.values()):
        raise ValueError(f"family provenance/config mismatch: {match_checks}")

    intensity_sets: dict[str, tuple[float, ...]] = {}
    for label in labels:
        grids = [
            tuple(float(row["intensity"]) for row in by_key[(label, seed)]["intensities"])
            for seed in seeds
        ]
        if any(grid != grids[0] for grid in grids[1:]):
            raise ValueError(f"intensity grid varies across checkpoints for {label!r}")
        intensity_sets[label] = grids[0]

    release = analyzer_provenance()
    release_policy = _analysis_release_policy(records, analyses, release)
    release_checks = {
        "analyzer_sha256_valid": release_policy["checks"]["analyzer_sha256_valid"],
        "analyzer_tracked": release_policy["checks"]["analyzer_tracked"],
        "analyzer_local_matches_head": release_policy["checks"][
            "analyzer_local_matches_head"
        ],
        "analyzer_git_clean": release_policy["checks"]["analyzer_git_clean"],
        "analyzer_revision_matches_acquisition": release_policy["pass"],
        "analysis_hotfix_provenance_valid": release_policy["pass"],
    }
    common_checks = {
        "checkpoint_sets_exact_42_to_51": all(
            seeds_for_label == set(PRIMARY_SEEDS) for seeds_for_label in checkpoint_sets.values()
        ),
        "all_acquisitions_pass_design": all(item["design"]["pass"] for item in analyses),
        "all_cells_full_intensity_grid": all(
            grid == PRIMARY_INTENSITIES for grid in intensity_sets.values()
        ),
        **match_checks,
        **release_checks,
    }
    if design_family == HISTORICAL_ADAPTATION_FAMILY:
        manifest_name = "dm10-historical-adaptation-replication-v1"
        family_specific_checks = {
            "exact_historical_label_set": set(labels)
            == {"adaptation_off", "adaptation_prior", "adaptation_shipped"},
            "shipped_anchor": baseline_label == "adaptation_shipped",
            "compound_cells_not_scalar_attribution": all(
                item["factor_id"] == "compound_adaptation_replication" for item in analyses
            ),
        }
    else:
        manifest_name = "dm10-formal-rmi-baseline-v1"
        family_specific_checks = {
            "scalar_baseline_anchor": baseline_label == "baseline" and "baseline" in labels,
            "all_labels_scalar_registered": all(
                label in MECHANISM_CELL_REGISTRY[SCALAR_RMI_FAMILY] for label in labels
            ),
        }
    design_checks = {**common_checks, **family_specific_checks}
    design_pass = all(design_checks.values())
    family_design = {
        "manifest": manifest_name,
        "design_family": design_family,
        "anchor_label": baseline_label,
        "pass": design_pass,
        "checks": design_checks,
        "failed_checks": [name for name, passed in design_checks.items() if not passed],
        "primary_intensity": 0.05,
        "family_intensities": list(PRIMARY_INTENSITIES),
        "q_levels": list(Q_LEVELS),
        "n_trials_per_cell": PRIMARY_N_TRIALS,
        "soa_ms": 0.0,
        "horizon_ms": PRIMARY_HORIZON_MS,
        "analyzer_release": release,
        "analysis_hotfix_provenance": release_policy,
    }

    condition_cells: list[dict[str, Any]] = []
    condition_columns: list[np.ndarray] = []
    condition_qc: list[bool] = []
    missing_g = False
    for label in labels:
        for intensity in intensity_sets[label]:
            for q in Q_LEVELS:
                values: list[float] = []
                qc_pass = True
                for seed in seeds:
                    rows = {
                        float(row["intensity"]): row for row in by_key[(label, seed)]["intensities"]
                    }
                    value = _g_lookup(rows[intensity])[q]
                    if value is None:
                        missing_g = True
                    else:
                        values.append(value)
                    qc_pass = qc_pass and bool(rows[intensity]["qc"]["formal_latency_qc_pass"])
                condition_cells.append({"condition": label, "intensity": intensity, "q": q})
                condition_qc.append(qc_pass)
                if len(values) == len(seeds):
                    condition_columns.append(np.asarray(values, dtype=np.float64))
    if missing_g or len(condition_columns) != len(condition_cells):
        condition_inference = _not_run_inference(
            "insufficient_hits: at least one preregistered Q_bound or Q_AV is unreachable"
        )
    else:
        condition_inference = checkpoint_sign_flip_max_stat(
            np.column_stack(condition_columns), condition_cells
        )
        any_claim = False
        for cell, qc_pass in zip(condition_inference["cells"], condition_qc):
            cell["qc_pass_all_checkpoints"] = qc_pass
            cell["design_pass"] = design_pass
            cell["formal_rmi_violation"] = formal_claim_gate(
                cell["p_one_sided_global_max_stat"],
                cell["approx_simultaneous_lower_G_ms"],
                qc_pass=qc_pass,
                integrity_pass=True,
                design_pass=design_pass,
            )
            any_claim = any_claim or cell["formal_rmi_violation"]
        condition_inference["formal_rmi_violation"] = any_claim

    change_cells: list[dict[str, Any]] = []
    change_columns: list[np.ndarray] = []
    change_missing = False
    baseline_intensities = set(intensity_sets[analysis_anchor])
    contrast_labels = [] if design_family == HISTORICAL_ADAPTATION_FAMILY else labels
    for label in contrast_labels:
        if label == analysis_anchor:
            continue
        if not set(intensity_sets[label]).issubset(baseline_intensities):
            raise ValueError(f"baseline lacks an intensity required by {label!r}")
        for intensity in intensity_sets[label]:
            for q in Q_LEVELS:
                values = []
                for seed in seeds:
                    condition_rows = {
                        float(row["intensity"]): row for row in by_key[(label, seed)]["intensities"]
                    }
                    baseline_rows = {
                        float(row["intensity"]): row
                        for row in by_key[(analysis_anchor, seed)]["intensities"]
                    }
                    condition_g = _g_lookup(condition_rows[intensity])[q]
                    baseline_g = _g_lookup(baseline_rows[intensity])[q]
                    if condition_g is None or baseline_g is None:
                        change_missing = True
                    else:
                        values.append(condition_g - baseline_g)
                change_cells.append(
                    {"contrast": f"{label} - {analysis_anchor}", "intensity": intensity, "q": q}
                )
                if len(values) == len(seeds):
                    change_columns.append(np.asarray(values, dtype=np.float64))
    if design_family == HISTORICAL_ADAPTATION_FAMILY:
        change_inference = _not_run_inference(
            "historical compound contrast directions/tails are not preregistered; no aM-vs-dM attribution"
        )
        change_inference["role"] = "exploratory compound contrast withheld"
    elif not change_cells:
        change_inference = None
    elif change_missing or len(change_columns) != len(change_cells):
        change_inference = _not_run_inference("insufficient_hits in matched G contrast")
    else:
        change_inference = checkpoint_sign_flip_max_stat(
            np.column_stack(change_columns), change_cells
        )

    inverse_effectiveness: dict[str, Any]
    if 0.05 in baseline_intensities and 1.0 in baseline_intensities:
        differences = []
        for seed in seeds:
            rows = {
                float(row["intensity"]): row
                for row in by_key[(analysis_anchor, seed)]["intensities"]
            }
            differences.append(
                rows[0.05]["descriptive_windows"]["full"]["positive_area_ms"]
                - rows[1.0]["descriptive_windows"]["full"]["positive_area_ms"]
            )
        ie = checkpoint_sign_flip_max_stat(
            np.asarray(differences, dtype=np.float64)[:, None],
            [{"contrast": "Aplus_full(I=.05)-Aplus_full(I=1.0)"}],
        )
        inverse_effectiveness = {
            "role": "paired descriptive secondary contrast; does not gate formal_rmi",
            "directional_hypothesis": "Aplus_full(.05) > Aplus_full(1.0)",
            "inference": ie,
        }
    else:
        inverse_effectiveness = {"status": "not_run", "reason": "required intensities absent"}

    formal_claim = bool(condition_inference.get("formal_rmi_violation", False))
    historical_replication_claim = (
        {
            "status": "not_run",
            "reason": "compound contrast directions/tails are not preregistered",
            "scalar_attribution_allowed": False,
        }
        if design_family == HISTORICAL_ADAPTATION_FAMILY
        else {"status": "not_applicable"}
    )
    result = {
        "schema_version": FAMILY_SCHEMA,
        "design_family": design_family,
        "formal_rmi_violation": formal_claim,
        "claim_gate": {
            "definition": "corrected p<.05 AND approximate simultaneous lower G band>0 AND QC/integrity/design/analyzer-release pass",
            "positive_area_is_descriptive_only": True,
            "passed": formal_claim,
        },
        "baseline_label": baseline_label,
        "condition_labels": labels,
        "checkpoint_seeds": seeds,
        "design": family_design,
        "analysis_hotfix_provenance": release_policy,
        "condition_violation_inference": condition_inference,
        "change_from_baseline_inference": change_inference,
        "historical_replication_claim": historical_replication_claim,
        "within_cell_rmi_interpretation": (
            "A historical compound cell may violate Miller within-cell; this provides no "
            "separate aM or dM attribution."
            if design_family == HISTORICAL_ADAPTATION_FAMILY
            else "Registered scalar cells retain their declared factor identity."
        ),
        "inverse_effectiveness": inverse_effectiveness,
        "nested_trial_checkpoint_bootstrap": {
            "status": "not_implemented",
            "role": "complementary sensitivity; required before publication-strength interval claims",
            "pairing_requirement": "preserve common-random-number cell pairing",
        },
        "alternate_endpoint_execution": {"status": "not_implemented"},
        "common_horizon_sensitivity_execution": {"status": "not_implemented"},
        "directional_mechanism_matrix_runner": {"status": "not_implemented"},
        "input_provenance": {
            f"{item['condition_label']}|seed{item['checkpoint_seed']}": copy.deepcopy(
                item["input_provenance"]
            )
            for item in analyses
        },
        "analyzer_provenance": copy.deepcopy(release),
        "disclaimer": DISCLAIMER,
        "individual_analyses": analyses,
    }
    return _attach_derived_checksum(result)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return _json_safe_float(value)
    return value


def _atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    path = validate_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(value), handle, indent=1, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="raw acquisition JSON file(s)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = validate_output_path(Path(args.out))
    records = []
    for path_string in args.inputs:
        with Path(path_string).open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    result = (
        analyze_acquisition(records[0])
        if len(records) == 1
        else analyze_family(records, baseline_label=args.baseline_label)
    )
    result["cli_input_files"] = [
        {"path": str(Path(path_string).resolve()), "sha256": _sha256_file(Path(path_string))}
        for path_string in args.inputs
    ]
    _attach_derived_checksum(result)
    _atomic_json_dump(result, output)
    print(f"[race-model-analysis] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
