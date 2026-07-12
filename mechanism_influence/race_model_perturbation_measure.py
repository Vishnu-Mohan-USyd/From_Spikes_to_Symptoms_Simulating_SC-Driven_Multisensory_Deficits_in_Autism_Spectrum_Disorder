#!/usr/bin/env python3
"""Acquire the frozen three-intensity Phase-2 perturbation matrix.

This module is a narrow orchestration wrapper around
``mechanism_influence.race_model_measure``.  It deliberately imports the
validated Phase-1 registry, RNG derivation, trial encoding, measurement
kernel, checkpoint loader, state guards, readout guards, endpoint constants,
and atomic JSON writer instead of redefining them.  One invocation acquires
one preregistered perturbation cell for one checkpoint seed.

Latencies are measured in milliseconds with a 0.1-ms resolution and a
400-ms censoring horizon.  Randomness is fixed by the shared Phase-1
``derive_stream_seed`` function using ten independent 100-trial substreams per
sensory condition.  The wrapper accepts no free mechanism values, intensity
grid, trial count, or run seed.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from mechanism_influence import race_model_measure as phase1


ROOT = phase1.ROOT
MANIFEST_PATH = ROOT / "mechanism_influence" / "race_model_perturbation_manifest_v1.json"
EXPECTED_MANIFEST_FILE_SHA256 = (
    "f2b786d4c8fea128119701b723c25be8061168931ef038b3b937aef6c42e00d6"
)
EXPECTED_MANIFEST_SEMANTIC_SHA256 = (
    "3f6e15209f0e65ef5575d751d84ef729178651c3411624a6975e2db17778acae"
)
PROTOCOL_ID = "dm10-phase2-frozen-perturbation-rmi-v1"
RAW_SCHEMA_VERSION = "fsts-race-perturbation-trials-v1"
MANIFEST_SCHEMA_VERSION = "fsts-race-perturbation-manifest-v1"


@dataclass(frozen=True)
class PreparedRequest:
    """A fully validated request that is safe to pass to the CUDA runtime.

    Attributes:
        manifest: Frozen declarative protocol manifest.
        block: Semantic block containing the requested cell.
        cell: Exact manifest cell, including its mechanism vector and tails.
        registry_cell: Matching Phase-1 registry resolution.
        checkpoint: Canonical checkpoint provenance for this seed.
        checkpoint_path: Absolute checkpoint file path.
        output_path: Validated external or git-ignored final JSON path.
        source_provenance: Clean, tracked, HEAD-byte-equal source ledger.
        gpu_preflight: Read-only NVIDIA identity obtained before model loading.
    """

    manifest: Mapping[str, Any]
    block: Mapping[str, Any]
    cell: Mapping[str, Any]
    registry_cell: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    checkpoint_path: Path
    output_path: Path
    source_provenance: Mapping[str, Any]
    gpu_preflight: Mapping[str, Any]


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def _manifest_hashes(path: Path | None = None) -> tuple[str, str, dict[str, Any]]:
    """Return literal and semantic SHA-256 values plus duplicate-safe JSON."""

    path = MANIFEST_PATH if path is None else path
    payload = path.read_bytes()
    literal_sha256 = hashlib.sha256(payload).hexdigest()
    manifest = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    semantic_sha256 = phase1.canonical_sha256(manifest)
    return literal_sha256, semantic_sha256, manifest


def load_manifest() -> dict[str, Any]:
    """Load the immutable Phase-2 manifest and enforce both pinned hashes."""

    literal, semantic, manifest = _manifest_hashes()
    if literal != EXPECTED_MANIFEST_FILE_SHA256:
        raise ValueError("perturbation manifest literal SHA-256 mismatch")
    if semantic != EXPECTED_MANIFEST_SEMANTIC_SHA256:
        raise ValueError("perturbation manifest semantic SHA-256 mismatch")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("perturbation manifest schema mismatch")
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("protocol_version") != 1:
        raise ValueError("perturbation protocol identity mismatch")
    if manifest.get("raw_schema_version") != RAW_SCHEMA_VERSION:
        raise ValueError("perturbation raw schema mismatch")
    if manifest.get("status") != "preregistered":
        raise ValueError("perturbation protocol is not preregistered")
    return manifest


def _cell_index(
    manifest: Mapping[str, Any],
) -> dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]]:
    index: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for block_value in manifest["blocks"]:
        block = dict(block_value)
        for cell_value in block["cells"]:
            cell = dict(cell_value)
            cell_id = str(cell["cell_id"])
            if cell_id in index:
                raise ValueError(f"duplicate perturbation cell: {cell_id}")
            index[cell_id] = (block, cell)
    if len(index) != int(manifest["expected_counts"]["new_perturbation_cells"]):
        raise ValueError("perturbation manifest cell count mismatch")
    if str(manifest["shipped_anchor"]["source_label"]) in index:
        raise ValueError("shipped anchor must not be a Phase-2 acquisition cell")
    return index


def _resolve_cell(
    manifest: Mapping[str, Any], cell_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve and semantically reconcile one exact manifest/registry cell."""

    index = _cell_index(manifest)
    if cell_id not in index:
        raise ValueError(f"cell_id is not in the preregistered Phase-2 matrix: {cell_id}")
    block_raw, cell_raw = index[cell_id]
    block = copy.deepcopy(dict(block_raw))
    cell = copy.deepcopy(dict(cell_raw))
    vector = {name: float(cell["vector"][name]) for name in phase1.BASELINE_MECHANISMS}
    registry_cell = phase1.registered_mechanism_cell(vector, str(block["design_family"]))
    if registry_cell["cell_id"] != cell_id:
        raise ValueError("manifest cell label disagrees with the frozen Phase-1 registry")
    if registry_cell["factor_id"] != block["factor_id"]:
        raise ValueError("manifest factor_id disagrees with the frozen Phase-1 registry")
    if dict(registry_cell["requested"]) != vector:
        raise ValueError("manifest vector disagrees with the frozen Phase-1 registry")
    if block["semantic_mode"] == "scalar_one_factor":
        if registry_cell["scalar_one_factor"] is not True:
            raise ValueError("scalar block contains a non-one-factor mechanism vector")
        if block["scalar_attribution_allowed"] is not True:
            raise ValueError("scalar block must explicitly allow its one-factor attribution")
        if len(registry_cell["changed_scalars"]) != 1:
            raise ValueError("scalar perturbation must change exactly one mechanism")
    elif block["semantic_mode"] == "exact_compound_adaptation":
        if block["scalar_attribution_allowed"] is not False:
            raise ValueError("historical compound adaptation cannot allow scalar attribution")
        if registry_cell["factor_id"] != "compound_adaptation_replication":
            raise ValueError("historical compound adaptation factor identity mismatch")
        if set(registry_cell["changed_scalars"]) != {"aM", "dM"}:
            raise ValueError("historical compound adaptation must change exactly aM and dM")
    else:
        raise ValueError(f"unknown perturbation semantic mode: {block['semantic_mode']}")
    return block, cell, registry_cell


def _git_bytes(*arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _tracked_head_entry(path: Path) -> dict[str, Any]:
    """Require one local source file to be tracked and byte-equal to HEAD."""

    resolved = path.resolve()
    relative = resolved.relative_to(ROOT).as_posix()
    subprocess.run(
        ("git", "ls-files", "--error-unmatch", "--", relative),
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    head_payload = _git_bytes("show", f"HEAD:{relative}")
    local_sha256 = phase1.sha256_file(resolved)
    head_sha256 = hashlib.sha256(head_payload).hexdigest()
    if local_sha256 != head_sha256:
        raise ValueError(f"local source is not byte-equal to HEAD: {relative}")
    return {
        "path": str(resolved),
        "relative_path": relative,
        "sha256": local_sha256,
        "head_blob_sha256": head_sha256,
        "tracked": True,
        "local_matches_head": True,
    }


def _source_preflight(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all source pins and clean Git provenance before CUDA loading."""

    git = phase1.git_provenance()
    if git.get("clean") is not True:
        raise ValueError("Phase-2 acquisition requires a clean committed checkout")
    driver = _tracked_head_entry(Path(__file__))
    manifest_entry = _tracked_head_entry(MANIFEST_PATH)
    shared = _tracked_head_entry(Path(phase1.__file__))
    network = _tracked_head_entry(ROOT / "routec_net_io.py")
    latency = _tracked_head_entry(ROOT / "response_latency_routec.py")
    expected = manifest["frozen_phase1_sources"]
    reconciliations = {
        "shared_acquisition_kernel": (
            shared["sha256"] == expected["shared_acquisition_kernel_sha256"]
        ),
        "network_io": network["sha256"] == expected["network_io_sha256"],
        "latency_module": latency["sha256"] == expected["latency_module_sha256"],
        "manifest_literal": manifest_entry["sha256"] == EXPECTED_MANIFEST_FILE_SHA256,
        "manifest_semantic": (
            phase1.canonical_sha256(manifest) == EXPECTED_MANIFEST_SEMANTIC_SHA256
        ),
    }
    if not all(reconciliations.values()):
        raise ValueError(f"Phase-2 source pin mismatch: {reconciliations}")
    return {
        "root": str(ROOT),
        "driver": driver,
        "shared_acquisition_kernel": shared,
        "network_io": network,
        "latency_module": latency,
        "manifest": {
            **manifest_entry,
            "literal_sha256": EXPECTED_MANIFEST_FILE_SHA256,
            "semantic_sha256": EXPECTED_MANIFEST_SEMANTIC_SHA256,
            "protocol_id": PROTOCOL_ID,
        },
        "source_sha256": {
            "driver": driver["sha256"],
            "shared_acquisition_kernel": shared["sha256"],
            "network_io": network["sha256"],
            "latency_module": latency["sha256"],
        },
        "git": git,
        "reconciliations": reconciliations,
    }


def _query_gpu_identity(nvidia_uuid: str) -> dict[str, str]:
    """Read the selected physical GPU identity without loading the model."""

    completed = subprocess.run(
        (
            "nvidia-smi",
            "-i",
            nvidia_uuid,
            "--query-gpu=uuid,name",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    fields = [field.strip() for field in completed.stdout.strip().split(",", 1)]
    if len(fields) != 2:
        raise ValueError("nvidia-smi returned malformed GPU identity")
    return {"nvidia_smi_uuid": fields[0], "name": fields[1]}


def _checkpoint_entry(manifest: Mapping[str, Any], seed: int) -> dict[str, Any]:
    matches = [
        dict(entry)
        for entry in manifest["validated_anchor_acquisitions"]
        if int(entry["seed"]) == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"checkpoint seed is not preregistered exactly once: {seed}")
    return matches[0]


def prepare_request(args: argparse.Namespace) -> PreparedRequest:
    """Fail closed on every caller-controlled field before loading CUDA state."""

    manifest = load_manifest()
    block, cell, registry_cell = _resolve_cell(manifest, str(args.cell_id))
    seed = int(args.checkpoint_seed)
    if seed not in tuple(int(value) for value in manifest["acquisition"]["checkpoint_seeds"]):
        raise ValueError("checkpoint seed must be one of 42..51")
    if str(args.device) != manifest["device"]["requested"]:
        raise ValueError("Phase-2 acquisition requires the fixed cuda:0 device")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != manifest["device"]["cuda_visible_devices"]:
        raise ValueError("CUDA_VISIBLE_DEVICES does not select the approved RTX 5090 UUID")
    output = phase1.validate_output_path(Path(args.out))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing Phase-2 raw: {output}")
    checkpoint = _checkpoint_entry(manifest, seed)
    checkpoint_path = Path(args.ckpt).resolve()
    expected_checkpoint_path = (ROOT / "checkpoint" / checkpoint["checkpoint_name"]).resolve()
    if checkpoint_path != expected_checkpoint_path or not checkpoint_path.is_file():
        raise ValueError("checkpoint path/name is not the canonical preregistered file")
    if phase1.md5_file(checkpoint_path) != checkpoint["checkpoint_md5"]:
        raise ValueError("checkpoint MD5 mismatch")
    if phase1.sha256_file(checkpoint_path) != checkpoint["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA-256 mismatch")
    source_provenance = _source_preflight(manifest)
    gpu_preflight = _query_gpu_identity(manifest["device"]["cuda_visible_devices"])
    if gpu_preflight != {
        "nvidia_smi_uuid": manifest["device"]["cuda_visible_devices"],
        "name": manifest["device"]["name"],
    }:
        raise ValueError("physical GPU identity does not match the approved RTX 5090")
    return PreparedRequest(
        manifest=manifest,
        block=block,
        cell=cell,
        registry_cell=registry_cell,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        output_path=output,
        source_provenance=source_provenance,
        gpu_preflight=gpu_preflight,
    )


def _mechanism_vector(runtime_values: Mapping[str, float]) -> dict[str, float]:
    return {name: float(runtime_values[name]) for name in phase1.BASELINE_MECHANISMS}


def _validate_device_provenance(
    device: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    for key in (
        "requested",
        "type",
        "visible_index",
        "name",
        "capability",
        "total_memory_bytes",
        "uuid",
        "cuda_visible_devices",
        "torch",
        "torch_cuda",
    ):
        if device.get(key) != expected.get(key):
            raise ValueError(f"runtime device provenance mismatch: {key}")


def acquire(prepared: PreparedRequest) -> dict[str, Any]:
    """Acquire one cell/checkpoint record using only frozen Phase-1 mechanics.

    The returned record contains three intensity rows in the exact order
    ``[0.05, 0.2, 1.0]``.  Each row contains four 1000-trial condition blocks.
    No state-dict tensor, frozen readout, or plasticity setting may change.
    """

    manifest = prepared.manifest
    acquisition = manifest["acquisition"]
    settings = {
        name: float(prepared.cell["vector"][name]) for name in phase1.BASELINE_MECHANISMS
    }
    phase1._configure_environment()
    network_io, latency_module = phase1._load_runtime()
    device = phase1._device_provenance(str(manifest["device"]["requested"]))
    _validate_device_provenance(device, manifest["device"])
    readouts_before = network_io.assert_frozen_readouts("PHASE2 RACE BEFORE")
    net, load_result, epoch, g_rec, comment = network_io.load_ckpt(
        str(prepared.checkpoint_path),
        int(prepared.checkpoint["seed"]),
        int(acquisition["n_trials_per_condition"]),
        str(manifest["device"]["requested"]),
    )
    loaded_effective = phase1._effective_scalars(net)
    loaded_mechanisms = _mechanism_vector(loaded_effective)
    shipped = {
        name: float(manifest["shipped_anchor"]["vector"][name])
        for name in phase1.BASELINE_MECHANISMS
    }
    if loaded_mechanisms != shipped:
        raise AssertionError("checkpoint did not load the exact shipped mechanism vector")
    if float(g_rec) != loaded_effective["g_rec"]:
        raise AssertionError("checkpoint loader g_rec disagrees with the live scalar configuration")
    loaded_hash = phase1.state_dict_sha256(net)
    applied = phase1._apply_registered_cell(net, settings, str(prepared.block["design_family"]))
    if applied != dict(prepared.registry_cell):
        raise AssertionError("registered mechanism cell changed during post-load application")
    net.eval()
    net.plasticity_enabled = False
    effective = phase1._effective_scalars(net)
    if _mechanism_vector(effective) != settings:
        raise AssertionError("effective mechanism vector differs from the manifest request")
    before_hash = phase1.state_dict_sha256(net)
    if before_hash != loaded_hash:
        raise AssertionError("post-load scalar override changed the state_dict")
    snapshot = phase1.snapshot_state_dict(net)

    frame_ms = float(net.dt) * int(net.n_substeps)
    horizon_ms = float(latency_module.N_FRAMES) * frame_ms
    if not math.isclose(horizon_ms, float(acquisition["horizon_ms"]), abs_tol=1e-9):
        raise AssertionError("runtime censoring horizon differs from 400 ms")
    if not math.isclose(float(net.dt), float(acquisition["dt_ms"]), abs_tol=1e-12):
        raise AssertionError("runtime integration step differs from 0.1 ms")
    rows: list[dict[str, Any]] = []
    for intensity_value in acquisition["intensities"]:
        intensity = float(intensity_value)
        conditions: dict[str, Any] = {}
        for condition in acquisition["condition_order"]:
            conditions[str(condition)] = phase1._measure_condition(
                net,
                latency_module,
                str(condition),
                intensity,
                float(net.sigma_in),
                int(acquisition["n_trials_per_condition"]),
                str(manifest["device"]["requested"]),
                int(acquisition["run_seed"]),
                int(prepared.checkpoint["seed"]),
                horizon_ms,
            )
        rows.append({"intensity": intensity, "conditions": conditions})

    after_hash = phase1.assert_state_dict_unchanged(net, snapshot)
    effective_after = phase1._effective_scalars(net)
    if effective_after != effective:
        raise AssertionError("effective live scalar configuration changed during frozen acquisition")
    readouts_after = network_io.assert_frozen_readouts("PHASE2 RACE AFTER")
    if list(readouts_before) != list(readouts_after):
        raise AssertionError("frozen readout hashes changed during Phase-2 acquisition")
    endpoint = copy.deepcopy(dict(acquisition["endpoint"]))
    endpoint["resolution_ms"] = float(net.dt)
    configuration = {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": int(manifest["protocol_version"]),
        "manifest_file_sha256": EXPECTED_MANIFEST_FILE_SHA256,
        "manifest_semantic_sha256": EXPECTED_MANIFEST_SEMANTIC_SHA256,
        "cell_id": prepared.cell["cell_id"],
        "block_id": prepared.block["block_id"],
        "baseline": shipped,
        "requested": settings,
        "effective": effective,
        "effective_after": effective_after,
        "environment": copy.deepcopy(phase1.BASE_ENV),
        "intensities": [float(value) for value in acquisition["intensities"]],
        "n_trials": int(acquisition["n_trials_per_condition"]),
        "run_seed": int(acquisition["run_seed"]),
        "soa_ms": float(acquisition["soa_ms"]),
        "horizon_ms": horizon_ms,
        "device": copy.deepcopy(dict(manifest["device"])),
    }
    changed_scalars = list(prepared.registry_cell["changed_scalars"])
    record: dict[str, Any] = {
        "schema_version": RAW_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_version": int(manifest["protocol_version"]),
        "manifest": {
            "path": str(MANIFEST_PATH.resolve()),
            "literal_sha256": EXPECTED_MANIFEST_FILE_SHA256,
            "semantic_sha256": EXPECTED_MANIFEST_SEMANTIC_SHA256,
            "status": manifest["status"],
        },
        "condition": {
            "label": prepared.cell["cell_id"],
            "cell_id": prepared.cell["cell_id"],
            "block_id": prepared.block["block_id"],
            "design_family": prepared.block["design_family"],
            "factor_id": prepared.block["factor_id"],
            "semantic_mode": prepared.block["semantic_mode"],
            "scalar_attribution_allowed": prepared.block["scalar_attribution_allowed"],
            "role": prepared.cell["role"],
            "changed_scalar": changed_scalars[0] if len(changed_scalars) == 1 else None,
            "changed_scalars": changed_scalars,
            "anchor_id": prepared.block["anchor_id"],
            "delta_definition": manifest["delta_definition"],
            "delta_tails": copy.deepcopy(dict(prepared.cell["delta_tails"])),
            "within_cell_rmi_tail": prepared.cell["within_cell_rmi_tail"],
            "response_collapse_policy": copy.deepcopy(
                prepared.cell.get("response_collapse_policy")
            ),
            "baseline": shipped,
            "requested": settings,
            "registry_resolution": copy.deepcopy(dict(prepared.registry_cell)),
        },
        "checkpoint": {
            "path": str(prepared.checkpoint_path),
            "name": prepared.checkpoint["checkpoint_name"],
            "seed": int(prepared.checkpoint["seed"]),
            "epoch": None if epoch is None else int(epoch),
            "md5": prepared.checkpoint["checkpoint_md5"],
            "sha256": prepared.checkpoint["checkpoint_sha256"],
            "load_missing_keys": list(load_result.missing_keys),
            "load_unexpected_keys": list(load_result.unexpected_keys),
            "comment": comment,
        },
        "measurement": {
            "intensities": [float(value) for value in acquisition["intensities"]],
            "condition_order": list(acquisition["condition_order"]),
            "n_trials_per_condition": int(acquisition["n_trials_per_condition"]),
            "run_seed": int(acquisition["run_seed"]),
            "substream_count": int(acquisition["substream_count"]),
            "trials_per_substream": int(acquisition["trials_per_substream"]),
            "stream_seed_derivation": acquisition["stream_seed_derivation"],
            "centre_deg": float(latency_module.CENTRE_DEG),
            "sigma_in_neurons": float(net.sigma_in),
            "pulse_frames": int(latency_module.PULSE_FRAMES),
            "n_frames": int(latency_module.N_FRAMES),
            "frame_ms": frame_ms,
            "dt_ms": float(net.dt),
            "n_substeps": int(net.n_substeps),
            "horizon_ms": horizon_ms,
            "soa_ms": float(acquisition["soa_ms"]),
            "endpoint": endpoint,
        },
        "loaded_effective": loaded_effective,
        "requested": settings,
        "effective": effective,
        "effective_after": effective_after,
        "device": device,
        "integrity": {
            "state_sha256_loaded": loaded_hash,
            "state_sha256_before": before_hash,
            "state_sha256_after": after_hash,
            "state_bit_identical": loaded_hash == before_hash == after_hash,
            "readout_md5_before": list(readouts_before),
            "readout_md5_after": list(readouts_after),
            "readouts_unchanged": list(readouts_before) == list(readouts_after),
            "plasticity_enabled": bool(net.plasticity_enabled),
            "g_rec": effective_after["g_rec"],
        },
        "code": {
            **copy.deepcopy(dict(prepared.source_provenance)),
            "argv": list(sys.argv),
        },
        "configuration": configuration,
        "configuration_sha256": phase1.canonical_sha256(configuration),
        "disclaimer": (
            "RMI and perturbation effects are model endpoint phenotypes; they do not "
            "prove a unique mechanism, clinical ASD, an SC locus, or human reaction time."
        ),
        "rows": rows,
    }
    if record["measurement"]["intensities"] != [0.05, 0.2, 1.0]:
        raise AssertionError("Phase-2 acquisition row grid/order changed")
    if record["requested"] != _mechanism_vector(record["effective"]):
        raise AssertionError("requested mechanism projection differs from the live configuration")
    if record["effective"] != record["effective_after"]:
        raise AssertionError("effective live scalar configurations differ")
    if record["integrity"]["g_rec"] != record["effective_after"]["g_rec"]:
        raise AssertionError("integrity g_rec differs from the final live configuration")
    if (
        record["integrity"]["state_bit_identical"] is not True
        or record["integrity"]["readouts_unchanged"] is not True
        or record["integrity"]["plasticity_enabled"] is not False
    ):
        raise AssertionError("frozen Phase-2 acquisition integrity failed")
    record["integrity"]["raw_payload_sha256"] = phase1.raw_payload_sha256(record)
    return record


def write_record(record: Mapping[str, Any], output: Path) -> None:
    """Write one strict JSON record atomically without replacing an existing raw."""

    resolved = phase1.validate_output_path(output)
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing Phase-2 raw: {resolved}")
    phase1.atomic_json_dump(record, resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--checkpoint-seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prepared = prepare_request(args)
    record = acquire(prepared)
    write_record(record, prepared.output_path)
    print(
        f"[phase2-race] wrote {prepared.output_path} "
        f"cell={prepared.cell['cell_id']} seed={prepared.checkpoint['seed']} "
        "intensities=[0.05, 0.2, 1.0]",
        flush=True,
    )


if __name__ == "__main__":
    main()
