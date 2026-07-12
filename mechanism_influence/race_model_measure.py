#!/usr/bin/env python3
"""Acquire frozen trial-level latency data for formal audiovisual race analysis.

This driver changes no network weights or dynamics implementation.  It loads one
shipped dm10 checkpoint with the canonical measurement loader, applies either
one registered scalar cell or one explicitly labelled historical adaptation
tuple, and reuses
``response_latency_routec.first_spike_latency`` for A, V, AV, and noise-only
catch trials.  Latencies are in milliseconds on the model's 0.1-ms substep grid.

Silence is logically ``+Inf``.  JSON does not permit IEEE infinity, so silent
latencies are serialized as the literal string ``"+Inf"`` together with a
``hit=false`` flag and the finite censoring horizon.  The CPU analyzer restores
that representation to logical infinity without dropping the trial.

One fresh process per checkpoint/condition is required.  This keeps CUDA state,
runtime network state, and RNG streams isolated across mechanism conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "fsts-race-trials-v1"
DECLARED_INTENSITIES = (0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.6)
CONDITION_ORDER = ("A", "V", "AV", "catch")
PRIMARY_SEEDS = tuple(range(42, 52))
PRIMARY_HORIZON_MS = 400.0
PRIMARY_N_TRIALS = 1000
PRIMARY_SUBSTREAMS = 10
TRIALS_PER_SUBSTREAM = 100
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
ENDPOINT_RESPONSE_RULE = "first substep with MSI population spike count > 0"
ENDPOINT_THRESHOLD = "MSI population spike count > 0"
ENDPOINT_FORMULA = "(_first_spike_substep + 1) * dt_ms"
ENDPOINT_INTERPRETATION = (
    "model neural first-spike latency surrogate; not human reaction time"
)


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

# Values required to reconstruct the shipped dm10 forward.  TAU_GABA/GNMDA
# must remain at their trained values during loading; lesions are applied only
# after the loader's checkpoint-vs-environment guards have passed.
BASE_ENV = {
    "DEND_COUPLING_ALPHA": "2",
    "MG_VHALF": "-48",
    "MG_VHALF_INH": "-30",
    "MG_K": "0.15",
    "GABA_SHUNT_SURR": "1",
    "K_SHUNT_SURR": "0.026",
    "E_GABA": "-70.0",
    "TAU_GABA": "18.0",
    "G_GABA": "5.56",
    "PV_GABA_SCALE": "1.0",
    "SIGMA_DL_FRAMES": "3",
    "GNMDA": "0.51",
    "TAU_NMDA": "40",
    "G_REC": "0.03",
    "EXP_G_REC": "0.03",
    "ISTDP_BASELINE": "0.56",
    "AFFERENT_JITTER_MS": "4",
    "U_STP_A": "0.2",
    "U_STP_V": "0.2",
    "NMDA_STD_SCALE": "0.8",
    "TAU_REC": "400.0",
    "K_DVDT": "0.0",
    "TAU_DVDT": "3.0",
    "V_THRESH_FLOOR": "20.0",
    "DVDT_CAP": "50.0",
    "MPLBACKEND": "Agg",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    """Return the MD5 provenance digest used by the shipped checkpoint manifest."""

    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Return SHA-256 of deterministic strict-JSON serialization."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def raw_payload_sha256(record_without_checksum: Mapping[str, Any]) -> str:
    """Hash a complete raw record before its self-referential checksum is added."""

    return canonical_sha256(record_without_checksum)


def validate_output_path(path: Path) -> Path:
    """Require raw artifacts to be outside the repo or under a git-ignore rule."""

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


def git_provenance() -> dict[str, Any]:
    """Capture exact source revision and require a clean formal acquisition tree."""

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    head = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    clean = run("status", "--porcelain") == ""
    return {"head": head, "branch": branch, "clean": clean}


def registered_mechanism_cell(
    settings: Mapping[str, float], design_family: str
) -> dict[str, Any]:
    """Resolve an exact registered cell and reject arbitrary/mixed perturbations."""

    changed_mechanisms(settings)
    if design_family not in MECHANISM_CELL_REGISTRY:
        raise ValueError(f"unknown mechanism design family: {design_family}")
    matches = []
    for cell_id, specification in MECHANISM_CELL_REGISTRY[design_family].items():
        requested = specification["requested"]
        if all(
            math.isclose(
                float(settings[name]), float(requested[name]), rel_tol=0.0, abs_tol=1e-12
            )
            for name in BASELINE_MECHANISMS
        ):
            matches.append((cell_id, specification))
    if len(matches) != 1:
        raise ValueError(
            f"unregistered mechanism vector for design family {design_family}: {dict(settings)}"
        )
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


def atomic_json_dump(record: Mapping[str, Any], output: Path) -> None:
    """Validate strict JSON, fsync a sibling temp file, then atomically promote it."""

    output = validate_output_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def state_dict_sha256(module: torch.nn.Module) -> str:
    """Hash every state-dict tensor, including names, shapes, and dtypes.

    The digest is device-independent because tensors are copied contiguously to
    CPU before their raw bytes are hashed.  Units are not applicable; this is a
    bit-identity provenance value.
    """

    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode("utf-8"))
        if not torch.is_tensor(value):
            digest.update(repr(value).encode("utf-8"))
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def snapshot_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone a module state dict for an exact post-inference mutation guard."""

    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def assert_state_dict_unchanged(
    module: torch.nn.Module, snapshot: Mapping[str, torch.Tensor]
) -> str:
    """Assert exact tensor equality against *snapshot* and return SHA-256."""

    current = module.state_dict()
    if set(current) != set(snapshot):
        raise AssertionError("state_dict key set changed during frozen inference")
    changed = [name for name in sorted(snapshot) if not torch.equal(snapshot[name], current[name])]
    if changed:
        raise AssertionError(f"state_dict tensors changed during frozen inference: {changed}")
    return state_dict_sha256(module)


def changed_mechanisms(settings: Mapping[str, float]) -> list[str]:
    """Return live scalar names that differ from the shipped dm10 baseline."""

    if set(settings) != set(BASELINE_MECHANISMS):
        raise ValueError("mechanism settings must have the exact dm10 one-factor keyset")
    for name, value in settings.items():
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"mechanism setting outside declared domain: {name}={numeric}")
        if name == "tau_gaba" and numeric <= 0:
            raise ValueError(f"mechanism setting must be positive: {name}={numeric}")
    return [
        name
        for name, baseline in BASELINE_MECHANISMS.items()
        if not math.isclose(float(settings[name]), baseline, rel_tol=0.0, abs_tol=1e-12)
    ]


def validate_one_factor(
    settings: Mapping[str, float], design_family: str = SCALAR_RMI_FAMILY
) -> str | None:
    """Validate one exact scalar-RMI registry cell and return its changed scalar."""

    cell = registered_mechanism_cell(settings, design_family)
    if design_family != SCALAR_RMI_FAMILY or not cell["scalar_one_factor"]:
        raise ValueError("validate_one_factor accepts scalar_rmi_v1 cells only")
    changed = cell["changed_scalars"]
    return changed[0] if changed else None


def derive_stream_seed(
    run_seed: int,
    checkpoint_seed: int,
    intensity: float,
    condition_label: str,
    substream_index: int = 0,
) -> int:
    """Derive a stable independent RNG stream for one vectorized trial batch.

    The intensity value, rather than its position in a requested subset, is in
    the hash.  A baseline full-grid run and a perturbation subset therefore use
    identical random streams at every shared intensity.
    """

    if condition_label not in CONDITION_ORDER:
        raise ValueError(f"unknown sensory condition label: {condition_label}")
    if not 0 <= substream_index < PRIMARY_SUBSTREAMS:
        raise ValueError("substream_index outside preregistered range")
    intensity_token = format(float(intensity), ".17g")
    payload = (
        f"{run_seed}:{checkpoint_seed}:{intensity_token}:{condition_label}:{substream_index}"
    ).encode("ascii")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (2**31 - 1)


def _rng_sha256(device: str) -> str:
    """Hash current CPU and selected-device Torch RNG states."""

    digest = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if str(device).startswith("cuda") and torch.cuda.is_available():
        digest.update(torch.cuda.get_rng_state(device).cpu().numpy().tobytes())
    return digest.hexdigest()


def _seed_torch(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _configure_environment() -> None:
    for name, value in BASE_ENV.items():
        os.environ[name] = value
    # Adaptation is restored from the checkpoint and changed, if requested,
    # only after load.  Removing these prevents stale caller values.
    for name in ("ADAPT_A", "ADAPT_DM", "ADAPT_A_INH", "ADAPT_DM_INH", "TAU_NMDA_V"):
        os.environ.pop(name, None)


def _load_runtime() -> tuple[Any, Any]:
    """Import current-checkout loader and latency modules after env setup."""

    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import routec_net_io as network_io
    import response_latency_routec as latency

    for module in (network_io, latency):
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(ROOT):
            raise AssertionError(f"module escaped current checkout: {module_path}")
    return network_io, latency


def _effective_scalars(net: Any) -> dict[str, float]:
    names = (
        "aM",
        "dM",
        "pv_gaba_scale",
        "tau_gaba",
        "gNMDA",
        "g_GABA",
        "g_rec",
        "tau_nmda",
        "u_stp_a",
        "u_stp_v",
        "nmda_std_scale",
        "tau_rec",
        "afferent_jitter_ms",
    )
    return {name: float(getattr(net, name)) for name in names}


def _apply_registered_cell(
    net: Any, settings: Mapping[str, float], design_family: str
) -> dict[str, Any]:
    """Apply one exact registered scalar or historical adaptation cell post-load."""

    cell = registered_mechanism_cell(settings, design_family)
    loaded = {name: float(getattr(net, name)) for name in BASELINE_MECHANISMS}
    for name, baseline in BASELINE_MECHANISMS.items():
        if not math.isclose(loaded[name], baseline, rel_tol=0.0, abs_tol=1e-9):
            raise AssertionError(f"loaded net.{name}={loaded[name]} != dm10 baseline {baseline}")
    for name in BASELINE_MECHANISMS:
        setattr(net, name, float(settings[name]))
    for name in BASELINE_MECHANISMS:
        got = float(getattr(net, name))
        if not math.isclose(got, float(settings[name]), rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"post-load override failed: net.{name}={got}")
    return cell


def _device_provenance(device: str) -> dict[str, Any]:
    if not str(device).startswith("cuda"):
        return {
            "requested": device,
            "type": "cpu",
            "name": "CPU",
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    props = torch.cuda.get_device_properties(index)
    return {
        "requested": device,
        "type": "cuda",
        "visible_index": int(index),
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": int(props.total_memory),
        "uuid": str(getattr(props, "uuid", "unavailable")),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _encode_trials(latencies: Sequence[float], horizon_ms: float) -> tuple[list[float | str], list[bool]]:
    """Encode ``0<t<=H`` hits; censor late/nonfinite silence and reject invalid time."""

    if not math.isfinite(horizon_ms) or horizon_ms <= 0:
        raise ValueError("censoring horizon must be finite and positive")
    encoded: list[float | str] = []
    hits: list[bool] = []
    for value in latencies:
        numeric = float(value)
        if numeric == -math.inf or (math.isfinite(numeric) and numeric <= 0):
            raise ValueError(f"latency must be positive or censored, got {numeric}")
        hit = math.isfinite(numeric) and numeric <= horizon_ms
        hits.append(hit)
        encoded.append(numeric if hit else "+Inf")
    if len(encoded) != len(hits):
        raise AssertionError("latency/hit encoding length mismatch")
    return encoded, hits


def _measure_condition(
    net: Any,
    latency_module: Any,
    condition: str,
    intensity: float,
    sigma_in: float,
    n_trials: int,
    device: str,
    run_seed: int,
    checkpoint_seed: int,
    horizon_ms: float,
) -> dict[str, Any]:
    if n_trials != PRIMARY_N_TRIALS:
        raise ValueError("formal acquisition requires exactly 1000 trials per condition")
    modality = {"A": "A", "V": "V", "AV": "B", "catch": "B"}[condition]
    saved_intensity = float(latency_module.INTENSITY)
    latencies: list[float] = []
    substreams: list[dict[str, Any]] = []
    n_spiked = 0
    count = 0
    try:
        latency_module.INTENSITY = 0.0 if condition == "catch" else float(intensity)
        for substream_index in range(PRIMARY_SUBSTREAMS):
            stream_seed = derive_stream_seed(
                run_seed, checkpoint_seed, intensity, condition, substream_index
            )
            _seed_torch(stream_seed)
            rng_before = _rng_sha256(device)
            _, substream_hits, substream_count, substream_latencies = (
                latency_module.first_spike_latency(
                    net,
                    modality,
                    sigma_in,
                    TRIALS_PER_SUBSTREAM,
                    device,
                    a_onset_frame=0,
                )
            )
            rng_after = _rng_sha256(device)
            start = len(latencies)
            latencies.extend(float(value) for value in substream_latencies)
            n_spiked += int(substream_hits)
            count += int(substream_count)
            substreams.append(
                {
                    "index": substream_index,
                    "stream_seed": int(stream_seed),
                    "trial_start": start,
                    "trial_stop_exclusive": len(latencies),
                    "state_before_sha256": rng_before,
                    "state_after_sha256": rng_after,
                }
            )
    finally:
        latency_module.INTENSITY = saved_intensity
    encoded, hits = _encode_trials(latencies, horizon_ms)
    if count != n_trials or len(encoded) != n_trials or n_spiked != sum(hits):
        raise AssertionError("trial-count or hit-count mismatch from latency probe")
    finite = np.asarray([x for x, hit in zip(latencies, hits) if hit], dtype=float)
    return {
        "condition": condition,
        "stimulus_intensity": 0.0 if condition == "catch" else float(intensity),
        "n_trials": int(n_trials),
        "n_hits": int(sum(hits)),
        "hit_rate": float(sum(hits) / n_trials),
        "latency_ms": encoded,
        "hit": hits,
        "trial_id": [
            f"s{substream_index:02d}:t{trial_index:03d}"
            for substream_index in range(PRIMARY_SUBSTREAMS)
            for trial_index in range(TRIALS_PER_SUBSTREAM)
        ],
        "trial_index": {"start": 0, "stop_exclusive": int(n_trials)},
        "censor_ms": float(horizon_ms),
        "silence_encoding": "+Inf",
        "mean_hit_latency_ms": float(np.mean(finite)) if finite.size else None,
        "median_hit_latency_ms": float(np.median(finite)) if finite.size else None,
        "rng": {
            "algorithm": RNG_ALGORITHM,
            "stable_cell_key": {
                "checkpoint_seed": int(checkpoint_seed),
                "intensity": float(intensity),
                "sensory_condition": condition,
            },
            "substreams": substreams,
        },
    }


def acquire(args: argparse.Namespace) -> dict[str, Any]:
    """Run one frozen checkpoint/condition acquisition and return a JSON-safe record."""

    _configure_environment()
    network_io, latency_module = _load_runtime()
    checkpoint = Path(args.ckpt).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    intensities = tuple(float(value) for value in args.intensities)
    if intensities != DECLARED_INTENSITIES:
        raise ValueError("formal acquisition requires the exact seven-intensity preregistered grid")
    if args.n_trials != PRIMARY_N_TRIALS:
        raise ValueError("formal acquisition requires exactly 1000 trials per condition")
    if int(args.checkpoint_seed) not in PRIMARY_SEEDS:
        raise ValueError("formal acquisition checkpoint seed must be one of 42..51")
    git = git_provenance()
    if git["clean"] is not True:
        raise ValueError("formal acquisition requires a clean committed git checkout")

    settings = {
        "aM": float(args.aM),
        "dM": float(args.dM),
        "pv_gaba_scale": float(args.pv_gaba_scale),
        "tau_gaba": float(args.tau_gaba),
        "gNMDA": float(args.gNMDA),
    }
    cell = registered_mechanism_cell(settings, args.design_family)
    changed_scalars = list(cell["changed_scalars"])
    changed = changed_scalars[0] if len(changed_scalars) == 1 else None
    expected_label = str(cell["cell_id"])
    if args.label != expected_label:
        raise ValueError(f"condition label must be canonical {expected_label!r}")
    readouts_before = network_io.assert_frozen_readouts("RACE BEFORE")
    net, load_result, epoch, g_rec, comment = network_io.load_ckpt(
        str(checkpoint), int(args.checkpoint_seed), int(args.n_trials), args.device
    )
    loaded_effective = _effective_scalars(net)
    loaded_hash = state_dict_sha256(net)
    applied_cell = _apply_registered_cell(net, settings, args.design_family)
    if applied_cell != cell:
        raise AssertionError("registered mechanism cell identity changed during application")
    net.eval()
    net.plasticity_enabled = False
    effective = _effective_scalars(net)
    before_hash = state_dict_sha256(net)
    if before_hash != loaded_hash:
        raise AssertionError("live scalar override changed state_dict")
    snapshot = snapshot_state_dict(net)

    frame_ms = float(net.dt) * int(net.n_substeps)
    horizon_ms = float(latency_module.N_FRAMES) * frame_ms
    if not math.isclose(horizon_ms, PRIMARY_HORIZON_MS, rel_tol=0.0, abs_tol=1e-9):
        raise AssertionError(f"formal censor horizon must be 400 ms, got {horizon_ms}")
    rows: list[dict[str, Any]] = []
    for intensity in intensities:
        conditions: dict[str, Any] = {}
        for condition in CONDITION_ORDER:
            conditions[condition] = _measure_condition(
                net,
                latency_module,
                condition,
                intensity,
                float(net.sigma_in),
                int(args.n_trials),
                args.device,
                int(args.run_seed),
                int(args.checkpoint_seed),
                horizon_ms,
            )
        rows.append({"intensity": float(intensity), "conditions": conditions})

    after_hash = assert_state_dict_unchanged(net, snapshot)
    effective_after = _effective_scalars(net)
    if effective_after != effective:
        raise AssertionError("effective live scalar configuration changed during acquisition")
    readouts_after = network_io.assert_frozen_readouts("RACE AFTER")
    if list(readouts_after) != list(readouts_before):
        raise AssertionError("frozen readout hashes changed")
    record = {
        "schema_version": SCHEMA_VERSION,
        "condition": {
            "label": args.label,
            "changed_scalar": changed,
            "changed_scalars": changed_scalars,
            "design_family": cell["design_family"],
            "cell_id": cell["cell_id"],
            "factor_id": cell["factor_id"],
            "baseline": BASELINE_MECHANISMS,
            "requested": settings,
            "one_factor": bool(cell["scalar_one_factor"]),
            "mechanism_manifest": {
                "version": "dm10-registered-mechanisms-v2",
                **cell,
                "pv_gaba_scale_name": "MSI-inhibitory GABA scale",
            },
        },
        "checkpoint": {
            "path": str(checkpoint),
            "name": checkpoint.name,
            "seed": int(args.checkpoint_seed),
            "epoch": None if epoch is None else int(epoch),
            "md5": md5_file(checkpoint),
            "sha256": sha256_file(checkpoint),
            "load_missing_keys": list(load_result.missing_keys),
            "load_unexpected_keys": list(load_result.unexpected_keys),
            "comment": comment,
        },
        "measurement": {
            "intensities": list(intensities),
            "condition_order": list(CONDITION_ORDER),
            "n_trials_per_condition": int(args.n_trials),
            "run_seed": int(args.run_seed),
            "substream_count": PRIMARY_SUBSTREAMS,
            "trials_per_substream": TRIALS_PER_SUBSTREAM,
            "stream_seed_derivation": STREAM_SEED_DERIVATION,
            "centre_deg": float(latency_module.CENTRE_DEG),
            "sigma_in_neurons": float(net.sigma_in),
            "pulse_frames": int(latency_module.PULSE_FRAMES),
            "n_frames": int(latency_module.N_FRAMES),
            "frame_ms": frame_ms,
            "dt_ms": float(net.dt),
            "n_substeps": int(net.n_substeps),
            "horizon_ms": horizon_ms,
            "soa_ms": 0.0,
            "endpoint": {
                "response_rule_id": "first_msi_population_spike_v1",
                "response_rule": ENDPOINT_RESPONSE_RULE,
                "threshold": ENDPOINT_THRESHOLD,
                "latency_formula": ENDPOINT_FORMULA,
                "latency_domain_ms": "0 < latency <= 400",
                "resolution_ms": float(net.dt),
                "interpretation": ENDPOINT_INTERPRETATION,
                "sensitivity_horizons_ms": [300.0, 350.0, 400.0],
            },
        },
        "effective": effective,
        "effective_after": effective_after,
        "loaded_effective": loaded_effective,
        "device": _device_provenance(args.device),
        "integrity": {
            "state_sha256_loaded": loaded_hash,
            "state_sha256_before": before_hash,
            "state_sha256_after": after_hash,
            "state_bit_identical": bool(before_hash == after_hash),
            "readout_md5_before": list(readouts_before),
            "readout_md5_after": list(readouts_after),
            "readouts_unchanged": bool(list(readouts_before) == list(readouts_after)),
            "plasticity_enabled": bool(net.plasticity_enabled),
            "g_rec": float(g_rec),
        },
        "code": {
            "root": str(ROOT),
            "driver": str(Path(__file__).resolve()),
            "driver_sha256": sha256_file(Path(__file__).resolve()),
            "network_io": str(Path(network_io.__file__).resolve()),
            "latency_module": str(Path(latency_module.__file__).resolve()),
            "source_sha256": {
                "driver": sha256_file(Path(__file__).resolve()),
                "network_io": sha256_file(Path(network_io.__file__).resolve()),
                "latency_module": sha256_file(Path(latency_module.__file__).resolve()),
            },
            "git": git,
            "argv": list(sys.argv),
        },
        "configuration": {
            "baseline": BASELINE_MECHANISMS,
            "requested": settings,
            "effective": effective,
            "environment": BASE_ENV,
            "intensities": list(intensities),
            "n_trials": int(args.n_trials),
            "soa_ms": 0.0,
            "horizon_ms": horizon_ms,
        },
        "configuration_sha256": canonical_sha256(
            {
                "baseline": BASELINE_MECHANISMS,
                "requested": settings,
                "effective": effective,
                "environment": BASE_ENV,
                "intensities": list(intensities),
                "n_trials": int(args.n_trials),
                "soa_ms": 0.0,
                "horizon_ms": horizon_ms,
            }
        ),
        "disclaimer": (
            "RMI violation rejects race plus context invariance; it does not prove unique "
            "coactivation, an SC locus, or human reaction time."
        ),
        "rows": rows,
    }
    if not record["integrity"]["state_bit_identical"] or not record["integrity"]["readouts_unchanged"]:
        raise AssertionError("frozen inference integrity failed")
    record["integrity"]["raw_payload_sha256"] = raw_payload_sha256(record)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--checkpoint-seed", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--design-family",
        choices=sorted(MECHANISM_CELL_REGISTRY),
        default=SCALAR_RMI_FAMILY,
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-seed", type=int, default=20260712)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--intensities", type=float, nargs="+", default=list(DECLARED_INTENSITIES))
    parser.add_argument("--aM", type=float, default=BASELINE_MECHANISMS["aM"])
    parser.add_argument("--dM", type=float, default=BASELINE_MECHANISMS["dM"])
    parser.add_argument("--pv-gaba-scale", type=float, default=BASELINE_MECHANISMS["pv_gaba_scale"])
    parser.add_argument("--tau-gaba", type=float, default=BASELINE_MECHANISMS["tau_gaba"])
    parser.add_argument("--gNMDA", type=float, default=BASELINE_MECHANISMS["gNMDA"])
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = validate_output_path(Path(args.out))
    record = acquire(args)
    atomic_json_dump(record, output)
    print(
        f"[race-model] wrote {output} label={args.label} seed={args.checkpoint_seed} "
        f"changed={record['condition']['changed_scalar']} intensities={record['measurement']['intensities']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
