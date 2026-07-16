#!/usr/bin/env python3
"""Frozen dm10 perturbations for official Gate 4 MEI and Gate 5 CRE.

This driver deliberately reuses the repository's validated measurement kernels:

* :func:`inverse_effectiveness_routec.measure_ie` for Gate 4; and
* :func:`diag_136b_cre_5seed.capture_profiles`, ``cre_block``, plus
  :func:`measure.measure_ens_cre.fit_center_surround` for Gate 5.

It does not edit the network, checkpoints, learned readouts, or official gate
implementations.  One ``cell`` invocation loads a fresh frozen checkpoint for
each endpoint, applies exactly one registered inference-time perturbation after
loading (and after Gate 5 restores its checkpoint NMDA values), and atomically
writes one JSON record.  ``launch`` executes checkpoint/setting cells in fresh
subprocesses, so CUDA state is never shared between scientific cells.

Scientific CUDA execution is restricted to the physical RTX 5090 identified by
both its immutable NVIDIA UUID and its reported product name.  Absolute A, V,
and AV responses accompany every normalized endpoint so collapse and small
denominators cannot masquerade as altered multisensory enhancement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

# These caps must be installed before NumPy, SciPy, or Torch load their native
# thread pools.  Gate 4/5 are GPU-bound; hundreds of CPU threads only create
# context-switch contention when independent cells are running.
THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
for _thread_variable, _thread_value in THREAD_CAPS.items():
    os.environ[_thread_variable] = _thread_value

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "fsts-response-gain-perturbation-v1"
PROTOCOL_ID = "dm10-official-gate4-gate5-frozen-perturbations-v1"
RTX5090_UUID = "GPU-c6b8df49-a222-338c-bc8a-02e4e43833c3"
RTX5090_NAME_TOKEN = "RTX 5090"
CHECKPOINT_SEEDS = tuple(range(42, 52))
CHECKPOINT_TEMPLATE = "ckpt_ep79_seed{seed}_bs250_delay52_tau18_dL3.pt"
BASELINE = {
    "aM": 0.02,
    "dM": 10.0,
    "pv_gaba_scale": 1.0,
    "tau_gaba": 18.0,
    "gNMDA": 0.51,
}


def _vector(**updates: float) -> dict[str, float]:
    values = dict(BASELINE)
    values.update({key: float(value) for key, value in updates.items()})
    return values


# The thirteen primary cells are unique operating points.  Shipped values are
# not repeated under the PV=1, tau=18, or gNMDA=.51 aliases.
PRIMARY_REGISTRY: dict[str, dict[str, Any]] = {
    "shipped": {"factor": "baseline", "values": _vector()},
    "adaptation_off": {
        "factor": "historical_compound_adaptation",
        "values": _vector(aM=0.0, dM=0.0),
    },
    "adaptation_prior": {
        "factor": "historical_compound_adaptation",
        "values": _vector(aM=0.008, dM=8.0),
    },
    "pv_gaba_scale_0": {"factor": "pv_gaba_scale", "values": _vector(pv_gaba_scale=0.0)},
    "pv_gaba_scale_0p5": {"factor": "pv_gaba_scale", "values": _vector(pv_gaba_scale=0.5)},
    "pv_gaba_scale_1p5": {"factor": "pv_gaba_scale", "values": _vector(pv_gaba_scale=1.5)},
    "pv_gaba_scale_4": {"factor": "pv_gaba_scale", "values": _vector(pv_gaba_scale=4.0)},
    "tau_gaba_10": {"factor": "tau_gaba", "values": _vector(tau_gaba=10.0)},
    "tau_gaba_40": {"factor": "tau_gaba", "values": _vector(tau_gaba=40.0)},
    "tau_gaba_60": {"factor": "tau_gaba", "values": _vector(tau_gaba=60.0)},
    "gNMDA_0": {"factor": "gNMDA", "values": _vector(gNMDA=0.0)},
    "gNMDA_0p255": {"factor": "gNMDA", "values": _vector(gNMDA=0.255)},
    "gNMDA_0p765": {"factor": "gNMDA", "values": _vector(gNMDA=0.765)},
}

# These are acquired only if the compound adaptation cell has a valid,
# familywise-significant effect.  They distinguish pre-spike aM from post-spike
# dM without retrospectively expanding the primary screen.
TRIGGERED_REGISTRY: dict[str, dict[str, Any]] = {
    "aM_0p008_dM_10": {"factor": "aM_attribution", "values": _vector(aM=0.008)},
    "aM_0p02_dM_8": {"factor": "dM_attribution", "values": _vector(dM=8.0)},
}
ALL_REGISTRY = {**PRIMARY_REGISTRY, **TRIGGERED_REGISTRY}


def validate_registry() -> None:
    """Fail closed if labels, vectors, or one-factor semantics drift."""

    if len(PRIMARY_REGISTRY) != 13 or len(ALL_REGISTRY) != 15:
        raise AssertionError("response-gain perturbation registry cardinality drift")
    ordered_names = tuple(BASELINE)
    seen: set[tuple[float, ...]] = set()
    for label, cell in ALL_REGISTRY.items():
        values = cell.get("values")
        if tuple(values) != ordered_names:
            raise AssertionError(f"{label}: mechanism vector keys/order drift")
        vector = tuple(float(values[name]) for name in ordered_names)
        if vector in seen:
            raise AssertionError(f"{label}: duplicate mechanism vector")
        seen.add(vector)
        changed = [name for name in ordered_names if float(values[name]) != BASELINE[name]]
        if label == "shipped" and changed:
            raise AssertionError("shipped cell is not the dm10 operating point")
        if label in {"adaptation_off", "adaptation_prior"} and changed != ["aM", "dM"]:
            raise AssertionError(f"{label}: historical adaptation must change aM and dM")
        if label not in {"shipped", "adaptation_off", "adaptation_prior"} and len(changed) != 1:
            raise AssertionError(f"{label}: scalar cell must change exactly one mechanism")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write strict JSON with fsync + atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
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


def _strict_json(value: Any) -> Any:
    """Convert NumPy values and nonfinite floats to portable strict JSON."""

    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_strict_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def checkpoint_path(seed: int) -> Path:
    if int(seed) not in CHECKPOINT_SEEDS:
        raise ValueError(f"checkpoint seed must be one of {CHECKPOINT_SEEDS}")
    path = ROOT / "checkpoint" / CHECKPOINT_TEMPLATE.format(seed=int(seed))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _physical_gpu_rows() -> list[dict[str, str]]:
    completed = subprocess.run(
        ("nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader,nounits"),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        uuid, name = (part.strip() for part in line.split(",", 1))
        rows.append({"uuid": uuid, "name": name})
    return rows


def assert_rtx5090(device: str = "cuda:0") -> dict[str, Any]:
    """Require UUID-selected physical RTX 5090 before scientific CUDA work."""

    if device != "cuda:0":
        raise RuntimeError("scientific protocol requires visible cuda:0")
    rows = _physical_gpu_rows()
    matches = [row for row in rows if row["uuid"] == RTX5090_UUID]
    if len(matches) != 1 or RTX5090_NAME_TOKEN not in matches[0]["name"]:
        raise RuntimeError(f"required RTX 5090 UUID/name not found: {matches}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != RTX5090_UUID:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be the physical RTX 5090 UUID; "
            f"got {visible!r}"
        )
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits this setting only before inter-op work begins.  A
        # fresh scientific cell normally takes the first branch; the fallback
        # is relevant only to direct in-process unit probes.
        if torch.get_num_interop_threads() != 1:
            raise
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("RTX 5090 must be the sole CUDA-visible device")
    props = torch.cuda.get_device_properties(0)
    if RTX5090_NAME_TOKEN not in props.name:
        raise RuntimeError(f"visible cuda:0 is not RTX 5090: {props.name}")
    return {
        "physical_uuid": RTX5090_UUID,
        "physical_name": matches[0]["name"],
        "visible_index": 0,
        "torch_name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": int(props.total_memory),
        "cuda_visible_devices": visible,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
    }


def effective_mechanisms(net: Any) -> dict[str, float]:
    return {name: float(getattr(net, name)) for name in BASELINE}


def apply_registered_setting(net: Any, setting: str) -> dict[str, float]:
    """Assert the shipped load, then apply one live inference-time vector."""

    validate_registry()
    if setting not in ALL_REGISTRY:
        raise ValueError(f"unknown registered setting: {setting}")
    loaded = effective_mechanisms(net)
    for name, expected in BASELINE.items():
        if not math.isclose(loaded[name], expected, rel_tol=0.0, abs_tol=1e-9):
            raise AssertionError(f"loaded net.{name}={loaded[name]} != shipped {expected}")
    requested = ALL_REGISTRY[setting]["values"]
    for name, value in requested.items():
        setattr(net, name, float(value))
    effective = effective_mechanisms(net)
    for name, expected in requested.items():
        if not math.isclose(effective[name], expected, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(
                f"post-load override failed: net.{name}={effective[name]} != {expected}"
            )
    return effective


def restore_gate5_then_apply(
    net: Any, mutable_hparams: Mapping[str, Any], setting: str, g_rec: float = 0.03
) -> dict[str, float]:
    """Mirror official Gate 5 restoration, then apply the registered lesion.

    Gate 5 explicitly restores checkpoint ``gNMDA``/``tau_nmda`` after loading.
    Keeping this ordering in one tested hook prevents an NMDA perturbation from
    being silently overwritten.
    """

    net.gNMDA = float(mutable_hparams.get("gNMDA", BASELINE["gNMDA"]))
    net.tau_nmda = float(mutable_hparams.get("tau_nmda", 40.0))
    net.g_rec = float(g_rec)
    return apply_registered_setting(net, setting)


def _state_dict_sha256(net: Any) -> str:
    """Hash state-dict names, dtypes, shapes, and exact tensor bytes."""

    digest = hashlib.sha256()
    for name, tensor in sorted(net.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _configure_import_environment() -> None:
    """Install the shipped dm10 constructor/runtime environment."""

    values = {
        "VAL36_BUILD": "delayfix",
        "TAU_GABA": "18",
        "GNMDA": "0.51",
        "TAU_NMDA": "40",
        "G_REC": "0.03",
        "EXP_G_REC": "0.03",
        "DEND_COUPLING_ALPHA": "2",
        "MG_VHALF": "-48",
        "MG_VHALF_INH": "-30",
        "MG_K": "0.15",
        "GABA_SHUNT_SURR": "1",
        "K_SHUNT_SURR": "0.026",
        "E_GABA": "-70.0",
        "SIGMA_DL_FRAMES": "3",
        "MPLBACKEND": "Agg",
        # Official Gate 5 uses one fixed geometry stream across checkpoints.
        "SEED": "42",
    }
    for key, value in values.items():
        os.environ[key] = value
    os.environ.pop("TAU_NMDA_V", None)


def _load_official_modules() -> tuple[Any, Any, Any, Any]:
    """Import official kernels, then correct their legacy tau10 import env."""

    _configure_import_environment()
    root = str(ROOT)
    measure_dir = str(ROOT / "measure")
    for path in (measure_dir, root):
        if path not in sys.path:
            sys.path.insert(0, path)
    # This import configures the official dense Gate-5 grid and NT=60.  It also
    # writes legacy tau10 values into os.environ, which the dm10 shim has always
    # corrected after import and before load_ckpt.
    from measure import measure_ens_cre as gate5

    _configure_import_environment()
    import inverse_effectiveness_routec as gate4
    import torch

    network_io = gate5.N
    return torch, network_io, gate4, gate5


def _git_provenance() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ("git", *arguments), cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout.strip()

    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "status_porcelain": git("status", "--porcelain"),
    }


def _load_frozen(
    torch: Any, network_io: Any, path: Path, seed: int, batch_size: int, device: str
) -> tuple[Any, Any, Mapping[str, Any], dict[str, float], str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    mutable_hparams = checkpoint.get("mutable_hparams", {})
    loaded = network_io.load_ckpt(str(path), int(seed), int(batch_size), device)
    net, load_result = loaded[0], loaded[1]
    if bool(net.plasticity_enabled):
        raise AssertionError("plasticity must be disabled for frozen inference")
    effective = effective_mechanisms(net)
    for name, expected in BASELINE.items():
        if not math.isclose(effective[name], expected, rel_tol=0.0, abs_tol=1e-9):
            raise AssertionError(f"checkpoint restored net.{name}={effective[name]} != {expected}")
    state_sha = _state_dict_sha256(net)
    return net, load_result, mutable_hparams, effective, state_sha


def _gate4_record(torch: Any, network_io: Any, gate4: Any, path: Path, seed: int,
                  setting: str, device: str) -> dict[str, Any]:
    net, load_result, _, loaded_effective, state_before = _load_frozen(
        torch, network_io, path, seed, 1, device
    )
    effective = apply_registered_setting(net, setting)
    effective_before_measure = effective_mechanisms(net)
    with torch.inference_mode():
        measured = gate4.measure_ie(
            net, float(net.sigma_in), int(gate4.DEFAULT_NTRIALS), device
        )
    effective_after = effective_mechanisms(net)
    state_after = _state_dict_sha256(net)
    if state_after != state_before:
        raise AssertionError("Gate 4 mutated the checkpoint state_dict")
    if effective_after != effective_before_measure:
        raise AssertionError("Gate 4 changed the live perturbation vector")

    intensities = np.asarray(measured["intensities"], dtype=float)
    r_a = np.asarray(measured["R_A"], dtype=float)
    r_v = np.asarray(measured["R_V"], dtype=float)
    r_av = np.asarray(measured["R_B"], dtype=float)
    best = np.maximum(r_a, r_v)
    mei = np.asarray(measured["mei"], dtype=float)
    finite = np.isfinite(mei)
    slope = float("nan")
    intercept = float("nan")
    if finite.sum() >= 3:
        slope, intercept = (float(value) for value in np.polyfit(np.log(intensities[finite]), mei[finite], 1))
    return _strict_json({
        "protocol": "official_gate4_inverse_effectiveness_routec.measure_ie",
        "intensities": intensities,
        "n_trials": int(gate4.DEFAULT_NTRIALS),
        "pulse_frames": int(gate4.PULSE_LEN),
        "total_frames": int(gate4.N_FRAMES),
        "sigma_in": float(net.sigma_in),
        "R_A": r_a,
        "R_V": r_v,
        "R_AV": r_av,
        "best_unisensory": best,
        "mei": mei,
        "summary": {
            "mei_vs_log_intensity_slope": slope,
            "mei_vs_log_intensity_intercept": intercept,
            "low_minus_high_mei": float(mei[0] - mei[-1]),
        },
        "qc": {
            "responses_nonnegative": bool(np.all(r_a >= 0) and np.all(r_v >= 0) and np.all(r_av >= 0)),
            "all_denominators_positive": bool(np.all(best > 0)),
            "all_mei_finite": bool(np.all(finite)),
            "fit_finite": bool(math.isfinite(slope) and math.isfinite(intercept)),
            "zero_response_cells": int(np.sum(np.concatenate((r_a, r_v, r_av)) <= 0)),
        },
        "loaded_effective": loaded_effective,
        "effective": effective,
        "effective_after": effective_after,
        "load_missing_keys": list(load_result.missing_keys),
        "load_unexpected_keys": list(load_result.unexpected_keys),
        "state_sha256_before": state_before,
        "state_sha256_after": state_after,
        "state_bit_identical": state_before == state_after,
    })


def capture_gate5_profiles_chunked(net: Any, gate5_kernel: Any, chunk_size: int) -> tuple[Any, ...]:
    """Acquire Gate-5 profiles through the exact serial compatibility path.

    The official kernel serializes 33 disparities and three reset-state
    conditions, producing 1,980 Python calls into a 100-substep CUDA update.
    The dedicated path reproduces every scientific detail while making reset
    stochasticity explicit and auditable:

    * the exact ``default_rng(42)`` spatial draws are generated one disparity
      at a time;
    * the exact two 60-element afferent-jitter draws are generated in original
      disparity/AV/A/V order and injected after each batched reset;
    * AV, A, and V still start from separate freshly reset network states;
    * the official inputs, duration, response profiles, and downstream
      ratio-of-means computation are unchanged.

    ``chunk_size=1`` is the only confirmatory mode.  Larger batches change the
    floating-point reduction path and were empirically non-identical to the
    official kernel, so they are rejected rather than treated as equivalent.
    """

    if int(chunk_size) != 1:
        raise ValueError("confirmatory Gate 5 requires exact serial chunk_size=1")
    import torch

    n, space_size = int(net.n), int(net.space_size)
    n_trials = int(gate5_kernel.NT)
    device = net.device
    xs = torch.arange(n, device=device, dtype=torch.float32)
    initial_g_ff, initial_step = net.g_FFinh, net.step_counter
    rng = np.random.default_rng(int(gate5_kernel.SEED))
    geometries: list[tuple[Any, Any, np.ndarray, np.ndarray]] = []
    jitter: list[dict[str, tuple[Any, Any]]] = []

    # load_ckpt restores afferent_jitter_ms=4 after construction, while the
    # constructor may still have allocated zero-margin delay buffers.  The
    # official first reset recomputes these margins before drawing jitter.  We
    # must do the same before precomputing the exact serial draws; otherwise
    # every draw is incorrectly clipped to zero.
    net._reset_delay_buffers()
    if float(getattr(net, "afferent_jitter_ms", 0.0)) > 0.0:
        if int(net._aff_margin_a) <= 0 or int(net._aff_margin_v) <= 0:
            raise AssertionError(
                "positive afferent jitter requires positive A/V delay-buffer margins"
            )

    def to_index(degrees: np.ndarray) -> Any:
        return torch.round(
            torch.as_tensor(degrees, device=device, dtype=torch.float32)
            * (n - 1) / (space_size - 1)
        ).long()

    def gaussian(indices: Any) -> Any:
        return torch.exp(-0.5 * ((xs - indices[:, None]) / net.sigma_in) ** 2) * float(
            gate5_kernel.INTENSITY
        )

    def draw_jitter() -> tuple[Any, Any]:
        if float(getattr(net, "afferent_jitter_ms", 0.0)) <= 0.0:
            return None, None
        sigma = float(net.afferent_jitter_ms) / float(net.dt)
        jitter_a = torch.round(torch.randn(n_trials, device=device) * sigma).long().clamp_(
            -int(net._aff_margin_a), int(net._aff_margin_a)
        )
        jitter_v = torch.round(torch.randn(n_trials, device=device) * sigma).long().clamp_(
            -int(net._aff_margin_v), int(net._aff_margin_v)
        )
        return jitter_a, jitter_v

    # Geometry and reset-time stochastic state are generated in the official
    # serial order before execution.  update_all_layers_batch itself draws no
    # random values in this inference protocol.
    for separation in gate5_kernel.SEPS:
        base = rng.integers(0, space_size, size=n_trials)
        loc_a = np.asarray(base)
        loc_v = np.asarray(
            gate5_kernel.V.SBW._second_stim_loc(base, separation, space_size, linear=False)
        )
        g_a, g_v = gaussian(to_index(loc_a)), gaussian(to_index(loc_v))
        geometries.append((g_a, g_v, loc_a, loc_v))
        jitter.append({condition: draw_jitter() for condition in ("AV", "A", "V")})

    profiles: dict[str, list[Any | None]] = {
        condition: [None] * len(geometries) for condition in ("AV", "A", "V")
    }
    for start in range(0, len(geometries), int(chunk_size)):
        stop = min(start + int(chunk_size), len(geometries))
        indices = list(range(start, stop))
        for condition in ("AV", "A", "V"):
            g_a = torch.cat([geometries[index][0] for index in indices], dim=0)
            g_v = torch.cat([geometries[index][1] for index in indices], dim=0)
            zeros = torch.zeros_like(g_a)
            stimulus_a = g_a if condition in ("AV", "A") else zeros
            stimulus_v = g_v if condition in ("AV", "V") else zeros
            batch_size = len(indices) * n_trials

            net.g_FFinh, net.step_counter = initial_g_ff, initial_step
            # reset_state must allocate the normal jitter-aware ring buffers,
            # but its larger-batch random draws are not part of the official
            # serial stream.  Restore the RNG immediately and inject the exact
            # precomputed per-disparity draws.
            cuda_rng = torch.cuda.get_rng_state(device)
            net.reset_state(batch_size=batch_size)
            torch.cuda.set_rng_state(cuda_rng, device)
            if float(getattr(net, "afferent_jitter_ms", 0.0)) > 0.0:
                net._aff_jitter_a = torch.cat([jitter[index][condition][0] for index in indices])
                net._aff_jitter_v = torch.cat([jitter[index][condition][1] for index in indices])
                net._aff_batch_arange = torch.arange(batch_size, device=device)
            accumulated = torch.zeros(batch_size, n, device=device)
            for _ in range(int(gate5_kernel.DURATION)):
                accumulated += net.update_all_layers_batch(
                    stimulus_a, stimulus_v, return_spike_sum=True
                )[-1]
            shaped = accumulated.reshape(len(indices), n_trials, n).cpu().numpy()
            for local_index, separation_index in enumerate(indices):
                profiles[condition][separation_index] = shaped[local_index]
        print(
            f"[gate5] completed disparities {start + 1}-{stop}/{len(geometries)} "
            f"(chunk={chunk_size})",
            flush=True,
        )
    net.g_FFinh, net.step_counter = initial_g_ff, initial_step
    loc_a_all = np.stack([geometry[2] for geometry in geometries])
    loc_v_all = np.stack([geometry[3] for geometry in geometries])
    return (
        np.stack(profiles["AV"]), np.stack(profiles["A"]), np.stack(profiles["V"]),
        loc_a_all, loc_v_all, n, space_size,
    )


def _gate5_record(torch: Any, network_io: Any, gate5: Any, path: Path, seed: int,
                  setting: str, device: str, chunk_size: int) -> dict[str, Any]:
    net, load_result, mutable_hparams, loaded_effective, state_before = _load_frozen(
        torch, network_io, path, seed, int(gate5.G.NT), device
    )
    effective = restore_gate5_then_apply(net, mutable_hparams, setting, float(gate5.ENV["G_REC"]))
    effective_before_measure = effective_mechanisms(net)
    with torch.inference_mode():
        av, auditory, visual, _, _, _, _ = capture_gate5_profiles_chunked(
            net, gate5.G, int(chunk_size)
        )
    seps = np.asarray(gate5.G.SEPS, dtype=float)
    r_av_trials = av.max(2)
    r_a_trials = auditory.max(2)
    r_v_trials = visual.max(2)
    block = gate5.G.cre_block(r_av_trials, r_a_trials, r_v_trials)
    cre = np.asarray(block["cre_rom"], dtype=float)
    if np.all(np.isfinite(cre)):
        fit = gate5.fit_center_surround(seps, cre)
    else:
        # The official normalized curve and its spatial fit are undefined when
        # a lesion removes the unisensory denominator.  Preserve the absolute
        # responses and encode derived quantities as null; this is a scientific
        # collapse outcome, not an acquisition failure.
        fit = {
            "peak_cre": float("nan"), "peak_d": float("nan"),
            "zero_cross": None, "cen_hwhm": float("nan"),
            "surr_min": float("nan"), "surr_min_d": float("nan"),
            "far_floor": float("nan"),
        }
    effective_after = effective_mechanisms(net)
    state_after = _state_dict_sha256(net)
    if state_after != state_before:
        raise AssertionError("Gate 5 mutated the checkpoint state_dict")
    if effective_after != effective_before_measure:
        raise AssertionError("Gate 5 changed the live perturbation vector")

    r_a = np.asarray(block["mRA"], dtype=float)
    r_v = np.asarray(block["mRV"], dtype=float)
    r_av = np.asarray(block["mRAV"], dtype=float)
    best = np.maximum(r_a, r_v)
    return _strict_json({
        "protocol": "official_gate5_peak_population_CRE",
        "disparities_deg": seps,
        "n_trials": int(gate5.G.NT),
        "duration_frames": int(gate5.G.DURATION),
        "intensity": float(gate5.G.INTENSITY),
        "geometry_rng_seed": int(gate5.G.SEED),
        "response_definition": "peak neuron of time-integrated MSI population profile",
        "execution": {
            "disparity_chunk_size": int(chunk_size),
            "semantics": "official Gate5 conditions with exact serial geometry/jitter streams",
        },
        "R_A": r_a,
        "R_V": r_v,
        "R_AV": r_av,
        "best_unisensory": best,
        "cre_percent_ratio_of_means": cre,
        "summary": {
            "peak_cre": fit["peak_cre"],
            "peak_disparity_deg": fit["peak_d"],
            "zero_cross_deg": fit["zero_cross"],
            "center_hwhm_deg": fit["cen_hwhm"],
            "surround_min_cre": fit["surr_min"],
            "surround_min_disparity_deg": fit["surr_min_d"],
            "far_floor_cre": fit["far_floor"],
        },
        "qc": {
            "responses_nonnegative": bool(np.all(r_a >= 0) and np.all(r_v >= 0) and np.all(r_av >= 0)),
            "all_denominators_positive": bool(np.all(best > 0)),
            "all_cre_finite": bool(np.all(np.isfinite(cre))),
            "center_fit_finite": bool(math.isfinite(float(fit["cen_hwhm"]))),
            "zero_response_cells": int(np.sum(np.concatenate((r_a, r_v, r_av)) <= 0)),
        },
        "loaded_effective": loaded_effective,
        "effective": effective,
        "effective_after": effective_after,
        "load_missing_keys": list(load_result.missing_keys),
        "load_unexpected_keys": list(load_result.unexpected_keys),
        "state_sha256_before": state_before,
        "state_sha256_after": state_after,
        "state_bit_identical": state_before == state_after,
    })


def _array_sha256(array: Any) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii") + b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def run_gate5_equivalence_probe(
    seed: int, setting: str, output: Path, device: str = "cuda:0"
) -> dict[str, Any]:
    """Live zero-tolerance one-disparity official-vs-custom regression."""

    validate_registry()
    path = checkpoint_path(seed)
    device_record = assert_rtx5090(device)
    torch, network_io, _, gate5 = _load_official_modules()
    readouts_before = network_io.assert_frozen_readouts("GAIN PROBE BEFORE")
    original_seps = list(gate5.G.SEPS)
    gate5.G.SEPS = [0.0]
    try:
        official_net, _, official_hparams, _, official_state = _load_frozen(
            torch, network_io, path, seed, int(gate5.G.NT), device
        )
        restore_gate5_then_apply(
            official_net, official_hparams, setting, float(gate5.ENV["G_REC"])
        )
        with torch.inference_mode():
            official = gate5.G.capture_profiles(official_net)
        if _state_dict_sha256(official_net) != official_state:
            raise AssertionError("official one-disparity probe mutated state_dict")
        del official_net
        torch.cuda.empty_cache()

        custom_net, _, custom_hparams, _, custom_state = _load_frozen(
            torch, network_io, path, seed, int(gate5.G.NT), device
        )
        restore_gate5_then_apply(custom_net, custom_hparams, setting, float(gate5.ENV["G_REC"]))
        with torch.inference_mode():
            custom = capture_gate5_profiles_chunked(custom_net, gate5.G, 1)
        if _state_dict_sha256(custom_net) != custom_state:
            raise AssertionError("custom one-disparity probe mutated state_dict")
        del custom_net
        torch.cuda.empty_cache()
    finally:
        gate5.G.SEPS = original_seps

    labels = ("AV_profiles", "A_profiles", "V_profiles", "loc_A", "loc_V")
    comparisons = {}
    for index, label in enumerate(labels):
        official_array = np.asarray(official[index])
        custom_array = np.asarray(custom[index])
        equal = bool(np.array_equal(official_array, custom_array))
        comparisons[label] = {
            "equal": equal,
            "official_sha256": _array_sha256(official_array),
            "custom_sha256": _array_sha256(custom_array),
            "shape": list(official_array.shape),
        }
    comparisons["n"] = {"equal": int(official[5]) == int(custom[5])}
    comparisons["space_size"] = {"equal": int(official[6]) == int(custom[6])}
    passed = bool(all(item["equal"] for item in comparisons.values()))
    if not passed:
        raise AssertionError(f"live Gate5 official/custom regression failed: {comparisons}")
    readouts_after = network_io.assert_frozen_readouts("GAIN PROBE AFTER")
    if readouts_after != readouts_before:
        raise AssertionError("frozen readouts changed during Gate5 equivalence probe")
    result = {
        "schema_version": "fsts-gate5-live-equivalence-v1",
        "protocol_id": PROTOCOL_ID,
        "checkpoint_seed": int(seed),
        "setting": setting,
        "disparities_deg": [0.0],
        "n_trials": int(gate5.G.NT),
        "device": device_record,
        "comparisons": comparisons,
        "zero_tolerance_pass": passed,
        "readout_md5_before": readouts_before,
        "readout_md5_after": readouts_after,
        "driver_sha256": sha256_file(Path(__file__).resolve()),
    }
    result["payload_sha256"] = canonical_sha256(result)
    atomic_write_json(output, result)
    print(
        f"[gate5-equivalence] PASS seed{seed}/{setting} payload={result['payload_sha256']}",
        flush=True,
    )
    return result


def run_cell(
    seed: int, setting: str, output: Path, device: str = "cuda:0", gate5_chunk_size: int = 1
) -> dict[str, Any]:
    """Acquire one checkpoint/setting cell using two fresh network loads."""

    validate_registry()
    if setting not in ALL_REGISTRY:
        raise ValueError(f"unregistered setting: {setting}")
    if int(gate5_chunk_size) != 1:
        raise ValueError("confirmatory Gate 5 requires gate5_chunk_size=1")
    path = checkpoint_path(seed)
    started = time.monotonic()
    print(
        f"[cell seed{seed}/{setting}] START chunk={gate5_chunk_size} "
        f"pid={os.getpid()}",
        flush=True,
    )
    device_record = assert_rtx5090(device)
    torch, network_io, gate4, gate5 = _load_official_modules()
    readouts_before = network_io.assert_frozen_readouts("GAIN BEFORE")
    gate4_started = time.monotonic()
    print(f"[cell seed{seed}/{setting}] Gate4 START", flush=True)
    gate4_record = _gate4_record(torch, network_io, gate4, path, seed, setting, device)
    print(
        f"[cell seed{seed}/{setting}] Gate4 DONE "
        f"elapsed={time.monotonic() - gate4_started:.1f}s",
        flush=True,
    )
    torch.cuda.empty_cache()
    gate5_started = time.monotonic()
    print(f"[cell seed{seed}/{setting}] Gate5 START", flush=True)
    gate5_record = _gate5_record(
        torch, network_io, gate5, path, seed, setting, device, int(gate5_chunk_size)
    )
    print(
        f"[cell seed{seed}/{setting}] Gate5 DONE "
        f"elapsed={time.monotonic() - gate5_started:.1f}s",
        flush=True,
    )
    torch.cuda.empty_cache()
    readouts_after = network_io.assert_frozen_readouts("GAIN AFTER")
    if readouts_after != readouts_before:
        raise AssertionError("frozen readout hashes changed")

    sources = {
        "driver": sha256_file(Path(__file__).resolve()),
        "gate4": sha256_file(ROOT / "inverse_effectiveness_routec.py"),
        "gate5": sha256_file(ROOT / "measure" / "measure_ens_cre.py"),
        "gate5_capture": sha256_file(ROOT / "diag_136b_cre_5seed.py"),
        "network_io": sha256_file(ROOT / "routec_net_io.py"),
    }
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "checkpoint": {
            "seed": int(seed),
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        },
        "setting": {
            "id": setting,
            "factor": ALL_REGISTRY[setting]["factor"],
            "requested": dict(ALL_REGISTRY[setting]["values"]),
            "primary_registry": setting in PRIMARY_REGISTRY,
            "triggered_registry": setting in TRIGGERED_REGISTRY,
        },
        "device": device_record,
        "gate4": gate4_record,
        "gate5": gate5_record,
        "integrity": {
            "readout_md5_before": readouts_before,
            "readout_md5_after": readouts_after,
            "readouts_unchanged": readouts_before == readouts_after,
            "plasticity_enabled": False,
        },
        "code": {"source_sha256": sources, "git": _git_provenance()},
    }
    record = _strict_json(record)
    record["integrity"]["scientific_payload_sha256"] = canonical_sha256(record)
    atomic_write_json(output, record)
    print(
        f"[cell seed{seed}/{setting}] COMPLETE elapsed={time.monotonic() - started:.1f}s "
        f"payload={record['integrity']['scientific_payload_sha256']}",
        flush=True,
    )
    return record


def cell_output(run_dir: Path, seed: int, setting: str) -> Path:
    return run_dir / "raw" / f"seed{int(seed)}" / f"{setting}.json"


def _valid_existing(
    path: Path, seed: int, setting: str, *, expected_chunk_size: int | None = None,
    require_current_driver: bool = False,
) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        expected_hash = record["integrity"].pop("scientific_payload_sha256")
        actual_hash = canonical_sha256(record)
        record["integrity"]["scientific_payload_sha256"] = expected_hash
        valid = (
            record["schema_version"] == SCHEMA_VERSION
            and record["protocol_id"] == PROTOCOL_ID
            and int(record["checkpoint"]["seed"]) == int(seed)
            and record["setting"]["id"] == setting
            and expected_hash == actual_hash
        )
        if expected_chunk_size is not None:
            valid = valid and int(record["gate5"]["execution"]["disparity_chunk_size"]) == int(
                expected_chunk_size
            )
        if require_current_driver:
            valid = valid and record["code"]["source_sha256"]["driver"] == sha256_file(
                Path(__file__).resolve()
            )
        return bool(valid)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


class _ShutdownRequested(Exception):
    """Internal control-flow exception raised by launcher signal handlers."""


def _stop_processes(active: Sequence[tuple[subprocess.Popen[Any], Any, int, str]]) -> None:
    """Terminate active cell sessions and close their logs without replacements."""

    for process, _, _, _ in active:
        if process.poll() is None:
            process.terminate()
    for process, handle, _, _ in active:
        if process.poll() is None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        handle.close()


def launch_matrix(
    run_dir: Path, seeds: Sequence[int], settings: Sequence[str], workers: int,
    gate5_chunk_size: int,
) -> None:
    """Run cells in independent processes, checkpointing status and logs."""

    validate_registry()
    if workers < 1 or workers > 4:
        raise ValueError("workers must be in [1,4]")
    if gate5_chunk_size != 1:
        raise ValueError("confirmatory Gate 5 requires gate5_chunk_size=1")
    if any(int(seed) not in CHECKPOINT_SEEDS for seed in seeds):
        raise ValueError("unregistered checkpoint seed")
    if any(setting not in ALL_REGISTRY for setting in settings):
        raise ValueError("unregistered perturbation setting")
    # Hardware preflight without initializing CUDA in the orchestration process.
    matches = [row for row in _physical_gpu_rows() if row["uuid"] == RTX5090_UUID]
    if len(matches) != 1 or RTX5090_NAME_TOKEN not in matches[0]["name"]:
        raise RuntimeError("required physical RTX 5090 is unavailable")

    run_dir = run_dir.resolve()
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [(int(seed), setting) for seed in seeds for setting in settings]
    pending = [
        job for job in jobs
        if not _valid_existing(
            cell_output(run_dir, *job), *job, expected_chunk_size=gate5_chunk_size,
            require_current_driver=True,
        )
    ]
    state: dict[str, Any] = {
        "schema_version": "fsts-response-gain-launch-status-v1",
        "protocol_id": PROTOCOL_ID,
        "workers": int(workers),
        "gate5_chunk_size": int(gate5_chunk_size),
        "total_cells": len(jobs),
        "completed_cells": len(jobs) - len(pending),
        "failed_cells": [],
        "running_cells": [],
        "pending_cells": [f"seed{seed}/{setting}" for seed, setting in pending],
        "rtx5090_uuid": RTX5090_UUID,
        "phase": "starting",
        "shutdown_requested": False,
    }
    status_path = run_dir / "status.json"
    atomic_write_json(status_path, state)
    active: list[tuple[subprocess.Popen[Any], Any, int, str]] = []
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = RTX5090_UUID
    environment["PYTHONUNBUFFERED"] = "1"
    environment.update(THREAD_CAPS)
    shutdown_signal: list[int] = []

    def request_shutdown(signum: int, _frame: Any) -> None:
        shutdown_signal.append(int(signum))
        raise _ShutdownRequested

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    for signum in previous_handlers:
        signal.signal(signum, request_shutdown)
    try:
        while pending or active:
            while pending and len(active) < workers and not shutdown_signal:
                seed, setting = pending.pop(0)
                output = cell_output(run_dir, seed, setting)
                output.parent.mkdir(parents=True, exist_ok=True)
                log_path = logs / f"seed{seed}__{setting}.log"
                handle = log_path.open("w", encoding="utf-8")
                command = [
                    sys.executable, str(Path(__file__).resolve()), "cell",
                    "--seed", str(seed), "--setting", setting,
                    "--output", str(output), "--device", "cuda:0",
                    "--gate5-chunk-size", str(gate5_chunk_size),
                ]
                process = subprocess.Popen(
                    command, cwd=ROOT, env=environment, stdout=handle,
                    stderr=subprocess.STDOUT, text=True, start_new_session=True,
                )
                active.append((process, handle, seed, setting))
                print(
                    f"[launcher] START seed{seed}/{setting} pid={process.pid} "
                    f"active={len(active)}/{workers}",
                    flush=True,
                )
            if shutdown_signal:
                raise _ShutdownRequested
            state["running_cells"] = [f"seed{seed}/{setting}" for _, _, seed, setting in active]
            state["pending_cells"] = [f"seed{seed}/{setting}" for seed, setting in pending]
            state["phase"] = "running"
            atomic_write_json(status_path, state)

            # Event-driven wait: block on one active process rather than polling.
            process, handle, seed, setting = active[0]
            return_code = process.wait()
            active.pop(0)
            handle.close()
            if return_code != 0:
                state["failed_cells"].append(
                    {"seed": seed, "setting": setting, "return_code": return_code}
                )
                for other, other_handle, _, _ in active:
                    other.terminate()
                    other.wait()
                    other_handle.close()
                state["running_cells"] = []
                atomic_write_json(status_path, state)
                raise RuntimeError(
                    f"cell seed{seed}/{setting} failed rc={return_code}; "
                    f"see {logs / f'seed{seed}__{setting}.log'}"
                )
            output = cell_output(run_dir, seed, setting)
            if not _valid_existing(
                output, seed, setting, expected_chunk_size=gate5_chunk_size,
                require_current_driver=True,
            ):
                raise RuntimeError(f"cell produced invalid payload: {output}")
            state["completed_cells"] += 1
            print(
                f"[launcher] DONE seed{seed}/{setting} "
                f"completed={state['completed_cells']}/{len(jobs)}",
                flush=True,
            )
    except _ShutdownRequested:
        state["shutdown_requested"] = True
        state["shutdown_signal"] = shutdown_signal[-1] if shutdown_signal else None
        state["phase"] = "stopping"
        atomic_write_json(status_path, state)
        _stop_processes(active)
        active.clear()
        state["phase"] = "stopped"
        raise SystemExit(128 + (shutdown_signal[-1] if shutdown_signal else signal.SIGTERM))
    finally:
        state["running_cells"] = []
        state["pending_cells"] = [f"seed{seed}/{setting}" for seed, setting in pending]
        if state["completed_cells"] == len(jobs):
            state["phase"] = "complete"
        atomic_write_json(status_path, state)
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    cell_parser = subparsers.add_parser("cell")
    cell_parser.add_argument("--seed", type=int, required=True)
    cell_parser.add_argument("--setting", choices=tuple(ALL_REGISTRY), required=True)
    cell_parser.add_argument("--output", type=Path, required=True)
    cell_parser.add_argument("--device", default="cuda:0")
    cell_parser.add_argument("--gate5-chunk-size", type=int, default=1)
    probe_parser = subparsers.add_parser("gate5-equivalence-probe")
    probe_parser.add_argument("--seed", type=int, default=42)
    probe_parser.add_argument("--setting", choices=tuple(ALL_REGISTRY), default="shipped")
    probe_parser.add_argument("--output", type=Path, required=True)
    probe_parser.add_argument("--device", default="cuda:0")
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--run-dir", type=Path, required=True)
    launch_parser.add_argument("--seeds", nargs="+", type=int, default=list(CHECKPOINT_SEEDS))
    launch_parser.add_argument("--settings", nargs="+", choices=tuple(ALL_REGISTRY),
                               default=list(PRIMARY_REGISTRY))
    launch_parser.add_argument("--workers", type=int, default=1)
    launch_parser.add_argument("--gate5-chunk-size", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "cell":
        run_cell(
            args.seed, args.setting, args.output, args.device, args.gate5_chunk_size
        )
        print(args.output, flush=True)
    elif args.command == "gate5-equivalence-probe":
        run_gate5_equivalence_probe(args.seed, args.setting, args.output, args.device)
    else:
        launch_matrix(
            args.run_dir, args.seeds, args.settings, args.workers, args.gate5_chunk_size
        )


validate_registry()


if __name__ == "__main__":
    main()
