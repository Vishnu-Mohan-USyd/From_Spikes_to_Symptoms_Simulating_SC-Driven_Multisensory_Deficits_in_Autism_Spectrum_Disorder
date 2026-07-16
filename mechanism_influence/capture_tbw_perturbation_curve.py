#!/usr/bin/env python3
"""Capture one canonical dm10 TBW perturbation curve from a frozen checkpoint.

This is the curve-preserving counterpart of :mod:`mechanism_influence.tbw_point`.
It uses the same stimulus path, trial count, SOA grid, measurement seed and
three repeat streams, but classifies the retained raw MSI temporal rasters
with the absolute sham-calibrated observer.  The former max-normalised
classifier is emitted only as a named regression control.  One process
measures one condition so inference-time scalar overrides cannot leak between
conditions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Exact dm10 constructor environment used by tbw_point.py.  The checkpoint's
# trained tau_GABA/gNMDA values remain in the environment for guarded loading;
# each lesion is applied to the live scalar only after the checkpoint loads.
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
    "SIGMA_DL_FRAMES": "3",
    "GNMDA": "0.51",
    "TAU_NMDA": "40",
    "G_REC": "0.03",
    "EXP_G_REC": "0.03",
    "K_DVDT": "0.0",
    "TAU_DVDT": "3.0",
    "V_THRESH_FLOOR": "20.0",
    "DVDT_CAP": "50.0",
    "AFFERENT_JITTER_MS": "4",
    "MPLBACKEND": "Agg",
    "U_STP_A": "0.2",
    "U_STP_V": "0.2",
    "NMDA_STD_SCALE": "0.8",
    "TAU_REC": "400.0",
}
for _key, _value in BASE_ENV.items():
    os.environ[_key] = _value
for _key in (
    "TAU_NMDA_V", "ADAPT_A", "ADAPT_DM", "ADAPT_A_INH", "ADAPT_DM_INH",
    "TAU_GABA_LOCAL", "TAU_GABA_SURROUND", "GNMDA_EXC", "GNMDA_INH",
):
    os.environ.pop(_key, None)
os.environ["SEED"] = "44"

import torch  # noqa: E402
import routec_net_io as net_io  # noqa: E402
import measure_107_convergence as convergence  # noqa: E402
from mechanism_influence.tbw_temporal_fusion_observer import (  # noqa: E402
    classify_temporal_rasters,
)


MDC = convergence.MDC
NT = 50
RUN_SEEDS = (0, 1, 2)
SIGMA_DL_FRAMES = 3.0
EXPECTED_READOUTS = {
    "TBW_test.py": "80d33465c4bf55d6e85b5990acb92da7",
    "SBW_test.py": "73b7d13626964d851cc090818b728311",
}
CONDITIONS = {
    "shipped": {
        "settings": {
            "aM": 0.02, "dM": 10.0, "pv_gaba_scale": 1.0,
            "tau_gaba": 18.0, "gNMDA": 0.51,
        },
        "expected_widths_ms": [260.0, 240.0, 240.0],
    },
    "adaptation_off": {
        "settings": {
            "aM": 0.0, "dM": 0.0, "pv_gaba_scale": 1.0,
            "tau_gaba": 18.0, "gNMDA": 0.51,
        },
        "expected_widths_ms": [280.0, 300.0, 300.0],
    },
    "tau_gaba_60": {
        "settings": {
            "aM": 0.02, "dM": 10.0, "pv_gaba_scale": 1.0,
            "tau_gaba": 60.0, "gNMDA": 0.51,
        },
        "expected_widths_ms": [300.0, 300.0, 300.0],
    },
    "gnmda_0p765": {
        "settings": {
            "aM": 0.02, "dM": 10.0, "pv_gaba_scale": 1.0,
            "tau_gaba": 18.0, "gNMDA": 0.765,
        },
        "expected_widths_ms": [260.0, 240.0, 260.0],
    },
    "adaptation_0p6": {
        "settings": {
            "aM": 0.012, "dM": 6.0, "pv_gaba_scale": 1.0,
            "tau_gaba": 18.0, "gNMDA": 0.51,
        },
        "expected_widths_ms": None,
    },
    "pv_gaba_0p5": {
        "settings": {
            "aM": 0.02, "dM": 10.0, "pv_gaba_scale": 0.5,
            "tau_gaba": 18.0, "tau_gaba_local": 18.0,
            "tau_gaba_surround": 18.0, "gNMDA": 0.51,
        },
        "expected_widths_ms": None,
    },
    "tau_gaba_local_10": {
        "settings": {
            "aM": 0.02, "dM": 10.0, "pv_gaba_scale": 1.0,
            "tau_gaba": 18.0, "tau_gaba_local": 10.0,
            "tau_gaba_surround": 18.0, "gNMDA": 0.51,
        },
        "expected_widths_ms": None,
    },
    "tau_gaba_local_40": {
        "settings": {
            "aM": 0.02, "dM": 10.0, "pv_gaba_scale": 1.0,
            "tau_gaba": 18.0, "tau_gaba_local": 40.0,
            "tau_gaba_surround": 18.0, "gNMDA": 0.51,
        },
        "expected_widths_ms": None,
    },
    "gnmda_0p383": {
        "settings": {
            "aM": 0.02, "dM": 10.0, "pv_gaba_scale": 1.0,
            "tau_gaba": 18.0, "gNMDA": 0.383,
        },
        "expected_widths_ms": None,
    },
}


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weight_fingerprint(net: torch.nn.Module) -> str:
    digest = hashlib.md5()
    for name in sorted(net.state_dict()):
        tensor = net.state_dict()[name]
        if torch.is_tensor(tensor):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def gpu_provenance() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Canonical TBW capture requires the RTX 5090, but CUDA is unavailable")
    name = torch.cuda.get_device_name(0)
    if "RTX 5090" not in name:
        raise RuntimeError(f"Canonical TBW capture requires RTX 5090; visible cuda:0 is {name!r}")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    physical = next((line for line in query if "RTX 5090" in line), "")
    fields = [item.strip() for item in physical.split(",", 3)]
    return {
        "torch_device": "cuda:0",
        "torch_device_name": name,
        "physical_index": int(fields[0]) if len(fields) == 4 else None,
        "uuid": fields[1] if len(fields) == 4 else None,
        "driver_version": fields[3] if len(fields) == 4 else None,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def raw_half_max_width(offsets: np.ndarray, probability: np.ndarray) -> tuple[float, float, float]:
    peak = float(np.max(probability))
    above = offsets[probability >= 0.5 * peak]
    if above.size < 2:
        return float("nan"), float("nan"), float("nan")
    left, right = float(np.min(above)), float(np.max(above))
    return right - left, left, right


@torch.no_grad()
def capture_temporal_rasters(
    net: torch.nn.Module,
    sigma_dL_frames: float,
    *,
    run_seed: int,
    n_trials: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the established Gate-2 stimulus path and retain raw MSI rasters.

    This is the forward section of ``MDC.tbw_fused_counts_jit`` with its
    seeds, timing jitter and state restoration unchanged.  Classification is
    intentionally deferred so the corrected and legacy-regression observers
    consume the exact same trials.

    Returns
    -------
    offsets_ms
        Nominal SOAs, shape ``(n_SOA,)``, in milliseconds.
    raster_msi
        Raw MSI population spikes, shape ``(60, n_SOA, n_trials)``.
    effective_offsets_frames
        Trial-specific A-to-V onset offsets after jitter, shape
        ``(n_SOA, n_trials)``, in 10-ms frames.
    """
    n_offsets = len(MDC.OFFS)
    initial_g_ff = net.g_FFinh
    initial_step = net.step_counter
    with MDC.H.seeded_default_rng(int(run_seed)):
        location_rng = np.random.default_rng()
        all_locations = location_rng.integers(
            0, net.space_size, size=(n_offsets, n_trials)
        )
    jitter_rng = np.random.default_rng([int(run_seed), MDC.JIT_RNG_TAG])
    location_sequences: list[list[int]] = []
    modality_sequences: list[list[str]] = []
    effective_offsets: list[int] = []
    for offset_index, offset in enumerate(MDC.OFFS):
        for trial_index in range(n_trials):
            location_sequence, modality_sequence = MDC.gen2(
                loc=int(all_locations[offset_index, trial_index]),
                T=MDC.T_STEPS,
                D=MDC.D,
                offset=offset,
                space_size=net.space_size,
                sigma_dL_frames=float(sigma_dL_frames),
                rng=jitter_rng,
            )
            location_sequences.append(location_sequence)
            modality_sequences.append(modality_sequence)
            effective_offsets.append(MDC.eff_offset_from_modseq(modality_sequence))

    total_batch = n_offsets * n_trials
    torch.manual_seed(int(run_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(run_seed))
    x_a, x_v, valid = MDC.genav(
        location_sequences,
        modality_sequences,
        [False] * total_batch,
        n=net.n,
        space_size=net.space_size,
        sigma_in=net.sigma_in,
        noise_std=0.0,
        device=net.device,
        max_len=MDC.T_STEPS,
        stimulus_intensity=MDC.STIM_IN,
    )
    net.g_FFinh = initial_g_ff
    net.step_counter = initial_step
    net.reset_state(total_batch)
    raster = torch.zeros((MDC.T_STEPS, total_batch), device=net.device)
    with torch.inference_mode():
        for frame in range(MDC.T_STEPS):
            returned = net.update_all_layers_batch(
                x_a[:, frame], x_v[:, frame], valid[:, frame],
                return_spike_sum=True,
            )
            raster[frame] = returned[-1].sum(dim=1)
    raster_msi = raster.cpu().numpy().reshape(MDC.T_STEPS, n_offsets, n_trials)
    offsets_ms = np.asarray(MDC.OFFS, dtype=float) * 10.0
    effective = np.asarray(effective_offsets, dtype=np.int64).reshape(n_offsets, n_trials)
    del raster, x_a, x_v, valid
    if net.device.type == "cuda":
        torch.cuda.empty_cache()
    net.g_FFinh = initial_g_ff
    net.step_counter = initial_step
    return offsets_ms, raster_msi, effective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--require-conductance-gaba",
        action="store_true",
        help=(
            "require the checkpoint-declared calibrated local GABA conductance "
            "equation; the old current-checkpoint legacy-width regression is then "
            "reported but not used as an acceptance gate"
        ),
    )
    args = parser.parse_args()

    checkpoint = args.ckpt.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    hardware = gpu_provenance()
    print(f"[{args.condition}] hardware={hardware['torch_device_name']} uuid={hardware['uuid']}", flush=True)

    before_readouts = net_io.assert_frozen_readouts(f"{args.condition} before")
    settings = CONDITIONS[args.condition]["settings"]
    net, load_result, epoch, _, _ = net_io.load_ckpt(str(checkpoint), 44, NT, "cuda:0")
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint did not load cleanly: missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    if args.require_conductance_gaba:
        expected_local_gaba = {
            "local_gaba_mode": "conductance",
            "local_gaba_conductance_per_mv": 0.0461574,
            "local_gaba_reversal_mv": -70.0,
        }
        actual_local_gaba = {
            key: getattr(net, key) for key in expected_local_gaba
        }
        if actual_local_gaba != expected_local_gaba:
            raise RuntimeError(
                "Checkpoint local-GABA configuration mismatch: "
                f"expected={expected_local_gaba}, actual={actual_local_gaba}"
            )
    net.g_rec = 0.03
    net.aM = float(settings["aM"])
    net.dM = float(settings["dM"])
    net.pv_gaba_scale = float(settings["pv_gaba_scale"])
    net.tau_gaba = float(settings["tau_gaba"])
    net.tau_gaba_local = settings.get("tau_gaba_local")
    net.tau_gaba_surround = settings.get("tau_gaba_surround")
    net.gNMDA = float(settings["gNMDA"])
    net.u_stp_a = 0.2
    net.u_stp_v = 0.2
    net.nmda_std_scale = 0.8
    net.u_a.fill_(0.2)
    net.u_v.fill_(0.2)
    net.plasticity_enabled = False

    effective = {
        "aM": float(net.aM),
        "dM": float(net.dM),
        "pv_gaba_scale": float(net.pv_gaba_scale),
        "tau_gaba_ms": float(net.tau_gaba),
        "gNMDA": float(net.gNMDA),
        "u_stp_a": float(net.u_stp_a),
        "u_stp_v": float(net.u_stp_v),
        "nmda_std_scale": float(net.nmda_std_scale),
        "g_rec": float(net.g_rec),
        "local_gaba_mode": str(net.local_gaba_mode),
        "local_gaba_conductance_per_mv": float(net.local_gaba_conductance_per_mv),
        "local_gaba_reversal_mv": float(net.local_gaba_reversal_mv),
        "resolved_path_controls": net._scientific_path_controls(),
    }
    weights_before = weight_fingerprint(net)
    runs: list[dict[str, Any]] = []
    for run_seed in RUN_SEEDS:
        offsets, raster_msi, effective_offsets = capture_temporal_rasters(
            net,
            SIGMA_DL_FRAMES,
            run_seed=run_seed,
            n_trials=NT,
        )
        corrected = classify_temporal_rasters(
            raster_msi, offsets, effective_offsets, observer="corrected"
        )
        legacy = classify_temporal_rasters(
            raster_msi, offsets, effective_offsets, observer="legacy_regression"
        )
        probability = np.asarray(corrected["p_fusion"], dtype=float)
        width, left, right = raw_half_max_width(offsets, probability)
        legacy_probability = np.asarray(legacy["p_fusion"], dtype=float)
        legacy_width, legacy_left, legacy_right = raw_half_max_width(
            offsets, legacy_probability
        )
        runs.append(
            {
                "run_seed": run_seed,
                "offsets_ms": offsets.tolist(),
                "fused_counts": corrected["fused_counts"],
                "fused_one_counts": corrected["fused_one_counts"],
                "fused_two_counts": corrected["fused_two_counts"],
                "separate_counts": corrected["separate_counts"],
                "dropout_counts": corrected["dropout_counts"],
                "dropout_ambiguous_counts": corrected["dropout_ambiguous_counts"],
                "no_response_counts": corrected["no_response_counts"],
                "n_trials": corrected["n_trials"],
                "p_fusion": probability.tolist(),
                "raw_half_max_width_ms": width,
                "raw_half_max_left_ms": left,
                "raw_half_max_right_ms": right,
                "peak_p_fusion": float(np.max(probability)),
                "overall_dropout_fraction": corrected["overall_dropout_fraction"],
                "overall_separate_fraction": corrected["overall_separate_fraction"],
                "effective_offset_mean_frames": float(np.mean(effective_offsets)),
                "effective_offset_sd_frames": float(np.std(effective_offsets, ddof=1)),
                "legacy_regression": {
                    "fused_counts": legacy["fused_counts"],
                    "p_fusion": legacy_probability.tolist(),
                    "raw_half_max_width_ms": legacy_width,
                    "raw_half_max_left_ms": legacy_left,
                    "raw_half_max_right_ms": legacy_right,
                    "peak_p_fusion": float(np.max(legacy_probability)),
                },
            }
        )

    weights_after = weight_fingerprint(net)
    after_readouts = net_io.assert_frozen_readouts(f"{args.condition} after")
    if weights_before != weights_after:
        raise RuntimeError(f"Frozen weight state changed: {weights_before} != {weights_after}")
    if before_readouts != after_readouts:
        raise RuntimeError(f"Frozen readout hashes changed: {before_readouts} != {after_readouts}")

    measured = [run["raw_half_max_width_ms"] for run in runs]
    legacy_measured = [run["legacy_regression"]["raw_half_max_width_ms"] for run in runs]
    legacy_expected = (
        None
        if args.require_conductance_gaba
        else CONDITIONS[args.condition]["expected_widths_ms"]
    )
    legacy_matches = None if legacy_expected is None else legacy_measured == legacy_expected
    output = {
        "protocol": "dm10-corrected-temporal-fusion-curves-v2",
        "condition": args.condition,
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint),
            "seed": 42,
            "epoch": int(epoch),
            "md5": file_hash(checkpoint, "md5"),
            "sha256": file_hash(checkpoint),
        },
        "measurement": {
            "measurement_seed": 44,
            "run_seeds": list(RUN_SEEDS),
            "n_trials_per_soa_per_run": NT,
            "sigma_dL_frames": SIGMA_DL_FRAMES,
            "soa_units": "ms",
            "fusion_definition": (
                "absolute sham-calibrated temporal-fusion observer on the raw 60-frame "
                "MSI population spike-count raster"
            ),
            "observer": {
                "default": "corrected",
                "legacy_mode": "legacy_regression",
                "profile_normalization": "none",
                "smoothing_sigma_frames": 2.0,
                "minimum_total_spikes": 10.0,
                "minimum_peak_separation_frames": 3,
                "two_peak_valley_threshold": 0.4,
                "one_peak_absolute_floor_spikes": 98.83377430574426,
                "auditory_latency_frames": 5,
                "visual_latency_frames": 7,
                "maximum_merged_latency_separation_frames": 6,
                "peak_interval_margin_frames": 1,
                "p_fusion_denominator": "all AV trials, including dropout/no-response",
            },
            "width_definition": "raw SOA span with P(fusion) >= 0.5 * within-run peak",
        },
        "effective_settings": effective,
        "hardware": hardware,
        "provenance": {
            "readout_md5": {
                "TBW_test.py": before_readouts[0],
                "SBW_test.py": before_readouts[1],
            },
            "source_sha256": {
                "capture_script": file_hash(Path(__file__).resolve()),
                "temporal_fusion_observer": file_hash(
                    ROOT / "mechanism_influence" / "tbw_temporal_fusion_observer.py"
                ),
                "tbw_point.py": file_hash(ROOT / "mechanism_influence" / "tbw_point.py"),
                "measure_develop_check.py": file_hash(ROOT / "measure_develop_check.py"),
                "Training_delayfix_d52.py": file_hash(ROOT / "Training_delayfix_d52.py"),
            },
            "weight_fingerprint_before": weights_before,
            "weight_fingerprint_after": weights_after,
            "weights_bit_identical": weights_before == weights_after,
        },
        "runs": runs,
        "summary": {
            "raw_half_max_widths_ms": measured,
            "median_raw_half_max_width_ms": float(np.median(measured)),
            "peak_p_fusion": [run["peak_p_fusion"] for run in runs],
            "overall_dropout_fraction": [run["overall_dropout_fraction"] for run in runs],
            "legacy_regression_raw_half_max_widths_ms": legacy_measured,
            "legacy_regression_expected_widths_ms": legacy_expected,
            "legacy_regression_widths_match_exactly": legacy_matches,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        f"[{args.condition}] widths={measured} median={np.median(measured):.0f} ms "
        f"legacy_widths={legacy_measured} legacy_expected={legacy_expected} "
        f"legacy_exact={legacy_matches} -> {args.out}",
        flush=True,
    )
    if legacy_matches is False:
        raise RuntimeError(
            "Legacy-regression widths differ from the canonical scalar artifact; "
            "stop and route to tbw_curve_debug"
        )


if __name__ == "__main__":
    main()
