"""Regression coverage for physical-time, state-frozen Fano evaluation."""

from pathlib import Path

import numpy as np
import torch

import fano_factor_test as fano


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
BASELINE_FRAMES = 30
STIM_FRAMES = 30
SEED_BASE = 0


def _bitwise_value(value) -> tuple[str, bytes]:
    """Return a dtype-tagged byte representation for scalar state comparison."""
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return array.dtype.str, array.tobytes()


def test_zero_mean_active_neuron_fano_is_undefined() -> None:
    """Inactive neurons must be NaN in the primary statistic, not epsilon-zero."""
    silent_counts = np.zeros((1, 2, 3), dtype=float)
    assert np.isnan(fano.active_neuron_fano_factor(silent_counts)[0])
    assert fano.fano_factor(silent_counts)[0] == 0.0


def test_fano_physical_time_m00(monkeypatch) -> None:
    """M00 must show an active-ROI Fano drop without mutating trained state."""
    assert torch.cuda.is_available(), "This regression requires the A6000 GPU"
    device_name = torch.cuda.get_device_name(0)
    assert "A6000" in device_name, f"Expected A6000, got {device_name}"
    print(f"CUDA device: {device_name}")

    original_loader = fano._load_msi_model
    captured: dict[str, object] = {}

    def state_capturing_loader(checkpoint: Path, *, device: str):
        """Capture the freshly loaded network before assay configuration."""
        net = original_loader(checkpoint, device=device)
        captured["net"] = net
        captured["parameters"] = {
            name: parameter.detach().clone()
            for name, parameter in net.named_parameters()
        }
        captured["g_FFinh"] = _bitwise_value(net.g_FFinh)
        captured["n_substeps"] = net.n_substeps
        return net

    monkeypatch.setattr(fano, "_load_msi_model", state_capturing_loader)
    result = fano.run_fano_factor_test_bio(
        [CHECKPOINT],
        n_trials=32,
        baseline_frames=BASELINE_FRAMES,
        stim_frames=STIM_FRAMES,
        device="cuda",
        seed_base=SEED_BASE,
        return_diagnostics=True,
    )

    net = captured["net"]
    assert captured["n_substeps"] == 100
    assert net.n_substeps == 100
    assert result["frame_ms"] == 10.0
    assert result["seeds"] == (SEED_BASE,)
    assert net.plasticity_enabled is False
    assert net.freeze_g_FFinh is True

    parameters_after = dict(net.named_parameters())
    parameters_before = captured["parameters"]
    assert parameters_after.keys() == parameters_before.keys()
    for name, parameter_before in parameters_before.items():
        assert torch.equal(parameter_before, parameters_after[name]), (
            f"Parameter changed during frozen Fano evaluation: {name}"
        )
    assert _bitwise_value(net.g_FFinh) == captured["g_FFinh"]

    expected_centre = int(round(90.0 * (net.n - 1) / (net.space_size - 1)))
    expected_radius = int(round(2.0 * net.sigma_in))
    assert result["roi"]["centre_index"] == expected_centre
    assert result["roi"]["radius_neurons"] == expected_radius

    first_100ms_frames = int(round(100.0 / result["frame_ms"]))
    primary_fano = result["primary"]["fano_mean"]
    roi_rate = result["primary"]["rate_mean"]
    conventional_roi = result["secondary"]["roi_conventional_fano_mean"]

    baseline_active = float(np.nanmean(primary_fano[:BASELINE_FRAMES]))
    early_active = float(np.nanmean(
        primary_fano[BASELINE_FRAMES:BASELINE_FRAMES + first_100ms_frames]
    ))
    baseline_rate = float(roi_rate[:BASELINE_FRAMES].mean())
    early_rate = float(
        roi_rate[BASELINE_FRAMES:BASELINE_FRAMES + first_100ms_frames].mean()
    )
    baseline_conventional = float(conventional_roi[:BASELINE_FRAMES].mean())
    early_conventional = float(
        conventional_roi[BASELINE_FRAMES:BASELINE_FRAMES + first_100ms_frames].mean()
    )

    print(
        "M00 Fano: "
        f"active={baseline_active:.6f}->{early_active:.6f}, "
        f"ROI rate={baseline_rate:.6f}->{early_rate:.6f}, "
        f"conventional ROI={baseline_conventional:.6f}->{early_conventional:.6f}"
    )
    assert 0.7 <= baseline_active <= 1.5
    assert 0.4 <= early_active <= 1.0
    assert baseline_active - early_active > 0.2
    assert 0.05 <= baseline_rate <= 0.2
    assert 0.08 <= early_rate <= 0.25
    assert 0.7 <= baseline_conventional <= 1.3
    assert 0.0 <= early_conventional <= 0.6


def test_fano_full_ensemble_quenches_and_rate_rises(monkeypatch) -> None:
    """The deterministic ten-model ensemble must quench FF and raise ROI rate."""
    assert torch.cuda.is_available(), "This regression requires the A6000 GPU"
    device_name = torch.cuda.get_device_name(0)
    assert "A6000" in device_name, f"Expected A6000, got {device_name}"
    print(f"CUDA device: {device_name}")

    checkpoints = sorted((ROOT / "checkpoint").glob("msi_model_surr_10_*.pt"))[:10]
    assert len(checkpoints) == 10
    original_loader = fano._load_msi_model
    state_records: list[dict[str, object]] = []

    def state_capturing_loader(checkpoint: Path, *, device: str):
        """Record each fresh ensemble member before assay configuration."""
        net = original_loader(checkpoint, device=device)
        state_records.append({
            "net": net,
            "parameters": {
                name: parameter.detach().clone()
                for name, parameter in net.named_parameters()
            },
            "g_FFinh": _bitwise_value(net.g_FFinh),
        })
        return net

    monkeypatch.setattr(fano, "_load_msi_model", state_capturing_loader)
    active_baselines, active_early = [], []
    rate_baselines, rate_early = [], []

    for model_index, checkpoint in enumerate(checkpoints):
        result = fano.run_fano_factor_test_bio(
            [checkpoint],
            n_trials=32,
            baseline_frames=BASELINE_FRAMES,
            stim_frames=STIM_FRAMES,
            device="cuda",
            seed_base=SEED_BASE + model_index,
            return_diagnostics=True,
        )
        first_100ms_frames = int(round(100.0 / result["frame_ms"]))
        active_fano = result["primary"]["fano_mean"]
        roi_rate = result["primary"]["rate_mean"]
        active_baselines.append(float(np.nanmean(active_fano[:BASELINE_FRAMES])))
        active_early.append(float(np.nanmean(
            active_fano[BASELINE_FRAMES:BASELINE_FRAMES + first_100ms_frames]
        )))
        rate_baselines.append(float(roi_rate[:BASELINE_FRAMES].mean()))
        rate_early.append(float(
            roi_rate[BASELINE_FRAMES:BASELINE_FRAMES + first_100ms_frames].mean()
        ))

        record = state_records[-1]
        net = record["net"]
        assert net.n_substeps == 100
        assert net.plasticity_enabled is False
        assert net.freeze_g_FFinh is True
        parameters_after = dict(net.named_parameters())
        parameters_before = record["parameters"]
        assert parameters_after.keys() == parameters_before.keys()
        for name, parameter_before in parameters_before.items():
            assert torch.equal(parameter_before, parameters_after[name]), (
                f"M{model_index:02d} parameter changed during Fano evaluation: {name}"
            )
        assert _bitwise_value(net.g_FFinh) == record["g_FFinh"]

    active_baselines = np.asarray(active_baselines)
    active_early = np.asarray(active_early)
    rate_baselines = np.asarray(rate_baselines)
    rate_early = np.asarray(rate_early)
    ensemble_active_baseline = float(active_baselines.mean())
    ensemble_active_early = float(active_early.mean())
    ensemble_rate_baseline = float(rate_baselines.mean())
    ensemble_rate_early = float(rate_early.mean())
    rate_rise_count = int(np.sum(rate_early > rate_baselines))

    print(
        "Full ensemble Fano: "
        f"active={ensemble_active_baseline:.6f}->{ensemble_active_early:.6f}, "
        f"ROI rate={ensemble_rate_baseline:.6f}->{ensemble_rate_early:.6f}, "
        f"FF quenches={int(np.sum(active_early < active_baselines))}/10, "
        f"rate rises={rate_rise_count}/10"
    )
    assert len(state_records) == 10
    assert np.all(active_early < active_baselines)
    assert ensemble_active_baseline - ensemble_active_early > 0.2
    assert ensemble_rate_early > ensemble_rate_baseline
    assert rate_rise_count >= 8
