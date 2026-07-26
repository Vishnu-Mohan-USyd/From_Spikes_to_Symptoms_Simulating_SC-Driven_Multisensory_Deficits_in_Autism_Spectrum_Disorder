"""Regression coverage for state-isolated inverse-effectiveness evaluation."""

from pathlib import Path

import numpy as np
import torch

from inverse_effectiveness_test import configure_inverse_eval, load_msi_model


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
LOC_DEG = 90
SIGMA_IN = 5.0
PULSE_LEN = 10
N_FRAMES = 20


@torch.no_grad()
def _integrated_spikes(net, condition: str, intensity: float) -> float:
    """Return total MSI spikes using the production A/V/B assay mechanic."""
    centre = LOC_DEG * (net.n - 1) / (net.space_size - 1)
    gaussian = torch.exp(
        -0.5
        * ((torch.arange(net.n, device=net.device) - centre) / SIGMA_IN) ** 2
    ) * intensity
    x_a = torch.zeros(N_FRAMES, net.n, device=net.device)
    x_v = torch.zeros_like(x_a)
    if condition in ("A", "B"):
        x_a[:PULSE_LEN] = gaussian
    if condition in ("V", "B"):
        x_v[:PULSE_LEN] = gaussian

    net.reset_state(batch_size=1)
    population_spikes = 0.0
    for frame in range(N_FRAMES):
        *_, spike_sum = net.update_all_layers_batch(
            x_a[frame:frame + 1],
            x_v[frame:frame + 1],
            return_spike_sum=True,
        )
        population_spikes += spike_sum.sum().item()
    return population_spikes


def _bitwise_value(value) -> tuple[str, bytes]:
    """Return a dtype-tagged byte representation for scalar state comparison."""
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return array.dtype.str, array.tobytes()


def test_frozen_eval_preserves_state_and_inverse_effectiveness() -> None:
    """M00 must retain parameters while MEI falls from low to high intensity."""
    assert torch.cuda.is_available(), "This regression requires the A6000 GPU"
    device_name = torch.cuda.get_device_name(0)
    assert "A6000" in device_name, f"Expected A6000, got {device_name}"
    print(f"CUDA device: {device_name}")

    torch.manual_seed(0)
    np.random.seed(0)
    mei_by_intensity: dict[float, float] = {}

    for intensity in (0.05, 1.6):
        net = load_msi_model(CHECKPOINT, device="cuda")
        configure_inverse_eval(net)

        assert net.freeze_g_FFinh is True
        assert net.plasticity_enabled is False
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in net.named_parameters()
        }
        g_ffinh_before = _bitwise_value(net.g_FFinh)

        response_a = _integrated_spikes(net, "A", intensity)
        response_v = _integrated_spikes(net, "V", intensity)
        response_av = _integrated_spikes(net, "B", intensity)
        denominator = max(response_a, response_v)
        assert denominator > 0.0
        mei_by_intensity[intensity] = (response_av - denominator) / denominator

        parameters_after = dict(net.named_parameters())
        assert parameters_after.keys() == parameters_before.keys()
        for name, parameter_before in parameters_before.items():
            assert torch.equal(parameter_before, parameters_after[name]), (
                f"Parameter changed during frozen evaluation: {name}"
            )
        assert _bitwise_value(net.g_FFinh) == g_ffinh_before

        del net
        torch.cuda.empty_cache()

    low_mei = mei_by_intensity[0.05]
    high_mei = mei_by_intensity[1.6]
    margin = low_mei - high_mei
    print(f"M00 MEI: low={low_mei:.6f}, high={high_mei:.6f}, margin={margin:.6f}")
    assert low_mei > high_mei
    assert margin > 0.75
