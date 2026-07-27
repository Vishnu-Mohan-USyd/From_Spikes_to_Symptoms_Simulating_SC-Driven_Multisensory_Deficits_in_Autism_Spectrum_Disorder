"""Regression for presynaptic A/V short-term depression into MSI inhibition."""

import torch
import torch.nn.functional as F

from Training import MultiBatchAudVisMSINetworkTime


def test_inhibitory_input_stp_releases_before_projection(monkeypatch) -> None:
    """High synchrony consumes each presynaptic terminal once and stays finite.

    The production update receives binary delayed spikes with shape ``(B, n)``.
    Dimensionless resource/utilization state must have that same presynaptic
    shape; released resources are projected by ``(n_inh, n)`` AMPA weights into
    a downstream current with shape ``(B, n_inh)``.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    batch_size, n_neurons = 3, 10
    net = MultiBatchAudVisMSINetworkTime(
        n_neurons=n_neurons,
        batch_size=batch_size,
        n_substeps=1,
        noise_std=0.0,
    )

    expected_pre_shape = (batch_size, n_neurons)
    for state_name in ("R_a_inh", "u_a_inh", "R_v_inh", "u_v_inh"):
        assert tuple(getattr(net, state_name).shape) == expected_pre_shape

    net.reset_state(batch_size=batch_size)
    for state_name in ("R_a_inh", "u_a_inh", "R_v_inh", "u_v_inh"):
        assert tuple(getattr(net, state_name).shape) == expected_pre_shape

    utilization_a = torch.linspace(0.05, 0.50, n_neurons).repeat(batch_size, 1)
    utilization_v = torch.linspace(0.50, 0.05, n_neurons).repeat(batch_size, 1)
    synchronous_spikes = torch.ones(expected_pre_shape)

    weights_a = torch.arange(
        1, net.n_inh * n_neurons + 1, dtype=torch.float32
    ).reshape(net.n_inh, n_neurons) / 100.0
    weights_v = torch.flip(weights_a, dims=(1,)) / 2.0

    with torch.no_grad():
        net.u_a_inh.copy_(utilization_a)
        net.u_v_inh.copy_(utilization_v)
        net.R_a_inh.fill_(1.0)
        net.R_v_inh.fill_(1.0)
        net.W_a2msiInh_AMPA.copy_(weights_a)
        net.W_v2msiInh_AMPA.copy_(weights_v)
        net.W_a2msiInh_NMDA.zero_()
        net.W_v2msiInh_NMDA.zero_()

        pos_a = net._delay_positions["buffer_a2msi_inh"]
        pos_v = net._delay_positions["buffer_v2msi_inh"]
        net.buffer_a2msi_inh[pos_a].copy_(synchronous_spikes)
        net.buffer_v2msi_inh[pos_v].copy_(synchronous_spikes)

    expected_current = (
        F.linear(utilization_a * synchronous_spikes, weights_a)
        + F.linear(utilization_v * synchronous_spikes, weights_v)
    )

    net.gNMDA = 0.0
    net.plasticity_enabled = False
    net.allow_inhib_plasticity = False
    net.freeze_g_FFinh = True
    zeros = torch.zeros(expected_pre_shape)
    outputs = net.update_all_layers_batch(zeros, zeros)

    assert tuple(net.I_M_inh.shape) == (batch_size, net.n_inh)
    torch.testing.assert_close(net.I_M_inh, expected_current, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(net.R_a_inh, 1.0 - utilization_a)
    torch.testing.assert_close(net.R_v_inh, 1.0 - utilization_v)

    for resource_name in ("R_a", "R_v", "R_a_inh", "R_v_inh"):
        resource = getattr(net, resource_name)
        assert torch.isfinite(resource).all()
        assert torch.all((resource >= 0.0) & (resource <= 1.0))

    assert len(outputs) == 4
    assert all(tuple(output.shape) == expected_pre_shape for output in outputs)
