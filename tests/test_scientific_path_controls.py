from __future__ import annotations

import unittest

import torch

from Training_delayfix_d52 import MultiBatchAudVisMSINetworkTime


def control_shell() -> MultiBatchAudVisMSINetworkTime:
    net = object.__new__(MultiBatchAudVisMSINetworkTime)
    net.dt = 0.1
    net.tau_gaba = 18.0
    net.gNMDA = 0.51
    net.gaba_shunt_surr = True
    net._tau_gaba_local_override = None
    net._tau_gaba_surround_override = None
    net._gNMDA_exc_override = None
    net._gNMDA_inh_override = None
    return net


class SplitControlTests(unittest.TestCase):
    def test_unset_split_controls_are_live_legacy_aliases(self) -> None:
        net = control_shell()
        net.tau_gaba = 40.0
        net.gNMDA = 0.765
        resolved = net._scientific_path_controls()
        self.assertEqual(resolved["tau_gaba_local_ms"], 40.0)
        self.assertEqual(resolved["tau_gaba_surround_ms"], 40.0)
        self.assertEqual(resolved["gNMDA_exc"], 0.765)
        self.assertEqual(resolved["gNMDA_inh"], 0.765)
        self.assertEqual(resolved["gNMDA_rec"], 0.765)

    def test_local_and_surround_gaba_timing_are_isolated(self) -> None:
        net = control_shell()
        baseline = net._scientific_path_controls()
        net.tau_gaba_local = 10.0
        local = net._scientific_path_controls()
        self.assertNotEqual(local["gaba_decay_local"], baseline["gaba_decay_local"])
        self.assertEqual(local["gaba_decay_surround"], baseline["gaba_decay_surround"])
        net.tau_gaba_local = None
        net.tau_gaba_surround = 40.0
        surround = net._scientific_path_controls()
        self.assertEqual(surround["gaba_decay_local"], baseline["gaba_decay_local"])
        self.assertNotEqual(surround["gaba_decay_surround"], baseline["gaba_decay_surround"])

    def test_excitatory_and_inhibitory_nmda_are_isolated_from_recurrence(self) -> None:
        net = control_shell()
        net.gNMDA_exc = 0.383
        excitatory = net._scientific_path_controls()
        self.assertEqual(excitatory["gNMDA_exc"], 0.383)
        self.assertEqual(excitatory["gNMDA_inh"], 0.51)
        self.assertEqual(excitatory["gNMDA_rec"], 0.51)
        net.gNMDA_exc = None
        net.gNMDA_inh = 0.765
        inhibitory = net._scientific_path_controls()
        self.assertEqual(inhibitory["gNMDA_exc"], 0.51)
        self.assertEqual(inhibitory["gNMDA_inh"], 0.765)
        self.assertEqual(inhibitory["gNMDA_rec"], 0.51)

    def test_invalid_controls_fail_closed(self) -> None:
        net = control_shell()
        with self.assertRaises(ValueError):
            net.tau_gaba_local = 0.1
            net._scientific_path_controls()
        net.tau_gaba_local = None
        with self.assertRaises(ValueError):
            net.gNMDA_exc = float("nan")

    def test_explicit_shipped_values_match_live_alias_network_output(self) -> None:
        torch.manual_seed(44)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(44)
        net = MultiBatchAudVisMSINetworkTime(
            n_neurons=8,
            batch_size=2,
            noise_std=0.0,
            n_substeps=10,
        )
        net.plasticity_enabled = False
        net.afferent_jitter_ms = 0.0
        pulse = torch.full((2, 8), 0.25, device=net.device)
        silence = torch.zeros_like(pulse)
        initial_g_ff = float(net.g_FFinh)

        def trajectory() -> torch.Tensor:
            net.g_FFinh = initial_g_ff
            net.step_counter = 0
            net.reset_state(batch_size=2)
            frames = []
            with torch.inference_mode():
                for frame in range(8):
                    stimulus = pulse if frame < 2 else silence
                    returned = net.update_all_layers_batch(
                        stimulus, stimulus, return_spike_sum=True
                    )
                    frames.append(returned[-1].clone())
            return torch.stack(frames)

        aliases = trajectory()
        net.tau_gaba_local = float(net.tau_gaba)
        net.tau_gaba_surround = float(net.tau_gaba)
        net.gNMDA_exc = float(net.gNMDA)
        net.gNMDA_inh = float(net.gNMDA)
        explicit = trajectory()
        self.assertTrue(torch.equal(aliases, explicit))


if __name__ == "__main__":
    unittest.main()
