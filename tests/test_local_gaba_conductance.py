from __future__ import annotations

import os
import unittest
from contextlib import contextmanager

import torch

import Training_delayfix_d52 as eager
import Training_graphdf_d52 as training


@contextmanager
def local_gaba_mode(mode: str):
    previous = os.environ.get("LOCAL_GABA_MODE")
    os.environ["LOCAL_GABA_MODE"] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LOCAL_GABA_MODE", None)
        else:
            os.environ["LOCAL_GABA_MODE"] = previous


def configuration_shell(network_type):
    return object.__new__(network_type)


class LocalGabaEquationTests(unittest.TestCase):
    def test_conductance_zero_at_reversal_and_signs_around_reversal(self) -> None:
        state = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        voltage = torch.tensor([[-70.0, -60.0, -80.0]], dtype=torch.float64)

        for module in (training, eager):
            current = module.compute_local_gaba_current(
                state,
                voltage,
                mode=module.LOCAL_GABA_MODE_CONDUCTANCE,
            )
            self.assertEqual(float(current[0, 0]), 0.0)
            self.assertGreater(float(current[0, 1]), 0.0)
            self.assertLess(float(current[0, 2]), 0.0)

    def test_training_and_eager_conductance_equations_are_exactly_equal(self) -> None:
        state = torch.tensor([[0.0, 0.25, 1.5], [2.0, 4.0, 8.0]])
        voltage = torch.tensor([[-70.0, -69.5, -55.0], [-80.0, -65.0, 25.0]])
        train_current = training.compute_local_gaba_current(
            state, voltage, mode=training.LOCAL_GABA_MODE_CONDUCTANCE
        )
        eager_current = eager.compute_local_gaba_current(
            state, voltage, mode=eager.LOCAL_GABA_MODE_CONDUCTANCE
        )
        self.assertTrue(torch.equal(train_current, eager_current))

    def test_legacy_current_mode_is_the_exact_original_state(self) -> None:
        state = torch.tensor([[1.0, 3.0]])
        voltage = torch.tensor([[-90.0, 30.0]])
        for module in (training, eager):
            current = module.compute_local_gaba_current(
                state, voltage, mode=module.LOCAL_GABA_MODE_LEGACY
            )
            self.assertIs(current, state)


class LocalGabaCheckpointTests(unittest.TestCase):
    def test_legacy_checkpoint_selects_current_mode_and_partial_triplet_fails(self) -> None:
        for module in (training, eager):
            net = configuration_shell(module.MultiBatchAudVisMSINetworkTime)
            restored = net.restore_local_gaba_hparams({})
            self.assertFalse(restored)
            self.assertEqual(net.local_gaba_mode, module.LOCAL_GABA_MODE_LEGACY)
            with self.assertRaises(ValueError):
                net.restore_local_gaba_hparams({"local_gaba_mode": "conductance"})

    def test_new_checkpoint_triplet_round_trips_in_both_builds(self) -> None:
        for module in (training, eager):
            source = configuration_shell(module.MultiBatchAudVisMSINetworkTime)
            source.set_local_gaba_configuration(module.LOCAL_GABA_MODE_CONDUCTANCE)
            saved = {
                key: getattr(source, key)
                for key in module.LOCAL_GABA_HPARAM_KEYS
            }
            target = configuration_shell(module.MultiBatchAudVisMSINetworkTime)
            self.assertTrue(target.restore_local_gaba_hparams(saved))
            self.assertEqual(target.local_gaba_mode, module.LOCAL_GABA_MODE_CONDUCTANCE)
            self.assertEqual(
                target.local_gaba_conductance_per_mv,
                module.LOCAL_GABA_CONDUCTANCE_PER_MV,
            )
            self.assertEqual(target.local_gaba_reversal_mv, module.LOCAL_GABA_REVERSAL_MV)

    def test_make_checkpoint_records_complete_triplet(self) -> None:
        for module in (training, eager):
            with local_gaba_mode(module.LOCAL_GABA_MODE_CONDUCTANCE):
                torch.manual_seed(9)
                net = module.MultiBatchAudVisMSINetworkTime(
                    n_neurons=6,
                    batch_size=1,
                    noise_std=0.0,
                    n_substeps=1,
                )
            checkpoint = module.make_checkpoint(net, epoch=0, rng_tag=False)
            mutable = checkpoint["mutable_hparams"]
            self.assertEqual(
                tuple(key for key in module.LOCAL_GABA_HPARAM_KEYS if key in mutable),
                module.LOCAL_GABA_HPARAM_KEYS,
            )
            self.assertEqual(mutable["local_gaba_mode"], "conductance")
            self.assertEqual(
                mutable["local_gaba_conductance_per_mv"],
                module.LOCAL_GABA_CONDUCTANCE_PER_MV,
            )
            self.assertEqual(
                mutable["local_gaba_reversal_mv"], module.LOCAL_GABA_REVERSAL_MV
            )


class LocalGabaForwardTests(unittest.TestCase):
    def test_conductance_forward_is_finite_in_training_and_eager_builds(self) -> None:
        for module in (training, eager):
            with local_gaba_mode(module.LOCAL_GABA_MODE_CONDUCTANCE):
                torch.manual_seed(17)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(17)
                net = module.MultiBatchAudVisMSINetworkTime(
                    n_neurons=6,
                    batch_size=2,
                    noise_std=0.0,
                    n_substeps=2,
                )
            net.plasticity_enabled = False
            net.reset_state(batch_size=2)
            stimulus = torch.full((2, 6), 0.2, device=net.device)
            with torch.no_grad():
                outputs = net.update_all_layers_batch(
                    stimulus, stimulus, return_spike_sum=True
                )
            for tensor in (*outputs, net.v_msi, net.I_M_gaba):
                self.assertTrue(bool(torch.isfinite(tensor).all().item()))


if __name__ == "__main__":
    unittest.main()
