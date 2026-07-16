"""CPU-only tests for frozen Gate 4/Gate 5 perturbation infrastructure."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from mechanism_influence import response_gain_analysis as analysis
from mechanism_influence import response_gain_perturbation as measure


class FakeNet:
    def __init__(self) -> None:
        for name, value in measure.BASELINE.items():
            setattr(self, name, value)
        self.tau_nmda = 999.0
        self.g_rec = 999.0


def gate_record(*, response: float = 10.0, effective: dict[str, float] | None = None) -> dict:
    vector = dict(measure.BASELINE if effective is None else effective)
    return {
        "effective": vector,
        "effective_after": dict(vector),
        "state_bit_identical": True,
        "R_A": [response, response],
        "R_V": [response, response],
        "R_AV": [response * 2, response * 2],
        "mei": [1.0, 1.0],
        "cre_percent_ratio_of_means": [100.0, 100.0],
    }


class RegistryTests(unittest.TestCase):
    def test_primary_registry_has_thirteen_unique_vectors(self) -> None:
        measure.validate_registry()
        vectors = {
            tuple(cell["values"].values()) for cell in measure.PRIMARY_REGISTRY.values()
        }
        self.assertEqual(len(measure.PRIMARY_REGISTRY), 13)
        self.assertEqual(len(vectors), 13)
        self.assertEqual(measure.PRIMARY_REGISTRY["shipped"]["values"], measure.BASELINE)

    def test_triggered_scalar_cells_are_not_primary(self) -> None:
        self.assertTrue(set(measure.TRIGGERED_REGISTRY).isdisjoint(measure.PRIMARY_REGISTRY))
        self.assertEqual(measure.TRIGGERED_REGISTRY["aM_0p008_dM_10"]["values"]["dM"], 10.0)
        self.assertEqual(measure.TRIGGERED_REGISTRY["aM_0p02_dM_8"]["values"]["aM"], 0.02)

    def test_confirmatory_runner_rejects_nonserial_gate5_chunks(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk_size=1"):
            measure.run_cell(42, "shipped", Path("unused.json"), gate5_chunk_size=2)

    def test_native_thread_pools_are_strictly_capped(self) -> None:
        for variable, expected in measure.THREAD_CAPS.items():
            self.assertEqual(os.environ[variable], expected)


class OverrideTests(unittest.TestCase):
    def test_gate5_restores_then_applies_nmda_override(self) -> None:
        net = FakeNet()
        effective = measure.restore_gate5_then_apply(
            net, {"gNMDA": 0.51, "tau_nmda": 40.0}, "gNMDA_0p765"
        )
        self.assertEqual(net.gNMDA, 0.765)
        self.assertEqual(net.tau_nmda, 40.0)
        self.assertEqual(net.g_rec, 0.03)
        self.assertEqual(effective["gNMDA"], 0.765)

    def test_override_rejects_nonshipped_loaded_state(self) -> None:
        net = FakeNet()
        net.tau_gaba = 10.0
        with self.assertRaisesRegex(AssertionError, "loaded net.tau_gaba"):
            measure.apply_registered_setting(net, "tau_gaba_40")


class QCTests(unittest.TestCase):
    def paired(self, pert_response: float, base_response: float = 10.0) -> dict:
        pert = {"gate4": gate_record(response=pert_response), "gate5": gate_record(response=pert_response)}
        base = {"gate4": gate_record(response=base_response), "gate5": gate_record(response=base_response)}
        return analysis._paired_qc(pert, base, "gate4")

    def test_qc_accepts_viable_absolute_responses(self) -> None:
        result = self.paired(2.0)
        self.assertTrue(result["eligible_for_normalized_inference"])
        self.assertEqual(result["exclusion_reasons"], [])

    def test_qc_rejects_response_collapse(self) -> None:
        result = self.paired(0.9)
        self.assertFalse(result["eligible_for_normalized_inference"])
        self.assertIn("response_below_10_percent_of_paired_shipped", result["exclusion_reasons"])

    def test_qc_rejects_zero_denominator(self) -> None:
        result = self.paired(0.0)
        self.assertFalse(result["eligible_for_normalized_inference"])
        self.assertIn("nonpositive_perturbation_response", result["exclusion_reasons"])


class ExactInferenceTests(unittest.TestCase):
    def test_constant_effect_has_exact_two_over_eight_p_and_zero_width_band(self) -> None:
        result = analysis.exact_row_sign_flip_max_t(np.ones((3, 1)), ["effect"])
        endpoint = result["endpoints"][0]
        self.assertEqual(result["n_sign_patterns"], 8)
        self.assertEqual(endpoint["adjusted_p_numerator"], 2)
        self.assertEqual(endpoint["adjusted_p_denominator"], 8)
        self.assertEqual(endpoint["simultaneous_lower"], 1.0)
        self.assertEqual(endpoint["simultaneous_upper"], 1.0)

    def test_zero_effect_is_not_significant(self) -> None:
        result = analysis.exact_row_sign_flip_max_t(np.zeros((4, 2)), ["a", "b"])
        self.assertTrue(all(row["adjusted_p"] == 1.0 for row in result["endpoints"]))

    def test_analysis_is_byte_deterministic(self) -> None:
        values = np.asarray([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]])
        first = analysis.exact_row_sign_flip_max_t(values, ["positive", "negative"])
        second = analysis.exact_row_sign_flip_max_t(values, ["positive", "negative"])
        self.assertEqual(measure.canonical_sha256(first), measure.canonical_sha256(second))


class PayloadAndProvenanceTests(unittest.TestCase):
    def fixture(self, checkpoint: Path) -> dict:
        gate4 = gate_record()
        gate5 = gate_record()
        record = {
            "schema_version": measure.SCHEMA_VERSION,
            "protocol_id": measure.PROTOCOL_ID,
            "checkpoint": {
                "seed": 42,
                "sha256": measure.sha256_file(checkpoint),
            },
            "setting": {
                "id": "shipped",
                "requested": dict(measure.BASELINE),
            },
            "device": {
                "physical_uuid": measure.RTX5090_UUID,
                "physical_name": "NVIDIA GeForce RTX 5090",
            },
            "integrity": {
                "readouts_unchanged": True,
                "plasticity_enabled": False,
            },
            "gate4": gate4,
            "gate5": gate5,
        }
        record["integrity"]["scientific_payload_sha256"] = measure.canonical_sha256(record)
        return record

    def test_payload_and_gpu_provenance_validate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint fixture")
            payload_path = root / "cell.json"
            payload_path.write_text(json.dumps(self.fixture(checkpoint)), encoding="utf-8")
            with mock.patch.object(measure, "checkpoint_path", return_value=checkpoint):
                loaded = analysis._validate_record(payload_path)
            self.assertEqual(loaded["device"]["physical_uuid"], measure.RTX5090_UUID)

    def test_payload_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint fixture")
            record = self.fixture(checkpoint)
            record["gate4"]["R_AV"][0] += 1.0
            payload_path = root / "cell.json"
            payload_path.write_text(json.dumps(record), encoding="utf-8")
            with mock.patch.object(measure, "checkpoint_path", return_value=checkpoint):
                with self.assertRaisesRegex(ValueError, "payload SHA-256 mismatch"):
                    analysis._validate_record(payload_path)


if __name__ == "__main__":
    unittest.main()
