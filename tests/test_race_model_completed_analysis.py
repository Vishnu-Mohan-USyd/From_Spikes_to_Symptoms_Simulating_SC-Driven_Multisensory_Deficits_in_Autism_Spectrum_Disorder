"""Regression tests for the portable completed-workflow race-model analyzer."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from mechanism_influence.race_model_completed_analysis import (
    AnalysisValidationError,
    _attach_summary_checksum,
    _require,
    _resolve_recorded_file,
    _validate_candidate_design,
    canonical_sha256,
    exact_family_inference,
    reconstruct_endpoint,
    verify_self_checksum,
)


ADAPTATION_G = [
    [21.29999542236328, 21.599998474121094, 21.70000457763672, 22.0, 21.900001525878906, 21.89999771118164, 21.900009155273438],
    [22.699996948242188, 22.900001525878906, 23.099998474121094, 23.0, 23.0, 22.900001525878906, 22.800003051757812],
    [24.099998474121094, 23.79999542236328, 23.599994659423828, 23.499996185302734, 23.299999237060547, 23.200000762939453, 23.10000228881836],
    [25.399993896484375, 25.5, 25.399993896484375, 25.400001525878906, 25.29999542236328, 25.20000457763672, 25.10000228881836],
    [23.50000762939453, 23.79999542236328, 23.699996948242188, 23.600006103515625, 23.599998474121094, 23.600006103515625, 23.299999237060547],
    [24.199996948242188, 24.29999542236328, 24.300003051757812, 24.5, 24.600006103515625, 24.400001525878906, 24.299999237060547],
    [23.79999542236328, 23.900001525878906, 23.89999771118164, 24.099998474121094, 24.0, 23.900001525878906, 23.89999771118164],
    [23.0, 23.100006103515625, 23.29999542236328, 23.20000457763672, 23.29999542236328, 23.300003051757812, 23.199996948242188],
    [24.800003051757812, 24.800003051757812, 24.699996948242188, 24.699996948242188, 24.70000457763672, 24.599994659423828, 24.50000762939453],
]

GNMDA_G = [
    [-18.000003814697266, -16.70000457763672, -15.999996185302734, -15.499996185302734, -15.200000762939453, -14.900001525878906, -14.599994659423828],
    [-15.500003814697266, -14.300003051757812, -13.700004577636719, -13.300003051757812, -13.200000762939453, -13.000003814697266, -12.599994659423828],
    [-19.200000762939453, -18.60000228881836, -17.60000228881836, -17.299999237060547, -16.800003051757812, -16.300003051757812, -15.800003051757812],
    [-16.900005340576172, -15.700000762939453, -15.200004577636719, -14.700000762939453, -14.300003051757812, -13.900001525878906, -13.700000762939453],
    [-16.599998474121094, -15.60000228881836, -14.900001525878906, -14.5, -14.10000228881836, -13.499996185302734, -13.400005340576172],
    [-18.500003814697266, -17.10000228881836, -16.299999237060547, -15.5, -15.099994659423828, -14.599998474121094, -14.599998474121094],
    [-16.39999771118164, -15.699996948242188, -14.89999771118164, -14.200004577636719, -13.900001525878906, -13.500003814697266, -13.200004577636719],
    [-17.10000228881836, -15.999996185302734, -15.400001525878906, -14.89999771118164, -14.60000228881836, -14.10000228881836, -13.700000762939453],
    [-15.89999771118164, -15.10000228881836, -14.700000762939453, -14.300003051757812, -14.0, -13.800003051757812, -13.39999771118164],
]

ADAPTATION_AREA = [
    21.742499992370618,
    22.4899001235962,
    22.853199802398677,
    24.77460010528566,
    23.000299873352063,
    23.96119991683964,
    23.646300010681152,
    22.785199916839602,
    24.128300178527827,
]

GNMDA_AREA = [
    -13.92800078201294,
    -12.0329006576538,
    -14.922400730133063,
    -13.07700063705444,
    -12.88660094833374,
    -13.806900878906239,
    -12.738900833129886,
    -13.155300907135016,
    -12.77310065078736,
]


def heldout_g_matrix() -> list[list[float]]:
    return [adaptation + [0.0] * 14 + gnmda for adaptation, gnmda in zip(ADAPTATION_G, GNMDA_G)]


def adaptation_g_matrix() -> list[list[float]]:
    return [adaptation + [0.0] * 7 for adaptation in ADAPTATION_G]


def _rng() -> dict[str, object]:
    return {"substreams": [{"index": index} for index in range(10)]}


def _condition_block(
    condition: str,
    *,
    intensity: float,
    n_trials: int = 20,
) -> dict[str, object]:
    if condition == "catch":
        hits = [False] * n_trials
        latencies: list[float | str] = ["+Inf"] * n_trials
    else:
        offset = {"A": 100.0, "V": 110.0, "AV": 80.0}[condition]
        hits = [True] * n_trials
        latencies = [offset + index for index in range(n_trials)]
    return {
        "condition": condition,
        "stimulus_intensity": intensity,
        "n_trials": n_trials,
        "n_hits": sum(hits),
        "hit_rate": sum(hits) / n_trials,
        "latency_ms": latencies,
        "hit": hits,
        "trial_id": [f"s00:t{index:03d}" for index in range(n_trials)],
        "trial_index": {"start": 0, "stop_exclusive": n_trials},
        "censor_ms": 400.0,
        "silence_encoding": "+Inf",
        "rng": _rng(),
    }


def actual_shaped_record() -> dict[str, object]:
    return {
        "measurement": {"horizon_ms": 400.0},
        "rows": [
            {
                "intensity": 0.05,
                "conditions": {
                    "A": _condition_block("A", intensity=0.05),
                    "V": _condition_block("V", intensity=0.05),
                    "AV": _condition_block("AV", intensity=0.05),
                    "catch": _condition_block("catch", intensity=0.0),
                },
            }
        ],
    }


class SavedNumericalRegressionTests(unittest.TestCase):
    def test_heldout_known_deltas_exact_p_values_and_bands(self) -> None:
        primary = exact_family_inference(heldout_g_matrix())
        secondary = exact_family_inference(
            [
                [adaptation, 0.0, 0.0, gnmda]
                for adaptation, gnmda in zip(ADAPTATION_AREA, GNMDA_AREA)
            ]
        )

        self.assertEqual(primary["global_p_numerator"], 2)
        self.assertEqual(primary["global_p_denominator"], 512)
        self.assertAlmostEqual(primary["simultaneous_band"]["critical_value"], 3.064192002399747, places=12)
        self.assertAlmostEqual(primary["means"][0], 23.64444308810764, places=12)
        self.assertAlmostEqual(primary["means"][21], -17.122223748101128, places=12)
        self.assertEqual(primary["adjusted_p_numerators"][0], 2)
        self.assertEqual(primary["adjusted_p_numerators"][7], 512)
        self.assertEqual(primary["adjusted_p_numerators"][21], 2)
        self.assertEqual(secondary["global_p_numerator"], 2)
        self.assertAlmostEqual(secondary["simultaneous_band"]["critical_value"], 2.712561032459204, places=12)
        self.assertAlmostEqual(secondary["means"][0], 23.264611102210164, places=12)
        self.assertAlmostEqual(secondary["means"][3], -13.257900780571832, places=12)

    def test_adaptation_known_deltas_exact_p_values_and_bands(self) -> None:
        primary = exact_family_inference(adaptation_g_matrix())
        secondary = exact_family_inference(
            [[adaptation, 0.0] for adaptation in ADAPTATION_AREA]
        )

        self.assertEqual(primary["global_p_numerator"], 2)
        self.assertAlmostEqual(primary["simultaneous_band"]["critical_value"], 2.583782634269584, places=12)
        self.assertAlmostEqual(primary["means"][0], 23.64444308810764, places=12)
        self.assertEqual(primary["adjusted_p_numerators"][:7], [2] * 7)
        self.assertEqual(primary["adjusted_p_numerators"][7:], [512] * 7)
        self.assertEqual(secondary["global_p_numerator"], 2)
        self.assertAlmostEqual(secondary["simultaneous_band"]["critical_value"], 2.4204998857655537, places=12)
        self.assertAlmostEqual(secondary["means"][0], 23.264611102210164, places=12)
        self.assertEqual(secondary["adjusted_p_numerators"], [2, 512])

    def test_checkpoint_and_endpoint_input_order_invariance(self) -> None:
        matrix = heldout_g_matrix()
        original = exact_family_inference(matrix)
        row_reversed = exact_family_inference(list(reversed(matrix)))
        column_reversed = exact_family_inference([list(reversed(row)) for row in matrix])

        self.assertEqual(original["global_p_numerator"], row_reversed["global_p_numerator"])
        self.assertEqual(original["adjusted_p_numerators"], row_reversed["adjusted_p_numerators"])
        for observed, expected in zip(original["means"], row_reversed["means"]):
            self.assertAlmostEqual(observed, expected, places=12)
        self.assertEqual(
            original["adjusted_p_numerators"],
            list(reversed(column_reversed["adjusted_p_numerators"])),
        )
        self.assertAlmostEqual(
            original["simultaneous_band"]["critical_value"],
            column_reversed["simultaneous_band"]["critical_value"],
            places=12,
        )

    def test_payload_checksum_is_deterministic(self) -> None:
        result = exact_family_inference(adaptation_g_matrix())
        first = _attach_summary_checksum({"result": result})
        second = _attach_summary_checksum(copy.deepcopy({"result": result}))
        self.assertEqual(first, second)
        self.assertEqual(first["summary_payload_sha256"], canonical_sha256({"result": result}))


class ValidationAndRelocationTests(unittest.TestCase):
    def test_actual_catch_zero_drive_is_required(self) -> None:
        record = actual_shaped_record()
        endpoint = reconstruct_endpoint(record)
        self.assertTrue(endpoint["qc_pass"])

        forged = copy.deepcopy(record)
        forged["rows"][0]["conditions"]["catch"]["stimulus_intensity"] = 0.05
        with self.assertRaisesRegex(AnalysisValidationError, "condition intensity mismatch"):
            reconstruct_endpoint(forged)

    def test_qc_failure_closes_confirmatory_use(self) -> None:
        record = actual_shaped_record()
        block = record["rows"][0]["conditions"]["A"]
        block["hit"][-2:] = [False, False]
        block["latency_ms"][-2:] = ["+Inf", "+Inf"]
        block["n_hits"] = 18
        block["hit_rate"] = 0.9
        endpoint = reconstruct_endpoint(record)
        self.assertFalse(endpoint["qc_pass"])
        self.assertEqual(endpoint["qc_classification"], "joint_detection_latency")
        with self.assertRaisesRegex(AnalysisValidationError, "confirmatory inference is closed"):
            _require(endpoint["qc_pass"], "confirmatory inference is closed")

    def test_recorded_absolute_path_relocates_by_pinned_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relocated = root / "candidate_seed43.json"
            relocated.write_text("{}\n", encoding="utf-8")
            resolved = _resolve_recorded_file(
                root,
                "/obsolete/acquisition/root/raw/candidate_seed43.json",
            )
            self.assertEqual(resolved, relocated.resolve())

    def test_payload_forgery_fails_closed(self) -> None:
        record = {"value": 7, "integrity": {}}
        record["integrity"]["raw_payload_sha256"] = canonical_sha256(record)
        self.assertEqual(
            verify_self_checksum(record, "integrity", "raw_payload_sha256"),
            record["integrity"]["raw_payload_sha256"],
        )
        forged = copy.deepcopy(record)
        forged["value"] = 8
        with self.assertRaisesRegex(AnalysisValidationError, "payload checksum mismatch"):
            verify_self_checksum(forged, "integrity", "raw_payload_sha256")

    def test_seed42_is_excluded_from_exact_candidate_design(self) -> None:
        cells = ("cell_a", "cell_b")
        entries = [
            {"cell_id": cell, "seed": seed}
            for cell in cells
            for seed in range(43, 52)
        ]
        self.assertEqual(len(_validate_candidate_design(entries, cells)), 18)
        entries.append({"cell_id": "cell_a", "seed": 42})
        with self.assertRaisesRegex(AnalysisValidationError, "design changed"):
            _validate_candidate_design(entries, cells)


if __name__ == "__main__":
    unittest.main()
