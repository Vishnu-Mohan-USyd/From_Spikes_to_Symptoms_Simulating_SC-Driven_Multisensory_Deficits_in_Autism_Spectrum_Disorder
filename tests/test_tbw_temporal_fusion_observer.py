from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from mechanism_influence.tbw_temporal_fusion_observer import (
    DEFAULT_CALIBRATION,
    classify_temporal_profile,
    classify_temporal_profile_legacy_regression,
    classify_temporal_rasters,
    half_peak_width,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tbw_temporal_fusion_heldout.json"


def heldout_cases() -> dict[str, dict[str, object]]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {case["name"]: case for case in payload["cases"]}


class HeldoutAnchorTests(unittest.TestCase):
    def test_heldout_profiles_reproduce_all_observer_outcomes(self) -> None:
        for name, case in heldout_cases().items():
            with self.subTest(name=name):
                decision = classify_temporal_profile(
                    np.asarray(case["profile"], dtype=float),
                    int(case["effective_offset_frames"]),
                )
                self.assertEqual(decision.outcome, case["expected_outcome"])

    def test_suppressed_single_response_is_not_fusion(self) -> None:
        case = heldout_cases()["pv4_suppression_not_fusion"]
        profile = np.asarray(case["profile"], dtype=float)
        offset = int(case["effective_offset_frames"])
        corrected = classify_temporal_profile(profile, offset)
        legacy = classify_temporal_profile_legacy_regression(profile, offset)
        self.assertEqual(corrected.outcome, "dropout_ambiguous")
        self.assertFalse(corrected.fused)
        self.assertTrue(legacy.fused)


class FrozenCalibrationTests(unittest.TestCase):
    def test_frozen_sham_threshold_is_absolute_across_response_scale(self) -> None:
        case = heldout_cases()["shipped_fused_one"]
        profile = np.asarray(case["profile"], dtype=float)
        offset = int(case["effective_offset_frames"])
        strong = classify_temporal_profile(profile, offset)
        suppressed = classify_temporal_profile(0.5 * profile, offset)
        self.assertEqual(strong.outcome, "fused_one")
        self.assertEqual(suppressed.outcome, "dropout_ambiguous")
        self.assertEqual(
            DEFAULT_CALIBRATION.one_peak_amplitude_floor_spikes,
            98.83377430574426,
        )

    def test_no_per_condition_max_normalization(self) -> None:
        case = heldout_cases()["shipped_fused_one"]
        profile = np.asarray(case["profile"], dtype=float)
        offset = int(case["effective_offset_frames"])
        corrected_full = classify_temporal_profile(profile, offset)
        corrected_half = classify_temporal_profile(0.5 * profile, offset)
        legacy_full = classify_temporal_profile_legacy_regression(profile, offset)
        legacy_half = classify_temporal_profile_legacy_regression(0.5 * profile, offset)
        self.assertTrue(corrected_full.fused)
        self.assertFalse(corrected_half.fused)
        self.assertTrue(legacy_full.fused)
        self.assertTrue(legacy_half.fused)


class CurveAuditTests(unittest.TestCase):
    def test_every_trial_remains_in_probability_denominator(self) -> None:
        cases = heldout_cases()
        fused = np.asarray(cases["shipped_fused_one"]["profile"], dtype=float)
        dropout = np.asarray(cases["pv4_suppression_not_fusion"]["profile"], dtype=float)
        rasters = np.stack((fused, dropout), axis=1)[:, None, :]
        effective = np.asarray([
            [
                cases["shipped_fused_one"]["effective_offset_frames"],
                cases["pv4_suppression_not_fusion"]["effective_offset_frames"],
            ]
        ])
        result = classify_temporal_rasters(rasters, np.asarray([0.0]), effective)
        self.assertEqual(result["fused_counts"], [1])
        self.assertEqual(result["dropout_counts"], [1])
        self.assertEqual(result["separate_counts"], [0])
        self.assertEqual(result["n_trials"], [2])
        self.assertEqual(result["p_fusion"], [0.5])

    def test_sampled_half_peak_estimator_is_unchanged(self) -> None:
        width, left, right = half_peak_width(
            np.asarray([-40.0, -20.0, 0.0, 20.0, 40.0]),
            np.asarray([0.1, 0.5, 1.0, 0.5, 0.1]),
        )
        self.assertEqual((width, left, right), (40.0, -20.0, 20.0))

    def test_corrected_observer_is_the_default(self) -> None:
        case = heldout_cases()["pv4_suppression_not_fusion"]
        profile = np.asarray(case["profile"], dtype=float)
        rasters = profile[:, None, None]
        effective = np.asarray([[case["effective_offset_frames"]]])
        result = classify_temporal_rasters(rasters, np.asarray([0.0]), effective)
        self.assertEqual(result["observer"], "corrected")
        self.assertEqual(result["p_fusion"], [0.0])


if __name__ == "__main__":
    unittest.main()
