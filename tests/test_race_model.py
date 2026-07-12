"""Deterministic CPU falsifications for formal Miller race-model analysis."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from mechanism_influence import race_model_analysis as analysis
from mechanism_influence import race_model_measure as measure


def condition_block(values: list[float], horizon_ms: float) -> dict:
    encoded: list[float | str] = []
    hit: list[bool] = []
    for value in values:
        finite = math.isfinite(value)
        hit.append(finite)
        encoded.append(float(value) if finite else "+Inf")
    return {
        "n_trials": len(values),
        "n_hits": sum(hit),
        "latency_ms": encoded,
        "hit": hit,
        "censor_ms": float(horizon_ms),
    }


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_analyzer_release(
    *,
    clean: bool = True,
    tracked: bool = True,
    local_matches_head: bool = True,
    head: str = "a" * 40,
) -> dict:
    sha256 = _digest("released-analyzer")
    return {
        "path": str(Path(analysis.__file__).resolve()),
        "sha256": sha256,
        "tracked": tracked,
        "head_blob_sha256": sha256 if local_matches_head else _digest("other-blob"),
        "local_matches_head": local_matches_head,
        "git": {"head": head, "branch": "release", "clean": clean},
    }


def rehash_raw(record: dict) -> dict:
    record["integrity"].pop("raw_payload_sha256", None)
    record["integrity"]["raw_payload_sha256"] = analysis.raw_payload_sha256(record)
    return record


def acquisition_record(
    *,
    a: list[float],
    v: list[float],
    av: list[float],
    catch: list[float] | None = None,
    checkpoint_seed: int = 1,
    intensities: tuple[float, ...] = (0.2,),
    horizon_ms: float = 60.0,
    requested: dict[str, float] | None = None,
    design_family: str = analysis.SCALAR_RMI_FAMILY,
    git_clean: bool = False,
) -> dict:
    """Build one checksum-valid raw fixture; design validity is independently gated."""

    settings = dict(analysis.BASELINE_MECHANISMS if requested is None else requested)
    cell = analysis.registered_mechanism_cell(settings, design_family)
    changed = list(cell["changed_scalars"])
    target = changed[0] if len(changed) == 1 else None
    label = str(cell["cell_id"])
    catch_values = [math.inf] * len(a) if catch is None else catch
    endpoint = {
        "response_rule_id": "first_msi_population_spike_v1",
        "response_rule": "first substep with MSI population spike count > 0",
        "threshold": "MSI population spike count > 0",
        "latency_formula": "(_first_spike_substep + 1) * dt_ms",
        "latency_domain_ms": f"0 < latency <= {int(horizon_ms)}",
        "resolution_ms": 0.1,
        "interpretation": "model neural first-spike latency surrogate; not human reaction time",
        "sensitivity_horizons_ms": [300.0, 350.0, 400.0],
    }
    measurement = {
        "intensities": list(intensities),
        "condition_order": list(measure.CONDITION_ORDER),
        "n_trials_per_condition": len(a),
        "run_seed": 20260712,
        "substream_count": analysis.PRIMARY_SUBSTREAMS,
        "trials_per_substream": analysis.TRIALS_PER_SUBSTREAM,
        "stream_seed_derivation": analysis.STREAM_SEED_DERIVATION,
        "dt_ms": 0.1,
        "horizon_ms": float(horizon_ms),
        "soa_ms": 0.0,
        "endpoint": endpoint,
    }
    configuration = {
        "baseline": dict(analysis.BASELINE_MECHANISMS),
        "requested": settings,
        "effective": settings,
        "environment": {"fixture": "dm10"},
        "intensities": list(intensities),
        "n_trials": len(a),
        "soa_ms": 0.0,
        "horizon_ms": float(horizon_ms),
    }
    state_hash = _digest(f"state-{checkpoint_seed}")
    readout_hashes = [hashlib.md5(b"tbw").hexdigest(), hashlib.md5(b"sbw").hexdigest()]
    record = {
        "schema_version": analysis.ACQUISITION_SCHEMA,
        "condition": {
            "label": label,
            "changed_scalar": target,
            "changed_scalars": changed,
            "design_family": design_family,
            "cell_id": cell["cell_id"],
            "factor_id": cell["factor_id"],
            "baseline": dict(analysis.BASELINE_MECHANISMS),
            "requested": settings,
            "one_factor": cell["scalar_one_factor"],
            "mechanism_manifest": {
                "version": "dm10-registered-mechanisms-v2",
                **cell,
            },
        },
        "checkpoint": {
            "seed": checkpoint_seed,
            "sha256": _digest(f"checkpoint-{checkpoint_seed}"),
            "load_missing_keys": [],
            "load_unexpected_keys": [],
        },
        "measurement": measurement,
        "loaded_effective": dict(analysis.BASELINE_MECHANISMS),
        "effective": settings,
        "effective_after": settings,
        "integrity": {
            "state_sha256_loaded": state_hash,
            "state_sha256_before": state_hash,
            "state_sha256_after": state_hash,
            "state_bit_identical": True,
            "readout_md5_before": readout_hashes,
            "readout_md5_after": readout_hashes,
            "readouts_unchanged": True,
            "plasticity_enabled": False,
        },
        "code": {
            "driver_sha256": _digest("driver"),
            "source_sha256": {
                "driver": _digest("driver"),
                "network_io": _digest("network"),
                "latency_module": _digest("latency"),
            },
            "git": {"head": "a" * 40, "branch": "fixture", "clean": git_clean},
        },
        "device": {"type": "cpu", "name": "fixture CPU"},
        "configuration": configuration,
        "configuration_sha256": analysis._canonical_sha256(configuration),
        "rows": [
            {
                "intensity": intensity,
                "conditions": {
                    "A": condition_block(a, horizon_ms),
                    "V": condition_block(v, horizon_ms),
                    "AV": condition_block(av, horizon_ms),
                    "catch": condition_block(catch_values, horizon_ms),
                },
            }
            for intensity in intensities
        ],
    }
    if (
        len(a) == analysis.PRIMARY_N_TRIALS
        and intensities == analysis.PRIMARY_INTENSITIES
        and horizon_ms == analysis.PRIMARY_HORIZON_MS
    ):
        trial_ids = [
            f"s{substream:02d}:t{trial:03d}"
            for substream in range(analysis.PRIMARY_SUBSTREAMS)
            for trial in range(analysis.TRIALS_PER_SUBSTREAM)
        ]
        for row in record["rows"]:
            for condition in measure.CONDITION_ORDER:
                substreams = []
                for substream_index in range(analysis.PRIMARY_SUBSTREAMS):
                    substreams.append(
                        {
                            "index": substream_index,
                            "stream_seed": measure.derive_stream_seed(
                                measurement["run_seed"],
                                checkpoint_seed,
                                row["intensity"],
                                condition,
                                substream_index,
                            ),
                            "trial_start": substream_index * analysis.TRIALS_PER_SUBSTREAM,
                            "trial_stop_exclusive": (substream_index + 1)
                            * analysis.TRIALS_PER_SUBSTREAM,
                            "state_before_sha256": _digest(
                                f"before-{checkpoint_seed}-{row['intensity']}-{condition}-{substream_index}"
                            ),
                            "state_after_sha256": _digest(
                                f"after-{checkpoint_seed}-{row['intensity']}-{condition}-{substream_index}"
                            ),
                        }
                    )
                block = row["conditions"][condition]
                block["trial_id"] = trial_ids
                block["rng"] = {
                    "algorithm": analysis.RNG_ALGORITHM,
                    "stable_cell_key": {
                        "checkpoint_seed": checkpoint_seed,
                        "intensity": row["intensity"],
                        "sensory_condition": condition,
                    },
                    "substreams": substreams,
                }
    return rehash_raw(record)


def strong_design_records() -> list[dict]:
    return [
        acquisition_record(
            a=[20.0] * analysis.PRIMARY_N_TRIALS,
            v=[20.0] * analysis.PRIMARY_N_TRIALS,
            av=[5.0] * analysis.PRIMARY_N_TRIALS,
            checkpoint_seed=seed,
            intensities=analysis.PRIMARY_INTENSITIES,
            horizon_ms=analysis.PRIMARY_HORIZON_MS,
            git_clean=True,
        )
        for seed in analysis.PRIMARY_SEEDS
    ]


def historical_design_records(
    labels: tuple[str, ...] = ("adaptation_off", "adaptation_prior", "adaptation_shipped"),
    seeds: tuple[int, ...] = analysis.PRIMARY_SEEDS,
) -> list[dict]:
    registry = analysis.MECHANISM_CELL_REGISTRY[analysis.HISTORICAL_ADAPTATION_FAMILY]
    return [
        acquisition_record(
            a=[20.0] * analysis.PRIMARY_N_TRIALS,
            v=[20.0] * analysis.PRIMARY_N_TRIALS,
            av=[5.0] * analysis.PRIMARY_N_TRIALS,
            checkpoint_seed=seed,
            intensities=analysis.PRIMARY_INTENSITIES,
            horizon_ms=analysis.PRIMARY_HORIZON_MS,
            requested=registry[label]["requested"],
            design_family=analysis.HISTORICAL_ADAPTATION_FAMILY,
            git_clean=True,
        )
        for label in labels
        for seed in seeds
    ]


class QuantileGainAndAreaTests(unittest.TestCase):
    def test_tied_early_shift_uses_G_not_D_at_bound_and_passes_full_gate(self) -> None:
        with mock.patch.object(
            analysis, "_analyzer_provenance", return_value=clean_analyzer_release()
        ):
            result = analysis.analyze_family(strong_design_records())
        per_seed = result["individual_analyses"][0]["intensities"][0]
        q_entry = per_seed["quantile_summaries"][0]
        self.assertEqual(q_entry["Q_bound_ms"], 20.0)
        self.assertEqual(q_entry["Q_AV_ms"], 5.0)
        self.assertEqual(q_entry["G_ms"], 15.0)
        self.assertEqual(q_entry["D_at_Q_bound"], 0.0)
        self.assertEqual(per_seed["descriptive_shape"]["max_D"], 1.0)
        self.assertEqual(per_seed["descriptive_windows"]["full"]["positive_area_ms"], 15.0)

        inference = result["condition_violation_inference"]
        self.assertEqual(inference["n_sign_patterns"], 1024)
        self.assertEqual(inference["global_p_one_sided_max_stat"], 1.0 / 1024.0)
        self.assertTrue(all(cell["approx_simultaneous_lower_G_ms"] > 0 for cell in inference["cells"]))
        self.assertTrue(result["design"]["pass"])
        self.assertTrue(result["formal_rmi_violation"])
        self.assertEqual(result["nested_trial_checkpoint_bootstrap"]["status"], "not_implemented")
        self.assertEqual(len(result["input_provenance"]), 10)
        self.assertEqual(len(result["analyzer_provenance"]["sha256"]), 64)

    def test_positive_rse_mean_benefit_can_have_zero_formal_G(self) -> None:
        result = analysis.analyze_acquisition(
            acquisition_record(a=[10.0, 20.0], v=[10.0, 20.0], av=[10.0, 10.0])
        )["intensities"][0]
        self.assertGreater(result["rse_mean_benefit"]["benefit_ms"], 0.0)
        self.assertEqual(
            [entry["G_ms"] for entry in result["quantile_summaries"]],
            [0.0] * len(analysis.Q_LEVELS),
        )

    def test_exact_step_area_and_normalization(self) -> None:
        support = np.asarray([0.0, 10.0, 20.0, 30.0])
        difference = np.asarray([0.2, -0.1, 0.0, 0.0])
        positive = analysis.integrate_step(
            support, difference, 0.0, 30.0, positive_part=True
        )
        signed = analysis.integrate_step(
            support, difference, 0.0, 30.0, positive_part=False
        )
        self.assertAlmostEqual(positive, 2.0)
        self.assertAlmostEqual(signed, 1.0)

        row = analysis.analyze_acquisition(
            acquisition_record(a=[20.0] * 4, v=[20.0] * 4, av=[5.0] * 4)
        )["intensities"][0]
        full = row["descriptive_windows"]["full"]
        self.assertEqual(full["window_width_ms"], 60.0)
        self.assertAlmostEqual(full["positive_area_normalized"], full["positive_area_ms"] / 60.0)
        self.assertFalse(full["inferential_role"])

    def test_ties_use_exact_quantiles_without_interpolation(self) -> None:
        result = analysis.analyze_acquisition(
            acquisition_record(a=[10.0] * 4, v=[10.0] * 4, av=[8.0] * 4)
        )["intensities"][0]
        self.assertEqual(
            [entry["Q_bound_ms"] for entry in result["quantile_summaries"]],
            [10.0] * len(analysis.Q_LEVELS),
        )
        self.assertEqual(
            analysis.exact_step_quantile(
                np.asarray([0.0, 10.0, 20.0]), np.asarray([0.0, 0.5, 1.0]), 0.35
            ),
            10.0,
        )


class HistoricalFamilyDesignTests(unittest.TestCase):
    def test_shipped_only_is_cell_eligible_but_replication_incomplete(self) -> None:
        records = historical_design_records(labels=("adaptation_shipped",))
        with mock.patch.object(
            analysis, "_analyzer_provenance", return_value=clean_analyzer_release()
        ):
            result = analysis.analyze_family(
                records, baseline_label="adaptation_shipped"
            )
        self.assertTrue(all(item["design"]["pass"] for item in result["individual_analyses"]))
        self.assertEqual(
            result["design"]["manifest"], "dm10-historical-adaptation-replication-v1"
        )
        self.assertFalse(result["design"]["pass"])
        self.assertFalse(result["formal_rmi_violation"])

    def test_exact_historical_family_passes_without_scalar_attribution(self) -> None:
        records = historical_design_records()
        with mock.patch.object(
            analysis, "_analyzer_provenance", return_value=clean_analyzer_release()
        ):
            result = analysis.analyze_family(
                records, baseline_label="adaptation_shipped"
            )
        self.assertTrue(result["design"]["pass"])
        self.assertEqual(result["design"]["anchor_label"], "adaptation_shipped")
        self.assertEqual(result["condition_violation_inference"]["n_family_cells"], 147)
        self.assertTrue(result["formal_rmi_violation"])
        self.assertEqual(result["historical_replication_claim"]["status"], "not_run")
        self.assertFalse(result["historical_replication_claim"]["scalar_attribution_allowed"])
        self.assertEqual(result["change_from_baseline_inference"]["status"], "not_run")

    def test_missing_label_seed_or_wrong_anchor_suppresses_historical_claim(self) -> None:
        for index in range(3):
            if index == 0:
                records = historical_design_records(
                    labels=("adaptation_prior", "adaptation_shipped")
                )
                anchor = "adaptation_shipped"
            elif index == 1:
                records = [
                    record
                    for record in historical_design_records()
                    if not (
                        record["condition"]["label"] == "adaptation_prior"
                        and record["checkpoint"]["seed"] == analysis.PRIMARY_SEEDS[-1]
                    )
                ]
                anchor = "adaptation_shipped"
            else:
                records = historical_design_records()
                anchor = "adaptation_off"
            with self.subTest(index=index), mock.patch.object(
                analysis, "_analyzer_provenance", return_value=clean_analyzer_release()
            ):
                result = analysis.analyze_family(records, baseline_label=anchor)
                self.assertFalse(result["design"]["pass"])
                self.assertFalse(result["formal_rmi_violation"])


class AnalyzerReleaseGateTests(unittest.TestCase):
    def test_dirty_untracked_blob_or_revision_mismatch_suppresses_claim(self) -> None:
        records = strong_design_records()
        cases = {
            "dirty": clean_analyzer_release(clean=False),
            "untracked": clean_analyzer_release(tracked=False),
            "blob_mismatch": clean_analyzer_release(local_matches_head=False),
            "revision_mismatch": clean_analyzer_release(head="b" * 40),
        }
        expected_failures = {
            "dirty": "analyzer_git_clean",
            "untracked": "analyzer_tracked",
            "blob_mismatch": "analyzer_local_matches_head",
            "revision_mismatch": "analyzer_revision_matches_acquisition",
        }
        for name, provenance in cases.items():
            with self.subTest(name=name), mock.patch.object(
                analysis, "_analyzer_provenance", return_value=provenance
            ):
                result = analysis.analyze_family(records)
                self.assertFalse(result["design"]["pass"])
                self.assertFalse(result["formal_rmi_violation"])
                self.assertIn(expected_failures[name], result["design"]["failed_checks"])

    def test_clean_tracked_matching_release_allows_statistical_claim(self) -> None:
        with mock.patch.object(
            analysis, "_analyzer_provenance", return_value=clean_analyzer_release()
        ):
            result = analysis.analyze_family(strong_design_records())
        self.assertTrue(result["design"]["pass"])
        self.assertTrue(result["formal_rmi_violation"])

    def test_output_preflight_accepts_outside_or_ignored_and_rejects_unignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "race.json"
            self.assertEqual(measure.validate_output_path(outside), outside.resolve())
            self.assertEqual(analysis.validate_output_path(outside), outside.resolve())
        ignored = measure.ROOT / "out" / "race-model-test.json"
        self.assertEqual(measure.validate_output_path(ignored), ignored.resolve())
        self.assertEqual(analysis.validate_output_path(ignored), ignored.resolve())
        unignored = measure.ROOT / "mechanism_influence" / "race-model-output.json"
        with self.assertRaises(ValueError):
            measure.validate_output_path(unignored)
        with self.assertRaises(ValueError):
            analysis.validate_output_path(unignored)


class AnalysisHotfixProvenanceTests(unittest.TestCase):
    @staticmethod
    def _git(repository: Path, *arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _hotfix_fixture(
        self,
        directory: str,
        *,
        extra_changed_path: bool = False,
        critical_source_change: bool = False,
        non_ancestor_head: bool = False,
    ) -> dict:
        repository = Path(directory) / "release"
        analyzer_path = repository / analysis.ANALYZER_RELATIVE_PATH
        test_path = repository / "tests/test_race_model.py"
        driver_path = repository / "mechanism_influence/race_model_measure.py"
        network_path = repository / "routec_net_io.py"
        latency_path = repository / "response_latency_routec.py"
        checkpoint_path = repository / "checkpoint/seed1.pt"
        for path in (
            analyzer_path,
            test_path,
            driver_path,
            network_path,
            latency_path,
            checkpoint_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
        analyzer_path.write_text("base analyzer\n", encoding="utf-8")
        test_path.write_text("base tests\n", encoding="utf-8")
        driver_path.write_text("frozen driver\n", encoding="utf-8")
        network_path.write_text("frozen network\n", encoding="utf-8")
        latency_path.write_text("frozen latency\n", encoding="utf-8")
        checkpoint_path.write_bytes(b"frozen checkpoint")

        self._git(repository, "init", "-q")
        self._git(repository, "config", "user.name", "Race Test")
        self._git(repository, "config", "user.email", "race@example.invalid")
        self._git(repository, "add", ".")
        self._git(repository, "commit", "-q", "-m", "initial")
        initial_commit = self._git(repository, "rev-parse", "HEAD")
        (repository / "acquisition-base.txt").write_text("base\n", encoding="utf-8")
        self._git(repository, "add", "acquisition-base.txt")
        self._git(repository, "commit", "-q", "-m", "acquisition base")
        base_commit = self._git(repository, "rev-parse", "HEAD")
        branch = self._git(repository, "rev-parse", "--abbrev-ref", "HEAD")

        record = acquisition_record(
            a=[20.0, 21.0, 22.0, 23.0],
            v=[24.0, 25.0, 26.0, 27.0],
            av=[10.0, 11.0, 12.0, 13.0],
            git_clean=True,
        )
        source_paths = {
            "driver": driver_path,
            "network_io": network_path,
            "latency_module": latency_path,
        }
        source_hashes = {
            name: self._file_sha256(path) for name, path in source_paths.items()
        }
        record["code"].update(
            {
                "root": str(repository.resolve()),
                **{name: str(path.resolve()) for name, path in source_paths.items()},
                "driver_sha256": source_hashes["driver"],
                "source_sha256": source_hashes,
                "git": {"head": base_commit, "branch": branch, "clean": True},
            }
        )
        record["checkpoint"]["path"] = str(checkpoint_path.resolve())
        record["checkpoint"]["sha256"] = self._file_sha256(checkpoint_path)
        rehash_raw(record)
        derived = analysis.analyze_acquisition(record)

        if non_ancestor_head:
            self._git(repository, "checkout", "-q", "-b", "sibling", initial_commit)
        analyzer_path.write_text("hotfix analyzer\n", encoding="utf-8")
        test_path.write_text("hotfix tests\n", encoding="utf-8")
        if extra_changed_path:
            (repository / "README.md").write_text("not allowlisted\n", encoding="utf-8")
        if critical_source_change:
            network_path.write_text("changed network\n", encoding="utf-8")
        self._git(repository, "add", ".")
        self._git(repository, "commit", "-q", "-m", "analysis hotfix")
        head_commit = self._git(repository, "rev-parse", "HEAD")
        release_sha = self._file_sha256(analyzer_path)
        release = {
            "path": str(analyzer_path.resolve()),
            "sha256": release_sha,
            "tracked": True,
            "head_blob_sha256": release_sha,
            "local_matches_head": True,
            "git": {
                "head": head_commit,
                "branch": self._git(repository, "rev-parse", "--abbrev-ref", "HEAD"),
                "clean": self._git(repository, "status", "--porcelain") == "",
            },
        }
        return {
            "repository": repository,
            "record": record,
            "derived": derived,
            "release": release,
            "base_commit": base_commit,
            "head_commit": head_commit,
            "checkpoint_path": checkpoint_path,
        }

    def test_verified_analysis_only_hotfix_records_complete_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._hotfix_fixture(directory)
            policy = analysis._analysis_release_policy(
                [fixture["record"]],
                [fixture["derived"]],
                fixture["release"],
                repository_root=fixture["repository"],
            )
        self.assertTrue(policy["pass"])
        self.assertEqual(policy["mode"], "analysis_only_hotfix")
        self.assertEqual(policy["status"], "verified")
        self.assertEqual(policy["base_commit"], fixture["base_commit"])
        self.assertEqual(policy["head_commit"], fixture["head_commit"])
        self.assertEqual(
            policy["changed_paths"], list(analysis.ANALYSIS_HOTFIX_ALLOWED_PATHS)
        )
        self.assertTrue(
            all(
                entry["preserved"]
                for entry in policy["preserved_hashes"][
                    "acquisition_critical_files"
                ].values()
            )
        )
        self.assertTrue(
            next(
                iter(policy["preserved_hashes"]["validated_raw_artifacts"].values())
            )["exact"]
        )

    def test_hotfix_gate_rejects_disallowed_or_unpreserved_state(self) -> None:
        cases = (
            "extra_path",
            "critical_source",
            "non_ancestor",
            "dirty_release",
            "blob_mismatch",
            "recorded_source_mismatch",
            "checkpoint_mismatch",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = self._hotfix_fixture(
                    directory,
                    extra_changed_path=case == "extra_path",
                    critical_source_change=case == "critical_source",
                    non_ancestor_head=case == "non_ancestor",
                )
                release = copy.deepcopy(fixture["release"])
                record = copy.deepcopy(fixture["record"])
                derived = fixture["derived"]
                if case == "dirty_release":
                    release["git"]["clean"] = False
                elif case == "blob_mismatch":
                    release["local_matches_head"] = False
                    release["head_blob_sha256"] = _digest("wrong analyzer blob")
                elif case == "recorded_source_mismatch":
                    record["code"]["source_sha256"]["network_io"] = "f" * 64
                    rehash_raw(record)
                    derived = analysis.analyze_acquisition(record)
                elif case == "checkpoint_mismatch":
                    fixture["checkpoint_path"].write_bytes(b"changed checkpoint")
                policy = analysis._analysis_release_policy(
                    [record],
                    [derived],
                    release,
                    repository_root=fixture["repository"],
                )
                self.assertFalse(policy["pass"])
                self.assertEqual(policy["status"], "rejected")
                self.assertTrue(policy["failed_checks"])


class QualityControlTests(unittest.TestCase):
    def test_all_censored_is_explicit_insufficient_hits_and_inference_not_run(self) -> None:
        records = [
            acquisition_record(
                a=[math.inf] * 4,
                v=[math.inf] * 4,
                av=[math.inf] * 4,
                checkpoint_seed=seed,
            )
            for seed in range(1, 11)
        ]
        individual = analysis.analyze_acquisition(records[0])["intensities"][0]
        self.assertEqual(individual["qc"]["classification"], "insufficient_hits")
        family = analysis.analyze_family(records)
        self.assertEqual(family["condition_violation_inference"]["status"], "not_run")
        self.assertFalse(family["formal_rmi_violation"])

    def test_hit_rate_imbalance_suppresses_strong_statistical_false_positive(self) -> None:
        records = [
            acquisition_record(
                a=[20.0, math.inf, math.inf, math.inf],
                v=[20.0, math.inf, math.inf, math.inf],
                av=[5.0, 5.0, 5.0, math.inf],
                checkpoint_seed=seed,
            )
            for seed in range(1, 11)
        ]
        result = analysis.analyze_family(records)
        row = result["individual_analyses"][0]["intensities"][0]
        self.assertEqual(row["qc"]["classification"], "joint_detection_latency")
        first_cell = result["condition_violation_inference"]["cells"][0]
        self.assertEqual(first_cell["p_one_sided_global_max_stat"], 1.0 / 1024.0)
        self.assertFalse(first_cell["qc_pass_all_checkpoints"])
        self.assertFalse(first_cell["formal_rmi_violation"])
        self.assertFalse(result["formal_rmi_violation"])

    def test_catch_failure_is_uninterpretable(self) -> None:
        result = analysis.analyze_acquisition(
            acquisition_record(
                a=[20.0] * 4,
                v=[20.0] * 4,
                av=[5.0] * 4,
                catch=[9.0, math.inf, math.inf, math.inf],
            )
        )["intensities"][0]
        self.assertEqual(result["qc"]["classification"], "uninterpretable")
        self.assertFalse(result["qc"]["formal_latency_qc_pass"])


class IntegrityAndBoundaryTests(unittest.TestCase):
    def test_exact_scalar_and_historical_registries_are_non_mixable(self) -> None:
        for family, cells in measure.MECHANISM_CELL_REGISTRY.items():
            for expected_cell_id, specification in cells.items():
                with self.subTest(family=family, cell=expected_cell_id):
                    measured = measure.registered_mechanism_cell(
                        specification["requested"], family
                    )
                    analyzed = analysis.registered_mechanism_cell(
                        specification["requested"], family
                    )
                    self.assertEqual(measured["cell_id"], expected_cell_id)
                    self.assertEqual(analyzed, measured)

        a_zero = dict(measure.BASELINE_MECHANISMS)
        a_zero["aM"] = 0.0
        d_zero = dict(measure.BASELINE_MECHANISMS)
        d_zero["dM"] = 0.0
        self.assertEqual(measure.validate_one_factor(a_zero), "aM")
        self.assertEqual(measure.validate_one_factor(d_zero), "dM")

        historical_off = dict(measure.BASELINE_MECHANISMS)
        historical_off.update(aM=0.0, dM=0.0)
        historical = measure.registered_mechanism_cell(
            historical_off, measure.HISTORICAL_ADAPTATION_FAMILY
        )
        self.assertEqual(historical["cell_id"], "adaptation_off")
        self.assertFalse(historical["scalar_one_factor"])
        with self.assertRaises(ValueError):
            measure.registered_mechanism_cell(historical_off, measure.SCALAR_RMI_FAMILY)

        arbitrary_pair = dict(measure.BASELINE_MECHANISMS)
        arbitrary_pair.update(aM=0.0, dM=8.0)
        with self.assertRaises(ValueError):
            measure.registered_mechanism_cell(
                arbitrary_pair, measure.HISTORICAL_ADAPTATION_FAMILY
            )
        historical_plus_lesion = dict(historical_off)
        historical_plus_lesion["gNMDA"] = 0.0
        with self.assertRaises(ValueError):
            measure.registered_mechanism_cell(
                historical_plus_lesion, measure.HISTORICAL_ADAPTATION_FAMILY
            )
        arbitrary_scalar = dict(measure.BASELINE_MECHANISMS)
        arbitrary_scalar["aM"] = 0.01
        with self.assertRaises(ValueError):
            measure.registered_mechanism_cell(arbitrary_scalar, measure.SCALAR_RMI_FAMILY)

        record = acquisition_record(
            a=[20.0] * 4,
            v=[20.0] * 4,
            av=[5.0] * 4,
            requested=historical_off,
            design_family=analysis.HISTORICAL_ADAPTATION_FAMILY,
        )
        derived = analysis.analyze_acquisition(record)
        self.assertEqual(derived["design_family"], analysis.HISTORICAL_ADAPTATION_FAMILY)
        self.assertEqual(derived["cell_id"], "adaptation_off")
        self.assertFalse(derived["design"]["pass"])

        scalar_records = [
            acquisition_record(
                a=[20.0] * 4,
                v=[20.0] * 4,
                av=[5.0] * 4,
                checkpoint_seed=seed,
            )
            for seed in (1, 2)
        ]
        historical_shipped = dict(measure.BASELINE_MECHANISMS)
        historical_records = [
            acquisition_record(
                a=[20.0] * 4,
                v=[20.0] * 4,
                av=[5.0] * 4,
                checkpoint_seed=seed,
                requested=historical_shipped,
                design_family=analysis.HISTORICAL_ADAPTATION_FAMILY,
            )
            for seed in (1, 2)
        ]
        with self.assertRaises(ValueError):
            analysis.analyze_family(scalar_records + historical_records)

    def test_zero_negative_and_late_acquisition_latencies(self) -> None:
        for value in (0.0, -1.0, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                measure._encode_trials([value], 60.0)
        encoded, hit = measure._encode_trials([61.0, math.nan, math.inf], 60.0)
        self.assertEqual(encoded, ["+Inf", "+Inf", "+Inf"])
        self.assertEqual(hit, [False, False, False])

        zero = condition_block([1.0], 60.0)
        zero["latency_ms"][0] = 0.0
        with self.assertRaises(ValueError):
            analysis.restore_trial_sample(zero, 60.0)
        late = condition_block([1.0], 60.0)
        late["latency_ms"][0] = 61.0
        with self.assertRaises(ValueError):
            analysis.restore_trial_sample(late, 60.0)

    def test_integrity_and_one_factor_forgery_each_fail_closed(self) -> None:
        base = acquisition_record(a=[20.0] * 4, v=[20.0] * 4, av=[5.0] * 4)
        mutations = []

        one_factor = copy.deepcopy(base)
        one_factor["condition"]["one_factor"] = False
        mutations.append(one_factor)
        plasticity = copy.deepcopy(base)
        plasticity["integrity"]["plasticity_enabled"] = True
        mutations.append(plasticity)
        state = copy.deepcopy(base)
        state["integrity"]["state_sha256_after"] = "b" * 64
        mutations.append(state)
        readout = copy.deepcopy(base)
        readout["integrity"]["readout_md5_after"] = ["c" * 32, "d" * 32]
        mutations.append(readout)
        two_targets = copy.deepcopy(base)
        two_targets["condition"]["requested"]["aM"] = 0.03
        two_targets["condition"]["requested"]["dM"] = 11.0
        mutations.append(two_targets)
        non_target = copy.deepcopy(base)
        non_target["effective"]["tau_gaba"] = 12.0
        mutations.append(non_target)
        bad_configuration = copy.deepcopy(base)
        bad_configuration["configuration_sha256"] = "e" * 64
        mutations.append(bad_configuration)
        bad_source = copy.deepcopy(base)
        bad_source["code"]["source_sha256"]["driver"] = "bad"
        mutations.append(bad_source)
        mismatched_driver = copy.deepcopy(base)
        mismatched_driver["code"]["driver_sha256"] = "f" * 64
        mutations.append(mismatched_driver)

        for index, record in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                analysis.analyze_acquisition(rehash_raw(record))

    def test_state_snapshot_guard_and_stable_label_rng(self) -> None:
        module = torch.nn.Linear(3, 2)
        snapshot = measure.snapshot_state_dict(module)
        before = measure.state_dict_sha256(module)
        self.assertEqual(measure.assert_state_dict_unchanged(module, snapshot), before)
        with torch.no_grad():
            module.weight[0, 0].add_(1.0)
        with self.assertRaises(AssertionError):
            measure.assert_state_dict_unchanged(module, snapshot)

        a_first = measure.derive_stream_seed(7, 42, 0.2, "A", 3)
        v_middle = measure.derive_stream_seed(7, 42, 0.2, "V", 3)
        a_after_reordering = measure.derive_stream_seed(7, 42, 0.2, "A", 3)
        self.assertEqual(a_first, a_after_reordering)
        self.assertNotEqual(a_first, v_middle)

    def test_raw_checksum_detects_any_post_acquisition_tamper(self) -> None:
        record = acquisition_record(a=[20.0] * 4, v=[20.0] * 4, av=[5.0] * 4)
        record["rows"][0]["conditions"]["AV"]["latency_ms"][0] = 6.0
        with self.assertRaises(ValueError):
            analysis.analyze_acquisition(record)

    def test_every_endpoint_semantic_field_is_reconciled(self) -> None:
        base = acquisition_record(a=[20.0] * 4, v=[20.0] * 4, av=[5.0] * 4)
        replacements = {
            "response_rule_id": "forged",
            "response_rule": "last spike",
            "threshold": "MSI population spike count > 999",
            "latency_formula": "frame * 10",
            "latency_domain_ms": "0 <= latency <= 60",
            "resolution_ms": 10.0,
            "interpretation": "human reaction time",
            "sensitivity_horizons_ms": [60.0],
        }
        for key, replacement in replacements.items():
            forged = copy.deepcopy(base)
            forged["measurement"]["endpoint"][key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                analysis.analyze_acquisition(rehash_raw(forged))

        extra_key = copy.deepcopy(base)
        extra_key["measurement"]["endpoint"]["extra"] = True
        with self.assertRaises(ValueError):
            analysis.analyze_acquisition(rehash_raw(extra_key))

    def test_rng_ledger_omissions_and_mutations_fail_primary_design(self) -> None:
        base = acquisition_record(
            a=[20.0] * analysis.PRIMARY_N_TRIALS,
            v=[20.0] * analysis.PRIMARY_N_TRIALS,
            av=[5.0] * analysis.PRIMARY_N_TRIALS,
            checkpoint_seed=analysis.PRIMARY_SEEDS[0],
            intensities=analysis.PRIMARY_INTENSITIES,
            horizon_ms=analysis.PRIMARY_HORIZON_MS,
            git_clean=True,
        )
        self.assertTrue(analysis.analyze_acquisition(base)["design"]["pass"])

        def first_substream(record: dict) -> dict:
            return record["rows"][0]["conditions"]["A"]["rng"]["substreams"][0]

        mutations = []
        missing_before = copy.deepcopy(base)
        del first_substream(missing_before)["state_before_sha256"]
        mutations.append(missing_before)
        missing_after = copy.deepcopy(base)
        del first_substream(missing_after)["state_after_sha256"]
        mutations.append(missing_after)
        bad_algorithm = copy.deepcopy(base)
        bad_algorithm["rows"][0]["conditions"]["A"]["rng"]["algorithm"] = "unknown"
        mutations.append(bad_algorithm)
        bad_derivation = copy.deepcopy(base)
        bad_derivation["measurement"]["stream_seed_derivation"] = "index-derived"
        mutations.append(bad_derivation)
        bad_order = copy.deepcopy(base)
        bad_order["measurement"]["condition_order"] = ["V", "A", "AV", "catch"]
        mutations.append(bad_order)
        bad_label = copy.deepcopy(base)
        bad_label["rows"][0]["conditions"]["A"]["rng"]["stable_cell_key"][
            "sensory_condition"
        ] = "V"
        mutations.append(bad_label)
        bad_index = copy.deepcopy(base)
        first_substream(bad_index)["index"] = 1
        mutations.append(bad_index)
        bad_range = copy.deepcopy(base)
        first_substream(bad_range)["trial_stop_exclusive"] = 99
        mutations.append(bad_range)
        bad_seed = copy.deepcopy(base)
        first_substream(bad_seed)["stream_seed"] += 1
        mutations.append(bad_seed)

        for index, record in enumerate(mutations):
            with self.subTest(index=index):
                derived = analysis.analyze_acquisition(rehash_raw(record))
                self.assertFalse(derived["design"]["pass"])
                self.assertIn(
                    "stable_label_rng_and_trial_ids", derived["design"]["failed_checks"]
                )

    def test_derived_provenance_links_raw_config_sources_git_and_analyzer(self) -> None:
        record = acquisition_record(a=[20.0] * 4, v=[20.0] * 4, av=[5.0] * 4)
        derived = analysis.analyze_acquisition(record)
        provenance = derived["input_provenance"]
        self.assertEqual(
            provenance["raw_payload_sha256"], record["integrity"]["raw_payload_sha256"]
        )
        self.assertEqual(provenance["configuration_sha256"], record["configuration_sha256"])
        self.assertEqual(provenance["acquisition_source_sha256"], record["code"]["source_sha256"])
        self.assertEqual(provenance["acquisition_git"], record["code"]["git"])
        self.assertEqual(len(derived["analyzer_provenance"]["sha256"]), 64)

        changed_git = copy.deepcopy(record)
        changed_git["code"]["git"]["branch"] = "different-provenance"
        changed_derived = analysis.analyze_acquisition(rehash_raw(changed_git))
        self.assertNotEqual(
            derived["derived_payload_sha256"], changed_derived["derived_payload_sha256"]
        )
        self.assertEqual(
            changed_derived["input_provenance"]["acquisition_git"]["branch"],
            "different-provenance",
        )


class CheckpointInferenceTests(unittest.TestCase):
    def test_exact_ties_are_layout_invariant_and_conservative(self) -> None:
        cell_7 = np.asarray(
            [
                -0.40000152587890625,
                -0.5,
                0.6999969482421875,
                -0.3999977111816406,
                0.7999992370605469,
                0.3000030517578125,
                0.20000076293945312,
                0.0,
                1.0,
                0.7000007629394531,
            ],
            dtype=np.float64,
        )
        cell_47 = np.asarray(
            [
                -0.20000076293945312,
                0.20000076293945312,
                -0.10000038146972656,
                -0.39999961853027344,
                0.6000003814697266,
                0.0,
                -0.20000076293945312,
                0.0,
                -0.10000038146972656,
                0.5999984741210938,
            ],
            dtype=np.float64,
        )
        labels = [{"cell": 7}, {"cell": 47}]
        c_values = np.ascontiguousarray(np.column_stack((cell_7, cell_47)))
        f_values = np.asfortranarray(c_values)

        c_result = analysis.checkpoint_sign_flip_max_stat(c_values, labels)
        f_result = analysis.checkpoint_sign_flip_max_stat(f_values, labels)
        for result in (c_result, f_result):
            counts = [
                round(cell["p_one_sided_uncorrected"] * result["n_sign_patterns"])
                for cell in result["cells"]
            ]
            self.assertEqual(counts, [102, 404])
        for key in (
            "status",
            "n_checkpoints",
            "n_family_cells",
            "n_sign_patterns",
            "exact_raw_sign_flip",
            "global_p_one_sided_max_stat",
        ):
            self.assertEqual(c_result[key], f_result[key])
        for c_cell, f_cell in zip(c_result["cells"], f_result["cells"]):
            self.assertEqual(c_cell["cell"], f_cell["cell"])
            self.assertEqual(
                c_cell["p_one_sided_uncorrected"],
                f_cell["p_one_sided_uncorrected"],
            )
            self.assertEqual(
                c_cell["p_one_sided_global_max_stat"],
                f_cell["p_one_sided_global_max_stat"],
            )
            np.testing.assert_allclose(
                [
                    c_cell["checkpoint_mean_G_ms"],
                    c_cell["checkpoint_se_G_ms"],
                    c_cell["checkpoint_t"],
                    c_cell["approx_simultaneous_lower_G_ms"],
                ],
                [
                    f_cell["checkpoint_mean_G_ms"],
                    f_cell["checkpoint_se_G_ms"],
                    f_cell["checkpoint_t"],
                    f_cell["approx_simultaneous_lower_G_ms"],
                ],
                rtol=4.0 * np.finfo(np.float64).eps,
                atol=4.0 * np.finfo(np.float64).eps,
            )

        observed = np.asarray([1.0, math.inf, -math.inf])
        within = observed.copy()
        within[0] -= 32.0 * np.finfo(np.float64).eps
        outside = observed.copy()
        outside[0] -= 128.0 * np.finfo(np.float64).eps
        with np.errstate(invalid="raise"):
            self.assertEqual(
                analysis._permutation_exceeds_or_ties(within, observed).tolist(),
                [True, True, True],
            )
            self.assertEqual(
                analysis._permutation_exceeds_or_ties(outside, observed).tolist(),
                [False, True, True],
            )

    def test_zero_strong_and_mixed_max_stat_matrices_are_deterministic(self) -> None:
        labels = [{"q": 0.05}, {"q": 0.10}]
        zero = analysis.checkpoint_sign_flip_max_stat(np.zeros((10, 2)), labels)
        self.assertEqual(zero["global_p_one_sided_max_stat"], 1.0)
        self.assertTrue(all(cell["approx_simultaneous_lower_G_ms"] == 0 for cell in zero["cells"]))

        strong = analysis.checkpoint_sign_flip_max_stat(np.full((10, 2), 15.0), labels)
        self.assertEqual(strong["global_p_one_sided_max_stat"], 1.0 / 1024.0)
        self.assertTrue(all(cell["approx_simultaneous_lower_G_ms"] == 15.0 for cell in strong["cells"]))
        self.assertTrue(strong["exact_raw_sign_flip"])
        self.assertTrue(strong["simultaneous_lower_band"]["approximate"])

        mixed_values = np.column_stack((np.asarray([-1.0, 1.0] * 5), np.full(10, 2.0)))
        first = analysis.checkpoint_sign_flip_max_stat(mixed_values, labels)
        second = analysis.checkpoint_sign_flip_max_stat(mixed_values.copy(), labels)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        for cell in first["cells"]:
            self.assertGreaterEqual(
                cell["p_one_sided_global_max_stat"], cell["p_one_sided_uncorrected"]
            )

    def test_formal_claim_is_exact_conjunction(self) -> None:
        self.assertTrue(
            analysis.formal_claim_gate(
                0.01, 1.0, qc_pass=True, integrity_pass=True, design_pass=True
            )
        )
        failures = [
            (0.05, 1.0, True, True, True),
            (0.01, 0.0, True, True, True),
            (0.01, 1.0, False, True, True),
            (0.01, 1.0, True, False, True),
            (0.01, 1.0, True, True, False),
        ]
        for p_value, lower, qc, integrity, design in failures:
            with self.subTest(failure=(p_value, lower, qc, integrity, design)):
                self.assertFalse(
                    analysis.formal_claim_gate(
                        p_value,
                        lower,
                        qc_pass=qc,
                        integrity_pass=integrity,
                        design_pass=design,
                    )
                )


if __name__ == "__main__":
    unittest.main()
