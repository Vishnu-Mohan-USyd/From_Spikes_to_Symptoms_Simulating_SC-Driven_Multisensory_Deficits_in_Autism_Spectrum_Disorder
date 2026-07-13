"""Subprocess regressions for the public route-C network/checkpoint interface."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]

# Isolate every import from shell/session experiment settings.  Several of these values are read
# while val36 imports the network definition, so deleting them after import would be too late.
ROUTEC_ENV_KEYS = (
    "TAU_GABA",
    "DEND_COUPLING_ALPHA",
    "MG_VHALF",
    "MG_VHALF_INH",
    "GABA_SHUNT_SURR",
    "K_SHUNT_SURR",
    "ISTDP_BASELINE",
    "SIGMA_DL_FRAMES",
    "G_REC",
    "ROUTEC_CKPT_TMPL",
    "GNMDA",
    "TAU_NMDA",
    "TAU_NMDA_V",
    "MG_K",
    "PV_GABA_SCALE",
    "G_GABA",
    "AFFERENT_JITTER_MS",
    "CONDUCTION_DELAY_V2MSI",
    "TAU_REC",
)


class RouteCNetIOTests(unittest.TestCase):
    maxDiff = None

    def clean_env(self, **overrides: str) -> dict[str, str]:
        env = os.environ.copy()
        for key in ROUTEC_ENV_KEYS:
            env.pop(key, None)
        env.update(overrides)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def run_python(
        self,
        source: str,
        *,
        env: dict[str, str] | None = None,
        timeout: int = 240,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            cwd=ROOT,
            env=self.clean_env() if env is None else env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"subprocess failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        return result

    def test_clean_import_resolves_all_ten_dm10_checkpoints(self) -> None:
        result = self.run_python(
            """
            import os
            import routec_net_io as N

            expected_env = {
                "DEND_COUPLING_ALPHA": "2",
                "MG_VHALF": "-48",
                "MG_VHALF_INH": "-30",
                "GABA_SHUNT_SURR": "1",
                "K_SHUNT_SURR": "0.026",
                "ISTDP_BASELINE": "0.56",
                "SIGMA_DL_FRAMES": "3",
                "G_REC": "0.03",
            }
            assert os.environ["TAU_GABA"] == "18"
            assert {key: os.environ.get(key) for key in expected_env} == expected_env
            assert N.CKPT_TMPL == "ckpt_ep79_seed{seed}_bs250_delay52_tau18_dL3.pt"
            assert N.SEEDS == list(range(42, 52))

            paths = [N.ckpt_path_for_seed(seed) for seed in N.SEEDS]
            assert len(paths) == 10
            assert all(os.path.isfile(path) for path in paths), paths
            print("[routec-import] PASS all 10 dm10 checkpoint paths")
            """
        )
        self.assertIn("PASS all 10", result.stdout)

    def test_explicit_tau_and_caller_overrides_are_preserved(self) -> None:
        tau10 = self.clean_env(TAU_GABA="10")
        result = self.run_python(
            """
            import os
            import routec_net_io as N

            dm10_only = (
                "DEND_COUPLING_ALPHA", "MG_VHALF", "MG_VHALF_INH",
                "GABA_SHUNT_SURR", "K_SHUNT_SURR", "ISTDP_BASELINE",
                "SIGMA_DL_FRAMES", "G_REC",
            )
            assert os.environ["TAU_GABA"] == "10"
            assert all(key not in os.environ for key in dm10_only)
            assert N.CKPT_TMPL == "ckpt_ep79_seed{seed}_bs250_delay52_tau10_dL3.pt"
            print("[routec-override] PASS explicit tau10 legacy defaults")
            """,
            env=tau10,
        )
        self.assertIn("PASS explicit tau10", result.stdout)

        custom = self.clean_env(
            TAU_GABA="18.0",
            G_REC="0.07",
            MG_VHALF="-44",
            ROUTEC_CKPT_TMPL="custom_seed{seed}.pt",
        )
        result = self.run_python(
            """
            import os
            import routec_net_io as N

            assert os.environ["TAU_GABA"] == "18.0"
            assert os.environ["G_REC"] == "0.07"
            assert os.environ["MG_VHALF"] == "-44"
            assert N.CKPT_TMPL == "custom_seed{seed}.pt"
            assert N.ckpt_path_for_seed(49).endswith("checkpoint/custom_seed49.pt")
            print("[routec-override] PASS explicit tau18 values and custom template")
            """,
            env=custom,
        )
        self.assertIn("PASS explicit tau18", result.stdout)

        result = self.run_python(
            """
            import routec_net_io as N
            assert N.CKPT_TMPL == "ckpt_ep79_seed{seed}_bs250_delay52_tau18_dL3.pt"
            print("[routec-override] PASS normalized 18.0 checkpoint tag")
            """,
            env=self.clean_env(TAU_GABA="18.0"),
        )
        self.assertIn("PASS normalized 18.0", result.stdout)

    def test_cpu_loads_exact_operating_point_for_all_ten_seeds(self) -> None:
        result = self.run_python(
            """
            import routec_net_io as N

            expected = {
                "aM": 0.02,
                "dM": 10.0,
                "pv_gaba_scale": 1.0,
                "tau_gaba": 18.0,
                "gNMDA": 0.51,
                "g_GABA": 5.56,
                "g_rec": 0.03,
                "tau_nmda": 40.0,
                "tau_nmda_v": 40.0,
                "tau_nmda_inh": 21.6,
                "u_stp_a": 0.2,
                "u_stp_v": 0.2,
                "nmda_std_scale": 0.8,
                "tau_rec": 400.0,
                "afferent_jitter_ms": 4.0,
                "dend_coupling_alpha": 2.0,
                "mg_vhalf": -48.0,
                "mg_vhalf_inh": -30.0,
                "mg_k": 0.15,
                "gaba_shunt_surr": True,
                "k_shunt_surr": 0.026,
                "istdp_baseline": 0.56,
                "sigma_dL_frames": 3.0,
                "conduction_delay_v2msi": 400.0,
                "conduction_delay_v2msi_inh": 420.0,
                "conduction_delay_v2msi_ms": 40.0,
                "conduction_delay_v2msi_inh_ms": 42.0,
            }

            for seed in N.SEEDS:
                net, load, epoch, g_rec, _ = N.load_ckpt(
                    N.ckpt_path_for_seed(seed), seed, 1, "cpu"
                )
                assert epoch == 79
                assert list(load.missing_keys) == []
                assert list(load.unexpected_keys) == []
                assert abs(float(g_rec) - 0.03) < 1e-12
                for attribute, wanted in expected.items():
                    got = getattr(net, attribute)
                    if isinstance(wanted, bool):
                        assert got is wanted, (seed, attribute, got, wanted)
                    else:
                        assert abs(float(got) - wanted) < 1e-12, (
                            seed, attribute, got, wanted
                        )
                print(f"[cpu-load s{seed}] epoch=79 strict=True attrs=27/27 exact")

            print("[cpu-load] PASS all 10 seeds; 27/27 operating-point attrs exact")
            """,
            timeout=300,
        )
        self.assertIn("PASS all 10 seeds", result.stdout)

    def test_public_dm10_shim_preflight_and_cpu_load_smoke(self) -> None:
        result = self.run_python(
            """
            import sys
            import measure.val394_dm10_stage2 as shim

            seeds = list(range(42, 52))
            derived = shim.preflight(seeds)
            assert derived == {"TAU_GABA": "18", "GNMDA": "0.51"}

            sys.path.insert(0, shim.MEASURE)
            sys.path.insert(0, shim.BUNDLE)
            import measure_ens_main as official

            shim._install_patches(official.N, derived)
            net, load, epoch, g_rec, _ = official.N.load_ckpt(
                shim.dm10_ckpt(42), 42, 1, "cpu"
            )
            assert epoch == 79
            assert list(load.missing_keys) == []
            assert list(load.unexpected_keys) == []
            assert abs(float(g_rec) - 0.03) < 1e-12
            assert [float(getattr(net, key)) for key in shim.EXPECT] == [
                shim.EXPECT[key] for key in shim.EXPECT
            ]
            print("[val394-shim] PASS ten-seed preflight and patched CPU load")
            """,
            timeout=300,
        )
        self.assertIn("PASS — all 10 seeds", result.stdout)
        self.assertIn("PASS ten-seed preflight", result.stdout)


if __name__ == "__main__":
    unittest.main()
