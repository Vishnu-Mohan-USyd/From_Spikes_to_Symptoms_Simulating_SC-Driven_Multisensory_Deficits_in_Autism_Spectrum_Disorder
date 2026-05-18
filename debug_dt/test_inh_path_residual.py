"""H_g: test if I_M_inh path contributes residual at dt=0.05.

At dt=0.05 + delays×2 + gNMDA=1.30 baseline = 127.84 ms.
Test: turn Form 2 OFF on I_M_inh path only (revert to bare-add). If this
moves HW noticeably, the I_M_inh path has a non-trivial residual.
"""
from __future__ import annotations
import inspect
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from TBW_test import (
    load_msi_model,
    compute_tbw_temporal_fusion_persep,
    fit_psychometric_curve_improved,
)
import debug_dt.patch_variants as PV
import Training as TRN

OFFSETS = list(range(-50, 51, 2))
OFFSETS_MS = [o * 10 for o in OFFSETS]
TRIALS = 50
SEED = 12345
CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"


def install_inh_form2_off():
    """Patch: keep MSI-exc NMDA Form 2 ON, but revert MSI-inh NMDA to bare add."""
    src = inspect.getsource(PV.ORIG_UPDATE)
    src = textwrap.dedent(src)
    src = src.replace("def update_all_layers_batch(", "def _inh_off(", 1)
    # The I_M_inh block has its own if/else; replace with unconditional bare add.
    old_block = ("if self.dt_correct_nmda:\n            self.I_M_inh.add_(I_nmda_inh * nmda_source_scale)\n        else:\n            self.I_M_inh.add_(I_nmda_inh)")
    new_block = "self.I_M_inh.add_(I_nmda_inh)"
    cnt = src.count(old_block)
    assert cnt == 1, f"Expected 1 occurrence of inh block, got {cnt}"
    src = src.replace(old_block, new_block)
    ns = {}
    glb = TRN.__dict__.copy()
    exec(src, glb, ns)
    TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch = ns["_inh_off"]


def install_inh_form2_double():
    """Patch: keep MSI-exc NMDA Form 2 ON, but DOUBLE MSI-inh NMDA injection.
    Tests sensitivity in the opposite direction."""
    src = inspect.getsource(PV.ORIG_UPDATE)
    src = textwrap.dedent(src)
    src = src.replace("def update_all_layers_batch(", "def _inh_2x(", 1)
    old = "self.I_M_inh.add_(I_nmda_inh * nmda_source_scale)"
    new = "self.I_M_inh.add_(I_nmda_inh * nmda_source_scale * 2.0)"
    cnt = src.count(old)
    assert cnt == 1, f"Expected 1 occurrence, got {cnt}"
    src = src.replace(old, new)
    ns = {}
    glb = TRN.__dict__.copy()
    exec(src, glb, ns)
    TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch = ns["_inh_2x"]


def measure_hw(dt: float, nsub: int, *, delay_scale: float = 1.0,
                gNMDA: float = 1.30, seed: int = 12345):
    rng = np.random.default_rng(seed)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        net = load_msi_model(CKPT, device="cuda")
        net.dt = dt
        net.n_substeps = nsub
        net.dt_correct_nmda = True
        net.gNMDA = gNMDA
        net.plasticity_enabled = False
        net.freeze_g_FFinh = True
        if delay_scale != 1.0:
            for attr in ["conduction_delay_a2msi", "conduction_delay_v2msi",
                         "conduction_delay_inA_inh", "conduction_delay_inV_inh",
                         "conduction_delay_a2msi_inh", "conduction_delay_v2msi_inh",
                         "conduction_delay_msi_inh2exc", "conduction_delay_msi2out"]:
                setattr(net, attr, int(round(getattr(net, attr) * delay_scale)))
            import torch as _t
            B = TRIALS
            for buf_name, delay_attr, shape in [
                ("buffer_a2msi", "conduction_delay_a2msi", (B, net.n)),
                ("buffer_v2msi", "conduction_delay_v2msi", (B, net.n)),
                ("buffer_inA_inh", "conduction_delay_inA_inh", (B, net.n)),
                ("buffer_inV_inh", "conduction_delay_inV_inh", (B, net.n)),
                ("buffer_a2msi_inh", "conduction_delay_a2msi_inh", (B, net.n)),
                ("buffer_v2msi_inh", "conduction_delay_v2msi_inh", (B, net.n)),
                ("buffer_msi_inh2exc", "conduction_delay_msi_inh2exc", (B, net.n_inh)),
                ("buffer_msi2out", "conduction_delay_msi2out", (B, net.n)),
            ]:
                d = getattr(net, delay_attr)
                if d > 0:
                    setattr(net, buf_name, _t.zeros((d,) + shape, device=net.device))
                    net._delay_positions[buf_name] = 0
        p, _ = compute_tbw_temporal_fusion_persep(
            net, OFFSETS, n_trials=TRIALS, T=60, D=5, stim_in=1.0,
            sigma=2.0, valley_threshold=0.4, min_peak_height=0.2,
            min_peak_separation=3, min_total=10.0,
        )
        del net
        torch.cuda.empty_cache()
    finally:
        np.random.default_rng = orig
    try:
        fit = fit_psychometric_curve_improved(np.asarray(OFFSETS_MS), p)
        hw = float(fit.get("tbw", float("nan")))
    except Exception:
        hw = float("nan")
    return hw


def main():
    print("H_g secondary tests at dt=0.05 + delays×2, gNMDA=1.30")
    print()
    PV.restore_original()
    h_base = measure_hw(0.05, 200, delay_scale=2.0, gNMDA=1.30)
    print(f"  Baseline (Form2 ON both paths)     : HW = {h_base:.2f}")

    PV.restore_original(); install_inh_form2_off()
    h_off = measure_hw(0.05, 200, delay_scale=2.0, gNMDA=1.30)
    PV.restore_original()
    print(f"  MSI-inh NMDA bare-add (Form2 OFF)  : HW = {h_off:.2f}  (Δ = {h_off-h_base:+.2f})")

    PV.restore_original(); install_inh_form2_double()
    h_2x = measure_hw(0.05, 200, delay_scale=2.0, gNMDA=1.30)
    PV.restore_original()
    print(f"  MSI-inh NMDA × 2 (over-correct)    : HW = {h_2x:.2f}  (Δ = {h_2x-h_base:+.2f})")


if __name__ == "__main__":
    main()
