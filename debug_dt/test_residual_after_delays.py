"""Test residual hypotheses ON TOP of (Form 2 ON + gNMDA=1.30 + delays×2) at dt=0.05.

Baseline after H_d fix: HW(dt=0.05, delays×2) = 127.84 ms. Target = 149 ms.
Residual gap = 21 ms.

Hypotheses to test as monkey-patches:
  H_j_nmda_decay : replace nmda_m linear-Euler decay (1 - dt/tau_nmda)
                   with exp(-dt/tau_nmda).
  H_j_ampa_decay : same for ampa_m.
  H_j_v_nmda     : replace `self.v_nmda += dt * dv_nmda` with exact step
                   `v_nmda = v_nmda + (target - v_nmda) * (1 - exp(-dt/tau))`
  H_i_R_recovery : replace `R_a += (1 - R_a) * (dt/tau_rec)` with exact
                   exponential recovery.
  H_g_inh_form2  : MSI_inh_NMDA is already Form 2 patched. Check if MSI_inh
                   excitation `b_msi_inh + AMPA` has any issue (biases = 0,
                   so this is spike-driven; should be invariant).

Each test reports HW. Look for one that moves HW from 128 → 149.
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
GNMDA = 1.30


# ------------------------------------------------------------------------
# Patch installers
# ------------------------------------------------------------------------
def install_replace(replacements, tag):
    """Generic source-rewrite installer.

    replacements: list of (old_str, new_str). Each old_str must appear exactly
    once in the source.
    """
    src = inspect.getsource(PV.ORIG_UPDATE)
    src = textwrap.dedent(src)
    src = src.replace("def update_all_layers_batch(", f"def _{tag}(", 1)
    for old, new in replacements:
        cnt = src.count(old)
        assert cnt == 1, f"[{tag}] expected 1 occurrence of {old!r}, got {cnt}"
        src = src.replace(old, new)
    ns = {}
    glb = TRN.__dict__.copy()
    exec(src, glb, ns)
    TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch = ns[f"_{tag}"]


def install_nmda_decay_exp():
    install_replace([
        ("nmda_decay = 1.0 - self.dt / self.tau_nmda",
         "nmda_decay = float(torch.exp(torch.tensor(-self.dt / self.tau_nmda)).item())"),
    ], tag="ndecay_exp")


def install_ampa_decay_exp():
    install_replace([
        ("ampa_decay = 1.0 - self.dt / self.tau_ampa_lp",
         "ampa_decay = float(torch.exp(torch.tensor(-self.dt / self.tau_ampa_lp)).item())"),
    ], tag="adecay_exp")


def install_IM_decay_exp():
    install_replace([
        ("decay_factor = 1.0 - self.dt / self.tau_syn",
         "decay_factor = float(torch.exp(torch.tensor(-self.dt / self.tau_syn)).item())"),
    ], tag="im_decay_exp")


def install_all_decays_exp():
    install_replace([
        ("decay_factor = 1.0 - self.dt / self.tau_syn",
         "decay_factor = float(torch.exp(torch.tensor(-self.dt / self.tau_syn)).item())"),
        ("ampa_decay = 1.0 - self.dt / self.tau_ampa_lp",
         "ampa_decay = float(torch.exp(torch.tensor(-self.dt / self.tau_ampa_lp)).item())"),
        ("nmda_decay = 1.0 - self.dt / self.tau_nmda",
         "nmda_decay = float(torch.exp(torch.tensor(-self.dt / self.tau_nmda)).item())"),
    ], tag="all_decay_exp")


def install_R_a_recovery_exact():
    """Replace R_a, R_v linear-Euler recovery with exact exponential.

    Original:  self.R_a += (1.0 - self.R_a) * (self.dt / self.tau_rec)
    Exact:     R_a_new = 1.0 + (R_a - 1.0) * exp(-dt/tau_rec)
                       = 1.0 - (1.0 - R_a) * exp(-dt/tau_rec)
    """
    install_replace([
        ("self.R_a += (1.0 - self.R_a) * (self.dt / self.tau_rec)",
         "self.R_a.copy_(1.0 - (1.0 - self.R_a) * float(torch.exp(torch.tensor(-self.dt / self.tau_rec)).item()))"),
        ("self.R_v += (1.0 - self.R_v) * (self.dt / self.tau_rec)",
         "self.R_v.copy_(1.0 - (1.0 - self.R_v) * float(torch.exp(torch.tensor(-self.dt / self.tau_rec)).item()))"),
        ("self.R_a_inh += (1.0 - self.R_a_inh) * (self.dt / self.tau_rec)",
         "self.R_a_inh.copy_(1.0 - (1.0 - self.R_a_inh) * float(torch.exp(torch.tensor(-self.dt / self.tau_rec)).item()))"),
        ("self.R_v_inh += (1.0 - self.R_v_inh) * (self.dt / self.tau_rec)",
         "self.R_v_inh.copy_(1.0 - (1.0 - self.R_v_inh) * float(torch.exp(torch.tensor(-self.dt / self.tau_rec)).item()))"),
    ], tag="R_exp")


def install_v_nmda_exact():
    """Replace `v_nmda += dt * dv_nmda` with exact step.

    Indentation after textwrap.dedent: 8 spaces (was 12 in source).
    """
    install_replace([
        ("dv_nmda = ((self.v_msi + self.nmda_vrest_offset) - self.v_nmda) / self.tau_nmdaVolt\n        self.v_nmda += self.dt * dv_nmda",
         "_tgt_n = self.v_msi + self.nmda_vrest_offset\n        _a_n = float(torch.exp(torch.tensor(-self.dt / self.tau_nmdaVolt)).item())\n        self.v_nmda.copy_(_tgt_n + (self.v_nmda - _tgt_n) * _a_n)"),
        ("dv_nmda_inh = ((self.v_msi_inh + self.nmda_vrest_offset) - self.v_nmda_inh) / self.tau_nmdaVolt\n        self.v_nmda_inh += self.dt * dv_nmda_inh",
         "_tgt_ni = self.v_msi_inh + self.nmda_vrest_offset\n        _a_ni = float(torch.exp(torch.tensor(-self.dt / self.tau_nmdaVolt)).item())\n        self.v_nmda_inh.copy_(_tgt_ni + (self.v_nmda_inh - _tgt_ni) * _a_ni)"),
    ], tag="v_nmda_exact")


def install_v_dend_exact():
    """Replace v_dend_A, v_dend_V, v_dend_inhA, v_dend_inhV with exact step.

    Indentation after textwrap.dedent: 8 spaces (was 12 in source).
    """
    install_replace([
        ("d_va = self.dend_coupling_alpha * (self.v_msi - self.v_dend_A) / self.tau_m\n        d_vv = self.dend_coupling_alpha * (self.v_msi - self.v_dend_V) / self.tau_m\n        self.v_dend_A += self.dt * d_va\n        self.v_dend_V += self.dt * d_vv",
         "_dr = float(torch.exp(torch.tensor(-self.dt * self.dend_coupling_alpha / self.tau_m).float()).item())\n        self.v_dend_A.copy_(self.v_msi + (self.v_dend_A - self.v_msi) * _dr)\n        self.v_dend_V.copy_(self.v_msi + (self.v_dend_V - self.v_msi) * _dr)"),
        ("d_viA = self.dend_coupling_alpha * (self.v_msi_inh - self.v_dend_inhA) / self.tau_m\n        d_viV = self.dend_coupling_alpha * (self.v_msi_inh - self.v_dend_inhV) / self.tau_m\n        self.v_dend_inhA += self.dt * d_viA\n        self.v_dend_inhV += self.dt * d_viV",
         "_dri = float(torch.exp(torch.tensor(-self.dt * self.dend_coupling_alpha / self.tau_m).float()).item())\n        self.v_dend_inhA.copy_(self.v_msi_inh + (self.v_dend_inhA - self.v_msi_inh) * _dri)\n        self.v_dend_inhV.copy_(self.v_msi_inh + (self.v_dend_inhV - self.v_msi_inh) * _dri)"),
    ], tag="v_dend_exact")


# ------------------------------------------------------------------------
# Measurement
# ------------------------------------------------------------------------
def measure_hw(dt: float, nsub: int, delay_scale: float = 1.0):
    rng = np.random.default_rng(SEED)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        net = load_msi_model(CKPT, device="cuda")
        net.dt = dt
        net.n_substeps = nsub
        net.dt_correct_nmda = True
        net.gNMDA = GNMDA
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
        t0 = time.time()
        p, _ = compute_tbw_temporal_fusion_persep(
            net, OFFSETS, n_trials=TRIALS, T=60, D=5, stim_in=1.0,
            sigma=2.0, valley_threshold=0.4, min_peak_height=0.2,
            min_peak_separation=3, min_total=10.0,
        )
        elapsed = time.time() - t0
        del net
        torch.cuda.empty_cache()
    finally:
        np.random.default_rng = orig
    try:
        fit = fit_psychometric_curve_improved(np.asarray(OFFSETS_MS), p)
        hw = float(fit.get("tbw", float("nan")))
    except Exception:
        hw = float("nan")
    return hw, elapsed


def banner(s):
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def main():
    banner("Baselines (Form 2 ON, gNMDA=1.30)")
    PV.restore_original()
    h01, t = measure_hw(0.1, 100); print(f"  dt=0.1 raw      : HW={h01:.2f} ms ({t:.1f}s)")
    h05, t = measure_hw(0.05, 200); print(f"  dt=0.05 raw     : HW={h05:.2f} ms ({t:.1f}s)")
    h05d, t = measure_hw(0.05, 200, delay_scale=2.0); print(f"  dt=0.05 delays×2: HW={h05d:.2f} ms ({t:.1f}s)")
    print(f"  Gap (149 vs 128 after H_d): residual {h01 - h05d:.2f} ms")

    banner("Residual hypothesis 1: nmda_m decay exp-Euler")
    PV.restore_original(); install_nmda_decay_exp()
    h, t = measure_hw(0.05, 200, delay_scale=2.0); PV.restore_original()
    print(f"  HW(dt=0.05, delays×2, nmda_decay=exp) = {h:.2f} ms ({t:.1f}s)")
    print(f"  Δ vs 128 baseline (after H_d) = {h - h05d:+.2f} ms")

    banner("Residual hypothesis 2: ampa_m decay exp-Euler")
    PV.restore_original(); install_ampa_decay_exp()
    h, t = measure_hw(0.05, 200, delay_scale=2.0); PV.restore_original()
    print(f"  HW(dt=0.05, delays×2, ampa_decay=exp) = {h:.2f} ms ({t:.1f}s)")
    print(f"  Δ vs 128 baseline = {h - h05d:+.2f} ms")

    banner("Residual hypothesis 3: I_M decay exp-Euler")
    PV.restore_original(); install_IM_decay_exp()
    h, t = measure_hw(0.05, 200, delay_scale=2.0); PV.restore_original()
    print(f"  HW(dt=0.05, delays×2, IM_decay=exp) = {h:.2f} ms ({t:.1f}s)")
    print(f"  Δ vs 128 baseline = {h - h05d:+.2f} ms")

    banner("Residual hypothesis 4: ALL decays exp-Euler")
    PV.restore_original(); install_all_decays_exp()
    h, t = measure_hw(0.05, 200, delay_scale=2.0); PV.restore_original()
    print(f"  HW(dt=0.05, delays×2, all decay=exp) = {h:.2f} ms ({t:.1f}s)")
    print(f"  Δ vs 128 baseline = {h - h05d:+.2f} ms")

    banner("Residual hypothesis 5: R_a/v recovery exact exp")
    PV.restore_original(); install_R_a_recovery_exact()
    h, t = measure_hw(0.05, 200, delay_scale=2.0); PV.restore_original()
    print(f"  HW(dt=0.05, delays×2, R_recovery=exp) = {h:.2f} ms ({t:.1f}s)")
    print(f"  Δ vs 128 baseline = {h - h05d:+.2f} ms")

    banner("Residual hypothesis 6: v_nmda exact step")
    PV.restore_original(); install_v_nmda_exact()
    h, t = measure_hw(0.05, 200, delay_scale=2.0); PV.restore_original()
    print(f"  HW(dt=0.05, delays×2, v_nmda=exact) = {h:.2f} ms ({t:.1f}s)")
    print(f"  Δ vs 128 baseline = {h - h05d:+.2f} ms")

    banner("Residual hypothesis 7: v_dend exact step")
    PV.restore_original(); install_v_dend_exact()
    h, t = measure_hw(0.05, 200, delay_scale=2.0); PV.restore_original()
    print(f"  HW(dt=0.05, delays×2, v_dend=exact) = {h:.2f} ms ({t:.1f}s)")
    print(f"  Δ vs 128 baseline = {h - h05d:+.2f} ms")


if __name__ == "__main__":
    main()
