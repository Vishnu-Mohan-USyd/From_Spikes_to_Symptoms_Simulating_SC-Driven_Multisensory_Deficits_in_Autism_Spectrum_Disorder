#!/usr/bin/env python
"""
TASK #192 — observational ep40 retrain after the 5-phase disynaptic-inh
rip-and-rebuild on branch sci-minimal-inhibition (@ d262e16):

  Phase A — RIP direct FF-inh shortcut + Mexican-hat learning + AGC
  Phase B — split tau_syn: tau_ampa=2.5 ms, tau_gaba=50 ms (biology)
  Phase C — delays: 4 ms exc / 9.2 ms disynaptic-inh (Whyland-Bickford 2018)
  Phase D — Hebbian + Oja on W_*2msiInh_{AMPA,NMDA} (input -> interneuron)
  Phase E — D'Amour-Froemke +/-10 ms symmetric iSTDP on W_msiInh2Exc_GABA
            (eta=5e-4, clamp [0, 0.5])

Per Team-Lead dispatch:
  - 5 ckpts parallel on H200 dev2 (run separately via --model_idx 0..4)
  - NO smoke-gates, NO auto-kill — observational only
  - Calibrator SKIPPED (auto_calibrate_W_msiInh2Exc_GABA NOT called) per
    spec §5.4b allowance: iSTDP self-balance during training is expected
  - Rolling 5-epoch profile dump (ep5/10/15/20/25/30/35/40)
    + per-profile-point checkpoint save
  - Alert thresholds emit "[ALERT]"-prefixed lines on stdout but do NOT
    terminate training (team-lead decides kill/continue out-of-band)
  - TBW/SBW HW+R^2 NOT computed inline; the saved profile ckpts allow
    post-hoc measurement via TBW_test.py / SBW_test.py (lightweight at
    full-dataset cost — kept off the hot path)

Code dir: /scratch/fsts_retrain_20260526/code_task192/Training.py
         md5 expected: 11eb993f4f8f3cc5215b0f247ccf19a4
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

CODE_DIR = "/scratch/fsts_retrain_asym_20260530/code"  # task#47: import PATCHED Training.py
RETRAIN_ROOT = Path("/scratch/fsts_retrain_asym_20260530")  # task#47: outputs -> new root
CKPT_DIR = RETRAIN_ROOT / "checkpoints_asymdelay"  # task#47: lead-specified asym output dir
LOG_DIR = RETRAIN_ROOT / "logs"

sys.path.insert(0, CODE_DIR)

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

from Training import (
    MultiBatchAudVisMSINetworkTime,
    assign_unimodal_preferred_locations,
    _init,
    make_checkpoint,
)

# ----------------------------- config -----------------------------
SIGMA_IN = 6.5
N_EPOCHS = 80
BATCH_SIZE = 1000
BASE_SEED = 42
PROFILE_EVERY = 5
PROFILE_EPOCHS = list(range(PROFILE_EVERY, N_EPOCHS + 1, PROFILE_EVERY))

# Match the long-standing biology calibration carried across recent retrains.
DM_OVERRIDE = 4.0
GNMDA_TRAIN = 0.7

# Alert thresholds (NOTE: these do NOT auto-kill; they emit "[ALERT]" lines)
ALERT_MSI_HZ_LOW = 5.0
ALERT_MSI_HZ_HIGH = 200.0
ALERT_W_GABA_NEAR_CLAMP = 0.45      # mean approaches 0.45 -> saturating
ALERT_W_GABA_CLAMP = 0.5            # nominal clamp
ALERT_N_ACTIVE_DROP_PCT = 0.50      # epoch-to-epoch drop >50 %
ALERT_W_INH_INPUT_NEAR_CLAMP = 4.5  # default W_inh_input_clamp = 5.0
# ------------------------------------------------------------------


@torch.no_grad()
def _wstat(t: torch.Tensor, clamp_max: float | None = None):
    """Return (mean_abs, max_abs, std_abs, frac_at_clamp)."""
    a = t.abs()
    m = float(a.mean().item())
    M = float(a.max().item())
    s = float(a.std().item())
    if clamp_max is None:
        cf = float("nan")
    else:
        cf = float(((a >= 0.999 * clamp_max).float().mean()).item())
    return m, M, s, cf


# ---- monkey-patch print_epoch_spike_summary to stash MSI rates on self ----
import Training as _T
_orig_pes = _T.MultiBatchAudVisMSINetworkTime.print_epoch_spike_summary


def _patched_pes(self, tag=""):
    """Stash MSI scalar + per-neuron Hz on self BEFORE original resets."""
    try:
        if getattr(self, "_dbg_steps", 0) > 0:
            norm = self._dbg_steps * self.n
            ms_per_frame = self.n_substeps * self.dt
            hz_fact = 1000.0 / ms_per_frame
            self._last_msi_rate_hz = float((self._dbg_spk_MSI / norm) * hz_fact)
            pn = getattr(self, "_dbg_spk_MSI_perN", None)
            if pn is not None and pn.numel() == self.n:
                self._last_msi_perN_hz = (
                    pn.float() / float(self._dbg_steps) * hz_fact
                ).detach().cpu().clone()
                pn.zero_()
            else:
                self._last_msi_perN_hz = None
        else:
            self._last_msi_rate_hz = float("nan")
            self._last_msi_perN_hz = None
    except Exception:
        self._last_msi_rate_hz = float("nan")
        self._last_msi_perN_hz = None
    _orig_pes(self, tag)


_T.MultiBatchAudVisMSINetworkTime.print_epoch_spike_summary = _patched_pes


@torch.no_grad()
def _profile_dump(net, model_idx, epoch_1idx, csv_writer, prev_n_active):
    """Append a profile row and emit ALERT lines for misbehavior."""
    msi_hz = float(getattr(net, "_last_msi_rate_hz", float("nan")))
    perN = getattr(net, "_last_msi_perN_hz", None)
    if perN is not None:
        n_total = int(perN.numel())
        n_active_1 = int((perN > 1.0).sum().item())
        n_active_5 = int((perN > 5.0).sum().item())
        msi_peak = float(perN.max().item())
        msi_std = float(perN.std().item())
        active_mask = perN > 1.0
        msi_active_mean = (
            float(perN[active_mask].mean().item()) if active_mask.any() else 0.0
        )
        n_active_pct = 100.0 * n_active_1 / max(n_total, 1)
    else:
        n_total = -1
        n_active_1 = -1
        n_active_5 = -1
        msi_peak = float("nan")
        msi_std = float("nan")
        msi_active_mean = float("nan")
        n_active_pct = float("nan")

    # MSI_inh firing-rate snapshot (per debugger #193 §Q1.5 remaining unknown +
    # sci_inhibition_phase_d_revision.md §6.3). Training.py does not accumulate
    # MSI_inh spikes across the epoch (only MSI exc via _dbg_spk_MSI), so we
    # take a single-substep snapshot from _latest_sMSI_inh. Single-substep noise
    # is reduced by averaging over batch=256; usable as a sanity signal for the
    # frozen FF→PV+ drive producing the biology-expected 20-40 Hz PV+ rate.
    try:
        s_inh = net._latest_sMSI_inh.float()              # (B, n_inh)
        per_neuron_frac = s_inh.mean(dim=0)               # (n_inh,)
        msi_inh_grand_frac = float(s_inh.mean().item())
        dt_s = float(getattr(net, "dt", 0.1)) / 1000.0    # ms → s
        msi_inh_hz_instant = (
            msi_inh_grand_frac / dt_s if dt_s > 0 else float("nan")
        )
        msi_inh_active_pct = float(
            (per_neuron_frac > 0).float().mean().item()
        ) * 100.0
    except Exception:
        msi_inh_hz_instant = float("nan")
        msi_inh_active_pct = float("nan")

    # Phase D plastic edges (input -> interneuron), Phase E plastic edge
    # (interneuron -> MSI exc), and the surround Mexican-hat (fixed shape).
    # Clamp values pulled from net so the runner doesn't drift if the spec
    # tunes them later.
    W_inh_input_clamp = float(getattr(net, "W_inh_input_clamp", 5.0))
    W_gaba_clamp = float(getattr(net, "W_gaba_clamp", 0.5))

    aA_aA_m, aA_aA_M, aA_aA_s, aA_aA_cf = _wstat(net.W_a2msi_AMPA)
    aV_aA_m, aV_aA_M, aV_aA_s, aV_aA_cf = _wstat(net.W_v2msi_AMPA)
    aA_nN_m, aA_nN_M, aA_nN_s, aA_nN_cf = _wstat(net.W_a2msi_NMDA)
    aV_nN_m, aV_nN_M, aV_nN_s, aV_nN_cf = _wstat(net.W_v2msi_NMDA)

    iA_aA_m, iA_aA_M, iA_aA_s, iA_aA_cf = _wstat(net.W_a2msiInh_AMPA, W_inh_input_clamp)
    iV_aA_m, iV_aA_M, iV_aA_s, iV_aA_cf = _wstat(net.W_v2msiInh_AMPA, W_inh_input_clamp)
    iA_nN_m, iA_nN_M, iA_nN_s, iA_nN_cf = _wstat(net.W_a2msiInh_NMDA, W_inh_input_clamp)
    iV_nN_m, iV_nN_M, iV_nN_s, iV_nN_cf = _wstat(net.W_v2msiInh_NMDA, W_inh_input_clamp)

    gaba_m, gaba_M, gaba_s, gaba_cf = _wstat(net.W_msiInh2Exc_GABA, W_gaba_clamp)
    lat_m, lat_M, lat_s, _ = _wstat(net.W_MSI_inh)

    # Numerical health
    v_msi_min = float(net.v_msi.min().item())
    v_msi_max = float(net.v_msi.max().item())
    n_nan = 0
    for name, t in [
        ("v_msi", net.v_msi),
        ("I_M", net.I_M),
        ("I_M_gaba", getattr(net, "I_M_gaba", torch.zeros(1))),
        ("W_a2msi_AMPA", net.W_a2msi_AMPA),
        ("W_v2msi_AMPA", net.W_v2msi_AMPA),
        ("W_a2msi_NMDA", net.W_a2msi_NMDA),
        ("W_v2msi_NMDA", net.W_v2msi_NMDA),
        ("W_a2msiInh_AMPA", net.W_a2msiInh_AMPA),
        ("W_v2msiInh_AMPA", net.W_v2msiInh_AMPA),
        ("W_msiInh2Exc_GABA", net.W_msiInh2Exc_GABA),
        ("W_MSI_inh", net.W_MSI_inh),
    ]:
        if t.numel() and not torch.isfinite(t).all():
            n_nan += int((~torch.isfinite(t)).sum().item())

    gff = float(getattr(net, "g_FFinh", 0.0))   # orphan post-Phase A
    gtonic = float(getattr(net, "g_tonic", 0.0))
    eta_istdp = float(getattr(net, "eta_istdp", 0.0))

    row = {
        "ep": epoch_1idx,
        "MSI_Hz": msi_hz,
        "MSI_peak": msi_peak,
        "MSI_std": msi_std,
        "MSI_active_mean": msi_active_mean,
        "n_active_gt1Hz": n_active_1,
        "n_active_gt5Hz": n_active_5,
        "n_active_pct": n_active_pct,
        # MSI_inh single-substep snapshot (no Training.py accumulator);
        # mean over batch=256 makes this a usable sanity signal for the
        # frozen Whyland-Bickford FF→PV+ drive (~30 Hz biology target).
        "MSI_inh_Hz_instant": msi_inh_hz_instant,
        "MSI_inh_active_pct": msi_inh_active_pct,
        # plastic excit -> MSI
        "W_a2msi_AMPA_m": aA_aA_m, "W_a2msi_AMPA_M": aA_aA_M, "W_a2msi_AMPA_s": aA_aA_s,
        "W_v2msi_AMPA_m": aV_aA_m, "W_v2msi_AMPA_M": aV_aA_M, "W_v2msi_AMPA_s": aV_aA_s,
        "W_a2msi_NMDA_m": aA_nN_m, "W_a2msi_NMDA_M": aA_nN_M, "W_a2msi_NMDA_s": aA_nN_s,
        "W_v2msi_NMDA_m": aV_nN_m, "W_v2msi_NMDA_M": aV_nN_M, "W_v2msi_NMDA_s": aV_nN_s,
        # Phase D: plastic excit -> interneuron
        "W_a2msiInh_AMPA_m": iA_aA_m, "W_a2msiInh_AMPA_M": iA_aA_M,
        "W_a2msiInh_AMPA_s": iA_aA_s, "W_a2msiInh_AMPA_cf": iA_aA_cf,
        "W_v2msiInh_AMPA_m": iV_aA_m, "W_v2msiInh_AMPA_M": iV_aA_M,
        "W_v2msiInh_AMPA_s": iV_aA_s, "W_v2msiInh_AMPA_cf": iV_aA_cf,
        "W_a2msiInh_NMDA_m": iA_nN_m, "W_a2msiInh_NMDA_M": iA_nN_M,
        "W_a2msiInh_NMDA_s": iA_nN_s, "W_a2msiInh_NMDA_cf": iA_nN_cf,
        "W_v2msiInh_NMDA_m": iV_nN_m, "W_v2msiInh_NMDA_M": iV_nN_M,
        "W_v2msiInh_NMDA_s": iV_nN_s, "W_v2msiInh_NMDA_cf": iV_nN_cf,
        # Phase E: plastic interneuron -> MSI exc (PRIORITY)
        "W_msiInh2Exc_GABA_m": gaba_m, "W_msiInh2Exc_GABA_M": gaba_M,
        "W_msiInh2Exc_GABA_s": gaba_s, "W_msiInh2Exc_GABA_cf": gaba_cf,
        # surround Mexican-hat (geometry only, no plasticity)
        "W_MSI_inh_m": lat_m, "W_MSI_inh_M": lat_M, "W_MSI_inh_s": lat_s,
        # scalars
        "g_FFinh": gff, "g_tonic": gtonic, "eta_istdp": eta_istdp,
        "v_msi_min": v_msi_min, "v_msi_max": v_msi_max, "n_nan_total": n_nan,
    }
    csv_writer.writerow(row)

    # -------------------- ALERTs (stdout, non-fatal) --------------------
    alerts = []
    if msi_hz == msi_hz:  # not NaN
        if msi_hz < ALERT_MSI_HZ_LOW:
            alerts.append(f"MSI_Hz={msi_hz:.2f} < {ALERT_MSI_HZ_LOW} (under-activity)")
        if msi_hz > ALERT_MSI_HZ_HIGH:
            alerts.append(f"MSI_Hz={msi_hz:.2f} > {ALERT_MSI_HZ_HIGH} (runaway)")
    if gaba_m >= ALERT_W_GABA_NEAR_CLAMP:
        alerts.append(
            f"W_msiInh2Exc_GABA mean={gaba_m:.4f} approaching clamp "
            f"{ALERT_W_GABA_CLAMP}  (iSTDP saturation risk)"
        )
    if gaba_cf >= 0.05:
        alerts.append(
            f"W_msiInh2Exc_GABA clamp%={gaba_cf*100:.1f}% (>=5% saturating)"
        )
    for tag, cf in [
        ("W_a2msiInh_AMPA", iA_aA_cf),
        ("W_v2msiInh_AMPA", iV_aA_cf),
        ("W_a2msiInh_NMDA", iA_nN_cf),
        ("W_v2msiInh_NMDA", iV_nN_cf),
    ]:
        if cf >= 0.05:
            alerts.append(f"{tag} clamp%={cf*100:.1f}% (>=5% saturating at {W_inh_input_clamp})")
    if (
        prev_n_active is not None and prev_n_active > 0 and n_active_1 >= 0
        and (prev_n_active - n_active_1) / prev_n_active >= ALERT_N_ACTIVE_DROP_PCT
    ):
        alerts.append(
            f"n_active dropped {prev_n_active} -> {n_active_1} "
            f"({100*(prev_n_active - n_active_1)/prev_n_active:.0f}% epoch-to-epoch)"
        )
    if n_nan > 0:
        alerts.append(f"NaN/Inf detected in {n_nan} elements across monitored tensors")

    for msg in alerts:
        print(f"[ALERT m{model_idx} ep{epoch_1idx:02d}] {msg}", flush=True)
    # ---- soft-bound smoke GATE probe (debugger killcriteria FINAL 3ed7d2aa) ----
    # print-only (no CSV schema change). Corrected per debugger: ~5% pinned AT
    # cap is HEALTHY (Gutig asymptotes TO cap); cap-engagement discriminator is
    # ep25 (p99>0.026); ep5 has a WEIGHT FLOOR (p50<0.001 = no FF learning).
    _cap = 0.024
    _wt = torch.cat([
        (net.W_a2msi_AMPA + net.W_a2msi_NMDA).detach().flatten(),
        (net.W_v2msi_AMPA + net.W_v2msi_NMDA).detach().flatten(),
    ]).double().cpu()
    _q = torch.quantile(_wt, torch.tensor([0.5, 0.99], dtype=torch.float64))
    _wp50 = float(_q[0]); _wp99 = float(_q[1]); _wmax = float(_wt.max())
    _pin = float((_wt > 0.9 * _cap).double().mean()) * 100.0
    _wnan = bool(torch.isnan(_wt).any() or torch.isinf(_wt).any())
    # training-time BC on perN (debugger BC formula; NOTE: training-perN basis,
    # NOT meas() eval -> absolute value not directly comparable to the 0.40-0.59/
    # 0.94-0.96 meas() reference; live tripwire only, authoritative BC = debugger
    # meas() apparatus at ep30/ep80).
    if perN is not None and int(perN.numel()) > 3:
        _a = perN.detach().double(); _mn = _a.mean(); _s = _a.std()
        if float(_s) > 1e-9:
            _g = ((_a - _mn) ** 3).mean() / _s ** 3
            _k = ((_a - _mn) ** 4).mean() / _s ** 4
            _bc = float((_g * _g + 1) / _k)
        else:
            _bc = float('nan')
    else:
        _bc = float('nan')
    _pv = msi_inh_hz_instant
    # pre-registered weight gates (retrain2_gate_spec / addendum d3e4de94):
    #  - FF->PV frozen-identity: max|W - W_init| must be 0 (any drift = KILL, any epoch)
    #  - disynaptic GABA iSTDP (W_msiInh2Exc_GABA, clamp 0.5): KILL 100%-pinned OR mean->0
    if hasattr(net, '_ffpv_ref'):
        _ffpv_drift = max(float((getattr(net, k).detach() - net._ffpv_ref[k]).abs().max())
                          for k in net._ffpv_ref)
    else:
        _ffpv_drift = float('nan')
    _gaba = net.W_msiInh2Exc_GABA.detach().double().cpu()
    _gaba_clamp = float(getattr(net, 'W_gaba_clamp', 0.5))
    _gaba_mean = float(_gaba.mean())
    _gaba_pin = float((_gaba >= 0.999 * _gaba_clamp).double().mean()) * 100.0
    _ep = epoch_1idx; _kill = []; _flag = []
    # FF->PV must never drift (frozen-identity) — checked every epoch
    if _ffpv_drift == _ffpv_drift and _ffpv_drift > 1e-9:
        _kill.append(f'FF->PV drift={_ffpv_drift:.2e}>0 (frozen-identity violated)')
    if _ep == 5:
        if _wp50 < 0.001: _kill.append(f'ep5 Wtot_p50={_wp50:.5f}<0.001 (no FF learning)')
        if _pin > 50.0: _kill.append(f'ep5 pct_pinned={_pin:.1f}%>50 (mass-pin)')
        if _pv == _pv and _pv > 500.0: _flag.append(f'PV(snap)={_pv:.0f}>500 single-substep NOISE (NOT a kill; 2ms-refractory hard-bounds sustained PV<=500; authoritative PV=debugger eval-basis ckpt-load ~343)')
        if _pv == _pv and _pv < 200.0: _flag.append(f'PV={_pv:.0f}<200 (low; 346 expected)')
        if _bc == _bc and _bc > 0.85: _flag.append(f'BC_train={_bc:.3f}>0.85 ADVISORY (training-perN basis; authoritative=debugger meas() ckpt-load)')
    if _ep >= 25 and _wp99 > 0.026: _kill.append(f'ep{_ep} Wtot_p99={_wp99:.4f}>0.026 (cap not bounding)')
    if _wmax > 0.026: _kill.append(f'ep{_ep} Wtot_max={_wmax:.4f}>0.026 (HARD cap breach)')
    if _wnan: _kill.append(f'ep{_ep} NaN/Inf in FF->MSI_exc weights')
    if _ep >= 30:
        if msi_hz == msi_hz and msi_hz < 3.0: _kill.append(f'ep{_ep} MSI_mean={msi_hz:.1f}<3 (collapse; coarse in-proc mean, authoritative=median ckpt-load)')
        if msi_hz == msi_hz and msi_hz > 100.0: _kill.append(f'ep{_ep} MSI_mean={msi_hz:.1f}>100 (runaway; confirm climbing vs prior ep)')
        # staged Sarle-BC (spec): PASS<=0.55 / FLAG 0.55-0.62 / KILL>0.62 (or climbing toward 0.90)
        if _bc == _bc and _bc > 0.85: _flag.append(f'BC_train={_bc:.3f}>0.85 ADVISORY (training-perN basis untested vs meas(); authoritative BC=debugger ckpt-load eval-basis 0.62)')
        _na_floor = 150 if _ep == 30 else 120
        if n_active_1 >= 0 and n_active_1 < _na_floor: _kill.append(f'ep{_ep} n_active={n_active_1}<{_na_floor} (mass-silencing; authoritative n_silent ckpt-load)')
        if msi_peak == msi_peak and msi_peak > 200.0: _kill.append(f'ep{_ep} peak={msi_peak:.0f}>200 (runaway clique; primary live WTA flag)')
        if _pin > 25.0: _flag.append(f'pct_pinned={_pin:.1f}%>25 (KILL only if CLIMBING across 2 checks)')
        # disynaptic GABA iSTDP health (pre-registered Stage-2 KILL)
        if _gaba_pin >= 99.5: _kill.append(f'ep{_ep} GABA {_gaba_pin:.0f}%-pinned@clamp{_gaba_clamp} (disynaptic iSTDP saturated)')
        if _gaba_mean < 1e-4: _kill.append(f'ep{_ep} GABA mean={_gaba_mean:.2e}->0 (disynaptic iSTDP collapsed)')
    _verdict = 'KILL' if _kill else 'ok'
    print(f'[GATE m{model_idx} ep{epoch_1idx:02d}] Wtot p50={_wp50:.5f} p99={_wp99:.5f} '
          f'max={_wmax:.5f} pct_pinned={_pin:.1f}% nan={_wnan} | BC_train={_bc:.3f} '
          f'MSI_mean={msi_hz:.1f} PV={_pv:.0f} n_active={n_active_1} peak={msi_peak:.0f} | '
          f'ffpv_drift={_ffpv_drift:.1e} GABA(mean={_gaba_mean:.4f} pin={_gaba_pin:.0f}%) | VERDICT={_verdict}'
          + ('' if not _kill else ' :: KILL ' + '; '.join(_kill))
          + ('' if not _flag else ' :: FLAG ' + '; '.join(_flag)), flush=True)
    return row, n_active_1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_idx", type=int, required=True, help="0..4")
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--prefix", type=str, default="task192", help="output prefix for ckpt/csv/workdir")
    # task#200 Phase-6 (option b): homeostatic exc-scaling knobs -> set on net post-construction
    # (mutable_hparams). r*/alpha default None => keep Training.py default (no override) so
    # Training.py md5 stays stable across r* tweaks. r_target is in TRACKER units (calibrated).
    parser.add_argument("--hss_sensor", type=str, default="net_rate",
                        choices=["net_rate", "exc_drive"], help="task#200 homeostatic sensor")
    parser.add_argument("--hss_r_target", type=float, required=True,
                        help="task#200 calibrated r*_tracker (tracker units); REQUIRED (lead decision 3) "
                             "-- NO None->1.17 placeholder fallback; a forgotten flag crashes loudly "
                             "instead of silently using the catastrophic 1.17.")
    parser.add_argument("--hss_alpha", type=float, default=None,
                        help="task#200 per-batch scale gain; None=Training.py default")
    parser.add_argument("--hss_step_clip", type=float, default=None,
                        help="task#200 per-batch +/- clamp on the scale factor; None=Training.py default")
    args = parser.parse_args()
    model_idx = args.model_idx
    n_epochs = args.n_epochs
    if model_idx not in range(5):
        print(f"!! model_idx must be 0..4; got {model_idx}", file=sys.stderr)
        sys.exit(1)

    seed = BASE_SEED + model_idx
    prefix = args.prefix
    work_dir = RETRAIN_ROOT / f"work_{prefix}_full_m{model_idx:02d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)

    csv_path = LOG_DIR / f"{prefix}_profile_m{model_idx}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"=== TASK #192 disynaptic-inh ep{n_epochs} retrain "
        f"(post 5-phase rebuild d262e16) — model_idx={model_idx} seed={seed} ===",
        flush=True,
    )
    print(
        f"=== sigma_in={SIGMA_IN}  dM={DM_OVERRIDE}  gNMDA={GNMDA_TRAIN}  "
        f"n_epochs={n_epochs}  batch={BATCH_SIZE}  profile-every={PROFILE_EVERY} ===",
        flush=True,
    )
    print(f"=== code_dir: {CODE_DIR}  work_dir: {work_dir} ===", flush=True)
    print(f"=== csv: {csv_path} ===", flush=True)
    _t_md5 = os.popen(f"md5sum {CODE_DIR}/Training.py").read().split()[0]
    print(f"=== Training.py md5: {_t_md5} (expect 11eb993f4f8f3cc5215b0f247ccf19a4) ===", flush=True)
    _st = os.popen(
        f"grep -cE 'Erev_gaba|gaba_m' {CODE_DIR}/Training.py"
    ).read().strip()
    print(f"=== Training.py pre-S-tier invariant (Erev_gaba|gaba_m, must be 0): {_st} ===", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    net = MultiBatchAudVisMSINetworkTime(
        n_neurons=180, batch_size=BATCH_SIZE,
        lr_unimodal=2e-2, lr_msi=2e-2, lr_readout=8e-4,
        sigma_in=SIGMA_IN, sigma_teacher=2.0, noise_std=0.02,
        single_modality_prob=0.5,
        v_thresh=0.3, dt=0.1, tau_m=20.0, n_substeps=100, loc_jitter_std=0,
        space_size=180,
        # task#47: __init__ now USES these kwargs (f92d04a override reverted) =>
        # asymmetric A/V delays a2msi=250 (25.0 ms) / v2msi=400 (40.0 ms, visual slower);
        # inh legs +20 substeps; msi_inh2exc = max(a2msi,v2msi)+50.
        conduction_delay_a2msi=250, conduction_delay_v2msi=400,
    )
    assign_unimodal_preferred_locations(net)
    net.b_uniA.data.fill_(0.0)
    net.b_uniV.data.fill_(0.0)
    _init(net.W_a2msi_AMPA, 0.004)
    _init(net.W_v2msi_AMPA, 0.004)
    _init(net.W_a2msi_NMDA, 0.004)
    _init(net.W_v2msi_NMDA, 0.004)

    with torch.no_grad():
        net.gNMDA = GNMDA_TRAIN
        net.tau_nmda = 80.0
        net.nmda_alpha = 0.1
        net.Erev_nmda = 20.0
        net.tau_nmdaVolt = 100.0
        net.v_nmda_rest = -65.0
        net.nmda_vrest_offset = 7.0
        net.mg_vhalf = -35.0
        net.dM = DM_OVERRIDE

    net.u_a.fill_(0.7)
    net.u_v.fill_(0.7)
    net.tau_rec = 400.0
    net.input_scaling = 400
    net.g_GABA = 10

    # ----- task#200 Phase-6: homeostatic exc-scaling config (option b, argparse) -----
    # Override Training.py defaults on the net; these are exported in mutable_hparams, so
    # they persist into every ckpt. r_target is in TRACKER units (trial-averaged Hz, ~25x
    # deflated vs evoked) -> the CALIBRATED r*_tracker, NOT 30. When overriding r_target,
    # RE-SEED both persistent trackers to it so deficit~0 at the epoch-26 onset (no
    # cold-start scaling kick). None => keep the Training.py default untouched.
    net.hss_sensor = args.hss_sensor
    if args.hss_r_target is not None:
        net.hss_r_target = float(args.hss_r_target)
        net.msi_exc_rate_persistent.fill_(float(args.hss_r_target))
        net.msi_exc_drive_persistent.fill_(float(args.hss_r_target))
    if args.hss_alpha is not None:
        net.hss_alpha = float(args.hss_alpha)
    if args.hss_step_clip is not None:
        net.hss_step_clip = float(args.hss_step_clip)
    print(f"[net cfg] task#200 HSS: sensor={net.hss_sensor} r_target={net.hss_r_target} "
          f"alpha={net.hss_alpha} step_clip={net.hss_step_clip} beta={net.hss_beta}", flush=True)

    # ----- frozen-identity reference: FF->PV (W_*2msiInh_*) must NOT drift -----
    # pre-registered Stage-1 KILL (retrain2_gate_spec): max|d vs init|=0. These are
    # requires_grad=False, so drift is expected EXACTLY 0; snapshot true post-
    # construction init for the [GATE] frozen-identity check every profile epoch.
    net._ffpv_ref = {k: getattr(net, k).detach().clone()
                     for k in ('W_a2msiInh_AMPA', 'W_a2msiInh_NMDA',
                               'W_v2msiInh_AMPA', 'W_v2msiInh_NMDA')}

    # ----- diagnostic preflight log -----
    print(
        f"[net cfg] dM={net.dM} aM={net.aM} gNMDA={net.gNMDA} "
        f"g_FFinh={getattr(net,'g_FFinh',0.0)} (orphan post-Phase A) "
        f"g_tonic={getattr(net, 'g_tonic', 0.0)}",
        flush=True,
    )
    print(
        f"[net cfg] Phase B: tau_ampa={getattr(net,'tau_ampa',float('nan'))} "
        f"tau_gaba={getattr(net,'tau_gaba',float('nan'))}",
        flush=True,
    )
    print(
        f"[net cfg] Phase C: delays a2msi={net.conduction_delay_a2msi} "
        f"v2msi={net.conduction_delay_v2msi} "
        f"a2msi_inh={net.conduction_delay_a2msi_inh} "
        f"v2msi_inh={net.conduction_delay_v2msi_inh} "
        f"msi_inh2exc={net.conduction_delay_msi_inh2exc}",
        flush=True,
    )
    print(
        f"[net cfg] Phase E: tau_istdp_pre={getattr(net,'tau_istdp_pre',float('nan'))} "
        f"tau_istdp_post={getattr(net,'tau_istdp_post',float('nan'))} "
        f"eta_istdp={getattr(net,'eta_istdp',float('nan'))} "
        f"istdp_baseline={getattr(net,'istdp_baseline',float('nan'))} "
        f"W_gaba_clamp={getattr(net,'W_gaba_clamp',float('nan'))}",
        flush=True,
    )
    print(
        f"[net cfg] W_a2msiInh_AMPA mean={net.W_a2msiInh_AMPA.mean().item():.4f} "
        f"max={net.W_a2msiInh_AMPA.max().item():.4f}",
        flush=True,
    )
    print(
        f"[net cfg] W_msiInh2Exc_GABA mean={net.W_msiInh2Exc_GABA.mean().item():.5f} "
        f"max={net.W_msiInh2Exc_GABA.max().item():.5f} "
        f"sum={net.W_msiInh2Exc_GABA.sum().item():.3f}",
        flush=True,
    )

    # ----- per-ckpt CSV -----
    CSV_FIELDS = [
        "ep", "MSI_Hz", "MSI_peak", "MSI_std", "MSI_active_mean",
        "n_active_gt1Hz", "n_active_gt5Hz", "n_active_pct",
        "MSI_inh_Hz_instant", "MSI_inh_active_pct",
        "W_a2msi_AMPA_m", "W_a2msi_AMPA_M", "W_a2msi_AMPA_s",
        "W_v2msi_AMPA_m", "W_v2msi_AMPA_M", "W_v2msi_AMPA_s",
        "W_a2msi_NMDA_m", "W_a2msi_NMDA_M", "W_a2msi_NMDA_s",
        "W_v2msi_NMDA_m", "W_v2msi_NMDA_M", "W_v2msi_NMDA_s",
        "W_a2msiInh_AMPA_m", "W_a2msiInh_AMPA_M", "W_a2msiInh_AMPA_s",
        "W_a2msiInh_AMPA_cf",
        "W_v2msiInh_AMPA_m", "W_v2msiInh_AMPA_M", "W_v2msiInh_AMPA_s",
        "W_v2msiInh_AMPA_cf",
        "W_a2msiInh_NMDA_m", "W_a2msiInh_NMDA_M", "W_a2msiInh_NMDA_s",
        "W_a2msiInh_NMDA_cf",
        "W_v2msiInh_NMDA_m", "W_v2msiInh_NMDA_M", "W_v2msiInh_NMDA_s",
        "W_v2msiInh_NMDA_cf",
        "W_msiInh2Exc_GABA_m", "W_msiInh2Exc_GABA_M", "W_msiInh2Exc_GABA_s",
        "W_msiInh2Exc_GABA_cf",
        "W_MSI_inh_m", "W_MSI_inh_M", "W_MSI_inh_s",
        "g_FFinh", "g_tonic", "eta_istdp",
        "v_msi_min", "v_msi_max", "n_nan_total",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    csv_writer.writeheader()

    # ----- training loop -----
    print(
        f"\n--- training (unsupervised) — {n_epochs} epochs, NO SMOKE GATES, NO AUTO-KILL ---",
        flush=True,
    )
    unsup_start = time.time()
    diverged = False
    prev_n_active = None

    for epoch in range(n_epochs):
        epoch_start = time.time()
        try:
            net.train_unsupervised_batch(1000, batch_size=256, debug=False, epoch_idx=epoch)
        except Exception as e:
            print(
                f"!! EXCEPTION during epoch {epoch}: {e!r}. Continuing "
                f"(no auto-kill — team-lead decides via [ALERT] grep).",
                flush=True,
            )
            print(f"[ALERT m{model_idx} ep{epoch+1:02d}] training exception: {e!r}", flush=True)
            diverged = True

        net.print_epoch_spike_summary(f"unsup {epoch + 1:02d}")
        epoch_time = time.time() - epoch_start

        # Quick one-liner every epoch (trajectory)
        with torch.no_grad():
            wAA_m = float(net.W_a2msi_AMPA.abs().mean().item())
            wAA_M = float(net.W_a2msi_AMPA.abs().max().item())
            wInhA_m = float(net.W_a2msiInh_AMPA.abs().mean().item())
            wInhA_M = float(net.W_a2msiInh_AMPA.abs().max().item())
            wGABA_m = float(net.W_msiInh2Exc_GABA.abs().mean().item())
            wGABA_M = float(net.W_msiInh2Exc_GABA.abs().max().item())
        msi_hz = float(getattr(net, "_last_msi_rate_hz", float("nan")))
        print(
            f"  Epoch {epoch + 1}/{n_epochs} time={epoch_time:.1f}s  "
            f"MSI={msi_hz:.2f}Hz  "
            f"W_AMPA|m|={wAA_m:.4f}|M|={wAA_M:.4f}  "
            f"W_a2Inh|m|={wInhA_m:.4f}|M|={wInhA_M:.4f}  "
            f"W_GABA|m|={wGABA_m:.4f}|M|={wGABA_M:.4f}",
            flush=True,
        )

        if (epoch + 1) in PROFILE_EPOCHS:
            print(f"\n=== PROFILE m{model_idx} ep{epoch+1:02d} ===", flush=True)
            row, n_active_now = _profile_dump(
                net, model_idx, epoch + 1, csv_writer, prev_n_active
            )
            csv_file.flush()
            prev_n_active = n_active_now
            print(
                f"  MSI: grand={row['MSI_Hz']:.2f}Hz peak={row['MSI_peak']:.1f} "
                f"active_mean={row['MSI_active_mean']:.2f}Hz "
                f"n_active>1Hz={row['n_active_gt1Hz']}/180 "
                f"({row['n_active_pct']:.1f}%)",
                flush=True,
            )
            print(
                f"  W_a2msiInh_AMPA: m={row['W_a2msiInh_AMPA_m']:.4f} "
                f"M={row['W_a2msiInh_AMPA_M']:.4f} "
                f"clamp%={100*row['W_a2msiInh_AMPA_cf']:.2f}",
                flush=True,
            )
            print(
                f"  W_v2msiInh_AMPA: m={row['W_v2msiInh_AMPA_m']:.4f} "
                f"M={row['W_v2msiInh_AMPA_M']:.4f} "
                f"clamp%={100*row['W_v2msiInh_AMPA_cf']:.2f}",
                flush=True,
            )
            print(
                f"  W_msiInh2Exc_GABA: m={row['W_msiInh2Exc_GABA_m']:.4f} "
                f"M={row['W_msiInh2Exc_GABA_M']:.4f} "
                f"clamp%={100*row['W_msiInh2Exc_GABA_cf']:.2f}  "
                f"(clamp={getattr(net,'W_gaba_clamp',0.5)})",
                flush=True,
            )
            print(
                f"  V_msi: min={row['v_msi_min']:.2f} max={row['v_msi_max']:.2f}  "
                f"n_NaN={row['n_nan_total']}",
                flush=True,
            )

            # Save profile-point ckpt
            try:
                ckpt_path = CKPT_DIR / f"{prefix}_m{model_idx}_ep{epoch+1:02d}.pt"
                ckpt = make_checkpoint(
                    net, epoch=epoch, optim=None,
                    comment=(
                        f"TASK#192 disynaptic-inh ep{epoch+1} m{model_idx} "
                        f"dM={DM_OVERRIDE} gNMDA={GNMDA_TRAIN} "
                        f"post-5phase-rebuild d262e16"
                    ),
                    rng_tag=True,
                )
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(ckpt, ckpt_path)
                print(f"  ckpt saved: {ckpt_path}", flush=True)
            except Exception as e:
                print(f"  !! ckpt save failed: {e!r}  (continuing)", flush=True)

    csv_file.close()
    print(f"\n--- training done in {time.time() - unsup_start:.1f}s ---", flush=True)
    print(f"Profile CSV: {csv_path}", flush=True)
    print(f"Profile ckpts: {CKPT_DIR}/{prefix}_m{model_idx}_ep??.pt", flush=True)
    print(f"[task192 m{model_idx:02d}] DONE  diverged={diverged}", flush=True)


if __name__ == "__main__":
    main()
