"""Task #29 — comprehensive dt-variance sweep.

For each metric, value@dt=0.1 (n_substeps=100) vs value@dt=0.05 (n_substeps=200),
on the canonical 10-checkpoint pool. modify_net pattern sets dt + n_substeps
+ gNMDA + per-condition extras, then calls _reset_delay_buffers().

Scope (per team-lead task #29):
  Section A: TBW for 4 remaining conditions (ff_inhibition, adaptation,
             nmda, nmda_increase). Control done in task #28 → 107.6 / 109.1.
  Section B: SBW for all 5 conditions with floor-subtraction.
  Section C: Aux metrics (control settings, gNMDA=1.30):
             - E/I ratio  (run_ei_balance pipeline, evoked window scaled to n_substeps)
             - Fano factor baseline + stim  (main_fano_fast_with_bio path)
             - Cue reliability R² + MAE     (reliability_sweep_batched)
             - Precision σ_A, σ_V, σ_B      (compute_hybrid_sensitivity_fast, CONTROL block)
             - Inverse-effectiveness MEI    (low + high intensities)
             - Response latency A, V, B     (run_latency_test; preserves line-685 Izh override)

Two temp one-line edits applied:
  - inverse_effectiveness_test.py line 2: `from typing import Sequence`
  - response_latency_test.py line 2: same
Both tagged with "task #29 TEMP" in a trailing comment for easy revert.

NO other production code changes.

Output: dt_variance_sweep_summary.json + verbatim print log.
Runtime ~75-90 min on the canonical 10-ckpt pool at default settings.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from TBW_test import run_fusion_across_models, _find_crossings, load_msi_model
from SBW_test import run_spatial_binding_across_models
from replot_all_cosmetic import (
    plot_tbw, plot_sbw, subtract_control_floor, _load_pooled,
)

BASE = ROOT / "checkpoint"
MODELS = [BASE / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
CACHE = Path(__file__).parent / "dt_sweep_cache"
CACHE.mkdir(exist_ok=True)
PLOTDIR = CACHE / "_plots"
PLOTDIR.mkdir(exist_ok=True)

# Canonical settings — match generate_all_fresh.py
TBW_OFFSETS = list(range(-50, 51, 2))         # 51
SBW_SEPARATIONS = tuple(range(-80, 85, 5))    # 33
ENH_THRESHOLD = 10.0
N_TRIALS_TBW = 50
N_TRIALS_SBW = 50

# gNMDA per-condition (generate_all_fresh.py CONDITIONS Stage F):
GNMDA = {
    "control": 1.30,
    "ff_inhibition": 1.30,
    "adaptation": 1.30,
    "nmda": 0.50,
    "nmda_increase": 5.00,
}

CONDS_TBW = ["ff_inhibition", "adaptation", "nmda", "nmda_increase"]  # control already in task #28
CONDS_SBW = ["control", "ff_inhibition", "adaptation", "nmda", "nmda_increase"]


def make_modify_net(condition: str, dt_val: float, n_sub: int):
    """Return modify_net callable for given condition + dt + n_substeps.

    Mirrors generate_all_fresh.py CONDITIONS exactly, with dt and n_substeps
    additionally overridden. _reset_delay_buffers() must be called *after*
    dt change so buffer sizes use the new dt.
    """
    g = GNMDA[condition]

    def _mod(n):
        n.dt = float(dt_val)
        n.n_substeps = int(n_sub)
        n.gNMDA = float(g)
        if condition == "ff_inhibition":
            n.pv_nmda = 0.8
            n.targ_ratio = 0.8
        elif condition == "adaptation":
            n.aM = 0.001
            n.bM = 0.2
            n.cM = -60.0
            n.dM = 0.01
        n._reset_delay_buffers()
    return _mod


def _save_pooled(pooled, path):
    saveable = {k: np.asarray(v) for k, v in pooled.items() if v is not None}
    np.savez(path, **saveable)


# ──────────────────────────────────────────────────────────────────
# SECTION A — TBW
# ──────────────────────────────────────────────────────────────────
def run_tbw_one(cond: str, dt_val: float, n_sub: int) -> float:
    label = f"{cond}_dt{int(dt_val*100):03d}"
    print(f"\n  TBW {cond} @ dt={dt_val}, n_sub={n_sub}")
    cache_path = CACHE / f"tbw_{label}.npz"
    t0 = time.time()
    pooled = run_fusion_across_models(
        MODELS, TBW_OFFSETS, device="cuda",
        fusion_method='temporal_fusion',
        n_trials=N_TRIALS_TBW,
        modify_net=make_modify_net(cond, dt_val, n_sub),
    )
    _save_pooled(pooled, cache_path)
    el = time.time() - t0

    pooled = _load_pooled(cache_path)
    fig, ax, fit = plot_tbw(pooled, reference_fit=None,
                            out_path=str(PLOTDIR / f"TBW_{label}.svg"))
    plt.close(fig)
    xs_f, ys_f = fit["xs"], fit["ys"]
    cross = _find_crossings(xs_f, ys_f, 0.5)
    hw = (cross[-1] - cross[0]) / 2 if len(cross) >= 2 else float("nan")
    print(f"    done in {el:.0f}s -> HW = {hw:.1f} ms")
    torch.cuda.empty_cache()
    return hw


def run_section_a():
    print("=" * 60)
    print("SECTION A — TBW for 4 remaining conditions × 2 dt values")
    print("=" * 60)
    rows = {}
    for cond in CONDS_TBW:
        hw01 = run_tbw_one(cond, 0.1,  100)
        hw005 = run_tbw_one(cond, 0.05, 200)
        rows[cond] = {"dt0.1": hw01, "dt0.05": hw005,
                      "delta": hw005 - hw01,
                      "pct": (hw005 - hw01) / hw01 * 100.0 if hw01 else float("nan")}
    return rows


# ──────────────────────────────────────────────────────────────────
# SECTION B — SBW
# ──────────────────────────────────────────────────────────────────
def run_sbw_one(cond: str, dt_val: float, n_sub: int):
    label = f"{cond}_dt{int(dt_val*100):03d}"
    cache_path = CACHE / f"sbw_{label}.npz"
    print(f"\n  SBW {cond} @ dt={dt_val}, n_sub={n_sub}")
    t0 = time.time()
    pooled = run_spatial_binding_across_models(
        MODELS, separations_deg=SBW_SEPARATIONS, device="cuda",
        method="enhancement", n_trials=N_TRIALS_SBW,
        enhancement_threshold=ENH_THRESHOLD,
        modify_net=make_modify_net(cond, dt_val, n_sub),
    )
    _save_pooled(pooled, cache_path)
    el = time.time() - t0
    print(f"    done in {el:.0f}s -> {cache_path.name}")
    torch.cuda.empty_cache()
    return _load_pooled(cache_path)


def fit_hw_sbw(pooled, out_label: str):
    fig, ax, fit_info = plot_sbw(pooled, reference_fit=None,
                                 out_path=str(PLOTDIR / f"SBW_{out_label}.svg"))
    plt.close(fig)
    xs_f, ys_f = fit_info[0], fit_info[1]
    cross = _find_crossings(xs_f, ys_f, 0.5)
    return (cross[-1] - cross[0]) / 2 if len(cross) >= 2 else float("nan")


def run_section_b():
    print("\n" + "=" * 60)
    print("SECTION B — SBW for 5 conditions × 2 dt values (with floor-subtraction per dt)")
    print("=" * 60)
    raw_01 = {}
    raw_005 = {}
    for cond in CONDS_SBW:
        raw_01[cond]  = run_sbw_one(cond, 0.1,  100)
        raw_005[cond] = run_sbw_one(cond, 0.05, 200)

    print("\n  Subtracting per-dt control floor...")
    sbw_01,  floor_01  = subtract_control_floor(raw_01)
    sbw_005, floor_005 = subtract_control_floor(raw_005)
    print(f"    floor @ dt=0.10: {floor_01:.4f}")
    print(f"    floor @ dt=0.05: {floor_005:.4f}")

    rows = {}
    for cond in CONDS_SBW:
        hw01  = fit_hw_sbw(sbw_01[cond],  f"{cond}_dt010")
        hw005 = fit_hw_sbw(sbw_005[cond], f"{cond}_dt005")
        rows[cond] = {"dt0.1": hw01, "dt0.05": hw005,
                      "delta": hw005 - hw01,
                      "pct": (hw005 - hw01) / hw01 * 100.0 if hw01 else float("nan")}
        print(f"    SBW {cond:20s}  dt=0.1: {hw01:.2f}°  dt=0.05: {hw005:.2f}°  Δ={hw005-hw01:+.2f}°")
    rows["_floor_dt0.1"] = floor_01
    rows["_floor_dt0.05"] = floor_005
    return rows


# ──────────────────────────────────────────────────────────────────
# SECTION C1 — E/I balance
# ──────────────────────────────────────────────────────────────────
def run_ei_one(dt_val: float, n_sub: int) -> dict:
    """Inline run_ei_evoked logic with modify_net + scaled EVOKED_SUBSTEPS."""
    from EI_balance_test import run_ei_probe_separated

    PULSE_FRAMES = 5
    N_FRAMES = 20
    EVOKED_FRAMES = 12.5
    evoked_substeps = int(EVOKED_FRAMES * n_sub)

    mod_fn = make_modify_net("control", dt_val, n_sub)
    rows = []
    for i, p in enumerate(MODELS):
        net = load_msi_model(Path(p), device="cuda")
        mod_fn(net)
        res = run_ei_probe_separated(net, centre_deg=90.0,
                                     pulse_frames=PULSE_FRAMES,
                                     n_frames=N_FRAMES, intensity=1.0)
        traces = res["traces"]
        I_E = traces["I_E_mean"][:evoked_substeps]
        I_I = traces["I_I_mean"][:evoked_substeps]
        exc_mean = float(I_E.mean())
        inh_mean = float(I_I.mean())
        ei = exc_mean / (inh_mean + 1e-12)
        rows.append({"E": exc_mean, "I": inh_mean, "ei": ei})
        del net; torch.cuda.empty_cache()
    arr_e = np.array([r["E"] for r in rows])
    arr_i = np.array([r["I"] for r in rows])
    arr_r = np.array([r["ei"] for r in rows])
    return {
        "E_mean": float(arr_e.mean()), "E_sem": float(arr_e.std(ddof=1) / np.sqrt(len(rows))),
        "I_mean": float(arr_i.mean()), "I_sem": float(arr_i.std(ddof=1) / np.sqrt(len(rows))),
        "EI_mean": float(arr_r.mean()), "EI_sem": float(arr_r.std(ddof=1) / np.sqrt(len(rows))),
    }


def run_section_c1():
    print("\n" + "=" * 60)
    print("SECTION C1 — E/I balance (evoked window, control)")
    print("=" * 60)
    t0 = time.time()
    a01 = run_ei_one(0.1, 100)
    a005 = run_ei_one(0.05, 200)
    el = time.time() - t0
    print(f"  done in {el:.0f}s")
    print(f"  E/I @ dt=0.1:  {a01['EI_mean']:.3f} ± {a01['EI_sem']:.3f}")
    print(f"  E/I @ dt=0.05: {a005['EI_mean']:.3f} ± {a005['EI_sem']:.3f}")
    return {"dt0.1": a01, "dt0.05": a005}


# ──────────────────────────────────────────────────────────────────
# SECTION C2 — Fano (uses main_fano_fast_with_bio path)
# ──────────────────────────────────────────────────────────────────
def run_fano_one(dt_val: float, n_sub: int) -> dict:
    """Inline run_fano_factor_test_bio loop with modify_net pattern.

    Note: fano_factor_test.simulate_batch_trials internally overrides
    n_substeps to fast_substeps=25 (line 60-62) — this is a script-internal
    speedup hack. dt is preserved, n_substeps gets clipped down. We measure
    what the canonical script outputs.
    """
    from fano_factor_test import simulate_batch_trials, fano_factor

    mod_fn = make_modify_net("control", dt_val, n_sub)

    fano_curves, rate_curves = [], []
    for p in MODELS:
        net = load_msi_model(Path(p), device="cuda")
        mod_fn(net)
        data = simulate_batch_trials(net, n_trials=32, baseline_frames=30, stim_frames=30)
        fano_curves.append(fano_factor(data))
        rate_curves.append(data.mean(axis=(1, 2)))
        del net; torch.cuda.empty_cache()

    F = np.vstack(fano_curves)
    meanF = F.mean(0)
    base = float(meanF[:30].mean())   # baseline = first 30 frames
    stim = float(meanF[30:].mean())   # stim     = last 30 frames
    return {"baseline": base, "stim": stim,
            "fano_curve_dim": list(meanF.shape)}


def run_section_c2():
    print("\n" + "=" * 60)
    print("SECTION C2 — Fano factor (baseline vs stim, main_fano_fast_with_bio path)")
    print("=" * 60)
    t0 = time.time()
    a01 = run_fano_one(0.1, 100)
    a005 = run_fano_one(0.05, 200)
    el = time.time() - t0
    print(f"  done in {el:.0f}s")
    print(f"  Fano baseline  dt=0.1: {a01['baseline']:.3f}  dt=0.05: {a005['baseline']:.3f}")
    print(f"  Fano stim      dt=0.1: {a01['stim']:.3f}  dt=0.05: {a005['stim']:.3f}")
    return {"dt0.1": a01, "dt0.05": a005}


# ──────────────────────────────────────────────────────────────────
# SECTION C3 — Cue reliability (R², MAE)
# ──────────────────────────────────────────────────────────────────
def run_cue_one(dt_val: float, n_sub: int) -> dict:
    """Inline reliability_sweep_batched loop with modify_net pattern."""
    from cue_reliability_test import (
        make_dataset, reliability_sweep_batched,
        pool_to_mean_sem, load_msi_model as cue_load,
    )

    mod_fn = make_modify_net("control", dt_val, n_sub)
    xA, xV, meta = make_dataset(n_trials=20, device="cuda")

    all_results = []
    for p in MODELS:
        net = cue_load(p, device="cuda")
        mod_fn(net)
        net._probe = None
        net.allow_inhib_plasticity = False
        res = reliability_sweep_batched(net, xA, xV, meta)
        all_results.append(res)
        del net; torch.cuda.empty_cache()

    pooled = pool_to_mean_sem(all_results)
    import math
    w_pred_all, w_emp_all, abs_errs, sq_errs = [], [], [], []
    for d in pooled:
        var_a, var_v = d["sigma_a"] ** 2, d["sigma_v"] ** 2
        w_pred = 1.0 / var_v / (1.0 / var_a + 1.0 / var_v)
        w_emp = d["mean_w"]
        w_pred_all.append(w_pred); w_emp_all.append(w_emp)
        abs_errs.append(abs(w_emp - w_pred)); sq_errs.append((w_emp - w_pred) ** 2)
    mae = float(np.mean(abs_errs))
    rmse = float(np.sqrt(np.mean(sq_errs)))
    r2 = float(np.corrcoef(w_pred_all, w_emp_all)[0, 1] ** 2)
    return {"R2": r2, "MAE": mae, "RMSE": rmse}


def run_section_c3():
    print("\n" + "=" * 60)
    print("SECTION C3 — Cue reliability (R², MAE)")
    print("=" * 60)
    t0 = time.time()
    a01 = run_cue_one(0.1, 100)
    a005 = run_cue_one(0.05, 200)
    el = time.time() - t0
    print(f"  done in {el:.0f}s")
    print(f"  R²   dt=0.1: {a01['R2']:.3f}   dt=0.05: {a005['R2']:.3f}")
    print(f"  MAE  dt=0.1: {a01['MAE']:.3f}   dt=0.05: {a005['MAE']:.3f}")
    return {"dt0.1": a01, "dt0.05": a005}


# ──────────────────────────────────────────────────────────────────
# SECTION C4 — Precision σ_A, σ_V, σ_B (control block only)
# ──────────────────────────────────────────────────────────────────
def run_prec_one(dt_val: float, n_sub: int) -> dict:
    """Loop compute_hybrid_sensitivity_fast per model with modify_net,
    aggregate σ across models (sigma is what the report needs)."""
    from precision_hist_test import (
        compute_hybrid_sensitivity_fast,
        load_msi_model as prec_load,
    )

    mod_fn = make_modify_net("control", dt_val, n_sub)
    sigmas = []  # rows of (σA, σV, σB)
    sens = []
    for p in MODELS:
        net = prec_load(p, device="cuda")
        mod_fn(net)
        out = compute_hybrid_sensitivity_fast(net, batch_size=1024)
        sigmas.append(out["sigma"])
        sens.append(out["sensitivity"])
        del net; torch.cuda.empty_cache()

    S = np.array(sigmas)  # (n_models, 3)
    SE = np.array(sens)   # (n_models, 4)
    return {
        "sigma_A": float(S[:, 0].mean()), "sigma_V": float(S[:, 1].mean()), "sigma_B": float(S[:, 2].mean()),
        "sigma_A_sem": float(S[:, 0].std(ddof=1) / np.sqrt(len(S))),
        "sigma_V_sem": float(S[:, 1].std(ddof=1) / np.sqrt(len(S))),
        "sigma_B_sem": float(S[:, 2].std(ddof=1) / np.sqrt(len(S))),
        "sA": float(SE[:, 0].mean()), "sV": float(SE[:, 1].mean()), "sB": float(SE[:, 2].mean()),
    }


def run_section_c4():
    print("\n" + "=" * 60)
    print("SECTION C4 — Precision σ_A, σ_V, σ_B (CONTROL block, gNMDA=1.30)")
    print("=" * 60)
    t0 = time.time()
    a01 = run_prec_one(0.1, 100)
    a005 = run_prec_one(0.05, 200)
    el = time.time() - t0
    print(f"  done in {el:.0f}s")
    print(f"  σA  dt=0.1: {a01['sigma_A']:.3f}  dt=0.05: {a005['sigma_A']:.3f}")
    print(f"  σV  dt=0.1: {a01['sigma_V']:.3f}  dt=0.05: {a005['sigma_V']:.3f}")
    print(f"  σB  dt=0.1: {a01['sigma_B']:.3f}  dt=0.05: {a005['sigma_B']:.3f}")
    return {"dt0.1": a01, "dt0.05": a005}


# ──────────────────────────────────────────────────────────────────
# SECTION C5 — Inverse-effectiveness MEI
# ──────────────────────────────────────────────────────────────────
def run_ie_one(dt_val: float, n_sub: int) -> dict:
    """Inline inverse_effectiveness_test.main() integrated-spikes path,
    using our modify_net (sets dt, n_sub, gNMDA=1.30)."""
    LOC_DEG = 90; SIGMA_IN = 5.0; PULSE_LEN = 10; N_FRAMES = 20
    INTENSITIES = np.array([0.05, .1, .2, .4, .8, 1.6], dtype=float)
    DEVICE = "cuda"
    mod_fn = make_modify_net("control", dt_val, n_sub)

    resp_A = np.zeros((len(MODELS), INTENSITIES.size))
    resp_V = np.zeros_like(resp_A)
    resp_AV = np.zeros_like(resp_A)

    def integrated_spikes(net, cond, intensity):
        gauss = lambda: torch.exp(
            -0.5 * ((torch.arange(net.n, device=DEVICE) -
                     (LOC_DEG * (net.n - 1) / (net.space_size - 1))) / SIGMA_IN) ** 2
        ) * intensity
        xA = torch.zeros(N_FRAMES, net.n, device=DEVICE)
        xV = torch.zeros_like(xA)
        if cond in ("A", "B"):
            xA[:PULSE_LEN] = gauss()
        if cond in ("V", "B"):
            xV[:PULSE_LEN] = gauss()
        net.reset_state(batch_size=1)
        pop = 0.0
        for t in range(N_FRAMES):
            *_, sSum = net.update_all_layers_batch(
                xA[t:t + 1], xV[t:t + 1], return_spike_sum=True)
            pop += sSum.sum().item()
        return pop

    for m_i, path in enumerate(MODELS):
        net = load_msi_model(Path(path), device=DEVICE)
        mod_fn(net)
        for j, I in enumerate(INTENSITIES):
            resp_A[m_i, j]  = integrated_spikes(net, "A", I)
            resp_V[m_i, j]  = integrated_spikes(net, "V", I)
            resp_AV[m_i, j] = integrated_spikes(net, "B", I)
        del net; torch.cuda.empty_cache()

    max_uni = np.maximum(resp_A, resp_V)
    mei = (resp_AV - max_uni) / np.maximum(max_uni, 1e-9)
    mei_mean = mei.mean(0)
    return {
        "intensities": INTENSITIES.tolist(),
        "MEI_mean": mei_mean.tolist(),
        "MEI_at_0.05": float(mei_mean[0]),
        "MEI_at_1.6": float(mei_mean[-1]),
        "MEI_grand_mean": float(mei_mean.mean()),
    }


def run_section_c5():
    print("\n" + "=" * 60)
    print("SECTION C5 — Inverse effectiveness (MEI sweep)")
    print("=" * 60)
    t0 = time.time()
    a01 = run_ie_one(0.1, 100)
    a005 = run_ie_one(0.05, 200)
    el = time.time() - t0
    print(f"  done in {el:.0f}s")
    print(f"  MEI@I=0.05  dt=0.1: {a01['MEI_at_0.05']:.3f}  dt=0.05: {a005['MEI_at_0.05']:.3f}")
    print(f"  MEI@I=1.6   dt=0.1: {a01['MEI_at_1.6']:.3f}  dt=0.05: {a005['MEI_at_1.6']:.3f}")
    print(f"  MEI grand mean dt=0.1: {a01['MEI_grand_mean']:.3f}  dt=0.05: {a005['MEI_grand_mean']:.3f}")
    return {"dt0.1": a01, "dt0.05": a005}


# ──────────────────────────────────────────────────────────────────
# SECTION C6 — Response latency (A, V, B)
# ──────────────────────────────────────────────────────────────────
def run_lat_one(dt_val: float, n_sub: int) -> dict:
    """Inline run_latency_test loop, preserving the script's line-685 Izh override.

    Per Lead: "line 685 already applies its own Izhikevich override
    (paper-likely-intentional); leave that alone."
    """
    from response_latency_test import measure_latency, load_msi_model as lat_load
    mod_fn = make_modify_net("control", dt_val, n_sub)

    rows = []
    for i in range(10):
        ckpt_path = BASE / f"msi_model_surr_10_{i:02d}.pt"
        net = lat_load(ckpt_path, device="cuda")
        # apply our dt/n_sub/gNMDA modify_net BEFORE the script's intentional Izh override
        mod_fn(net)
        # script's intentional override (line 685) — apply AFTER our mod_fn so it wins
        net.aM, net.bM, net.cM, net.dM = 0.001, 0.2, -60.0, 0.1
        rows.append({
            "A": measure_latency(net, modality="A"),
            "V": measure_latency(net, modality="V"),
            "B": measure_latency(net, modality="B"),
        })
        del net; torch.cuda.empty_cache()

    A = np.array([r["A"] for r in rows], dtype=float)
    V = np.array([r["V"] for r in rows], dtype=float)
    B = np.array([r["B"] for r in rows], dtype=float)
    return {
        "A_mean": float(np.nanmean(A)), "V_mean": float(np.nanmean(V)), "B_mean": float(np.nanmean(B)),
        "A_n_nan": int(np.isnan(A).sum()),
        "V_n_nan": int(np.isnan(V).sum()),
        "B_n_nan": int(np.isnan(B).sum()),
    }


def run_section_c6():
    print("\n" + "=" * 60)
    print("SECTION C6 — Response latency (A, V, B)")
    print("=" * 60)
    t0 = time.time()
    a01 = run_lat_one(0.1, 100)
    a005 = run_lat_one(0.05, 200)
    el = time.time() - t0
    print(f"  done in {el:.0f}s")
    print(f"  Lat A  dt=0.1: {a01['A_mean']:.1f}  dt=0.05: {a005['A_mean']:.1f}")
    print(f"  Lat V  dt=0.1: {a01['V_mean']:.1f}  dt=0.05: {a005['V_mean']:.1f}")
    print(f"  Lat B  dt=0.1: {a01['B_mean']:.1f}  dt=0.05: {a005['B_mean']:.1f}")
    return {"dt0.1": a01, "dt0.05": a005}


# ──────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_total = time.time()
    print(f"start = {time.strftime('%Y-%m-%dT%H:%M:%S')}")

    results = {}

    # SBW first — heaviest; if budget is tight, this is the priority
    results["TBW"] = run_section_a()
    results["SBW"] = run_section_b()
    results["EI"]  = run_section_c1()
    results["Fano"] = run_section_c2()
    results["Cue"] = run_section_c3()
    results["Precision"] = run_section_c4()
    results["IE"]  = run_section_c5()
    results["Latency"] = run_section_c6()

    # Persist
    with open(CACHE / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    el = time.time() - t_total
    print("\n" + "=" * 60)
    print(f"TOTAL ELAPSED: {el:.0f} s ({el/60:.1f} min)")
    print("=" * 60)
    print(f"Saved summary -> {CACHE / 'summary.json'}")
