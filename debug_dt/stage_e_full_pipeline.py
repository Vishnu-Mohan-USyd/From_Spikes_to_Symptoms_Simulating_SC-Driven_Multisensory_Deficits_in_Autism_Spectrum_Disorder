"""Stage E for task #27: qualitative preservation across all 5 conditions
with the dt-correct fix enabled.

Setup (per team-lead's task #27 brief + Option B confirmation):
  - flag self.dt_correct_nmda = True
  - control gNMDA = 1.30 (×25 the legacy 0.05 — preserves effective NMDA gain)
  - perturbations rescaled by ×25:
      nmda          = 0.50 (was legacy 0.02)
      nmda_increase = 5.00 (was legacy 0.20)
  - ff_inhibition and adaptation perturbations unchanged (no NMDA component)
  - delays-in-ms via _reset_delay_buffers (no effect at dt=0.1, just consistency)

Pipeline: 10 models × 50 trials × 51 TBW offsets + 33 SBW separations at dt=0.1.

Qualitative criteria (team-lead's task #27 brief):
  TBW rank order: control < nmda < nmda_increase ≤ ff_inhibition < adaptation
  SBW rank order: nmda < control < ff_inhibition < adaptation < nmda_increase
  Magnitudes within ~25% of paper.

Reference paper values:
  TBW (ms): control 107, ff_inh 146, adaptation 216, nmda 94, nmda_increase 108
  SBW (°,  floor-subtracted): control 24.3, ff_inh 27.7, adaptation 29.4,
                              nmda 13.1, nmda_increase 30.0

Results saved to debug_dt/stage_e_cache/ (NOT production cache/).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Headless matplotlib for the plot_*_ functions (they need it).
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from TBW_test import run_fusion_across_models, _find_crossings
from SBW_test import run_spatial_binding_across_models
from replot_all_cosmetic import (
    plot_tbw, plot_sbw, subtract_control_floor, _load_pooled,
)

BASE = ROOT / "checkpoint"
MODELS = [BASE / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
CACHE = Path(__file__).parent / "stage_e_cache"
CACHE.mkdir(exist_ok=True)
TMP_PLOT_DIR = CACHE / "_plots"
TMP_PLOT_DIR.mkdir(exist_ok=True)

TBW_OFFSETS = list(range(-50, 51, 2))           # 51 values
SBW_SEPARATIONS = tuple(range(-80, 85, 5))      # 33 values
ENH_THRESHOLD = 10.0
N_TRIALS = 50

GNMDA_CONTROL       = 1.30
GNMDA_NMDA          = 0.50  # 25× legacy 0.02
GNMDA_NMDA_INCREASE = 5.00  # 25× legacy 0.20


def _apply_dt_correct(n, gNMDA=GNMDA_CONTROL):
    """Enable Form 2 + delays-in-ms + set gNMDA. Reset buffers for safety.
    Called per-net by every CONDITIONS modify_net.
    """
    n.dt_correct_nmda = True
    n.gNMDA = float(gNMDA)
    n._reset_delay_buffers()
    return None


def _set_attrs(n, **kw):
    for k, v in kw.items():
        setattr(n, k, v)
    return None


# Perturbation directions per generate_all_fresh.py:CONDITIONS but with
# (a) Form 2 / delays / new control gNMDA always applied, and
# (b) NMDA perturbations rescaled ×25 per team-lead's Option B confirmation.
CONDITIONS = {
    "control":       lambda n: _apply_dt_correct(n),
    "ff_inhibition": lambda n: (_apply_dt_correct(n),
                                _set_attrs(n, pv_nmda=0.8, targ_ratio=0.8)),
    "adaptation":    lambda n: (_apply_dt_correct(n),
                                _set_attrs(n, aM=0.001, bM=0.2, cM=-60.0, dM=0.01)),
    "nmda":          lambda n: _apply_dt_correct(n, gNMDA=GNMDA_NMDA),
    "nmda_increase": lambda n: _apply_dt_correct(n, gNMDA=GNMDA_NMDA_INCREASE),
}

COND_ORDER = ["control", "ff_inhibition", "adaptation", "nmda", "nmda_increase"]

PAPER_TBW = {"control": 107, "ff_inhibition": 146, "adaptation": 216,
             "nmda": 94, "nmda_increase": 108}
PAPER_SBW = {"control": 24.3, "ff_inhibition": 27.7, "adaptation": 29.4,
             "nmda": 13.1, "nmda_increase": 30.0}


def _save_pooled(pooled, path):
    saveable = {k: np.asarray(v) for k, v in pooled.items() if v is not None}
    np.savez(path, **saveable)


def run_tbw():
    print("=" * 60)
    print(f"TBW — dt-correct, 10 models × {N_TRIALS} trials × {len(TBW_OFFSETS)} offsets")
    print("=" * 60)
    tbw_ref_fit = None
    summary = {}
    for cname in COND_ORDER:
        mod_fn = CONDITIONS[cname]
        cache_path = CACHE / f"tbw_{cname}.npz"
        print(f"\n  --- TBW {cname.upper()} ---")
        t0 = time.time()
        pooled = run_fusion_across_models(
            MODELS, TBW_OFFSETS, device="cuda",
            fusion_method='temporal_fusion',
            n_trials=N_TRIALS,
            modify_net=mod_fn,
        )
        _save_pooled(pooled, cache_path)
        dt = time.time() - t0
        print(f"  done in {dt:.0f}s -> {cache_path.name}")
        pooled = _load_pooled(cache_path)
        if cname == "control":
            fig, ax, fit_info = plot_tbw(
                pooled, reference_fit=None,
                out_path=str(TMP_PLOT_DIR / f"TBW_{cname}.svg"))
            tbw_ref_fit = fit_info
        else:
            fig, ax, fit_info = plot_tbw(
                pooled, reference_fit=tbw_ref_fit,
                out_path=str(TMP_PLOT_DIR / f"TBW_{cname}.svg"))
        plt.close(fig)
        xs_f, ys_f = fit_info["xs"], fit_info["ys"]
        cross = _find_crossings(xs_f, ys_f, 0.5)
        hw = (cross[-1] - cross[0]) / 2 if len(cross) >= 2 else float("nan")
        summary[cname] = hw
        print(f"  TBW {cname:20s}: HW={hw:.1f} ms")
        torch.cuda.empty_cache()
    return summary


def run_sbw():
    print("\n" + "=" * 60)
    print(f"SBW — dt-correct, 10 models × {N_TRIALS} trials × {len(SBW_SEPARATIONS)} separations")
    print("=" * 60)
    all_sbw_raw = {}
    for cname in COND_ORDER:
        mod_fn = CONDITIONS[cname]
        cache_path = CACHE / f"sbw_{cname}_t{ENH_THRESHOLD:.0f}.npz"
        print(f"\n  --- SBW {cname.upper()} ---")
        t0 = time.time()
        pooled = run_spatial_binding_across_models(
            MODELS, separations_deg=SBW_SEPARATIONS, device="cuda",
            method="enhancement", n_trials=N_TRIALS,
            enhancement_threshold=ENH_THRESHOLD,
            modify_net=mod_fn,
        )
        _save_pooled(pooled, cache_path)
        dt = time.time() - t0
        print(f"  done in {dt:.0f}s -> {cache_path.name}")
        all_sbw_raw[cname] = _load_pooled(cache_path)
        torch.cuda.empty_cache()
    all_sbw, floor = subtract_control_floor(all_sbw_raw)
    print(f"\n  Control floor subtracted: {floor:.4f}")
    sbw_ref_fit = None
    summary = {}
    for cname in COND_ORDER:
        pooled = all_sbw[cname]
        if cname == "control":
            fig, ax, fit_info = plot_sbw(
                pooled, reference_fit=None,
                out_path=str(TMP_PLOT_DIR / f"SBW_{cname}.svg"))
            sbw_ref_fit = (fit_info[0], fit_info[1])
        else:
            fig, ax, fit_info = plot_sbw(
                pooled, reference_fit=sbw_ref_fit,
                out_path=str(TMP_PLOT_DIR / f"SBW_{cname}.svg"))
        plt.close(fig)
        xs_f, ys_f = fit_info[0], fit_info[1]
        cross = _find_crossings(xs_f, ys_f, 0.5)
        hw = (cross[-1] - cross[0]) / 2 if len(cross) >= 2 else float("nan")
        summary[cname] = hw
        print(f"  SBW {cname:20s}: HW={hw:.2f}°")
    return summary, floor


def _rank_check(summary, expected_order, label):
    """Check rank ordering with ≤ tolerance (use < for strict, ≤ allows ties).
    expected_order is a list of condition names, lowest first.
    """
    vals = [summary[c] for c in expected_order]
    ok = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    return ok


def _pct_off(value, paper):
    if paper == 0 or np.isnan(value):
        return float("nan")
    return 100.0 * abs(value - paper) / abs(paper)


def report(tbw, sbw, floor):
    print("\n" + "=" * 60)
    print("STAGE E REPORT — qualitative preservation across 5 conditions")
    print("=" * 60)

    print("\nTBW (ms):")
    print(f"  {'condition':<20s} {'HW (ms)':>10s} {'paper':>8s} {'%off':>8s}")
    for c in COND_ORDER:
        v = tbw[c]; p = PAPER_TBW[c]
        print(f"  {c:<20s} {v:>10.1f} {p:>8.0f} {_pct_off(v, p):>8.1f}")
    tbw_rank = _rank_check(tbw, ["control", "nmda", "nmda_increase",
                                 "ff_inhibition", "adaptation"], "TBW")
    print(f"  Rank order (control < nmda < nmda_increase ≤ ff_inhibition < adaptation):"
          f" {'PASS' if tbw_rank else 'FAIL'}")
    tbw_pct_max = max(_pct_off(tbw[c], PAPER_TBW[c]) for c in COND_ORDER)
    print(f"  Max %off paper: {tbw_pct_max:.1f}%   (criterion ≤ 25%)")

    print(f"\nSBW (deg, floor-subtracted; floor = {floor:.4f}):")
    print(f"  {'condition':<20s} {'HW (°)':>10s} {'paper':>8s} {'%off':>8s}")
    for c in COND_ORDER:
        v = sbw[c]; p = PAPER_SBW[c]
        print(f"  {c:<20s} {v:>10.2f} {p:>8.1f} {_pct_off(v, p):>8.1f}")
    sbw_rank = _rank_check(sbw, ["nmda", "control", "ff_inhibition",
                                 "adaptation", "nmda_increase"], "SBW")
    print(f"  Rank order (nmda < control < ff_inhibition < adaptation < nmda_increase):"
          f" {'PASS' if sbw_rank else 'FAIL'}")
    sbw_pct_max = max(_pct_off(sbw[c], PAPER_SBW[c]) for c in COND_ORDER)
    print(f"  Max %off paper: {sbw_pct_max:.1f}%   (criterion ≤ 25%)")

    # Save final summary
    np.savez(CACHE / "stage_e_summary.npz",
             tbw=np.array([tbw[c] for c in COND_ORDER]),
             sbw=np.array([sbw[c] for c in COND_ORDER]),
             tbw_rank_pass=tbw_rank, sbw_rank_pass=sbw_rank,
             tbw_pct_max=tbw_pct_max, sbw_pct_max=sbw_pct_max,
             cond_order=np.array(COND_ORDER),
             sbw_floor=floor,
             gNMDA_control=GNMDA_CONTROL,
             gNMDA_nmda=GNMDA_NMDA,
             gNMDA_nmda_increase=GNMDA_NMDA_INCREASE)
    print(f"\nSaved summary -> {CACHE / 'stage_e_summary.npz'}")

    overall_pass = tbw_rank and sbw_rank and tbw_pct_max <= 25 and sbw_pct_max <= 25
    print(f"\nOVERALL: {'STAGE E PASS' if overall_pass else 'STAGE E FAIL — escalate'}")


if __name__ == "__main__":
    t_total = time.time()
    print(f"Stage E start, t0 = {time.strftime('%H:%M:%S')}")
    tbw_summary = run_tbw()
    sbw_summary, floor = run_sbw()
    report(tbw_summary, sbw_summary, floor)
    print(f"\nTotal elapsed: {time.time() - t_total:.0f}s "
          f"({(time.time() - t_total) / 60:.1f} min)")
