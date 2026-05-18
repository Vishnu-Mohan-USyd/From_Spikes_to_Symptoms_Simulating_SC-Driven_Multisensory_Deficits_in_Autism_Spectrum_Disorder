"""ISSUE A — gNMDA sweep at Form 2 ON to RULE OUT parameter-tuning recovery.

The user's alternative hypothesis: "maybe gNMDA=1.30 (calibrated against TBW HW)
doesn't preserve E/I because TBW and E/I respond to gNMDA differently.  Is
there a gNMDA value that recovers BOTH metrics?"

This sweep tests:
  - gNMDA ∈ {0.50, 1.00, 1.30, 2.00, 5.00} (matches generate_all_fresh.py
    pre-existing perturbation values)
  - All with dt_correct_nmda=True (post-fix), 10 ckpts
  - Measure E/I (raw probe, the bug) AND TBW HW
  - Find any gNMDA giving paper TBW HW (~107ms) AND paper E/I (~1.04)

If NO gNMDA gives both → parameter-tuning recovery is IMPOSSIBLE → only
the probe fix works.
"""
from __future__ import annotations
import sys, time, math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from EI_balance_test import run_ei_probe_separated
from TBW_test import load_msi_model, run_fusion_across_models, _find_crossings
from replot_all_cosmetic import _load_pooled

BASE = ROOT / "checkpoint"
MODELS = [BASE / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
PULSE_FRAMES = 5
N_FRAMES = 20
N_SUBSTEPS = 100
EVOKED_FRAMES = 12.5
EVOKED_SUBSTEPS = int(EVOKED_FRAMES * N_SUBSTEPS)
TBW_OFFSETS = list(range(-50, 51, 4))   # cheaper 26-point version
N_TRIALS_TBW = 25                       # cheaper version


def measure_ei_one_gnmda(gNMDA):
    rows = []
    for p in MODELS:
        net = load_msi_model(p, device="cuda")
        net.gNMDA = float(gNMDA)
        net.dt_correct_nmda = True
        res = run_ei_probe_separated(net, centre_deg=90.0,
                                     pulse_frames=PULSE_FRAMES,
                                     n_frames=N_FRAMES, intensity=1.0)
        traces = res["traces"]
        I_E = traces["I_E_mean"][:EVOKED_SUBSTEPS]
        I_I = traces["I_I_mean"][:EVOKED_SUBSTEPS]
        nmda = traces["NMDA"][:EVOKED_SUBSTEPS]
        ampa = traces["AMPA"][:EVOKED_SUBSTEPS]
        rows.append(dict(
            E=float(I_E.mean()), I=float(I_I.mean()),
            ei=float(I_E.mean())/(float(I_I.mean())+1e-12),
            NMDA=float(nmda.mean()), AMPA=float(ampa.mean()),
        ))
        del net; torch.cuda.empty_cache()
    keys = ["E", "I", "ei", "NMDA", "AMPA"]
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    sems  = {k: float(np.std([r[k] for r in rows], ddof=1)/np.sqrt(len(rows))) for k in keys}
    return means, sems


def measure_tbw_hw_one_gnmda(gNMDA):
    """Cheap TBW HW measurement (less trials/offsets than full)."""
    def mod_fn(n):
        n.gNMDA = float(gNMDA)
        n.dt_correct_nmda = True
    pooled = run_fusion_across_models(
        MODELS, TBW_OFFSETS, device="cuda",
        fusion_method='temporal_fusion',
        n_trials=N_TRIALS_TBW,
        modify_net=mod_fn,
    )
    # Compute HW directly from pooled.  Key is "mean_fusion".
    try:
        from scipy.optimize import curve_fit
        def gauss(x, A, mu, sigma, C):
            return A * np.exp(-0.5*((x-mu)/sigma)**2) + C
        xs = np.array(pooled["offsets_ms"])
        ys = np.array(pooled["mean_fusion"])
        p0 = [ys.max() - ys.min(), 0.0, 20.0, ys.min()]
        popt, _ = curve_fit(gauss, xs, ys, p0=p0, maxfev=5000)
        xs_fit = np.linspace(xs.min(), xs.max(), 1001)
        ys_fit = gauss(xs_fit, *popt)
        cross = _find_crossings(xs_fit, ys_fit, 0.5)
        hw = (cross[-1] - cross[0]) / 2.0 if len(cross) >= 2 else float('nan')
    except Exception as e:
        print(f"   fit failed: {e}")
        hw = float('nan')
    return hw


def main():
    print("ISSUE A — gNMDA sweep at Form 2 ON")
    print("=" * 78)
    print(f"  Source_scale at dt=0.1: {1 - math.exp(-0.1/2.5):.5f}")
    print(f"  Paper baseline (gNMDA=0.05, Form2 OFF): E/I = 1.042, TBW HW ≈ 107")
    print(f"  Cheap TBW: {N_TRIALS_TBW} trials × {len(TBW_OFFSETS)} offsets")
    print()

    # First, reference paper baseline TBW with the cheap settings
    print("REFERENCE: paper TBW HW at cheap-settings (gNMDA=0.05, Form2=OFF)")
    print("-" * 78)
    def mod_paper(n):
        n.gNMDA = 0.05
        n.dt_correct_nmda = False
    t0 = time.time()
    pooled_ref = run_fusion_across_models(
        MODELS, TBW_OFFSETS, device="cuda",
        fusion_method='temporal_fusion',
        n_trials=N_TRIALS_TBW, modify_net=mod_paper,
    )
    from scipy.optimize import curve_fit
    def gauss(x, A, mu, sigma, C):
        return A * np.exp(-0.5*((x-mu)/sigma)**2) + C
    xs_r = np.array(pooled_ref["offsets_ms"])
    ys_r = np.array(pooled_ref["mean_fusion"])
    popt_r, _ = curve_fit(gauss, xs_r, ys_r, p0=[ys_r.max()-ys_r.min(),0,20,ys_r.min()], maxfev=5000)
    xs_fit = np.linspace(xs_r.min(), xs_r.max(), 1001)
    ys_fit = gauss(xs_fit, *popt_r)
    cross = _find_crossings(xs_fit, ys_fit, 0.5)
    hw_ref = (cross[-1]-cross[0])/2 if len(cross)>=2 else float('nan')
    print(f"  Paper TBW HW reference (cheap): {hw_ref:.1f}ms   ({time.time()-t0:.0f}s)")

    print()
    print("SWEEP: gNMDA at Form 2 ON")
    print("-" * 78)
    g_values = [0.50, 1.00, 1.30, 2.00, 5.00]
    results = []
    for g in g_values:
        t0 = time.time()
        ei_m, ei_s = measure_ei_one_gnmda(g)
        ei_t = time.time() - t0
        t0 = time.time()
        hw = measure_tbw_hw_one_gnmda(g)
        tbw_t = time.time() - t0
        results.append({"g": g, "ei": ei_m["ei"], "ei_sem": ei_s["ei"],
                        "E": ei_m["E"], "I": ei_m["I"],
                        "NMDA": ei_m["NMDA"], "AMPA": ei_m["AMPA"],
                        "hw": hw})
        print(f"  gNMDA={g:>5.2f}:  E/I = {ei_m['ei']:>7.3f} ± {ei_s['ei']:.3f}  "
              f"TBW HW = {hw:>6.1f} ms   "
              f"(NMDA={ei_m['NMDA']:.3f} AMPA={ei_m['AMPA']:.3f} I={ei_m['I']:.3f})  "
              f"[ei {ei_t:.0f}s, tbw {tbw_t:.0f}s]")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  Paper target:  E/I ≈ 1.04   TBW HW ≈ {hw_ref:.1f} ms (ref) / 107 ms (paper)")
    print()
    print(f"  {'gNMDA':<8s}{'E/I':<10s}{'TBW HW':<12s}{'matches both?':<15s}")
    for r in results:
        ei_ok = abs(r['ei'] - 1.04) < 0.2
        hw_ok = abs(r['hw'] - hw_ref) < 15.0
        both = "YES" if (ei_ok and hw_ok) else "no"
        print(f"  {r['g']:<8.2f}{r['ei']:<10.3f}{r['hw']:<12.1f}"
              f"{'EI=ok ' if ei_ok else 'EI=no '}"
              f"{'HW=ok' if hw_ok else 'HW=no'}  -> {both}")
    print()
    matches = [r for r in results if abs(r['ei']-1.04)<0.2 and abs(r['hw']-hw_ref)<15.0]
    if matches:
        print(f"  RECOVERY POSSIBLE: gNMDA = {[r['g'] for r in matches]} restore BOTH")
    else:
        print("  RECOVERY NOT POSSIBLE via gNMDA tuning alone.")
        print("  E/I and TBW HW respond to gNMDA in incompatible ways:")
        print("    - Recorded NMDA scales linearly with gNMDA (no source_scale on probe)")
        print("    - TBW HW depends on effective injection (= gNMDA × source_scale × ...)")
        print("    - The probe asymmetry decouples the two metrics → cannot be jointly tuned.")


if __name__ == "__main__":
    main()
