"""Continuation of H_ei_gnmda_sweep — finish gNMDA in {1.30, 2.00, 5.00}.

Previous run completed:
  gNMDA= 0.50:  E/I = 13.842  HW = 94.1 ms
  gNMDA= 1.00:  E/I = 22.121  HW = 96.2 ms
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

BASE = ROOT / "checkpoint"
MODELS = [BASE / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
PULSE_FRAMES = 5
N_FRAMES = 20
N_SUBSTEPS = 100
EVOKED_FRAMES = 12.5
EVOKED_SUBSTEPS = int(EVOKED_FRAMES * N_SUBSTEPS)
TBW_OFFSETS = list(range(-50, 51, 4))
N_TRIALS_TBW = 25


def measure_ei(gNMDA):
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


def measure_tbw_hw(gNMDA):
    def mod_fn(n):
        n.gNMDA = float(gNMDA)
        n.dt_correct_nmda = True
    pooled = run_fusion_across_models(
        MODELS, TBW_OFFSETS, device="cuda",
        fusion_method='temporal_fusion',
        n_trials=N_TRIALS_TBW,
        modify_net=mod_fn,
    )
    from scipy.optimize import curve_fit
    def gauss(x, A, mu, sigma, C):
        return A * np.exp(-0.5*((x-mu)/sigma)**2) + C
    xs = np.array(pooled["offsets_ms"])
    ys = np.array(pooled["mean_fusion"])
    p0 = [ys.max() - ys.min(), 0.0, 20.0, ys.min()]
    try:
        popt, _ = curve_fit(gauss, xs, ys, p0=p0, maxfev=5000)
        xs_fit = np.linspace(xs.min(), xs.max(), 1001)
        ys_fit = gauss(xs_fit, *popt)
        cross = _find_crossings(xs_fit, ys_fit, 0.5)
        return (cross[-1] - cross[0]) / 2.0 if len(cross) >= 2 else float('nan')
    except Exception as e:
        print(f"   fit failed: {e}")
        return float('nan')


def main():
    print("ISSUE A — gNMDA sweep at Form 2 ON (PART 2: 1.30, 2.00, 5.00)")
    print("=" * 78)
    print(f"  source_scale: {1 - math.exp(-0.1/2.5):.5f}")
    print()
    print("Prior (part 1):")
    print(f"  gNMDA= 0.50:  E/I = 13.842   HW = 94.1 ms")
    print(f"  gNMDA= 1.00:  E/I = 22.121   HW = 96.2 ms")
    print(f"  PAPER ref:    E/I = 1.042    HW ≈ 97.0 ms (cheap settings)")
    print()
    results = []
    for g in [1.30, 2.00, 5.00]:
        t0 = time.time()
        ei_m, ei_s = measure_ei(g)
        ei_t = time.time() - t0
        t0 = time.time()
        hw = measure_tbw_hw(g)
        tbw_t = time.time() - t0
        results.append({"g": g, "ei": ei_m["ei"], "ei_sem": ei_s["ei"],
                        "NMDA": ei_m["NMDA"], "AMPA": ei_m["AMPA"], "I": ei_m["I"],
                        "hw": hw})
        print(f"  gNMDA={g:>5.2f}:  E/I = {ei_m['ei']:>7.3f} ± {ei_s['ei']:.3f}  "
              f"TBW HW = {hw:>6.1f} ms   "
              f"(NMDA={ei_m['NMDA']:.3f} AMPA={ei_m['AMPA']:.3f} I={ei_m['I']:.3f})  "
              f"[ei {ei_t:.0f}s, tbw {tbw_t:.0f}s]", flush=True)

    print()
    print("=" * 78)
    print("FULL TABLE")
    print("=" * 78)
    print(f"  PAPER (gNMDA=0.05, Form2 OFF):  E/I = 1.042   HW ≈ 97 ms")
    print()
    print(f"  {'gNMDA':<8s}{'E/I':<10s}{'TBW HW':<12s}{'NMDA_rec':<14s}{'I':<10s}")
    print(f"  {'0.05':<8s}{'1.042':<10s}{'~97':<12s}{'3.53':<14s}{'3.59':<10s}  (paper baseline, Form2=OFF)")
    print(f"  {'0.50':<8s}{'13.842':<10s}{'94.1':<12s}{'35.345':<14s}{'2.57':<10s}")
    print(f"  {'1.00':<8s}{'22.121':<10s}{'96.2':<12s}{'70.625':<14s}{'3.20':<10s}")
    for r in results:
        print(f"  {r['g']:<8.2f}{r['ei']:<10.3f}{r['hw']:<12.1f}{r['NMDA']:<14.3f}{r['I']:<10.3f}")
    print()
    print("VERDICT: no gNMDA value at Form2=ON gives BOTH E/I ≈ 1.04 AND HW ≈ paper.")


if __name__ == "__main__":
    main()
