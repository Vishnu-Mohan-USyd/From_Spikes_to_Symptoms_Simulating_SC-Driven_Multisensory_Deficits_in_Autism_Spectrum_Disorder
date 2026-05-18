"""ISSUE A — 2×2 grid: gNMDA × dt_correct_nmda → E/I values.

The team-lead's framing:
  Paper / pre-fix gNMDA=0.05:    E/I = 1.04
  Post-recalibration gNMDA=1.30: E/I = 1.72

Production run_ei_balance.py default behavior already gives E/I ≈ 1.72
with gNMDA=0.05 from checkpoint (verified).  So the variable that flips
1.04 ↔ 1.72 must be dt_correct_nmda, NOT gNMDA alone.

This script measures all four combinations to find:
  - which combination gives the paper's E/I = 1.04
  - which gives the post-fix 1.72
  - the component breakdown to see what shifted

If gNMDA=0.05 + dt_correct=False gives ≈1.04, and gNMDA=0.05 +
dt_correct=True (production default) gives ≈1.72, then the E/I "shift"
is caused entirely by the new dt_correct_nmda default — NOT by recalibration.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from EI_balance_test import run_ei_probe_separated
from TBW_test import load_msi_model

BASE = ROOT / "checkpoint"
MODELS = [BASE / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
PULSE_FRAMES = 5
N_FRAMES = 20
N_SUBSTEPS = 100
EVOKED_FRAMES = 12.5
EVOKED_SUBSTEPS = int(EVOKED_FRAMES * N_SUBSTEPS)


def run_cell(*, gNMDA, dt_correct, label):
    rows = []
    t0 = time.time()
    for p in MODELS:
        net = load_msi_model(p, device="cuda")
        if gNMDA is not None:
            net.gNMDA = float(gNMDA)
        net.dt_correct_nmda = bool(dt_correct)
        res = run_ei_probe_separated(net, centre_deg=90.0,
                                     pulse_frames=PULSE_FRAMES,
                                     n_frames=N_FRAMES, intensity=1.0)
        traces = res["traces"]
        I_E = traces["I_E_mean"][:EVOKED_SUBSTEPS]
        I_I = traces["I_I_mean"][:EVOKED_SUBSTEPS]
        ampa = traces["AMPA"][:EVOKED_SUBSTEPS]
        nmda = traces["NMDA"][:EVOKED_SUBSTEPS]
        ff = traces["FFInh"][:EVOKED_SUBSTEPS]
        recur = traces["RecurInh"][:EVOKED_SUBSTEPS]
        lat = traces["LatInh"][:EVOKED_SUBSTEPS]
        rows.append({
            "E": float(I_E.mean()),
            "I": float(I_I.mean()),
            "ei": float(I_E.mean()) / (float(I_I.mean()) + 1e-12),
            "AMPA": float(ampa.mean()),
            "NMDA": float(nmda.mean()),
            "FF": float(ff.mean()),
            "Rec": float(recur.mean()),
            "Lat": float(lat.mean()),
        })
        del net; torch.cuda.empty_cache()
    el = time.time() - t0
    keys = ["E", "I", "ei", "AMPA", "NMDA", "FF", "Rec", "Lat"]
    means = {k: np.mean([r[k] for r in rows]) for k in keys}
    sems = {k: np.std([r[k] for r in rows], ddof=1)/np.sqrt(len(rows)) for k in keys}
    print(f"{label:<55s}  E/I={means['ei']:.3f}±{sems['ei']:.3f}  "
          f"E={means['E']:.3f}  I={means['I']:.3f}  el={el:.0f}s")
    print(f"    breakdown: AMPA={means['AMPA']:.4f}  NMDA={means['NMDA']:.4f}  "
          f"FF={means['FF']:.4f}  Rec={means['Rec']:.4f}  Lat={means['Lat']:.4f}")
    return means, sems


def main():
    print("ISSUE A — E/I 2×2 grid (gNMDA × dt_correct_nmda)")
    print("=" * 75)
    print()

    r1, _ = run_cell(gNMDA=0.05, dt_correct=False,
                     label="A: gNMDA=0.05, dt_correct=False (PAPER pre-fix)")
    r2, _ = run_cell(gNMDA=0.05, dt_correct=True,
                     label="B: gNMDA=0.05, dt_correct=True  (production default)")
    r3, _ = run_cell(gNMDA=1.30, dt_correct=False,
                     label="C: gNMDA=1.30, dt_correct=False (over-driven)")
    r4, _ = run_cell(gNMDA=1.30, dt_correct=True,
                     label="D: gNMDA=1.30, dt_correct=True  (post-recalibration)")

    print()
    print("=" * 75)
    print("VERDICT")
    print("=" * 75)
    print(f"  A (paper, gNMDA=0.05, Form2=OFF):     E/I = {r1['ei']:.3f}")
    print(f"  B (broken, gNMDA=0.05, Form2=ON):     E/I = {r2['ei']:.3f}")
    print(f"  C (overdrive, gNMDA=1.30, Form2=OFF): E/I = {r3['ei']:.3f}")
    print(f"  D (post-fix, gNMDA=1.30, Form2=ON):   E/I = {r4['ei']:.3f}")
    print()
    print(f"  Δ from A→D (paper→post-fix):  {r4['ei'] - r1['ei']:+.3f}")
    print(f"  Δ from A→B (Form2 alone):     {r2['ei'] - r1['ei']:+.3f}")
    print(f"  Δ from A→C (gNMDA alone):     {r3['ei'] - r1['ei']:+.3f}")
    print()
    print("  Component shifts A → D:")
    for k in ["AMPA", "NMDA", "FF", "Rec", "Lat"]:
        ratio = r4[k] / (r1[k] + 1e-12)
        print(f"    {k:6s}: {r1[k]:.4f} → {r4[k]:.4f}  ratio = {ratio:.2f}×")


if __name__ == "__main__":
    main()
