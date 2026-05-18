"""Task #53 — Experiment 3: gNMDA modulation at sym_40 + I=1.0.

Researcher #43 attributed multisensory latency facilitation to the NMDA Mg-gate
(mg_A + mg_V) at Training.py:2094-2098. At canonical gNMDA=0.05 (ckpt default)
the substrate produced only 1 ms of facilitation under sym_40 delays (Exp 4 +
Exp 4b). This experiment varies gNMDA to test whether:

  gNMDA = 0.00  →  facilitation vanishes (NMDA is responsible)
  gNMDA = 0.05  →  baseline 1 ms (already measured)
  gNMDA = 0.50  →  facilitation enlarges (NMDA dose-response)
  gNMDA = 1.30  →  facilitation peaks at biological magnitude? (wrapper default)
  gNMDA = 5.00  →  saturation / instability?

If NMDA boost can enlarge facilitation to ~10-20 ms (biological magnitude),
then "regime = sym_40 + high gNMDA + I=1.0" is the answer.
If facilitation stays ~1 ms regardless of gNMDA, the substrate is structurally
incapable of paper-magnitude facilitation.

Test config:
  delays  = sym_40 (40 ms both — the only regime showing any facilitation)
  intens  = 1.0    (the only intensity showing any facilitation, per Exp 4b)
  gNMDA   = {0.00, 0.05, 0.50, 1.30, 5.00}
  measurement: 1 ms resolution (n_substeps=10)
  10 ckpts
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from response_latency_test import load_msi_model, measure_latency
from task53_exp4_delay_symmetry import apply_canonical_overrides, set_delays

CKPT_TEMPLATE = "checkpoint/msi_model_surr_10_{:02d}.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GNMDA_LEVELS = [0.0, 0.05, 0.50, 1.30, 5.00]
DELAY_A_MS = 40.0
DELAY_V_MS = 40.0
INTENSITY = 1.0
N_SUBSTEPS = 10
PULSE_FRAMES = 100
N_FRAMES = 400


def measure(net) -> dict:
    A = measure_latency(net, modality="A", intensity=INTENSITY,
                        pulse_frames=PULSE_FRAMES, n_frames=N_FRAMES)
    V = measure_latency(net, modality="V", intensity=INTENSITY,
                        pulse_frames=PULSE_FRAMES, n_frames=N_FRAMES)
    B = measure_latency(net, modality="B", intensity=INTENSITY,
                        pulse_frames=PULSE_FRAMES, n_frames=N_FRAMES)
    minAV = float(np.nanmin([A, V]))
    facil = minAV - B if not np.isnan(B) else np.nan
    return {"A_ms": A, "V_ms": V, "B_ms": B,
            "min(A,V)_ms": minAV, "B_vs_minAV_ms": facil}


def run_one_ckpt(ckpt_path: Path) -> list[dict]:
    rows = []
    for g in GNMDA_LEVELS:
        net = load_msi_model(ckpt_path, device=DEVICE)
        apply_canonical_overrides(net)
        set_delays(net, DELAY_A_MS, DELAY_V_MS)
        net.n_substeps = N_SUBSTEPS
        net.gNMDA = float(g)
        t0 = time.time()
        r = measure(net)
        el = time.time() - t0
        r["gNMDA"] = g
        r["Model"] = ckpt_path.stem
        r["elapsed_s"] = round(el, 2)
        rows.append(r)
        print(
            f"    gNMDA={g:>5.2f}  A={r['A_ms']!s:>7}  V={r['V_ms']!s:>7}  "
            f"B={r['B_ms']!s:>7}  min={r['min(A,V)_ms']!s:>5}  "
            f"B−min={r['B_vs_minAV_ms']!s:>6}  el={el:.1f}s"
        )
        del net
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_models", type=int, default=10)
    args = ap.parse_args()

    print("=" * 84)
    print(f"TASK #53 EXP-3 — gNMDA modulation @ sym_40, I=1.0, 1 ms resolution")
    print(f"   n_models={args.n_models}  delays=({DELAY_A_MS},{DELAY_V_MS})  "
          f"intensity={INTENSITY}")
    print(f"   gNMDA levels: {GNMDA_LEVELS}")
    print("=" * 84)

    all_rows: list[dict] = []
    t0 = time.time()
    for i in range(args.n_models):
        ckpt = ROOT / CKPT_TEMPLATE.format(i)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        print(f"\n[{i:02d}] {ckpt.name}")
        all_rows.extend(run_one_ckpt(ckpt))
    df = pd.DataFrame(all_rows)
    print(f"\nElapsed: {time.time() - t0:.1f}s")

    print("\n" + "=" * 84)
    print("PER-gNMDA SUMMARY (mean ± std across ckpts, at sym_40 + I=1.0)")
    print("=" * 84)
    g = df.groupby("gNMDA")
    summary = pd.DataFrame({
        "n":            g["A_ms"].count(),
        "A_mean":       g["A_ms"].mean(),
        "V_mean":       g["V_ms"].mean(),
        "B_mean":       g["B_ms"].mean(),
        "minAV_mean":   g["min(A,V)_ms"].mean(),
        "B_minAV_mean": g["B_vs_minAV_ms"].mean(),
        "B_minAV_std":  g["B_vs_minAV_ms"].std(),
    })
    print(summary.to_string(float_format=lambda x: f"{x:7.2f}", na_rep="  nan  "))

    out_csv = Path(__file__).parent / "task53_exp3_gNMDA_at_sym40_I1.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nRaw saved → {out_csv.name}")


if __name__ == "__main__":
    main()
