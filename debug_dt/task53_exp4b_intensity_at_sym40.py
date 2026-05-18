"""Task #53 — Experiment 4b: intensity sweep at sym_40 delays.

Exp 4 (delay symmetry) showed:
  asym_25_40 (canonical):  B − min(A,V) = +1 ms  (no facilitation)
  sym_25:                  B − min(A,V) =  0 ms
  sym_40:                  B − min(A,V) = -1 ms  (1 ms facilitation, all 10 ckpts)
  sym_32:                  B − min(A,V) = -0.2 ms

So *symmetric delays unlock the substrate*, but only marginally (1 ms vs the
paper's claimed ~19 ms). Researcher #43's primary hypothesis was that the
*inverse-effectiveness* regime (low intensity) is where facilitation peaks.
Under asymmetric delays Exp 1 ruled this out, BUT under symmetric delays the
NMDA-AND-gate may finally have time to integrate before threshold-crossing.

Test: at sym_40 delays (the only regime showing facilitation), sweep
intensity ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4} on 10 ckpts at
1 ms resolution. Look for ΔLat (= min(A,V) − B) > 1 ms in the low-intensity
regime.

If facilitation > 5 ms appears at low intensity → "regime" found, report.
If still ~1 ms at every intensity → substrate fundamentally cannot reach
biological-magnitude facilitation under any tested config.
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

INTENSITIES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4]

DELAY_A_MS = 40.0
DELAY_V_MS = 40.0
N_SUBSTEPS = 10
PULSE_FRAMES = 100
N_FRAMES = 400  # window 400 ms


def measure(net, intensity: float) -> dict:
    A = measure_latency(net, modality="A", intensity=intensity,
                        pulse_frames=PULSE_FRAMES, n_frames=N_FRAMES)
    V = measure_latency(net, modality="V", intensity=intensity,
                        pulse_frames=PULSE_FRAMES, n_frames=N_FRAMES)
    B = measure_latency(net, modality="B", intensity=intensity,
                        pulse_frames=PULSE_FRAMES, n_frames=N_FRAMES)
    uni_mean = float(np.nanmean([A, V]))
    minAV = float(np.nanmin([A, V]))
    facil = minAV - B if not np.isnan(B) else np.nan
    dlat = uni_mean - B if not np.isnan(B) else np.nan
    return {"A_ms": A, "V_ms": V, "B_ms": B,
            "min(A,V)_ms": minAV, "B_vs_minAV_ms": facil,
            "UniMean_ms": uni_mean, "ΔLat_ms": dlat}


def run_one_ckpt(ckpt_path: Path) -> list[dict]:
    rows = []
    for I in INTENSITIES:
        # Re-load fresh so delay buffers reset clean for each intensity
        net = load_msi_model(ckpt_path, device=DEVICE)
        apply_canonical_overrides(net)
        set_delays(net, DELAY_A_MS, DELAY_V_MS)
        net.n_substeps = N_SUBSTEPS
        t0 = time.time()
        r = measure(net, I)
        el = time.time() - t0
        r["intensity"] = I
        r["Model"] = ckpt_path.stem
        r["elapsed_s"] = round(el, 2)
        rows.append(r)
        print(
            f"    I={I:>5.2f}  A={r['A_ms']!s:>7}  V={r['V_ms']!s:>7}  "
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
    print(f"TASK #53 EXP-4b — intensity sweep @ sym_40 delays, 1 ms resolution")
    print(f"   n_models = {args.n_models}    delays = ({DELAY_A_MS}, {DELAY_V_MS}) ms")
    print(f"   intensities = {INTENSITIES}")
    print(f"   measurement: n_substeps=10, pulse=100ms, window=400ms")
    print(f"   canonical Izh override applied; gNMDA from ckpt (default 0.05)")
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
    print("PER-INTENSITY SUMMARY (mean ± std across ckpts, at sym_40)")
    print("=" * 84)
    g = df.groupby("intensity")
    summary = pd.DataFrame({
        "n":             g["A_ms"].count(),
        "A_mean":        g["A_ms"].mean(),
        "V_mean":        g["V_ms"].mean(),
        "B_mean":        g["B_ms"].mean(),
        "minAV_mean":    g["min(A,V)_ms"].mean(),
        "B_minAV_mean":  g["B_vs_minAV_ms"].mean(),
        "B_minAV_std":   g["B_vs_minAV_ms"].std(),
    })
    print(summary.to_string(float_format=lambda x: f"{x:7.2f}", na_rep="  nan  "))

    out_csv = Path(__file__).parent / "task53_exp4b_intensity_at_sym40.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nRaw saved → {out_csv.name}")


if __name__ == "__main__":
    main()
