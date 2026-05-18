"""Task #55 — find latency-facilitation regime via brief-pulse + intensity grid.

Task #53 ruled out 4 of 5 researcher hypotheses but tested only sustained
pulses (100 ms). Biology uses BRIEF flashes/clicks. Brief pulses force the
network into a "sub-threshold accumulation tips the balance" regime instead
of "continuous drive saturates" regime — which is where biological MSI
facilitation lives (Cuppini 2010, Rowland 2007, Frens 1995).

Stage 1 (this wrapper, ckpt 00 only): brief-pulse × intensity grid at
canonical asymmetric delays (25/40), 1 ms measurement resolution, gNMDA from
ckpt default (0.05).

  pulse_ms  ∈ {5, 10, 20, 30, 50, 100}    (5 ms = biological transient;
                                            100 ms = task-53 baseline)
  intensity ∈ {0.1, 0.3, 0.7, 1.0}        (team-lead's priority grid)

Facilitation criterion: B < min(A, V) (NOT race-model averaging).
Biological magnitude target: 10-20 ms.

Resolution note: at n_substeps=10, dt=0.1, frame_ms=1. So pulse_frames =
pulse_ms (1 frame per ms). Window = 400 frames = 400 ms wall time.

Skip frame counting issue: if pulse_ms < first-spike-time, the latency is
still detected via the persistent decaying NMDA/AMPA from the (brief) pulse;
all measurements use the SAME n_frames=400 window to capture late spikes.
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

CKPT_TEMPLATE = "checkpoint/msi_model_surr_10_{:02d}.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PULSE_MS_LEVELS = [5, 10, 20, 30, 50, 100]
INTENSITY_LEVELS = [0.1, 0.3, 0.7, 1.0]

N_SUBSTEPS = 10          # frame_ms = 1
N_FRAMES = 400           # window = 400 ms


def apply_canonical_overrides(net) -> None:
    """Replicate response_latency_test.py:692 — only Izh adapt params."""
    net.aM, net.bM, net.cM, net.dM = 0.001, 0.2, -60.0, 0.1


def measure(net, pulse_ms: int, intensity: float) -> dict:
    """measure_latency at given pulse + intensity at 1 ms resolution."""
    pulse_frames = pulse_ms  # since frame_ms=1
    A = measure_latency(net, modality="A", intensity=intensity,
                        pulse_frames=pulse_frames, n_frames=N_FRAMES)
    V = measure_latency(net, modality="V", intensity=intensity,
                        pulse_frames=pulse_frames, n_frames=N_FRAMES)
    B = measure_latency(net, modality="B", intensity=intensity,
                        pulse_frames=pulse_frames, n_frames=N_FRAMES)
    minAV = float(np.nanmin([A, V])) if not (np.isnan(A) and np.isnan(V)) else np.nan
    facil = minAV - B if (not np.isnan(B) and not np.isnan(minAV)) else np.nan
    return {"A_ms": A, "V_ms": V, "B_ms": B,
            "min(A,V)_ms": minAV, "B_vs_minAV_ms": facil}


def run_one_ckpt(ckpt_path: Path) -> list[dict]:
    rows = []
    for pulse in PULSE_MS_LEVELS:
        for I in INTENSITY_LEVELS:
            net = load_msi_model(ckpt_path, device=DEVICE)
            apply_canonical_overrides(net)
            net.n_substeps = N_SUBSTEPS
            net._reset_delay_buffers()
            t0 = time.time()
            r = measure(net, pulse, I)
            el = time.time() - t0
            r["pulse_ms"] = pulse
            r["intensity"] = I
            r["Model"] = ckpt_path.stem
            r["elapsed_s"] = round(el, 2)
            rows.append(r)
            print(
                f"    pulse={pulse:>3} ms  I={I:>4.2f}  "
                f"A={r['A_ms']!s:>7}  V={r['V_ms']!s:>7}  "
                f"B={r['B_ms']!s:>7}  min={r['min(A,V)_ms']!s:>5}  "
                f"B−min={r['B_vs_minAV_ms']!s:>6}"
            )
            del net
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_models", type=int, default=1)
    args = ap.parse_args()

    print("=" * 86)
    print("TASK #55 STAGE 1 — brief-pulse × intensity @ canonical asymmetric delays")
    print(f"   n_models={args.n_models}  pulse_ms={PULSE_MS_LEVELS}  "
          f"intensity={INTENSITY_LEVELS}")
    print(f"   1 ms resolution (n_substeps=10), window=400 ms, gNMDA=ckpt-default (0.05)")
    print(f"   canonical Izh override: aM=0.001 bM=0.2 cM=-60 dM=0.1")
    print(f"   delays from ckpt: A=25 ms, V=40 ms (canonical asymmetric)")
    print("=" * 86)

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

    # 2-D summary: rows = pulse_ms, cols = intensity, values = B − min(A,V)
    print("\n" + "=" * 86)
    print("FACILITATION MATRIX  (B − min(A,V) in ms, averaged across ckpts)")
    print("  negative = facilitation; 0 = race-model floor; positive = slowdown")
    print("=" * 86)
    facil_pivot = df.pivot_table(
        index="pulse_ms", columns="intensity", values="B_vs_minAV_ms",
        aggfunc="mean",
    )
    print(facil_pivot.to_string(float_format=lambda x: f"{x:7.2f}",
                                na_rep="  nan  "))

    print("\nB_mean matrix (latencies, ms):")
    b_pivot = df.pivot_table(
        index="pulse_ms", columns="intensity", values="B_ms",
        aggfunc="mean",
    )
    print(b_pivot.to_string(float_format=lambda x: f"{x:7.2f}", na_rep="  nan  "))

    print("\nA_mean matrix:")
    a_pivot = df.pivot_table(
        index="pulse_ms", columns="intensity", values="A_ms", aggfunc="mean",
    )
    print(a_pivot.to_string(float_format=lambda x: f"{x:7.2f}", na_rep="  nan  "))

    print("\nV_mean matrix:")
    v_pivot = df.pivot_table(
        index="pulse_ms", columns="intensity", values="V_ms", aggfunc="mean",
    )
    print(v_pivot.to_string(float_format=lambda x: f"{x:7.2f}", na_rep="  nan  "))

    # Highlight any cell with facilitation > 5 ms
    print("\n" + "=" * 86)
    facil_threshold = -5.0
    interesting = df[df["B_vs_minAV_ms"] < facil_threshold]
    if len(interesting):
        print(f"INTERESTING REGIMES (B − min(A,V) ≤ {facil_threshold} ms):")
        print(interesting.to_string(index=False))
    else:
        print(f"NO REGIME with B − min(A,V) ≤ {facil_threshold} ms found.")
        print(f"Best (lowest) facilitation: {df['B_vs_minAV_ms'].min():.2f} ms")
        bestrow = df.loc[df["B_vs_minAV_ms"].idxmin()]
        print(f"  at pulse_ms={bestrow['pulse_ms']}, intensity={bestrow['intensity']}, "
              f"A={bestrow['A_ms']}, V={bestrow['V_ms']}, B={bestrow['B_ms']}")

    out_csv = Path(__file__).parent / (
        f"task55_brief_pulse_sweep_{args.n_models}ckpts.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"\nRaw saved → {out_csv.name}")


if __name__ == "__main__":
    main()
