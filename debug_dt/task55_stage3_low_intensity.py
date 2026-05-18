"""Task #55 Stage 3 — very-low intensity sweep (near-threshold regime).

Stages 1 & 2 found at-most 1 ms facilitation regardless of pulse duration,
delay config, or gNMDA. The remaining hypothesis: at NEAR-THRESHOLD intensities
(below the level where A's AMPA alone reliably triggers a spike), V's
contribution becomes ESSENTIAL — bimodal "rescues" a sub-threshold unimodal.

This is the regime where Rowland 2007 documented biological facilitation:
near the unimodal response threshold, multisensory inputs combine
super-additively. Brief-pulse and pulse-duration are irrelevant for first-spike
timing in this model, so we use sustained 100 ms pulse.

Grid:
  intensity ∈ {0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15}   (very-low → low)
  delays    ∈ {asym_25_40, sym_25, sym_40}
  gNMDA     ∈ {0.05 (ckpt default), 1.30 (boost)}
  pulse_ms  = 100  (sustained)

Total: 7 × 3 × 2 = 42 conditions × 3 modalities × 1 ckpt = 126 measurements.

Key cases:
  • A spikes, V spikes, B spikes faster → ordinary facilitation
  • A=nan, V=nan, B=spikes → BIMODAL RESCUE (real biological facilitation)
  • A=spikes (late), V=nan, B=spikes much earlier → ALSO RESCUE

For ranking, define:
  • If A or V are nan, set their effective latency to N_FRAMES = 400 (window edge)
  • facil = min(eff_A, eff_V) − B   POSITIVE = facilitation
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
from task53_exp4_delay_symmetry import set_delays

CKPT_TEMPLATE = "checkpoint/msi_model_surr_10_{:02d}.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_SUBSTEPS = 10
N_FRAMES = 400
PULSE_MS = 100

DELAY_CASES = [
    ("asym_25_40", 25.0, 40.0),
    ("sym_25",     25.0, 25.0),
    ("sym_40",     40.0, 40.0),
]
INTENSITY_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
GNMDA_LEVELS = [0.05, 1.30]


def apply_canonical_overrides(net) -> None:
    net.aM, net.bM, net.cM, net.dM = 0.001, 0.2, -60.0, 0.1


def measure(net, intensity: float) -> dict:
    A = measure_latency(net, modality="A", intensity=intensity,
                        pulse_frames=PULSE_MS, n_frames=N_FRAMES)
    V = measure_latency(net, modality="V", intensity=intensity,
                        pulse_frames=PULSE_MS, n_frames=N_FRAMES)
    B = measure_latency(net, modality="B", intensity=intensity,
                        pulse_frames=PULSE_MS, n_frames=N_FRAMES)
    eff_A = N_FRAMES if np.isnan(A) else A
    eff_V = N_FRAMES if np.isnan(V) else V
    eff_B = N_FRAMES if np.isnan(B) else B
    minAV_eff = min(eff_A, eff_V)
    facil = minAV_eff - eff_B
    return {
        "A_ms": A, "V_ms": V, "B_ms": B,
        "eff_A": eff_A, "eff_V": eff_V, "eff_B": eff_B,
        "min(A,V)_eff": minAV_eff, "facil_ms": facil,
    }


def fmt(x):
    if isinstance(x, float) and np.isnan(x):
        return "  nan "
    return f"{x:>5}"


def run_one_ckpt(ckpt_path: Path) -> list[dict]:
    rows = []
    for d_label, a_ms, v_ms in DELAY_CASES:
        for gNMDA in GNMDA_LEVELS:
            for I in INTENSITY_LEVELS:
                net = load_msi_model(ckpt_path, device=DEVICE)
                apply_canonical_overrides(net)
                set_delays(net, a_ms, v_ms)
                net.n_substeps = N_SUBSTEPS
                net.gNMDA = float(gNMDA)
                t0 = time.time()
                r = measure(net, I)
                el = time.time() - t0
                r["delays"] = d_label
                r["gNMDA"] = gNMDA
                r["intensity"] = I
                r["Model"] = ckpt_path.stem
                r["elapsed_s"] = round(el, 2)
                rows.append(r)
                if r["facil_ms"] >= 5.0:
                    marker = "  ◀── ≥5 ms FACIL!"
                elif r["facil_ms"] >= 1.0:
                    marker = "  ◀── facil"
                else:
                    marker = ""
                print(
                    f"    {d_label:<10} gNMDA={gNMDA:>5.2f}  I={I:>5.3f}  "
                    f"A={fmt(r['A_ms'])} V={fmt(r['V_ms'])} B={fmt(r['B_ms'])} "
                    f"min(A,V)eff={r['min(A,V)_eff']:>5.0f}  facil={r['facil_ms']:>6.1f}"
                    f"{marker}"
                )
                del net
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_models", type=int, default=1)
    args = ap.parse_args()

    print("=" * 100)
    print("TASK #55 STAGE 3 — very-low intensity, sustained pulse, all delay configs")
    print(f"   n_models={args.n_models}  intensities={INTENSITY_LEVELS}  "
          f"delays={[c[0] for c in DELAY_CASES]}  gNMDA={GNMDA_LEVELS}")
    print(f"   pulse=100 ms, n_substeps=10 (1 ms res), window=400 ms")
    print(f"   facil = min(eff_A, eff_V) − eff_B   (nan→400; POSITIVE = facilitation)")
    print("=" * 100)

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

    # facil pivot per (delays, gNMDA)
    for d_label, _, _ in DELAY_CASES:
        for gNMDA in GNMDA_LEVELS:
            sub = df[(df["delays"] == d_label) & (df["gNMDA"] == gNMDA)]
            piv = sub.set_index("intensity")[["A_ms", "V_ms", "B_ms", "facil_ms"]]
            print(f"\n=== delays={d_label}  gNMDA={gNMDA} ===")
            print(piv.to_string(float_format=lambda x: f"{x:6.1f}", na_rep="  nan "))

    # Top-5 facilitation
    print("\n" + "=" * 100)
    print("TOP-5 facilitation cells:")
    top = df.nlargest(5, "facil_ms")
    print(top[["delays", "gNMDA", "intensity", "A_ms", "V_ms", "B_ms", "facil_ms"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.2f}", na_rep="  nan  "))

    out_csv = Path(__file__).parent / f"task55_stage3_{args.n_models}ckpts.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nRaw saved → {out_csv.name}")


if __name__ == "__main__":
    main()
