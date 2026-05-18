"""Task #55 Stage 2 — brief-pulse × intensity at SYMMETRIC delays.

Stage 1 found no facilitation at canonical asymmetric (25/40) delays with any
brief pulse + intensity combo. Stage 2 explores team-lead's priorities 3 & 4:

  Priority 3: sym_25 delays + brief pulses (pulse_frames ∈ {2, 3} → 20, 30 ms)
  Priority 4: gNMDA boost to 1.30 (in case ckpt-default 0.05 is too low)

Combined grid:
  delays   ∈ {sym_25, sym_40}
  pulse_ms ∈ {5, 10, 20, 30, 50}
  intens   ∈ {0.1, 0.3, 0.7, 1.0}
  gNMDA    ∈ {0.05 (ckpt default), 1.30 (wrapper-canonical boost)}

Total: 2 × 5 × 4 × 2 = 80 conditions × 3 modalities × 1 ckpt = 240 measurements.

If ANY cell shows B − min(A,V) ≤ -5 ms (5+ ms facilitation), report as
candidate regime. If best is ~1 ms, the substrate ceiling is confirmed.

Reference (Stage 1):
  asym_25_40 + any pulse + any intens → B − min(A,V) ∈ {-1, 0, +1} ms
  i.e., race-model floor or slight slowdown. Never facilitation.

Convention: facilitation = min(A,V) − B (POSITIVE = B faster than min(A,V)).
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

DELAY_CASES = [
    ("sym_25", 25.0, 25.0),
    ("sym_40", 40.0, 40.0),
]
PULSE_MS_LEVELS = [5, 10, 20, 30, 50]
INTENSITY_LEVELS = [0.1, 0.3, 0.7, 1.0]
GNMDA_LEVELS = [0.05, 1.30]


def apply_canonical_overrides(net) -> None:
    net.aM, net.bM, net.cM, net.dM = 0.001, 0.2, -60.0, 0.1


def measure(net, pulse_ms: int, intensity: float) -> dict:
    pulse_frames = pulse_ms
    A = measure_latency(net, modality="A", intensity=intensity,
                        pulse_frames=pulse_frames, n_frames=N_FRAMES)
    V = measure_latency(net, modality="V", intensity=intensity,
                        pulse_frames=pulse_frames, n_frames=N_FRAMES)
    B = measure_latency(net, modality="B", intensity=intensity,
                        pulse_frames=pulse_frames, n_frames=N_FRAMES)
    if np.isnan(A) and np.isnan(V):
        return {"A_ms": A, "V_ms": V, "B_ms": B,
                "min(A,V)_ms": np.nan, "facil_ms": np.nan}
    minAV = float(np.nanmin([A, V]))
    facil = minAV - B if not np.isnan(B) else np.nan
    return {"A_ms": A, "V_ms": V, "B_ms": B,
            "min(A,V)_ms": minAV, "facil_ms": facil}


def run_one_ckpt(ckpt_path: Path) -> list[dict]:
    rows = []
    for d_label, a_ms, v_ms in DELAY_CASES:
        for gNMDA in GNMDA_LEVELS:
            for pulse in PULSE_MS_LEVELS:
                for I in INTENSITY_LEVELS:
                    net = load_msi_model(ckpt_path, device=DEVICE)
                    apply_canonical_overrides(net)
                    set_delays(net, a_ms, v_ms)
                    net.n_substeps = N_SUBSTEPS
                    net.gNMDA = float(gNMDA)
                    t0 = time.time()
                    r = measure(net, pulse, I)
                    el = time.time() - t0
                    r["delays"] = d_label
                    r["gNMDA"] = gNMDA
                    r["pulse_ms"] = pulse
                    r["intensity"] = I
                    r["Model"] = ckpt_path.stem
                    r["elapsed_s"] = round(el, 2)
                    rows.append(r)
                    if r["facil_ms"] is not None and \
                            not np.isnan(r["facil_ms"]) and r["facil_ms"] > 0:
                        marker = "  ◀── FACIL"
                    else:
                        marker = ""
                    print(
                        f"    {d_label:<6}  gNMDA={gNMDA:>5.2f}  "
                        f"pulse={pulse:>3}ms  I={I:>4.2f}  "
                        f"A={r['A_ms']!s:>6}  V={r['V_ms']!s:>6}  "
                        f"B={r['B_ms']!s:>6}  facil={r['facil_ms']!s:>6}{marker}"
                    )
                    del net
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_models", type=int, default=1)
    args = ap.parse_args()

    print("=" * 96)
    print("TASK #55 STAGE 2 — brief-pulse × intensity at SYMMETRIC delays + gNMDA boost")
    print(f"   n_models={args.n_models}  "
          f"delays={[c[0] for c in DELAY_CASES]}  pulses={PULSE_MS_LEVELS}  "
          f"intensities={INTENSITY_LEVELS}  gNMDA={GNMDA_LEVELS}")
    print(f"   1 ms resolution (n_substeps=10), canonical Izh override applied")
    print(f"   facil = min(A,V) − B  (POSITIVE = B faster = facilitation)")
    print("=" * 96)

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

    # Per-delay × gNMDA × pulse pivot of facil
    for d_label, _, _ in DELAY_CASES:
        for gNMDA in GNMDA_LEVELS:
            sub = df[(df["delays"] == d_label) & (df["gNMDA"] == gNMDA)]
            piv = sub.pivot_table(
                index="pulse_ms", columns="intensity", values="facil_ms",
                aggfunc="mean",
            )
            print(f"\n=== facil matrix  delays={d_label}  gNMDA={gNMDA} ===")
            print(piv.to_string(float_format=lambda x: f"{x:7.2f}",
                                na_rep="  nan  "))

    # Best-facilitation cell
    valid = df[~df["facil_ms"].isna()]
    if len(valid):
        best_idx = valid["facil_ms"].idxmax()
        best = df.loc[best_idx]
        print(f"\nBEST facilitation cell: facil = {best['facil_ms']:.2f} ms")
        print(f"  delays={best['delays']}  gNMDA={best['gNMDA']}  "
              f"pulse_ms={best['pulse_ms']}  intensity={best['intensity']}")
        print(f"  A={best['A_ms']}  V={best['V_ms']}  B={best['B_ms']}")
        if best["facil_ms"] >= 5.0:
            print(f"\n◀── BIOLOGICAL-MAGNITUDE candidate (≥5 ms). Expand to 10 ckpts.")

    out_csv = Path(__file__).parent / f"task55_stage2_{args.n_models}ckpts.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nRaw saved → {out_csv.name}")


if __name__ == "__main__":
    main()
