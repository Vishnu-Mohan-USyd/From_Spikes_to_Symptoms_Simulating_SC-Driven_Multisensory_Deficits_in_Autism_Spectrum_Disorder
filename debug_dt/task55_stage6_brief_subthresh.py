"""Task #55 Stage 6 — close the gap: brief pulse × very-low intensity.

Team-lead noted that prior stages tested:
  Stage 1, 2: brief pulse × intensity ∈ {0.1, 0.3, 0.7, 1.0}  — too high; A always spikes
  Stage 3:    intensity ∈ {0.01, …, 0.15} × pulse=100 ms      — sustained, not brief

The MISSING combination is *brief pulse + very-low intensity*, where the
team-lead's mechanistic prediction lives:

  At pulse_ms=10 + intensity=0.02 (or similar), A's brief AMPA transient
  may decay before threshold-crossing, leaving sub-threshold depolarisation
  that opens NMDA Mg-gate via v_dend_A. NMDA τ=40 ms persists past A's
  AMPA decay. When V arrives at +15 ms (asymmetric) or +0 ms (symmetric),
  V's AMPA finds the NMDA channels OPEN and adds enough current to tip
  past threshold. B fires earlier than A or V alone.

Grid:
  pulse_ms    ∈ {5, 10, 20, 30}           — brief biological pulse range
  intensity   ∈ {0.01, 0.02, 0.03, 0.05, 0.07, 0.10}  — sub- to near-threshold
  delays      ∈ {asym_25_40, sym_25, sym_40}
  gNMDA       = 1.30  (forced per team-lead — production default 0.05 is debugger-flagged)
  1 ms resolution, canonical Izh override
  ckpt 00 only

Total: 4 × 6 × 3 = 72 conditions × 3 modalities = 216 measurements.
If any cell shows facilitation ≥ 5 ms → expand to 10 ckpts.
If max stays ≤ 1 ms → confirm NO-FACILITATION exhaustively.

Reporting convention:
  facil_ms = min(A, V) − B   (POSITIVE = B faster = facilitation)
  If A or V are nan, treat as N_FRAMES=400 (i.e., "didn't spike in window").
"""
from __future__ import annotations

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

CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_SUBSTEPS = 10
N_FRAMES = 400
GNMDA = 1.30  # forced per team-lead

PULSE_MS_LEV   = [5, 10, 20, 30]
INTENSITY_LEV  = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
DELAY_CASES    = [
    ("asym_25_40", 25.0, 40.0),
    ("sym_25",     25.0, 25.0),
    ("sym_40",     40.0, 40.0),
]


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
    eff_A = N_FRAMES if np.isnan(A) else A
    eff_V = N_FRAMES if np.isnan(V) else V
    eff_B = N_FRAMES if np.isnan(B) else B
    minAV = min(eff_A, eff_V)
    facil = minAV - eff_B
    return {"A_ms": A, "V_ms": V, "B_ms": B,
            "min(A,V)_eff": minAV, "facil_ms": facil}


def fmt(x):
    if isinstance(x, float) and np.isnan(x):
        return "  nan "
    return f"{x:>5}"


def main():
    print("=" * 100)
    print("TASK #55 STAGE 6 — brief pulse × very-low intensity × delays @ gNMDA=1.30")
    print(f"   pulse_ms={PULSE_MS_LEV}   intensity={INTENSITY_LEV}")
    print(f"   delays={[c[0] for c in DELAY_CASES]}   gNMDA={GNMDA} (forced)")
    print(f"   1 ms resolution (n_substeps=10), window=400 ms, ckpt 00")
    print(f"   facil = min(eff_A, eff_V) − eff_B  (POSITIVE = facilitation)")
    print("=" * 100)

    rows = []
    t0 = time.time()
    for d_label, a_ms, v_ms in DELAY_CASES:
        print(f"\n--- delays = {d_label} ({a_ms},{v_ms}) ms ---")
        for pulse in PULSE_MS_LEV:
            for I in INTENSITY_LEV:
                net = load_msi_model(CKPT, device=DEVICE)
                apply_canonical_overrides(net)
                set_delays(net, a_ms, v_ms)
                net.n_substeps = N_SUBSTEPS
                net.gNMDA = float(GNMDA)
                t1 = time.time()
                r = measure(net, pulse, I)
                el = time.time() - t1
                r["delays"] = d_label
                r["pulse_ms"] = pulse
                r["intensity"] = I
                rows.append(r)
                if r["facil_ms"] >= 5.0:
                    marker = "  ◀── ≥5 ms FACIL!"
                elif r["facil_ms"] >= 2.0:
                    marker = "  ◀── facil"
                else:
                    marker = ""
                print(
                    f"   pulse={pulse:>3}ms I={I:>4.2f}  "
                    f"A={fmt(r['A_ms'])} V={fmt(r['V_ms'])} B={fmt(r['B_ms'])}  "
                    f"facil={r['facil_ms']:>6.1f}{marker}"
                )
                del net
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    print(f"\nElapsed: {time.time() - t0:.1f}s")

    print("\n" + "=" * 100)
    print("Per-delay facilitation matrix (rows=pulse_ms, cols=intensity):")
    for d_label, _, _ in DELAY_CASES:
        sub = df[df["delays"] == d_label]
        piv = sub.pivot_table(index="pulse_ms", columns="intensity",
                              values="facil_ms", aggfunc="mean")
        print(f"\n=== delays={d_label} (gNMDA={GNMDA}) ===")
        print(piv.to_string(float_format=lambda x: f"{x:7.2f}", na_rep="  nan  "))

    print("\nTOP-10 facilitation cells:")
    top = df.nlargest(10, "facil_ms")
    print(top[["delays", "pulse_ms", "intensity",
               "A_ms", "V_ms", "B_ms", "facil_ms"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.2f}", na_rep="  nan  "))

    out_csv = Path(__file__).parent / "task55_stage6_brief_subthresh.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nRaw saved → {out_csv.name}")


if __name__ == "__main__":
    main()
