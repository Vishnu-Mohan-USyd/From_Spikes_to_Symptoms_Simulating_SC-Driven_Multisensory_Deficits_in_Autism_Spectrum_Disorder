"""Task #55 Stage 7 — drive intensity into the SUB-THRESHOLD A regime.

Team-lead's mechanistic prediction (Cuppini 2010): facilitation operates when
A alone is sub-threshold at the MSI layer (no spike from A's AMPA alone),
but A's NMDA-mediated dendritic depolarisation opens the Mg-gate. V then
arrives and exploits the open NMDA channels to push the cell past threshold,
producing a spike LATER than A's arrival but FAR EARLIER than V alone would
trigger.

Stage 6 found that at intensity=0.01 (already very low), A still spikes at
29 ms — meaning MSI's AMPA from A alone is still suprathreshold. To enter
the genuine sub-threshold-A regime we must push intensity even lower.

Grid:
  intensity ∈ {0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02}
              (down to 50× lower than the lowest in Stage 6)
  pulse_ms  ∈ {5, 10, 20, 30, 100}    (5-30 ms = biological; 100 ms = sanity)
  delays    ∈ {asym_25_40, sym_25, sym_40}
  gNMDA     = 1.30  (forced per team-lead)
  ckpt 00, 1 ms resolution, n_frames=400 (window 400 ms)

Key cases to watch:
  (1) A=nan and V=nan and B=spike  → BIMODAL-RESCUE (highest-magnitude facilitation)
  (2) A=nan, V=spikes, B<V          → V-rescued bimodal
  (3) A=spike(late), V=spike(later), B<<min(A,V)  → mechanistic facilitation

Total: 8 × 5 × 3 = 120 conditions × 3 modalities = 360 measurements. ~25 min.
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
GNMDA = 1.30

PULSE_MS_LEV   = [5, 10, 20, 30, 100]
INTENSITY_LEV  = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02]
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
    A_silent = np.isnan(A)
    V_silent = np.isnan(V)
    B_silent = np.isnan(B)
    return {"A_ms": A, "V_ms": V, "B_ms": B,
            "min(A,V)_eff": minAV, "facil_ms": facil,
            "A_silent": A_silent, "V_silent": V_silent, "B_silent": B_silent}


def fmt(x):
    if isinstance(x, float) and np.isnan(x):
        return "  nan "
    return f"{x:>5}"


def main():
    print("=" * 100)
    print("TASK #55 STAGE 7 — sub-threshold A regime (very-very-low intensity)")
    print(f"   intensity={INTENSITY_LEV}")
    print(f"   pulse_ms={PULSE_MS_LEV}   delays={[c[0] for c in DELAY_CASES]}")
    print(f"   gNMDA={GNMDA} (forced)   1 ms res, ckpt 00")
    print(f"   facil = min(eff_A, eff_V) − eff_B  (POSITIVE = facil; nan → 400)")
    print("=" * 100)

    rows = []
    t0 = time.time()
    for d_label, a_ms, v_ms in DELAY_CASES:
        print(f"\n=== delays={d_label} ({a_ms},{v_ms}) ms ===")
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
                elif r["A_silent"] and r["V_silent"] and not r["B_silent"]:
                    marker = "  ◀── BIMODAL RESCUE!"
                else:
                    marker = ""
                print(
                    f"   pulse={pulse:>3}ms I={I:>7.4f}  "
                    f"A={fmt(r['A_ms'])} V={fmt(r['V_ms'])} B={fmt(r['B_ms'])}  "
                    f"facil={r['facil_ms']:>6.1f}{marker}"
                )
                del net
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    print(f"\nElapsed: {time.time() - t0:.1f}s")

    print("\n" + "=" * 100)
    print("Threshold-crossing summary (lowest intensity at which A still spikes):")
    for d_label, _, _ in DELAY_CASES:
        sub = df[df["delays"] == d_label]
        # find lowest intensity where A is NOT silent
        A_spikers = sub[~sub["A_silent"]]
        if len(A_spikers):
            min_I = A_spikers["intensity"].min()
            print(f"  {d_label}: A still spikes down to intensity ≥ {min_I}")
        else:
            print(f"  {d_label}: A never spikes in tested range")

    print("\nPer-delay facilitation matrix (rows=pulse_ms, cols=intensity):")
    for d_label, _, _ in DELAY_CASES:
        sub = df[df["delays"] == d_label]
        piv = sub.pivot_table(index="pulse_ms", columns="intensity",
                              values="facil_ms", aggfunc="mean")
        print(f"\n=== delays={d_label} ===")
        print(piv.to_string(float_format=lambda x: f"{x:8.1f}", na_rep="   nan  "))

    print("\nTOP-15 facilitation cells:")
    top = df.nlargest(15, "facil_ms")
    print(top[["delays", "pulse_ms", "intensity",
               "A_ms", "V_ms", "B_ms", "facil_ms"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.2f}", na_rep="  nan  "))

    # Highlight bimodal rescue cells
    rescue = df[df["A_silent"] & df["V_silent"] & ~df["B_silent"]]
    if len(rescue):
        print(f"\n◀── {len(rescue)} BIMODAL-RESCUE CELLS FOUND:")
        print(rescue[["delays", "pulse_ms", "intensity", "B_ms"]]
              .to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    else:
        print("\nNo bimodal-rescue cells found (every cell either spikes from A and V"
              " independently, or none spike at all).")

    out_csv = Path(__file__).parent / "task55_stage7_subthreshold_A.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nRaw saved → {out_csv.name}")


if __name__ == "__main__":
    main()
