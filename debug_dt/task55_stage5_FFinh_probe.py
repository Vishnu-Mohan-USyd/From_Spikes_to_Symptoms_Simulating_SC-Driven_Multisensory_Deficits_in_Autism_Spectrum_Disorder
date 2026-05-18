"""Task #55 Stage 5 — feedforward-inhibition disabling probe.

Stage 4 revealed that weakening AMPA produces NEGATIVE multisensory effects
(B 24-29 ms SLOWER than min(A,V)). The mechanism appears to be feedforward
inhibition (g_FFinh × (I_inA_inh + I_inV_inh) at Training.py:2157) — bimodal
recruits BOTH inhibitory streams, doubling inhibition and suppressing the
spike below unimodal speeds.

Final probe: disable feedforward inhibition entirely (g_FFinh = 0) under the
most-likely-facilitating regime. Test:
  delays      = sym_25 (synchronous A+V arrival)
  pulse_ms    = 100
  gAMPA scale ∈ {1.0, 0.2}    (default and weakened)
  gNMDA       ∈ {0.05, 1.30}
  intensity   ∈ {0.1, 0.3, 1.0}
  g_FFinh     ∈ {ckpt-default ~0.56, 0.0 (disabled)}

This isolates AMPA-NMDA summation from inhibitory suppression. If facilitation
emerges at g_FFinh=0 → architectural recommendation includes "reduce FFinh
during retraining for latency-facilitation experiments". If not → the
substrate is fundamentally limited beyond just FFinh.

ckpt 00 only. ~3 min total.
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
PULSE_MS = 100

DELAY_A_MS = 25.0
DELAY_V_MS = 25.0  # sym_25
INTENSITIES = [0.1, 0.3, 1.0]
GNMDA_LEV = [0.05, 1.30]
GAMPA_LEV = [1.0, 0.2]
GFFINH_LEV = ["default", "off"]


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
    minAV = min(eff_A, eff_V)
    facil = minAV - eff_B
    return {"A_ms": A, "V_ms": V, "B_ms": B,
            "min(A,V)": minAV, "facil_ms": facil}


def fmt(x):
    if isinstance(x, float) and np.isnan(x):
        return "  nan "
    return f"{x:>5}"


def main():
    print("=" * 100)
    print("TASK #55 STAGE 5 — feedforward-inhibition (g_FFinh) disabling probe")
    print(f"   delays = sym_25, pulse = {PULSE_MS} ms, 1 ms resolution, ckpt 00")
    print(f"   gAMPA={GAMPA_LEV}  gNMDA={GNMDA_LEV}  intens={INTENSITIES}  "
          f"FFinh={GFFINH_LEV}")
    print(f"   ckpt-default g_FFinh ≈ 0.56  (loaded from msi_model_surr_10_00.pt)")
    print("=" * 100)

    rows = []
    t0 = time.time()
    for gampa_scale in GAMPA_LEV:
        for gNMDA in GNMDA_LEV:
            for ffinh_setting in GFFINH_LEV:
                for I in INTENSITIES:
                    net = load_msi_model(CKPT, device=DEVICE)
                    apply_canonical_overrides(net)
                    set_delays(net, DELAY_A_MS, DELAY_V_MS)
                    net.n_substeps = N_SUBSTEPS
                    net.gAMPA = float(gampa_scale)
                    net.gNMDA = float(gNMDA)
                    if ffinh_setting == "off":
                        net.g_FFinh = 0.0
                    # else: leave at ckpt-default ~0.56
                    g_ffinh_value = float(net.g_FFinh)
                    t1 = time.time()
                    r = measure(net, I)
                    el = time.time() - t1
                    r["gAMPA"] = gampa_scale
                    r["gNMDA"] = gNMDA
                    r["g_FFinh"] = g_ffinh_value
                    r["FFinh_setting"] = ffinh_setting
                    r["intensity"] = I
                    rows.append(r)
                    if r["facil_ms"] >= 5.0:
                        marker = "  ◀── ≥5 ms FACIL!"
                    elif r["facil_ms"] >= 2.0:
                        marker = "  ◀── facil"
                    else:
                        marker = ""
                    print(
                        f"   gAMPA={gampa_scale:>4.2f} gNMDA={gNMDA:>4.2f} "
                        f"FFinh={ffinh_setting:>7s}(={g_ffinh_value:>4.2f}) "
                        f"I={I:>4.2f}  A={fmt(r['A_ms'])} V={fmt(r['V_ms'])} "
                        f"B={fmt(r['B_ms'])}  facil={r['facil_ms']:>6.1f}{marker}"
                    )
                    del net
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    print(f"\nElapsed: {time.time() - t0:.1f}s")

    print("\n" + "=" * 100)
    print("Per-setting facilitation pivot (cols=intensity, rows=FFinh setting):")
    for gA in GAMPA_LEV:
        for gN in GNMDA_LEV:
            sub = df[(df["gAMPA"] == gA) & (df["gNMDA"] == gN)]
            piv = sub.pivot_table(index="FFinh_setting", columns="intensity",
                                  values="facil_ms", aggfunc="mean")
            print(f"\n--- gAMPA={gA}  gNMDA={gN} ---")
            print(piv.to_string(float_format=lambda x: f"{x:7.2f}",
                                na_rep="  nan  "))

    print("\nTOP-5 facilitation cells:")
    top = df.nlargest(5, "facil_ms")
    print(top[["gAMPA", "gNMDA", "FFinh_setting", "intensity",
               "A_ms", "V_ms", "B_ms", "facil_ms"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.2f}", na_rep="  nan  "))

    out_csv = Path(__file__).parent / "task55_stage5_FFinh_probe.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nRaw saved → {out_csv.name}")


if __name__ == "__main__":
    main()
