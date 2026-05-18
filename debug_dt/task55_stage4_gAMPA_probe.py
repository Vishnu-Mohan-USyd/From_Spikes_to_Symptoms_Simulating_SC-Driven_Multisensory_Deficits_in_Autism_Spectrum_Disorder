"""Task #55 Stage 4 — gAMPA scaling probe (architectural-capacity test).

Stages 1-3 (sustained pulse intensity sweep, brief-pulse + sym delays + gNMDA,
near-threshold low-intensity probe) all hit a 1 ms facilitation ceiling. The
mechanistic reason: AMPA gain × stimulus × Izhikevich threshold combination
causes the cell to cross threshold within ~2 ms of input arrival, regardless
of intensity or delay config. There is no opportunity for sub-threshold
integration to participate in first-spike timing.

This stage tests the SUBSTRATE'S CAPACITY by directly weakening AMPA:
multiply `net.gAMPA` by a scale factor. If the substrate CAN produce >5 ms
multisensory latency facilitation under sufficiently weak AMPA, then the
architectural recommendation is: "retrain with reduced gAMPA (or equivalently
reduced W_*_AMPA)".

Caveat: trained weights are matched to gAMPA=1.0. Scaling gAMPA at inference
puts the network OUT OF DISTRIBUTION — results probe SUBSTRATE CAPACITY but
do NOT predict a retrained network's behaviour.

Grid:
  gAMPA_scale ∈ {1.0, 0.5, 0.2, 0.1, 0.05}   # default down to 5%
  intensity   ∈ {0.3, 1.0}                   # moderate + paper default
  gNMDA       ∈ {0.05, 1.30}
  delays      = canonical asym_25_40 (what the model was trained on)
  pulse_ms    = 100
  ckpt 00 only

Total: 5 × 2 × 2 = 20 conditions × 3 modalities = 60 measurements.

If facilitation > 5 ms emerges at any cell → architectural recommendation
empirically supported. Else → substrate fundamentally limited even with
arbitrary architectural changes.
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

N_SUBSTEPS = 10
N_FRAMES = 400
PULSE_MS = 100

GAMPA_SCALES   = [1.0, 0.5, 0.2, 0.1, 0.05]
INTENSITY_LEV  = [0.3, 1.0]
GNMDA_LEV      = [0.05, 1.30]


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
            "min(A,V)_eff": minAV, "facil_ms": facil}


def fmt(x):
    if isinstance(x, float) and np.isnan(x):
        return "  nan "
    return f"{x:>5}"


def run_one_ckpt(ckpt_path: Path) -> list[dict]:
    rows = []
    for gamp_scale in GAMPA_SCALES:
        for gNMDA in GNMDA_LEV:
            for I in INTENSITY_LEV:
                net = load_msi_model(ckpt_path, device=DEVICE)
                apply_canonical_overrides(net)
                net.n_substeps = N_SUBSTEPS
                # NB: don't touch delays — use trained asymmetric (25/40)
                net.gNMDA = float(gNMDA)
                net.gAMPA = float(gamp_scale)  # default was 1.0 → scale directly
                t0 = time.time()
                r = measure(net, I)
                el = time.time() - t0
                r["gAMPA"] = gamp_scale
                r["gNMDA"] = gNMDA
                r["intensity"] = I
                r["Model"] = ckpt_path.stem
                r["elapsed_s"] = round(el, 2)
                rows.append(r)
                if r["facil_ms"] >= 5.0:
                    marker = "  ◀── ≥5 ms FACIL!"
                elif r["facil_ms"] >= 2.0:
                    marker = "  ◀── facil"
                else:
                    marker = ""
                print(
                    f"    gAMPA={gamp_scale:>5.2f}  gNMDA={gNMDA:>5.2f}  I={I:>4.2f}  "
                    f"A={fmt(r['A_ms'])} V={fmt(r['V_ms'])} B={fmt(r['B_ms'])} "
                    f"facil={r['facil_ms']:>6.1f}{marker}"
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
    print("TASK #55 STAGE 4 — gAMPA-scaling architectural-capacity probe")
    print(f"   n_models={args.n_models}  gAMPA_scales={GAMPA_SCALES}  "
          f"intensity={INTENSITY_LEV}  gNMDA={GNMDA_LEV}")
    print(f"   delays = trained asymmetric (25/40)  pulse = {PULSE_MS} ms  "
          f"1 ms resolution")
    print(f"   CAVEAT: gAMPA scaling at inference puts trained weights")
    print(f"           OUT-OF-DISTRIBUTION. Probes SUBSTRATE CAPACITY only.")
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

    print("\nFacilitation matrix (rows=gAMPA, cols=intensity, panels=gNMDA):")
    for gN in GNMDA_LEV:
        sub = df[df["gNMDA"] == gN]
        piv = sub.pivot_table(
            index="gAMPA", columns="intensity", values="facil_ms",
            aggfunc="mean",
        )
        print(f"\n=== gNMDA = {gN} ===")
        print(piv.to_string(float_format=lambda x: f"{x:7.2f}", na_rep="  nan  "))

    print("\nA/V/B matrix per cell:")
    for gN in GNMDA_LEV:
        sub = df[df["gNMDA"] == gN]
        for I in INTENSITY_LEV:
            sub2 = sub[sub["intensity"] == I][["gAMPA", "A_ms", "V_ms", "B_ms", "facil_ms"]]
            print(f"\n--- gNMDA={gN}, I={I} ---")
            print(sub2.to_string(index=False, float_format=lambda x: f"{x:6.1f}", na_rep="  nan "))

    print("\n" + "=" * 100)
    print("TOP-5 facilitation cells (descending):")
    top = df.nlargest(5, "facil_ms")
    print(top[["gAMPA", "gNMDA", "intensity", "A_ms", "V_ms", "B_ms", "facil_ms"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.2f}", na_rep="  nan  "))

    out_csv = Path(__file__).parent / f"task55_stage4_{args.n_models}ckpts.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nRaw saved → {out_csv.name}")


if __name__ == "__main__":
    main()
