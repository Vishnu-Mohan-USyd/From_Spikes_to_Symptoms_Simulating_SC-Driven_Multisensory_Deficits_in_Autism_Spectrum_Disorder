"""Task #53 — Experiment 4: conduction-delay symmetry.

Key finding from Exp 1 & 2: model has no multisensory latency facilitation
at any intensity (Exp 1) and no sub-frame facilitation (Exp 2). Direct
inspection of `msi_model_surr_10_00.pt` reveals asymmetric trained delays:
    conduction_delay_a2msi_ms = 25.0  (A reaches MSI at t = 25 ms)
    conduction_delay_v2msi_ms = 40.0  (V reaches MSI at t = 40 ms)

A first spike emerges at t ≈ 26 ms (1 ms after A arrives). By the time V
arrives at t = 40 ms, the MSI neuron has already spiked (and refractoried
or moved past peak). V cannot contribute to the FIRST spike under this
configuration → no integration window → race-model dominates.

Hypothesis (researcher's H4): equalizing delays so A and V arrive
simultaneously will allow temporal summation → B's first spike earlier than
A's. This tests whether the NMDA-AND-gate + AMPA-summation substrate at
Training.py:2094-2098 is actually capable of producing facilitation.

Test cases (symmetric delays, applied at inference; the trained weights
remain unchanged):
    asymmetric (canonical): A=25 ms, V=40 ms    [baseline]
    sym_25:                  A=25 ms, V=25 ms    [V brought to A speed]
    sym_40:                  A=40 ms, V=40 ms    [A slowed to V speed]
    sym_32:                  A=32 ms, V=32 ms    [meet in middle]

Caveat: the network was TRAINED with asymmetric delays; changing delays at
inference puts the weights out-of-distribution. This is a *substrate test*,
not a "what the model would learn" test.

Measurement at 1 ms resolution (n_substeps=10, frame_ms=1) to avoid frame
quantization. Pulse=100 ms wall time (pulse_frames=100), window=400 ms.
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


def apply_canonical_overrides(net) -> None:
    net.aM, net.bM, net.cM, net.dM = 0.001, 0.2, -60.0, 0.1


def set_delays(net, a_ms: float, v_ms: float) -> None:
    """Set A and V conduction delays (and dependent inhibitory delays) to the
    requested ms values, then reset buffers.

    Both *_ms and substep-count attributes are updated to keep the model's
    `dt_correct_nmda=True` and legacy paths consistent. Inhibitory delays
    are kept at +20 substeps offset (matching the construction-time rule).
    """
    dt = net.dt
    a_steps = int(round(a_ms / dt))
    v_steps = int(round(v_ms / dt))

    # primary excitatory delays
    net.conduction_delay_a2msi = a_steps
    net.conduction_delay_v2msi = v_steps
    net.conduction_delay_a2msi_ms = a_steps * dt
    net.conduction_delay_v2msi_ms = v_steps * dt

    # dependent inhibitory delays (a+20, v+20 substeps — match construction)
    net.conduction_delay_inA_inh = a_steps + 20
    net.conduction_delay_inV_inh = v_steps + 20
    net.conduction_delay_inA_inh_ms = (a_steps + 20) * dt
    net.conduction_delay_inV_inh_ms = (v_steps + 20) * dt
    net.conduction_delay_a2msi_inh = a_steps + 20
    net.conduction_delay_v2msi_inh = v_steps + 20
    net.conduction_delay_a2msi_inh_ms = (a_steps + 20) * dt
    net.conduction_delay_v2msi_inh_ms = (v_steps + 20) * dt

    # inh→exc delay (max + 50 substeps — match construction)
    net.conduction_delay_msi_inh2exc = max(a_steps, v_steps) + 50
    net.conduction_delay_msi_inh2exc_ms = (max(a_steps, v_steps) + 50) * dt

    net._reset_delay_buffers()


# (label, A delay ms, V delay ms)
DELAY_CASES = [
    ("asym_25_40",  25.0, 40.0),    # canonical trained values
    ("sym_25",      25.0, 25.0),    # both at A's current speed
    ("sym_40",      40.0, 40.0),    # both at V's current speed
    ("sym_32",      32.0, 32.0),    # mean
]


def measure(net, intensity: float, n_substeps: int = 10,
            pulse_frames: int = 100, n_frames: int = 400) -> dict:
    net.n_substeps = int(n_substeps)
    A = measure_latency(net, modality="A", intensity=intensity,
                        pulse_frames=pulse_frames, n_frames=n_frames)
    V = measure_latency(net, modality="V", intensity=intensity,
                        pulse_frames=pulse_frames, n_frames=n_frames)
    B = measure_latency(net, modality="B", intensity=intensity,
                        pulse_frames=pulse_frames, n_frames=n_frames)
    uni_mean = float(np.nanmean([A, V]))
    dlat = uni_mean - B if not np.isnan(B) else np.nan
    # B vs min(A,V) — the *real* facilitation criterion (must beat the fastest unimodal)
    minAV = float(np.nanmin([A, V]))
    facil = minAV - B if not np.isnan(B) else np.nan
    return {"A_ms": A, "V_ms": V, "B_ms": B,
            "UniMean_ms": uni_mean, "ΔLat_ms": dlat,
            "min(A,V)_ms": minAV, "B_vs_minAV_ms": facil}


def run_one_ckpt(ckpt_path: Path, intensity: float) -> list[dict]:
    rows = []
    for label, a_ms, v_ms in DELAY_CASES:
        net = load_msi_model(ckpt_path, device=DEVICE)
        apply_canonical_overrides(net)
        set_delays(net, a_ms, v_ms)
        t0 = time.time()
        r = measure(net, intensity=intensity)
        el = time.time() - t0
        r["case"] = label
        r["delay_A_ms"] = a_ms
        r["delay_V_ms"] = v_ms
        r["intensity"] = intensity
        r["Model"] = ckpt_path.stem
        r["elapsed_s"] = round(el, 2)
        rows.append(r)
        print(
            f"    {label:<12s} A={r['A_ms']!s:>7}  V={r['V_ms']!s:>7}  "
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
    ap.add_argument("--intensity", type=float, default=1.0)
    args = ap.parse_args()

    print("=" * 84)
    print("TASK #53 EXP-4 — conduction-delay symmetry @ 1 ms resolution")
    print(f"   n_models = {args.n_models}    intensity = {args.intensity}")
    print(f"   measurement: n_substeps=10, pulse=100ms, window=400ms")
    print(f"   canonical Izh override: aM=0.001 bM=0.2 cM=-60 dM=0.1")
    print(f"   delay cases: {[c[0] for c in DELAY_CASES]}")
    print("=" * 84)

    all_rows: list[dict] = []
    t0 = time.time()
    for i in range(args.n_models):
        ckpt = ROOT / CKPT_TEMPLATE.format(i)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        print(f"\n[{i:02d}] {ckpt.name}")
        all_rows.extend(run_one_ckpt(ckpt, args.intensity))
    df = pd.DataFrame(all_rows)
    print(f"\nElapsed: {time.time() - t0:.1f}s")

    print("\n" + "=" * 84)
    print("PER-CASE SUMMARY (mean ± std across ckpts)")
    print("=" * 84)
    g = df.groupby("case", sort=False)
    summary = pd.DataFrame({
        "n":           g["A_ms"].count(),
        "A_mean":      g["A_ms"].mean(),
        "V_mean":      g["V_ms"].mean(),
        "B_mean":      g["B_ms"].mean(),
        "minAV_mean":  g["min(A,V)_ms"].mean(),
        "B_minAV_mean": g["B_vs_minAV_ms"].mean(),
        "B_minAV_std":  g["B_vs_minAV_ms"].std(),
        "ΔLat_mean":   g["ΔLat_ms"].mean(),
    })
    print(summary.to_string(float_format=lambda x: f"{x:7.2f}", na_rep="  nan  "))

    out_csv = Path(__file__).parent / f"task53_exp4_delay_symmetry_I{args.intensity}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nRaw saved → {out_csv.name}")


if __name__ == "__main__":
    main()
