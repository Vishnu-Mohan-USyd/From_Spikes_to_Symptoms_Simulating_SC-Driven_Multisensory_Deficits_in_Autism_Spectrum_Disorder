"""Task #53 — Experiment 1: intensity sweep to test inverse-effectiveness.

Researcher #43 hypothesised: paper's default intensity=1.0 is suprathreshold,
and biological multisensory latency facilitation (Rowland 2007, Stein &
Meredith) appears specifically in the *near-threshold* regime. At suprathreshold
inputs, A and V each independently push past threshold quickly, leaving no room
for sub-threshold summation to bring the bimodal first spike *earlier* than
either unimodal — race-model dominates.

Test: sweep intensity ∈ {0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4}, measure
A/V/B latency, compute ΔLatency = (A+V)/2 − B. Look for ΔLat > 0 emerging at
low intensity, vanishing at high.

This wrapper imports `measure_latency` and `load_msi_model` from
response_latency_test (so it picks up Coder's task-#40 measurement-bug fix
at line 651 automatically). The line-692 Izh override is replicated exactly
to match canonical behaviour. No production code is edited.

Stages:
  S1: ckpt 00 only (1 model × 8 intensities × 3 modalities = 24 trials,
      ~30-60 s).
      → if ΔLat curve clearly emerges at low I, expand to all 10 ckpts.
  S2: (controlled by --all flag) all 10 ckpts × 8 intensities → mean ± SEM.

Pulse_frames=10, n_frames=40, sigma_in=5.0, centre_deg=90.0 — matches
response_latency_test.py::measure_latency defaults.
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

INTENSITIES = [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4]


def apply_canonical_overrides(net) -> None:
    """Replicate response_latency_test.py::run_latency_test() line-692 override.

    Only mutation made by canonical script: Izhikevich adaptation params.
    dt, n_substeps, dt_correct_nmda, gNMDA, plasticity_enabled keep
    whatever the checkpoint mutable_hparams set them to.
    """
    net.aM, net.bM, net.cM, net.dM = 0.001, 0.2, -60.0, 0.1


def latency_at_intensity(net, intensity: float) -> dict[str, float]:
    A = measure_latency(net, modality="A", intensity=intensity)
    V = measure_latency(net, modality="V", intensity=intensity)
    B = measure_latency(net, modality="B", intensity=intensity)
    uni_mean = float(np.nanmean([A, V]))
    dlat = uni_mean - B if not np.isnan(B) else np.nan
    return {
        "intensity": intensity,
        "A_ms": A,
        "V_ms": V,
        "B_ms": B,
        "UniMean_ms": uni_mean,
        "ΔLat_ms": dlat,
    }


def run_one_ckpt(ckpt_path: Path) -> pd.DataFrame:
    net = load_msi_model(ckpt_path, device=DEVICE)
    apply_canonical_overrides(net)
    rows = []
    for I in INTENSITIES:
        r = latency_at_intensity(net, I)
        r["Model"] = ckpt_path.stem
        rows.append(r)
        print(
            f"    I={I:>5.2f}  A={r['A_ms']!s:>6}  V={r['V_ms']!s:>6}  "
            f"B={r['B_ms']!s:>6}  ΔLat={r['ΔLat_ms']!s:>6}"
        )
    del net
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--all", action="store_true",
        help="sweep all 10 ckpts (else just ckpt 00 — stage S1).",
    )
    args = ap.parse_args()

    n_models = 10 if args.all else 1
    print("=" * 78)
    print("TASK #53 EXP-1 — intensity sweep for latency facilitation")
    print(f"   n_models = {n_models}    intensities = {INTENSITIES}")
    print(f"   canonical Izh override: aM=0.001 bM=0.2 cM=-60 dM=0.1")
    print(f"   (dt, n_substeps, gNMDA, plasticity from ckpt mutable_hparams)")
    print("=" * 78)

    all_rows = []
    t0 = time.time()
    for i in range(n_models):
        ckpt = ROOT / CKPT_TEMPLATE.format(i)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        print(f"\n[{i:02d}] {ckpt.name}")
        df_i = run_one_ckpt(ckpt)
        all_rows.append(df_i)
    df_all = pd.concat(all_rows, ignore_index=True)

    print(f"\nElapsed: {time.time() - t0:.1f}s")

    # group by intensity → mean ± SEM
    print("\n" + "=" * 78)
    print("PER-INTENSITY SUMMARY")
    print("=" * 78)
    g = df_all.groupby("intensity")
    summary = pd.DataFrame({
        "n": g["A_ms"].count(),
        "A_mean": g["A_ms"].mean(),
        "V_mean": g["V_ms"].mean(),
        "B_mean": g["B_ms"].mean(),
        "ΔLat_mean": g["ΔLat_ms"].mean(),
        "ΔLat_sem": g["ΔLat_ms"].sem() if n_models > 1 else 0.0,
    })
    print(summary.to_string(float_format=lambda x: f"{x:6.2f}", na_rep=" nan "))

    # save raw
    out_csv = Path(__file__).parent / (
        f"task53_exp1_intensity_sweep_{'10ckpts' if args.all else 'ckpt00'}.csv"
    )
    df_all.to_csv(out_csv, index=False)
    print(f"\nRaw saved → {out_csv.name}")


if __name__ == "__main__":
    main()
