"""task #41 — Action #2: latency under TRUE-CONTROL Izhikevich parameters.

Replicates `response_latency_test.py::run_latency_test()` BUT replaces the
canonical line-686 adaptation-condition Izhikevich override
    aM = 0.001, bM = 0.2,  cM = -60.0, dM = 0.1
with the TRUE-CONTROL Regular-Spiking defaults specified by team-lead:
    aM = 0.02,  bM = 0.2,  cM = -65.0, dM = 8.0

Everything else (dt=0.1, n_substeps=100, gNMDA=1.30, dt_correct_nmda=True,
plasticity_enabled=False, first-spike detection via the (post-fix) per-frame
return_spike_sum readout) matches the canonical pipeline. The wrapper does NOT
edit production code; it monkey-patches the four Izh attributes after
load_msi_model and BEFORE latency_profile_for_model is called.

Action #1 of task #41 (`python response_latency_test.py`) is the
canonical-script run for the line-686-adaptation case; this wrapper is the
true-control counterpart.

Imports latency_profile_for_model directly so it picks up coder's task-#40
measurement-bug fix at line 649 (and 82/137/252/347) automatically.
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

from response_latency_test import load_msi_model, latency_profile_for_model

CKPT_TEMPLATE = "checkpoint/msi_model_surr_10_{:02d}.pt"
N_MODELS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Regular-Spiking Izhikevich defaults — per team-lead's task #41 brief.
IZH_TRUE_CONTROL = dict(aM=0.02, bM=0.2, cM=-65.0, dM=8.0)


def run_true_control() -> pd.DataFrame:
    rows: list[dict] = []
    for i in range(N_MODELS):
        ckpt = ROOT / CKPT_TEMPLATE.format(i)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        print(f"[{i}] loading {ckpt.name} ...")
        net = load_msi_model(ckpt, device=DEVICE)

        # ensure canonical dt and n_substeps (frame_ms = 10 ms) and gNMDA control.
        net.dt = 0.1
        net.n_substeps = 100
        net.dt_correct_nmda = True
        net.gNMDA = 1.30
        net.plasticity_enabled = False
        net._reset_delay_buffers()

        # APPLY TRUE-CONTROL Izh (replaces canonical line-686 override).
        for k, v in IZH_TRUE_CONTROL.items():
            setattr(net, k, v)

        tic = time.time()
        row = latency_profile_for_model(net)
        row["Model"] = ckpt.stem
        rows.append(row)
        print(
            f"    done in {time.time() - tic:.2f}s  ->  "
            f"A={row['A_ms']!s:>6}  V={row['V_ms']!s:>6}  "
            f"B={row['B_ms']!s:>6}  dLat={row['ΔLatency_ms']!s:>6}"
        )
        del net
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows).set_index("Model")


def summarise(df: pd.DataFrame, label: str) -> None:
    mean = df.mean()
    sem = df.sem()
    print(f"\n=== {label} ===")
    print(df.to_string(float_format=lambda x: f"{x:6.1f}", na_rep=" n/a "))
    print("\nMean ± SEM (ms):")
    for col in ["A_ms", "V_ms", "B_ms", "UniMean_ms", "ΔLatency_ms"]:
        print(f"  {col:<14s}: {mean[col]:6.2f} ± {sem[col]:5.2f}")


def main() -> None:
    print("=" * 72)
    print("TASK #41 ACTION #2 — latency at dt=0.1, post-meas-fix,")
    print("                     TRUE-CONTROL Izh (RS defaults).")
    print("                     aM=0.02  bM=0.2  cM=-65  dM=8")
    print("                     gNMDA=1.30  n_substeps=100  frame_ms=10")
    print("=" * 72)
    t0 = time.time()
    df = run_true_control()
    summarise(df, "TRUE-CONTROL Izh latency summary across 10 models")
    print(f"\nTotal elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
