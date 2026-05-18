"""Task #53 — Experiment 2: sub-frame measurement.

Hypothesis: a sub-10-ms multisensory latency facilitation exists but is hidden
by the 10-ms frame quantum. Test by re-sampling the network at finer frame
granularity while preserving total stimulus duration and simulation window.

Method (researcher's option (c)): keep dt=0.1 ms, REDUCE n_substeps from 100
to a smaller value. Each `update_all_layers_batch` then advances fewer
substeps → finer first-spike resolution. The model's per-substep dynamics
(Izh dt, conduction-delay counts, synaptic decays) are unchanged because dt
is fixed at 0.1 ms and conduction delays live in substep units.

To keep the experiment comparable to canonical:
  pulse_frames × frame_ms = 100 ms  (stimulus duration)
  n_frames    × frame_ms = 400 ms  (simulation window)

  CANONICAL   (frame_ms=10):  pulse_frames=10,   n_frames=40
  RESOLUTION 1 (frame_ms=1):   pulse_frames=100,  n_frames=400
  RESOLUTION 0.1 (frame_ms=0.1): pulse_frames=1000, n_frames=4000 (optional)

Each row: per-ckpt latency at the requested resolution, plus ΔLat.

Default: 10 ckpts × intensity=1.0 × resolution {10, 1, 0.1} ms.

Imports `measure_latency` and `load_msi_model` from response_latency_test
(picks up Coder #40 fix). Line-692 Izh override replicated. No production
code edited.
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


# (frame_ms_target, n_substeps, pulse_frames, n_frames)
RESOLUTIONS = {
    "10ms":  (10,    100,   10,   40),
    "1ms":   ( 1,     10,  100,  400),
    "0.1ms": ( 0.1,    1, 1000, 4000),
}


def latency_at_resolution(
    net, n_substeps: int, pulse_frames: int, n_frames: int,
    intensity: float = 1.0,
) -> dict[str, float]:
    net.n_substeps = int(n_substeps)
    net._reset_delay_buffers()   # delay buffers length is in substeps — re-init
    A = measure_latency(net, modality="A", intensity=intensity,
                        pulse_frames=pulse_frames, n_frames=n_frames)
    V = measure_latency(net, modality="V", intensity=intensity,
                        pulse_frames=pulse_frames, n_frames=n_frames)
    B = measure_latency(net, modality="B", intensity=intensity,
                        pulse_frames=pulse_frames, n_frames=n_frames)
    uni_mean = float(np.nanmean([A, V]))
    dlat = uni_mean - B if not np.isnan(B) else np.nan
    return {"A_ms": A, "V_ms": V, "B_ms": B, "UniMean_ms": uni_mean,
            "ΔLat_ms": dlat}


def run_one_ckpt(
    ckpt_path: Path, resolutions: list[str], intensity: float,
) -> list[dict]:
    net = load_msi_model(ckpt_path, device=DEVICE)
    apply_canonical_overrides(net)
    rows: list[dict] = []
    for res in resolutions:
        fm, n_sub, pf, nf = RESOLUTIONS[res]
        t0 = time.time()
        r = latency_at_resolution(net, n_sub, pf, nf, intensity=intensity)
        el = time.time() - t0
        r["resolution_ms"] = fm
        r["n_substeps"] = n_sub
        r["pulse_frames"] = pf
        r["n_frames"] = nf
        r["intensity"] = intensity
        r["Model"] = ckpt_path.stem
        r["elapsed_s"] = round(el, 2)
        rows.append(r)
        print(
            f"    res={res:>6}  A={r['A_ms']!s:>8}  V={r['V_ms']!s:>8}  "
            f"B={r['B_ms']!s:>8}  ΔLat={r['ΔLat_ms']!s:>8}  el={el:.1f}s"
        )
    del net
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_models", type=int, default=10)
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument(
        "--resolutions", nargs="+", default=["10ms", "1ms"],
        choices=list(RESOLUTIONS.keys()),
        help="which sub-frame resolutions to test",
    )
    args = ap.parse_args()

    print("=" * 78)
    print("TASK #53 EXP-2 — sub-frame latency measurement")
    print(f"   n_models = {args.n_models}    intensity = {args.intensity}")
    print(f"   resolutions = {args.resolutions}")
    print(f"   (preserved stimulus=100ms, window=400ms across resolutions)")
    print(f"   canonical Izh override: aM=0.001 bM=0.2 cM=-60 dM=0.1")
    print("=" * 78)

    all_rows: list[dict] = []
    t0 = time.time()
    for i in range(args.n_models):
        ckpt = ROOT / CKPT_TEMPLATE.format(i)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        print(f"\n[{i:02d}] {ckpt.name}")
        all_rows.extend(run_one_ckpt(ckpt, args.resolutions, args.intensity))
    df = pd.DataFrame(all_rows)
    print(f"\nElapsed: {time.time() - t0:.1f}s")

    # group by resolution
    print("\n" + "=" * 78)
    print("PER-RESOLUTION SUMMARY")
    print("=" * 78)
    g = df.groupby("resolution_ms")
    summary = pd.DataFrame({
        "n":       g["A_ms"].count(),
        "A_mean":  g["A_ms"].mean(),
        "A_std":   g["A_ms"].std(),
        "V_mean":  g["V_ms"].mean(),
        "V_std":   g["V_ms"].std(),
        "B_mean":  g["B_ms"].mean(),
        "B_std":   g["B_ms"].std(),
        "ΔLat_mean": g["ΔLat_ms"].mean(),
        "ΔLat_std":  g["ΔLat_ms"].std(),
    })
    print(summary.to_string(float_format=lambda x: f"{x:7.2f}", na_rep="  nan  "))

    out_csv = Path(__file__).parent / (
        f"task53_exp2_subframe_{'_'.join(args.resolutions)}.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"\nRaw saved → {out_csv.name}")


if __name__ == "__main__":
    main()
