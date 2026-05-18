"""Task #57 — Phase 3.

Given that I=0 baseline ≈ 0 (no spontaneous firing), the lead's H_baseline must
be ruled out (the ensemble Phase 2 confirms this).  Test alternative methodology-side
hypotheses on M00 (single ckpt for fast iteration):

  H_window_during      response counted only during the stim pulse (frames 0..PULSE_LEN)
  H_window_evoked      response counted only AFTER stim onset+latency (frames 2..PULSE_LEN+5)
  H_local_strict       response counted only at the RF-centre neuron (1 neuron)
  H_local_5            response counted in ±5 neurons (=stim sigma)
  H_local_10           response counted in ±10 neurons
  H_peak_rate          peak per-frame spike count (instead of integrated)
  H_peak_local         peak per-frame spike count within ±10 of centre
  H_first_5frames      sum over first 5 frames only (avoid late-buildup at high I)

For each: compute MEI per intensity, report direction (rising vs falling vs flat).
A method that RECOVERS the paper's monotonic-decreasing direction is the candidate
methodology fix.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Training import MultiBatchAudVisMSINetworkTime

MODEL_DIR = ROOT / "checkpoint"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOC_DEG = 90
SIGMA_IN = 5.0
PULSE_LEN = 10
N_FRAMES = 20
INTENSITIES = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]


def load_msi_model(ckpt_path: Path, *, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
    net.load_state_dict(ckpt["model_state"])
    for k, v in ckpt["mutable_hparams"].items():
        setattr(net, k, v)
    net.to(device).eval()
    net.device = torch.device(device)
    return net


@torch.no_grad()
def spike_matrix(net, cond, intensity):
    """Return [T, N] per-frame spike sum matrix for one trial."""
    gauss = lambda: torch.exp(
        -0.5 * ((torch.arange(net.n, device=DEVICE) -
                 (LOC_DEG * (net.n - 1) / (net.space_size - 1))) / SIGMA_IN) ** 2
    ) * intensity
    xA = torch.zeros(N_FRAMES, net.n, device=DEVICE)
    xV = torch.zeros_like(xA)
    if cond in ("A", "B"):
        xA[:PULSE_LEN] = gauss()
    if cond in ("V", "B"):
        xV[:PULSE_LEN] = gauss()
    net.reset_state(batch_size=1)
    out = np.zeros((N_FRAMES, net.n))
    for t in range(N_FRAMES):
        *_, sSum = net.update_all_layers_batch(
            xA[t:t + 1], xV[t:t + 1], return_spike_sum=True
        )
        out[t] = sSum[0].cpu().numpy()
    return out


def compute_R(mat, method, centre):
    """Given a [T, N] spike matrix, return scalar R per the chosen method."""
    if method == "full_pop_all_frames":
        return float(mat.sum())
    if method == "during_stim_full_pop":
        return float(mat[:PULSE_LEN].sum())
    if method == "evoked_window_full_pop":
        return float(mat[2:PULSE_LEN + 5].sum())  # latency + during + 5-frame tail
    if method == "first_5frames_full_pop":
        return float(mat[:5].sum())
    if method.startswith("local_"):
        radius = int(method.split("_")[1])
        lo = max(0, centre - radius); hi = min(mat.shape[1], centre + radius + 1)
        if method.endswith("_during"):
            return float(mat[:PULSE_LEN, lo:hi].sum())
        return float(mat[:, lo:hi].sum())
    if method == "peak_per_frame_full":
        return float(mat.sum(axis=1).max())
    if method == "peak_per_frame_local10":
        lo = max(0, centre - 10); hi = min(mat.shape[1], centre + 11)
        return float(mat[:, lo:hi].sum(axis=1).max())
    raise ValueError(method)


METHODS = [
    "full_pop_all_frames",          # current script
    "during_stim_full_pop",
    "evoked_window_full_pop",
    "first_5frames_full_pop",
    "local_0",                       # only centre neuron
    "local_3",
    "local_5",                       # = sigma_in
    "local_10",
    "local_20",
    "local_5_during",
    "local_10_during",
    "peak_per_frame_full",
    "peak_per_frame_local10",
]


def main():
    ckpt_path = MODEL_DIR / "msi_model_surr_10_00.pt"
    torch.manual_seed(0); np.random.seed(0)
    net = load_msi_model(ckpt_path, device=DEVICE)
    setattr(net, "gNMDA", 1.30)

    centre = int(round(LOC_DEG * (net.n - 1) / (net.space_size - 1)))
    print(f"\n{'='*88}")
    print(f"Phase 3 — methodology sweep on M00 (centre neuron idx = {centre})")
    print(f"{'='*88}")
    print(f"  goal: identify R-count methodology that RECOVERS monotonic-decreasing MEI")
    print()

    # Pre-compute spike matrices once per (cond, intensity)
    mats = {}
    for cond in ("A", "V", "B"):
        for I in INTENSITIES:
            mats[(cond, I)] = spike_matrix(net, cond, I)
            print(f"  cached cond={cond} I={I:.2f}: total={mats[(cond, I)].sum():.0f}")
    print()

    # Compute MEI per method
    print(f"{'='*88}")
    print(f"MEI per method × intensity (last col = direction)")
    print(f"{'='*88}")
    hdr = f"  {'method':<26s} | " + " ".join(f"I={I:<5.2f}" for I in INTENSITIES) + " | direction"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for method in METHODS:
        meis = []
        for I in INTENSITIES:
            RA = compute_R(mats[("A", I)], method, centre)
            RV = compute_R(mats[("V", I)], method, centre)
            RB = compute_R(mats[("B", I)], method, centre)
            d = max(RA, RV)
            mei = (RB - d) / d if d > 0 else float('nan')
            meis.append(mei)
        # direction: is MEI decreasing with intensity?
        diffs = np.diff(meis)
        n_down = (diffs < -0.01).sum()
        n_up   = (diffs > 0.01).sum()
        if n_down >= 4:
            dirn = "FALLING ✓"
        elif n_up >= 4:
            dirn = "RISING (bad)"
        else:
            dirn = "FLAT/MIXED"
        print(f"  {method:<26s} | " +
              " ".join(f"{m:>+6.2f}" for m in meis) +
              f" | {dirn}  (MEI[0]/MEI[-1]={meis[0]/meis[-1] if meis[-1]!=0 else float('nan'):.2f})")

    print()
    print(f"{'='*88}")
    print(f"DIAGNOSTIC: per-method R_A vs intensity (does R_A monotonically increase?)")
    print(f"{'='*88}")
    print(f"  {'method':<26s} | " + " ".join(f"I={I:<5.2f}" for I in INTENSITIES))
    print("  " + "-" * 90)
    for method in METHODS:
        RAs = [compute_R(mats[("A", I)], method, centre) for I in INTENSITIES]
        is_mono = all(RAs[i+1] >= RAs[i] - 1.0 for i in range(len(RAs)-1))
        marker = "✓" if is_mono else "DIP"
        print(f"  {method:<26s} | " + " ".join(f"{r:>6.0f}" for r in RAs) + f"  {marker}")


if __name__ == "__main__":
    main()
