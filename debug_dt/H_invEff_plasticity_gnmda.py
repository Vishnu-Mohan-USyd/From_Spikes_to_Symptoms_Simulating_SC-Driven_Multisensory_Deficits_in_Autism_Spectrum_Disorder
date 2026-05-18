"""Task #57 — broader sweep: AGC + plasticity + gNMDA.

Cell B/C (frozen AGC only) did NOT recover paper-direction MEI on M00:
  Cell B MEI = +0.15, +0.24, +0.23, -0.01, -0.01, +0.49  (non-monotonic, dips negative)
  Cell C (frozen 0.6) similar.

That means AGC is not the only culprit.  Test the next layer of state-drift
(plasticity) and the parameter regime (gNMDA) together.

Cells:
  [P]: plasticity_enabled = False, freeze_g_FFinh = True, gNMDA = {0.05, 0.6, 1.30}
       — both drifts disabled, vary gNMDA

If any (plasticity, AGC, gNMDA) cell produces decreasing MEI matching paper,
that's the methodology + parameter prescription.  If NONE does, the inversion
is a fundamental network-behavior issue (not a measurement bug), and the
deliverable is "MEI inversion cannot be recovered by methodology fix; requires
either retraining or different params (which the team-lead must decide)."
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
def integrated_spikes(net, cond, intensity):
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
    pop = 0.0
    for t in range(N_FRAMES):
        *_, sSum = net.update_all_layers_batch(
            xA[t:t + 1], xV[t:t + 1], return_spike_sum=True
        )
        pop += sSum.sum().item()
    return pop


def run_cell(*, label, ckpt_path, gNMDA, freeze_agc, disable_plast, g_ff_seed=None,
             load_fresh_per_intensity=False):
    """Returns dict with R_A, R_V, R_AV, MEI per intensity + meta."""
    if not load_fresh_per_intensity:
        net = load_msi_model(ckpt_path, device=DEVICE)
        net.gNMDA = float(gNMDA)
        if g_ff_seed is not None:
            net.g_FFinh = float(g_ff_seed)
        if freeze_agc:
            net.freeze_g_FFinh = True
        if disable_plast:
            net.plasticity_enabled = False

    rows = {}
    for I in INTENSITIES:
        if load_fresh_per_intensity:
            net = load_msi_model(ckpt_path, device=DEVICE)
            net.gNMDA = float(gNMDA)
            if g_ff_seed is not None:
                net.g_FFinh = float(g_ff_seed)
            if freeze_agc:
                net.freeze_g_FFinh = True
            if disable_plast:
                net.plasticity_enabled = False
        rA = integrated_spikes(net, "A", I)
        rV = integrated_spikes(net, "V", I)
        rB = integrated_spikes(net, "B", I)
        d = max(rA, rV)
        mei = (rB - d) / d if d > 0 else float('nan')
        rows[I] = dict(A=rA, V=rV, AV=rB, MEI=mei)
        if load_fresh_per_intensity:
            del net; torch.cuda.empty_cache()

    if not load_fresh_per_intensity:
        del net; torch.cuda.empty_cache()
    meis = [rows[I]["MEI"] for I in INTENSITIES]
    rAs  = [rows[I]["A"]  for I in INTENSITIES]
    rVs  = [rows[I]["V"]  for I in INTENSITIES]
    rBs  = [rows[I]["AV"] for I in INTENSITIES]
    print(f"  {label}")
    print(f"    R_A:  " + " ".join(f"{r:>6.0f}" for r in rAs))
    print(f"    R_V:  " + " ".join(f"{r:>6.0f}" for r in rVs))
    print(f"    R_AV: " + " ".join(f"{r:>6.0f}" for r in rBs))
    print(f"    MEI:  " + " ".join(f"{m:>+6.2f}" for m in meis))
    # direction
    diffs = np.diff(meis)
    n_down = (diffs < -0.01).sum(); n_up = (diffs > 0.01).sum()
    if n_down >= 4:
        dirn = "FALLING ✓ matches paper"
    elif n_up >= 4:
        dirn = "RISING (inverted vs paper)"
    elif (n_down + n_up) <= 1:
        dirn = "FLAT"
    else:
        dirn = "NON-MONOTONIC"
    print(f"    direction: {dirn}    R_A_mono?: {all(rAs[i+1]>=rAs[i]-1 for i in range(len(rAs)-1))}")
    return dict(meis=meis, rAs=rAs, rVs=rVs, rBs=rBs, dirn=dirn)


def main():
    ckpt = MODEL_DIR / "msi_model_surr_10_00.pt"
    torch.manual_seed(0); np.random.seed(0)
    print(f"\n{'='*88}")
    print(f"Task #57 H_PLAST+gNMDA — broader recovery sweep on M00")
    print(f"{'='*88}\n")

    results = {}

    print("───── group 1: vary gNMDA, both drifts disabled (frozen AGC + no plast) ─────")
    for g in [0.05, 0.3, 0.6, 1.0, 1.3]:
        key = f"frozen+noplast g={g}"
        results[key] = run_cell(
            label=key, ckpt_path=ckpt, gNMDA=g,
            freeze_agc=True, disable_plast=True, g_ff_seed=0.6,
        )

    print()
    print("───── group 2: gNMDA=1.30 (production), peel back one drift at a time ─────")
    results["agc=drift  plast=on   g=1.30"] = run_cell(
        label="agc=drift  plast=on   g=1.30  (=production default)",
        ckpt_path=ckpt, gNMDA=1.30, freeze_agc=False, disable_plast=False,
    )
    results["agc=frozen plast=on   g=1.30"] = run_cell(
        label="agc=frozen plast=on   g=1.30",
        ckpt_path=ckpt, gNMDA=1.30, freeze_agc=True, disable_plast=False, g_ff_seed=0.6,
    )
    results["agc=frozen plast=off  g=1.30"] = run_cell(
        label="agc=frozen plast=off  g=1.30",
        ckpt_path=ckpt, gNMDA=1.30, freeze_agc=True, disable_plast=True, g_ff_seed=0.6,
    )
    results["agc=frozen plast=off  g=1.30  fresh-per-I"] = run_cell(
        label="agc=frozen plast=off  g=1.30  load_fresh_per_intensity (no carryover)",
        ckpt_path=ckpt, gNMDA=1.30, freeze_agc=True, disable_plast=True, g_ff_seed=0.6,
        load_fresh_per_intensity=True,
    )

    print()
    print(f"{'='*88}")
    print(f"SUMMARY — which cells produce monotonically-decreasing MEI?")
    print(f"{'='*88}")
    falling = [k for k, v in results.items() if v["dirn"].startswith("FALLING")]
    if falling:
        print(f"  ✓ MEI FALLING (paper-like) in: {falling}")
    else:
        print(f"  ✗ NO cell produces falling MEI on M00.")
        print()
        print(f"  This means the inversion is a NETWORK-BEHAVIOR issue, not a")
        print(f"  measurement-script bug.  The trained network (with any gNMDA")
        print(f"  in 0.05..1.30, with/without AGC, with/without plasticity)")
        print(f"  does not exhibit the inverse-effectiveness regime.")
    print()
    for k, v in results.items():
        print(f"  {k:<48s}  {v['dirn']}")
    print()


if __name__ == "__main__":
    main()
