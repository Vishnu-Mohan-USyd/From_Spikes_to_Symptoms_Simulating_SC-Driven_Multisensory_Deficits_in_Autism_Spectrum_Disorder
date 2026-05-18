"""ISSUE B (Task #57) — reproduce the lead's 10-ckpt dip and check baseline on all 10.

Goal: confirm the lead's data (resp_A 361/311/295/315/358/457) and verify whether
any ckpt has nonzero spontaneous activity at I=0.

Also: per-ckpt breakdown so we can see which ckpts contribute the dip.
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

INTENSITIES = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]


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


def main():
    paths = sorted(MODEL_DIR.glob("msi_model_surr_10_*.pt"))
    print(f"\n{'='*78}")
    print(f"Reproduce lead's 10-ckpt MEI inversion + I=0 spontaneous probe")
    print(f"{'='*78}")
    print(f"  ckpts: {len(paths)}")
    print(f"  intensities: {INTENSITIES}")
    print()

    resp_A = np.zeros((len(paths), len(INTENSITIES)))
    resp_V = np.zeros_like(resp_A)
    resp_AV = np.zeros_like(resp_A)

    for m_i, p in enumerate(paths):
        net = load_msi_model(p, device=DEVICE)
        setattr(net, "gNMDA", 1.30)  # matches inverse_effectiveness_test.py
        for j, I in enumerate(INTENSITIES):
            resp_A[m_i, j]  = integrated_spikes(net, "A", I)
            resp_V[m_i, j]  = integrated_spikes(net, "V", I)
            resp_AV[m_i, j] = integrated_spikes(net, "B", I)
        del net; torch.cuda.empty_cache()
        print(f"  M{m_i:02d}  R_A={resp_A[m_i]}  R_V={resp_V[m_i]}", flush=True)

    print()
    print(f"{'='*78}")
    print("PER-CKPT R_A at all intensities (col headers I=0/0.05/0.1/0.2/0.4/0.8/1.6)")
    print(f"{'='*78}")
    print(f"  {'ckpt':>6s} | " + " ".join(f"{I:>7.3f}" for I in INTENSITIES))
    print("  " + "-" * 70)
    for m_i in range(len(paths)):
        cells = " ".join(f"{resp_A[m_i, j]:>7.0f}" for j in range(len(INTENSITIES)))
        print(f"  M{m_i:02d}    | {cells}")
    print(f"  {'mean':>6s} | " + " ".join(f"{resp_A.mean(0)[j]:>7.1f}" for j in range(len(INTENSITIES))))

    print()
    print(f"{'='*78}")
    print("ENSEMBLE MEAN responses & MEI")
    print(f"{'='*78}")
    mA = resp_A.mean(0); mV = resp_V.mean(0); mAV = resp_AV.mean(0)
    print(f"  {'I':>6s} | {'R_A':>7s} {'R_V':>7s} {'R_AV':>7s} {'maxAV':>7s} {'MEI':>7s}")
    for j, I in enumerate(INTENSITIES):
        d = max(mA[j], mV[j])
        mei = (mAV[j] - d) / d if d > 0 else float('nan')
        print(f"  {I:>6.3f} | {mA[j]:>7.1f} {mV[j]:>7.1f} {mAV[j]:>7.1f} {d:>7.1f} {mei:>7.3f}")

    print()
    print(f"{'='*78}")
    print("BASELINE-SUBTRACTED MEI (using I=0 as the floor)")
    print(f"{'='*78}")
    bg_A = mA[0]; bg_V = mV[0]; bg_AV = mAV[0]
    print(f"  baseline (I=0): R_A={bg_A:.1f}  R_V={bg_V:.1f}  R_AV={bg_AV:.1f}")
    print()
    print(f"  {'I':>6s} | {'evk_A':>7s} {'evk_V':>7s} {'evk_AV':>7s} {'maxAV':>7s} {'MEI_c':>7s}")
    for j in range(1, len(INTENSITIES)):
        eA = mA[j] - bg_A
        eV = mV[j] - bg_V
        eAV = mAV[j] - bg_AV
        d  = max(eA, eV)
        mei_c = (eAV - d) / d if d > 0 else float('nan')
        print(f"  {INTENSITIES[j]:>6.3f} | {eA:>7.1f} {eV:>7.1f} {eAV:>7.1f} {d:>7.1f} {mei_c:>7.3f}")

    print()
    print(f"{'='*78}")
    print("KEY CHECK — is there any spontaneous activity at I=0?")
    print(f"{'='*78}")
    for m_i in range(len(paths)):
        print(f"  M{m_i:02d}:  R_A(I=0)={resp_A[m_i,0]:.0f}  R_V(I=0)={resp_V[m_i,0]:.0f}  R_AV(I=0)={resp_AV[m_i,0]:.0f}")
    print()
    print(f"  Ensemble baseline: R_A={bg_A:.2f}  R_V={bg_V:.2f}  R_AV={bg_AV:.2f}")
    if bg_A < 1 and bg_V < 1 and bg_AV < 1:
        print(f"  → CONFIRMED: spontaneous activity ≈ 0 across the ensemble.")
        print(f"  → H_baseline (lead's primary hypothesis) RULED OUT.")
    else:
        print(f"  → spontaneous activity NONZERO — lead's H_baseline still in play.")


if __name__ == "__main__":
    main()
