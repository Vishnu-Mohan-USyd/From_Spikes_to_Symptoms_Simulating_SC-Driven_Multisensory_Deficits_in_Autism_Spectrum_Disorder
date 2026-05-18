"""Task #57 — fast 3-ckpt confirmation of the FIX (AGC frozen + plast off).

The full 10-ckpt run with CONTROL+FIX (H_invEff_ensemble_fix.py) ran out
of time mid-CONTROL.  Focus here on validating ONLY the FIX cell on 3 ckpts,
to confirm M00's FALLING-MEI result generalizes.
"""
from __future__ import annotations
import sys, time
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


def main():
    paths = sorted(MODEL_DIR.glob("msi_model_surr_10_*.pt"))[:4]  # 4 ckpts
    torch.manual_seed(0); np.random.seed(0)

    print(f"\n{'='*88}")
    print(f"Task #57 — fast FIX cell on {len(paths)} ckpts (AGC frozen + plast off, g=1.30)")
    print(f"{'='*88}\n")

    nM = len(paths); nI = len(INTENSITIES)
    rA = np.zeros((nM, nI)); rV = np.zeros((nM, nI)); rAV = np.zeros((nM, nI))
    t0 = time.time()
    for m_i, p in enumerate(paths):
        net = load_msi_model(p, device=DEVICE)
        net.gNMDA = 1.30
        net.g_FFinh = 0.6
        net.freeze_g_FFinh = True
        net.plasticity_enabled = False
        for j, I in enumerate(INTENSITIES):
            rA[m_i, j]  = integrated_spikes(net, "A", I)
            rV[m_i, j]  = integrated_spikes(net, "V", I)
            rAV[m_i, j] = integrated_spikes(net, "B", I)
        del net; torch.cuda.empty_cache()
        meis_m = (rAV[m_i] - np.maximum(rA[m_i], rV[m_i])) / np.maximum(rA[m_i], rV[m_i])
        print(f"  M{m_i:02d}  R_A={rA[m_i]}  MEI={[f'{m:+.2f}' for m in meis_m]}  el={time.time()-t0:.0f}s",
              flush=True)

    mA = rA.mean(0); mV = rV.mean(0); mAV = rAV.mean(0)
    mei = (rAV - np.maximum(rA, rV)) / np.maximum(rA, rV)
    mei_m = mei.mean(0); mei_s = mei.std(0, ddof=1) / np.sqrt(nM)

    print(f"\n{'='*88}")
    print(f"ENSEMBLE RESULT (FIX cell, n={nM})")
    print(f"{'='*88}")
    print(f"  {'I':>6s} | {'R_A':>8s} {'R_V':>8s} {'R_AV':>8s} | {'MEI':>8s} {'±SEM':>6s}")
    for j, I in enumerate(INTENSITIES):
        print(f"  {I:>6.3f} | {mA[j]:>8.1f} {mV[j]:>8.1f} {mAV[j]:>8.1f} | "
              f"{mei_m[j]:>+8.3f} {mei_s[j]:>6.3f}")

    diffs = np.diff(mei_m)
    n_down = (diffs < -0.01).sum(); n_up = (diffs > 0.01).sum()
    dirn = "FALLING ✓ matches paper" if n_down >= 4 else "NOT FALLING"

    from scipy.stats import spearmanr
    rho, p = spearmanr(INTENSITIES, mei_m)
    print()
    print(f"  Direction: {dirn}")
    print(f"  Spearman ρ(intensity, MEI) = {rho:.3f}  p={p:.3g}  (tolerance ρ < -0.8)")
    print(f"  MEI[I=0.05] = {mei_m[0]:+.3f}  vs paper +1.04")
    print(f"  MEI[I=1.60] = {mei_m[-1]:+.3f}  vs paper +0.78")
    print(f"  validator pass criterion (docs/validation_protocol.md):")
    print(f"    MEI(0.05) > MEI(1.6):  {mei_m[0]:.3f} > {mei_m[-1]:.3f}  → {mei_m[0] > mei_m[-1]}")
    print(f"    Spearman ρ < -0.8:      ρ = {rho:.3f}      → {rho < -0.8}")


if __name__ == "__main__":
    main()
