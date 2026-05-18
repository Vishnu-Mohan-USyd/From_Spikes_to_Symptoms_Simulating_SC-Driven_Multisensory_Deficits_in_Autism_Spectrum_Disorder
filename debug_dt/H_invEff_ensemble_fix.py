"""Task #57 — final ensemble validation of the proposed fix.

On M00:
  control (AGC drifts, plast on, g=1.30) MEI = +0.23/+0.36/+0.47/+0.50/+0.56/+0.59 → RISING
  fix     (AGC frozen, plast off, g=1.30) MEI = +0.89/+0.93/+0.64/+0.52/+0.25/-0.06 → FALLING
  paper                                    MEI ≈ +1.04 → +0.78

Now validate on the 10-ckpt ensemble (matching the production test paradigm).
Report ensemble mean + SEM for both control and fix.  If fix gives ensemble
mean MEI that decreases monotonically, the fix is validated.

Time budget: ~10 min on cuda.  Single-pass, no repetition.
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


def run_ensemble(*, label, ckpt_paths, freeze_agc, disable_plast, g_ff_seed=None):
    """Returns (resp_A, resp_V, resp_AV) shape (n_ckpts, n_intensities)."""
    print(f"\n  === {label} ===  (freeze_agc={freeze_agc}, disable_plast={disable_plast})")
    nM = len(ckpt_paths); nI = len(INTENSITIES)
    resp_A  = np.zeros((nM, nI))
    resp_V  = np.zeros((nM, nI))
    resp_AV = np.zeros((nM, nI))
    t0 = time.time()
    for m_i, p in enumerate(ckpt_paths):
        net = load_msi_model(p, device=DEVICE)
        net.gNMDA = 1.30
        if g_ff_seed is not None:
            net.g_FFinh = float(g_ff_seed)
        if freeze_agc:
            net.freeze_g_FFinh = True
        if disable_plast:
            net.plasticity_enabled = False
        for j, I in enumerate(INTENSITIES):
            resp_A[m_i, j]  = integrated_spikes(net, "A", I)
            resp_V[m_i, j]  = integrated_spikes(net, "V", I)
            resp_AV[m_i, j] = integrated_spikes(net, "B", I)
        del net; torch.cuda.empty_cache()
        if (m_i + 1) % 2 == 0 or m_i == nM - 1:
            print(f"     ckpt {m_i+1}/{nM}  el={time.time()-t0:.0f}s")
    return resp_A, resp_V, resp_AV


def summarize(label, rA, rV, rAV):
    """Print per-intensity mean+SEM of MEI + R_A / R_V / R_AV."""
    nM = rA.shape[0]
    mA = rA.mean(0); sA = rA.std(0, ddof=1)/np.sqrt(nM)
    mV = rV.mean(0); sV = rV.std(0, ddof=1)/np.sqrt(nM)
    mAV = rAV.mean(0); sAV = rAV.std(0, ddof=1)/np.sqrt(nM)
    mei = (rAV - np.maximum(rA, rV)) / np.where(np.maximum(rA, rV) == 0, 1, np.maximum(rA, rV))
    mei_m = mei.mean(0); mei_s = mei.std(0, ddof=1)/np.sqrt(nM)

    print(f"\n  === {label} ===")
    print(f"    {'I':>6s} | {'R_A_mean':>10s} {'R_V_mean':>10s} {'R_AV_mean':>10s} | {'MEI mean':>10s} {'MEI sem':>8s}")
    for j, I in enumerate(INTENSITIES):
        print(f"    {I:>6.3f} | {mA[j]:>10.1f} {mV[j]:>10.1f} {mAV[j]:>10.1f} | "
              f"{mei_m[j]:>+10.3f} {mei_s[j]:>8.3f}")

    diffs = np.diff(mei_m)
    n_down = (diffs < -0.01).sum(); n_up = (diffs > 0.01).sum()
    if n_down >= 4:
        dirn = "FALLING ✓ matches paper"
    elif n_up >= 4:
        dirn = "RISING (inverted)"
    else:
        dirn = "NON-MONOTONIC"
    from scipy.stats import spearmanr
    rho, p = spearmanr(INTENSITIES, mei_m)
    print(f"    direction: {dirn}")
    print(f"    Spearman ρ(intensity, MEI) = {rho:.3f}  p={p:.3g}    [tolerance: ρ < -0.8]")
    print(f"    MEI[low]/MEI[high] = {mei_m[0]/mei_m[-1] if mei_m[-1]!=0 else float('nan'):.2f}")
    return mei_m


def main():
    ckpt_paths = sorted(MODEL_DIR.glob("msi_model_surr_10_*.pt"))
    torch.manual_seed(0); np.random.seed(0)
    print(f"\n{'='*88}")
    print(f"Task #57 — FINAL: ensemble validation of fix (AGC frozen + plast off)")
    print(f"{'='*88}")
    print(f"  ckpts: {len(ckpt_paths)}")
    print(f"  paper target: MEI = +1.04 (I=0.05) → +0.78 (I=1.6), monotonically DECREASING")
    print(f"  validation tolerance (docs/validation_protocol.md): Spearman ρ < -0.8")

    # CONTROL: current production methodology
    t0 = time.time()
    rA_ctl, rV_ctl, rAV_ctl = run_ensemble(
        label="CONTROL (current production: AGC drifts, plast on)",
        ckpt_paths=ckpt_paths, freeze_agc=False, disable_plast=False,
    )
    print(f"  ctl total: {time.time()-t0:.0f}s")

    # FIX
    t0 = time.time()
    rA_fix, rV_fix, rAV_fix = run_ensemble(
        label="FIX (proposed: AGC frozen 0.6, plast off)",
        ckpt_paths=ckpt_paths, freeze_agc=True, disable_plast=True, g_ff_seed=0.6,
    )
    print(f"  fix total: {time.time()-t0:.0f}s")

    print(f"\n{'='*88}")
    print(f"ENSEMBLE RESULTS")
    print(f"{'='*88}")
    mei_ctl = summarize("CONTROL", rA_ctl, rV_ctl, rAV_ctl)
    mei_fix = summarize("FIX",     rA_fix, rV_fix, rAV_fix)

    print()
    print(f"{'='*88}")
    print(f"FINAL VERDICT")
    print(f"{'='*88}")
    print(f"  Control MEI direction: {'RISING (inverted)' if mei_ctl[-1] > mei_ctl[0] else 'OK'}")
    print(f"  Fix     MEI direction: {'FALLING ✓ matches paper' if mei_fix[-1] < mei_fix[0] else 'NOT FIXED'}")
    print()
    print(f"  Paper target:           +1.04 → +0.78")
    print(f"  Control:                {mei_ctl[0]:+.3f} → {mei_ctl[-1]:+.3f}")
    print(f"  Fix:                    {mei_fix[0]:+.3f} → {mei_fix[-1]:+.3f}")


if __name__ == "__main__":
    main()
