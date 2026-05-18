"""Task #54 metric extraction.

Imports existing helpers from the 6 patched test scripts and prints the
specific numerical values the debugger predicted (Fano baseline/stim, MEI,
σ_A/V/AV). Does NOT modify the production scripts.

Run from the project root:
    python task54_logs/extract_metrics.py
"""
import sys
import os
import numpy as np
import torch
from pathlib import Path

# Make the project root importable so we can pull in the test modules.
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)  # so internal Path("checkpoint") resolves correctly

CKPTS = sorted(Path("checkpoint").glob("msi_model_surr_10_*.pt"))[:10]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================================================================
# 1) Fano factor — baseline (pre-stim) vs evoked (post-stim) means
# ===================================================================
print("=" * 60)
print("FANO FACTOR EXTRACTION")
print("=" * 60)
from fano_factor_test import run_fano_factor_test_bio  # noqa: E402

t, meanF, semF, meanR, semR, dt = run_fano_factor_test_bio(
    CKPTS,
    n_trials=32,
    baseline_frames=30,
    stim_frames=30,
    device=DEVICE,
)
# Convention in script: stim_onset = 30 (after the baseline frames),
# so baseline = first 30 frames, stim = next 30 frames.
fano_baseline = meanF[:30].mean()
fano_stim = meanF[30:].mean()
fano_baseline_sem = semF[:30].mean()
fano_stim_sem = semF[30:].mean()
print(f"Fano baseline (pre-stim, mean of first 30 frames): {fano_baseline:.3f} ± {fano_baseline_sem:.3f}")
print(f"Fano stim     (post-stim, mean of next  30 frames): {fano_stim:.3f} ± {fano_stim_sem:.3f}")
print(f"Mean rate baseline: {meanR[:30].mean():.4f} spikes/frame/neuron")
print(f"Mean rate stim:     {meanR[30:].mean():.4f} spikes/frame/neuron")

# ===================================================================
# 2) Inverse effectiveness — MEI at each intensity
# ===================================================================
print()
print("=" * 60)
print("INVERSE EFFECTIVENESS MEI EXTRACTION")
print("=" * 60)
from inverse_effectiveness_test import load_msi_model  # noqa: E402

# Replicated from inverse_effectiveness_test.main() (lines 620-650):
MODEL_DIR = Path("checkpoint")
INTENSITIES = np.array([0.05, .1, .2, .4, .8, 1.6], dtype=float)
LOC_DEG = 90
SIGMA_IN = 5.0
PULSE_LEN = 10
N_FRAMES = 20


def integrated_spikes(net, cond, intensity):
    """Return Σ MSI spikes for one trial (copied from inv_eff main())."""
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


models = sorted(MODEL_DIR.glob("msi_model_surr_10_*.pt"))
resp_A = np.zeros((len(models), INTENSITIES.size))
resp_V = np.zeros_like(resp_A)
resp_AV = np.zeros_like(resp_A)

for m_i, path in enumerate(models):
    for j, I in enumerate(INTENSITIES):
        # Task #60 principled fix mirror: reload per intensity so each intensity
        # starts from the same trained-network state (no AGC/plastic drift).
        net = load_msi_model(path, device=DEVICE)
        setattr(net, "gNMDA", 1.30)  # task #42 fix mirror
        resp_A[m_i, j] = integrated_spikes(net, "A", I)
        resp_V[m_i, j] = integrated_spikes(net, "V", I)
        resp_AV[m_i, j] = integrated_spikes(net, "B", I)
        del net
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

max_uni = np.maximum(resp_A, resp_V)
mei = (resp_AV - max_uni) / np.where(max_uni == 0, 1, max_uni)
mei_mean = mei.mean(0)
mei_sem = mei.std(0, ddof=1) / np.sqrt(mei.shape[0])
print(f"INTENSITIES  = {list(INTENSITIES)}")
print(f"MEI mean     = {[f'{x:.3f}' for x in mei_mean]}")
print(f"MEI sem      = {[f'{x:.3f}' for x in mei_sem]}")
print(f"resp_A mean  = {[f'{x:.2f}' for x in resp_A.mean(0)]}")
print(f"resp_V mean  = {[f'{x:.2f}' for x in resp_V.mean(0)]}")
print(f"resp_AV mean = {[f'{x:.2f}' for x in resp_AV.mean(0)]}")

# ===================================================================
# 3) Precision histogram — σ values for A / V / AV (control)
# ===================================================================
print()
print("=" * 60)
print("PRECISION HISTOGRAM σ EXTRACTION (control gNMDA=1.30)")
print("=" * 60)
from precision_hist_test import (  # noqa: E402
    load_msi_model as load_msi_model_ph,
    compute_hybrid_sensitivity_fast,
)

paths_ph = sorted(Path("checkpoint").glob("msi_model_surr_10_*.pt"))[:10]
all_err_A = []
all_err_V = []
all_err_AV = []
for p in paths_ph:
    rep_net = load_msi_model_ph(p)
    setattr(rep_net, "gNMDA", 1.30)  # task #42 fix mirror
    out = compute_hybrid_sensitivity_fast(rep_net)
    all_err_A.extend(out["err_A"])
    all_err_V.extend(out["err_V"])
    all_err_AV.extend(out["err_AV"])
    del rep_net
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

sigma_A = float(np.std(all_err_A, ddof=1))
sigma_V = float(np.std(all_err_V, ddof=1))
sigma_AV = float(np.std(all_err_AV, ddof=1))
print(f"σ_A  = {sigma_A:.2f}°  (paper target: 8.47)")
print(f"σ_V  = {sigma_V:.2f}°  (paper target: 9.59)")
print(f"σ_AV = {sigma_AV:.2f}° (paper target: 7.08)")

# MLE prediction for σ_AV from σ_A and σ_V (optimal cue integration)
sigma_mle = (sigma_A * sigma_V) / np.sqrt(sigma_A ** 2 + sigma_V ** 2)
print(f"σ_AV(MLE prediction from σ_A,σ_V) = {sigma_mle:.2f}°")

print()
print("=" * 60)
print("DONE")
print("=" * 60)
