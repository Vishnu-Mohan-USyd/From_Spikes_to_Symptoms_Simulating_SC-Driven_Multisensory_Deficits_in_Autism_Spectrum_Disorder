"""Task #138 E5: Toggle `dt_correct_nmda=False` on CURRENT Training.py.

If the dt_correct_nmda code path (post-pristine) is the carry-over difference
that breaks SBW, then forcing dt_correct_nmda=False should restore pristine-
like A_roi spikes (~64) instead of the current ~0.66.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from SBW_test import load_msi_model, compute_sbw_enhancement_persep

CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
DEVICE = "cuda"


def probe(seps, label, modifier=None):
    net = load_msi_model(CKPT, device=DEVICE)
    if modifier is not None:
        modifier(net)
    print(f"\n--- {label}  dt_correct_nmda={getattr(net, 'dt_correct_nmda', 'MISSING')} ---")
    mean_enh, all_trial_enh = compute_sbw_enhancement_persep(
        net, separations_deg=seps, n_trials=50, intensity=1.0, duration=20,
    )
    p_fusion = np.array([(t > 10.0).mean() for t in all_trial_enh])
    print(f"{'sep':>5} {'mean_enh':>10} {'p(enh>10)':>11}")
    for s, m, p in zip(seps, mean_enh, p_fusion):
        print(f"{s:>5} {m:10.2f} {p:11.3f}")
    # report HW
    # symmetric — average mirror separations
    sym = {}
    for i, s in enumerate(seps):
        sym.setdefault(abs(s), []).append(p_fusion[i])
    mags = sorted(sym.keys())
    p_sym = [np.mean(sym[m]) for m in mags]
    # find crossing of 0.5
    cross = None
    for i in range(len(mags) - 1):
        if (p_sym[i] - 0.5) * (p_sym[i + 1] - 0.5) < 0:
            t = (0.5 - p_sym[i]) / (p_sym[i + 1] - p_sym[i])
            cross = mags[i] + t * (mags[i + 1] - mags[i])
            break
    print(f"HW (one-sided 0.5 crossing): {cross}")


def main():
    seps = list(range(-80, 85, 5))

    print("=" * 90)
    print("E5: CURRENT Training.py with dt_correct_nmda toggled")
    print("=" * 90)

    probe(seps, "CURRENT (dt_correct_nmda=True, default)", modifier=None)

    probe(seps, "CURRENT (dt_correct_nmda=False forced)",
          modifier=lambda n: setattr(n, "dt_correct_nmda", False))


if __name__ == "__main__":
    main()
