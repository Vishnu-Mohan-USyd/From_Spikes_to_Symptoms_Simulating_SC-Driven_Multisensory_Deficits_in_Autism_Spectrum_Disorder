"""Step-3 validator probe — single-ckpt SBW control with CURRENT Training.py.

Loads checkpoint/msi_model_surr_10_00.pt and runs the STOCK
`compute_sbw_enhancement_persep` from SBW_test (no monkey-patches) at
separations [0, 30, 60, 80] degrees. The AGC fix lives inside
`net.reset_state`, so calling the stock API exercises the production path.

Reports pooled `mean_prob` (= fraction of trials with enhancement > 10 spikes)
at each separation. AGC bug signature: P=1.0 saturation across all seps.
AGC fix working: P drops at large separations.

Does NOT touch any cache/sbw_*_t10.npz files (fresh sim only).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from SBW_test import load_msi_model, compute_sbw_enhancement_persep

CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
DEVICE = "cuda"
SEPS = [0, 30, 60, 80]
N_TRIALS = 50
THRESH = 10.0

def main():
    print("=" * 92)
    print("Validator Step-3 SBW probe — CURRENT Training.py, stock API, no monkey-patches")
    print("=" * 92)
    print(f"  ckpt:  {CKPT}")
    print(f"  seps:  {SEPS}")
    print(f"  n_trials: {N_TRIALS}, p-fusion thresh: enh > {THRESH}")
    print()

    t0 = time.time()
    net = load_msi_model(CKPT, device=DEVICE)
    print(f"  loaded ckpt in {time.time() - t0:.1f}s; g_FFinh={float(net.g_FFinh):.4f}")
    print(f"  has _last_agc_*_t? fast={hasattr(net,'_last_agc_fast_t')} slow={hasattr(net,'_last_agc_slow_t')}")
    if hasattr(net, "_last_agc_fast_t"):
        print(f"  initial _last_agc_fast_t={float(net._last_agc_fast_t):.3f}  _last_agc_slow_t={float(net._last_agc_slow_t):.3f}")

    t0 = time.time()
    # API: compute_sbw_enhancement_persep returns (mean_enhancement, all_trial_enh)
    # all_trial_enh is list[np.ndarray(n_trials,)] per separation
    out = compute_sbw_enhancement_persep(
        net,
        separations_deg=SEPS,
        n_trials=N_TRIALS,
        intensity=1.0,
        duration=20,
    )
    if isinstance(out, tuple):
        mean_enh, all_trial_enh = out
    else:
        # in case API returns only the per-trial array
        all_trial_enh = out
        mean_enh = np.array([np.mean(x) for x in all_trial_enh]) if isinstance(out, list) else np.asarray(out).mean(axis=1)
    print(f"  sim time: {time.time() - t0:.1f}s")

    print()
    print(f"{'sep_deg':>8} {'mean_enh':>10} {'mean_prob':>11} {'std_enh':>10}")
    print("  " + "-" * 44)
    mean_probs = []
    for k, sep in enumerate(SEPS):
        trials = np.asarray(all_trial_enh[k])
        me = float(np.mean(trials))
        mp = float((trials > THRESH).mean())
        se = float(np.std(trials))
        mean_probs.append(mp)
        print(f"{sep:>8} {me:10.3f} {mp:11.3f} {se:10.3f}")

    print()
    print(f"  pooled mean_prob array: {mean_probs}")
    if all(p >= 0.99 for p in mean_probs):
        print("  ⚠ SATURATED — P=1.0 across all seps. AGC fix is NOT working for SBW.")
    elif mean_probs[0] > mean_probs[-1]:
        print(f"  ✓ Curve drops: P({SEPS[0]})={mean_probs[0]:.3f} → P({SEPS[-1]})={mean_probs[-1]:.3f}.")
        print("    AGC fix IS working for SBW (no saturation).")
    else:
        print(f"  ? Unexpected pattern: P({SEPS[0]})={mean_probs[0]:.3f} P({SEPS[-1]})={mean_probs[-1]:.3f}")

if __name__ == "__main__":
    main()
