"""Task #56: Recalibrate SBW enhancement threshold post-task-#40 measurement fix.

Strategy: run all 5 conditions × 10 models once, caching per-trial enhancement
values. Then post-hoc sweep threshold values and compute the resulting
SBW HW for each (threshold, condition). Pick the threshold value that gives
the best overall fit to paper HW values.

Paper-target SBW HW (°):
    control: 24.3
    ff_inhibition: 27.7
    adaptation: 29.4
    nmda: 13.1
    nmda_increase: 30.0

Run from project root:
    python task56_logs/calibrate_sbw_threshold.py
"""
import os
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)

from SBW_test import load_msi_model, compute_sbw_enhancement_persep  # noqa
from replot_all_cosmetic import subtract_control_floor, fit_sbw_pedestal  # noqa
from TBW_test import _find_crossings  # noqa

# Conditions from generate_all_fresh.py (post-task-#58: freeze flags included)
CONDITIONS = {
    "control": lambda n: (setattr(n, "gNMDA", 1.30)
                          or setattr(n, "freeze_g_FFinh", True)
                          or setattr(n, "plasticity_enabled", False)),
    "ff_inhibition": lambda n: (setattr(n, "gNMDA", 1.30)
                                or setattr(n, "freeze_g_FFinh", True)
                                or setattr(n, "plasticity_enabled", False)
                                or setattr(n, "pv_nmda", 0.8)
                                or setattr(n, "targ_ratio", 0.8)),
    "adaptation": lambda n: (setattr(n, "gNMDA", 1.30)
                             or setattr(n, "freeze_g_FFinh", True)
                             or setattr(n, "plasticity_enabled", False)
                             or setattr(n, "aM", 0.001)
                             or setattr(n, "bM", 0.2)
                             or setattr(n, "cM", -60.0)
                             or setattr(n, "dM", 0.01)),
    "nmda": lambda n: (setattr(n, "gNMDA", 0.50)
                       or setattr(n, "freeze_g_FFinh", True)
                       or setattr(n, "plasticity_enabled", False)),
    "nmda_increase": lambda n: (setattr(n, "gNMDA", 5.00)
                                or setattr(n, "freeze_g_FFinh", True)
                                or setattr(n, "plasticity_enabled", False)),
}
COND_ORDER = ["control", "ff_inhibition", "adaptation", "nmda", "nmda_increase"]
PAPER_HW = {
    "control": 24.3,
    "ff_inhibition": 27.7,
    "adaptation": 29.4,
    "nmda": 13.1,
    "nmda_increase": 30.0,
}

BASE = Path("checkpoint")
MODELS = [BASE / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
SBW_SEPARATIONS = tuple(range(-80, 85, 5))  # 33 values
N_TRIALS = 50

CACHE = Path("task58_logs/raw_enh_cache_freeze.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────
# Step 1: Simulate all 5 conditions × 10 models, cache per-trial enhancement
# ─────────────────────────────────────────────────────────────────────
def run_all_simulations():
    """Run 5 conditions × 10 models, return dict[cond] -> list of model curves.
    Each model's curve is a list (one per separation) of np.ndarray (n_trials,).
    """
    if CACHE.exists():
        print(f"Loading cached enhancement values from {CACHE}")
        with open(CACHE, "rb") as f:
            return pickle.load(f)

    all_raw = {}
    for cname in COND_ORDER:
        print(f"\n=== Running {cname} ===")
        t0 = time.time()
        modify_net = CONDITIONS[cname]
        per_model_list = []
        for i, path in enumerate(MODELS):
            net = load_msi_model(path, device=DEVICE)
            modify_net(net)
            mean_enh, all_trial_enh = compute_sbw_enhancement_persep(
                net,
                separations_deg=SBW_SEPARATIONS,
                n_trials=N_TRIALS,
                intensity=1,
                duration=20,
            )
            # Convert torch tensors -> numpy if needed
            converted = []
            for arr in all_trial_enh:
                if torch.is_tensor(arr):
                    converted.append(arr.detach().cpu().numpy())
                else:
                    converted.append(np.asarray(arr))
            per_model_list.append(converted)
            del net
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            print(f"  model {i}: done in {time.time()-t0:.1f}s cum")
        all_raw[cname] = per_model_list
        print(f"  {cname} total: {time.time()-t0:.1f}s")

    CACHE.parent.mkdir(exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(all_raw, f)
    print(f"Cached enhancement values to {CACHE}")
    return all_raw


# ─────────────────────────────────────────────────────────────────────
# Step 2: For a given threshold, compute pooled dict per condition
# (mimics run_spatial_binding_across_models post-processing)
# ─────────────────────────────────────────────────────────────────────
def pool_for_threshold(all_raw, threshold):
    """Apply threshold to per-trial enhancement values, build pooled dicts."""
    sep_arr = np.asarray(SBW_SEPARATIONS, float)
    n_sep = len(SBW_SEPARATIONS)

    # mirror-separation mapping
    mag_to_cols = {}
    for i, s in enumerate(SBW_SEPARATIONS):
        mag = abs(s)
        mag_to_cols.setdefault(mag, []).append(i)
    mags = np.array(sorted(mag_to_cols.keys()), dtype=float)
    n_mags = len(mags)

    pooled_per_cond = {}
    for cname, per_model in all_raw.items():
        n_models = len(per_model)
        curves = np.zeros((n_models, n_sep))
        for mi, per_sep_trials in enumerate(per_model):
            for si, trial_enh in enumerate(per_sep_trials):
                curves[mi, si] = (trial_enh > threshold).mean()

        # symmetrize
        curves_sym = np.zeros((n_models, n_mags))
        for j, mag in enumerate(mags):
            cols = mag_to_cols[mag]
            curves_sym[:, j] = curves[:, cols].mean(axis=1)
        mean_onesided = curves_sym.mean(0)
        all_onesided = curves_sym

        # mirror back
        full_sep = np.concatenate([-mags[::-1], mags[1:]])
        full_mean = np.concatenate([mean_onesided[::-1], mean_onesided[1:]])
        full_sem_parts = all_onesided.std(0, ddof=1) / np.sqrt(n_models)
        full_sem = np.concatenate([full_sem_parts[::-1], full_sem_parts[1:]])
        full_all = np.concatenate([all_onesided[:, ::-1], all_onesided[:, 1:]], axis=1)

        pooled_per_cond[cname] = {
            "separations_deg": full_sep,
            "mean_prob": full_mean,
            "sem_prob": full_sem,
            "all_prob": full_all,
        }
    return pooled_per_cond


def hw_for_pooled(pooled):
    """Apply pedestal fit + 50% crossings to extract HW."""
    try:
        xs_fit, ys_fit, popt, r2 = fit_sbw_pedestal(pooled)
        cross = _find_crossings(xs_fit, ys_fit, 0.5)
        if len(cross) >= 2:
            return (cross[-1] - cross[0]) / 2.0, r2
        else:
            return 0.0, r2
    except Exception as e:
        return float("nan"), 0.0


def sweep(all_raw, thresholds):
    print("\n" + "=" * 70)
    print(f"{'Threshold':>10s}", end="")
    for c in COND_ORDER:
        print(f" {c[:8]:>9s}", end="")
    print(f" {'L1err':>8s} {'L2err':>8s}")
    print("=" * 70)

    paper_vec = np.array([PAPER_HW[c] for c in COND_ORDER])

    best_T = None
    best_err = float("inf")
    rows = []
    for T in thresholds:
        pooled = pool_for_threshold(all_raw, T)
        # subtract control floor (mimics generate_all_fresh.py post-processing)
        try:
            pooled_corr, floor = subtract_control_floor(pooled)
        except Exception as e:
            print(f"{T:>10.2f}  ERROR subtract_floor: {e}")
            continue
        hws = []
        for c in COND_ORDER:
            hw, r2 = hw_for_pooled(pooled_corr[c])
            hws.append(hw)
        hws = np.array(hws)
        l1 = float(np.mean(np.abs(hws - paper_vec)))
        l2 = float(np.sqrt(np.mean((hws - paper_vec) ** 2)))
        rows.append((T, hws.tolist(), l1, l2, floor))

        print(f"{T:>10.2f}", end="")
        for h in hws:
            print(f" {h:>9.1f}", end="")
        print(f" {l1:>8.2f} {l2:>8.2f}  (floor={floor:.4f})")

        # use L2 for picking best
        if l2 < best_err:
            best_err = l2
            best_T = T

    print("=" * 70)
    print(f"Paper:    ", end="")
    for c in COND_ORDER:
        print(f" {PAPER_HW[c]:>9.1f}", end="")
    print()
    print(f"\nBest threshold (L2 across all 5 conditions): T = {best_T:.2f}")
    return best_T, rows


if __name__ == "__main__":
    all_raw = run_all_simulations()

    print("\n" + "=" * 70)
    print("COARSE SWEEP (post-#58 freeze regime — team-lead recommended range)")
    print("=" * 70)
    coarse = [0, 10, 50, 100, 150, 200, 300, 500, 800, 1500]
    best_T_coarse, _ = sweep(all_raw, coarse)

    # Refine around the best
    print("\n" + "=" * 70)
    print("FINE SWEEP (around best from coarse)")
    print("=" * 70)
    fine_low = best_T_coarse * 0.3 if best_T_coarse > 0 else 1.0
    fine_high = best_T_coarse * 3.0 if best_T_coarse > 0 else 100.0
    fine = list(np.linspace(fine_low, fine_high, 11))
    best_T_fine, _ = sweep(all_raw, fine)

    print("\n" + "=" * 70)
    print(f"FINAL recommended threshold: {best_T_fine:.2f}")
    print("=" * 70)
