"""Stage C fine sweep around gNMDA = 1.25."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from debug_dt.stage_c_gnmda_sweep import measure_hw, TARGET_HW


def main():
    ckpt = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
    print(f"Stage C fine sweep — flag ON, dt=0.1, M00 fixed-seed.")
    print(f"  Target HW = {TARGET_HW:.2f} ms")
    print()
    sweep = [1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40]
    results = {}
    print(f"  {'gNMDA':>7s}  {'HW (ms)':>10s}  {'Δ vs target':>12s}")
    for g in sweep:
        hw, _, _ = measure_hw(ckpt, dt=0.1, nsub=100, flag=True, gNMDA=g)
        results[g] = hw
        print(f"  {g:>7.3f}  {hw:>10.2f}  {hw - TARGET_HW:>+12.2f}")
    closest = min(results.keys(), key=lambda g: abs(results[g] - TARGET_HW))
    print()
    print(f"  Closest: gNMDA = {closest}, HW = {results[closest]:.2f}, "
          f"Δ = {results[closest] - TARGET_HW:+.2f} ms")
    print(f"  Full: {results}")


if __name__ == "__main__":
    main()
