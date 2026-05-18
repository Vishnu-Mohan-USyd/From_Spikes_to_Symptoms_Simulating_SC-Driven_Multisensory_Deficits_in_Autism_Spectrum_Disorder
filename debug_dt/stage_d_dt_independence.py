"""Stage D: verify dt-independence at dt=0.05 with flag ON, gNMDA=1.30.

Expected: HW @ dt=0.05 ≈ HW @ dt=0.1 (within ~10 ms).
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from debug_dt.stage_c_gnmda_sweep import measure_hw, TARGET_HW

GNMDA = 1.30


def main():
    ckpt = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
    print(f"Stage D dt-independence check — flag ON, gNMDA={GNMDA}, M00 fixed-seed.")
    print()
    print(f"  Reference (flag OFF, dt=0.1, gNMDA=0.05): {TARGET_HW:.2f} ms")
    print()
    hw_01, _, t_01 = measure_hw(ckpt, dt=0.1,  nsub=100, flag=True, gNMDA=GNMDA)
    hw_005, _, t_005 = measure_hw(ckpt, dt=0.05, nsub=200, flag=True, gNMDA=GNMDA)
    print(f"  dt=0.1   (nsub=100): HW = {hw_01:.2f} ms  ({t_01:.1f}s)")
    print(f"  dt=0.05  (nsub=200): HW = {hw_005:.2f} ms  ({t_005:.1f}s)")
    print(f"  ΔHW (dt0.05 - dt0.1) = {hw_005 - hw_01:+.2f} ms")
    if abs(hw_005 - hw_01) <= 10:
        print(f"  → PASS: dt-independence within ±10 ms.")
    else:
        print(f"  → FAIL: dt-dependence persists ({abs(hw_005-hw_01):.1f} ms gap).")


if __name__ == "__main__":
    main()
