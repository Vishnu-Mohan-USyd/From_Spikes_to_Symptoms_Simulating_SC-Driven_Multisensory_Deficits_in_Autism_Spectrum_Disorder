"""Stage D sanity: verify the debugger's empirical claim reproduces on the
current patched code.

Debugger claim (debug_dt/final_proof.py Fix B): with gNMDA UNCHANGED (legacy
0.05) and Fix B applied, HW@dt=0.1 ≈ HW@dt=0.05.

If this reproduces, the dt=0.05 mismatch in Stage D is caused by the gNMDA
recalibration, not by the patch implementation. If it doesn't reproduce, the
patch differs from the debugger's monkey-patch.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from debug_dt.stage_c_gnmda_sweep import measure_hw


def main():
    ckpt = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
    print(f"Stage D sanity — does Fix B + gNMDA=0.05 (legacy) give dt-invariance?")
    print()
    # Legacy gNMDA = 0.05 is what run_training assigns and what the checkpoint
    # has after training (verified via run_training:3468 -> net.gNMDA = 0.05).
    # But load_msi_model restores mutable_hparams, which may or may not include
    # gNMDA. To be safe, set it explicitly.
    gNMDA_legacy = 0.05

    # Baseline: flag OFF with legacy gNMDA at both dt
    hw_OFF_01,  _, _ = measure_hw(ckpt, dt=0.1,  nsub=100, flag=False, gNMDA=gNMDA_legacy)
    hw_OFF_005, _, _ = measure_hw(ckpt, dt=0.05, nsub=200, flag=False, gNMDA=gNMDA_legacy)
    print(f"  Flag OFF (legacy bug):")
    print(f"    dt=0.1  HW = {hw_OFF_01:.2f} ms")
    print(f"    dt=0.05 HW = {hw_OFF_005:.2f} ms   Δ = {hw_OFF_005 - hw_OFF_01:+.2f}")
    print()
    # Fix B (flag ON) with legacy gNMDA at both dt
    hw_ON_01,  _, _ = measure_hw(ckpt, dt=0.1,  nsub=100, flag=True, gNMDA=gNMDA_legacy)
    hw_ON_005, _, _ = measure_hw(ckpt, dt=0.05, nsub=200, flag=True, gNMDA=gNMDA_legacy)
    print(f"  Flag ON (Fix B), legacy gNMDA = {gNMDA_legacy}:")
    print(f"    dt=0.1  HW = {hw_ON_01:.2f} ms")
    print(f"    dt=0.05 HW = {hw_ON_005:.2f} ms   Δ = {hw_ON_005 - hw_ON_01:+.2f}")
    print()
    print(f"  If |Δ| < ~10 ms → debugger's claim reproduces with current patch.")


if __name__ == "__main__":
    main()
