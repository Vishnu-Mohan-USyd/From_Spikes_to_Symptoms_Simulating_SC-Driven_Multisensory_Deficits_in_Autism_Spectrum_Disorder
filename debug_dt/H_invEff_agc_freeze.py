"""Task #57 H_AGC — CAUSAL PROOF: g_FFinh drift via runaway AGC is the methodology bug.

Evidence:
  * The current inverse_effectiveness_test.py + integrated_spikes() does NOT
    set net.freeze_g_FFinh = True.  Default is False (Training.py:1464).
  * AGC (Training.py:2461-2476) updates self.g_FFinh every substep based on
    running means of exc/inh.  It is NOT reset by reset_state.
  * Therefore g_FFinh DRIFTS across trials within a single net load — values
    of g_FFinh at trial N depend on all prior trials.
  * Test confirmed: running the IDENTICAL integrated_spikes mechanic but in
    different orders gave wildly different totals (R_A I=1.6 = 347 vs 1969,
    a 5.7× difference) for the same M00 ckpt.  The only difference is the
    order of calls → therefore the only stateful drift is between calls.

This script tests:
  CELL A (control, current default): freeze_g_FFinh = False (AGC drifts)
  CELL B (proposed fix):             freeze_g_FFinh = True  (AGC frozen)

For each cell, measure:
  - R_A, R_V, R_AV at each intensity
  - MEI per intensity
  - Direction of MEI (rising / falling / flat)
  - g_FFinh trajectory across the test (start vs end)
  - reproducibility: run twice, see if result is identical when AGC frozen

Goal: prove that freezing AGC (a) gives reproducible results and (b) restores
the paper's monotonically-decreasing MEI direction.
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
def integrated_spikes(net, cond, intensity, *, track_agc=False):
    """Replica of inverse_effectiveness_test.integrated_spikes()."""
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
    agc_trace = []
    if track_agc:
        agc_trace.append(float(net.g_FFinh))
    for t in range(N_FRAMES):
        *_, sSum = net.update_all_layers_batch(
            xA[t:t + 1], xV[t:t + 1], return_spike_sum=True
        )
        pop += sSum.sum().item()
        if track_agc:
            agc_trace.append(float(net.g_FFinh))
    return pop, agc_trace


def run_cell(*, freeze_agc, label, ckpt_path, do_twice=False, reseed_agc=None):
    """One full A/V/B × intensities sweep using the literal integrated_spikes API."""
    print(f"\n  {label}")
    print(f"  freeze_g_FFinh = {freeze_agc}")

    def _one_pass():
        net = load_msi_model(ckpt_path, device=DEVICE)
        setattr(net, "gNMDA", 1.30)
        if reseed_agc is not None:
            net.g_FFinh = float(reseed_agc)
        if freeze_agc:
            net.freeze_g_FFinh = True
        else:
            net.freeze_g_FFinh = False
        g0 = float(net.g_FFinh)
        rA = {}; rV = {}; rB = {}
        for I in INTENSITIES:
            rA[I], _ = integrated_spikes(net, "A", I)
            rV[I], _ = integrated_spikes(net, "V", I)
            rB[I], _ = integrated_spikes(net, "B", I)
        g_end = float(net.g_FFinh)
        del net; torch.cuda.empty_cache()
        return rA, rV, rB, g0, g_end

    rA1, rV1, rB1, g0_1, gend_1 = _one_pass()
    print(f"   pass 1:  g_FFinh start={g0_1:.4f}  end={gend_1:.4f}  Δ={gend_1-g0_1:+.4f}")

    if do_twice:
        rA2, rV2, rB2, g0_2, gend_2 = _one_pass()
        print(f"   pass 2:  g_FFinh start={g0_2:.4f}  end={gend_2:.4f}  Δ={gend_2-g0_2:+.4f}")
        # check reproducibility
        equal_A = all(abs(rA1[I] - rA2[I]) < 0.5 for I in INTENSITIES)
        equal_V = all(abs(rV1[I] - rV2[I]) < 0.5 for I in INTENSITIES)
        equal_B = all(abs(rB1[I] - rB2[I]) < 0.5 for I in INTENSITIES)
        print(f"   reproducible (pass1==pass2)?  A:{equal_A} V:{equal_V} B:{equal_B}")
        # average over both passes
        rA = {I: 0.5*(rA1[I]+rA2[I]) for I in INTENSITIES}
        rV = {I: 0.5*(rV1[I]+rV2[I]) for I in INTENSITIES}
        rB = {I: 0.5*(rB1[I]+rB2[I]) for I in INTENSITIES}
    else:
        rA, rV, rB = rA1, rV1, rB1

    print(f"   {'I':>6s} | {'R_A':>7s} {'R_V':>7s} {'R_AV':>7s} {'max':>7s} {'MEI':>7s}")
    meis = []
    for I in INTENSITIES:
        d = max(rA[I], rV[I])
        mei = (rB[I] - d) / d if d > 0 else float('nan')
        meis.append(mei)
        print(f"   {I:>6.3f} | {rA[I]:>7.1f} {rV[I]:>7.1f} {rB[I]:>7.1f} {d:>7.1f} {mei:>+7.3f}")
    diffs = np.diff(meis)
    n_down = (diffs < -0.01).sum(); n_up = (diffs > 0.01).sum()
    if n_down >= 4:
        dirn = "FALLING ✓ (matches paper)"
    elif n_up >= 4:
        dirn = "RISING (bad; opposite of paper)"
    else:
        dirn = "FLAT/MIXED"
    print(f"   MEI direction: {dirn}")
    print(f"   MEI[low]/MEI[high] = {meis[0]/meis[-1] if meis[-1]!=0 else float('nan'):.3f}  "
          f"(paper ≈ 1.04/0.78 = 1.33)")
    return rA, rV, rB, meis


def main():
    ckpt_path = MODEL_DIR / "msi_model_surr_10_00.pt"
    torch.manual_seed(0); np.random.seed(0)

    print(f"\n{'='*88}")
    print(f"Task #57 H_AGC — g_FFinh drift causal proof on M00")
    print(f"{'='*88}")
    print(f"  ckpt: {ckpt_path.name}")
    print(f"  integrated_spikes is bit-identical to inverse_effectiveness_test.py")
    print(f"  Differ only by: freeze_g_FFinh flag")
    print()
    print("─" * 88)
    print(" CELL A — control (default freeze_g_FFinh = False, current production)")
    print("─" * 88)
    rA_A, rV_A, rB_A, meis_A = run_cell(
        freeze_agc=False, label="A: default (AGC drifts)",
        ckpt_path=ckpt_path, do_twice=True,
    )

    print()
    print("─" * 88)
    print(" CELL B — proposed fix: freeze AGC during eval")
    print("─" * 88)
    rA_B, rV_B, rB_B, meis_B = run_cell(
        freeze_agc=True, label="B: frozen AGC",
        ckpt_path=ckpt_path, do_twice=True,
    )

    # Optional: also try with g_FFinh reseeded to a canonical post-train value
    print()
    print("─" * 88)
    print(" CELL C — frozen AGC + g_FFinh reseeded to 0.6 (Training.py:3559 cal default)")
    print("─" * 88)
    rA_C, rV_C, rB_C, meis_C = run_cell(
        freeze_agc=True, label="C: frozen AGC, g_FFinh=0.6",
        ckpt_path=ckpt_path, do_twice=True, reseed_agc=0.6,
    )

    print()
    print(f"{'='*88}")
    print(f"CAUSAL VERDICT")
    print(f"{'='*88}")
    print(f"  Cell A (AGC drifts):  MEI = {[f'{m:+.3f}' for m in meis_A]}")
    print(f"  Cell B (AGC frozen):  MEI = {[f'{m:+.3f}' for m in meis_B]}")
    print(f"  Cell C (frozen+0.6):  MEI = {[f'{m:+.3f}' for m in meis_C]}")
    print()
    print(f"  Paper target:         MEI = +1.04 → +0.78 (DECREASING, log-axis)")
    print()
    print("  If B has consistent reproducibility AND A does not, AGC drift is the bug.")
    print("  If B's MEI direction matches paper, the methodology fix is freeze_g_FFinh=True.")


if __name__ == "__main__":
    main()
