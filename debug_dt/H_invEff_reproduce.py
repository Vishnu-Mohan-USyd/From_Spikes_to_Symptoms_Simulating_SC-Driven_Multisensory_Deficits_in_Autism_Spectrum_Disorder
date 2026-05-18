"""ISSUE — MEI inverse-effectiveness direction inverted.

PHASE 1: REPRODUCE the inversion on a single fixed checkpoint and probe
baseline-floor / time-window candidates BEFORE committing to a full hypothesis set.

We use the EXACT integrated_spikes mechanic from inverse_effectiveness_test.py
(single-batch, sum-all-MSI-spikes-over-N_FRAMES, gNMDA=1.30 override).  Then we add
instrumentation:

  - intensity = 0 (TRUE BASELINE — no stimulus at all)
  - per-window breakdown: spikes pre-stim (none here), during stim (frames 0..PULSE_LEN),
    post-stim (frames PULSE_LEN..N_FRAMES)
  - spatial restriction:  only neurons within ±20 idx of LOC_DEG=90 center
  - per-frame spike trace so we can see the temporal pattern

This script reads ONLY from M00 (single ckpt) but tests the full 6-intensity sweep
plus baseline.  No production code is modified.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Training import MultiBatchAudVisMSINetworkTime

# ──────────────────────────────────────────────────────────────────────
# Setup identical to inverse_effectiveness_test.py main()
# ──────────────────────────────────────────────────────────────────────
MODEL_DIR = ROOT / "checkpoint"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOC_DEG = 90
SIGMA_IN = 5.0
PULSE_LEN = 10
N_FRAMES = 20

INTENSITIES = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]


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
def integrated_spikes_instrumented(net, cond, intensity):
    """Replica of integrated_spikes() with per-frame and spatial breakdown."""
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
    per_frame_total = np.zeros(N_FRAMES, dtype=np.float64)
    per_frame_local = np.zeros(N_FRAMES, dtype=np.float64)  # within ±20 idx of center
    centre_idx = int(round(LOC_DEG * (net.n - 1) / (net.space_size - 1)))
    lo = max(0, centre_idx - 20)
    hi = min(net.n, centre_idx + 21)
    for t in range(N_FRAMES):
        *_, sSum = net.update_all_layers_batch(
            xA[t:t + 1], xV[t:t + 1], return_spike_sum=True
        )
        per_frame_total[t] = sSum.sum().item()
        per_frame_local[t] = sSum[0, lo:hi].sum().item()
    return per_frame_total, per_frame_local


def main():
    ckpt_path = MODEL_DIR / "msi_model_surr_10_00.pt"
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"\n{'='*78}")
    print("ISSUE B (Task #57) — MEI inversion REPRODUCER on M00")
    print(f"{'='*78}")
    print(f"  ckpt: {ckpt_path.name}")
    print(f"  PULSE_LEN={PULSE_LEN}  N_FRAMES={N_FRAMES}")
    print(f"  intensities: {INTENSITIES}")
    print(f"  measuring: total spikes, AND local ±20 idx around centre")
    print()

    net = load_msi_model(ckpt_path, device=DEVICE)
    setattr(net, "gNMDA", 1.30)  # matches inverse_effectiveness_test.py line 660
    print(f"  net.gNMDA = {net.gNMDA}  n={net.n}  space={net.space_size}")
    print(f"  dt_correct_nmda = {getattr(net, 'dt_correct_nmda', 'absent')}")
    print()

    # ── table headers ─────────────────────────────────────────────────
    print(f"  {'I':>6s} | "
          f"{'R_A_tot':>9s} {'R_V_tot':>9s} {'R_AV_tot':>9s} {'MEI_tot':>8s} | "
          f"{'R_A_loc':>8s} {'R_V_loc':>8s} {'R_AVloc':>8s} {'MEI_loc':>8s} | "
          f"{'A_pre':>6s} {'A_dur':>6s} {'A_post':>6s}")
    print(f"  {'-'*6} | {'-'*39} | {'-'*36} | {'-'*22}")

    results = []
    for I in INTENSITIES:
        rA_t, rA_l = integrated_spikes_instrumented(net, "A", I)
        rV_t, rV_l = integrated_spikes_instrumented(net, "V", I)
        rB_t, rB_l = integrated_spikes_instrumented(net, "B", I)

        R_A_tot, R_V_tot, R_AV_tot = float(rA_t.sum()), float(rV_t.sum()), float(rB_t.sum())
        R_A_loc, R_V_loc, R_AV_loc = float(rA_l.sum()), float(rV_l.sum()), float(rB_l.sum())

        denom_t = max(R_A_tot, R_V_tot)
        denom_l = max(R_A_loc, R_V_loc)
        MEI_tot = (R_AV_tot - denom_t) / denom_t if denom_t > 0 else float('nan')
        MEI_loc = (R_AV_loc - denom_l) / denom_l if denom_l > 0 else float('nan')

        # time-window decomposition for the audio-only trace
        A_pre  = float(rA_t[:0].sum())            # there is no pre-stim window
        A_dur  = float(rA_t[:PULSE_LEN].sum())
        A_post = float(rA_t[PULSE_LEN:].sum())

        results.append(dict(
            I=I,
            R_A_tot=R_A_tot, R_V_tot=R_V_tot, R_AV_tot=R_AV_tot, MEI_tot=MEI_tot,
            R_A_loc=R_A_loc, R_V_loc=R_V_loc, R_AV_loc=R_AV_loc, MEI_loc=MEI_loc,
            A_dur=A_dur, A_post=A_post,
            rA_t=rA_t, rV_t=rV_t, rB_t=rB_t,
        ))

        print(f"  {I:>6.3f} | "
              f"{R_A_tot:>9.1f} {R_V_tot:>9.1f} {R_AV_tot:>9.1f} {MEI_tot:>8.3f} | "
              f"{R_A_loc:>8.1f} {R_V_loc:>8.1f} {R_AV_loc:>8.1f} {MEI_loc:>8.3f} | "
              f"{A_pre:>6.0f} {A_dur:>6.1f} {A_post:>6.1f}")

    print()
    print(f"{'='*78}")
    print("BASELINE-SUBTRACTED MEI (using I=0 as the true spontaneous floor)")
    print(f"{'='*78}")
    bg = results[0]  # I=0
    bg_A, bg_V, bg_AV = bg["R_A_tot"], bg["R_V_tot"], bg["R_AV_tot"]
    bg_A_l, bg_V_l, bg_AV_l = bg["R_A_loc"], bg["R_V_loc"], bg["R_AV_loc"]
    print(f"  Baseline (I=0):  R_A={bg_A:.1f}  R_V={bg_V:.1f}  R_AV={bg_AV:.1f}")
    print(f"               loc R_A={bg_A_l:.1f}  R_V={bg_V_l:.1f}  R_AV={bg_AV_l:.1f}")
    print()
    print(f"  {'I':>6s} | {'R_A_e':>8s} {'R_V_e':>8s} {'R_AVe':>8s} {'MEI_corr_tot':>13s} | "
          f"{'R_A_eL':>8s} {'R_V_eL':>8s} {'R_AVeL':>8s} {'MEI_corr_loc':>13s}")
    print(f"  {'-'*6} | {'-'*44} | {'-'*43}")
    for r in results[1:]:
        eA = r["R_A_tot"] - bg_A
        eV = r["R_V_tot"] - bg_V
        eAV = r["R_AV_tot"] - bg_AV
        eA_l = r["R_A_loc"] - bg_A_l
        eV_l = r["R_V_loc"] - bg_V_l
        eAV_l = r["R_AV_loc"] - bg_AV_l
        d  = max(eA, eV)
        dl = max(eA_l, eV_l)
        MEI_c  = (eAV - d) / d if d  > 0 else float('nan')
        MEI_cl = (eAV_l - dl) / dl if dl > 0 else float('nan')
        print(f"  {r['I']:>6.3f} | {eA:>8.1f} {eV:>8.1f} {eAV:>8.1f} {MEI_c:>13.3f} | "
              f"{eA_l:>8.1f} {eV_l:>8.1f} {eAV_l:>8.1f} {MEI_cl:>13.3f}")

    print()
    print(f"{'='*78}")
    print("PER-FRAME TIME COURSE for the AUDIO-ONLY condition")
    print(f"{'='*78}")
    print("  (rows = intensity, cols = frame 0..19, stim is ON for frames 0..9)")
    print()
    hdr = "  I    | " + " ".join(f"{t:>5d}" for t in range(N_FRAMES))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        cells = " ".join(f"{r['rA_t'][t]:>5.0f}" for t in range(N_FRAMES))
        print(f"  {r['I']:>4.2f} | {cells}")

    print()
    print(f"{'='*78}")
    print("VERDICT (Phase 1)")
    print(f"{'='*78}")
    # check if MEI_tot inverts after baseline subtraction
    raw_seq = [r["MEI_tot"] for r in results[1:]]
    print(f"  Raw MEI (total) across intensities: {[f'{x:.3f}' for x in raw_seq]}")
    # check if sequence is monotonically increasing (= inverted vs paper)
    is_increasing = all(raw_seq[i+1] >= raw_seq[i] - 0.02 for i in range(len(raw_seq)-1))
    print(f"  Monotonically increasing (INVERTED vs paper): {is_increasing}")
    print(f"  Paper expected: ~1.04 → ~0.81  (DECREASING)")
    print()
    print(f"  Spontaneous baseline (I=0):  R_A_tot={bg_A:.1f}  R_V_tot={bg_V:.1f}  R_AV_tot={bg_AV:.1f}")
    print(f"  Lowest stim R_A (I=0.2):     R_A_tot={results[3]['R_A_tot']:.1f}")
    print(f"  Δ = R_A(I=0.2) − R_A(I=0) = {results[3]['R_A_tot'] - bg_A:+.1f}  ← small Δ means baseline dominates")


if __name__ == "__main__":
    main()
