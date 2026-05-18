"""ISSUE A H1 — CAUSAL PROOF: NMDA probe asymmetry is the entire mechanism.

Training.py line 2254-2256 records:
   _I_E_ampa = torch.clamp(I_AMPA_curr, min=0.0)
   _I_E_nmda = torch.clamp(I_nmda, min=0.0)
   _I_E = _I_E_ampa + _I_E_nmda

Asymmetry: AMPA is bare-added to I_M (line 2062), so I_AMPA_curr equals
the effective per-substep injection.  NMDA with dt_correct_nmda=True is
added as I_M.add_(I_nmda * nmda_source_scale) (line 2104), so I_nmda
is the SOURCE current, NOT the effective injection.  Source is ~1/source_scale
≈ 25× larger than effective injection.

At gNMDA=0.05 + Form2 OFF (paper): no source_scale, recording = injection.  Symmetric.
At gNMDA=1.30 + Form2 ON  (post-fix): source_scale = 0.0392, recording = 25×injection.

This patch records `I_nmda * nmda_source_scale` when Form2 ON, which
is the effective injection (apples-to-apples with AMPA).

PROOF:
  cell D (gNMDA=1.30, Form2 ON) raw E/I = 25.36
  cell D + probe fix predicted E/I ≈ 1.05 (paper's 1.04 ± 0.005)

If the patched cell D gives E/I ≈ 1.05 in this experiment, hypothesis CONFIRMED.
"""
from __future__ import annotations
import sys, time, inspect, textwrap, math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import Training as TRN
from EI_balance_test import run_ei_probe_separated
from TBW_test import load_msi_model

BASE = ROOT / "checkpoint"
MODELS = [BASE / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
PULSE_FRAMES = 5
N_FRAMES = 20
N_SUBSTEPS = 100
EVOKED_FRAMES = 12.5
EVOKED_SUBSTEPS = int(EVOKED_FRAMES * N_SUBSTEPS)


# ─────────────────────────────────────────────────────────────────────
# Monkey-patch update_all_layers_batch's NMDA recording
# Replace:
#   _I_E_nmda = torch.clamp(I_nmda, min=0.0)
# with:
#   _I_E_nmda = torch.clamp(I_nmda * nmda_source_scale, min=0.0) if self.dt_correct_nmda else torch.clamp(I_nmda, min=0.0)
# ─────────────────────────────────────────────────────────────────────
OLD_LINE = "_I_E_nmda = torch.clamp(I_nmda, min=0.0)"
NEW_LINE = ("_I_E_nmda = torch.clamp(I_nmda * nmda_source_scale, min=0.0) "
            "if self.dt_correct_nmda else torch.clamp(I_nmda, min=0.0)")

ORIG_METHOD = TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch


def install_probe_fix():
    src = inspect.getsource(ORIG_METHOD)
    src = textwrap.dedent(src)
    assert OLD_LINE in src, "OLD_LINE not found in update_all_layers_batch"
    src = src.replace(OLD_LINE, NEW_LINE)
    ns: dict = {}
    # Need same globals as Training so torch, F, math, etc resolve
    ns.update(TRN.__dict__)
    exec(src, ns)
    patched = ns["update_all_layers_batch"]
    TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch = patched


def uninstall_probe_fix():
    TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch = ORIG_METHOD


def run_cell(*, gNMDA, dt_correct, label):
    rows = []
    t0 = time.time()
    for p in MODELS:
        net = load_msi_model(p, device="cuda")
        if gNMDA is not None:
            net.gNMDA = float(gNMDA)
        net.dt_correct_nmda = bool(dt_correct)
        res = run_ei_probe_separated(net, centre_deg=90.0,
                                     pulse_frames=PULSE_FRAMES,
                                     n_frames=N_FRAMES, intensity=1.0)
        traces = res["traces"]
        I_E = traces["I_E_mean"][:EVOKED_SUBSTEPS]
        I_I = traces["I_I_mean"][:EVOKED_SUBSTEPS]
        ampa = traces["AMPA"][:EVOKED_SUBSTEPS]
        nmda = traces["NMDA"][:EVOKED_SUBSTEPS]
        ff = traces["FFInh"][:EVOKED_SUBSTEPS]
        rec = traces["RecurInh"][:EVOKED_SUBSTEPS]
        lat = traces["LatInh"][:EVOKED_SUBSTEPS]
        rows.append(dict(
            E=float(I_E.mean()), I=float(I_I.mean()),
            ei=float(I_E.mean())/(float(I_I.mean())+1e-12),
            AMPA=float(ampa.mean()), NMDA=float(nmda.mean()),
            FF=float(ff.mean()), Rec=float(rec.mean()), Lat=float(lat.mean()),
        ))
        del net; torch.cuda.empty_cache()
    el = time.time() - t0
    keys = ["E", "I", "ei", "AMPA", "NMDA", "FF", "Rec", "Lat"]
    means = {k: np.mean([r[k] for r in rows]) for k in keys}
    sems  = {k: np.std([r[k] for r in rows], ddof=1)/np.sqrt(len(rows)) for k in keys}
    print(f"{label:<60s}  E/I={means['ei']:.3f}±{sems['ei']:.3f}  "
          f"E={means['E']:.3f}  I={means['I']:.3f}  el={el:.0f}s")
    print(f"   AMPA={means['AMPA']:.4f}  NMDA_rec={means['NMDA']:.4f}  "
          f"FF={means['FF']:.4f}  Rec={means['Rec']:.4f}  Lat={means['Lat']:.4f}")
    return means


def main():
    print("ISSUE A H1 — probe-fix causal proof")
    print("=" * 78)
    print()
    print("Phase 1: BASELINE (orig probe, unmodified Training.py)")
    print("-" * 78)
    print(f"  source_scale at dt=0.1, tau_syn=2.5 = {1 - math.exp(-0.1/2.5):.5f}")
    print()
    a_orig = run_cell(gNMDA=0.05, dt_correct=False,
                      label="A_orig: gNMDA=0.05, Form2=OFF (paper)")
    d_orig = run_cell(gNMDA=1.30, dt_correct=True,
                      label="D_orig: gNMDA=1.30, Form2=ON  (post-recalib)")

    print()
    print("Phase 2: PATCHED probe (NMDA records I_nmda*source_scale when Form2=ON)")
    print("-" * 78)
    install_probe_fix()
    try:
        a_patch = run_cell(gNMDA=0.05, dt_correct=False,
                           label="A_patch: gNMDA=0.05, Form2=OFF (no-op, no scale)")
        d_patch = run_cell(gNMDA=1.30, dt_correct=True,
                           label="D_patch: gNMDA=1.30, Form2=ON  (scaled)")
    finally:
        uninstall_probe_fix()

    print()
    print("=" * 78)
    print("CAUSAL VERDICT — probe asymmetry is the mechanism")
    print("=" * 78)
    print(f"  Paper baseline (cell A):                 E/I = {a_orig['ei']:.3f}")
    print(f"  Post-recalib raw (cell D, orig probe):   E/I = {d_orig['ei']:.3f}")
    print(f"  Paper with patched probe (cell A_patch): E/I = {a_patch['ei']:.3f}  (should equal A_orig)")
    print(f"  Post-recalib with patched probe (D_patch): E/I = {d_patch['ei']:.3f}  (should equal A_orig ≈1.04)")
    print()
    print(f"  Δ from paper to post-fix (orig probe):   {d_orig['ei'] - a_orig['ei']:+.3f}")
    print(f"  Δ from paper to post-fix (patched probe): {d_patch['ei'] - a_orig['ei']:+.3f}")
    print()
    print("  If D_patch ≈ A_orig (≈1.04), the entire E/I jump is a probe asymmetry,")
    print("  proven by changing ONLY the probe (not the network).")
    print()
    print(f"  NMDA component check:")
    print(f"    A_orig NMDA = {a_orig['NMDA']:.4f}  (recorded ≈ effective in paper)")
    print(f"    D_orig NMDA = {d_orig['NMDA']:.4f}  (recorded = SOURCE, 26× too big)")
    print(f"    D_patch NMDA = {d_patch['NMDA']:.4f}  (recorded = effective injection ≈ A_orig)")


if __name__ == "__main__":
    main()
