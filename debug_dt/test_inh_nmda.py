"""Test whether the SECOND continuous NMDA injection (I_M_inh.add_(I_nmda_inh)
at Training.py:2138) also contributes to dt-dependence.
"""
from __future__ import annotations
import inspect, textwrap, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from TBW_test import load_msi_model, compute_tbw_temporal_fusion_persep, fit_psychometric_curve_improved
import debug_dt.patch_variants as PV
import Training as TRN

CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
OFFSETS = list(range(-50, 51, 2))
OFFSETS_MS = [o*10 for o in OFFSETS]


def measure_hw(dt, nsub, seed=12345):
    rng = np.random.default_rng(seed)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        net = load_msi_model(CKPT, device="cuda")
        net.dt = dt; net.n_substeps = nsub
        net.plasticity_enabled = False; net.freeze_g_FFinh = True
        p, _ = compute_tbw_temporal_fusion_persep(
            net, OFFSETS, n_trials=50, T=60, D=5, stim_in=1.0,
            sigma=2.0, valley_threshold=0.4, min_peak_height=0.2,
            min_peak_separation=3, min_total=10.0,
        )
        del net; torch.cuda.empty_cache()
    finally:
        np.random.default_rng = orig
    try:
        fit = fit_psychometric_curve_improved(np.asarray(OFFSETS_MS), p)
        return float(fit.get("tbw", float("nan")))
    except Exception:
        return float("nan")


def install_patch(scale_msi_exc: float, scale_msi_inh: float):
    src = inspect.getsource(PV.ORIG_UPDATE)
    src = textwrap.dedent(src)
    src = src.replace("def update_all_layers_batch(", "def _ptest(", 1)
    old1 = "self.I_M.add_(I_nmda)"
    new1 = f"self.I_M.add_(I_nmda * {scale_msi_exc!r})"
    assert src.count(old1) == 1
    src = src.replace(old1, new1)
    old2 = "self.I_M_inh.add_(I_nmda_inh)"
    new2 = f"self.I_M_inh.add_(I_nmda_inh * {scale_msi_inh!r})"
    assert src.count(old2) == 1
    src = src.replace(old2, new2)
    ns = {}; glb = TRN.__dict__.copy()
    exec(src, glb, ns)
    TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch = ns["_ptest"]


def main():
    print(f"{'scale_exc':>10s}  {'scale_inh':>10s}  {'HW (ms)':>10s}")
    # dt=0.1 baseline (no patch)
    PV.restore_original()
    h_ref = measure_hw(0.1, 100); print(f"{'1.00':>10s}  {'1.00':>10s}  {h_ref:>10.1f}  (dt=0.1 ref)")

    # dt=0.05 raw
    PV.restore_original()
    h05_raw = measure_hw(0.05, 200); print(f"{'1.00':>10s}  {'1.00':>10s}  {h05_raw:>10.1f}  (dt=0.05 raw)")

    # dt=0.05 with only MSI-exc NMDA scaled
    install_patch(0.6, 1.0); h_a = measure_hw(0.05, 200); PV.restore_original()
    print(f"{'0.60':>10s}  {'1.00':>10s}  {h_a:>10.1f}  (scale MSI-exc NMDA)")

    # dt=0.05 with only MSI-inh NMDA scaled
    install_patch(1.0, 0.6); h_b = measure_hw(0.05, 200); PV.restore_original()
    print(f"{'1.00':>10s}  {'0.60':>10s}  {h_b:>10.1f}  (scale MSI-inh NMDA)")

    # dt=0.05 with BOTH scaled
    install_patch(0.6, 0.6); h_c = measure_hw(0.05, 200); PV.restore_original()
    print(f"{'0.60':>10s}  {'0.60':>10s}  {h_c:>10.1f}  (scale BOTH)")


if __name__ == "__main__":
    main()
