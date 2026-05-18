"""Quick verification: AGC and plasticity are actually disabled."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from TBW_test import load_msi_model, compute_tbw_temporal_fusion_persep

CKPT = ROOT / "checkpoint" / "msi_model_surr_10_00.pt"
OFFSETS = list(range(-50, 51, 10))


def main():
    # Save initial weight states and g_FFinh, then run TBW, check if anything changed
    rng = np.random.default_rng(12345)
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        net = load_msi_model(CKPT, device="cuda")
        net.dt = 0.05
        net.n_substeps = 200
        net.dt_correct_nmda = True
        net.gNMDA = 1.30
        net.plasticity_enabled = False
        net.freeze_g_FFinh = True

        print(f"plasticity_enabled = {net.plasticity_enabled}")
        print(f"freeze_g_FFinh = {net.freeze_g_FFinh}")
        print(f"g_FFinh (initial) = {net.g_FFinh}")
        W_save_a = net.W_a2msi_AMPA.detach().clone()
        W_save_v = net.W_v2msi_AMPA.detach().clone()
        W_save_inA = net.W_inA_inh.detach().clone()
        W_save_inV = net.W_inV_inh.detach().clone()

        p, _ = compute_tbw_temporal_fusion_persep(
            net, OFFSETS, n_trials=10, T=60, D=5, stim_in=1.0,
            sigma=2.0, valley_threshold=0.4, min_peak_height=0.2,
            min_peak_separation=3, min_total=10.0,
        )

        print(f"g_FFinh (final)   = {net.g_FFinh}")
        print(f"W_a2msi_AMPA diff = {(net.W_a2msi_AMPA - W_save_a).abs().max().item():.3e}")
        print(f"W_v2msi_AMPA diff = {(net.W_v2msi_AMPA - W_save_v).abs().max().item():.3e}")
        print(f"W_inA_inh diff    = {(net.W_inA_inh - W_save_inA).abs().max().item():.3e}")
        print(f"W_inV_inh diff    = {(net.W_inV_inh - W_save_inV).abs().max().item():.3e}")

        del net
        torch.cuda.empty_cache()
    finally:
        np.random.default_rng = orig


if __name__ == "__main__":
    main()
