"""Task #138 E6: 1-outer-frame trace on CURRENT and PRISTINE, dump intermediate
state to find divergence.

Loads surr_10_00 with both versions. With identical reset_state + identical
stim + same RNG, runs ONE outer frame call. Compares MSI mean spike count,
I_M mean, v_msi mean, nmda_m mean.
"""
from __future__ import annotations
import sys
from pathlib import Path
import importlib.util
import numpy as np
import torch


def load_from_root(root_path: str, label: str):
    root = Path(root_path)
    spec = importlib.util.spec_from_file_location(
        f"sbw_{label}", root / "SBW_test.py"
    )
    sbw = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(sbw)
    finally:
        sys.path.pop(0)
    return sbw


def trace_one_frame(sbw_module, ckpt_path, label):
    print(f"\n========== {label} ==========")
    net = sbw_module.load_msi_model(ckpt_path, device="cuda")
    print(f"  loaded: g_FFinh={float(net.g_FFinh):.6f}, dt={net.dt}, n_substeps={net.n_substeps}")
    print(f"  tau_syn={net.tau_syn}, tau_nmda={net.tau_nmda}, tau_nmdaVolt={net.tau_nmdaVolt}")

    # Set known state, identical stim
    n_trials = 8
    net.reset_state(batch_size=n_trials)
    torch.manual_seed(12345)

    N = net.n
    S = net.space_size
    xs = torch.arange(N, device=net.device, dtype=torch.float32)
    # Single A-only stimulus, centered at S/2
    idxA = torch.full((n_trials,), int(S/2), device=net.device, dtype=torch.long)
    gA = torch.exp(-0.5 * ((xs - idxA[:, None]) / net.sigma_in) ** 2) * 1.0
    zero = torch.zeros_like(gA)

    # Print state BEFORE
    print(f"  BEFORE: I_M.mean={net.I_M.mean().item():.6e}, "
          f"v_msi.mean={net.v_msi.mean().item():.4f}, "
          f"nmda_m.mean={net.nmda_m.mean().item():.6e}")

    # Run ONE outer frame call (A-only)
    ret = net.update_all_layers_batch(gA, zero, return_spike_sum=True)
    sA, sV, sM, sO, sum_sM = ret

    print(f"  AFTER FRAME 1 (A-only):")
    print(f"    new_sM.sum     = {sM.sum().item():.4f}")
    print(f"    sum_sM.sum     = {sum_sM.sum().item():.4f}")
    print(f"    sum_sM/n_trials = {sum_sM.sum().item()/n_trials:.4f}")
    print(f"    I_M.mean       = {net.I_M.mean().item():.6e}")
    print(f"    v_msi.mean     = {net.v_msi.mean().item():.4f}")
    print(f"    v_msi.max      = {net.v_msi.max().item():.4f}")
    print(f"    nmda_m.mean    = {net.nmda_m.mean().item():.6e}")
    print(f"    nmda_m.max     = {net.nmda_m.max().item():.6e}")
    print(f"    g_FFinh        = {float(net.g_FFinh):.6f}")

    # Run 5 more outer frames to see accumulation
    total_sum_M = sum_sM.sum().item()
    for k in range(5):
        ret = net.update_all_layers_batch(gA, zero, return_spike_sum=True)
        sA, sV, sM, sO, sum_sM = ret
        total_sum_M += sum_sM.sum().item()
    print(f"  TOTAL after 6 frames: sum_sM = {total_sum_M:.4f}, "
          f"g_FFinh={float(net.g_FFinh):.6f}")

    return {
        "label": label,
        "sum_sM_frame1": float(sum_sM.sum().item()),
        "I_M_mean": float(net.I_M.mean().item()),
        "v_msi_mean": float(net.v_msi.mean().item()),
    }


def main():
    print("=" * 90)
    print("E6: 1-frame trace — CURRENT vs PRISTINE on surr_10_00, A-only stim")
    print("=" * 90)

    ROOT_CUR = Path(__file__).resolve().parent.parent
    sbw_cur = load_from_root(str(ROOT_CUR), "cur")
    res_cur = trace_one_frame(sbw_cur, ROOT_CUR / "checkpoint/msi_model_surr_10_00.pt", "CURRENT")

    # Purge modules
    torch.cuda.empty_cache()
    for mod in list(sys.modules.keys()):
        if mod.startswith("sbw_") or mod in ("Training", "SBW_test", "TBW_test"):
            del sys.modules[mod]

    sbw_pri = load_from_root("/tmp/fsts_pristine", "pri")
    res_pri = trace_one_frame(
        sbw_pri, Path("/tmp/fsts_pristine/checkpoint/msi_model_surr_10_00.pt"), "PRISTINE"
    )


if __name__ == "__main__":
    main()
