"""Task #138 E4: load surr_10_00 on CURRENT vs PRISTINE Training.py and dump
post-load state to find which attributes differ.
"""
from __future__ import annotations
import sys
from pathlib import Path
import importlib
import importlib.util
import numpy as np
import torch


def load_module_from(label, root_path):
    """Load Training/SBW_test from a specific filesystem root."""
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


def dump_state(net, label):
    print(f"\n--- {label} STATE ---")
    print(f"  dt={net.dt}, n_substeps={net.n_substeps}")
    print(f"  g_FFinh={float(net.g_FFinh):.6f}")
    print(f"  gNMDA={net.gNMDA}")
    print(f"  g_GABA={net.g_GABA}")
    print(f"  pv_nmda={net.pv_nmda}, targ_ratio={net.targ_ratio}")
    print(f"  freeze_g_FFinh={getattr(net, 'freeze_g_FFinh', 'MISSING')}")
    print(f"  plasticity_enabled={getattr(net, 'plasticity_enabled', 'MISSING')}")
    print(f"  dt_correct_nmda={getattr(net, 'dt_correct_nmda', 'MISSING')}")
    print(f"  step_counter={net.step_counter}")
    # Conduction delays
    for attr in ["conduction_delay_a2msi", "conduction_delay_v2msi",
                 "conduction_delay_inA_inh", "conduction_delay_inV_inh",
                 "conduction_delay_a2msi_inh", "conduction_delay_v2msi_inh",
                 "conduction_delay_msi_inh2exc", "conduction_delay_msi2out"]:
        val = getattr(net, attr, 'MISSING')
        print(f"  {attr}={val}")
    # Physical-time *_ms attrs (current only)
    for attr in ["conduction_delay_a2msi_ms", "conduction_delay_v2msi_ms",
                 "T_AGC_FAST_MS", "T_AGC_SLOW_MS"]:
        if hasattr(net, attr):
            print(f"  {attr}={getattr(net, attr)}")
    # Weights
    for w in ["W_a2msi_AMPA", "W_v2msi_AMPA", "W_a2msi_NMDA", "W_v2msi_NMDA",
              "W_inA_inh", "W_inV_inh", "W_inA", "W_inV", "W_msiInh2Exc_GABA"]:
        wt = getattr(net, w, None)
        if wt is None:
            continue
        wd = wt.data if hasattr(wt, "data") else wt
        print(f"  {w}: shape={tuple(wd.shape)}, mean={wd.mean().item():.6f}, "
              f"std={wd.std().item():.6f}, sum={wd.sum().item():.4f}, "
              f"min={wd.min().item():.4f}, max={wd.max().item():.4f}")


def main():
    ROOT_CUR = Path(__file__).resolve().parent.parent
    ROOT_PRI = Path("/tmp/fsts_pristine")

    print("=" * 90)
    print("E4: post-load state comparison — CURRENT vs PRISTINE")
    print("  ckpt: msi_model_surr_10_00.pt (SAME .pt for both)")
    print("=" * 90)

    sbw_cur = load_module_from("cur", ROOT_CUR)
    net_cur = sbw_cur.load_msi_model(ROOT_CUR / "checkpoint" / "msi_model_surr_10_00.pt",
                                     device="cuda")
    dump_state(net_cur, "CURRENT v2 Training.py")
    del net_cur
    torch.cuda.empty_cache()

    # Need to reset module cache for pristine
    for mod in list(sys.modules.keys()):
        if mod.startswith("sbw_") or mod in ("Training", "SBW_test", "TBW_test"):
            del sys.modules[mod]

    sbw_pri = load_module_from("pri", ROOT_PRI)
    net_pri = sbw_pri.load_msi_model(ROOT_PRI / "checkpoint" / "msi_model_surr_10_00.pt",
                                     device="cuda")
    dump_state(net_pri, "PRISTINE fb6d3f6 Training.py")


if __name__ == "__main__":
    main()
