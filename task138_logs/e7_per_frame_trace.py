"""Task #138 E7: per-frame trace at sep=80 A-only, RANDOM idxA (e1 seed=42).
Logs sum_sM.sum, A_roi(sum_sM), A_roi(_latest_sMSI), g_FFinh, I_M.mean.
Identifies the frame where CURRENT and PRISTINE diverge.
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


def run(sbw_module, ckpt_path, label, sep_deg=80, n_trials=50, n_frames=20,
        intensity=1.0, roi_half=20, seed=42):
    print(f"\n========== {label} ==========")
    net = sbw_module.load_msi_model(ckpt_path, device="cuda")
    print(f"  loaded: g_FFinh_initial={float(net.g_FFinh):.6f}, dt={net.dt}, "
          f"n_substeps={net.n_substeps}, step_counter={net.step_counter}")

    initial_g_FFinh = float(net.g_FFinh)
    initial_step_counter = net.step_counter

    N = net.n
    S = net.space_size
    rng = np.random.default_rng(seed)
    xs = torch.arange(N, device=net.device, dtype=torch.float32)

    def to_idx(deg):
        return torch.round(
            torch.as_tensor(deg, device=net.device, dtype=torch.float32)
            * (N - 1) / (S - 1)).long()

    def make_gauss(idx_centres):
        return torch.exp(
            -0.5 * ((xs - idx_centres[:, None]) / net.sigma_in) ** 2
        ) * intensity

    base = rng.integers(0, S, size=n_trials)
    locA = base
    idxA = to_idx(locA)
    gA = make_gauss(idxA)
    zero = torch.zeros_like(gA)

    # ROI gather indices
    offsets = torch.arange(-roi_half, roi_half + 1, device=net.device)
    roi_indices = (idxA[:, None] + offsets[None, :]) % N

    # Reset state for A-only pass (matches e1's run_pass semantics)
    net.g_FFinh = initial_g_FFinh
    net.step_counter = initial_step_counter
    net.reset_state(batch_size=n_trials)

    print(f"  frame |  sum_sM.sum  | A_roi(sum_sM) | A_roi(_latest) | g_FFinh  | I_M.mean")
    print(f"  ------+-------------+---------------+----------------+----------+---------")
    for f in range(n_frames):
        ret = net.update_all_layers_batch(gA, zero, return_spike_sum=True)
        sum_sM = ret[-1]
        latest = net._latest_sMSI
        ssM_total = float(sum_sM.sum().item())
        Aroi_ssM = float(sum_sM.gather(1, roi_indices).sum().item())
        Aroi_lat = float(latest.gather(1, roi_indices).sum().item())
        g_ffi = float(net.g_FFinh)
        I_M_mean = float(net.I_M.mean().item())
        print(f"  {f:3d}   | {ssM_total:11.2f} | {Aroi_ssM:13.4f} | "
              f"{Aroi_lat:14.4f} | {g_ffi:.6f} | {I_M_mean:+.4e}")


def main():
    print("=" * 100)
    print("E7: per-frame trace (sep=80, A-only, random idxA seed=42) — CURRENT vs PRISTINE")
    print("=" * 100)

    ROOT_CUR = Path(__file__).resolve().parent.parent
    sbw_cur = load_from_root(str(ROOT_CUR), "cur")
    run(sbw_cur, ROOT_CUR / "checkpoint/msi_model_surr_10_00.pt", "CURRENT")

    torch.cuda.empty_cache()
    for mod in list(sys.modules.keys()):
        if mod.startswith("sbw_") or mod in ("Training", "SBW_test", "TBW_test"):
            del sys.modules[mod]

    sbw_pri = load_from_root("/tmp/fsts_pristine", "pri")
    run(sbw_pri, Path("/tmp/fsts_pristine/checkpoint/msi_model_surr_10_00.pt"), "PRISTINE")


if __name__ == "__main__":
    main()
