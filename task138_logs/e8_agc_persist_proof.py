"""Task #138 E8: causal proof that _last_agc_fast_t / _last_agc_slow_t
persistence across reset_state() causes AGC suppression in 2nd/3rd
within-separation pass, which causes SBW saturation in CURRENT but
not PRISTINE.

Procedure (CURRENT Training.py only):
  Variant A: Triplet AV→A→V identical to SBW_test pattern.
             Measure A_roi(_latest_sMSI) and g_FFinh trajectory in A-only pass.
  Variant B: Same triplet but explicitly reset _last_agc_*_t to 0 between passes.
             Measure same quantities.

If Variant A shows g_FFinh = 0.5648 flat (AGC suppressed) and A_roi << pristine,
while Variant B shows g_FFinh drift and A_roi ≈ pristine → AGC-persistence is
the root cause.

Then repeat with PRISTINE Training.py as a control (which has no such state)
to demonstrate that pristine triplet does NOT suffer this issue.
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


def triplet_trace(sbw_module, ckpt_path, label, reset_agc_t=False,
                  sep_deg=80, n_trials=50, n_frames=20,
                  intensity=1.0, roi_half=20, seed=42):
    print(f"\n========== {label}  reset_agc_t={reset_agc_t} ==========")
    net = sbw_module.load_msi_model(ckpt_path, device="cuda")
    initial_g_FFinh = float(net.g_FFinh)
    initial_step_counter = net.step_counter

    has_agc_t = hasattr(net, "_last_agc_fast_t")
    if has_agc_t:
        print(f"  pre-init: _last_agc_fast_t={net._last_agc_fast_t:.3f}, "
              f"_last_agc_slow_t={net._last_agc_slow_t:.3f}")
    else:
        print(f"  (no _last_agc_*_t attribute — pristine has none)")

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
    locV = (base + sep_deg) % S
    idxA = to_idx(locA)
    idxV = to_idx(locV)
    gA = make_gauss(idxA)
    gV = make_gauss(idxV)
    zero = torch.zeros_like(gA)

    offsets = torch.arange(-roi_half, roi_half + 1, device=net.device)
    roi_indices = (idxA[:, None] + offsets[None, :]) % N

    def reset_and_optionally_reset_agc():
        net.g_FFinh = initial_g_FFinh
        net.step_counter = initial_step_counter
        net.reset_state(batch_size=n_trials)
        if reset_agc_t and has_agc_t:
            net._last_agc_fast_t = 0.0
            net._last_agc_slow_t = 0.0

    def run_pass(stim_A, stim_V, tag):
        reset_and_optionally_reset_agc()
        sum_lat = torch.zeros(n_trials, N, device=net.device)
        g_first = float(net.g_FFinh)
        for f in range(n_frames):
            ret = net.update_all_layers_batch(stim_A, stim_V, return_spike_sum=True)
            sum_lat += net._latest_sMSI
        g_last = float(net.g_FFinh)
        Aroi_lat = float(sum_lat.gather(1, roi_indices).sum().item() / n_trials)
        post_agc_t = (net._last_agc_fast_t, net._last_agc_slow_t) if has_agc_t else (None, None)
        return Aroi_lat, g_first, g_last, post_agc_t

    av_Aroi, av_g0, av_g1, av_t = run_pass(gA, gV, "AV")
    print(f"  AV  pass: A_roi(latest)/trial={av_Aroi:.3f}  g_FFinh: {av_g0:.6f} -> {av_g1:.6f}  "
          f"post _last_agc_t={av_t}")
    a_Aroi, a_g0, a_g1, a_t = run_pass(gA, zero, "A")
    print(f"  A   pass: A_roi(latest)/trial={a_Aroi:.3f}  g_FFinh: {a_g0:.6f} -> {a_g1:.6f}  "
          f"post _last_agc_t={a_t}")
    v_Aroi, v_g0, v_g1, v_t = run_pass(zero, gV, "V")
    print(f"  V   pass: A_roi(latest)/trial={v_Aroi:.3f}  g_FFinh: {v_g0:.6f} -> {v_g1:.6f}  "
          f"post _last_agc_t={v_t}")

    enh = av_Aroi - max(a_Aroi, v_Aroi)
    print(f"  --> enh = AV_roi - max(A_roi, V_roi) = {av_Aroi:.3f} - "
          f"{max(a_Aroi, v_Aroi):.3f} = {enh:+.3f}")
    print(f"  --> enh > threshold(10)?  {enh > 10}")


def main():
    print("=" * 100)
    print("E8: AGC-persistence causal-proof — triplet AV->A->V on surr_10_00 sep=80")
    print("=" * 100)

    ROOT_CUR = Path(__file__).resolve().parent.parent

    # Variant 1: CURRENT, default behavior (_last_agc_*_t persists)
    sbw_cur = load_from_root(str(ROOT_CUR), "cur_default")
    triplet_trace(sbw_cur, ROOT_CUR / "checkpoint/msi_model_surr_10_00.pt",
                  "CURRENT (default, _last_agc_*_t persists)", reset_agc_t=False)
    del sbw_cur
    torch.cuda.empty_cache()
    for mod in list(sys.modules.keys()):
        if mod.startswith("sbw_") or mod in ("Training", "SBW_test", "TBW_test"):
            del sys.modules[mod]

    # Variant 2: CURRENT but explicitly zero _last_agc_*_t between passes
    sbw_cur2 = load_from_root(str(ROOT_CUR), "cur_reset")
    triplet_trace(sbw_cur2, ROOT_CUR / "checkpoint/msi_model_surr_10_00.pt",
                  "CURRENT (with _last_agc_*_t reset between passes)", reset_agc_t=True)
    del sbw_cur2
    torch.cuda.empty_cache()
    for mod in list(sys.modules.keys()):
        if mod.startswith("sbw_") or mod in ("Training", "SBW_test", "TBW_test"):
            del sys.modules[mod]

    # Variant 3: PRISTINE, no such attribute — control
    sbw_pri = load_from_root("/tmp/fsts_pristine", "pri")
    triplet_trace(sbw_pri, Path("/tmp/fsts_pristine/checkpoint/msi_model_surr_10_00.pt"),
                  "PRISTINE (control)", reset_agc_t=False)


if __name__ == "__main__":
    main()
