"""
Minimal forensic reproducer for TBW dt-dependence.

Loads checkpoint(s) and runs the canonical temporal_fusion P(fusion) curve at
dt = 0.1 ms (n_substeps = 100) and at dt = 0.05 ms (n_substeps = 200).  Fits
the asymmetric pedestal exactly the same way TBW_test.plot_psychometric_tbw_ax
does and reports the half-width (HW).

Interventions (toggle individually):
    --freeze-agc          Set net.freeze_g_FFinh = True (kill AGC drift).
    --preserve-delays-ms  Rescale all conduction delays so their PHYSICAL
                          duration in ms matches the dt=0.1 baseline regardless
                          of dt.  Specifically: every delay_substeps is
                          multiplied by (0.1 / dt_target) and rounded.
    --exp-decay           Monkey-patch the substep loop to use exact
                          exp(-dt/tau) for the leaky-integrator decays of
                          I_A, I_V, I_M, I_M_inh, I_O, ampa_m, nmda_m,
                          and nmda_m_inh.
    --izhi-half           Sub-step the Izhikevich voltage / recovery update
                          twice per substep with dt/2 each (effective 2x finer
                          Izhikevich integration without touching delays/decays).

You can combine flags; each is applied independently.

Usage:
    python debug_dt/repro_dt_tbw.py --ckpts 0 --trials 50
    python debug_dt/repro_dt_tbw.py --ckpts 0 --trials 50 --freeze-agc
    python debug_dt/repro_dt_tbw.py --ckpts 0 --trials 50 --preserve-delays-ms
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure project root is importable when run from the project dir.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from TBW_test import (
    load_msi_model,
    compute_tbw_temporal_fusion_persep,
    fit_psychometric_curve_improved,
)
import Training as TRN


# ──────────────────────────────────────────────────────────────────────
# Intervention helpers
# ──────────────────────────────────────────────────────────────────────
DELAY_ATTRS = (
    "conduction_delay_a2msi",
    "conduction_delay_v2msi",
    "conduction_delay_msi2out",
    "conduction_delay_inA_inh",
    "conduction_delay_inV_inh",
    "conduction_delay_a2msi_inh",
    "conduction_delay_v2msi_inh",
    "conduction_delay_msi_inh2exc",
)


def configure_dt(net, dt: float, n_substeps: int) -> None:
    """Hot-swap dt and n_substeps for an already-loaded network."""
    net.dt = dt
    net.n_substeps = n_substeps


def preserve_delays_ms(net, ref_dt: float = 0.1) -> None:
    """Multiply each substep-indexed delay by (ref_dt / net.dt) so the
    physical delay in ms matches the dt=ref_dt baseline."""
    factor = ref_dt / float(net.dt)
    for attr in DELAY_ATTRS:
        old = int(getattr(net, attr))
        new = max(1, int(round(old * factor)))
        setattr(net, attr, new)
    # Re-allocate ring buffers with new sizes
    net._reset_delay_buffers()


# Save the original update_all_layers_batch so we can monkey-patch one or
# more behaviour variants on top of it without losing the reference.
_orig_update = TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch


def _wrap_exp_decay():
    """Replace `1 - dt/tau` linear decay factors with `exp(-dt/tau)`.
    The simplest way: monkey-patch the function to overwrite the locals
    via a re-compiled version.  Instead we wrap via subclassing.
    """
    # Strategy: subclass via runtime patch on the class' update method.
    # Easier: read the source, replace `decay_factor = 1.0 - self.dt / self.tau_syn`
    # etc., exec, and assign.
    import inspect
    src = inspect.getsource(_orig_update)
    # Cut the leading method indentation (4 spaces) so it's a top-level fn
    src_lines = src.splitlines()
    base_indent = len(src_lines[0]) - len(src_lines[0].lstrip())
    src = "\n".join(line[base_indent:] for line in src_lines)
    # Rename to avoid colliding
    src = src.replace("def update_all_layers_batch(",
                      "def _patched_update_all_layers_batch(", 1)
    # Patch each of the four declaration lines
    repls = {
        "decay_factor = 1.0 - self.dt / self.tau_syn":
            "decay_factor = float(torch.exp(torch.tensor(-self.dt / self.tau_syn)).item())",
        "ampa_decay = 1.0 - self.dt / self.tau_ampa_lp":
            "ampa_decay = float(torch.exp(torch.tensor(-self.dt / self.tau_ampa_lp)).item())",
        "nmda_decay = 1.0 - self.dt / self.tau_nmda":
            "nmda_decay = float(torch.exp(torch.tensor(-self.dt / self.tau_nmda)).item())",
    }
    for old, new in repls.items():
        assert old in src, f"Could not find: {old}"
        src = src.replace(old, new)
    ns = {}
    # Need access to torch, F, etc. from the original module's namespace.
    glb = TRN.__dict__.copy()
    exec(src, glb, ns)
    new_fn = ns["_patched_update_all_layers_batch"]
    TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch = new_fn


def _wrap_izhi_half():
    """Replace each `v += dt * dV; u += dt * (a*(b*v - u))` with two
    half-steps of dt/2.  Implemented via source rewrite as in _wrap_exp_decay.
    """
    import inspect
    src = inspect.getsource(_orig_update)
    src_lines = src.splitlines()
    base_indent = len(src_lines[0]) - len(src_lines[0].lstrip())
    src = "\n".join(line[base_indent:] for line in src_lines)
    src = src.replace("def update_all_layers_batch(",
                      "def _patched_update_all_layers_batch(", 1)

    # Pattern for each layer: 4 lines (dV computation, v update, u update, spike-detect)
    # We replace the v and u updates with sub-stepping (dt/2 twice).  We keep the
    # dV computation untouched; we just *re-compute* dV at the half-step.
    # NOTE: this is a clean two-substep Euler for v alone; u uses dt full step.
    # The simplest robust approach: replace the entire 4-line block per layer.

    # Layer A
    blockA_old = (
        "            dVA = (0.04 * self.v_uniA.pow(2) + 5.0 * self.v_uniA + 140.0\n"
        "                   - self.u_uniA + self.I_A)\n"
        "            self.v_uniA += self.dt * dVA\n"
        "            self.u_uniA += self.dt * (self.aA * (self.bA * self.v_uniA - self.u_uniA))"
    )
    blockA_new = (
        "            for _hs in range(2):\n"
        "                _dt_h = self.dt / 2.0\n"
        "                dVA = (0.04 * self.v_uniA.pow(2) + 5.0 * self.v_uniA + 140.0\n"
        "                       - self.u_uniA + self.I_A)\n"
        "                self.v_uniA += _dt_h * dVA\n"
        "                self.u_uniA += _dt_h * (self.aA * (self.bA * self.v_uniA - self.u_uniA))"
    )

    blockV_old = (
        "            dVV = (0.04 * self.v_uniV.pow(2) + 5.0 * self.v_uniV + 140.0\n"
        "                   - self.u_uniV + self.I_V)\n"
        "            self.v_uniV += self.dt * dVV\n"
        "            self.u_uniV += self.dt * (self.aV * (self.bV * self.v_uniV - self.u_uniV))"
    )
    blockV_new = (
        "            for _hs in range(2):\n"
        "                _dt_h = self.dt / 2.0\n"
        "                dVV = (0.04 * self.v_uniV.pow(2) + 5.0 * self.v_uniV + 140.0\n"
        "                       - self.u_uniV + self.I_V)\n"
        "                self.v_uniV += _dt_h * dVV\n"
        "                self.u_uniV += _dt_h * (self.aV * (self.bV * self.v_uniV - self.u_uniV))"
    )

    blockM_old = (
        "            dVM = (0.04 * self.v_msi.pow(2) + 5.0 * self.v_msi + 140.0 - self.u_msi + self.I_M)\n"
        "            self.v_msi += self.dt * dVM\n"
        "            self.u_msi += self.dt * (self.aM * (self.bM * self.v_msi - self.u_msi))"
    )
    blockM_new = (
        "            for _hs in range(2):\n"
        "                _dt_h = self.dt / 2.0\n"
        "                dVM = (0.04 * self.v_msi.pow(2) + 5.0 * self.v_msi + 140.0 - self.u_msi + self.I_M)\n"
        "                self.v_msi += _dt_h * dVM\n"
        "                self.u_msi += _dt_h * (self.aM * (self.bM * self.v_msi - self.u_msi))"
    )

    blockMi_old = (
        "            dVMi = (0.04 * self.v_msi_inh.pow(2) + 5.0 * self.v_msi_inh + 140.0 - self.u_msi_inh + self.I_M_inh)\n"
        "            self.v_msi_inh += self.dt * dVMi\n"
        "            self.u_msi_inh += self.dt * (self.aMi * (self.bMi * self.v_msi_inh - self.u_msi_inh))"
    )
    blockMi_new = (
        "            for _hs in range(2):\n"
        "                _dt_h = self.dt / 2.0\n"
        "                dVMi = (0.04 * self.v_msi_inh.pow(2) + 5.0 * self.v_msi_inh + 140.0 - self.u_msi_inh + self.I_M_inh)\n"
        "                self.v_msi_inh += _dt_h * dVMi\n"
        "                self.u_msi_inh += _dt_h * (self.aMi * (self.bMi * self.v_msi_inh - self.u_msi_inh))"
    )

    for old, new in [(blockA_old, blockA_new), (blockV_old, blockV_new),
                     (blockM_old, blockM_new), (blockMi_old, blockMi_new)]:
        assert old in src, f"Could not find layer block:\n{old}"
        src = src.replace(old, new)

    ns = {}
    glb = TRN.__dict__.copy()
    exec(src, glb, ns)
    new_fn = ns["_patched_update_all_layers_batch"]
    TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch = new_fn


def reset_update_method():
    TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch = _orig_update


# ──────────────────────────────────────────────────────────────────────
# Measurement
# ──────────────────────────────────────────────────────────────────────
def measure_pfusion(net, offsets, n_trials, T, D, stim_in, seed):
    rng = np.random.default_rng(seed)
    orig_default_rng = np.random.default_rng
    np.random.default_rng = lambda *a, **k: rng
    try:
        p_fusion, all_fused = compute_tbw_temporal_fusion_persep(
            net, offsets, n_trials=n_trials, T=T, D=D, stim_in=stim_in,
            sigma=2.0, valley_threshold=0.4, min_peak_height=0.2,
            min_peak_separation=3, min_total=10.0,
        )
    finally:
        np.random.default_rng = orig_default_rng
    return p_fusion, all_fused


def fit_hw(offsets_ms, mean_fusion):
    fit = fit_psychometric_curve_improved(np.asarray(offsets_ms),
                                          np.asarray(mean_fusion))
    return fit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", type=int, nargs="+", default=[0])
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--offset_step_frames", type=int, default=2)
    p.add_argument("--offset_max_frames", type=int, default=50)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--mode", choices=["both", "dt01", "dt005"], default="both")
    p.add_argument("--T", type=int, default=60)
    p.add_argument("--D", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--freeze-agc", action="store_true")
    p.add_argument("--preserve-delays-ms", action="store_true")
    p.add_argument("--exp-decay", action="store_true")
    p.add_argument("--izhi-half", action="store_true")
    p.add_argument("--tag", type=str, default="default",
                   help="Label for output file")
    return p.parse_args()


def main():
    args = parse_args()

    # Apply substep-loop patches first (must be before any net is built)
    if args.exp_decay:
        _wrap_exp_decay()
    if args.izhi_half:
        # Need to start from the (possibly already-patched) version; re-patch
        # from the freshly imported function so they compose.
        if args.exp_decay:
            # If both, re-wrap on top of exp_decay
            _orig2 = TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch
            # We do izhi_half via source rewrite from _orig (the original),
            # losing exp_decay.  Combine explicitly by re-applying both.
            reset_update_method()
            _wrap_izhi_half_and_exp_decay()
        else:
            _wrap_izhi_half()

    base_dir = ROOT / "checkpoint"
    paths = [base_dir / f"msi_model_surr_10_{i:02d}.pt" for i in args.ckpts]

    step = args.offset_step_frames
    mx = args.offset_max_frames
    offsets = list(range(-mx, mx + 1, step))
    offsets_ms = [o * 10 for o in offsets]

    results = {}

    for tag, dt, nsub in [("dt01", 0.1, 100), ("dt005", 0.05, 200)]:
        if args.mode in ("both", tag):
            per_model_fusion = []
            per_model_hw = []
            t0 = time.time()
            for path in paths:
                net = load_msi_model(path, device=args.device)
                configure_dt(net, dt, nsub)
                if args.preserve_delays_ms:
                    preserve_delays_ms(net, ref_dt=0.1)
                net.plasticity_enabled = False
                if args.freeze_agc:
                    net.freeze_g_FFinh = True
                p_fusion, _ = measure_pfusion(
                    net, offsets, args.trials, args.T, args.D,
                    stim_in=1.0, seed=args.seed,
                )
                per_model_fusion.append(p_fusion)
                try:
                    fit = fit_hw(offsets_ms, p_fusion)
                    hw_single = float(fit.get("tbw", float("nan")))
                except Exception:
                    hw_single = float("nan")
                per_model_hw.append(hw_single)
                del net
                if str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()
            elapsed = time.time() - t0
            per_model_fusion = np.vstack(per_model_fusion)
            mean_fusion = per_model_fusion.mean(0)
            sem_fusion = per_model_fusion.std(0, ddof=1) / np.sqrt(len(paths)) \
                          if len(paths) > 1 else np.zeros_like(mean_fusion)
            fit = fit_hw(offsets_ms, mean_fusion)
            results[tag] = {
                "dt": dt,
                "n_substeps": nsub,
                "elapsed_s": elapsed,
                "per_model_fusion": per_model_fusion,
                "mean_fusion": mean_fusion,
                "sem_fusion": sem_fusion,
                "hw_pooled": float(fit.get("tbw", float("nan"))),
                "per_model_hw": np.asarray(per_model_hw, dtype=float),
                "offsets_ms": np.asarray(offsets_ms, dtype=float),
            }
            print(f"[{tag}] dt={dt} n_substeps={nsub} elapsed={elapsed:.1f}s")
            print(f"      pooled HW = {results[tag]['hw_pooled']:.1f} ms")
            for i, hw in zip(args.ckpts, per_model_hw):
                print(f"        ckpt{i:02d} HW = {hw:.1f} ms")

    if "dt01" in results and "dt005" in results:
        a = results["dt01"]["hw_pooled"]
        b = results["dt005"]["hw_pooled"]
        print()
        print(f"ΔHW (dt005 − dt01) = {b - a:+.1f} ms ({(b-a)/a*100:+.1f}%)")

    out = ROOT / "debug_dt" / f"repro_{args.tag}.npz"
    npz_payload = {}
    for tag, d in results.items():
        for k, v in d.items():
            key = f"{tag}_{k}"
            if isinstance(v, np.ndarray):
                npz_payload[key] = v
            elif np.isscalar(v):
                npz_payload[key] = np.asarray(v)
    np.savez(out, **npz_payload)
    print(f"Saved results to {out}")

    # Reset class method to avoid contamination on re-import
    reset_update_method()


def _wrap_izhi_half_and_exp_decay():
    """Apply both source rewrites: exp_decay first, then izhi_half on top."""
    _wrap_exp_decay()
    # Now izhi_half wraps the already-patched function — but our _wrap_izhi_half
    # reads from _orig_update (unpatched).  We re-do it from the now-patched
    # function via re-source-rewrite.
    import inspect
    src = inspect.getsource(
        TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch)
    src_lines = src.splitlines()
    base_indent = len(src_lines[0]) - len(src_lines[0].lstrip())
    src = "\n".join(line[base_indent:] for line in src_lines)
    src = src.replace("def _patched_update_all_layers_batch(",
                      "def _patched2_update_all_layers_batch(", 1)
    pairs = [
        ("            dVA = (0.04 * self.v_uniA.pow(2) + 5.0 * self.v_uniA + 140.0\n"
         "                   - self.u_uniA + self.I_A)\n"
         "            self.v_uniA += self.dt * dVA\n"
         "            self.u_uniA += self.dt * (self.aA * (self.bA * self.v_uniA - self.u_uniA))",
         "            for _hs in range(2):\n"
         "                _dt_h = self.dt / 2.0\n"
         "                dVA = (0.04 * self.v_uniA.pow(2) + 5.0 * self.v_uniA + 140.0\n"
         "                       - self.u_uniA + self.I_A)\n"
         "                self.v_uniA += _dt_h * dVA\n"
         "                self.u_uniA += _dt_h * (self.aA * (self.bA * self.v_uniA - self.u_uniA))"),
        ("            dVV = (0.04 * self.v_uniV.pow(2) + 5.0 * self.v_uniV + 140.0\n"
         "                   - self.u_uniV + self.I_V)\n"
         "            self.v_uniV += self.dt * dVV\n"
         "            self.u_uniV += self.dt * (self.aV * (self.bV * self.v_uniV - self.u_uniV))",
         "            for _hs in range(2):\n"
         "                _dt_h = self.dt / 2.0\n"
         "                dVV = (0.04 * self.v_uniV.pow(2) + 5.0 * self.v_uniV + 140.0\n"
         "                       - self.u_uniV + self.I_V)\n"
         "                self.v_uniV += _dt_h * dVV\n"
         "                self.u_uniV += _dt_h * (self.aV * (self.bV * self.v_uniV - self.u_uniV))"),
        ("            dVM = (0.04 * self.v_msi.pow(2) + 5.0 * self.v_msi + 140.0 - self.u_msi + self.I_M)\n"
         "            self.v_msi += self.dt * dVM\n"
         "            self.u_msi += self.dt * (self.aM * (self.bM * self.v_msi - self.u_msi))",
         "            for _hs in range(2):\n"
         "                _dt_h = self.dt / 2.0\n"
         "                dVM = (0.04 * self.v_msi.pow(2) + 5.0 * self.v_msi + 140.0 - self.u_msi + self.I_M)\n"
         "                self.v_msi += _dt_h * dVM\n"
         "                self.u_msi += _dt_h * (self.aM * (self.bM * self.v_msi - self.u_msi))"),
        ("            dVMi = (0.04 * self.v_msi_inh.pow(2) + 5.0 * self.v_msi_inh + 140.0 - self.u_msi_inh + self.I_M_inh)\n"
         "            self.v_msi_inh += self.dt * dVMi\n"
         "            self.u_msi_inh += self.dt * (self.aMi * (self.bMi * self.v_msi_inh - self.u_msi_inh))",
         "            for _hs in range(2):\n"
         "                _dt_h = self.dt / 2.0\n"
         "                dVMi = (0.04 * self.v_msi_inh.pow(2) + 5.0 * self.v_msi_inh + 140.0 - self.u_msi_inh + self.I_M_inh)\n"
         "                self.v_msi_inh += _dt_h * dVMi\n"
         "                self.u_msi_inh += _dt_h * (self.aMi * (self.bMi * self.v_msi_inh - self.u_msi_inh))"),
    ]
    for old, new in pairs:
        assert old in src, f"Could not find layer block in patched src:\n{old}"
        src = src.replace(old, new)
    ns = {}
    glb = TRN.__dict__.copy()
    exec(src, glb, ns)
    TRN.MultiBatchAudVisMSINetworkTime.update_all_layers_batch = \
        ns["_patched2_update_all_layers_batch"]


if __name__ == "__main__":
    main()
