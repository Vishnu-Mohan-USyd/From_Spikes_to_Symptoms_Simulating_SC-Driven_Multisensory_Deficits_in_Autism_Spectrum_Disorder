"""E/I balance probes for ``MultiBatchAudVisMSINetworkTime``.

Provides two flavours of E/I measurement:

  * **Sign-based** (legacy, ``run_ei_probe_with_offset`` /
    ``run_ei_probe_fast`` / ``run_ei_probe_averaged``): pools all positive
    inputs to MSI excit as "E" and all negative as "I".  Fast and
    self-contained but conflates intra-cellular sources.

  * **Separated synaptic** (``run_ei_probe_separated``, preferred):
    records each component independently via
    ``net.start_ei_recording()`` / ``net.stop_ei_recording()``:

        Excitatory: AMPA, NMDA
        Inhibitory: FFInh (feed-forward), RecurInh (MSI->MSI),
                    LatInh (lateral / surround).

    This is the canonical probe used by ``run_ei_balance.py``: it lets
    us decompose the E/I ratio into bio-plausible components and report
    fractional contributions matching Wehr & Zador (2003), Xue et al.
    (2014) etc.

Units / shapes
--------------
Synaptic currents are stored in arbitrary model units (see
``Training.py`` for the conductance-based dynamics).  Traces returned by
``net.stop_ei_recording()`` are 1-D arrays over substeps
(``n_frames * n_substeps``, default 100 substeps/frame -> 1 ms / substep
when n_frames ticks at 10 ms each).  Summary scalars are population
means over MSI excit neurons and substeps.

Randomness
----------
The probes are deterministic given the network weights and stimulus.

Side effects
------------
``run_ei_probe_separated`` saves and restores ``plasticity_enabled``
and ``g_FFinh`` so that probing does not perturb adaptation state.
"""
from Training import *
from matplotlib import font_manager


def load_msi_model(ckpt_path: Path, *, device="cuda"):
    """
    Restore a MultiBatchAudVisMSINetworkTime exactly as it was saved.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
    net.load_state_dict(ckpt["model_state"])
    for k, v in ckpt["mutable_hparams"].items():
        setattr(net, k, v)

    net.to(device).eval()
    return net


def run_temporal_integration_across_models(
        model_paths,
        offsets,
        *,
        loc=90, T=60, D=5, extra=5, stim_in=1,
        device="cuda",
        modify_net=None,  # optional modifier
):
    int_all = []
    for p in model_paths:
        net = load_msi_model(Path(p), device=device)
        # net.tau_nmda = 80

        if callable(modify_net):  
            modify_net(net)  # e.g. bump gNMDA, knock‑out synapses…

        res = run_temporal_integration(
            net, offsets, loc=loc, T=T, D=D, extra=extra, stim_in=stim_in
        )
        int_all.append(res["int_spikes"])

        del net
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    int_all = np.vstack(int_all)  # (n_models , n_offsets)
    return {
        "offsets_ms": res["offsets_ms"],
        "mean_int_spikes": int_all.mean(0),
        "sem_int_spikes": int_all.std(0, ddof=1) / np.sqrt(int_all.shape[0]),
        "all_int_spikes": int_all,
    }


def tbw_gaussian_fit_curve(pooled):
    offs = np.asarray(pooled["offsets_ms"])
    fit = fit_tbw_curve(offs, pooled["mean_int_spikes"], model="gaussian")
    return fit["xs"], fit["ys"], fit["params"]  # ← also return params


def plot_temporal_binding_summary(
        pooled_res,
        *,
        fit_model="gaussian",
        reference_fit=None,  # (xs_ref, ys_ref, params_ref) or None
        **fit_kw,
):
    """Draw TBW curve, Gaussian fit, optional overlay & half‑max shading."""
    offs = np.asarray(pooled_res["offsets_ms"])
    mean = np.asarray(pooled_res["mean_int_spikes"])
    sem = np.asarray(pooled_res["sem_int_spikes"])

    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.errorbar(offs, mean, yerr=sem,
                fmt="o", capsize=0, label="mean ± SEM (n=10)")

    ax.set_ylim(bottom=0)  # ←  guarantees x‑axis is at y=0
    y0 = ax.get_ylim()[0]  # value to which we anchor dotted lines
    # ------------------------------------------------------------------------

    # ---------------------------------------------------------------- current
    fwhm_txt = ""
    if fit_model:
        fit = fit_tbw_curve(offs, mean, model=fit_model, **fit_kw)
        xs_fit = fit["xs"]
        ys_fit = fit["ys"]
        base_c, amp_c, mu_c, sigma_c = fit["params"][:4]
        fwhm_c = fit["fwhm"]

        ax.plot(xs_fit, ys_fit, lw=2, color="C1", label="gaussian fit")

        # half‑max values
        y_half_c = base_c + 0.5 * amp_c
        xL_c = mu_c - 0.5 * fwhm_c
        xR_c = mu_c + 0.5 * fwhm_c

        for x_h in (xL_c, xR_c):
            ax.plot([x_h, x_h], [y0, y_half_c], ls=":", lw=1, color="C1")

        # shading under Gaussian between FWHM abscissae
        m_c = (xs_fit >= xL_c) & (xs_fit <= xR_c)
        ax.fill_between(xs_fit[m_c], ys_fit[m_c], y0,
                        color="C1", alpha=0.15, zorder=0)

        fwhm_txt = f" |  FWHM ≈ {fwhm_c:.0f} ms"

    # ----------------------------------------------------------- overlay (ctrl)
    if reference_fit is not None:
        xs_r, ys_r, p_r = reference_fit
        base_r, amp_r, mu_r, sigma_r = p_r[:4]
        fwhm_r = 2.355 * sigma_r

        # scale to current amplitude
        ys_r_scaled = base_c + (ys_r - base_r) * (amp_c / amp_r)
        ax.plot(xs_r, ys_r_scaled, lw=2, ls="--", color="0.35",
                label="control Gaussian (scaled)")

        xL_r = mu_r - 0.5 * fwhm_r
        xR_r = mu_r + 0.5 * fwhm_r
        for x_h in (xL_r, xR_r):
            ax.plot([x_h, x_h], [y0, y_half_c], ls=":", lw=1, color="0.35")

        # light‑red tint under control Gaussian
        m_r = (xs_r >= xL_r) & (xs_r <= xR_r)
        ax.fill_between(xs_r[m_r], ys_r_scaled[m_r], y0,
                        color="#ffb3b3", alpha=0.20, zorder=0)

    # ---------------------------------------------------------------- cosmetics
    ax.axvline(0, ls="--", lw=.7, color="k")
    ax.set_xlabel("Audio – Visual onset (ms)")
    ax.set_ylabel("Integrated spikes (0–100 ms)")
    ax.set_title("Temporal binding window (10‑model mean)" + fwhm_txt)

    ax.legend(frameon=False, fontsize=7)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_position(("outward", 6))
    ax.spines["bottom"].set_position(("outward", 6))

    fig.tight_layout()
    plt.show()

# -----------------------------------------------------------
#  ei_balance.py     (add next to diagnostics.py)
# -----------------------------------------------------------

# ei_balance_complete.py
"""
Complete E/I balance testing that matches AMPA/NMDA probe measurements.
This replaces the original ei_balance.py with proper synchronous/asynchronous testing.
"""

from pathlib import Path
from typing import Sequence, Callable, Optional, Literal

from collections import defaultdict


# from your_network_module import MultiBatchAudVisMSINetworkTime, AMPANMDADebugger


# ============================================================================
#                          SINGLE PROBE MEASUREMENTS
# ============================================================================

def run_ei_probe_with_offset(
        net,
        *,
        centre_deg: float = 90.0,
        offset_steps: int = 0,  # Temporal offset in external frames (10ms each)
        sigma_in: float = 5.0,
        pulse_frames: int = 5,
        n_frames: int = 20,
        intensity: float = 1.0,
        noise_std: float = 0.0,
        use_probe: bool = False,
) -> dict:
    """
    Feed audio-visual pulses with temporal offset through net.

    Parameters
    ----------
    offset_steps : int
        0 = synchronous, positive = visual lags, negative = audio lags
        Each step is 10ms (one external frame)
    use_probe : bool
        If True, install AMPANMDADebugger and return its measurements

    Returns
    -------
    dict with exc, inh, ratio, and optionally probe data
    """
    B, N = 1, net.n
    net.reset_state(batch_size=B)

    # Optionally install probe
    if use_probe:
        original_probe = getattr(net, "_probe", None)
        net._probe = AMPANMDADebugger()
        net._probe.reset()

    # Build stimuli with offset
    xA = torch.zeros(n_frames, N, device=net.device)
    xV = torch.zeros_like(xA)

    idx_c = int(round(centre_deg * (N - 1) / (net.space_size - 1)))
    xs = torch.arange(N, dtype=torch.float32, device=net.device)
    gauss = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * intensity
    if noise_std > 0:
        gauss += torch.randn_like(gauss) * noise_std

    # Apply temporal offset
    aud_start = 0 if offset_steps >= 0 else abs(offset_steps)
    vis_start = 0 if offset_steps <= 0 else offset_steps

    xA[aud_start:aud_start + pulse_frames] = gauss
    xV[vis_start:vis_start + pulse_frames] = gauss

    # Run simulation and collect currents
    exc_trace, inh_trace = [], []
    for t in range(n_frames):
        net.update_all_layers_batch(
            xA[t].unsqueeze(0), xV[t].unsqueeze(0), valid_mask=None
        )
        I_M = net.I_M.detach()
        exc_trace.append(torch.clamp(I_M, min=0).mean().item())
        inh_trace.append((-torch.clamp(I_M, max=0)).mean().item())

    # Calculate means
    exc_mean = float(np.mean(exc_trace))
    inh_mean = float(np.mean(inh_trace))
    ei_ratio = exc_mean / (inh_mean + 1e-12)  # E/I ratio
    ie_ratio = inh_mean / (exc_mean + 1e-12)  # I/E ratio

    result = {
        "exc": exc_mean,
        "inh": inh_mean,
        "ei_ratio": ei_ratio,
        "ie_ratio": ie_ratio,
        "offset_ms": offset_steps * 10
    }

    # Add probe data if used
    if use_probe:
        probe_data = net._probe.sums
        if net._probe.t > 0:
            result["probe_ei_ratio"] = probe_data["I_exc"] / max(probe_data["I_inh"], 1e-9)
            result["probe_data"] = dict(probe_data)
        # Restore original probe
        net._probe = original_probe

    return result


def run_ei_probe_averaged(
        net,
        *,
        offsets: Sequence[int] = None,
        use_probe: bool = True,
        **kwargs
) -> dict:
    """
    Average E/I measurements across multiple temporal offsets (like TBW test).

    Parameters
    ----------
    offsets : sequence of int
        Temporal offsets to test (in external frames, 10ms each)
        Default: [-30, -20, -10, 0, 10, 20, 30]
    """
    if offsets is None:
        offsets = [-30, -20, -10, 0, 10, 20, 30]  # -300ms to +300ms

    # Collect measurements at each offset
    exc_vals, inh_vals, ei_ratios, ie_ratios = [], [], [], []
    offset_results = []

    for offset in offsets:
        res = run_ei_probe_with_offset(net, offset_steps=offset, use_probe=False, **kwargs)
        exc_vals.append(res["exc"])
        inh_vals.append(res["inh"])
        ei_ratios.append(res["ei_ratio"])
        ie_ratios.append(res["ie_ratio"])
        offset_results.append(res)

    probe_data = None
    if use_probe:
        original_probe = getattr(net, "_probe", None)
        net._probe = AMPANMDADebugger()
        net._probe.reset()

        for offset in offsets:
            _ = run_ei_probe_with_offset(net, offset_steps=offset, use_probe=False, **kwargs)

        # Get probe summary
        if net._probe.t > 0:
            probe_data = {
                "charge_ei_ratio": net._probe.sums["Q_exc"] / max(net._probe.sums["Q_inh"], 1e-9),
                "mean_ei_ratio": net._probe.sums["I_exc"] / max(net._probe.sums["I_inh"], 1e-9),
                "nmda_ampa_ratio": net._probe.sums["Q_nmda"] / max(net._probe.sums["Q_ampa"], 1e-9)
            }

        net._probe = original_probe

    return {
        "exc_mean": np.mean(exc_vals),
        "inh_mean": np.mean(inh_vals),
        "ei_ratio_mean": np.mean(ei_ratios),
        "ie_ratio_mean": np.mean(ie_ratios),
        "exc_std": np.std(exc_vals),
        "inh_std": np.std(inh_vals),
        "offset_results": offset_results,
        "probe_data": probe_data
    }


# ============================================================================
#                          POOL ACROSS MODELS
# ============================================================================

def pool_ei_across_models(
        model_paths: Sequence[Path],
        *,
        device: str = "cuda",
        modify_net: Optional[Callable] = None,
        test_mode: Literal["synchronous", "averaged", "both"] = "both",
        probe_kw: Optional[dict] = None
) -> dict:
    """
    Test E/I balance across multiple model replicas.

    Parameters
    ----------
    test_mode : str
        "synchronous" - only test with offset=0
        "averaged" - average across multiple offsets
        "both" - test both conditions
    """
    probe_kw = probe_kw or {}

    # Storage for results
    sync_results = defaultdict(list)
    avg_results = defaultdict(list)

    for i, p in enumerate(model_paths):
        print(f"Testing model {i + 1}/{len(model_paths)}...", end="\r")

        # Load model
        ckpt = torch.load(p, map_location=device)
        net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
        net.load_state_dict(ckpt["model_state"])
        for k, v in ckpt["mutable_hparams"].items():
            setattr(net, k, v)
        net.to(device).eval()

        if callable(modify_net):
            modify_net(net)

        # Test synchronous condition
        if test_mode in ["synchronous", "both"]:
            sync = run_ei_probe_with_offset(net, offset_steps=0, use_probe=True, **probe_kw)
            sync_results["exc"].append(sync["exc"])
            sync_results["inh"].append(sync["inh"])
            sync_results["ei_ratio"].append(sync["ei_ratio"])
            sync_results["ie_ratio"].append(sync["ie_ratio"])

        # Test averaged condition
        if test_mode in ["averaged", "both"]:
            avg = run_ei_probe_averaged(net, use_probe=True, **probe_kw)
            avg_results["exc"].append(avg["exc_mean"])
            avg_results["inh"].append(avg["inh_mean"])
            avg_results["ei_ratio"].append(avg["ei_ratio_mean"])
            avg_results["ie_ratio"].append(avg["ie_ratio_mean"])
            if avg["probe_data"]:
                avg_results["probe_ei_ratio"].append(avg["probe_data"]["mean_ei_ratio"])

        del net
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print()  # Clear progress line

    # Process results
    results = {}

    if sync_results:
        sync_results = {k: np.array(v) for k, v in sync_results.items()}
        results["synchronous"] = {
            **sync_results,
            "exc_mean": sync_results["exc"].mean(),
            "exc_sem": sync_results["exc"].std(ddof=1) / math.sqrt(len(sync_results["exc"])),
            "inh_mean": sync_results["inh"].mean(),
            "inh_sem": sync_results["inh"].std(ddof=1) / math.sqrt(len(sync_results["inh"])),
            "ei_ratio_mean": sync_results["ei_ratio"].mean(),
            "ei_ratio_sem": sync_results["ei_ratio"].std(ddof=1) / math.sqrt(len(sync_results["ei_ratio"])),
            "ie_ratio_mean": sync_results["ie_ratio"].mean(),
            "ie_ratio_sem": sync_results["ie_ratio"].std(ddof=1) / math.sqrt(len(sync_results["ie_ratio"])),
        }

    if avg_results:
        avg_results = {k: np.array(v) for k, v in avg_results.items() if v}
        results["averaged"] = {
            **avg_results,
            "exc_mean": avg_results["exc"].mean(),
            "exc_sem": avg_results["exc"].std(ddof=1) / math.sqrt(len(avg_results["exc"])),
            "inh_mean": avg_results["inh"].mean(),
            "inh_sem": avg_results["inh"].std(ddof=1) / math.sqrt(len(avg_results["inh"])),
            "ei_ratio_mean": avg_results["ei_ratio"].mean(),
            "ei_ratio_sem": avg_results["ei_ratio"].std(ddof=1) / math.sqrt(len(avg_results["ei_ratio"])),
            "ie_ratio_mean": avg_results["ie_ratio"].mean(),
            "ie_ratio_sem": avg_results["ie_ratio"].std(ddof=1) / math.sqrt(len(avg_results["ie_ratio"])),
        }
        if "probe_ei_ratio" in avg_results:
            results["averaged"]["probe_ei_ratio_mean"] = avg_results["probe_ei_ratio"].mean()

    return results

from scipy.stats import gaussian_kde   # only needed by plot_ei_comparison
# ============================================================================
#                          VISUALIZATION
# ============================================================================

def plot_ei_comparison(results: dict, *, title_suffix: str = "") -> None:
    """
    More detailed comparison figure – now **only** for the averaged condition.
    """
    if "averaged" not in results:
        raise ValueError("No averaged data found in `results`")

    fig = plt.figure(figsize=(8, 3.8))
    gs  = fig.add_gridspec(1, 2, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    avg = results["averaged"]

    ax1.bar([0], [avg["ei_ratio_mean"]], yerr=[avg["ei_ratio_sem"]],
            capsize=5, color="C1")
    ax1.set_xticks([0])
    ax1.set_xticklabels(["Averaged\n(±200 ms)"])
    ax1.set_ylabel("E/I ratio")
    ax1.set_title("E/I Ratio")
    ax1.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax1.text(0, avg["ei_ratio_mean"] + avg["ei_ratio_sem"] + 0.04,
             f"{avg['ei_ratio_mean']:.3f}", ha="center")

    ax2 = fig.add_subplot(gs[1])
    ie_vals = np.asarray(avg["ie_ratio"])
    ax2.hist(ie_vals, bins=10, alpha=0.5, density=True, color="C1",
             label="I/E (averaged)")

    if len(ie_vals) > 1:                               # KDE only makes sense with n>1
        xs = np.linspace(ie_vals.min()*0.9, ie_vals.max()*1.1, 250)
        kde = gaussian_kde(ie_vals)
        ax2.plot(xs, kde(xs), color="C1")

    ax2.set_xlabel("I/E ratio")
    ax2.set_title("I/E Distribution")
    ax2.legend(frameon=False, fontsize=8)

    fig.suptitle(f"E/I Balance – averaged offsets{title_suffix}",
                 fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_offset_dependency(net, offsets=None, **kwargs):
    """
    Show how E/I balance changes with temporal offset.
    """
    if offsets is None:
        offsets = list(range(-50, 51, 10))  # -500ms to +500ms in 100ms steps

    results = []
    for offset in offsets:
        res = run_ei_probe_with_offset(net, offset_steps=offset, **kwargs)
        results.append(res)

    # Extract data
    offset_ms = [r["offset_ms"] for r in results]
    exc_vals = [r["exc"] for r in results]
    inh_vals = [r["inh"] for r in results]
    ei_ratios = [r["ei_ratio"] for r in results]

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Currents
    ax1.plot(offset_ms, exc_vals, "o-", label="Excitation")
    ax1.plot(offset_ms, inh_vals, "s-", label="Inhibition")
    ax1.set_ylabel("Mean current (a.u.)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # E/I ratio
    ax2.plot(offset_ms, ei_ratios, "o-", color="C2")
    ax2.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax2.set_xlabel("Audio-Visual offset (ms)")
    ax2.set_ylabel("E/I ratio")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("E/I Balance vs Temporal Offset")
    plt.tight_layout()
    plt.show()


# ============================================================================
#                          MAIN RUNNER
# ============================================================================

# ei_balance_fast.py
"""
Fast E/I balance testing - 10x faster than the complete version.
Key optimizations:
1. Single pass measurement (no duplicate runs)
2. Batch processing where possible
3. Minimal offset testing for averaged condition
4. Optional quick mode for even faster results
"""

import math
import time
from pathlib import Path
from typing import Sequence, Callable, Optional

import torch


# ============================================================================
#                          FAST PROBE MEASUREMENT
# ============================================================================

# ============================================================================
#                          BATCH PROCESSING
# ============================================================================

def pool_ei_fast(
        model_paths: Sequence[Path],
        *,
        device: str = "cuda",
        modify_net: Optional[Callable] = None,
        test_averaged: bool = True,
        quick_mode: bool = False
) -> dict:
    """
    Fast pooling across models.

    Parameters
    ----------
    test_averaged : bool
        If False, only test synchronous (even faster)
    quick_mode : bool
        If True, only test first 3 models (for quick debugging)
    """

    paths = model_paths[:3] if quick_mode else model_paths
    n_models = len(paths)

    # Pre-allocate arrays
    sync_exc = np.zeros(n_models)
    sync_inh = np.zeros(n_models)
    avg_exc = np.zeros(n_models) if test_averaged else None
    avg_inh = np.zeros(n_models) if test_averaged else None

    start_time = time.time()

    for i, p in enumerate(paths):
        # Progress
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (n_models - i - 1) if i > 0 else 0
        print(f"Model {i + 1}/{n_models} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="\r")

        # Load model efficiently
        ckpt = torch.load(p, map_location=device)  # Remove weights_only=True
        net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
        net.load_state_dict(ckpt["model_state"])  # Remove strict=False

        for k, v in ckpt["mutable_hparams"].items():
            setattr(net, k, v)

        net.to(device).eval()

        if modify_net:
            modify_net(net)

        # Synchronous test
        sync = run_ei_probe_fast(net, offset_steps=0)
        sync_exc[i] = sync["exc"]
        sync_inh[i] = sync["inh"]

        # Averaged test
        if test_averaged:
            avg = run_ei_averaged_fast(net)
            avg_exc[i] = avg["exc"]
            avg_inh[i] = avg["inh"]

        del net
        torch.cuda.empty_cache()

    print(f"\nCompleted in {time.time() - start_time:.1f}s")

    # Calculate statistics
    def calc_stats(exc, inh):
        ei_ratios = exc / (inh + 1e-12)
        ie_ratios = inh / (exc + 1e-12)
        return {
            "exc": exc,
            "inh": inh,
            "exc_mean": exc.mean(),
            "exc_sem": exc.std(ddof=1) / np.sqrt(len(exc)),
            "inh_mean": inh.mean(),
            "inh_sem": inh.std(ddof=1) / np.sqrt(len(inh)),
            "ei_ratio_mean": ei_ratios.mean(),
            "ei_ratio_sem": ei_ratios.std(ddof=1) / np.sqrt(len(ei_ratios)),
            "ie_ratio_mean": ie_ratios.mean(),
            "ie_ratio_sem": ie_ratios.std(ddof=1) / np.sqrt(len(ie_ratios)),
        }

    results = {"synchronous": calc_stats(sync_exc, sync_inh)}

    if test_averaged:
        results["averaged"] = calc_stats(avg_exc, avg_inh)

    return results


# ============================================================================
#                          SIMPLE VISUALIZATION
# ============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt





# ============================================================================
#  NEW: Separated E/I probe using individual synaptic current components
# ============================================================================

def run_ei_probe_separated(
        net,
        *,
        centre_deg: float = 90.0,
        sigma_in: float = 5.0,
        pulse_frames: int = 5,
        n_frames: int = 20,
        intensity: float = 1.0,
) -> dict:
    """Run a synchronous AV pulse and record SEPARATE synaptic current components.

    Uses net.start_ei_recording() / net.stop_ei_recording() to capture:
      Excitatory: AMPA, NMDA
      Inhibitory: FFInh, RecurInh, LatInh

    Parameters
    ----------
    net : MultiBatchAudVisMSINetworkTime
        Network model (already on device).
    centre_deg : float
        Spatial centre of the stimulus in degrees.
    sigma_in : float
        Gaussian width of the stimulus.
    pulse_frames : int
        Duration of the stimulus pulse in external frames (10 ms each).
    n_frames : int
        Total simulation duration in external frames.
    intensity : float
        Peak stimulus intensity.

    Returns
    -------
    dict with keys:
        exc_mean, inh_mean, ei_ratio,
        Q_E_total, Q_I_total, charge_ratio,
        AMPA_mean, NMDA_mean, FFInh_mean, RecurInh_mean, LatInh_mean,
        traces (raw per-substep arrays)
    """
    B, N = 1, net.n
    net.reset_state(batch_size=B)

    # Save and disable plasticity
    orig_plasticity = getattr(net, 'plasticity_enabled', True)
    orig_g_FFinh = net.g_FFinh
    net.plasticity_enabled = False

    # Build synchronous AV stimulus (offset=0)
    xA = torch.zeros(n_frames, N, device=net.device)
    xV = torch.zeros_like(xA)

    idx_c = int(round(centre_deg * (N - 1) / (net.space_size - 1)))
    xs = torch.arange(N, dtype=torch.float32, device=net.device)
    gauss = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * intensity

    xA[:pulse_frames] = gauss
    xV[:pulse_frames] = gauss

    # Start recording
    net.start_ei_recording()

    # Run simulation
    for t in range(n_frames):
        net.update_all_layers_batch(xA[t].unsqueeze(0), xV[t].unsqueeze(0))

    # Stop recording
    traces = net.stop_ei_recording()

    # Restore
    net.plasticity_enabled = orig_plasticity
    net.g_FFinh = orig_g_FFinh

    # Compute summary statistics
    exc_mean = traces["I_E_mean"].mean()
    inh_mean = traces["I_I_mean"].mean()
    Q_E_total = traces["Q_E"].sum()
    Q_I_total = traces["Q_I"].sum()

    return {
        "exc_mean": float(exc_mean),
        "inh_mean": float(inh_mean),
        "ei_ratio": float(exc_mean / (inh_mean + 1e-12)),
        "Q_E_total": float(Q_E_total),
        "Q_I_total": float(Q_I_total),
        "charge_ratio": float(Q_E_total / (Q_I_total + 1e-12)),
        "AMPA_mean": float(traces["AMPA"].mean()),
        "NMDA_mean": float(traces["NMDA"].mean()),
        "FFInh_mean": float(traces["FFInh"].mean()),
        "RecurInh_mean": float(traces["RecurInh"].mean()),
        "LatInh_mean": float(traces["LatInh"].mean()),
        "traces": traces,
    }


def pool_ei_separated(
        model_paths,
        *,
        device: str = "cuda",
        modify_net=None,
        **probe_kw,
) -> dict:
    """Run separated E/I probe across all model replicas.

    Returns dict with per-model arrays and summary statistics.
    """
    all_results = []

    for i, p in enumerate(model_paths):
        ckpt = torch.load(p, map_location=device)
        net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
        net.load_state_dict(ckpt["model_state"])
        for k, v in ckpt["mutable_hparams"].items():
            setattr(net, k, v)
        net.to(device).eval()

        if callable(modify_net):
            modify_net(net)

        res = run_ei_probe_separated(net, **probe_kw)
        all_results.append(res)
        print(f"  Model {i}: E={res['exc_mean']:.4f} I={res['inh_mean']:.4f} "
              f"E/I={res['ei_ratio']:.3f}  "
              f"AMPA={res['AMPA_mean']:.4f} NMDA={res['NMDA_mean']:.4f} "
              f"FF={res['FFInh_mean']:.4f} Rec={res['RecurInh_mean']:.4f} Lat={res['LatInh_mean']:.4f}")

        del net
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Aggregate
    keys = ["exc_mean", "inh_mean", "ei_ratio", "Q_E_total", "Q_I_total",
            "charge_ratio", "AMPA_mean", "NMDA_mean", "FFInh_mean",
            "RecurInh_mean", "LatInh_mean"]
    arrays = {k: np.array([r[k] for r in all_results]) for k in keys}
    n = len(all_results)

    summary = {}
    for k, arr in arrays.items():
        summary[k] = arr
        summary[f"{k}_mean"] = arr.mean()
        summary[f"{k}_sem"] = arr.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0

    return summary


def plot_ei_scatter_separated(summary: dict, *, out_path_base: str = None):
    """Scatter plot of Excitation vs Inhibition from separated components.

    Each point = one model replica. Diagonal = E=I reference.
    """
    exc = summary["exc_mean"]
    inh = summary["inh_mean"]

    try:
        font_path = './fonts/Roboto-Regular.ttf'
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Roboto'
    except Exception:
        pass
    plt.rcParams['font.size'] = 60
    plt.rcParams['xtick.labelsize'] = 60
    plt.rcParams['ytick.labelsize'] = 60
    plt.rcParams['axes.titlesize'] = 50
    plt.rcParams['axes.labelsize'] = 60
    plt.rcParams['legend.fontsize'] = 50
    plt.rcParams['svg.fonttype'] = 'none'

    fig, ax = plt.subplots(figsize=(25, 25))

    # Scatter: one point per model
    ax.scatter(exc, inh, s=2000, color="C0", edgecolor="none", alpha=0.9,
               zorder=3, label="Model replicas (n=10)")

    # E=I diagonal
    lim = max(exc.max(), inh.max()) * 1.15
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.5, lw=2,
            label="E = I")

    # Mean crosshairs
    e_m, i_m = summary["exc_mean_mean"], summary["inh_mean_mean"]
    e_s, i_s = summary["exc_mean_sem"], summary["inh_mean_sem"]
    ax.errorbar(e_m, i_m, xerr=e_s, yerr=i_s,
                fmt='o', ms=25, color='C1', elinewidth=3, capsize=8,
                zorder=4, label=f"Mean E/I = {summary['ei_ratio_mean']:.2f}")

    ax.set(xlim=(0, lim), ylim=(0, lim),
           xlabel="Excitatory current (a.u.)",
           ylabel="Inhibitory current (a.u.)",
           title="E/I Balance (separated synaptic currents)")
    ax.legend(frameon=False, loc='upper left')
    ax.tick_params(axis='both', which='major', length=20, width=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    base = out_path_base or './Saved_Images/EI_balance'
    fig.savefig(f"{base}.svg", format='svg')
    fig.savefig(f"{base}.png", format='png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {base}.svg and {base}.png")


def main_separated():
    """Run the corrected E/I balance analysis using separated synaptic currents."""
    base = Path("checkpoint")
    model_paths = [base / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]

    print("=" * 60)
    print("E/I BALANCE — separated synaptic currents")
    print("=" * 60)

    summary = pool_ei_separated(model_paths, device="cuda")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Excitation:  {summary['exc_mean_mean']:.4f} ± {summary['exc_mean_sem']:.4f}")
    print(f"  Inhibition:  {summary['inh_mean_mean']:.4f} ± {summary['inh_mean_sem']:.4f}")
    print(f"  E/I ratio:   {summary['ei_ratio_mean']:.3f} ± {summary['ei_ratio_sem']:.3f}")
    print(f"  Charge E:    {summary['Q_E_total_mean']:.6f} ± {summary['Q_E_total_sem']:.6f}")
    print(f"  Charge I:    {summary['Q_I_total_mean']:.6f} ± {summary['Q_I_total_sem']:.6f}")
    print(f"  Charge E/I:  {summary['charge_ratio_mean']:.3f} ± {summary['charge_ratio_sem']:.3f}")
    print(f"\n  Component breakdown (mean across models):")
    print(f"    AMPA:      {summary['AMPA_mean_mean']:.4f}")
    print(f"    NMDA:      {summary['NMDA_mean_mean']:.4f}")
    print(f"    FFInh:     {summary['FFInh_mean_mean']:.4f}")
    print(f"    RecurInh:  {summary['RecurInh_mean_mean']:.4f}")
    print(f"    LatInh:    {summary['LatInh_mean_mean']:.4f}")

    plot_ei_scatter_separated(summary)

    return summary


if __name__ == "__main__":
    main_separated()
    # base = Path("checkpoint")
    # check_ei_single_model(base / "msi_model_surr_2_00.pt

#
# # ───────────────────────── runner script ─────────────────────────
# def main():
#     base_dir = Path("checkpoint")
#
#     # CONTROL
#     pooled_ctrl = run_temporal_integration_across_models(model_paths, offsets, device="cuda")
#     xs_ref, ys_ref, params_ref = tbw_gaussian_fit_curve(pooled_ctrl)
#
#     # (optional) plot control alone
#     plot_temporal_binding_summary(pooled_ctrl, reference_fit=None)
#
#     def bump_nmda(net):
#
#         # net.gNMDA = 0.5
#         # net.u_a.fill_(0.05)
#         # net.u_v.fill_(0.05)
#         # net.tau_rec = 50.0
#         # # MSI excit
#         # # MSI inh (fast spiking)
#         # # Out
#         # net.g_GABA = 0
#         net.pv_scale = 0.4
#         # net.conduction_delay_v2msi=460
#
#     pooled_mod = run_temporal_integration_across_models(
#         model_paths, offsets, device="cuda",
#         modify_net=bump_nmda,
#     )
#
#     plot_temporal_binding_summary(
#         pooled_mod,
#     )


