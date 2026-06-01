"""TBW (Temporal Binding Window) measurement pipeline.

This module measures and plots the temporal binding window of trained
``MultiBatchAudVisMSINetworkTime`` checkpoints.  The canonical metric is
``temporal_fusion`` P(fusion), evaluated by the ``is_temporally_fused``
peak/valley classifier on the MSI population spike-count time-series; an
``enhancement`` (per-trial AV-vs-best-unimodal threshold) variant is also
provided for cross-checks.

Pipeline (used by ``generate_all_fresh.py`` and ``replot_all_cosmetic.py``):

  1. ``load_msi_model``                  — restore checkpoint, plasticity off.
  2. ``compute_tbw_temporal_fusion_persep`` — batched forward pass over
     n_offsets x n_trials; classify each MSI time-series as fused/not.
  3. ``run_fusion_across_models``        — pool across checkpoints, returns
     mean +/- SEM P(fusion) per SOA.
  4. ``fit_psychometric_curve_improved`` — asymmetric pedestal fit;
     ``_find_crossings`` extracts the 50%-fusion x-coordinates that define
     the TBW half-width.

Units / shapes / randomness
---------------------------
Time:  one external timestep = 10 ms (n_substeps = 100 substeps per frame).
SOA:   passed as integer ``offsets`` in macro-steps (10 ms each); plotted
       and stored in ms (offsets_ms).
Trials: random spatial location per trial, drawn from
       ``np.random.default_rng()`` (no global seed pinned in this module).
Outputs are returned as ``np.ndarray`` (P(fusion) per offset) plus the
per-trial binary fusion flags for diagnostics.

Side effects
------------
Functions that load a checkpoint move the network to ``device``, set
``net.eval()`` and ``net.plasticity_enabled = False``.  GPU memory is
explicitly freed (``torch.cuda.empty_cache()``) after each model in the
pooled drivers.
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
    # net.g_FFinh = 0.1
    net.to(device).eval()
    net.plasticity_enabled = False  # disable plasticity during evaluation
    return net


# ──────────────────────────────────────────────────────────────────
# task#185 (minimal-arch §5.5.2) — FF inh perturbation helper
# ──────────────────────────────────────────────────────────────────
def has_agc(net) -> bool:
    """True if `net` has AGC state (pre-task#185 architecture).

    Used by perturbation entry points to dispatch between the legacy
    pv_nmda/targ_ratio modification path and the new W-scaling path.
    The minimal-architecture (post-task#185) network has no AGC state.
    """
    return hasattr(net, "targ_ratio")


def apply_ff_inh_perturbation(net, scale: float = 1.2) -> None:
    """Multiplicatively scale FF inhibitory weights onto MSI excit.

    Replaces the pre-redesign `setattr(n, "pv_nmda", 0.8); setattr(n,
    "targ_ratio", 0.8)` AGC-tweak approach, which becomes a structural
    NO-OP after task#185 removes AGC entirely. Per minimal-architecture
    spec §5.5.2 (review_response/sci_inhibition_minimal_architecture.md).

    Args:
        net: a MultiBatchAudVisMSINetworkTime instance.
        scale: multiplicative scale factor for W_inA_inh + W_inV_inh.
               Default 1.2 (+20%) per spec §5.5.3 initial value.
               scale > 1.0 → strengthen FF inhibition (narrower TBW/SBW).
               scale < 1.0 → weaken FF inhibition (wider TBW/SBW).

    Note: this is an IN-PLACE modification (matches CONDITIONS-dict lambda
    convention in the perturbation runner scripts). The caller is
    responsible for re-loading a fresh checkpoint between conditions, or
    saving/restoring via `_ff_inh_save()` / `_ff_inh_restore()` if reusing
    the same network across conditions.

    Backward compatibility: on legacy AGC-active networks (`has_agc()` =
    True), this function STILL applies the W-scaling. The legacy
    pv_nmda/targ_ratio path is no longer recommended because it relies on
    the AGC feedback loop to express the perturbation, and that feedback
    is convolved with the AGC's own gain control. If strict legacy
    behaviour is required, the caller can branch on `has_agc(net)` and
    invoke the legacy lambdas explicitly.
    """
    import torch as _t
    with _t.no_grad():
        net.W_inA_inh.mul_(float(scale))
        net.W_inV_inh.mul_(float(scale))


def _ff_inh_save(net):
    """Save (W_inA_inh, W_inV_inh) for later restoration."""
    return (net.W_inA_inh.clone(), net.W_inV_inh.clone())


def _ff_inh_restore(net, saved) -> None:
    """Restore W_inA_inh, W_inV_inh from a tuple produced by _ff_inh_save()."""
    import torch as _t
    sa, sv = saved
    with _t.no_grad():
        net.W_inA_inh.copy_(sa)
        net.W_inV_inh.copy_(sv)


# ──────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────
def run_temporal_integration_across_models(
        model_paths,
        offsets,
        *,
        loc=90,
        T=60,
        D=5,
        extra=5,
        stim_in=1.0,
        device="cuda",
        modify_net=None,
        log_charges: bool = False,           # log charge totals
):
    """
    Identical API to the original helper **plus** an optional live print‑out
    of NMDA/AMPA and E/I charge ratios for *each* network that is evaluated.

    Parameters
    ----------
    log_charges : bool
        Forwarded to `run_temporal_integration(..., log_charges=…)`.

    Returns
    -------
    pooled : dict
        {
          "offsets_ms"      : list[int],
          "mean_int_spikes" : ndarray (n_offsets,),
          "sem_int_spikes"  : ndarray (n_offsets,),
          "all_int_spikes"  : ndarray (n_models , n_offsets),
        }
        (unchanged from the original implementation)
    """
    all_int = []

    for path in model_paths:
        net = load_msi_model(Path(path), device=device)

        # Optional one‑off tweak
        if callable(modify_net):
            modify_net(net)

        res = run_temporal_integration(
            net, offsets,
            loc=loc, T=T, D=D, extra=extra, stim_in=stim_in,
            log_charges=log_charges
        )
        all_int.append(res["int_spikes"])

        # explicit clean‑up for big GPU models
        del net
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    all_int = np.vstack(all_int)                         # (n_models , n_offsets)
    return {
        "offsets_ms":      res["offsets_ms"],
        "mean_int_spikes": all_int.mean(0),
        "sem_int_spikes":  all_int.std(0, ddof=1) / np.sqrt(all_int.shape[0]),
        "all_int_spikes":  all_int,
    }


def fit_tbw_curve_improved(offs_ms, int_spikes, *, model="gaussian", p0=None,
                           weight_by_value=True, robust_baseline=True,
                           smooth_before_fit=False, smooth_sigma=1.0):
    """
    Improved fit of temporal binding window with better peak capture.

    Parameters
    ----------
    offs_ms : array-like
        Audio-visual onset asynchronies in milliseconds
    int_spikes : array-like
        Integrated spike counts
    model : str
        "gaussian" or "flattop"
    p0 : tuple, optional
        Initial parameter guesses
    weight_by_value : bool
        If True, weight points by their y-value to emphasize peak
    robust_baseline : bool
        If True, use robust baseline estimation
    smooth_before_fit : bool
        If True, apply light smoothing before fitting
    smooth_sigma : float
        Smoothing parameter if smooth_before_fit is True
    """
    offs = np.asarray(offs_ms, dtype=float)
    ints = np.asarray(int_spikes, dtype=float)

    # Optional smoothing to reduce noise
    if smooth_before_fit:
        ints_smooth = gaussian_filter1d(ints, sigma=smooth_sigma)
    else:
        ints_smooth = ints.copy()

    # Robust baseline estimation using percentiles
    if robust_baseline:
        baseline_est = np.percentile(ints_smooth, 5)
    else:
        baseline_est = ints_smooth.min()

    # Better amplitude estimation
    peak_idx = np.argmax(ints_smooth)
    peak_val = ints_smooth[peak_idx]
    amp_est = peak_val - baseline_est

    peak_region = np.abs(offs - offs[peak_idx]) < 50  # ±50ms around peak
    if peak_region.sum() > 0:
        weights = ints_smooth[peak_region] - baseline_est
        weights = np.maximum(weights, 0)
        if weights.sum() > 0:
            mu_est = np.average(offs[peak_region], weights=weights)
        else:
            mu_est = offs[peak_idx]
    else:
        mu_est = offs[peak_idx]

    # Better sigma estimation using FWHM
    half_max = baseline_est + 0.5 * amp_est
    above_half = ints_smooth >= half_max
    if above_half.sum() >= 2:
        indices = np.where(above_half)[0]
        fwhm_est = offs[indices[-1]] - offs[indices[0]]
        sigma_est = fwhm_est / 2.355  # Convert FWHM to sigma
    else:
        sigma_est = 60.0  # fallback

    if model == "gaussian":
        def _f(x, base, amp, mu, sigma):
            return base + amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        if p0 is None:
            p0 = [baseline_est, amp_est, mu_est, sigma_est]

        # Set reasonable bounds
        bounds = ([0, 0, -200, 10],  # lower bounds
                  [np.inf, np.inf, 200, 300])  # upper bounds

    elif model == "flattop":
        def _f(x, base, amp, lc, lk, rc, rk):
            left = 1.0 / (1.0 + np.exp(-(x - lc) / lk))
            right = 1.0 / (1.0 + np.exp((x - rc) / rk))
            return base + amp * left * right

        if p0 is None:
            p0 = [baseline_est, amp_est, -80.0, 10.0, 80.0, 10.0]

        bounds = ([0, 0, -300, 1, -100, 1],
                  [np.inf, np.inf, -10, 100, 300, 100])

    else:
        raise ValueError("model must be 'gaussian' or 'flattop'")

    # Prepare weights for fitting
    if weight_by_value:
        weights = (ints - baseline_est) / amp_est
        weights = np.maximum(weights, 0.1)  # minimum weight
        weights = np.sqrt(weights)  # square root to avoid over-weighting
    else:
        weights = None

    # Fit with bounds and weights
    try:
        popt, pcov = curve_fit(_f, offs, ints, p0=p0, bounds=bounds,
                               sigma=None if weights is None else 1 / weights,
                               absolute_sigma=True, maxfev=5000)
    except:
        popt, pcov = curve_fit(_f, offs, ints, p0=p0,
                               sigma=None if weights is None else 1 / weights)

    # Generate smooth curve for plotting
    xs = np.linspace(offs.min(), offs.max(), 600)
    ys = _f(xs, *popt)

    # Calculate FWHM for Gaussian
    fwhm = None
    if model == "gaussian":
        sigma = popt[3]
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

    # Calculate goodness of fit metrics
    residuals = ints - _f(offs, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((ints - np.mean(ints)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    peak_pred = _f(offs[peak_idx], *popt)
    peak_error = abs(peak_val - peak_pred) / peak_val

    return {
        "xs": xs,
        "ys": ys,
        "params": popt,
        "cov": pcov,
        "fwhm": fwhm,
        "r_squared": r_squared,
        "peak_error": peak_error,
        "peak_actual": peak_val,
        "peak_fitted": peak_pred
    }


def fit_tbw_double_gaussian(offs_ms, int_spikes, p0=None):
    """
    Fit with two half-Gaussians to allow asymmetry.
    """
    offs = np.asarray(offs_ms, dtype=float)
    ints = np.asarray(int_spikes, dtype=float)

    def _f(x, base, amp, mu, sigma_left, sigma_right):
        y = np.zeros_like(x)
        left_mask = x <= mu
        right_mask = x > mu

        # Left side
        y[left_mask] = base + amp * np.exp(-(x[left_mask] - mu) ** 2 / (2 * sigma_left ** 2))
        # Right side
        y[right_mask] = base + amp * np.exp(-(x[right_mask] - mu) ** 2 / (2 * sigma_right ** 2))

        return y

    # Initial guesses
    baseline = np.percentile(ints, 5)
    peak_idx = np.argmax(ints)
    peak_val = ints[peak_idx]
    amp = peak_val - baseline
    mu = offs[peak_idx]

    if p0 is None:
        p0 = [baseline, amp, mu, 50.0, 50.0]

    # Bounds
    bounds = ([0, 0, -100, 10, 10],
              [np.inf, np.inf, 100, 200, 200])

    # Weight by value
    weights = (ints - baseline) / amp
    weights = np.maximum(weights, 0.1)
    weights = np.sqrt(weights)

    popt, pcov = curve_fit(_f, offs, ints, p0=p0, bounds=bounds,
                           sigma=1 / weights, absolute_sigma=True)

    xs = np.linspace(offs.min(), offs.max(), 600)
    ys = _f(xs, *popt)

    # Calculate effective FWHM
    half_max = popt[0] + 0.5 * popt[1]
    above_half = ys >= half_max
    if above_half.sum() >= 2:
        indices = np.where(above_half)[0]
        fwhm = xs[indices[-1]] - xs[indices[0]]
    else:
        fwhm = None

    return {
        "xs": xs,
        "ys": ys,
        "params": popt,
        "cov": pcov,
        "fwhm": fwhm
    }


# Helper function to diagnose fitting issues
def diagnose_tbw_fit(offs_ms, int_spikes, fit_result):
    """
    Print diagnostics about the fit quality.
    """
    print("\n=== TBW Fit Diagnostics ===")
    print(f"R² = {fit_result.get('r_squared', 'N/A'):.4f}")
    print(f"Peak error = {fit_result.get('peak_error', 'N/A'):.2%}")
    print(f"Actual peak = {fit_result.get('peak_actual', 'N/A'):.1f}")
    print(f"Fitted peak = {fit_result.get('peak_fitted', 'N/A'):.1f}")

    params = fit_result['params']
    if len(params) >= 4:  # Gaussian
        print(f"\nFitted parameters:")
        print(f"  Baseline = {params[0]:.1f}")
        print(f"  Amplitude = {params[1]:.1f}")
        print(f"  Center = {params[2]:.1f} ms")
        print(f"  Sigma = {params[3]:.1f} ms")
        print(f"  FWHM = {fit_result.get('fwhm', 'N/A'):.1f} ms")


def tbw_gaussian_fit_curve(pooled):
    """Updated to use improved fitting"""
    offs = np.asarray(pooled["offsets_ms"])
    fit = fit_tbw_curve_improved(offs, pooled["mean_int_spikes"],
                                 model="gaussian",
                                 weight_by_value=True,
                                 robust_baseline=True)

    # Print diagnostics
    diagnose_tbw_fit(offs, pooled["mean_int_spikes"], fit)

    return fit["xs"], fit["ys"], fit["params"]


def plot_temporal_binding_summary(
        pooled_res,
        *,
        fit_model="gaussian",
        reference_fit=None,
        compare_methods=False,  # New parameter
        **fit_kw,
):
    """Updated with improved fitting and optional method comparison"""
    offs = np.asarray(pooled_res["offsets_ms"])
    mean = np.asarray(pooled_res["mean_int_spikes"])
    sem = np.asarray(pooled_res["sem_int_spikes"])

    font_path = './fonts/Roboto-Regular.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.size'] = 50
    plt.rcParams['xtick.labelsize'] = 50
    plt.rcParams['ytick.labelsize'] = 50
    plt.rcParams['axes.titlesize'] = 50
    plt.rcParams['axes.labelsize'] = 50
    plt.rcParams['legend.fontsize'] = 50

    fig, ax = plt.subplots(figsize=(55, 40))
    ax.errorbar(offs, mean, yerr=sem,
                fmt="o", capsize=0, markersize=5, label="mean ± SEM (n=10)")

    ax.set_ylim(bottom=0)
    y0 = ax.get_ylim()[0]

    # Current improved fit
    fwhm_txt = ""
    if fit_model:
        fit = fit_tbw_curve_improved(offs, mean, model=fit_model,
                                     weight_by_value=True,
                                     robust_baseline=True,
                                     **fit_kw)
        xs_fit = fit["xs"]
        ys_fit = fit["ys"]
        base_c, amp_c, mu_c, sigma_c = fit["params"][:4]
        fwhm_c = fit["fwhm"]
        peak_error = fit.get("peak_error", 0) * 100

        ax.plot(xs_fit, ys_fit, lw=2, color="C1", label="improved gaussian fit")

        if compare_methods:
            fit_old = fit_tbw_curve(offs, mean, model=fit_model)
            ax.plot(fit_old["xs"], fit_old["ys"], lw=1.5, ls=":",
                    color="gray", alpha=0.7, label="original fit")

        # Half-max visualization
        y_half_c = base_c + 0.5 * amp_c
        xL_c = mu_c - 0.5 * fwhm_c
        xR_c = mu_c + 0.5 * fwhm_c

        for x_h in (xL_c, xR_c):
            ax.plot([x_h, x_h], [y0, y_half_c], ls=":", lw=1, color="C1")

        m_c = (xs_fit >= xL_c) & (xs_fit <= xR_c)
        ax.fill_between(xs_fit[m_c], ys_fit[m_c], y0,
                        color="C1", alpha=0.15, zorder=0)

        fwhm_txt = f" | FWHM ≈ {fwhm_c:.0f} ms | Peak err = {peak_error:.1f}%"

    # Reference overlay
    if reference_fit is not None:
        xs_r, ys_r, p_r = reference_fit
        base_r, amp_r, mu_r, sigma_r = p_r[:4]
        fwhm_r = 2.355 * sigma_r

        # Scale to current amplitude
        ys_r_scaled = base_c + (ys_r - base_r) * (amp_c / amp_r)
        ax.plot(xs_r, ys_r_scaled, lw=2, ls="--", color="0.35",
                label="control Gaussian (scaled)")

        xL_r = mu_r - 0.5 * fwhm_r
        xR_r = mu_r + 0.5 * fwhm_r
        for x_h in (xL_r, xR_r):
            ax.plot([x_h, x_h], [y0, y_half_c], ls=":", lw=1, color="0.35")

        m_r = (xs_r >= xL_r) & (xs_r <= xR_r)
        ax.fill_between(xs_r[m_r], ys_r_scaled[m_r], y0,
                        color="#ffb3b3", alpha=0.20, zorder=0)

    # Cosmetics
    ax.axvline(0, ls="--", lw=.7, color="k")
    ax.set_xlabel("Audio – Visual onset (ms)")
    ax.set_ylabel("Integrated spikes (0–100 ms)")
    ax.set_title("Temporal binding window (10-model mean)" + fwhm_txt)

    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_position(("outward", 6))
    ax.spines["bottom"].set_position(("outward", 6))

    fig.tight_layout()
    plt.show()

    # Print diagnostics
    if fit_model:
        diagnose_tbw_fit(offs, mean, fit)


# Fast psychophysical TBW using existing infrastructure

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


def calculate_fusion_from_integration(integration_results, method='robust_sigmoid'):
    """
    Convert spike integration results to fusion probabilities.

    Parameters
    ----------
    integration_results : dict
        Must contain "offsets_ms" and "mean_int_spikes"
    method : str
        'linear': Simple normalization (original)
        'robust_sigmoid': Sigmoid with percentile-based scaling
        'preserve_peak': Direct mapping that preserves peak structure
    """
    offsets_ms = np.array(integration_results["offsets_ms"])
    int_spikes = integration_results["mean_int_spikes"]

    if method == 'linear':
        baseline = np.min(int_spikes)
        peak = int_spikes.max()
        fusion_probs = (int_spikes - baseline) / (peak - baseline + 1e-6)

    elif method == 'robust_sigmoid':
        # Use percentiles for more robust scaling
        p5 = np.percentile(int_spikes, 5)
        p95 = np.percentile(int_spikes, 95)

        # Normalize to roughly 0-1 range
        normalized = (int_spikes - p5) / (p95 - p5 + 1e-6)

        # Apply gentle sigmoid that preserves peaks
        k = 3.0  # Steepness parameter
        center = 0.5  # Sigmoid center
        fusion_probs = 1 / (1 + np.exp(-k * (normalized - center)))

    elif method == 'preserve_peak':
        # Use robust baseline estimation
        baseline = np.percentile(int_spikes, 10)

        range_val = np.percentile(int_spikes, 90) - baseline
        normalized = (int_spikes - baseline) / range_val

        fusion_probs = np.tanh(normalized * 0.8) * 1.1
        fusion_probs = np.clip(fusion_probs, 0, 1)

    else:
        raise ValueError(f"Unknown method: {method}")

    return fusion_probs


# ──────────────────────────────────────────────────────────────────
#  MEI-based fusion probability  (Stein & Stanford 2008)
# ──────────────────────────────────────────────────────────────────
@torch.inference_mode()
def _run_unimodal_reference(net, *, loc=90, T=60, D=5, extra=5, stim_in=1):
    """
    Run A-only and V-only reference trials at the given location.

    Returns
    -------
    a_spikes : float
        Integrated MSI spike count for audio-only stimulus.
    v_spikes : float
        Integrated MSI spike count for visual-only stimulus.
    """
    from Training import generate_av_batch_tensor

    n, S, sigma = net.n, net.space_size, net.sigma_in

    # Build A-only and V-only sequences (batch of 2)
    loc_seq = [999] * T
    mod_A   = ['X'] * T
    mod_V   = ['X'] * T
    for t in range(D):
        loc_seq[t] = loc
        mod_A[t] = 'A'
        mod_V[t] = 'V'

    loc_seqs = [list(loc_seq), list(loc_seq)]
    mod_seqs = [list(mod_A), list(mod_V)]

    xA, xV, mask = generate_av_batch_tensor(
        loc_seqs, mod_seqs, [False, False],
        n=n, space_size=S, sigma_in=sigma,
        noise_std=0.0, device=net.device, max_len=T,
        stimulus_intensity=stim_in,
    )

    net.reset_state(2)
    rast = torch.zeros((T, 2), device=net.device)
    for t in range(T):
        ret = net.update_all_layers_batch(xA[:, t], xV[:, t], mask[:, t], return_spike_sum=True)
        sum_sM = ret[-1]
        rast[t] = sum_sM.sum(dim=1)

    # Integrate over the same window as bimodal trials (onset → onset+D+extra)
    win_start = 0
    win_stop = min(D + extra, T)
    a_spikes = rast[win_start:win_stop, 0].sum().item()
    v_spikes = rast[win_start:win_stop, 1].sum().item()
    return a_spikes, v_spikes


def calculate_fusion_mei(av_int_spikes, a_ref, v_ref, threshold=0.1):
    """
    Multisensory Enhancement Index (MEI) fusion probability.

    Parameters
    ----------
    av_int_spikes : ndarray (n_offsets,)
        Integrated AV spike counts per offset.
    a_ref : float
        A-only integrated spike count.
    v_ref : float
        V-only integrated spike count.
    threshold : float
        MEI threshold for "fused" (default 0.1 = 10% enhancement).

    Returns
    -------
    fusion_probs : ndarray (n_offsets,)
        P(fusion) at each offset — here 0 or 1 per offset since we have
        one trial per model; averaging across models gives smooth curve.
    mei_values : ndarray (n_offsets,)
        Raw MEI values for diagnostics.
    """
    best_uni = max(a_ref, v_ref)
    if best_uni < 1e-6:
        # No unimodal response — can't compute MEI
        return np.zeros_like(av_int_spikes), np.zeros_like(av_int_spikes)

    mei = (av_int_spikes - best_uni) / best_uni
    fused = (mei > threshold).astype(float)
    return fused, mei


def fit_psychometric_curve(offsets_ms, fusion_probs, p0=None,
                           use_weights=True, smooth_data=False,
                           robust_fit=True):
    """
    Fit a Gaussian curve for TBW fusion data with improved peak capture.

    Parameters
    ----------
    offsets_ms : array-like
        Audio-visual onset asynchronies in milliseconds
    fusion_probs : array-like
        Fusion probabilities (0-1)
    p0 : tuple, optional
        Initial parameters [base, amp, mu, sigma]
    use_weights : bool
        Weight points by their value to emphasize peak
    smooth_data : bool
        Apply light smoothing before fitting
    robust_fit : bool
        Use robust fitting with outlier detection
    """
    offsets_ms = np.asarray(offsets_ms)
    fusion_probs = np.asarray(fusion_probs)

    # Optional smoothing
    if smooth_data:
        fusion_probs = gaussian_filter1d(fusion_probs, sigma=1.0)

    def gaussian(x, base, amp, mu, sigma):
        return base + amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    if p0 is None:
        # Robust baseline estimation
        baseline_percentile = 10 if robust_fit else 0
        base_guess = np.percentile(fusion_probs, baseline_percentile)

        # Find peak
        peak_idx = np.argmax(fusion_probs)
        peak_val = fusion_probs[peak_idx]
        mu_guess = offsets_ms[peak_idx]

        # Amplitude must reach the peak
        amp_guess = peak_val - base_guess

        # Estimate sigma from width at half-maximum
        half_max = base_guess + 0.5 * amp_guess
        above_half = fusion_probs >= half_max

        if above_half.sum() >= 2:
            indices = np.where(above_half)[0]
            fwhm_est = offsets_ms[indices[-1]] - offsets_ms[indices[0]]
            sigma_guess = fwhm_est / 2.355
        else:
            sigma_guess = 60.0

        p0 = [base_guess, amp_guess, mu_guess, sigma_guess]

    # CRITICAL: Set appropriate bounds
    max_amp = 1.5 * (fusion_probs.max() - np.percentile(fusion_probs, 10))

    bounds = ([0, 0, -300, 5],  # lower bounds
              [0.5, max_amp, 300, 200])  # upper bounds

    # Prepare weights
    if use_weights:
        base_est = np.percentile(fusion_probs, 10)
        weights = fusion_probs - base_est
        weights = np.maximum(weights, 0.05)  # Minimum weight
        weights = np.sqrt(weights)  # Square root for gentler weighting
        sigma_weights = 1 / weights
    else:
        sigma_weights = None

    # Robust fitting with multiple attempts
    fit_success = False
    methods = ['trf', 'dogbox', 'lm'] if robust_fit else ['lm']

    for method in methods:
        try:
            popt, pcov = curve_fit(
                gaussian, offsets_ms, fusion_probs,
                p0=p0, bounds=bounds,
                sigma=sigma_weights,
                absolute_sigma=True,
                maxfev=5000,
                method=method
            )
            fit_success = True
            break
        except Exception as e:
            if method == methods[-1]:  # Last method
                print(f"Warning: All fitting methods failed. Using initial guess.")
                popt = p0
                pcov = np.eye(4)

    # Generate smooth curve
    xs = np.linspace(offsets_ms.min(), offsets_ms.max(), 600)
    ys = gaussian(xs, *popt)

    base, amp, mu, sigma = popt

    criterion_level = base + 0.5 * amp

    # Find where curve crosses criterion
    above_criterion = ys >= criterion_level
    if above_criterion.any():
        indices = np.where(above_criterion)[0]
        x_left = xs[indices[0]]
        x_right = xs[indices[-1]]
        tbw = x_right - x_left
    else:
        x_left = x_right = mu
        tbw = 0

    # FWHM
    fwhm = 2.355 * sigma

    # Calculate goodness of fit
    residuals = fusion_probs - gaussian(offsets_ms, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((fusion_probs - np.mean(fusion_probs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Peak capture error
    peak_idx = np.argmax(fusion_probs)
    actual_peak = fusion_probs[peak_idx]
    fitted_peak = gaussian(offsets_ms[peak_idx], *popt)
    peak_error = abs(actual_peak - fitted_peak)

    return {
        "xs": xs,
        "ys": ys,
        "params": popt,
        "cov": pcov,
        "tbw": tbw,
        "fwhm": fwhm,
        "mu": mu,
        "sigma": sigma,
        "base": base,
        "amp": amp,
        "x_left": x_left,
        "x_right": x_right,
        "r_squared": r_squared,
        "peak_error": peak_error,
        "actual_peak": actual_peak,
        "fitted_peak": fitted_peak
    }


def is_temporally_fused(temporal_profile, sigma=2.0, valley_threshold=0.4,
                        min_peak_height=0.2, min_peak_separation=3,
                        min_total=10.0):
    """Temporal analog of spatial is_fused().

    Smooths the MSI population spike-count time series, finds peaks,
    and checks whether the two tallest peaks are separated by a deep
    valley (= two distinct events) or merged (= one fused response).

    The normalized profile is zero-padded before peak detection so that
    peaks at the first/last timestep (common at extreme SOAs where one
    stimulus response falls at the trial boundary) are properly detected
    by scipy.signal.find_peaks.

    Parameters
    ----------
    temporal_profile : 1D array
        MSI population spike counts per timestep (1 step = 10 ms).
    sigma : float
        Gaussian smoothing kernel width in timesteps.
    valley_threshold : float
        Valley/smaller-peak ratio above which peaks are considered merged.
    min_peak_height : float
        Minimum peak height as fraction of max for peak detection.
    min_peak_separation : int
        Minimum timestep separation between peaks.
    min_total : float
        Minimum total spike count (noise floor). Trials below this are
        classified as not-fused. Set low (default 10) to reject only
        near-silent trials; temporal classification handles the rest.

    Returns
    -------
    bool
        True if response is a single fused event, False if two separate events.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    sm = gaussian_filter1d(np.asarray(temporal_profile, dtype=float), sigma=sigma)
    if sm.max() < 1e-6:
        return False  # silent → not fused (no response)
    if sm.sum() < min_total:
        return False  # too weak → not fused (noise)
    sm_norm = sm / sm.max()

    # Zero-pad so find_peaks can detect peaks at trial boundaries.
    # At extreme SOAs one stimulus response may fall at the first or last
    # timestep; without padding these edge peaks are invisible to find_peaks,
    # causing single-modality responses to be misclassified as "fused."
    pad_w = min_peak_separation + 1
    sm_padded = np.pad(sm_norm, pad_w, mode='constant', constant_values=0)
    peaks, props = find_peaks(sm_padded, height=min_peak_height,
                              distance=min_peak_separation)
    peaks = peaks - pad_w                         # shift to original indices
    valid = (peaks >= 0) & (peaks < len(sm_norm))
    peaks = peaks[valid]
    heights = props['peak_heights'][valid]

    if len(peaks) <= 1:
        return True  # one or no peak → fused

    # Check valley between two tallest peaks
    top2_idx = np.argsort(heights)[-2:]
    p1, p2 = sorted(peaks[top2_idx])
    valley = sm_norm[p1:p2 + 1].min()
    smaller_peak = min(sm_norm[p1], sm_norm[p2])

    return valley / smaller_peak > valley_threshold  # high valley = merged = fused


def compute_tbw_temporal_fusion_persep(
    net,
    offsets,
    *,
    n_trials: int = 50,
    T: int = 60,
    D: int = 5,
    stim_in: float = 1.0,
    sigma: float = 2.0,
    valley_threshold: float = 0.4,
    min_peak_height: float = 0.2,
    min_peak_separation: int = 3,
    min_total: float = 10.0,
):
    """Compute per-trial temporal fusion probability at each offset.

    Batches ALL offsets × trials into a single GPU forward pass for
    massive speedup (51 offsets × 50 trials = 2550 batch elements).

    For each offset, runs n_trials with random spatial locations.
    Each trial's full temporal profile is classified as fused/not-fused
    using is_temporally_fused().

    Parameters
    ----------
    net : MultiBatchAudVisMSINetworkTime
        Loaded model (plasticity_enabled=False).
    offsets : list of int
        AV onset asynchronies in macro-steps (10 ms each).
    n_trials : int
        Number of trials per offset (random locations).
    T, D, stim_in : int/float
        Temporal integration parameters.
    sigma, valley_threshold, min_peak_height, min_peak_separation, min_total
        Parameters for is_temporally_fused().

    Returns
    -------
    p_fusion : np.ndarray (n_offsets,)
        P(fusion) at each offset.
    all_fused : list of np.ndarray
        Per-trial binary fusion labels. all_fused[k] has shape (n_trials,).
    """
    from Training import (
        generate_two_event_offset_seq,
        generate_av_batch_tensor,
    )

    initial_g_FFinh = net.g_FFinh
    initial_step_counter = net.step_counter
    rng = np.random.default_rng()

    n_offsets = len(offsets)
    total_batch = n_offsets * n_trials

    # ── Generate ALL stimuli for all offsets × trials at once ──
    # Layout: [off0_trial0, off0_trial1, ..., off0_trialN-1,
    #          off1_trial0, ..., offK_trialN-1]
    all_locs = rng.integers(0, net.space_size, size=(n_offsets, n_trials))

    loc_seqs = []
    mod_seqs = []
    for k, off in enumerate(offsets):
        for i in range(n_trials):
            ls, ms = generate_two_event_offset_seq(
                loc=int(all_locs[k, i]), T=T, D=D, offset=off,
                space_size=net.space_size,
            )
            loc_seqs.append(ls)
            mod_seqs.append(ms)

    off_flags = [False] * total_batch
    xA, xV, mask = generate_av_batch_tensor(
        loc_seqs, mod_seqs, off_flags,
        n=net.n, space_size=net.space_size, sigma_in=net.sigma_in,
        noise_std=0.0, device=net.device, max_len=T,
        stimulus_intensity=stim_in,
    )

    # ── Single forward pass for ALL 2550 batch elements ──
    net.g_FFinh = initial_g_FFinh
    net.step_counter = initial_step_counter
    net.reset_state(total_batch)
    rast = torch.zeros((T, total_batch), device=net.device)
    with torch.inference_mode():
        for t in range(T):
            ret = net.update_all_layers_batch(xA[:, t], xV[:, t], mask[:, t], return_spike_sum=True)
            sum_sM = ret[-1]
            rast[t] = sum_sM.sum(dim=1)

    # ── Reshape to (T, n_offsets, n_trials) and classify ──
    rast_np = rast.cpu().numpy().reshape(T, n_offsets, n_trials)
    del rast, xA, xV, mask  # free GPU memory early

    p_fusion = np.zeros(n_offsets)
    all_fused = [None] * n_offsets
    for k in range(n_offsets):
        fused_arr = np.zeros(n_trials)
        for i in range(n_trials):
            fused_arr[i] = float(is_temporally_fused(
                rast_np[:, k, i],
                sigma=sigma,
                valley_threshold=valley_threshold,
                min_peak_height=min_peak_height,
                min_peak_separation=min_peak_separation,
                min_total=min_total,
            ))
        p_fusion[k] = fused_arr.mean()
        all_fused[k] = fused_arr

    if net.device.type == "cuda":
        torch.cuda.empty_cache()

    # Restore state
    net.g_FFinh = initial_g_FFinh
    net.step_counter = initial_step_counter
    return p_fusion, all_fused


def compute_tbw_enhancement_persep(
    net,
    offsets,
    *,
    n_trials: int = 50,
    loc: int = 90,
    T: int = 60,
    D: int = 5,
    extra: int = 5,
    stim_in: float = 1.0,
):
    """Compute per-trial enhancement at each temporal offset.

    For each offset, runs n_trials trials (all at the same spatial location)
    with 3 passes each: AV bimodal, A-only, V-only.

    Parameters
    ----------
    net : MultiBatchAudVisMSINetworkTime
        Loaded model (plasticity_enabled=False).
    offsets : list of int
        AV onset asynchronies in macro-steps (10 ms each).
    n_trials : int
        Number of independent trials per offset.
    loc : int
        Spatial location in degrees.
    T, D, extra, stim_in : int/float
        Temporal integration parameters.

    Returns
    -------
    mean_enhancement : np.ndarray (n_offsets,)
        Mean enhancement across trials at each offset.
    all_trial_enh : list of np.ndarray
        Per-trial enhancement values. all_trial_enh[k] has shape (n_trials,).
    """
    from Training import (
        generate_two_event_offset_seq,
        generate_av_batch_tensor,
    )

    initial_g_FFinh = net.g_FFinh
    initial_step_counter = net.step_counter
    rng = np.random.default_rng()

    n_offsets = len(offsets)
    mean_enhancement = np.zeros(n_offsets)
    all_trial_enh = [None] * n_offsets

    for k, off in enumerate(offsets):
        bs = n_trials
        # Random spatial locations per trial (same as SBW) for trial-to-trial variability
        locs = rng.integers(0, net.space_size, size=bs)

        # Build per-trial stimulus sequences for AV, A-only, V-only
        loc_seqs_av, mod_seqs_av = [], []
        loc_seqs_a, mod_seqs_a = [], []
        loc_seqs_v, mod_seqs_v = [], []

        aud_on = 0 if off >= 0 else abs(off)
        vis_on = 0 if off <= 0 else off

        for trial_loc in locs:
            trial_loc = int(trial_loc)

            # AV bimodal
            loc_seq, mod_seq = generate_two_event_offset_seq(
                loc=trial_loc, T=T, D=D, offset=off, space_size=net.space_size
            )
            loc_seqs_av.append(loc_seq)
            mod_seqs_av.append(mod_seq)

            # A-only
            ls_a = [999] * T
            ms_a = ['X'] * T
            for t in range(aud_on, aud_on + D):
                ls_a[t] = trial_loc
                ms_a[t] = 'A'
            loc_seqs_a.append(ls_a)
            mod_seqs_a.append(ms_a)

            # V-only
            ls_v = [999] * T
            ms_v = ['X'] * T
            for t in range(vis_on, vis_on + D):
                ls_v[t] = trial_loc
                ms_v[t] = 'V'
            loc_seqs_v.append(ls_v)
            mod_seqs_v.append(ms_v)

        # Concatenate all 3 conditions into a single batch of 3*bs
        all_loc_seqs = loc_seqs_av + loc_seqs_a + loc_seqs_v
        all_mod_seqs = mod_seqs_av + mod_seqs_a + mod_seqs_v
        off_flags = [False] * (3 * bs)

        xA, xV, mask = generate_av_batch_tensor(
            all_loc_seqs, all_mod_seqs, off_flags,
            n=net.n, space_size=net.space_size, sigma_in=net.sigma_in,
            noise_std=0.0, device=net.device, max_len=T,
            stimulus_intensity=stim_in,
        )

        # Run network with fresh state
        net.g_FFinh = initial_g_FFinh
        net.step_counter = initial_step_counter
        net.reset_state(3 * bs)
        rast = torch.zeros((T, 3 * bs), device=net.device)
        with torch.inference_mode():
            for t in range(T):
                ret = net.update_all_layers_batch(xA[:, t], xV[:, t], mask[:, t], return_spike_sum=True)
                sum_sM = ret[-1]
                rast[t] = sum_sM.sum(dim=1)

        # Integration window
        later_onset = abs(off)
        win_start = later_onset
        win_stop = min(win_start + D + extra, T)
        int_spikes = rast[win_start:win_stop].sum(dim=0).cpu().numpy()  # (3*bs,)

        av_spikes = int_spikes[:bs]
        a_spikes = int_spikes[bs:2*bs]
        v_spikes = int_spikes[2*bs:]

        enh = av_spikes - np.maximum(a_spikes, v_spikes)
        mean_enhancement[k] = enh.mean()
        all_trial_enh[k] = enh

        if net.device.type == "cuda":
            torch.cuda.empty_cache()

    # Restore state
    net.g_FFinh = initial_g_FFinh
    net.step_counter = initial_step_counter
    return mean_enhancement, all_trial_enh


def _normalize_spike_profile(spikes, n_extreme=5):
    """Baseline-subtract and peak-normalize a spike count profile.

    Parameters
    ----------
    spikes : ndarray (n_points,)
        Raw integrated spike counts across offsets or separations.
    n_extreme : int
        Number of points at each tail used to estimate the baseline
        (mean of the *n_extreme* lowest points at each end).

    Returns
    -------
    ndarray (n_points,)
        Normalized profile in [0, 1]: 0 at extremes, 1 at peak.
    """
    spikes = np.asarray(spikes, dtype=float)
    # Baseline = max of left and right tail means (handles asymmetric A/V drive)
    baseline = max(spikes[:n_extreme].mean(), spikes[-n_extreme:].mean())
    shifted = spikes - baseline
    peak = shifted.max()
    if peak < 1e-9:
        return np.zeros_like(spikes)
    return np.clip(shifted / peak, 0.0, 1.0)


def run_fusion_across_models(model_paths, offsets, device="cuda",
                             modify_net=None, fusion_method='spike_profile',
                             mei_threshold=0.1,
                             enhancement_threshold=10.0,
                             n_trials=50,
                             loc=90, T=60, D=5, extra=5, stim_in=1.0):
    """
    Run AV temporal integration across model checkpoints and compute
    fusion probability at each offset.

    Parameters
    ----------
    fusion_method : str
        'spike_profile'  : Normalized AV spike count profile (default).
                           Baseline = mean of extreme offsets, peak-normalized
                           to [0,1].  No unimodal passes needed.
        'enhancement'    : P(fusion) via enhancement thresholding.
                           Per-trial: enh = AV - max(A,V) in integration window.
                           P(fusion) = fraction of trials where enh > threshold.
        'mei'            : Multisensory Enhancement Index (Stein & Stanford 2008).
                           Compares AV spikes to best unimodal baseline.
        'preserve_peak'  : Legacy percentile-based normalization.
    enhancement_threshold : float
        Spike-count threshold for 'enhancement' method (default 10.0).
    n_trials : int
        Number of trials per offset for 'enhancement' method (default 50).
    mei_threshold : float
        Enhancement fraction above best-unimodal to count as fused (default 0.1).
    loc, T, D, extra, stim_in : passed to run_temporal_integration / unimodal ref.
    """
    from Training import run_temporal_integration

    # Enhancement method uses a completely different path (multi-trial per offset)
    if fusion_method == 'enhancement':
        all_pfusion = []  # (n_models, n_offsets) — P(fusion) per model
        n_models = len(model_paths)

        for path in model_paths:
            net = load_msi_model(Path(path), device=device)
            if callable(modify_net):
                modify_net(net)

            _mean_enh, all_trial_enh = compute_tbw_enhancement_persep(
                net, offsets, n_trials=n_trials, loc=loc,
                T=T, D=D, extra=extra, stim_in=stim_in,
            )
            # P(fusion) per offset for this model
            p_fusion = np.array([
                (trial_enh > enhancement_threshold).mean()
                for trial_enh in all_trial_enh
            ])
            all_pfusion.append(p_fusion)

            del net
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

        all_pfusion = np.vstack(all_pfusion)  # (n_models, n_offsets)
        result = {
            "offsets_ms": [o * 10 for o in offsets],
            "mean_fusion": all_pfusion.mean(0),
            "sem_fusion": all_pfusion.std(0, ddof=1) / np.sqrt(n_models),
            "all_fusion": all_pfusion,
            "mean_int_spikes": all_pfusion.mean(0),  # placeholder
            "sem_int_spikes": all_pfusion.std(0, ddof=1) / np.sqrt(n_models),
            "enhancement_threshold": float(enhancement_threshold),
        }
        return result

    # Temporal fusion method: multi-trial per offset, is_temporally_fused classifier
    if fusion_method == 'temporal_fusion':
        import time as _time
        all_pfusion = []  # (n_models, n_offsets)
        n_models = len(model_paths)

        for mi, path in enumerate(model_paths):
            _t0 = _time.time()
            net = load_msi_model(Path(path), device=device)
            if callable(modify_net):
                modify_net(net)

            p_fusion, _all_fused = compute_tbw_temporal_fusion_persep(
                net, offsets, n_trials=n_trials, T=T, D=D, stim_in=stim_in,
                sigma=2.0, valley_threshold=0.4, min_peak_height=0.2,
                min_peak_separation=3, min_total=10.0,
            )
            all_pfusion.append(p_fusion)
            _dt = _time.time() - _t0
            idx0 = offsets.index(0) if 0 in offsets else len(offsets) // 2
            print(f"  Model {mi}/{n_models}: P(fus@0ms)={p_fusion[idx0]:.3f}, "
                  f"peak={p_fusion.max():.3f}, {_dt:.1f}s")

            del net
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

        all_pfusion = np.vstack(all_pfusion)  # (n_models, n_offsets)
        result = {
            "offsets_ms": [o * 10 for o in offsets],
            "mean_fusion": all_pfusion.mean(0),
            "sem_fusion": all_pfusion.std(0, ddof=1) / np.sqrt(n_models),
            "all_fusion": all_pfusion,
            "mean_int_spikes": all_pfusion.mean(0),  # alias for compatibility
            "sem_int_spikes": all_pfusion.std(0, ddof=1) / np.sqrt(n_models),
            "enhancement_threshold": None,
        }
        return result

    # Standard path: one trial per offset per model (spike_profile, mei, etc.)
    all_fusion = []
    all_int = []
    all_mei = []

    for path in model_paths:
        net = load_msi_model(Path(path), device=device)
        if callable(modify_net):
            modify_net(net)

        initial_g_FFinh = net.g_FFinh

        # 1. AV bimodal integration
        res = run_temporal_integration(
            net, offsets, loc=loc, T=T, D=D, extra=extra, stim_in=stim_in
        )
        av_spikes = res["int_spikes"]  # (n_offsets,)
        all_int.append(av_spikes)

        if fusion_method == 'spike_profile':
            pass
        elif fusion_method == 'mei':
            net.g_FFinh = initial_g_FFinh
            a_ref, v_ref = _run_unimodal_reference(
                net, loc=loc, T=T, D=D, extra=extra, stim_in=stim_in
            )
            fused, mei = calculate_fusion_mei(
                av_spikes, a_ref, v_ref, threshold=mei_threshold
            )
            all_fusion.append(fused)
            all_mei.append(mei)
        else:
            single_res = {
                "offsets_ms": res["offsets_ms"],
                "mean_int_spikes": av_spikes
            }
            fusion = calculate_fusion_from_integration(
                single_res, method=fusion_method
            )
            all_fusion.append(fusion)

        net.g_FFinh = initial_g_FFinh
        del net
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    all_int = np.vstack(all_int)
    n_models = len(model_paths)

    if fusion_method == 'spike_profile':
        mean_raw = all_int.mean(0)
        mean_fusion = _normalize_spike_profile(mean_raw)

        baseline = max(mean_raw[:5].mean(), mean_raw[-5:].mean())
        peak_shift = (mean_raw - baseline).max()
        if peak_shift > 1e-9:
            all_norm = np.clip((all_int - baseline) / peak_shift, 0.0, 1.0)
        else:
            all_norm = np.zeros_like(all_int)
        sem_fusion = all_norm.std(0, ddof=1) / np.sqrt(n_models)

        result = {
            "offsets_ms": [o * 10 for o in offsets],
            "mean_fusion": mean_fusion,
            "sem_fusion": sem_fusion,
            "all_fusion": all_norm,
            "mean_int_spikes": all_int.mean(0),
            "sem_int_spikes": all_int.std(0, ddof=1) / np.sqrt(n_models),
        }
    else:
        all_fusion = np.vstack(all_fusion)
        result = {
            "offsets_ms": [o * 10 for o in offsets],
            "mean_fusion": all_fusion.mean(0),
            "sem_fusion": all_fusion.std(0, ddof=1) / np.sqrt(n_models),
            "all_fusion": all_fusion,
            "mean_int_spikes": all_int.mean(0),
            "sem_int_spikes": all_int.std(0, ddof=1) / np.sqrt(n_models),
        }
        if all_mei:
            result["mean_mei"] = np.vstack(all_mei).mean(0)
    return result


def print_fit_diagnostics(fit, data_fusion=None):
    """
    Print comprehensive diagnostics about the fit quality.
    """
    print("\n" + "=" * 60)
    print("TEMPORAL BINDING WINDOW FIT DIAGNOSTICS")
    print("=" * 60)

    print(f"\nFit Quality Metrics:")
    print(f"  R² = {fit.get('r_squared', 0):.4f}")
    print(f"  Peak error = {fit.get('peak_error', 0):.4f} ({fit.get('peak_error', 0) * 100:.1f}%)")

    if 'actual_peak' in fit and 'fitted_peak' in fit:
        print(f"  Actual data peak = {fit['actual_peak']:.3f}")
        print(f"  Fitted curve peak = {fit['fitted_peak']:.3f}")
        print(f"  Peak difference = {fit['fitted_peak'] - fit['actual_peak']:+.3f}")

    print(f"\nFitted Parameters:")
    print(f"  Baseline = {fit['base']:.3f}")
    print(f"  Amplitude = {fit['amp']:.3f}")
    print(f"  Peak location (μ) = {fit['mu']:.1f} ms")
    print(f"  Width (σ) = {fit['sigma']:.1f} ms")

    print(f"\nTemporal Window Metrics:")
    print(f"  FWHM = {fit['fwhm']:.1f} ms")
    print(f"  TBW at 50% = {fit['tbw']:.1f} ms")
    print(f"  TBW boundaries = [{fit['x_left']:.1f}, {fit['x_right']:.1f}] ms")

    # Data range check
    if data_fusion is not None:
        print(f"\nData Range Check:")
        print(f"  Data range = [{data_fusion.min():.3f}, {data_fusion.max():.3f}]")
        print(f"  Fitted range = [{fit['base']:.3f}, {fit['base'] + fit['amp']:.3f}]")

        if fit['base'] + fit['amp'] < data_fusion.max() * 0.95:
            print("  ⚠️  WARNING: Fitted peak is >5% below data peak!")
            print("     Consider adjusting bounds or fusion calculation method")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------
def _find_crossings(xs, ys, y0):
    """Return x-coordinates where the polyline ``(xs, ys)`` crosses ``y = y0``.

    Linearly interpolates between successive samples to locate every
    sign-change of ``ys - y0``.  Used to extract the 50%-fusion left/right
    crossings that define the TBW (right_x - left_x) on the fitted
    psychometric curve.

    Inputs
    ------
    xs, ys : 1D array-likes of equal length, monotonic in ``xs``.
    y0 : float — the criterion (typically 0.5).

    Returns
    -------
    list of float — interpolated crossing x-values, in xs-order.
    Empty list if no crossing exists.

    Notes
    -----
    Pure numpy, deterministic.  Adjacent equal samples are skipped to
    avoid divide-by-zero.
    """
    xs, ys = np.asarray(xs), np.asarray(ys)
    out = []
    for i in range(len(xs) - 1):
        y1, y2 = ys[i], ys[i + 1]
        if (y1 - y0) * (y2 - y0) <= 0 and y1 != y2:          # sign‑change
            x = xs[i] + (y0 - y1) * (xs[i + 1] - xs[i]) / (y2 - y1)
            out.append(x)
    return out
# ---------------------------------------------------------------------
def plot_psychometric_tbw(
        pooled_res,
        reference_fit=None,              # dict or (xs,ys,params) or None
        title_suffix="",
        criterion=0.5,                   # ← the horizontal 50 % line
        fusion_method='preserve_peak',
        show_diagnostics=True
):
    """
    Plot the manipulated TBW (orange) and – optionally – the *unaltered*
    control curve (grey dashed).  The control outline is now drawn **exactly
    as it was in the single‑condition plot** – no amplitude rescaling, no
    baseline shifting – so the two figures are directly comparable.
    """
    # ───────── 0)  pull fusion‑probability data ─────────────────────────
    offs = np.asarray(pooled_res["offsets_ms"])
    if "mean_fusion" in pooled_res:                      # ready‑made
        mean_fus, sem_fus = (np.asarray(pooled_res["mean_fusion"]),
                             np.asarray(pooled_res["sem_fusion"]))
    else:                                                # derive from spikes
        mean_fus = calculate_fusion_from_integration(
            pooled_res, method=fusion_method)
        if "all_int_spikes" in pooled_res:
            all_f = np.vstack([
                calculate_fusion_from_integration(
                    {"offsets_ms": offs, "mean_int_spikes": arr},
                    method=fusion_method)
                for arr in pooled_res["all_int_spikes"]])
            sem_fus = all_f.std(0, ddof=1) / np.sqrt(all_f.shape[0])
        else:
            sem_fus = np.zeros_like(mean_fus)

    fit = fit_psychometric_curve_improved(offs, mean_fus)
    xs_M, ys_M = fit["xs"], fit["ys"]
    xL_M, xR_M = _find_crossings(xs_M, ys_M, criterion)[:2]   # first & last

    have_ctrl = reference_fit is not None
    if have_ctrl:
        if isinstance(reference_fit, dict):
            xs_C = np.asarray(reference_fit["xs"])
            ys_C = np.asarray(reference_fit["ys"])        # UNTOUCHED!
            p_C  = reference_fit["params"]
        else:                                             # tuple
            xs_C, ys_C, p_C = reference_fit
        crosses_C = _find_crossings(xs_C, ys_C, criterion)
        xL_C, xR_C = (crosses_C[0], crosses_C[-1]) if len(crosses_C) >= 2 else (None, None)

    font_path = './fonts/Roboto-Regular.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.size'] = 50
    plt.rcParams['xtick.labelsize'] = 50
    plt.rcParams['ytick.labelsize'] = 50
    plt.rcParams['axes.titlesize'] = 50
    plt.rcParams['axes.labelsize'] = 50
    plt.rcParams['legend.fontsize'] = 50

    # ───────── 3)  draw everything ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(40, 26))

    # data points
    ax.errorbar(offs, mean_fus, yerr=sem_fus,
                fmt='o', ms=6, capsize=3, color='C0', label='Data', zorder=3)

    # orange fit
    ax.plot(xs_M, ys_M, lw=2.5, color='C1',
            label=f'Gaussian fit (TBW={fit["tbw"]:.0f} ms)', zorder=2)

    # 50 % criterion
    ax.axhline(criterion, ls=':', lw=1.2, color='0.35', label='50 % criterion')

    # orange shading + verticals
    ax.fill_between(xs_M, 0, ys_M, where=(xs_M >= xL_M) & (xs_M <= xR_M),
                    color='C1', alpha=.15, zorder=0)
    for x in (xL_M, xR_M):
        ax.axvline(x, ls='--', lw=1, color='C1', alpha=.35)

    if have_ctrl:
        ax.plot(xs_C, ys_C, ls='--', lw=2, color='0.35',
                label=f'Control (TBW={2.355 * p_C[3]:.0f} ms)', zorder=2)
        if xL_C is not None and xR_C is not None:
            ax.fill_between(xs_C, 0, ys_C,
                            where=(xs_C >= xL_C) & (xs_C <= xR_C),
                            color='#ffb3b3', alpha=.22, zorder=0)
            for x in (xL_C, xR_C):
                ax.plot([x, x], [0, 1], ls=':', lw=1.2, color='0.45', zorder=1)

    # cosmetics
    ax.axvline(0, ls='--', lw=.8, color='k', alpha=.3)
    ax.set(xlabel='Audio – Visual onset (ms)',
           ylabel='Fusion probability',
           ylim=(-.05, 1.05),
           xlim=(offs.min() - 20, offs.max() + 20),
           title=f'Temporal Binding Window{title_suffix}')
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    if show_diagnostics:
        print_fit_diagnostics(fit, mean_fus)

    return fit



# Ultra-fast main function
def main_psychometric():
    """
    Fast psychometric analysis reusing existing infrastructure.
    """
    base_dir = Path("checkpoint")
    model_paths = [base_dir / f"msi_model_surr_2_{i:02d}.pt" for i in range(10)]
    offsets = list(range(-50, 51, 2))  # -500 to +500ms

    print("CONTROL condition...")
    t0 = time.time()

    # Reuse the existing fast batch infrastructure
    pooled_ctrl = run_fusion_across_models(model_paths, offsets, device="cuda")

    print(f"Done in {time.time() - t0:.1f}s")
    ctrl_fit = plot_psychometric_tbw(pooled_ctrl)

    print("\nMANIPULATED condition...")
    t0 = time.time()

    def manip_fn(net):
        # net.pv_nmda = 0.2
        # net.targ_ratio = 1
        net.gNMDA = 0.015
        # net.u_a.fill_(0.05)
        # net.u_v.fill_(0.05)
        # net.tau_rec = 50.0
        # # MSI excit
        # # MSI inh (fast spiking)
        # # Out
        # net.g_GABA = 30
        # net.pv_scale = 0.4

    pooled_manip = run_fusion_across_models(
        model_paths, offsets, device="cuda", modify_net=manip_fn
    )

    print(f"Done in {time.time() - t0:.1f}s")
    manip_fit = plot_psychometric_tbw(pooled_manip, reference_fit=ctrl_fit,
                                      title_suffix=" - Comparison")

    print(f"\nΔTBW = {manip_fit['tbw'] - ctrl_fit['tbw']:+.0f}ms")


# ────────────────────────────────────────────────────────────────
#  ❶  MSI TIME‑COURSE  (single AV asynchrony)
# ────────────────────────────────────────────────────────────────
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np, torch, matplotlib.pyplot as plt


@torch.inference_mode()
def msi_timecourse_at_offset(net,
                             offset_steps: int,
                             *,
                             loc_deg: float = 90.0,
                             T: int = 60,
                             D: int = 5,
                             intensity: float = 1.0):
    """
    Run one flash‑sound sequence with AUDIO at t=0 (or VISUAL if offset<0)
    and the other modality lagging/leading by *offset_steps* external frames
    (10 ms each).  Returns a NumPy vector of summed MSI spikes per frame.
    """
    n = net.n
    dev = net.device
    xs = torch.arange(n, dtype=torch.float32, device=dev)
    centre_idx = int(round(loc_deg * (n - 1) / (net.space_size - 1)))
    gauss = torch.exp(-0.5 * ((xs - centre_idx) / net.sigma_in) ** 2) * intensity

    # Build external‑input tensors -----------------------------------------
    xA = torch.zeros(T, n, device=dev)
    xV = torch.zeros_like(xA)

    aud_on = 0 if offset_steps >= 0 else abs(offset_steps)
    vis_on = 0 if offset_steps <= 0 else offset_steps

    xA[aud_on:aud_on + D] = gauss
    xV[vis_on:vis_on + D] = gauss

    # Run the network ------------------------------------------------------
    net.reset_state(batch_size=1)
    spike_sum = torch.zeros(T, device=dev)

    for t in range(T):
        ret = net.update_all_layers_batch(xA[t][None, :], xV[t][None, :], return_spike_sum=True)
        sum_sM = ret[-1]
        spike_sum[t] = sum_sM[0].sum()

    return spike_sum.cpu().numpy()


# ────────────────────────────────────────────────────────────────
#  ❷  Small inset helper (time‑axis plot)
# ────────────────────────────────────────────────────────────────
def add_timecourse_inset(ax_parent,
                         timecourse: np.ndarray,
                         *,
                         title: str,
                         loc: int = 1,
                         width: str = "32%",
                         height: str = "35%",
                         color: str = "C0"):
    """
    Draw *timecourse* (vector) as an inset inside *ax_parent* (TBW panel).

    loc : 1=UR, 2=UL, 3=LL, 4=LR.
    """
    ax_in = inset_axes(ax_parent, width=width, height=height,
                       loc=loc, borderpad=1.2)

    # NORMALIZE THE TIMECOURSE
    if timecourse.max() > 0:
        timecourse_norm = timecourse / timecourse.max()
    else:
        timecourse_norm = timecourse

    ax_in.plot(range(len(timecourse)), timecourse_norm, color=color, lw=1.6)  # Use normalized data
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_ylim(0, 1.05)  # Set ylim to normalized range
    ax_in.set_title(title, fontsize=7, pad=1.2)

    for spine in ax_in.spines.values():
        spine.set_linewidth(.6)
        spine.set_color("0.4")
    return ax_in


def _asymmetric_pedestal(x, base, top, center, hw_left, hw_right, k):
    """Asymmetric pedestal: product of two sigmoids with independent half-widths.

    Parameters
    ----------
    x : array
        SOA values in ms.
    base : float
        Floor P(fusion) at extreme offsets.
    top : float
        Ceiling P(fusion) near the center.
    center : float
        Center offset (ms) — allows peak to shift from 0.
    hw_left : float
        Half-width on the audio-leading (negative) side.
    hw_right : float
        Half-width on the visual-leading (positive) side.
    k : float
        Edge steepness (ms). Smaller = sharper transition.
    """
    left = 1.0 / (1.0 + np.exp(-(x - center + hw_left) / k))
    right = 1.0 / (1.0 + np.exp((x - center - hw_right) / k))
    return base + (top - base) * left * right


def fit_psychometric_curve_improved(offsets_ms, fusion_probs, p0=None, **kwargs):
    """Fit an asymmetric pedestal (difference-of-sigmoids) to P(fusion) vs SOA.

    Uses independent left/right half-widths to capture the audio-visual
    asymmetry that a symmetric Gaussian cannot represent.

    Returns a dict compatible with plot_psychometric_tbw_ax and _find_crossings.
    """
    from scipy.optimize import curve_fit

    x = np.asarray(offsets_ms, dtype=float)
    y = np.asarray(fusion_probs, dtype=float)

    # Initial guesses
    base0 = float(np.mean(np.concatenate([y[:3], y[-3:]])))
    top0 = float(y.max())
    center0 = float(x[np.argmax(y)])
    hw0 = float((x.max() - x.min()) / 6)
    k0 = 30.0  # moderate steepness for ms-scale x-axis

    if p0 is None:
        p0 = [base0, top0, center0, hw0, hw0, k0]

    bounds = (
        [-0.05, 0.3,  x.min(), 10,  10,  5],   # lower
        [ 0.5,  1.05, x.max(), 400, 400, 150],  # upper
    )

    try:
        popt, pcov = curve_fit(_asymmetric_pedestal, x, y, p0=p0,
                               bounds=bounds, maxfev=10000)
    except RuntimeError:
        # Fallback: fix k, reduce free params
        popt = np.array(p0)
        pcov = np.zeros((6, 6))

    base, top, center, hw_left, hw_right, k = popt

    xs = np.linspace(x.min(), x.max(), 600)
    ys = _asymmetric_pedestal(xs, *popt)

    # Find 50% crossings
    criterion = 0.5
    crossings = _find_crossings(xs, ys, criterion)
    if len(crossings) >= 2:
        x_left = crossings[0]
        x_right = crossings[-1]
        tbw = x_right - x_left
    else:
        x_left = center - hw_left
        x_right = center + hw_right
        tbw = hw_left + hw_right

    # R-squared
    y_pred = _asymmetric_pedestal(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "xs": xs,
        "ys": ys,
        "params": popt,
        "cov": pcov,
        "tbw": tbw,
        "fwhm": hw_left + hw_right,
        "mu": center,
        "sigma": (hw_left + hw_right) / 2,  # compat
        "base": base,
        "amp": top - base,
        "x_left": x_left,
        "x_right": x_right,
        "r_squared": r_squared,
        "hw_left": hw_left,
        "hw_right": hw_right,
        "k": k,
        "peak_error": 0,
        "actual_peak": float(y.max()),
        "fitted_peak": float(top),
    }
# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
def plot_psychometric_tbw_ax(pooled_res,
                             *,
                             reference_fit=None,
                             title_suffix="",
                             criterion=0.5,
                             cont=False,
                             out_path=None):
    """
    Identical visuals to plot_psychometric_tbw() but returns
    handles so callers can add extra graphics before plt.show().
    """
    offs = np.asarray(pooled_res["offsets_ms"])
    mean_fus = pooled_res["mean_fusion"]
    sem_fus = pooled_res["sem_fusion"]

    # Fit manipulated
    # Fit manipulated
    fit_M = fit_psychometric_curve_improved(offs, mean_fus)
    xs_M, ys_M = fit_M["xs"], fit_M["ys"]

    crossings_M = _find_crossings(xs_M, ys_M, criterion)
    if len(crossings_M) >= 2:
        xL_M, xR_M = crossings_M[0], crossings_M[-1]
    else:
        # Fallback if no crossings found
        xL_M = xR_M = fit_M["mu"]

    print(f"Manipulated crossing points: [{xL_M:.1f}, {xR_M:.1f}] ms")

    have_ctrl = isinstance(reference_fit, dict)
    if have_ctrl:
        fit_C = reference_fit
        xs_C, ys_C = fit_C["xs"], fit_C["ys"]
        crossings_C = _find_crossings(xs_C, ys_C, criterion)
        if len(crossings_C) >= 2:
            xL_C, xR_C = crossings_C[0], crossings_C[-1]
        else:
            xL_C = xR_C = fit_C["mu"]

        print(f"Control crossing points: [{xL_C:.1f}, {xR_C:.1f}] ms")

    font_path = './fonts/Roboto-Regular.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.size'] = 60
    plt.rcParams['xtick.labelsize'] = 60
    plt.rcParams['ytick.labelsize'] = 60
    plt.rcParams['axes.titlesize'] = 50
    plt.rcParams['axes.labelsize'] = 60
    plt.rcParams['legend.fontsize'] = 60

    # ::::: canvas :::::::::::::::::::::::::::::::::::::::::::::::
    fig, ax = plt.subplots(figsize=(25, 14))

    ax.errorbar(offs, mean_fus, yerr=sem_fus,
                fmt='o', ms=5, capsize=3, color='C0',
                label='mean ± SEM (n=10)', zorder=3, elinewidth=0.5)

    # manipulated curve + guides
    ax.plot(xs_M, ys_M, lw=2.5, color='C1',
            label='Pedestal fit (manipulated)', zorder=2)
    mask_M = (xs_M >= xL_M) & (xs_M <= xR_M)
    ax.fill_between(xs_M, 0, criterion, where=mask_M,
                    color='C1', alpha=.18, zorder=1)
    ax.vlines([xL_M, xR_M], 0, criterion, ls=':', lw=1, color='C1')

    # control overlay (grey dashed + guides)
    if have_ctrl:
        ax.plot(xs_C, ys_C, ls='--', lw=2, color='.45',
                label='Control fit', zorder=1.5)
        mask_C = (xs_C >= xL_C) & (xs_C <= xR_C)
        ax.fill_between(xs_C, 0, criterion, where=mask_C,
                        color='.45', alpha=.12, zorder=0.8)
        ax.vlines([xL_C, xR_C], 0, criterion, ls=':', lw=1, color='.45')

    # cosmetics
    ax.axhline(criterion, ls=':', lw=1, color='.4')
    ax.axvline(0, ls='--', lw=.8, color='k', alpha=.3)
    ax.set(xlabel='Audio – Visual onset (ms)',
           ylabel='Fusion probability',
           ylim=(-.05, 1.05),
           xlim=(offs.min() - 20, offs.max() + 20),
           title=f'Temporal Binding Window{title_suffix}')
    # ax.legend(frameon=False, fontsize=60, loc='upper right')
    # ax.legend(None)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    ax.tick_params(axis='both', which='major', length=20, width=1)
    plt.rcParams['pdf.fonttype'] = 42      # TrueType in PDF/PS
    plt.rcParams['ps.fonttype']  = 42
    plt.rcParams['svg.fonttype'] = 'none'
    save_path = out_path or './Saved_Images/TBW_curve.svg'
    plt.savefig(save_path, format='svg')
    return fig, ax, fit_M

# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
def average_msi_timecourse(model_paths,
                           offset_steps: int,
                           *,
                           device: str = "cuda",
                           **kwargs):
    """
    For every model in *model_paths* (control parameters),
    run `msi_timecourse_at_offset()` with the provided kwargs and
    return the **mean** time‑course (NumPy vector).
    """
    tcs = []
    for p in model_paths:
        net = load_msi_model(p, device=device)
        tc = msi_timecourse_at_offset(net, offset_steps, **kwargs)
        tcs.append(tc)
    return np.mean(tcs, axis=0)


# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
def plot_timecourse_figure(timecourse: np.ndarray,
                           *,
                           title: str,
                           color: str = "C0",
                           dt_ms: int = 10,
                           figsize=(25, 14)):
    """
    Display *timecourse* (1‑D NumPy) as a separate matplotlib figure.

    dt_ms   : temporal resolution per frame (default 10 ms).
    Returns : (fig, ax)
    """

    font_path = './fonts/Roboto-Regular.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.size'] = 80
    plt.rcParams['xtick.labelsize'] = 80
    plt.rcParams['ytick.labelsize'] = 80
    plt.rcParams['axes.titlesize'] = 80
    plt.rcParams['axes.labelsize'] = 80
    plt.rcParams['legend.fontsize'] = 80

    xs = np.arange(len(timecourse)) * dt_ms

    # NORMALIZE THE TIMECOURSE
    if timecourse.max() > 0:
        timecourse_norm = timecourse / timecourse.max()
    else:
        timecourse_norm = timecourse

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs, timecourse_norm, color=color, lw=0.5)  # Use normalized data
    ax.set(xlabel="Time (ms)",
           ylabel="Normalized MSI spikes",  # Update ylabel
           title=title,
           xlim=(xs.min(), xs.max()),
           ylim=(0, 1.05))  # Set ylim to normalized range
    ax.grid(False)
    fig.tight_layout()
    ax.tick_params(axis='both', which='major', length=30, width=1)
    plt.savefig('./Saved_Images/tbw_inset.svg', format='svg')
    return fig, ax


# ────────────────────────────────────────────────────────────────
#          ① TBW – CONTROL
# ────────────────────────────────────────────────────────────────
def main_tbw_profiles():
    base_dir = Path("checkpoint")
    model_paths = [base_dir / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]

    offsets = list(range(-50, 51, 2))
    max_off = offsets[-1]                        # +50  →  +500 ms

    # ---------- 1)  CONTROL TBW ------------------------------------------
    pooled_ctrl = run_fusion_across_models(model_paths,
                                           offsets,
                                           device="cuda")
    fig_c, ax_c, fit_ctrl = plot_psychometric_tbw_ax(pooled_ctrl, cont=True)
    try:
        fig_c.canvas.manager.set_window_title("TBW – CONTROL")
    except Exception:
        pass

    # ---------- 2)  MANIPULATED TBW --------------------------------------
    def tweak_fn(net):
        # FF Inhibition manipulation
        # net.pv_nmda = 1
        # net.targ_ratio = 1.5

        # NMDA manipulation
        net.gNMDA = 0.02

        # Adaptation manipulation

    pooled_manip = run_fusion_across_models(model_paths,
                                            offsets,
                                            device="cuda",
                                            modify_net=tweak_fn)
    fig_m, ax_m, _ = plot_psychometric_tbw_ax(
        pooled_manip,
        reference_fit=fit_ctrl,
        title_suffix="  (comparison)",
        cont=False
    )
    try:
        fig_m.canvas.manager.set_window_title(
            "TBW – MANIPULATED  (+ control overlay)")
    except Exception:
        pass

    # tc_fusion = average_msi_timecourse(model_paths,
    #                                    offset_steps=0,
    #                                    device="cuda",
    #                                    loc_deg=90, T=60, D=5, intensity=1.0)
    # tc_sep = average_msi_timecourse(model_paths,
    #                                 offset_steps=30,
    #                                 device="cuda",
    #                                 loc_deg=90, T=60, D=5, intensity=1.0)

    # plot_timecourse_figure(tc_fusion,
    #                        color="C0")
    # plot_timecourse_figure(tc_sep,
    #                        color="C0")

    # ---------- 4)  show everything -------------------------------------
    plt.show()


# keep the entry‑point guard
if __name__ == "__main__":
    main_tbw_profiles()

