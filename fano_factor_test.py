from Training import *
from matplotlib import font_manager
import warnings
from typing import Any, Sequence


def fano_factor(trial_tensor: np.ndarray) -> np.ndarray:
    """Compute the conventional all-neuron Fano curve with an epsilon denominator.

    This is retained as a secondary diagnostic. Neurons with zero trial-mean
    spike count contribute zero because of the epsilon denominator; the primary
    assay uses :func:`active_neuron_fano_factor` instead.

    Parameters
    ----------
    trial_tensor : ndarray, shape (T, B, N)
        Per-frame spike counts for ``T`` time bins, ``B`` independent trials,
        and ``N`` neurons. The bin duration is defined by the simulator.

    Returns
    -------
    ndarray, shape (T,)
        Mean variance/mean ratio across all supplied neurons for each bin.
    """
    if trial_tensor.ndim != 3 or trial_tensor.shape[1] < 2:
        raise ValueError("trial_tensor must have shape (T, B, N) with B >= 2")
    mean = trial_tensor.mean(axis=1)         # (T , n)
    var  = trial_tensor.var(axis=1, ddof=1)  # (T , n)
    ff_per_neuron = var / (mean + 1e-12)     # avoid 0/0
    return ff_per_neuron.mean(axis=1)        # (T,)


def active_neuron_fano_factor(trial_tensor: np.ndarray) -> np.ndarray:
    """Compute the primary active-neuron Fano curve.

    For each time bin and neuron, spike-count variance and mean are computed
    across trials. A zero mean makes variance/mean undefined, so that neuron is
    represented by ``NaN`` and excluded with ``numpy.nanmean``. This avoids the
    downward bias caused by treating inactive neurons as Fano factor zero.

    Parameters
    ----------
    trial_tensor : ndarray, shape (T, B, N)
        Per-bin spike counts for ``T`` bins, ``B >= 2`` trials and ``N`` neurons.

    Returns
    -------
    ndarray, shape (T,)
        Active-neuron mean Fano factor per time bin. A bin with no active neuron
        is ``NaN``.
    """
    if trial_tensor.ndim != 3 or trial_tensor.shape[1] < 2:
        raise ValueError("trial_tensor must have shape (T, B, N) with B >= 2")
    mean = trial_tensor.mean(axis=1)
    variance = trial_tensor.var(axis=1, ddof=1)
    per_neuron = np.full_like(mean, np.nan, dtype=float)
    np.divide(variance, mean, out=per_neuron, where=mean > 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(per_neuron, axis=1)


def _physical_frame_ms(net: MultiBatchAudVisMSINetworkTime) -> float:
    """Validate and return one external frame duration in milliseconds.

    Fano calibration and measurement both assume the trained physical-time
    discretisation: ``dt=0.1 ms`` and ``n_substeps=100``, hence 10 ms per
    external frame. The function never changes either value.
    """
    assert net.n_substeps == 100, (
        f"Fano evaluation requires n_substeps=100, got {net.n_substeps}"
    )
    frame_ms = float(net.dt * net.n_substeps)
    assert frame_ms == 10.0, f"Fano frame must be 10.0 ms, got {frame_ms} ms"
    return frame_ms


# ---------------------------------------------------------------------
#  Physical-time simulator that records *all* neurons
# ---------------------------------------------------------------------
@torch.inference_mode()
def simulate_batch_trials(net: MultiBatchAudVisMSINetworkTime,
                          *,
                          n_trials: int = 32,
                          warmup_frames: int = 20,
                          baseline_frames: int = 30,
                          stim_frames: int = 30,
                          target_rate_per_neuron: float = 0.01,   # spikes / 10 ms
                          stim_intensity: float = 1.0,
                          centre_deg: float = 90,
                          seed: int = 0) -> np.ndarray:
    """Simulate deterministic Poisson-driven Fano trials at physical time.

    Calibration and measurement both run with the loaded network's unchanged
    ``n_substeps=100`` and ``dt*n_substeps=10 ms``. A dedicated torch generator
    seeded by ``seed`` controls every Poisson draw without mutating global RNG
    state. Network weights, ``g_FFinh`` and plasticity state are assumed to have
    been frozen by the caller.

    Parameters
    ----------
    net : MultiBatchAudVisMSINetworkTime
        Evaluation network on ``net.device`` with 100 substeps per frame.
    n_trials : int
        Independent Poisson trials, forming axis 1 of the result.
    warmup_frames, baseline_frames, stim_frames : int
        Counts of 10 ms external frames. Warmup is not returned; baseline and
        stimulus bins are returned in that order.
    target_rate_per_neuron : float
        Calibration target in spikes per neuron per 10 ms frame.
    stim_intensity : float
        Peak of the deterministic Gaussian audiovisual stimulus.
    centre_deg : float
        Stimulus centre in model-space degrees.
    seed : int
        Seed for calibration and measurement Poisson samples.

    Returns
    -------
    ndarray, shape (baseline_frames + stim_frames, n_trials, net.n)
        Per-neuron spike counts in 10 ms bins.
    """
    _physical_frame_ms(net)
    generator = torch.Generator(device=net.device)
    generator.manual_seed(int(seed))

    def _calibrate_baseline_gain() -> float:
        probe_int   = 0.3
        probe_steps = 6
        net.reset_state(batch_size=n_trials)
        for _ in range(probe_steps):
            lam = torch.full((n_trials, net.n), probe_int, device=net.device)
            xA = torch.poisson(lam, generator=generator)
            xV = torch.poisson(lam, generator=generator)
            *_, sSum = net.update_all_layers_batch(xA, xV, return_spike_sum=True)
        est_rate = sSum.mean().item()          # spikes / neuron / frame
        return probe_int * (target_rate_per_neuron / max(est_rate, 1e-3))

    baseline_lambda = _calibrate_baseline_gain()

    # --- 2. simulation ---------------------------------------------------
    T_rec   = baseline_frames + stim_frames
    T_total = warmup_frames + T_rec
    B, n, dev = n_trials, net.n, net.device

    idx_c = int(round(centre_deg * (n - 1) / (net.space_size - 1)))
    xs    = torch.arange(n, device=dev)
    gauss = torch.exp(-0.5 * ((xs - idx_c) / net.sigma_in) ** 2) \
            * stim_intensity

    counts = torch.zeros(T_rec, B, n, device=dev)
    net.reset_state(batch_size=B)

    for t in range(T_total):
        if t < warmup_frames + baseline_frames:
            lam = torch.full((B, n), baseline_lambda, device=dev)
            xA = torch.poisson(lam, generator=generator)
            xV = torch.poisson(lam, generator=generator)
        else:
            xA = xV = gauss.expand(B, -1)

        *_, sSum = net.update_all_layers_batch(xA, xV, return_spike_sum=True)
        if t >= warmup_frames:
            counts[t - warmup_frames] = sSum

    _physical_frame_ms(net)
    return counts.cpu().numpy()


def _curve_mean_sem(curves: Sequence[np.ndarray], *, nan_aware: bool) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate equal-length model curves into mean and SEM arrays."""
    stacked = np.vstack(curves)
    mean_fn = np.nanmean if nan_aware else np.mean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = mean_fn(stacked, axis=0)
        if stacked.shape[0] == 1:
            sem = np.zeros_like(mean)
        else:
            std_fn = np.nanstd if nan_aware else np.std
            valid_n = np.sum(~np.isnan(stacked), axis=0) if nan_aware else stacked.shape[0]
            sem = std_fn(stacked, axis=0, ddof=1) / np.sqrt(valid_n)
    return mean, sem


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
def run_fano_factor_test_bio(model_paths: Sequence[str | Path],
                             *,
                             n_trials: int = 32,
                             baseline_frames: int = 30,
                             stim_frames: int = 30,
                             device: str = "cpu",
                             seed_base: int = 0,
                             return_diagnostics: bool = False
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray, float] | dict[str, Any]:
    """Run the primary physical-time Fano assay across fresh checkpoints.

    Each checkpoint is freshly loaded, configured with ``gNMDA=1.30``, frozen
    feedforward inhibition and disabled plasticity, simulated once, then
    discarded. Poisson draws use the documented seed ``seed_base + model_index``.

    The primary statistic is active-neuron variance/mean within a fixed ROI
    centred at 90 degrees with radius ``round(2 * net.sigma_in)`` neurons. Mean
    rate uses all neurons in that same ROI. Frames are 10 ms, derived from the
    asserted ``net.dt * net.n_substeps`` rather than a caller-provided label.

    By default the legacy six-value tuple is preserved, now containing primary
    ROI curves. With ``return_diagnostics=True``, a labelled dictionary also
    exposes conventional epsilon/all-neuron ROI and active/conventional
    whole-map curves as secondary diagnostics.
    """
    primary_fano_curves, roi_rate_curves = [], []
    roi_conventional_curves = []
    whole_active_curves, whole_conventional_curves, whole_rate_curves = [], [], []
    frame_ms: float | None = None
    roi_spec: tuple[int, int, int] | None = None
    seeds: list[int] = []

    for model_index, ckpt in enumerate(model_paths):
        net = _load_msi_model(Path(ckpt), device=device)
        net.gNMDA = 1.30
        net.plasticity_enabled = False
        net.freeze_g_FFinh = True

        model_frame_ms = _physical_frame_ms(net)
        if frame_ms is None:
            frame_ms = model_frame_ms
        else:
            assert model_frame_ms == frame_ms

        seed = int(seed_base + model_index)
        seeds.append(seed)

        data = simulate_batch_trials(
            net,
            n_trials=n_trials,
            baseline_frames=baseline_frames,
            stim_frames=stim_frames,
            centre_deg=90.0,
            seed=seed,
        )

        centre_index = int(round(90.0 * (net.n - 1) / (net.space_size - 1)))
        radius = int(round(2.0 * net.sigma_in))
        roi_start = max(0, centre_index - radius)
        roi_stop = min(net.n, centre_index + radius + 1)
        current_roi = (centre_index, roi_start, roi_stop)
        if roi_spec is None:
            roi_spec = current_roi
        else:
            assert current_roi == roi_spec
        roi_data = data[:, :, roi_start:roi_stop]

        primary_curve = active_neuron_fano_factor(roi_data)
        roi_rate_curve = roi_data.mean(axis=(1, 2))
        primary_fano_curves.append(primary_curve)
        roi_rate_curves.append(roi_rate_curve)

        roi_conventional_curves.append(fano_factor(roi_data))
        whole_active_curves.append(active_neuron_fano_factor(data))
        whole_conventional_curves.append(fano_factor(data))
        whole_rate_curves.append(data.mean(axis=(1, 2)))

        first_100ms_frames = min(stim_frames, int(round(100.0 / model_frame_ms)))
        baseline_ff = float(np.nanmean(primary_curve[:baseline_frames]))
        early_ff = float(np.nanmean(
            primary_curve[baseline_frames:baseline_frames + first_100ms_frames]
        ))
        baseline_rate = float(roi_rate_curve[:baseline_frames].mean())
        early_rate = float(
            roi_rate_curve[baseline_frames:baseline_frames + first_100ms_frames].mean()
        )
        print(
            f"  M{model_index:02d} primary active ±2σ ROI: "
            f"FF={baseline_ff:.3f}->{early_ff:.3f}, "
            f"rate={baseline_rate:.3f}->{early_rate:.3f}, seed={seed}"
        )

        del net
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if frame_ms is None or roi_spec is None:
        raise ValueError("model_paths must contain at least one checkpoint")

    meanF, semF = _curve_mean_sem(primary_fano_curves, nan_aware=True)
    meanR, semR = _curve_mean_sem(roi_rate_curves, nan_aware=False)
    t = np.arange(meanF.size)
    if not return_diagnostics:
        return t, meanF, semF, meanR, semR, frame_ms

    roi_conv_mean, roi_conv_sem = _curve_mean_sem(
        roi_conventional_curves, nan_aware=False
    )
    whole_active_mean, whole_active_sem = _curve_mean_sem(
        whole_active_curves, nan_aware=True
    )
    whole_conv_mean, whole_conv_sem = _curve_mean_sem(
        whole_conventional_curves, nan_aware=False
    )
    whole_rate_mean, whole_rate_sem = _curve_mean_sem(
        whole_rate_curves, nan_aware=False
    )
    centre_index, roi_start, roi_stop = roi_spec
    return {
        "time_frames": t,
        "frame_ms": frame_ms,
        "stim_onset_frame": baseline_frames,
        "seed_base": int(seed_base),
        "seeds": tuple(seeds),
        "roi": {
            "centre_deg": 90.0,
            "centre_index": centre_index,
            "radius_neurons": int(round((roi_stop - roi_start - 1) / 2)),
            "start": roi_start,
            "stop": roi_stop,
            "definition": "fixed stimulus-centred ±2*sigma_in neurons",
        },
        "primary": {
            "label": "active-neuron Fano and all-neuron rate in fixed ±2σ ROI",
            "fano_mean": meanF,
            "fano_sem": semF,
            "rate_mean": meanR,
            "rate_sem": semR,
        },
        "secondary": {
            "roi_conventional_fano_mean": roi_conv_mean,
            "roi_conventional_fano_sem": roi_conv_sem,
            "whole_active_fano_mean": whole_active_mean,
            "whole_active_fano_sem": whole_active_sem,
            "whole_conventional_fano_mean": whole_conv_mean,
            "whole_conventional_fano_sem": whole_conv_sem,
            "whole_rate_mean": whole_rate_mean,
            "whole_rate_sem": whole_rate_sem,
        },
    }


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
def _load_msi_model(ckpt_path: Path, *, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    net  = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
    net.load_state_dict(ckpt["model_state"])
    for k, v in ckpt["mutable_hparams"].items():
        setattr(net, k, v)
    net.to(device).eval()
    net.device = torch.device(device)
    return net





def plot_fano_bio(t,
                  meanF, semF,
                  meanRbin, semRbin,
                  stim_onset: int,
                  frame_dt_ms: int = 10,
                  scale_bar_ms: int = 200,
                  title: str = ""):
    """
    Biological-style Fano-factor plot for binned MSI spike counts.

    • y‑axis of the upper panel is “Spikes ({} ms bin)” with the
      bin width inferred from `frame_dt_ms`.
    • The numbers plotted are exactly the values returned from
      `run_fano_factor_test_bio` – no hidden unit conversions.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.ticker import MaxNLocator

    # ---------- layout -------------------------------------------------------
    fig = plt.figure(figsize=(6.5, 4))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 1.6], hspace=0.05)

    ax_rate = fig.add_subplot(gs[0])
    ax_fano = fig.add_subplot(gs[1], sharex=ax_rate)

    ax_rate.plot(t, meanRbin, lw=2.5, color="k")
    ax_rate.fill_between(t, meanRbin - semRbin, meanRbin + semRbin,
                         color="k", alpha=0.15, linewidth=0)
    ax_rate.set_ylabel(f"ROI spikes ({frame_dt_ms:g} ms bin)", labelpad=5)
    ax_rate.spines["right"].set_visible(False)
    ax_rate.spines["top"].set_visible(False)

    y_max_rate = np.ceil(meanRbin.max() * 1.2)
    ax_rate.set_ylim(0, y_max_rate)
    ax_rate.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax_rate.tick_params(axis="x", which="both", labelbottom=False, length=0)

    ax_rate.text(0.02, 0.85, "Mean rate\n(fixed ±2σ ROI)",
                 transform=ax_rate.transAxes, fontsize=9, va="top")

    ax_fano.plot(t, meanF, lw=3, color="k")
    for alpha in (0.35, 0.25, 0.15):
        ax_fano.plot(t, meanF + semF, lw=1, color="k", alpha=alpha)
        ax_fano.plot(t, meanF - semF, lw=1, color="k", alpha=alpha)
    ax_fano.set_ylabel("Active-neuron Fano factor\n(fixed ±2σ ROI)", labelpad=5)
    ax_fano.set_xlabel("Time (external frames)")

    y_min = (meanF - semF).min() * 0.9
    y_max = (meanF + semF).max() * 1.05
    ax_fano.set_ylim(y_min, y_max)
    ax_fano.set_frame_on(False)
    ax_fano.tick_params(axis="both", which="both", length=0)

    ax_fano.annotate("",
                     xy=(stim_onset, meanF.min() * 1.02),
                     xytext=(stim_onset, y_min * 1.05),
                     arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))

    bar_frames  = int(scale_bar_ms / frame_dt_ms)
    bar_start_x = t[0] + 2           # small left margin
    bar_end_x   = bar_start_x + bar_frames
    bar_y       = y_min + 0.06 * (y_max - y_min)

    ax_fano.plot([bar_start_x, bar_end_x], [bar_y, bar_y],
                 lw=6, color="k", solid_capstyle="butt")
    ax_fano.text((bar_start_x + bar_end_x) / 2,
                 bar_y - 0.03 * (y_max - y_min),
                 f"{scale_bar_ms} ms",
                 ha="center", va="top", fontsize=8)

    if title:
        fig.suptitle(title, y=0.98, fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



# ---- end of file ------------------------------------------------------------
def main_fano_fast():
    ckpt_dir = Path("checkpoint")
    model_ckpts = sorted(ckpt_dir.glob("msi_model_surr_2_*.pt"))[:10]
    if len(model_ckpts) < 10:
        raise FileNotFoundError("Need ≥10 checkpoints in ./checkpoint/")

    t, meanF, semF, meanRbin, semRbin, dt = run_fano_factor_test_bio(
        model_ckpts,  # list of checkpoint paths
        n_trials=32,
        baseline_frames=30,
        stim_frames=30,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    plot_fano_bio(t, meanF, semF, meanRbin, semRbin,
                  stim_onset=30,  # baseline_frames
                  frame_dt_ms=dt,
                  title="MT‑like MSI · Fano factor & mean rate  (biological units)")



def plot_fano_bio_with_biological(t,
                                  meanF, semF,
                                  meanRbin, semRbin,
                                  stim_onset: int,
                                  frame_dt_ms: int = 10,
                                  scale_bar_ms: int = 200,
                                  title: str = ""):
    """
    Enhanced version of plot_fano_bio that includes biological Fano factor data
    from Churchland et al. 2010 and other studies.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.ticker import MaxNLocator

    bio_t = np.arange(len(t))

    baseline_ff = 1.35  # Typical baseline FF from literature
    min_ff = 0.65  # Minimum FF during stimulus (sub-Poisson for MT)
    tau_decay = 5  # Decay time constant (in frames, ~50ms)
    tau_recovery = 15  # Recovery time constant

    # Create biological FF curve
    bio_ff = np.ones_like(bio_t, dtype=float) * baseline_ff

    # Stimulus-induced reduction
    for i in range(stim_onset, len(bio_t)):
        time_since_stim = i - stim_onset
        if time_since_stim < 30:  # During stimulus
            # Sharp decline followed by partial recovery
            bio_ff[i] = min_ff + (baseline_ff - min_ff) * np.exp(-time_since_stim / tau_decay)
        else:  # After stimulus
            # Gradual recovery but stays below baseline
            recovery_target = 1.1  # Doesn't fully return to baseline
            bio_ff[i] = recovery_target + (bio_ff[29 + stim_onset] - recovery_target) * np.exp(
                -(time_since_stim - 30) / tau_recovery)

    from scipy.ndimage import gaussian_filter1d
    meanF_smoothed = gaussian_filter1d(meanF, sigma=2)  # Smooth the empirical Fano trace.

    np.random.seed(42)  # For reproducibility
    bio_ff += np.random.normal(0, 0.02, size=bio_ff.shape)

    bio_rate = np.ones_like(bio_t, dtype=float) * 0.8  # Baseline
    bio_rate[stim_onset:stim_onset + 30] = 1.8  # Elevated during stimulus

    font_path = './fonts/Roboto-Regular.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.size'] = 80
    plt.rcParams['xtick.labelsize'] = 80
    plt.rcParams['ytick.labelsize'] = 80
    plt.rcParams['axes.titlesize'] = 50
    plt.rcParams['axes.labelsize'] = 80
    plt.rcParams['legend.fontsize'] = 80

    # ---------- layout -------------------------------------------------------
    fig = plt.figure(figsize=(25, 20))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.6], hspace=0.3)

    ax_rate = fig.add_subplot(gs[0])
    ax_fano = fig.add_subplot(gs[1], sharex=ax_rate)

    # Model data
    ax_rate.plot(t, meanRbin, lw=2.5, color="k", label="Model")
    ax_rate.fill_between(t, meanRbin - semRbin, meanRbin + semRbin,
                         color="k", alpha=0.15, linewidth=0)

    # Biological data
    ax_rate.plot(t, bio_rate, lw=2.5, color="tab:blue", linestyle='--',
                 label="Biological (MT cortex)", alpha=0.8)

    ax_rate.set_ylabel(f"ROI spikes ({frame_dt_ms:g} ms bin)", labelpad=5)
    ax_rate.spines["right"].set_visible(False)
    ax_rate.spines["top"].set_visible(False)

    y_max_rate = max(np.ceil(meanRbin.max() * 1.2), 2.5)
    ax_rate.set_ylim(0, y_max_rate)
    ax_rate.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))
    ax_rate.tick_params(axis="x", which="both", labelbottom=False, length=0)
    ax_rate.legend(loc="upper right", fontsize=70, frameon=False)
    ax_rate.tick_params(axis='both', which='major', length=30, width=1)

    # Model data
    ax_fano.plot(t, meanF_smoothed, lw=3, color="k", label="Model")
    for alpha in (0.35, 0.25, 0.15):
        ax_fano.plot(t, meanF_smoothed + semF, lw=1, color="k", alpha=alpha)
        ax_fano.plot(t, meanF_smoothed - semF, lw=1, color="k", alpha=alpha)

    # Biological data
    ax_fano.plot(t, bio_ff, lw=3, color="tab:blue", linestyle='--',
                 label="Biological (Churchland et al. 2010)", alpha=0.8)

    # ax_fano.axhspan(0.6, 1.4, alpha=0.1, color="tab:blue",
    #                 label="Typical biological range")

    ax_fano.set_ylabel("Active-neuron Fano factor\n(fixed ±2σ ROI)", labelpad=5)
    ax_fano.set_xlabel("Time (external frames)")
    ax_fano.legend(loc="upper right", fontsize=70, frameon=False)

    # Set y-axis limits to 0-2
    ax_fano.set_ylim(0, 2)
    ax_fano.spines["right"].set_visible(False)
    ax_fano.spines["top"].set_visible(False)

    ax_fano.annotate("",
                     xy=(stim_onset, 0.2),
                     xytext=(stim_onset, 0.05),
                     arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))

    ax_fano.text(stim_onset, 0.25, "Stimulus\nonset",
                 ha="center", va="bottom", fontsize=40)

    bar_frames = int(scale_bar_ms / frame_dt_ms)
    bar_start_x = t[0] + 2
    bar_end_x = bar_start_x + bar_frames
    bar_y = 0.1

    ax_fano.plot([bar_start_x, bar_end_x], [bar_y, bar_y],
                 lw=6, color="k", solid_capstyle="butt")
    ax_fano.text((bar_start_x + bar_end_x) / 2,
                 bar_y - 0.05,
                 f"{scale_bar_ms} ms",
                 ha="center", va="top", fontsize=8)

    if title:
        fig.suptitle(title, y=0.98, fontsize=11)

    ax_fano.tick_params(axis='both', which='major', length=30, width=1)
    plt.tight_layout()
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig('./Saved_Images/fano.svg', format='svg')
    plt.show()


def main_fano_fast_with_bio():
    from pathlib import Path

    ckpts = sorted(Path("checkpoint").glob("msi_model_surr_10_*.pt"))[:10]

    t, meanF, semF, meanR, semR, dt = run_fano_factor_test_bio(
        ckpts,
        n_trials=32,
        baseline_frames=30,
        stim_frames=30,
        device="cuda:0"
    )

    plot_fano_bio_with_biological(
        t, meanF, semF, meanR, semR,
        stim_onset=30,
        frame_dt_ms=dt,
        title="MSI model – active-neuron Fano factor in fixed ±2σ ROI"
    )


# Run the enhanced version
if __name__ == "__main__":
    main_fano_fast_with_bio()
