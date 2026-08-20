"""SBW (Spatial Binding Window) measurement pipeline.

Measures the spatial binding window of trained
``MultiBatchAudVisMSINetworkTime`` checkpoints.  The canonical metric is
P(fusion) via an *enhancement* threshold:

    enhancement = AV_roi_spikes - max(A_roi_spikes, V_roi_spikes)
    fused      <=>  enhancement > threshold      (default threshold = 10)

ROI is +/- ``roi_half`` neurons around the auditory stimulus position.

Pipeline (used by ``generate_all_fresh.py`` and ``replot_all_cosmetic.py``):

  1. ``load_msi_model``                       — restore checkpoint.
  2. ``compute_sbw_enhancement_persep``       — batched AV / A-only / V-only
     forward passes; per-trial enhancement; threshold to P(fusion).
     ``compute_sbw_fused_persep`` is the legacy peak/valley classifier
     kept for cross-checks.
  3. ``run_spatial_binding_across_models``    — drive across all
     checkpoints, return per-model curves.
  4. ``fit_pedestal_curve`` / ``plot_spatial_binding_pedestal`` — symmetric
     pedestal (difference-of-sigmoids) fit; SBW = full-width at 50%.

Units / shapes / randomness
---------------------------
Spatial axis: degrees on a circular ``space_size`` (default 180 deg)
neuron ring; separations are signed integers.
Stimulus:     Gaussian over ``net.n`` neurons with ``sigma = net.sigma_in``,
              amplitude ``intensity``, duration ``duration`` timesteps.
Trials:       random base location per trial via ``np.random.default_rng()``.
Outputs:      ``p_fusion`` ndarray of shape ``(n_separations,)`` per model.

Side effects
------------
``g_FFinh`` and ``step_counter`` are saved before each scan and restored
afterwards so that SBW evaluation does not perturb adaptation state.
GPU memory is freed between blocks.
"""
from Training import *
from matplotlib import font_manager
from typing import Sequence


def spatial_binding_diagnostics(
        net,
        *,
        separations_deg: Sequence[int] | None = None,
        example_seps: Sequence[int] | None = None,
        n_trials: int = 100,
        n_examples: int = 3,
        intensity: float = 0.5,
        duration: int = 20,
        method: str = "ratio"):
    """GPU-batched P(fusion) curve + illustrative rasters/profiles.

    Inputs
    ------
    net : MultiBatchAudVisMSINetworkTime
        Loaded model (plasticity_enabled handled by caller).
    separations_deg : sequence of int, optional
        Disparities in degrees (default ``range(0, 61, 5)``).
    example_seps : sequence of int, optional
        Disparities for which raster/profile examples are drawn.
    n_trials : int
        Trials per separation (random base location each trial).
    n_examples : int
        Number of example trials shown per ``example_seps`` value.
    intensity, duration : stimulus amplitude and duration (timesteps).
    method : {"ratio", ...}
        Fusion-classifier mode (kept for backward-compat).

    Returns
    -------
    dict — figure handles + the per-separation P(fusion) curve, plus
    a linear fit and the marked 50%-fusion threshold.

    Side effects
    ------------
    Allocates B = n_sep * n_trials batch elements on ``net.device`` for
    a single forward pass; renders matplotlib figures (no file write).

    Randomness
    ----------
    Uses ``np.random.default_rng()`` (no seed pinned).
    """
    # ───── helpers ────────────────────────────────────────────────────────
    N = net.n

    def idx(deg_arr):
        a = np.asarray(deg_arr, dtype=float)
        return np.round(a * (N - 1) / (net.space_size - 1)).astype(int)

    def make_gauss(idx_centres: torch.Tensor) -> torch.Tensor:
        xs = torch.arange(N, device=net.device, dtype=torch.float32)
        return torch.exp(-0.5 * ((xs - idx_centres.unsqueeze(1)) /
                                 net.sigma_in) ** 2) * intensity

    def is_fused(profile: np.ndarray) -> bool:
        sm = gaussian_filter1d(profile, sigma=2, mode='wrap')
        if sm.max() < 1e-6:  # silent
            return True
        sm /= sm.max()
        peaks, props = find_peaks(sm, height=0.2, distance=10)
        if len(peaks) <= 1:
            return True
        p1, p2 = peaks[np.argsort(props['peak_heights'])[::-1][:2]]

        # Two peaks within 15 neuron indices (~15 deg) = single fused blob
        # (SC receptive fields are ~20-40 deg wide; Meredith & Stein 1986)
        peak_sep = min(abs(p2 - p1), N - abs(p2 - p1))
        if peak_sep <= 15:
            return True

        def valley(i, j):
            direct = abs(j - i)
            seg = sm[min(i, j): max(i, j) + 1] if direct <= N - direct else \
                np.r_[sm[max(i, j):], sm[:min(i, j) + 1]]
            return seg.min()

        return valley(p1, p2) / (min(sm[p1], sm[p2]) + 1e-12) > 0.4

    # ───── default lists ─────────────────────────────────────────────────
    if separations_deg is None:
        separations_deg = list(range(0, 61, 5))  # 0 … 60°
    if example_seps is None:
        example_seps = [0, 20, 40, 60]

    # ───── 1) GPU-batched fusion-probability curve ───────────────────────
    n_sep = len(separations_deg)
    B = n_sep * n_trials
    rng = np.random.default_rng()

    base_deg = rng.integers(30, 150, size=B)
    sep_rep = np.repeat(separations_deg, n_trials)
    locA_deg = base_deg
    locV_deg = (base_deg + sep_rep) % net.space_size

    idxA = torch.as_tensor(idx(locA_deg), device=net.device)
    idxV = torch.as_tensor(idx(locV_deg), device=net.device)

    gA = make_gauss(idxA)
    gV = make_gauss(idxV)

    xA = torch.zeros(B, duration, N, device=net.device)
    xV = torch.zeros_like(xA)
    xA[:, :duration] = gA.unsqueeze(1)
    xV[:, :duration] = gV.unsqueeze(1)

    net.reset_state(batch_size=B)
    msi_sum = torch.zeros(B, N, device=net.device)
    for t in range(duration):
        ret = net.update_all_layers_batch(xA[:, t], xV[:, t], return_spike_sum=True)
        sum_sM = ret[-1]
        msi_sum += sum_sM

    flags = np.zeros((n_sep, n_trials), dtype=bool)
    profs = msi_sum.cpu().numpy()
    for k in range(n_sep):
        s, e = k * n_trials, (k + 1) * n_trials
        for j in range(n_trials):
            flags[k, j] = is_fused(profs[s + j])

    fusion_prob = flags.mean(1)

    #  -- summary plot ----------------------------------------------------
    x = np.asarray(separations_deg, dtype=float)
    y = fusion_prob
    m, c = np.polyfit(x, y, 1)  # slope & intercept
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = m * x_fit + c

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=60, color='C0', zorder=3, label='data')
    plt.plot(x_fit, y_fit, color='C1', lw=2, label=f'slope = {m:.3f}')

    if m != 0:  # avoid divide-by-zero if slope is 0
        x50 = (0.5 - c) / m
        plt.axhline(0.5, ls='--', color='0.4', lw=1)
        plt.axvline(x50, ls='--', color='0.4', lw=1)
        plt.scatter([x50], [0.5], color='red', s=90, zorder=4)
        plt.text(x50, 0.53, f'50 % @ {x50:.1f}°', ha='center', va='bottom',
                 color='red', fontsize=8)

    plt.xlabel('Spatial disparity (°)')
    plt.ylabel('P(fusion)')
    plt.ylim(-0.05, 1.05)
    plt.xlim(0, x.max())
    plt.legend(frameon=False)
    plt.grid(alpha=.3)
    plt.title('Spatial binding window (linear fit)')
    plt.tight_layout()
    plt.show()

    for sep in example_seps:
        print(f"\n=== Raster & profile examples — separation {sep}° ===")
        for ex in range(n_examples):
            base = rng.integers(40, 140)
            loc1, loc2 = base, (base + sep) % net.space_size
            i1, i2 = idx(loc1), idx(loc2)

            g1 = make_gauss(torch.tensor([i1], device=net.device))[0]
            g2 = make_gauss(torch.tensor([i2], device=net.device))[0]

            net.reset_state(1)
            xA_ex, xV_ex = g1.unsqueeze(0), g2.unsqueeze(0)
            hist = []
            for _ in range(duration):
                net.update_all_layers_batch(xA_ex, xV_ex)
                hist.append(net._latest_sMSI[0].cpu().numpy())
            hist = np.stack(hist)
            prof = hist.sum(0)
            fused = is_fused(prof)

            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(8, 3),
                gridspec_kw={'width_ratios': [2, 1]}
            )
            ax1.imshow(hist.T, aspect='auto', cmap='hot')
            ax1.set(
                title=f'A@{loc1}°  V@{loc2}°  →  {"FUSED" if fused else "SEPARATED"}',
                xlabel='time-step', ylabel='neuron'
            )
            ax2.plot(prof)
            ax2.axvline(i1, ls='--', c='r', label=f'A {loc1}°')
            ax2.axvline(i2, ls='--', c='g', label=f'V {loc2}°')
            ax2.set(xlabel='neuron index', ylabel='Σ spikes')
            ax2.legend()
            ax2.grid(alpha=.3)
            plt.tight_layout()
            plt.show()

    return {
        "separations_deg": separations_deg,
        "fusion_prob": fusion_prob.tolist(),
        "raw_flags": flags
    }




# Example usage (main execution):

# ───────────────────────── imports ─────────────────────────
from pathlib import Path
from typing import Sequence
import math
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# ─────────────────── 0. checkpoint loader ───────────────────
def load_msi_model(ckpt_path: Path, *, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
    net.load_state_dict(ckpt["model_state"])
    for k, v in ckpt["mutable_hparams"].items():
        setattr(net, k, v)
    net.to(device).eval()
    net.device = torch.device(device)  # make sure helpers pick this up
    net.plasticity_enabled = False  # disable plasticity during evaluation
    return net


@torch.inference_mode()
def spatial_binding_curve_fast(
        net,
        separations_deg=range(0, 61, 5),
        n_trials=100,
        duration=20,
        intensity=0.5):
    """Compute mean AV spike count profile vs spatial disparity.

    Processes each separation as a SEPARATE forward pass with g_FFinh
    reset before each one, eliminating batch cross-talk where the scalar
    g_FFinh adapts to the batch-mean excitation and inflates the SBW.

    Parameters
    ----------
    net : MultiBatchAudVisMSINetworkTime
    separations_deg : iterable of int
        Spatial disparities in degrees.
    n_trials : int
        Trials per separation (all processed as one batch per sep).
    duration : int
        Stimulus duration in timesteps.
    intensity : float
        Peak amplitude of Gaussian stimulus.

    Returns
    -------
    np.ndarray
        Mean AV spike count for each separation (shape: [n_sep]).
    """
    N, S = net.n, net.space_size
    rng = np.random.default_rng()
    separations_deg = list(separations_deg)
    n_sep = len(separations_deg)

    initial_g_FFinh = net.g_FFinh
    initial_step_counter = net.step_counter
    xs = torch.arange(N, device=net.device, dtype=torch.float32)

    def to_idx(deg):
        return torch.round(
            torch.as_tensor(deg, device=net.device, dtype=torch.float32)
            * (N - 1) / (S - 1)).long()

    def make_gauss(idx):
        return torch.exp(-0.5 * ((xs - idx[:, None]) / net.sigma_in) ** 2) * intensity

    mean_spikes = np.zeros(n_sep)

    for k, sep in enumerate(separations_deg):
        # Fresh g_FFinh + step_counter for each separation — eliminates cross-talk
        net.g_FFinh = initial_g_FFinh
        net.step_counter = initial_step_counter

        base_deg = rng.integers(0, S, size=n_trials)
        locA_deg = base_deg
        locV_deg = (base_deg + sep) % S

        idxA, idxV = to_idx(locA_deg), to_idx(locV_deg)
        gA, gV = make_gauss(idxA), make_gauss(idxV)

        net.reset_state(batch_size=n_trials)
        total = torch.zeros(n_trials, device=net.device)
        for t in range(duration):
            ret = net.update_all_layers_batch(gA, gV, return_spike_sum=True)
            sum_sM = ret[-1]
            total += sum_sM.sum(dim=1)

        mean_spikes[k] = total.mean().item()

    # Restore g_FFinh + step_counter
    net.g_FFinh = initial_g_FFinh
    net.step_counter = initial_step_counter
    return mean_spikes


@torch.inference_mode()
def compute_spatial_binding_curve(
        net,
        *,
        separations_deg: Sequence[int],
        n_trials: int = 100,
        intensity: float = 0.5,
        duration: int = 20,
        block_size: int = 32,  # max #trials simultaneously on GPU
):
    """
    Low‑memory computation of P(fusion) vs. spatial disparity.

    Each separation is handled in independent mini‑batches of size
    ≤ `block_size`, so GPU memory usage is essentially constant.
    """
    N = net.n
    space_deg = net.space_size
    rng = np.random.default_rng()

    # ---------- helper functions -------------------------------------
    def idx(deg_arr):
        a = np.asarray(deg_arr, dtype=float)
        return np.round(a * (N - 1) / (space_deg - 1)).astype(int)

    def make_gauss(idx_centres: torch.Tensor):
        xs = torch.arange(N, device=net.device, dtype=torch.float32)
        return torch.exp(
            -0.5 * ((xs - idx_centres.unsqueeze(1)) / net.sigma_in) ** 2
        ) * intensity

    def is_fused(profile: np.ndarray) -> bool:
        sm = gaussian_filter1d(profile, sigma=2, mode="wrap")
        if sm.max() < 1e-6:
            return True
        sm /= sm.max()
        peaks, props = find_peaks(sm, height=0.2, distance=10)
        if len(peaks) <= 1:
            return True
        p1, p2 = peaks[np.argsort(props["peak_heights"])[::-1][:2]]

        # Two peaks within 15 neuron indices (~15 deg) = single fused blob
        peak_sep = min(abs(p2 - p1), N - abs(p2 - p1))
        if peak_sep <= 15:
            return True

        def valley(i, j):
            direct = abs(j - i)
            seg = (
                sm[min(i, j): max(i, j) + 1]
                if direct <= N - direct
                else np.r_[sm[max(i, j):], sm[: min(i, j) + 1]]
            )
            return seg.min()

        return valley(p1, p2) / (min(sm[p1], sm[p2]) + 1e-12) > 0.4

    # ---------- main loop over separations ---------------------------
    fusion_prob = []

    for sep in separations_deg:
        fused_trials = 0

        n_blocks = math.ceil(n_trials / block_size)
        for blk in range(n_blocks):
            bs = min(block_size, n_trials - blk * block_size)

            base_deg = rng.integers(30, 150, size=bs)
            locA_deg = base_deg
            locV_deg = (base_deg + sep) % space_deg

            idxA = torch.as_tensor(idx(locA_deg), device=net.device)
            idxV = torch.as_tensor(idx(locV_deg), device=net.device)

            gA = make_gauss(idxA)
            gV = make_gauss(idxV)

            xA = torch.zeros(bs, duration, N, device=net.device)
            xV = torch.zeros_like(xA)
            xA[:, :duration] = gA.unsqueeze(1)
            xV[:, :duration] = gV.unsqueeze(1)

            net.reset_state(batch_size=bs)
            msi_sum = torch.zeros(bs, N, device=net.device)

            for t in range(duration):
                ret = net.update_all_layers_batch(xA[:, t], xV[:, t], return_spike_sum=True)
                sum_sM = ret[-1]
                msi_sum += sum_sM

            profs = msi_sum.cpu().numpy()
            for pr in profs:
                fused_trials += is_fused(pr)

            # Release temporary tensors before clearing the CUDA cache.
            del xA, xV, msi_sum, gA, gV, idxA, idxV
            if net.device.type == "cuda":
                torch.cuda.empty_cache()

        fusion_prob.append(fused_trials / n_trials)

    return np.asarray(fusion_prob, dtype=float)


@torch.inference_mode()
def compute_sbw_fused_persep(
        net,
        *,
        separations_deg: Sequence[int],
        n_trials: int = 50,
        intensity: float = 0.5,
        duration: int = 20,
        block_size: int = 32,
):
    """P(fusion) via is_fused() classifier with per-separation processing.

    Each separation runs as a separate forward pass with g_FFinh and
    step_counter restored to initial values, eliminating cross-talk.

    Parameters
    ----------
    net : MultiBatchAudVisMSINetworkTime
    separations_deg : sequence of int
        Spatial disparities in degrees (e.g., range(-80, 85, 5)).
    n_trials : int
        Trials per separation.
    intensity : float
        Stimulus amplitude.
    duration : int
        Stimulus duration in timesteps.
    block_size : int
        Max trials per GPU batch.

    Returns
    -------
    np.ndarray
        P(fusion) for each separation (shape: [n_sep]).
    """
    N = net.n
    S = net.space_size
    rng = np.random.default_rng()

    initial_g_FFinh = net.g_FFinh
    initial_step_counter = net.step_counter

    def idx(deg_arr):
        a = np.asarray(deg_arr, dtype=float)
        return np.round(a * (N - 1) / (S - 1)).astype(int)

    def make_gauss(idx_centres: torch.Tensor):
        xs = torch.arange(N, device=net.device, dtype=torch.float32)
        return torch.exp(
            -0.5 * ((xs - idx_centres.unsqueeze(1)) / net.sigma_in) ** 2
        ) * intensity

    def is_fused(profile: np.ndarray) -> bool:
        sm = gaussian_filter1d(profile, sigma=2, mode="wrap")
        if sm.max() < 1e-6:
            return True
        sm /= sm.max()
        peaks, props = find_peaks(sm, height=0.2, distance=10)
        if len(peaks) <= 1:
            return True
        p1, p2 = peaks[np.argsort(props["peak_heights"])[::-1][:2]]

        peak_sep = min(abs(p2 - p1), N - abs(p2 - p1))
        if peak_sep <= 15:
            return True

        def valley(i, j):
            direct = abs(j - i)
            seg = (
                sm[min(i, j): max(i, j) + 1]
                if direct <= N - direct
                else np.r_[sm[max(i, j):], sm[: min(i, j) + 1]]
            )
            return seg.min()

        return valley(p1, p2) / (min(sm[p1], sm[p2]) + 1e-12) > 0.4

    fusion_prob = []
    for sep in separations_deg:
        # Restore initial state for each separation
        net.g_FFinh = initial_g_FFinh
        net.step_counter = initial_step_counter

        fused_trials = 0
        n_blocks = math.ceil(n_trials / block_size)

        for blk in range(n_blocks):
            bs = min(block_size, n_trials - blk * block_size)

            base_deg = rng.integers(0, S, size=bs)
            locA_deg = base_deg
            locV_deg = (base_deg + sep) % S

            idxA = torch.as_tensor(idx(locA_deg), device=net.device)
            idxV = torch.as_tensor(idx(locV_deg), device=net.device)
            gA = make_gauss(idxA)
            gV = make_gauss(idxV)

            net.reset_state(batch_size=bs)
            msi_sum = torch.zeros(bs, N, device=net.device)

            for t in range(duration):
                ret = net.update_all_layers_batch(gA, gV, return_spike_sum=True)
                sum_sM = ret[-1]
                msi_sum += sum_sM

            profs = msi_sum.cpu().numpy()
            for pr in profs:
                fused_trials += is_fused(pr)

            del gA, gV, idxA, idxV, msi_sum
            if net.device.type == "cuda":
                torch.cuda.empty_cache()

        fusion_prob.append(fused_trials / n_trials)

    # Restore initial state
    net.g_FFinh = initial_g_FFinh
    net.step_counter = initial_step_counter
    return np.asarray(fusion_prob, dtype=float)


@torch.inference_mode()
def compute_sbw_enhancement_persep(
        net,
        *,
        separations_deg: Sequence[int],
        n_trials: int = 50,
        intensity: float = 0.5,
        duration: int = 20,
        block_size: int = 32,
        roi_half: int = 20,
):
    """Absolute multisensory enhancement vs spatial disparity.

    Batches ALL separations × trials into a single forward pass for each
    of the 3 conditions (AV, A-only, V-only), giving ~33× speedup over
    the per-separation sequential approach.

    For each separation and trial, computes:

        enhancement = av_roi_spikes - max(a_roi_spikes, v_roi_spikes)

    where ROI is ±roi_half neurons around the auditory stimulus position.
    g_FFinh and step_counter are saved/restored before each of the 3 passes.

    Parameters
    ----------
    net : MultiBatchAudVisMSINetworkTime
    separations_deg : sequence of int
        Spatial disparities in degrees.
    n_trials : int
        Trials per separation.
    intensity : float
        Stimulus amplitude.
    duration : int
        Stimulus duration in timesteps.
    block_size : int
        Ignored (kept for API compatibility). All separations × trials
        are batched into a single forward pass.
    roi_half : int
        Half-width of ROI in neurons (~4*sigma_in). Default: 20.

    Returns
    -------
    mean_enhancement : np.ndarray
        Mean enhancement for each separation (shape: [n_sep]).
    all_trial_enh : list of np.ndarray
        Per-trial enhancement values for each separation.
        all_trial_enh[k] has shape (n_trials,).
    """
    N = net.n
    S = net.space_size
    rng = np.random.default_rng()

    initial_g_FFinh = net.g_FFinh
    initial_step_counter = net.step_counter

    xs = torch.arange(N, device=net.device, dtype=torch.float32)

    n_sep = len(separations_deg)
    total_batch = n_sep * n_trials

    def to_idx(deg):
        return torch.round(
            torch.as_tensor(deg, device=net.device, dtype=torch.float32)
            * (N - 1) / (S - 1)).long()

    def make_gauss(idx_centres):
        return torch.exp(
            -0.5 * ((xs - idx_centres[:, None]) / net.sigma_in) ** 2
        ) * intensity

    def roi_spikes(msi_sum, center_idx):
        """Sum spikes in ±roi_half neurons around center_idx (with wrapping)."""
        offsets = torch.arange(-roi_half, roi_half + 1, device=msi_sum.device)
        indices = (center_idx[:, None] + offsets[None, :]) % N
        return msi_sum.gather(1, indices).sum(dim=1)

    # ── Generate ALL stimuli for ALL separations × trials at once ──
    # Layout: [sep0_trial0, ..., sep0_trialN-1, sep1_trial0, ..., sepK_trialN-1]
    all_locA = np.zeros(total_batch, dtype=int)
    all_locV = np.zeros(total_batch, dtype=int)
    for k, sep in enumerate(separations_deg):
        s, e = k * n_trials, (k + 1) * n_trials
        base = rng.integers(0, S, size=n_trials)
        all_locA[s:e] = base
        all_locV[s:e] = (base + sep) % S

    idxA = to_idx(all_locA)
    idxV = to_idx(all_locV)
    gA = make_gauss(idxA)
    gV = make_gauss(idxV)
    zeros = torch.zeros_like(gA)

    def run_pass(stim_A, stim_V):
        """Run one full-batch forward pass with fresh state."""
        net.g_FFinh = initial_g_FFinh
        net.step_counter = initial_step_counter
        net.reset_state(batch_size=total_batch)
        msi_sum = torch.zeros(total_batch, N, device=net.device)
        for _ in range(duration):
            ret = net.update_all_layers_batch(stim_A, stim_V, return_spike_sum=True)
            sum_sM = ret[-1]
            msi_sum += sum_sM
        return msi_sum

    # ── 3 passes total (AV, A-only, V-only) ──
    av_sum = run_pass(gA, gV)
    a_sum = run_pass(gA, zeros)
    v_sum = run_pass(zeros, gV)

    # ── Compute enhancement per trial ──
    av_roi = roi_spikes(av_sum, idxA)
    a_roi = roi_spikes(a_sum, idxA)
    v_roi = roi_spikes(v_sum, idxA)
    enh_all = (av_roi - torch.max(a_roi, v_roi)).cpu().numpy()

    del gA, gV, zeros, av_sum, a_sum, v_sum
    if net.device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Reshape to per-separation results ──
    enh_reshaped = enh_all.reshape(n_sep, n_trials)
    mean_enhancement = enh_reshaped.mean(axis=1)
    all_trial_enh = [enh_reshaped[k] for k in range(n_sep)]

    net.g_FFinh = initial_g_FFinh
    net.step_counter = initial_step_counter
    return mean_enhancement, all_trial_enh


def fit_pedestal_curve(pooled, k_edge=4.0):
    x, y = pooled["separations_deg"], pooled["mean_prob"]
    p0 = [y.min(), y.max(), 25.0]  # base, top, half‑width
    popt, _ = curve_fit(
        lambda X, b, t, w: pedestal(X, b, t, w, k_edge), x, y, p0=p0
    )
    xs = np.linspace(x.min(), x.max(), 600)
    ys = pedestal(xs, *popt, k_edge)
    return xs, ys, popt  # fitted curve and pedestal parameters


# ─────────────────── 2. run across checkpoints ───────────────────
def run_spatial_binding_across_models(
        model_paths,
        *,
        separations_deg,
        n_trials=20,
        intensity=1,
        duration=20,
        device="cpu",
        modify_net=None,  # optional modifier
        method="spike_profile",  # "spike_profile", "is_fused", or "enhancement"
        block_size=32,
        ref_baseline=None,
        ref_peak_shift=None,
        enhancement_threshold=0.0,
):
    """Run SBW across model checkpoints.

    Parameters
    ----------
    method : str
        "spike_profile" — raw AV spike count profile (default).
        "is_fused" — peak/valley classifier P(fusion).
        "enhancement" — P(fusion) via enhancement thresholding.
            Per-trial: enhancement = AV_roi - max(A_roi, V_roi).
            P(fusion) = fraction of trials where enhancement > threshold.
    enhancement_threshold : float
        Spike-count threshold for the "enhancement" method.
        A trial is classified as "fused" if enhancement > threshold.
        Default: 0.0 (any positive enhancement counts as fusion).
    ref_baseline : float or None
        (Unused for enhancement method — kept for backward compatibility.)
    ref_peak_shift : float or None
        (Unused for enhancement method — kept for backward compatibility.)
    """
    curves = []
    for p in model_paths:
        net = load_msi_model(Path(p), device=device)

        if callable(modify_net):
            modify_net(net)  # tweak parameters *in‑place*

        if method == "is_fused":
            curves.append(
                compute_sbw_fused_persep(
                    net,
                    separations_deg=separations_deg,
                    n_trials=n_trials,
                    intensity=intensity,
                    duration=duration,
                    block_size=block_size,
                )
            )
        elif method == "enhancement":
            mean_enh, all_trial_enh = compute_sbw_enhancement_persep(
                net,
                separations_deg=separations_deg,
                n_trials=n_trials,
                intensity=intensity,
                duration=duration,
                block_size=block_size,
            )
            # P(fusion) per separation: fraction of trials with enhancement > threshold
            p_fusion = np.array([
                (trial_enh > enhancement_threshold).mean()
                for trial_enh in all_trial_enh
            ])
            curves.append(p_fusion)
        else:
            curves.append(
                spatial_binding_curve_fast(
                    net,
                    separations_deg=separations_deg,
                    n_trials=n_trials,
                    intensity=intensity,
                    duration=duration,
                )
            )
        del net
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    curves = np.vstack(curves)  # (n_models, n_sep)
    sep_arr = np.asarray(separations_deg, float)

    # ── Symmetrize: average mirror separations (+s and -s) ──
    mag_to_cols = {}
    for i, s in enumerate(separations_deg):
        mag = abs(s)
        mag_to_cols.setdefault(mag, []).append(i)

    mags = np.array(sorted(mag_to_cols.keys()), dtype=float)
    n_mags = len(mags)

    curves_sym = np.zeros((curves.shape[0], n_mags))
    for j, mag in enumerate(mags):
        cols = mag_to_cols[mag]
        curves_sym[:, j] = curves[:, cols].mean(axis=1)

    if method == "enhancement":
        # P(fusion) is already in [0, 1] — no normalization needed.
        # Average across models and mirror the one-sided curve.
        mean_onesided = curves_sym.mean(0)
        all_onesided = curves_sym
        baseline = 0.0
        peak_shift = 1.0
    else:
        # Legacy normalization path for spike_profile / is_fused methods.
        mean_sym = curves_sym.mean(0)
        baseline = ref_baseline if ref_baseline is not None else mean_sym[-1]
        peak_shift = ref_peak_shift if ref_peak_shift is not None else (mean_sym - baseline).max()

        if peak_shift > 1e-9:
            mean_onesided = (mean_sym - baseline) / peak_shift
            all_onesided = (curves_sym - baseline) / peak_shift
            if ref_baseline is None:
                mean_onesided = np.clip(mean_onesided, 0.0, 1.0)
                all_onesided = np.clip(all_onesided, 0.0, 1.0)
        else:
            mean_onesided = np.zeros_like(mean_sym)
            all_onesided = np.zeros_like(curves_sym)

    # ── Mirror back to full symmetric curve for plotting ──
    full_sep = np.concatenate([-mags[::-1], mags[1:]])
    full_mean = np.concatenate([mean_onesided[::-1], mean_onesided[1:]])
    full_sem_parts = all_onesided.std(0, ddof=1) / np.sqrt(curves.shape[0])
    full_sem = np.concatenate([full_sem_parts[::-1], full_sem_parts[1:]])
    full_all = np.concatenate([all_onesided[:, ::-1], all_onesided[:, 1:]], axis=1)

    return {
        "separations_deg": full_sep,
        "mean_prob": full_mean,
        "sem_prob": full_sem,
        "all_prob": full_all,
        "ref_baseline": float(baseline),
        "ref_peak_shift": float(peak_shift),
        "enhancement_threshold": float(enhancement_threshold) if method == "enhancement" else None,
    }


def gaussian(x, base, amp, mu, sigma):
    """General Gaussian: base + amp * exp(-0.5*((x-mu)/sigma)^2)."""
    return base + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _gaussian_centered(x, amp, sigma):
    """Zero-centered Gaussian for symmetric SBW data: amp * exp(-0.5*(x/sigma)^2)."""
    return amp * np.exp(-0.5 * (x / sigma) ** 2)


def fit_gaussian_curve(pooled):
    """Fit a zero-centered Gaussian to the symmetrized SBW profile.

    After symmetrization + baseline subtraction, the data is centered
    at 0° with baseline ≈ 0, so we fit: P(sep) = amp * exp(-0.5*(sep/sigma)^2).

    Returns (xs, ys, popt) where popt = (amp, sigma).
    """
    x, y = pooled["separations_deg"], pooled["mean_prob"]
    p0 = [y.max(), 20.0]
    popt, _ = curve_fit(_gaussian_centered, x, y, p0=p0,
                        bounds=([0.1, 5], [3.0, 80]))
    xs = np.linspace(x.min(), x.max(), 600)
    ys = _gaussian_centered(xs, *popt)
    return xs, ys, popt


# ---------------- pedestal (3-parameter flattop) ----------------
def pedestal(x, base, top, half_width, k=4.0):
    """
    Smooth symmetric flattop:

        base                     (outside |x| > half_width)
        top                      (inside  |x| < half_width, up to logistic softness)
        half_width: positive; full width ≈ 2*half_width
        k : fixed edge steepness (deg). Smaller = sharper edges.
    """
    left = 1.0 / (1.0 + np.exp(-(x + half_width) / k))
    right = 1.0 / (1.0 + np.exp((x - half_width) / k))
    return base + (top - base) * left * right


def plot_spatial_binding_pedestal(
        pooled,
        *,
        k_edge=4.0,
        level=0.5,
        reference_fit=None,  # (xs_ref, ys_ref) or None
        ref_tint_color="lightcoral",
        ref_tint_alpha=0.18,
):
    import numpy as np, matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    x, y, err = (pooled[k] for k in ("separations_deg",
                                     "mean_prob",
                                     "sem_prob"))

    fit = lambda X, b, t, w: pedestal(X, b, t, w, k_edge)
    p0 = [y.min(), y.max(), 25.0]
    popt, _ = curve_fit(fit, x, y, p0=p0)
    xs_fit = np.linspace(x.min(), x.max(), 600)
    ys_fit = fit(xs_fit, *popt)

    signs = ys_fit - level
    inside = signs[:-1] * signs[1:] <= 0
    cross_m = [xs_fit[i] + (level - ys_fit[i]) *
               (xs_fit[i + 1] - xs_fit[i]) / (ys_fit[i + 1] - ys_fit[i])
               for i in np.where(inside)[0]]

    # ---------- canvas --------------------------------------------------
    y_bottom = -0.05
    y_top = max(1.05, ys_fit.max() * 1.05)
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(x, y, yerr=err, fmt="o", capsize=0,
                elinewidth=1.2, markeredgewidth=0.8,
                label="mean ± SEM (n=10)", zorder=3)
    ax.plot(xs_fit, ys_fit, color="C1", lw=2,
            label="Pedestal fit (manipulated)", zorder=2)

    # ---------- CONTROL pedestal (grey dashed) --------------------------
    if reference_fit is not None:
        xs_ref, ys_ref = reference_fit
        ax.plot(xs_ref, ys_ref, ls="--", color="0.5", lw=2,
                label="Control pedestal", zorder=1.5)

        ref_signs = ys_ref - level
        inside_r = ref_signs[:-1] * ref_signs[1:] <= 0
        cross_r = [xs_ref[i] + (level - ys_ref[i]) *
                   (xs_ref[i + 1] - xs_ref[i]) / (ys_ref[i + 1] - ys_ref[i])
                   for i in np.where(inside_r)[0]]

        for xc in cross_r:
            ax.vlines(xc, y_bottom, level, ls=":", color="0.4",  # Changed to grey
                      lw=1.3, zorder=1)
        if len(cross_r) == 2:
            mask_r = (xs_ref >= min(cross_r)) & (xs_ref <= max(cross_r))
            ax.fill_between(xs_ref, y_bottom, level, where=mask_r,
                            color=ref_tint_color, alpha=ref_tint_alpha,
                            zorder=0)

    mask_m = (xs_fit >= min(cross_m)) & (xs_fit <= max(cross_m))
    ax.fill_between(xs_fit, y_bottom, level, where=mask_m,
                    color="C1", alpha=0.15, zorder=1)

    # ---------- cosmetics ----------------------------------------------
    ax.axhline(level, ls=":", color="0.4")
    for xc in cross_m:
        ax.vlines(xc, y_bottom, level, ls=":", color="C1")  # Changed to orange (C1)


    xticks = np.arange(-80, 85, 20)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(t):d}" for t in xticks])
    ax.set(xlabel="Spatial disparity (°)",
           ylabel="P(fusion)",
           title="Spatial binding window – Pedestal fits",
           ylim=(y_bottom, y_top),
           xlim=(xs_fit.min(), xs_fit.max()))
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.grid(False)
    plt.tight_layout()
    plt.show()


def plot_spatial_binding_gaussian(pooled):
    """
    Averaged SBW curve with Gaussian fit, central tint down to the x-axis,
    and vertical dotted lines that touch the axis.
    """
    # ---------- raw means & SEM ------------------------------------------
    x, y, err = pooled["separations_deg"], pooled["mean_prob"], pooled["sem_prob"]
    level = 0.5  # 50 % reference

    # ---------- Gaussian fit --------------------------------------------
    p0 = [y.min(), y.max() - y.min(), 0.0, 20.0]
    popt, _ = curve_fit(gaussian, x, y, p0=p0)
    base, amp, mu, sigma = popt
    xs_fit = np.linspace(x.min(), x.max(), 600)
    ys_fit = gaussian(xs_fit, *popt)

    # two x-values where Gaussian crosses 0.5
    inside = (ys_fit - level)[:-1] * (ys_fit - level)[1:] <= 0
    crossings = []
    for i in np.where(inside)[0]:
        x1, x2, y1, y2 = xs_fit[i], xs_fit[i + 1], ys_fit[i], ys_fit[i + 1]
        crossings.append(x1 + (level - y1) * (x2 - x1) / (y2 - y1))
    crossings = np.asarray(crossings)  # left & right

    # ---------- figure ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))

    # mean ± SEM data
    ax.errorbar(
        x, y, yerr=err,
        fmt="o", capsize=0,
        elinewidth=1.2, markeredgewidth=0.8,
        label="mean ± SEM (n=10)", zorder=3,
    )

    # Gaussian fit
    ax.plot(xs_fit, ys_fit, color="C1", lw=2, label="Gaussian fit", zorder=2)

    y_bottom = -0.05
    y_top = max(1.05, ys_fit.max() * 1.05)
    ax.set_ylim(y_bottom, y_top)

    mask_cent = (xs_fit >= crossings.min()) & (xs_fit <= crossings.max())
    ax.fill_between(
        xs_fit, y_bottom, ys_fit,
        where=mask_cent,
        color="C1", alpha=0.15, zorder=1,
    )

    # 50 % reference line
    ax.axhline(level, ls=":", color="0.4")

    for xc in crossings:
        ax.vlines(xc, ymin=y_bottom, ymax=level, ls=":", color="0.4")
        ax.text(
            xc, level + 0.03,
            f"{abs(xc):.1f}°",
            ha="center", va="bottom", fontsize=8, color="0.25",
        )

    # x-ticks with absolute labels
    xticks = np.arange(-80, 85, 20)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(t):d}" for t in xticks])

    # remaining cosmetics
    ax.set(
        xlabel="Spatial disparity (°)",
        ylabel="P(fusion)",
        title="Spatial binding window – 10-model mean (Gaussian fit)",
        xlim=(xs_fit.min(), xs_fit.max()),
    )
    ax.legend(frameon=False)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Numeric fit summary for interactive diagnostics.
    print(f"Gaussian fit: base={base:.3f}, amp={amp:.3f}, μ={mu:.2f}, σ={sigma:.2f}")
    if len(crossings) == 2:
        print(f"Spatial 50 % window: ±{abs(crossings[1]):.1f}°")


# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
import numpy as np


def plot_msi_profile(profile: np.ndarray,
                     *,
                     title: str,
                     color: str = "C0",
                     figsize=(20, 15)):
    """
    Render one MSI‑layer population response as a separate figure.

    Parameters
    ----------
    profile : (n,) NumPy array – summed spikes per neuron
    title   : str  – figure title
    color   : matplotlib colour spec
    figsize : (w,h) inches
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
    fig, ax = plt.subplots(figsize=figsize)
    xs = np.linspace(0, 180, profile.size, endpoint=False)
    if profile.max() > 0:
        ax.plot(xs, profile / profile.max(), color=color, lw=0.5)
    else:
        ax.plot(xs, profile, color=color, lw=0.5)

    ax.set(
        xlabel="Azimuth (°)",
        ylabel="Normalised spikes",
        title=title,
        ylim=(0, 1.05),
        xlim=(0, 180),
    )
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    ax.tick_params(axis='both', which='major', length=20, width=1)
    plt.savefig('./Saved_Images/sbw_inset.svg', format='svg')
    return fig, ax

# ───────────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np, torch, matplotlib.pyplot as plt
from scipy.optimize import curve_fit   # already imported higher up

@torch.inference_mode()
def msi_profile_at_disparity(net,
                             disparity_deg: float,
                             *,
                             centre_deg: float = 90.0,
                             duration: int = 20,
                             intensity: float = .5) -> np.ndarray:
    """
    Run a brief bimodal pulse (Audio at *centre_deg*, Visual shifted by
    *disparity_deg*) through *net* and return the summed MSI‑layer spikes
    as a 1‑D NumPy array (length = n neurons).
    """
    n, S, dev = net.n, net.space_size, net.device
    to_idx = lambda deg: int(round(deg * (n - 1) / (S - 1))) % n

    iA, iV = to_idx(centre_deg), to_idx(centre_deg + disparity_deg)

    xs = torch.arange(n, dtype=torch.float32, device=dev)
    g = lambda idx: torch.exp(-.5 * ((xs - idx) / net.sigma_in) ** 2) * intensity
    gauss_A, gauss_V = g(iA), g(iV)

    net.reset_state(batch_size=1)
    hist = torch.zeros(duration, n, device=dev)

    for _ in range(duration):
        ret = net.update_all_layers_batch(gauss_A.unsqueeze(0),
                                          gauss_V.unsqueeze(0),
                                          return_spike_sum=True)
        sum_sM = ret[-1]
        hist[_] = sum_sM[0]

    return hist.sum(0).cpu().numpy()


def add_msi_inset(parent_ax,
                  profile: np.ndarray,
                  *,
                  title: str,
                  loc: int = 1,
                  width: str = "28%",
                  height: str = "38%",
                  color: str = "C0"):
    """
    Embed a *profile* (already normalised or not) into *parent_ax*.
    loc: 1=UR, 2=UL, 3=LL, 4=LR (same as matplotlib legend codes).
    """
    ax_in = inset_axes(parent_ax, width=width, height=height, loc=loc,
                       borderpad=1.2)
    xs = np.linspace(0, 180, profile.size, endpoint=False)
    if profile.max() > 0:
        ax_in.plot(xs, profile / profile.max(), color=color, lw=1.4)
    else:  # Silent profile: plot zeros so the frame remains visible.
        ax_in.plot(xs, profile, color=color, lw=1.4)

    ax_in.set_xticks([]), ax_in.set_yticks([])
    ax_in.set_ylim(0, 1.05)
    ax_in.set_title(title, fontsize=7, pad=1.0)
    for sp in ax_in.spines.values():
        sp.set_linewidth(.6), sp.set_color("0.4")
    return ax_in


# ───────────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────
#  Improved SBW plotting helper
# ───────────────────────────────────────────────────────────────────
def plot_spatial_binding_pedestal_ax(
        pooled,
        *,
        k_edge: float = 4.0,
        level: float = .5,
        reference_fit=None,
        ref_tint_alpha: float = .12,
        manip_tint_alpha: float = .18,
        cont: False,
        out_path=None):
    """
    Returns (fig, ax) — draws pedestal fit for *pooled* data.
    If *reference_fit* ≠ None (tuple of xs, ys) the control curve
    gets its own vertical dotted guides and grey shaded area.
    """
    # ---------------- raw data ----------------
    x, y, err = (pooled[k] for k in ("separations_deg",
                                     "mean_prob",
                                     "sem_prob"))

    # manipulated‑condition fit
    fit = lambda X, b, t, w: pedestal(X, b, t, w, k_edge)
    p0 = [y.min(), y.max(), 25.0]
    popt, _ = curve_fit(fit, x, y, p0=p0)
    xs_fit = np.linspace(x.min(), x.max(), 600)
    ys_fit = fit(xs_fit, *popt)

    font_path = './fonts/Roboto-Regular.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.size'] = 60
    plt.rcParams['xtick.labelsize'] = 60
    plt.rcParams['ytick.labelsize'] = 60
    plt.rcParams['axes.titlesize'] = 50
    plt.rcParams['axes.labelsize'] = 60
    plt.rcParams['legend.fontsize'] = 60

    # crossing points for manipulated
    sgn_m = ys_fit - level
    xsect_m = xs_fit[:-1][sgn_m[:-1] * sgn_m[1:] <= 0]
    cross_m = [xs_fit[i] + (level - ys_fit[i]) *
               (xs_fit[i + 1] - xs_fit[i]) / (ys_fit[i + 1] - ys_fit[i])
               for i in np.nonzero(sgn_m[:-1] * sgn_m[1:] <= 0)[0]]

    if cross_m:
        print(f"Manipulated crossing points: {cross_m}")

    # ---------------- canvas ----------------
    y_bottom = -.05
    fig, ax = plt.subplots(figsize=(25, 14))

    # data points
    ax.errorbar(x, y, yerr=err,
                fmt='o', ms=5, capsize=3, color='C0', zorder=3, elinewidth=0.5)

    # manipulated curve
    ax.plot(xs_fit, ys_fit, color="C1", lw=2.5, zorder=2)

    # control / reference curve (optional)
    cross_r = None
    if reference_fit is not None:
        xs_ref, ys_ref = reference_fit
        ax.plot(xs_ref, ys_ref, ls="--", color=".5", lw=2, zorder=1.5)

        # crossing points for control
        ys_ref_interp = np.interp(xs_fit, xs_ref, ys_ref)
        sgn_r = ys_ref_interp - level
        cross_r = [xs_fit[i] + (level - ys_ref_interp[i]) *
                   (xs_fit[i + 1] - xs_fit[i]) /
                   (ys_ref_interp[i + 1] - ys_ref_interp[i])
                   for i in np.nonzero(sgn_r[:-1] * sgn_r[1:] <= 0)[0]]

    if cross_r:
            print(f"Control crossing points: {cross_r}")

    # ---------------- guides & shading ----------------
    # horizontal 50 % line
    ax.axhline(level, ls=":", color=".4")

    # manipulated guides/shading
    if len(cross_m) == 2:
        mask_m = (xs_fit >= min(cross_m)) & (xs_fit <= max(cross_m))
        ax.fill_between(xs_fit, y_bottom, level, where=mask_m,
                        color="C1", alpha=manip_tint_alpha, zorder=0.5)
        for xc in cross_m:
            ax.vlines(xc, y_bottom, level, ls=":", color="C1")

    # control guides/shading
    if cross_r and len(cross_r) == 2:
        mask_r = (xs_fit >= min(cross_r)) & (xs_fit <= max(cross_r))
        ax.fill_between(xs_fit, y_bottom, level, where=mask_r,
                        color=".5", alpha=ref_tint_alpha, zorder=0.4)
        for xc in cross_r:
            ax.vlines(xc, y_bottom, level, ls=":", color=".5")

    # ---------------- cosmetics ----------------
    xticks = np.arange(-80, 85, 20)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(t):d}" for t in xticks])
    ax.set(xlabel="Spatial disparity (°)",
           ylabel="P(fusion)",
           title="Spatial binding window – Pedestal fits",
           ylim=(y_bottom, max(1.05, ys_fit.max() * 1.05)),
           xlim=(xs_fit.min(), xs_fit.max()))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.grid(False)
    plt.tight_layout()
    ax.tick_params(axis='both', which='major', length=20, width=1)
    save_path = out_path or './Saved_Images/SBW_curve.svg'
    plt.savefig(save_path, format='svg')
    return fig, ax



# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
#  Revised MAIN  – shows:
#      ①  SBW (Control‑only)
#      ②  SBW (Manipulated + control overlay)
# ────────────────────────────────────────────────────────────────
def main():
    # ----- paths & common parameters -------------------------------------
    base_dir = Path("checkpoint")
    model_paths = [base_dir / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]
    separations = tuple(range(-80, 85, 5))
    max_sep = max(abs(s) for s in separations)          # 80 °

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    pooled_ctrl = run_spatial_binding_across_models(
        model_paths,
        separations_deg=separations,
        device="cuda:0"
    )

    xs_ref, ys_ref, _ = fit_pedestal_curve(pooled_ctrl)

    fig_ctrl, _ = plot_spatial_binding_pedestal_ax(
        pooled_ctrl,
        reference_fit=None,      # nothing overlaid here,
        cont=True
    )
    try:                                    # nicer window‐title if backend supports it
        fig_ctrl.canvas.manager.set_window_title("SBW – CONTROL")
    except Exception:
        pass

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    def manipulation(net):
        # FF Inhibition manipulation
        # net.pv_nmda = 1
        # net.targ_ratio = 1.5

        # NMDA manipulation
        net.gNMDA = 0.02

        # Adaptation manipulation


    pooled_mod = run_spatial_binding_across_models(
        model_paths,
        separations_deg=separations,
        modify_net=manipulation,
        device="cuda:0"
    )

    fig_mod, ax_mod = plot_spatial_binding_pedestal_ax(
        pooled_mod,
        reference_fit=(xs_ref, ys_ref),
        cont=False
    )
    try:
        fig_mod.canvas.manager.set_window_title("SBW – MANIPULATED (+ control overlay)")
    except Exception:
        pass

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    sample_net = load_msi_model(model_paths[0], device="cuda:0")
    manipulation(sample_net)                       # apply same manipulation

    prof_fusion = msi_profile_at_disparity(sample_net,
                                           disparity_deg=0,
                                           centre_deg=90,
                                           duration=20,
                                           intensity=.5)
    prof_sep = msi_profile_at_disparity(sample_net,
                                        disparity_deg=max_sep,
                                        centre_deg=45,
                                        duration=20,
                                        intensity=.5)

    plot_msi_profile(prof_fusion,
                     title="MSI activity – Δ azimuth 0°",
                     color="C0")
    # plot_msi_profile(prof_sep,
    #                  title=f"MSI activity – Δ azimuth {max_sep}°",
    #                  color="C0")

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    import matplotlib.pyplot as plt
    plt.show()



# usual guard
if __name__ == "__main__":
    main()


